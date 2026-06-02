#!/usr/bin/env python3
"""Measure baseline FP16 OPT KV-cache growth on CUDA.

This is intentionally a Hugging Face baseline, not an Oaken evaluation. It
loads local OPT checkpoints, runs random input tokens, and compares the
theoretical KV-cache formula with the actual `output.past_key_values` tensors.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


SEQ_LENGTHS = [128, 256, 512, 1024, 2048]
BATCH_SIZES = [1, 2, 4]
NO_CACHE_SEQ_LENGTHS = [128, 512, 1024]
DTYPE = torch.float16
DTYPE_NAME = "float16"
DTYPE_BYTES = 2


CSV_COLUMNS = [
    "model",
    "gpu_name",
    "num_layers",
    "hidden_size",
    "batch_size",
    "sequence_length",
    "dtype",
    "use_cache",
    "theoretical_kv_bytes",
    "theoretical_kv_mib",
    "actual_past_key_values_bytes",
    "actual_past_key_values_mib",
    "theoretical_actual_ratio",
    "torch_cuda_max_memory_allocated_bytes",
    "torch_cuda_max_memory_allocated_mib",
    "torch_cuda_max_memory_reserved_bytes",
    "torch_cuda_max_memory_reserved_mib",
    "elapsed_sec",
    "success",
    "error_type",
    "error_message",
]


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def mib(value: int | float | None) -> str:
    if value is None:
        return ""
    return f"{float(value) / (1024 ** 2):.6f}"


def ratio(theoretical: int, actual: int | None) -> str:
    if actual is None or actual == 0:
        return ""
    return f"{theoretical / actual:.6f}"


def model_label(model_path: Path) -> str:
    return model_path.name


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cuda-unavailable"


def run_text(command: list[str], timeout: int | None = None) -> str:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip()


def query_vram() -> dict[str, int] | None:
    try:
        output = run_text(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if not output:
        return None
    parts = [part.strip() for part in output.splitlines()[0].split(",")]
    if len(parts) != 3:
        return None
    try:
        used, free, total = (int(part) for part in parts)
    except ValueError:
        return None
    return {"used_mib": used, "free_mib": free, "total_mib": total}


def monitor_vram(stop: threading.Event, samples: list[dict[str, object]], interval_s: float = 0.05) -> None:
    while not stop.is_set():
        sample = query_vram()
        if sample is not None:
            samples.append({"timestamp": now_iso(), **sample})
        stop.wait(interval_s)


def write_vram_csv(path: Path, samples: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "used_mib", "free_mib", "total_mib"])
        writer.writeheader()
        writer.writerows(samples)


def theoretical_kv_bytes(num_layers: int, batch_size: int, sequence_length: int, hidden_size: int) -> int:
    return 2 * num_layers * batch_size * sequence_length * hidden_size * DTYPE_BYTES


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def past_key_values_bytes(past_key_values: Any, log_lines: list[str]) -> int:
    if past_key_values is None:
        log_lines.append("past_key_values is None")
        return 0

    total = 0
    for layer_idx, layer_cache in enumerate(past_key_values):
        if layer_cache is None:
            log_lines.append(f"layer {layer_idx}: layer cache is None, skipped")
            continue
        if len(layer_cache) < 2:
            log_lines.append(f"layer {layer_idx}: expected key/value tensors, got {len(layer_cache)} entries")
            continue
        key, value = layer_cache[0], layer_cache[1]
        if key is None:
            log_lines.append(f"layer {layer_idx}: key tensor is None, skipped")
        if value is None:
            log_lines.append(f"layer {layer_idx}: value tensor is None, skipped")
        key_bytes = tensor_bytes(key)
        value_bytes = tensor_bytes(value)
        total += key_bytes + value_bytes
        key_shape = tuple(key.shape) if key is not None else None
        value_shape = tuple(value.shape) if value is not None else None
        log_lines.append(
            f"layer {layer_idx}: key_shape={key_shape} value_shape={value_shape} "
            f"key_bytes={key_bytes} value_bytes={value_bytes}"
        )
    return total


def error_type_from_exception(exc: BaseException) -> str:
    message = str(exc).lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message:
        return "CUDA_OOM"
    return type(exc).__name__


def run_case(
    model: torch.nn.Module,
    config: Any,
    model_name: str,
    batch_size: int,
    sequence_length: int,
    use_cache: bool,
    output_dir: Path,
) -> dict[str, object]:
    log_path = output_dir / "raw_logs" / (
        f"{model_name}_bs{batch_size}_seq{sequence_length}_cache{str(use_cache).lower()}.log"
    )
    vram_path = output_dir / "raw_logs" / (
        f"{model_name}_bs{batch_size}_seq{sequence_length}_cache{str(use_cache).lower()}_vram.csv"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_lines = [
        f"run_start={now_iso()}",
        f"model={model_name}",
        f"batch_size={batch_size}",
        f"sequence_length={sequence_length}",
        f"use_cache={use_cache}",
    ]
    theory = theoretical_kv_bytes(config.num_hidden_layers, batch_size, sequence_length, config.hidden_size)

    actual_bytes: int | None = None
    elapsed = 0.0
    success = False
    error_type = ""
    error_message = ""
    max_allocated = 0
    max_reserved = 0
    samples: list[dict[str, object]] = []
    stop = threading.Event()

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_allocated = torch.cuda.memory_allocated()
        start_reserved = torch.cuda.memory_reserved()
        log_lines.append(f"start_memory_allocated_bytes={start_allocated}")
        log_lines.append(f"start_memory_reserved_bytes={start_reserved}")

        monitor = threading.Thread(target=monitor_vram, args=(stop, samples), daemon=True)
        monitor.start()

        input_ids = torch.randint(
            low=0,
            high=int(config.vocab_size),
            size=(batch_size, sequence_length),
            device="cuda",
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids, device="cuda")

        start = time.monotonic()
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        actual_bytes = past_key_values_bytes(getattr(output, "past_key_values", None), log_lines)

        max_allocated = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()
        success = True

        del output
        del input_ids
        del attention_mask
    except BaseException as exc:  # preserve rows for OOM and unexpected failures
        elapsed = elapsed or 0.0
        error_type = error_type_from_exception(exc)
        error_message = " ".join(str(exc).split())
        log_lines.append(f"error_type={error_type}")
        log_lines.append(f"error_message={error_message}")
        log_lines.append("traceback:")
        log_lines.extend(traceback.format_exc().splitlines())
        try:
            max_allocated = torch.cuda.max_memory_allocated()
            max_reserved = torch.cuda.max_memory_reserved()
        except Exception:
            max_allocated = 0
            max_reserved = 0
    finally:
        stop.set()
        if "monitor" in locals():
            monitor.join(timeout=5)
        write_vram_csv(vram_path, samples)
        gc.collect()
        torch.cuda.empty_cache()

    if not use_cache and actual_bytes in (None, 0):
        actual_bytes = 0

    log_lines.extend(
        [
            f"run_end={now_iso()}",
            f"success={success}",
            f"theoretical_kv_bytes={theory}",
            f"actual_past_key_values_bytes={actual_bytes if actual_bytes is not None else ''}",
            f"torch_cuda_max_memory_allocated_bytes={max_allocated}",
            f"torch_cuda_max_memory_reserved_bytes={max_reserved}",
            f"elapsed_sec={elapsed:.6f}",
            f"vram_csv={vram_path.as_posix()}",
        ]
    )
    log_path.write_text("\n".join(log_lines) + "\n")

    return {
        "model": model_name,
        "gpu_name": gpu_name(),
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "dtype": DTYPE_NAME,
        "use_cache": str(use_cache),
        "theoretical_kv_bytes": theory,
        "theoretical_kv_mib": mib(theory),
        "actual_past_key_values_bytes": actual_bytes if actual_bytes is not None else "",
        "actual_past_key_values_mib": mib(actual_bytes),
        "theoretical_actual_ratio": ratio(theory, actual_bytes),
        "torch_cuda_max_memory_allocated_bytes": max_allocated,
        "torch_cuda_max_memory_allocated_mib": mib(max_allocated),
        "torch_cuda_max_memory_reserved_bytes": max_reserved,
        "torch_cuda_max_memory_reserved_mib": mib(max_reserved),
        "elapsed_sec": f"{elapsed:.6f}",
        "success": str(success),
        "error_type": error_type,
        "error_message": error_message[:500],
    }


def load_model(model_path: Path) -> tuple[Any, Any, Any]:
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.eval()
    return config, tokenizer, model


def write_csv(output_dir: Path, rows: list[dict[str, object]]) -> None:
    with (output_dir / "kv_cache_growth.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def bool_row(row: dict[str, object], key: str) -> bool:
    return str(row[key]).lower() == "true"


def as_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_logs_md(output_dir: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# KV Cache Growth Logs",
        "",
        "| Model | Batch | Seq len | use_cache | Success | Error | Raw log | VRAM CSV |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        model_name = str(row["model"])
        batch = row["batch_size"]
        seq = row["sequence_length"]
        use_cache = row["use_cache"]
        stem = f"{model_name}_bs{batch}_seq{seq}_cache{str(use_cache).lower()}"
        log = f"raw_logs/{stem}.log"
        vram = f"raw_logs/{stem}_vram.csv"
        lines.append(
            f"| {model_name} | {batch} | {seq} | {use_cache} | {row['success']} | "
            f"{row['error_type']} | [{log}]({log}) | [{vram}]({vram}) |"
        )
    (output_dir / "logs.md").write_text("\n".join(lines) + "\n")


def linearity_summary(rows: list[dict[str, object]], field: str, model_name: str, use_cache: bool) -> list[str]:
    subset = [
        row for row in rows
        if row["model"] == model_name and bool_row(row, "use_cache") == use_cache and row["success"] == "True"
    ]
    lines: list[str] = []
    for batch in sorted({int(row["batch_size"]) for row in subset}):
        points = sorted(
            (int(row["sequence_length"]), as_float(row, field))
            for row in subset
            if int(row["batch_size"]) == batch and as_float(row, field) is not None
        )
        points = [(seq, value) for seq, value in points if value is not None]
        if len(points) < 2:
            continue
        values = ", ".join(f"{seq}: {value:.2f} MiB" for seq, value in points)
        first_seq, first_value = points[0]
        last_seq, last_value = points[-1]
        expected = last_seq / first_seq
        observed = last_value / first_value if first_value else math.nan
        lines.append(
            f"- batch={batch}: {values}; {first_seq}->{last_seq} expected {expected:.2f}x, observed {observed:.2f}x."
        )
    return lines


def write_summary(output_dir: Path, rows: list[dict[str, object]], models: list[Path]) -> None:
    model_names = [model_label(path) for path in models]
    oom_rows = [row for row in rows if row["error_type"] == "CUDA_OOM"]
    success_rows = [row for row in rows if row["success"] == "True"]
    ratio_values = [
        as_float(row, "theoretical_actual_ratio")
        for row in success_rows
        if bool_row(row, "use_cache") and as_float(row, "theoretical_actual_ratio") is not None
    ]
    ratio_values = [value for value in ratio_values if value is not None]

    lines = [
        "# RTX 5060 FP16 KV-Cache Growth Baseline",
        "",
        "This experiment measures the baseline FP16 KV-cache memory growth that motivates Oaken-like KV-cache quantization.",
        "",
        "This is not an Oaken reproduction. This experiment isolates KV-cache memory pressure before evaluating Oaken's quantized path.",
        "",
        "## Setup",
        "",
        f"- GPU: {gpu_name()}",
        f"- Models: {', '.join(path.as_posix() for path in models)}",
        "- Inputs: random token IDs, not accuracy data.",
        "- Runtime: Hugging Face Transformers, PyTorch, CUDA fp16, `model.eval()`, `torch.no_grad()`.",
        "- Main sweep: sequence lengths 128, 256, 512, 1024, 2048 with batch sizes 1, 2, 4 and `use_cache=True`.",
        "- Extra comparison: `use_cache=False` for batch size 1 and sequence lengths 128, 512, 1024.",
        "",
        "## Headline Results",
        "",
    ]

    if ratio_values:
        min_ratio = min(ratio_values)
        max_ratio = max(ratio_values)
        lines.append(
            f"- Actual `past_key_values` size matches the theoretical formula closely: theoretical/actual ratio range `{min_ratio:.6f}` to `{max_ratio:.6f}` for successful `use_cache=True` runs."
        )
    else:
        lines.append("- No successful `use_cache=True` run produced a theoretical/actual ratio.")

    if oom_rows:
        lines.append("- CUDA OOM occurred at:")
        for row in oom_rows:
            lines.append(
                f"  - {row['model']} batch={row['batch_size']} seq={row['sequence_length']} use_cache={row['use_cache']}"
            )
    else:
        lines.append("- No CUDA OOM occurred in the completed sweep.")

    if success_rows:
        peak_row = max(success_rows, key=lambda row: int(row["torch_cuda_max_memory_reserved_bytes"]))
        lines.append(
            f"- Largest successful reserved-memory peak: `{peak_row['model']}` batch={peak_row['batch_size']} seq={peak_row['sequence_length']} use_cache={peak_row['use_cache']} at `{peak_row['torch_cuda_max_memory_reserved_mib']} MiB` reserved."
        )

    lines.extend(["", "## Theoretical vs Actual KV Cache", ""])
    lines.append("The formula used is:")
    lines.append("")
    lines.append("```text")
    lines.append("KV bytes = 2 * num_layers * batch_size * sequence_length * hidden_size * bytes_per_element")
    lines.append("```")
    lines.append("")
    lines.append("For successful `use_cache=True` rows, actual bytes are computed by summing `numel() * element_size()` for every key and value tensor in `output.past_key_values`.")
    lines.append("")

    lines.extend(["## Sequence Length Scaling", ""])
    for model_name in model_names:
        lines.append(f"### {model_name}")
        seq_lines = linearity_summary(rows, "actual_past_key_values_mib", model_name, True)
        lines.extend(seq_lines or ["- Not enough successful rows to summarize sequence-length scaling."])
        lines.append("")

    lines.extend(["## Batch Size Scaling", ""])
    for model_name in model_names:
        subset = [
            row for row in rows
            if row["model"] == model_name and bool_row(row, "use_cache") and row["success"] == "True"
        ]
        lines.append(f"### {model_name}")
        for seq in sorted({int(row["sequence_length"]) for row in subset}):
            points = sorted(
                (int(row["batch_size"]), as_float(row, "actual_past_key_values_mib"))
                for row in subset
                if int(row["sequence_length"]) == seq and as_float(row, "actual_past_key_values_mib") is not None
            )
            points = [(batch, value) for batch, value in points if value is not None]
            if len(points) < 2:
                continue
            values = ", ".join(f"batch={batch}: {value:.2f} MiB" for batch, value in points)
            first_batch, first_value = points[0]
            last_batch, last_value = points[-1]
            expected = last_batch / first_batch
            observed = last_value / first_value if first_value else math.nan
            lines.append(f"- seq={seq}: {values}; expected {expected:.2f}x, observed {observed:.2f}x.")
        lines.append("")

    lines.extend(["## use_cache=False Comparison", ""])
    for model_name in model_names:
        lines.append(f"### {model_name}")
        for seq in NO_CACHE_SEQ_LENGTHS:
            true_row = next(
                (
                    row for row in rows
                    if row["model"] == model_name
                    and int(row["batch_size"]) == 1
                    and int(row["sequence_length"]) == seq
                    and row["use_cache"] == "True"
                ),
                None,
            )
            false_row = next(
                (
                    row for row in rows
                    if row["model"] == model_name
                    and int(row["batch_size"]) == 1
                    and int(row["sequence_length"]) == seq
                    and row["use_cache"] == "False"
                ),
                None,
            )
            if true_row and false_row and true_row["success"] == "True" and false_row["success"] == "True":
                true_peak = as_float(true_row, "torch_cuda_max_memory_allocated_mib")
                false_peak = as_float(false_row, "torch_cuda_max_memory_allocated_mib")
                if true_peak is not None and false_peak is not None:
                    lines.append(
                        f"- seq={seq}: use_cache=True allocated peak `{true_peak:.2f} MiB`; use_cache=False allocated peak `{false_peak:.2f} MiB`; delta `{true_peak - false_peak:.2f} MiB`."
                    )
        lines.append("")

    lines.extend(
        [
            "## Why Torch Peak Memory Is Larger Than The KV Formula",
            "",
            "The theoretical KV formula covers only stored key/value tensors. Torch peak memory also includes FP16 model weights, input tensors, attention masks, intermediate activations that exist during the forward pass, logits/loss-related outputs, CUDA allocator fragmentation, reserved caching allocator blocks, and framework workspaces. Therefore peak allocated/reserved memory is expected to be larger than pure `past_key_values` storage.",
            "",
            "## RTX 5060 8GB Boundary",
            "",
        ]
    )
    if oom_rows:
        lines.append("The observed boundary is represented by the first CUDA OOM rows listed above. Earlier successful rows are the last feasible points in this sweep.")
    elif success_rows:
        peak_reserved = max(success_rows, key=lambda row: int(row["torch_cuda_max_memory_reserved_bytes"]))
        lines.append(
            f"No OOM occurred. The closest observed condition to the 8GB boundary was `{peak_reserved['model']}` batch={peak_reserved['batch_size']} seq={peak_reserved['sequence_length']} use_cache={peak_reserved['use_cache']}, with `{peak_reserved['torch_cuda_max_memory_reserved_mib']} MiB` reserved."
        )

    lines.extend(
        [
            "",
            "## Connection To Oaken",
            "",
            "This baseline shows the FP16 KV-cache term that Oaken-like KV-cache quantization targets. Because actual `past_key_values` bytes match the theoretical linear formula, reducing the bytes per KV element is a direct way to reduce the cache component. This experiment does not evaluate Oaken's quantized path; it establishes the baseline memory pressure before that comparison.",
            "",
            "## Limitations",
            "",
            "- Inputs are random token IDs, so this is a memory-growth test rather than an accuracy or perplexity test.",
            "- This is prefill-style full-context forward measurement, not an incremental decoding loop with one token appended at a time.",
            "- Peak torch memory includes more than KV cache, so the KV formula should be compared against actual `past_key_values` bytes, not total peak memory.",
            "- `nvidia-smi` VRAM sampling is best-effort and can miss short peaks; torch peak memory is the primary measurement.",
            "",
            "## Artifacts",
            "",
            "- `kv_cache_growth.csv`: machine-readable results.",
            "- `logs.md`: per-run raw log and VRAM CSV index.",
            "- `raw_logs/*.log`: per-run tensor shapes, byte counts, and errors.",
            "- `raw_logs/*_vram.csv`: best-effort `nvidia-smi` samples.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def sweep_model(model_path: Path, output_dir: Path) -> list[dict[str, object]]:
    name = model_label(model_path)
    print(f"Loading {model_path}", flush=True)
    config, tokenizer, model = load_model(model_path)
    del tokenizer

    rows: list[dict[str, object]] = []
    for batch_size in BATCH_SIZES:
        for sequence_length in SEQ_LENGTHS:
            print(f"run model={name} batch={batch_size} seq={sequence_length} use_cache=True", flush=True)
            row = run_case(model, config, name, batch_size, sequence_length, True, output_dir)
            rows.append(row)
            if row["error_type"] == "CUDA_OOM":
                print(f"CUDA OOM at model={name} batch={batch_size} seq={sequence_length}; continuing", flush=True)

    for sequence_length in NO_CACHE_SEQ_LENGTHS:
        print(f"run model={name} batch=1 seq={sequence_length} use_cache=False", flush=True)
        rows.append(run_case(model, config, name, 1, sequence_length, False, output_dir))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure FP16 KV-cache growth for local OPT models.")
    parser.add_argument("--models", nargs="+", required=True, type=Path, help="Local Hugging Face model directories.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for csv/md/log artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this experiment.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_logs").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for model_path in args.models:
        rows.extend(sweep_model(model_path, output_dir))
        write_csv(output_dir, rows)
        write_logs_md(output_dir, rows)
        write_summary(output_dir, rows, args.models[: len({row["model"] for row in rows})])

    write_csv(output_dir, rows)
    write_logs_md(output_dir, rows)
    write_summary(output_dir, rows, args.models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
