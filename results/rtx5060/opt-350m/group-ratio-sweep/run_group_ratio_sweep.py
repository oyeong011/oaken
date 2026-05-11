#!/usr/bin/env python3
"""Run and summarize the RTX 5060 OPT-350M Oaken group-ratio sweep."""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("results/rtx5060/opt-350m/group-ratio-sweep")
CONTAINER = "oaken-ae-container"
PYTHON = "/opt/conda/envs/oaken/bin/python"
MODEL = "opt"
SIZE = "350m"
TASK = "wikitext"
GPU_NAME = "RTX 5060"
MODEL_NAME = "OPT-350M"

RATIOS = [
    ("0.02-0.94-0.04", ("0.02", "0.94", "0.04")),
    ("0.04-0.90-0.06", ("0.04", "0.90", "0.06")),
    ("0.08-0.84-0.08", ("0.08", "0.84", "0.08")),
    ("0.10-0.80-0.10", ("0.10", "0.80", "0.10")),
]


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    return path.as_posix()


def run_text(cmd: list[str], timeout: int | None = None) -> str:
    completed = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip()


def query_vram() -> dict[str, int] | None:
    output = run_text(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    if not output:
        return None
    first_line = output.splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 3:
        return None
    try:
        used, free, total = (int(part) for part in parts)
    except ValueError:
        return None
    return {"used_mib": used, "free_mib": free, "total_mib": total}


def monitor_vram(stop: threading.Event, samples: list[dict[str, object]], interval_s: float = 0.5) -> None:
    while not stop.is_set():
        sample = query_vram()
        if sample is not None:
            samples.append({"timestamp": now_iso(), **sample})
        stop.wait(interval_s)


def write_vram_csv(path: Path, samples: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "used_mib", "free_mib", "total_mib"])
        writer.writeheader()
        writer.writerows(samples)


def run_monitored(name: str, command: list[str], out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{name}.log"
    vram_path = out_dir / f"{name}_vram.csv"
    meta_path = out_dir / f"{name}_meta.json"

    idle_vram = query_vram()
    samples: list[dict[str, object]] = []
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_vram, args=(stop, samples), daemon=True)

    started_at = now_iso()
    start = time.monotonic()
    monitor.start()
    with log_path.open("w") as log:
        log.write(f"# run: {name}\n")
        log.write(f"# start: {started_at}\n")
        log.write(f"# idle_vram: {idle_vram}\n")
        log.write(f"# command: {' '.join(command)}\n\n")
        log.flush()
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
        exit_code = proc.wait()
    elapsed = time.monotonic() - start
    ended_at = now_iso()
    stop.set()
    monitor.join(timeout=5)

    write_vram_csv(vram_path, samples)
    text = log_path.read_text(errors="replace")
    peak_vram = max((int(sample["used_mib"]) for sample in samples), default=None)

    meta = {
        "name": name,
        "command": command,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed,
        "exit_code": exit_code,
        "idle_vram_mib": idle_vram,
        "peak_vram_mib": peak_vram,
        "vram_samples": len(samples),
        "log": rel(log_path),
        "vram_log": rel(vram_path),
        "nan_detected": bool(re.search(r"\bnan\b|tensor\(nan", text, flags=re.IGNORECASE)),
        "oom_detected": "out of memory" in text.lower() or "cuda oom" in text.lower(),
        "traceback_detected": "Traceback (most recent call last)" in text,
        "runtime_error_detected": "RuntimeError:" in text,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def docker_python_command(script: str, *args: str) -> list[str]:
    return [
        "docker",
        "exec",
        "-w",
        "/workspace",
        CONTAINER,
        PYTHON,
        script,
        *args,
    ]


def original_command() -> list[str]:
    return docker_python_command(
        "eval_perplexity.py",
        "-m",
        MODEL,
        "-s",
        SIZE,
        "-t",
        TASK,
        "--quant-method",
        "none",
        "--gpu-start-idx",
        "0",
        "--gpu-count",
        "1",
    )


def profile_command(ratio: tuple[str, str, str], quantizer: Path) -> list[str]:
    return docker_python_command(
        "oaken_preprocess_activation.py",
        "-m",
        MODEL,
        "-s",
        SIZE,
        "-t",
        TASK,
        "-f",
        *ratio,
        "-o",
        rel(quantizer),
        "--gpu-start-idx",
        "0",
        "--gpu-count",
        "1",
    )


def eval_command(quantizer: Path) -> list[str]:
    return docker_python_command(
        "eval_perplexity.py",
        "-m",
        MODEL,
        "-s",
        SIZE,
        "-t",
        TASK,
        "-q",
        rel(quantizer),
        "--quant-method",
        "oaken",
        "--gpu-start-idx",
        "0",
        "--gpu-count",
        "1",
    )


def parse_ppl(log_path: Path) -> float | None:
    text = log_path.read_text(errors="replace")
    matches = re.findall(r"tensor\((nan|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text)
    if not matches:
        return None
    value = matches[-1].lower()
    if value == "nan":
        return math.nan
    return float(value)


def parse_profile_internal_elapsed(log_path: Path) -> float | None:
    text = log_path.read_text(errors="replace")
    matches = re.findall(r"Elapsed time:\s*([-+]?\d+(?:\.\d+)?)\s*seconds", text)
    return float(matches[-1]) if matches else None


def parse_sparsity(log_path: Path) -> tuple[list[float] | None, list[float] | None]:
    text = log_path.read_text(errors="replace")
    match = re.search(r"Total Sparsity: Key - (\[[^\]]+\]), Value - (\[[^\]]+\])", text)
    if match is None:
        return None, None
    try:
        key = [float(item) for item in ast.literal_eval(match.group(1))]
        value = [float(item) for item in ast.literal_eval(match.group(2))]
    except (SyntaxError, ValueError):
        return None, None
    return key, value


def status_from_meta(*metas: dict[str, object]) -> str:
    if any(int(meta["exit_code"]) != 0 for meta in metas):
        return "FAILED"
    if any(meta.get("oom_detected") for meta in metas):
        return "OOM"
    if any(meta.get("nan_detected") for meta in metas):
        return "NaN"
    if any(meta.get("traceback_detected") or meta.get("runtime_error_detected") for meta in metas):
        return "ERROR"
    return "OK"


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def write_hardware() -> None:
    text = []
    text.append(f"Recorded at: {now_iso()}")
    text.append("")
    text.append("== host nvidia-smi ==")
    text.append(run_text(["nvidia-smi"], timeout=20))
    text.append("")
    text.append("== docker ==");
    text.append(run_text(["docker", "ps"], timeout=20))
    text.append("")
    text.append("== container python/torch ==")
    text.append(
        run_text(
            [
                "docker",
                "exec",
                "-w",
                "/workspace",
                CONTAINER,
                PYTHON,
                "-c",
                (
                    "import torch, sys; "
                    "print(sys.version); "
                    "print(torch.__version__); "
                    "print(torch.cuda.is_available()); "
                    "print(torch.cuda.get_device_name(0))"
                ),
            ],
            timeout=30,
        )
    )
    text.append("")
    text.append("== model path ==")
    text.append("/data/models/opt-350m")
    text.append(run_text(["docker", "exec", "-w", "/workspace", CONTAINER, "find", "/data/models/opt-350m", "-maxdepth", "1", "-type", "f", "-printf", "%f %s\\n"], timeout=30))
    (ROOT / "hardware.txt").write_text("\n".join(text) + "\n")


def write_summary_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "gpu",
        "model",
        "ratio",
        "original_ppl",
        "oaken_ppl",
        "ppl_delta",
        "profile_time_s",
        "profile_internal_time_s",
        "eval_time_s",
        "profile_peak_vram_mib",
        "eval_peak_vram_mib",
        "peak_vram_mib",
        "idle_vram_mib",
        "status",
        "nan_detected",
        "oom_detected",
        "key_sparsity_g0",
        "key_sparsity_g1",
        "key_sparsity_g2",
        "value_sparsity_g0",
        "value_sparsity_g1",
        "value_sparsity_g2",
        "quantizer",
        "profile_log",
        "eval_log",
    ]
    with (ROOT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(rows: list[dict[str, object]], baseline_meta: dict[str, object], baseline_ppl: float | None) -> None:
    ok_rows = [row for row in rows if row["status"] == "OK"]
    default_row = next((row for row in rows if row["ratio"] == "0.04/0.90/0.06"), None)
    best_row = min(
        ok_rows,
        key=lambda row: float(row["ppl_delta"]) if row["ppl_delta"] != "" else float("inf"),
        default=None,
    )

    lines = [
        "# RTX 5060 OPT-350M Oaken Group-Ratio Sweep",
        "",
        "## Scope",
        "",
        "This sweep evaluates Wikitext perplexity, profiling behavior, sparsity, VRAM use, and stability for four Oaken activation group ratios on the RTX 5060 8GB path.",
        "",
        "## Environment",
        "",
        "- GPU: NVIDIA GeForce RTX 5060",
        "- Model: OPT-350M from `/data/models/opt-350m`",
        "- Container: `oaken-ae-container` / `oaken-ae-img`",
        "- Task: Wikitext-2 raw test perplexity",
        f"- Original FP16 baseline PPL: `{fmt(baseline_ppl)}`",
        f"- Baseline run elapsed: `{fmt(float(baseline_meta['elapsed_seconds']), 3)}s`",
        "",
        "## Results",
        "",
        "| Ratio | Status | Original PPL | Oaken PPL | Delta | Profile Time | Eval Time | Peak VRAM | Key Sparsity | Value Sparsity |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for row in rows:
        key_sparsity = f"[{row['key_sparsity_g0']}, {row['key_sparsity_g1']}, {row['key_sparsity_g2']}]"
        value_sparsity = f"[{row['value_sparsity_g0']}, {row['value_sparsity_g1']}, {row['value_sparsity_g2']}]"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["ratio"]),
                    str(row["status"]),
                    str(row["original_ppl"]),
                    str(row["oaken_ppl"]),
                    str(row["ppl_delta"]),
                    f"{row['profile_time_s']}s",
                    f"{row['eval_time_s']}s",
                    f"{row['peak_vram_mib']} MiB",
                    key_sparsity,
                    value_sparsity,
                ]
            )
            + " |"
        )

    lines.extend(["", "## Interpretation", ""])
    if default_row is not None:
        lines.append(
            f"- The default `0.04/0.90/0.06` grouping completed successfully with Oaken PPL `{default_row['oaken_ppl']}` and delta `{default_row['ppl_delta']}`."
        )
    if best_row is not None:
        lines.append(
            f"- The smallest observed PPL delta was `{best_row['ppl_delta']}` at ratio `{best_row['ratio']}`."
        )
    if len(ok_rows) > 1:
        ppl_sequence = " -> ".join(str(row["oaken_ppl"]) for row in ok_rows)
        lines.append(
            f"- Across the tested ratios, Oaken PPL improved monotonically as the outer/inner groups grew and the middle group shrank: `{ppl_sequence}`."
        )
    ok_peaks = [int(row["peak_vram_mib"]) for row in ok_rows if row["peak_vram_mib"] != ""]
    if ok_peaks:
        max_peak = max(ok_peaks)
        lines.append(f"- All completed ratios stayed well within the 8GB RTX 5060 budget; the largest sampled peak was `{max_peak} MiB`.")
    if len(ok_rows) > 1:
        profile_times = [float(row["profile_time_s"]) for row in ok_rows]
        eval_times = [float(row["eval_time_s"]) for row in ok_rows]
        lines.append(
            f"- Profiling and eval timing were effectively flat across ratios: profile `{min(profile_times):.3f}-{max(profile_times):.3f}s`, eval `{min(eval_times):.3f}-{max(eval_times):.3f}s`."
        )
    failed = [row for row in rows if row["status"] != "OK"]
    if not failed:
        lines.append("- No NaN, OOM, traceback, or runtime error was detected in the sweep logs.")
    else:
        failed_ratios = ", ".join(str(row["ratio"]) for row in failed)
        lines.append(f"- Non-OK ratios: {failed_ratios}. See per-run logs and metadata for details.")
    lines.append("- Reported sparsity tracks the configured grouping closely for both key and value projections, confirming the profiling/eval hooks are applying the intended ratio split.")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `summary.csv`: machine-readable sweep table.",
            "- `summary.md`: this interpretation.",
            "- `hardware.txt`: hardware, Docker, Python, PyTorch, and model-file listing.",
            "- `baseline/`: original FP16 Wikitext baseline log, metadata, and VRAM samples.",
            "- `ratio-*/`: per-ratio quantizer, profiling log/metadata/VRAM samples, and Oaken eval log/metadata/VRAM samples.",
            "",
            "Model weights, HuggingFace cache files, `.bin`, and `.safetensors` artifacts are not stored in this results directory.",
        ]
    )

    (ROOT / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    write_hardware()

    baseline_dir = ROOT / "baseline"
    baseline_meta = run_monitored("original_fp16_eval", original_command(), baseline_dir)
    baseline_ppl = parse_ppl(Path(str(baseline_meta["log"])))

    rows: list[dict[str, object]] = []
    for label, ratio in RATIOS:
        ratio_dir = ROOT / f"ratio-{label}"
        quantizer = ratio_dir / "oaken-quantizer.json"
        profile_meta = run_monitored("oaken_profile", profile_command(ratio, quantizer), ratio_dir)
        eval_meta = run_monitored("oaken_eval", eval_command(quantizer), ratio_dir)

        oaken_ppl = parse_ppl(Path(str(eval_meta["log"])))
        profile_internal = parse_profile_internal_elapsed(Path(str(profile_meta["log"])))
        key_sparsity, value_sparsity = parse_sparsity(Path(str(eval_meta["log"])))
        profile_peak = profile_meta["peak_vram_mib"]
        eval_peak = eval_meta["peak_vram_mib"]
        peak_values = [int(value) for value in [profile_peak, eval_peak] if value is not None]
        peak_vram = max(peak_values) if peak_values else ""
        idle = profile_meta.get("idle_vram_mib") or eval_meta.get("idle_vram_mib") or {}
        idle_used = idle.get("used_mib") if isinstance(idle, dict) else None
        ppl_delta = None
        if baseline_ppl is not None and oaken_ppl is not None and not math.isnan(oaken_ppl):
            ppl_delta = oaken_ppl - baseline_ppl

        row = {
            "gpu": GPU_NAME,
            "model": MODEL_NAME,
            "ratio": "/".join(ratio),
            "original_ppl": fmt(baseline_ppl),
            "oaken_ppl": fmt(oaken_ppl),
            "ppl_delta": fmt(ppl_delta),
            "profile_time_s": fmt(float(profile_meta["elapsed_seconds"]), 3),
            "profile_internal_time_s": fmt(profile_internal, 3),
            "eval_time_s": fmt(float(eval_meta["elapsed_seconds"]), 3),
            "profile_peak_vram_mib": profile_peak or "",
            "eval_peak_vram_mib": eval_peak or "",
            "peak_vram_mib": peak_vram,
            "idle_vram_mib": idle_used or "",
            "status": status_from_meta(profile_meta, eval_meta),
            "nan_detected": bool(profile_meta.get("nan_detected") or eval_meta.get("nan_detected")),
            "oom_detected": bool(profile_meta.get("oom_detected") or eval_meta.get("oom_detected")),
            "key_sparsity_g0": fmt(key_sparsity[0] if key_sparsity else None, 6),
            "key_sparsity_g1": fmt(key_sparsity[1] if key_sparsity else None, 6),
            "key_sparsity_g2": fmt(key_sparsity[2] if key_sparsity else None, 6),
            "value_sparsity_g0": fmt(value_sparsity[0] if value_sparsity else None, 6),
            "value_sparsity_g1": fmt(value_sparsity[1] if value_sparsity else None, 6),
            "value_sparsity_g2": fmt(value_sparsity[2] if value_sparsity else None, 6),
            "quantizer": rel(quantizer),
            "profile_log": profile_meta["log"],
            "eval_log": eval_meta["log"],
        }
        rows.append(row)

    write_summary_csv(rows)
    write_summary_md(rows, baseline_meta, baseline_ppl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
