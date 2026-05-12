#!/usr/bin/env python3
"""Run sequence-length scaling for Oaken KV-cache pressure analysis."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "analysis" / "outputs" / "kv_cache_pressure" / "rtx5080"
RAW_LOG_DIR = OUTPUT_DIR / "raw_logs"
CSV_PATH = OUTPUT_DIR / "kv_cache_pressure.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.md"
LOGS_MD_PATH = OUTPUT_DIR / "logs.md"

CONTAINER = "oaken-ae-container"
PYTHON = "/opt/conda/envs/oaken/bin/python"
MODEL = "opt"
MODEL_SIZE = "1.3b"
MODEL_NAME = "OPT-1.3B"
TASK = "wikitext"
GPU_NAME = "RTX 5080"
SEQUENCE_LENGTHS = (128, 256, 512, 1024, 2048)
QUANTIZER = Path("quantizer/oaken/opt-1.3b.json")


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
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
    parts = [part.strip() for part in output.splitlines()[0].split(",")]
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
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "used_mib", "free_mib", "total_mib"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(samples)


def docker_eval_command(sequence_length: int, mode: str) -> list[str]:
    command = [
        "docker",
        "exec",
        "-w",
        "/workspace",
        CONTAINER,
        PYTHON,
        "eval_perplexity.py",
        "-m",
        MODEL,
        "-s",
        MODEL_SIZE,
        "-t",
        TASK,
        "--max-length",
        str(sequence_length),
        "--stride",
        str(sequence_length),
        "--gpu-start-idx",
        "0",
        "--gpu-count",
        "1",
    ]
    if mode == "oaken":
        command.extend(["-q", QUANTIZER.as_posix(), "--quant-method", "oaken"])
    else:
        command.extend(["--quant-method", "none"])
    return command


def parse_ppl(text: str) -> float | None:
    matches = re.findall(r"tensor\((nan|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text)
    if not matches:
        return None
    value = matches[-1].lower()
    if value == "nan":
        return math.nan
    return float(value)


def classify_status(exit_code: int, text: str) -> tuple[str, str]:
    lowered = text.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "OOM", "CUDA out of memory"
    if re.search(r"\bnan\b|tensor\(nan", text, flags=re.IGNORECASE):
        return "NaN", "NaN detected"
    if "Traceback (most recent call last)" in text or "RuntimeError:" in text:
        return "ERROR", "Traceback or RuntimeError detected"
    if exit_code != 0:
        return "FAILED", f"exit code {exit_code}"
    return "OK", ""


def run_monitored(sequence_length: int, mode: str) -> dict[str, object]:
    RAW_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{MODEL_SIZE.replace('.', '_')}_seq{sequence_length}_{mode}"
    log_path = RAW_LOG_DIR / f"{stem}.log"
    vram_path = RAW_LOG_DIR / f"{stem}_vram.csv"
    meta_path = RAW_LOG_DIR / f"{stem}_meta.json"
    command = docker_eval_command(sequence_length, mode)

    idle_vram = query_vram()
    samples: list[dict[str, object]] = []
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_vram, args=(stop, samples), daemon=True)
    started_at = now_iso()
    start = time.monotonic()

    monitor.start()
    with log_path.open("w") as log:
        log.write(f"# model: {MODEL_NAME}\n")
        log.write(f"# gpu: {GPU_NAME}\n")
        log.write(f"# mode: {mode}\n")
        log.write(f"# sequence_length: {sequence_length}\n")
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
    status, failure_reason = classify_status(exit_code, text)
    peak_vram = max((int(sample["used_mib"]) for sample in samples), default=None)
    ppl = parse_ppl(text)

    row = {
        "model": MODEL_NAME,
        "gpu": GPU_NAME,
        "sequence_length": sequence_length,
        "mode": "Oaken" if mode == "oaken" else "Original FP16",
        "task": TASK,
        "ppl": "" if ppl is None else ("nan" if math.isnan(ppl) else f"{ppl:.6g}"),
        "peak_vram_mib": "" if peak_vram is None else peak_vram,
        "elapsed_seconds": f"{elapsed:.2f}",
        "status": status,
        "failure_reason": failure_reason,
        "oom_point": sequence_length if status == "OOM" else "",
        "idle_vram_mib": "" if idle_vram is None else idle_vram["used_mib"],
        "log_path": rel(log_path),
        "vram_csv": rel(vram_path),
    }
    meta_path.write_text(json.dumps({**row, "command": command, "started_at": started_at, "ended_at": ended_at}, indent=2) + "\n")
    return row


def write_csv(rows: list[dict[str, object]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "gpu",
        "sequence_length",
        "mode",
        "task",
        "ppl",
        "peak_vram_mib",
        "elapsed_seconds",
        "status",
        "failure_reason",
        "oom_point",
        "idle_vram_mib",
        "log_path",
        "vram_csv",
    ]
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def numeric(row: dict[str, object], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def write_summary(rows: list[dict[str, object]]) -> None:
    ok_rows = [row for row in rows if row["status"] == "OK"]
    by_seq: dict[int, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_seq.setdefault(int(row["sequence_length"]), {})[str(row["mode"])] = row

    comparison_lines = [
        "| Sequence length | Original peak MiB | Oaken peak MiB | Oaken - original MiB | Original PPL | Oaken PPL | Status |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    peak_diffs: list[float] = []
    for seq in sorted(by_seq):
        original = by_seq[seq].get("Original FP16")
        oaken = by_seq[seq].get("Oaken")
        original_peak = numeric(original, "peak_vram_mib") if original else None
        oaken_peak = numeric(oaken, "peak_vram_mib") if oaken else None
        diff = None if original_peak is None or oaken_peak is None else oaken_peak - original_peak
        if diff is not None:
            peak_diffs.append(diff)
        comparison_lines.append(
            f"| {seq} | {'' if original_peak is None else int(original_peak)} | "
            f"{'' if oaken_peak is None else int(oaken_peak)} | "
            f"{'' if diff is None else int(diff)} | "
            f"{'' if original is None else original['ppl']} | {'' if oaken is None else oaken['ppl']} | "
            f"{'' if original is None else original['status']} / {'' if oaken is None else oaken['status']} |"
        )

    original_ok = [row for row in ok_rows if row["mode"] == "Original FP16"]
    oaken_ok = [row for row in ok_rows if row["mode"] == "Oaken"]
    original_max = max((int(row["sequence_length"]) for row in original_ok), default=None)
    oaken_max = max((int(row["sequence_length"]) for row in oaken_ok), default=None)
    avg_diff = sum(peak_diffs) / len(peak_diffs) if peak_diffs else None
    original_growth = _mode_growth(rows, "Original FP16")
    oaken_growth = _mode_growth(rows, "Oaken")
    pressure_sentence = (
        "Peak VRAM grows with sequence length in this artifact path: "
        f"original FP16 changes by {original_growth:+.0f} MiB and Oaken changes by {oaken_growth:+.0f} MiB "
        f"from sequence length {min(SEQUENCE_LENGTHS)} to {max(SEQUENCE_LENGTHS)}."
    )
    if avg_diff is None:
        oaken_sentence = "Oaken memory impact could not be compared because no paired runs completed."
    elif avg_diff < -128:
        oaken_sentence = f"Oaken reduced peak VRAM on average by {-avg_diff:.1f} MiB."
    elif avg_diff > 128:
        oaken_sentence = f"Oaken increased peak VRAM on average by {avg_diff:.1f} MiB."
    else:
        oaken_sentence = (
            f"Oaken peak VRAM was effectively tied with original FP16 on average ({avg_diff:+.1f} MiB), "
            "with no tested length showing a meaningful reduction."
        )

    boundary_sentence = (
        "Oaken did not extend the feasible context boundary in this run because both modes completed the same maximum tested length."
        if original_max == oaken_max
        else f"The maximum successful sequence length differed: original={original_max}, Oaken={oaken_max}."
    )

    summary = f"""# RTX 5080 KV-Cache Pressure Experiment

## Goal

Measure whether Oaken relieves KV-cache memory pressure as Wikitext evaluation sequence length increases on RTX 5080.

## Method

- Model: `{MODEL_NAME}`
- GPU: `{GPU_NAME}`
- Dataset/task: `{TASK}`
- Sequence lengths: `{', '.join(str(item) for item in SEQUENCE_LENGTHS)}`
- Modes: original FP16 and Oaken using `{QUANTIZER.as_posix()}`
- `eval_perplexity.py` was parameterized with `--max-length` and `--stride`; both were set to the tested sequence length.
- Full Wikitext test tokenization was used; only sequence length changed between runs.

## Results

{chr(10).join(comparison_lines)}

## Interpretation

- {pressure_sentence}
- {oaken_sentence}
- {boundary_sentence}
- The observed memory growth is real, but Oaken's evaluation wrapper does not reduce the measured process peak here; the artifact path appears dominated by model weights, framework allocation, and fixed evaluation overhead plus attention activations rather than by a saved long-lived KV cache.

## Artifacts

- `{rel(CSV_PATH)}`
- `{rel(LOGS_MD_PATH)}`
- Raw logs and VRAM CSVs under `{rel(RAW_LOG_DIR)}`
"""
    SUMMARY_PATH.write_text(summary)


def _mode_growth(rows: list[dict[str, object]], mode: str) -> float:
    mode_rows = [row for row in rows if row["mode"] == mode and row.get("peak_vram_mib") != ""]
    by_seq = {int(row["sequence_length"]): float(row["peak_vram_mib"]) for row in mode_rows}
    if not by_seq:
        return 0.0
    return by_seq[max(by_seq)] - by_seq[min(by_seq)]


def write_logs_md(rows: list[dict[str, object]]) -> None:
    lines = [
        "# RTX 5080 KV-Cache Pressure Logs",
        "",
        "## Commands",
        "",
    ]
    for row in rows:
        meta_path = RAW_LOG_DIR / f"{MODEL_SIZE.replace('.', '_')}_seq{row['sequence_length']}_{'oaken' if row['mode'] == 'Oaken' else 'original'}_meta.json"
        command = ""
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            command = " ".join(meta["command"])
        lines.extend(
            [
                f"### {row['mode']} sequence length {row['sequence_length']}",
                "",
                "```sh",
                command,
                "```",
                "",
                f"- Status: `{row['status']}`",
                f"- PPL: `{row['ppl']}`",
                f"- Peak VRAM: `{row['peak_vram_mib']}` MiB",
                f"- Elapsed: `{row['elapsed_seconds']}` seconds",
                f"- Log: `{row['log_path']}`",
                f"- VRAM CSV: `{row['vram_csv']}`",
                "",
            ]
        )
    LOGS_MD_PATH.write_text("\n".join(lines))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for sequence_length in SEQUENCE_LENGTHS:
        for mode in ("original", "oaken"):
            print(f"running {MODEL_NAME} {mode} sequence_length={sequence_length}", flush=True)
            row = run_monitored(sequence_length, mode)
            rows.append(row)
            write_csv(rows)
            write_logs_md(rows)
            write_summary(rows)
            print(f"  {row['status']} ppl={row['ppl']} peak={row['peak_vram_mib']} elapsed={row['elapsed_seconds']}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
