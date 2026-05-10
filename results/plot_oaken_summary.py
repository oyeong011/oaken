#!/usr/bin/env python3
"""Generate simple plots from oaken_consumer_gpu_summary.csv when matplotlib works."""

from __future__ import annotations

import csv
import contextlib
import io
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "oaken_consumer_gpu_summary.csv"
PLOT_DIR = ROOT / "plots"


def _float_or_none(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _import_pyplot():
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib may be installed but ABI-incompatible.
        print(f"matplotlib unavailable; skipping plot generation: {exc}")
        return None
    return plt


def _bar_plot(plt, rows: list[dict[str, str]], column: str, title: str, ylabel: str, output: Path) -> None:
    filtered = [(f"{row['gpu']}\n{row['model']}", _float_or_none(row[column])) for row in rows]
    filtered = [(label, value) for label, value in filtered if value is not None]
    labels = [label for label, _ in filtered]
    values = [value for _, value in filtered]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#3b6ea8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"wrote {output}")


def main() -> int:
    rows = _load_rows()
    plt = _import_pyplot()
    if plt is None:
        return 0

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _bar_plot(plt, rows, "ppl_delta", "Oaken PPL Delta vs Original FP16", "PPL delta", PLOT_DIR / "ppl_delta.png")
    _bar_plot(plt, rows, "peak_vram_gb", "Peak VRAM During Oaken Path", "Peak VRAM (GB decimal)", PLOT_DIR / "peak_vram.png")
    _bar_plot(plt, rows, "profiling_time_s", "Oaken Profiling Time", "Profiling time (s)", PLOT_DIR / "profiling_time.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
