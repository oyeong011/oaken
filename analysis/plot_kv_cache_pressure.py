#!/usr/bin/env python3
"""Plot Oaken/original KV-cache pressure CSVs.

This is a lightweight plotting helper for
`analysis/outputs/kv_cache_pressure/<machine>/kv_cache_pressure.csv`.
It does not rerun benchmarks; it regenerates figures from existing CSV rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in ["sequence_length", "ppl", "peak_vram_mib", "elapsed_seconds"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def successful(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"].astype(str).str.upper() == "OK"].copy()


def line_plot(df: pd.DataFrame, y: str, ylabel: str, title: str, output: Path) -> None:
    if y not in df.columns or df[y].dropna().empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for (model, mode), group in df.groupby(["model", "mode"]):
        group = group.sort_values("sequence_length")
        ax.plot(group["sequence_length"], group[y], marker="o", label=f"{model} {mode}")
    ax.set_title(title)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = successful(numeric(pd.read_csv(args.input)))
    if df.empty:
        raise SystemExit("No status=OK rows to plot")

    line_plot(
        df,
        "peak_vram_mib",
        "Peak VRAM (MiB)",
        "KV-cache pressure: peak VRAM vs sequence length",
        args.output_dir / "peak_vram_vs_seq_len.png",
    )
    line_plot(
        df,
        "ppl",
        "Wikitext perplexity",
        "KV-cache pressure: perplexity vs sequence length",
        args.output_dir / "ppl_vs_seq_len.png",
    )
    elapsed_column = "elapsed_seconds" if "elapsed_seconds" in df.columns else "elapsed_sec"
    line_plot(
        df,
        elapsed_column,
        "Elapsed seconds",
        "KV-cache pressure: elapsed time vs sequence length",
        args.output_dir / "elapsed_vs_seq_len.png",
    )
    print(f"wrote KV-cache pressure plots to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
