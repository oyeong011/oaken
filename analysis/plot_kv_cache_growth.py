#!/usr/bin/env python3
"""Plot the pure FP16 KV-cache growth baseline from CSV.

The baseline CSV is the source of truth.  This script only regenerates PNG
figures from `analysis/outputs/kv_cache_growth/<machine>/kv_cache_growth.csv`.
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
    for column in [
        "batch_size",
        "sequence_length",
        "theoretical_kv_mib",
        "actual_past_key_values_mib",
        "theoretical_actual_ratio",
        "torch_cuda_max_memory_allocated_mib",
        "torch_cuda_max_memory_reserved_mib",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def successful_cache_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["success"].astype(str).str.lower() == "true") & (df["use_cache"].astype(str) == "True")].copy()


def plot_kv_vs_seq(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for (model, batch), group in df.groupby(["model", "batch_size"]):
        group = group.sort_values("sequence_length")
        label = f"{model}, bs={int(batch)}"
        ax.plot(group["sequence_length"], group["theoretical_kv_mib"], marker="o", label=f"theory {label}")
        ax.plot(
            group["sequence_length"],
            group["actual_past_key_values_mib"],
            marker="x",
            linestyle="--",
            label=f"actual {label}",
        )
    ax.set_title("Pure KV-cache growth: theoretical vs actual")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("KV-cache size (MiB)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_actual_vs_theory(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for model, group in df.groupby("model"):
        ax.scatter(group["theoretical_kv_mib"], group["actual_past_key_values_mib"], label=model, alpha=0.8)
    max_value = max(df["theoretical_kv_mib"].max(), df["actual_past_key_values_mib"].max())
    ax.plot([0, max_value], [0, max_value], color="black", linestyle="--", linewidth=1, label="y=x")
    ax.set_title("Actual `past_key_values` bytes vs KV formula")
    ax.set_xlabel("Theoretical KV-cache (MiB)")
    ax.set_ylabel("Actual past_key_values (MiB)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_peak_memory(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for (model, batch), group in df.groupby(["model", "batch_size"]):
        group = group.sort_values("sequence_length")
        ax.plot(
            group["sequence_length"],
            group["torch_cuda_max_memory_allocated_mib"],
            marker="o",
            label=f"{model}, bs={int(batch)}",
        )
    ax.set_title("Pure KV-cache growth: peak CUDA allocated")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("torch.cuda.max_memory_allocated (MiB)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = successful_cache_rows(numeric(pd.read_csv(args.input)))
    if df.empty:
        raise SystemExit("No successful use_cache=True rows to plot")

    plot_kv_vs_seq(df, args.output_dir / "kv_theory_actual_vs_seq_len.png")
    plot_actual_vs_theory(df, args.output_dir / "actual_vs_theoretical_kv.png")
    plot_peak_memory(df, args.output_dir / "peak_cuda_allocated_vs_seq_len.png")
    print(f"wrote KV-cache growth plots to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
