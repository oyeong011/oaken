#!/usr/bin/env python3
"""Create plots and boundary summaries from KV-cache sweep CSV output."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def save_peak_vs_seq(ok: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(9, 6))
    for (cache_mode, batch_size), group in ok.groupby(["cache_mode", "batch_size"]):
        group = group.sort_values("seq_len")
        plt.plot(
            group["seq_len"].to_numpy(),
            group["peak_memory_mb"].to_numpy(),
            marker="o",
            label=f"{cache_mode}, B={batch_size}",
        )
    plt.xlabel("Sequence length")
    plt.ylabel("Peak CUDA allocated memory (MiB)")
    plt.title("Peak memory vs sequence length")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "peak_memory_vs_seq_len.png", dpi=180)
    plt.close()


def save_throughput_vs_memory(ok: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(8, 6))
    for cache_mode, group in ok.groupby("cache_mode"):
        plt.scatter(
            group["peak_memory_mb"].to_numpy(),
            group["tokens_per_sec"].to_numpy(),
            s=45,
            alpha=0.8,
            label=cache_mode,
        )
    plt.xlabel("Peak CUDA allocated memory (MiB)")
    plt.ylabel("Generated tokens / second")
    plt.title("Throughput vs peak memory")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "throughput_vs_peak_memory.png", dpi=180)
    plt.close()


def save_boundary_matrix(df: pd.DataFrame, outdir: Path) -> None:
    matrix = df.pivot_table(
        index=["batch_size", "seq_len"],
        columns="cache_mode",
        values="status",
        aggfunc="first",
    ).reset_index()
    matrix.to_csv(outdir / "status_boundary_matrix.csv", index=False)

    oom_rows = df[df["status"].astype(str).str.lower().eq("oom")].copy()
    oom_rows.to_csv(outdir / "oom_cases.csv", index=False)

    if "dynamic" in matrix.columns:
        mode_columns = [column for column in matrix.columns if column not in {"batch_size", "seq_len"}]
        rescued = matrix[matrix["dynamic"].astype(str).str.lower().eq("oom")].copy()
        for column in mode_columns:
            rescued[f"{column}_rescues_dynamic"] = rescued[column].astype(str).str.lower().eq("ok")
        rescued.to_csv(outdir / "dynamic_oom_rescue_cases.csv", index=False)


def save_summary(ok: pd.DataFrame, outdir: Path) -> None:
    summary = ok.groupby("cache_mode", as_index=False).agg(
        rows=("status", "count"),
        mean_tokens_per_sec=("tokens_per_sec", "mean"),
        median_latency_ms=("latency_ms", "median"),
        max_peak_memory_mb=("peak_memory_mb", "max"),
        mean_kv_actual_over_theory=("kv_actual_over_theory", "mean"),
        mean_non_kv_overhead_mb=("non_kv_overhead_mb", "mean"),
    )
    summary.to_csv(outdir / "summary_by_cache_mode.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", "--input", dest="csv", required=True)
    parser.add_argument("--outdir", "--output-dir", dest="outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = numeric(
        df,
        [
            "batch_size",
            "seq_len",
            "peak_memory_mb",
            "tokens_per_sec",
            "latency_ms",
            "kv_actual_over_theory",
            "non_kv_overhead_mb",
        ],
    )
    save_boundary_matrix(df, outdir)

    ok = df[df["status"].astype(str).str.lower().eq("ok")].copy()
    if ok.empty:
        raise SystemExit("No status=ok rows to plot; wrote boundary CSVs only.")

    save_peak_vs_seq(ok, outdir)
    save_throughput_vs_memory(ok, outdir)
    save_summary(ok, outdir)
    print(f"Wrote plots and summaries to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
