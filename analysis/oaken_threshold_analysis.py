#!/usr/bin/env python3
"""Summarize Oaken key/value threshold JSON files into CSV and Markdown."""

from __future__ import annotations

import csv
import contextlib
import io
import json
import statistics
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
LONG_CSV = OUTPUT_ROOT / "thresholds_long.csv"
SUMMARY_MD = OUTPUT_ROOT / "threshold_summary.md"
SUCCESSFUL_MODELS = {
    ("rtx5060", "opt-350m"),
    ("rtx5060", "opt-1.3b"),
    ("rtx5060", "opt-2.7b"),
    ("rtx5080", "opt-125m"),
    ("rtx5080", "opt-350m"),
    ("rtx5080", "opt-1.3b"),
    ("rtx5080", "opt-2.7b"),
}


def _display_gpu(directory_name: str) -> str:
    return directory_name.replace("rtx", "RTX ")


def _parse_quantizer(path: Path) -> list[dict[str, object]]:
    model = path.parent.name.replace("opt-", "OPT-").upper()
    gpu_dir = path.parent.parent.name
    gpu = _display_gpu(gpu_dir)
    with path.open() as handle:
        data = json.load(handle)

    rows: list[dict[str, object]] = []
    for tensor_type in ("key", "value"):
        lower_by_layer = data[tensor_type]["lower_threshold"]
        upper_by_layer = data[tensor_type]["upper_threshold"]
        for layer, (lower_groups, upper_groups) in enumerate(zip(lower_by_layer, upper_by_layer)):
            for group_index, (lower, upper) in enumerate(zip(lower_groups, upper_groups)):
                lower_f = float(lower)
                upper_f = float(upper)
                rows.append(
                    {
                        "model": model,
                        "gpu": gpu,
                        "layer": layer,
                        "tensor_type": tensor_type,
                        "group_index": group_index,
                        "lower_threshold": lower_f,
                        "upper_threshold": upper_f,
                        "width": upper_f - lower_f,
                        "abs_max": max(abs(lower_f), abs(upper_f)),
                    }
                )
    return rows


def _iter_successful_quantizers() -> list[Path]:
    paths: list[Path] = []
    for gpu_dir, model_dir in sorted(SUCCESSFUL_MODELS):
        path = RESULTS_ROOT / gpu_dir / model_dir / "oaken-quantizer.json"
        if path.exists():
            paths.append(path)
    return paths


def _write_csv(rows: list[dict[str, object]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "gpu",
        "layer",
        "tensor_type",
        "group_index",
        "lower_threshold",
        "upper_threshold",
        "width",
        "abs_max",
    ]
    with LONG_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summary_table(rows: list[dict[str, object]]) -> str:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["gpu"]), str(row["model"]), str(row["tensor_type"]))].append(row)

    lines = [
        "| GPU | Model | Tensor | Rows | Mean width | Max abs threshold |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for (gpu, model, tensor_type), group_rows in sorted(grouped.items()):
        widths = [float(row["width"]) for row in group_rows]
        abs_maxes = [float(row["abs_max"]) for row in group_rows]
        lines.append(
            f"| {gpu} | {model} | {tensor_type} | {len(group_rows)} | "
            f"{statistics.mean(widths):.6f} | {max(abs_maxes):.6f} |"
        )
    return "\n".join(lines)


def _write_summary(rows: list[dict[str, object]], source_paths: list[Path]) -> None:
    layers = sorted({int(row["layer"]) for row in rows})
    groups = sorted({int(row["group_index"]) for row in rows})
    summary = f"""# Oaken Threshold Observation Summary

This scaffold parses successful Wikitext Oaken quantizer artifacts from the current consumer GPU runs and emits a long-form threshold table for Section 4 observation work.

## Sources

{chr(10).join(f"- `{path.relative_to(REPO_ROOT)}`" for path in source_paths)}

## Output Files

- `{LONG_CSV.relative_to(REPO_ROOT)}`: one row per model, GPU, layer, key/value tensor type, and quantization group.
- `analysis/outputs/plots/`: optional plots when matplotlib imports successfully.

## Coverage

- Rows: {len(rows)}
- Layers observed: {layers[0]} through {layers[-1]}
- Quantization groups: {", ".join(str(group) for group in groups)}
- Tensor types: key, value

## Aggregate Threshold Statistics

{_summary_table(rows)}

## Interpretation Notes

- `width` is `upper_threshold - lower_threshold`; narrow groups indicate tighter retained activation ranges.
- `abs_max` is the larger absolute bound and is useful for comparing key/value magnitude behavior by layer.
- RTX 5080 OPT-6.7B is excluded from this successful-run table because Oaken evaluation reached CUDA OOM even though profiling produced a quantizer.
- Cross-dataset stability is not measured yet; see `analysis/dataset_stability_plan.md` for the comparison scaffold.
"""
    SUMMARY_MD.write_text(summary)


def _import_pyplot():
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable; skipping threshold plots: {exc}")
        return None
    return plt


def _plot_by_layer(plt, rows: list[dict[str, object]], metric: str, output: Path, ylabel: str) -> None:
    by_layer: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer"])].append(float(row[metric]))
    layers = sorted(by_layer)
    values = [statistics.mean(by_layer[layer]) for layer in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, values, marker="o", linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"wrote {output}")


def _plot_key_vs_value_absmax(plt, rows: list[dict[str, object]], output: Path) -> None:
    by_layer_tensor: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        by_layer_tensor[(int(row["layer"]), str(row["tensor_type"]))].append(float(row["abs_max"]))
    layers = sorted({layer for layer, _ in by_layer_tensor})
    key_values = [statistics.mean(by_layer_tensor[(layer, "key")]) for layer in layers]
    value_values = [statistics.mean(by_layer_tensor[(layer, "value")]) for layer in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, key_values, marker="o", label="key")
    ax.plot(layers, value_values, marker="s", label="value")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean abs_max")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"wrote {output}")


def _write_plots(rows: list[dict[str, object]]) -> None:
    plt = _import_pyplot()
    if plt is None:
        return
    plot_dir = OUTPUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _plot_by_layer(plt, rows, "abs_max", plot_dir / "threshold_absmax_by_layer.png", "Mean abs_max")
    _plot_by_layer(plt, rows, "width", plot_dir / "threshold_width_by_layer.png", "Mean threshold width")
    _plot_key_vs_value_absmax(plt, rows, plot_dir / "key_vs_value_absmax.png")


def main() -> int:
    source_paths = _iter_successful_quantizers()
    rows: list[dict[str, object]] = []
    for path in source_paths:
        rows.extend(_parse_quantizer(path))
    if not rows:
        raise SystemExit("no successful oaken-quantizer.json files found")

    _write_csv(rows)
    _write_summary(rows, source_paths)
    _write_plots(rows)
    print(f"wrote {LONG_CSV}")
    print(f"wrote {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
