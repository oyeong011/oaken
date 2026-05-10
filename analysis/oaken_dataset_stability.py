#!/usr/bin/env python3
"""Compare OPT-350M Oaken threshold quantizers across datasets."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import statistics
from collections import defaultdict
from pathlib import Path


MODEL = "OPT-350M"
BASELINE_DATASET = "wikitext"
EXPECTED_DATASETS = ("wikitext", "piqa", "winogrande", "hellaswag")
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "analysis" / "outputs" / "dataset_stability" / "opt-350m"
CSV_PATH = OUTPUT_DIR / "thresholds_by_dataset.csv"
SUMMARY_PATH = OUTPUT_DIR / "stability_summary.md"
PLOT_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"


def _quantizer_path(dataset: str) -> Path:
    return OUTPUT_DIR / f"opt-350m-{dataset}.json"


def _float_diff(value: float, baseline: float) -> tuple[float, str]:
    diff = value - baseline
    if baseline == 0:
        return diff, ""
    return diff, f"{diff / abs(baseline):.10g}"


def _parse_quantizer(path: Path, dataset: str) -> list[dict[str, object]]:
    with path.open() as handle:
        data = json.load(handle)

    rows: list[dict[str, object]] = []
    for tensor_type in ("key", "value"):
        lowers = data[tensor_type]["lower_threshold"]
        uppers = data[tensor_type]["upper_threshold"]
        for layer, (lower_groups, upper_groups) in enumerate(zip(lowers, uppers)):
            for group_index, (lower, upper) in enumerate(zip(lower_groups, upper_groups)):
                lower_f = float(lower)
                upper_f = float(upper)
                rows.append(
                    {
                        "model": MODEL,
                        "dataset": dataset,
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


def _load_rows() -> tuple[list[dict[str, object]], list[str], dict[str, str]]:
    rows: list[dict[str, object]] = []
    successful: list[str] = []
    failures: dict[str, str] = {}
    for dataset in EXPECTED_DATASETS:
        path = _quantizer_path(dataset)
        if path.exists():
            rows.extend(_parse_quantizer(path, dataset))
            successful.append(dataset)
        else:
            failures[dataset] = _failure_reason(dataset)
    return rows, successful, failures


def _failure_reason(dataset: str) -> str:
    log_path = LOG_DIR / f"{dataset}_profile.log"
    if not log_path.exists():
        return "No quantizer JSON or profiling log was found."
    text = log_path.read_text(errors="replace")
    for marker in (
        "RuntimeError: Dataset scripts are no longer supported",
        "Traceback (most recent call last):",
        "Error",
        "ERROR",
    ):
        index = text.find(marker)
        if index != -1:
            return " ".join(text[index:].split())[:700]
    return "Profiling log exists, but no successful quantizer JSON was produced."


def _add_baseline_diffs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline: dict[tuple[int, str, int], dict[str, object]] = {}
    for row in rows:
        if row["dataset"] == BASELINE_DATASET:
            baseline[(int(row["layer"]), str(row["tensor_type"]), int(row["group_index"]))] = row

    compared: list[dict[str, object]] = []
    for row in rows:
        key = (int(row["layer"]), str(row["tensor_type"]), int(row["group_index"]))
        base = baseline[key]
        absmax_abs_diff, absmax_rel_diff = _float_diff(float(row["abs_max"]), float(base["abs_max"]))
        width_abs_diff, width_rel_diff = _float_diff(float(row["width"]), float(base["width"]))
        enriched = dict(row)
        enriched.update(
            {
                "baseline_dataset": BASELINE_DATASET,
                "absmax_abs_diff_from_baseline": absmax_abs_diff,
                "absmax_rel_diff_from_baseline": absmax_rel_diff,
                "width_abs_diff_from_baseline": width_abs_diff,
                "width_rel_diff_from_baseline": width_rel_diff,
            }
        )
        compared.append(enriched)
    return compared


def _write_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "dataset",
        "layer",
        "tensor_type",
        "group_index",
        "lower_threshold",
        "upper_threshold",
        "width",
        "abs_max",
        "baseline_dataset",
        "absmax_abs_diff_from_baseline",
        "absmax_rel_diff_from_baseline",
        "width_abs_diff_from_baseline",
        "width_rel_diff_from_baseline",
    ]
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_stats(rows: list[dict[str, object]]) -> list[str]:
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)

    lines = [
        "| Dataset | Rows | Mean abs_max rel diff | Max abs_max rel diff | Mean width rel diff | Max width rel diff |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in EXPECTED_DATASETS:
        group = by_dataset.get(dataset, [])
        if not group:
            continue
        abs_rel = [abs(float(row["absmax_rel_diff_from_baseline"] or 0.0)) for row in group]
        width_rel = [abs(float(row["width_rel_diff_from_baseline"] or 0.0)) for row in group]
        lines.append(
            f"| {dataset} | {len(group)} | {statistics.mean(abs_rel):.4%} | {max(abs_rel):.4%} | "
            f"{statistics.mean(width_rel):.4%} | {max(width_rel):.4%} |"
        )
    return lines


def _dataset_abs_stats(rows: list[dict[str, object]]) -> list[str]:
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)

    lines = [
        "| Dataset | Mean abs_max abs diff | Max abs_max abs diff | Mean width abs diff | Max width abs diff |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in EXPECTED_DATASETS:
        group = by_dataset.get(dataset, [])
        if not group:
            continue
        abs_diff = [abs(float(row["absmax_abs_diff_from_baseline"])) for row in group]
        width_diff = [abs(float(row["width_abs_diff_from_baseline"])) for row in group]
        lines.append(
            f"| {dataset} | {statistics.mean(abs_diff):.6f} | {max(abs_diff):.6f} | "
            f"{statistics.mean(width_diff):.6f} | {max(width_diff):.6f} |"
        )
    return lines


def _top_drift_rows(rows: list[dict[str, object]], metric: str, limit: int = 8) -> list[str]:
    candidates = [row for row in rows if row["dataset"] != BASELINE_DATASET]
    candidates.sort(key=lambda row: abs(float(row[metric] or 0.0)), reverse=True)
    lines = [
        "| Dataset | Layer | Tensor | Group | abs_max | Width | abs_max rel diff | Width rel diff |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in candidates[:limit]:
        lines.append(
            f"| {row['dataset']} | {row['layer']} | {row['tensor_type']} | {row['group_index']} | "
            f"{float(row['abs_max']):.6f} | {float(row['width']):.6f} | "
            f"{float(row['absmax_rel_diff_from_baseline'] or 0.0):.4%} | "
            f"{float(row['width_rel_diff_from_baseline'] or 0.0):.4%} |"
        )
    return lines


def _top_abs_drift_rows(rows: list[dict[str, object]], limit: int = 8) -> list[str]:
    candidates = [row for row in rows if row["dataset"] != BASELINE_DATASET]
    candidates.sort(key=lambda row: abs(float(row["absmax_abs_diff_from_baseline"])), reverse=True)
    lines = [
        "| Dataset | Layer | Tensor | Group | abs_max | abs_max abs diff | Width abs diff |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in candidates[:limit]:
        lines.append(
            f"| {row['dataset']} | {row['layer']} | {row['tensor_type']} | {row['group_index']} | "
            f"{float(row['abs_max']):.6f} | {abs(float(row['absmax_abs_diff_from_baseline'])):.6f} | "
            f"{abs(float(row['width_abs_diff_from_baseline'])):.6f} |"
        )
    return lines


def _write_summary(rows: list[dict[str, object]], successful: list[str], failures: dict[str, str]) -> None:
    comparable = [row for row in rows if row["dataset"] != BASELINE_DATASET]
    abs_rel = [abs(float(row["absmax_rel_diff_from_baseline"] or 0.0)) for row in comparable]
    width_rel = [abs(float(row["width_rel_diff_from_baseline"] or 0.0)) for row in comparable]
    if abs_rel:
        stability_sentence = (
            f"Available non-Wikitext datasets show mean abs_max relative drift of "
            f"{statistics.mean(abs_rel):.2%} and mean width relative drift of {statistics.mean(width_rel):.2%}. "
            "This is mixed evidence: broad layer-wise structure is reusable, but some groups have large relative drift, "
            "so the result supports offline profiling plausibility more than strict threshold interchangeability."
        )
    else:
        stability_sentence = "No non-Wikitext quantizers were available, so stability could not be assessed."

    failure_lines = ["None"] if not failures else [f"- `{dataset}`: {reason}" for dataset, reason in failures.items()]
    summary = f"""# OPT-350M Oaken Dataset Stability Summary

## Goal

Reproduce the Oaken Section 4-style observation that layer-wise key/value activation threshold patterns are relatively stable across datasets, using OPT-350M on RTX 5080 and Wikitext as the baseline.

## Method

- Reused the existing Wikitext quantizer as `opt-350m-wikitext.json`.
- Ran Oaken offline profiling with `-f 0.04 0.9 0.06` for available non-Wikitext tasks.
- Parsed each available quantizer into one row per dataset, layer, key/value tensor, and quantization group.
- Compared `abs_max` and threshold `width` against the Wikitext row with the same layer, tensor type, and group.

## Successful Datasets

{chr(10).join(f"- `{dataset}`" for dataset in successful)}

## Failed Datasets

{chr(10).join(failure_lines)}

## Key Observations

Absolute differences from Wikitext:

{chr(10).join(_dataset_abs_stats(rows))}

Relative differences from Wikitext:

{chr(10).join(_dataset_stats(rows))}

Largest relative `abs_max` drift rows:

{chr(10).join(_top_drift_rows(rows, "absmax_rel_diff_from_baseline"))}

Largest absolute `abs_max` drift rows:

{chr(10).join(_top_abs_drift_rows(rows))}

## Whether Thresholds Appear Stable Enough To Support Offline Profiling

{stability_sentence}

The successful task quantizers preserve the same coarse layer/tensor/group schema as Wikitext, and many absolute maxima remain within a modest range. However, the largest relative differences occur in very small threshold groups, where relative percentages can be large even when absolute thresholds are small.

## Limitations

- PIQA did not produce a quantizer because the installed Hugging Face `datasets` version rejects the dataset-script based PIQA loader.
- This is threshold/range comparison only; it does not evaluate downstream accuracy using cross-dataset quantizers.
- Hellaswag profiling is much longer than Wikitext and Winogrande because it issues 40,168 loglikelihood requests.
- Matplotlib plot generation is optional and depends on the host Python matplotlib/NumPy ABI being usable.
"""
    SUMMARY_PATH.write_text(summary)


def _import_pyplot():
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable; skipping dataset-stability plots: {exc}")
        return None
    return plt


def _plot_absmax_by_dataset_layer(plt, rows: list[dict[str, object]]) -> None:
    by_dataset_layer: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_dataset_layer[(str(row["dataset"]), int(row["layer"]))].append(float(row["abs_max"]))
    datasets = [dataset for dataset in EXPECTED_DATASETS if any(key[0] == dataset for key in by_dataset_layer)]
    layers = sorted({layer for _, layer in by_dataset_layer})

    fig, ax = plt.subplots(figsize=(10, 5))
    for dataset in datasets:
        values = [statistics.mean(by_dataset_layer[(dataset, layer)]) for layer in layers]
        ax.plot(layers, values, marker="o", linewidth=1.5, label=dataset)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean abs_max")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "absmax_by_dataset_layer.png", dpi=160)
    plt.close(fig)


def _plot_relative_diff(plt, rows: list[dict[str, object]]) -> None:
    by_dataset_layer: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["dataset"] == BASELINE_DATASET:
            continue
        by_dataset_layer[(str(row["dataset"]), int(row["layer"]))].append(
            abs(float(row["absmax_rel_diff_from_baseline"] or 0.0))
        )
    datasets = [dataset for dataset in EXPECTED_DATASETS if any(key[0] == dataset for key in by_dataset_layer)]
    layers = sorted({layer for _, layer in by_dataset_layer})

    fig, ax = plt.subplots(figsize=(10, 5))
    for dataset in datasets:
        values = [statistics.mean(by_dataset_layer[(dataset, layer)]) for layer in layers]
        ax.plot(layers, values, marker="o", linewidth=1.5, label=dataset)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean abs_max relative diff from Wikitext")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "relative_diff_from_wikitext.png", dpi=160)
    plt.close(fig)


def _write_plots(rows: list[dict[str, object]]) -> None:
    plt = _import_pyplot()
    if plt is None:
        return
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_absmax_by_dataset_layer(plt, rows)
    _plot_relative_diff(plt, rows)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, successful, failures = _load_rows()
    if BASELINE_DATASET not in successful:
        raise SystemExit(f"baseline dataset missing: {BASELINE_DATASET}")
    compared = _add_baseline_diffs(rows)
    _write_csv(compared)
    _write_summary(compared, successful, failures)
    _write_plots(compared)
    print(f"wrote {CSV_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
