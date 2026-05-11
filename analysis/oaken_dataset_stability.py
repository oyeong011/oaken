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
from xml.sax.saxutils import escape


MODEL = "OPT-350M"
BASELINE_DATASET = "wikitext"
EXPECTED_DATASETS = ("wikitext", "piqa", "winogrande", "hellaswag")
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "analysis" / "outputs" / "dataset_stability" / "opt-350m"
CSV_PATH = OUTPUT_DIR / "thresholds_by_dataset.csv"
SUMMARY_PATH = OUTPUT_DIR / "stability_summary.md"
PLOT_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"
SVG_WIDTH = 1200
SVG_HEIGHT = 900
PANEL_HEIGHT = 320
PANEL_GAP = 48
LEFT_MARGIN = 86
RIGHT_MARGIN = 28
TOP_MARGIN = 56
BOTTOM_MARGIN = 68
LEGEND_Y = 26
COLORS = {
    "wikitext": "#2f6fdb",
    "winogrande": "#d94841",
    "hellaswag": "#2ca25f",
}


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


def _series_by_dataset_layer(rows: list[dict[str, object]], metric: str, absolute: bool = False) -> dict[str, list[float | None]]:
    by_dataset_layer: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["dataset"]), int(row["layer"]))
        value = float(row[metric] or 0.0)
        if absolute:
            value = abs(value)
        by_dataset_layer[key].append(value)

    series: dict[str, list[float | None]] = {}
    layers = sorted({int(row["layer"]) for row in rows})
    for dataset in EXPECTED_DATASETS:
        if not any(key[0] == dataset for key in by_dataset_layer):
            continue
        values: list[float | None] = []
        for layer in layers:
            group = by_dataset_layer.get((dataset, layer))
            values.append(statistics.mean(group) if group else None)
        series[dataset] = values
    return series


def _panel_series(rows: list[dict[str, object]], metric: str, absolute: bool = False) -> tuple[list[int], dict[str, dict[str, list[float | None]]]]:
    layers = sorted({int(row["layer"]) for row in rows})
    panels: dict[str, dict[str, list[float | None]]] = {}
    for tensor_type in ("key", "value"):
        panel_rows = [row for row in rows if str(row["tensor_type"]) == tensor_type]
        panels[tensor_type] = _series_by_dataset_layer(panel_rows, metric, absolute=absolute)
    return layers, panels


def _svg_path_commands(points: list[tuple[float, float]]) -> str:
    return " ".join((f"M {points[0][0]:.2f} {points[0][1]:.2f} " + " ".join(f"L {x:.2f} {y:.2f}" for x, y in points[1:])) if points else "")


def _series_max(series: dict[str, list[float | None]]) -> float:
    values = [value for series_values in series.values() for value in series_values if value is not None]
    return max(values) if values else 0.0


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f"<title>{escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def _svg_text(x: float, y: float, text: str, *, size: int = 14, anchor: str = "start", weight: str = "normal", fill: str = "#1f2937", rotate: float | None = None) -> str:
    transform = f' transform="rotate({rotate:.2f} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{transform}>'
        f"{escape(text)}</text>"
    )


def _render_panel(
    lines: list[str],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    y_label: str,
    layers: list[int],
    series: dict[str, list[float | None]],
    y_max: float,
    show_zero_line: bool = False,
    dashed_zero: bool = False,
) -> None:
    left = x + LEFT_MARGIN
    top = y + TOP_MARGIN
    chart_w = width - LEFT_MARGIN - RIGHT_MARGIN
    chart_h = height - TOP_MARGIN - BOTTOM_MARGIN
    lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="#fafafa" stroke="#e5e7eb"/>')
    lines.append(_svg_text(x + 12, y + 24, title, size=18, weight="bold"))
    lines.append(_svg_text(x + 12, y + 44, y_label, size=12, fill="#4b5563"))
    if show_zero_line:
        zero_y = top + chart_h
        dash_attr = ' stroke-dasharray="5 4"' if dashed_zero else ""
        lines.append(
            f'<line x1="{left:.2f}" y1="{zero_y:.2f}" x2="{left + chart_w:.2f}" y2="{zero_y:.2f}" stroke="#9ca3af" '
            f'stroke-width="1.2"{dash_attr}/>'
        )

    step_x = chart_w / max(len(layers) - 1, 1)
    for idx, layer in enumerate(layers):
        px = left + idx * step_x
        if idx % 3 == 0 or idx == len(layers) - 1:
            lines.append(f'<line x1="{px:.2f}" y1="{top + chart_h:.2f}" x2="{px:.2f}" y2="{top + chart_h + 6:.2f}" stroke="#6b7280" stroke-width="1"/>')
            lines.append(_svg_text(px, top + chart_h + 22, str(layer), size=10, anchor="middle", fill="#6b7280"))

    for tick in range(5):
        value = y_max * tick / 4 if y_max else 0.0
        py = top + chart_h - (value / y_max * chart_h if y_max else 0.0)
        lines.append(f'<line x1="{left - 5:.2f}" y1="{py:.2f}" x2="{left:.2f}" y2="{py:.2f}" stroke="#6b7280" stroke-width="1"/>')
        lines.append(f'<line x1="{left:.2f}" y1="{py:.2f}" x2="{left + chart_w:.2f}" y2="{py:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(_svg_text(left - 10, py + 4, f"{value:.2f}", size=10, anchor="end", fill="#6b7280"))

    for idx, (name, values) in enumerate(series.items()):
        color = COLORS.get(name, "#6b7280")
        points: list[tuple[float, float]] = []
        for layer_idx, value in enumerate(values):
            if value is None or y_max == 0:
                continue
            px = left + layer_idx * step_x
            py = top + chart_h - (min(value, y_max) / y_max * chart_h)
            points.append((px, py))
        if not points:
            continue
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" '
            f'stroke-linecap="round" points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in points)}"/>'
        )
        for px, py in points:
            lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.8" fill="{color}"/>')

    legend_x = x + width - 200
    for idx, name in enumerate(series):
        ly = y + LEGEND_Y + idx * 20
        color = COLORS.get(name, "#6b7280")
        lines.append(f'<rect x="{legend_x:.2f}" y="{ly - 10:.2f}" width="12" height="12" fill="{color}"/>')
        lines.append(_svg_text(legend_x + 18, ly, name, size=12, fill="#374151"))


def _write_svg_file(path: Path, title: str, panels: list[tuple[str, str, dict[str, list[float | None]], float]], layers: list[int]) -> None:
    width = SVG_WIDTH
    height = 90 + len(panels) * (PANEL_HEIGHT + PANEL_GAP) - PANEL_GAP
    lines = _svg_header(width, height, title)
    lines.append(_svg_text(24, 34, title, size=22, weight="bold"))
    for idx, (panel_title, y_label, series, y_max) in enumerate(panels):
        panel_y = 60 + idx * (PANEL_HEIGHT + PANEL_GAP)
        _render_panel(
            lines,
            x=24,
            y=panel_y,
            width=width - 48,
            height=PANEL_HEIGHT,
            title=panel_title,
            y_label=y_label,
            layers=layers,
            series=series,
            y_max=y_max,
            show_zero_line="relative" in title.lower(),
            dashed_zero=True,
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def _write_svg_plots(rows: list[dict[str, object]]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    layers, abs_panels = _panel_series(rows, "abs_max", absolute=False)
    _, rel_panels = _panel_series(rows, "absmax_rel_diff_from_baseline", absolute=True)

    abs_series = [
        ("key", "Mean abs_max", abs_panels["key"], _series_max(abs_panels["key"])),
        ("value", "Mean abs_max", abs_panels["value"], _series_max(abs_panels["value"])),
    ]
    rel_series = [
        ("key", "Mean abs_max relative diff from Wikitext", rel_panels["key"], _series_max(rel_panels["key"])),
        ("value", "Mean abs_max relative diff from Wikitext", rel_panels["value"], _series_max(rel_panels["value"])),
    ]
    _write_svg_file(PLOT_DIR / "absmax_by_dataset_layer.svg", "OPT-350M abs_max by dataset and layer", abs_series, layers)
    _write_svg_file(PLOT_DIR / "relative_diff_from_wikitext.svg", "OPT-350M relative difference from Wikitext", rel_series, layers)


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
        _write_svg_plots(rows)
        return
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_absmax_by_dataset_layer(plt, rows)
    _plot_relative_diff(plt, rows)
    _write_svg_plots(rows)


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
