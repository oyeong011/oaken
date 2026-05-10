# Oaken Threshold Stability Plan

## Goal

Measure whether Oaken threshold profiles are stable across Wikitext, PIQA, Winogrande, and Hellaswag for the same model, GPU, sparsity fractions, and software environment.

## Fixed Conditions

- Use one model size per comparison round before scaling up, starting with OPT-350M or OPT-1.3B.
- Keep GPU, driver, container image, PyTorch, Transformers, and Oaken commit fixed.
- Keep Oaken sparsity fractions fixed at the same key/value group split used in the current reproduction: `0.04 0.9 0.06`.
- Run profiling only after an idle VRAM check and a tiny CUDA allocation gate.
- Store each dataset-specific quantizer as `analysis/threshold_runs/<gpu>/<model>/<dataset>/oaken-quantizer.json`.

## Datasets

| Dataset | Purpose in stability check | Planned artifact |
| --- | --- | --- |
| Wikitext | Current baseline threshold source and perplexity path | `wikitext/oaken-quantizer.json` |
| PIQA | Commonsense physical reasoning activation profile | `piqa/oaken-quantizer.json` |
| Winogrande | Pronoun/coreference-style commonsense profile | `winogrande/oaken-quantizer.json` |
| Hellaswag | Activity continuation and broader context profile | `hellaswag/oaken-quantizer.json` |

## Metrics

- Per-layer `abs_max` for key and value thresholds.
- Per-layer threshold `width = upper_threshold - lower_threshold`.
- Per-group drift from Wikitext: absolute difference and relative percentage difference.
- Rank correlation of layer-level `abs_max` between datasets.
- Key-vs-value separation by layer and group.

## Comparison Procedure

1. Generate one quantizer per dataset with the same model/GPU/configuration.
2. Convert all quantizers to the long CSV schema used by `analysis/oaken_threshold_analysis.py`.
3. Join rows by `model`, `gpu`, `layer`, `tensor_type`, and `group_index`.
4. Use Wikitext as the initial baseline and compute dataset deltas for PIQA, Winogrande, and Hellaswag.
5. Summarize drift by dataset, tensor type, layer band, and quantization group.
6. Flag layers where any dataset exceeds the chosen drift threshold.

## Initial Drift Gates

- Low drift: mean relative `abs_max` delta below 5%.
- Moderate drift: mean relative `abs_max` delta from 5% to 15%.
- High drift: mean relative `abs_max` delta above 15% or repeated layer-level outliers.

These gates are starting points for observation, not pass/fail claims. They should be tightened only after at least one complete four-dataset sweep.

## Reporting Shape

The final stability report should include:

- A dataset-by-dataset threshold summary table.
- Per-layer key/value `abs_max` plots.
- Per-layer width plots.
- Dataset delta heatmaps by layer and tensor type.
- A short note on whether Wikitext thresholds appear transferable to PIQA, Winogrande, and Hellaswag for the tested model/GPU pair.

## Risks

- Dataset-specific prompt formatting or evaluation harness changes can affect activation distributions independently of task semantics.
- Larger OPT models may expose VRAM limits before all four dataset profiles are collected.
- Matplotlib is currently unavailable in the host Python environment because the installed build fails against NumPy 2.2.6; scripts should continue to emit CSV and Markdown without plots.
