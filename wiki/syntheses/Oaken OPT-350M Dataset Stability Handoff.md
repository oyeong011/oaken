---
type: synthesis
status: active
created: 2026-05-11
updated: 2026-05-11
tags: [oaken, opt-350m, dataset-stability, offline-profiling, consumer-gpu]
sources:
  - analysis/outputs/dataset_stability/opt-350m/stability_summary.md
  - analysis/outputs/dataset_stability/opt-350m/interpretation.md
  - analysis/outputs/dataset_stability/opt-350m/overview_ko.md
  - analysis/outputs/dataset_stability/opt-350m/thresholds_by_dataset.csv
  - results/oaken_consumer_gpu_summary.md
  - results/oaken_consumer_gpu_summary.csv
---

# Oaken OPT-350M Dataset Stability Handoff

## Summary

This work checked the Oaken Section 4-style assumption that KV threshold profiles are reasonably stable across datasets, using OPT-350M on RTX 5080.

It also updated the consumer GPU summary to include the RTX 5060 OPT-2.7B final result from commit `404d9fbc`, and refreshed the derived threshold summary artifacts.

## What was done

- Reused the Wikitext quantizer as the baseline.
- Ran offline profiling for `winogrande` and `hellaswag` with the `0.04 0.9 0.06` group ratio.
- Recorded the `piqa` failure caused by the current datasets loader rejecting dataset scripts.
- Produced `thresholds_by_dataset.csv` and `stability_summary.md`.
- Added `interpretation.md` for a concise evidence-based reading.
- Added `overview_ko.md` for a plain-language Korean explanation of the experiment.
- Added SVG plot fallbacks:
  - `analysis/outputs/dataset_stability/opt-350m/plots/absmax_by_dataset_layer.svg`
  - `analysis/outputs/dataset_stability/opt-350m/plots/relative_diff_from_wikitext.svg`

## Key result

The coarse layer/tensor/group structure is shared across datasets, which supports Oaken's offline profiling idea as a practical approximation.

The thresholds are not identical enough to call dataset-invariant. Key group 0 shows the largest absolute drift, and group 2 shows the largest relative drift because baseline widths are very small.

## Consumer GPU summary update

The RTX 5060 OPT-2.7B result is now included in `results/oaken_consumer_gpu_summary.csv` and `results/oaken_consumer_gpu_summary.md`.

That result says OPT-2.7B remains viable on an 8GB-class RTX 5060 for the Oaken Wikitext path, but the memory headroom is tight.

## Open points

- PIQA remains blocked by the dataset-script loader incompatibility.
- The current stability result is OPT-350M only.
- The comparison measures threshold similarity, not cross-dataset downstream accuracy.
- If the environment later gets a working matplotlib stack, the SVG plots can be mirrored to PNGs without changing the underlying analysis.
