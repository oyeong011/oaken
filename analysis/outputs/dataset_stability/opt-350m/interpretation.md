# OPT-350M Oaken Dataset Stability Interpretation

## Dataset Outcomes

- Succeeded: `wikitext`, `winogrande`, `hellaswag`.
- Failed: `piqa`. The runner failed at dataset loading with `RuntimeError: Dataset scripts are no longer supported, but found piqa.py`.

Wikitext was reused as the baseline quantizer. Winogrande and Hellaswag produced new OPT-350M Oaken quantizers. The comparison table contains 432 rows: 3 datasets x 24 layers x 2 tensor types x 3 quantization groups.

## Stability Readout

The result is mixed. Wikitext, Winogrande, and Hellaswag preserve the same layer-wise/key-value/group structure, and their large-range groups are in the same broad magnitude regime. That supports Oaken's offline profiling assumption at a coarse level: a Wikitext profile is not structurally unrelated to the other task profiles.

The thresholds are not close enough to call directly interchangeable. Relative drift is large, especially for group 2, where Wikitext threshold widths are often near zero. Mean absolute `abs_max` drift is moderate for Winogrande and Hellaswag, but relative drift is high.

## Quantitative Evidence

Absolute drift from Wikitext:

| Dataset | Mean abs_max abs diff | Max abs_max abs diff | Mean width abs diff | Max width abs diff |
| --- | ---: | ---: | ---: | ---: |
| Winogrande | 0.143692 | 1.749381 | 0.290616 | 3.385388 |
| Hellaswag | 0.094686 | 1.125693 | 0.195060 | 2.282006 |

Relative drift from Wikitext:

| Dataset | Mean abs_max rel diff | Max abs_max rel diff | Mean width rel diff | Max width rel diff |
| --- | ---: | ---: | ---: | ---: |
| Winogrande | 100.0884% | 4572.1910% | 2277.6457% | 32748.5948% |
| Hellaswag | 72.3968% | 3880.6106% | 823.1002% | 7551.7720% |

The relative values overstate practical movement in some rows because several baseline widths are extremely small. For example, group 2 rows can have absolute width differences around `1e-5` to `1e-4` while reporting thousands of percent relative drift.

## Key vs Value Thresholds

Key thresholds vary more than value thresholds in absolute terms.

| Dataset | Tensor | Mean abs_max abs diff | Max abs_max abs diff | Mean width abs diff |
| --- | --- | ---: | ---: | ---: |
| Winogrande | key | 0.246542 | 1.749381 | 0.501467 |
| Winogrande | value | 0.040841 | 0.194689 | 0.079766 |
| Hellaswag | key | 0.163701 | 1.125693 | 0.341652 |
| Hellaswag | value | 0.025670 | 0.119397 | 0.048469 |

This suggests the cross-dataset instability is mainly in key projection activation ranges, especially the widest retained range group. Value thresholds are comparatively more stable in absolute magnitude, though their small group-2 ranges can still show large relative percentages.

## Large Deviations

The largest absolute deviations are concentrated in key tensor group 0:

| Dataset | Layer | Tensor | Group | abs_max abs diff | Width abs diff |
| --- | ---: | --- | ---: | ---: | ---: |
| Winogrande | 6 | key | 0 | 1.749381 | 3.385388 |
| Winogrande | 5 | key | 0 | 1.312190 | 2.885721 |
| Winogrande | 1 | key | 0 | 1.258083 | 2.557271 |
| Winogrande | 21 | key | 0 | 1.191081 | 2.262138 |
| Hellaswag | 6 | key | 0 | 1.125693 | 2.282006 |

The largest relative deviations are concentrated in group 2:

| Dataset | Layer | Tensor | Group | abs_max rel diff | Width rel diff |
| --- | ---: | --- | ---: | ---: | ---: |
| Winogrande | 8 | key | 2 | 4572.1910% | 6397.8558% |
| Hellaswag | 1 | value | 2 | 3880.6106% | 3316.1105% |
| Winogrande | 1 | value | 2 | 3323.2362% | 11214.2276% |
| Hellaswag | 8 | key | 2 | 1063.0390% | 2128.2633% |

These group-2 rows should be interpreted cautiously because their absolute thresholds and widths are tiny.

## Implication For Oaken Offline Profiling

The experiment supports a weaker version of Oaken's offline profiling claim: threshold profiles learned on Wikitext appear to capture reusable layer-wise structure for OPT-350M on RTX 5080, and value thresholds are fairly stable in absolute terms. This makes offline profiling plausible as a practical approximation.

It weakens a stronger claim that one dataset's thresholds are numerically stable enough to transfer without concern. Winogrande and Hellaswag both show meaningful absolute drift in key group 0, and very large relative drift in near-zero group-2 ranges. A Wikitext-derived quantizer may work as an approximation, but the data does not prove dataset-invariant thresholds.

## Limitations

- PIQA is missing because the current task stack rejects its dataset-script loader.
- This is OPT-350M only; it does not establish the same behavior for OPT-1.3B, OPT-2.7B, or the RTX 5080 OPT-6.7B boundary.
- The comparison measures threshold similarity only. It does not test downstream accuracy when applying one dataset's quantizer to another dataset.
- Relative drift can be misleading for near-zero baseline groups; absolute drift should be read alongside it.
- The result uses the current Oaken runner and local evaluation harness behavior, including task formatting and request construction.
