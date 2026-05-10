# OPT-350M Oaken Dataset Stability Summary

## Goal

Reproduce the Oaken Section 4-style observation that layer-wise key/value activation threshold patterns are relatively stable across datasets, using OPT-350M on RTX 5080 and Wikitext as the baseline.

## Method

- Reused the existing Wikitext quantizer as `opt-350m-wikitext.json`.
- Ran Oaken offline profiling with `-f 0.04 0.9 0.06` for available non-Wikitext tasks.
- Parsed each available quantizer into one row per dataset, layer, key/value tensor, and quantization group.
- Compared `abs_max` and threshold `width` against the Wikitext row with the same layer, tensor type, and group.

## Successful Datasets

- `wikitext`
- `winogrande`
- `hellaswag`

## Failed Datasets

- `piqa`: RuntimeError: Dataset scripts are no longer supported, but found piqa.py

## Key Observations

Absolute differences from Wikitext:

| Dataset | Mean abs_max abs diff | Max abs_max abs diff | Mean width abs diff | Max width abs diff |
| --- | ---: | ---: | ---: | ---: |
| wikitext | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| winogrande | 0.143692 | 1.749381 | 0.290616 | 3.385388 |
| hellaswag | 0.094686 | 1.125693 | 0.195060 | 2.282006 |

Relative differences from Wikitext:

| Dataset | Rows | Mean abs_max rel diff | Max abs_max rel diff | Mean width rel diff | Max width rel diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| wikitext | 144 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| winogrande | 144 | 100.0884% | 4572.1910% | 2277.6457% | 32748.5948% |
| hellaswag | 144 | 72.3968% | 3880.6106% | 823.1002% | 7551.7720% |

Largest relative `abs_max` drift rows:

| Dataset | Layer | Tensor | Group | abs_max | Width | abs_max rel diff | Width rel diff |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| winogrande | 8 | key | 2 | 0.004857 | 0.000199 | 4572.1910% | 6397.8558% |
| hellaswag | 1 | value | 2 | 0.002433 | 0.000013 | 3880.6106% | 3316.1105% |
| winogrande | 1 | value | 2 | 0.002092 | 0.000044 | 3323.2362% | 11214.2276% |
| hellaswag | 8 | key | 2 | 0.001209 | 0.000068 | 1063.0390% | 2128.2633% |
| winogrande | 10 | value | 2 | 0.001737 | 0.000021 | 743.9324% | 2702.6327% |
| hellaswag | 18 | key | 2 | 0.059259 | 0.000050 | 693.5323% | 3010.7235% |
| winogrande | 18 | key | 2 | 0.049572 | 0.000136 | 563.8162% | 8353.4142% |
| hellaswag | 7 | key | 2 | 0.049932 | 0.000062 | 412.1379% | 2104.0916% |

Largest absolute `abs_max` drift rows:

| Dataset | Layer | Tensor | Group | abs_max | abs_max abs diff | Width abs diff |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| winogrande | 6 | key | 0 | 4.678977 | 1.749381 | 3.385388 |
| winogrande | 5 | key | 0 | 3.919941 | 1.312190 | 2.885721 |
| winogrande | 1 | key | 0 | 4.736653 | 1.258083 | 2.557271 |
| winogrande | 21 | key | 0 | 3.583465 | 1.191081 | 2.262138 |
| winogrande | 20 | key | 0 | 3.642438 | 1.138424 | 2.299193 |
| hellaswag | 6 | key | 0 | 5.302665 | 1.125693 | 2.282006 |
| winogrande | 4 | key | 0 | 4.255476 | 1.056026 | 2.390978 |
| winogrande | 8 | key | 0 | 4.633096 | 1.033598 | 2.039066 |

## Whether Thresholds Appear Stable Enough To Support Offline Profiling

Available non-Wikitext datasets show mean abs_max relative drift of 86.24% and mean width relative drift of 1550.37%. This is mixed evidence: broad layer-wise structure is reusable, but some groups have large relative drift, so the result supports offline profiling plausibility more than strict threshold interchangeability.

The successful task quantizers preserve the same coarse layer/tensor/group schema as Wikitext, and many absolute maxima remain within a modest range. However, the largest relative differences occur in very small threshold groups, where relative percentages can be large even when absolute thresholds are small.

## Limitations

- PIQA did not produce a quantizer because the installed Hugging Face `datasets` version rejects the dataset-script based PIQA loader.
- This is threshold/range comparison only; it does not evaluate downstream accuracy using cross-dataset quantizers.
- Hellaswag profiling is much longer than Wikitext and Winogrande because it issues 40,168 loglikelihood requests.
- Matplotlib plot generation is optional and depends on the host Python matplotlib/NumPy ABI being usable.
