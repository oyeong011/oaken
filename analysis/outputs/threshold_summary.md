# Oaken Threshold Observation Summary

This scaffold parses successful Wikitext Oaken quantizer artifacts from the current consumer GPU runs and emits a long-form threshold table for Section 4 observation work.

## Sources

- `results/rtx5060/opt-1.3b/oaken-quantizer.json`
- `results/rtx5060/opt-350m/oaken-quantizer.json`
- `results/rtx5080/opt-1.3b/oaken-quantizer.json`
- `results/rtx5080/opt-125m/oaken-quantizer.json`
- `results/rtx5080/opt-2.7b/oaken-quantizer.json`
- `results/rtx5080/opt-350m/oaken-quantizer.json`

## Output Files

- `analysis/outputs/thresholds_long.csv`: one row per model, GPU, layer, key/value tensor type, and quantization group.
- `analysis/outputs/plots/`: optional plots when matplotlib imports successfully.

## Coverage

- Rows: 840
- Layers observed: 0 through 31
- Quantization groups: 0, 1, 2
- Tensor types: key, value

## Aggregate Threshold Statistics

| GPU | Model | Tensor | Rows | Mean width | Max abs threshold |
| --- | --- | --- | ---: | ---: | ---: |
| RTX 5060 | OPT-1.3B | key | 72 | 4.163223 | 14.975122 |
| RTX 5060 | OPT-1.3B | value | 72 | 0.906980 | 2.688844 |
| RTX 5060 | OPT-350M | key | 72 | 2.989748 | 6.428524 |
| RTX 5060 | OPT-350M | value | 72 | 1.047489 | 1.846271 |
| RTX 5080 | OPT-1.3B | key | 72 | 4.163242 | 14.975122 |
| RTX 5080 | OPT-1.3B | value | 72 | 0.906984 | 2.688802 |
| RTX 5080 | OPT-125M | key | 36 | 2.551920 | 4.544326 |
| RTX 5080 | OPT-125M | value | 36 | 0.368890 | 0.823952 |
| RTX 5080 | OPT-2.7B | key | 96 | 4.530290 | 12.027427 |
| RTX 5080 | OPT-2.7B | value | 96 | 0.906375 | 2.753823 |
| RTX 5080 | OPT-350M | key | 72 | 2.989750 | 6.428358 |
| RTX 5080 | OPT-350M | value | 72 | 1.047490 | 1.846257 |

## Interpretation Notes

- `width` is `upper_threshold - lower_threshold`; narrow groups indicate tighter retained activation ranges.
- `abs_max` is the larger absolute bound and is useful for comparing key/value magnitude behavior by layer.
- RTX 5080 OPT-6.7B is excluded from this successful-run table because Oaken evaluation reached CUDA OOM even though profiling produced a quantizer.
- Cross-dataset stability is not measured yet; see `analysis/dataset_stability_plan.md` for the comparison scaffold.
