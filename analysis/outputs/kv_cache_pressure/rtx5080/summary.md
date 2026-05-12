# RTX 5080 KV-Cache Pressure Experiment

## Goal

Measure whether Oaken relieves KV-cache memory pressure as Wikitext evaluation sequence length increases on RTX 5080.

## Method

- Model: `OPT-1.3B`
- GPU: `RTX 5080`
- Dataset/task: `wikitext`
- Sequence lengths: `128, 256, 512, 1024, 2048`
- Modes: original FP16 and Oaken using `quantizer/oaken/opt-1.3b.json`
- `eval_perplexity.py` was parameterized with `--max-length` and `--stride`; both were set to the tested sequence length.
- Full Wikitext test tokenization was used; only sequence length changed between runs.

## Results

| Sequence length | Original peak MiB | Oaken peak MiB | Oaken - original MiB | Original PPL | Oaken PPL | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 3421 | 3427 | 6 | 35.375 | 36.6562 | OK / OK |
| 256 | 3471 | 3477 | 6 | 25.8906 | 26.9219 | OK / OK |
| 512 | 3369 | 3373 | 4 | 20.2031 | 21.0938 | OK / OK |
| 1024 | 3733 | 3749 | 16 | 16.7812 | 17.5938 | OK / OK |
| 2048 | 4673 | 4689 | 16 | 14.6406 | 15.4297 | OK / OK |

## Interpretation

- Peak VRAM grows with sequence length in this artifact path: original FP16 changes by +1252 MiB and Oaken changes by +1262 MiB from sequence length 128 to 2048.
- Oaken peak VRAM was effectively tied with original FP16 on average (+9.6 MiB), with no tested length showing a meaningful reduction.
- Oaken did not extend the feasible context boundary in this run because both modes completed the same maximum tested length.
- The observed memory growth is real, but Oaken's evaluation wrapper does not reduce the measured process peak here; the artifact path appears dominated by model weights, framework allocation, and fixed evaluation overhead plus attention activations rather than by a saved long-lived KV cache.

## Artifacts

- `analysis/outputs/kv_cache_pressure/rtx5080/kv_cache_pressure.csv`
- `analysis/outputs/kv_cache_pressure/rtx5080/logs.md`
- Raw logs and VRAM CSVs under `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs`
