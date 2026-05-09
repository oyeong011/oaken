# RTX 5080 OPT-350M Oaken Smoke Result

## Result

| Metric | Value |
| --- | ---: |
| Original Wikitext PPL | 22.0156 |
| Oaken Wikitext PPL | 22.1406 |
| PPL delta | +0.1250 |
| Profiling time | 16.54s |
| Profiling peak VRAM | 2112 MiB |
| Eval wall time | 13s |
| Eval peak VRAM | 2080 MiB |
| Status | OK |

## Artifacts

- `oaken-quantizer.json`: copied from `quantizer/oaken/opt-350m.json`.
- `profile-vram.csv`: 1s `nvidia-smi` samples during profiling.
- `eval-vram.csv`: 1s `nvidia-smi` samples during Oaken eval.
- `hardware.txt`: host and CUDA environment snapshot.

## Compatibility Fixes Required

- `src/util.py`: OPT-350M includes `model.decoder.project_in` and `model.decoder.project_out`; these modules must be included in the device map.
- `src/oaken/quantize.py`: zero-range quantization rows must preserve the original tensor values to avoid `nan` propagation in Oaken evaluation.

Before profiling and eval, the tiny CUDA allocation gate passed:

```text
True
NVIDIA GeForce RTX 5080
tensor([0.], device='cuda:0')
```
