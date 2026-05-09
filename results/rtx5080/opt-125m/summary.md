# RTX 5080 OPT-125M Oaken Smoke Result

## Result

| Metric | Value |
| --- | ---: |
| Original Wikitext PPL | 27.6719 |
| Oaken Wikitext PPL | 28.0 |
| PPL delta | +0.3281 |
| Profiling time | 16.69s initial, 10.69s retest |
| Profiling peak VRAM | 1584 MiB |
| Eval wall time | 10s |
| Eval peak VRAM | 1542 MiB |
| Status | OK |

## Artifacts

- `oaken-quantizer.json`: copied from `quantizer/oaken/opt-125m.json`.
- `profile-vram.csv`: 1s `nvidia-smi` samples during profiling retest.
- `eval-vram.csv`: 1s `nvidia-smi` samples during Oaken eval retest.
- `hardware.txt`: host and CUDA environment snapshot.

## Notes

Before the profiling and eval runs, the tiny CUDA allocation gate passed:

```text
True
NVIDIA GeForce RTX 5080
tensor([0.], device='cuda:0')
```
