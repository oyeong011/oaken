# RTX 5080 OPT-2.7B Oaken Result

## Result

| Metric | Value |
| --- | ---: |
| Original Wikitext PPL | 12.4688 |
| Oaken Wikitext PPL | 12.5703 |
| PPL delta | +0.1015 |
| Original eval wall time | 25s |
| Original eval peak VRAM | 7690 MiB |
| Profiling time | 60.50s |
| Profiling wall time | 64s |
| Profiling peak VRAM | 7830 MiB |
| Oaken eval wall time | 30s |
| Oaken eval peak VRAM | 7732 MiB |
| Idle VRAM after runs | 38 MiB |
| Status | OK |

## Artifacts

- `oaken-quantizer.json`: copied from `quantizer/oaken/opt-2.7b.json`.
- `original.log`: original FP16 Wikitext perplexity run.
- `profile.log`: Oaken offline profiling run.
- `eval.log`: Oaken Wikitext perplexity run.
- `original-vram.csv`, `profile-vram.csv`, `eval-vram.csv`: 1s `nvidia-smi` samples.
- `hardware.txt`: host and CUDA environment snapshot.

## Notes

Before each major CUDA run, the tiny CUDA allocation gate passed:

```text
True
NVIDIA GeForce RTX 5080
tensor([0.], device='cuda:0')
```

The OPT-2.7B model was prepared under `/data/models/opt-2.7b`; model weights and Hugging Face cache files are not part of this result artifact.
