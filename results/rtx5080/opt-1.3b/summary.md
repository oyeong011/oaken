# RTX 5080 OPT-1.3B Oaken Result

## Result

| Metric | Value |
| --- | ---: |
| Original Wikitext PPL | 14.6406 |
| Oaken Wikitext PPL | 15.3984 |
| PPL delta | +0.7578 |
| Original eval wall time | 18s |
| Original eval peak VRAM | 4580 MiB |
| Profiling time | 34.80s |
| Profiling wall time | 38s |
| Profiling peak VRAM | 4668 MiB |
| Oaken eval wall time | 20s |
| Oaken eval peak VRAM | 4612 MiB |
| Idle VRAM after runs | 38 MiB |
| Status | OK |

## Artifacts

- `oaken-quantizer.json`: copied from `quantizer/oaken/opt-1.3b.json`.
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

The OPT-1.3B model was prepared under `/data/models/opt-1.3b`; model weights and Hugging Face cache files are not part of this result artifact.
