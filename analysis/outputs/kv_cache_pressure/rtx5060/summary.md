# RTX 5060 KV-Cache Pressure Sequence-Length Scaling

## Scope

This experiment evaluates Wikitext perplexity while scaling the perplexity window length. It compares original FP16 OPT evaluation against the Oaken quantized path using the existing Wikitext Oaken quantizer when available.

All runs keep the dataset, model, quantizer, GPU, task, GPU count, and stride policy fixed. For each sequence length, `stride == sequence_length`.

## Result Table

| Model | Seq len | Mode | Status | PPL | Peak VRAM | Elapsed |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| OPT-350M | 128 | original_fp16 | OK | 53.7500 | 909 MiB | 22.916s |
| OPT-350M | 128 | oaken | OK | 53.9688 | 909 MiB | 54.929s |
| OPT-350M | 256 | original_fp16 | OK | 39.8750 | 967 MiB | 15.850s |
| OPT-350M | 256 | oaken | OK | 40.0312 | 973 MiB | 31.992s |
| OPT-350M | 512 | original_fp16 | OK | 31.1094 | 1111 MiB | 14.849s |
| OPT-350M | 512 | oaken | OK | 31.2344 | 1115 MiB | 23.378s |
| OPT-350M | 1024 | original_fp16 | OK | 25.5469 | 1359 MiB | 14.345s |
| OPT-350M | 1024 | oaken | OK | 25.6875 | 1363 MiB | 20.030s |
| OPT-350M | 2048 | original_fp16 | OK | 22.0156 | 1879 MiB | 14.681s |
| OPT-350M | 2048 | oaken | OK | 22.1875 | 1915 MiB | 19.434s |
| OPT-1.3B | 128 | original_fp16 | OK | 35.3750 | 3151 MiB | 35.118s |
| OPT-1.3B | 128 | oaken | OK | 36.6562 | 3157 MiB | 64.301s |
| OPT-1.3B | 256 | original_fp16 | OK | 25.8906 | 3201 MiB | 32.098s |
| OPT-1.3B | 256 | oaken | OK | 26.9219 | 3207 MiB | 47.732s |
| OPT-1.3B | 512 | original_fp16 | OK | 20.2031 | 3099 MiB | 30.717s |
| OPT-1.3B | 512 | oaken | OK | 21.0938 | 3103 MiB | 41.422s |
| OPT-1.3B | 1024 | original_fp16 | OK | 16.7812 | 3463 MiB | 30.772s |
| OPT-1.3B | 1024 | oaken | OK | 17.5469 | 3479 MiB | 39.827s |
| OPT-1.3B | 2048 | original_fp16 | OK | 14.6406 | 4403 MiB | 31.126s |
| OPT-1.3B | 2048 | oaken | OK | 15.3984 | 4419 MiB | 42.835s |

## Quantizer Notes

- OPT-350M: reused existing quantizer `quantizer/oaken/opt-350m.json`.
- OPT-1.3B: reused existing quantizer `results/rtx5060/opt-1.3b/oaken-quantizer.json`.

## Interpretation

### OPT-350M
- `original_fp16` peak VRAM by sequence length: 128=909 MiB, 256=967 MiB, 512=1111 MiB, 1024=1359 MiB, 2048=1879 MiB.
- `oaken` peak VRAM by sequence length: 128=909 MiB, 256=973 MiB, 512=1115 MiB, 1024=1363 MiB, 2048=1915 MiB.
- Oaken minus original peak VRAM: 128=+0 MiB, 256=+6 MiB, 512=+4 MiB, 1024=+4 MiB, 2048=+36 MiB.
- No OOM was observed through sequence length 2048.

### OPT-1.3B
- `original_fp16` peak VRAM by sequence length: 128=3151 MiB, 256=3201 MiB, 512=3099 MiB, 1024=3463 MiB, 2048=4403 MiB.
- `oaken` peak VRAM by sequence length: 128=3157 MiB, 256=3207 MiB, 512=3103 MiB, 1024=3479 MiB, 2048=4419 MiB.
- Oaken minus original peak VRAM: 128=+6 MiB, 256=+6 MiB, 512=+4 MiB, 1024=+16 MiB, 2048=+16 MiB.
- No OOM was observed through sequence length 2048.

## Answer

The answer should be based on the result table above. In this artifact path, Oaken quantizes KV activations inside the model forward pass, but the measured process peak can still be dominated by FP16 model weights, logits/loss tensors, allocator behavior, and temporary tensors. Therefore a lack of peak-VRAM reduction does not by itself disprove KV-cache compression; it means this Wikitext perplexity artifact path is not a pure long-generation KV-cache storage benchmark.

## Artifacts

- `kv_cache_pressure.csv`: machine-readable results.
- `logs.md`: log index.
- `hardware.txt`: hardware and container snapshot.
- `logs/*.log`: raw per-run logs.
- `logs/*_vram.csv`: VRAM samples from `nvidia-smi`.
- `logs/*_meta.json`: run metadata.
