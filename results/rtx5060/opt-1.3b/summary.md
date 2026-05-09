# RTX 5060 OPT-1.3B Oaken Smoke

## Scope

This run checks the official Oaken Python artifact path on an RTX 5060 8GB machine for OPT-1.3B:

- `/data/models/opt-1.3b` model preparation
- tiny CUDA sanity test
- original FP16 Wikitext perplexity
- Oaken offline threshold profiling
- Oaken Wikitext perplexity
- idle VRAM, elapsed time, peak VRAM, and OOM boundary recording

PIQA, Winogrande, Hellaswag zero-shot accuracy and Figure 11 throughput reproduction were not run here. Those remain outside this small-model 5060 artifact smoke scope.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 |
| VRAM | 8151 MiB |
| Driver | 590.48.01 |
| CUDA reported by driver | 13.1 |
| Container | `oaken-ae-container` from `oaken-ae-img` |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu130 |
| Baseline idle VRAM | 33 MiB used / 7668 MiB free |

See `hardware.txt` for the raw `nvidia-smi`, Docker, Python, and torch output.

## Results

| Run | Status | PPL | Elapsed | Peak VRAM | Idle VRAM |
| --- | --- | --- | --- | --- | --- |
| Model download | OK | N/A | 346.75s | 33 MiB | 33 MiB |
| Tiny CUDA test | OK | N/A | 1.56s | 185 MiB | 33 MiB |
| Original FP16 Wikitext eval | OK | 14.6406 | 31.06s | 4403 MiB | 33 MiB |
| Oaken offline profiling | OK | 14.6406 during profiling pass | 85.07s wrapper / 81.83s script | 4491 MiB | 33 MiB |
| Oaken Wikitext eval | OK | 15.3984 | 42.57s | 4419 MiB | 33 MiB |

No CUDA OOM occurred for OPT-1.3B on this RTX 5060 8GB run.

## Notes

- The quantizer was written to `oaken-quantizer.json`.
- The final Oaken sparsity summary was:
  - Key: `[0.04021852180784476, 0.8998964562765531, 0.059885022088436164]`
  - Value: `[0.039985373083405176, 0.9000418194994576, 0.05997280757455926]`
- The model was downloaded PyTorch-only with `allow_patterns` to avoid unnecessary TF/Flax weight downloads.
- The first broad `snapshot_download` attempt started unused TF/Flax incomplete cache files under `/data/models/opt-1.3b/.cache`; those cache files are outside the git checkout and are not committed.
- Model files and HuggingFace cache remain under `/data/models` / `/home/ssu/models`, not in this repository.

## Artifacts

- `hardware.txt`
- `oaken-quantizer.json`
- `logs/download.log`
- `logs/download_meta.json`
- `logs/download_vram.csv`
- `logs/model_files.txt`
- `logs/tiny_cuda.log`
- `logs/tiny_cuda_meta.json`
- `logs/tiny_cuda_vram.csv`
- `logs/original_fp16_eval.log`
- `logs/original_fp16_eval_meta.json`
- `logs/original_fp16_eval_vram.csv`
- `logs/oaken_profile.log`
- `logs/oaken_profile_meta.json`
- `logs/oaken_profile_vram.csv`
- `logs/oaken_eval.log`
- `logs/oaken_eval_meta.json`
- `logs/oaken_eval_vram.csv`
