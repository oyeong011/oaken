# RTX 5060 OPT-2.7B Oaken Boundary Experiment

## Scope

This run tests whether OPT-2.7B can complete the official Oaken Python artifact Wikitext accuracy path on an RTX 5060 8GB GPU:

- `/data/models/opt-2.7b` model preparation
- tiny CUDA gate before each major run
- original FP16 Wikitext perplexity
- Oaken offline profiling with group ratio `0.04 0.9 0.06`
- Oaken Wikitext perplexity
- default CUDA allocator vs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- elapsed time, peak VRAM, idle VRAM, OOM, NaN, and error-signal recording

PIQA, Winogrande, Hellaswag zero-shot accuracy and Figure 11 throughput reproduction were not run here. This is the OPT-2.7B Wikitext accuracy boundary check requested for the RTX 5060.

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
| Model path | `/data/models/opt-2.7b` |
| Model weight | `/data/models/opt-2.7b/pytorch_model.bin`, 5,303,359,381 bytes |

See `hardware.txt` for raw `nvidia-smi`, Docker, Python, Torch, and tiny CUDA output.

## Boundary Result

OPT-2.7B is inside the RTX 5060 8GB boundary for this Oaken official artifact Wikitext accuracy path.

Both allocator modes completed:

- tiny CUDA gates
- original FP16 Wikitext eval
- Oaken offline profiling
- Oaken Wikitext eval

No CUDA OOM, traceback, runtime error, CUDA error, `NaN`, or `tensor(nan)` was found in the recorded OPT-2.7B logs.

## Allocator Comparison

`Idle VRAM after` is recorded as the next wrapper's pre-run idle sample where available; the final post-run idle sample was `33 MiB used / 7668 MiB free`.

| Allocator | Step | Status | PPL | Elapsed | Peak VRAM | Idle VRAM after | Error signal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Default | Model download | OK | N/A | 3012.34s | 33 MiB | 33 MiB | None |
| Default | Tiny CUDA before original eval | OK | N/A | 1.57s | 173 MiB | 33 MiB | None |
| Default | Original FP16 Wikitext eval | OK | 12.4688 | 54.11s | 7515 MiB | 33 MiB | None |
| Default | Tiny CUDA before Oaken profiling | OK | N/A | 1.54s | 173 MiB | 33 MiB | None |
| Default | Oaken offline profiling | OK | 12.4688 during profiling pass | 144.14s wrapper / 140.44s script | 7653 MiB | 33 MiB | None |
| Default | Tiny CUDA before Oaken eval | OK | N/A | 1.52s | 173 MiB | 33 MiB | None |
| Default | Oaken Wikitext eval | OK | 12.5703 | 75.59s | 7535 MiB | 33 MiB | None |
| `expandable_segments:True` | Tiny CUDA before original eval | OK | N/A | 1.56s | 173 MiB | 33 MiB | None |
| `expandable_segments:True` | Original FP16 Wikitext eval | OK | 12.4688 | 54.59s | 7187 MiB | 33 MiB | None |
| `expandable_segments:True` | Tiny CUDA before Oaken profiling | OK | N/A | 1.57s | 173 MiB | 33 MiB | None |
| `expandable_segments:True` | Oaken offline profiling | OK | 12.4688 during profiling pass | 144.16s wrapper / 140.47s script | 7187 MiB | 33 MiB | None |
| `expandable_segments:True` | Tiny CUDA before Oaken eval | OK | N/A | 1.56s | 173 MiB | 33 MiB | None |
| `expandable_segments:True` | Oaken Wikitext eval | OK | 12.5703 | 76.13s | 7427 MiB | 33 MiB | None |

## Interpretation

- OPT-2.7B successfully completes the requested Oaken path on RTX 5060 8GB.
- This establishes the RTX 5060 can go beyond the previously completed OPT-1.3B path for this artifact.
- The default allocator peak was close to the available 8GB budget:
  - Original FP16 eval: 7515 MiB
  - Oaken profiling: 7653 MiB
  - Oaken eval: 7535 MiB
- `expandable_segments:True` lowered peak VRAM for the original eval and profiling paths:
  - Original FP16 eval: 7515 MiB -> 7187 MiB
  - Oaken profiling: 7653 MiB -> 7187 MiB
  - Oaken eval: 7535 MiB -> 7427 MiB
- PPL was unchanged between allocator modes:
  - Original: 12.4688
  - Oaken: 12.5703

## Oaken Sparsity

Both allocator modes produced the same total sparsity summary:

- Key: `[0.03983654623152688, 0.9000914422861227, 0.06007202468310475]`
- Value: `[0.04006396042404313, 0.8999179913299091, 0.060018061292820106]`

## Notes

- The model was downloaded PyTorch-only with `huggingface_hub.snapshot_download(..., allow_patterns=[...])`.
- Model files and Hugging Face cache remain under `/data/models`, not in this repository.
- The default quantizer was written to `oaken-quantizer.json`.
- The expandable allocator quantizer was written to `oaken-quantizer-expandable.json`.
- Result file scan for committed model payloads is expected to return nothing:
  `find results/rtx5060/opt-2.7b -type f \( -name "*.bin" -o -name "*.safetensors" \)`

## Artifacts

- `summary.md`
- `hardware.txt`
- `oaken-quantizer.json`
- `oaken-quantizer-expandable.json`
- `logs/download.log`
- `logs/download_meta.json`
- `logs/download_vram.csv`
- `logs/model_files.txt`
- `logs/default_before_original_tiny_cuda.log`
- `logs/default_before_original_tiny_cuda_meta.json`
- `logs/default_before_original_tiny_cuda_vram.csv`
- `logs/default_original_fp16_eval.log`
- `logs/default_original_fp16_eval_meta.json`
- `logs/default_original_fp16_eval_vram.csv`
- `logs/default_before_profile_tiny_cuda.log`
- `logs/default_before_profile_tiny_cuda_meta.json`
- `logs/default_before_profile_tiny_cuda_vram.csv`
- `logs/default_oaken_profile.log`
- `logs/default_oaken_profile_meta.json`
- `logs/default_oaken_profile_vram.csv`
- `logs/default_before_oaken_eval_tiny_cuda.log`
- `logs/default_before_oaken_eval_tiny_cuda_meta.json`
- `logs/default_before_oaken_eval_tiny_cuda_vram.csv`
- `logs/default_oaken_eval.log`
- `logs/default_oaken_eval_meta.json`
- `logs/default_oaken_eval_vram.csv`
- `logs/expandable_before_original_tiny_cuda.log`
- `logs/expandable_before_original_tiny_cuda_meta.json`
- `logs/expandable_before_original_tiny_cuda_vram.csv`
- `logs/expandable_original_fp16_eval.log`
- `logs/expandable_original_fp16_eval_meta.json`
- `logs/expandable_original_fp16_eval_vram.csv`
- `logs/expandable_before_profile_tiny_cuda.log`
- `logs/expandable_before_profile_tiny_cuda_meta.json`
- `logs/expandable_before_profile_tiny_cuda_vram.csv`
- `logs/expandable_oaken_profile.log`
- `logs/expandable_oaken_profile_meta.json`
- `logs/expandable_oaken_profile_vram.csv`
- `logs/expandable_before_oaken_eval_tiny_cuda.log`
- `logs/expandable_before_oaken_eval_tiny_cuda_meta.json`
- `logs/expandable_before_oaken_eval_tiny_cuda_vram.csv`
- `logs/expandable_oaken_eval.log`
- `logs/expandable_oaken_eval_meta.json`
- `logs/expandable_oaken_eval_vram.csv`
