# RTX 5060 OPT-350M Oaken Smoke

## Environment

- GPU: NVIDIA GeForce RTX 5060
- VRAM: 8151 MiB
- Driver: 590.48.01
- Container: oaken-ae-container / oaken-ae-img
- Python: 3.10.20
- PyTorch: 2.11.0+cu130
- Transformers: 4.47.0

## Result

| Run | Status | PPL | Elapsed | Peak VRAM |
| --- | --- | ---: | ---: | ---: |
| Oaken profiling | OK | 22.0156 during profiling runner | 39.090s internal / 41.758s wrapper | 1937 MiB |
| Original FP16 eval | OK | 22.0156 | 14.921s wrapper | 1879 MiB |
| Oaken eval after zero-range fix | OK | 22.1875 | 19.332s wrapper | 1915 MiB |

Before the zero-range guard, OPT-350M Oaken eval exited with status 0 but printed `tensor(nan, device='cuda:0', dtype=torch.float16)`.
After the guard, Oaken eval is finite and close to the original FP16 baseline.

## Fix Summary

- `src/oaken/quantize.py`: guarded uniform quantization against zero or non-finite ranges before dividing by `rangeval`.
- `src/util.py`: added config-aware OPT device-map entries for `project_in` and `project_out`, needed by OPT-350M because `word_embed_proj_dim != hidden_size`.
- `src/model.py`: passes the loaded config into `get_model_device_map()` so OPT-350M gets projection layers while OPT-125M keeps the original final layer norm mapping.

`kivi_main.py` and `tender_main.py` were reverted because they are not needed for this Oaken OPT-350M smoke.

## Artifacts

- `hardware.txt`: hardware and software environment
- `oaken-quantizer.json`: copied Oaken threshold profile for OPT-350M
- `logs/oaken_profile_final.log`
- `logs/oaken_profile_final_vram.csv`
- `logs/original_fp16_eval.log`
- `logs/original_fp16_eval_vram.csv`
- `logs/oaken_eval_after_zero_range_fix.log`
- `logs/oaken_eval_after_zero_range_fix_vram.csv`

## Notes

No local RTX 5080 Oaken checkout was found under `/home/ssu` or `/tmp`, so the numeric guard was applied directly to the matching zero-range failure point in `src/oaken/quantize.py`.
