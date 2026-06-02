# Two-machine KV-cache Results

Scope: two experiments in `/home/ssu/oaken`, with RTX 5060 and RTX 5080 evidence kept separate.

- **Experiment A:** Pure KV-cache Growth Baseline.
- **Experiment B:** Quantized Cache Rescue / Oaken-inspired KV-cache Boundary.

Result labels:

- **verified from existing file**: benchmark CSV/log existed and was checked.
- **newly regenerated**: plot/derived CSV regenerated from CSV during this update.
- **missing / not reproducible**: unavailable in this repository or not rerun in this session.

## Machine A — RTX 5060 8GB

### Environment

| Item | Value | Evidence |
| --- | --- | --- |
| Hostname | `ssu-22663-09` | `analysis/outputs/kv_cache_artifact_verification_20260602T003111Z.log` |
| GPU | NVIDIA GeForce RTX 5060 | same log; CSV `detected_gpu_name` fields |
| VRAM | 8151 MiB by current `nvidia-smi` / 7.52 GiB usable in CUDA OOM messages | same log; Qwen OOM messages |
| Driver | 590.48.01 | same log; `analysis/outputs/kv_cache_pressure/rtx5060/hardware.txt` |
| CUDA | 13.1 by `nvidia-smi`; PyTorch CUDA 13.0 in current venv | same log |
| PyTorch | 2.12.0+cu130 in current verification venv; 2.11.0+cu130 in older Docker pressure artifact | same log; `analysis/outputs/kv_cache_pressure/rtx5060/hardware.txt` |
| Transformers | 5.8.1 in current verification venv | same log |

### Experiment A: Pure KV-cache Growth Baseline

**Status:** benchmark rows are **verified from existing file**; PNG plots are **newly regenerated** from CSV.

Purpose: isolate the FP16 KV-cache term and verify that actual `past_key_values` tensor bytes follow the theoretical formula as batch size and sequence length grow.

Exact benchmark command recorded for this artifact family:

```bash
python analysis/kv_cache_growth.py \
  --models /home/ssu/models/opt-350m /home/ssu/models/opt-1.3b \
  --output-dir analysis/outputs/kv_cache_growth/rtx5060
```

Plot regeneration command:

```bash
/home/ssu/kv-cache-consumer-gpu-bench/.venv/bin/python analysis/plot_kv_cache_growth.py \
  --input analysis/outputs/kv_cache_growth/rtx5060/kv_cache_growth.csv \
  --output-dir analysis/outputs/kv_cache_growth/rtx5060
```

Models tested: `opt-350m`, `opt-1.3b` local Hugging Face checkpoints.

Cache modes tested: `use_cache=True`; plus `use_cache=False` for batch 1 at selected sequence lengths as a lower-bound comparison.

Batch sizes: `1, 2, 4` for `use_cache=True`.

Sequence lengths: `128, 256, 512, 1024, 2048` for `use_cache=True`; `128, 512, 1024` for `use_cache=False` batch 1.

CSV output paths:

- `analysis/outputs/kv_cache_growth/rtx5060/kv_cache_growth.csv`
- `analysis/outputs/kv_cache_growth/rtx5060/raw_logs/*_vram.csv`

Plot output paths:

- `analysis/outputs/kv_cache_growth/rtx5060/kv_theory_actual_vs_seq_len.png`
- `analysis/outputs/kv_cache_growth/rtx5060/actual_vs_theoretical_kv.png`
- `analysis/outputs/kv_cache_growth/rtx5060/peak_cuda_allocated_vs_seq_len.png`

Logs/docs:

- `analysis/outputs/kv_cache_growth/rtx5060/logs.md`
- `analysis/outputs/kv_cache_growth/rtx5060/raw_logs/*.log`
- `analysis/outputs/kv_cache_growth/rtx5060/summary.md`

OOM boundary table:

| Model | Cache mode | Batch sizes | Seq lengths | OOM cases | Evidence |
| --- | --- | --- | --- | --- | --- |
| `opt-350m` | `use_cache=True` | 1,2,4 | 128..2048 | none | `kv_cache_growth.csv` |
| `opt-1.3b` | `use_cache=True` | 1,2,4 | 128..2048 | none | `kv_cache_growth.csv` |
| `opt-350m`, `opt-1.3b` | `use_cache=False` | 1 | 128,512,1024 | none | `kv_cache_growth.csv` |

Actual vs theoretical KV size:

- File-backed claim: all successful `use_cache=True` rows in `kv_cache_growth.csv` have `theoretical_actual_ratio=1.0`.
- Interpretation: for the tested OPT models, the pure FP16 KV-cache tensor footprint scales linearly with `batch_size * sequence_length` and matches the formula used by the script.
- Limitation: CUDA peak allocator memory is larger than KV bytes because it includes model weights, activations, workspaces, and allocator behavior.

### Experiment B: RTX 5060 Qwen2.5 dynamic vs quantized boundary/rescue

**Status:** benchmark rows are **verified from existing file**; PNGs and derived CSVs are **newly regenerated** from `results/rtx5060_qwen25_15b_combined.csv`.

Purpose: test whether a quantized KV-cache can make dynamic-cache OOM cases feasible on the 8GB RTX 5060, using a position-valid long-context Qwen model.

Exact dynamic boundary command:

```bash
python scripts/run_kv_cache_sweep.py \
  --model /home/ssu/models/Qwen2.5-1.5B-Instruct \
  --gpu-name rtx5060-8gb \
  --dtype fp16 \
  --batch-sizes 1 2 4 8 \
  --seq-lens 1024 2048 4096 8192 12288 16384 \
  --cache-modes dynamic \
  --output results/rtx5060_qwen25_15b_dynamic_boundary.csv \
  --chunk-size 128
```

Exact rescue command:

```bash
python scripts/run_kv_cache_sweep.py \
  --model /home/ssu/models/Qwen2.5-1.5B-Instruct \
  --gpu-name rtx5060-8gb \
  --dtype fp16 \
  --batch-sizes 8 \
  --seq-lens 12288 16384 \
  --cache-modes quantized no_cache \
  --output results/rtx5060_qwen25_15b_rescue_cases.csv \
  --chunk-size 128 \
  --quant-backend hqq \
  --quant-bits 4
```

Plot regeneration command:

```bash
/home/ssu/kv-cache-consumer-gpu-bench/.venv/bin/python scripts/plot_kv_cache_sweep.py \
  --input results/rtx5060_qwen25_15b_combined.csv \
  --output-dir results/plots_rtx5060_qwen25_15b_combined
```

Model tested: `/home/ssu/models/Qwen2.5-1.5B-Instruct` (Qwen2.5-1.5B-Instruct local checkpoint).

Cache modes tested: `dynamic`, `quantized` (HQQ, 4-bit), and `no_cache` lower-bound ablation.

Batch sizes: dynamic `1, 2, 4, 8`; rescue `8`.

Sequence lengths: dynamic `1024, 2048, 4096, 8192, 12288, 16384`; rescue `12288, 16384`.

CSV output paths:

- `results/rtx5060_qwen25_15b_sanity.csv`
- `results/rtx5060_qwen25_15b_dynamic_boundary.csv`
- `results/rtx5060_qwen25_15b_rescue_cases.csv`
- `results/rtx5060_qwen25_15b_combined.csv`
- `results/plots_rtx5060_qwen25_15b_combined/status_boundary_matrix.csv`
- `results/plots_rtx5060_qwen25_15b_combined/oom_cases.csv`
- `results/plots_rtx5060_qwen25_15b_combined/dynamic_oom_rescue_cases.csv`

Plot output paths:

- `results/plots_rtx5060_qwen25_15b_combined/peak_memory_vs_seq_len.png`
- `results/plots_rtx5060_qwen25_15b_combined/throughput_vs_peak_memory.png`

OOM boundary table:

| Cache mode | Batch | Largest OK seq_len | First OOM seq_len | Evidence |
| --- | ---: | ---: | ---: | --- |
| dynamic | 1 | 16384 | none tested | `results/rtx5060_qwen25_15b_dynamic_boundary.csv` |
| dynamic | 2 | 16384 | none tested | same |
| dynamic | 4 | 16384 | none tested | same |
| dynamic | 8 | 8192 | 12288 | `results/plots_rtx5060_qwen25_15b_combined/oom_cases.csv` |
| quantized | 8 | 16384 | none in rescue cases | `results/rtx5060_qwen25_15b_rescue_cases.csv` |

Dynamic vs quantized comparison:

| Batch | Seq len | Dynamic status | Quantized status | Quantized `kv_actual_over_theory` | Quantized peak VRAM MiB | Evidence |
| ---: | ---: | --- | --- | ---: | ---: | --- |
| 8 | 12288 | OOM | OK | 0.288737 | 5903 | `results/rtx5060_qwen25_15b_rescue_cases.csv` |
| 8 | 16384 | OOM | OK | 0.286865 | 6795 | `results/rtx5060_qwen25_15b_rescue_cases.csv` |

Interpretation:

- File-backed claim: on RTX 5060, Qwen2.5 dynamic cache OOMs at `batch=8, seq_len=12288` and `batch=8, seq_len=16384`; quantized cache completes both cases.
- File-backed claim: Qwen sanity rows report `position_valid=True`, `max_position_embeddings=32768`, `kv_formula_type=gqa_mqa`, and dynamic `kv_actual_over_theory=1.0`.
- Do not overclaim: the ~0.287 quantized ratio is a KV-cache tensor footprint ratio, not a total VRAM reduction ratio. `no_cache` is an ablation/lower-bound, not a practical serving policy.

## Machine B — RTX 5080

### Environment

| Item | Value | Evidence |
| --- | --- | --- |
| Hostname | `ssu-04` | `results/rtx5080/opt-1.3b/hardware.txt` and sibling `hardware.txt` files |
| GPU | NVIDIA GeForce RTX 5080 | same |
| VRAM | 16303 MiB | same |
| Driver | 580.142 | same |
| CUDA | 13.0 by `nvidia-smi` and PyTorch CUDA | same |
| PyTorch | 2.11.0+cu130 | same |
| Container | `oaken-ae-container` from image `oaken-ae-img` | same |

### Experiment B: RTX 5080 OPT-family / Oaken-style boundary evidence

**Status:** benchmark rows and logs are **verified from existing file**; pressure plots are **newly regenerated** from CSV.  No fresh RTX 5080 benchmark was rerun from the current RTX 5060 host.

This is **not** RTX 5080 Qwen evidence. It is OPT-family evidence and is kept separate from the RTX 5060 Qwen claims.

#### OPT-1.3B pressure sweep

Purpose: compare Original FP16 vs Oaken as Wikitext sequence length increases on RTX 5080.

Exact commands: every row's Docker command is listed in `analysis/outputs/kv_cache_pressure/rtx5080/logs.md`. Example pattern for Original FP16:

```bash
docker exec -w /workspace oaken-ae-container /opt/conda/envs/oaken/bin/python eval_perplexity.py \
  -m opt -s 1.3b -t wikitext \
  --max-length 2048 --stride 2048 \
  --gpu-start-idx 0 --gpu-count 1 --quant-method none
```

Example pattern for Oaken:

```bash
docker exec -w /workspace oaken-ae-container /opt/conda/envs/oaken/bin/python eval_perplexity.py \
  -m opt -s 1.3b -t wikitext \
  --max-length 2048 --stride 2048 \
  --gpu-start-idx 0 --gpu-count 1 \
  -q quantizer/oaken/opt-1.3b.json --quant-method oaken
```

Model tested: `OPT-1.3B`.

Modes tested: `Original FP16`, `Oaken`.

Sequence lengths: `128, 256, 512, 1024, 2048`.

CSV/log output paths:

- `analysis/outputs/kv_cache_pressure/rtx5080/kv_cache_pressure.csv`
- `analysis/outputs/kv_cache_pressure/rtx5080/logs.md`
- `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/*.log`
- `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/*_vram.csv`

Plot output paths:

- `analysis/outputs/kv_cache_pressure/rtx5080/peak_vram_vs_seq_len.png`
- `analysis/outputs/kv_cache_pressure/rtx5080/ppl_vs_seq_len.png`
- `analysis/outputs/kv_cache_pressure/rtx5080/elapsed_vs_seq_len.png`

OOM boundary table for this sweep:

| Mode | Largest OK seq_len | OOM cases | Evidence |
| --- | ---: | --- | --- |
| Original FP16 | 2048 | none in this sweep | `analysis/outputs/kv_cache_pressure/rtx5080/kv_cache_pressure.csv` |
| Oaken | 2048 | none in this sweep | same |

Original vs Oaken comparison in the pressure sweep:

| Seq len | Original peak MiB | Oaken peak MiB | Original PPL | Oaken PPL | Evidence |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 3421 | 3427 | 35.3750 | 36.6562 | `analysis/outputs/kv_cache_pressure/rtx5080/kv_cache_pressure.csv` |
| 256 | 3471 | 3477 | 25.8906 | 26.9219 | same |
| 512 | 3369 | 3373 | 20.2031 | 21.0938 | same |
| 1024 | 3733 | 3749 | 16.7812 | 17.5938 | same |
| 2048 | 4673 | 4689 | 14.6406 | 15.4297 | same |

Interpretation: in this RTX 5080 OPT-1.3B pressure path, Oaken did **not** lower peak VRAM or extend the tested sequence boundary; both modes completed all tested lengths, and Oaken peak VRAM was slightly higher in each row. This is file-backed only for OPT-1.3B and the listed Wikitext/evaluation path.

#### OPT-family larger-model boundary confirmation

Purpose: record a larger-model RTX 5080 capacity boundary separately from the RTX 5060 Qwen evidence.

CSV/docs:

- `results/oaken_consumer_gpu_summary.csv`
- `results/rtx5080/opt-1.3b/summary.md`
- `results/rtx5080/opt-2.7b/summary.md`
- `results/rtx5080/opt-6.7b/summary.md`
- `results/rtx5080/opt-6.7b/logs.md`
- `results/rtx5080/opt-*/hardware.txt`

Boundary table:

| Model | Original FP16 | Oaken / Oaken-style path | Peak VRAM GB | Status | Evidence |
| --- | --- | --- | ---: | --- | --- |
| OPT-1.3B | OK | OK | 4.67 | OK | `results/oaken_consumer_gpu_summary.csv`, `results/rtx5080/opt-1.3b/summary.md` |
| OPT-2.7B | OK | OK | 7.83 | OK | `results/oaken_consumer_gpu_summary.csv`, `results/rtx5080/opt-2.7b/summary.md` |
| OPT-6.7B | OK with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Oaken eval OOM | 15.83 | Boundary | `results/rtx5080/opt-6.7b/summary.md` |

Exact OPT-6.7B boundary commands are in `results/rtx5080/opt-6.7b/logs.md`; the successful original/profiling runs used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and the Oaken eval still failed with CUDA OOM.

### RTX 5080 limitations

- No RTX 5080 Qwen dynamic/quantized CSV exists in this repository; no Qwen 5080 claim is made.
- The RTX 5080 pressure sweep is Original FP16 vs Oaken, not Hugging Face `dynamic` vs `quantized` cache mode.
- The OPT-6.7B result is a boundary artifact, not a quantized rescue claim.
- The current session ran on RTX 5060, so RTX 5080 benchmark rows were verified from files rather than rerun.

## Cross-machine interpretation

- RTX 5060-backed evidence: Qwen2.5 dynamic OOM and quantized rescue at `batch=8`, `seq_len=12288/16384`; OPT pure KV growth baseline with theory/actual ratio 1.0.
- RTX 5080-backed evidence: OPT-family Oaken-style runs scale to OPT-2.7B and expose an OPT-6.7B boundary; OPT-1.3B pressure sweep shows no peak-memory rescue by Oaken in that evaluation path.
- These are separate claims. RTX 5060 Qwen rescue is not used to claim RTX 5080 Qwen behavior, and RTX 5080 OPT evidence is not used to claim Qwen rescue.
