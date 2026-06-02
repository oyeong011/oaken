# KV-cache Experiment Inventory

Verification date: 2026-06-02 (Asia/Seoul).  Repository: `/home/ssu/oaken`.

This inventory separates file-backed evidence from missing evidence.  Labels mean:

- **verified from existing file**: benchmark CSV/log already existed and was checked for path, timestamp, schema, row count, and supporting plot script.
- **newly regenerated**: produced in this update from an existing CSV, not hand drawn.
- **missing / not reproducible**: not present in this repository or not rerun in this session.

## Relevant commits

| Commit | Date | Relevance |
| --- | --- | --- |
| `dd2ea6a` | 2026-05-25 | Recovered RTX 5060 briefing evidence; current base commit before this publication update. |
| `e94c581` | 2026-05-25 | Added position-valid RTX 5060 Qwen2.5 boundary/rescue analysis. |
| `89c83dc` | 2026-05-25 | Added RTX 5060 OPT-1.3B boundary/rescue analysis. |
| `ef80cc2` | 2026-05-12 | Measured Oaken-style KV-cache pressure. |
| `26b18bc` / earlier RTX 5080 commits | 2026-05-10 to 2026-05-11 | Added RTX 5080 OPT-family summaries under `results/rtx5080/`. |
| publication commit from this update | 2026-06-02 | Commits the new required docs, growth plots, pressure plots, plot scripts, verification log, and any previously untracked relevant CSV/log/plot artifacts. |

## Scripts

| Path | Label | Purpose |
| --- | --- | --- |
| `analysis/kv_cache_growth.py` | verified from existing file | Pure FP16 KV-cache growth baseline. Writes theoretical vs actual `past_key_values` bytes, raw logs, and VRAM CSVs. |
| `analysis/plot_kv_cache_growth.py` | newly regenerated support script | Regenerates Experiment A PNGs from `kv_cache_growth.csv`. |
| `scripts/run_kv_cache_sweep.py` | verified from existing file | Chunked cache-growth boundary sweep for `dynamic`, `quantized`, `offloaded`, and `no_cache`. |
| `scripts/plot_kv_cache_sweep.py` | verified from existing file | Regenerates boundary/rescue PNGs and derived CSVs from combined sweep CSVs. |
| `analysis/plot_kv_cache_pressure.py` | newly regenerated support script | Regenerates RTX 5080/5060 Oaken-style pressure plots from `kv_cache_pressure.csv`. |

## Experiment A — Pure KV-cache Growth Baseline artifacts

Primary machine: RTX 5060 8GB.

| Path | Type | Label | Notes |
| --- | --- | --- | --- |
| `analysis/outputs/kv_cache_growth/rtx5060/kv_cache_growth.csv` | CSV | verified from existing file | 36 data rows; models `opt-350m`, `opt-1.3b`; batch sizes 1,2,4; sequence lengths 128..2048; no OOM; `theoretical_actual_ratio=1.0` for successful cached rows. |
| `analysis/outputs/kv_cache_growth/rtx5060/logs.md` | log index | verified from existing file | Index of raw per-case logs and VRAM sample CSVs. |
| `analysis/outputs/kv_cache_growth/rtx5060/raw_logs/*.log` | logs | verified from existing file | Per-case tensor shapes, byte counts, errors if any. |
| `analysis/outputs/kv_cache_growth/rtx5060/raw_logs/*_vram.csv` | CSV logs | verified from existing file | Best-effort `nvidia-smi` samples for each case. |
| `analysis/outputs/kv_cache_growth/rtx5060/summary.md` | docs | verified from existing file | Baseline interpretation and limitations. |
| `analysis/outputs/kv_cache_growth/rtx5060/kv_theory_actual_vs_seq_len.png` | PNG | newly regenerated | From `kv_cache_growth.csv` using `analysis/plot_kv_cache_growth.py`. |
| `analysis/outputs/kv_cache_growth/rtx5060/actual_vs_theoretical_kv.png` | PNG | newly regenerated | From `kv_cache_growth.csv`; visual formula check. |
| `analysis/outputs/kv_cache_growth/rtx5060/peak_cuda_allocated_vs_seq_len.png` | PNG | newly regenerated | From `kv_cache_growth.csv`; peak allocator context. |

## Experiment B — RTX 5060 Qwen2.5 dynamic vs quantized boundary/rescue artifacts

| Path | Type | Label | Notes |
| --- | --- | --- | --- |
| `results/rtx5060_qwen25_15b_sanity.csv` | CSV | verified from existing file | Confirms Qwen position limit and GQA/MQA metadata; dynamic `kv_actual_over_theory=1.0`. |
| `results/rtx5060_qwen25_15b_dynamic_boundary.csv` | CSV | verified from existing file | Dynamic cache sweep; 24 rows; OOM at batch 8 sequence 12288 and 16384. |
| `results/rtx5060_qwen25_15b_rescue_cases.csv` | CSV | verified from existing file | Quantized and no-cache rows for dynamic OOM cases; quantized succeeds at both OOM points. |
| `results/rtx5060_qwen25_15b_combined.csv` | CSV | verified from existing file | Combined dynamic + rescue rows; source for regenerated plots. |
| `results/plots_rtx5060_qwen25_15b_combined/peak_memory_vs_seq_len.png` | PNG | newly regenerated | From `results/rtx5060_qwen25_15b_combined.csv`. |
| `results/plots_rtx5060_qwen25_15b_combined/throughput_vs_peak_memory.png` | PNG | newly regenerated | From `results/rtx5060_qwen25_15b_combined.csv`. |
| `results/plots_rtx5060_qwen25_15b_combined/status_boundary_matrix.csv` | CSV summary | newly regenerated | Status table from combined CSV. |
| `results/plots_rtx5060_qwen25_15b_combined/oom_cases.csv` | CSV summary | newly regenerated | Dynamic OOM rows from combined CSV. |
| `results/plots_rtx5060_qwen25_15b_combined/dynamic_oom_rescue_cases.csv` | CSV summary | newly regenerated | Dynamic OOM cases rescued by quantized/no-cache rows. |

## Experiment B — RTX 5080 evidence artifacts

This repository has RTX 5080 **OPT-family / Oaken-style** evidence, not RTX 5080 Qwen dynamic/quantized evidence.

| Path | Type | Label | Notes |
| --- | --- | --- | --- |
| `analysis/outputs/kv_cache_pressure/rtx5080/kv_cache_pressure.csv` | CSV | verified from existing file | OPT-1.3B Original FP16 vs Oaken over seq 128,256,512,1024,2048; all rows OK. |
| `analysis/outputs/kv_cache_pressure/rtx5080/logs.md` | logs/doc | verified from existing file | Contains the exact Docker commands for every RTX 5080 pressure row. |
| `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/*.log` | logs | verified from existing file | Raw command outputs for each Original/Oaken run. |
| `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/*_vram.csv` | CSV logs | verified from existing file | Per-run VRAM samples. |
| `analysis/outputs/kv_cache_pressure/rtx5080/summary.md` | docs | verified from existing file | Interpretation of pressure sweep. |
| `analysis/outputs/kv_cache_pressure/rtx5080/peak_vram_vs_seq_len.png` | PNG | newly regenerated | From `kv_cache_pressure.csv`. |
| `analysis/outputs/kv_cache_pressure/rtx5080/ppl_vs_seq_len.png` | PNG | newly regenerated | From `kv_cache_pressure.csv`. |
| `analysis/outputs/kv_cache_pressure/rtx5080/elapsed_vs_seq_len.png` | PNG | newly regenerated | From `kv_cache_pressure.csv`. |
| `results/oaken_consumer_gpu_summary.csv` | CSV summary | verified from existing file | OPT-family summary across RTX 5060/5080, including RTX 5080 OPT-6.7B boundary. |
| `results/rtx5080/opt-1.3b/summary.md` | docs | verified from existing file | RTX 5080 OPT-1.3B Original/Oaken OK. |
| `results/rtx5080/opt-2.7b/summary.md` | docs | verified from existing file | RTX 5080 OPT-2.7B Original/Oaken OK. |
| `results/rtx5080/opt-6.7b/summary.md` | docs | verified from existing file | RTX 5080 OPT-6.7B boundary: original/profiling OK with allocator setting, Oaken eval OOM. |
| `results/rtx5080/opt-*/hardware.txt` | logs | verified from existing file | Host/GPU/driver/CUDA/PyTorch environment for RTX 5080 OPT-family runs. |

## Verification / regeneration logs from this update

| Path | Label | Contents |
| --- | --- | --- |
| `analysis/outputs/plot_regeneration_20260602T002937Z.log` | newly regenerated | Commands used to regenerate Experiment A and RTX 5060 boundary plots from CSV. |
| `analysis/outputs/plot_regeneration_pressure_20260602T003034Z.log` | newly regenerated | Commands used to regenerate Oaken-style pressure plots from CSV. |
| `analysis/outputs/kv_cache_artifact_verification_20260602T003111Z.log` | newly regenerated | File timestamps, sizes, CSV schemas, row counts, and current RTX 5060 environment snapshot. |


## Auxiliary / diagnostic artifacts in this repository

These files are related to KV-cache/Oaken work but are not the primary backing files for the two required experiment sections above.

| Path | Type | Label | Notes |
| --- | --- | --- | --- |
| `analysis/outputs/kv_cache_pressure/rtx5060/kv_cache_pressure.csv` | CSV | verified from existing file | RTX 5060 Wikitext Original/Oaken pressure artifact; auxiliary to the required 5060 Qwen boundary claim. |
| `analysis/outputs/kv_cache_pressure/rtx5060/logs.md` | logs/doc | verified from existing file | Command/log index for the auxiliary RTX 5060 pressure artifact. |
| `analysis/outputs/kv_cache_pressure/rtx5060/peak_vram_vs_seq_len.png` | PNG | newly regenerated | From `analysis/outputs/kv_cache_pressure/rtx5060/kv_cache_pressure.csv`. |
| `analysis/outputs/kv_cache_pressure/rtx5060/ppl_vs_seq_len.png` | PNG | newly regenerated | From `analysis/outputs/kv_cache_pressure/rtx5060/kv_cache_pressure.csv`. |
| `results/rtx5060_opt125m_dynamic_sanity.csv` | CSV | verified from existing file | Small OPT dynamic sanity diagnostic. |
| `results/plots_rtx5060_dynamic/peak_memory_vs_seq_len.png` | PNG | verified from existing file | Dynamic-only OPT diagnostic plot. |
| `results/plots_rtx5060_dynamic/throughput_vs_peak_memory.png` | PNG | verified from existing file | Dynamic-only OPT diagnostic plot. |
| `results/plots_rtx5060_dynamic/status_boundary_matrix.csv` | CSV | verified from existing file | Dynamic-only OPT diagnostic summary. |
| `results/plots_rtx5060_dynamic/oom_cases.csv` | CSV | verified from existing file | Dynamic-only OPT diagnostic OOM cases. |
| `results/plots_rtx5060_dynamic/dynamic_oom_rescue_cases.csv` | CSV | verified from existing file | Empty/diagnostic rescue table for dynamic-only run. |

## Missing / not reproducible in this repository

- **RTX 5080 Qwen dynamic-vs-quantized CSV:** missing / not reproducible here. No Qwen 5080 claim is made in these docs.
- **RTX 5060 offloaded rescue for Qwen:** missing / not reproducible here. The file-backed Qwen rescue evidence is quantized and no-cache lower-bound only.
- **A fresh benchmark rerun on RTX 5080 during this session:** missing / not reproducible from the current RTX 5060 host. Existing RTX 5080 files are verified instead of rerun.
- **Full Oaken paper reproduction:** not claimed. The repo contains Oaken-inspired / Oaken-style memory-pressure evidence only.
