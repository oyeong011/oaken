# Experiment Inventory

## 1. Repository Purpose

1. Consumer NVIDIA GPU에서 LLM inference KV-cache memory pressure를 관찰한 실험 저장소이다.
2. Oaken 논문을 동기로 삼았지만, 현재 repo는 Oaken paper full reproduction을 증명하지 않는다.
3. 핵심 산출물은 batch size, sequence length, cache policy 변화에 따른 OOM boundary와 rescue 사례이다.
4. RTX 5060 8GB에서는 OPT-1.3B stress 결과와 Qwen2.5-1.5B position-valid long-context 결과가 있다.
5. RTX 5080에는 Oaken-style accuracy/boundary summary가 있으나, repo 안에는 Qwen cache-policy cross-GPU CSV가 없다.

## 2. Relevant Directory Tree

```text
.
├── AGENTS.md
├── README.md
├── scripts/
│   ├── run_kv_cache_sweep.py
│   └── plot_kv_cache_sweep.py
├── results/
│   ├── oaken_consumer_gpu_summary.csv
│   ├── oaken_consumer_gpu_summary.md
│   ├── rtx5060_opt13b_dynamic_boundary.csv
│   ├── rtx5060_opt13b_rescue_cases.csv
│   ├── rtx5060_opt13b_combined.csv
│   ├── rtx5060_qwen25_15b_sanity.csv
│   ├── rtx5060_qwen25_15b_dynamic_boundary.csv
│   ├── rtx5060_qwen25_15b_rescue_cases.csv
│   ├── rtx5060_qwen25_15b_combined.csv
│   ├── plots_rtx5060_combined/
│   ├── plots_rtx5060_qwen25_15b_combined/
│   ├── rtx5060/
│   └── rtx5080/
└── analysis/
    └── outputs/
        ├── kv_cache_growth/rtx5060/
        └── kv_cache_pressure/rtx5060/
```

Low-level `*_vram.csv` files exist under `results/rtx5060`, `results/rtx5080`, and `analysis/outputs`. They are raw sampling logs, not the main briefing tables.

## 3. Experiment Scripts

| Path | Role |
| --- | --- |
| `scripts/run_kv_cache_sweep.py` | Chunked cache-growth sweep for dynamic / quantized / offloaded / no_cache. Emits metadata such as `position_valid`, `kv_formula_type`, and KV theory/actual columns. |
| `scripts/plot_kv_cache_sweep.py` | Creates peak-memory and throughput plots plus OOM/rescue derived CSVs from sweep CSVs. |
| `analysis/kv_cache_growth.py` | Earlier FP16 OPT KV-cache growth sanity experiment. |
| `analysis/oaken_dataset_stability.py` | Dataset stability analysis, not the main KV-cache boundary result. |
| `analysis/oaken_threshold_analysis.py` | Threshold analysis, not the main KV-cache boundary result. |
| `scripts/accuracy_oaken.py` and sibling accuracy scripts | Accuracy evaluation entrypoints from the base Oaken-style repository. |

## 4. Main Result CSV Files

| Path | Rows | Columns | Model | GPU | Cache modes | Max batch | Max seq | OOM count |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| `results/rtx5060_opt13b_dynamic_boundary.csv` | 20 | sweep columns incl. `status`, `theoretical_dynamic_kv_mib` | `facebook/opt-1.3b` | `rtx5060-8gb` | dynamic | 8 | 8192 | 4 |
| `results/rtx5060_opt13b_rescue_cases.csv` | 8 | sweep columns | `facebook/opt-1.3b` | `rtx5060-8gb` | quantized, no_cache | 8 | 8192 | 2 |
| `results/rtx5060_opt13b_combined.csv` | 28 | sweep columns | `facebook/opt-1.3b` | `rtx5060-8gb` | dynamic, quantized, no_cache | 8 | 8192 | 6 |
| `results/plots_rtx5060_combined/dynamic_oom_rescue_cases.csv` | 6 | rescue summary columns | `facebook/opt-1.3b` | not found in file | quantized, no_cache rescue rows | 8 | 8192 | not found in file |
| `results/plots_rtx5060_combined/oom_cases.csv` | 6 | OOM summary columns | `facebook/opt-1.3b` | not found in file | dynamic, quantized | 8 | 8192 | 6 |
| `results/plots_rtx5060_combined/status_boundary_matrix.csv` | 28 | status matrix columns | `facebook/opt-1.3b` | not found in file | dynamic, quantized, no_cache | 8 | 8192 | 6 |
| `results/rtx5060_qwen25_15b_sanity.csv` | 4 | sweep columns incl. `position_valid`, `kv_formula_type` | `/home/ssu/models/Qwen2.5-1.5B-Instruct` | `rtx5060-8gb` | dynamic, quantized | 1 | 2048 | 0 |
| `results/rtx5060_qwen25_15b_dynamic_boundary.csv` | 24 | sweep columns incl. Qwen metadata | `/home/ssu/models/Qwen2.5-1.5B-Instruct` | `rtx5060-8gb` | dynamic | 8 | 16384 | 2 |
| `results/rtx5060_qwen25_15b_rescue_cases.csv` | 4 | sweep columns incl. Qwen metadata | `/home/ssu/models/Qwen2.5-1.5B-Instruct` | `rtx5060-8gb` | quantized, no_cache | 8 | 16384 | 0 |
| `results/rtx5060_qwen25_15b_combined.csv` | 28 | sweep columns incl. Qwen metadata | `/home/ssu/models/Qwen2.5-1.5B-Instruct` | `rtx5060-8gb` | dynamic, quantized, no_cache | 8 | 16384 | 2 |
| `results/plots_rtx5060_qwen25_15b_combined/dynamic_oom_rescue_cases.csv` | 4 | rescue summary columns | Qwen local path | not found in file | quantized, no_cache rescue rows | 8 | 16384 | not found in file |
| `results/plots_rtx5060_qwen25_15b_combined/oom_cases.csv` | 2 | OOM summary columns | Qwen local path | not found in file | dynamic | 8 | 16384 | 2 |
| `results/plots_rtx5060_qwen25_15b_combined/status_boundary_matrix.csv` | 28 | status matrix columns | Qwen local path | not found in file | dynamic, quantized, no_cache | 8 | 16384 | 2 |
| `analysis/outputs/kv_cache_growth/rtx5060/kv_cache_growth.csv` | 36 | KV theory/actual growth columns | `opt-1.3b`, `opt-350m` | `NVIDIA GeForce RTX 5060` | `use_cache` True/False | 4 | 2048 | 0 |
| `analysis/outputs/kv_cache_pressure/rtx5060/kv_cache_pressure.csv` | 20 | PPL/memory pressure columns | `OPT-1.3B`, `OPT-350M` | `RTX 5060 8GB` in `gpu` column | original_fp16, oaken | not found in file | 2048 | 0 |
| `results/oaken_consumer_gpu_summary.csv` | 9 | Oaken-style summary columns | OPT-125M through OPT-6.7B | RTX 5060, RTX 5080 | not cache-policy sweep | not found in file | not found in file | boundary status only |

## 5. Plot/Image Files

| Path | Supported result |
| --- | --- |
| `results/plots_rtx5060_combined/peak_memory_vs_seq_len.png` | RTX 5060 OPT-1.3B combined sweep |
| `results/plots_rtx5060_combined/throughput_vs_peak_memory.png` | RTX 5060 OPT-1.3B combined sweep |
| `results/plots_rtx5060_qwen25_15b_combined/peak_memory_vs_seq_len.png` | RTX 5060 Qwen2.5-1.5B combined sweep |
| `results/plots_rtx5060_qwen25_15b_combined/throughput_vs_peak_memory.png` | RTX 5060 Qwen2.5-1.5B combined sweep |
| `results/plots_rtx5060_dynamic/*.png` | Dynamic-only OPT diagnostic plots; untracked at time of inventory |
| `results/_tmp_plots_smoke/*.png` | Smoke-test plots; not briefing evidence |

## 6. README / Summary Sections

| Path | Summary content |
| --- | --- |
| `README.md` | RTX 5060 OPT boundary result and Qwen position-valid long-context result |
| `results/oaken_consumer_gpu_summary.md` | Consumer GPU Oaken-style result summary |
| `results/rtx5080/opt-1.3b/summary.md` | RTX 5080 OPT-1.3B Oaken-style result |
| `results/rtx5080/opt-2.7b/summary.md` | RTX 5080 OPT-2.7B Oaken-style result |
| `results/rtx5080/opt-6.7b/summary.md` | RTX 5080 OPT-6.7B boundary: original/profiling OK, Oaken eval OOM |
| `analysis/outputs/kv_cache_growth/rtx5060/summary.md` | FP16 OPT KV-cache theoretical-vs-actual sanity |
| `analysis/outputs/kv_cache_pressure/rtx5060/summary.md` | Wikitext perplexity pressure summary |

## 7. Evidence Mapping

| Evidence target | Supporting files |
| --- | --- |
| RTX 5060 OPT-1.3B result | `results/rtx5060_opt13b_dynamic_boundary.csv`, `results/rtx5060_opt13b_rescue_cases.csv`, `results/plots_rtx5060_combined/*`, `README.md` |
| RTX 5060 Qwen result | `results/rtx5060_qwen25_15b_sanity.csv`, `results/rtx5060_qwen25_15b_dynamic_boundary.csv`, `results/rtx5060_qwen25_15b_rescue_cases.csv`, `results/plots_rtx5060_qwen25_15b_combined/*`, `README.md` |
| RTX 5080 result | `results/oaken_consumer_gpu_summary.csv`, `results/rtx5080/opt-*/summary.md`. Qwen cache-policy 5080 CSV is not found in this repository. |
| dynamic vs quantized vs offloaded vs no_cache comparison | RTX 5060 OPT/Qwen files cover dynamic, quantized, no_cache. Offloaded comparison is not available in the committed sweep CSVs; README records host-memory limitation. |
| theoretical vs actual KV-cache size check | `analysis/outputs/kv_cache_growth/rtx5060/kv_cache_growth.csv`, `results/rtx5060_qwen25_15b_sanity.csv`, `results/rtx5060_qwen25_15b_dynamic_boundary.csv` |

## 8. Missing or Ambiguous Files

- `results/status_boundary_matrix.csv`, `results/oom_cases.csv`, and `results/dynamic_oom_rescue_cases.csv` are not found at top-level `results/`; corresponding files exist under `results/plots_rtx5060_combined/` and `results/plots_rtx5060_qwen25_15b_combined/`.
- The prior-note RTX 5080 Qwen cache-policy file with `80 total / 76 OK / 4 OOM` is not found under this repository. A different local directory may contain it, but this inventory treats repository files as authority.
- Offloaded RTX 5060 rescue rows are not present in `results/rtx5060_opt13b_rescue_cases.csv`; README states the run hit host RAM pressure.
- `results/plots_rtx5060_dynamic/` and several `_tmp_*` files are untracked diagnostics, not primary evidence.

## 9. Korean Summary

repo 안에서 강하게 뒷받침되는 증거는 두 가지다. 첫째, RTX 5060 8GB에서 OPT-1.3B memory stress sweep은 dynamic OOM과 일부 quantized rescue를 보여준다. 둘째, Qwen2.5-1.5B-Instruct는 `position_valid=True`인 12288/16384 context에서 dynamic OOM이 발생했고 quantized cache가 두 케이스를 모두 rescue했다. 부족한 증거는 RTX 5080 Qwen cache-policy cross-GPU 비교 CSV와 offloaded rescue row이다.
