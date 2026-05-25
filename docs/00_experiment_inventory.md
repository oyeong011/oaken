# Experiment Inventory

## 1. Repository Purpose

1. 이 저장소는 Oaken artifact와 consumer GPU 기반 LLM inference/KV-cache 관련 실험 결과를 모은다.
2. 현재 `oaken` 내부에는 Oaken accuracy/VRAM artifact와 RTX 5080 KV-cache pressure 결과가 있다.
3. 별도 로컬 저장소 `/home/ssu/kv_cache_consumer_gpu_bench`에는 Hugging Face Qwen2.5 KV-cache mode sweep 결과가 있다.
4. 발표에서 강하게 사용할 수 있는 수치는 로컬 CSV/MD 파일로 확인된 값으로 제한해야 한다.
5. RTX 5060 OPT/Qwen boundary-rescue CSV는 현재 `oaken` 저장소에서 발견되지 않았다.

## 2. Relevant File Tree

```text
.
├── AGENTS.md
├── README.md
├── analysis/
│   ├── run_kv_cache_pressure.py
│   └── outputs/
│       └── kv_cache_pressure/rtx5080/
│           ├── kv_cache_pressure.csv
│           ├── summary.md
│           └── logs.md
├── results/
│   ├── oaken_consumer_gpu_summary.csv
│   ├── oaken_consumer_gpu_summary.md
│   ├── rtx5060/
│   └── rtx5080/
├── scripts/
│   ├── run_kv_cache_sweep.py
│   └── plot_kv_cache_sweep.py
└── docs/
```

Relevant external local evidence:

```text
/home/ssu/kv_cache_consumer_gpu_bench/
├── results/results_5080_qwen25_1p5b.csv
├── analysis/rtx5080_qwen25_analysis.md
├── analysis/rtx5080_status_count.csv
├── analysis/rtx5080_oom_cases.csv
├── analysis/rtx5080_ratios_vs_dynamic.csv
├── analysis/rtx5080_key_examples.csv
└── plots_5080/*.png
```

## 3. Experiment Scripts

| Path | Role |
| --- | --- |
| `analysis/run_kv_cache_pressure.py` | RTX 5080 Oaken vs Original FP16 sequence-length pressure run. |
| `scripts/run_kv_cache_sweep.py` | HF cache-mode sweep harness with dynamic/quantized/offloaded/no_cache support. |
| `scripts/plot_kv_cache_sweep.py` | Plot/status summary generator for sweep CSVs. |
| `/home/ssu/kv_cache_consumer_gpu_bench/run_kv_cache_bench.py` | External local Qwen/TinyLlama KV-cache benchmark harness. |
| `/home/ssu/kv_cache_consumer_gpu_bench/analyze_results.py` | External local Qwen result analyzer. |

## 4. Result CSV Files

| Path | Columns | Rows | Model | GPU | Cache modes | Max batch | Max seq | OOM count |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| `results/oaken_consumer_gpu_summary.csv` | 11 | 9 | OPT-125M/350M/1.3B/2.7B/6.7B | RTX 5060, RTX 5080 | Oaken vs Original summary | not found in repository | not found in repository | boundary row 1 |
| `analysis/outputs/kv_cache_pressure/rtx5080/kv_cache_pressure.csv` | 14 | 10 | OPT-1.3B | RTX 5080 | Original FP16, Oaken | not found in repository | 2048 | 0 |
| `/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv` | 26 | 80 | Qwen/Qwen2.5-1.5B-Instruct | NVIDIA GeForce RTX 5080 | dynamic, quantized, offloaded, no_cache | 8 | 8192 | 4 |
| `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_status_count.csv` | 2 | 2 | Qwen2.5 analysis summary | RTX 5080 | all modes summarized | not found in repository | not found in repository | 4 |
| `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_oom_cases.csv` | 4 | 4 | Qwen2.5 analysis summary | RTX 5080 | dynamic, quantized, offloaded, no_cache | 8 | 8192 | 4 |
| `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_ratios_vs_dynamic.csv` | 6 | 3 | Qwen2.5 analysis summary | RTX 5080 | quantized, offloaded, no_cache | not found in repository | not found in repository | not found in repository |

## 5. Plot/Image Files

| Path | Meaning |
| --- | --- |
| `/home/ssu/kv_cache_consumer_gpu_bench/plots_5080/throughput_vs_seq_len.png` | Qwen2.5 RTX 5080 throughput by sequence length. |
| `/home/ssu/kv_cache_consumer_gpu_bench/plots_5080/latency_vs_seq_len.png` | Qwen2.5 RTX 5080 latency by sequence length. |
| `/home/ssu/kv_cache_consumer_gpu_bench/plots_5080/peak_delta_memory_vs_seq_len.png` | Qwen2.5 RTX 5080 peak allocated delta by sequence length. |
| `/home/ssu/kv_cache_consumer_gpu_bench/plots_5080/actual_kv_vs_theoretical_kv.png` | Qwen2.5 actual vs theoretical KV tensor footprint. |
| `/home/ssu/kv_cache_consumer_gpu_bench/plots_5080/kv_actual_over_theory_vs_seq_len.png` | Qwen2.5 actual/theory sanity plot. |
| `analysis/outputs/dataset_stability/opt-350m/plots/*.svg` | OPT-350M threshold stability plots, adjacent but not the main KV-cache boundary evidence. |

## 6. README / Result Summary Sections

| Path | Relevant Sections |
| --- | --- |
| `README.md` | Section 6, KV-cache Capacity Boundary Sweep; contains planned commands and interpretation rules. |
| `results/oaken_consumer_gpu_summary.md` | Oaken accuracy artifact scope, hardware, reproduction status, main results, limitations. |
| `analysis/outputs/kv_cache_pressure/rtx5080/summary.md` | RTX 5080 OPT-1.3B sequence-length pressure summary. |
| `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_qwen25_analysis.md` | Main RTX 5080 Qwen2.5 cache-mode benchmark analysis. |

## 7. Evidence Mapping

| Topic | Supporting files | Status |
| --- | --- | --- |
| RTX 5060 OPT-1.3B result | `results/oaken_consumer_gpu_summary.csv`, `results/rtx5060/opt-1.3b/summary.md` | Oaken accuracy/VRAM result exists; dynamic/quantized rescue CSV not found. |
| RTX 5060 Qwen result | not found in repository | Missing or not yet exported to CSV. |
| RTX 5080 result | `/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`, analysis CSV/MD files | Strong file-backed evidence. |
| dynamic vs quantized vs offloaded vs no_cache | `/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`, `analysis/rtx5080_ratios_vs_dynamic.csv` | Strong for RTX 5080 Qwen2.5. |
| theoretical vs actual KV-cache size check | `/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`, `analysis/rtx5080_kv_theory_check.csv` | Strong for RTX 5080 Qwen2.5. |

## 8. Missing or Ambiguous Files

- `results/rtx5060_opt13b_dynamic_boundary.csv`: not found in repository.
- `results/rtx5060_opt13b_rescue_cases.csv`: not found in repository.
- `results/rtx5060_qwen25_15b_dynamic_boundary.csv`: not found in repository.
- `results/rtx5060_qwen25_15b_rescue_cases.csv`: not found in repository.
- `results/plots_rtx5060_combined/dynamic_oom_rescue_cases.csv`: not found in repository.
- Current `oaken` README contains future/planned sweep commands, but corresponding CSVs are not present.

## Korean Summary

현재 파일로 강하게 뒷받침되는 증거는 두 축입니다. 첫째, `results/oaken_consumer_gpu_summary.csv`는 RTX 5060/5080에서 Oaken-inspired accuracy/VRAM artifact 결과를 제공합니다. 둘째, `/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`는 RTX 5080에서 Qwen2.5-1.5B의 dynamic/quantized/offloaded/no_cache 비교와 KV 이론식 검증을 제공합니다. 반면, RTX 5060에서 dynamic OOM을 quantized가 rescue했다는 핵심 주장은 현재 저장소 CSV로 확인되지 않으므로 발표에서는 “추가 확인 필요”로 낮춰야 합니다.
