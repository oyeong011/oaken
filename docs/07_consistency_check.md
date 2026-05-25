# Consistency Check

## 1. Numerical Claims Table

| Claim | Appears in | Source CSV/MD | Consistent? | Notes |
| --- | --- | --- | --- | --- |
| RTX 5080 Qwen sweep has 80 rows, 76 OK, 4 OOM | docs 01/02/03/06 | `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_status_count.csv` | Yes | 76+4=80. |
| RTX 5080 Qwen OOM at B=8, S=8192 for all modes | docs 01/03/06 | `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_oom_cases.csv` | Yes | dynamic/offloaded/no_cache/quantized all listed. |
| `kv_actual_over_theory=1.0` for successful RTX 5080 Qwen rows | docs 01/02/03/06 | `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_qwen25_analysis.md` | Yes | analysis table reports mean/min/max 1.0. |
| Quantized throughput ratio 0.744150 | docs 01/02/03 | `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_ratios_vs_dynamic.csv` | Yes | Same exact value used. |
| Offloaded throughput ratio 0.594451 | docs 01/02/03 | `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_ratios_vs_dynamic.csv` | Yes | Same exact value used. |
| no_cache throughput ratio 0.093942 | docs 01/02/03 | `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_ratios_vs_dynamic.csv` | Yes | Same exact value used. |
| RTX 5060 OPT-1.3B dynamic OOM/rescue | AGENTS prior context only | not found in repository | Not supported | Docs weaken this to missing evidence. |
| RTX 5060 Qwen dynamic OOM/rescue | AGENTS prior context only | not found in repository | Not supported | Docs weaken this to missing evidence. |

## 2. Terminology Check

| Risk | Status |
| --- | --- |
| Full Oaken reproduction overclaim | Avoided. Docs use "Oaken-inspired" and explicitly say not full reproduction. |
| Universal quantization improvement | Avoided. Docs frame quantized cache as memory-throughput trade-off. |
| Offloading as complete solution | Avoided. Docs frame offloading as GPU/host/transfer trade-off. |
| no_cache as practical serving method | Avoided. Docs call no_cache ablation/lower-bound. |

## 3. Missing Evidence

- `results/rtx5060_opt13b_dynamic_boundary.csv`: missing.
- `results/rtx5060_opt13b_rescue_cases.csv`: missing.
- `results/rtx5060_qwen25_15b_dynamic_boundary.csv`: missing.
- `results/rtx5060_qwen25_15b_rescue_cases.csv`: missing.
- `results/plots_rtx5060_combined/dynamic_oom_rescue_cases.csv`: missing.
- Qwen `position_valid=True`, `max_position_embeddings=32768` are not currently file-backed in a result CSV in `oaken`.

## 4. Fixes Applied

- Created `AGENTS.md` to prevent overclaiming and require file-backed numbers.
- Created docs using only file-backed strong claims.
- Weakened RTX 5060 OPT/Qwen rescue claims to "not found in repository" / "확인 필요".
- Separated Oaken artifact accuracy/VRAM results from HF Qwen cache-mode sweep results.

## 5. Final Confidence

**MOSTLY READY**

한국어 설명: RTX 5080 Qwen cache-mode 결과와 Oaken artifact summary는 파일 근거가 있어 발표에 사용할 수 있습니다. 다만 사용자가 가장 강조하고 싶은 RTX 5060 dynamic OOM/rescue와 Qwen position-valid rescue 결과는 현재 저장소 CSV로 확인되지 않았습니다. 따라서 내일 발표는 "5080 file-backed evidence + 5060 rescue future work/확인 필요" 프레임으로 가면 방어 가능하지만, 5060 rescue를 핵심 성과로 말하려면 수동으로 CSV를 확보해야 합니다.
