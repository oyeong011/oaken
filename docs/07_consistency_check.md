# Consistency Check

검증 기준은 repository 안의 `README.md`, `docs/00_experiment_inventory.md`부터 `docs/06_one_page_summary.md`, 그리고 `results/` 아래 CSV/summary file이다. Prior note에만 있고 repository 안에서 source file을 찾지 못한 숫자는 확정 claim으로 쓰지 않았다.

## 1. Numerical claims table

| Claim | Appears in | Source CSV / file | Consistent? | Notes |
|---|---|---|---|---|
| Qwen sanity rows have `position_valid=True` | README, docs/00-06 | `results/rtx5060_qwen25_15b_sanity.csv` | Yes | All 4 sanity rows record `position_valid=True`. |
| Qwen `max_position_embeddings=32768` | README, docs/00-06 | `results/rtx5060_qwen25_15b_sanity.csv` | Yes | Used to frame 12288/16384 as position-valid. |
| Qwen GQA/MQA metadata is `num_attention_heads=12`, `num_key_value_heads=2`, `head_dim=128` | docs/01, docs/03, docs/04 | `results/rtx5060_qwen25_15b_sanity.csv` | Yes | Supports key/value-head formula instead of naive full-head formula. |
| Qwen `kv_formula_type=gqa_mqa` | README, docs/00-06 | `results/rtx5060_qwen25_15b_sanity.csv` | Yes | All sanity rows agree. |
| Qwen dynamic `kv_actual_over_theory=1.0` | README, docs/01-06 | `results/rtx5060_qwen25_15b_sanity.csv` | Yes | CSV value is `1.000000`; docs round to `1.0`. |
| Qwen dynamic boundary file has 24 rows | docs/00 | `results/rtx5060_qwen25_15b_dynamic_boundary.csv` | Yes | 4 batch sizes x 6 sequence lengths. |
| Qwen dynamic OOM at `batch=8`, `seq_len=12288` and `16384` | README, docs/01-06 | `results/rtx5060_qwen25_15b_dynamic_boundary.csv` | Yes | Both rows have `status=OOM` and `oom=True`. |
| Qwen dynamic `batch=8`, `seq_len=8192` is OK | docs/05 | `results/rtx5060_qwen25_15b_dynamic_boundary.csv` | Yes | Row has `status=OK` and `oom=False`. |
| Qwen quantized rescues `8x12288` and `8x16384` | README, docs/01-06 | `results/rtx5060_qwen25_15b_rescue_cases.csv` | Yes | Both quantized rows have `status=OK` and `oom=False`. |
| Qwen quantized `kv_actual_over_theory` is 0.288737 and 0.286865 | README, docs/01, docs/03, docs/06 | `results/rtx5060_qwen25_15b_rescue_cases.csv` | Yes | Interpreted only as KV tensor footprint ratio, not total CUDA memory reduction. |
| OPT dynamic boundary file has 20 rows | docs/00, docs/01 | `results/rtx5060_opt13b_dynamic_boundary.csv` | Yes | 4 batch sizes x 5 sequence lengths. |
| OPT dynamic OOM at `4x8192`, `8x4096`, `8x6144`, `8x8192` | README, docs/01, docs/03 | `results/rtx5060_opt13b_dynamic_boundary.csv` | Yes | OPT CSV has no `oom` column; consistency checked by `status=OOM`. |
| OPT quantized OK at `4x8192` and `8x4096` | README, docs/01, docs/03 | `results/rtx5060_opt13b_rescue_cases.csv` | Yes | Both rows have `status=OK`. |
| OPT quantized OOM at `8x6144` and `8x8192` | README, docs/01, docs/03 | `results/rtx5060_opt13b_rescue_cases.csv` | Yes | Both rows have `status=OOM`. |
| Offloaded RTX 5060 attempt hit 15 GiB RAM / no swap host-memory pressure | README, docs/01-04 | `README.md` | Partially | README records this limitation; no valid offloaded result row exists in the inspected result CSVs. |
| RTX 5080 OPT-125M, OPT-350M, OPT-1.3B, OPT-2.7B completed Oaken Wikitext evaluation | docs/01, docs/02, docs/03, docs/05, docs/06 | `results/oaken_consumer_gpu_summary.csv`, `results/rtx5080/opt-125m/summary.md`, `results/rtx5080/opt-350m/summary.md`, `results/rtx5080/opt-1.3b/summary.md`, `results/rtx5080/opt-2.7b/summary.md` | Yes | This is Oaken-style Wikitext/OPT evidence, not Qwen cache-policy sweep evidence. |
| RTX 5080 OPT-6.7B is the upper boundary case | docs/01, docs/02, docs/03, docs/05, docs/06 | `results/oaken_consumer_gpu_summary.csv`, `results/rtx5080/opt-6.7b/summary.md` | Yes | Summary marks status `Boundary`; per-run summary says Oaken eval failed with CUDA OOM. |
| RTX 5080 OPT-6.7B original eval and profiling completed only with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | docs/01, docs/03, docs/05 | `results/oaken_consumer_gpu_summary.md`, `results/rtx5080/opt-6.7b/summary.md` | Yes | Without allocator tuning, original FP16 eval failed near the same boundary. |
| RTX 5080 OPT-6.7B original peak 15806 MiB and Oaken eval peak 15826 MiB | docs/01 | `results/rtx5080/opt-6.7b/summary.md` | Yes | This is repo-backed 5080 OPT evidence, not Qwen cache-policy evidence. |
| Prior-note RTX 5080 Qwen `80 total / 76 OK / 4 OOM` and throughput ratios | docs/00, docs/01 | Not found in this repository | Not claimed | Docs explicitly mark these as missing from `/home/ssu/oaken`. |

## 2. Terminology check

| Risk area | Check result |
|---|---|
| Full Oaken reproduction | The docs use "Oaken-inspired" or explicitly say this is not full paper reproduction. |
| Universal quantization improvement | The docs frame quantized cache as a memory-capacity trade-off and explicitly reject universal speedup claims. |
| Offloading as complete solution | The docs describe offloading as a trade-off that may move pressure to host memory and transfer overhead. |
| no_cache as practical serving method | The docs consistently call `no_cache` a lower-bound / ablation, not a practical serving policy. |
| Total memory reduction overclaim | Qwen 0.288737 / 0.286865 is described as KV-cache tensor footprint ratio, not total CUDA peak memory reduction. |
| OPT long-context overclaim | OPT >2048 sequence results are described as memory stress evidence, while Qwen is used for position-valid long-context evidence. |
| RTX 5080 evidence type overclaim | Docs separate RTX 5060 cache-policy sweep from RTX 5080 OPT Oaken-style Wikitext/VRAM boundary evidence. |

## 3. Missing evidence

1. RTX 5080 Qwen cache-policy sweep CSV is not present under this repository, so the prior-note `80 total / 76 OK / 4 OOM` and ratio claims are not used as file-backed claims.
2. Qwen quality, perplexity, or accuracy degradation under quantized cache is not measured in the inspected files.
3. Prefill and decode latency are not separated in the inspected sweep outputs.
4. A valid RTX 5060 offloaded rescue CSV row is missing; the repository only supports a host-memory limitation statement through `README.md`.
5. Transfer overhead / PCIe bandwidth measurements for offloading are not present.
6. Exact environment lockfile or complete package version manifest was not found in the inspected docs/results.

## 4. Fixes applied

Updated the KAIRI and lab-meeting documents to separate evidence types: RTX 5060 is now framed as the dynamic/quantized/no_cache cache-policy sweep with Qwen position-valid rescue, while RTX 5080 is framed as Oaken-style Wikitext accuracy and OPT model-size VRAM boundary evidence. Added the RTX 5080 slide wording "Oaken-style accuracy path and 16GB boundary" and removed/avoided any implication that RTX 5080 has repo-backed Qwen cache-policy sweep results. The unsupported prior-note RTX 5080 `80 total / 76 OK / 4 OOM` numbers remain explicitly non-file-backed.

## 5. Final confidence

**MOSTLY READY**

RTX 5060 Qwen position-valid dynamic OOM / quantized rescue evidence is strong and file-backed. RTX 5060 OPT memory-stress evidence is also consistent with the CSVs. RTX 5080 evidence is file-backed for the Oaken-style OPT Wikitext accuracy path and OPT-6.7B 16GB-class boundary, but not for a Qwen cache-policy sweep. Readiness remains MOSTLY READY until RTX 5080 Qwen cache-policy CSV, quality/perplexity for Qwen quantized cache, and prefill/decode latency separation are added.
