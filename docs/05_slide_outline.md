# Slide Outline

## Slide 1. Title

- **Main message:** Consumer GPU에서 LLM inference KV-cache memory pressure를 계측했다.
- **Bullets:** Oaken-inspired, RTX 5060/5080, cache mode trade-off.
- **Suggested figure/table:** 없음.
- **Source:** `docs/01_kv_cache_experiment_brief.md`
- **Speaker note:** full Oaken reproduction이 아니라고 처음부터 명확히 말한다.

## Slide 2. Motivation

- **Main message:** Long-context serving은 memory system 문제다.
- **Bullets:** KV-cache grows with sequence/batch; VRAM boundary; serving policy trade-off.
- **Suggested figure/table:** KV formula text.
- **Source:** `docs/01_kv_cache_experiment_brief.md`
- **Speaker note:** 선형 증가 자체가 아니라 boundary와 policy가 핵심이라고 말한다.

## Slide 3. KV-cache Background

- **Main message:** KV-cache는 재계산을 줄이지만 memory를 누적한다.
- **Bullets:** K/V per layer, autoregressive decode, MHA vs GQA formula.
- **Suggested figure/table:** formula table.
- **Source:** `scripts/run_kv_cache_sweep.py`, `/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`
- **Speaker note:** Qwen2.5는 KV heads=2, attention heads=12.

## Slide 4. Experiment Setup

- **Main message:** 두 evidence stream을 분리한다.
- **Bullets:** Oaken artifact summary; RTX 5080 Qwen cache sweep; missing RTX 5060 rescue CSV.
- **Suggested figure/table:** inventory table.
- **Source:** `docs/00_experiment_inventory.md`
- **Speaker note:** 없는 파일은 없다고 말한다.

## Slide 5. Theory vs Actual KV

- **Main message:** RTX 5080 Qwen에서 actual/theory sanity 통과.
- **Bullets:** `kv_actual_over_theory=1.0`; GQA-aware formula.
- **Suggested figure/table:** `actual_kv_vs_theoretical_kv.png`
- **Source file:** `/home/ssu/kv_cache_consumer_gpu_bench/plots_5080/actual_kv_vs_theoretical_kv.png`
- **Speaker note:** contribution이 아니라 measurement sanity.

## Slide 6. OOM Boundary

- **Main message:** RTX 5080 Qwen에서 최대 grid point가 OOM.
- **Bullets:** 80 rows; 76 OK; 4 OOM; B=8/S=8192.
- **Suggested figure/table:** OOM cases table.
- **Source file:** `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_oom_cases.csv`
- **Speaker note:** quantized도 해당 point에서는 rescue 실패.

## Slide 7. Throughput vs Sequence Length

- **Main message:** cache mode별 throughput cost가 다르다.
- **Bullets:** dynamic baseline; quantized moderate cost; offloaded larger cost; no_cache collapse.
- **Suggested figure/table:** throughput plot.
- **Source file:** `/home/ssu/kv_cache_consumer_gpu_bench/plots_5080/throughput_vs_seq_len.png`
- **Speaker note:** no_cache는 practical option이 아니다.

## Slide 8. Peak Memory vs Sequence Length

- **Main message:** memory delta는 cache mode에 따라 달라진다.
- **Bullets:** quantized/offloaded lower peak delta; throughput penalty.
- **Suggested figure/table:** peak memory plot.
- **Source file:** `/home/ssu/kv_cache_consumer_gpu_bench/plots_5080/peak_delta_memory_vs_seq_len.png`
- **Speaker note:** memory만 보고 좋은 방법이라고 말하지 않는다.

## Slide 9. Representative Long-context Case

- **Main message:** B=4/S=8192에서 policy trade-off가 선명하다.
- **Bullets:** dynamic 124.957829 tokens/s; quantized 108.237820; offloaded 45.451941; no_cache 2.951665.
- **Suggested figure/table:** key examples table.
- **Source file:** `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_key_examples.csv`
- **Speaker note:** exact values are file-backed.

## Slide 10. Oaken Artifact Context

- **Main message:** Oaken artifact는 accuracy/VRAM boundary context를 제공한다.
- **Bullets:** RTX 5060/5080 OPT summary; 5080 OPT-6.7B boundary.
- **Suggested figure/table:** Oaken summary table.
- **Source file:** `results/oaken_consumer_gpu_summary.csv`
- **Speaker note:** HF Qwen sweep과 같은 실험으로 섞지 않는다.

## Slide 11. Limitations

- **Main message:** 현재 주장의 강도는 제한적이다.
- **Bullets:** no full Oaken reproduction; 5060 Qwen missing; quality missing; no repeated statistics.
- **Suggested figure/table:** limitations table.
- **Source:** `docs/04_defense_qna.md`
- **Speaker note:** 먼저 인정하면 방어가 쉬워진다.

## Slide 12. Next Steps

- **Main message:** 다음은 position-valid 5060 Qwen rescue를 file-backed로 만드는 것.
- **Bullets:** dynamic boundary CSV; quantized/no_cache rescue; fixed decode length; prefill/decode split.
- **Suggested figure/table:** Missing plot: RTX 5060 Qwen combined `dynamic_oom_rescue_cases.csv`.
- **Source:** generate with `scripts/run_kv_cache_sweep.py` and `scripts/plot_kv_cache_sweep.py`.
- **Speaker note:** 내일 발표에서는 future work로 둔다.
