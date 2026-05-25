# Slide Outline

## Slide 1. Title

- **Main message:** Consumer GPU에서도 LLM KV-cache memory pressure는 직접 관찰 가능한 systems bottleneck이다.
- **Bullet points:** Oaken-inspired experiment; RTX 5060 cache-policy sweep; RTX 5080 Oaken-style OPT boundary; full reproduction 아님.
- **Suggested figure/table:** 없음.
- **Source file:** `README.md`
- **Speaker note:** Full Oaken reproduction이 아니라 memory-pressure experiment라고 먼저 선 긋기.

## Slide 2. Motivation

- **Main message:** Long-context inference는 cache memory 때문에 VRAM boundary에 부딪힌다.
- **Bullet points:** KV-cache grows with sequence length; grows with batch size; consumer GPUs have tight VRAM; cache policy matters.
- **Suggested figure/table:** KV formula text box.
- **Source file:** `scripts/run_kv_cache_sweep.py`, `results/rtx5060_qwen25_15b_sanity.csv`
- **Speaker note:** "왜 KV-cache인가"를 systems language로 설명.

## Slide 3. KV-cache Formula

- **Main message:** Qwen 같은 GQA/MQA 모델은 key/value heads 기준 공식이 필요하다.
- **Bullet points:** `num_attention_heads=12`; `num_key_value_heads=2`; `head_dim=128`; dynamic ratio 1.0.
- **Suggested figure/table:** Sanity metadata table.
- **Source file:** `results/rtx5060_qwen25_15b_sanity.csv`
- **Speaker note:** hidden_size-only 공식 과대평가를 피해야 한다고 강조.

## Slide 4. Method

- **Main message:** Dynamic boundary를 먼저 찾고, 실패 조건만 rescue한다.
- **Bullet points:** chunked cache growth; batch/seq sweep; dynamic then quantized/no_cache; plot/derived CSV generation.
- **Suggested figure/table:** experiment pipeline diagram.
- **Source file:** `scripts/run_kv_cache_sweep.py`, `scripts/plot_kv_cache_sweep.py`
- **Speaker note:** 처음부터 모든 policy를 섞지 않은 이유를 설명.

## Slide 5. OPT Stress Result

- **Main message:** OPT-1.3B는 memory stress evidence이지만 position-valid long-context claim은 아니다.
- **Bullet points:** dynamic OOM 4 cases; quantized rescue 2 cases; quantized still OOM 2 cases; offloaded host RAM issue.
- **Suggested figure/table:** OPT OOM/rescue table.
- **Source file:** `results/rtx5060_opt13b_dynamic_boundary.csv`, `results/rtx5060_opt13b_rescue_cases.csv`, `README.md`
- **Speaker note:** OPT max position limitation을 먼저 인정.

## Slide 6. Qwen Sanity Result

- **Main message:** Qwen experiment is position-valid and GQA-aware.
- **Bullet points:** `position_valid=True`; `max_position_embeddings=32768`; `kv_formula_type=gqa_mqa`; dynamic `kv_actual_over_theory=1.0`.
- **Suggested figure/table:** sanity CSV row table.
- **Source file:** `results/rtx5060_qwen25_15b_sanity.csv`
- **Speaker note:** 이 slide가 OPT stress 공격을 방어하는 핵심.

## Slide 7. Qwen Dynamic Boundary

- **Main message:** Dynamic cache fails under position-valid long-context/big-batch Qwen inference on RTX 5060.
- **Bullet points:** OOM at `8x12288`; OOM at `8x16384`; both position-valid; OK up to `8x8192`.
- **Suggested figure/table:** `peak_memory_vs_seq_len.png`
- **Source file:** `results/plots_rtx5060_qwen25_15b_combined/peak_memory_vs_seq_len.png`, `results/rtx5060_qwen25_15b_dynamic_boundary.csv`
- **Speaker note:** "정상 long-context 모델에서도 죽었다"가 핵심.

## Slide 8. Qwen Quantized Rescue

- **Main message:** Quantized cache rescues both dynamic OOM Qwen cases.
- **Bullet points:** `8x12288` dynamic OOM -> quantized OK; `8x16384` dynamic OOM -> quantized OK; KV footprint ratio about 28.7%; not total memory reduction.
- **Suggested figure/table:** rescue cases table.
- **Source file:** `results/plots_rtx5060_qwen25_15b_combined/dynamic_oom_rescue_cases.csv`, `results/rtx5060_qwen25_15b_rescue_cases.csv`
- **Speaker note:** "속도 개선"이 아니라 "memory-capacity rescue"라고 말하기.

## Slide 9. Throughput vs Memory

- **Main message:** Cache policies should be interpreted as trade-offs, not universal wins.
- **Bullet points:** Dynamic is fastest when it succeeds; quantized can rescue OOM; no_cache is ablation; offloaded missing valid row in 5060 sweep.
- **Suggested figure/table:** `throughput_vs_peak_memory.png`
- **Source file:** `results/plots_rtx5060_qwen25_15b_combined/throughput_vs_peak_memory.png`
- **Speaker note:** no_cache throughput caveat: chunked sweep is not real decode serving.

## Slide 10. RTX 5080: Oaken-style accuracy path and 16GB boundary

- **Main message:** RTX 5080 evidence is an OPT Oaken-style Wikitext/VRAM boundary artifact, not a Qwen cache-policy sweep.
- **Bullet points:** OPT-125M~2.7B: Oaken Wikitext evaluation completed; OPT-6.7B: original eval + profiling completed only with expandable allocator; OPT-6.7B Oaken eval still OOM near 15.8GB peak VRAM; 5080 shifts the memory boundary to larger models.
- **Suggested figure/table:** summary table from `results/oaken_consumer_gpu_summary.csv`.
- **Source file:** `results/oaken_consumer_gpu_summary.csv`, `results/oaken_consumer_gpu_summary.md`, `results/rtx5080/opt-125m/summary.md`, `results/rtx5080/opt-350m/summary.md`, `results/rtx5080/opt-1.3b/summary.md`, `results/rtx5080/opt-2.7b/summary.md`, `results/rtx5080/opt-6.7b/summary.md`
- **Speaker note:** 5080을 5060식 dynamic/quantized/no_cache sweep으로 말하지 말고, Oaken-style accuracy path와 16GB boundary로 설명.

## Slide 11. Limitations

- **Main message:** The current result is useful but bounded.
- **Bullet points:** not full Oaken reproduction; limited models/hardware; Qwen quality not measured; prefill/decode not separated; RTX 5080 Qwen cache-policy CSV missing; offloading host RAM issue.
- **Suggested figure/table:** limitation table.
- **Source file:** `docs/04_defense_qna.md`
- **Speaker note:** 한계를 먼저 인정하면 공격을 줄일 수 있음.

## Slide 12. Next Steps

- **Main message:** Cross-GPU Qwen cache-policy comparison remains the most important missing experiment.
- **Bullet points:** run Qwen 5080 recheck; find 5080 Qwen boundary; quantized rescue around boundary; add quality/perplexity and prefill/decode latency distribution.
- **Suggested figure/table:** planned comparison table.
- **Source file:** Missing plot: RTX 5080 Qwen combined plot. Generate after `results/rtx5080_qwen25_15b_combined.csv` exists using `scripts/plot_kv_cache_sweep.py`.
- **Speaker note:** 다음 질문이 명확해야 발표가 연구처럼 보임.
