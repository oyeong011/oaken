# Project Instructions for Codex

## Project identity

This repository contains my own KV-cache / LLM inference memory-pressure experiments on consumer NVIDIA GPUs, mainly RTX 5060 8GB and RTX 5080.

The purpose is to prepare:
1. a KAIRI / KAIST AI Research Internship briefing,
2. a lab-meeting briefing for tomorrow,
3. a concise technical narrative based strictly on actual experiment artifacts.

## Non-negotiable rules

- Do not invent numbers.
- Do not claim full reproduction of the Oaken paper unless the repository directly proves it.
- Use the phrase "Oaken-inspired" or "Oaken-style KV-cache quantization experiment" when appropriate.
- Distinguish clearly between:
  - measured results,
  - theoretical KV-cache formula,
  - implementation limitation,
  - interpretation,
  - future work.
- If README, CSV, plots, and scripts disagree, report the discrepancy instead of silently choosing one.
- Any table or claim must cite the local file path it came from.
- Korean output is preferred for KAIRI/lab-meeting material.
- Keep the tone technical, honest, and defensible.

## Expected final artifacts

Create or update these files if possible:

- docs/00_experiment_inventory.md
- docs/01_kv_cache_experiment_brief.md
- docs/02_kairi_briefing.md
- docs/03_lab_meeting_talk.md
- docs/04_defense_qna.md
- docs/05_slide_outline.md
- docs/06_one_page_summary.md

## Known experiment context from the researcher

Use this only as initial context. Verify against repository files.

- Main theme: consumer-GPU KV-cache memory pressure during LLM inference.
- Motivation: LLM inference becomes memory-capacity and memory-bandwidth constrained as sequence length and batch size grow.
- Theoretical KV-cache size formula:
  KV bytes = 2 * num_layers * batch_size * sequence_length * hidden_size * bytes_per_element
  where 2 accounts for K and V.
- For GQA/MQA models, the formula must use key/value heads rather than blindly using full attention heads.
- Relevant modes include dynamic, quantized, offloaded, and no_cache.
- no_cache should be treated as an ablation / lower-bound reference, not as a practical serving solution.
- offloaded may reduce GPU memory pressure but can stress host memory and transfer overhead.

Known RTX 5080 result summary from prior notes:
- 80 total cases / 76 OK / 4 OOM.
- OOM boundary observed at batch_size=8, seq_len=8192.
- kv_actual_over_theory = 1.0.
- Relative throughput vs dynamic:
  - quantized: 0.744x
  - offloaded: 0.594x
  - no_cache: 0.094x
- Relative peak memory delta vs dynamic:
  - quantized: 0.786x
  - offloaded: 0.705x
  - no_cache: 0.739x

Known RTX 5060 OPT-1.3B result summary from prior notes:
- dynamic OOM at batch=4 seq_len=8192.
- dynamic OOM at batch=8 seq_len=4096, 6144, 8192.
- quantized rescue succeeded for batch=4 seq_len=8192 and batch=8 seq_len=4096.
- quantized still OOM at batch=8 seq_len=6144 and 8192.
- offloaded hit host RAM pressure on a 15 GiB RAM / no-swap system and caused kernel OOM kill.
- no_cache should be described as lower-bound / ablation.

Known RTX 5060 Qwen sanity summary from prior notes:
- position_valid=True.
- kv_formula_type=gqa_mqa.
- max_position_embeddings=32768.
- dynamic kv_actual_over_theory=1.0.
- dynamic OOM at batch=8 seq_len=12288 and 16384.
- quantized rescue succeeded for both listed OOM cases.

Important output files that may exist:
- results/rtx5060_opt13b_dynamic_boundary.csv
- results/rtx5060_opt13b_rescue_cases.csv
- results/plots_rtx5060_combined/peak_memory_vs_seq_len.png
- results/plots_rtx5060_combined/throughput_vs_peak_memory.png
- results/status_boundary_matrix.csv
- results/oom_cases.csv
- results/dynamic_oom_rescue_cases.csv
- README.md
- oaken/scripts/run_kv_cache_sweep.py
- oaken/scripts/plot_kv_cache_sweep.py

## Style for presentation

Use the following framing:

"저는 모델 구조 자체보다는 LLM inference가 실제 GPU 메모리 계층에서 어떻게 병목을 만드는지에 관심을 두고, consumer GPU 환경에서 KV-cache 증가와 VRAM pressure를 직접 계측했습니다."

Avoid overclaiming:

Bad:
- "Oaken을 완전히 재현했다."
- "KV-cache quantization이 항상 성능을 개선한다."
- "offloading이 메모리 문제를 해결한다."

Better:
- "Oaken 논문을 동기로 삼아, consumer GPU에서 KV-cache quantization/offloading/no-cache의 메모리-성능 trade-off를 관찰했습니다."
- "실험 결과, quantized cache는 일부 OOM boundary에서 rescue 가능성을 보였지만, 큰 batch/long-context에서는 여전히 한계가 남았습니다."
- "offloading은 GPU VRAM pressure는 줄일 수 있지만, host memory pressure와 전송 비용을 새 병목으로 만들 수 있었습니다."
