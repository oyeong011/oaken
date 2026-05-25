# Consumer GPU 기반 LLM KV-cache memory pressure 실험 요약

## 1. 연구 관심

저는 모델 구조 자체보다 LLM inference가 실제 GPU memory hierarchy에서 어떤 병목을 만드는지에 관심이 있다. 이 repo의 실험은 Oaken full reproduction이 아니라 Oaken-inspired KV-cache memory-pressure 관찰이다. 5060은 Qwen/OPT cache-policy sweep 근거이고, 5080은 OPT Oaken-style Wikitext accuracy path와 VRAM boundary 근거이다.

## 2. 문제의식

Autoregressive inference는 이전 token의 key/value를 layer별로 저장한다. 이 KV-cache는 sequence length와 batch size에 따라 커져 consumer GPU의 VRAM boundary를 만들 수 있다. 따라서 dynamic cache가 언제 OOM이 나는지, quantized/no_cache가 어떤 의미를 갖는지 측정했다. no_cache는 practical serving solution이 아니라 lower-bound / ablation이다.

## 3. 실험 설정

RTX 5060 8GB에서는 OPT-1.3B memory stress sweep과 Qwen2.5-1.5B-Instruct position-valid cache-policy sweep을 수행했다. RTX 5080에서는 OPT-125M~OPT-6.7B Oaken-style Wikitext/VRAM boundary artifact가 있다 (`results/oaken_consumer_gpu_summary.csv`). Qwen sanity는 `position_valid=True`, `max_position_embeddings=32768`, `kv_formula_type=gqa_mqa`이다 (`results/rtx5060_qwen25_15b_sanity.csv`).

## 4. 핵심 결과

1. Qwen dynamic cache는 `batch=8`, `seq_len=12288/16384`에서 OOM이고, quantized cache는 두 조건 모두 OK였다 (`results/rtx5060_qwen25_15b_dynamic_boundary.csv`, `results/rtx5060_qwen25_15b_rescue_cases.csv`).
2. Quantized KV footprint ratio는 0.288737, 0.286865였다 (`results/rtx5060_qwen25_15b_rescue_cases.csv`).
3. RTX 5080 OPT-125M~2.7B는 Oaken Wikitext eval 완료, OPT-6.7B는 약 15.8GB peak VRAM 근처에서 Oaken eval CUDA OOM boundary였다 (`results/oaken_consumer_gpu_summary.csv`, `results/rtx5080/opt-6.7b/summary.md`).

## 5. 해석

Qwen 결과는 position-valid long-context cache-policy evidence이고, OPT-1.3B는 memory stress evidence이다. RTX 5080 결과는 Qwen cache-policy sweep이 아니라 Oaken-style OPT accuracy/boundary evidence이다. Quantized cache는 speedup claim이 아니라 memory-capacity optimization으로 해석해야 한다. 5080은 memory-boundary 문제를 제거한 것이 아니라 더 큰 model size로 boundary를 이동시킨 것으로 해석한다.

## 6. 한계

1. Oaken paper full reproduction이 아니다.
2. Qwen sweep에는 품질/perplexity 평가가 없다.
3. RTX 5080 Qwen cross-GPU cache-policy CSV는 현재 repo에 없다.
이 한계들은 `README.md`, `docs/00_experiment_inventory.md`, `docs/01_kv_cache_experiment_brief.md`에 명시했다.

## 7. 다음 계획

가장 중요한 다음 실험은 RTX 5080에서 같은 Qwen2.5-1.5B cache-policy 조건을 재실행하는 것이다. 5060에서 dynamic OOM이던 `8x12288`, `8x16384`가 5080에서 dynamic OK로 바뀌는지 확인한다. 그 다음 5080 Qwen 자체 boundary를 찾고, quality/perplexity와 prefill/decode latency를 추가한다.

## 8. KAIRI / 연구실에서 더 배우고 싶은 점

이 실험은 작은 consumer GPU 계측이지만, AI systems 연구에서 필요한 가설 설정, 측정, 실패 케이스 정리의 연습이었다. KAIRI/연구실에서는 이런 실험을 더 재현 가능하고 엄밀한 benchmark로 만들고, serving latency, quality, scheduling까지 연결하는 방법을 배우고 싶다.

**Spoken closing:**
제가 보여주고 싶은 것은 큰 모델을 학습했다는 것이 아니라, LLM inference에서 실제 memory bottleneck을 계측하고 방어 가능한 evidence로 정리했다는 점입니다. 다음 단계에서는 RTX 5080 Qwen cache-policy 비교와 품질/latency 측정을 추가해 systems 연구에 더 가깝게 확장하고 싶습니다.
