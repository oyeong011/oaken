# Consumer GPU 기반 LLM KV-cache memory pressure 실험 요약

## 1. 연구 관심

저는 모델 구조 자체보다 LLM inference가 실제 GPU memory hierarchy에서 어떤 병목을 만드는지에 관심이 있다. 이 repo의 실험은 Oaken full reproduction이 아니라 Oaken-inspired KV-cache memory-pressure 관찰이다. 주요 근거 파일은 `results/rtx5060_qwen25_15b_dynamic_boundary.csv`, `results/rtx5060_qwen25_15b_rescue_cases.csv`, `results/rtx5060_opt13b_dynamic_boundary.csv`이다.

## 2. 문제의식

Autoregressive inference는 이전 token의 key/value를 layer별로 저장한다. 이 KV-cache는 sequence length와 batch size에 따라 커져 consumer GPU의 VRAM boundary를 만들 수 있다. 따라서 dynamic cache가 언제 OOM이 나는지, quantized/no_cache가 어떤 의미를 갖는지 측정했다. no_cache는 practical serving solution이 아니라 lower-bound / ablation이다.

## 3. 실험 설정

RTX 5060 8GB에서 OPT-1.3B memory stress sweep과 Qwen2.5-1.5B-Instruct position-valid sweep을 수행했다. Qwen sanity 결과는 `position_valid=True`, `max_position_embeddings=32768`, `kv_formula_type=gqa_mqa`, dynamic `kv_actual_over_theory=1.0`이다 (`results/rtx5060_qwen25_15b_sanity.csv`).

## 4. 핵심 결과

1. Qwen dynamic cache는 `batch=8`, `seq_len=12288/16384`에서 OOM이었다 (`results/rtx5060_qwen25_15b_dynamic_boundary.csv`).  
2. 같은 두 조건은 quantized cache로 모두 OK였다 (`results/rtx5060_qwen25_15b_rescue_cases.csv`).  
3. Quantized KV footprint ratio는 0.288737, 0.286865였다 (`results/rtx5060_qwen25_15b_rescue_cases.csv`).

## 5. 해석

Qwen 결과는 OPT stress 결과보다 강하다. OPT는 position limit 밖 stress로 해석해야 하지만, Qwen은 32768 context 안에서 12288/16384를 사용했으므로 position-valid long-context evidence이다. Quantized cache는 speedup claim이 아니라 memory-capacity optimization으로 해석해야 한다. 0.287 ratio는 KV-cache footprint 비율이지 total CUDA peak memory 감소율이 아니다.

## 6. 한계

1. Oaken paper full reproduction이 아니다.  
2. Qwen sweep에는 품질/perplexity 평가가 없다.  
3. RTX 5080 Qwen cross-GPU cache-policy CSV는 현재 repo에 없다.  
이 한계들은 `README.md`, `docs/00_experiment_inventory.md`, `docs/01_kv_cache_experiment_brief.md`에 명시했다.

## 7. 다음 계획

가장 중요한 다음 실험은 RTX 5080에서 같은 Qwen2.5-1.5B 조건을 재실행하는 것이다. 5060에서 dynamic OOM이던 `8x12288`, `8x16384`가 5080에서 dynamic OK로 바뀌는지 확인한다. 그 다음 5080 자체 boundary를 더 큰 batch/seq에서 찾고 quantized rescue를 반복한다.

## 8. KAIRI / 연구실에서 더 배우고 싶은 점

이 실험은 작은 consumer GPU 계측이지만, AI systems 연구에서 필요한 가설 설정, 측정, 실패 케이스 정리의 연습이었다. KAIRI/연구실에서는 이런 실험을 더 재현 가능하고 엄밀한 benchmark로 만들고, serving latency, quality, scheduling까지 연결하는 방법을 배우고 싶다.

**Spoken closing:**  
제가 보여주고 싶은 것은 큰 모델을 학습했다는 것이 아니라, LLM inference에서 실제 memory bottleneck을 계측하고 방어 가능한 evidence로 정리했다는 점입니다. 다음 단계에서는 RTX 5080 cross-GPU 비교와 품질/latency 측정을 추가해 systems 연구에 더 가깝게 확장하고 싶습니다.
