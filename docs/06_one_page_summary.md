# Consumer GPU 기반 LLM KV-cache memory pressure 실험 요약

## 1. 연구 관심

저는 모델 구조 자체보다 LLM inference가 실제 GPU memory hierarchy에서 어떻게 병목을 만드는지에 관심이 있습니다. 특히 long-context serving과 batching에서 KV-cache가 VRAM pressure를 만드는 지점을 계측하고 싶었습니다.

## 2. 문제의식

KV-cache는 autoregressive decoding에서 과거 token의 key/value를 재사용하게 해주지만, sequence length와 batch size가 커질수록 memory footprint가 증가합니다. Consumer GPU에서는 이 증가가 OOM boundary로 직접 나타날 수 있습니다.

## 3. 실험 설정

파일로 확인되는 주요 결과는 RTX 5080에서 Qwen/Qwen2.5-1.5B-Instruct를 fp16으로 dynamic, quantized, offloaded, no_cache mode에서 sweep한 결과입니다. 출처는 `/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`입니다.

## 4. 핵심 결과

RTX 5080 Qwen sweep은 80 rows 중 76 OK, 4 OOM입니다. OOM은 모든 cache mode에서 batch_size=8, seq_len=8192에 발생했습니다. 성공 row의 `kv_actual_over_theory`는 1.0입니다. 출처는 `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_status_count.csv`, `rtx5080_oom_cases.csv`, `rtx5080_qwen25_analysis.md`입니다.

## 5. 해석

이 결과는 KV-cache 크기 계산이 실제 tensor footprint와 맞는다는 sanity check를 제공하고, cache mode가 memory와 throughput 사이의 trade-off를 만든다는 점을 보여줍니다. Quantized/offloaded는 universal speedup이 아니라 capacity technique으로 보는 것이 안전합니다.

## 6. 한계

Full Oaken reproduction이 아닙니다. RTX 5060 Qwen dynamic/rescue CSV가 현재 저장소에 없습니다. HF Qwen cache sweep에는 quality/perplexity 평가와 prefill/decode 분리 측정이 없습니다.

## 7. 다음 계획

RTX 5060에서 Qwen2.5 position-valid long-context dynamic boundary와 quantized rescue를 CSV로 남기겠습니다. 이후 fixed decode length, quality metric, prefill/decode timing을 추가해 더 공정한 comparison으로 확장하겠습니다.

## 8. KAIRI / 연구실에서 더 배우고 싶은 점

개인 실험을 넘어서 재현 가능한 methodology, rigorous measurement, systems-level interpretation을 배우고 싶습니다. LLM inference serving에서 memory capacity, bandwidth, scheduling, cache policy를 함께 고려하는 연구 방향을 더 배우고 싶습니다.

## Spoken Closing

제가 한 작업은 Oaken을 완전히 재현했다는 주장이 아니라, Oaken의 문제의식인 KV-cache memory pressure를 consumer GPU에서 직접 관찰한 준비 실험입니다. 앞으로는 이 실험을 더 엄밀한 AI systems/infrastructure 연구 질문으로 발전시키고 싶습니다.
