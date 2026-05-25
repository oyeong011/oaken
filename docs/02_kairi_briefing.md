# KAIRI 브리핑: KV-cache memory pressure experiment

## 1. 30초 자기소개 버전

안녕하세요. 저는 숭실대학교 컴퓨터학부 4학년이고, AI model 자체보다는 AI system과 infrastructure에서 실제 병목이 어디서 생기는지에 관심이 있습니다. 최근에는 LLM inference에서 sequence length와 batch size가 커질 때 KV-cache가 GPU memory pressure를 어떻게 만드는지 consumer GPU에서 직접 계측했습니다. RTX 5060 8GB에서 Qwen2.5-1.5B-Instruct long-context 조건을 실험했고, dynamic cache가 OOM 나는 조건을 quantized cache가 rescue하는 사례를 확인했습니다. KAIRI에서는 이런 실험을 더 엄밀한 연구 방법론으로 발전시키는 법을 배우고 싶습니다.

## 2. 1분 연구경험 설명

처음에는 Oaken 논문이 제기한 KV-cache quantization 문제의식에서 출발했습니다. 다만 제가 한 것은 논문 전체 재현이 아니라, Oaken-inspired 방식으로 consumer GPU에서 KV-cache memory pressure를 직접 관찰한 실험입니다. RTX 5060 8GB에서 OPT-1.3B와 Qwen2.5-1.5B-Instruct를 대상으로 batch size와 sequence length를 바꾸며 dynamic, quantized, no_cache를 비교했습니다. Qwen 실험에서는 `position_valid=True`, `kv_formula_type=gqa_mqa`, dynamic `kv_actual_over_theory=1.0`을 확인한 뒤, `batch=8, seq_len=12288/16384`에서 dynamic OOM과 quantized rescue를 기록했습니다 (`results/rtx5060_qwen25_15b_*`). 배운 점은 KV-cache optimization은 단순 speedup이 아니라 memory-capacity trade-off로 보아야 한다는 것입니다. 부족한 점은 아직 5080 cross-GPU Qwen 비교와 품질/latency 분포 측정이 정리되지 않았다는 점입니다.

## 3. INA Research Group 정렬

INA는 AI/ML systems와 internet-scale service infrastructure를 다루는 연구실로 이해하고 있습니다. 제 실험은 model architecture 설계 연구가 아니라, 실제 LLM inference가 GPU memory capacity와 cache policy에 의해 어떻게 제한되는지를 계측한 작은 systems experiment입니다. Consumer GPU에서의 long-context/batch 실험은 비용 효율적인 serving, resource-aware scheduling, batching policy, memory-capacity limit과 연결됩니다. 특히 KV-cache pressure는 long-context serving에서 직접적인 병목이므로, dynamic/quantized/offloaded/no_cache의 trade-off를 관찰하는 것은 AI infrastructure 연구를 준비하는 구체적인 연습이라고 생각합니다. 다만 이것을 INA의 연구와 동등한 수준이라고 주장하지 않고, 제가 연구실에서 더 체계적으로 배우기 위한 준비 과정으로 설명하겠습니다.

## 4. 지원동기 문단

저는 LLM이 실제 서비스 환경에서 동작할 때 발생하는 memory capacity, memory bandwidth, batching, latency 같은 systems-level bottleneck에 관심이 있습니다. 최근 진행한 KV-cache 실험에서는 Oaken 논문을 동기로 삼아 consumer GPU에서 dynamic cache가 long-context/big-batch 조건에서 OOM boundary를 만들고, quantized cache가 일부 실패 조건을 rescue할 수 있음을 관찰했습니다. 이 과정에서 단순히 모델을 실행하는 것보다, 가설을 세우고 계측하고 실패 조건을 기록하며 해석의 강도를 조절하는 능력이 중요하다는 것을 느꼈습니다. KAIRI를 통해 AI systems 연구에서 요구되는 실험 설계, 재현성, 성능 분석 방법론을 더 엄밀하게 배우고 싶습니다.

## 5. 예상 질문과 답변

### Q1. 왜 KV-cache를 보았나?

KV-cache는 autoregressive decoding에서 이전 token의 key/value를 저장해 재계산을 줄이는 핵심 구조입니다. Sequence length와 batch size가 커질수록 cache memory가 커져 consumer GPU에서 OOM boundary를 만들 수 있기 때문에 systems 관점에서 중요합니다.

### Q2. Oaken을 정확히 재현한 것인가?

아닙니다. 이 repo는 Oaken paper full reproduction이 아니라 Oaken-inspired KV-cache quantization experiment입니다. README와 docs에서는 paper reproduction claim을 하지 않고, memory pressure와 cache-policy trade-off 관찰로 범위를 제한했습니다.

### Q3. quantized cache가 항상 좋은가?

아닙니다. Qwen 결과에서는 dynamic OOM 조건 두 개를 quantized가 rescue했지만, 이것을 universal speedup으로 해석하지 않습니다. Quantized cache는 memory-capacity optimization이며, latency/throughput cost가 있을 수 있습니다.

### Q4. offloading은 왜 한계가 있었나?

README에는 RTX 5060 host가 15 GiB RAM / no swap이었고 offloaded run이 host memory pressure로 killed되었다고 기록되어 있습니다. 따라서 offloading은 GPU VRAM pressure를 줄일 수 있지만, host memory와 transfer overhead를 새 병목으로 만들 수 있습니다.

### Q5. no_cache 결과는 무슨 의미인가?

no_cache는 practical serving policy가 아니라 lower-bound / ablation입니다. KV-cache 저장을 피하면 memory는 줄 수 있지만, 실제 autoregressive serving에서는 과거 context 재계산 때문에 throughput이 크게 나빠질 수 있습니다.

### Q6. 이론식과 실제값이 맞았다는 것이 왜 중요한가?

실험이 실제로 KV-cache tensor footprint를 측정하고 있는지 확인하는 sanity check입니다. Qwen dynamic rows에서 `kv_actual_over_theory=1.0`이 기록되어 있어 GQA/MQA 공식이 실제 cache tensor 크기와 맞음을 확인했습니다 (`results/rtx5060_qwen25_15b_sanity.csv`).

### Q7. Qwen에서는 왜 GQA/MQA formula sanity가 필요한가?

Qwen2.5-1.5B는 `num_attention_heads=12`, `num_key_value_heads=2`로 기록됩니다. Full attention heads 기준으로 계산하면 KV-cache 이론값을 과대평가하므로, key/value heads 기준의 GQA/MQA 공식 검증이 필요합니다.

### Q8. RTX 5060과 RTX 5080 비교에서 무엇을 배웠나?

현재 repo-backed Qwen cross-GPU 비교는 아직 없습니다. Repo 안의 RTX 5080 evidence는 OPT Oaken-style summary이며, OPT-6.7B에서 16GB-class boundary가 나타납니다 (`results/rtx5080/opt-6.7b/summary.md`). 다음 단계는 같은 Qwen sweep을 5080에서 실행해 boundary 이동을 보는 것입니다.

### Q9. 이 실험이 AI 대학원 연구와 무슨 관련이 있나?

AI systems 연구는 모델 accuracy뿐 아니라 serving cost, memory hierarchy, throughput, latency를 함께 다룹니다. 이 실험은 작은 규모지만 LLM inference 병목을 가설화하고 계측하고 실패 케이스를 정리한 systems-oriented 준비입니다.

### Q10. 다음 실험은 무엇인가?

Qwen2.5-1.5B를 RTX 5080에서 같은 조건으로 실행해 5060에서 OOM이던 `8x12288`, `8x16384`가 dynamic으로 살아나는지 확인하는 것입니다. 이후 5080 자체의 새로운 OOM boundary를 찾고 quantized rescue를 반복해야 합니다.

### Q11. 본인이 직접 한 부분은 무엇인가?

Sweep script를 작성/수정해 dynamic/quantized/no_cache 조건을 실행하고, CSV/plot artifact를 생성했습니다. 또한 README와 docs에서 어떤 claim이 파일로 뒷받침되는지 정리했습니다.

### Q12. 부족한 점은 무엇인가?

아직 Qwen 5080 cross-GPU result가 repo에 없고, quality/perplexity degradation도 Qwen sweep에는 포함되어 있지 않습니다. Prefill과 decode latency 분리도 더 정교하게 해야 합니다.

## 6. 면담에서 쓸 수 있는 핵심 문장 10개

1. 저는 LLM inference가 실제 GPU memory hierarchy에서 어떻게 병목을 만드는지에 관심이 있습니다.
2. 이 실험은 Oaken full reproduction이 아니라 Oaken-inspired KV-cache memory-pressure 실험입니다.
3. KV-cache는 sequence length와 batch size에 따라 선형적으로 커집니다.
4. Qwen2.5-1.5B에서는 GQA 구조 때문에 key/value heads 기준 공식이 필요했습니다.
5. Sanity run에서 dynamic cache의 실제 KV footprint가 이론값과 일치했습니다.
6. RTX 5060 8GB에서 Qwen `batch=8`, `seq_len=12288/16384`는 dynamic cache로 OOM이 났습니다.
7. 같은 조건을 quantized cache로 바꾸자 두 케이스 모두 실행됐습니다.
8. 저는 이 결과를 speedup이 아니라 memory-capacity trade-off로 해석합니다.
9. no_cache는 실용 정책이 아니라 ablation baseline입니다.
10. 다음 단계는 RTX 5080에서 같은 Qwen 조건을 돌려 capacity scaling을 확인하는 것입니다.

## 7. 금지 표현 / 대체 표현

| Risky expression | Why risky | Safer expression |
| --- | --- | --- |
| Oaken을 재현했습니다 | repo가 full paper reproduction을 증명하지 않음 | Oaken 논문을 동기로 삼은 Oaken-inspired KV-cache 실험입니다 |
| quantization으로 성능을 개선했습니다 | throughput/latency가 항상 좋아지는 것이 아님 | quantized cache가 일부 OOM 조건의 feasible region을 확장했습니다 |
| offloading으로 해결했습니다 | host RAM pressure와 transfer bottleneck 가능 | offloading은 GPU VRAM pressure를 줄일 수 있지만 새 병목을 만들 수 있습니다 |
| no_cache가 좋았습니다 | serving에서는 재계산 비용이 큼 | no_cache는 lower-bound / ablation으로만 사용했습니다 |
| 5080에서도 Qwen 비교가 끝났습니다 | repo에 Qwen 5080 cache-policy CSV 없음 | 5080 Qwen cross-GPU 비교는 다음 실험입니다 |
