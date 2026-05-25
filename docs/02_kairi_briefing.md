# KAIRI 브리핑: KV-cache memory pressure experiment

## 1. 30초 자기소개 버전

안녕하세요. 저는 숭실대학교 컴퓨터학부 4학년으로, AI 모델 구조 자체보다 실제 inference가 GPU memory와 system resource 위에서 어떻게 병목을 만드는지에 관심이 있습니다. 최근에는 Oaken-inspired KV-cache 실험을 통해 consumer GPU에서 sequence length, batch size, cache mode가 VRAM pressure와 throughput에 어떤 영향을 주는지 계측했습니다. 아직 완전한 논문 재현은 아니지만, 실패 case와 OOM boundary까지 정리하면서 AI systems/infrastructure 연구 방법을 더 배우고 싶어 KAIRI에 지원하고자 합니다.

## 2. 1분 연구경험 설명

Oaken 논문을 보면서 LLM inference 최적화가 단순히 연산량 문제가 아니라 KV-cache와 memory hierarchy 문제라는 점에 관심을 갖게 되었습니다. 그래서 consumer GPU 환경에서 dynamic, quantized, offloaded, no_cache cache mode를 비교하는 실험을 구성했습니다. 파일로 확인되는 RTX 5080 Qwen2.5 sweep은 80개 case 중 76개가 성공했고 4개가 OOM이었으며, OOM은 모든 cache mode에서 `batch_size=8`, `seq_len=8192`에 나타났습니다 (`/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_status_count.csv`, `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_oom_cases.csv`). 또한 성공 row에서 `kv_actual_over_theory=1.0`으로 이론식과 실제 KV tensor 크기가 맞는 것을 확인했습니다. 다만 RTX 5060 Qwen rescue 결과는 현재 CSV로 확인되지 않기 때문에, 발표에서는 추가 실험/확인 필요 항목으로 분리할 생각입니다.

## 3. INA Research Group 정렬

INA Research Group이 AI/ML systems와 internet-scale infrastructure를 다룬다면, 제 실험은 model design 연구라기보다 LLM inference가 실제 GPU memory constraint에서 어떻게 행동하는지를 보는 작은 준비 작업입니다. Long-context serving과 batching은 throughput을 높이는 데 중요하지만, 동시에 KV-cache memory pressure를 키웁니다. Consumer GPU 실험은 production-scale system은 아니지만, cost-efficient serving과 resource-aware scheduling 문제를 이해하는 출발점이 될 수 있습니다. 따라서 이 경험은 INA 연구와 동등하다고 주장하기보다, AI systems/infrastructure 관점으로 문제를 바라보려는 구체적인 준비라고 설명하는 것이 안전합니다.

## 4. 지원동기 문단

저는 LLM이 실제 서비스 환경에서 효율적으로 동작하려면 모델 정확도뿐 아니라 inference memory, batching, cache policy, hardware resource constraint를 함께 이해해야 한다고 생각합니다. 최근 진행한 KV-cache memory pressure 실험에서는 Oaken을 완전히 재현하기보다, 그 문제의식을 바탕으로 consumer GPU에서 cache mode별 memory-throughput trade-off와 OOM boundary를 직접 계측했습니다. KAIRI에서는 이런 개인 실험을 더 엄밀한 연구 질문, 재현 가능한 methodology, 그리고 시스템 관점의 분석으로 발전시키는 방법을 배우고 싶습니다.

## 5. 예상 질문과 답변

### Q1. 왜 KV-cache를 보았나?

KV-cache는 autoregressive decoding에서 이전 token의 key/value를 저장해 재계산을 줄이는 핵심 구조입니다. 하지만 sequence length와 batch size에 따라 memory가 선형 증가하므로 long-context serving에서 VRAM bottleneck이 됩니다.

### Q2. Oaken을 정확히 재현한 것인가?

아닙니다. 제 작업은 full Oaken reproduction이 아니라 Oaken-inspired consumer GPU memory-pressure 실험입니다. Oaken의 hardware/software co-design 전체를 재현했다고 말하지 않습니다.

### Q3. quantized cache가 항상 좋은가?

아닙니다. RTX 5080 Qwen2.5 결과에서 quantized의 평균 throughput ratio는 dynamic 대비 0.744150이었습니다 (`/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_ratios_vs_dynamic.csv`). 따라서 memory 절감과 throughput 손실의 trade-off로 봐야 합니다.

### Q4. offloading은 왜 한계가 있었나?

파일로 확인되는 RTX 5080 Qwen2.5에서는 offloaded가 dynamic 대비 peak delta ratio 0.705081을 보였지만 throughput ratio는 0.594451이었습니다 (`/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_ratios_vs_dynamic.csv`). GPU memory pressure를 줄일 수 있어도 host transfer 비용이 새 병목이 될 수 있습니다.

### Q5. no_cache 결과는 무슨 의미인가?

no_cache는 practical serving policy라기보다 ablation/lower-bound입니다. RTX 5080 대표 case에서 no_cache는 `batch_size=4`, `seq_len=8192`에서 2.951665 tokens/s로 dynamic 124.957829 tokens/s보다 매우 낮았습니다 (`/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_key_examples.csv`).

### Q6. 이론식과 실제값이 맞았다는 것이 왜 중요한가?

OOM이나 throughput 해석 전에, 내가 계산한 KV footprint가 실제 tensor와 맞는지 검증해야 합니다. RTX 5080 Qwen2.5 성공 row에서는 `kv_actual_over_theory=1.0`으로 sanity check가 통과했습니다 (`/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_qwen25_analysis.md`).

### Q7. Qwen에서는 왜 GQA/MQA formula sanity가 필요한가?

Qwen2.5 CSV에는 `num_attention_heads=12`, `num_key_value_heads=2`, `head_dim=128`이 기록되어 있습니다 (`/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`). 따라서 full attention head 기준이 아니라 KV head 기준으로 계산해야 합니다.

### Q8. RTX 5060과 RTX 5080 비교에서 무엇을 배웠나?

현재 파일로 강하게 말할 수 있는 것은 RTX 5080 Qwen2.5 cache-mode sweep과 Oaken artifact summary입니다. RTX 5060 dynamic/quantized rescue CSV는 현재 발견되지 않아 비교 결론은 보류해야 합니다.

### Q9. 이 실험이 AI 대학원 연구와 무슨 관련이 있나?

AI systems 연구는 모델이 실제 hardware와 serving stack에서 어떤 병목을 만나는지 다룹니다. KV-cache memory pressure는 long-context serving, batching, latency/throughput trade-off와 직접 연결됩니다.

### Q10. 다음 실험은 무엇인가?

RTX 5060에서 Qwen2.5 position-valid long-context dynamic boundary와 quantized rescue를 CSV로 남기는 것입니다. 현재는 그 파일이 없어 발표에서 future work로 처리해야 합니다.

### Q11. 본인이 직접 한 부분은 무엇인가?

로컬 artifact를 정리하고, HF cache mode sweep harness와 plot script를 추가했으며, GQA/MQA-aware formula와 `position_valid` 기록을 반영했습니다 (`scripts/run_kv_cache_sweep.py`).

### Q12. 부족한 점은 무엇인가?

RTX 5060 Qwen 결과가 아직 file-backed artifact로 남아 있지 않습니다. 또한 quality/perplexity, prefill/decode 분리, fixed generation length control이 부족합니다.

## 6. 면담에서 쓸 수 있는 핵심 문장 10개

1. 저는 LLM inference의 실제 GPU memory bottleneck에 관심이 있습니다.
2. 이 작업은 Oaken 완전 재현이 아니라 Oaken-inspired KV-cache 실험입니다.
3. KV-cache는 sequence length와 batch size에 따라 선형으로 증가합니다.
4. Qwen 같은 GQA 모델은 KV head 수 기준으로 cache 크기를 계산해야 합니다.
5. RTX 5080 Qwen2.5 결과에서는 성공 row의 `kv_actual_over_theory`가 1.0이었습니다.
6. 같은 결과에서 OOM은 `batch_size=8`, `seq_len=8192`에서 나타났습니다.
7. Quantized cache는 항상 빠른 방법이 아니라 memory-throughput trade-off입니다.
8. Offloading은 GPU VRAM pressure를 줄일 수 있지만 transfer/host memory 병목을 만들 수 있습니다.
9. no_cache는 practical policy가 아니라 ablation으로 해석했습니다.
10. 다음 단계는 RTX 5060 Qwen long-context 결과를 재현 가능한 CSV로 남기는 것입니다.

## 7. 금지 표현 / 대체 표현

| Risky expression | Why risky | Safer expression |
| --- | --- | --- |
| Oaken을 재현했습니다 | hardware co-design과 full throughput reproduction이 없음 | Oaken-inspired KV-cache memory-pressure 실험을 했습니다 |
| quantization으로 성능을 개선했습니다 | throughput은 낮아질 수 있음 | quantized cache는 일부 memory pressure를 줄이는 trade-off입니다 |
| offloading으로 해결했습니다 | host memory/transfer 병목 가능 | offloading은 GPU pressure를 host/transfer 비용으로 옮길 수 있습니다 |
| no_cache가 좋은 방법입니다 | throughput collapse 가능 | no_cache는 ablation/lower-bound입니다 |
| RTX 5060 Qwen rescue를 확인했습니다 | 현재 CSV 파일 없음 | RTX 5060 Qwen rescue는 파일 근거 확보가 필요합니다 |
