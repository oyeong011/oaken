# 내일 랩미팅 발표 스크립트

## Slide 1. Title

**Title:** Consumer GPU 환경에서 LLM inference KV-cache memory pressure 관찰  
**Subtitle:** RTX 5060/5080 기반 dynamic / quantized / offloaded / no_cache 비교

- Oaken-inspired 실험
- Consumer GPU VRAM pressure 관찰
- 실패 case까지 결과로 기록

**Script:** 오늘 발표에서는 제가 Oaken 논문을 동기로 삼아 진행한 KV-cache memory pressure 실험을 공유드리겠습니다. 핵심은 Oaken을 완전히 재현했다는 주장이 아니라, LLM inference에서 KV-cache가 실제 GPU VRAM boundary를 어떻게 만드는지 consumer GPU에서 직접 계측해 본 것입니다.

**Possible question:** Oaken reproduction인가요?  
**Answer:** 아닙니다. Oaken-inspired artifact/benchmark 성격이며 full hardware throughput reproduction은 아닙니다.

## Slide 2. Motivation

- LLM inference는 compute-bound만이 아님
- Long context에서 KV-cache memory 증가
- Batch 증가도 memory pressure 증가
- Consumer GPU는 capacity boundary가 빨리 드러남

**Script:** LLM inference에서는 attention 계산 자체도 중요하지만, 실제 serving에서는 KV-cache를 저장하고 읽는 memory cost가 커집니다. 특히 context length와 batch size가 커지면 VRAM pressure가 커지고, consumer GPU에서는 OOM boundary가 비교적 빨리 드러납니다.

**Possible question:** 왜 consumer GPU인가요?  
**Answer:** production GPU는 아니지만 제한된 자원에서 병목이 명확히 드러나 resource-aware serving 질문을 보기 좋습니다.

## Slide 3. Background: KV-cache

- Autoregressive decoding에서 이전 token K/V 재사용
- Layer마다 key/value tensor 저장
- 재계산을 줄이지만 memory는 누적
- 이론적으로 batch와 sequence length에 선형 증가

**Script:** KV-cache는 이전 token의 key/value projection을 저장해 다음 token 생성 시 재사용하는 구조입니다. 이 덕분에 매번 과거 전체를 재계산하지 않아도 되지만, layer별로 key와 value를 저장하므로 context와 batch가 커질수록 memory footprint가 커집니다.

**Possible question:** 이건 너무 당연한 결과 아닌가요?  
**Answer:** 선형 증가는 당연합니다. 그래서 실험의 초점은 증가 자체가 아니라 실제 OOM boundary와 cache policy trade-off입니다.

## Slide 4. Experimental Questions

1. Theory vs actual size
2. OOM boundary
3. Quantized rescue
4. Offloading/no_cache trade-off

**Script:** 실험 질문은 네 가지입니다. 계산한 KV-cache 크기가 실제 tensor 크기와 맞는지, 어느 batch/sequence에서 OOM이 나는지, quantized cache가 실패 case를 살릴 수 있는지, 그리고 offloading/no_cache가 실용적인 대안인지 확인하는 것입니다.

**Possible question:** 품질 평가는 있나요?  
**Answer:** HF cache-mode sweep에는 품질 평가가 없습니다. Oaken artifact 쪽에는 Wikitext PPL summary가 있습니다.

## Slide 5. Method

- Models: OPT 계열, Qwen2.5-1.5B
- GPUs: RTX 5060, RTX 5080
- Modes: dynamic, quantized, offloaded, no_cache
- Metrics: status, OOM, peak memory, tokens/s, latency, actual/theory KV

**Script:** 파일로 확인 가능한 주요 결과는 두 종류입니다. `results/oaken_consumer_gpu_summary.csv`는 Oaken artifact accuracy/VRAM 결과이고, `/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`는 RTX 5080 Qwen2.5 cache-mode sweep입니다.

**Possible question:** RTX 5060 Qwen 결과는 있나요?  
**Answer:** 현재 저장소에서 해당 CSV는 발견되지 않았습니다. 발표에서는 future work로 분리하겠습니다.

## Slide 6. Results: Theory vs Actual

- RTX 5080 Qwen 성공 row에서 actual/theory 일치
- `kv_actual_over_theory=1.0`
- GQA 모델은 KV head 기준으로 계산

**Script:** RTX 5080 Qwen2.5 결과에서는 성공한 row의 `kv_actual_over_theory`가 1.0으로 기록되어 있습니다. Qwen2.5는 `num_attention_heads=12`, `num_key_value_heads=2`인 GQA 구조라 KV head 기준 계산이 필요합니다.

**Possible question:** 왜 중요한가요?  
**Answer:** 이 sanity check가 있어야 이후 memory pressure 해석이 단순 추정이 아니라 실제 tensor footprint와 연결됩니다.

## Slide 7. Results: OOM Boundary

- RTX 5080 Qwen: 80 cases
- 76 OK, 4 OOM
- 모든 mode가 B=8, S=8192에서 OOM

**Script:** RTX 5080 Qwen2.5 sweep은 80개 case 중 76개가 성공했고 4개가 OOM입니다. OOM은 dynamic, quantized, offloaded, no_cache 모두 `batch_size=8`, `seq_len=8192`에서 발생했습니다.

**Possible question:** quantized도 OOM이면 의미가 없나요?  
**Answer:** 이 boundary에서는 rescue하지 못했습니다. 다만 non-OOM 영역에서 memory-throughput trade-off는 관찰됩니다.

## Slide 8. Results: Quantized Rescue

- RTX 5080 최대 OOM case는 rescue 실패
- Quantized throughput ratio: 0.744150
- Peak delta ratio: 0.786496
- RTX 5060 rescue claim은 CSV 확보 필요

**Script:** RTX 5080 결과에서 quantized는 dynamic 대비 평균 throughput이 0.744150, peak memory delta가 0.786496입니다. 하지만 가장 큰 OOM case인 B=8, S=8192는 quantized도 OOM이었습니다. 5060 rescue 주장은 현재 파일이 없어 확인 필요로 남깁니다.

**Possible question:** 발표에서 rescue라고 말해도 되나요?  
**Answer:** RTX 5080 파일 기준으로는 rescue라고 말하면 안 됩니다. 5060 CSV 확보 후에만 말할 수 있습니다.

## Slide 9. Results: Offloading and no_cache

- Offloaded throughput ratio: 0.594451
- no_cache throughput ratio: 0.093942
- no_cache는 ablation/lower-bound

**Script:** Offloaded는 peak memory delta를 줄였지만 dynamic 대비 throughput ratio가 0.594451로 낮았습니다. no_cache는 0.093942로 throughput이 크게 무너져 practical policy보다는 ablation으로 해석하는 것이 맞습니다.

**Possible question:** offloading은 실패인가요?  
**Answer:** 실패라기보다 trade-off입니다. GPU memory를 줄이는 대신 transfer/host-side cost를 감수합니다.

## Slide 10. Interpretation

1. KV-cache formula는 pressure 예측에 유용
2. Quantization은 universal fix가 아님
3. Serving은 GPU/host/transfer/throughput을 함께 봐야 함

**Script:** 결론적으로 KV-cache 이론식은 실제 footprint sanity check에 유용했고, cache quantization/offloading은 속도 향상이라기보다 feasible region을 넓히기 위한 capacity technique으로 보는 것이 안전합니다.

**Possible question:** 그래서 연구적으로 무엇이 남나요?  
**Answer:** 더 엄밀한 prefill/decode 분리, model 다양화, quality 영향 측정, 그리고 scheduler/policy 관점 확장이 남습니다.

## Slide 11. Limitations

- Full Oaken reproduction 아님
- RTX 5060 Qwen rescue CSV 없음
- 품질/perplexity는 HF sweep에 없음
- prefill/decode 분리 부족
- offloading은 host 환경 영향 큼

**Script:** 이 실험은 아직 제한이 많습니다. 특히 Oaken full reproduction이 아니고, RTX 5060 Qwen rescue 결과는 현재 파일 근거가 없습니다. 따라서 발표에서는 측정한 것과 아직 측정하지 못한 것을 명확히 분리하겠습니다.

**Possible question:** 가장 큰 약점은?  
**Answer:** 5060 long-context rescue 결과가 file-backed artifact로 정리되지 않은 점입니다.

## Slide 12. Next Steps

- RTX 5060 Qwen CSV 확보
- fixed generation length
- prefill/decode 분리
- quality metric 추가
- 환경 문서화

**Script:** 다음 단계는 Qwen2.5를 사용해 position-valid long-context 조건에서 RTX 5060 dynamic boundary와 quantized rescue를 CSV로 남기는 것입니다. 그 다음 fixed decode length와 quality metric을 추가하겠습니다.

**Possible question:** 내일까지 가능한 것은?  
**Answer:** 현재 파일 기반 발표는 가능하고, 5060 Qwen 수치는 확보 전까지 future work로 두는 것이 안전합니다.

## Slide 13. Closing

- 실험의 핵심은 거대 모델 학습이 아님
- 실제 시스템 병목 가설화/계측/실패 정리
- KAIRI/연구실에서 방법론 발전 희망

**Script:** 제가 이 실험에서 보여주고 싶은 것은 거대한 모델을 학습했다는 것이 아니라, LLM inference에서 실제 시스템 병목을 가설화하고, 계측하고, 실패 케이스까지 정리하는 방식입니다. 앞으로는 이런 실험을 더 재현 가능하고 연구 질문이 분명한 형태로 발전시키고 싶습니다.

**Possible question:** 한 문장으로 기여를 말하면?  
**Answer:** Consumer GPU에서 KV-cache memory pressure와 cache policy trade-off를 파일 기반으로 계측하고, 과장 없이 한계까지 정리한 준비 실험입니다.
