# 내일 랩미팅 발표 스크립트

## Slide 1. Title

**Title:** Consumer GPU 환경에서 LLM inference KV-cache memory pressure 관찰
**Subtitle:** RTX 5060 cache-policy sweep와 RTX 5080 Oaken-style OPT boundary 근거

- Oaken-inspired KV-cache memory-pressure experiment
- RTX 5060 8GB 중심 boundary/rescue 관찰
- RTX 5080은 OPT Oaken-style Wikitext/VRAM boundary evidence

**Spoken script:**
오늘 발표는 큰 모델을 새로 학습했다는 이야기가 아니라, LLM inference에서 실제 GPU memory가 어디서 한계가 되는지 계측한 실험입니다. Oaken 논문을 동기로 삼았지만 full reproduction은 아닙니다. RTX 5060에서는 dynamic, quantized, no_cache cache-policy sweep을 보았고, RTX 5080에서는 OPT model size를 키운 Oaken-style Wikitext accuracy path와 VRAM boundary artifact를 정리했습니다. 특히 RTX 5060 8GB에서 Qwen2.5-1.5B의 position-valid long-context 조건에서 dynamic OOM과 quantized rescue를 확인했습니다.

**Possible question:** Oaken 재현인가요?
**Short answer:** 아닙니다. Oaken-inspired memory-pressure experiment로 범위를 제한합니다.

## Slide 2. Motivation

- LLM inference는 compute-bound만이 아니라 memory-capacity/bandwidth-bound가 될 수 있음
- KV-cache는 context length와 batch size에 따라 증가
- Consumer GPU에서는 VRAM limit이 빨리 드러남
- Cache compression/offloading은 serving feasibility와 직결됨

**Spoken script:**
Autoregressive LLM inference에서는 이전 token의 key/value를 저장해야 합니다. 이 cache가 없으면 매번 과거 context를 다시 계산해야 하므로 비효율적이지만, cache를 저장하면 sequence length와 batch size가 커질수록 memory pressure가 커집니다. Consumer GPU는 VRAM이 제한되어 있기 때문에, long-context serving이나 큰 batch에서 먼저 boundary가 드러납니다. 그래서 cache quantization이나 offloading이 단순 최적화가 아니라 feasible region을 넓히는 방법인지 확인하고 싶었습니다.

**Possible question:** 너무 당연한 linear growth 아닌가요?
**Short answer:** Linear growth 자체는 예상 가능하지만, 어느 조건에서 실제 GPU OOM이 나고 어떤 policy가 rescue하는지는 계측해야 합니다.

## Slide 3. Background: KV-cache

- Autoregressive decoding은 token을 순차 생성
- 각 layer에서 key/value tensor를 저장
- Cache는 과거 token 재계산을 줄임
- 저장량은 token 수와 batch 수에 선형 증가

**Spoken script:**
KV-cache는 transformer attention에서 이전 token들의 key와 value를 저장하는 구조입니다. 다음 token을 만들 때 이전 token 전체를 다시 forward하지 않고 저장된 key/value를 재사용합니다. 그래서 latency에는 유리하지만, layer 수, batch size, sequence length, key/value head 수에 비례해 memory가 증가합니다. 이 실험에서는 실제 `past_key_values` tensor footprint가 이론식과 맞는지 먼저 확인했습니다.

**Possible question:** 왜 hidden size만 쓰면 안 되나요?
**Short answer:** GQA/MQA 모델은 key/value head 수가 query head 수보다 작아서 KV head 기준으로 계산해야 합니다.

## Slide 4. Experimental Questions

1. Theory vs actual KV-cache size가 맞는가?
2. OOM boundary는 어디서 발생하는가?
3. Quantized cache가 dynamic OOM을 rescue하는가?
4. Offloading/no_cache는 어떤 trade-off인가?

**Spoken script:**
질문은 네 가지였습니다. 첫째, 이론식과 실제 `past_key_values` 크기가 맞는지 sanity check를 합니다. 둘째, batch와 sequence를 키울 때 OOM boundary를 찾습니다. 셋째, dynamic이 실패한 조건만 quantized로 다시 돌려 rescue가 되는지 봅니다. 넷째, offloading과 no_cache는 memory를 줄일 수 있지만 다른 병목을 만드는지 구분합니다.

**Possible question:** 왜 처음부터 모든 모드를 다 돌리지 않았나요?
**Short answer:** 먼저 dynamic boundary를 찾아야 rescue 실험의 의미가 생기기 때문입니다.

## Slide 5. Method

- Models: OPT-1.3B, Qwen2.5-1.5B-Instruct
- GPU: RTX 5060 8GB cache-policy sweep; RTX 5080 OPT Oaken-style summaries
- Metrics: status, OOM, peak memory, throughput, KV theory/actual
- Output: CSV, derived OOM/rescue CSV, plots
- Sanity: Qwen `position_valid=True`, `kv_formula_type=gqa_mqa`

**Spoken script:**
OPT-1.3B는 RTX 5060 memory stress test로 사용했고, Qwen2.5-1.5B는 32768 context를 지원하는 position-valid long-context 모델로 사용했습니다. Sweep script는 chunked cache-growth 방식으로 cache를 늘리며 OOM을 찾고, 결과를 CSV로 저장합니다. Qwen에서는 `position_valid`, `max_position_embeddings`, `kv_formula_type`, `kv_actual_over_theory`를 기록해 GQA 공식이 맞는지 확인했습니다. RTX 5080 결과는 같은 Qwen sweep이 아니라, `results/oaken_consumer_gpu_summary.csv`와 `results/rtx5080/*/summary.md`에 정리된 OPT Oaken-style Wikitext/VRAM boundary입니다.

**Possible question:** OPT 8192 결과는 정상 long-context인가요?
**Short answer:** 아닙니다. OPT 결과는 memory stress evidence이고, 정상 long-context 해석은 Qwen 결과가 담당합니다.

## Slide 6. Results: Theory vs Actual

- Qwen dynamic sanity: `kv_actual_over_theory=1.0`
- Qwen formula type: `gqa_mqa`
- Qwen max positions: 32768
- Source: `results/rtx5060_qwen25_15b_sanity.csv`

**Spoken script:**
Qwen sanity run에서 dynamic cache의 actual KV footprint가 theory와 일치했습니다. 특히 Qwen은 GQA 구조라 `num_key_value_heads=2`를 사용해야 하며, sanity CSV에 `kv_formula_type=gqa_mqa`로 기록되어 있습니다. 이 sanity check는 contribution이라기보다 이후 OOM boundary 해석이 KV-cache 기반이라는 것을 확인하는 단계입니다.

**Possible question:** quantized ratio가 1.0이 아닌 건 문제인가요?
**Short answer:** 아닙니다. quantized cache는 KV tensor representation이 달라지므로 FP16 theory보다 작아지는 것이 목적입니다.

## Slide 7. Results: OOM Boundary

- OPT-1.3B dynamic OOM: `4x8192`, `8x4096`, `8x6144`, `8x8192`
- Qwen dynamic OOM: `8x12288`, `8x16384`
- Qwen rows are `position_valid=True`
- Source: `results/rtx5060_opt13b_dynamic_boundary.csv`, `results/rtx5060_qwen25_15b_dynamic_boundary.csv`

**Spoken script:**
OPT-1.3B에서는 position limit 밖까지 밀어붙인 memory stress test에서 dynamic OOM boundary를 관찰했습니다. 더 중요한 Qwen 결과에서는 모델이 지원하는 context 범위 안에서 `batch=8`, `seq_len=12288`와 `16384`가 dynamic OOM이었습니다. 이 차이가 중요합니다. OPT는 stress evidence이고, Qwen은 real long-context evidence로 방어할 수 있습니다.

**Possible question:** Qwen 32768까지는 왜 안 갔나요?
**Short answer:** 16384에서 이미 dynamic OOM과 quantized rescue가 나왔고, 우선 boundary/rescue evidence 확보를 목표로 했습니다.

## Slide 8. Results: Quantized Rescue

- OPT quantized OK: `4x8192`, `8x4096`
- OPT quantized OOM: `8x6144`, `8x8192`
- Qwen quantized OK: `8x12288`, `8x16384`
- Qwen quantized KV footprint ratio: 0.288737, 0.286865

**Spoken script:**
Rescue 실험은 dynamic에서 실패한 조건만 다시 돌렸습니다. OPT에서는 일부 조건이 quantized로 살아났지만 더 큰 조건은 여전히 OOM이었습니다. Qwen에서는 dynamic OOM이던 두 position-valid 조건이 quantized로 모두 성공했습니다. 다만 0.287이라는 ratio는 KV-cache tensor footprint가 FP16 theory 대비 줄었다는 뜻이지 total CUDA peak memory가 같은 비율로 줄었다는 뜻은 아닙니다.

**Possible question:** quantized가 더 빠른가요?
**Short answer:** 이 실험에서는 speedup claim을 하지 않습니다. 핵심은 memory-capacity rescue입니다.

## Slide 9. Results: Offloading and no_cache

- Offloading: GPU VRAM pressure를 host memory/transfer로 옮길 수 있음
- RTX 5060 offloaded attempt: host RAM pressure, no valid row
- no_cache: lower-bound / ablation only
- Source: `README.md`, `results/rtx5060_*_rescue_cases.csv`

**Spoken script:**
Offloading은 GPU memory를 아낄 수 있지만 공짜가 아닙니다. 이번 RTX 5060 환경은 system RAM이 작고 swap이 없어 offloaded run이 host memory pressure로 유효 row 없이 실패했습니다. no_cache는 cache memory를 줄이는 lower-bound로 볼 수 있지만, 실제 generation에서는 과거 context 재계산이 필요하므로 practical serving solution이라고 말하면 안 됩니다.

**Possible question:** no_cache throughput이 높게 보이는 row가 있는데요?
**Short answer:** 현재 chunked memory sweep의 처리량이고, autoregressive decode serving의 재계산 비용을 대표하지 않습니다.

## Slide 10. RTX 5080: Oaken-style accuracy path and 16GB boundary

- OPT-125M~2.7B: Oaken Wikitext evaluation completed
- OPT-6.7B: original eval + profiling completed only with expandable allocator
- OPT-6.7B Oaken eval still OOM near 15.8GB peak VRAM
- Interpretation: 5080 does not remove the memory-boundary problem; it shifts the boundary to larger models

**Spoken script:**
RTX 5080 결과는 5060 Qwen cache-policy sweep과 같은 종류가 아닙니다. Repo에 있는 5080 evidence는 OPT model size를 키우며 Oaken-style Wikitext accuracy path가 어디까지 가능한지 본 결과입니다. OPT-125M부터 OPT-2.7B까지는 Oaken Wikitext evaluation이 완료됐고, OPT-6.7B는 original FP16 eval과 Oaken profiling이 expandable allocator에서 완료됐지만 Oaken eval은 CUDA OOM으로 실패했습니다. 이때 peak VRAM이 약 15.8GB였으므로, 5080은 memory-boundary 문제를 없앤 것이 아니라 boundary를 더 큰 model size로 이동시킨 것으로 해석하는 것이 안전합니다.

**Possible question:** 5080에서도 Qwen dynamic/quantized sweep을 돌린 건가요?
**Short answer:** 아닙니다. 현재 repo-backed 5080 evidence는 OPT Oaken-style Wikitext/VRAM boundary이고, Qwen cache-policy sweep은 다음 실험입니다.

## Slide 11. Interpretation

1. KV-cache formula is useful for predicting pressure.
2. Quantization helps at some boundary cases but is not universal.
3. Serving systems need GPU memory, host memory, transfer, throughput together.

**Spoken script:**
결론은 세 가지입니다. 첫째, KV-cache 이론식과 actual footprint가 맞는지 확인하면 boundary 해석이 훨씬 방어 가능해집니다. 둘째, quantized cache는 일부 OOM 조건을 rescue하지만 모든 조건을 해결하는 universal fix가 아닙니다. 셋째, serving system에서는 GPU VRAM만 볼 것이 아니라 host memory, transfer overhead, throughput까지 같이 봐야 합니다.

**Possible question:** 연구적으로 새로운 점은 무엇인가요?
**Short answer:** 새로운 알고리즘 제안은 아니고, consumer GPU에서 file-backed boundary/rescue evidence를 정리한 systems preparation입니다.

## Slide 12. Limitations

- Limited models and hardware
- Not full Oaken reproduction
- Qwen cache-policy quality/perplexity not measured
- RTX 5080 Qwen cache-policy CSV not present
- Prefill vs decode not separated
- Offloading affected by host RAM/no swap

**Spoken script:**
한계는 명확합니다. 모델과 hardware가 제한적이고, Oaken full reproduction이 아닙니다. Qwen sweep은 memory-focused synthetic token 실험이라 quality나 perplexity를 포함하지 않습니다. 또한 RTX 5080에 대해서는 Qwen cache-policy CSV가 아니라 OPT Oaken-style boundary artifact만 있습니다. Prefill과 decode latency를 분리하지 않았고, offloading 결과는 host RAM/no swap 환경에 영향을 받았습니다. 이 점을 먼저 인정하는 것이 안전하다고 봅니다.

**Possible question:** 품질 저하를 보지 않았으면 실용성이 부족하지 않나요?
**Short answer:** 맞습니다. 현재는 memory feasibility 실험이고, 다음 단계에서 quality/perplexity를 붙여야 합니다.

## Slide 13. Next Steps

- RTX 5080 Qwen cross-GPU sweep
- 5060 OOM 조건이 5080에서 dynamic OK인지 확인
- 5080 Qwen 자체 boundary 탐색
- Prefill/decode 분리, token latency distribution
- 환경/스크립트 재현성 정리

**Spoken script:**
다음 단계는 같은 Qwen2.5-1.5B cache-policy 실험을 RTX 5080에서 반복하는 것입니다. 5060에서 dynamic OOM이던 `8x12288`, `8x16384`가 5080에서 살아나는지 보면 VRAM capacity scaling을 보여줄 수 있습니다. 이후 더 큰 batch/sequence에서 5080 Qwen 자체 boundary를 찾고, quantized rescue가 반복되는지 확인하겠습니다.

**Possible question:** 왜 5080이 필요한가요?
**Short answer:** 같은 모델/조건에서 GPU memory capacity가 feasible region을 어떻게 이동시키는지 볼 수 있기 때문입니다.

## Slide 14. Closing

- 핵심은 큰 모델 학습이 아니라 병목의 가설화/계측/정리
- 실패 케이스까지 기록
- KAIRI/연구실에서는 더 엄밀한 methodology를 배우고 싶음

**Spoken script:**
제가 이 실험에서 보여주고 싶은 것은 거대한 모델을 학습했다는 것이 아니라, LLM inference에서 실제 시스템 병목을 가설화하고, 계측하고, 실패 케이스까지 정리하는 방식입니다. 현재 결과는 작은 consumer GPU 실험이지만, long-context serving과 resource-aware inference라는 더 큰 AI systems 문제로 연결된다고 생각합니다.

**Possible question:** 이걸 어떻게 연구로 확장할 건가요?
**Short answer:** 더 다양한 GPU/model에서 boundary를 계측하고, scheduling/cache policy와 quality/latency까지 포함한 reproducible benchmark로 확장하고 싶습니다.
