# Defense Q&A

## 1. 공격 질문 20개

1. 이게 새로운가, 그냥 KV-cache가 선형 증가한다는 당연한 얘기 아닌가?
2. Oaken을 재현했다고 볼 수 있나?
3. RTX 5080 Qwen 결과와 Oaken artifact 결과가 같은 실험인가?
4. theoretical KV formula가 정확하다는 근거는 무엇인가?
5. GQA/MQA 모델에서 formula를 잘못 쓰면 어떻게 되는가?
6. actual `past_key_values` 측정이 정말 KV-cache 전체를 잡는가?
7. CUDA peak memory가 allocator나 temporary buffer 때문에 왜곡될 수 있지 않나?
8. peak memory와 allocated/reserved memory를 구분했나?
9. batch size와 sequence length 선택 근거는 무엇인가?
10. dynamic과 quantized 비교가 공정한가?
11. quantized가 EOS로 더 적은 token을 생성하면 latency 비교가 흔들리지 않나?
12. offloading 비교는 host RAM 환경에 종속적이지 않나?
13. no_cache를 왜 넣었나?
14. RTX 5060 rescue 결과가 파일로 없으면 발표에서 말할 수 있나?
15. throughput 측정은 충분히 반복했나?
16. prefill과 decode를 분리했나?
17. quality degradation이나 perplexity는 측정했나?
18. 왜 Qwen2.5-1.5B인가?
19. consumer GPU 결과를 연구실/KAIRI 연구와 연결할 수 있나?
20. 다음 연구 질문은 무엇인가?

## 2. 방어 답변

### 1. 선형 증가가 당연하지 않나?

**Short answer:** 맞습니다. 선형 증가는 기여가 아니라 sanity check입니다. 핵심은 실제 OOM boundary와 cache mode trade-off입니다.

**Technical answer:** KV-cache formula 자체는 알려진 내용입니다. 그래서 문서에서는 `kv_actual_over_theory=1.0`을 최종 기여가 아니라 측정 harness가 맞는지 확인하는 sanity check로 둡니다. RTX 5080 Qwen2.5에서는 80 rows 중 76 OK, 4 OOM이었고, OOM은 모든 mode에서 `batch_size=8`, `seq_len=8192`에 발생했습니다. 이 boundary와 throughput/memory trade-off가 발표의 중심입니다. 출처는 `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_status_count.csv`와 `rtx5080_oom_cases.csv`입니다.

**What not to say:** "KV-cache 선형 증가를 새로 발견했습니다."

### 2. Oaken 재현인가?

**Short answer:** 아닙니다. Oaken-inspired 실험입니다.

**Technical answer:** Oaken은 hardware/software co-design과 dedicated quant/dequant module까지 포함합니다. 이 저장소에는 Oaken accuracy artifact와 consumer GPU VRAM characterization은 있지만, full hardware throughput reproduction은 없습니다. 따라서 "Oaken을 완전히 재현했다"는 표현은 쓰지 않습니다. `results/oaken_consumer_gpu_summary.md`도 full hardware throughput reproduction이 아니라고 명시합니다.

**What not to say:** "Oaken 논문 결과를 재현했습니다."

### 3. RTX 5080 Qwen과 Oaken artifact가 같은 실험인가?

**Short answer:** 아닙니다. 서로 다른 evidence stream입니다.

**Technical answer:** `results/oaken_consumer_gpu_summary.csv`는 Oaken artifact accuracy/VRAM summary입니다. `/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`는 HF Qwen2.5 cache-mode benchmark입니다. 둘 다 KV-cache/memory pressure 주제와 관련되지만 같은 실험으로 합치면 안 됩니다.

**What not to say:** "두 결과가 같은 실험을 증명합니다."

### 4. 이론식 정확성 근거는?

**Short answer:** RTX 5080 Qwen 성공 row에서 actual/theory가 1.0입니다.

**Technical answer:** Qwen2.5 result CSV에는 `theoretical_kv_bytes`, `actual_prefill_kv_bytes`, `kv_actual_over_theory`가 있습니다. 분석 파일은 모든 성공 row에서 cache mode별 mean/min/max가 1.0이라고 요약합니다. 출처는 `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_qwen25_analysis.md`입니다.

**What not to say:** "모든 모델에서 항상 맞습니다."

### 5. GQA/MQA formula 문제는?

**Short answer:** KV head 수를 써야 합니다.

**Technical answer:** Qwen2.5 CSV는 `num_attention_heads=12`, `num_key_value_heads=2`, `head_dim=128`을 기록합니다. full attention head를 쓰면 KV-cache를 과대추정합니다. 그래서 `scripts/run_kv_cache_sweep.py`는 `num_key_value_heads * head_dim` 기반으로 계산합니다.

**What not to say:** "hidden size만 쓰면 됩니다."

### 6. actual past_key_values 측정은 충분한가?

**Short answer:** tensor footprint sanity로는 유용하지만 전체 CUDA peak를 설명하지는 않습니다.

**Technical answer:** actual KV tensor bytes는 `past_key_values` 객체 내부 tensor를 합산합니다. 하지만 peak CUDA memory에는 model weights, activations, temporary allocation, allocator behavior가 포함됩니다. 그래서 새 script에는 `non_kv_overhead_mb`도 기록하도록 했습니다. 이 값은 future analysis에서 KV 외 overhead를 분리하는 데 필요합니다.

**What not to say:** "CUDA peak memory는 전부 KV-cache입니다."

### 7. CUDA peak memory 왜곡 가능성?

**Short answer:** 있습니다. 그래서 peak memory는 allocator-level measurement로 해석해야 합니다.

**Technical answer:** PyTorch CUDA peak allocated는 실제 tensor allocation peak를 보여주지만 reserved memory나 fragmentation과 다를 수 있습니다. 따라서 `peak_allocated_bytes`, `peak_reserved_bytes`, `free_before_bytes`, `free_after_bytes`를 함께 봐야 합니다. Qwen CSV에는 이 컬럼들이 있습니다.

**What not to say:** "peak memory가 순수 KV-cache입니다."

### 8. peak/allocated/reserved 구분?

**Short answer:** Qwen CSV에는 관련 컬럼이 있습니다.

**Technical answer:** `/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`에는 `peak_allocated_bytes`, `peak_reserved_bytes`, `base_allocated_bytes`, `free_before_bytes`, `free_after_bytes`가 있습니다. 발표에서는 peak delta를 중심으로 쓰되 allocator caveat을 붙입니다.

**What not to say:** "memory metric은 하나면 충분합니다."

### 9. sweep 범위 근거?

**Short answer:** RTX 5080 Qwen은 batch 1/2/4/8과 seq 512~8192로 구성되어 있습니다.

**Technical answer:** 해당 범위는 `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_qwen25_analysis.md`에 명시되어 있습니다. RTX 5060 Qwen long-context 확장 범위는 아직 CSV 근거가 없습니다. 따라서 내일 발표에서는 5080 범위만 measured로 말합니다.

**What not to say:** "모든 long-context 범위를 탐색했습니다."

### 10. dynamic/quantized 공정성?

**Short answer:** 같은 sweep 안의 비교지만 완전 공정성에는 caveat이 있습니다.

**Technical answer:** 동일 모델, GPU, batch/seq grid에서 비교했습니다. 다만 quantized run은 EOS로 생성 token 수가 달라질 수 있어 raw latency 비교는 주의해야 합니다. 분석 파일도 `tokens_per_sec`를 더 안전한 normalized metric으로 보라고 적습니다.

**What not to say:** "latency만 보면 됩니다."

### 11. EOS 문제?

**Short answer:** 맞습니다. limitation입니다.

**Technical answer:** `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_qwen25_analysis.md`는 quantized run이 EOS로 `max_new_tokens`보다 적게 생성할 수 있다고 명시합니다. 다음 실험에서는 fixed decode length나 EOS 비활성화를 고려해야 합니다.

**What not to say:** "모든 latency가 완전히 apples-to-apples입니다."

### 12. offloading host 의존성?

**Short answer:** 큽니다.

**Technical answer:** Offloading은 GPU KV-cache를 CPU 쪽으로 옮겨 GPU VRAM을 줄이는 전략입니다. 따라서 host RAM, PCIe transfer, page/cache behavior의 영향을 받습니다. RTX 5080 파일에서는 throughput ratio가 0.594451로 낮습니다.

**What not to say:** "offloading은 메모리 문제를 해결합니다."

### 13. no_cache 의미?

**Short answer:** ablation/lower-bound입니다.

**Technical answer:** no_cache는 KV reuse를 포기하므로 memory pressure 일부는 줄일 수 있지만 decode throughput이 크게 낮아집니다. RTX 5080 대표 case에서 no_cache 2.951665 tokens/s, dynamic 124.957829 tokens/s입니다.

**What not to say:** "no_cache도 serving 후보입니다."

### 14. RTX 5060 rescue 말할 수 있나?

**Short answer:** 현재 파일 근거로는 강하게 말하면 안 됩니다.

**Technical answer:** `docs/00_experiment_inventory.md` 기준으로 `results/rtx5060_opt13b_dynamic_boundary.csv`와 Qwen rescue CSV가 발견되지 않았습니다. 따라서 "prior notes에 있음, repository evidence 필요"로만 말해야 합니다.

**What not to say:** "5060에서 rescue를 증명했습니다."

### 15. throughput 반복 측정?

**Short answer:** 현재 CSV는 sweep result이지 반복 통계 실험은 아닙니다.

**Technical answer:** 평균/최소/최대는 grid 내 group summary입니다. 동일 condition 반복으로 confidence interval을 낸 것은 아닙니다. 발표에서는 trend와 boundary 관찰로 제한합니다.

**What not to say:** "통계적으로 유의합니다."

### 16. prefill/decode 분리?

**Short answer:** 부족합니다.

**Technical answer:** current benchmark는 prefill KV tensor sanity와 generate latency/throughput을 함께 기록합니다. Prefill latency와 decode per-token latency distribution은 분리하지 않았습니다. next step으로 분리해야 합니다.

**What not to say:** "decode bottleneck을 완전히 분해했습니다."

### 17. quality/perplexity?

**Short answer:** HF Qwen cache-mode sweep에는 없습니다.

**Technical answer:** Oaken artifact summary에는 Wikitext PPL이 있습니다. 그러나 Qwen dynamic/quantized/offloaded/no_cache sweep은 memory/throughput 중심이며 quality degradation을 평가하지 않았습니다.

**What not to say:** "품질 손실이 없습니다."

### 18. 모델 선택?

**Short answer:** Qwen2.5는 GQA와 long-context를 보기 위한 타깃입니다.

**Technical answer:** 현재 file-backed Qwen result는 RTX 5080에서 8192까지입니다. Qwen은 GQA 구조라 formula sanity가 중요하고, OPT 2048 position limit 문제를 피하기 위한 후보입니다. 다만 RTX 5060 Qwen long-context CSV는 아직 없습니다.

**What not to say:** "모델 선택이 완전합니다."

### 19. KAIRI/연구 관련성?

**Short answer:** AI systems/infrastructure preparation입니다.

**Technical answer:** 이 실험은 model architecture contribution이 아닙니다. 대신 inference serving에서 memory capacity, batching, context length, cache strategy가 만드는 execution bottleneck을 계측합니다. 이는 resource-aware serving과 infrastructure 연구의 작은 준비입니다.

**What not to say:** "이 자체가 완성된 시스템 연구입니다."

### 20. 다음 연구 질문?

**Short answer:** position-valid long-context rescue를 file-backed로 확인하는 것입니다.

**Technical answer:** RTX 5060 Qwen2.5 dynamic boundary와 quantized rescue CSV를 만들고, fixed decode length, quality metric, prefill/decode 분리, host memory instrumentation을 추가해야 합니다. 그래야 feasible-region expansion 주장을 더 안전하게 할 수 있습니다.

**What not to say:** "이미 충분히 끝났습니다."

## 3. 먼저 인정하면 오히려 안전한 한계

| Limitation | Why it matters | Next step |
| --- | --- | --- |
| Full Oaken reproduction 아님 | 과장하면 바로 반박됨 | Oaken-inspired라고 명시 |
| RTX 5060 rescue CSV 없음 | 핵심 rescue 주장의 file-backed evidence 부족 | CSV 생성 후 문서 갱신 |
| Qwen position_valid CSV 없음 | long-context-valid claim 약함 | `position_valid` 포함 sweep 실행 |
| HF Qwen quality metric 없음 | quantization 품질 영향 미확인 | PPL 또는 task score 추가 |
| Prefill/decode 분리 부족 | 병목 위치가 불분명 | phase별 timing 추가 |
| 반복 실험 부족 | variance/신뢰구간 없음 | 동일 조건 반복 측정 |
| offloading host instrumentation 부족 | host OOM/transfer 병목 설명 약함 | RAM/swap/PCIe monitoring 추가 |
| Consumer GPU 한정 | generalization 제한 | 더 다양한 GPU/model로 확장 |
