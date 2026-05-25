# Defense Q&A

## 1. 공격 질문 20개

1. 이 실험의 novelty가 무엇인가? 그냥 옵션을 돌려본 것 아닌가?
2. KV-cache가 선형으로 증가한다는 건 이미 알려진 사실인데, 새로 확인한 의미가 있나?
3. Oaken을 재현했다고 볼 수 있나?
4. 이론식이 정확한가? GQA/MQA는 어떻게 처리했나?
5. actual KV-cache 크기는 어떻게 쟀고, 정말 `past_key_values`만 잰 것인가?
6. CUDA peak memory 측정은 allocator reserved memory 때문에 불안정하지 않나?
7. peak allocated, peak reserved, nvidia-smi VRAM 중 무엇을 신뢰해야 하나?
8. batch size와 sequence length 선택이 임의적인 것 아닌가?
9. dynamic과 quantized 비교가 공정한가?
10. quantized cache가 latency를 악화시킬 수 있는데 왜 rescue라고 하나?
11. offloading은 제대로 비교하지 못한 것 아닌가?
12. no_cache 결과를 왜 넣었나? 실제 serving과 다르지 않나?
13. host RAM / no swap 환경 때문에 offloading 결론이 일반화되지 않는 것 아닌가?
14. throughput 측정이 random token chunked sweep이라 실제 decode throughput과 다른 것 아닌가?
15. prefill과 decode를 분리하지 않았는데 KV-cache boundary라고 말할 수 있나?
16. quality degradation이나 perplexity를 측정하지 않았는데 quantized cache를 평가했다고 볼 수 있나?
17. 왜 OPT와 Qwen만 선택했나?
18. RTX 5080 cross-GPU Qwen 비교가 없는데 GPU capacity scaling을 말할 수 있나?
19. 재현성은 충분한가? 환경과 모델 weight, script version이 고정되어 있나?
20. 다음 연구 방향은 단순 benchmark 반복 말고 무엇인가?

## 2. 방어 답변

### 1. Novelty가 무엇인가?

**Short answer:** 새 알고리즘 제안은 아닙니다. Consumer GPU에서 KV-cache boundary와 quantized rescue를 file-backed artifact로 정리한 systems preparation입니다.

**Technical answer:** 이 실험의 novelty를 algorithmic novelty로 주장하지 않습니다. 목적은 Oaken-inspired 문제의식을 consumer GPU 환경에서 직접 계측하는 것입니다. 특히 OPT stress 결과만으로는 position-limit 공격을 받을 수 있어, Qwen2.5-1.5B position-valid long-context 실험으로 보강했습니다. `results/rtx5060_qwen25_15b_dynamic_boundary.csv`와 `results/rtx5060_qwen25_15b_rescue_cases.csv`는 dynamic OOM과 quantized rescue를 같은 조건에서 보여줍니다. 따라서 기여는 "새 방법"이 아니라 "명확한 실패 조건과 rescue 조건을 계측하고 해석한 것"입니다.

**What not to say:** "새로운 quantization 방법을 만들었습니다."

### 2. 선형 증가 확인의 의미가 있나?

**Short answer:** 선형 증가는 알려진 사실이지만, 실제 sweep이 그 병목을 제대로 측정하는지 확인하는 sanity check입니다.

**Technical answer:** Theory vs actual check는 final contribution이 아닙니다. 하지만 이 검증이 없으면 OOM이 KV-cache 때문인지 script/position bug 때문인지 방어하기 어렵습니다. Qwen dynamic sanity에서 `kv_actual_over_theory=1.0`이고 `kv_formula_type=gqa_mqa`가 기록되어 있습니다. 이는 GQA/MQA 공식이 실제 `past_key_values` footprint와 맞는다는 의미입니다. 따라서 이후 OOM boundary 해석이 단순 speculation이 아니라 측정 기반이라는 근거가 됩니다.

**What not to say:** "KV-cache 선형 증가를 제가 발견했습니다."

### 3. Oaken 재현인가?

**Short answer:** 아닙니다. Full Oaken reproduction이 아니라 Oaken-style KV-cache quantization experiment입니다.

**Technical answer:** Repo에는 Oaken accuracy path와 quantizer 관련 파일이 있지만, 논문 전체 workload, hardware co-design, 모든 benchmark를 재현했다는 증거는 없습니다. 따라서 README와 발표에서는 "Oaken-inspired"라고 표현해야 합니다. 이 실험은 Oaken의 문제의식인 KV-cache memory pressure와 quantization/offloading trade-off를 consumer GPU에서 관찰한 것입니다. Full reproduction claim은 과장입니다.

**What not to say:** "Oaken 논문을 완전히 재현했습니다."

### 4. 이론식과 GQA/MQA 처리?

**Short answer:** MHA는 hidden size 기준이 가능하지만, GQA/MQA는 key/value heads 기준으로 계산해야 합니다.

**Technical answer:** Qwen sanity CSV는 `num_attention_heads=12`, `num_key_value_heads=2`, `head_dim=128`, `kv_formula_type=gqa_mqa`를 기록합니다. 따라서 KV bytes는 `2 * layers * batch * seq * num_key_value_heads * head_dim * bytes` 구조로 계산해야 합니다. `results/rtx5060_qwen25_15b_sanity.csv`에서 dynamic cache `kv_actual_over_theory=1.0`이어서 이 공식이 실제 tensor footprint와 맞았음을 확인했습니다.

**What not to say:** "모든 모델에서 hidden_size 공식만 쓰면 됩니다."

### 5. actual KV-cache 측정 방식?

**Short answer:** cache object 안의 tensor들을 순회하며 tensor footprint를 합산했습니다.

**Technical answer:** `scripts/run_kv_cache_sweep.py`는 cache object를 순회해 tensor `numel * element_size`를 합산하는 `tensor_bytes_by_device` 경로를 사용합니다. 이 값은 `cache_tensor_total_mib`, `cache_tensor_cuda_mib`, `kv_actual_mb`로 기록됩니다. Dynamic sanity에서 theory와 actual이 맞으므로 측정 경로가 적어도 dynamic cache에 대해서는 일관적입니다. Quantized cache에서는 representation이 달라져 ratio가 1보다 작아지는 것이 기대됩니다.

**What not to say:** "전체 GPU memory가 곧 KV-cache 크기입니다."

### 6. CUDA peak memory 신뢰성?

**Short answer:** Peak memory는 allocator 영향이 있으므로 KV tensor footprint와 구분해서 해석해야 합니다.

**Technical answer:** CUDA peak allocated/reserved와 nvidia-smi VRAM은 모델 weight, activation, temp buffer, allocator reserved memory까지 포함합니다. 그래서 quantized `kv_actual_over_theory=0.287`을 total memory 71% 감소로 해석하면 안 됩니다. README에도 이 제한을 명시했습니다. 신뢰할 수 있는 것은 "KV tensor footprint ratio"와 "해당 조건이 OK/OOM인지"입니다.

**What not to say:** "GPU memory가 71% 줄었습니다."

### 7. 어떤 memory metric을 믿나?

**Short answer:** 목적에 따라 다릅니다. KV footprint는 `kv_actual_mb`, capacity boundary는 OOM status와 peak memory를 함께 봅니다.

**Technical answer:** `kv_actual_mb`는 cache tensor 자체의 크기입니다. `peak_memory_allocated_mib`는 PyTorch allocator가 실제 할당한 peak입니다. `peak_vram_used_mib`는 nvidia-smi sample 기반이라 sampling resolution의 한계가 있습니다. 결론에는 OOM status와 KV theory/actual sanity를 중심으로 쓰고, peak memory는 보조 지표로 써야 합니다.

**What not to say:** "nvidia-smi peak가 항상 정확합니다."

### 8. sweep 범위가 임의적인가?

**Short answer:** 일부는 실용적 탐색 범위입니다. Qwen은 position-valid range 안에서 OOM을 찾기 위해 12288/16384까지 확장했습니다.

**Technical answer:** OPT는 max position 2048 밖까지 밀었기 때문에 stress test로 제한해 해석합니다. Qwen은 max position 32768이 sanity CSV에 기록되어 있어 12288/16384가 position-valid입니다. 따라서 Qwen 결과는 정상 long-context range 안의 OOM/rescue evidence입니다. 다음 단계에서는 5080에서 같은 조건과 더 큰 boundary를 반복해야 합니다.

**What not to say:** "OPT 8192는 정상 long-context 결과입니다."

### 9. dynamic vs quantized 공정성?

**Short answer:** 공정한 speed benchmark라기보다 boundary rescue test입니다.

**Technical answer:** Dynamic이 실패한 같은 batch/seq 조건을 quantized로 다시 실행했습니다. 목적은 latency superiority가 아니라 failed configuration의 feasibility 확인입니다. Qwen에서 dynamic OOM인 `8x12288`, `8x16384`가 quantized OK로 바뀌었습니다. 따라서 claim은 "quantized가 더 빠르다"가 아니라 "quantized가 일부 feasible region을 확장했다"입니다.

**What not to say:** "quantized가 dynamic보다 항상 낫습니다."

### 10. latency 악화 가능성?

**Short answer:** 맞습니다. 그래서 speedup이 아니라 memory-capacity optimization으로 해석합니다.

**Technical answer:** Quantized cache는 low precision representation과 quant/dequant overhead가 있을 수 있습니다. 현재 Qwen rescue는 OOM을 OK로 바꾼 점이 핵심입니다. Throughput 수치는 기록되어 있지만, random token chunked sweep이라 serving latency 결론으로 과장하지 않습니다. Future work는 fixed decode length와 latency distribution 측정입니다.

**What not to say:** "quantized는 latency도 항상 좋습니다."

### 11. offloading 비교 부족?

**Short answer:** 맞습니다. 현재 repo에는 valid offloaded rescue row가 없습니다.

**Technical answer:** README는 RTX 5060 host가 15 GiB RAM/no swap이었고 offloaded attempt가 kernel OOM kill을 유발했다고 기록합니다. 따라서 offloading을 실패한 방법으로 일반화할 수는 없습니다. 다만 이 환경에서는 GPU memory 문제를 host memory pressure로 옮길 수 있다는 limitation evidence가 됩니다. 더 큰 host RAM과 pinned transfer profiling 환경에서 재실험해야 합니다.

**What not to say:** "offloading은 쓸모없습니다."

### 12. no_cache 의미?

**Short answer:** Ablation/lower-bound입니다. Practical serving method로 해석하면 안 됩니다.

**Technical answer:** no_cache는 cache를 저장하지 않아 memory 측면의 lower-bound처럼 보일 수 있습니다. 그러나 autoregressive serving에서는 과거 token을 재계산해야 하므로 long-context throughput이 크게 나빠질 수 있습니다. 현재 chunked memory sweep의 throughput은 real serving no_cache cost를 대표하지 않습니다. 따라서 no_cache는 비교 기준이지 제안 방법이 아닙니다.

**What not to say:** "no_cache가 가장 좋습니다."

### 13. host RAM/no swap 일반화 문제?

**Short answer:** 일반화할 수 없습니다. 환경 limitation으로 먼저 인정해야 합니다.

**Technical answer:** Offloaded cache는 host memory capacity와 PCIe transfer에 의존합니다. 이번 RTX 5060 host는 15 GiB RAM/no swap이라는 강한 제약이 있었습니다. 따라서 offloading failure는 이 환경의 limitation으로 기록하고, 일반적인 offloading 성능 결론으로 확장하지 않습니다. 다음 실험은 더 큰 RAM과 transfer metrics가 필요합니다.

**What not to say:** "offloading은 모든 환경에서 실패합니다."

### 14. throughput reliability?

**Short answer:** Memory sweep throughput은 참고 지표이지 serving benchmark로 과장하지 않습니다.

**Technical answer:** Script는 chunked cache-growth 방식입니다. 이것은 full prefill attention OOM을 피하고 cache boundary를 보기 위한 설계입니다. 따라서 recorded throughput은 동일 script 내 비교에는 참고 가능하지만, production decode throughput으로 직접 일반화하면 안 됩니다. Fixed token decode와 prefill/decode 분리 측정이 필요합니다.

**What not to say:** "이 throughput이 실제 serving throughput입니다."

### 15. prefill vs decode 미분리?

**Short answer:** 한계입니다. 현재는 KV-cache capacity boundary 중심 실험입니다.

**Technical answer:** Prefill은 attention activation과 seq^2 비용이 섞일 수 있습니다. 이 실험은 chunked cache-growth로 이 영향을 줄이려 했지만, prefill/decode를 완전히 분리한 것은 아닙니다. 다음 단계에서는 prompt prefill 후 one-token decode loop를 분리해 token latency와 cache growth를 측정해야 합니다.

**What not to say:** "decode와 prefill 병목을 완전히 분리했습니다."

### 16. 품질 평가 부재?

**Short answer:** Qwen sweep에는 품질 평가가 없습니다.

**Technical answer:** Qwen boundary/rescue 실험은 random token 기반 memory feasibility 실험입니다. 따라서 perplexity나 generation quality claim은 하지 않습니다. Repo에는 OPT Oaken-style PPL summaries가 있지만, Qwen quantized rescue에 대한 quality degradation은 측정하지 않았습니다. KAIRI/랩미팅에서는 memory feasibility와 quality evaluation을 분리해서 말해야 합니다.

**What not to say:** "품질 손실 없이 rescue했습니다."

### 17. 모델 선택?

**Short answer:** OPT는 stress test, Qwen은 position-valid long-context evidence를 위해 선택했습니다.

**Technical answer:** OPT-1.3B는 memory stress boundary를 빠르게 찾는 데 유용했지만 max position 2048 때문에 4096 이상을 정상 long-context로 해석하기 어렵습니다. Qwen2.5-1.5B는 sanity CSV에서 max position 32768과 GQA metadata가 확인되어 12288/16384를 position-valid long-context로 볼 수 있습니다. 두 모델은 서로 다른 역할입니다.

**What not to say:** "OPT long seq 결과도 정상 long-context입니다."

### 18. 5080 cross-GPU 부재?

**Short answer:** 현재 repo에는 Qwen 5080 cache-policy sweep이 없습니다.

**Technical answer:** Repo에는 RTX 5080 OPT Oaken-style summaries가 있고, OPT-6.7B boundary evidence가 있습니다. 하지만 Qwen 5060 OOM 조건이 5080에서 살아나는지 보여주는 CSV는 없습니다. 따라서 capacity scaling은 future work입니다. 문서에서는 이 missing evidence를 명시했습니다.

**What not to say:** "5080 Qwen 비교도 끝났습니다."

### 19. 재현성?

**Short answer:** CSV와 scripts는 남아 있지만, 더 정리할 부분이 있습니다.

**Technical answer:** `scripts/run_kv_cache_sweep.py`, result CSV, plot artifacts는 repo에 있습니다. README에는 software stack 일부가 기록되어 있습니다. 그러나 exact package lockfile, model download hash, seed, host RAM state, driver snapshot을 완전하게 묶은 것은 아닙니다. 다음 단계에서는 environment capture와 deterministic run metadata를 추가해야 합니다.

**What not to say:** "완벽히 재현 가능합니다."

### 20. 다음 연구 방향?

**Short answer:** Cross-GPU Qwen sweep, decode/prefill 분리, quality/latency 측정입니다.

**Technical answer:** 먼저 RTX 5080에서 Qwen `8x12288`, `8x16384`를 dynamic/quantized로 재확인합니다. 그 다음 5080 자체 boundary를 더 큰 batch/seq에서 찾습니다. 이후 quality/perplexity, latency distribution, host transfer metrics를 추가합니다. 이 방향은 resource-aware long-context serving benchmark로 확장될 수 있습니다.

**What not to say:** "이제 연구가 완성됐습니다."

## 3. 먼저 인정하면 오히려 안전한 한계

| Limitation | Why it matters | Next step |
| --- | --- | --- |
| Full Oaken reproduction 아님 | 논문 결과와 동일하다고 주장하면 방어 불가 | Oaken-inspired로 표현하고 scope 제한 |
| Qwen quality metric 없음 | quantized rescue가 품질을 유지하는지 모름 | PPL 또는 fixed evaluation 추가 |
| RTX 5080 Qwen CSV 없음 | capacity scaling claim이 미완성 | 같은 Qwen sweep을 5080에서 실행 |
| Offloading valid row 없음 | offloading 비교 결론이 약함 | RAM 큰 host에서 offload 재실험 |
| no_cache serving realism 부족 | chunked throughput이 real serving no_cache 비용을 대표하지 않음 | one-token decode loop로 재측정 |
| Prefill/decode 미분리 | 병목 원인 분해가 제한됨 | prefill peak와 decode latency 분리 |
| 모델 수 제한 | 일반화 범위가 좁음 | Llama/Qwen larger model 등 추가 |
| 환경 기록 부족 | 재현성 약점 | driver, package lock, seed, host RAM state 기록 |
