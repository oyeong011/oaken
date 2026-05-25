# 내일 발표 최종 압축 브리핑

## 1. 30초 opening

저는 모델 구조 자체보다 LLM inference가 실제 GPU memory hierarchy에서 어떤 병목을 만드는지에 관심이 있습니다. 이번에는 Oaken의 문제의식을 동기로 삼아 consumer GPU에서 KV-cache memory pressure를 계측했습니다. 핵심은 Oaken을 완전히 재현했다는 주장이 아니라, RTX 5060 8GB에서 dynamic cache의 OOM boundary와 quantized cache의 rescue 가능성을 파일 근거로 확인한 것입니다.

## 2. 90초 experiment explanation

Autoregressive decoding에서는 이전 token의 key/value tensor를 cache로 저장해서 재계산을 줄입니다. 하지만 KV-cache는 sequence length와 batch size가 커질수록 증가하므로, long-context나 큰 batch에서는 compute보다 VRAM capacity가 먼저 병목이 될 수 있습니다.

이 실험에서는 dynamic cache를 기준으로 quantized, offloaded, no_cache 전략을 비교했습니다. 핵심은 "quantization이 항상 좋다"가 아니라, OOM boundary 근처에서 어떤 전략이 memory를 줄이고 그 대신 throughput이나 host/transfer cost를 얼마나 희생하는지 보는 것입니다.

파일 근거가 가장 강한 cache-policy 결과는 RTX 5060 Qwen2.5-1.5B와 OPT-1.3B boundary/rescue CSV입니다. Qwen sanity file은 `position_valid=True`, `max_position_embeddings=32768`, `kv_formula_type=gqa_mqa`를 기록합니다 (`results/rtx5060_qwen25_15b_sanity.csv`). Qwen dynamic boundary에서는 `batch=8`, `seq_len=12288/16384`가 OOM이고, rescue file에서는 quantized cache가 두 조건 모두 OK입니다 (`results/rtx5060_qwen25_15b_dynamic_boundary.csv`, `results/rtx5060_qwen25_15b_rescue_cases.csv`).

RTX 5080 근거는 같은 Qwen cache-policy sweep이 아니라 Oaken-style Wikitext accuracy path와 OPT model-size boundary입니다. OPT-125M, OPT-350M, OPT-1.3B, OPT-2.7B는 Oaken Wikitext evaluation이 완료됐고, OPT-6.7B는 original eval과 profiling이 expandable allocator에서 완료됐지만 Oaken eval은 약 15.8GB peak VRAM 근처에서 CUDA OOM이었습니다 (`results/oaken_consumer_gpu_summary.csv`, `results/rtx5080/opt-6.7b/summary.md`).

즉, KV-cache 최적화는 universal speedup이라기보다 limited VRAM 환경에서 feasible execution region을 넓히는 memory-capacity technique으로 해석하는 것이 안전합니다.

## 3. Key claims

1. 이 작업은 full Oaken reproduction이 아니라 Oaken-inspired KV-cache memory-pressure 실험이다.
2. Qwen2.5-1.5B sanity에서 `position_valid=True`, `max_position_embeddings=32768`, `kv_formula_type=gqa_mqa`가 확인된다 (`results/rtx5060_qwen25_15b_sanity.csv`).
3. Qwen dynamic cache는 `batch=8`, `seq_len=12288/16384`에서 OOM이다 (`results/rtx5060_qwen25_15b_dynamic_boundary.csv`).
4. Qwen quantized cache는 해당 두 dynamic OOM 조건을 OK로 rescue했다 (`results/rtx5060_qwen25_15b_rescue_cases.csv`).
5. RTX 5080은 Qwen cache-policy sweep 근거가 아니라 OPT Oaken-style Wikitext/VRAM boundary 근거이며, OPT-6.7B Oaken eval은 약 15.8GB peak VRAM 근처에서 CUDA OOM이었다 (`results/oaken_consumer_gpu_summary.csv`, `results/rtx5080/opt-6.7b/summary.md`).

## 4. Limitations

1. Full Oaken hardware/software co-design reproduction이 아니다.
2. no_cache는 OK row가 있어도 practical serving policy가 아니라 ablation/lower-bound다.
3. offloaded는 host memory pressure와 transfer overhead가 새 병목이 될 수 있으며, README는 15 GiB RAM/no swap 환경의 host-memory issue를 기록한다 (`README.md`).
4. Qwen boundary sweep은 random token 기반 memory sweep이며 quality/perplexity 평가가 없다.
5. RTX 5080 Qwen cache-policy CSV와 prefill/decode latency 분리 측정이 아직 없다.

## 5. Likely questions

**Q1. 그냥 KV-cache 선형 증가 아닌가요?**
맞습니다. 선형 증가는 contribution이 아니라 sanity check입니다. 핵심은 실제 OOM boundary와 cache policy trade-off를 계측한 점입니다.

**Q2. Oaken 재현인가요?**
아닙니다. Oaken-inspired 실험이며 full hardware throughput reproduction이라고 말하지 않습니다.

**Q3. 왜 Qwen을 썼나요?**
OPT는 position limit이 짧아서 long-context 해석이 synthetic stress가 될 수 있습니다. Qwen2.5-1.5B는 `max_position_embeddings=32768`로 기록되어 position-valid long-context boundary를 보기 좋습니다 (`results/rtx5060_qwen25_15b_sanity.csv`).

**Q4. quantized cache가 항상 좋은가요?**
아닙니다. Qwen에서는 dynamic OOM이던 두 조건을 rescue했지만, 이는 speedup claim이 아니라 memory-capacity claim입니다.

**Q5. OPT에서도 rescue가 됐나요?**
부분적으로 됐습니다. quantized는 `4x8192`, `8x4096`을 OK로 만들었지만, `8x6144`, `8x8192`는 여전히 OOM입니다 (`results/rtx5060_opt13b_rescue_cases.csv`).

**Q6. no_cache는 좋은 방법인가요?**
아닙니다. no_cache는 cache memory lower-bound를 보는 ablation입니다. Practical serving policy 후보로 말하지 않습니다.

**Q7. offloading은 해결책인가요?**
완전한 해결책이 아닙니다. GPU VRAM pressure를 줄일 수 있지만 host memory와 transfer 비용으로 병목이 이동할 수 있습니다.

**Q8. GQA/MQA formula는 왜 중요한가요?**
Qwen2.5는 KV head 수가 attention head 수보다 작으므로 KV-cache 계산에는 key/value head 기준 formula가 필요합니다. CSV는 `num_attention_heads=12`, `num_key_value_heads=2`, `head_dim=128`을 기록합니다 (`results/rtx5060_qwen25_15b_sanity.csv`).

**Q9. 품질 손실은 봤나요?**
Qwen boundary sweep에는 없습니다. 다음 실험에서 PPL 또는 downstream score를 추가해야 합니다.

**Q10. 다음 단계는 무엇인가요?**
RTX 5080에서 같은 Qwen sweep을 돌려 capacity scaling을 보고, fixed decode length, prefill/decode 분리, quality metric을 추가하는 것입니다.

**Q11. RTX 5080 결과는 무엇인가요?**
RTX 5080은 Qwen cache-policy sweep이 아니라 OPT Oaken-style Wikitext accuracy path와 model-size boundary 결과입니다. OPT-125M부터 OPT-2.7B까지 Oaken eval이 완료됐고, OPT-6.7B는 original eval/profiling은 expandable allocator로 완료됐지만 Oaken eval은 약 15.8GB peak VRAM 근처에서 OOM이었습니다.

## 6. 5080 질문을 받았을 때

현재 `/home/ssu/oaken` repo 안의 RTX 5080 evidence는 Oaken-style OPT summary가 중심입니다. 이전에 언급한 80-row RTX 5080 Qwen cache-policy sweep은 현재 repo 안의 source CSV로는 확인되지 않으므로 file-backed claim으로 쓰지 않겠습니다. 대신 5080은 OPT-125M~2.7B Oaken Wikitext evaluation 완료와 OPT-6.7B 16GB-class boundary evidence로 설명하겠습니다.

## 7. Final speaking script

제가 한 작업은 Oaken을 완전히 재현했다는 주장이 아니라, Oaken의 문제의식인 KV-cache memory pressure를 consumer GPU에서 계측해 본 준비 실험입니다. RTX 5060 8GB에서 Qwen2.5-1.5B dynamic cache가 `batch=8`, `seq_len=12288/16384`에서 OOM이 나는 것을 확인했고, quantized cache가 두 조건을 실행 가능하게 만든 것을 CSV로 확인했습니다. RTX 5080에서는 Qwen cache-policy sweep이 아니라 OPT Oaken-style Wikitext path를 보았고, OPT-6.7B가 약 15.8GB VRAM 근처에서 boundary가 됐습니다. no_cache는 lower-bound ablation이고 offloading은 host memory/transfer 병목을 만들 수 있어 조심스럽게 해석해야 합니다. 앞으로는 RTX 5080 Qwen cache-policy sweep, quality metric, prefill/decode 분리까지 추가해 더 엄밀한 systems measurement로 발전시키고 싶습니다.
