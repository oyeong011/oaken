# KV-cache 실험 브리핑

## 1. 한 문장 요약

이 실험은 consumer GPU에서 LLM inference KV-cache가 batch size와 sequence length 증가에 따라 VRAM boundary를 만들고, 일부 dynamic OOM 조건을 quantized cache가 rescue할 수 있음을 파일 기반으로 확인한 Oaken-inspired KV-cache memory-pressure 실험이다.

## 2. 실험 동기

LLM autoregressive inference는 이전 token의 key/value tensor를 layer별로 저장한다. Sequence length가 길어지면 저장해야 할 token 수가 늘기 때문에 KV-cache memory가 선형적으로 증가한다. Batch size가 커져도 batch마다 KV-cache가 필요하므로 memory가 다시 선형적으로 증가한다. Consumer GPU는 VRAM capacity가 제한되어 있어 long-context와 larger batch 조건에서 OOM boundary가 빠르게 나타날 수 있다. Dynamic, quantized, offloaded, no_cache 같은 cache strategy는 throughput/latency와 memory capacity 사이의 trade-off를 만든다.

## 3. 실험 질문

1. 이론적 KV-cache 크기와 실제 `past_key_values` 크기가 일치하는가?
2. Batch size와 sequence length가 커질 때 OOM boundary는 어디서 발생하는가?
3. Quantized cache가 dynamic cache의 OOM case를 rescue할 수 있는가?
4. Offloading은 GPU VRAM 문제를 해결하는가, 아니면 host memory/transfer 병목을 새로 만드는가?
5. no_cache는 어떤 lower-bound / ablation 의미를 가지는가?

## 4. 실험 환경

| 항목 | repo-backed 값 | 근거 파일 |
| --- | --- | --- |
| RTX 5060 GPU label | `rtx5060-8gb` | `results/rtx5060_qwen25_15b_dynamic_boundary.csv` |
| 실제 감지 GPU | `NVIDIA GeForce RTX 5060` | `results/rtx5060_qwen25_15b_dynamic_boundary.csv` |
| Qwen model | `/home/ssu/models/Qwen2.5-1.5B-Instruct` | `results/rtx5060_qwen25_15b_sanity.csv` |
| Qwen precision | `fp16` | `results/rtx5060_qwen25_15b_sanity.csv` |
| Qwen cache modes | dynamic, quantized, no_cache | `results/rtx5060_qwen25_15b_combined.csv` |
| Qwen dynamic batch sizes | 1, 2, 4, 8 | `results/rtx5060_qwen25_15b_dynamic_boundary.csv` |
| Qwen dynamic sequence lengths | 1024, 2048, 4096, 8192, 12288, 16384 | `results/rtx5060_qwen25_15b_dynamic_boundary.csv` |
| OPT-1.3B model | `facebook/opt-1.3b` | `results/rtx5060_opt13b_dynamic_boundary.csv` |
| OPT-1.3B cache modes | dynamic, quantized, no_cache | `results/rtx5060_opt13b_combined.csv` |
| Software stack | PyTorch 2.12.0+cu130, Transformers 5.8.1 | `README.md` |
| RTX 5080 Oaken-style evidence | OPT-125M, OPT-350M, OPT-1.3B, OPT-2.7B OK; OPT-6.7B boundary | `results/oaken_consumer_gpu_summary.csv`, `results/rtx5080/opt-6.7b/summary.md` |

## 5. 이론식

기본 MHA 모델의 FP16 KV-cache 이론식은 다음과 같이 둘 수 있다.

```text
KV bytes = 2 * num_layers * batch_size * sequence_length * hidden_size * bytes_per_element
```

여기서 `2`는 key와 value를 의미한다. 다만 GQA/MQA 모델에서는 full attention head 수를 그대로 쓰면 안 된다. Qwen2.5-1.5B sanity 결과는 `num_attention_heads=12`, `num_key_value_heads=2`, `head_dim=128`, `kv_formula_type=gqa_mqa`로 기록되어 있으며, 이 경우 KV-cache 공식은 key/value heads 기준으로 계산해야 한다 (`results/rtx5060_qwen25_15b_sanity.csv`).

## 6. RTX 5080 결과

현재 repo 안의 RTX 5080 evidence는 RTX 5060 Qwen/OPT cache-policy sweep과 같은 유형이 아니다. 5080 근거는 Oaken-style Wikitext accuracy artifact와 OPT model-size별 VRAM boundary이다. `results/oaken_consumer_gpu_summary.csv`와 per-model summary는 RTX 5080에서 OPT-125M, OPT-350M, OPT-1.3B, OPT-2.7B의 Oaken Wikitext evaluation이 완료되었고, OPT-6.7B가 upper boundary case임을 기록한다.

특히 `results/rtx5080/opt-6.7b/summary.md`는 original FP16 Wikitext eval과 Oaken profiling이 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`에서 완료되었지만, Oaken Wikitext eval은 CUDA OOM으로 실패했다고 기록한다. 같은 파일의 peak VRAM은 original eval 15806 MiB, Oaken eval 15826 MiB이다. 따라서 5080 결과는 "5080에서도 Qwen cache-policy sweep이 끝났다"가 아니라, 16GB-class consumer GPU에서도 model size가 커지면 VRAM capacity boundary가 실질적 한계가 된다는 근거로 사용해야 한다.

Prior note의 RTX 5080 `80 total / 76 OK / 4 OOM`, relative throughput, relative peak memory delta 숫자는 현재 `/home/ssu/oaken` repository 안에서 source CSV를 찾지 못했다. 따라서 이 문서에서는 그 숫자를 file-backed 확정 claim으로 사용하지 않는다.

## 7. RTX 5060 OPT-1.3B 결과

OPT-1.3B dynamic sweep은 `results/rtx5060_opt13b_dynamic_boundary.csv`에 20 rows로 기록되어 있고, dynamic OOM은 4건이다. OOM 조건은 `batch=4, seq_len=8192`, `batch=8, seq_len=4096`, `batch=8, seq_len=6144`, `batch=8, seq_len=8192`이다. Rescue file `results/rtx5060_opt13b_rescue_cases.csv`에는 quantized OK가 `batch=4, seq_len=8192`와 `batch=8, seq_len=4096`에서 기록되어 있다. 같은 rescue file에서 quantized OOM은 `batch=8, seq_len=6144`와 `batch=8, seq_len=8192`이다.

README는 offloaded cache가 15 GiB system RAM / no swap 환경에서 host memory pressure로 kernel OOM kill을 유발했다고 기록한다 (`README.md`). no_cache는 `results/rtx5060_opt13b_rescue_cases.csv`에는 OK로 기록되어 있지만, practical serving policy가 아니라 lower-bound / ablation으로 해석해야 한다.

## 8. RTX 5060 Qwen sanity 결과

Qwen sanity file `results/rtx5060_qwen25_15b_sanity.csv`는 `position_valid=True`, `max_position_embeddings=32768`, `kv_formula_type=gqa_mqa`를 기록한다. Dynamic cache sanity rows에서 `kv_actual_over_theory=1.0`이다. Dynamic boundary file `results/rtx5060_qwen25_15b_dynamic_boundary.csv`는 `batch=8, seq_len=12288`와 `batch=8, seq_len=16384`에서 OOM을 기록한다. Rescue file `results/rtx5060_qwen25_15b_rescue_cases.csv`는 quantized cache가 두 조건 모두 OK임을 기록한다.

Quantized rescue row의 `kv_actual_over_theory`는 `8x12288`에서 0.288737, `8x16384`에서 0.286865이다 (`results/rtx5060_qwen25_15b_rescue_cases.csv`). 이 숫자는 measured KV-cache tensor footprint가 FP16 theoretical KV-cache size의 약 28.7%였다는 의미이지, total CUDA peak memory가 같은 비율로 줄었다는 의미가 아니다.

## 9. 핵심 해석

- Theory vs actual: Qwen dynamic sanity와 boundary OK rows에서 `kv_actual_over_theory=1.0`이므로 GQA/MQA KV 공식이 실제 cache tensor footprint와 맞는지 확인했다 (`results/rtx5060_qwen25_15b_sanity.csv`).
- OOM boundary: RTX 5060 Qwen dynamic sweep은 `batch=8, seq_len=12288/16384`에서 OOM을 보였다 (`results/rtx5060_qwen25_15b_dynamic_boundary.csv`).
- Quantized rescue: Qwen quantized cache는 dynamic OOM이던 `8x12288`, `8x16384` 두 조건을 모두 OK로 rescue했다 (`results/rtx5060_qwen25_15b_rescue_cases.csv`).
- Offloading limitation: README에는 offloaded run이 host RAM pressure 때문에 유효 row 없이 실패했다고 기록되어 있어, offloading은 GPU memory issue를 host memory/transfer issue로 옮길 수 있다 (`README.md`).
- AI systems implication: 이 결과는 model architecture 자체가 아니라 long-context serving에서 GPU memory capacity, cache representation, host memory, throughput을 함께 고려해야 함을 보여준다.

## 10. 주장 강도 조절

### 측정한 것

- RTX 5060에서 OPT-1.3B와 Qwen2.5-1.5B의 dynamic OOM boundary.
- Qwen2.5-1.5B의 position-valid long-context 조건에서 quantized cache rescue.
- Dynamic cache의 실제 KV tensor footprint와 이론식 일치.

### 측정하지 않은 것

- Oaken paper full reproduction.
- 모든 모델/모든 GPU에서 quantized cache가 항상 유리하다는 일반 명제.
- Qwen 5080 cross-GPU cache-policy sweep; 해당 CSV는 현재 repo에서 찾지 못했다.
- RTX 5080 Qwen cache-policy sweep 또는 RTX 5080 dynamic/quantized/no_cache 비교; 현재 repo의 5080 evidence는 OPT Oaken-style Wikitext/VRAM boundary이다.
- 품질 degradation이나 perplexity 변화; Qwen boundary sweep은 random token 기반 memory sweep이다.

### Future work

- RTX 5080에서 같은 Qwen sweep을 실행해 capacity scaling을 확인한다.
- Prefill과 decode를 분리하고 token latency distribution을 측정한다.
- Quality/perplexity metric을 붙여 memory rescue와 품질 trade-off를 함께 본다.

## 11. File-backed Evidence Table

| Claim | Evidence file | Exact value | Confidence |
| --- | --- | --- | --- |
| Qwen supports position-valid tested range | `results/rtx5060_qwen25_15b_sanity.csv` | `position_valid=True`, `max_position_embeddings=32768` | HIGH |
| Qwen uses GQA/MQA formula | `results/rtx5060_qwen25_15b_sanity.csv` | `kv_formula_type=gqa_mqa`, `num_attention_heads=12`, `num_key_value_heads=2`, `head_dim=128` | HIGH |
| Qwen dynamic theory/actual matches | `results/rtx5060_qwen25_15b_sanity.csv` | dynamic `kv_actual_over_theory=1.0` | HIGH |
| Qwen dynamic OOM at 8x12288 and 8x16384 | `results/rtx5060_qwen25_15b_dynamic_boundary.csv` | OOM rows at `batch_size=8`, `seq_len=12288/16384` | HIGH |
| Qwen quantized rescues both dynamic OOM cases | `results/rtx5060_qwen25_15b_rescue_cases.csv` | quantized OK at `8x12288`, `8x16384` | HIGH |
| Qwen quantized KV footprint ratio | `results/rtx5060_qwen25_15b_rescue_cases.csv` | 0.288737 and 0.286865 | HIGH |
| OPT dynamic OOM cases | `results/rtx5060_opt13b_dynamic_boundary.csv` | 4 OOM rows | HIGH |
| OPT quantized rescue partial success | `results/rtx5060_opt13b_rescue_cases.csv` | quantized OK at `4x8192`, `8x4096`; OOM at `8x6144`, `8x8192` | HIGH |
| Offloaded host-memory issue | `README.md` | 15 GiB RAM / no swap; offloaded runs killed before valid rows | MEDIUM |
| RTX 5080 OPT Oaken Wikitext eval completed through OPT-2.7B | `results/oaken_consumer_gpu_summary.csv`, `results/rtx5080/opt-125m/summary.md`, `results/rtx5080/opt-350m/summary.md`, `results/rtx5080/opt-1.3b/summary.md`, `results/rtx5080/opt-2.7b/summary.md` | OPT-125M, OPT-350M, OPT-1.3B, OPT-2.7B status `OK` | HIGH |
| RTX 5080 OPT-6.7B is the upper boundary case | `results/oaken_consumer_gpu_summary.csv`, `results/rtx5080/opt-6.7b/summary.md` | status `Boundary`; original eval/profiling completed with expandable allocator; Oaken eval CUDA OOM | HIGH |
| RTX 5080 OPT-6.7B reached about 15.8GB peak VRAM | `results/rtx5080/opt-6.7b/summary.md` | original eval 15806 MiB; Oaken eval 15826 MiB | HIGH |
| RTX 5080 Qwen 80-row cache-policy result | not found in repository | not found in repository | LOW |
