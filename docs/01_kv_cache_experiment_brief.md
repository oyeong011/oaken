# KV-cache 실험 브리핑

## 1. 한 문장 요약

이 실험은 Oaken-inspired 관점에서 consumer GPU의 LLM inference가 batch size와 sequence length 증가에 따라 KV-cache memory pressure와 OOM boundary를 어떻게 만나는지, 그리고 cache mode가 어떤 trade-off를 만드는지 로컬 artifact로 확인한 작업이다.

## 2. 실험 동기

Autoregressive LLM inference에서는 이전 token의 key/value tensor를 layer별로 저장해 다음 token 계산에서 재사용한다. 이 KV-cache는 sequence length가 길어질수록 선형으로 커지고, batch size가 커져도 선형으로 증가한다. Consumer GPU는 VRAM 여유가 제한적이므로 long-context 또는 larger-batch serving에서 cache가 memory-capacity pressure를 빠르게 만든다. 따라서 dynamic, quantized, offloaded, no_cache 같은 cache strategy는 단순 속도 옵션이 아니라 memory와 throughput/latency 사이의 trade-off로 해석해야 한다.

## 3. 실험 질문

1. 이론적 KV-cache 크기와 실제 `past_key_values` 크기가 일치하는가?
2. batch size와 sequence length가 커질 때 OOM boundary는 어디서 발생하는가?
3. quantized cache가 dynamic cache의 OOM case를 rescue할 수 있는가?
4. offloading은 GPU VRAM 문제를 해결하는가, 아니면 host memory/transfer 병목을 새로 만드는가?
5. no_cache는 어떤 lower-bound / ablation 의미를 가지는가?

## 4. 실험 환경

| Item | File-backed value |
| --- | --- |
| RTX 5080 Qwen GPU | NVIDIA GeForce RTX 5080 (`/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`) |
| RTX 5080 Qwen model | `Qwen/Qwen2.5-1.5B-Instruct` (`/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`) |
| RTX 5080 Qwen dtype | fp16 (`/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`) |
| RTX 5080 Qwen cache modes | dynamic, quantized, offloaded, no_cache (`/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`) |
| RTX 5080 Qwen batch sizes | 1, 2, 4, 8 (`/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_qwen25_analysis.md`) |
| RTX 5080 Qwen sequence lengths | 512, 1024, 2048, 4096, 8192 (`/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_qwen25_analysis.md`) |
| Oaken artifact GPUs | RTX 5060 and RTX 5080 (`results/oaken_consumer_gpu_summary.csv`) |
| Software stack | repository에서 확인 필요; 일부 local run에서 Transformers/PyTorch 환경은 command output에만 있음. |

## 5. 이론식

일반 MHA 모델의 단순 KV-cache 식:

```text
KV bytes = 2 * num_layers * batch_size * sequence_length * hidden_size * bytes_per_element
```

GQA/MQA 모델에서는 full hidden size를 그대로 쓰면 안 되고, key/value head 수를 사용해야 한다.

```text
KV bytes = 2 * num_layers * batch_size * sequence_length * num_key_value_heads * head_dim * bytes_per_element
```

RTX 5080 Qwen2.5 CSV는 `num_attention_heads=12`, `num_key_value_heads=2`, `head_dim=128`을 기록한다 (`/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`). 따라서 Qwen2.5 결과는 GQA/MQA-aware formula로 해석해야 한다.

## 6. RTX 5080 결과

RTX 5080 Qwen2.5 sweep은 총 80 rows, 76 OK, 4 OOM을 기록했다 (`/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_status_count.csv`). OOM은 모든 cache mode에서 `batch_size=8`, `seq_len=8192`에서 발생했다 (`/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_oom_cases.csv`).

성공 row의 `kv_actual_over_theory`는 cache mode별 평균/최소/최대가 모두 1.0이었다 (`/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_qwen25_analysis.md`). Dynamic 대비 평균 throughput ratio는 quantized 0.744150, offloaded 0.594451, no_cache 0.093942였다 (`/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_ratios_vs_dynamic.csv`). Dynamic 대비 평균 peak memory delta ratio는 quantized 0.786496, offloaded 0.705081, no_cache 0.738751였다 (`/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_ratios_vs_dynamic.csv`).

대표 long-context case `batch_size=4`, `seq_len=8192`에서 dynamic은 124.957829 tokens/s, quantized는 108.237820 tokens/s, offloaded는 45.451941 tokens/s, no_cache는 2.951665 tokens/s였다 (`/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_key_examples.csv`).

## 7. RTX 5060 OPT-1.3B 결과

파일로 확인 가능한 RTX 5060 OPT-1.3B 결과는 Oaken artifact accuracy/VRAM summary다. `results/oaken_consumer_gpu_summary.csv`에 따르면 RTX 5060 OPT-1.3B는 original PPL 14.6406, Oaken PPL 15.3984, peak VRAM 4.49GB, status OK로 기록되어 있다.

사용자 메모에는 dynamic OOM과 quantized rescue case가 있으나, 현재 저장소에서 `results/rtx5060_opt13b_dynamic_boundary.csv`와 `results/rtx5060_opt13b_rescue_cases.csv`가 발견되지 않았다. 따라서 dynamic OOM/rescue 수치는 이 문서에서 강한 measured claim으로 쓰지 않는다.

## 8. RTX 5060 Qwen sanity 결과

현재 저장소에는 RTX 5060 Qwen sanity/boundary/rescue CSV가 없다. `results/rtx5060_qwen25_15b_dynamic_boundary.csv`와 `results/rtx5060_qwen25_15b_rescue_cases.csv`는 not found in repository다.

Qwen2.5의 GQA 구조 자체는 RTX 5080 Qwen CSV에서 `num_attention_heads=12`, `num_key_value_heads=2`, `head_dim=128`로 확인된다 (`/home/ssu/kv_cache_consumer_gpu_bench/results/results_5080_qwen25_1p5b.csv`). 다만 `position_valid`, `max_position_embeddings=32768`, RTX 5060 dynamic OOM/rescue 결과는 현재 파일 근거가 없다.

## 9. 핵심 해석

- 이론식 vs 실제값: RTX 5080 Qwen2.5 성공 row에서 `kv_actual_over_theory=1.0`이므로 KV-cache tensor footprint 계산은 file-backed sanity check를 통과했다.
- OOM boundary: RTX 5080 Qwen2.5에서는 모든 cache mode가 `batch_size=8`, `seq_len=8192`에서 OOM을 기록했다.
- Quantized rescue: RTX 5080 Qwen2.5에서는 가장 큰 OOM case를 quantized도 rescue하지 못했다; RTX 5060 rescue 주장은 현재 CSV 근거가 없어 확인 필요다.
- Offloading limitation: RTX 5080 Qwen2.5에서 offloaded는 memory delta를 낮췄지만 throughput ratio가 dynamic의 0.594451로 낮아졌다.
- AI systems 의미: 이 실험은 model architecture 개선보다 serving에서 memory hierarchy, batching, context length, cache policy가 만드는 실제 병목을 계측하는 방향의 준비로 해석하는 것이 안전하다.

## 10. 주장 강도 조절

### What I measured

- RTX 5080 Qwen2.5 cache-mode sweep: 80 rows, 76 OK, 4 OOM (`/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_status_count.csv`).
- RTX 5060/5080 Oaken artifact accuracy/VRAM summary (`results/oaken_consumer_gpu_summary.csv`).

### What I did not measure

- Full Oaken hardware throughput reproduction.
- RTX 5060 Qwen dynamic OOM/rescue result with local CSV evidence.
- Quality/perplexity impact for HF Qwen cache-mode sweep.
- Prefill and decode latency separated as independent distributions.

### Future work

- Export RTX 5060 Qwen sanity/dynamic/rescue CSVs.
- Add position-valid long-context analysis to the result table.
- Separate prefill/decode and add fixed generation length controls.

## 11. File-backed Evidence Table

| Claim | Evidence file | Exact value | Confidence |
| --- | --- | --- | --- |
| RTX 5080 Qwen sweep row count | `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_status_count.csv` | 76 OK, 4 OOM | HIGH |
| RTX 5080 Qwen OOM boundary | `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_oom_cases.csv` | all modes OOM at batch_size=8, seq_len=8192 | HIGH |
| KV actual/theory check | `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_qwen25_analysis.md` | mean/min/max 1.0 for each cache mode | HIGH |
| Throughput ratio quantized | `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_ratios_vs_dynamic.csv` | 0.744150 | HIGH |
| Throughput ratio offloaded | `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_ratios_vs_dynamic.csv` | 0.594451 | HIGH |
| Throughput ratio no_cache | `/home/ssu/kv_cache_consumer_gpu_bench/analysis/rtx5080_ratios_vs_dynamic.csv` | 0.093942 | HIGH |
| RTX 5060 OPT-1.3B Oaken PPL | `results/oaken_consumer_gpu_summary.csv` | 15.3984 | HIGH |
| RTX 5060 OPT-1.3B dynamic OOM/rescue | not found in repository | not found in repository | LOW |
| RTX 5060 Qwen dynamic OOM/rescue | not found in repository | not found in repository | LOW |
