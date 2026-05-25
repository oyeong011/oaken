# RTX 5060 Evidence Recovery

## 1. 결론

RTX 5060 OPT/Qwen boundary-rescue artifact를 로컬에서 확인했다. 원격 `origin/main`에 추가된 커밋을 fetch/rebase하면서 다음 파일들이 repository 안에 들어왔고, 발표 문서의 5060 claim은 이제 file-backed로 강화할 수 있다.

중요한 결론:

- OPT-1.3B dynamic OOM boundary와 partial quantized rescue가 CSV로 확인된다.
- Qwen2.5-1.5B는 `position_valid=True`, `kv_formula_type=gqa_mqa`, `max_position_embeddings=32768`가 CSV로 확인된다.
- Qwen dynamic OOM at `batch=8`, `seq_len=12288/16384`와 quantized rescue success가 CSV로 확인된다.
- no_cache도 rescue table에 OK로 나오지만, practical serving policy가 아니라 ablation/lower-bound로만 해석해야 한다.

## 2. 실행한 검색/확인 명령

```sh
find /home/ssu/oaken /home/ssu/kv_cache_consumer_gpu_bench /home/ssu/kv-cache-consumer-gpu-bench -type f \( -iname '*5060*' -o -iname '*rtx5060*' -o -iname '*qwen*' -o -iname '*rescue*' -o -iname '*boundary*' -o -iname '*oom*' -o -iname '*kv_cache*' -o -iname '*kv-cache*' -o -iname '*sweep*' -o -iname '*opt*' \) 2>/dev/null
```

```sh
rg -n "position_valid|kv_formula_type|12288|16384|0\.288737|0\.286865|rescue|rtx5060|5060" /home/ssu/oaken /home/ssu/kv_cache_consumer_gpu_bench --glob '!**/.venv/**' --glob '!**/transformers/**' --glob '!**/lm-evaluation-harness/**' --glob '!**/__pycache__/**'
```

```sh
find results -maxdepth 4 -type f \( -iname '*5060*' -o -iname '*qwen*' -o -iname '*rescue*' -o -iname '*boundary*' -o -iname '*oom*' \) -print
```

```sh
head -5 results/rtx5060_qwen25_15b_dynamic_boundary.csv
head -5 results/rtx5060_qwen25_15b_rescue_cases.csv
cat results/plots_rtx5060_qwen25_15b_combined/dynamic_oom_rescue_cases.csv
head -5 results/rtx5060_opt13b_dynamic_boundary.csv
cat results/plots_rtx5060_combined/dynamic_oom_rescue_cases.csv
```

## 3. Copied evidence inventory

Relevant files were copied into `results/imported_5060_evidence/` for presentation-oriented access. The original files remain in their canonical result paths.

| Original path | Copied path | Type | Supports | Confidence | Presentation sufficient? |
| --- | --- | --- | --- | --- | --- |
| `results/rtx5060_opt13b_dynamic_boundary.csv` | `results/imported_5060_evidence/opt13b/rtx5060_opt13b_dynamic_boundary.csv` | CSV | OPT dynamic OOM boundary | HIGH | Yes |
| `results/rtx5060_opt13b_rescue_cases.csv` | `results/imported_5060_evidence/opt13b/rtx5060_opt13b_rescue_cases.csv` | CSV | OPT quantized/no_cache rescue cases | HIGH | Yes |
| `results/plots_rtx5060_combined/dynamic_oom_rescue_cases.csv` | `results/imported_5060_evidence/opt13b/dynamic_oom_rescue_cases.csv` | CSV | OPT dynamic OOM cases rescued by other modes | HIGH | Yes |
| `results/rtx5060_qwen25_15b_sanity.csv` | `results/imported_5060_evidence/qwen25_15b/rtx5060_qwen25_15b_sanity.csv` | CSV | Qwen position-valid, GQA/MQA formula sanity | HIGH | Yes |
| `results/rtx5060_qwen25_15b_dynamic_boundary.csv` | `results/imported_5060_evidence/qwen25_15b/rtx5060_qwen25_15b_dynamic_boundary.csv` | CSV | Qwen dynamic OOM boundary | HIGH | Yes |
| `results/rtx5060_qwen25_15b_rescue_cases.csv` | `results/imported_5060_evidence/qwen25_15b/rtx5060_qwen25_15b_rescue_cases.csv` | CSV | Qwen quantized/no_cache rescue cases | HIGH | Yes |
| `results/plots_rtx5060_qwen25_15b_combined/dynamic_oom_rescue_cases.csv` | `results/imported_5060_evidence/qwen25_15b/dynamic_oom_rescue_cases.csv` | CSV | Qwen dynamic OOM cases rescued by other modes | HIGH | Yes |

## 4. Verified claims

| Claim | Evidence file | Exact value | Confidence |
| --- | --- | --- | --- |
| Qwen position window is valid for tested rows | `results/rtx5060_qwen25_15b_sanity.csv` | `position_valid=True`, `max_position_embeddings=32768` | HIGH |
| Qwen formula is GQA/MQA-aware | `results/rtx5060_qwen25_15b_sanity.csv` | `kv_formula_type=gqa_mqa`, `num_attention_heads=12`, `num_key_value_heads=2`, `head_dim=128` | HIGH |
| Dynamic Qwen actual/theory sanity | `results/rtx5060_qwen25_15b_sanity.csv` | `kv_actual_over_theory=1.0` for dynamic sanity rows | HIGH |
| Qwen dynamic OOM boundary | `results/rtx5060_qwen25_15b_dynamic_boundary.csv` | OOM at `batch_size=8`, `seq_len=12288` and `16384` | HIGH |
| Qwen quantized rescue | `results/rtx5060_qwen25_15b_rescue_cases.csv` | quantized OK at `8x12288` and `8x16384` | HIGH |
| Qwen quantized actual/theory ratio | `results/rtx5060_qwen25_15b_rescue_cases.csv` | `0.288737` and `0.286865` | HIGH |
| OPT dynamic OOM boundary | `results/rtx5060_opt13b_dynamic_boundary.csv` | OOM at `4x8192`, `8x4096`, `8x6144`, `8x8192` | HIGH |
| OPT quantized partial rescue | `results/rtx5060_opt13b_rescue_cases.csv` | quantized OK at `4x8192`, `8x4096`; quantized OOM at `8x6144`, `8x8192` | HIGH |

## 5. Presentation use

이제 RTX 5060 Qwen rescue는 발표에서 사용할 수 있다. 다만 문장 강도는 다음처럼 유지한다.

안전한 문장:

> RTX 5060 8GB의 position-valid Qwen2.5-1.5B 실험에서 dynamic cache가 OOM이던 `batch=8, seq_len=12288/16384` 조건을 quantized cache가 실행 가능하게 만든 것을 CSV로 확인했습니다.

피해야 할 문장:

> quantized cache가 모든 long-context 문제를 해결했습니다.

no_cache는 rescue table에 OK로 나오지만 throughput/serving 해석상 ablation이다. offloaded는 host memory pressure와 transfer bottleneck 관점의 limitation으로 설명한다.

