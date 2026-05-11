# Oaken OPT-350M 데이터셋 안정성 실험 개요

## 이 논문이 뭔가

이 실험은 Oaken 논문, **“Oaken: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization”**(ISCA 2025)을 검증하기 위한 재현 실험입니다.  
논문의 핵심 아이디어는 LLM의 KV cache를 온라인/오프라인 혼합 방식으로 양자화해서 메모리 사용량과 지연시간을 줄이는 것입니다.

논문에서 특히 중요한 가정은 다음입니다.

- 모델과 레이어별로 KV 분포는 다르지만
- 같은 모델 안에서는 입력 데이터셋이 달라도 offline profiling으로 얻은 threshold가 크게 흔들리지 않는다
- 그래서 한 번 오프라인으로 threshold를 추정해 두고, 이후 입력에서는 그 threshold를 재사용할 수 있다

이 실험은 그 가정을 직접 확인하는 용도입니다.

## 왜 이 실험을 했나

Oaken의 offline profiling이 실제로 성립하려면, 기준 데이터셋에서 얻은 threshold가 다른 데이터셋에서도 어느 정도 재사용 가능해야 합니다.  
즉, 이 실험은 다음 질문에 답하려고 합니다.

1. Wikitext로 얻은 threshold를 baseline으로 잡을 수 있는가
2. Winogrande, Hellaswag에서도 layer-wise key/value range가 비슷하게 유지되는가
3. 그 정도면 offline profiling이 실용적인 근사치라고 말할 수 있는가

요약하면, **“한 번만 프로파일링해도 되는가”**를 확인하는 실험입니다.

## 무엇을 측정했나

- 모델: `OPT-350M`
- baseline dataset: `wikitext`
- 비교 dataset: `winogrande`, `hellaswag`
- 실패 dataset: `piqa`
- group ratio: `0.04 0.9 0.06`
- 비교 대상: layer별 key/value threshold

핵심 산출물은 [`thresholds_by_dataset.csv`](./thresholds_by_dataset.csv)와 [`stability_summary.md`](./stability_summary.md)입니다.

## 결과 그래프

matplotlib 환경 문제로 PNG 그래프는 생성되지 않았습니다.  
대신 아래의 텍스트 그래프처럼 보면 됩니다.

### Wikitext 대비 평균 절대 변화

```text
abs_max mean abs diff
Wikitext    | 0.000000
Winogrande  | 0.143692
Hellaswag   | 0.094686

width mean abs diff
Wikitext    | 0.000000
Winogrande  | 0.290616
Hellaswag   | 0.195060
```

### key vs value

```text
Mean abs_max abs diff
Winogrande key   | 0.246542
Winogrande value | 0.040841
Hellaswag  key   | 0.163701
Hellaswag  value | 0.025670
```

이 패턴은 key 쪽이 value 쪽보다 더 흔들리고, 특히 큰 range를 담는 group 0에서 차이가 크다는 뜻입니다.

## 결과를 어떻게 읽어야 하나

- `wikitext`는 baseline이라 차이가 0입니다.
- `winogrande`와 `hellaswag`는 Wikitext와 완전히 동일하지는 않지만, 같은 layer/tensor/group 구조는 유지합니다.
- 절대값 기준으로는 대체 가능성이 보이지만, 상대값 기준으로는 특히 작은 group 2에서 차이가 크게 보입니다.
- 따라서 이 실험은 **offline profiling이 아예 틀렸다는 증거는 아니지만**, **dataset-independent threshold라고 단정할 정도로 안정적이지도 않다**는 쪽에 가깝습니다.

## 이 실험이 무엇을 위한 건가

이 실험은 Oaken의 offline profiling 가정을 데이터셋 축에서 점검하기 위한 것입니다.  
즉, Oaken이 말하는 “미리 threshold를 뽑아두고 나중에 재사용하는 방식”이 실제로 얼마나 일반화되는지 확인하는 용도입니다.

이 결과가 의미하는 바는 다음과 같습니다.

- 실용성: Wikitext 기반 profiling은 OPT-350M에서 다른 benchmark에도 어느 정도 재사용 가능해 보인다
- 한계: key range와 작은 group에서는 데이터셋에 따라 drift가 커질 수 있다
- 결론: offline profiling은 plausibility를 주지만, 엄밀한 dataset invariance는 증명하지 못한다

## 한계

- PIQA는 현재 runner의 dataset-script loader 문제로 실패했습니다.
- 이 결과는 OPT-350M만 다룹니다.
- threshold similarity만 본 것이고, cross-dataset quantizer를 실제 accuracy에 적용한 평가는 아닙니다.
- 따라서 이 문서는 “threshold가 비슷한가”를 말할 뿐, “다른 데이터셋에 그대로 써도 정확도가 보장되는가”까지는 말하지 않습니다.

## 같이 보면 좋은 파일

- [interpretation.md](./interpretation.md)
- [stability_summary.md](./stability_summary.md)
- [thresholds_by_dataset.csv](./thresholds_by_dataset.csv)
