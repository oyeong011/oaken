# RTX 5080 OPT-125M Run Log

## Tiny CUDA Gate

```sh
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.zeros(1, device="cuda"))
PY
```

Output:

```text
True
NVIDIA GeForce RTX 5080
tensor([0.], device='cuda:0')
```

## Profiling

```sh
python oaken_preprocess_activation.py \
  -m opt \
  -s 125m \
  -t wikitext \
  -f 0.04 0.9 0.06 \
  -o quantizer/oaken/opt-125m.json \
  --gpu-start-idx 0 \
  --gpu-count 1
```

Observed:

```text
Original PPL: tensor(27.6719, device='cuda:0', dtype=torch.float16)
Elapsed time: 16.69216263300018 seconds
Retest elapsed time: 10.69117203199994 seconds
Retest wall time: 14 seconds
Retest peak VRAM: 1584 MiB
```

## Oaken Evaluation

```sh
python eval_perplexity.py \
  -m opt \
  -s 125m \
  -t wikitext \
  -q quantizer/oaken/opt-125m.json \
  --quant-method oaken \
  --gpu-start-idx 0 \
  --gpu-count 1
```

Observed:

```text
Oaken PPL: tensor(28., device='cuda:0', dtype=torch.float16)
Eval wall time: 10 seconds
Eval peak VRAM: 1542 MiB
Total Sparsity:
  Key   [0.040422578194744485, 0.8996245212518295, 0.05995292292057974]
  Value [0.03992296642146071, 0.9000447796521175, 0.060032276278163516]
```
