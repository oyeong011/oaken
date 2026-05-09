# RTX 5080 OPT-350M Run Log

## Model Download

The Hugging Face Python downloader fetched metadata but stalled on `pytorch_model.bin`.
The weight file was downloaded via the direct resolve URL:

```sh
curl -L --fail --progress-bar \
  -o /data/models/opt-350m/pytorch_model.bin \
  https://huggingface.co/facebook/opt-350m/resolve/main/pytorch_model.bin
```

Model directory after download:

```text
/data/models/opt-350m: 634M
pytorch_model.bin: 632M
```

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
  -s 350m \
  -t wikitext \
  -f 0.04 0.9 0.06 \
  -o quantizer/oaken/opt-350m.json \
  --gpu-start-idx 0 \
  --gpu-count 1
```

Observed:

```text
Original PPL: tensor(22.0156, device='cuda:0', dtype=torch.float16)
Elapsed time: 16.536413129000266 seconds
Wall time: 20 seconds
Peak VRAM: 2112 MiB
```

## Original Evaluation

```sh
python eval_perplexity.py \
  -m opt \
  -s 350m \
  -t wikitext \
  --quant-method none \
  --gpu-start-idx 0 \
  --gpu-count 1
```

Observed:

```text
Original PPL: tensor(22.0156, device='cuda:0', dtype=torch.float16)
```

## Oaken Evaluation

```sh
python eval_perplexity.py \
  -m opt \
  -s 350m \
  -t wikitext \
  -q quantizer/oaken/opt-350m.json \
  --quant-method oaken \
  --gpu-start-idx 0 \
  --gpu-count 1
```

Observed after the zero-range quantization fix:

```text
Oaken PPL: tensor(22.1406, device='cuda:0', dtype=torch.float16)
Eval wall time: 13 seconds
Eval peak VRAM: 2080 MiB
Total Sparsity:
  Key   [0.040546790189462015, 0.8996099501883448, 0.059843259768881975]
  Value [0.04002316862820311, 0.8999973365389707, 0.05997949503097974]
```

Before the zero-range fix, the Oaken eval process exited with status 0 but produced:

```text
Oaken PPL: tensor(nan, device='cuda:0', dtype=torch.float16)
```
