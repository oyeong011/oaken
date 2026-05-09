# RTX 5080 OPT-1.3B Run Log

## Model Preparation

Small metadata and tokenizer files were downloaded with `huggingface_hub`.
The PyTorch weight was downloaded by direct resolve URL:

```sh
curl -L --fail --progress-bar \
  -o /data/models/opt-1.3b/pytorch_model.bin \
  https://huggingface.co/facebook/opt-1.3b/resolve/main/pytorch_model.bin
```

Model directory after download:

```text
/data/models/opt-1.3b: 2.5G
pytorch_model.bin: 2.5G
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

## Original Evaluation

```sh
python eval_perplexity.py \
  -m opt \
  -s 1.3b \
  -t wikitext \
  --quant-method none \
  --gpu-start-idx 0 \
  --gpu-count 1
```

Observed:

```text
Original PPL: tensor(14.6406, device='cuda:0', dtype=torch.float16)
Wall time: 18 seconds
Peak VRAM: 4580 MiB
```

## Oaken Offline Profiling

```sh
python oaken_preprocess_activation.py \
  -m opt \
  -s 1.3b \
  -t wikitext \
  -f 0.04 0.9 0.06 \
  -o quantizer/oaken/opt-1.3b.json \
  --gpu-start-idx 0 \
  --gpu-count 1
```

Observed:

```text
Profiling pass PPL: tensor(14.6406, device='cuda:0', dtype=torch.float16)
Elapsed time: 34.80459634399995 seconds
Wall time: 38 seconds
Peak VRAM: 4668 MiB
```

## Oaken Evaluation

```sh
python eval_perplexity.py \
  -m opt \
  -s 1.3b \
  -t wikitext \
  -q quantizer/oaken/opt-1.3b.json \
  --quant-method oaken \
  --gpu-start-idx 0 \
  --gpu-count 1
```

Observed:

```text
Oaken PPL: tensor(15.3984, device='cuda:0', dtype=torch.float16)
Wall time: 20 seconds
Peak VRAM: 4612 MiB
Total Sparsity:
  Key   [0.04021413909407866, 0.89989575007424, 0.05989011101552394]
  Value [0.039989044290943056, 0.9000348530833039, 0.059976102774368316]
```
