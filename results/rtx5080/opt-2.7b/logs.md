# RTX 5080 OPT-2.7B Run Log

## Model Preparation

Small metadata and tokenizer files were downloaded with `huggingface_hub`.
The PyTorch weight was downloaded by direct resolve URL:

```sh
curl -L --fail --progress-bar \
  -o /data/models/opt-2.7b/pytorch_model.bin \
  https://huggingface.co/facebook/opt-2.7b/resolve/main/pytorch_model.bin
```

Model directory after download:

```text
/data/models/opt-2.7b: 5.0G
pytorch_model.bin: 5.0G
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
  -s 2.7b \
  -t wikitext \
  --quant-method none \
  --gpu-start-idx 0 \
  --gpu-count 1
```

Observed:

```text
Original PPL: tensor(12.4688, device='cuda:0', dtype=torch.float16)
Wall time: 25 seconds
Peak VRAM: 7690 MiB
```

## Oaken Offline Profiling

```sh
python oaken_preprocess_activation.py \
  -m opt \
  -s 2.7b \
  -t wikitext \
  -f 0.04 0.9 0.06 \
  -o quantizer/oaken/opt-2.7b.json \
  --gpu-start-idx 0 \
  --gpu-count 1
```

Observed:

```text
Profiling pass PPL: tensor(12.4688, device='cuda:0', dtype=torch.float16)
Elapsed time: 60.50044054799946 seconds
Wall time: 64 seconds
Peak VRAM: 7830 MiB
```

## Oaken Evaluation

```sh
python eval_perplexity.py \
  -m opt \
  -s 2.7b \
  -t wikitext \
  -q quantizer/oaken/opt-2.7b.json \
  --quant-method oaken \
  --gpu-start-idx 0 \
  --gpu-count 1
```

Observed:

```text
Oaken PPL: tensor(12.5703, device='cuda:0', dtype=torch.float16)
Wall time: 30 seconds
Peak VRAM: 7732 MiB
Total Sparsity:
  Key   [0.03983435272866636, 0.9000911041037409, 0.06007455684020058]
  Value [0.040053794917184866, 0.8999215492463492, 0.060024669257666356]
```
