# RTX 5060 FP16 KV-Cache Growth Baseline

This experiment measures the baseline FP16 KV-cache memory growth that motivates Oaken-like KV-cache quantization.

This is not an Oaken reproduction. This experiment isolates KV-cache memory pressure before evaluating Oaken's quantized path.

## Setup

- GPU: NVIDIA GeForce RTX 5060
- Models: /data/models/opt-350m, /data/models/opt-1.3b
- Inputs: random token IDs, not accuracy data.
- Runtime: Hugging Face Transformers, PyTorch, CUDA fp16, `model.eval()`, `torch.no_grad()`.
- Main sweep: sequence lengths 128, 256, 512, 1024, 2048 with batch sizes 1, 2, 4 and `use_cache=True`.
- Extra comparison: `use_cache=False` for batch size 1 and sequence lengths 128, 512, 1024.

## Headline Results

- Actual `past_key_values` size matches the theoretical formula closely: theoretical/actual ratio range `1.000000` to `1.000000` for successful `use_cache=True` runs.
- No CUDA OOM occurred in the completed sweep.
- Largest successful reserved-memory peak: `opt-1.3b` batch=4 seq=2048 use_cache=True at `5584.000000 MiB` reserved.

## Theoretical vs Actual KV Cache

The formula used is:

```text
KV bytes = 2 * num_layers * batch_size * sequence_length * hidden_size * bytes_per_element
```

For successful `use_cache=True` rows, actual bytes are computed by summing `numel() * element_size()` for every key and value tensor in `output.past_key_values`.

## Sequence Length Scaling

### opt-350m
- batch=1: 128: 12.00 MiB, 256: 24.00 MiB, 512: 48.00 MiB, 1024: 96.00 MiB, 2048: 192.00 MiB; 128->2048 expected 16.00x, observed 16.00x.
- batch=2: 128: 24.00 MiB, 256: 48.00 MiB, 512: 96.00 MiB, 1024: 192.00 MiB, 2048: 384.00 MiB; 128->2048 expected 16.00x, observed 16.00x.
- batch=4: 128: 48.00 MiB, 256: 96.00 MiB, 512: 192.00 MiB, 1024: 384.00 MiB, 2048: 768.00 MiB; 128->2048 expected 16.00x, observed 16.00x.

### opt-1.3b
- batch=1: 128: 24.00 MiB, 256: 48.00 MiB, 512: 96.00 MiB, 1024: 192.00 MiB, 2048: 384.00 MiB; 128->2048 expected 16.00x, observed 16.00x.
- batch=2: 128: 48.00 MiB, 256: 96.00 MiB, 512: 192.00 MiB, 1024: 384.00 MiB, 2048: 768.00 MiB; 128->2048 expected 16.00x, observed 16.00x.
- batch=4: 128: 96.00 MiB, 256: 192.00 MiB, 512: 384.00 MiB, 1024: 768.00 MiB, 2048: 1536.00 MiB; 128->2048 expected 16.00x, observed 16.00x.

## Batch Size Scaling

### opt-350m
- seq=128: batch=1: 12.00 MiB, batch=2: 24.00 MiB, batch=4: 48.00 MiB; expected 4.00x, observed 4.00x.
- seq=256: batch=1: 24.00 MiB, batch=2: 48.00 MiB, batch=4: 96.00 MiB; expected 4.00x, observed 4.00x.
- seq=512: batch=1: 48.00 MiB, batch=2: 96.00 MiB, batch=4: 192.00 MiB; expected 4.00x, observed 4.00x.
- seq=1024: batch=1: 96.00 MiB, batch=2: 192.00 MiB, batch=4: 384.00 MiB; expected 4.00x, observed 4.00x.
- seq=2048: batch=1: 192.00 MiB, batch=2: 384.00 MiB, batch=4: 768.00 MiB; expected 4.00x, observed 4.00x.

### opt-1.3b
- seq=128: batch=1: 24.00 MiB, batch=2: 48.00 MiB, batch=4: 96.00 MiB; expected 4.00x, observed 4.00x.
- seq=256: batch=1: 48.00 MiB, batch=2: 96.00 MiB, batch=4: 192.00 MiB; expected 4.00x, observed 4.00x.
- seq=512: batch=1: 96.00 MiB, batch=2: 192.00 MiB, batch=4: 384.00 MiB; expected 4.00x, observed 4.00x.
- seq=1024: batch=1: 192.00 MiB, batch=2: 384.00 MiB, batch=4: 768.00 MiB; expected 4.00x, observed 4.00x.
- seq=2048: batch=1: 384.00 MiB, batch=2: 768.00 MiB, batch=4: 1536.00 MiB; expected 4.00x, observed 4.00x.

## use_cache=False Comparison

### opt-350m
- seq=128: use_cache=True allocated peak `666.14 MiB`; use_cache=False allocated peak `654.14 MiB`; delta `12.00 MiB`.
- seq=512: use_cache=True allocated peak `740.25 MiB`; use_cache=False allocated peak `692.25 MiB`; delta `48.00 MiB`.
- seq=1024: use_cache=True allocated peak `836.94 MiB`; use_cache=False allocated peak `740.94 MiB`; delta `96.00 MiB`.

### opt-1.3b
- seq=128: use_cache=True allocated peak `2555.51 MiB`; use_cache=False allocated peak `2531.51 MiB`; delta `24.00 MiB`.
- seq=512: use_cache=True allocated peak `2666.74 MiB`; use_cache=False allocated peak `2570.74 MiB`; delta `96.00 MiB`.
- seq=1024: use_cache=True allocated peak `2812.94 MiB`; use_cache=False allocated peak `2620.94 MiB`; delta `192.00 MiB`.

## Why Torch Peak Memory Is Larger Than The KV Formula

The theoretical KV formula covers only stored key/value tensors. Torch peak memory also includes FP16 model weights, input tensors, attention masks, intermediate activations that exist during the forward pass, logits/loss-related outputs, CUDA allocator fragmentation, reserved caching allocator blocks, and framework workspaces. Therefore peak allocated/reserved memory is expected to be larger than pure `past_key_values` storage.

## RTX 5060 8GB Boundary

No OOM occurred. The closest observed condition to the 8GB boundary was `opt-1.3b` batch=4 seq=2048 use_cache=True, with `5584.000000 MiB` reserved.

## Connection To Oaken

This baseline shows the FP16 KV-cache term that Oaken-like KV-cache quantization targets. Because actual `past_key_values` bytes match the theoretical linear formula, reducing the bytes per KV element is a direct way to reduce the cache component. This experiment does not evaluate Oaken's quantized path; it establishes the baseline memory pressure before that comparison.

## Limitations

- Inputs are random token IDs, so this is a memory-growth test rather than an accuracy or perplexity test.
- This is prefill-style full-context forward measurement, not an incremental decoding loop with one token appended at a time.
- Peak torch memory includes more than KV cache, so the KV formula should be compared against actual `past_key_values` bytes, not total peak memory.
- `nvidia-smi` VRAM sampling is best-effort and can miss short peaks; torch peak memory is the primary measurement.

## Artifacts

- `kv_cache_growth.csv`: machine-readable results.
- `logs.md`: per-run raw log and VRAM CSV index.
- `raw_logs/*.log`: per-run tensor shapes, byte counts, and errors.
- `raw_logs/*_vram.csv`: best-effort `nvidia-smi` samples.
