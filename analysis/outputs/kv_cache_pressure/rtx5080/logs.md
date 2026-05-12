# RTX 5080 KV-Cache Pressure Logs

## Commands

### Original FP16 sequence length 128

```sh
docker exec -w /workspace oaken-ae-container /opt/conda/envs/oaken/bin/python eval_perplexity.py -m opt -s 1.3b -t wikitext --max-length 128 --stride 128 --gpu-start-idx 0 --gpu-count 1 --quant-method none
```

- Status: `OK`
- PPL: `35.375`
- Peak VRAM: `3421` MiB
- Elapsed: `23.54` seconds
- Log: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq128_original.log`
- VRAM CSV: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq128_original_vram.csv`

### Oaken sequence length 128

```sh
docker exec -w /workspace oaken-ae-container /opt/conda/envs/oaken/bin/python eval_perplexity.py -m opt -s 1.3b -t wikitext --max-length 128 --stride 128 --gpu-start-idx 0 --gpu-count 1 -q quantizer/oaken/opt-1.3b.json --quant-method oaken
```

- Status: `OK`
- PPL: `36.6562`
- Peak VRAM: `3427` MiB
- Elapsed: `55.57` seconds
- Log: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq128_oaken.log`
- VRAM CSV: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq128_oaken_vram.csv`

### Original FP16 sequence length 256

```sh
docker exec -w /workspace oaken-ae-container /opt/conda/envs/oaken/bin/python eval_perplexity.py -m opt -s 1.3b -t wikitext --max-length 256 --stride 256 --gpu-start-idx 0 --gpu-count 1 --quant-method none
```

- Status: `OK`
- PPL: `25.8906`
- Peak VRAM: `3471` MiB
- Elapsed: `18.86` seconds
- Log: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq256_original.log`
- VRAM CSV: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq256_original_vram.csv`

### Oaken sequence length 256

```sh
docker exec -w /workspace oaken-ae-container /opt/conda/envs/oaken/bin/python eval_perplexity.py -m opt -s 1.3b -t wikitext --max-length 256 --stride 256 --gpu-start-idx 0 --gpu-count 1 -q quantizer/oaken/opt-1.3b.json --quant-method oaken
```

- Status: `OK`
- PPL: `26.9219`
- Peak VRAM: `3477` MiB
- Elapsed: `33.80` seconds
- Log: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq256_oaken.log`
- VRAM CSV: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq256_oaken_vram.csv`

### Original FP16 sequence length 512

```sh
docker exec -w /workspace oaken-ae-container /opt/conda/envs/oaken/bin/python eval_perplexity.py -m opt -s 1.3b -t wikitext --max-length 512 --stride 512 --gpu-start-idx 0 --gpu-count 1 --quant-method none
```

- Status: `OK`
- PPL: `20.2031`
- Peak VRAM: `3369` MiB
- Elapsed: `17.61` seconds
- Log: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq512_original.log`
- VRAM CSV: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq512_original_vram.csv`

### Oaken sequence length 512

```sh
docker exec -w /workspace oaken-ae-container /opt/conda/envs/oaken/bin/python eval_perplexity.py -m opt -s 1.3b -t wikitext --max-length 512 --stride 512 --gpu-start-idx 0 --gpu-count 1 -q quantizer/oaken/opt-1.3b.json --quant-method oaken
```

- Status: `OK`
- PPL: `21.0938`
- Peak VRAM: `3373` MiB
- Elapsed: `25.72` seconds
- Log: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq512_oaken.log`
- VRAM CSV: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq512_oaken_vram.csv`

### Original FP16 sequence length 1024

```sh
docker exec -w /workspace oaken-ae-container /opt/conda/envs/oaken/bin/python eval_perplexity.py -m opt -s 1.3b -t wikitext --max-length 1024 --stride 1024 --gpu-start-idx 0 --gpu-count 1 --quant-method none
```

- Status: `OK`
- PPL: `16.7812`
- Peak VRAM: `3733` MiB
- Elapsed: `16.73` seconds
- Log: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq1024_original.log`
- VRAM CSV: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq1024_original_vram.csv`

### Oaken sequence length 1024

```sh
docker exec -w /workspace oaken-ae-container /opt/conda/envs/oaken/bin/python eval_perplexity.py -m opt -s 1.3b -t wikitext --max-length 1024 --stride 1024 --gpu-start-idx 0 --gpu-count 1 -q quantizer/oaken/opt-1.3b.json --quant-method oaken
```

- Status: `OK`
- PPL: `17.5938`
- Peak VRAM: `3749` MiB
- Elapsed: `22.01` seconds
- Log: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq1024_oaken.log`
- VRAM CSV: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq1024_oaken_vram.csv`

### Original FP16 sequence length 2048

```sh
docker exec -w /workspace oaken-ae-container /opt/conda/envs/oaken/bin/python eval_perplexity.py -m opt -s 1.3b -t wikitext --max-length 2048 --stride 2048 --gpu-start-idx 0 --gpu-count 1 --quant-method none
```

- Status: `OK`
- PPL: `14.6406`
- Peak VRAM: `4673` MiB
- Elapsed: `16.70` seconds
- Log: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq2048_original.log`
- VRAM CSV: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq2048_original_vram.csv`

### Oaken sequence length 2048

```sh
docker exec -w /workspace oaken-ae-container /opt/conda/envs/oaken/bin/python eval_perplexity.py -m opt -s 1.3b -t wikitext --max-length 2048 --stride 2048 --gpu-start-idx 0 --gpu-count 1 -q quantizer/oaken/opt-1.3b.json --quant-method oaken
```

- Status: `OK`
- PPL: `15.4297`
- Peak VRAM: `4689` MiB
- Elapsed: `21.23` seconds
- Log: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq2048_oaken.log`
- VRAM CSV: `analysis/outputs/kv_cache_pressure/rtx5080/raw_logs/1_3b_seq2048_oaken_vram.csv`
