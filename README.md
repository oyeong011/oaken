# Oaken: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization

Oaken is an accleration solution that achieves high accuracy and high performance simultaneously through co-designing algorithm and hardware, leveraging online-offline hybrid KV cache quantization algorithm and dedicated quantization/dequantization hardware modules.

This repository provides source code for evaluating the inference accuracy of Oaken and other baselines.
Running the evaluation code requires Python 3.10 or later and CUDA 12.1 or later.

## RTX 5060 8GB Boundary Result

### 1. Experiment Goal

This project empirically characterizes KV-cache memory pressure across GPU memory capacities and evaluates whether alternative cache policies can expand the feasible long-context inference region.

The RTX 5060 8GB experiment isolates the practical memory-capacity boundary of KV-cache based inference. The sweep uses chunked synthetic token inputs to grow the cache while avoiding a full-context prefill attention OOM that would hide the KV-cache boundary.

### 2. Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 |
| VRAM | 8151 MiB |
| Driver | 590.48.01 |
| PyTorch | 2.12.0+cu130 |
| Transformers | 5.8.1 |
| Model | OPT-1.3B FP16 |
| Measurement | Chunked cache-growth inference |

### 3. Dynamic KV-cache OOM Boundary

Dynamic cache showed the highest throughput among successful runs, but failed at larger batch-size and sequence-length combinations.

| Batch size | Largest successful seq_len | First observed OOM seq_len |
| ---: | ---: | ---: |
| 1 | 8192 | N/A |
| 2 | 8192 | N/A |
| 4 | 6144 | 8192 |
| 8 | 2048 | 4096 |

The dynamic OOM cases were:

- `batch=4, seq_len=8192`
- `batch=8, seq_len=4096`
- `batch=8, seq_len=6144`
- `batch=8, seq_len=8192`

### 4. Quantized Rescue Cases

The dynamic OOM cases were re-tested with HQQ-backed quantized cache. Quantized cache expanded the feasible execution region for two cases:

| Batch size | Seq len | Dynamic status | Quantized status | Quantized throughput |
| ---: | ---: | --- | --- | ---: |
| 4 | 8192 | OOM | OK | 157.93 tok/s |
| 8 | 4096 | OOM | OK | 317.22 tok/s |
| 8 | 6144 | OOM | OOM | N/A |
| 8 | 8192 | OOM | OOM | N/A |

`no_cache` was also run as a memory lower-bound/ablation baseline. It should not be interpreted as a practical serving policy: it avoids KV-cache storage, but generation would need to recompute prior context and throughput would collapse in a real autoregressive serving path.

### 5. Offloaded Cache Limitation

Offloaded cache could not be completed on this host. The RTX 5060 machine has 15 GiB system RAM and no swap, and the offloaded runs were killed by host memory pressure before producing valid rows. This result should therefore be interpreted as quantized-cache rescue evidence, not as an offloaded-cache comparison.

### 6. Interpretation

The key observation is that KV-cache optimization is not necessarily a universal speedup technique. Instead, it acts as a memory-capacity extension mechanism for long-context or larger-batch inference, trading throughput for successful execution under limited VRAM.

The RTX 5060 result shows that the boundary arrives quickly on an 8GB GPU. The RTX 5080 results show the same pressure scaling up: OPT-1.3B and OPT-2.7B complete on the larger GPU, while OPT-6.7B reaches the 16GB-class boundary, where allocator tuning is required for the original evaluation and the Oaken evaluation still OOMs.

Korean summary:

이 프로젝트는 GPU VRAM 용량이 다른 환경에서 KV-cache가 long-context inference의 실행 가능 영역을 어떻게 제한하는지 실험적으로 분석하고, quantized/offloaded/no-cache 정책이 그 한계를 얼마나 완화할 수 있는지 평가하는 실험 프레임워크입니다.

### 7. Generated Artifacts

- `results/rtx5060_opt13b_dynamic_boundary.csv`
- `results/rtx5060_opt13b_rescue_cases.csv`
- `results/rtx5060_opt13b_combined.csv`
- `results/plots_rtx5060_combined/peak_memory_vs_seq_len.png`
- `results/plots_rtx5060_combined/throughput_vs_peak_memory.png`
- `results/plots_rtx5060_combined/status_boundary_matrix.csv`
- `results/plots_rtx5060_combined/oom_cases.csv`
- `results/plots_rtx5060_combined/dynamic_oom_rescue_cases.csv`

## Qwen2.5-1.5B Position-valid Long-context Result on RTX 5060 8GB

To avoid over-interpreting the OPT-1.3B memory stress results beyond its position limit, I repeated the KV-cache boundary experiment with Qwen2.5-1.5B-Instruct. This model supports a 32,768-token context window and uses GQA with 12 query heads and 2 key/value heads.

The sanity run verified that the sweep records the expected model metadata:

- `position_valid=True`
- `max_position_embeddings=32768`
- `kv_formula_type=gqa_mqa`
- `kv_actual_over_theory=1.0` for dynamic cache

The dynamic KV-cache sweep found GPU OOM at:

| batch_size | seq_len | dynamic |
| ---: | ---: | --- |
| 8 | 12288 | OOM |
| 8 | 16384 | OOM |

I then re-ran the failed configurations with quantized KV-cache. Quantized cache successfully rescued both dynamic OOM cases:

| batch_size | seq_len | dynamic | quantized | quantized kv_actual_over_theory |
| ---: | ---: | --- | --- | ---: |
| 8 | 12288 | OOM | OK | 0.288737 |
| 8 | 16384 | OOM | OK | 0.286865 |

Quantized cache reduced the measured KV-cache tensor footprint to about 28.7% of the fp16 theoretical KV-cache size in the rescued cases. This should not be interpreted as the same reduction ratio for total CUDA peak memory, which also includes model weights, temporary buffers, activations, and allocator-reserved memory.

This result shows that the dynamic OOM / quantized rescue pattern is not limited to synthetic OPT memory stress. It also appears in a position-valid long-context model. In this setting, KV-cache quantization is best interpreted as a memory-capacity optimization: it trades cache representation overhead and potential latency cost for a larger feasible long-context/batch execution region.

Korean summary:

이 프로젝트는 LLM 추론에서 KV-cache가 GPU 메모리 용량을 어떻게 압박하는지 실험적으로 분석했다. RTX 5060 8GB에서 Qwen2.5-1.5B-Instruct를 대상으로 batch=8, seq_len=12288/16384 조건에서 dynamic KV-cache는 OOM으로 실패했지만, quantized KV-cache는 두 조건을 모두 실행 가능하게 만들었다.

Generated artifacts:

- `results/rtx5060_qwen25_15b_sanity.csv`
- `results/rtx5060_qwen25_15b_dynamic_boundary.csv`
- `results/rtx5060_qwen25_15b_rescue_cases.csv`
- `results/rtx5060_qwen25_15b_combined.csv`
- `results/plots_rtx5060_qwen25_15b_combined/peak_memory_vs_seq_len.png`
- `results/plots_rtx5060_qwen25_15b_combined/throughput_vs_peak_memory.png`
- `results/plots_rtx5060_qwen25_15b_combined/status_boundary_matrix.csv`
- `results/plots_rtx5060_qwen25_15b_combined/oom_cases.csv`
- `results/plots_rtx5060_qwen25_15b_combined/dynamic_oom_rescue_cases.csv`

## 1. Preparing environments

### 1.1. Build Docker image

We provide a Dockerfile to build a container with Python 3.10, CUDA 12.1, and Miniconda pre-installed.
Run the following command to build the Docker image and create the corresponding container.

```sh
$ docker build --force-rm -t oaken-ae-img .
$ docker run                              \
        --name oaken-ae-container         \
        -it                               \
        --gpus '"device=[GPU LIST]"'      \
        oaken-ae-img 
```

### 1.2. Configurating Huggingface Model Path

1. Open `src/model.py`.
2. Set `MODEL_STORAGE_PREFIX`, and `MODEL_STORATE_POSTFIX` to the directory where huggingface models are stored.
3. Huggingface model directory names should follow the format `{model_name}-{model_size}`, such as llama2-7b.

### 1.3. Download Huggingface Model

1. Put your Huggingface access token to the variable `HF_TOKEN` in `model_downloader.py`. (Some of the models might require indiviual access grant) 
2. Select models to download by setting the variables `DOWNLOAD_*` in `model_downloader.py`.
3. Run the following command to download LLMs from Huggingface.

```sh
$ pip install huggingface_hub
$ python3 model_downloader.py
``` 

## 2. Oaken, KVQuant, QServe, Tender

### 2.1.1. Install Dependencies
Oaken, KVQuant, QServe, and Tender share the same installation commands.

```sh
$ git submodule update --init --recursive

$ conda create -n oaken python=3.10
$ conda activate oaken
$ pip install torch protobuf sentencepiece

$ pushd transformers
$ pip install -e .
$ popd

$ pushd lm-evaluation-harness
$ pip install -e .
$ popd

$ pushd kvquant/quant
$ pip install -e .
$ popd
$ pip install flash-attn --no-build-isolation
```

### 2.2. All-in-one Scripts
You can run the entire accuracy evaluation to get the results for Table 2 with the following command.
Please configure for list of models and number of GPUs that will be used for evaluation.
**Note that running the entire accuracy evaluation with this script takes very long time.**

```sh
$ python3 scripts/accuracy_oaken.py
$ python3 scripts/accuracy_kvquant.py
$ python3 scripts/accuracy_qserve.py
$ python3 scripts/accuracy_tender.py
```

Use the following command to get the results for Figure 12(a).

```sh
$ python3 explore_oaken.py
```

The following sections describe the instructions for running individual models and benchmarks.

### 2.3.1. Preparing Oaken
You can run offline profiling for Oaken with the following command.

```sh
$ python3 oaken_preprocess_activation.py \
    -m [MODEL NAME]     \
    -s [MODEL SIZE]     \
    -t wikitext         \
    -f 0.04 0.9 0.06    \
    -o quantizer/oaken/[OUTPUT QUANTIZER].json 
```

Example command:
```sh
$ python3 oaken_preprocess_activation.py \
    -m llama2           \
    -s 7b               \
    -t wikitext         \
    -f 0.04 0.9 0.06    \
    -o quantizer/oaken/llama2-7b.json
```

### 2.3.2 Preparing KVQuant

Run profiling process for KVQuant with the following command.
Set `[MODEL PATH]` to the directory where you downloaded huggingface model.

```sh
$ cd kvquant/quant
$ CUDA_VISIBLLE_DEVICES=0 python3 llama_simquant.py \
    [MODEL PATH]                \
    --abits 4                   \
    --nsamples 16               \
    --seqlen 2048               \
    --nuq                       \
    --quantize                  \
    --include_sparse            \
    --sparsity-threshold 0.99   \
    --quantizer-path quantizer/kvquant/[OUTPUT QUANTIZER].pickle
```

Please refer to detailed instruction at https://github.com/SqueezeAILab/KVQuant/tree/main/quant.

### 2.3.3. Preparing QServe
Run profiling process for QServe with the following command.

```sh
$ python3 qserve_preprocess_activation.py \
    -m [MODEL NAME] \
    -s [MODEL SIZE] \
    -t wikitext \
    -o quantizer/qserve/[OUTPUT QUANTIZER].json 
```

### 2.3.4. Preparing Tender

You should download `val.jsonl.zst` file with the following command.

```sh
wget https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst
```

You can run preprocessing for Tender with the following command.

```sh
$ python3 tender_preprocess_activation.py \
    -m [MODEL NAME] \
    -s [MODEL SIZE] \
    -d val.jsonl.zst \
    -o quantizer/tender/[OUTPUT QUANTIZER].pt
```

### 2.4.1 Accuracy Evaluation (Perplexity)
Use `--quant-method` option to select quantization method.
You can use `--gpu-start-idx` and `--gpu-count` option to choose which GPUs will be used for the evaluation.
Specify the path for the quantizer files generated by the above commands, if required by the quantization algorithm.

```sh
$ python3 eval_perplexity.py \
    -m [MODEL NAME] \
    -s [MODEL SIZE] \
    -t wikitext \
    -q quantizer/[QUANTIZER PATH] \
    --quant-method [oaken | qserve | kvquant | tender] \
    --gpu-start-idx 0 \
    --gpu-count 1
```

### 2.4.2 Accuracy Evaluation (Zero-shot Accuracy)
Use `--quant-method` option to select quantization method.
Use `-t` option to select evaluation dataset.
You can use `--gpu-start-idx` and `--gpu-count` option to choose which GPUs will be used for the evaluation.

```sh
$ python3 eval_workload.py \
    -m [MODEL NAME] \
    -s [MODEL SIZE] \
    -t [piqa | winogrande | hellaswag] \
    -q quantizer/[QUANTIZER PATH] \
    --quant-method [oaken | qserve | kvquant | tender] \
    --gpu-start-idx 0 \
    --gpu-count 1
```

## 3. KIVI

### 3.1. Install Dependencies
```sh
$ conda create -n kivi python=3.10
$ conda activate kivi

$ cd KIVI
$ pip install -e .
$ cd quant
$ pip install -e .

$ cd ../..
$ pip install datasets

$ pushd transformers
$ pip install -e .
$ popd

$ pushd lm-evaluation-harness
$ pip install -e .
$ popd
```

### 3.2. All-in-one Script
You can run the entire accuracy evaluation to get the results for Table 2 with the following command.
Please configure for list of models and number of GPUs that will be used for evaluation.
**Note that running the entire accuracy evaluation with this script takes very long time.**

```sh
$ python3 scripts/accuracy_kivi.py
```

### 3.3.1. Accuracy Evaluation (Perplexity)

```sh
$ python3 eval_perplexity.py \
    -m [MODEL NAME] \
    -s [MODEL SIZE] \
    -t wikitext \
    --quant-method kivi \
    --gpu-start-idx 0 \
    --gpu-count 1
```

### 3.3.2. Accuracy Evaluation (Zero-shot Accuracy)

```sh
$ python3 eval_workload.py \
    -m [MODEL NAME] \
    -s [MODEL SIZE] \
    -t [piqa | winogrande | hellaswag] \
    --quant-method kivi \
    --gpu-start-idx 0 \
    --gpu-count 1
```

## 4. Atom

### 4.1. Install Dependencies

```sh
conda create -n atom python=3.10
conda activate atom
cd Atom/model
pip install -r requirements.txt
```

### 4.2.1. Accuracy Evaluation (Perplexity)

```sh
$ cd Atom
$ ./scripts/run_atom_ppl.sh [MODEL NAME] [# of GPUs]
```

Example command: 
```sh
$ ./scripts/run_atom_ppl.sh llama2-7b 1
$ ./scripts/run_atom_ppl.sh llama2-70b 4
```

### 4.2.2. Accuracy Evaluation (Zero-shot Accuracy)

```sh
$ cd Atom
$ ./scripts/run_atom_zeroshot_acc.sh [MODEL NAME] [# of GPUs]
```

## 5. Troubleshooting

### 5.1. If you encounter: `nvcc was not found`

Configure the environment variables with the following commands.

```sh
export PATH="/usr/local/cuda-[CUDA_VERSION]/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-[CUDA_VERSION]/lib64:${LD_LIBRARY_PATH}"
```

### 5.2. If you encounter: `RuntimeError: quantile() input tensor is too large`

Use smaller sampling rate rather than 1.0 by giving option `--sample-rate` or modifying `SAMPLING_RATE` variable in the script.

### 5.3. Failing KVQuant profiling with the error: `Expected all tensors to be on the same device ...`

Modify `freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)` at `src/transformers/models/llama/modeling_llama.py` to
`freqs = (inv_freq_expanded.float().to(position_ids_expanded.device) @ position_ids_expanded.float()).transpose(1, 2)`.

### 5.4. If you encounter: `cannot import name 'EncoderDecoderCache' from 'transformers'`

Install the package with the following command.

```sh
$ pip install peft==0.10.0
```
