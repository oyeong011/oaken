# Oaken Consumer GPU Accuracy Artifact Summary

## Experiment Scope

This is not a full Oaken hardware throughput reproduction. It is an Oaken official accuracy artifact reproduction and VRAM-boundary characterization on consumer RTX GPUs.

The runs cover Wikitext perplexity for original FP16 OPT evaluation, Oaken offline threshold profiling, and Oaken Wikitext evaluation where the available consumer GPU memory allowed it. PIQA, Winogrande, Hellaswag, and Figure 11 throughput reproduction were not run in this result set.

## Hardware

| GPU | VRAM | Driver | CUDA reported by driver | Notes |
| --- | ---: | --- | --- | --- |
| NVIDIA GeForce RTX 5060 | 8151 MiB | 590.48.01 | 13.1 | 8GB-class consumer GPU; baseline idle VRAM for OPT-1.3B run was 33 MiB. |
| NVIDIA GeForce RTX 5080 | 16GB-class | See per-run `hardware.txt` | See per-run `hardware.txt` | 16GB-class consumer GPU; baseline idle VRAM after larger runs was 38 MiB. |

## Reproduction Status

| GPU | Model | Status | Evidence |
| --- | --- | --- | --- |
| RTX 5060 | OPT-125M | OK | Known small-model smoke: Oaken PPL about 28, profiling 23.58s, peak about 1.4GB. Detailed artifact directory is not present in the current tree. |
| RTX 5060 | OPT-350M | OK | Original FP16 eval, Oaken profiling, and Oaken eval completed after compatibility fixes. |
| RTX 5060 | OPT-1.3B | OK | Original FP16 eval, Oaken profiling, and Oaken eval completed on 8GB VRAM with both default and expandable allocator settings. |
| RTX 5080 | OPT-125M | OK | Oaken profiling and eval retest completed. |
| RTX 5080 | OPT-350M | OK | Original FP16 eval, Oaken profiling, and Oaken eval completed after compatibility fixes. |
| RTX 5080 | OPT-1.3B | OK | Original FP16 eval, Oaken profiling, and Oaken eval completed. |
| RTX 5080 | OPT-2.7B | OK | Original FP16 eval, Oaken profiling, and Oaken eval completed. |
| RTX 5080 | OPT-6.7B | Boundary | Original FP16 eval and Oaken profiling completed with allocator tuning; Oaken eval failed with CUDA OOM. |

## Main Results Table

Peak VRAM is reported as decimal GB from the per-run MiB samples, matching the existing per-model summaries.

| GPU | Model | Original PPL | Oaken PPL | Delta | Profiling Time | Eval Time | Peak VRAM | Idle VRAM | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RTX 5060 | OPT-125M | N/A | ~28.0 | N/A | 23.58s | N/A | ~1.40GB | N/A | OK |
| RTX 5060 | OPT-350M | 22.0156 | 22.1875 | +0.1719 | 39.09s | 19.332s | 1.94GB | N/A | OK |
| RTX 5060 | OPT-1.3B | 14.6406 | 15.3984 | +0.7578 | 81.83s script / 85.07s wrapper | 42.57s | 4.49GB default / 4.35GB expandable eval | 33 MiB | OK |
| RTX 5080 | OPT-125M | 27.6719 | 28.0 | +0.3281 | 10.69s retest / 16.69s initial | 10s | 1.58GB | N/A | OK |
| RTX 5080 | OPT-350M | 22.0156 | 22.1406 | +0.1250 | 16.54s | 13s | 2.11GB | N/A | OK |
| RTX 5080 | OPT-1.3B | 14.6406 | 15.3984 | +0.7578 | 34.80s | ~20s | 4.67GB | 38 MiB | OK |
| RTX 5080 | OPT-2.7B | 12.4688 | 12.5703 | +0.1015 | 60.50s | ~30s | 7.83GB | 38 MiB | OK |
| RTX 5080 | OPT-6.7B | 10.8594 | OOM | N/A | 108.86s | 12s before OOM | 15.83GB | 38 MiB | Boundary |

## Accuracy Impact

For completed paired original/Oaken runs, Oaken Wikitext perplexity stayed close to the original FP16 baseline:

| Model / GPU | PPL Delta |
| --- | ---: |
| OPT-350M / RTX 5060 | +0.1719 |
| OPT-1.3B / RTX 5060 | +0.7578 |
| OPT-125M / RTX 5080 | +0.3281 |
| OPT-350M / RTX 5080 | +0.1250 |
| OPT-1.3B / RTX 5080 | +0.7578 |
| OPT-2.7B / RTX 5080 | +0.1015 |

The largest observed delta in this set is OPT-1.3B at +0.7578 on both GPUs. The 6.7B RTX 5080 case has no Oaken PPL because evaluation reached CUDA OOM after profiling succeeded.

## VRAM Boundary

The RTX 5060 run shows that OPT-1.3B remains viable for the Oaken accuracy path on an 8GB-class consumer GPU. The default allocator peaked at 4.49GB during profiling and 4.42GB during Oaken eval; the expandable allocator repeat preserved the same PPL while reducing the profiling and eval peaks to 4.13GB and 4.35GB.

The RTX 5080 OPT-6.7B run is the meaningful 16GB boundary. Original FP16 Wikitext evaluation and Oaken offline profiling completed only with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; Oaken Wikitext evaluation still failed with CUDA OOM after reaching a 15.83GB peak. Without allocator tuning, the original FP16 eval also failed near the same boundary.

## Compatibility Fixes

- `src/oaken/quantize.py` now guards zero or non-finite quantization ranges before division, preventing `nan` propagation in OPT-350M Oaken evaluation.
- `src/util.py` includes OPT projection-layer device-map entries for configurations where `word_embed_proj_dim != hidden_size`.
- `src/model.py` passes the loaded OPT config into device-map construction so OPT-350M gets projection layers while OPT-125M keeps the original mapping.
- RTX 5080 OPT-6.7B required `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for original FP16 eval and Oaken profiling to complete.

## Limitations

- This summary is based on the current local result artifacts under `results/rtx5060/` and `results/rtx5080/`, plus the known RTX 5060 OPT-125M smoke numbers supplied in the experiment notes.
- The RTX 5060 OPT-125M detailed artifact directory is not present in the current result tree.
- Hardware throughput, kernel-level timing, Figure 11 throughput curves, PIQA, Winogrande, and Hellaswag were not reproduced here.
- The threshold artifacts are Wikitext-derived; cross-dataset threshold stability remains future work.
- Local matplotlib currently fails to import against the installed NumPy 2.2.6, so plot scripts are present but PNG generation is environment-dependent.

## Next Experiments

1. Run the threshold observation analysis over the successful Wikitext quantizer artifacts and inspect per-layer key/value threshold widths and absolute maxima.
2. Repeat Oaken profiling for Wikitext, PIQA, Winogrande, and Hellaswag with fixed model, GPU, sparsity fractions, and software stack.
3. Compare threshold stability across datasets by layer, tensor type, and quantization group.
4. Revisit the RTX 5080 OPT-6.7B boundary with lower batch pressure or evaluation changes only if the goal is specifically to characterize OOM sensitivity rather than claim full Oaken success.
