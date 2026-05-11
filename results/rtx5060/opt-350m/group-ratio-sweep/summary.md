# RTX 5060 OPT-350M Oaken Group-Ratio Sweep

## Scope

This sweep evaluates Wikitext perplexity, profiling behavior, sparsity, VRAM use, and stability for four Oaken activation group ratios on the RTX 5060 8GB path.

## Environment

- GPU: NVIDIA GeForce RTX 5060
- Model: OPT-350M from `/data/models/opt-350m`
- Container: `oaken-ae-container` / `oaken-ae-img`
- Task: Wikitext-2 raw test perplexity
- Original FP16 baseline PPL: `22.0156`
- Baseline run elapsed: `14.743s`

## Results

| Ratio | Status | Original PPL | Oaken PPL | Delta | Profile Time | Eval Time | Peak VRAM | Key Sparsity | Value Sparsity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0.02/0.94/0.04 | OK | 22.0156 | 22.2344 | 0.2188 | 40.883s | 19.341s | 1937 MiB | [0.020413, 0.939707, 0.039880] | [0.020018, 0.939999, 0.039984] |
| 0.04/0.90/0.06 | OK | 22.0156 | 22.1875 | 0.1719 | 40.881s | 19.378s | 1937 MiB | [0.040545, 0.899612, 0.059842] | [0.040029, 0.899989, 0.059982] |
| 0.08/0.84/0.08 | OK | 22.0156 | 22.1406 | 0.1250 | 41.037s | 19.518s | 1937 MiB | [0.080526, 0.839592, 0.079882] | [0.080140, 0.839893, 0.079967] |
| 0.10/0.80/0.10 | OK | 22.0156 | 22.1094 | 0.0938 | 40.978s | 19.513s | 1937 MiB | [0.100684, 0.799507, 0.099809] | [0.099984, 0.800003, 0.100013] |

## Interpretation

- The default `0.04/0.90/0.06` grouping completed successfully with Oaken PPL `22.1875` and delta `0.1719`.
- The smallest observed PPL delta was `0.0938` at ratio `0.10/0.80/0.10`.
- Across the tested ratios, Oaken PPL improved monotonically as the outer/inner groups grew and the middle group shrank: `22.2344 -> 22.1875 -> 22.1406 -> 22.1094`.
- All completed ratios stayed well within the 8GB RTX 5060 budget; the largest sampled peak was `1937 MiB`.
- Profiling and eval timing were effectively flat across ratios: profile `40.881-41.037s`, eval `19.341-19.518s`.
- No NaN, OOM, traceback, or runtime error was detected in the sweep logs.
- Reported sparsity tracks the configured grouping closely for both key and value projections, confirming the profiling/eval hooks are applying the intended ratio split.

## Artifacts

- `summary.csv`: machine-readable sweep table.
- `summary.md`: this interpretation.
- `hardware.txt`: hardware, Docker, Python, PyTorch, and model-file listing.
- `baseline/`: original FP16 Wikitext baseline log, metadata, and VRAM samples.
- `ratio-*/`: per-ratio quantizer, profiling log/metadata/VRAM samples, and Oaken eval log/metadata/VRAM samples.

Model weights, HuggingFace cache files, `.bin`, and `.safetensors` artifacts are not stored in this results directory.
