---
type: log
status: active
created: 2026-05-11
updated: 2026-05-11
tags: [oaken, handoff, dataset-stability]
sources: []
---

# Log

## [2026-05-11] synthesis | Oaken OPT-350M dataset stability and RTX 5060 OPT-2.7B handoff

- Added a Korean overview for the OPT-350M dataset-stability experiment.
- Added SVG fallback plots for abs_max by dataset/layer and relative difference from Wikitext.
- The summary artifacts currently indicate:
  - Wikitext baseline reused successfully.
  - Winogrande and Hellaswag profiled successfully.
  - PIQA failed because the installed datasets stack rejected the dataset-script loader.
  - Offline profiling looks plausible at a coarse level, but thresholds are not strictly dataset-invariant.
- The RTX 5060 OPT-2.7B consumer GPU result is already reflected in the summary CSV/MD and derived threshold artifacts.
