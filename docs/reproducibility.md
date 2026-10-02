# Reproducibility Notes

## What is shipped
- `results/predictions/rerun/`: every answer of the ten open-source models in the version-pinned re-run
  (615 items x original + 5 variants).
- `results/predictions/robustness/<condition>/`: the eight robustness conditions in the paper.
- `results/*.json`, `results/tables/`, `results/figures/`: everything the paper reports, regenerated from the above
  by `reproduce/` (README, step 1). On our machine the regenerated tables and JSON files are byte-identical to the
  shipped ones and the figure PDFs differ only in their creation date.
- `results/original_run/`: three tables and six figures kept as they appear in the paper, computed from the
  original evaluation run, whose raw predictions are not included.

## Environment of the paper
Python 3.11.16, PyTorch 2.5.1 (CUDA 12.1), Transformers 4.46.3, one NVIDIA V100 32 GB; exact versions in
`requirements.txt`. Model weights are pinned to the Hugging Face commits in `mei_benchmark/configs/models_open.yaml`
(`revision`); `download_models.py` fetches those commits, and evaluating with `HF_HUB_OFFLINE=1` makes sure they
are the ones loaded. Decoding is greedy (temperature 0).

## Determinism
Blur and gray-out variants are deterministic given the stored boxes. Content swap uses a random block shuffle; see
`docs/intervention_generation.md`. The re-run reproduces the original correct set and evidence-flip count to within
3 of 615 items for every model (exactly for six of the nine models evaluated in both runs).
