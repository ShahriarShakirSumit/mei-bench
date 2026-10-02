# Reproducibility Notes

## What is shipped
- `model_outputs/rerun/`: every answer of the ten open-source models in the version-pinned re-run
  (615 items x original + 5 variants).
- `model_outputs/robustness/<condition>/`: their answers in the eight robustness conditions of the paper.
- `reproduce/inputs/`: two small inputs the scripts need: the per-model metrics of our original evaluation run (whose
  raw answers are not included) and the comparison of the re-run with that original run.

The scripts in `reproduce/` compute everything the paper reports from these files and write it to `outputs/paper/`.
On our machine the resulting tables and numbers match the paper exactly.

## Environment of the paper
Python 3.11.16, PyTorch 2.5.1 (CUDA 12.1), Transformers 4.46.3, one NVIDIA V100 32 GB; exact versions in
`requirements.txt`. Model weights are pinned to the Hugging Face commits in `mei_benchmark/configs/models_open.yaml`
(`revision`); `download_models.py` fetches those commits, and evaluating with `HF_HUB_OFFLINE=1` makes sure they
are the ones loaded. Decoding is greedy (temperature 0).

## Determinism
Blur and gray-out variants are deterministic given the stored boxes. Content swap uses a random block shuffle; see
`docs/intervention_generation.md`. The re-run reproduces the original correct set and evidence-flip count to within
3 of 615 items for every model (exactly for six of the nine models evaluated in both runs).
