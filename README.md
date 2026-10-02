# MEI-Bench

**Controlled Evidence Interventions for Diagnosing Visual Grounding in Vision-Language Models**

Shahriar Shakir Sumit, Murat Tahtali and Seyedali Mirjalili. ACML 2026 (PMLR).

This repository has everything behind our ACML 2026 paper: the benchmark annotations, the evaluation toolkit, every
model output we collected, and the scripts that rebuild each table and figure in the paper.

The benchmark is also on Hugging Face, where you can browse the questions and load them with one line:
[huggingface.co/datasets/Shahriar10/mei-bench](https://huggingface.co/datasets/Shahriar10/mei-bench).

## Why we built this

A vision-language model can give the right answer to a question about an image without ever looking at the part of
the image that holds the answer. It might be leaning on language priors, or on patterns it picked up in training.
Accuracy alone can't tell these cases apart.

So we test it the way you would test a treatment. For each question we know where the evidence is. We blur (or gray
out, or scramble) that region and check whether the answer changes. Then we apply exactly the same edit to a region of
the same size somewhere else in the image, a "sham" region that holds none of the evidence. If a model really uses the
evidence, its answer should change when the evidence is degraded and stay put when the sham is. Models that change
their answer in both cases are fragile, not grounded, and the sham is what lets us see the difference.

## What we measure

Everything is computed on the questions a model answers correctly on the untouched image.

| Metric | What it means |
|---|---|
| Causal Sensitivity (CS) | how often the answer changes when the evidence region is degraded |
| Sham Sensitivity (SI) | how often the answer changes when the sham region is degraded |
| Grounding Gap (GG) | CS minus SI: the extra change caused by touching the evidence |
| Sham Robustness (SR) | 1 minus SI |
| Grounding Specificity (GS) | GG divided by CS: the share of the sensitivity that is specific to the evidence |

The main results use Gaussian blur with a kernel of 51 pixels.

## What we found

Across ten open-source models on 615 questions from COCO val2017:

* Every model has a positive grounding gap (GG between 0.14 and 0.27, GS between 0.57 and 0.82).
* Between 18 and 43% of the sensitivity we measure is general fragility rather than grounding. Without the sham, you
  would overestimate how much a model relies on the evidence by 5 to 12 points.
* The gap stays positive when we redraw the sham at random, match it to the evidence on saliency, texture or
  position, or use the object's exact mask instead of its box.
* Accuracy tells you very little about grounding (r = 0.13).

## What's in the repository

```
mei_benchmark/          the toolkit: data loaders, image edits, model wrappers, metrics and scripts
  configs/              base.yaml and models_open.yaml (ten models, pinned to the exact weights we used)
data/annotations/       mei_bench.jsonl, the 615 benchmark items
data/metadata/          schemas, a dataset summary, COCO source info and the filtering report (959 items down to 615)
data/sample/            25 items to look at before you download anything
data/rerun/             the items of our version-pinned re-run
data/robustness_conditions/     the eight extra sham and mask conditions used in the paper's robustness checks
reproduce/           the scripts that produce every number, table and figure in the paper
results/                all model predictions, the result files, and the tables and figures they produce
tests/                  unit tests for the image edits and the metrics
tools/                  two small checkers for the annotation file
docs/                   quickstart, data provenance, how the edits are made, licence notes
DATASET_CARD.md, DATASHEET.md, croissant.json, MAINTENANCE.md
```

We don't redistribute the COCO images themselves. You'll download COCO val2017 from the official site (see below).

## Getting started

We ran everything with Python 3.11, PyTorch 2.5.1 (CUDA 12.1) and Transformers 4.46.3 on a single NVIDIA V100 with
32 GB of memory.

```bash
git clone https://github.com/ShahriarShakirSumit/mei-bench.git
cd mei-bench
conda create -n mei-benchmark python=3.11 -y && conda activate mei-benchmark
pip install -r requirements.txt          # the exact versions we used
pip install -e .
pytest -q tests                          # 39 tests
python tools/validate_dataset.py --annotations data/annotations/mei_bench.jsonl
python tools/validate_regions.py --annotations data/annotations/mei_bench.jsonl
```

If you want to evaluate Claude or Gemini as well, also install `requirements-api.txt`.

## Reproduce the paper from our saved outputs (CPU, a few minutes)

You don't need a GPU, model weights or COCO images for this part. Every answer every model gave is saved in
`results/predictions/`, and these scripts rebuild the paper's numbers, tables and figures from them:

```bash
python reproduce/unpack_results.py          # unpack the saved predictions into outputs/
python reproduce/copy_original_run.py       # tables and figures from our original run (see below)
python reproduce/molmo_extract.py           # read Molmo-7B-D's answers with its own answer parser
python reproduce/molmo_stratified_tables.py     # Molmo-7B-D rows of the per-task tables
python reproduce/make_main_tables.py               # main results and metric correlations
python reproduce/make_figures.py              # the figures in the main paper
python reproduce/robustness_checks.py           # random, saliency, position and texture shams, and mask evidence
python reproduce/make_robustness_figure.py       # the robustness figure
python reproduce/intersection_analysis.py \
       --pred-dir outputs/rerun_molmo_fixed/predictions --out results/intersection_10open.json
python reproduce/clean_label_subset.py      # results without the noisier labels
python reproduce/make_supplementary_tables.py          # the supplementary tables
```

The results land in `results/tables/`, `results/figures/` and `results/*.json`, on top of the copies we ship. Run
`git diff --stat results/` afterwards to see whether anything changed. When we did this on a fresh copy, every table
and result file came out byte for byte the same, and the figures differed only in the date stamp inside the PDF.

A few tables and figures (dataset statistics, area effects, the comparison of the three edits, the method diagram, the
example images and four plots) come from our original evaluation run, whose raw
outputs are no longer available, so we ship them as they are. `results/original_run/README.md` lists them.

## Evaluate a model on MEI-Bench (GPU)

**1. Get COCO val2017.** Put the images in `data/coco/val2017/` and the annotation file in
`data/coco/annotations/instances_val2017.json` (see `docs/coco_download_instructions.md`). If you already have a copy
somewhere else, point to it with `--images-dir`.

**2. Make the edited images.** This uses the evidence and sham boxes stored in the annotations:

```bash
python -m mei_benchmark.scripts.render_variants \
       --annotations data/annotations/mei_bench.jsonl --images-dir data/coco/val2017 --out-dir data/mei_bench
```

You get five edited versions of every image: evidence blur, evidence gray-out, evidence swap, sham blur and sham
gray-out. The blur and gray-out images always come out the same. The swap (colour inversion plus a random shuffle of
blocks) is seeded here, but the images in the paper were made without a fixed seed, so swap results can differ a
little from ours.

**3. Download the models and run them.** The download script fetches the exact weights we used. Running offline
afterwards makes sure those are the weights that get loaded:

```bash
python -m mei_benchmark.scripts.download_models                 # all ten, or name the ones you want, e.g. qwen2-vl-7b
export HF_HUB_OFFLINE=1
python -m mei_benchmark.scripts.run_evaluation --model qwen2-vl-7b \
       --config mei_benchmark/configs/base.yaml --model-config mei_benchmark/configs/models_open.yaml \
       --data-dir data/mei_bench --output-dir outputs/mei_bench
```

The ten models are `llava-1.5-7b`, `llava-1.5-13b`, `llava-v1.6-7b`, `qwen2-vl-7b`, `internvl2-8b`,
`internvl2.5-8b`, `phi-3.5-vision`, `llama-3.2-11b-vision`, `deepseek-vl-7b` and `molmo-7b`. Answers are saved to
`<output-dir>/predictions/<Model>_predictions.jsonl`, and if a run stops halfway it picks up where it left off.

**Trying your own model?** Write a small wrapper class based on `mei_benchmark.models.base.BaseVLM` (it needs a
`load_model` and a `predict` method), add it to `configs/models_open.yaml` with
`wrapper_class: "your_module:YourWrapper"`, and run it the same way.

**Claude and Gemini:** run `python -m mei_benchmark.scripts.run_api_eval --provider claude` with your
`ANTHROPIC_API_KEY` set, or `--provider gemini` with `GOOGLE_API_KEY`.

## Re-run the robustness checks yourself (GPU)

```bash
# edited images for the re-run items (these carry the re-run's own sham boxes)
python -m mei_benchmark.scripts.render_variants \
       --annotations data/rerun/items.jsonl --images-dir data/coco/val2017 --out-dir data/rerun
# build the eight extra conditions (seed 20260926); the mask condition needs the COCO instance annotations
python reproduce/build_robustness_conditions.py
# reuse each model's answers on the original images, so only the new edited images are sent to the model
python reproduce/unpack_results.py --only rerun && python reproduce/seed_originals.py
for COND in sham_r1 sham_r2 sham_r3 sham_r4 sham_r5 sham_salience sham_position mask; do
  python -m mei_benchmark.scripts.run_evaluation --mode inference-only --model qwen2-vl-7b \
         --config mei_benchmark/configs/base.yaml --model-config mei_benchmark/configs/models_open.yaml \
         --data-dir data/robustness_conditions/$COND --output-dir outputs/robustness/$COND --resume
done
```

Repeat the loop for each model, then run the reproduction scripts above from `molmo_extract.py` onward. One thing to
watch: use `--only rerun` when you unpack here. Without it, our saved results for these conditions would be copied
into `outputs/robustness/` and the evaluation would think there is nothing left to do.

## About the data

`data/annotations/mei_bench.jsonl` has 615 questions built from COCO val2017, spread over five kinds of question:
spatial relations, counting, object identification, attribute checks and reading text in the image. Each item has
the question, the accepted answers, the evidence box and the sham box. `DATASET_CARD.md` and `DATASHEET.md` explain
how it was made, what it's good for and where its limits are, and `croissant.json` has the same information in a
machine-readable form.

## Licence

The code is under the MIT licence and the annotations under CC BY 4.0. The COCO images stay under the COCO terms of
use. Details are in `LICENSE`.

## Citing our work

If MEI-Bench or this code is useful in your research, we'd really appreciate it if you cite our paper:

```bibtex
@inproceedings{sumit2026mei,
  title     = {Controlled Evidence Interventions for Diagnosing Visual Grounding in Vision-Language Models},
  author    = {Sumit, Shahriar Shakir and Tahtali, Murat and Mirjalili, Seyedali},
  booktitle = {Proceedings of the 18th Asian Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year      = {2026}
}
```

## Questions and feedback

We're happy to hear from you. Open an issue on GitHub, or write to Shahriar Shakir Sumit at m.sumit@unsw.edu.au.
Pull requests are welcome too.

## Acknowledgement

Experiments were performed on the computational facility of the National Computational Infrastructure (NCI) through
the UNSW HPC Scheme (DOI: 10.26190/PMN5-7J50).
