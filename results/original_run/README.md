# Tables and figures from our original run

These tables and figures come from our original evaluation run. Its raw model outputs are no longer available, so we
ship the tables and figures themselves. The predictions in `results/predictions/` come from a later, version-pinned
re-run, which reproduces the original correct sets and evidence flips to within 3 of 615 items (see
`results/tables/tab_reproduction.tex`).

| File | What it is |
|---|---|
| `tables/tab_dataset_stats.tex` | benchmark statistics (`mei_benchmark/scripts/build_paper_tables.py`) |
| `tables/tab_area_effects.tex` | effect of evidence size (`mei_benchmark/scripts/build_appendix_tables.py`) |
| `tables/tab_ablation_intervention.tex` | blur, gray-out and swap compared (`build_appendix_tables.py`) |
| `tables/tab_stratified.tex`, `tables/tab_full_stratified.tex` | per-task results; `reproduce/molmo_stratified_tables.py` replaces the Molmo-7B-D rows with values from the re-run |
| `figures/fig1_method_overview.pdf` | the method diagram |
| `figures/fig_qualitative_examples.pdf` | example items (COCO images) |
| `figures/fig_area_bar_chart.pdf`, `fig_area_effects.pdf`, `fig_dose_response.pdf`, `fig_intervention_comparison.pdf` | drawn by `mei_benchmark/scripts/regen_figures.py` |

`reproduce/copy_original_run.py` copies them into `results/tables/` and `results/figures/`.
