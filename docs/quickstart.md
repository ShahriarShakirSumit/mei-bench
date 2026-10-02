# Quickstart

```bash
pip install -r requirements.txt && pip install -e .
pytest -q tests

# reproduce the paper's tables and figures from the shipped predictions (CPU)
python reproduce/unpack_results.py && python reproduce/copy_original_run.py
python reproduce/molmo_extract.py && python reproduce/molmo_stratified_tables.py
python reproduce/make_main_tables.py && python reproduce/robustness_checks.py

# evaluate a model (GPU; COCO val2017 in data/coco/val2017)
python -m mei_benchmark.scripts.render_variants --out-dir data/mei_bench
python -m mei_benchmark.scripts.download_models qwen2-vl-7b
HF_HUB_OFFLINE=1 python -m mei_benchmark.scripts.run_evaluation --model qwen2-vl-7b \
    --config mei_benchmark/configs/base.yaml --model-config mei_benchmark/configs/models_open.yaml \
    --data-dir data/mei_bench --output-dir outputs/mei_bench
```

The full list of steps is in the top-level `README.md`.
