from __future__ import annotations
#!/usr/bin/env python3
"""
Recompute MEIEvaluator metrics for every prediction file in outputs/predictions
against the (newly filtered) items.jsonl. Writes per-model JSON to
outputs/metrics/<model>_metrics.json and a combined summary CSV.
"""
from mei_benchmark.paths import DATA_DIR, OUTPUT_DIR
import json
from pathlib import Path

from mei_benchmark.data.loader import MEIDatasetLoader, PredictionLoader
from mei_benchmark.evaluation.metrics import MEIEvaluator, StratifiedEvaluator

OUT = OUTPUT_DIR
DATA = DATA_DIR


def main():
    loader = MEIDatasetLoader(DATA)
    items = loader.load_items()
    print(f"items: {len(items)}")

    pred_loader = PredictionLoader(str(OUT / "predictions"))
    models = pred_loader.available_models()
    print(f"models: {models}")

    evaluator = MEIEvaluator()
    strat = StratifiedEvaluator()
    rows = []
    strat_all = {}

    for m in models:
        pred_sets = pred_loader.load_predictions(m)
        metrics = evaluator.evaluate(items, pred_sets)
        out_path = OUT / "metrics" / f"{m}_metrics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        rows.append(metrics.to_dict())

        # Stratified
        s = strat.evaluate_stratified(items, pred_sets, m)
        # Capture per-task GG, CS, SI, n
        per_task = {}
        for task, tm in s.by_task.items():
            per_task[str(task)] = {
                "n": tm.n_items,
                "acc": tm.original_accuracy,
                "CS_blur": tm.causal_sensitivity_blur,
                "CS_gray": tm.causal_sensitivity_gray,
                "SI_blur": tm.spurious_invariance_blur,
                "SI_gray": tm.spurious_invariance_gray,
                "GG": tm.grounding_gap,
                "GS": tm.grounding_specificity,
            }
        strat_all[m] = per_task
        print(f"  {m}: n={metrics.n_items} acc={metrics.original_accuracy:.3f} "
              f"CS={metrics.causal_sensitivity_blur:.3f} SI={metrics.spurious_invariance_blur:.3f} "
              f"GG={metrics.grounding_gap:.3f} GS={metrics.grounding_specificity:.3f}")

    with open(OUT / "metrics" / "_summary.json", "w") as f:
        json.dump({"models": rows, "stratified": strat_all}, f, indent=2)
    print("wrote", OUT / "metrics" / "_summary.json")


if __name__ == "__main__":
    main()
