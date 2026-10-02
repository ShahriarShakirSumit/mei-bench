from __future__ import annotations
#!/usr/bin/env python3
"""
Post-processing script for MEI benchmark results.      

After new model evaluations complete, this script:
1. Reads all metrics from <MEI_OUTPUTS>/metrics/
2. Computes bootstrap CIs for new models
3. Prints a summary table suitable for the paper
4. Outputs LaTeX table rows

Usage:
    python scripts/postprocess_results.py
"""

from mei_benchmark.paths import DATA_DIR, OUTPUT_DIR

import json
import sys
from pathlib import Path

import numpy as np


def load_all_metrics(metrics_dir: str) -> dict:
    """Load all model metrics from JSONL files."""
    metrics = {}
    p = Path(metrics_dir)
    for f in sorted(p.glob("*_metrics.json")):
        with open(f) as fh:
            data = json.load(fh)
        model = data["model"]
        metrics[model] = data
    return metrics


def compute_bootstrap_ci_from_predictions(
    predictions_dir: str,
    data_dir: str,
    model_name: str,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> dict:
    """Compute bootstrap CIs for GG and GS from raw predictions."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from mei_benchmark.data.loader import MEIDatasetLoader, PredictionLoader
    from mei_benchmark.evaluation.metrics import MEIEvaluator

    loader = MEIDatasetLoader(data_dir)
    items = loader.load_items()

    pred_loader = PredictionLoader(predictions_dir)
    all_models = pred_loader.available_models()

    target = None
    for m in all_models:
        if model_name.lower() in m.lower():
            target = m
            break

    if target is None:
        print(f"WARNING: No predictions found for {model_name}")
        return {}

    pred_sets = pred_loader.load_predictions(target)

    # Build per-item GG values for bootstrap
    from mei_benchmark.data.schema import InterventionType

    pred_index = {ps.item_id: ps for ps in pred_sets}

    cs_values = []  # per-item CS (blur)
    si_values = []  # per-item SI (blur)

    for item in items:
        pset = pred_index.get(item.item_id)
        if pset is None:
            continue

        original = pset.original_prediction
        if original is None:
            continue

        correct = original.answer == item.correct_answer

        evidence_pred = pset.get_prediction(InterventionType.EVIDENCE_BLUR)
        sham_pred = pset.get_prediction(InterventionType.SHAM_BLUR)

        if correct and evidence_pred is not None:
            flipped = evidence_pred.answer != item.correct_answer
            cs_values.append(1.0 if flipped else 0.0)
        else:
            cs_values.append(0.0)

        if correct and sham_pred is not None:
            flipped = sham_pred.answer != item.correct_answer
            si_values.append(1.0 if flipped else 0.0)
        else:
            si_values.append(0.0)

    cs_arr = np.array(cs_values)
    si_arr = np.array(si_values)

    rng = np.random.default_rng(42)
    n = len(cs_arr)

    gg_samples = []
    gs_samples = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        cs_boot = cs_arr[idx].mean()
        si_boot = si_arr[idx].mean()
        gg_boot = cs_boot - si_boot
        gs_boot = max(gg_boot, 0.0) / max(cs_boot, 1e-8)
        gg_samples.append(gg_boot)
        gs_samples.append(gs_boot)

    alpha = 1 - ci_level
    gg_lo = np.percentile(gg_samples, 100 * alpha / 2)
    gg_hi = np.percentile(gg_samples, 100 * (1 - alpha / 2))
    gs_lo = np.percentile(gs_samples, 100 * alpha / 2)
    gs_hi = np.percentile(gs_samples, 100 * (1 - alpha / 2))

    return {
        "gg_ci": [round(gg_lo, 3), round(gg_hi, 3)],
        "gs_ci": [round(gs_lo, 3), round(gs_hi, 3)],
    }


def format_latex_row(model: str, m: dict, ci: dict = None) -> str:
    """Format a single model's LaTeX table row."""
    gg_ci_str = ""
    if ci and "gg_ci" in ci:
        gg_ci_str = f" & \\scriptsize[{ci['gg_ci'][0]:.3f},{ci['gg_ci'][1]:.3f}]"
    else:
        gg_ci_str = " & "

    return (
        f"{model:20s} & {m['original_accuracy']:.3f} "
        f"& {m['CS_blur']:.3f} & {m['CS_gray']:.3f} "
        f"& {m['SI_blur']:.3f} & {m['GG']:.3f}{gg_ci_str} "
        f"& {m['SR_blur']:.3f} & {m['GS']:.3f} \\\\"
    )


def main():
    metrics_dir = str(OUTPUT_DIR / "metrics")
    predictions_dir = str(OUTPUT_DIR / "predictions")
    data_dir = str(DATA_DIR)

    print("=" * 80)
    print("MEI Benchmark - All Model Results Summary")
    print("=" * 80)

    metrics = load_all_metrics(metrics_dir)

    if not metrics:
        print("No metrics files found!")
        return

    # Sort by model family
    family_order = [
        "LLaVA-1.5-7B", "LLaVA-1.5-13B", "LLaVA-NeXT-7B",
        "Qwen2-VL-7B",
        "InternVL2-8B", "InternVL2.5-8B",
        "Phi-3.5-Vision", "Llama-3.2-11B-Vision",
        "DeepSeek-VL-7B", "Molmo-7B-D",
    ]

    # Print plain-text table
    print(f"\n{'Model':25s} {'Acc':>6s} {'CS_b':>6s} {'CS_g':>6s} "
          f"{'SI_b':>6s} {'GG':>6s} {'SR_b':>6s} {'GS':>6s}")
    print("-" * 80)

    ordered = []
    for name in family_order:
        if name in metrics:
            ordered.append((name, metrics[name]))
    # Add any models not in the predefined order
    for name, m in metrics.items():
        if name not in [o[0] for o in ordered]:
            ordered.append((name, m))

    for name, m in ordered:
        print(f"{name:25s} {m['original_accuracy']:6.3f} {m['CS_blur']:6.3f} "
              f"{m['CS_gray']:6.3f} {m['SI_blur']:6.3f} {m['GG']:6.3f} "
              f"{m['SR_blur']:6.3f} {m['GS']:6.3f}")

    # Print LaTeX table
    print("\n" + "=" * 80)
    print("LaTeX Table Rows (for main.tex Table 1):")
    print("=" * 80)

    for name, m in ordered:
        print(format_latex_row(name, m))

    # Summary statistics
    all_gg = [m["GG"] for _, m in ordered]
    all_gs = [m["GS"] for _, m in ordered]
    print(f"\n{'GG range:':20s} {min(all_gg):.3f} -- {max(all_gg):.3f}")
    print(f"{'GS range:':20s} {min(all_gs):.3f} -- {max(all_gs):.3f}")
    print(f"{'Models evaluated:':20s} {len(ordered)}")
    print(f"{'Model families:':20s} "
          f"{len(set(n.split('-')[0] for n, _ in ordered))}")


if __name__ == "__main__":
    main()
