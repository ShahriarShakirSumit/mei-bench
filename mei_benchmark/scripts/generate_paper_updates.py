from __future__ import annotations
#!/usr/bin/env python3
"""
Generate updated LaTeX tables and text for 10-model MEI-Bench paper.

Run this AFTER all 4 new model evaluations complete.
Reads metrics from <MEI_OUTPUTS>/metrics/
and generates updated paper sections.

Usage:
    python scripts/generate_paper_updates.py
"""

from mei_benchmark.paths import DATA_DIR as _DATA_DIR, OUTPUT_DIR

import json
import sys
from pathlib import Path

import numpy as np

METRICS_DIR = str(OUTPUT_DIR / "metrics")
PREDICTIONS_DIR = str(OUTPUT_DIR / "predictions")
DATA_DIR = str(_DATA_DIR)

# Expected model order for tables
MODEL_ORDER = [
    "LLaVA-1.5-7B",
    "LLaVA-1.5-13B",
    "LLaVA-NeXT-7B",
    "Qwen2-VL-7B",
    "InternVL2-8B",
    "InternVL2.5-8B",
    "Phi-3.5-Vision",
    "Llama-3.2-11B-Vision",
    "DeepSeek-VL-7B",
    "Molmo-7B-D",
]

# LaTeX-friendly display names
DISPLAY_NAMES = {
    "LLaVA-1.5-7B": "LLaVA-1.5-7B",
    "LLaVA-1.5-13B": "LLaVA-1.5-13B",
    "LLaVA-NeXT-7B": "LLaVA-NeXT-7B",
    "Qwen2-VL-7B": "Qwen2-VL-7B",
    "InternVL2-8B": "InternVL2-8B",
    "InternVL2.5-8B": "InternVL2.5-8B",
    "Phi-3.5-Vision": "Phi-3.5-Vision",
    "Llama-3.2-11B-Vision": "Llama-3.2-11B-Vision",
    "DeepSeek-VL-7B": "DeepSeek-VL-7B",
    "Molmo-7B-D": "Molmo-7B-D",
}

# Known bootstrap CIs from previous run (6 models)
KNOWN_BOOTSTRAP_CIS = {
    "LLaVA-1.5-7B": {"gg_ci": [0.121, 0.227], "gs_ci": [0.527, 0.776]},
    "LLaVA-1.5-13B": {"gg_ci": [0.116, 0.228], "gs_ci": [0.516, 0.773]},
    "LLaVA-NeXT-7B": {"gg_ci": [0.148, 0.251], "gs_ci": [0.635, 0.867]},
    "Qwen2-VL-7B": {"gg_ci": [0.176, 0.289], "gs_ci": [0.598, 0.802]},
    "InternVL2-8B": {"gg_ci": [0.170, 0.286], "gs_ci": [0.579, 0.794]},
    "InternVL2.5-8B": {"gg_ci": [0.186, 0.297], "gs_ci": [0.644, 0.826]},
}


def load_metrics():
    metrics = {}
    p = Path(METRICS_DIR)
    for f in sorted(p.glob("*_metrics.json")):
        with open(f) as fh:
            data = json.load(fh)
        metrics[data["model"]] = data
    return metrics


def compute_bootstrap_ci(predictions_dir, data_dir, model_name, n_bootstrap=1000):
    """Compute bootstrap CIs for GG and GS."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from mei_benchmark.data.loader import MEIDatasetLoader, PredictionLoader
    from mei_benchmark.data.schema import InterventionType

    loader = MEIDatasetLoader(data_dir)
    items = loader.load_items()
    pred_loader = PredictionLoader(predictions_dir)

    target = None
    for m in pred_loader.available_models():
        if model_name.lower() in m.lower():
            target = m
            break
    if not target:
        return None

    pred_sets = pred_loader.load_predictions(target)
    pred_index = {ps.item_id: ps for ps in pred_sets}

    cs_values, si_values = [], []
    for item in items:
        pset = pred_index.get(item.item_id)
        if not pset or not pset.original_prediction:
            continue
        correct = pset.original_prediction.answer == item.correct_answer
        ev = pset.get_prediction(InterventionType.EVIDENCE_BLUR)
        sh = pset.get_prediction(InterventionType.SHAM_BLUR)
        cs_values.append(1.0 if (correct and ev and ev.answer != item.correct_answer) else 0.0)
        si_values.append(1.0 if (correct and sh and sh.answer != item.correct_answer) else 0.0)

    cs_arr, si_arr = np.array(cs_values), np.array(si_values)
    rng = np.random.default_rng(42)
    n = len(cs_arr)
    gg_samples, gs_samples = [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        cs_b, si_b = cs_arr[idx].mean(), si_arr[idx].mean()
        gg_b = cs_b - si_b
        gs_b = max(gg_b, 0.0) / max(cs_b, 1e-8)
        gg_samples.append(gg_b)
        gs_samples.append(gs_b)

    return {
        "gg_ci": [round(np.percentile(gg_samples, 2.5), 3), round(np.percentile(gg_samples, 97.5), 3)],
        "gs_ci": [round(np.percentile(gs_samples, 2.5), 3), round(np.percentile(gs_samples, 97.5), 3)],
    }


def generate_main_table(metrics, bootstrap_cis):
    """Generate Table 1 (main results)."""
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Main \\mei{}-Bench results across ten open-source models spanning seven model families. "
                 "Acc = original accuracy, CS = Causal Sensitivity ($\\uparrow$), SI = Sham Invariance ($\\downarrow$), "
                 "GG = Grounding Gap ($\\uparrow$), SR = Sham Robustness ($\\uparrow$), GS = Grounding Specificity ($\\uparrow$). "
                 "Best values in \\textbf{bold}, worst \\underline{underlined}. "
                 "GG and GS computed on blur interventions; 95\\% bootstrap CIs ($B{=}1000$) reported for GG and GS.}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{@{}lcccccccc@{}}")
    lines.append("\\toprule")
    lines.append("Model & Acc & CS$_\\text{blur}$ & CS$_\\text{gray}$ & SI$_\\text{blur}$ & GG & GG 95\\% CI & SR & GS \\\\")
    lines.append("\\midrule")

    # Get available models in order
    available = [(name, metrics[name]) for name in MODEL_ORDER if name in metrics]

    # Find best/worst for bold/underline
    vals = {k: [m[k] for _, m in available] for k in ["original_accuracy", "CS_blur", "CS_gray", "SI_blur", "GG", "SR_blur", "GS"]}

    for name, m in available:
        ci = bootstrap_cis.get(name, {})
        gg_ci_str = f"\\scriptsize[{ci['gg_ci'][0]:.3f},{ci['gg_ci'][1]:.3f}]" if "gg_ci" in ci else ""

        def fmt(key, val, higher_better=True):
            v = f".{int(val*1000):03d}" if val < 1 else f"{val:.3f}"
            all_vals = vals[key]
            if higher_better:
                if val == max(all_vals):
                    return f"\\textbf{{{v}}}"
                if val == min(all_vals):
                    return f"\\underline{{{v}}}"
            else:  # lower is better (SI)
                if val == min(all_vals):
                    return f"\\textbf{{{v}}}"
                if val == max(all_vals):
                    return f"\\underline{{{v}}}"
            return v

        row = (f"{DISPLAY_NAMES.get(name, name):20s} & "
               f"{fmt('original_accuracy', m['original_accuracy'])} & "
               f"{fmt('CS_blur', m['CS_blur'])} & "
               f"{fmt('CS_gray', m['CS_gray'])} & "
               f"{fmt('SI_blur', m['SI_blur'], False)} & "
               f"{fmt('GG', m['GG'])} & "
               f"{gg_ci_str} & "
               f"{fmt('SR_blur', m['SR_blur'])} & "
               f"{fmt('GS', m['GS'])} \\\\")
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_summary(metrics):
    """Generate summary text for abstract/conclusion updates."""
    available = [(name, metrics[name]) for name in MODEL_ORDER if name in metrics]

    all_gg = [m["GG"] for _, m in available]
    all_gs = [m["GS"] for _, m in available]
    all_acc = [m["original_accuracy"] for _, m in available]

    best_gs_name = available[np.argmax(all_gs)][0]
    worst_gs_name = available[np.argmin(all_gs)][0]
    best_acc_name = available[np.argmax(all_acc)][0]

    fragility_pct_low = int((1 - max(all_gs)) * 100)
    fragility_pct_high = int((1 - min(all_gs)) * 100)

    print(f"\n{'='*80}")
    print("SUMMARY FOR PAPER TEXT UPDATES")
    print(f"{'='*80}")
    print(f"Models evaluated: {len(available)}")
    print(f"GG range: {min(all_gg):.2f} -- {max(all_gg):.2f}")
    print(f"GS range: {min(all_gs):.2f} -- {max(all_gs):.2f}")
    print(f"Best GS: {best_gs_name} ({max(all_gs):.3f})")
    print(f"Worst GS: {worst_gs_name} ({min(all_gs):.3f})")
    print(f"Fragility range: {fragility_pct_low}--{fragility_pct_high}% of CS attributable to fragility")
    print(f"Best accuracy: {best_acc_name} ({max(all_acc):.3f})")


def main():
    metrics = load_metrics()
    print(f"Loaded metrics for {len(metrics)} models: {sorted(metrics.keys())}")

    missing = [m for m in MODEL_ORDER if m not in metrics]
    if missing:
        print(f"\nWARNING: Missing metrics for: {missing}")
        print("Run evaluation jobs first, then re-run this script.")

    # Get bootstrap CIs (use known ones + compute for new)
    bootstrap_cis = dict(KNOWN_BOOTSTRAP_CIS)
    new_models = [m for m in MODEL_ORDER if m not in KNOWN_BOOTSTRAP_CIS and m in metrics]

    for model_name in new_models:
        print(f"\nComputing bootstrap CIs for {model_name}...")
        ci = compute_bootstrap_ci(PREDICTIONS_DIR, DATA_DIR, model_name)
        if ci:
            bootstrap_cis[model_name] = ci
            print(f"  GG CI: {ci['gg_ci']}, GS CI: {ci['gs_ci']}")

    # Generate Table 1
    print(f"\n{'='*80}")
    print("TABLE 1 (Main Results):")
    print(f"{'='*80}")
    table = generate_main_table(metrics, bootstrap_cis)
    print(table)

    # Generate summary
    generate_summary(metrics)

    # Save bootstrap CIs
    ci_path = Path(METRICS_DIR) / "bootstrap_cis_all.json"
    with open(ci_path, "w") as f:
        json.dump(bootstrap_cis, f, indent=2)
    print(f"\nBootstrap CIs saved to: {ci_path}")


if __name__ == "__main__":
    main()
