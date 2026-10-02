from __future__ import annotations
#!/usr/bin/env python3
"""Generate the three appendix tables (full stratified, area effects,
intervention ablation) from current 615-item, 12-model results."""
from mei_benchmark.paths import DATA_DIR, OUTPUT_DIR
import json
from pathlib import Path
import numpy as np

from mei_benchmark.data.loader import MEIDatasetLoader, PredictionLoader
from mei_benchmark.evaluation.metrics import (
    MEIEvaluator, StratifiedEvaluator,
    answers_match, answer_flipped,
)

DATA = DATA_DIR
OUT = OUTPUT_DIR
PAPER_AUTO = Path(__file__).resolve().parents[1] / "paper" / "auto"
PAPER_AUTO.mkdir(parents=True, exist_ok=True)


def model_short(m: str) -> str:
    return m.replace("Llama-3.2-11B-Vision", "Llama-3.2-11B")\
            .replace("Phi-3.5-Vision", "Phi-3.5-V")


def per_item_flips(items, pred_sets, ev_label, sh_label):
    pmap = {p.item_id: p for p in pred_sets}
    out = []
    for it in items:
        ps = pmap.get(it.item_id)
        if not ps: continue
        ans = {}
        for pp in ps.predictions:
            itype = pp.intervention_type
            key = itype.value if hasattr(itype, "value") else str(itype)
            ans[key] = pp.answer
        orig = ans.get("original")
        if orig is None or not answers_match(orig, it.valid_answers):
            continue
        ev = ans.get(ev_label); sh = ans.get(sh_label)
        if ev is None or sh is None:
            continue
        out.append((it,
                    int(answer_flipped(orig, ev)),
                    int(answer_flipped(orig, sh))))
    return out


def main():
    items = MEIDatasetLoader(DATA).load_items()
    pred_loader = PredictionLoader(str(OUT / "predictions"))
    models = pred_loader.available_models()
    strat_eval = StratifiedEvaluator()

    # Order models by overall GS (matches main table)
    order = []
    for m in models:
        ps = pred_loader.load_predictions(m)
        flips = per_item_flips(items, ps, "evidence_blur", "sham_blur")
        if flips:
            ev = np.mean([f[1] for f in flips])
            sh = np.mean([f[2] for f in flips])
            gs = max(ev - sh, 0) / max(ev, 1e-9)
        else:
            gs = -1
        order.append((m, gs))
    order.sort(key=lambda x: -x[1])
    model_list = [m for m, _ in order]

    # ---------- FULL STRATIFIED TABLE -------------------------------------
    task_order = ["object_identification", "attribute_verification",
                  "spatial_relations", "counting", "text_in_image"]
    task_pretty = {
        "object_identification": "Obj.\\ ID",
        "attribute_verification": "Attr",
        "spatial_relations": "Spatial",
        "counting": "Count",
        "text_in_image": "Text",
    }
    lines = []
    lines.append(r"\begin{tabular}{@{}llcccccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Task & Model & Acc & $n_c$ & CS & SI & GG & SR & GS \\")
    lines.append(r"\midrule")
    for ti, t in enumerate(task_order):
        if ti > 0:
            lines.append(r"\midrule")
        for mi, m in enumerate(model_list):
            ps = pred_loader.load_predictions(m)
            s = strat_eval.evaluate_stratified(items, ps, m)
            tm = None
            for k, v in s.by_task.items():
                kk = k.value if hasattr(k, "value") else k
                if kk == t:
                    tm = v; break
            task_label = task_pretty[t] if mi == 0 else ""
            if tm is None:
                lines.append(f"{task_label} & {model_short(m)} & --- & 0 & --- & --- & --- & --- & --- \\\\")
                continue
            nc = int(round(tm.original_accuracy * tm.n_items))
            dagger = r"^\dagger" if nc < 30 else ""
            def f(x): return f"{x:.3f}".lstrip("0").replace("-0.", "-.") if isinstance(x, float) else x
            lines.append(
                f"{task_label} & {model_short(m)} & {f(tm.original_accuracy)} & "
                f"${nc}{dagger}$ & {f(tm.causal_sensitivity_blur)} & "
                f"{f(tm.spurious_invariance_blur)} & {f(tm.grounding_gap)} & "
                f"{f(tm.sham_robustness_blur)} & {f(tm.grounding_specificity)} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (PAPER_AUTO / "tab_full_stratified.tex").write_text("\n".join(lines))

    # ---------- AREA EFFECTS TABLE ----------------------------------------
    bins = [(0.005, 0.05, "Small ($<$5\\%)"),
            (0.05, 0.15, "Medium (5--15\\%)"),
            (0.15, 1.0, "Large ($>$15\\%)")]
    lines = []
    lines.append(r"\begin{tabular}{@{}lcccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Model & " + " & ".join(b[2] for b in bins) + r" & Items \\")
    lines.append(r"\midrule")
    for m in model_list:
        ps = pred_loader.load_predictions(m)
        flips = per_item_flips(items, ps, "evidence_blur", "sham_blur")
        if not flips:
            lines.append(f"{model_short(m)} & --- & --- & --- & 0 \\\\")
            continue
        af = np.array([f[0].evidence_region.area_fraction for f in flips])
        ev = np.array([f[1] for f in flips])
        sh = np.array([f[2] for f in flips])
        cells = []
        for lo, hi, _ in bins:
            mask = (af >= lo) & (af < hi)
            if mask.sum() < 5:
                cells.append(r"\scriptsize n/a")
            else:
                gg = ev[mask].mean() - sh[mask].mean()
                cells.append(f"{gg:.3f}".lstrip("0").replace("-0.", "-."))
        lines.append(f"{model_short(m)} & " + " & ".join(cells)
                     + f" & {len(flips)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (PAPER_AUTO / "tab_area_effects.tex").write_text("\n".join(lines))

    # ---------- INTERVENTION ABLATION TABLE -------------------------------
    rows = []
    for m in model_list:
        ps = pred_loader.load_predictions(m)
        b = per_item_flips(items, ps, "evidence_blur", "sham_blur")
        g = per_item_flips(items, ps, "evidence_gray", "sham_gray")
        sw = per_item_flips(items, ps, "evidence_swap", "sham_blur")
        def gg(fl):
            if not fl: return None
            return np.mean([x[1] for x in fl]) - np.mean([x[2] for x in fl])
        rows.append((m, gg(b), gg(g), gg(sw)))
    valid = [r for r in rows if r[1] is not None and r[2] is not None and r[3] is not None
             and len([f for f in per_item_flips(items, pred_loader.load_predictions(r[0]), "evidence_blur", "sham_blur")]) >= 30]
    if valid:
        gg_b = np.array([r[1] for r in valid])
        gg_g = np.array([r[2] for r in valid])
        gg_s = np.array([r[3] for r in valid])
        from scipy.stats import spearmanr
        rho_g, _ = spearmanr(gg_b, gg_g)
        rho_s, _ = spearmanr(gg_b, gg_s)
        lines = []
        lines.append(r"\begin{tabular}{@{}lccc@{}}")
        lines.append(r"\toprule")
        lines.append(r" & Blur & Grayscale & Swap \\")
        lines.append(r"\midrule")
        lines.append(f"Mean GG (n={len(valid)} models) & "
                     f"{gg_b.mean():.3f} & {gg_g.mean():.3f} & {gg_s.mean():.3f} \\\\".replace("0.", "."))
        lines.append(f"Spearman $\\rho$ vs.\\ blur & 1.000 & {rho_g:.3f} & {rho_s:.3f} \\\\".replace("1.000", "1.000"))
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        (PAPER_AUTO / "tab_ablation_intervention.tex").write_text("\n".join(lines))

    print("wrote tab_full_stratified.tex, tab_area_effects.tex, tab_ablation_intervention.tex")


if __name__ == "__main__":
    main()
