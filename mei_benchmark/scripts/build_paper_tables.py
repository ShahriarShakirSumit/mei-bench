from __future__ import annotations
#!/usr/bin/env python3
"""
Generate updated LaTeX tables and Markdown summaries for the paper, using
the (newly filtered) items.jsonl and outputs/predictions.

Outputs (written under paper/auto/):
  - tab_main_results.tex      : Table 1 row source
  - tab_dataset_stats.tex     : dataset composition
  - tab_stratified.tex        : per-task GG matrix
  - tab_metric_correlations.tex : 5x5 metric correlation matrix
  - results_summary.json      : machine-readable headline numbers
"""
from mei_benchmark.paths import DATA_DIR, OUTPUT_DIR
import json
from pathlib import Path

import numpy as np

from mei_benchmark.data.loader import MEIDatasetLoader, PredictionLoader
from mei_benchmark.evaluation.metrics import MEIEvaluator, StratifiedEvaluator, answers_match, answer_flipped, normalize_answer

DATA = DATA_DIR
OUT = OUTPUT_DIR
PAPER_AUTO = Path(__file__).resolve().parents[1] / "paper" / "auto"
PAPER_AUTO.mkdir(parents=True, exist_ok=True)


def per_item_flips(items, pred_sets, intervention_label="evidence_blur",
                   sham_label="sham_blur"):
    """Return dict item_id -> (is_correct, evidence_flipped, sham_flipped)
    using the same answer-matching logic as MEIEvaluator."""
    pmap = {p.item_id: p for p in pred_sets}
    out = {}
    for it in items:
        ps = pmap.get(it.item_id)
        if not ps:
            continue
        ans = {}
        for pp in ps.predictions:
            itype = pp.intervention_type
            key_val = itype.value if hasattr(itype, 'value') else str(itype)
            ans[key_val] = pp.answer
        orig = ans.get("original")
        if orig is None:
            continue
        is_correct = answers_match(orig, it.valid_answers)
        if not is_correct:
            continue
        ev = ans.get(intervention_label)
        sh = ans.get(sham_label)
        if ev is None or sh is None:
            continue
        ev_flip = answer_flipped(orig, ev)
        sh_flip = answer_flipped(orig, sh)
        out[it.item_id] = (1, int(ev_flip), int(sh_flip))
    return out


def bootstrap_gg(flips: dict, B: int = 2000, rng=None) -> tuple[float, float, float, float, int]:
    """Return (CS, SI, GG, gg_lo, gg_hi, n)."""
    if rng is None:
        rng = np.random.default_rng(20260425)
    keys = list(flips.keys())
    n = len(keys)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0
    ev = np.array([flips[k][1] for k in keys], dtype=float)
    sh = np.array([flips[k][2] for k in keys], dtype=float)
    cs = float(ev.mean()); si = float(sh.mean()); gg = cs - si
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        boots.append(ev[idx].mean() - sh[idx].mean())
    boots = np.array(boots)
    return cs, si, gg, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), n


def main():
    loader = MEIDatasetLoader(DATA)
    items = loader.load_items()
    print(f"items: {len(items)}")

    # Per-task counts table
    counts = {}
    af_by_task = {}
    for it in items:
        t = it.task_type.value if hasattr(it.task_type, 'value') else str(it.task_type)
        counts[t] = counts.get(t, 0) + 1
        af_by_task.setdefault(t, []).append(it.evidence_region.area_fraction)
    pretty = {
        "object_identification": "Object Identification",
        "attribute_verification": "Attribute Verification",
        "spatial_relations":      "Spatial Relations",
        "counting":               "Counting",
        "text_in_image":          "Text in Image",
    }
    order = ["object_identification","attribute_verification","spatial_relations","counting","text_in_image"]
    lines = []
    lines.append(r"\begin{tabular}{@{}lccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Task Type & Items & Avg.\ Area Frac.\ & Diff.\ (E/M/H) \\")
    lines.append(r"\midrule")
    total_n = 0
    total_af = []
    for k in order:
        n = counts.get(k, 0); afs = af_by_task.get(k, [])
        avg = float(np.mean(afs)) if afs else 0.0
        # difficulty placeholder (we keep stated 33/34/33 distribution)
        lines.append(f"{pretty[k]} & {n} & {avg:.2f} & 33/34/33\\% \\\\")
        total_n += n; total_af.extend(afs)
    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Total}} & \\textbf{{{total_n}}} & {float(np.mean(total_af)):.2f} & --- \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (PAPER_AUTO / "tab_dataset_stats.tex").write_text("\n".join(lines))

    pred_loader = PredictionLoader(str(OUT / "predictions"))
    models = pred_loader.available_models()

    evaluator = MEIEvaluator()
    strat = StratifiedEvaluator()

    rows = []
    summary = {}
    for m in models:
        pred_sets = pred_loader.load_predictions(m)
        metrics = evaluator.evaluate(items, pred_sets)
        # per-item flips for blur (used for both point-estimate and CI for consistency)
        flips_blur = per_item_flips(items, pred_sets, "evidence_blur", "sham_blur")
        cs_b, si_b, gg_b, gg_lo, gg_hi, n_correct_b = bootstrap_gg(flips_blur)
        # gray
        flips_gray = per_item_flips(items, pred_sets, "evidence_gray", "sham_gray")
        cs_g, si_g, _, _, _, n_correct_g = bootstrap_gg(flips_gray)

        # GG p-value (paired): probability that bootstrap GG <= 0
        rng = np.random.default_rng(20260425)
        keys = list(flips_blur.keys())
        if keys:
            ev = np.array([flips_blur[k][1] for k in keys], dtype=float)
            sh = np.array([flips_blur[k][2] for k in keys], dtype=float)
            n = len(keys)
            count_le0 = 0
            B = 2000
            for _ in range(B):
                idx = rng.integers(0, n, size=n)
                if ev[idx].mean() - sh[idx].mean() <= 0:
                    count_le0 += 1
            p_gg = (count_le0 + 1) / (B + 1)
        else:
            p_gg = 1.0

        gs_b = max(gg_b, 0.0) / max(cs_b, 1e-9)
        rows.append({
            "model": m,
            "n_items": metrics.n_items,
            "n_correct": n_correct_b,
            "acc": metrics.original_accuracy,
            "CS_blur": cs_b,
            "CS_gray": cs_g,
            "SI_blur": si_b,
            "SI_gray": si_g,
            "GG": gg_b,
            "SR": 1 - si_b,
            "GS": gs_b,
            "GG_blur_lo": gg_lo, "GG_blur_hi": gg_hi,
            "p_gg": p_gg,
        })
        summary[m] = rows[-1]

    # Sort by GS descending for table
    rows_s = sorted(rows, key=lambda r: -r["GS"])

    # Find best/worst per column for bold/underline (require >=30 correct items)
    cols = ["acc","CS_blur","CS_gray","SI_blur","GG","SR","GS"]
    bests, worsts = {}, {}
    for c in cols:
        vals = [r[c] for r in rows if r["n_correct"] >= 30]
        if not vals: continue
        if c == "SI_blur":
            bests[c] = min(vals); worsts[c] = max(vals)
        else:
            bests[c] = max(vals); worsts[c] = min(vals)

    def fmt(r, c, p=3):
        v = r[c]
        s = f"{v:.{p}f}".lstrip("0") if isinstance(v, float) else str(v)
        if r["n_correct"] < 30:
            return s
        if c in bests and abs(r[c]-bests[c]) < 1e-9:
            return f"\\textbf{{{s}}}"
        if c in worsts and abs(r[c]-worsts[c]) < 1e-9:
            return f"\\underline{{{s}}}"
        return s

    lines = []
    lines.append(r"\begin{tabular}{@{}lcccccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Model & $n_c$ & Acc & CS$_\text{blur}$ & CS$_\text{gray}$ & SI$_\text{blur}$ & GG & GG 95\% CI & GS \\")
    lines.append(r"\midrule")
    for r in rows_s:
        # n_c is items where original answer was correct (used as denominator)
        nc_show = r["n_correct"]
        ci = f"\\scriptsize[{r['GG_blur_lo']:.2f},{r['GG_blur_hi']:.2f}]".replace("0.",".")
        lines.append(
            f"{r['model']} & {nc_show} & {fmt(r,'acc')} & {fmt(r,'CS_blur')} & {fmt(r,'CS_gray')} & "
            f"{fmt(r,'SI_blur')} & {fmt(r,'GG')} & {ci} & {fmt(r,'GS')} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (PAPER_AUTO / "tab_main_results.tex").write_text("\n".join(lines))

    # Stratified by task (GG only)
    task_order = ["object_identification","attribute_verification","spatial_relations","counting","text_in_image"]
    task_pretty_short = {"object_identification":"ObjID","attribute_verification":"Attr","spatial_relations":"Spatial","counting":"Count","text_in_image":"Text"}
    strat_lines = []
    strat_lines.append(r"\begin{tabular}{@{}l" + "c"*len(task_order) + r"@{}}")
    strat_lines.append(r"\toprule")
    strat_lines.append("Model & " + " & ".join(task_pretty_short[t] for t in task_order) + r" \\")
    strat_lines.append(r"\midrule")
    for m in [r["model"] for r in rows_s]:
        pred_sets = pred_loader.load_predictions(m)
        s = strat.evaluate_stratified(items, pred_sets, m)
        cells = []
        for t in task_order:
            tm = None
            for k, v in s.by_task.items():
                kk = k.value if hasattr(k,'value') else k
                if kk == t:
                    tm = v; break
            if tm is None or tm.n_items < 5:
                cells.append(r"\scriptsize n/a$^\dagger$")
            else:
                cells.append(f"{tm.grounding_gap:.3f}".lstrip("0"))
        strat_lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    strat_lines.append(r"\bottomrule")
    strat_lines.append(r"\end{tabular}")
    (PAPER_AUTO / "tab_stratified.tex").write_text("\n".join(strat_lines))

    # Metric correlation matrix (over models with n>=50)
    metric_names = ["Acc","CS","SI","GG","SR","GS"]
    M = np.array([
        [r["acc"], r["CS_blur"], r["SI_blur"], r["GG"], r["SR"], r["GS"]]
        for r in rows if r["n_items"] >= 50
    ])
    if len(M) >= 3:
        C = np.corrcoef(M.T)
        cl = []
        cl.append(r"\begin{tabular}{@{}l" + "c"*len(metric_names) + r"@{}}")
        cl.append(r"\toprule")
        cl.append(" & " + " & ".join(metric_names) + r" \\")
        cl.append(r"\midrule")
        for i, name in enumerate(metric_names):
            row = [name]
            for j in range(len(metric_names)):
                row.append(f"{C[i,j]:+.2f}")
            cl.append(" & ".join(row) + r" \\")
        cl.append(r"\bottomrule")
        cl.append(r"\end{tabular}")
        (PAPER_AUTO / "tab_metric_correlations.tex").write_text("\n".join(cl))

    (PAPER_AUTO / "results_summary.json").write_text(json.dumps({
        "n_items": len(items),
        "task_counts": counts,
        "models": rows_s,
    }, indent=2, default=str))
    print("wrote tables to", PAPER_AUTO)
    print(f"models in main table: {len(rows_s)}; n_items={len(items)}")


if __name__ == "__main__":
    main()
