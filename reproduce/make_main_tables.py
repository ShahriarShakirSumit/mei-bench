#!/usr/bin/env python
"""Single source of every cross-model number in the MEI-Bench paper (ACML 2026).

Rows: the values of the original run (reproduce/inputs/original_run_summary.json) for the eleven models, except
Molmo-7B-D, whose row is recomputed from the pinned re-run with model-specific answer extraction
(reproduce/molmo_extract.py) using the paper's OWN per_item_flips + bootstrap_gg (B=2000, seed 20260425).
Model sets (stated explicitly, used everywhere):
  H (headline ranges)   = the ten open-source models (all n_c >= 30)
  X (cross-model stats) = H + Claude-Sonnet-4.5  (Gemini-2.5-Pro is a preliminary truncated run: excluded)
Writes: outputs/paper/tables/tab_main_results.tex, outputs/paper/tables/tab_metric_correlations.tex,
        outputs/paper/main_numbers.json
"""
import json, sys
from pathlib import Path
import numpy as np
from repro_paths import RERUN_DATA, MOLMO_FIX, ORIGINAL_SUMMARY, RESULTS, TABLES
from mei_benchmark.scripts.build_paper_tables import per_item_flips, bootstrap_gg   # noqa: E402
from mei_benchmark.data.loader import MEIDatasetLoader, PredictionLoader            # noqa: E402
from mei_benchmark.evaluation.metrics import MEIEvaluator                            # noqa: E402


SUB = json.load(open(ORIGINAL_SUMMARY))["models"]
rows = {m["model"]: dict(m) for m in SUB}

# ---- Molmo row: paper's procedure on the re-run with model-specific extraction ----------------------
items = MEIDatasetLoader(str(RERUN_DATA)).load_items()
ps = PredictionLoader(str(MOLMO_FIX)).load_predictions("Molmo-7B-D")
met = MEIEvaluator().evaluate(items, ps)
fb = per_item_flips(items, ps, "evidence_blur", "sham_blur"); fg = per_item_flips(items, ps, "evidence_gray", "sham_gray")
cs_b, si_b, gg_b, lo, hi, n_b = bootstrap_gg(fb); cs_g, si_g, _, _, _, _ = bootstrap_gg(fg)
rng = np.random.default_rng(20260425); k = list(fb); ev = np.array([fb[x][1] for x in k], float); sh = np.array([fb[x][2] for x in k], float)
p_gg = (sum(ev[i].mean() - sh[i].mean() <= 0 for i in (rng.integers(0, len(k), len(k)) for _ in range(2000))) + 1) / 2001
rows["Molmo-7B-D"] = dict(model="Molmo-7B-D", n_items=615, n_correct=n_b, acc=met.original_accuracy, CS_blur=cs_b, CS_gray=cs_g,
                          SI_blur=si_b, SI_gray=si_g, GG=gg_b, SR=1 - si_b, GS=max(gg_b, 0) / max(cs_b, 1e-9),
                          GG_blur_lo=lo, GG_blur_hi=hi, p_gg=p_gg, source="pinned re-run + model-specific extraction")

OPEN = ["Qwen2-VL-7B", "InternVL2.5-8B", "InternVL2-8B", "LLaVA-NeXT-7B", "LLaVA-1.5-13B", "LLaVA-1.5-7B",
        "Phi-3.5-Vision", "Llama-3.2-11B-Vision", "DeepSeek-VL-7B", "Molmo-7B-D"]
H = [rows[m] for m in OPEN]; X = H + [rows["Claude-Sonnet-4.5"]]; G = rows["Gemini-2.5-Pro"]
assert all(r["n_correct"] >= 30 for r in X)

f3 = lambda v: f"{v:.3f}".replace("0.", ".", 1) if v >= 0 else f"-{abs(v):.3f}".replace("0.", ".", 1)
def line(r, mark=""):
    ci = f"\\scriptsize[{r['GG_blur_lo']:.2f},{r['GG_blur_hi']:.2f}]".replace("0.", ".")
    return (f"{r['model']}{mark} & {r['n_correct']} & {f3(r['acc'])} & {f3(r['CS_blur'])} & {f3(r['CS_gray'])} & "
            f"{f3(r['SI_blur'])} & {f3(r['GG'])} & {ci} & {f3(r['GS'])} \\\\")
L = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
     r"Model & $n_c$ & Acc & CS$_\text{blur}$ & CS$_\text{gray}$ & SI$_\text{blur}$ & GG & GG 95\% CI & GS \\", r"\midrule"]
for r in sorted(X, key=lambda r: -r["GS"]): L.append(line(r, "$^\\dagger$" if r["model"] == "Molmo-7B-D" else ""))
L += [r"\midrule", r"\multicolumn{9}{@{}l}{\scriptsize\textit{Preliminary (free-tier-truncated run; excluded from all ranges and comparisons)}} \\",
      line(G), r"\bottomrule", r"\end{tabular}"]
(TABLES / "tab_main_results.tex").write_text("\n".join(L) + "\n")

# ---- correlation matrix over X (Pearson), corrected ---------------------------------------------------
keys = [("Acc", "acc"), ("CS", "CS_blur"), ("SI", "SI_blur"), ("GG", "GG"), ("SR", "SR"), ("GS", "GS")]
M = np.array([[r[k] for _, k in keys] for r in X]); R = np.corrcoef(M.T)
T = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule", " & " + " & ".join(n for n, _ in keys) + r" \\", r"\midrule"]
for i, (n, _) in enumerate(keys): T.append(n + " & " + " & ".join(f"{R[i, j]:+.2f}" for j in range(len(keys))) + r" \\")
T += [r"\bottomrule", r"\end{tabular}"]
(TABLES / "tab_metric_correlations.tex").write_text("\n".join(T) + "\n")

rng_ = lambda S, k: (min(r[k] for r in S), max(r[k] for r in S))
out = dict(
    sets=dict(H=OPEN, X=OPEN + ["Claude-Sonnet-4.5"], excluded="Gemini-2.5-Pro (preliminary)"),
    molmo=rows["Molmo-7B-D"],
    H_ranges={k: rng_(H, k) for k in ("acc", "CS_blur", "SI_blur", "GG", "GS")},
    H_fragility_share=(1 - max(r["GS"] for r in H), 1 - min(r["GS"] for r in H)),
    X_corr={f"{a}~{b}": float(R[i, j]) for i, (a, _) in enumerate(keys) for j, (b, _) in enumerate(keys) if i < j},
    X_cs_gray_gt_blur=[r["model"] for r in X if r["CS_gray"] > r["CS_blur"]],
    X_cs_gray_gt_blur_count=f"{sum(r['CS_gray'] > r['CS_blur'] for r in X)}/{len(X)}",
    gemini_cs_gray_gt_blur=G["CS_gray"] > G["CS_blur"],
    clean_label_accuracy={r["model"]: r["n_correct"] / 529 for r in X},
    claude=rows["Claude-Sonnet-4.5"], gemini=G,
    table_order=[r["model"] for r in sorted(X, key=lambda r: -r["GS"])])
json.dump(out, open(RESULTS / "main_numbers.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("H_ranges", "H_fragility_share", "X_cs_gray_gt_blur_count", "table_order")}, indent=0))
print("molmo:", {k: round(v, 4) if isinstance(v, float) else v for k, v in rows["Molmo-7B-D"].items() if k not in ("model", "source")})
print("corr X:", {k: round(v, 3) for k, v in out["X_corr"].items() if k in ("Acc~GS", "Acc~GG", "CS~GG", "GG~GS", "SI~SR")})
