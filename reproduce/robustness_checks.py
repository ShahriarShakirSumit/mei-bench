#!/usr/bin/env python
"""Robustness of the sham control + instance-mask evidence (ACML 2026).
Per model, on its re-run correct set C (paper's answers_match / answer_flipped; Molmo via its extractor):
  draws   : SI and GG under 6 independent random sham draws (re-run draw + sham_r1..r5): mean, SD, range
  salience: SI/GG with the saliency-matched sham
  position: SI/GG with the mirrored sham on its 545-item subset, vs mean random-draw SI on the SAME items
  texture : per item, the draw (6 random + salience) closest to the evidence box in (edge density, entropy),
            z-scored; SI/GG with that texture-matched sham
  mask    : CS/SI/GG with instance-mask evidence and translated-mask sham vs box CS/SI/GG on the same items
Writes results/robustness_checks.json and results/tables/tab_robustness_{main,supp}.tex"""
import json, sys, statistics as st
from pathlib import Path
import numpy as np
from repro_paths import RERUN_DATA, RERUN_PRED, ROBUSTNESS_DATA, ROBUSTNESS_OUT, MOLMO_FIX, RESULTS, TABLES
sys.path.insert(0, str(Path(__file__).parent))
from mei_benchmark.evaluation.metrics import answers_match, answer_flipped   # noqa: E402
from molmo_extract import extract as molmo_extract                          # noqa: E402  (module body guarded below)


RERUN = RERUN_PRED; OUT = ROBUSTNESS_OUT
MODELS = ["Qwen2-VL-7B", "InternVL2.5-8B", "InternVL2-8B", "LLaVA-NeXT-7B", "LLaVA-1.5-13B", "LLaVA-1.5-7B",
          "Phi-3.5-Vision", "Llama-3.2-11B-Vision", "DeepSeek-VL-7B", "Molmo-7B-D"]
DRAWS = ["sham_r1", "sham_r2", "sham_r3", "sham_r4", "sham_r5"]
items = {json.loads(l)["item_id"]: json.loads(l) for l in open(RERUN_DATA / "items.jsonl")}
meta = {r["item_id"]: r for r in json.load(open(ROBUSTNESS_DATA / "robustness_conditions_meta.json"))["items"]}

def answers(path, model):
    out = {}
    if not path.exists(): return out
    for l in open(path):
        if not l.strip(): continue
        r = json.loads(l); a = molmo_extract(r["raw_response"]) if model == "Molmo-7B-D" else r["answer"]
        out[(r["item_id"], r["intervention_type"])] = a
    return out

def complete(model):
    exp = {"sham_position": 615 + 545, "mask": 615 + 2 * 613}
    for c in DRAWS + ["sham_salience", "sham_position", "mask"]:
        f = OUT / c / "predictions" / f"{model}_predictions.jsonl"
        if not f.exists() or sum(1 for _ in open(f)) < exp.get(c, 1230): return False
    return True

def z(v, mu, sd): return (v - mu) / sd if sd > 0 else 0.0
tex_keys = ("edge_density", "entropy")
allv = {k: [meta[i][t][k] for i in meta for t in ["tex_evidence"] + [f"tex_{d}" for d in DRAWS] if meta[i].get(t)] for k in tex_keys}
mu = {k: st.mean(v) for k, v in allv.items()}; sd = {k: st.pstdev(v) for k, v in allv.items()}
def tdist(a, b): return sum(abs(z(a[k], mu[k], sd[k]) - z(b[k], mu[k], sd[k])) for k in tex_keys)

res = {}
for m in MODELS:
    if not complete(m): continue
    base = answers(RERUN / f"{m}_predictions.jsonl", m) if m != "Molmo-7B-D" else answers(MOLMO_FIX / "Molmo-7B-D_predictions.jsonl", m)
    cond = {c: answers(OUT / c / "predictions" / f"{m}_predictions.jsonl", m) for c in DRAWS + ["sham_salience", "sham_position", "mask"]}
    C = [i for i in items if (i, "original") in base and answers_match(base[(i, "original")], items[i]["valid_answers"])
         and (i, "evidence_blur") in base and (i, "sham_blur") in base]
    o = {i: base[(i, "original")] for i in C}
    ev = {i: answer_flipped(o[i], base[(i, "evidence_blur")]) for i in C}
    CS = float(np.mean([ev[i] for i in C]))
    sh = {"draw0": {i: answer_flipped(o[i], base[(i, "sham_blur")]) for i in C}}
    for d in DRAWS: sh[d] = {i: answer_flipped(o[i], cond[d][(i, "sham_blur")]) for i in C if (i, "sham_blur") in cond[d]}
    sh["salience"] = {i: answer_flipped(o[i], cond["sham_salience"][(i, "sham_blur")]) for i in C if (i, "sham_blur") in cond["sham_salience"]}
    SI = {k: float(np.mean(list(v.values()))) for k, v in sh.items()}
    rnd = ["draw0"] + DRAWS; si_r = [SI[k] for k in rnd]
    # position (subset where a valid mirror exists) vs the random draws on the same items
    P = [i for i in C if (i, "sham_blur") in cond["sham_position"]]
    si_pos = float(np.mean([answer_flipped(o[i], cond["sham_position"][(i, "sham_blur")]) for i in P]))
    si_rnd_P = float(np.mean([np.mean([sh[k][i] for k in rnd]) for i in P])); cs_P = float(np.mean([ev[i] for i in P]))
    # texture-matched sham: per item the draw (6 random + salience) closest to the evidence box
    tm = []
    for i in C:
        cands = [(tdist(meta[i]["tex_evidence"], meta[i][f"tex_{d}"] if d != "draw0" else meta[i]["tex_orig_sham"]), d) for d in rnd]
        if meta[i].get("tex_sham_salience"): cands.append((tdist(meta[i]["tex_evidence"], meta[i]["tex_sham_salience"]), "salience"))
        tm.append(sh[min(cands)[1]][i])
    si_tex = float(np.mean(tm))
    gap_tex = [tdist(meta[i]["tex_evidence"], meta[i]["tex_orig_sham"]) for i in C]
    # mask vs box on the same items
    M = [i for i in C if (i, "evidence_blur") in cond["mask"] and (i, "sham_blur") in cond["mask"]]
    cs_mask = float(np.mean([answer_flipped(o[i], cond["mask"][(i, "evidence_blur")]) for i in M]))
    si_mask = float(np.mean([answer_flipped(o[i], cond["mask"][(i, "sham_blur")]) for i in M]))
    cs_box = float(np.mean([ev[i] for i in M])); si_box = float(np.mean([np.mean([sh[k][i] for k in rnd]) for i in M]))
    gs = lambda cs, si: max(cs - si, 0) / max(cs, 1e-9)
    res[m] = dict(n_c=len(C), CS=CS, SI_draws=si_r, SI_mean=float(np.mean(si_r)), SI_sd=float(np.std(si_r, ddof=1)),
                  GG_mean=CS - float(np.mean(si_r)), GG_min=CS - max(si_r), GG_max=CS - min(si_r), GS_mean=gs(CS, float(np.mean(si_r))),
                  SI_salience=SI["salience"], GG_salience=CS - SI["salience"],
                  n_position=len(P), SI_position=si_pos, SI_random_on_P=si_rnd_P, GG_position=cs_P - si_pos, GG_random_on_P=cs_P - si_rnd_P,
                  SI_texture=si_tex, GG_texture=CS - si_tex, texture_gap_median=float(np.median(gap_tex)),
                  n_mask=len(M), CS_mask=cs_mask, SI_mask=si_mask, GG_mask=cs_mask - si_mask, CS_box=cs_box, SI_box=si_box,
                  GG_box=cs_box - si_box, GS_mask=gs(cs_mask, si_mask), GS_box=gs(cs_box, si_box))
    r = res[m]
    print(f"{m:<22} n_c={r['n_c']:3d} CS={CS:.3f} | SI draws {min(si_r):.3f}-{max(si_r):.3f} (sd {r['SI_sd']:.3f}) GG {r['GG_min']:.3f}-{r['GG_max']:.3f} "
          f"| sal SI {r['SI_salience']:.3f} | pos SI {si_pos:.3f} vs rnd {si_rnd_P:.3f} (n={len(P)}) | tex SI {si_tex:.3f} "
          f"| mask CS/SI {cs_mask:.3f}/{si_mask:.3f} GG {r['GG_mask']:.3f} vs box GG {r['GG_box']:.3f}")
json.dump(res, open(RESULTS / "robustness_checks.json", "w"), indent=1)
print(f"models complete: {len(res)}/{len(MODELS)}")

# ---- cross-model agreement + tables ------------------------------------------------------------------
from scipy.stats import spearmanr
ms = list(res)
def rho(a, b): return float(spearmanr([res[m][a] for m in ms], [res[m][b] for m in ms]).correlation) if len(ms) >= 3 else float("nan")
for m in ms:
    r = res[m]; r["GG_texture"] = r["CS"] - r["SI_texture"]; r["GS_salience"] = max(r["GG_salience"], 0) / r["CS"]
summary = dict(n_models=len(ms),
    si_draw_sd=(min(res[m]["SI_sd"] for m in ms), max(res[m]["SI_sd"] for m in ms)),
    gg_min_over_draws=min(res[m]["GG_min"] for m in ms),
    gg_salience=(min(res[m]["GG_salience"] for m in ms), max(res[m]["GG_salience"] for m in ms)),
    gg_position=(min(res[m]["GG_position"] for m in ms), max(res[m]["GG_position"] for m in ms)),
    gg_texture=(min(res[m]["GG_texture"] for m in ms), max(res[m]["GG_texture"] for m in ms)),
    gg_mask=(min(res[m]["GG_mask"] for m in ms), max(res[m]["GG_mask"] for m in ms)),
    gg_box_same_items=(min(res[m]["GG_box"] for m in ms), max(res[m]["GG_box"] for m in ms)),
    si_increase_salience=(min(res[m]["SI_salience"] - res[m]["SI_mean"] for m in ms), max(res[m]["SI_salience"] - res[m]["SI_mean"] for m in ms)),
    si_increase_position=(min(res[m]["SI_position"] - res[m]["SI_random_on_P"] for m in ms), max(res[m]["SI_position"] - res[m]["SI_random_on_P"] for m in ms)),
    rho_GG_random_vs_salience=rho("GG_mean", "GG_salience"), rho_GG_random_vs_texture=rho("GG_mean", "GG_texture"),
    rho_GG_random_vs_position=rho("GG_random_on_P", "GG_position"), rho_GG_box_vs_mask=rho("GG_box", "GG_mask"),
    rho_GS_box_vs_mask=rho("GS_box", "GS_mask"), models_mask_gg_positive=sum(res[m]["GG_mask"] > 0 for m in ms))
json.dump(dict(per_model=res, summary=summary), open(RESULTS / "robustness_checks.json", "w"), indent=1)
print("SUMMARY:", json.dumps({k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v] if isinstance(v, tuple) else v) for k, v in summary.items()}, indent=0))
f3 = lambda v: (f"{v:.3f}".replace("0.", ".", 1) if v >= 0 else "$-$" + f"{abs(v):.3f}".replace("0.", ".", 1))
T = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
     r" & & \multicolumn{2}{c}{Random shams (6 draws)} & Saliency & Texture & \multicolumn{2}{c}{Mask vs box} \\",
     r"Model & $\CS$ & $\GG$ mean [min, max] & SD($\SI$) & $\GG$ & $\GG$ & $\GG_{\mathrm{mask}}$ & $\GG_{\mathrm{box}}$ \\", r"\midrule"]
for m in sorted(ms, key=lambda m: -res[m]["GG_mean"]):
    r = res[m]
    T.append(f"{m} & {f3(r['CS'])} & {f3(r['GG_mean'])} [{f3(r['GG_min'])}, {f3(r['GG_max'])}] & {f3(r['SI_sd'])} & "
             f"{f3(r['GG_salience'])} & {f3(r['GG_texture'])} & {f3(r['GG_mask'])} & {f3(r['GG_box'])} \\\\")
T += [r"\midrule", f"Spearman $\\rho$ with random-sham $\\GG$ & & & & {summary['rho_GG_random_vs_salience']:.2f} & {summary['rho_GG_random_vs_texture']:.2f} & "
      f"\\multicolumn{{2}}{{c}}{{{summary['rho_GG_box_vs_mask']:.2f} (mask vs box)}} \\\\", r"\bottomrule", r"\end{tabular}"]
(TABLES / "tab_robustness_supp.tex").write_text("\n".join(T) + "\n")
P = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Model & $n$ & $\GG$ random shams & $\GG$ position-matched \\", r"\midrule"]
for m in sorted(ms, key=lambda m: -res[m]["GG_mean"]):
    r = res[m]; P.append(f"{m} & {r['n_position']} & {f3(r['GG_random_on_P'])} & {f3(r['GG_position'])} \\\\")
P += [r"\bottomrule", r"\end{tabular}"]; (TABLES / "tab_robustness_position.tex").write_text("\n".join(P) + "\n")
