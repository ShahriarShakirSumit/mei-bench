#!/usr/bin/env python
"""Cross-model comparability analysis (ACML 2026, Appendix).

Reads the re-run predictions and reports, per model:
  * own-subset metrics (acc, n_c, CS, SI, GG, GS) -- same formulas as
    mei_benchmark.evaluation.metrics (blur operator, correct set C_m);
  * the same metrics on the INTERSECTION of correct sets across all completed
    open-weight models with n_c >= 30;
  * GS by item-difficulty tercile (difficulty = fraction of completed models
    that answer the original correctly);
  * Spearman rho between the own-subset GS ranking and the intersection ranking.
Also prints the paper's published values next to the re-run for a reproduction check.

Usage: python intersection_analysis.py [--pred-dir DIR] [--data-dir DIR] [--out FILE]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

from repro_paths import RERUN_DATA, RERUN_PRED, ORIGINAL_SUMMARY, RESULTS
from mei_benchmark.data.loader import MEIDatasetLoader, PredictionLoader          # noqa: E402
from mei_benchmark.data.schema import InterventionType                        # noqa: E402
from mei_benchmark.evaluation.metrics import answers_match, answer_flipped    # noqa: E402

PAPER_SUMMARY = ORIGINAL_SUMMARY
MODELS = ["Qwen2-VL-7B", "InternVL2.5-8B", "InternVL2-8B", "LLaVA-NeXT-7B", "LLaVA-1.5-13B",
          "LLaVA-1.5-7B", "Phi-3.5-Vision", "Llama-3.2-11B-Vision", "DeepSeek-VL-7B", "Molmo-7B-D"]


def per_item(items, psets):
    """item_id -> (correct, flip_blur, flip_sham_blur) ; None entries when a variant is missing."""
    idx = {p.item_id: p for p in psets}
    out = {}
    for it in items:
        ps = idx.get(it.item_id)
        if ps is None or ps.original_prediction is None:
            continue
        o = ps.original_prediction.answer
        eb = ps.get_prediction(InterventionType.EVIDENCE_BLUR)
        sb = ps.get_prediction(InterventionType.SHAM_BLUR)
        out[it.item_id] = (answers_match(o, it.valid_answers),
                           None if eb is None else answer_flipped(o, eb.answer),
                           None if sb is None else answer_flipped(o, sb.answer))
    return out


def metrics_on(pi, subset):
    """CS/SI/GG/GS over an explicit item subset (all must be correct on original)."""
    cs = [pi[i][1] for i in subset if pi[i][1] is not None]
    si = [pi[i][2] for i in subset if pi[i][2] is not None]
    if not cs or not si:
        return None
    CS, SI = float(np.mean(cs)), float(np.mean(si))
    GG = CS - SI
    return dict(n=len(subset), CS=CS, SI=SI, GG=GG, GS=max(GG, 0) / max(CS, 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=str(RERUN_PRED))
    ap.add_argument("--data-dir", default=str(RERUN_DATA))
    ap.add_argument("--out", default=str(RESULTS / "intersection_result.json"))
    ap.add_argument("--min-nc", type=int, default=30)
    a = ap.parse_args()

    items = MEIDatasetLoader(a.data_dir).load_items()
    n_items = len(items)
    pl = PredictionLoader(a.pred_dir)
    paper = {m["model"]: m for m in json.load(open(PAPER_SUMMARY))["models"]}

    per, own = {}, {}
    for m in MODELS:
        f = Path(a.pred_dir) / f"{m}_predictions.jsonl"
        if not f.exists():
            print(f"[skip] {m}: no predictions"); continue
        psets = pl.load_predictions(m)
        pi = per_item(items, psets)
        if len(pi) < n_items:
            print(f"[incomplete] {m}: {len(pi)}/{n_items} items with an original prediction")
        C = [i for i, v in pi.items() if v[0]]
        per[m] = pi
        own[m] = dict(n_items=len(pi), acc=len(C) / max(len(pi), 1), n_c=len(C),
                      **(metrics_on(pi, C) or {}))

    # ---- reproduction check vs the published table --------------------------
    print("\n== own-subset metrics: re-run vs paper (acc / n_c / CS / SI / GG / GS) ==")
    for m, o in own.items():
        p = paper.get(m, {})
        if "CS" not in o:
            print(f"{m:22s} rerun acc={o['acc']:.3f} n_c={o['n_c']}  (no flips computable)"); continue
        print(f"{m:22s} rerun {o['acc']:.3f}/{o['n_c']:4d}/{o['CS']:.3f}/{o['SI']:.3f}/{o['GG']:.3f}/{o['GS']:.3f}"
              f"   paper {p.get('acc',float('nan')):.3f}/{p.get('n_correct',0):4d}/{p.get('CS_blur',float('nan')):.3f}/"
              f"{p.get('SI_blur',float('nan')):.3f}/{p.get('GG',float('nan')):.3f}/{p.get('GS',float('nan')):.3f}")

    # ---- reproduction expressed in ITEMS (decimals hide how small the drift is) --
    print("\n== reproduction in item counts (re-run vs paper): correct set / CS flips / sham flips ==")
    repro = {}
    for m, o in own.items():
        p = paper.get(m, {})
        if "CS" not in o or o["n_items"] < n_items or not p:
            continue
        C = [i for i, v in per[m].items() if v[0]]
        cs_f = sum(1 for i in C if per[m][i][1]); si_f = sum(1 for i in C if per[m][i][2])
        pc = p["n_correct"]; p_cs = round(p["CS_blur"] * pc); p_si = round(p["SI_blur"] * pc)
        repro[m] = dict(n_c=len(C), n_c_paper=pc, cs_flips=cs_f, cs_flips_paper=p_cs,
                        si_flips=si_f, si_flips_paper=p_si)
        print(f"{m:22s} n_c {len(C):4d}/{pc:<4d} ({len(C)-pc:+d})   CS {cs_f:3d}/{p_cs:<3d} ({cs_f-p_cs:+d})"
              f"   sham {si_f:3d}/{p_si:<3d} ({si_f-p_si:+d})")
    if repro:
        mx = lambda k, kp: max(abs(v[k] - v[kp]) for v in repro.values())
        print(f"max |delta| over {len(repro)} models: correct set {mx('n_c','n_c_paper')} items, "
              f"CS flips {mx('cs_flips','cs_flips_paper')}, sham flips {mx('si_flips','si_flips_paper')} "
              f"(of {n_items} items)")

    # ---- intersection over eligible models ------------------------------------
    elig = [m for m, o in own.items() if o["n_c"] >= a.min_nc]
    inter = None
    for m in elig:
        C = {i for i, v in per[m].items() if v[0]}
        inter = C if inter is None else inter & C
    inter = sorted(inter or [])
    print(f"\n== intersection of correct sets over {len(elig)} models (n_c>={a.min_nc}): |C_cap| = {len(inter)} ==")
    inter_m = {m: metrics_on(per[m], inter) for m in elig} if inter else {}
    for m in elig:
        r = inter_m.get(m)
        if r:
            nb = sum(1 for i in inter if per[m][i][1]); ns = sum(1 for i in inter if per[m][i][2])
            r["cs_flips"], r["si_flips"] = nb, ns
            print(f"{m:22s} own GS={own[m]['GS']:.3f} (n={own[m]['n_c']})   intersection GS={r['GS']:.3f} "
                  f"CS={r['CS']:.3f} SI={r['SI']:.3f} GG={r['GG']:.3f}  [flips {nb} evidence / {ns} sham of {len(inter)}]")

    rho = None
    if len(elig) >= 3 and inter:
        from scipy.stats import spearmanr
        rho = float(spearmanr([own[m]["GS"] for m in elig], [inter_m[m]["GS"] for m in elig]).correlation)
        rho_gg = float(spearmanr([own[m]["GG"] for m in elig], [inter_m[m]["GG"] for m in elig]).correlation)
        print(f"Spearman rho own-vs-intersection: GS {rho:.3f}, GG {rho_gg:.3f}")

    # ---- difficulty terciles ---------------------------------------------------
    terc = {}
    if len(elig) >= 3:
        diff = {it.item_id: np.mean([per[m].get(it.item_id, (False,))[0] for m in elig]) for it in items}
        vals = np.array(list(diff.values()))
        q1, q2 = np.quantile(vals, [1/3, 2/3])
        def bin_of(d): return "hard" if d <= q1 else ("medium" if d <= q2 else "easy")
        bins = {b: [i for i, d in diff.items() if bin_of(d) == b] for b in ("hard", "medium", "easy")}
        print(f"\n== GS by item-difficulty tercile (difficulty = fraction of {len(elig)} models correct; cuts {q1:.2f}/{q2:.2f}) ==")
        print(f"tercile sizes: " + ", ".join(f"{b}={len(v)}" for b, v in bins.items()))
        for m in elig:
            row = {}
            for b, ids in bins.items():
                sub = [i for i in ids if i in per[m] and per[m][i][0]]
                r = metrics_on(per[m], sub) if sub else None
                row[b] = None if r is None else dict(n=len(sub), GS=r["GS"], GG=r["GG"])
            terc[m] = row
            print(f"{m:22s} " + "  ".join(f"{b}: GS={v['GS']:.3f} (n={v['n']})" if v else f"{b}: --" for b, v in row.items()))
        from scipy.stats import spearmanr as _sp
        for b in ("easy", "medium"):
            ok = [m for m in elig if terc[m].get(b)]
            if len(ok) >= 3:
                r = float(_sp([own[m]["GS"] for m in ok], [terc[m][b]["GS"] for m in ok]).correlation)
                print(f"Spearman rho own-subset GS vs {b}-tercile GS ({len(ok)} models): {r:.3f}")
                terc.setdefault("_spearman", {})[b] = r

    json.dump(dict(n_items=n_items, own=own, eligible=elig, intersection_size=len(inter),
                   intersection=inter_m, spearman_gs_own_vs_intersection=rho, terciles=terc,
                   reproduction=repro),
              open(a.out, "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
