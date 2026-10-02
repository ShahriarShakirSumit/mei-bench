from __future__ import annotations
#!/usr/bin/env python3
"""Regenerate publication figures from the current 615-item, 12-model results.

Writes PDFs into paper/figures/ overwriting the stale March-dated copies.
Pulls per-model headline numbers from outputs/metrics/_summary.json and
per-task GG numbers from paper/auto/tab_stratified.tex (parsed) plus the
raw items + predictions for any item-level breakdowns.
"""
from mei_benchmark.paths import DATA_DIR, OUTPUT_DIR
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[1]
SUM = OUTPUT_DIR / "metrics" / "_summary.json"
FIG = ROOT / "paper" / "figures"
AUTO = ROOT / "paper" / "auto"
FIG.mkdir(parents=True, exist_ok=True)


def load_summary() -> list[dict]:
    return json.loads(SUM.read_text())["models"]


def color_for(model: str) -> str:
    if "Claude" in model: return "#cc6644"
    if "Gemini" in model: return "#4477aa"
    if "InternVL2.5" in model: return "#117733"
    if "InternVL2-" in model: return "#44aa99"
    if "LLaVA-NeXT" in model: return "#88ccee"
    if "LLaVA-1.5-13" in model: return "#332288"
    if "LLaVA-1.5-7" in model: return "#6677aa"
    if "Qwen2-VL" in model: return "#ddcc77"
    if "Phi" in model: return "#aa4499"
    if "Llama" in model: return "#882255"
    if "DeepSeek" in model: return "#999933"
    if "Molmo" in model: return "#888888"
    return "#444444"


def short_name(model: str) -> str:
    return model.replace("Claude-Sonnet-4.5", "Claude-Sonnet-4.5")\
                .replace("Gemini-2.5-Pro", "Gemini-2.5-Pro")\
                .replace("InternVL2.5-8B", "InternVL2.5-8B")\
                .replace("Llama-3.2-11B-Vision", "Llama-3.2-11B")\
                .replace("Phi-3.5-Vision", "Phi-3.5-V")


def is_closed(m: str) -> bool:
    return "Claude" in m or "Gemini" in m


# --------------------------------------------------------------------------
def _numbered_scatter(models, xkey, ykey, xlabel, ylabel, title, outfile,
                     xlim=None, ylim=None, diag=False):
    """Scatter with numbered points and a side legend keyed by number."""
    rows = [m for m in models if m["model"] != "Molmo-7B-D"]
    rows = sorted(rows, key=lambda m: -m.get("GS", 0.0))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if diag:
        lim = max(xlim[1] if xlim else 0.6, ylim[1] if ylim else 0.6)
        ax.plot([0, lim], [0, lim], color="gray", linestyle="--",
                linewidth=0.8, zorder=1, label="GG = 0")
    handles = []
    for i, m in enumerate(rows, 1):
        x = m[xkey] if xkey != "SR" else 1 - m["SI_blur"]
        y = m[ykey] if ykey != "SR" else 1 - m["SI_blur"]
        col = color_for(m["model"])
        mk = "*" if is_closed(m["model"]) else "o"
        sz = 140 if is_closed(m["model"]) else 90
        ax.scatter(x, y, s=sz, color=col, marker=mk,
                   edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(str(i), (x, y), ha="center", va="center",
                    fontsize=7, color="white" if not is_closed(m["model"]) else "black",
                    fontweight="bold", zorder=4)
        handles.append((i, m["model"], col, mk))
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Side legend with numbered keys
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker=mk, linestyle="", color=col,
               markeredgecolor="black", markersize=10 if mk == "*" else 8,
               label=f"{i}. {short_name(name)}")
        for i, name, col, mk in handles
    ]
    if diag:
        legend_handles.append(Line2D([0], [0], color="gray", linestyle="--",
                                     linewidth=0.8, label="GG = 0"))
    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False,
              handletextpad=0.4, borderaxespad=0.0)
    fig.savefig(FIG / outfile)
    plt.close(fig)


def fig_grounding_scatter(models):
    """CS vs SI scatter with the GG=0 diagonal."""
    _numbered_scatter(
        models, xkey="SI_blur", ykey="CS_blur",
        xlabel="Sham Sensitivity (SI, lower is better)",
        ylabel="Causal Sensitivity (CS, higher is better)",
        title="Causal vs Sham sensitivity (blur)",
        outfile="fig_grounding_scatter.pdf",
        xlim=(0, 0.18), ylim=(0, 0.55), diag=True,
    )


def fig_model_comparison(models):
    """Grouped bar chart: CS, SI, GG, GS per model (sorted by GS)."""
    ms = sorted([m for m in models if m["model"] != "Molmo-7B-D"],
                key=lambda x: -x["GS"])
    names = [short_name(m["model"]) for m in ms]
    cs = [m["CS_blur"] for m in ms]
    si = [m["SI_blur"] for m in ms]
    gg = [m["GG"] for m in ms]
    gs = [m["GS"] for m in ms]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(ms)); w = 0.2
    ax.bar(x - 1.5 * w, cs, w, label="CS", color="#4477aa")
    ax.bar(x - 0.5 * w, si, w, label="SI", color="#cc6644")
    ax.bar(x + 0.5 * w, gg, w, label="GG", color="#117733")
    ax.bar(x + 1.5 * w, gs, w, label="GS", color="#aa4499")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Metric value")
    ax.set_title("Model comparison across MEI metrics (blur)")
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.0)
    fig.savefig(FIG / "fig_model_comparison.pdf")
    plt.close(fig)


def fig_cs_vs_gs(models):
    _numbered_scatter(
        models, xkey="CS_blur", ykey="GS",
        xlabel="Causal Sensitivity (CS)",
        ylabel="Grounding Specificity (GS)",
        title="CS vs GS: high sensitivity does not guarantee specificity",
        outfile="fig_cs_vs_gs.pdf",
        xlim=(0, 0.55), ylim=(0, 1.0), diag=False,
    )


def fig_intervention_comparison(models):
    """CS_blur vs CS_gray vs CS_swap per model."""
    ms = sorted([m for m in models if m["model"] != "Molmo-7B-D"],
                key=lambda x: -x["CS_blur"])
    names = [short_name(m["model"]) for m in ms]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(ms)); w = 0.27
    ax.bar(x - w, [m["CS_blur"] for m in ms], w, label="CS$_\\mathrm{blur}$", color="#4477aa")
    ax.bar(x,     [m["CS_gray"] for m in ms], w, label="CS$_\\mathrm{gray}$", color="#117733")
    ax.bar(x + w, [m["CS_swap"] for m in ms], w, label="CS$_\\mathrm{swap}$", color="#aa4499")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Causal Sensitivity")
    ax.set_title("CS by intervention operator")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(FIG / "fig_intervention_comparison.pdf")
    plt.close(fig)


def fig_metric_correlation(models):
    rows = [m for m in models if m["model"] not in ("Molmo-7B-D",)]
    M = np.array([[m["CS_blur"], m["SI_blur"], m["GG"],
                   1 - m["SI_blur"], m["GS"]] for m in rows])
    labels = ["CS", "SI", "GG", "SR", "GS"]
    R = np.corrcoef(M.T)
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels)
    ax.set_yticks(range(5)); ax.set_yticklabels(labels)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(R[i, j]) > 0.55 else "black",
                    fontsize=8)
    ax.set_title(f"Metric correlations (n={len(rows)} models)")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.savefig(FIG / "fig_metric_correlation.pdf")
    plt.close(fig)


# Per-task ----------------------------------------------------------------
def parse_stratified() -> dict[str, dict[str, float]]:
    txt = (AUTO / "tab_stratified.tex").read_text()
    out = {}
    for line in txt.splitlines():
        if "&" not in line or "\\toprule" in line or "\\bottomrule" in line:
            continue
        parts = [p.strip() for p in line.split("&")]
        if len(parts) != 6 or parts[0].startswith("Model"):
            continue
        name = parts[0]
        vals = [float(p.replace("\\\\", "").strip()) for p in parts[1:]]
        out[name] = dict(zip(["ObjID", "Attr", "Spatial", "Count", "Text"], vals))
    return out


def fig_gg_heatmap(models):
    strat = parse_stratified()
    order = sorted([m["model"] for m in models if m["model"] in strat
                    and m["model"] != "Molmo-7B-D"],
                   key=lambda n: -next(mm["GS"] for mm in models if mm["model"] == n))
    tasks = ["ObjID", "Attr", "Spatial", "Count", "Text"]
    M = np.array([[strat[m][t] for t in tasks] for m in order])
    fig, ax = plt.subplots(figsize=(5.5, max(2.8, 0.32 * len(order) + 1.2)))
    im = ax.imshow(M, cmap="viridis", vmin=-0.05, vmax=0.65, aspect="auto")
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([short_name(m) for m in order])
    for i, m in enumerate(order):
        for j, t in enumerate(tasks):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.3 else "black", fontsize=7)
    ax.set_title("Grounding Gap by task (blur)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="GG")
    fig.savefig(FIG / "fig_gg_heatmap.pdf")
    plt.close(fig)


def fig_task_radar(models):
    strat = parse_stratified()
    tasks = ["ObjID", "Attr", "Spatial", "Count", "Text"]
    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    fig, ax = plt.subplots(figsize=(5.0, 5.0), subplot_kw=dict(polar=True))
    highlight = ["InternVL2.5-8B", "Claude-Sonnet-4.5",
                 "Gemini-2.5-Pro", "LLaVA-1.5-7B"]
    for m in highlight:
        if m not in strat:
            continue
        vals = [max(strat[m][t], 0) for t in tasks]
        vals.append(vals[0])
        ax.plot(angles, vals, color=color_for(m), label=short_name(m), linewidth=1.4)
        ax.fill(angles, vals, color=color_for(m), alpha=0.10)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(tasks)
    ax.set_yticks([0.1, 0.3, 0.5])
    ax.set_ylim(0, 0.65)
    ax.set_title("Per-task Grounding Gap (blur)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=7,
              frameon=False)
    fig.savefig(FIG / "fig_task_radar.pdf")
    plt.close(fig)


# Item-level: area-binned figures -----------------------------------------
def load_item_flips(model: str):
    """Return list of (area_fraction, ev_flip, sh_flip) on correct items."""
    import sys
    sys.path.insert(0, str(ROOT))
    from mei_benchmark.data.loader import MEIDatasetLoader, PredictionLoader
    from mei_benchmark.evaluation.metrics import answers_match, answer_flipped
    DATA = DATA_DIR
    PRED_DIR = OUTPUT_DIR / "predictions"
    if not (PRED_DIR / f"{model}_predictions.jsonl").exists():
        return []
    items = MEIDatasetLoader(DATA).load_items()
    pred_sets = PredictionLoader(PRED_DIR).load_predictions(model)
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
        ev = ans.get("evidence_blur"); sh = ans.get("sham_blur")
        if ev is None or sh is None: continue
        out.append((it.evidence_region.area_fraction,
                    int(answer_flipped(orig, ev)),
                    int(answer_flipped(orig, sh))))
    return out


def fig_area_effects():
    """GG vs evidence area fraction, pooling 3 representative models."""
    models = ["InternVL2.5-8B", "Claude-Sonnet-4.5", "Qwen2-VL-7B"]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    bins = np.array([0.005, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0])
    centers = (bins[:-1] + bins[1:]) / 2
    for mname in models:
        flips = load_item_flips(mname)
        if not flips:
            continue
        af = np.array([f[0] for f in flips])
        ev = np.array([f[1] for f in flips])
        sh = np.array([f[2] for f in flips])
        gg_per = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (af >= lo) & (af < hi)
            if mask.sum() < 5:
                gg_per.append(np.nan)
            else:
                gg_per.append(ev[mask].mean() - sh[mask].mean())
        ax.plot(centers, gg_per, marker="o", color=color_for(mname),
                label=short_name(mname), linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("Evidence area fraction (log)")
    ax.set_ylabel("Grounding Gap (binned)")
    ax.set_title("GG vs evidence-region size")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "fig_area_effects.pdf")
    plt.close(fig)


def fig_area_bar_chart():
    """Per-area-bin CS and SI for a single representative model."""
    flips = load_item_flips("InternVL2.5-8B")
    bins = np.array([0.005, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0])
    af = np.array([f[0] for f in flips])
    ev = np.array([f[1] for f in flips])
    sh = np.array([f[2] for f in flips])
    cs = []; si = []; counts = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (af >= lo) & (af < hi)
        counts.append(mask.sum())
        if mask.sum() == 0:
            cs.append(0); si.append(0); continue
        cs.append(ev[mask].mean()); si.append(sh[mask].mean())
    labels = [f"[{lo:.3f},{hi:.2f})" for lo, hi in zip(bins[:-1], bins[1:])]
    x = np.arange(len(labels)); w = 0.4
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(x - w/2, cs, w, label="CS", color="#4477aa")
    ax.bar(x + w/2, si, w, label="SI", color="#cc6644")
    for i, c in enumerate(counts):
        ax.text(i, max(cs[i], si[i]) + 0.02, f"n={c}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Flip rate")
    ax.set_title("InternVL2.5-8B: CS, SI by evidence area bin")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "fig_area_bar_chart.pdf")
    plt.close(fig)


def fig_dose_response(models):
    """Pseudo-dose-response: CS_blur vs CS_gray vs CS_swap for top 6 models.
    Treats {original, blur, gray, swap} as increasing-severity intervention
    dose levels and plots CS at each level. This matches the v1 figure used
    in the paper, now with 12 models available."""
    top = sorted([m for m in models if m["model"] != "Molmo-7B-D"],
                 key=lambda x: -x["GS"])[:6]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    xs = ["original", "blur", "gray", "swap"]
    for m in top:
        ys = [0.0, m["CS_blur"], m["CS_gray"], m["CS_swap"]]
        ax.plot(xs, ys, marker="o", color=color_for(m["model"]),
                label=short_name(m["model"]), linewidth=1.4)
    ax.set_ylabel("Causal Sensitivity")
    ax.set_xlabel("Intervention operator (increasing severity)")
    ax.set_title("Dose response: CS by intervention operator")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(FIG / "fig_dose_response.pdf")
    plt.close(fig)


def main():
    models = load_summary()
    fig_grounding_scatter(models)
    fig_model_comparison(models)
    fig_cs_vs_gs(models)
    fig_intervention_comparison(models)
    fig_metric_correlation(models)
    fig_gg_heatmap(models)
    fig_task_radar(models)
    fig_dose_response(models)
    fig_area_effects()
    fig_area_bar_chart()
    print("wrote figures to", FIG)


if __name__ == "__main__":
    main()
