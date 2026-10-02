#!/usr/bin/env python
"""Figures 2a, 2b, 4 and the CS-vs-GS panel of the paper.
Reuses the paper's plotting helpers/style (scripts/regen_figures.py) unchanged; differences, all
made for the final paper: Molmo-7B-D is plotted with its fixed values (it was dropped by name before),
Gemini-2.5-Pro (preliminary) is excluded from comparisons, and accuracy is added to the correlation matrix.
Model set = X of make_main_tables.py. Writes into outputs/paper/figures/."""
import json, sys
from pathlib import Path
import numpy as np
from repro_paths import ORIGINAL_SUMMARY, RESULTS, FIGURES
import mei_benchmark.scripts.regen_figures as RF          # style + helpers only
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIG = FIGURES
N = json.load(open(RESULTS / "main_numbers.json"))
SUB = {m["model"]: m for m in json.load(open(ORIGINAL_SUMMARY))["models"]}
SUB["Molmo-7B-D"] = N["molmo"]
X = [SUB[m] for m in N["sets"]["X"]]

def scatter(rows, xkey, ykey, xlabel, ylabel, title, outfile, xlim, ylim, diag):
    rows = sorted(rows, key=lambda m: -m["GS"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2)); handles = []
    if diag:
        lim = max(xlim[1], ylim[1]); ax.plot([0, lim], [0, lim], color="gray", ls="--", lw=0.8, zorder=1)
    for i, m in enumerate(rows, 1):
        col = RF.color_for(m["model"]); mk = "*" if RF.is_closed(m["model"]) else "o"
        ax.scatter(m[xkey], m[ykey], s=140 if mk == "*" else 90, color=col, marker=mk, edgecolor="black", lw=0.6, zorder=3)
        ax.annotate(str(i), (m[xkey], m[ykey]), ha="center", va="center", fontsize=7,
                    color="black" if mk == "*" else "white", fontweight="bold", zorder=4)
        handles.append(Line2D([0], [0], marker=mk, ls="", color=col, markeredgecolor="black",
                              markersize=10 if mk == "*" else 8, label=f"{i}. {RF.short_name(m['model'])}"))
    if diag: handles.append(Line2D([0], [0], color="gray", ls="--", lw=0.8, label="GG = 0"))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False,
              handletextpad=0.4, borderaxespad=0.0)
    fig.savefig(FIG / outfile); plt.close(fig)

scatter(X, "SI_blur", "CS_blur", "Sham Sensitivity (SI, lower is better)", "Causal Sensitivity (CS, higher is better)",
        "Causal vs Sham sensitivity (blur)", "fig_grounding_scatter.pdf", (0, 0.18), (0, 0.55), True)
scatter(X, "CS_blur", "GS", "Causal Sensitivity (CS)", "Grounding Specificity (GS)",
        "CS vs GS: high sensitivity does not guarantee specificity", "fig_cs_vs_gs.pdf", (0, 0.55), (0, 1.0), False)

ms = sorted(X, key=lambda m: -m["GS"]); x = np.arange(len(ms)); w = 0.2
fig, ax = plt.subplots(figsize=(7.0, 3.6))
for off, k, lab, col in ((-1.5, "CS_blur", "CS", "#4477aa"), (-0.5, "SI_blur", "SI", "#cc6644"),
                         (0.5, "GG", "GG", "#117733"), (1.5, "GS", "GS", "#aa4499")):
    ax.bar(x + off * w, [m[k] for m in ms], w, label=lab, color=col)
ax.set_xticks(x); ax.set_xticklabels([RF.short_name(m["model"]) for m in ms], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Metric value"); ax.set_title("Model comparison across MEI metrics (blur)")
ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper right"); ax.set_ylim(0, 1.0)
fig.savefig(FIG / "fig_model_comparison.pdf"); plt.close(fig)

lab = ["Acc", "CS", "SI", "GG", "GS"]          # SR omitted: SR = 1 - SI (stated in the paper)
M = np.array([[m["acc"], m["CS_blur"], m["SI_blur"], m["GG"], m["GS"]] for m in X]); R = np.corrcoef(M.T)
fig, ax = plt.subplots(figsize=(4.2, 3.7)); im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(5)); ax.set_xticklabels(lab); ax.set_yticks(range(5)); ax.set_yticklabels(lab)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(R[i, j]) > 0.55 else "black")
ax.set_title(f"Metric correlations (n={len(X)} models)"); fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
fig.savefig(FIG / "fig_metric_correlation.pdf"); plt.close(fig)
print("wrote figures:", [p.name for p in FIG.glob("fig_*") if p.stat().st_mtime > __import__('time').time() - 120])
