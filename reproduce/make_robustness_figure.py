"""Poster/slide figure: GG per model under random shams (mean, min-max over 6 draws), saliency-, texture- and
position-matched shams, and instance-mask evidence. Data: outputs/paper/robustness_checks.json. Okabe-Ito palette."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
from repro_paths import RESULTS, FIGURES
R = json.load(open(RESULTS / "robustness_checks.json"))
res, S = R["per_model"], R["summary"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})
ms = sorted(res, key=lambda m: res[m]["GG_mean"]); y = np.arange(len(ms))
short = lambda m: m.replace("-Vision", "-V").replace("Llama-3.2-11B-V", "Llama-3.2-11B")
fig, ax = plt.subplots(figsize=(8.2, 5.4))
ax.hlines(y, [res[m]["GG_min"] for m in ms], [res[m]["GG_max"] for m in ms], color="#0072B2", lw=5, alpha=0.30, zorder=1)
series = [("GG_mean", "Random shams (mean; bar = range over 6 draws)", "#0072B2", "o"), ("GG_salience", "Saliency-matched", "#E69F00", "D"),
          ("GG_texture", "Texture-matched", "#009E73", "s"), ("GG_position", "Position-matched", "#CC79A7", "^"),
          ("GG_mask", "Instance-mask evidence", "#D55E00", "X")]
for k, lab, c, mk in series:
    ax.scatter([res[m][k] for m in ms], y, color=c, marker=mk, s=58 if mk != "X" else 70, label=lab, zorder=3, edgecolor="white", linewidth=0.6)
ax.axvline(0, color="#444", lw=1); ax.set_yticks(y); ax.set_yticklabels([short(m) for m in ms])
ax.set_xlabel("Grounding Gap  GG = CS − SI  (blur)"); ax.set_xlim(-0.01, 0.31)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3, fontsize=9, frameon=False, columnspacing=1.2, handletextpad=0.3)
ax.set_title(f"GG stays positive under every sham choice (ordering: ρ = {S['rho_GG_random_vs_salience']:.2f} saliency, "
             f"{S['rho_GG_random_vs_texture']:.2f} texture, {S['rho_GG_random_vs_position']:.2f} position)", fontsize=10.5)
fig.savefig(FIGURES / "fig_robustness_gg.pdf")
print("wrote fig_robustness_gg.pdf")
