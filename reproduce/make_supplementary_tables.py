"""Generate the supplementary's new tables from result files (no hand-typed numbers).
Inputs: reproduce/inputs/rerun_vs_original.json (reproduction, 9 models), intersection_10open.json,
clean_label_subset.json. Outputs: outputs/paper/tables/tab_{reproduction,intersection,terciles,clean_label}.tex"""
import json
from pathlib import Path
from repro_paths import REPRODUCTION, RESULTS, TABLES
REP = json.load(open(REPRODUCTION))["reproduction"]
INT = json.load(open(RESULTS / "intersection_10open.json"))
CLN = json.load(open(RESULTS / "clean_label_subset.json"))
f3 = lambda v: f"{v:.3f}".replace("0.", ".", 1)
d = lambda a, b: f"{a}/{b}" + ("" if a == b else f" ({a-b:+d})")
T = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Model & Correct set & Evidence flips & Sham flips \\", r"\midrule"]
for m, v in REP.items():
    T.append(f"{m} & {d(v['n_c'], v['n_c_paper'])} & {d(v['cs_flips'], v['cs_flips_paper'])} & {d(v['si_flips'], v['si_flips_paper'])} \\\\")
T += [r"\bottomrule", r"\end{tabular}"]; (TABLES / "tab_reproduction.tex").write_text("\n".join(T) + "\n")
own = INT["own"]; inter = INT["intersection"]
T = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r"Model & own $\GS$ ($n_c$) & $\GG$ on $\mathcal{C}_\cap$ & $\GS$ on $\mathcal{C}_\cap$ & flips: evidence / sham \\", r"\midrule"]
for m in sorted(inter, key=lambda m: -own[m]["GS"]):
    r = inter[m]; T.append(f"{m} & {f3(own[m]['GS'])} ({own[m]['n_c']}) & {f3(r['GG'])} & {f3(r['GS'])} & {r['cs_flips']} / {r['si_flips']} \\\\")
T += [r"\bottomrule", r"\end{tabular}"]; (TABLES / "tab_intersection.tex").write_text("\n".join(T) + "\n")
ter = INT["terciles"]; sp = ter.get("_spearman", {})
T = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Model & own $\GS$ & middle tercile $\GS$ ($n$) & easiest tercile $\GS$ ($n$) \\", r"\midrule"]
for m in sorted([k for k in ter if not k.startswith("_")], key=lambda m: -own[m]["GS"]):
    c = lambda b: f"{f3(ter[m][b]['GS'])} ({ter[m][b]['n']})" if ter[m].get(b) else "--"
    T.append(f"{m} & {f3(own[m]['GS'])} & {c('medium')} & {c('easy')} \\\\")
T += [r"\midrule", f"Spearman $\\rho$ with own $\\GS$ & & {sp['medium']:.2f} & {sp['easy']:.2f} \\\\", r"\bottomrule", r"\end{tabular}"]
(TABLES / "tab_terciles.tex").write_text("\n".join(T) + "\n")
T = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule", r" & \multicolumn{3}{c}{All items} & \multicolumn{3}{c}{Clean-label subset} \\",
     r"Model & $n_c$ & $\GG$ & $\GS$ & $n_c$ & $\GG$ & $\GS$ \\", r"\midrule"]
for m, v in sorted(CLN.items(), key=lambda kv: -kv[1]["all"]["GS"]):
    a, c = v["all"], v["clean"]; T.append(f"{m} & {a['n']} & {f3(a['GG'])} & {f3(a['GS'])} & {c['n']} & {f3(c['GG'])} & {f3(c['GS'])} \\\\")
T += [r"\bottomrule", r"\end{tabular}"]; (TABLES / "tab_clean_label.tex").write_text("\n".join(T) + "\n")
print("wrote tab_reproduction, tab_intersection, tab_terciles, tab_clean_label")
