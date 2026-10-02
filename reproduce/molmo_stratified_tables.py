"""Patch the Molmo-7B-D rows of the stratified tables with its corrected per-task metrics,
computed by the paper's own StratifiedEvaluator on the fixed predictions; format identical to
mei_benchmark/scripts/build_appendix_tables.py. Writes into results/tables/."""
import sys, json, re
from pathlib import Path
from repro_paths import RERUN_DATA, MOLMO_FIX, TABLES
from mei_benchmark.data.loader import MEIDatasetLoader, PredictionLoader
from mei_benchmark.evaluation.metrics import StratifiedEvaluator

items = MEIDatasetLoader(str(RERUN_DATA)).load_items()
ps = PredictionLoader(str(MOLMO_FIX)).load_predictions("Molmo-7B-D")
s = StratifiedEvaluator().evaluate_stratified(items, ps, "Molmo-7B-D")
bt = {(k.value if hasattr(k, "value") else k): v for k, v in s.by_task.items()}
order = ["object_identification", "attribute_verification", "spatial_relations", "counting", "text_in_image"]
f = lambda x: f"{x:.3f}".lstrip("0").replace("-0.", "-.")
full = (TABLES / "tab_full_stratified.tex").read_text().split("\n")
idx = [i for i, l in enumerate(full) if "Molmo-7B-D" in l]; assert len(idx) == 5, idx
gg = {}
for i, t in zip(idx, order):
    tm = bt[t]; nc = int(round(tm.original_accuracy * tm.n_items)); dag = r"^\dagger" if nc < 30 else ""
    full[i] = (f" & Molmo-7B-D$^\\ddagger$ & {f(tm.original_accuracy)} & ${nc}{dag}$ & {f(tm.causal_sensitivity_blur)} & "
               f"{f(tm.spurious_invariance_blur)} & {f(tm.grounding_gap)} & {f(tm.sham_robustness_blur)} & {f(tm.grounding_specificity)} \\\\")
    gg[t] = tm.grounding_gap; print(t, "n_c", nc, "GG", round(tm.grounding_gap, 3))
(TABLES / "tab_full_stratified.tex").write_text("\n".join(full))
st = (TABLES / "tab_stratified.tex").read_text().split("\n")
j = [i for i, l in enumerate(st) if l.startswith("Molmo-7B-D")]; assert len(j) == 1
st[j[0]] = "Molmo-7B-D & " + " & ".join(f(gg[t]) for t in order) + r" \\"
(TABLES / "tab_stratified.tex").write_text("\n".join(st))
print("patched Molmo rows")
