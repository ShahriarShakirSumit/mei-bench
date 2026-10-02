"""Clean-label subset (Appendix): GG and GS recomputed without the two task types whose labels are least certain
(attribute_verification, text_in_image), on each model's own correct set C, with the paper's per_item_flips.
Input: outputs/rerun_molmo_fixed/predictions (written by molmo_extract.py). Output: outputs/paper/clean_label_subset.json"""
import json
from repro_paths import RERUN_DATA, RERUN_PRED, RESULTS
from scipy.stats import spearmanr
from mei_benchmark.scripts.build_paper_tables import per_item_flips
from mei_benchmark.data.loader import MEIDatasetLoader, PredictionLoader

items = MEIDatasetLoader(str(RERUN_DATA)).load_items()
task = {i.item_id: i.task_type.value if hasattr(i.task_type, "value") else i.task_type for i in items}
pl = PredictionLoader(str(RERUN_PRED.parent.parent / "rerun_molmo_fixed" / "predictions"))
M = ["Qwen2-VL-7B", "InternVL2.5-8B", "InternVL2-8B", "LLaVA-NeXT-7B", "LLaVA-1.5-13B", "LLaVA-1.5-7B", "Phi-3.5-Vision",
     "Llama-3.2-11B-Vision", "DeepSeek-VL-7B", "Molmo-7B-D"]
out = {}
for m in M:
    f = per_item_flips(items, pl.load_predictions(m))
    def gg(ks):
        if not ks: return None
        cs = sum(f[k][1] for k in ks) / len(ks); si = sum(f[k][2] for k in ks) / len(ks)
        return dict(n=len(ks), CS=cs, SI=si, GG=cs - si, GS=max(cs - si, 0) / max(cs, 1e-9))
    allk = list(f); clean = [k for k in f if task[k] not in ("attribute_verification", "text_in_image")]
    out[m] = dict(all=gg(allk), clean=gg(clean), by_task={t: sum(1 for k in f if task[k] == t) for t in dict.fromkeys(task.values())})
    a, c = out[m]["all"], out[m]["clean"]
    print(f"{m:<22} all n={a['n']:3d} GG={a['GG']:.3f} GS={a['GS']:.3f} | clean n={c['n']:3d} GG={c['GG']:.3f} GS={c['GS']:.3f}")
d = [out[m]["clean"]["GG"] - out[m]["all"]["GG"] for m in M]
print("clean-minus-all GG: min %+.3f max %+.3f" % (min(d), max(d)))
print("Spearman GS all vs clean: %.3f" % spearmanr([out[m]["all"]["GS"] for m in M], [out[m]["clean"]["GS"] for m in M]).correlation)
json.dump(out, open(RESULTS / "clean_label_subset.json", "w"), indent=1)
