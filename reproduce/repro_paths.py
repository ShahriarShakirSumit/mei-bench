"""Locations used by the reproduction scripts (all relative to the repository; override with env vars).

model_outputs/                    the answers of every model, as shipped (.jsonl.gz)
reproduce/inputs/                 two small inputs: the original run's per-model metrics, and re-run versus original run
data/rerun/items.jsonl            items of the version-pinned re-run (same questions, answers and evidence as the released
                                  annotations; sham regions from the re-run's draw)
data/robustness_conditions/<cond>/  robustness conditions (items.jsonl; images rebuilt by build_robustness_conditions.py)
outputs/rerun/predictions/        re-run predictions        (unpack_results.py expands model_outputs/rerun)
outputs/robustness/<cond>/predictions/  robustness-condition predictions
outputs/molmo_fix/predictions/    Molmo-7B-D predictions with model-specific answer extraction (molmo_extract.py)
outputs/paper/                    everything the scripts produce: numbers (.json), tables (.tex) and figures (.pdf)
"""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
E = lambda k, d: Path(os.environ.get(k, d))
MODEL_OUTPUTS = REPO / "model_outputs"
INPUTS = REPO / "reproduce" / "inputs"
RERUN_DATA = E("MEI_RERUN_DATA", REPO / "data" / "rerun")
ROBUSTNESS_DATA = E("MEI_ROBUSTNESS_DATA", REPO / "data" / "robustness_conditions")
RERUN_PRED = E("MEI_RERUN_PRED", REPO / "outputs" / "rerun" / "predictions")
ROBUSTNESS_OUT = E("MEI_ROBUSTNESS_OUT", REPO / "outputs" / "robustness")
MOLMO_FIX = E("MEI_MOLMO_FIX", REPO / "outputs" / "molmo_fix" / "predictions")
COCO_ANN = E("MEI_COCO_ANN", REPO / "data" / "coco" / "annotations" / "instances_val2017.json")
RESULTS = E("MEI_PAPER_OUT", REPO / "outputs" / "paper"); TABLES = RESULTS / "tables"; FIGURES = RESULTS / "figures"
ORIGINAL_SUMMARY = INPUTS / "original_run_summary.json"        # per-model metrics of the original run
REPRODUCTION = INPUTS / "rerun_vs_original.json"              # re-run versus original run, in item counts
for d in (TABLES, FIGURES): d.mkdir(parents=True, exist_ok=True)
