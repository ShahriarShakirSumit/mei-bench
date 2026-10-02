"""Seed each robustness condition with the re-run's predictions on the ORIGINAL images.

The conditions change only the sham (or, for `mask`, the evidence operator), so the answer on the unmodified image,
and hence each model's correct set C, is the re-run's. This copies those `original` predictions into
outputs/robustness/<cond>/predictions/; `run_evaluation --resume` then skips them and evaluates only the new variants.
Existing files are left untouched."""
import json
from repro_paths import RERUN_PRED, ROBUSTNESS_DATA, ROBUSTNESS_OUT

conds = sorted(p.name for p in ROBUSTNESS_DATA.iterdir() if (p / "items.jsonl").exists())
for cond in conds:
    out = ROBUSTNESS_OUT / cond / "predictions"; out.mkdir(parents=True, exist_ok=True)
    for src in sorted(RERUN_PRED.glob("*_predictions.jsonl")):
        dst = out / src.name
        if dst.exists(): continue
        with open(src) as a, open(dst, "w") as b:
            for line in a:
                d = json.loads(line)
                if d["intervention_type"] == "original": b.write(line)          # all 615, as in the paper's runs
    print(f"{cond}: seeded {len(list(RERUN_PRED.glob('*_predictions.jsonl')))} models")
