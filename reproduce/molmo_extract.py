#!/usr/bin/env python
"""Model-specific answer extraction for Molmo-7B-D (ACML 2026).

Failure mode (verified on all 3,690 raw outputs of the pinned re-run): Molmo does not stop after
its answer; it keeps generating further sentences ON THE SAME LINE ("To the left.To the right....",
"Right of motorcycle.The chair is positioned ..."). The shared extractor keeps the first LINE, so the
normalised string never equals a valid answer -> 0/615 in the original table.

Rules below were fixed from the answer FORMAT only (before any metric was computed):
  1. keep the first sentence (text before the first '.', '!', '?' or newline);
  2. if it contains exactly one of {left, right} as a word -> that word;
  3. if it starts with a number word or digit -> the digit ("Two couches" -> "2");
  4. otherwise the sentence itself (the paper's normalize_answer then lower-cases, strips articles
     and punctuation).
Scores are then computed with the paper's unchanged MEIEvaluator.
"""
import json, re, sys
from pathlib import Path
from repro_paths import RERUN_DATA, RERUN_PRED, MOLMO_FIX, RESULTS
from mei_benchmark.data.loader import MEIDatasetLoader, PredictionLoader   # noqa: E402
from mei_benchmark.evaluation.metrics import MEIEvaluator                  # noqa: E402

NUM = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}

def extract(raw: str) -> str:
    s = re.split(r"[.!?\n]", (raw or "").strip(), maxsplit=1)[0].strip()
    words = re.findall(r"[a-z0-9]+", s.lower())
    lr = [w for w in words if w in ("left", "right")]
    if len(set(lr)) == 1:
        return lr[0]
    if words and (words[0] in NUM or words[0].isdigit()):
        return NUM.get(words[0], words[0])
    return s

def main():
    SRC = RERUN_PRED / "Molmo-7B-D_predictions.jsonl"
    OUT = MOLMO_FIX; OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    assert len(rows) == 3690, len(rows)
    with open(OUT / "Molmo-7B-D_predictions.jsonl", "w") as f:
        for r in rows:
            r = dict(r); r["answer"] = extract(r["raw_response"]); f.write(json.dumps(r) + "\n")
    items = MEIDatasetLoader(str(RERUN_DATA)).load_items()
    m = MEIEvaluator().evaluate(items, PredictionLoader(str(OUT)).load_predictions("Molmo-7B-D"))
    d = m.to_dict(); print(json.dumps({k: d[k] for k in d if k in ("n_items", "n_correct_original", "accuracy", "original_accuracy",
          "CS_blur", "CS_gray", "SI_blur", "SI_gray", "GG", "GS", "SR_blur")}, indent=1))
    json.dump(d, open(RESULTS / "molmo_fixed_metrics.json", "w"), indent=1)
    # the ten open models with Molmo's corrected answers, for intersection_analysis.py
    comb = RERUN_PRED.parent.parent / "rerun_molmo_fixed" / "predictions"; comb.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in RERUN_PRED.glob("*_predictions.jsonl"):
        shutil.copy(OUT / f.name if f.name.startswith("Molmo-7B-D") else f, comb / f.name)


if __name__ == "__main__":
    main()
