"""Expand the shipped predictions (results/predictions/**.jsonl.gz) into outputs/ for the analyses.

    python reproduce/unpack_results.py              # re-run + robustness conditions (reproduce the paper)
    python reproduce/unpack_results.py --only rerun # re-run only (before evaluating the conditions yourself)
"""
import argparse, gzip, shutil
from repro_paths import REPO, RERUN_PRED, ROBUSTNESS_OUT
ap = argparse.ArgumentParser(); ap.add_argument("--only", choices=["rerun", "robustness"]); a = ap.parse_args()
src = REPO / "results" / "predictions"
n = 0
for gz in sorted(src.rglob("*.jsonl.gz")):
    rel = gz.relative_to(src)
    if a.only and rel.parts[0] != a.only: continue
    dst = (RERUN_PRED if rel.parts[0] == "rerun" else ROBUSTNESS_OUT / rel.parts[1] / "predictions") / gz.name[:-3]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz, "rb") as f, open(dst, "wb") as g: shutil.copyfileobj(f, g)
    n += 1
print("unpacked", n, "prediction files into outputs/")
