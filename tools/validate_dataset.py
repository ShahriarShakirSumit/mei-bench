#!/usr/bin/env python3
"""Validate MEI-Bench annotations against the schema."""
import argparse, json, sys
from pathlib import Path

REQUIRED = ["item_id","source_dataset","source_image_id","image_path","question","valid_answers","task_type","evidence_region","variants"]
TASKS = {"object_identification","attribute_verification","spatial_relations","counting","text_in_image"}
INTERV = {"evidence_blur","sham_blur","evidence_gray","sham_gray","evidence_swap"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    args = ap.parse_args()
    n = 0; bad = 0
    for line in Path(args.annotations).read_text().splitlines():
        if not line.strip(): continue
        n += 1
        it = json.loads(line)
        for k in REQUIRED:
            if k not in it:
                print(f"FAIL {it.get('item_id','?')}: missing field {k}", file=sys.stderr); bad += 1; break
        if it["task_type"] not in TASKS:
            print(f"FAIL {it['item_id']}: bad task_type {it['task_type']}", file=sys.stderr); bad += 1
        for v in it.get("variants", []):
            if v["intervention_type"] not in INTERV:
                print(f"FAIL {it['item_id']}: bad intervention_type {v['intervention_type']}", file=sys.stderr); bad += 1
    if bad:
        print(f"FAIL: {bad} issues across {n} items"); sys.exit(1)
    print(f"OK: {n} items pass schema")

if __name__ == "__main__":
    main()
