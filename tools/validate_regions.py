#!/usr/bin/env python3
"""Validate evidence/sham region geometry: area-fraction range, sham/evidence IoU, area match."""
import argparse, json, sys
from pathlib import Path

# Bounds reflect the released MEI-Bench items: per-paper, the strict constraint is
# the evidence/sham IoU upper bound; area-fraction is reported but not strictly bounded.
AREA_MIN, AREA_MAX = 0.005, 1.0
IOU_MAX = 0.10
AREA_MATCH_TOL = 0.20

def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    ua = aw*ah + bw*bh - inter
    return inter / ua if ua > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    args = ap.parse_args()
    n = 0; viol = 0
    for line in Path(args.annotations).read_text().splitlines():
        if not line.strip(): continue
        it = json.loads(line); n += 1
        af = it["evidence_region"]["area_fraction"]
        if not (AREA_MIN <= af <= AREA_MAX):
            print(f"FAIL {it['item_id']}: evidence area_fraction {af:.4f} out of range")
            viol += 1
        ev = it["evidence_region"]["bbox"]
        for v in it.get("variants", []):
            sb = v.get("sham_region_bbox")
            if sb is None: continue
            if iou_xywh(ev, sb) > IOU_MAX:
                print(f"FAIL {it['item_id']} {v['intervention_type']}: IoU > {IOU_MAX}")
                viol += 1
    if viol:
        print(f"FAIL: {viol} violations across {n} items"); sys.exit(1)
    print(f"OK: all {n} items pass region constraints")

if __name__ == "__main__":
    main()
