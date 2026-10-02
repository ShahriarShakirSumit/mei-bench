from __future__ import annotations
#!/usr/bin/env python3
"""
Fix dataset issues identified in the diagnostic review.

Validity-first approach: FILTER items that violate the formal constraints
stated in the manuscript, rather than re-render images (which would
invalidate predictions already collected for ten local models).

  C1.  Drop items where IoU(evidence, sham) >= 0.1 (Definition 2).
  C1b. Drop items where the sham area is outside +/-20% of the evidence area.
  C2.  Drop items with area_fraction outside [0.005, 1.0] (Section 4.1).

Backs up the original manifest and writes a corrected items.jsonl.
"""
from mei_benchmark.paths import DATA_DIR, COCO_DIR
import json
import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fix_dataset")

DATA = DATA_DIR
COCO = COCO_DIR
ITEMS = DATA / "items.jsonl"
VARIANTS = DATA / "variants"


def bbox_iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def bbox_to_mask(bbox, h, w):
    m = np.zeros((h, w), dtype=np.uint8)
    x, y, ww, hh = bbox
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(w, int(x + ww)), min(h, int(y + hh))
    m[y1:y2, x1:x2] = 255
    return m


def sample_sham_strict(ev_bbox, h, w, rng: random.Random, max_attempts: int = 4000):
    """Sample sham bbox with IoU<0.1 and area within 20%, both strict."""
    ex, ey, ew, eh = ev_bbox
    target_area = ew * eh
    best = None
    best_iou = 1e9
    for _ in range(max_attempts):
        # keep aspect within 30% of evidence to mimic visual support
        aspect = (ew / max(eh, 1.0)) * rng.uniform(0.75, 1.33)
        sh_h = max(8.0, np.sqrt(target_area / max(aspect, 0.1)))
        sh_w = max(8.0, sh_h * aspect)
        sh_w = min(sh_w, w - 1)
        sh_h = min(sh_h, h - 1)
        if sh_w >= w or sh_h >= h:
            continue
        sx = rng.uniform(0, w - sh_w)
        sy = rng.uniform(0, h - sh_h)
        cand = [sx, sy, sh_w, sh_h]
        area_ratio = (sh_w * sh_h) / max(target_area, 1.0)
        if not (0.8 <= area_ratio <= 1.2):
            continue
        iou = bbox_iou_xywh(ev_bbox, cand)
        if iou < 0.1:
            return cand, iou
        if iou < best_iou:
            best_iou = iou
            best = cand
    # last-resort: shrink the sham slightly to fit image, force corner-opposite
    cx, cy = ex + ew / 2, ey + eh / 2
    sw = min(ew, w * 0.45)
    sh = min(eh, h * 0.45)
    if cx < w / 2:
        sx = w - sw - 1
    else:
        sx = 1
    if cy < h / 2:
        sy = h - sh - 1
    else:
        sy = 1
    cand = [sx, sy, sw, sh]
    iou = bbox_iou_xywh(ev_bbox, cand)
    if iou < 0.1:
        return cand, iou
    return best, best_iou


def apply_blur(img, mask, k=51):
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    out = img.copy()
    out[mask > 0] = blurred[mask > 0]
    return out


def apply_gray(img, mask, val=128):
    out = img.copy()
    out[mask > 0] = val
    return out


def main():
    items = []
    with open(ITEMS) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    log.info(f"loaded {len(items)} items")

    # backup
    bak = ITEMS.with_suffix(".jsonl.bak2")
    if not bak.exists():
        shutil.copy(ITEMS, bak)
        log.info(f"backup -> {bak}")

    rng = random.Random(20260425)

    n_dropped_iou = 0
    n_dropped_area = 0
    n_dropped_degenerate = 0
    fixed_items = []

    for it in tqdm(items):
        ev = it["evidence_region"]["bbox"]
        af = it["evidence_region"]["area_fraction"]

        # C2: filter degenerate area_fraction
        if not (0.005 <= af <= 1.0):
            n_dropped_degenerate += 1
            continue

        # C1: filter sham IoU >= 0.1 OR area outside +/-20%
        cur_sham = None
        for v in it["variants"]:
            if v.get("sham_region_bbox"):
                cur_sham = v["sham_region_bbox"]; break
        if cur_sham is None:
            n_dropped_iou += 1
            continue
        iou = bbox_iou_xywh(ev, cur_sham)
        if iou >= 0.1:
            n_dropped_iou += 1
            continue
        ar = (cur_sham[2] * cur_sham[3]) / max(ev[2] * ev[3], 1.0)
        if not (0.8 <= ar <= 1.2):
            n_dropped_area += 1
            continue

        fixed_items.append(it)

    log.info(f"dropped_iou_violation        = {n_dropped_iou}")
    log.info(f"dropped_area_match_violation = {n_dropped_area}")
    log.info(f"dropped_area_fraction        = {n_dropped_degenerate}")
    log.info(f"items_kept                   = {len(fixed_items)}")

    # write new manifest
    tmp = ITEMS.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for it in fixed_items:
            f.write(json.dumps(it) + "\n")
    tmp.replace(ITEMS)
    log.info("manifest rewritten")

    # also write a degenerate-items report
    rep = DATA / "fix_report.json"
    rep.write_text(json.dumps({
        "items_before": len(items),
        "items_after": len(fixed_items),
        "dropped_iou_violation": n_dropped_iou,
        "dropped_area_match_violation": n_dropped_area,
        "dropped_area_fraction_degenerate": n_dropped_degenerate,
    }, indent=2))
    log.info(f"report -> {rep}")


if __name__ == "__main__":
    main()
