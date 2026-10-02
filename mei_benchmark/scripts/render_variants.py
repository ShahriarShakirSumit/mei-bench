#!/usr/bin/env python3
"""Render the five intervention variants of the RELEASED benchmark from the boxes stored in the annotations.

Unlike generate_interventions.py (which samples new sham regions), this script uses each item's stored
evidence box and sham box, so evidence/sham blur and gray-out images are reproduced exactly as released.
Content swap applies a random texture scramble; it is seeded per item here, but the paper's swap images
were produced with an unseeded generator and are therefore not bit-identical.

Usage:
    python -m mei_benchmark.scripts.render_variants \
        --annotations data/annotations/mei_bench.jsonl --images-dir data/coco/val2017 --out-dir data/mei_bench
    python -m mei_benchmark.scripts.render_variants \
        --annotations data/rerun/items.jsonl --images-dir data/coco/val2017 --out-dir data/rerun     # the re-run's images
"""
from __future__ import annotations
import argparse, json, zlib
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm
from mei_benchmark.scripts.generate_interventions import bbox_to_mask
from mei_benchmark.intervention.generator import apply_blur_intervention, apply_gray_intervention, apply_swap_intervention

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotations", default="data/annotations/mei_bench.jsonl")
    ap.add_argument("--images-dir", default="data/coco/val2017")
    ap.add_argument("--out-dir", default="data/mei_bench")
    ap.add_argument("--blur-radius", type=int, default=51)
    ap.add_argument("--gray-value", type=int, default=128)
    a = ap.parse_args()
    out = Path(a.out_dir); (out / "variants").mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in open(a.annotations) if l.strip()]
    for it in tqdm(items, desc="rendering"):
        img = cv2.imread(str(Path(a.images_dir) / it["image_path"]))
        if img is None:
            raise FileNotFoundError(Path(a.images_dir) / it["image_path"])
        h, w = img.shape[:2]
        ev = bbox_to_mask(it["evidence_region"]["bbox"], h, w)
        sham_box = next(v["sham_region_bbox"] for v in it["variants"] if v.get("sham_region_bbox"))
        sh = bbox_to_mask(sham_box, h, w)
        np.random.seed(zlib.crc32(it["item_id"].encode()))          # deterministic swap scramble
        rendered = {"evidence_blur": apply_blur_intervention(img, ev, blur_radius=a.blur_radius).image,
                    "evidence_gray": apply_gray_intervention(img, ev, gray_value=a.gray_value).image,
                    "evidence_swap": apply_swap_intervention(img, ev).image,
                    "sham_blur": apply_blur_intervention(img, sh, blur_radius=a.blur_radius).image,
                    "sham_gray": apply_gray_intervention(img, sh, gray_value=a.gray_value).image}
        for v in it["variants"]:
            cv2.imwrite(str(out / v["image_path"]), rendered[v["intervention_type"]], [cv2.IMWRITE_JPEG_QUALITY, 95])
        link = out / it["image_path"]
        if not link.exists():
            link.symlink_to((Path(a.images_dir) / it["image_path"]).resolve())
    if Path(a.annotations).resolve() != (out / "items.jsonl").resolve():
        with open(out / "items.jsonl", "w") as f:
            for it in items: f.write(json.dumps(it) + "\n")
    print(f"rendered {len(items)} items x 5 variants into {out}")

if __name__ == "__main__":
    main()
