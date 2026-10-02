#!/usr/bin/env python3
"""
Generate intervention variant images for all MEI items.

Reads items.jsonl, loads source images, creates evidence masks from bounding boxes,
generates all 5 intervention types (blur, gray, swap, sham_blur, sham_gray),
saves images and updates items.jsonl with variant paths.

Usage:
    python -m mei_benchmark.scripts.generate_interventions \
        --data-dir data/mei_bench \
        --images-dir data/coco/val2017 \
        --output-dir data/mei_bench/variants
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from mei_benchmark.data.schema import (
    InterventionType,
    InterventionVariant,
    MEIItem,
)
from mei_benchmark.intervention.generator import (
    apply_blur_intervention,
    apply_gray_intervention,
    apply_swap_intervention,
)

logger = logging.getLogger(__name__)


def bbox_to_mask(bbox: list[float], height: int, width: int) -> np.ndarray:
    """Convert [x, y, w, h] bounding box to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    x, y, w, h = bbox
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(width, int(x + w))
    y2 = min(height, int(y + h))
    mask[y1:y2, x1:x2] = 255
    return mask


def generate_sham_region(
    bbox: list[float],
    height: int,
    width: int,
    rng: random.Random,
    max_attempts: int = 50,
) -> list[float]:
    """Generate a sham bounding box with matching area, no overlap with evidence.

    Returns [x, y, w, h] for sham region.
    """
    ex, ey, ew, eh = bbox
    target_area = ew * eh

    for _ in range(max_attempts):
        # Random aspect ratio similar to evidence
        aspect = (ew / max(eh, 1)) * rng.uniform(0.7, 1.3)
        sh = max(10, int(np.sqrt(target_area / max(aspect, 0.1))))
        sw = max(10, int(sh * aspect))

        # Clamp to image bounds
        sw = min(sw, width - 1)
        sh = min(sh, height - 1)

        # Random position
        sx = rng.randint(0, max(0, width - sw))
        sy = rng.randint(0, max(0, height - sh))

        # Check IoU with evidence bbox
        ix1 = max(ex, sx)
        iy1 = max(ey, sy)
        ix2 = min(ex + ew, sx + sw)
        iy2 = min(ey + eh, sy + sh)

        if ix2 > ix1 and iy2 > iy1:
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = ew * eh + sw * sh - inter
            iou = inter / max(union, 1)
        else:
            iou = 0.0

        if iou < 0.1:
            # Check area ratio
            area_ratio = (sw * sh) / max(target_area, 1)
            if 0.8 <= area_ratio <= 1.2:
                return [float(sx), float(sy), float(sw), float(sh)]

    # Fallback: place sham in corner opposite to evidence
    cx = ex + ew / 2
    cy = ey + eh / 2
    # Place in opposite corner
    if cx < width / 2:
        sx = max(0, width - int(ew) - 10)
    else:
        sx = 10
    if cy < height / 2:
        sy = max(0, height - int(eh) - 10)
    else:
        sy = 10

    return [float(sx), float(sy), float(min(ew, width - sx)), float(min(eh, height - sy))]


def generate_interventions_for_item(
    item: MEIItem,
    image: np.ndarray,
    output_dir: Path,
    rng: random.Random,
    blur_radius: int = 51,
    gray_value: int = 128,
) -> list[InterventionVariant]:
    """Generate all intervention variants for a single item."""
    h, w = image.shape[:2]
    bbox = item.evidence_region.bbox
    evidence_mask = bbox_to_mask(bbox, h, w)

    # Generate sham region
    sham_bbox = generate_sham_region(bbox, h, w, rng)
    sham_mask = bbox_to_mask(sham_bbox, h, w)

    variants = []

    # 1. Evidence Blur
    result = apply_blur_intervention(image, evidence_mask, blur_radius=blur_radius)
    fname = f"{item.item_id}_evidence_blur.jpg"
    cv2.imwrite(str(output_dir / fname), result.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    variants.append(InterventionVariant(
        intervention_type=InterventionType.EVIDENCE_BLUR,
        image_path=f"variants/{fname}",
        intervention_params={"blur_radius": blur_radius},
        sham_region_bbox=None,
    ))

    # 2. Evidence Gray
    result = apply_gray_intervention(image, evidence_mask, gray_value=gray_value)
    fname = f"{item.item_id}_evidence_gray.jpg"
    cv2.imwrite(str(output_dir / fname), result.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    variants.append(InterventionVariant(
        intervention_type=InterventionType.EVIDENCE_GRAY,
        image_path=f"variants/{fname}",
        intervention_params={"gray_value": gray_value},
        sham_region_bbox=None,
    ))

    # 3. Evidence Swap (content inversion within region)
    result = apply_swap_intervention(image, evidence_mask)
    fname = f"{item.item_id}_evidence_swap.jpg"
    cv2.imwrite(str(output_dir / fname), result.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    variants.append(InterventionVariant(
        intervention_type=InterventionType.EVIDENCE_SWAP,
        image_path=f"variants/{fname}",
        intervention_params={},
        sham_region_bbox=None,
    ))

    # 4. Sham Blur (control)
    result = apply_blur_intervention(image, sham_mask, blur_radius=blur_radius)
    fname = f"{item.item_id}_sham_blur.jpg"
    cv2.imwrite(str(output_dir / fname), result.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    variants.append(InterventionVariant(
        intervention_type=InterventionType.SHAM_BLUR,
        image_path=f"variants/{fname}",
        intervention_params={"blur_radius": blur_radius},
        sham_region_bbox=sham_bbox,
    ))

    # 5. Sham Gray (control)
    result = apply_gray_intervention(image, sham_mask, gray_value=gray_value)
    fname = f"{item.item_id}_sham_gray.jpg"
    cv2.imwrite(str(output_dir / fname), result.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    variants.append(InterventionVariant(
        intervention_type=InterventionType.SHAM_GRAY,
        image_path=f"variants/{fname}",
        intervention_params={"gray_value": gray_value},
        sham_region_bbox=sham_bbox,
    ))

    return variants


def main():
    parser = argparse.ArgumentParser(description="Generate intervention images")
    parser.add_argument("--data-dir", required=True, help="Directory with items.jsonl")
    parser.add_argument("--images-dir", required=True, help="Directory with source images")
    parser.add_argument("--output-dir", default=None, help="Output dir for variant images")
    parser.add_argument("--blur-radius", type=int, default=51)
    parser.add_argument("--gray-value", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    data_dir = Path(args.data_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "variants"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Load items
    items_file = data_dir / "items.jsonl"
    items = []
    with open(items_file) as f:
        for line in f:
            items.append(MEIItem(**json.loads(line.strip())))
    logger.info(f"Loaded {len(items)} items from {items_file}")

    # Generate interventions
    updated_items = []
    n_success = 0
    n_fail = 0

    for item in tqdm(items, desc="Generating interventions"):
        # Find source image
        img_path = images_dir / item.image_path
        if not img_path.exists():
            # Try with zero-padded COCO naming
            img_id = item.source_image_id
            img_path = images_dir / f"{int(img_id):012d}.jpg"

        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            updated_items.append(item)
            n_fail += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Failed to load: {img_path}")
            updated_items.append(item)
            n_fail += 1
            continue

        try:
            variants = generate_interventions_for_item(
                item, image, output_dir, rng,
                blur_radius=args.blur_radius,
                gray_value=args.gray_value,
            )
            # Update item with variants
            updated = item.model_copy(update={"variants": variants})
            updated_items.append(updated)
            n_success += 1
        except Exception as e:
            logger.error(f"Failed for {item.item_id}: {e}")
            updated_items.append(item)
            n_fail += 1

    # Save updated items
    output_items = data_dir / "items.jsonl"
    with open(output_items, "w") as f:
        for item in updated_items:
            f.write(item.model_dump_json() + "\n")

    logger.info(f"Done: {n_success} success, {n_fail} failures")
    logger.info(f"Updated items saved to {output_items}")
    logger.info(f"Variant images in {output_dir}")


if __name__ == "__main__":
    main()
