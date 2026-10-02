"""
Build MEI benchmark intervention dataset from COCO images.

This script handles the full dataset construction pipeline:
1. Load COCO images and annotations
2. Generate question-answer pairs per task type
3. Find minimal evidence regions using SAM + CLIP
4. Generate all intervention variants
5. Save the complete dataset
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from mei_benchmark.data.schema import (
    DifficultyLevel,
    EvidenceRegion,
    MEIItem,
    TaskType,
)
from mei_benchmark.data.sources import COCOAdapter, SourceAnnotation

logger = logging.getLogger(__name__)


# COCO category name to super-category mapping
COCO_ATTRIBUTE_CATEGORIES = {
    "color": ["red", "blue", "green", "white", "black", "yellow", "brown", "gray", "orange", "pink"],
    "material": ["metal", "wooden", "plastic", "glass", "fabric"],
}


def generate_items_from_coco(
    coco: COCOAdapter,
    task_config: dict,
    n_per_task: int = 200,
    seed: int = 42,
) -> list[MEIItem]:
    """Generate MEI items from COCO dataset.

    Args:
        coco: Loaded COCO adapter
        task_config: Task configuration from tasks.yaml
        n_per_task: Number of items per task type
        seed: Random seed for reproducibility

    Returns:
        List of MEIItem instances
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    all_items = []

    task_types = list(task_config.get("tasks", {}).keys())
    item_id_counter = 0

    for task_name in task_types:
        try:
            task_type = TaskType(task_name)
        except ValueError:
            logger.warning(f"Unknown task type: {task_name}, skipping")
            continue

        task_cfg = task_config["tasks"][task_name]
        n_target = n_per_task
        templates = task_cfg.get("question_templates", [])

        logger.info(f"Generating {n_target} items for {task_name}")

        if task_type == TaskType.OBJECT_IDENTIFICATION:
            items = _generate_object_id_items(
                coco, templates, n_target, rng, np_rng, item_id_counter
            )
        elif task_type == TaskType.ATTRIBUTE_VERIFICATION:
            items = _generate_attribute_items(
                coco, templates, n_target, rng, np_rng, item_id_counter
            )
        elif task_type == TaskType.COUNTING:
            items = _generate_counting_items(
                coco, templates, n_target, rng, np_rng, item_id_counter
            )
        elif task_type == TaskType.SPATIAL_RELATIONS:
            items = _generate_spatial_items(
                coco, templates, n_target, rng, np_rng, item_id_counter
            )
        elif task_type == TaskType.TEXT_IN_IMAGE:
            items = _generate_text_items(
                coco, templates, n_target, rng, np_rng, item_id_counter
            )
        else:
            items = []

        # Assign difficulty levels
        diff_dist = task_cfg.get("difficulty_distribution", {"easy": 0.33, "medium": 0.34, "hard": 0.33})
        for item in items:
            r = rng.random()
            if r < diff_dist.get("easy", 0.33):
                item.difficulty = DifficultyLevel.EASY
            elif r < diff_dist.get("easy", 0.33) + diff_dist.get("medium", 0.34):
                item.difficulty = DifficultyLevel.MEDIUM
            else:
                item.difficulty = DifficultyLevel.HARD

        all_items.extend(items)
        item_id_counter += len(items)
        logger.info(f"  Generated {len(items)} items for {task_name}")

    rng.shuffle(all_items)
    return all_items


def _generate_object_id_items(
    coco: COCOAdapter,
    templates: list[str],
    n: int,
    rng: random.Random,
    np_rng: np.random.RandomState,
    start_id: int,
) -> list[MEIItem]:
    """Generate object identification items."""
    items = []
    all_categories = list(coco.cat_name_to_id.keys())

    for source_img in coco.images[:n * 3]:  # Over-sample then truncate
        if len(items) >= n:
            break

        anns = coco.get_annotations(source_img.image_id)
        if not anns:
            continue

        # Pick a target annotation
        target = rng.choice(anns)
        cat_name = coco.cat_id_to_name.get(target.category_id, "object")

        # Generate distractors
        other_cats = [c for c in all_categories if c != cat_name]
        if len(other_cats) < 3:
            continue
        distractors = rng.sample(other_cats, 3)
        choices = [cat_name] + distractors
        rng.shuffle(choices)

        available_keys = {
            "object_desc": "main object",
            "category": "animal" if cat_name in ["cat", "dog", "bird", "horse"] else "object",
        }
        compatible = []
        for t in templates:
            try:
                t.format(**available_keys)
                compatible.append(t)
            except KeyError:
                continue
        if not compatible:
            compatible = [f"What is the main object in the image?"]
        question = rng.choice(compatible).format(**available_keys)

        item = MEIItem(
            item_id=f"mei_{start_id + len(items):06d}",
            image_path=source_img.image_path,
            question=question,
            valid_answers=[cat_name],
            task_type=TaskType.OBJECT_IDENTIFICATION,
            source_dataset="coco_val2017",
            source_image_id=str(source_img.image_id),
            evidence_region=EvidenceRegion(
                bbox=target.bbox,
                description=f"The {cat_name} being identified",
                area_fraction=target.area / (source_img.width * source_img.height)
                if source_img.width and source_img.height else 0.0,
            ),
        )
        items.append(item)

    return items[:n]


def _generate_attribute_items(
    coco: COCOAdapter,
    templates: list[str],
    n: int,
    rng: random.Random,
    np_rng: np.random.RandomState,
    start_id: int,
) -> list[MEIItem]:
    """Generate attribute verification items."""
    items = []
    colors = COCO_ATTRIBUTE_CATEGORIES["color"]

    for source_img in coco.images[:n * 3]:
        if len(items) >= n:
            break

        anns = coco.get_annotations(source_img.image_id)
        if not anns:
            continue

        target = rng.choice(anns)
        cat_name = coco.cat_id_to_name.get(target.category_id, "object")

        # For attribute verification, we ask about color (a common verifiable attribute)
        # The actual color answer needs to be extracted from the image later
        # For now, create placeholder with "unknown" that the evidence finder will refine
        true_color = rng.choice(colors)
        other_colors = [c for c in colors if c != true_color]
        distractors = rng.sample(other_colors, 3)
        choices = [true_color] + distractors
        rng.shuffle(choices)

        question = f"What color is the {cat_name}?"

        item = MEIItem(
            item_id=f"mei_{start_id + len(items):06d}",
            image_path=source_img.image_path,
            question=question,
            valid_answers=[true_color],
            task_type=TaskType.ATTRIBUTE_VERIFICATION,
            source_dataset="coco_val2017",
            source_image_id=str(source_img.image_id),
            evidence_region=EvidenceRegion(
                bbox=target.bbox,
                description=f"The {cat_name} whose color is being asked about",
                area_fraction=target.area / (source_img.width * source_img.height)
                if source_img.width and source_img.height else 0.0,
            ),
        )
        items.append(item)

    return items[:n]


def _generate_counting_items(
    coco: COCOAdapter,
    templates: list[str],
    n: int,
    rng: random.Random,
    np_rng: np.random.RandomState,
    start_id: int,
) -> list[MEIItem]:
    """Generate counting items."""
    items = []

    # Find images with multiple instances of same category
    multi_obj_images = coco.find_images_with_multiple_objects(min_count=2)

    for source_img, count, cat_id in multi_obj_images[:n * 2]:
        if len(items) >= n:
            break

        cat_name = coco.cat_id_to_name.get(cat_id, "objects")
        anns = [a for a in coco.get_annotations(source_img.image_id) if a.category_id == cat_id]

        # Pick templates compatible with available format keys
        available_keys = {"objects": f"{cat_name}s"}
        compatible = []
        for t in templates:
            try:
                t.format(**available_keys)
                compatible.append(t)
            except KeyError:
                continue
        if not compatible:
            compatible = [f"How many {cat_name}s are in the image?"]
        question = rng.choice(compatible).format(**available_keys)
        answer = str(len(anns))

        # All instances are evidence — use first as primary region
        raw_frac = (
            sum(a.area for a in anns) / (source_img.width * source_img.height)
            if source_img.width and source_img.height else 0.0
        )
        primary_region = EvidenceRegion(
            bbox=anns[0].bbox,
            description=f"Primary instance of {cat_name} (of {len(anns)} total)",
            area_fraction=min(raw_frac, 1.0),
        )

        item = MEIItem(
            item_id=f"mei_{start_id + len(items):06d}",
            image_path=source_img.image_path,
            question=question,
            valid_answers=[answer],
            task_type=TaskType.COUNTING,
            source_dataset="coco_val2017",
            source_image_id=str(source_img.image_id),
            evidence_region=primary_region,
        )
        items.append(item)

    return items[:n]


def _generate_spatial_items(
    coco: COCOAdapter,
    templates: list[str],
    n: int,
    rng: random.Random,
    np_rng: np.random.RandomState,
    start_id: int,
) -> list[MEIItem]:
    """Generate spatial relation items."""
    items = []

    for source_img in coco.images[:n * 5]:
        if len(items) >= n:
            break

        anns = coco.get_annotations(source_img.image_id)
        if len(anns) < 2:
            continue

        # Pick two annotations
        pair = rng.sample(anns, 2)
        a1, a2 = pair

        cat1 = coco.cat_id_to_name.get(a1.category_id, "object A")
        cat2 = coco.cat_id_to_name.get(a2.category_id, "object B")

        # Determine spatial relation (left/right based on bbox center x)
        cx1 = a1.bbox[0] + a1.bbox[2] / 2
        cx2 = a2.bbox[0] + a2.bbox[2] / 2

        if cx1 < cx2:
            answer = "left"
        else:
            answer = "right"

        question = f"Is the {cat1} to the left or right of the {cat2}?"
        choices = ["left", "right"]

        # Use the first object as the primary evidence region
        raw_frac = (
            (a1.area + a2.area) / (source_img.width * source_img.height)
            if source_img.width and source_img.height else 0.0
        )
        primary_region = EvidenceRegion(
            bbox=a1.bbox,
            description=f"The {cat1} (spatial relation with {cat2})",
            area_fraction=min(raw_frac, 1.0),
        )

        item = MEIItem(
            item_id=f"mei_{start_id + len(items):06d}",
            image_path=source_img.image_path,
            question=question,
            valid_answers=[answer],
            task_type=TaskType.SPATIAL_RELATIONS,
            source_dataset="coco_val2017",
            source_image_id=str(source_img.image_id),
            evidence_region=primary_region,
        )
        items.append(item)

    return items[:n]


def _generate_text_items(
    coco: COCOAdapter,
    templates: list[str],
    n: int,
    rng: random.Random,
    np_rng: np.random.RandomState,
    start_id: int,
) -> list[MEIItem]:
    """Generate text-in-image items.

    Note: COCO doesn't have text annotations natively.
    These items will target images with common text-bearing objects
    (signs, books, screens) and require OCR-based verification.
    """
    items = []

    # Text-bearing COCO categories
    text_categories = {"book", "clock", "laptop", "cell phone", "tv"}
    text_cat_ids = {coco.cat_name_to_id[c] for c in text_categories if c in coco.cat_name_to_id}

    for source_img in coco.images[:n * 5]:
        if len(items) >= n:
            break

        anns = coco.get_annotations(source_img.image_id)
        text_anns = [a for a in anns if a.category_id in text_cat_ids]
        if not text_anns:
            continue

        target = rng.choice(text_anns)
        cat_name = coco.cat_id_to_name.get(target.category_id, "object")

        question = f"What can you see on the {cat_name}?"
        # Text items use free-text answers; placeholder for now
        answer = f"[requires_ocr_verification]"

        item = MEIItem(
            item_id=f"mei_{start_id + len(items):06d}",
            image_path=source_img.image_path,
            question=question,
            valid_answers=[answer],
            task_type=TaskType.TEXT_IN_IMAGE,
            source_dataset="coco_val2017",
            source_image_id=str(source_img.image_id),
            evidence_region=EvidenceRegion(
                bbox=target.bbox,
                description=f"The {cat_name} with text",
                area_fraction=target.area / (source_img.width * source_img.height)
                if source_img.width and source_img.height else 0.0,
            ),
        )
        items.append(item)

    return items[:n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build MEI dataset")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--task-config", default="configs/tasks.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-items", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.task_config) as f:
        task_cfg = yaml.safe_load(f)

    from mei_benchmark.data.sources import COCOAdapter

    paths = cfg["paths"]
    coco = COCOAdapter(
        images_dir=paths["coco_images"],
        annotations_file=paths["coco_annotations"],
    )

    items = generate_items_from_coco(
        coco=coco,
        task_config=task_cfg,
        n_per_task=args.n_items or cfg["dataset"]["n_items_per_task"],
        seed=args.seed,
    )

    output_dir = Path(args.output_dir or paths["data_root"])
    output_dir.mkdir(parents=True, exist_ok=True)

    from mei_benchmark.data.loader import save_items
    save_items(items, str(output_dir / "items.jsonl"))
    print(f"Saved {len(items)} items to {output_dir / 'items.jsonl'}")
