"""
Data loading utilities for MEI benchmark.

Supports loading from JSONL format and integrating with source datasets (COCO).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import jsonlines

from mei_benchmark.data.schema import (
    MEIDatasetManifest,
    MEIItem,
    ModelPrediction,
    ModelPredictionSet,
    TaskType,
)


class MEIDatasetLoader:
    """Load and iterate over MEI benchmark items."""

    def __init__(self, data_dir: str | Path, split: str = "test"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.manifest: Optional[MEIDatasetManifest] = None
        self._items: Optional[list[MEIItem]] = None

        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = MEIDatasetManifest(**json.load(f))

    def load_items(self) -> list[MEIItem]:
        """Load all MEI items from JSONL file."""
        if self._items is not None:
            return self._items

        items_file = self.data_dir / f"{self.split}.jsonl"
        if not items_file.exists():
            items_file = self.data_dir / "items.jsonl"

        if not items_file.exists():
            raise FileNotFoundError(f"No items file found at {items_file}")

        self._items = []
        with jsonlines.open(items_file) as reader:
            for obj in reader:
                self._items.append(MEIItem(**obj))

        return self._items

    def iter_items(self) -> Iterator[MEIItem]:
        """Iterate over MEI items one at a time (memory-efficient)."""
        items_file = self.data_dir / f"{self.split}.jsonl"
        if not items_file.exists():
            items_file = self.data_dir / "items.jsonl"

        with jsonlines.open(items_file) as reader:
            for obj in reader:
                yield MEIItem(**obj)

    def filter_by_task(self, task_type: TaskType) -> list[MEIItem]:
        """Filter items by task type."""
        items = self.load_items()
        return [item for item in items if item.task_type == task_type]

    def filter_by_difficulty(self, difficulty: str) -> list[MEIItem]:
        """Filter items by difficulty level."""
        items = self.load_items()
        return [item for item in items if item.difficulty.value == difficulty]

    def get_human_verified(self) -> list[MEIItem]:
        """Get only human-verified items."""
        items = self.load_items()
        return [item for item in items if item.human_verified]

    def summary(self) -> dict:
        """Get summary statistics of the dataset."""
        items = self.load_items()
        task_counts = {}
        difficulty_counts = {}
        verified_count = 0

        for item in items:
            task_counts[item.task_type.value] = task_counts.get(item.task_type.value, 0) + 1
            difficulty_counts[item.difficulty.value] = (
                difficulty_counts.get(item.difficulty.value, 0) + 1
            )
            if item.human_verified:
                verified_count += 1

        return {
            "total_items": len(items),
            "task_distribution": task_counts,
            "difficulty_distribution": difficulty_counts,
            "human_verified": verified_count,
            "avg_evidence_area": sum(
                item.evidence_region.area_fraction for item in items
            )
            / max(len(items), 1),
            "avg_variants_per_item": sum(len(item.variants) for item in items)
            / max(len(items), 1),
        }


class PredictionLoader:
    """Load model predictions for evaluation."""

    def __init__(self, predictions_dir: str | Path):
        self.predictions_dir = Path(predictions_dir)

    def load_predictions(self, model_name: str) -> list[ModelPredictionSet]:
        """Load all predictions for a given model.

        Supports two formats:
        1. Flat JSONL: one ModelPrediction per line (written by inference.py)
        2. Grouped JSONL: one ModelPredictionSet per line

        Returns per-item ModelPredictionSets in both cases.
        """
        # Try both naming conventions
        pred_file = self.predictions_dir / f"{model_name}.jsonl"
        if not pred_file.exists():
            pred_file = self.predictions_dir / f"{model_name}_predictions.jsonl"
        if not pred_file.exists():
            # Try fuzzy match
            for f in self.predictions_dir.glob("*.jsonl"):
                if model_name.lower() in f.stem.lower():
                    pred_file = f
                    break
        if not pred_file.exists():
            raise FileNotFoundError(
                f"No predictions file for {model_name} at {self.predictions_dir}"
            )

        # Peek at first line to detect format
        with open(pred_file) as f:
            first_line = f.readline().strip()
            if not first_line:
                return []
            first_obj = json.loads(first_line)

        if "predictions" in first_obj:
            # Grouped format: one ModelPredictionSet per line
            prediction_sets = []
            with jsonlines.open(pred_file) as reader:
                for obj in reader:
                    prediction_sets.append(ModelPredictionSet(**obj))
            return prediction_sets
        else:
            # Flat format: one ModelPrediction per line -> group by item_id
            from collections import defaultdict
            preds_by_item: dict[str, list[ModelPrediction]] = defaultdict(list)
            inferred_model_name = model_name
            with jsonlines.open(pred_file) as reader:
                for obj in reader:
                    pred = ModelPrediction(**obj)
                    preds_by_item[pred.item_id].append(pred)
                    inferred_model_name = pred.model_name

            return [
                ModelPredictionSet(
                    item_id=item_id,
                    model_name=inferred_model_name,
                    predictions=preds,
                )
                for item_id, preds in preds_by_item.items()
            ]

    def load_all_models(self) -> dict[str, list[ModelPredictionSet]]:
        """Load predictions for all models found in the predictions directory."""
        all_preds = {}
        for pred_file in sorted(self.predictions_dir.glob("*.jsonl")):
            model_name = pred_file.stem
            all_preds[model_name] = self.load_predictions(model_name)
        return all_preds

    def available_models(self) -> list[str]:
        """List all models with available predictions."""
        models = []
        for f in sorted(self.predictions_dir.glob("*.jsonl")):
            name = f.stem
            # Strip _predictions suffix if present
            if name.endswith("_predictions"):
                name = name[:-len("_predictions")]
            models.append(name)
        return models


def save_items(items: list[MEIItem], output_path: str | Path) -> None:
    """Save MEI items to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(output_path, mode="w") as writer:
        for item in items:
            writer.write(item.model_dump())


def save_predictions(
    predictions: list[ModelPredictionSet], output_path: str | Path
) -> None:
    """Save model predictions to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(output_path, mode="w") as writer:
        for pred_set in predictions:
            writer.write(pred_set.model_dump())


def save_manifest(manifest: MEIDatasetManifest, output_path: str | Path) -> None:
    """Save dataset manifest to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(manifest.model_dump(), f, indent=2)
