"""
Batch inference pipeline for MEI benchmark.

Orchestrates running VLMs across all items and intervention variants,
saving predictions in standardized format.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from mei_benchmark.data.loader import MEIDatasetLoader, save_predictions
from mei_benchmark.data.schema import (
    InterventionType,
    MEIItem,
    ModelPrediction,
    ModelPredictionSet,
)
from mei_benchmark.models.base import BaseVLM, VLMConfig

logger = logging.getLogger(__name__)


# Full model registry (local + API)
MODEL_REGISTRY: dict[str, str] = {
    # Open-source local models
    "llava-1.5-7b": "mei_benchmark.models.llava:LLaVAWrapper",
    "llava-1.5-13b": "mei_benchmark.models.llava:LLaVAWrapper",
    "llava-v1.6-7b": "mei_benchmark.models.llava:LLaVAWrapper",
    "llava-v1.6-mistral": "mei_benchmark.models.llava:LLaVAWrapper",
    "qwen2-vl-7b": "mei_benchmark.models.qwen_vl:QwenVLWrapper",
    "qwen2-vl-72b": "mei_benchmark.models.qwen_vl:QwenVLWrapper",
    "internvl2-8b": "mei_benchmark.models.internvl:InternVL2Wrapper",
    "internvl2.5-8b": "mei_benchmark.models.internvl:InternVL2Wrapper",
    "internvl2_5-8b": "mei_benchmark.models.internvl:InternVL2Wrapper",
    # New open-source models (Phase 2)
    "phi-3.5-vision": "mei_benchmark.models.phi3v:Phi3VisionWrapper",
    "llama-3.2-11b-vision": "mei_benchmark.models.llama_vision:LlamaVisionWrapper",
    "deepseek-vl-7b": "mei_benchmark.models.deepseek_vl:DeepSeekVLWrapper",
    "molmo-7b": "mei_benchmark.models.molmo:MolmoWrapper",
    # API models
    "gpt-4o": "mei_benchmark.models.api_models:GPT4oWrapper",
    "claude-3.5-sonnet": "mei_benchmark.models.api_models:ClaudeWrapper",
    "gemini-1.5-pro": "mei_benchmark.models.api_models:GeminiWrapper",
}


def load_model_from_config(config: VLMConfig) -> BaseVLM:
    """Load a model wrapper from a config, using the registry.

    Args:
        config: VLMConfig with model_id set

    Returns:
        Initialized (but not loaded) model wrapper
    """
    # Try direct import path
    for key, import_path in MODEL_REGISTRY.items():
        if key in config.model_id.lower() or key in config.model_name.lower():
            module_path, class_name = import_path.rsplit(":", 1)
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls(config)

    raise ValueError(
        f"Cannot find model class for {config.model_id}. "
        f"Register it in MODEL_REGISTRY or use a direct import."
    )


def run_inference(
    model: BaseVLM,
    items: list[MEIItem],
    data_root: str | Path,
    output_dir: str | Path,
    intervention_types: Optional[list[InterventionType]] = None,
    resume: bool = True,
) -> list[ModelPredictionSet]:
    """Run inference on all items and their intervention variants.

    This is the main inference entry point. For each item, runs:
    1. Original image
    2. Each intervention variant

    Results are streamed to disk for crash recovery.

    Args:
        model: Loaded VLM wrapper
        items: List of MEIItem to evaluate
        data_root: Root directory containing images/masks
        output_dir: Directory to save predictions
        intervention_types: Which intervention types to evaluate
            (default: all available per item)
        resume: Whether to resume from existing predictions

    Returns:
        List of ModelPredictionSet (one per item)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_file = output_dir / f"{model.name}_predictions.jsonl"

    # Load existing predictions for resume
    existing: dict[str, ModelPrediction] = {}
    if resume and pred_file.exists():
        with open(pred_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pred = ModelPrediction(**json.loads(line))
                # Use .value for enum to ensure consistent keys
                itype = pred.intervention_type
                key_val = itype.value if hasattr(itype, 'value') else str(itype)
                key = f"{pred.item_id}_{key_val}"
                existing[key] = pred
        logger.info(f"Resuming: {len(existing)} predictions already completed")

    all_predictions: list[ModelPrediction] = list(existing.values())

    # Open file in append mode for streaming
    with open(pred_file, "a") as f:
        for item in tqdm(items, desc=f"Evaluating {model.name}"):
            # --- Original image ---
            orig_key = f"{item.item_id}_original"
            if orig_key not in existing:
                img_path = Path(data_root) / item.image_path
                if not img_path.exists():
                    logger.warning(f"Image not found: {img_path}")
                    continue

                image = cv2.imread(str(img_path))
                if image is None:
                    logger.warning(f"Failed to read image: {img_path}")
                    continue

                response = model.predict(
                    image, item.question, None
                )

                pred = ModelPrediction(
                    item_id=item.item_id,
                    model_name=model.name,
                    intervention_type="original",
                    answer=response.answer,
                    raw_response=response.raw_output,
                    confidence=response.confidence,
                    latency_ms=response.latency_ms,
                )
                all_predictions.append(pred)
                f.write(pred.model_dump_json() + "\n")
                f.flush()

            # --- Intervention variants ---
            for variant in item.variants:
                if intervention_types and variant.intervention_type not in intervention_types:
                    continue

                var_key = f"{item.item_id}_{variant.intervention_type.value}"
                if var_key in existing:
                    continue

                var_img_path = Path(data_root) / variant.image_path
                if not var_img_path.exists():
                    logger.warning(f"Variant image not found: {var_img_path}")
                    continue

                var_image = cv2.imread(str(var_img_path))
                if var_image is None:
                    continue

                response = model.predict(
                    var_image, item.question, None
                )

                pred = ModelPrediction(
                    item_id=item.item_id,
                    model_name=model.name,
                    intervention_type=variant.intervention_type.value,
                    answer=response.answer,
                    raw_response=response.raw_output,
                    confidence=response.confidence,
                    latency_ms=response.latency_ms,
                )
                all_predictions.append(pred)
                f.write(pred.model_dump_json() + "\n")
                f.flush()

    # Group predictions by item_id into per-item ModelPredictionSets
    from collections import defaultdict
    preds_by_item: dict[str, list[ModelPrediction]] = defaultdict(list)
    for pred in all_predictions:
        preds_by_item[pred.item_id].append(pred)

    pred_sets = [
        ModelPredictionSet(
            item_id=item_id,
            model_name=model.name,
            predictions=preds,
        )
        for item_id, preds in preds_by_item.items()
    ]

    logger.info(
        f"Inference complete: {len(all_predictions)} predictions "
        f"across {len(pred_sets)} items for {model.name}"
    )
    return pred_sets


def run_dose_response_inference(
    model: BaseVLM,
    items: list[MEIItem],
    data_root: str | Path,
    output_dir: str | Path,
    dose_dirs: dict[str, str],
) -> dict[str, list[ModelPrediction]]:
    """Run inference across dose-response levels.

    Args:
        model: Loaded VLM wrapper
        items: List of MEIItem
        data_root: Root data directory
        output_dir: Output directory for predictions
        dose_dirs: {dose_label: directory_containing_dose_images}

    Returns:
        {dose_label: list of ModelPrediction}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[ModelPrediction]] = {}

    for dose_label, dose_dir in dose_dirs.items():
        logger.info(f"Running dose level: {dose_label}")
        dose_preds = []

        for item in tqdm(items, desc=f"Dose {dose_label}"):
            # Look for dose-level image
            dose_img_path = Path(dose_dir) / f"{item.item_id}.jpg"
            if not dose_img_path.exists():
                dose_img_path = Path(dose_dir) / f"{item.item_id}.png"
            if not dose_img_path.exists():
                continue

            image = cv2.imread(str(dose_img_path))
            if image is None:
                continue

            response = model.predict(
                image, item.question, None
            )

            pred = ModelPrediction(
                item_id=item.item_id,
                model_name=model.name,
                intervention_type=f"dose_{dose_label}",
                answer=response.answer,
                raw_response=response.raw_output,
                confidence=response.confidence,
                latency_ms=response.latency_ms,
            )
            dose_preds.append(pred)

        results[dose_label] = dose_preds

        # Save per-dose predictions
        dose_file = output_dir / f"{model.name}_dose_{dose_label}.jsonl"
        with open(dose_file, "w") as f:
            for p in dose_preds:
                f.write(p.model_dump_json() + "\n")

    return results
