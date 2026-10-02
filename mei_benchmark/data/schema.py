"""
Pydantic schemas for MEI benchmark data items.

Each MEI item consists of:
- An original image with a question and ground-truth answer
- An evidence mask identifying the minimal region needed for correct answering
- A set of intervention variants (blur, gray, swap, distractor, sham)
- Metadata about the task type, difficulty, and evidence properties
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Task domains for MEI evaluation."""

    TEXT_IN_IMAGE = "text_in_image"
    ATTRIBUTE_VERIFICATION = "attribute_verification"
    SPATIAL_RELATIONS = "spatial_relations"
    COUNTING = "counting"
    OBJECT_IDENTIFICATION = "object_identification"


class InterventionType(str, Enum):
    """Types of minimal evidence interventions."""

    ORIGINAL = "original"
    EVIDENCE_BLUR = "evidence_blur"
    EVIDENCE_GRAY = "evidence_gray"
    EVIDENCE_SWAP = "evidence_swap"
    DISTRACTOR_ADDED = "distractor_added"
    SHAM_BLUR = "sham_blur"
    SHAM_GRAY = "sham_gray"


class DifficultyLevel(str, Enum):
    """Difficulty levels for benchmark items."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvidenceRegion(BaseModel):
    """Defines the minimal evidence region in an image."""

    bbox: list[float] = Field(
        description="Bounding box [x1, y1, x2, y2] in pixel coordinates"
    )
    mask_path: Optional[str] = Field(
        default=None, description="Path to binary mask PNG (255 = evidence region)"
    )
    area_fraction: float = Field(
        description="Fraction of total image area covered by evidence region",
        ge=0.0,
        le=1.0,
    )
    description: str = Field(
        description="Human-readable description of what the evidence region contains"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence that this region is truly the minimal evidence (0-1)",
        ge=0.0,
        le=1.0,
    )


class InterventionVariant(BaseModel):
    """A single intervention variant of an MEI item."""

    intervention_type: InterventionType
    image_path: str = Field(description="Path to the intervened image")
    intervention_params: dict = Field(
        default_factory=dict,
        description="Parameters used for this intervention (e.g., blur_radius, mask_color)",
    )
    sham_region_bbox: Optional[list[float]] = Field(
        default=None,
        description="Bounding box of sham region (for sham interventions only)",
    )


class MEIItem(BaseModel):
    """A single MEI benchmark item with all variants.

    Each item contains the original image+question, the evidence region,
    and all generated intervention variants.
    """

    item_id: str = Field(description="Unique identifier for this item")
    source_dataset: str = Field(
        default="coco", description="Source dataset (coco, openimages, etc.)"
    )
    source_image_id: Optional[str] = Field(
        default=None, description="Original image ID in source dataset"
    )

    # Original image and question
    image_path: str = Field(description="Path to the original image")
    question: str = Field(description="The question about the image")
    valid_answers: list[str] = Field(
        description="All valid answers to the question"
    )
    task_type: TaskType = Field(description="Task domain classification")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)

    # Evidence region
    evidence_region: EvidenceRegion = Field(
        description="The minimal evidence region for this question"
    )

    # Intervention variants
    variants: list[InterventionVariant] = Field(
        default_factory=list,
        description="List of intervention variants generated for this item",
    )

    # Metadata
    human_verified: bool = Field(
        default=False,
        description="Whether this item has been verified by a human annotator",
    )
    notes: Optional[str] = Field(default=None)

    def get_variant(self, intervention_type: InterventionType) -> Optional[InterventionVariant]:
        """Get a specific intervention variant by type."""
        for v in self.variants:
            if v.intervention_type == intervention_type:
                return v
        return None

    def get_all_image_paths(self) -> list[str]:
        """Get all image paths (original + all variants)."""
        paths = [self.image_path]
        for v in self.variants:
            paths.append(v.image_path)
        return paths


class ModelPrediction(BaseModel):
    """A model's prediction for a single image+question pair."""

    item_id: str
    intervention_type: InterventionType
    model_name: str
    answer: str = Field(description="Model's predicted answer")
    raw_response: Optional[str] = Field(
        default=None, description="Full raw model response before parsing"
    )
    confidence: Optional[float] = Field(
        default=None, description="Model's self-reported confidence (0-1)"
    )
    logprob: Optional[float] = Field(
        default=None, description="Log-probability of the predicted answer"
    )
    latency_ms: Optional[float] = Field(
        default=None, description="Inference latency in milliseconds"
    )


class ModelPredictionSet(BaseModel):
    """All predictions from a model for a single MEI item (original + variants)."""

    item_id: str
    model_name: str
    predictions: list[ModelPrediction] = Field(default_factory=list)

    def get_prediction(self, intervention_type: InterventionType) -> Optional[ModelPrediction]:
        """Get prediction for a specific intervention type."""
        for p in self.predictions:
            if p.intervention_type == intervention_type:
                return p
        return None

    @property
    def original_prediction(self) -> Optional[ModelPrediction]:
        """Get the prediction on the original (unmodified) image."""
        return self.get_prediction(InterventionType.ORIGINAL)


class MEIDatasetManifest(BaseModel):
    """Manifest for the full MEI benchmark dataset."""

    version: str = "1.0"
    name: str = "MEI-Benchmark"
    description: str = "Causal Visual Grounding Test via Minimal Evidence Interventions"
    total_items: int = 0
    task_distribution: dict[str, int] = Field(default_factory=dict)
    difficulty_distribution: dict[str, int] = Field(default_factory=dict)
    human_verified_count: int = 0
    data_dir: str = ""
    items_file: str = "items.jsonl"
    split: str = "test"
