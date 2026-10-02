"""
High-level intervention pipeline that orchestrates evidence finding and
intervention generation for the full MEI benchmark.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from mei_benchmark.data.schema import (
    EvidenceRegion,
    InterventionType,
    InterventionVariant,
    MEIItem,
)
from mei_benchmark.intervention.evidence_finder import EvidenceFinder
from mei_benchmark.intervention.generator import (
    apply_blur_intervention,
    apply_distractor_intervention,
    apply_gray_intervention,
    apply_sham_blur_intervention,
    apply_sham_gray_intervention,
    apply_swap_intervention,
)

logger = logging.getLogger(__name__)


class MEIInterventionPipeline:
    """End-to-end pipeline for generating all MEI intervention variants.

    For each benchmark item:
    1. Load the original image
    2. Identify or use the evidence mask
    3. Generate a sham mask (control)
    4. Apply all 5 intervention types
    5. Save results to output directory
    """

    def __init__(
        self,
        output_dir: str | Path,
        evidence_finder: Optional[EvidenceFinder] = None,
        blur_radius: int = 51,
        gray_value: int = 128,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_finder = evidence_finder
        self.blur_radius = blur_radius
        self.gray_value = gray_value
        self.rng = np.random.default_rng(seed)

    def process_item(self, item: MEIItem) -> MEIItem:
        """Generate all intervention variants for a single MEI item.

        Args:
            item: MEI item with image_path and evidence_region defined

        Returns:
            Updated MEI item with variants populated
        """
        # Load image
        image = cv2.imread(item.image_path)
        if image is None:
            logger.error(f"Cannot load image: {item.image_path}")
            return item

        h, w = image.shape[:2]

        # Get evidence mask
        if item.evidence_region.mask_path and Path(item.evidence_region.mask_path).exists():
            evidence_mask = cv2.imread(item.evidence_region.mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Create mask from bbox
            evidence_mask = EvidenceFinder.create_evidence_mask_from_bbox(
                (h, w), item.evidence_region.bbox
            )

        # Create sham mask (control)
        sham_mask = EvidenceFinder.create_sham_mask((h, w), evidence_mask, self.rng)

        # Create output directory for this item
        item_dir = self.output_dir / item.item_id
        item_dir.mkdir(parents=True, exist_ok=True)

        variants = []

        # 1. Evidence Blur
        result = apply_blur_intervention(image, evidence_mask, self.blur_radius)
        path = self._save_variant(result.image, item_dir, "evidence_blur")
        variants.append(
            InterventionVariant(
                intervention_type=InterventionType.EVIDENCE_BLUR,
                image_path=str(path),
                intervention_params=result.params,
            )
        )

        # 2. Evidence Gray
        result = apply_gray_intervention(image, evidence_mask, self.gray_value)
        path = self._save_variant(result.image, item_dir, "evidence_gray")
        variants.append(
            InterventionVariant(
                intervention_type=InterventionType.EVIDENCE_GRAY,
                image_path=str(path),
                intervention_params=result.params,
            )
        )

        # 3. Evidence Swap
        result = apply_swap_intervention(image, evidence_mask)
        path = self._save_variant(result.image, item_dir, "evidence_swap")
        variants.append(
            InterventionVariant(
                intervention_type=InterventionType.EVIDENCE_SWAP,
                image_path=str(path),
                intervention_params=result.params,
            )
        )

        # 4. Distractor Added
        result = apply_distractor_intervention(image, evidence_mask, rng=self.rng)
        path = self._save_variant(result.image, item_dir, "distractor_added")
        variants.append(
            InterventionVariant(
                intervention_type=InterventionType.DISTRACTOR_ADDED,
                image_path=str(path),
                intervention_params=result.params,
            )
        )

        # 5. Sham Blur (control)
        sham_bbox = EvidenceFinder._mask_to_bbox(sham_mask)
        result = apply_sham_blur_intervention(image, sham_mask, self.blur_radius)
        path = self._save_variant(result.image, item_dir, "sham_blur")
        variants.append(
            InterventionVariant(
                intervention_type=InterventionType.SHAM_BLUR,
                image_path=str(path),
                intervention_params=result.params,
                sham_region_bbox=sham_bbox,
            )
        )

        # 6. Sham Gray (control)
        result = apply_sham_gray_intervention(image, sham_mask, self.gray_value)
        path = self._save_variant(result.image, item_dir, "sham_gray")
        variants.append(
            InterventionVariant(
                intervention_type=InterventionType.SHAM_GRAY,
                image_path=str(path),
                intervention_params=result.params,
                sham_region_bbox=sham_bbox,
            )
        )

        # Save evidence and sham masks
        cv2.imwrite(str(item_dir / "evidence_mask.png"), evidence_mask)
        cv2.imwrite(str(item_dir / "sham_mask.png"), sham_mask)

        # Update item
        item.variants = variants
        item.evidence_region.mask_path = str(item_dir / "evidence_mask.png")

        return item

    def process_dataset(
        self, items: list[MEIItem], show_progress: bool = True
    ) -> list[MEIItem]:
        """Process all items in the dataset.

        Args:
            items: List of MEI items to process
            show_progress: Whether to show progress bar

        Returns:
            List of updated MEI items with variants
        """
        processed = []
        iterator = tqdm(items, desc="Generating interventions") if show_progress else items

        for item in iterator:
            try:
                processed_item = self.process_item(item)
                processed.append(processed_item)
            except Exception as e:
                logger.error(f"Error processing item {item.item_id}: {e}")
                processed.append(item)

        logger.info(f"Processed {len(processed)}/{len(items)} items successfully")
        return processed

    @staticmethod
    def _save_variant(image: np.ndarray, item_dir: Path, name: str) -> Path:
        """Save an intervention variant image."""
        path = item_dir / f"{name}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return path
