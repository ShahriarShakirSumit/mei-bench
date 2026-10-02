"""
Evidence identification pipeline using SAM + CLIP.

Identifies the minimal region in an image that contains the decisive
evidence needed to correctly answer a given question.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class EvidenceCandidate:
    """A candidate evidence region identified by the pipeline."""

    mask: np.ndarray  # Binary mask (H, W), uint8, 255 = evidence
    bbox: list[float]  # [x1, y1, x2, y2]
    relevance_score: float  # CLIP-based relevance to question+answer
    area_fraction: float  # Fraction of image area
    source: str = "sam"  # "sam", "annotation", "manual"


class EvidenceFinder:
    """Find minimal evidence regions using SAM + CLIP.

    Pipeline:
    1. SAM segments the image into candidate regions
    2. CLIP scores each region's relevance to the question+answer
    3. Select the smallest region with high relevance (minimal evidence)
    """

    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_h",
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        device: str = "cuda",
        min_area_fraction: float = 0.005,
        max_area_fraction: float = 0.40,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
    ):
        self.device = device
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh

        self._sam_predictor = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None

    def _load_sam(self) -> None:
        """Lazy-load SAM model."""
        if self._sam_predictor is not None:
            return

        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        if self.sam_checkpoint is None:
            logger.warning(
                "No SAM checkpoint provided. Using annotation-based evidence regions only."
            )
            return

        sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
        sam.to(self.device)
        self._sam_predictor = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=100,
        )
        logger.info(f"SAM model loaded: {self.sam_model_type}")

    def _load_clip(self) -> None:
        """Lazy-load CLIP model."""
        if self._clip_model is not None:
            return

        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name, pretrained=self.clip_pretrained
        )
        tokenizer = open_clip.get_tokenizer(self.clip_model_name)

        self._clip_model = model.to(self.device).eval()
        self._clip_preprocess = preprocess
        self._clip_tokenizer = tokenizer
        logger.info(f"CLIP model loaded: {self.clip_model_name}")

    def find_evidence(
        self,
        image: np.ndarray,
        question: str,
        answer: str,
        annotation_masks: Optional[list[np.ndarray]] = None,
    ) -> list[EvidenceCandidate]:
        """Find minimal evidence regions in the image.

        Args:
            image: BGR image (H, W, 3), uint8
            question: The question about the image
            answer: The correct answer
            annotation_masks: Optional pre-defined masks from dataset annotations

        Returns:
            List of evidence candidates ranked by relevance, smallest first
        """
        candidates = []
        h, w = image.shape[:2]
        total_area = h * w

        # Step 1: Generate candidate regions
        if annotation_masks is not None:
            # Use provided annotation masks
            for i, mask in enumerate(annotation_masks):
                bbox = self._mask_to_bbox(mask)
                area_frac = np.sum(mask > 0) / total_area
                candidates.append(
                    EvidenceCandidate(
                        mask=mask,
                        bbox=bbox,
                        relevance_score=0.0,
                        area_fraction=area_frac,
                        source="annotation",
                    )
                )
        else:
            # Use SAM for automatic segmentation
            self._load_sam()
            if self._sam_predictor is not None:
                sam_masks = self._sam_predictor.generate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                for sam_result in sam_masks:
                    mask = (sam_result["segmentation"].astype(np.uint8) * 255)
                    bbox = list(sam_result["bbox"])  # [x, y, w, h] -> convert
                    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                    area_frac = sam_result["area"] / total_area

                    if self.min_area_fraction <= area_frac <= self.max_area_fraction:
                        candidates.append(
                            EvidenceCandidate(
                                mask=mask,
                                bbox=bbox,
                                relevance_score=0.0,
                                area_fraction=area_frac,
                                source="sam",
                            )
                        )

        if not candidates:
            logger.warning("No candidate regions found, using center crop as fallback")
            center_mask = np.zeros((h, w), dtype=np.uint8)
            ch, cw = h // 4, w // 4
            center_mask[ch : 3 * ch, cw : 3 * cw] = 255
            candidates.append(
                EvidenceCandidate(
                    mask=center_mask,
                    bbox=[float(cw), float(ch), float(3 * cw), float(3 * ch)],
                    relevance_score=0.5,
                    area_fraction=0.25,
                    source="fallback",
                )
            )
            return candidates

        # Step 2: Score candidates by relevance to question+answer using CLIP
        self._load_clip()
        if self._clip_model is not None:
            candidates = self._score_candidates(image, question, answer, candidates)

        # Step 3: Sort by relevance (descending), then by area (ascending for minimality)
        candidates.sort(key=lambda c: (-c.relevance_score, c.area_fraction))

        return candidates

    def _score_candidates(
        self,
        image: np.ndarray,
        question: str,
        answer: str,
        candidates: list[EvidenceCandidate],
    ) -> list[EvidenceCandidate]:
        """Score evidence candidates using CLIP similarity."""
        import torch
        from PIL import Image as PILImage

        # Prepare text query
        text_query = f"{question} {answer}"
        text_tokens = self._clip_tokenizer([text_query]).to(self.device)

        with torch.no_grad():
            text_features = self._clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Score each candidate region
        for candidate in candidates:
            # Extract region from image using mask
            x1, y1, x2, y2 = [int(c) for c in candidate.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            if x2 - x1 < 10 or y2 - y1 < 10:
                candidate.relevance_score = 0.0
                continue

            crop = image[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_crop = PILImage.fromarray(crop_rgb)

            # CLIP encode
            image_input = self._clip_preprocess(pil_crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self._clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity
            similarity = (image_features @ text_features.T).item()
            candidate.relevance_score = float(similarity)

        return candidates

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> list[float]:
        """Convert binary mask to bounding box [x1, y1, x2, y2]."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    @staticmethod
    def create_evidence_mask_from_bbox(
        image_shape: tuple[int, int], bbox: list[float]
    ) -> np.ndarray:
        """Create a binary mask from a bounding box.

        Args:
            image_shape: (height, width) of the image
            bbox: [x1, y1, x2, y2] in pixel coordinates

        Returns:
            Binary mask (H, W), uint8, 255 = evidence region
        """
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = [int(c) for c in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = 255
        return mask

    @staticmethod
    def create_sham_mask(
        image_shape: tuple[int, int],
        evidence_mask: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Create a sham (control) mask with the same area but different location.

        The sham mask has the same total area as the evidence mask but is placed
        in a random location that does NOT overlap with the evidence region.

        Args:
            image_shape: (height, width)
            evidence_mask: The evidence region mask to match area of
            rng: Random number generator for reproducibility

        Returns:
            Binary sham mask (H, W), uint8, 255 = sham region
        """
        if rng is None:
            rng = np.random.default_rng()

        h, w = image_shape
        evidence_area = np.sum(evidence_mask > 0)

        # Get evidence bbox to compute region dimensions
        ys, xs = np.where(evidence_mask > 0)
        if len(xs) == 0:
            return np.zeros((h, w), dtype=np.uint8)

        ev_h = int(ys.max() - ys.min())
        ev_w = int(xs.max() - xs.min())

        # Try to place sham region in a non-overlapping location
        max_attempts = 100
        for _ in range(max_attempts):
            # Random top-left corner
            sx = rng.integers(0, max(1, w - ev_w))
            sy = rng.integers(0, max(1, h - ev_h))

            sham_mask = np.zeros((h, w), dtype=np.uint8)
            sham_mask[sy : sy + ev_h, sx : sx + ev_w] = 255

            # Check overlap with evidence
            overlap = np.logical_and(sham_mask > 0, evidence_mask > 0).sum()
            overlap_ratio = overlap / max(evidence_area, 1)

            if overlap_ratio < 0.1:  # Less than 10% overlap
                return sham_mask

        # Fallback: place at the opposite corner
        ev_cx = (xs.min() + xs.max()) / 2
        ev_cy = (ys.min() + ys.max()) / 2

        if ev_cx < w / 2:
            sx = min(w - ev_w, int(w * 0.7))
        else:
            sx = max(0, int(w * 0.1))

        if ev_cy < h / 2:
            sy = min(h - ev_h, int(h * 0.7))
        else:
            sy = max(0, int(h * 0.1))

        sham_mask = np.zeros((h, w), dtype=np.uint8)
        sham_mask[sy : sy + ev_h, sx : sx + ev_w] = 255
        return sham_mask
