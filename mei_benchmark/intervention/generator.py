"""
Intervention generators for MEI benchmark.

Implements all five types of minimal evidence interventions:
1. Evidence-Removed (Blur) — Gaussian blur inside evidence mask
2. Evidence-Removed (Gray) — Replace evidence region with gray
3. Evidence-Swapped — Replace key region with contradictory content (optional)
4. Distractor-Added — Insert visually similar distractor elsewhere
5. Sham (Control) — Blur/gray a random non-evidence region of same area
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InterventionResult:
    """Result of applying an intervention to an image."""

    image: np.ndarray  # BGR image (H, W, 3)
    intervention_type: str
    params: dict
    mask_used: np.ndarray  # The mask that was applied


def apply_blur_intervention(
    image: np.ndarray,
    mask: np.ndarray,
    blur_radius: int = 51,
    blur_sigma: float = 0.0,
) -> InterventionResult:
    """Apply Gaussian blur inside the evidence mask.

    This tests whether the model relies on precise visual details
    in the evidence region. The blur makes fine details unreadable
    while preserving rough structure and color.

    Args:
        image: BGR image (H, W, 3)
        mask: Binary mask (H, W), uint8, 255 = region to blur
        blur_radius: Gaussian blur kernel size (must be odd)
        blur_sigma: Gaussian blur sigma (0 = auto)

    Returns:
        InterventionResult with blurred image
    """
    # Ensure odd kernel size
    blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1

    # Apply Gaussian blur to entire image
    blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), blur_sigma)

    # Composite: use blurred pixels inside mask, original outside
    mask_3ch = np.stack([mask, mask, mask], axis=-1).astype(np.float32) / 255.0
    result = (blurred.astype(np.float32) * mask_3ch +
              image.astype(np.float32) * (1 - mask_3ch))
    result = np.clip(result, 0, 255).astype(np.uint8)

    return InterventionResult(
        image=result,
        intervention_type="evidence_blur",
        params={"blur_radius": blur_radius, "blur_sigma": blur_sigma},
        mask_used=mask,
    )


def apply_gray_intervention(
    image: np.ndarray,
    mask: np.ndarray,
    gray_value: int = 128,
) -> InterventionResult:
    """Replace evidence region with uniform gray.

    This is a stronger intervention than blur — it completely removes
    all visual information from the evidence region.

    Args:
        image: BGR image (H, W, 3)
        mask: Binary mask (H, W), uint8, 255 = region to gray out
        gray_value: Gray intensity value (0-255)

    Returns:
        InterventionResult with gray-masked image
    """
    result = image.copy()

    # Replace masked region with gray
    mask_bool = mask > 0
    result[mask_bool] = gray_value

    return InterventionResult(
        image=result,
        intervention_type="evidence_gray",
        params={"gray_value": gray_value},
        mask_used=mask,
    )


def apply_swap_intervention(
    image: np.ndarray,
    mask: np.ndarray,
    replacement_image: Optional[np.ndarray] = None,
    replacement_region: Optional[np.ndarray] = None,
) -> InterventionResult:
    """Swap evidence region with contradictory visual content.

    This tests whether the model's answer tracks the visual evidence.
    If the evidence is swapped with contradictory content and the model
    still gives the same answer, it's not grounding properly.

    When no replacement is provided, a simple color inversion + texture
    scramble is used as a default swap strategy.

    Args:
        image: BGR image (H, W, 3)
        mask: Binary mask (H, W), uint8, 255 = region to swap
        replacement_image: Optional full image to take replacement content from
        replacement_region: Optional direct replacement content (same size as mask bbox)

    Returns:
        InterventionResult with swapped image
    """
    result = image.copy()
    mask_bool = mask > 0

    if replacement_region is not None:
        # Direct replacement (e.g., from inpainting or another image)
        ys, xs = np.where(mask_bool)
        if len(xs) > 0:
            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1
            region_h, region_w = y2 - y1, x2 - x1

            # Resize replacement to fit
            replacement_resized = cv2.resize(replacement_region, (region_w, region_h))
            submask = mask[y1:y2, x1:x2] > 0
            result[y1:y2, x1:x2][submask] = replacement_resized[submask]

    elif replacement_image is not None:
        # Take corresponding region from a different image
        ys, xs = np.where(mask_bool)
        if len(xs) > 0:
            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1

            rep_h, rep_w = replacement_image.shape[:2]
            # Scale coordinates to replacement image
            sy1 = int(y1 * rep_h / image.shape[0])
            sy2 = int(y2 * rep_h / image.shape[0])
            sx1 = int(x1 * rep_w / image.shape[1])
            sx2 = int(x2 * rep_w / image.shape[1])

            rep_crop = replacement_image[sy1:sy2, sx1:sx2]
            if rep_crop.shape[0] > 0 and rep_crop.shape[1] > 0:
                rep_crop = cv2.resize(rep_crop, (x2 - x1, y2 - y1))
                submask = mask[y1:y2, x1:x2] > 0
                result[y1:y2, x1:x2][submask] = rep_crop[submask]

    else:
        # Default: invert colors + add noise in the evidence region
        inverted = cv2.bitwise_not(image)
        # Add texture scramble (shuffle blocks)
        ys, xs = np.where(mask_bool)
        if len(xs) > 0:
            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1
            region = inverted[y1:y2, x1:x2].copy()

            # Block shuffle for more natural-looking swap
            bh, bw = max(1, region.shape[0] // 4), max(1, region.shape[1] // 4)
            for i in range(0, region.shape[0] - bh, bh):
                for j in range(0, region.shape[1] - bw, bw):
                    ri = np.random.randint(0, max(1, region.shape[0] - bh))
                    rj = np.random.randint(0, max(1, region.shape[1] - bw))
                    region[i:i+bh, j:j+bw], region[ri:ri+bh, rj:rj+bw] = (
                        region[ri:ri+bh, rj:rj+bw].copy(),
                        region[i:i+bh, j:j+bw].copy(),
                    )

            submask = mask[y1:y2, x1:x2] > 0
            result[y1:y2, x1:x2][submask] = region[submask]

    return InterventionResult(
        image=result,
        intervention_type="evidence_swap",
        params={"method": "replacement" if replacement_region is not None else "invert_shuffle"},
        mask_used=mask,
    )


def apply_distractor_intervention(
    image: np.ndarray,
    evidence_mask: np.ndarray,
    distractor_patch: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> InterventionResult:
    """Insert a visually similar distractor elsewhere in the image.

    Tests whether the model is robust to confounders — visually similar
    objects that could mislead the model into using the wrong evidence.

    Args:
        image: BGR image (H, W, 3)
        evidence_mask: Binary mask of the evidence region
        distractor_patch: Optional patch to insert. If None, copies + augments evidence region
        rng: Random number generator

    Returns:
        InterventionResult with distractor added
    """
    if rng is None:
        rng = np.random.default_rng()

    result = image.copy()
    h, w = image.shape[:2]

    # Extract evidence region
    ys, xs = np.where(evidence_mask > 0)
    if len(xs) == 0:
        return InterventionResult(
            image=result,
            intervention_type="distractor_added",
            params={"distractor_placed": False},
            mask_used=evidence_mask,
        )

    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    ev_h, ev_w = y2 - y1, x2 - x1

    if distractor_patch is None:
        # Copy the evidence region and augment it
        distractor_patch = image[y1:y2, x1:x2].copy()
        # Apply slight augmentation (flip + brightness shift)
        distractor_patch = cv2.flip(distractor_patch, 1)  # Horizontal flip
        brightness = rng.integers(-30, 30)
        distractor_patch = np.clip(
            distractor_patch.astype(np.int16) + brightness, 0, 255
        ).astype(np.uint8)

    # Find placement location (non-overlapping with evidence)
    max_attempts = 50
    placed = False
    for _ in range(max_attempts):
        px = rng.integers(0, max(1, w - ev_w))
        py = rng.integers(0, max(1, h - ev_h))

        # Check overlap with evidence
        test_mask = np.zeros((h, w), dtype=bool)
        test_mask[py : py + ev_h, px : px + ev_w] = True
        overlap = np.logical_and(test_mask, evidence_mask > 0).sum()

        if overlap < 0.1 * (ev_h * ev_w):
            # Place distractor
            patch_resized = cv2.resize(distractor_patch, (ev_w, ev_h))
            # Alpha blend at edges for natural appearance
            alpha_mask = np.ones((ev_h, ev_w), dtype=np.float32)
            border = min(5, ev_h // 4, ev_w // 4)
            if border > 0:
                for b in range(border):
                    factor = (b + 1) / border
                    alpha_mask[b, :] *= factor
                    alpha_mask[-(b + 1), :] *= factor
                    alpha_mask[:, b] *= factor
                    alpha_mask[:, -(b + 1)] *= factor

            alpha_3ch = np.stack([alpha_mask] * 3, axis=-1)
            target = result[py : py + ev_h, px : px + ev_w].astype(np.float32)
            blended = (patch_resized.astype(np.float32) * alpha_3ch +
                       target * (1 - alpha_3ch))
            result[py : py + ev_h, px : px + ev_w] = np.clip(blended, 0, 255).astype(np.uint8)
            placed = True
            break

    distractor_mask = np.zeros((h, w), dtype=np.uint8)
    if placed:
        distractor_mask[py : py + ev_h, px : px + ev_w] = 255

    return InterventionResult(
        image=result,
        intervention_type="distractor_added",
        params={"distractor_placed": placed},
        mask_used=distractor_mask,
    )


def apply_sham_blur_intervention(
    image: np.ndarray,
    sham_mask: np.ndarray,
    blur_radius: int = 51,
    blur_sigma: float = 0.0,
) -> InterventionResult:
    """Apply blur to a sham (non-evidence) region as a control.

    Identical to evidence blur but applied to a random region.
    Expected result: model answer should NOT change (sham robustness).

    Args:
        image: BGR image (H, W, 3)
        sham_mask: Binary mask of the sham region
        blur_radius: Gaussian blur kernel size
        blur_sigma: Gaussian blur sigma

    Returns:
        InterventionResult with sham-blurred image
    """
    result = apply_blur_intervention(image, sham_mask, blur_radius, blur_sigma)
    result.intervention_type = "sham_blur"
    return result


def apply_sham_gray_intervention(
    image: np.ndarray,
    sham_mask: np.ndarray,
    gray_value: int = 128,
) -> InterventionResult:
    """Apply gray masking to a sham (non-evidence) region as a control.

    Identical to evidence gray-out but applied to a random region.
    Expected result: model answer should NOT change (sham robustness).

    Args:
        image: BGR image (H, W, 3)
        sham_mask: Binary mask of the sham region
        gray_value: Gray intensity value

    Returns:
        InterventionResult with sham-grayed image
    """
    result = apply_gray_intervention(image, sham_mask, gray_value)
    result.intervention_type = "sham_gray"
    return result


# Dose-response: varying intervention strength
BLUR_DOSE_LEVELS = [11, 21, 31, 51, 71, 101]  # Increasing blur radius
GRAY_ALPHA_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]  # Partial to full gray replacement


def apply_blur_dose_response(
    image: np.ndarray,
    mask: np.ndarray,
    dose_levels: Optional[list[int]] = None,
) -> list[InterventionResult]:
    """Generate dose-response curve for blur intervention.

    Applies increasing blur strength to measure graded causal effect.
    Analogous to dose-response curves in pharmacology.

    Args:
        image: BGR image (H, W, 3)
        mask: Binary mask (H, W)
        dose_levels: List of blur radii to try

    Returns:
        List of InterventionResults at each dose level
    """
    if dose_levels is None:
        dose_levels = BLUR_DOSE_LEVELS

    results = []
    for radius in dose_levels:
        result = apply_blur_intervention(image, mask, blur_radius=radius)
        result.params["dose_level"] = radius
        results.append(result)

    return results


def apply_gray_dose_response(
    image: np.ndarray,
    mask: np.ndarray,
    alpha_levels: Optional[list[float]] = None,
) -> list[InterventionResult]:
    """Generate dose-response curve for gray-out intervention.

    Applies increasing gray intensity (partial transparency) to measure
    graded causal effect.

    Args:
        image: BGR image (H, W, 3)
        mask: Binary mask (H, W)
        alpha_levels: List of blending alphas (0=original, 1=fully gray)

    Returns:
        List of InterventionResults at each alpha level
    """
    if alpha_levels is None:
        alpha_levels = GRAY_ALPHA_LEVELS

    results = []
    gray_value = 128

    for alpha in alpha_levels:
        result_img = image.copy()
        mask_bool = mask > 0
        gray_color = np.array([gray_value, gray_value, gray_value], dtype=np.float32)
        blended = (
            image[mask_bool].astype(np.float32) * (1 - alpha) +
            gray_color * alpha
        )
        result_img[mask_bool] = np.clip(blended, 0, 255).astype(np.uint8)

        results.append(
            InterventionResult(
                image=result_img,
                intervention_type="evidence_gray",
                params={"gray_value": gray_value, "alpha": alpha, "dose_level": alpha},
                mask_used=mask,
            )
        )

    return results
