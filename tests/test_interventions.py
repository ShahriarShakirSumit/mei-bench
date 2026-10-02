"""
Tests for the intervention module.

Tests evidence finding, intervention generation, and pipeline integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from mei_benchmark.data.schema import (
    EvidenceRegion,
    InterventionType,
    InterventionVariant,
    MEIItem,
    TaskType,
)
from mei_benchmark.intervention.generator import (
    apply_blur_intervention,
    apply_gray_intervention,
    apply_swap_intervention,
    apply_distractor_intervention,
    apply_sham_blur_intervention,
    apply_sham_gray_intervention,
    apply_blur_dose_response,
    BLUR_DOSE_LEVELS,
)


# ---------- Fixtures ----------

@pytest.fixture
def sample_image():
    """Create a simple test image (BGR, 200x300)."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    # Red region in the center
    img[50:150, 100:200, 2] = 255  # Red in BGR
    # Green region in top-left
    img[10:40, 10:60, 1] = 200
    # Blue region in bottom-right
    img[160:190, 220:280, 0] = 180
    return img


@pytest.fixture
def evidence_mask():
    """Create a mask covering the center red region."""
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[50:150, 100:200] = 255
    return mask


@pytest.fixture
def sham_mask():
    """Create a sham mask covering a non-evidence region."""
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[10:40, 10:60] = 255  # The green region
    return mask


@pytest.fixture
def sample_item():
    """Create a sample MEIItem for testing."""
    return MEIItem(
        item_id="test_001",
        image_path="test/image.jpg",
        question="What color is the central object?",
        valid_answers=["red"],
        task_type=TaskType.ATTRIBUTE_VERIFICATION,
        source_dataset="test",
        evidence_region=EvidenceRegion(
            bbox=[100, 50, 100, 100],
            description="The red central object",
            area_fraction=0.167,
        ),
    )


# ---------- Intervention Tests ----------

class TestBlurIntervention:
    def test_output_shape(self, sample_image, evidence_mask):
        result = apply_blur_intervention(sample_image, evidence_mask)
        assert result.image.shape == sample_image.shape

    def test_output_dtype(self, sample_image, evidence_mask):
        result = apply_blur_intervention(sample_image, evidence_mask)
        assert result.image.dtype == np.uint8

    def test_non_evidence_unchanged(self, sample_image, evidence_mask):
        result = apply_blur_intervention(sample_image, evidence_mask)
        # Pixels outside the evidence mask should be identical
        outside = evidence_mask == 0
        np.testing.assert_array_equal(
            result.image[outside], sample_image[outside]
        )

    def test_evidence_region_modified(self, sample_image, evidence_mask):
        result = apply_blur_intervention(sample_image, evidence_mask)
        inside = evidence_mask > 0
        # The intervened image should differ inside the evidence region
        assert not np.array_equal(result.image[inside], sample_image[inside])

    def test_blur_radius_effect(self, sample_image, evidence_mask):
        # Larger radius should produce more blur
        small = apply_blur_intervention(sample_image, evidence_mask, blur_radius=11)
        large = apply_blur_intervention(sample_image, evidence_mask, blur_radius=51)
        # Both should differ from original
        inside = evidence_mask > 0
        diff_small = np.abs(small.image[inside].astype(float) - sample_image[inside].astype(float)).mean()
        diff_large = np.abs(large.image[inside].astype(float) - sample_image[inside].astype(float)).mean()
        assert diff_large >= diff_small


class TestGrayIntervention:
    def test_output_shape(self, sample_image, evidence_mask):
        result = apply_gray_intervention(sample_image, evidence_mask)
        assert result.image.shape == sample_image.shape

    def test_full_gray(self, sample_image, evidence_mask):
        result = apply_gray_intervention(sample_image, evidence_mask, gray_value=128)
        # Inside the evidence region, all pixels should be gray (128)
        inside = evidence_mask > 0
        assert np.all(result.image[inside] == 128)

    def test_non_evidence_unchanged(self, sample_image, evidence_mask):
        result = apply_gray_intervention(sample_image, evidence_mask)
        outside = evidence_mask == 0
        np.testing.assert_array_equal(result.image[outside], sample_image[outside])

    def test_evidence_modified(self, sample_image, evidence_mask):
        result = apply_gray_intervention(sample_image, evidence_mask)
        inside = evidence_mask > 0
        # The evidence region should differ from original (unless it was already gray)
        assert not np.array_equal(result.image[inside], sample_image[inside])


class TestSwapIntervention:
    def test_output_shape(self, sample_image, evidence_mask):
        result = apply_swap_intervention(sample_image, evidence_mask)
        assert result.image.shape == sample_image.shape

    def test_non_evidence_unchanged(self, sample_image, evidence_mask):
        result = apply_swap_intervention(sample_image, evidence_mask)
        outside = evidence_mask == 0
        np.testing.assert_array_equal(result.image[outside], sample_image[outside])

    def test_evidence_modified(self, sample_image, evidence_mask):
        result = apply_swap_intervention(sample_image, evidence_mask)
        inside = evidence_mask > 0
        assert not np.array_equal(result.image[inside], sample_image[inside])


class TestDistractorIntervention:
    def test_output_shape(self, sample_image, evidence_mask):
        result = apply_distractor_intervention(sample_image, evidence_mask)
        assert result.image.shape == sample_image.shape

    def test_evidence_unchanged(self, sample_image, evidence_mask):
        result = apply_distractor_intervention(sample_image, evidence_mask)
        inside = evidence_mask > 0
        # Evidence region should be mostly unchanged
        # Distractor placement may slightly overlap at edges (up to 10% overlap allowed)
        diff = np.abs(
            result.image[inside].astype(float) - sample_image[inside].astype(float)
        )
        # At most ~5% of evidence pixels should be affected, with small magnitude
        assert (diff > 30).mean() < 0.05, "Too many evidence pixels significantly changed"


class TestShamInterventions:
    def test_sham_blur_output_shape(self, sample_image, evidence_mask, sham_mask):
        result = apply_sham_blur_intervention(sample_image, sham_mask)
        assert result.image.shape == sample_image.shape

    def test_sham_gray_output_shape(self, sample_image, evidence_mask, sham_mask):
        result = apply_sham_gray_intervention(sample_image, sham_mask)
        assert result.image.shape == sample_image.shape

    def test_sham_blur_evidence_unchanged(self, sample_image, evidence_mask, sham_mask):
        result = apply_sham_blur_intervention(sample_image, sham_mask)
        inside_evidence = evidence_mask > 0
        np.testing.assert_array_equal(result.image[inside_evidence], sample_image[inside_evidence])

    def test_sham_modifies_sham_region(self, sample_image, sham_mask):
        result = apply_sham_blur_intervention(sample_image, sham_mask)
        inside_sham = sham_mask > 0
        assert not np.array_equal(result.image[inside_sham], sample_image[inside_sham])


class TestDoseResponse:
    def test_returns_correct_levels(self, sample_image, evidence_mask):
        results = apply_blur_dose_response(sample_image, evidence_mask)
        assert len(results) == len(BLUR_DOSE_LEVELS)
        for result in results:
            assert result.image.shape == sample_image.shape

    def test_monotonic_blur(self, sample_image, evidence_mask):
        results = apply_blur_dose_response(sample_image, evidence_mask)
        inside = evidence_mask > 0
        prev_diff = 0
        for result in results:
            diff = np.abs(
                result.image[inside].astype(float) -
                sample_image[inside].astype(float)
            ).mean()
            assert diff >= prev_diff - 1e-5  # Should increase monotonically
            prev_diff = diff


# ---------- Schema Tests ----------

class TestMEIItem:
    def test_create_item(self, sample_item):
        assert sample_item.item_id == "test_001"
        assert sample_item.task_type == TaskType.ATTRIBUTE_VERIFICATION
        assert sample_item.evidence_region is not None
        assert sample_item.valid_answers == ["red"]

    def test_evidence_region(self, sample_item):
        er = sample_item.evidence_region
        assert er.bbox == [100, 50, 100, 100]
        assert 0 < er.area_fraction < 1

    def test_serialization(self, sample_item):
        json_str = sample_item.model_dump_json()
        restored = MEIItem.model_validate_json(json_str)
        assert restored.item_id == sample_item.item_id
        assert restored.task_type == sample_item.task_type
        assert restored.valid_answers == sample_item.valid_answers
