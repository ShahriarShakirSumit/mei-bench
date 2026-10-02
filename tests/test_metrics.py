"""
Tests for the evaluation metrics module.

Tests metric computation, stratification, and bootstrap CIs.
"""

from __future__ import annotations

import numpy as np
import pytest

from mei_benchmark.data.schema import (
    DifficultyLevel,
    EvidenceRegion,
    InterventionType,
    InterventionVariant,
    MEIItem,
    ModelPrediction,
    ModelPredictionSet,
    TaskType,
)
from mei_benchmark.evaluation.metrics import (
    MEIEvaluator,
    MEIMetrics,
    StratifiedEvaluator,
    answers_match,
    compute_bootstrap_ci,
    normalize_answer,
)


# ---------- Fixtures ----------

def _make_item(
    item_id: str,
    correct: str,
    task: TaskType = TaskType.OBJECT_IDENTIFICATION,
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
    area_fraction: float = 0.1,
) -> MEIItem:
    """Helper to create a test MEIItem."""
    return MEIItem(
        item_id=item_id,
        image_path=f"images/{item_id}.jpg",
        question="What is this?",
        valid_answers=[correct],
        task_type=task,
        difficulty=difficulty,
        source_dataset="test",
        evidence_region=EvidenceRegion(
            bbox=[10, 10, 50, 50],
            description="test evidence",
            area_fraction=area_fraction,
        ),
        variants=[
            InterventionVariant(intervention_type=InterventionType.EVIDENCE_BLUR, image_path=f"variants/{item_id}_blur.jpg"),
            InterventionVariant(intervention_type=InterventionType.EVIDENCE_GRAY, image_path=f"variants/{item_id}_gray.jpg"),
            InterventionVariant(intervention_type=InterventionType.SHAM_BLUR, image_path=f"variants/{item_id}_sham_blur.jpg"),
            InterventionVariant(intervention_type=InterventionType.SHAM_GRAY, image_path=f"variants/{item_id}_sham_gray.jpg"),
        ],
    )


def _make_pred(
    item_id: str,
    intervention: str,
    answer: str,
    model: str = "test_model",
) -> ModelPrediction:
    """Helper to create a test prediction."""
    return ModelPrediction(
        item_id=item_id,
        model_name=model,
        intervention_type=intervention,
        answer=answer,
        raw_response=answer,
    )


@pytest.fixture
def perfect_grounding_scenario():
    """Scenario: model always flips on evidence intervention, never on sham.

    This represents a perfectly grounded model.
    """
    items = [_make_item(f"item_{i}", "cat") for i in range(10)]

    pred_sets = []
    for item in items:
        item_preds = [
            _make_pred(item.item_id, "original", "cat"),
            _make_pred(item.item_id, "evidence_blur", "dog"),
            _make_pred(item.item_id, "evidence_gray", "dog"),
            _make_pred(item.item_id, "sham_blur", "cat"),
            _make_pred(item.item_id, "sham_gray", "cat"),
        ]
        pred_sets.append(ModelPredictionSet(
            item_id=item.item_id,
            model_name="test_model",
            predictions=item_preds,
        ))
    return items, pred_sets


@pytest.fixture
def no_grounding_scenario():
    """Scenario: model never flips on anything (memorized answers)."""
    items = [_make_item(f"item_{i}", "cat") for i in range(10)]

    pred_sets = []
    for item in items:
        item_preds = [
            _make_pred(item.item_id, "original", "cat"),
            _make_pred(item.item_id, "evidence_blur", "cat"),
            _make_pred(item.item_id, "evidence_gray", "cat"),
            _make_pred(item.item_id, "sham_blur", "cat"),
            _make_pred(item.item_id, "sham_gray", "cat"),
        ]
        pred_sets.append(ModelPredictionSet(
            item_id=item.item_id,
            model_name="test_model",
            predictions=item_preds,
        ))
    return items, pred_sets


@pytest.fixture
def fragile_scenario():
    """Scenario: model flips on both evidence AND sham (fragile, not grounded)."""
    items = [_make_item(f"item_{i}", "cat") for i in range(10)]

    pred_sets = []
    for item in items:
        item_preds = [
            _make_pred(item.item_id, "original", "cat"),
            _make_pred(item.item_id, "evidence_blur", "dog"),
            _make_pred(item.item_id, "evidence_gray", "dog"),
            _make_pred(item.item_id, "sham_blur", "dog"),
            _make_pred(item.item_id, "sham_gray", "dog"),
        ]
        pred_sets.append(ModelPredictionSet(
            item_id=item.item_id,
            model_name="test_model",
            predictions=item_preds,
        ))
    return items, pred_sets


# ---------- Answer Matching Tests ----------

class TestAnswerMatching:
    def test_exact_match(self):
        assert answers_match("cat", ["cat"])

    def test_case_insensitive(self):
        assert answers_match("Cat", ["cat"])

    def test_whitespace(self):
        assert answers_match("  cat  ", ["cat"])

    def test_period_stripping(self):
        assert answers_match("cat.", ["cat"])

    def test_article_stripping(self):
        assert answers_match("a cat", ["cat"])
        assert answers_match("the cat", ["cat"])

    def test_no_match(self):
        assert not answers_match("cat", ["dog"])

    def test_multiple_valid_answers(self):
        assert answers_match("cat", ["cat", "kitten", "feline"])
        assert answers_match("kitten", ["cat", "kitten", "feline"])
        assert not answers_match("dog", ["cat", "kitten", "feline"])

    def test_normalize(self):
        assert normalize_answer("The Cat.") == "cat"
        assert normalize_answer("  A DOG  ") == "dog"


# ---------- Metric Computation Tests ----------

class TestMEIEvaluator:
    def test_perfect_grounding_metrics(self, perfect_grounding_scenario):
        items, pred_sets = perfect_grounding_scenario
        evaluator = MEIEvaluator()
        metrics = evaluator.evaluate(items, pred_sets)

        assert metrics.original_accuracy == 1.0
        assert metrics.causal_sensitivity_blur == 1.0  # All flipped
        assert metrics.causal_sensitivity_gray == 1.0
        assert metrics.sham_robustness_blur == 1.0  # None flipped on sham
        assert metrics.sham_robustness_gray == 1.0
        # GG = CS - SI = 1.0 - 0.0 = 1.0 (perfect grounding: evidence flips, sham doesn't)
        assert metrics.grounding_gap == pytest.approx(1.0)
        assert metrics.spurious_invariance_blur == 0.0  # No sham flips among correct
        # GS = max(GG, 0) / max(CS, eps) = 1.0 / 1.0 = 1.0
        assert metrics.grounding_specificity == pytest.approx(1.0)

    def test_no_grounding_metrics(self, no_grounding_scenario):
        items, pred_sets = no_grounding_scenario
        evaluator = MEIEvaluator()
        metrics = evaluator.evaluate(items, pred_sets)

        assert metrics.original_accuracy == 1.0
        assert metrics.causal_sensitivity_blur == 0.0  # No flips
        assert metrics.sham_robustness_blur == 1.0  # No flips on sham
        assert metrics.grounding_gap == 0.0  # Both accuracies are 1.0
        assert metrics.grounding_specificity == 0.0  # No CS

    def test_fragile_metrics(self, fragile_scenario):
        items, pred_sets = fragile_scenario
        evaluator = MEIEvaluator()
        metrics = evaluator.evaluate(items, pred_sets)

        assert metrics.original_accuracy == 1.0
        assert metrics.causal_sensitivity_blur == 1.0
        assert metrics.sham_robustness_blur == 0.0  # All flipped on sham
        assert metrics.sham_robustness_gray == 0.0
        # GG = CS - SI = 1.0 - 1.0 = 0.0 (fragile model, no specific grounding)
        assert metrics.grounding_gap == pytest.approx(0.0)
        # GS = max(GG, 0) / max(CS, eps) = 0.0 / 1.0 = 0.0
        assert metrics.grounding_specificity == pytest.approx(0.0)

    def test_metrics_to_dict(self, perfect_grounding_scenario):
        items, pred_sets = perfect_grounding_scenario
        evaluator = MEIEvaluator()
        metrics = evaluator.evaluate(items, pred_sets)

        d = metrics.to_dict()
        assert "CS_blur" in d
        assert "GG" in d
        assert "GS" in d
        assert d["model"] == "test_model"

    def test_n_items(self, perfect_grounding_scenario):
        items, pred_sets = perfect_grounding_scenario
        evaluator = MEIEvaluator()
        metrics = evaluator.evaluate(items, pred_sets)
        assert metrics.n_items == 10


class TestStratifiedEvaluator:
    def test_stratify_by_task(self):
        """Test that stratification separates by task type."""
        items = [
            _make_item("item_0", "cat", task=TaskType.OBJECT_IDENTIFICATION),
            _make_item("item_1", "cat", task=TaskType.OBJECT_IDENTIFICATION),
            _make_item("item_2", "red", task=TaskType.ATTRIBUTE_VERIFICATION),
            _make_item("item_3", "red", task=TaskType.ATTRIBUTE_VERIFICATION),
        ]

        pred_sets = []
        for item in items:
            correct = item.valid_answers[0]
            item_preds = [
                _make_pred(item.item_id, "original", correct),
                _make_pred(item.item_id, "evidence_blur", "wrong"),
                _make_pred(item.item_id, "sham_blur", correct),
            ]
            pred_sets.append(ModelPredictionSet(
                item_id=item.item_id,
                model_name="test_model",
                predictions=item_preds,
            ))

        strat = StratifiedEvaluator()
        results = strat.evaluate_stratified(items, pred_sets, "test_model")

        assert "object_identification" in results.by_task
        assert "attribute_verification" in results.by_task
        assert results.by_task["object_identification"].n_items == 2
        assert results.by_task["attribute_verification"].n_items == 2


class TestBootstrapCI:
    def test_bootstrap_returns_ci(self):
        rng = np.random.default_rng(42)
        values = rng.random(100).tolist()
        mean, low, high = compute_bootstrap_ci(values, n_bootstrap=500)
        assert low <= mean <= high
        assert low >= 0

    def test_bootstrap_constant(self):
        values = [0.5] * 50
        mean, low, high = compute_bootstrap_ci(values, n_bootstrap=500)
        assert abs(mean - 0.5) < 0.01
        assert abs(low - 0.5) < 0.01
        assert abs(high - 0.5) < 0.01
