"""
MEI evaluation metrics.

Implements all causal diagnostic metrics for measuring whether VLLMs
genuinely ground their answers in visual evidence.

Primary Metrics:
    - Causal Sensitivity (CS): P(answer flips | evidence altered)
    - Spurious Invariance (SI): P(answer unchanged | evidence removed)
    - Grounding Gap (GG): Acc(original) - Acc(evidence-removed)
    - Sham Robustness (SR): P(answer unchanged | sham intervention)

Diagnostic Metrics:
    - Grounding Specificity: CS for evidence vs. sham region
    - Prior Reliance Score: CS stratified by answer frequency
    - Intervention Dose-Response: CS as function of intervention strength
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

from mei_benchmark.data.schema import (
    InterventionType,
    MEIItem,
    ModelPrediction,
    ModelPredictionSet,
    TaskType,
)

logger = logging.getLogger(__name__)


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Handles case, whitespace, articles, and common variations.
    """
    answer = answer.strip().lower()
    # Remove articles
    for article in ["a ", "an ", "the "]:
        if answer.startswith(article):
            answer = answer[len(article):]
    # Remove trailing punctuation
    answer = answer.rstrip(".,!?;:")
    # Normalize whitespace
    answer = " ".join(answer.split())
    return answer


def answers_match(pred_answer: str, valid_answers: list[str]) -> bool:
    """Check if predicted answer matches any valid answer."""
    pred_norm = normalize_answer(pred_answer)
    for valid in valid_answers:
        if normalize_answer(valid) == pred_norm:
            return True
    return False


def answer_flipped(
    original_answer: str,
    variant_answer: str,
) -> bool:
    """Check if the answer changed between original and variant."""
    return normalize_answer(original_answer) != normalize_answer(variant_answer)


@dataclass
class MEIMetrics:
    """Container for all MEI evaluation metrics for a single model."""

    model_name: str

    # Primary metrics
    causal_sensitivity_blur: float = 0.0  # CS for blur intervention
    causal_sensitivity_gray: float = 0.0  # CS for gray intervention
    causal_sensitivity_swap: float = 0.0  # CS for swap intervention
    spurious_invariance_blur: float = 0.0  # SI for blur (1 - CS_blur among correct)
    spurious_invariance_gray: float = 0.0  # SI for gray (1 - CS_gray among correct)
    grounding_gap: float = 0.0  # Acc(original) - Acc(evidence_removed)
    sham_robustness_blur: float = 0.0  # SR for sham blur
    sham_robustness_gray: float = 0.0  # SR for sham gray

    # Derived metrics
    grounding_specificity: float = 0.0  # GS = max(GG,0)/max(CS,eps) per paper Eq.7
    original_accuracy: float = 0.0
    evidence_removed_accuracy: float = 0.0

    # Per-task breakdown
    task_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    # Counts
    n_items: int = 0
    n_correct_original: int = 0

    def to_dict(self) -> dict:
        """Convert to flat dictionary for tabulation."""
        result = {
            "model": self.model_name,
            "n_items": self.n_items,
            "original_accuracy": round(self.original_accuracy, 4),
            "CS_blur": round(self.causal_sensitivity_blur, 4),
            "CS_gray": round(self.causal_sensitivity_gray, 4),
            "CS_swap": round(self.causal_sensitivity_swap, 4),
            "SI_blur": round(self.spurious_invariance_blur, 4),
            "SI_gray": round(self.spurious_invariance_gray, 4),
            "GG": round(self.grounding_gap, 4),
            "SR_blur": round(self.sham_robustness_blur, 4),
            "SR_gray": round(self.sham_robustness_gray, 4),
            "GS": round(self.grounding_specificity, 4),
        }
        return result


class MEIEvaluator:
    """Compute all MEI metrics from model predictions.

    Usage:
        evaluator = MEIEvaluator()
        metrics = evaluator.evaluate(items, prediction_sets)
    """

    def evaluate(
        self,
        items: list[MEIItem],
        prediction_sets: list[ModelPredictionSet],
        model_name: Optional[str] = None,
    ) -> MEIMetrics:
        """Compute all metrics for a model's predictions.

        Args:
            items: MEI benchmark items
            prediction_sets: Model predictions (one per item)
            model_name: Override model name (auto-detected from predictions if None)

        Returns:
            MEIMetrics with all computed metrics
        """
        if not prediction_sets:
            raise ValueError("No predictions provided")

        if model_name is None:
            model_name = prediction_sets[0].model_name

        # Index predictions by item_id
        pred_index = {ps.item_id: ps for ps in prediction_sets}

        # Accumulators
        n_items = 0
        n_correct_original = 0
        n_correct_evidence_removed = 0

        # Per-intervention-type count tracking (correct denominators)
        n_has_blur = 0
        n_has_gray = 0
        n_has_swap = 0

        # Answer flip counts (among ALL items with that variant)
        flip_blur = 0
        flip_gray = 0
        flip_swap = 0
        nonflip_sham_blur = 0
        nonflip_sham_gray = 0

        # Among originally correct items (for SI computation per paper)
        n_correct_has_blur = 0
        n_correct_has_gray = 0
        n_correct_has_sham_blur = 0
        n_correct_has_sham_gray = 0
        sham_flip_blur = 0   # flips on sham among originally correct
        sham_flip_gray = 0

        # Answer flip counts among originally correct items (for CS per paper)
        cs_flip_blur = 0  # evidence flips among originally correct
        cs_flip_gray = 0
        cs_flip_swap = 0

        # Sham counts
        n_sham_blur = 0
        n_sham_gray = 0

        # Per-task accumulators
        task_accum: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

        for item in items:
            pset = pred_index.get(item.item_id)
            if pset is None:
                continue

            original_pred = pset.original_prediction
            if original_pred is None:
                continue

            n_items += 1
            original_correct = answers_match(original_pred.answer, item.valid_answers)
            if original_correct:
                n_correct_original += 1

            task_key = item.task_type.value

            # Evaluate each intervention type
            for itype in [
                InterventionType.EVIDENCE_BLUR,
                InterventionType.EVIDENCE_GRAY,
                InterventionType.EVIDENCE_SWAP,
                InterventionType.SHAM_BLUR,
                InterventionType.SHAM_GRAY,
            ]:
                variant_pred = pset.get_prediction(itype)
                if variant_pred is None:
                    continue

                flipped = answer_flipped(original_pred.answer, variant_pred.answer)
                variant_correct = answers_match(variant_pred.answer, item.valid_answers)

                if itype == InterventionType.EVIDENCE_BLUR:
                    n_has_blur += 1
                    if flipped:
                        flip_blur += 1
                    if original_correct:
                        n_correct_has_blur += 1
                        if flipped:
                            cs_flip_blur += 1
                    task_accum[task_key]["cs_blur"].append(1.0 if flipped else 0.0)

                elif itype == InterventionType.EVIDENCE_GRAY:
                    n_has_gray += 1
                    if flipped:
                        flip_gray += 1
                    if original_correct:
                        n_correct_has_gray += 1
                        if flipped:
                            cs_flip_gray += 1
                    if variant_correct:
                        n_correct_evidence_removed += 1
                    task_accum[task_key]["cs_gray"].append(1.0 if flipped else 0.0)

                elif itype == InterventionType.EVIDENCE_SWAP:
                    n_has_swap += 1
                    if flipped:
                        flip_swap += 1
                    if original_correct:
                        if flipped:
                            cs_flip_swap += 1
                    task_accum[task_key]["cs_swap"].append(1.0 if flipped else 0.0)

                elif itype == InterventionType.SHAM_BLUR:
                    n_sham_blur += 1
                    if not flipped:
                        nonflip_sham_blur += 1
                    if original_correct:
                        n_correct_has_sham_blur += 1
                        if flipped:
                            sham_flip_blur += 1
                    task_accum[task_key]["sr_blur"].append(0.0 if flipped else 1.0)

                elif itype == InterventionType.SHAM_GRAY:
                    n_sham_gray += 1
                    if not flipped:
                        nonflip_sham_gray += 1
                    if original_correct:
                        n_correct_has_sham_gray += 1
                        if flipped:
                            sham_flip_gray += 1
                    task_accum[task_key]["sr_gray"].append(0.0 if flipped else 1.0)

            task_accum[task_key]["correct"].append(1.0 if original_correct else 0.0)

        # Compute metrics
        metrics = MEIMetrics(model_name=model_name)
        metrics.n_items = n_items
        metrics.n_correct_original = n_correct_original

        if n_items > 0:
            metrics.original_accuracy = n_correct_original / n_items
            metrics.evidence_removed_accuracy = n_correct_evidence_removed / n_items

        # CS: fraction of originally-correct items where evidence intervention flips answer
        # (per paper Eq. 3: computed over C = {i : f(I_i, Q_i) = A_i*})
        if n_correct_has_blur > 0:
            metrics.causal_sensitivity_blur = cs_flip_blur / n_correct_has_blur
        if n_correct_has_gray > 0:
            metrics.causal_sensitivity_gray = cs_flip_gray / n_correct_has_gray
        if n_correct_original > 0 and cs_flip_swap > 0:
            metrics.causal_sensitivity_swap = cs_flip_swap / n_correct_original

        # SI: fraction of originally-correct items where sham intervention flips answer
        # (per paper Eq. 4)
        if n_correct_has_sham_blur > 0:
            metrics.spurious_invariance_blur = sham_flip_blur / n_correct_has_sham_blur
        if n_correct_has_sham_gray > 0:
            metrics.spurious_invariance_gray = sham_flip_gray / n_correct_has_sham_gray

        # SR: 1 - SI (per paper Eq. 6)
        if n_sham_blur > 0:
            metrics.sham_robustness_blur = nonflip_sham_blur / n_sham_blur
        if n_sham_gray > 0:
            metrics.sham_robustness_gray = nonflip_sham_gray / n_sham_gray

        # GG: CS - SI (per paper Eq. 5)
        metrics.grounding_gap = metrics.causal_sensitivity_blur - metrics.spurious_invariance_blur

        # GS: max(GG, 0) / max(CS, epsilon) (per paper Eq. 7)
        # GS = 1 when all evidence sensitivity is specific (SI = 0)
        # GS = 0 when evidence and sham sensitivity are identical
        epsilon = 1e-8
        metrics.grounding_specificity = max(metrics.grounding_gap, 0.0) / max(metrics.causal_sensitivity_blur, epsilon)

        # Per-task metrics
        for task_key, acc in task_accum.items():
            task_metrics = {}
            for metric_key, values in acc.items():
                if values:
                    task_metrics[metric_key] = float(np.mean(values))
                    task_metrics[f"{metric_key}_n"] = len(values)
            metrics.task_metrics[task_key] = task_metrics

        return metrics

    def evaluate_dose_response(
        self,
        items: list[MEIItem],
        dose_predictions: dict[float, list[ModelPredictionSet]],
    ) -> dict[float, float]:
        """Compute CS at each dose level for dose-response analysis.

        Args:
            items: MEI benchmark items
            dose_predictions: Mapping from dose level to predictions at that dose

        Returns:
            Dict mapping dose level to causal sensitivity
        """
        dose_cs = {}

        for dose, psets in sorted(dose_predictions.items()):
            pred_index = {ps.item_id: ps for ps in psets}

            n_total = 0
            n_flipped = 0

            for item in items:
                pset = pred_index.get(item.item_id)
                if pset is None:
                    continue

                original = pset.original_prediction
                if original is None:
                    continue

                # Get the dose variant prediction (stored as EVIDENCE_BLUR)
                variant = pset.get_prediction(InterventionType.EVIDENCE_BLUR)
                if variant is None:
                    continue

                n_total += 1
                if answer_flipped(original.answer, variant.answer):
                    n_flipped += 1

            dose_cs[dose] = n_flipped / max(n_total, 1)

        return dose_cs


@dataclass
class StratifiedResults:
    """Results stratified by various dimensions."""

    by_task: dict[str, MEIMetrics] = field(default_factory=dict)
    by_difficulty: dict[str, MEIMetrics] = field(default_factory=dict)
    by_evidence_size: dict[str, MEIMetrics] = field(default_factory=dict)
    by_confidence: dict[str, MEIMetrics] = field(default_factory=dict)


class StratifiedEvaluator:
    """Compute MEI metrics stratified by various dimensions."""

    def __init__(self):
        self.base_evaluator = MEIEvaluator()

    def evaluate_stratified(
        self,
        items: list[MEIItem],
        prediction_sets: list[ModelPredictionSet],
        model_name: str,
    ) -> StratifiedResults:
        """Compute metrics stratified by task, difficulty, evidence size, and confidence.

        Args:
            items: MEI benchmark items
            prediction_sets: Model predictions
            model_name: Model name

        Returns:
            StratifiedResults with metrics broken down by each dimension
        """
        pred_index = {ps.item_id: ps for ps in prediction_sets}
        results = StratifiedResults()

        # By task type
        task_groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for item in items:
            ps = pred_index.get(item.item_id)
            if ps:
                task_groups[item.task_type.value][0].append(item)
                task_groups[item.task_type.value][1].append(ps)

        for task, (task_items, task_preds) in task_groups.items():
            results.by_task[task] = self.base_evaluator.evaluate(
                task_items, task_preds, model_name
            )

        # By difficulty
        diff_groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for item in items:
            ps = pred_index.get(item.item_id)
            if ps:
                diff_groups[item.difficulty.value][0].append(item)
                diff_groups[item.difficulty.value][1].append(ps)

        for diff, (diff_items, diff_preds) in diff_groups.items():
            results.by_difficulty[diff] = self.base_evaluator.evaluate(
                diff_items, diff_preds, model_name
            )

        # By evidence region size (small/medium/large)
        size_groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for item in items:
            ps = pred_index.get(item.item_id)
            if ps:
                area = item.evidence_region.area_fraction
                if area < 0.05:
                    size_cat = "small"
                elif area < 0.15:
                    size_cat = "medium"
                else:
                    size_cat = "large"
                size_groups[size_cat][0].append(item)
                size_groups[size_cat][1].append(ps)

        for size_cat, (size_items, size_preds) in size_groups.items():
            results.by_evidence_size[size_cat] = self.base_evaluator.evaluate(
                size_items, size_preds, model_name
            )

        return results


def compute_bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        values: Sample values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        rng: Random number generator

    Returns:
        (mean, ci_lower, ci_upper)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    arr = np.array(values)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_means, 100 * alpha))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha)))

    return float(np.mean(arr)), ci_lower, ci_upper
