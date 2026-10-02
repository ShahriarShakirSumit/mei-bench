"""
Visualization utilities for MEI benchmark results.

Generates publication-quality figures for the paper:
- Radar charts comparing models across metrics
- Dose-response curves
- Stratified bar charts by task/difficulty/evidence size
- Intervention example grids
- Grounding specificity scatter plots
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mei_benchmark.evaluation.metrics import MEIMetrics, StratifiedResults

matplotlib.use("Agg")  # Non-interactive backend for HPC
logger = logging.getLogger(__name__)

# Publication-quality defaults
FIGSIZE_SINGLE = (6, 4)
FIGSIZE_WIDE = (10, 4)
FIGSIZE_DOUBLE = (12, 5)
DPI = 300
FONT_SIZE = 11

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "legend.fontsize": FONT_SIZE - 1,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "font.family": "serif",
    "figure.dpi": DPI,
})


def plot_model_comparison_bar(
    all_metrics: list[MEIMetrics],
    output_path: str | Path,
    metric_keys: Optional[list[str]] = None,
) -> None:
    """Bar chart comparing multiple models across key metrics.

    Args:
        all_metrics: List of MEIMetrics for each model
        output_path: Path to save the figure
        metric_keys: Which metrics to include (default: core metrics)
    """
    if metric_keys is None:
        metric_keys = [
            "original_accuracy",
            "CS_blur",
            "CS_gray",
            "SI_blur",
            "SR_blur",
            "GG",
        ]

    labels = {
        "original_accuracy": "Orig. Acc.",
        "CS_blur": "CS (Blur)",
        "CS_gray": "CS (Gray)",
        "CS_swap": "CS (Swap)",
        "SI_blur": "SI (Blur)",
        "SI_gray": "SI (Gray)",
        "SR_blur": "SR (Blur)",
        "SR_gray": "SR (Gray)",
        "GG": "Grounding Gap",
        "GS": "Grounding Spec.",
    }

    model_names = [m.model_name for m in all_metrics]
    n_models = len(model_names)
    n_metrics = len(metric_keys)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    colors = sns.color_palette("Set2", n_models)

    for i, metrics in enumerate(all_metrics):
        metrics_dict = metrics.to_dict()
        values = [metrics_dict.get(k, 0.0) for k in metric_keys]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=metrics.model_name, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(k, k) for k in metric_keys], rotation=30, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("MEI Benchmark: Model Comparison")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved model comparison bar chart to {output_path}")


def plot_radar_chart(
    all_metrics: list[MEIMetrics],
    output_path: str | Path,
) -> None:
    """Radar/spider chart comparing models across all dimensions.

    Args:
        all_metrics: List of MEIMetrics for each model
        output_path: Path to save the figure
    """
    metric_keys = [
        "original_accuracy",
        "CS_blur",
        "CS_gray",
        "SI_blur",
        "SR_blur",
        "GG",
        "GS",
    ]
    labels = [
        "Accuracy",
        "CS (Blur)\n↑ better grounding",
        "CS (Gray)\n↑ better grounding",
        "SI (Blur)\n↓ better",
        "SR (Blur)\n↑ better",
        "Grounding\nGap",
        "Grounding\nSpecificity",
    ]

    n_metrics = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the plot

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette("Set2", len(all_metrics))

    for i, metrics in enumerate(all_metrics):
        metrics_dict = metrics.to_dict()
        values = [metrics_dict.get(k, 0.0) for k in metric_keys]
        values += values[:1]  # Close the plot
        ax.plot(angles, values, "o-", color=colors[i], linewidth=2, label=metrics.model_name)
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=FONT_SIZE - 2)
    ax.set_ylim(0, 1.2)
    ax.set_title("MEI Diagnostic Profile", y=1.08, fontsize=FONT_SIZE + 2)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved radar chart to {output_path}")


def plot_dose_response(
    dose_cs_curves: dict[str, dict[float, float]],
    output_path: str | Path,
    x_label: str = "Blur Radius",
    y_label: str = "Causal Sensitivity",
) -> None:
    """Plot dose-response curves for multiple models.

    Analogous to dose-response curves in pharmacology.

    Args:
        dose_cs_curves: {model_name: {dose_level: CS_value}}
        output_path: Path to save the figure
        x_label: X-axis label
        y_label: Y-axis label
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    colors = sns.color_palette("Set2", len(dose_cs_curves))

    for i, (model_name, curve) in enumerate(dose_cs_curves.items()):
        doses = sorted(curve.keys())
        cs_values = [curve[d] for d in doses]
        ax.plot(doses, cs_values, "o-", color=colors[i], linewidth=2, label=model_name)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Dose-Response: Evidence Blur Strength vs Causal Sensitivity")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved dose-response curve to {output_path}")


def plot_task_stratified_bar(
    stratified: StratifiedResults,
    output_path: str | Path,
    metric_key: str = "CS_blur",
) -> None:
    """Bar chart showing a single metric across task types.

    Args:
        stratified: StratifiedResults from StratifiedEvaluator
        output_path: Path to save the figure
        metric_key: Which metric to plot
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    tasks = sorted(stratified.by_task.keys())
    values = [stratified.by_task[t].to_dict().get(metric_key, 0.0) for t in tasks]
    counts = [stratified.by_task[t].n_items for t in tasks]

    colors = sns.color_palette("muted", len(tasks))
    bars = ax.bar(range(len(tasks)), values, color=colors)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE - 2,
        )

    task_labels = [t.replace("_", "\n") for t in tasks]
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(task_labels)
    ax.set_ylabel(metric_key)
    ax.set_title(f"{metric_key} by Task Type")
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved task-stratified bar chart to {output_path}")


def plot_grounding_specificity_scatter(
    all_metrics: list[MEIMetrics],
    output_path: str | Path,
) -> None:
    """Scatter plot: CS (evidence) vs. CS (sham) for each model.

    Models ABOVE the diagonal have specific grounding.
    Models ON the diagonal are just fragile to any masking.

    Args:
        all_metrics: List of MEIMetrics for each model
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    cs_evidence = [m.causal_sensitivity_blur for m in all_metrics]
    cs_sham = [1 - m.sham_robustness_blur for m in all_metrics]
    names = [m.model_name for m in all_metrics]

    colors = sns.color_palette("Set2", len(all_metrics))

    for i, (ce, cs, name) in enumerate(zip(cs_evidence, cs_sham, names)):
        ax.scatter(cs, ce, s=100, color=colors[i], zorder=5)
        ax.annotate(name, (cs, ce), textcoords="offset points",
                    xytext=(8, 5), fontsize=FONT_SIZE - 2)

    # Diagonal line (no specificity)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="No specificity")
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color="green")
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.05, color="red")

    ax.set_xlabel("CS (Sham) — Image fragility")
    ax.set_ylabel("CS (Evidence) — Causal sensitivity")
    ax.set_title("Grounding Specificity: Evidence vs Sham Sensitivity")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved grounding specificity scatter to {output_path}")


def plot_intervention_examples(
    original: np.ndarray,
    variants: dict[str, np.ndarray],
    evidence_mask: np.ndarray,
    output_path: str | Path,
    question: str = "",
    answer: str = "",
) -> None:
    """Create a grid showing original + all intervention variants.

    Args:
        original: Original BGR image
        variants: {intervention_name: intervened BGR image}
        evidence_mask: Binary mask of the evidence region
        output_path: Path to save the figure
        question: The question (for title)
        answer: The correct answer (for title)
    """
    import cv2

    n_images = 1 + len(variants)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    # Show original with evidence mask overlay
    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    overlay = orig_rgb.copy()
    overlay[evidence_mask > 0] = [255, 0, 0]  # Red overlay
    blended = cv2.addWeighted(orig_rgb, 0.7, overlay, 0.3, 0)

    axes[0].imshow(blended)
    axes[0].set_title("Original + Evidence", fontsize=FONT_SIZE - 1)
    axes[0].axis("off")

    # Show each variant
    for i, (name, variant_img) in enumerate(variants.items(), 1):
        if i < len(axes):
            variant_rgb = cv2.cvtColor(variant_img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(variant_rgb)
            axes[i].set_title(name.replace("_", " ").title(), fontsize=FONT_SIZE - 1)
            axes[i].axis("off")

    # Hide unused axes
    for j in range(n_images, len(axes)):
        axes[j].axis("off")

    title = f"Q: {question}" if question else "Intervention Examples"
    if answer:
        title += f"\nA: {answer}"
    fig.suptitle(title, fontsize=FONT_SIZE, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved intervention example grid to {output_path}")


def generate_latex_table(
    all_metrics: list[MEIMetrics],
    output_path: str | Path,
    caption: str = "MEI Benchmark Results",
) -> None:
    """Generate a LaTeX table of results.

    Args:
        all_metrics: List of MEIMetrics for each model
        output_path: Path to save the .tex file
        caption: Table caption
    """
    headers = [
        "Model", "Acc.", "CS↑ (Blur)", "CS↑ (Gray)", "SI↓ (Blur)",
        "SR↑ (Blur)", "GG↑", "GS↑"
    ]
    metric_keys = [
        "original_accuracy", "CS_blur", "CS_gray", "SI_blur",
        "SR_blur", "GG", "GS"
    ]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:mei_results}",
        "\\begin{tabular}{l" + "c" * len(metric_keys) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]

    for metrics in all_metrics:
        d = metrics.to_dict()
        values = [d.get(k, 0.0) for k in metric_keys]
        row = f"{metrics.model_name} & " + " & ".join(f"{v:.3f}" for v in values)
        lines.append(row + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved LaTeX table to {output_path}")
