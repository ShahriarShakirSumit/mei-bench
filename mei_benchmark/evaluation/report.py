"""
Report generation module for MEI benchmark.

Generates comprehensive evaluation reports in Markdown and LaTeX formats,
including analysis, figures, and statistical tests.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from mei_benchmark.evaluation.metrics import MEIMetrics, StratifiedResults

logger = logging.getLogger(__name__)


def generate_markdown_report(
    all_metrics: list[MEIMetrics],
    stratified: Optional[dict[str, StratifiedResults]] = None,
    output_path: str | Path = "outputs/report.md",
    figures_dir: Optional[str | Path] = None,
) -> str:
    """Generate a comprehensive Markdown report.

    Args:
        all_metrics: List of MEIMetrics for each model
        stratified: Optional {model_name: StratifiedResults}
        output_path: Path to save the report
        figures_dir: Directory containing generated figures

    Returns:
        Markdown content string
    """
    lines = [
        "# MEI Benchmark Evaluation Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Models evaluated:** {len(all_metrics)}",
        "",
        "---",
        "",
        "## 1. Overview",
        "",
        "The Minimal Evidence Intervention (MEI) benchmark evaluates whether",
        "Vision-Language Models actually ground their answers in visual evidence.",
        "It does this through targeted interventions on the *minimal evidence*",
        "needed to answer each question, combined with sham controls.",
        "",
        "### Key Metrics",
        "",
        "| Metric | Abbreviation | Ideal Value | Interpretation |",
        "|--------|-------------|-------------|----------------|",
        "| Causal Sensitivity | CS | High (→1.0) | Model relies on evidence |",
        "| Sham Invariance | SI | Low (→0.0) | Model robust to non-evidence changes |",
        "| Grounding Gap | GG | High (→1.0) | CS - SI: net causal grounding |",
        "| Sham Robustness | SR | High (→1.0) | 1 - SI: robustness score |",
        "| Grounding Specificity | GS | High (→1.0) | GG/CS: specificity of grounding |",
        "",
        "---",
        "",
        "## 2. Main Results",
        "",
    ]

    # Results table
    headers = ["Model", "Acc.", "CS↑(Blur)", "CS↑(Gray)", "SI↓(Blur)", "GG↑", "SR↑", "GS↑"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for m in sorted(all_metrics, key=lambda x: x.grounding_gap, reverse=True):
        row = [
            m.model_name,
            f"{m.original_accuracy:.3f}",
            f"{m.causal_sensitivity_blur:.3f}",
            f"{m.causal_sensitivity_gray:.3f}",
            f"{m.spurious_invariance_blur:.3f}",
            f"{m.grounding_gap:.3f}",
            f"{m.sham_robustness_blur:.3f}",
            f"{m.grounding_specificity:.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", ""])

    # Model-by-model analysis
    lines.extend([
        "## 3. Model Analysis",
        "",
    ])

    for m in all_metrics:
        lines.extend([
            f"### {m.model_name}",
            "",
            f"- **Original Accuracy:** {m.original_accuracy:.3f}",
            f"- **Causal Sensitivity (Blur):** {m.causal_sensitivity_blur:.3f}",
            f"- **Causal Sensitivity (Gray):** {m.causal_sensitivity_gray:.3f}",
            f"- **Spurious Invariance (Blur):** {m.spurious_invariance_blur:.3f}",
            f"- **Grounding Gap:** {m.grounding_gap:.3f}",
            f"- **Grounding Specificity:** {m.grounding_specificity:.3f}",
            "",
        ])

        # Interpretation
        if m.grounding_gap > 0.3:
            lines.append(f"✅ {m.model_name} shows **strong causal grounding** (GG={m.grounding_gap:.3f})")
        elif m.grounding_gap > 0.1:
            lines.append(f"⚠️ {m.model_name} shows **moderate causal grounding** (GG={m.grounding_gap:.3f})")
        else:
            lines.append(f"❌ {m.model_name} shows **weak causal grounding** (GG={m.grounding_gap:.3f})")

        if m.grounding_specificity > 0.7:
            lines.append(f"✅ Grounding is **highly specific** to evidence (GS={m.grounding_specificity:.3f})")
        elif m.spurious_invariance_blur > 0.3:
            lines.append(f"⚠️ Model is **fragile** to image perturbations (SI={m.spurious_invariance_blur:.3f})")

        lines.extend(["", ""])

    # Stratified results
    if stratified:
        lines.extend([
            "## 4. Stratified Analysis",
            "",
        ])

        for model_name, strat_result in stratified.items():
            lines.extend([
                f"### {model_name} - By Task Type",
                "",
                "| Task | n | CS(Blur) | SI(Blur) | GG | GS |",
                "|------|---|----------|----------|----|----|",
            ])

            for task, task_m in sorted(strat_result.by_task.items()):
                row = [
                    task,
                    str(task_m.n_items),
                    f"{task_m.causal_sensitivity_blur:.3f}",
                    f"{task_m.spurious_invariance_blur:.3f}",
                    f"{task_m.grounding_gap:.3f}",
                    f"{task_m.grounding_specificity:.3f}",
                ]
                lines.append("| " + " | ".join(row) + " |")

            lines.extend(["", ""])

    # Figures
    if figures_dir:
        fig_dir = Path(figures_dir)
        lines.extend([
            "## 5. Figures",
            "",
        ])

        figure_files = {
            "model_comparison": "Model Comparison Bar Chart",
            "radar_chart": "Diagnostic Profile (Radar)",
            "grounding_scatter": "Grounding Specificity (CS-evidence vs CS-sham)",
        }

        for fig_key, fig_title in figure_files.items():
            for ext in ["pdf", "png", "svg"]:
                fig_path = fig_dir / f"fig*{fig_key}*.{ext}"
                matches = list(fig_dir.glob(f"*{fig_key}*.{ext}"))
                if matches:
                    lines.extend([
                        f"### {fig_title}",
                        f"![{fig_title}]({matches[0]})",
                        "",
                    ])
                    break

    # Summary
    lines.extend([
        "---",
        "",
        "## Summary",
        "",
    ])

    if all_metrics:
        best = max(all_metrics, key=lambda m: m.grounding_gap)
        worst = min(all_metrics, key=lambda m: m.grounding_gap)
        lines.extend([
            f"- **Best grounded model:** {best.model_name} (GG={best.grounding_gap:.3f})",
            f"- **Least grounded model:** {worst.model_name} (GG={worst.grounding_gap:.3f})",
            "",
        ])

        # Overall finding
        avg_gg = sum(m.grounding_gap for m in all_metrics) / len(all_metrics)
        avg_gs = sum(m.grounding_specificity for m in all_metrics) / len(all_metrics)
        lines.extend([
            f"- **Average Grounding Gap:** {avg_gg:.3f}",
            f"- **Average Grounding Specificity:** {avg_gs:.3f}",
            "",
        ])

    content = "\n".join(lines)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)

    logger.info(f"Saved evaluation report to {output_path}")
    return content
