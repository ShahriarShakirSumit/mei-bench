"""
Generate publication figures for the MEI benchmark paper.

Creates all figures needed for the paper:
- Figure 1: Method overview (intervention examples)
- Figure 2: Model comparison bar chart
- Figure 3: Radar chart
- Figure 4: Dose-response curves
- Figure 5: Grounding specificity scatter
- Figure 6: Task-stratified results
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate MEI paper figures")
    parser.add_argument("--results-dir", required=True, help="Directory with metrics JSON files")
    parser.add_argument("--output-dir", default="outputs/figures", help="Output directory")
    parser.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from mei_benchmark.evaluation.metrics import MEIMetrics
    from mei_benchmark.evaluation.visualize import (
        plot_model_comparison_bar,
        plot_radar_chart,
        plot_grounding_specificity_scatter,
        generate_latex_table,
    )

    # Load all metrics files
    all_metrics = []
    for metrics_file in sorted(results_dir.glob("*_metrics.json")):
        with open(metrics_file) as f:
            data = json.load(f)

        metrics = MEIMetrics(
            model_name=data.get("model", metrics_file.stem.replace("_metrics", "")),
            n_items=data.get("n_items", 0),
            original_accuracy=data.get("original_accuracy", 0),
            causal_sensitivity_blur=data.get("CS_blur", 0),
            causal_sensitivity_gray=data.get("CS_gray", 0),
            causal_sensitivity_swap=data.get("CS_swap", 0),
            spurious_invariance_blur=data.get("SI_blur", 0),
            spurious_invariance_gray=data.get("SI_gray", 0),
            grounding_gap=data.get("GG", 0),
            sham_robustness_blur=data.get("SR_blur", 0),
            sham_robustness_gray=data.get("SR_gray", 0),
            grounding_specificity=data.get("GS", 0),
        )
        all_metrics.append(metrics)
        logger.info(f"Loaded metrics for {metrics.model_name}")

    if not all_metrics:
        logger.error("No metrics files found!")
        return

    ext = args.format

    # Figure 2: Model comparison bar chart
    logger.info("Generating model comparison bar chart...")
    plot_model_comparison_bar(
        all_metrics,
        output_dir / f"fig2_model_comparison.{ext}",
    )

    # Figure 3: Radar chart
    logger.info("Generating radar chart...")
    plot_radar_chart(
        all_metrics,
        output_dir / f"fig3_radar_chart.{ext}",
    )

    # Figure 5: Grounding specificity scatter
    logger.info("Generating grounding specificity scatter...")
    plot_grounding_specificity_scatter(
        all_metrics,
        output_dir / f"fig5_grounding_scatter.{ext}",
    )

    # LaTeX table
    logger.info("Generating LaTeX table...")
    generate_latex_table(
        all_metrics,
        output_dir / "table1_results.tex",
        caption="Main MEI benchmark results across models. CS = Causal Sensitivity (higher = better grounding), SI = Sham Invariance (lower = better), GG = Grounding Gap (CS - SI), SR = Sham Robustness (1 - SI), GS = Grounding Specificity.",
    )

    logger.info(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    main()
