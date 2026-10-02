"""
MEI Benchmark CLI.

Main command-line interface for building the dataset,
running evaluations, and generating reports.

Usage:
    mei build-dataset --config configs/base.yaml
    mei run-evaluation --model llava-1.5-7b --config configs/base.yaml
    mei generate-report --results outputs/results/ --output outputs/report/
    mei visualize --results outputs/results/ --output outputs/figures/
"""

from __future__ import annotations

from mei_benchmark.paths import MODEL_CACHE

import json
import logging
from pathlib import Path
from typing import Optional

import typer
import yaml

app = typer.Typer(
    name="mei",
    help="MEI Benchmark: Minimal Evidence Intervention for causal visual grounding evaluation",
    add_completion=False,
)

logger = logging.getLogger(__name__)


def _setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def _load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


@app.command()
def build_dataset(
    config: str = typer.Option("configs/base.yaml", help="Path to base config"),
    task_config: str = typer.Option("configs/tasks.yaml", help="Path to task config"),
    output_dir: Optional[str] = typer.Option(None, help="Override output directory"),
    n_items: Optional[int] = typer.Option(None, help="Override n_items per task"),
    seed: int = typer.Option(42, help="Random seed"),
    skip_interventions: bool = typer.Option(False, help="Skip intervention generation"),
):
    """Build the MEI benchmark dataset from COCO images.

    Steps:
    1. Load COCO images and annotations
    2. Generate questions per task type
    3. Find minimal evidence regions (SAM + CLIP)
    4. Generate intervention variants
    5. Save dataset manifest
    """
    _setup_logging()

    cfg = _load_config(config)
    task_cfg = _load_config(task_config)

    paths = cfg["paths"]
    data_root = output_dir or paths["data_root"]

    typer.echo(f"Building MEI dataset → {data_root}")
    typer.echo(f"COCO images: {paths['coco_images']}")
    typer.echo(f"Tasks: {list(task_cfg['tasks'].keys())}")

    # Import here to avoid slow imports at CLI level
    from mei_benchmark.data.sources import COCOAdapter
    from mei_benchmark.intervention.evidence_finder import EvidenceFinder
    from mei_benchmark.intervention.pipeline import MEIInterventionPipeline

    # Step 1: Load COCO
    typer.echo("\n[1/4] Loading COCO dataset...")
    coco = COCOAdapter(
        images_dir=paths["coco_images"],
        annotations_file=paths["coco_annotations"],
    )
    typer.echo(f"  Loaded {len(coco)} images")

    # Step 2: Generate items (question-answer pairs)
    typer.echo("\n[2/4] Generating question-answer pairs...")
    from mei_benchmark.scripts.build_interventions import generate_items_from_coco
    items = generate_items_from_coco(
        coco=coco,
        task_config=task_cfg,
        n_per_task=n_items or cfg["dataset"]["n_items_per_task"],
        seed=seed,
    )
    typer.echo(f"  Generated {len(items)} items")

    if skip_interventions:
        # Save items without interventions
        from mei_benchmark.data.loader import save_items
        items_path = Path(data_root) / "items.jsonl"
        items_path.parent.mkdir(parents=True, exist_ok=True)
        save_items(items, str(items_path))
        typer.echo(f"\nSaved {len(items)} items to {items_path}")
        return

    # Step 3: Find evidence regions
    typer.echo("\n[3/4] Finding evidence regions (SAM + CLIP)...")
    ef_cfg = cfg["evidence_finder"]
    finder = EvidenceFinder(
        sam_checkpoint=paths["sam_checkpoint"],
        sam_model_type=ef_cfg["sam_model_type"],
        clip_model_name=ef_cfg.get("clip_model_name", ef_cfg.get("clip_model", "ViT-B-32")),
        clip_pretrained=ef_cfg["clip_pretrained"],
        device=ef_cfg["device"],
    )

    # Step 4: Generate interventions
    typer.echo("\n[4/4] Generating interventions...")
    pipeline = MEIInterventionPipeline(
        output_dir=data_root,
        evidence_finder=finder,
        blur_radius=cfg["interventions"].get("blur_radius", cfg["interventions"].get("blur_kernel_size", 51)),
        gray_value=cfg["interventions"].get("gray_value", 128),
    )

    processed = pipeline.process_dataset(items)
    typer.echo(f"\nDataset built: {len(processed)} items with interventions")
    typer.echo(f"Output: {data_root}")


@app.command()
def run_evaluation(
    model: str = typer.Argument(..., help="Model key (e.g., llava-1.5-7b)"),
    config: str = typer.Option("configs/base.yaml", help="Base config path"),
    model_config: str = typer.Option("configs/models_open.yaml", help="Model config path"),
    data_dir: Optional[str] = typer.Option(None, help="Override data directory"),
    output_dir: Optional[str] = typer.Option(None, help="Override output directory"),
    resume: bool = typer.Option(True, help="Resume from existing predictions"),
):
    """Run a VLM on the MEI benchmark dataset.

    Loads the model, runs inference on all items (original + interventions),
    and saves predictions.
    """
    _setup_logging()

    cfg = _load_config(config)
    model_cfg = _load_config(model_config)

    if model not in model_cfg["models"]:
        typer.echo(f"Unknown model: {model}")
        typer.echo(f"Available: {list(model_cfg['models'].keys())}")
        raise typer.Exit(1)

    mcfg = model_cfg["models"][model]
    data_root = data_dir or cfg["paths"]["data_root"]
    out_root = output_dir or cfg["paths"]["output_root"]

    typer.echo(f"Evaluating model: {mcfg['model_name']}")
    typer.echo(f"Data: {data_root}")
    typer.echo(f"Output: {out_root}")

    # Load dataset
    from mei_benchmark.data.loader import MEIDatasetLoader
    loader = MEIDatasetLoader(data_root)
    items = loader.load_items()
    typer.echo(f"Loaded {len(items)} items")

    # Load model
    from mei_benchmark.models.base import VLMConfig
    from mei_benchmark.models.inference import load_model_from_config

    vlm_config = VLMConfig(
        model_name=mcfg["model_name"],
        model_id=mcfg["model_id"],
        device=mcfg.get("device", "cuda:0"),
        max_new_tokens=mcfg.get("max_new_tokens", 128),
        temperature=mcfg.get("temperature", 0.0),
        batch_size=mcfg.get("batch_size", 1),
        dtype=mcfg.get("dtype", "float16"),
        use_flash_attention=mcfg.get("use_flash_attention", True),
        cache_dir=cfg["paths"].get("model_cache", str(MODEL_CACHE)),
    )

    vlm = load_model_from_config(vlm_config)
    typer.echo(f"Loading model weights...")
    vlm.load_model()

    # Run inference
    from mei_benchmark.models.inference import run_inference

    pred_set = run_inference(
        model=vlm,
        items=items,
        data_root=data_root,
        output_dir=Path(out_root) / "predictions",
        resume=resume,
    )

    typer.echo(f"\nInference complete: {len(pred_set.predictions)} predictions")

    # Clean up
    vlm.unload_model()


@app.command()
def compute_metrics(
    results_dir: str = typer.Argument(..., help="Directory with prediction files"),
    data_dir: Optional[str] = typer.Option(None, help="Data directory with items.jsonl"),
    config: str = typer.Option("configs/base.yaml", help="Base config"),
    output_dir: Optional[str] = typer.Option(None, help="Override output directory"),
):
    """Compute MEI metrics from saved predictions.

    Reads prediction JSONL files and computes:
    - Causal Sensitivity (CS)
    - Sham Invariance (SI)
    - Grounding Gap (GG)
    - Sham Robustness (SR)
    - Grounding Specificity (GS)
    """
    _setup_logging()

    cfg = _load_config(config)
    data_root = data_dir or cfg["paths"]["data_root"]
    out_root = output_dir or cfg["paths"]["output_root"]

    # Load items
    from mei_benchmark.data.loader import MEIDatasetLoader, PredictionLoader
    loader = MEIDatasetLoader(data_root)
    items = loader.load_items()

    # Load all predictions
    pred_loader = PredictionLoader(results_dir)
    all_models = pred_loader.available_models()
    typer.echo(f"Found predictions for: {all_models}")

    from mei_benchmark.evaluation.metrics import MEIEvaluator, StratifiedEvaluator

    all_metrics = []

    for model_name in all_models:
        typer.echo(f"\nComputing metrics for {model_name}...")
        pred_sets = pred_loader.load_predictions(model_name)

        evaluator = MEIEvaluator()
        metrics = evaluator.evaluate(items, pred_sets)
        all_metrics.append(metrics)

        typer.echo(f"  Original Accuracy: {metrics.original_accuracy:.3f}")
        typer.echo(f"  CS (Blur): {metrics.causal_sensitivity_blur:.3f}")
        typer.echo(f"  CS (Gray): {metrics.causal_sensitivity_gray:.3f}")
        typer.echo(f"  SI (Blur): {metrics.spurious_invariance_blur:.3f}")
        typer.echo(f"  GG: {metrics.grounding_gap:.3f}")
        typer.echo(f"  SR (Blur): {metrics.sham_robustness_blur:.3f}")
        typer.echo(f"  GS: {metrics.grounding_specificity:.3f}")

        # Stratified analysis
        strat = StratifiedEvaluator()
        strat_results = strat.evaluate_stratified(items, pred_sets, model_name)

        # Save per-model metrics
        metrics_path = Path(out_root) / "metrics" / f"{model_name}_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

    # Generate comparative visualizations
    if len(all_metrics) > 1:
        typer.echo("\nGenerating comparison figures...")
        from mei_benchmark.evaluation.visualize import (
            plot_model_comparison_bar,
            plot_radar_chart,
            plot_grounding_specificity_scatter,
            generate_latex_table,
        )

        fig_dir = Path(out_root) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        plot_model_comparison_bar(all_metrics, fig_dir / "model_comparison.pdf")
        plot_radar_chart(all_metrics, fig_dir / "radar_chart.pdf")
        plot_grounding_specificity_scatter(all_metrics, fig_dir / "grounding_scatter.pdf")
        generate_latex_table(all_metrics, fig_dir / "results_table.tex")

    typer.echo("\nDone!")


@app.command()
def generate_pbs(
    model: str = typer.Argument(..., help="Model key"),
    config: str = typer.Option("configs/base.yaml", help="Base config"),
    model_config: str = typer.Option("configs/models_open.yaml", help="Model config"),
    output: str = typer.Option("scripts/", help="Output directory for PBS scripts"),
):
    """Generate PBS job scripts for NCI Gadi.

    Creates a .pbs file for running a specific model evaluation
    on the Gadi HPC system.
    """
    cfg = _load_config(config)
    model_cfg = _load_config(model_config)

    if model not in model_cfg["models"]:
        typer.echo(f"Unknown model: {model}")
        raise typer.Exit(1)

    mcfg = model_cfg["models"][model]
    pbs = mcfg.get("pbs", {})
    pbs_defaults = model_cfg.get("pbs_defaults", {})

    script = f"""#!/bin/bash
#PBS -N mei_{model}
#PBS -q {pbs.get('queue', 'gpuvolta')}
#PBS -l ncpus={pbs.get('ncpus', 12)}
#PBS -l ngpus={pbs.get('ngpus', 1)}
#PBS -l mem={pbs.get('mem', '48GB')}
#PBS -l walltime={pbs.get('walltime', '04:00:00')}
#PBS -l jobfs={pbs.get('jobfs', '10GB')}
#PBS -l storage={pbs_defaults.get('storage', 'scratch/YOUR_PROJECT')}
#PBS -P {pbs_defaults.get('project', 'YOUR_PROJECT')}
#PBS -l wd
#PBS -j oe

# Load modules
module load singularity

# Activate conda
CONDA_ENV="{pbs_defaults.get('conda_env', 'mei-benchmark')}"
export PATH="$CONDA_ENV/bin:$PATH"

# Run evaluation
cd $PBS_O_WORKDIR
python -m mei_benchmark.scripts.cli run-evaluation {model} \\
    --config {config} \\
    --model-config {model_config} \\
    --resume
"""

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / f"run_{model}.pbs"
    with open(script_path, "w") as f:
        f.write(script)

    typer.echo(f"Generated PBS script: {script_path}")
    typer.echo(f"Submit with: qsub {script_path}")


@app.command()
def info():
    """Show benchmark configuration and dataset info."""
    typer.echo("MEI Benchmark v1.0.0")
    typer.echo("=" * 40)
    typer.echo("Minimal Evidence Intervention for Causal Visual Grounding")
    typer.echo()
    typer.echo("Task Types:")
    typer.echo("  1. Text in Image")
    typer.echo("  2. Attribute Verification")
    typer.echo("  3. Spatial Relations")
    typer.echo("  4. Counting")
    typer.echo("  5. Object Identification")
    typer.echo()
    typer.echo("Intervention Types:")
    typer.echo("  - Evidence Blur (Gaussian blur over evidence region)")
    typer.echo("  - Evidence Gray (Grayscale conversion of evidence)")
    typer.echo("  - Evidence Swap (Content inversion of evidence)")
    typer.echo("  - Distractor Added (Duplicate object placed elsewhere)")
    typer.echo("  - Sham Blur (Control: blur non-evidence region)")
    typer.echo("  - Sham Gray (Control: grayscale non-evidence region)")
    typer.echo()
    typer.echo("Metrics:")
    typer.echo("  CS = Causal Sensitivity (evidence flip rate)")
    typer.echo("  SI = Sham Invariance (sham flip rate, lower better)")
    typer.echo("  GG = Grounding Gap (CS - SI)")
    typer.echo("  SR = Sham Robustness (1 - SI)")
    typer.echo("  GS = Grounding Specificity (GG / max(CS, ε))")


if __name__ == "__main__":
    app()
