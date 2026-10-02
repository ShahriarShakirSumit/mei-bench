"""
Run evaluation pipeline end-to-end.

Convenience script that chains: load data → load model → run inference → compute metrics.
Can also be used to only compute metrics from existing predictions.
"""

from __future__ import annotations

from mei_benchmark.paths import MODEL_CACHE

import argparse
import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run MEI evaluation pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "inference-only", "metrics-only"],
        default="full",
        help="Pipeline mode",
    )
    parser.add_argument("--model", required=True, help="Model key from config")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config")
    parser.add_argument("--model-config", default="configs/models_open.yaml", help="Model config")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    data_root = args.data_dir or paths["data_root"]
    out_root = args.output_dir or paths["output_root"]

    # Load dataset items
    from mei_benchmark.data.loader import MEIDatasetLoader

    loader = MEIDatasetLoader(data_root)
    items = loader.load_items()
    logger.info(f"Loaded {len(items)} items from {data_root}")

    if args.mode in ("full", "inference-only"):
        mcfg = model_cfg["models"][args.model]

        from mei_benchmark.models.base import VLMConfig
        from mei_benchmark.models.inference import load_model_from_config, run_inference

        vlm_config = VLMConfig(
            model_name=mcfg["model_name"],
            model_id=mcfg["model_id"],
            device=mcfg.get("device", "cuda:0"),
            max_new_tokens=mcfg.get("max_new_tokens", 128),
            temperature=mcfg.get("temperature", 0.0),
            dtype=mcfg.get("dtype", "float16"),
            use_flash_attention=mcfg.get("use_flash_attention", True),
            cache_dir=paths.get("model_cache", str(MODEL_CACHE)),
        )

        vlm = load_model_from_config(vlm_config)
        logger.info(f"Loading model: {mcfg['model_name']}")
        vlm.load_model()

        pred_dir = Path(out_root) / "predictions"
        run_inference(
            model=vlm,
            items=items,
            data_root=data_root,
            output_dir=pred_dir,
            resume=args.resume,
        )
        vlm.unload_model()

    if args.mode in ("full", "metrics-only"):
        pred_dir = Path(out_root) / "predictions"
        from mei_benchmark.data.loader import PredictionLoader
        from mei_benchmark.evaluation.metrics import MEIEvaluator, StratifiedEvaluator

        pred_loader = PredictionLoader(str(pred_dir))
        model_name = model_cfg["models"][args.model]["model_name"]

        # Try to find prediction file
        all_models = pred_loader.available_models()
        target_model = None
        for m in all_models:
            if args.model in m.lower() or model_name.lower() in m.lower():
                target_model = m
                break

        if target_model is None:
            logger.error(f"No predictions found for {args.model}")
            return

        pred_sets = pred_loader.load_predictions(target_model)
        evaluator = MEIEvaluator()
        metrics = evaluator.evaluate(items, pred_sets)

        # Print summary
        print("\n" + "=" * 60)
        print(f"MEI Benchmark Results: {metrics.model_name}")
        print("=" * 60)
        print(f"  Items evaluated:      {metrics.n_items}")
        print(f"  Original Accuracy:    {metrics.original_accuracy:.4f}")
        print(f"  CS (Blur):            {metrics.causal_sensitivity_blur:.4f}")
        print(f"  CS (Gray):            {metrics.causal_sensitivity_gray:.4f}")
        print(f"  SI (Blur):            {metrics.spurious_invariance_blur:.4f}")
        print(f"  SI (Gray):            {metrics.spurious_invariance_gray:.4f}")
        print(f"  Grounding Gap (GG):   {metrics.grounding_gap:.4f}")
        print(f"  SR (Blur):            {metrics.sham_robustness_blur:.4f}")
        print(f"  SR (Gray):            {metrics.sham_robustness_gray:.4f}")
        print(f"  Grounding Specificity:{metrics.grounding_specificity:.4f}")
        print("=" * 60)

        # Save metrics
        metrics_dir = Path(out_root) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"{target_model}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

        # Stratified analysis
        strat = StratifiedEvaluator()
        strat_results = strat.evaluate_stratified(items, pred_sets, target_model)

        print("\nStratified by Task Type:")
        for task, task_metrics in strat_results.by_task.items():
            print(f"  {task}: CS={task_metrics.causal_sensitivity_blur:.3f}, "
                  f"GG={task_metrics.grounding_gap:.3f}, n={task_metrics.n_items}")


if __name__ == "__main__":
    main()
