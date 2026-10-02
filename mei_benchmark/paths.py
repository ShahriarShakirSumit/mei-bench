"""Default locations, all overridable by environment variables.

MEI_DATA        items.jsonl + variants/ (default: <repo>/data/mei_bench)
MEI_OUTPUTS     predictions/ and metrics/ written by the evaluation (default: <repo>/outputs)
MEI_COCO        COCO val2017 images (default: <repo>/data/coco/val2017)
HF_HUB_CACHE    Hugging Face model cache (default: ~/.cache/huggingface/hub)
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MEI_DATA", REPO_ROOT / "data" / "mei_bench"))
OUTPUT_DIR = Path(os.environ.get("MEI_OUTPUTS", REPO_ROOT / "outputs"))
COCO_DIR = Path(os.environ.get("MEI_COCO", REPO_ROOT / "data" / "coco" / "val2017"))
MODEL_CACHE = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
