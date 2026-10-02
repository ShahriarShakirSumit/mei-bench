#!/usr/bin/env python3
"""Download the ten open-weight models at the exact commits used in the paper (configs/models_open.yaml: revision).

    python -m mei_benchmark.scripts.download_models            # all ten
    python -m mei_benchmark.scripts.download_models qwen2-vl-7b molmo-7b

The evaluation loads models by repository id. To make it resolve to the pinned commit, this script also points the
cache's refs/main at that commit; then evaluate with HF_HUB_OFFLINE=1 (as in the paper, whose GPU nodes were offline).
"""
import sys
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download

cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "models_open.yaml"))["models"]
for k in sys.argv[1:] or list(cfg):
    m = cfg[k]
    path = Path(snapshot_download(m["model_id"], revision=m["revision"], ignore_patterns=["*.gguf", "*.ggml"]))
    ref = path.parents[1] / "refs" / "main"                       # <cache>/models--org--name/refs/main
    ref.parent.mkdir(exist_ok=True); ref.write_text(m["revision"])
    print(f"{k}: {m['model_id']}@{m['revision'][:8]} -> {path}")
