"""
Async API evaluator for closed-source VLLMs (Claude Sonnet 4.5, Gemini 2.5 Pro).

Streams predictions to disk in the same JSONL format as the local-model pipeline,
so existing PredictionLoader and MEIEvaluator can consume the output directly.
Resumable, rate-limited, retry-on-error.
"""
from __future__ import annotations

from mei_benchmark.paths import REPO_ROOT

import argparse
import asyncio
import base64
import json
import logging
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("api_eval")

ROOT = REPO_ROOT
ITEMS_PATH = ROOT / "data" / "items.jsonl"
DATA_ROOT = ROOT / "data"
PRED_DIR = ROOT / "outputs" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)


def encode_image_b64(path: Path, max_side: int = 1024) -> str:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError(f"jpeg encode failed for {path}")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def format_prompt(question: str, valid_answers: list[str]) -> str:
    prompt = (
        "Answer the question based on the image. Give only the direct answer, "
        "no explanation, no full sentence.\n\n"
        f"Question: {question}\nAnswer:"
    )
    return prompt


def load_items() -> list[dict]:
    items = []
    with open(ITEMS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_existing(pred_file: Path) -> set[str]:
    keys = set()
    if pred_file.exists():
        with open(pred_file) as f:
            for line in f:
                try:
                    p = json.loads(line)
                    keys.add(f"{p['item_id']}_{p['intervention_type']}")
                except Exception:
                    continue
    return keys


# --------------------------------------------------------------------------
#                              Claude runner
# --------------------------------------------------------------------------
async def run_claude(model_name: str, model_id: str, items: list[dict],
                     concurrency: int = 8, max_calls: int | None = None) -> None:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    pred_file = PRED_DIR / f"{model_name}_predictions.jsonl"
    done = load_existing(pred_file)
    log.info(f"[{model_name}] resuming with {len(done)} predictions already done")

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    out = open(pred_file, "a")
    counter = {"n": 0, "errs": 0, "start": time.time()}

    async def one_call(item_id: str, itype: str, img_path: Path,
                       question: str):
        key = f"{item_id}_{itype}"
        if key in done:
            return
        async with sem:
            try:
                b64 = encode_image_b64(img_path)
            except Exception as e:
                log.warning(f"img read fail {img_path}: {e}")
                return
            for attempt in range(5):
                try:
                    t0 = time.perf_counter()
                    resp = await client.messages.create(
                        model=model_id,
                        max_tokens=128,
                        temperature=0.0,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image",
                                 "source": {"type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": b64}},
                                {"type": "text",
                                 "text": format_prompt(question, [])},
                            ],
                        }],
                    )
                    latency = (time.perf_counter() - t0) * 1000.0
                    raw = resp.content[0].text if resp.content else ""
                    pred = {
                        "item_id": item_id,
                        "model_name": model_name,
                        "intervention_type": itype,
                        "answer": raw.strip().lower(),
                        "raw_response": raw,
                        "confidence": None,
                        "latency_ms": latency,
                    }
                    async with write_lock:
                        out.write(json.dumps(pred) + "\n")
                        out.flush()
                        counter["n"] += 1
                        if counter["n"] % 50 == 0:
                            elapsed = time.time() - counter["start"]
                            rate = counter["n"] / max(elapsed, 1)
                            log.info(f"[{model_name}] +{counter['n']} preds "
                                     f"({rate:.2f}/s, errs={counter['errs']})")
                    return
                except Exception as e:
                    msg = str(e)[:160]
                    if "overloaded" in msg or "rate" in msg.lower() or "529" in msg:
                        wait = 2 ** attempt + random.random()
                        await asyncio.sleep(min(wait, 30))
                    elif attempt == 4:
                        log.warning(f"[{model_name}] {key} fail: {msg}")
                        counter["errs"] += 1
                        return
                    else:
                        await asyncio.sleep(1 + random.random())

    tasks = []
    for it in items:
        for itype, ipath in [("original", it["image_path"])] + [
            (v["intervention_type"], v["image_path"]) for v in it["variants"]
        ]:
            full = DATA_ROOT / ipath
            tasks.append(one_call(it["item_id"], itype, full, it["question"]))
            if max_calls and len(tasks) >= max_calls:
                break
        if max_calls and len(tasks) >= max_calls:
            break

    log.info(f"[{model_name}] launching {len(tasks)} tasks")
    await asyncio.gather(*tasks)
    out.close()
    log.info(f"[{model_name}] done. Total: {counter['n']} new preds, errs={counter['errs']}")


# --------------------------------------------------------------------------
#                              Gemini runner
# --------------------------------------------------------------------------
async def run_gemini(model_name: str, model_id: str, items: list[dict],
                     concurrency: int = 6, max_calls: int | None = None) -> None:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    pred_file = PRED_DIR / f"{model_name}_predictions.jsonl"
    done = load_existing(pred_file)
    log.info(f"[{model_name}] resuming with {len(done)} predictions already done")

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    out = open(pred_file, "a")
    counter = {"n": 0, "errs": 0, "start": time.time()}

    is_pro = "pro" in model_id.lower()
    thinking_budget = 256 if is_pro else 0
    cfg = types.GenerateContentConfig(
        max_output_tokens=512 if is_pro else 64,
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
    )

    async def one_call(item_id: str, itype: str, img_path: Path, question: str):
        key = f"{item_id}_{itype}"
        if key in done:
            return
        async with sem:
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
            except Exception as e:
                log.warning(f"img read fail {img_path}: {e}")
                return
            for attempt in range(5):
                try:
                    t0 = time.perf_counter()
                    resp = await asyncio.to_thread(
                        client.models.generate_content,
                        model=model_id,
                        contents=[
                            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                            format_prompt(question, []),
                        ],
                        config=cfg,
                    )
                    latency = (time.perf_counter() - t0) * 1000.0
                    raw = (resp.text or "").strip()
                    pred = {
                        "item_id": item_id,
                        "model_name": model_name,
                        "intervention_type": itype,
                        "answer": raw.lower(),
                        "raw_response": raw,
                        "confidence": None,
                        "latency_ms": latency,
                    }
                    async with write_lock:
                        out.write(json.dumps(pred) + "\n")
                        out.flush()
                        counter["n"] += 1
                        if counter["n"] % 50 == 0:
                            elapsed = time.time() - counter["start"]
                            rate = counter["n"] / max(elapsed, 1)
                            log.info(f"[{model_name}] +{counter['n']} preds "
                                     f"({rate:.2f}/s, errs={counter['errs']})")
                    return
                except Exception as e:
                    msg = str(e)[:160]
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate" in msg.lower() or "503" in msg or "UNAVAILABLE" in msg:
                        wait = 2 ** attempt + random.random()
                        await asyncio.sleep(min(wait, 30))
                    elif attempt == 4:
                        log.warning(f"[{model_name}] {key} fail: {msg}")
                        counter["errs"] += 1
                        return
                    else:
                        await asyncio.sleep(1 + random.random())

    tasks = []
    for it in items:
        for itype, ipath in [("original", it["image_path"])] + [
            (v["intervention_type"], v["image_path"]) for v in it["variants"]
        ]:
            full = DATA_ROOT / ipath
            tasks.append(one_call(it["item_id"], itype, full, it["question"]))
            if max_calls and len(tasks) >= max_calls:
                break
        if max_calls and len(tasks) >= max_calls:
            break

    log.info(f"[{model_name}] launching {len(tasks)} tasks")
    await asyncio.gather(*tasks)
    out.close()
    log.info(f"[{model_name}] done. Total: {counter['n']} new preds, errs={counter['errs']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["claude", "gemini", "both"], required=True)
    p.add_argument("--claude-model-name", default="Claude-Sonnet-4.5")
    p.add_argument("--claude-model-id", default="claude-sonnet-4-5-20250929")
    p.add_argument("--gemini-model-name", default="Gemini-2.5-Pro")
    p.add_argument("--gemini-model-id", default="gemini-2.5-pro")
    p.add_argument("--claude-concurrency", type=int, default=10)
    p.add_argument("--gemini-concurrency", type=int, default=6)
    p.add_argument("--max-calls", type=int, default=None)
    args = p.parse_args()

    items = load_items()
    log.info(f"Loaded {len(items)} items")

    async def go():
        coros = []
        if args.provider in ("claude", "both"):
            coros.append(run_claude(args.claude_model_name, args.claude_model_id,
                                    items, args.claude_concurrency, args.max_calls))
        if args.provider in ("gemini", "both"):
            coros.append(run_gemini(args.gemini_model_name, args.gemini_model_id,
                                    items, args.gemini_concurrency, args.max_calls))
        await asyncio.gather(*coros)

    asyncio.run(go())


if __name__ == "__main__":
    main()
