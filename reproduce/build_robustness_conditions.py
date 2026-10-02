#!/usr/bin/env python
"""Robustness conditions for MEI-Bench (ACML 2026).

Uses the benchmark's OWN rendering code, unchanged:
  mei_benchmark.scripts.generate_interventions.generate_sham_region  (sham sampler of the release)
  mei_benchmark.scripts.generate_interventions.bbox_to_mask
  mei_benchmark.intervention.generator.apply_blur_intervention       (Gaussian blur, k = 51)
and the paper's validity constraints (IoU(S,E) < 0.1, area ratio in [0.8, 1.2]).

Conditions (blur operator only, as in the main-paper GG):
  sham_r1..sham_r5 : five further randomised shams per item
  sham_salience    : sham matched to the evidence box on model-free saliency
  sham_position    : evidence box mirrored (horizontal / vertical / point)
  mask             : evidence blurred inside its COCO instance mask + the same
                     mask shape translated to a non-overlapping location (sham)
Each condition is written as a self-contained data dir (items.jsonl + variants/) so that the
paper's unchanged `mei_benchmark.scripts.run_evaluation` can evaluate it.  Box texture statistics
(edge density, entropy) are stored for the texture-matched sham.
"""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
import cv2, numpy as np

from repro_paths import RERUN_DATA, ROBUSTNESS_DATA, COCO_ANN
from mei_benchmark.scripts.generate_interventions import generate_sham_region, bbox_to_mask  # noqa: E402
from mei_benchmark.intervention.generator import apply_blur_intervention                    # noqa: E402

BLUR_K = 51; JPEG_Q = 95; N_SHAM = 5; N_SAL_CAND = 200

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)); ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih; union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def valid(ev, sh, W, H):
    """The paper's constraints (Def. 2 + Sec. 4.1) plus 'box lies inside the image'."""
    ar = (sh[2] * sh[3]) / max(ev[2] * ev[3], 1.0)
    inside = sh[0] >= 0 and sh[1] >= 0 and sh[0] + sh[2] <= W + 0.5 and sh[1] + sh[3] <= H + 0.5
    return iou(ev, sh) < 0.1 and 0.8 <= ar <= 1.2 and inside

def draw_valid_sham(ev, H, W, rng, tries=40):
    """Call the release sampler until it returns a box satisfying the constraints (its fallback may not)."""
    for _ in range(tries):
        sh = generate_sham_region(ev, H, W, rng)
        if valid(ev, sh, W, H):
            return sh
    return None

def spectral_residual_saliency(img_bgr):
    """Hou & Zhang (CVPR 2007) spectral-residual saliency, model-free, returned at image size in [0,1]."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape; w = 64; h = max(1, int(round(H * w / W)))
    s = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    F = np.fft.fft2(s); A = np.log(np.abs(F) + 1e-8); P = np.angle(F)
    R = A - cv2.blur(A, (3, 3))
    S = np.abs(np.fft.ifft2(np.exp(R + 1j * P))) ** 2
    S = cv2.GaussianBlur(S.astype(np.float32), (0, 0), 2.5)
    S = (S - S.min()) / (S.max() - S.min() + 1e-12)
    return cv2.resize(S, (W, H), interpolation=cv2.INTER_LINEAR)

def box_mean(m, b):
    x, y, w, h = [int(round(v)) for v in b]
    x2, y2 = min(m.shape[1], x + max(w, 1)), min(m.shape[0], y + max(h, 1))
    r = m[max(0, y):y2, max(0, x):x2]
    return float(r.mean()) if r.size else float("nan")

def texture(gray, edges, b):
    x, y, w, h = [int(round(v)) for v in b]
    r = gray[max(0, y):y + h, max(0, x):x + w]; e = edges[max(0, y):y + h, max(0, x):x + w]
    if r.size == 0: return dict(edge_density=float("nan"), entropy=float("nan"))
    p = np.bincount(r.ravel(), minlength=256).astype(np.float64); p = p[p > 0] / p.sum()
    return dict(edge_density=float((e > 0).mean()), entropy=float(-(p * np.log2(p)).sum()))

def mirrored(ev, W, H):
    x, y, w, h = ev
    return [("horizontal", [W - x - w, y, w, h]), ("vertical", [x, H - y - h, w, h]),
            ("point", [W - x - w, H - y - h, w, h])]

def ann_mask(ann, H, W):
    m = np.zeros((H, W), dtype=np.uint8)
    seg = ann["segmentation"]
    if isinstance(seg, list):
        for poly in seg:
            pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
        return m
    return None   # RLE (iscrowd) -> reported, not rendered

def translate_mask(mask, ev, H, W, rng, tries=4000):
    """Same shape, same size, moved to a location whose box has IoU < 0.1 with the evidence box."""
    x, y, w, h = ev
    for _ in range(tries):
        sx = rng.uniform(0, max(0.0, W - w)); sy = rng.uniform(0, max(0.0, H - h))
        cand = [sx, sy, w, h]
        if iou(ev, cand) < 0.1:
            M = np.float32([[1, 0, sx - x], [0, 1, sy - y]])
            return cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0), cand
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(RERUN_DATA))
    ap.add_argument("--coco-ann", default=str(COCO_ANN))
    ap.add_argument("--out", default=str(ROBUSTNESS_DATA))
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--limit", type=int, default=0, help="pilot: first N items only")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(Path(a.data_dir) / "items.jsonl") if l.strip()]
    if a.limit: items = items[:a.limit]
    coco = json.load(open(a.coco_ann)); by_img = {}
    for an in coco["annotations"]: by_img.setdefault(an["image_id"], []).append(an)
    conds = [f"sham_r{r}" for r in range(1, N_SHAM + 1)] + ["sham_salience", "sham_position", "mask"]
    out = Path(a.out); rows = {c: [] for c in conds}; meta = []
    for c in conds: (out / c / "variants").mkdir(parents=True, exist_ok=True)
    for it in items:
        iid = it["item_id"]; img = cv2.imread(str(Path(a.data_dir) / it["image_path"]))
        H, W = img.shape[:2]; ev = it["evidence_region"]["bbox"]
        orig_sham = next(v["sham_region_bbox"] for v in it["variants"] if v["intervention_type"] == "sham_blur")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); edges = cv2.Canny(gray, 100, 200)
        rec = dict(item_id=iid, W=W, H=H, evidence_bbox=ev, orig_sham_bbox=orig_sham,
                   tex_evidence=texture(gray, edges, ev), tex_orig_sham=texture(gray, edges, orig_sham))
        def emit(cond, itype, bbox_or_none, image, name):
            fn = f"{iid}_{name}.jpg"
            cv2.imwrite(str(out / cond / "variants" / fn), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            return dict(intervention_type=itype, image_path=f"variants/{fn}",
                        intervention_params={"blur_radius": BLUR_K, "condition": cond}, sham_region_bbox=bbox_or_none)
        # (i) five further randomised shams, independent seeds per replicate and item
        for r in range(1, N_SHAM + 1):
            rng = random.Random(f"{a.seed}|sham_r{r}|{iid}")
            sh = draw_valid_sham(ev, H, W, rng)
            rec[f"sham_r{r}"] = sh
            if sh is None: continue
            res = apply_blur_intervention(img, bbox_to_mask(sh, H, W), blur_radius=BLUR_K)
            rec[f"tex_sham_r{r}"] = texture(gray, edges, sh)
            rows[f"sham_r{r}"].append({**it, "variants": [emit(f"sham_r{r}", "sham_blur", sh, res.image, "sham_blur")]})
        # (ii) saliency-matched sham: among valid release-sampler draws, closest mean saliency to the evidence box
        sal = spectral_residual_saliency(img); s_ev = box_mean(sal, ev); rng = random.Random(f"{a.seed}|sal|{iid}")
        cands = [c for c in (generate_sham_region(ev, H, W, rng) for _ in range(N_SAL_CAND)) if valid(ev, c, W, H)]
        rec.update(sal_evidence=s_ev, sal_orig_sham=box_mean(sal, orig_sham), sal_n_candidates=len(cands))
        if cands:
            best = min(cands, key=lambda c: abs(box_mean(sal, c) - s_ev))
            rec.update(sham_salience=best, sal_matched=box_mean(sal, best), tex_sham_salience=texture(gray, edges, best))
            res = apply_blur_intervention(img, bbox_to_mask(best, H, W), blur_radius=BLUR_K)
            rows["sham_salience"].append({**it, "variants": [emit("sham_salience", "sham_blur", best, res.image, "sham_blur")]})
        # (iii) position-matched: mirror of the evidence box (preserves size and distance to centre)
        pm = next(((k, b) for k, b in mirrored(ev, W, H) if valid(ev, b, W, H)), None)
        rec["sham_position"] = pm
        if pm:
            res = apply_blur_intervention(img, bbox_to_mask(pm[1], H, W), blur_radius=BLUR_K)
            rec["tex_sham_position"] = texture(gray, edges, pm[1])
            rows["sham_position"].append({**it, "variants": [emit("sham_position", "sham_blur", pm[1], res.image, "sham_blur")]})
        # (iv) segmentation-mask evidence + translated-mask sham
        img_id = int(it["source_image_id"])
        match = [an for an in by_img.get(img_id, []) if all(abs(p - q) < 0.02 for p, q in zip(an["bbox"], ev))]
        rec["mask_ann_matches"] = len(match)
        if len(match) == 1:
            m = ann_mask(match[0], H, W)
            rec["mask_iscrowd"] = m is None
            if m is not None and (m > 0).sum() > 0:
                sm, sb = translate_mask(m, ev, H, W, random.Random(f"{a.seed}|mask|{iid}"))
                rec.update(mask_area=int((m > 0).sum()), mask_box_fill=float((m > 0).sum() / max(ev[2] * ev[3], 1)))
                if sm is not None:
                    e_img = apply_blur_intervention(img, m, blur_radius=BLUR_K).image
                    s_img = apply_blur_intervention(img, sm, blur_radius=BLUR_K).image
                    rec["mask_sham_box"] = sb
                    rows["mask"].append({**it, "variants": [emit("mask", "evidence_blur", None, e_img, "evidence_blur_mask"),
                                                             emit("mask", "sham_blur", sb, s_img, "sham_blur_mask")]})
        meta.append(rec)
    for c in conds:
        with open(out / c / "items.jsonl", "w") as f:
            for r in rows[c]: f.write(json.dumps(r) + "\n")
    json.dump(dict(seed=a.seed, blur_k=BLUR_K, n_items=len(items), counts={c: len(rows[c]) for c in conds}, items=meta),
              open(out / "robustness_conditions_meta.json", "w"))
    print(json.dumps({c: len(rows[c]) for c in conds}))

if __name__ == "__main__":
    main()
