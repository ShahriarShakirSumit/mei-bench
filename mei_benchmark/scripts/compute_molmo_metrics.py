#!/usr/bin/env python3
"""Compute all Molmo metrics once predictions are complete (5754 lines).

Outputs:
  1. Main metrics (Acc, CS_blur, CS_gray, SI_blur, GG, SR, GS)
  2. Bootstrap 95% CIs for GG and GS
  3. Task-stratified GG
  4. Area-effect GG (small/medium/large)
  5. Ablation (GG for gray and swap interventions)
  6. Updated Table 4 means across all 10 models
  7. Saves Molmo metrics JSON
"""

from mei_benchmark.paths import DATA_DIR as _DATA_DIR, OUTPUT_DIR
import json
import glob
import numpy as np
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

DATA_DIR = _DATA_DIR
PRED_DIR = OUTPUT_DIR / 'predictions'
METRICS_DIR = OUTPUT_DIR / 'metrics'

def fuzzy_match(a, b, threshold=0.8):
    a = a.lower().strip('., !?')
    b = b.lower().strip('., !?')
    return SequenceMatcher(None, a, b).ratio() >= threshold

def is_correct(answer, valid_answers):
    return any(fuzzy_match(answer, v) for v in valid_answers)

def load_items():
    items = {}
    with open(DATA_DIR / 'items.jsonl') as f:
        for line in f:
            d = json.loads(line)
            items[d['item_id']] = d
    return items

def load_predictions(model_name):
    by_item = defaultdict(dict)
    with open(PRED_DIR / f'{model_name}_predictions.jsonl') as f:
        for line in f:
            d = json.loads(line)
            by_item[d['item_id']][d['intervention_type']] = d['answer']
    return by_item

def compute_metrics(items, by_item):
    """Compute full metric suite."""
    correct_items = []
    for item_id, preds in by_item.items():
        if item_id not in items:
            continue
        item = items[item_id]
        orig = preds.get('original', '')
        if is_correct(orig, item['valid_answers']):
            correct_items.append((item_id, item, preds))
    
    n_total = len(by_item)
    n_correct = len(correct_items)
    acc = n_correct / n_total if n_total > 0 else 0
    
    # Compute flip rates
    cs_blur = cs_gray = si_blur = 0
    cs_swap = 0
    
    for item_id, item, preds in correct_items:
        valid = item['valid_answers']
        
        ev_blur = preds.get('evidence_blur', '')
        ev_gray = preds.get('evidence_gray', '')
        ev_swap = preds.get('evidence_swap', '')
        sh_blur = preds.get('sham_blur', '')
        
        if not is_correct(ev_blur, valid): cs_blur += 1
        if not is_correct(ev_gray, valid): cs_gray += 1
        if not is_correct(ev_swap, valid): cs_swap += 1
        if not is_correct(sh_blur, valid): si_blur += 1
    
    if n_correct == 0:
        return None
    
    cs_b = cs_blur / n_correct
    cs_g = cs_gray / n_correct
    cs_s = cs_swap / n_correct
    si_b = si_blur / n_correct
    gg = cs_b - si_b
    sr = 1 - si_b
    gs = max(gg, 0) / max(cs_b, 1e-8)
    
    gg_gray = cs_g - si_b  # using same sham baseline
    gg_swap = cs_s - si_b
    
    return {
        'n_total': n_total, 'n_correct': n_correct,
        'acc': acc, 'CS_blur': cs_b, 'CS_gray': cs_g, 'CS_swap': cs_s,
        'SI_blur': si_b, 'GG': gg, 'SR': sr, 'GS': gs,
        'GG_gray': gg_gray, 'GG_swap': gg_swap,
        'correct_items': correct_items,
    }

def bootstrap_ci(items_data, n_boot=1000, seed=42):
    """Bootstrap 95% CIs for GG and GS."""
    rng = np.random.RandomState(seed)
    n = len(items_data)
    
    gg_samples = []
    gs_samples = []
    
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sample = [items_data[i] for i in idx]
        
        cs_count = sum(1 for _, _, preds, item in sample 
                       if not is_correct(preds.get('evidence_blur', ''), item['valid_answers']))
        si_count = sum(1 for _, _, preds, item in sample 
                       if not is_correct(preds.get('sham_blur', ''), item['valid_answers']))
        
        cs = cs_count / n
        si = si_count / n
        gg = cs - si
        gs = max(gg, 0) / max(cs, 1e-8)
        
        gg_samples.append(gg)
        gs_samples.append(gs)
    
    gg_ci = (np.percentile(gg_samples, 2.5), np.percentile(gg_samples, 97.5))
    gs_ci = (np.percentile(gs_samples, 2.5), np.percentile(gs_samples, 97.5))
    
    return gg_ci, gs_ci

def task_stratified(items, correct_items):
    """GG by task type."""
    tasks = ['object_identification', 'attribute_verification', 'spatial_relations', 'counting', 'text_in_image']
    short = {'object_identification': 'ObjID', 'attribute_verification': 'Attr', 
             'spatial_relations': 'Spatial', 'counting': 'Count', 'text_in_image': 'Text'}
    
    results = {}
    for task in tasks:
        task_items = [(iid, item, preds) for iid, item, preds in correct_items if item['task_type'] == task]
        n = len(task_items)
        if n == 0:
            results[short[task]] = {'n': 0, 'GG': 0.0, 'CS': 0.0, 'SI': 0.0, 'Acc': 0.0, 'GS': 0.0, 'SR': 0.0}
            continue
        
        cs = sum(1 for _, item, preds in task_items if not is_correct(preds.get('evidence_blur', ''), item['valid_answers'])) / n
        si = sum(1 for _, item, preds in task_items if not is_correct(preds.get('sham_blur', ''), item['valid_answers'])) / n
        gg = cs - si
        sr = 1 - si
        gs = max(gg, 0) / max(cs, 1e-8)
        
        # Get total items for accuracy
        all_items_count = sum(1 for iid in items if items[iid]['task_type'] == task)
        acc = n / all_items_count if all_items_count > 0 else 0
        
        results[short[task]] = {'n': n, 'GG': gg, 'CS': cs, 'SI': si, 'Acc': acc, 'GS': gs, 'SR': sr}
    
    return results

def area_effects(items, correct_items):
    """GG by evidence area size."""
    bins = {'small': (0, 0.05), 'medium': (0.05, 0.15), 'large': (0.15, 1.0)}
    results = {}
    
    for bin_name, (lo, hi) in bins.items():
        bin_items = [(iid, item, preds) for iid, item, preds in correct_items
                     if lo <= item['evidence_region']['area_fraction'] < hi]
        n = len(bin_items)
        if n == 0:
            results[bin_name] = 0.0
            continue
        
        cs = sum(1 for _, item, preds in bin_items if not is_correct(preds.get('evidence_blur', ''), item['valid_answers'])) / n
        si = sum(1 for _, item, preds in bin_items if not is_correct(preds.get('sham_blur', ''), item['valid_answers'])) / n
        results[bin_name] = cs - si
    
    return results

def compute_ablation_means():
    """Compute mean GG across all 10 models for blur, gray, swap."""
    items = load_items()
    
    all_gg_blur = []
    all_gg_gray = []
    all_gg_swap = []
    
    for pred_file in sorted(PRED_DIR.glob('*_predictions.jsonl')):
        model = pred_file.stem.replace('_predictions', '')
        by_item = defaultdict(dict)
        with open(pred_file) as f:
            for line in f:
                d = json.loads(line)
                by_item[d['item_id']][d['intervention_type']] = d['answer']
        
        m = compute_metrics(items, by_item)
        if m is None:
            continue
        
        all_gg_blur.append(m['GG'])
        all_gg_gray.append(m['GG_gray'])
        all_gg_swap.append(m['GG_swap'])
        print(f"  {model}: GG_blur={m['GG']:.3f}  GG_gray={m['GG_gray']:.3f}  GG_swap={m['GG_swap']:.3f}")
    
    return {
        'mean_blur': np.mean(all_gg_blur),
        'mean_gray': np.mean(all_gg_gray),
        'mean_swap': np.mean(all_gg_swap),
        'n_models': len(all_gg_blur),
    }

def main():
    items = load_items()
    by_item = load_predictions('Molmo-7B-D')
    
    n_preds = len(by_item)
    n_expected = len(items)
    print(f"Molmo predictions: {sum(len(v) for v in by_item.values())} lines, {n_preds} unique items (expected {n_expected})")
    
    if n_preds < n_expected:
        print(f"WARNING: Only {n_preds}/{n_expected} items have predictions. Proceeding with available data.")
    
    # 1. Main metrics
    m = compute_metrics(items, by_item)
    if m is None:
        print("ERROR: No correct items found!")
        return
    
    print(f"\n=== MAIN METRICS ===")
    print(f"Acc={m['acc']:.3f}  CS_blur={m['CS_blur']:.3f}  CS_gray={m['CS_gray']:.3f}  SI_blur={m['SI_blur']:.3f}")
    print(f"GG={m['GG']:.3f}  SR={m['SR']:.3f}  GS={m['GS']:.3f}")
    print(f"GG_gray={m['GG_gray']:.3f}  GG_swap={m['GG_swap']:.3f}")
    
    # 2. Bootstrap CIs
    items_data = [(iid, item, preds, item) for iid, item, preds in m['correct_items']]
    gg_ci, gs_ci = bootstrap_ci(items_data)
    print(f"\n=== BOOTSTRAP CIs ===")
    print(f"GG 95% CI: [{gg_ci[0]:.3f}, {gg_ci[1]:.3f}]")
    print(f"GS 95% CI: [{gs_ci[0]:.3f}, {gs_ci[1]:.3f}]")
    
    # 3. Task-stratified
    ts = task_stratified(items, m['correct_items'])
    print(f"\n=== TASK-STRATIFIED GG ===")
    for task, vals in ts.items():
        print(f"  {task}: n={vals['n']}  Acc={vals['Acc']:.3f}  CS={vals['CS']:.3f}  SI={vals['SI']:.3f}  GG={vals['GG']:.3f}  SR={vals['SR']:.3f}  GS={vals['GS']:.3f}")
    
    # 4. Area effects
    ae = area_effects(items, m['correct_items'])
    print(f"\n=== AREA EFFECTS GG ===")
    for size, gg_val in ae.items():
        print(f"  {size}: GG={gg_val:.3f}")
    
    # 5. Save metrics JSON
    metrics_json = {
        'model': 'Molmo-7B-D',
        'accuracy': round(m['acc'], 4),
        'CS_blur': round(m['CS_blur'], 4),
        'CS_gray': round(m['CS_gray'], 4),
        'SI_blur': round(m['SI_blur'], 4),
        'GG': round(m['GG'], 4),
        'SR': round(m['SR'], 4),
        'GS': round(m['GS'], 4),
    }
    out_path = METRICS_DIR / 'Molmo-7B-D_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\nSaved metrics to {out_path}")
    
    # 6. Ablation means (all 10 models)
    print(f"\n=== ABLATION TABLE MEANS (all models) ===")
    abl = compute_ablation_means()
    print(f"\nMean GG blur={abl['mean_blur']:.3f}  gray={abl['mean_gray']:.3f}  swap={abl['mean_swap']:.3f}  (n={abl['n_models']} models)")
    
    # 7. Spearman correlation for ablation
    from scipy import stats
    all_gg = {'blur': [], 'gray': [], 'swap': []}
    for pred_file in sorted(PRED_DIR.glob('*_predictions.jsonl')):
        model = pred_file.stem.replace('_predictions', '')
        by_item_m = defaultdict(dict)
        with open(pred_file) as f:
            for line in f:
                d = json.loads(line)
                by_item_m[d['item_id']][d['intervention_type']] = d['answer']
        m2 = compute_metrics(items, by_item_m)
        if m2:
            all_gg['blur'].append(m2['GG'])
            all_gg['gray'].append(m2['GG_gray'])
            all_gg['swap'].append(m2['GG_swap'])
    
    rho_gray, _ = stats.spearmanr(all_gg['blur'], all_gg['gray'])
    rho_swap, _ = stats.spearmanr(all_gg['blur'], all_gg['swap'])
    print(f"Spearman rho (blur vs gray): {rho_gray:.3f}")
    print(f"Spearman rho (blur vs swap): {rho_swap:.3f}")

if __name__ == '__main__':
    main()
