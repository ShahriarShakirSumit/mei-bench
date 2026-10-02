#!/usr/bin/env python3
"""Compute all Molmo-7B-D metrics with robust verbose answer extraction."""
from mei_benchmark.paths import DATA_DIR, OUTPUT_DIR
import json, re, numpy as np
from collections import defaultdict
from difflib import SequenceMatcher

# Load benchmark items
items = {}
with open(DATA_DIR / 'items.jsonl') as f:
    for line in f:
        d = json.loads(line)
        items[d['item_id']] = d

# Load predictions
by_item = defaultdict(dict)
with open(OUTPUT_DIR / 'predictions' / 'Molmo-7B-D_predictions.jsonl') as f:
    for line in f:
        d = json.loads(line)
        by_item[d['item_id']][d['intervention_type']] = d['answer']

print(f"Loaded {len(items)} items, {len(by_item)} prediction groups")

def extract_core_answer(answer, task_type, valid_answers):
    """Extract the core answer from Molmo's verbose multi-sentence responses."""
    answer = answer.strip()
    if task_type == 'spatial_relations':
        a_lower = answer.lower()
        first_sent = a_lower.split('.')[0]
        if 'left' in first_sent and 'right' not in first_sent:
            return 'left'
        elif 'right' in first_sent and 'left' not in first_sent:
            return 'right'
        left_pos = a_lower.find('left')
        right_pos = a_lower.find('right')
        if left_pos >= 0 and (right_pos < 0 or left_pos < right_pos):
            return 'left'
        elif right_pos >= 0:
            return 'right'
        return answer
    if task_type == 'counting':
        num_words = {'zero':'0','one':'1','two':'2','three':'3','four':'4',
                     'five':'5','six':'6','seven':'7','eight':'8','nine':'9',
                     'ten':'10','eleven':'11','twelve':'12'}
        first_part = answer.split('.')[0].lower().strip()
        for word, digit in num_words.items():
            if word in first_part:
                return digit
        nums = re.findall(r'\d+', first_part)
        if nums:
            return nums[0]
        for word, digit in num_words.items():
            if word in answer.lower():
                return digit
        nums = re.findall(r'\d+', answer)
        if nums:
            return nums[0]
        return answer
    if task_type == 'object_identification':
        first_part = answer.split('.')[0].strip().lower()
        for v in valid_answers:
            if v.lower() in first_part:
                return v.lower()
        return first_part
    if task_type == 'attribute_verification':
        first_part = answer.split('.')[0].lower()
        for v in valid_answers:
            if v.lower() in first_part:
                return v.lower()
        return first_part
    # text_in_image or fallback
    return answer.split('.')[0].strip()

def match_answer(answer, valid_answers, task_type):
    """Match answer against valid answers using extraction + fuzzy matching."""
    core = extract_core_answer(answer, task_type, valid_answers)
    core_norm = core.lower().strip(' .,!?')
    for v in valid_answers:
        v_norm = v.lower().strip(' .,!?')
        if core_norm == v_norm:
            return True
        if v_norm in core_norm or core_norm in v_norm:
            return True
        if SequenceMatcher(None, core_norm, v_norm).ratio() >= 0.7:
            return True
    return False

# Compute metrics
intervention_types = ['evidence_blur','evidence_gray','evidence_swap','sham_blur','sham_gray']
n_total = 0
n_correct = 0
flips = {it: 0 for it in intervention_types}
task_stats = defaultdict(lambda: {'n':0,'correct':0,'ev_blur_flip':0,'sh_blur_flip':0,
                                   'ev_gray_flip':0,'sh_gray_flip':0})

for item_id, preds in sorted(by_item.items()):
    if item_id not in items:
        continue
    item = items[item_id]
    task = item['task_type']
    valid = item['valid_answers']
    n_total += 1
    orig = preds.get('original','')
    correct = match_answer(orig, valid, task)
    task_stats[task]['n'] += 1
    if correct:
        n_correct += 1
        task_stats[task]['correct'] += 1
        for it in intervention_types:
            int_ans = preds.get(it,'')
            if not match_answer(int_ans, valid, task):
                flips[it] += 1
                if it == 'evidence_blur': task_stats[task]['ev_blur_flip'] += 1
                elif it == 'sham_blur': task_stats[task]['sh_blur_flip'] += 1
                elif it == 'evidence_gray': task_stats[task]['ev_gray_flip'] += 1
                elif it == 'sham_gray': task_stats[task]['sh_gray_flip'] += 1

acc = n_correct/n_total if n_total > 0 else 0
cs_blur = flips['evidence_blur']/n_correct if n_correct > 0 else 0
cs_gray = flips['evidence_gray']/n_correct if n_correct > 0 else 0
si_blur = flips['sham_blur']/n_correct if n_correct > 0 else 0
gg = cs_blur - si_blur
sr = 1 - si_blur
gs = max(gg,0)/max(cs_blur,1e-8)

print(f'\n=== MOLMO OVERALL ===')
print(f'Total: {n_total}, Correct: {n_correct}, Acc: {acc:.3f}')
print(f'CS_blur: {cs_blur:.3f}, CS_gray: {cs_gray:.3f}, SI_blur: {si_blur:.3f}')
print(f'GG: {gg:.3f}, SR: {sr:.3f}, GS: {gs:.3f}')

# Show some example extractions for sanity checking
print(f'\n=== SAMPLE EXTRACTIONS (first 5 correct items) ===')
count = 0
for item_id, preds in sorted(by_item.items()):
    if item_id not in items: continue
    item = items[item_id]
    valid = item['valid_answers']
    task = item['task_type']
    orig = preds.get('original','')
    if match_answer(orig, valid, task):
        core = extract_core_answer(orig, task, valid)
        print(f'  Task={task}, Valid={valid}, Raw="{orig[:80]}...", Core="{core}"')
        count += 1
        if count >= 5: break

# Task-stratified
print(f'\n=== TASK-STRATIFIED ===')
for task in ['object_identification','attribute_verification','spatial_relations','counting','text_in_image']:
    s = task_stats[task]
    nc = s['correct']
    acc_t = nc/s['n'] if s['n'] > 0 else 0
    cs_t = s['ev_blur_flip']/nc if nc > 0 else 0
    si_t = s['sh_blur_flip']/nc if nc > 0 else 0
    gg_t = cs_t - si_t
    cs_g = s['ev_gray_flip']/nc if nc > 0 else 0
    si_g = s['sh_gray_flip']/nc if nc > 0 else 0
    gg_g = cs_g - si_g
    print(f'{task:25s}: n={s["n"]}, nc={nc}, acc={acc_t:.3f}, CS_blur={cs_t:.3f}, SI_blur={si_t:.3f}, GG={gg_t:.3f}, CS_gray={cs_g:.3f}, GG_gray={gg_g:.3f}')

# Area effects
print(f'\n=== AREA EFFECTS ===')
for bin_name, lo, hi in [('small',0,0.05),('medium',0.05,0.15),('large',0.15,1.0)]:
    ev_flip=sh_flip=nc=0
    for item_id, preds in by_item.items():
        if item_id not in items: continue
        item = items[item_id]
        area = item['evidence_region']['area_fraction']
        if not (lo <= area < hi): continue
        valid = item['valid_answers']
        task = item['task_type']
        if not match_answer(preds.get('original',''), valid, task): continue
        nc += 1
        if not match_answer(preds.get('evidence_blur',''), valid, task): ev_flip += 1
        if not match_answer(preds.get('sham_blur',''), valid, task): sh_flip += 1
    gg_a = (ev_flip-sh_flip)/nc if nc > 0 else 0
    print(f'{bin_name}: nc={nc}, GG={gg_a:.3f}')

# Bootstrap CIs
print(f'\n=== BOOTSTRAP CIs ===')
correct_items = []
for item_id, preds in sorted(by_item.items()):
    if item_id not in items: continue
    item = items[item_id]
    valid = item['valid_answers']
    task = item['task_type']
    if not match_answer(preds.get('original',''), valid, task): continue
    ev_f = 0 if match_answer(preds.get('evidence_blur',''), valid, task) else 1
    sh_f = 0 if match_answer(preds.get('sham_blur',''), valid, task) else 1
    correct_items.append((ev_f, sh_f))

correct_items = np.array(correct_items)
np.random.seed(42)
gg_boots = []
gs_boots = []
for _ in range(1000):
    idx = np.random.choice(len(correct_items), len(correct_items), replace=True)
    s = correct_items[idx]
    cs_b = s[:,0].mean()
    si_b = s[:,1].mean()
    gg_b = cs_b - si_b
    gs_b = max(gg_b,0)/max(cs_b,1e-8)
    gg_boots.append(gg_b)
    gs_boots.append(gs_b)
print(f'GG 95% CI: [{np.percentile(gg_boots,2.5):.3f}, {np.percentile(gg_boots,97.5):.3f}]')
print(f'GS 95% CI: [{np.percentile(gs_boots,2.5):.3f}, {np.percentile(gs_boots,97.5):.3f}]')

# Ablation values
gg_gray = (flips['evidence_gray']-flips['sham_gray'])/n_correct if n_correct > 0 else 0
gg_swap = flips['evidence_swap']/n_correct if n_correct > 0 else 0
print(f'\nGG_blur: {gg:.3f}, GG_gray: {gg_gray:.3f}, GG_swap: {gg_swap:.3f}')

# Save metrics
metrics = {
    'model': 'Molmo-7B-D',
    'accuracy': round(acc, 4),
    'CS_blur': round(cs_blur, 4),
    'CS_gray': round(cs_gray, 4),
    'SI_blur': round(si_blur, 4),
    'GG': round(gg, 4),
    'SR': round(sr, 4),
    'GS': round(gs, 4),
    'GG_CI_lower': round(float(np.percentile(gg_boots, 2.5)), 3),
    'GG_CI_upper': round(float(np.percentile(gg_boots, 97.5)), 3),
    'GG_gray': round(gg_gray, 4),
    'GG_swap': round(gg_swap, 4),
}
with open(OUTPUT_DIR / 'metrics' / 'Molmo-7B-D_metrics.json','w') as f:
    json.dump(metrics, f, indent=2)
print('\nMetrics saved to Molmo-7B-D_metrics.json')
