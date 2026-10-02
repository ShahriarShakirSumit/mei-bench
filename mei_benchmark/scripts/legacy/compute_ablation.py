#!/usr/bin/env python3
"""Compute GG_blur, GG_gray, GG_swap for all 10 models from prediction files."""
from mei_benchmark.paths import DATA_DIR, OUTPUT_DIR
import json, re, numpy as np
from collections import defaultdict
from difflib import SequenceMatcher
from scipy.stats import spearmanr

# Load benchmark items
items = {}
with open(DATA_DIR / 'items.jsonl') as f:
    for line in f:
        d = json.loads(line)
        items[d['item_id']] = d

def extract_core_answer(answer, task_type, valid_answers):
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
    return answer.split('.')[0].strip()

def match_answer(answer, valid_answers, task_type, use_extraction=False):
    if use_extraction:
        core = extract_core_answer(answer, task_type, valid_answers)
    else:
        core = answer
    core_norm = core.lower().strip(' .,!?')
    for v in valid_answers:
        v_norm = v.lower().strip(' .,!?')
        if core_norm == v_norm:
            return True
        if v_norm in core_norm or core_norm in v_norm:
            return True
        threshold = 0.7 if use_extraction else 0.8
        if SequenceMatcher(None, core_norm, v_norm).ratio() >= threshold:
            return True
    return False

models = [
    'LLaVA-1.5-7B', 'LLaVA-1.5-13B', 'LLaVA-NeXT-7B',
    'Qwen2-VL-7B', 'InternVL2-8B', 'InternVL2.5-8B',
    'Phi-3.5-Vision', 'Llama-3.2-11B-Vision', 'DeepSeek-VL-7B', 'Molmo-7B-D'
]

gg_blur_all = []
gg_gray_all = []
gg_swap_all = []

for model in models:
    pred_file = f'{OUTPUT_DIR}/predictions/{model}_predictions.jsonl'
    use_ext = (model == 'Molmo-7B-D')
    
    by_item = defaultdict(dict)
    with open(pred_file) as f:
        for line in f:
            d = json.loads(line)
            by_item[d['item_id']][d['intervention_type']] = d['answer']
    
    n_correct = 0
    flip_eb = flip_eg = flip_es = flip_sb = flip_sg = 0
    
    for item_id, preds in by_item.items():
        if item_id not in items: continue
        item = items[item_id]
        valid = item['valid_answers']
        task = item['task_type']
        orig = preds.get('original', '')
        if not match_answer(orig, valid, task, use_ext): continue
        n_correct += 1
        if not match_answer(preds.get('evidence_blur',''), valid, task, use_ext): flip_eb += 1
        if not match_answer(preds.get('evidence_gray',''), valid, task, use_ext): flip_eg += 1
        if not match_answer(preds.get('evidence_swap',''), valid, task, use_ext): flip_es += 1
        if not match_answer(preds.get('sham_blur',''), valid, task, use_ext): flip_sb += 1
        if not match_answer(preds.get('sham_gray',''), valid, task, use_ext): flip_sg += 1
    
    cs_blur = flip_eb/n_correct if n_correct else 0
    si_blur = flip_sb/n_correct if n_correct else 0
    gg_blur = cs_blur - si_blur
    
    cs_gray = flip_eg/n_correct if n_correct else 0
    si_gray = flip_sg/n_correct if n_correct else 0
    gg_gray = cs_gray - si_gray
    
    gg_swap = flip_es/n_correct if n_correct else 0  # no sham_swap
    
    display = model.replace('-Vision', '').replace('-11B-Vision', '-11B')
    print(f'{display:20s}: nc={n_correct:3d}, GG_blur={gg_blur:.3f}, GG_gray={gg_gray:.3f}, GG_swap={gg_swap:.3f}')
    
    gg_blur_all.append(gg_blur)
    gg_gray_all.append(gg_gray)
    gg_swap_all.append(gg_swap)

print(f'\nMean GG_blur: {np.mean(gg_blur_all):.3f}')
print(f'Mean GG_gray: {np.mean(gg_gray_all):.3f}')
print(f'Mean GG_swap: {np.mean(gg_swap_all):.3f}')

rho_gray, _ = spearmanr(gg_blur_all, gg_gray_all)
rho_swap, _ = spearmanr(gg_blur_all, gg_swap_all)
print(f'Spearman rho (gray vs blur): {rho_gray:.3f}')
print(f'Spearman rho (swap vs blur): {rho_swap:.3f}')
