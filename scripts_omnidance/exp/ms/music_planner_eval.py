"""Music-only native AR evaluation. Text diagnostics are not choreography scores."""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

BASELINE = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/music_planner_fixed2s_baseline_20261024_v1')
WINDOWS = ((0.0, 2.0), (2.0, 5.0625))
TIME_TOLERANCE_SECONDS = 0.05
# Unitless timestamps are rejected rather than silently treating milliseconds as seconds.
MARKER = re.compile(r'\[([^\[\]\n]*\d[^\[\]\n]*[-–—][^\[\]\n]*\d[^\[\]\n]*)\]')
TIME_PAIR = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*(s|sec|seconds?|ms)?\s*[-–—]\s*(\d+(?:\.\d+)?)\s*(s|sec|seconds?|ms)?\s*$', re.I)
WORDS = re.compile(r"[a-z]+(?:['’][a-z]+)?|[\u3400-\u9fff]", re.I)
ACTION_PATTERNS = {
    'step': r'\b(?:steps?|stepp?ing|stepped|walk\w*|footwork)\b|踏步|迈步',
    'turn': r'\b(?:turn\w*|pivot\w*|spin\w*|rotat\w*)\b|转身|旋转',
    'sway': r'\b(?:sway\w*|rock\w*|shift\w*)\b|摇摆|重心',
    'bend': r'\b(?:bend\w*|bent|crouch\w*|squat\w*|dip\w*)\b|屈膝|下蹲',
    'extend': r'\b(?:extend\w*|stretch\w*|reach\w*|open\w*)\b|伸展|展开',
    'lift': r'\b(?:lift\w*|rais\w*|jump\w*|hop\w*)\b|抬起|跳跃',
    'lower': r'\b(?:lower\w*|drop\w*)\b|放下|降低',
    'cross': r'\b(?:cross\w*|uncross\w*)\b|交叉',
    'sweep': r'\b(?:sweep\w*|swept|swing\w*|swung|wave\w*)\b|挥动|摆臂',
}
BODY_PATTERNS = {
    'arms': r'\b(?:arms?|elbows?|shoulders?)\b|手臂|肩|肘',
    'hands': r'\b(?:hands?|palms?|wrists?|fingers?)\b|手掌|手腕|手指',
    'legs': r'\b(?:legs?|knees?|thighs?)\b|腿|膝',
    'feet': r'\b(?:feet|foot|toes?|heels?)\b|脚|足',
    'torso': r'\b(?:torso|hips?|waist|chest|spine|trunk)\b|躯干|腰|胯|胸',
    'head': r'\b(?:head|neck|chin)\b|头|颈',
}


def file_ref(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': path.stat().st_size}


def checkpoint_identity(path, family=None, data=None):
    path = Path(path).resolve()
    marker = json.loads((path / 'trainer_state.json').read_text())
    if marker.get('schema') != 'music-planner-train-v1' or marker.get('complete') is not True:
        raise ValueError('Evaluation requires a completed portable training checkpoint')
    state = file_ref(path / 'trainable.pt')
    if state['sha256'] != marker['trainable_sha256']:
        raise ValueError('Checkpoint trainable bytes changed')
    if not (path / 'adapter').is_dir():
        raise ValueError('Portable native adapter export missing')
    identity = {'trainer_state': file_ref(path / 'trainer_state.json'), 'trainable': state,
                'fingerprint': marker['fingerprint'], 'cursor': marker['cursor']}
    if family is not None:
        config_path = path.parent / 'run_config.json'
        config = json.loads(config_path.read_text())
        canonical = {k: v for k, v in config.items() if k not in ('fingerprint', 'trainable_parameters')}
        fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if fingerprint != marker['fingerprint'] or config['fingerprint'] != fingerprint or config['family'] != family:
            raise ValueError('Checkpoint training config/family identity differs')
        if config['data'].get('test_used') is not False:
            raise ValueError('Checkpoint training/selection consumed test data')
        for split in ('train', 'val'):
            if config['data'][split]['sha256'] != file_ref(Path(data) / f'{split}.jsonl')['sha256']:
                raise ValueError('Checkpoint uses another training/validation benchmark')
        identity['run_config'] = file_ref(config_path)
        identity['training_config'] = config
    return identity


def verify_base_identity(path, trained_identity):
    path = Path(path).resolve()
    config_ref = file_ref(path / 'config.json')
    if config_ref['sha256'] != trained_identity['config_sha256']:
        raise ValueError('Evaluation base config differs from checkpoint training')
    weights = sorted(set(path.glob('*.safetensors')) | set(path.glob('*.bin')))
    actual = {p.name: p.stat().st_size for p in weights}
    expected = {item['name']: item['size'] for item in trained_identity['weights']}
    if not actual or actual != expected:
        raise ValueError('Evaluation base weights membership/size differs')
    processor_refs = {}
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json', 'tokenizer.json',
                 'tokenizer_config.json', 'preprocessor_config.json', 'processor_config.json', 'chat_template.jinja'):
        if name in trained_identity:
            processor_refs[name] = file_ref(path / name)
            if processor_refs[name]['sha256'] != trained_identity[name]:
                raise ValueError('Evaluation tokenizer/processor/index differs from checkpoint training')
    return {'path': str(path), 'config': config_ref, 'processor_refs': processor_refs,
            'weights': [{'name': name, 'size_bytes': size} for name, size in sorted(actual.items())],
            'weight_content_SHA_verified': False,
            'identity_method': 'config/processor SHA and weight membership/size only; relocated mtime ignored'}


def word_units(text):
    return WORDS.findall(text.lower())


def parse_windows(text, expected=WINDOWS, tolerance=TIME_TOLERANCE_SECONDS):
    matches = list(MARKER.finditer(text))
    parsed, invalid = [], []
    for marker in matches:
        value = TIME_PAIR.fullmatch(marker.group(1))
        if not value or not (value.group(2) or value.group(4)):
            invalid.append(marker.group(0))
            continue
        a, unit_a, b, unit_b = value.groups()
        unit_a = (unit_a or unit_b).lower()
        unit_b = (unit_b or unit_a).lower()
        start = float(a) / (1000 if unit_a == 'ms' else 1)
        end = float(b) / (1000 if unit_b == 'ms' else 1)
        next_start = next((m.start() for m in matches if m.start() > marker.start()), len(text))
        parsed.append({'start': start, 'end': end, 'text': text[marker.end():next_start].strip()})
    valid_ranges = all(math.isfinite(w['start']) and math.isfinite(w['end']) and w['end'] > w['start'] for w in parsed)
    ordered = valid_ranges and all(a['end'] <= b['start'] + tolerance for a, b in zip(parsed, parsed[1:]))
    boundaries = (len(parsed) == len(expected) and not invalid and valid_ranges and ordered and
                  all(abs(w['start'] - start) <= tolerance and abs(w['end'] - end) <= tolerance
                      for w, (start, end) in zip(parsed, expected)))
    return {'windows': parsed, 'invalid_time_markers': invalid, 'window_count': len(parsed),
            'valid_ranges': valid_ranges, 'ordered_nonoverlapping': ordered,
            'matches_fixed_windows': boundaries, 'missing_time_windows': not matches,
            'nonempty_window_text': bool(parsed) and all(w['text'] for w in parsed),
            'tolerance_seconds': tolerance}


def text_metrics(text):
    timed = parse_windows(text)
    body_text = MARKER.sub('', text)
    sentences = [re.sub(r'\s+', ' ', s).strip().lower()
                 for s in re.split(r'(?<=[.!?。！？])\s*|\n+', body_text)]
    sentences = [s for s in sentences if word_units(s)]
    counts = Counter(sentences)
    words = word_units(body_text)
    ngrams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    actions = sorted(k for k, pat in ACTION_PATTERNS.items() if re.search(pat, body_text, re.I))
    body = sorted(k for k, pat in BODY_PATTERNS.items() if re.search(pat, body_text, re.I))
    return {'char_count': len(text), 'word_count': len(words), 'sentence_count': len(sentences),
            'exact_sentence_max_count': max(counts.values(), default=0),
            'repeated_sentence_fraction': (len(sentences) - len(counts)) / len(sentences) if sentences else 0.0,
            'repeated_4gram_fraction': (len(ngrams) - len(set(ngrams))) / len(ngrams) if ngrams else 0.0,
            'distinct_1': len(set(words)) / len(words) if words else 0.0,
            'action_categories': actions, 'action_category_count': len(actions),
            'body_categories': body, 'body_category_count': len(body), 'temporal': timed,
            'diagnostic_only_not_quality_score': True}


def normalize_nll(nll_sum, assistant_tokens, target_text):
    if (not math.isfinite(nll_sum) or nll_sum < 0 or type(assistant_tokens) is not int or assistant_tokens <= 0):
        raise ValueError('Invalid teacher-forced NLL/count')
    chars, words = len(target_text), len(word_units(target_text))
    if not chars or not words:
        raise ValueError('Empty reference text cannot normalize NLL')
    return {'nll_sum': nll_sum, 'assistant_tokens': assistant_tokens, 'reference_chars': chars,
            'reference_words': words, 'within_model_nll_per_token': nll_sum / assistant_tokens,
            'nll_per_char': nll_sum / chars, 'nll_per_word': nll_sum / words,
            'interpretation': 'reference likelihood only; not choreography quality'}


def classify_stop(new_token_ids, eos_token_ids, budget, native_stop=False):
    if type(budget) is not int or budget <= 0 or len(new_token_ids) > budget:
        raise ValueError('Invalid generation budget/count')
    eos = set(eos_token_ids)
    if new_token_ids and new_token_ids[-1] in eos:
        return 'eos'
    if native_stop:
        return 'native_stop'
    if len(new_token_ids) == budget:
        return 'budget_truncation'
    return 'unknown_stop'


def select_rows(rows, mode, limit=None, selection=None, seed=42):
    if mode not in ('val', 'test'):
        raise ValueError('Only val/test evaluation is permitted')
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError('Limit must be a positive integer')
    if selection is not None and (not isinstance(selection, list) or any(not isinstance(sid, str) for sid in selection)):
        raise ValueError('Selection must be a JSON list of source IDs')
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate source IDs')
    if selection is not None:
        if len(set(selection)) != len(selection) or any(sid not in by_id for sid in selection):
            raise ValueError('Selection must be unique IDs from the declared split')
        selected = [by_id[sid] for sid in selection]
        if limit is not None:
            selected = selected[:limit]
    else:
        if mode == 'test' and limit is None:
            raise ValueError('Test needs explicit --limit (500 for full test) or --selection')
        n = 32 if limit is None else limit
        if type(n) is not int or n <= 0:
            raise ValueError('Limit must be a positive integer')
        selected = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())[:n]
    if not selected:
        raise ValueError('Empty evaluation selection')
    return selected


def validate_data(data):
    data = Path(data).resolve()
    summary = json.loads((data / 'summary.json').read_text())
    if (summary['schema'] != 'music_planner_fixed2s_weak_baseline_v1' or
            summary['split_counts'] != {'train': 10713, 'val': 490, 'test': 500} or
            summary['source_and_exact_audio_split_overlap'] != 0):
        raise ValueError('Not the cleaned fixed music planner baseline')
    exclusion_ref = summary['training_exclusion_receipt_ref']
    if file_ref(data / 'exclusion_receipt.json') != exclusion_ref:
        raise ValueError('Training exclusion receipt changed')
    exclusions = json.loads((data / 'exclusion_receipt.json').read_text())
    excluded_ids = {row['id'] for row in exclusions['removed']}
    if (exclusions['schema'] != 'music_planner_training_exclusion_v1' or
            len(excluded_ids) != 7 or exclusions['exclusion_count'] != 7 or
            exclusions['clean_train_count'] != 10713 or
            exclusions['val_and_test_bytes_unchanged'] is not True):
        raise ValueError('Clean training exclusion contract differs')
    parent_ref = exclusions['parent_summary_ref']
    if file_ref(parent_ref['path']) != parent_ref:
        raise ValueError('Parent baseline summary changed')
    parent = json.loads(Path(parent_ref['path']).read_text())
    parent_train = Path(parent['split_artifacts']['train']['path']).read_bytes()
    if hashlib.sha256(parent_train).hexdigest() != parent['split_artifacts']['train']['sha256']:
        raise ValueError('Parent training manifest changed')
    expected_train = b''.join(line for line in parent_train.splitlines(keepends=True)
                              if not line.strip() or json.loads(line)['id'] not in excluded_ids)
    if (data / 'train.jsonl').read_bytes() != expected_train:
        raise ValueError('Clean training manifest is not the exact ordered exclusion')
    for split in ('val', 'test'):
        if summary['split_artifacts'][split]['sha256'] != parent['split_artifacts'][split]['sha256']:
            raise ValueError('Held-out split changed during training cleanup')
    splits, source_sets, audio_sets = {}, {}, {}
    for split in ('train', 'val', 'test'):
        path = data / f'{split}.jsonl'
        ref = file_ref(path)
        if ref['sha256'] != summary['split_artifacts'][split]['sha256']:
            raise ValueError(f'{split} input split hash differs')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != summary['split_counts'][split] or len({r['id'] for r in rows}) != len(rows):
            raise ValueError(f'{split} input split membership/count differs')
        for row in rows:
            if row['audio_window_seconds'] != [0.0, 5.0625] or not parse_windows(row['target_text'])['matches_fixed_windows']:
                raise ValueError('Reference target/window differs from fixed two-window protocol')
        splits[split] = rows
        source_sets[split] = {r['source_group_id'] for r in rows}
        audio_sets[split] = {r['audio_sha256'] for r in rows}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if source_sets[a] & source_sets[b] or audio_sets[a] & audio_sets[b]:
            raise ValueError('Source or exact audio split leakage')
    return splits, summary


def make_conditions(rows, counterfactual=False):
    conditions = []
    for index, row in enumerate(rows):
        conditions.append((row, 'original', row))
        if counterfactual:
            donor = next((rows[(index + offset) % len(rows)] for offset in range(1, len(rows))
                          if rows[(index + offset) % len(rows)]['source_group_id'] != row['source_group_id'] and
                          rows[(index + offset) % len(rows)]['audio_sha256'] != row['audio_sha256']), None)
            if donor is None:
                raise ValueError('Counterfactual needs another source and nonidentical exact audio')
            conditions.extend(((row, 'mismatched', donor), (row, 'silence', row)))
    return conditions


def audio_only_input(source_row, donor_row):
    # Whitelist: no target, target_windows, reference image, or source video passed to generator.
    return {'id': source_row['id'], 'audio_path': donor_row['audio_path'],
            'audio_sha256': donor_row['audio_sha256'],
            'audio_window_seconds': list(donor_row['audio_window_seconds'])}


def counterfactual_metrics(records):
    groups = defaultdict(dict)
    for row in records:
        key, condition = row['id'], row['condition']
        if condition in groups[key]:
            raise ValueError('Duplicate source/condition output')
        groups[key][condition] = row
    pairs, incomplete = [], []
    for sid, group in groups.items():
        if set(group) != {'original', 'mismatched', 'silence'}:
            incomplete.append(sid)
            continue
        original, wrong, silent = (group[key] for key in ('original', 'mismatched', 'silence'))
        if (any(row['source_group_id'] != original['source_group_id'] for row in group.values()) or
                original['source_group_id'] == wrong['audio_source_group_id'] or
                original['audio_sha256'] == wrong['audio_sha256'] or
                silent['audio_sha256'] != original['audio_sha256'] or
                any(row.get('failure') for row in group.values())):
            incomplete.append(sid)
            continue
        for condition, other in (('mismatched', wrong), ('silence', silent)):
            a, b = word_units(original['raw_text']), word_units(other['raw_text'])
            sa, sb = set(a), set(b)
            pairs.append({'id': sid, 'condition': condition,
                          'text_edit_change': 1 - SequenceMatcher(None, a, b, autojunk=False).ratio(),
                          'word_set_jaccard_change': 1 - len(sa & sb) / len(sa | sb) if sa | sb else 0.0,
                          'raw_text_identical': original['raw_text'] == other['raw_text']})
    return {'pairs': pairs, 'incomplete_or_invalid_source_ids': incomplete,
            'interpretation': 'Audio responsiveness diagnostic only. More change is NOT better synchrony.',
            'music_synchronization_score': None,
            'donor_policy': 'Different known source group and exact audio SHA; song/near-duplicate separation is NOT verified'}


def summarize(records):
    complete = [r for r in records if not r.get('failure')]
    original = [r for r in complete if r['condition'] == 'original']
    all_words = [w for row in original for w in word_units(row['raw_text'])]
    all_bigrams = [pair for row in original for pair in zip(word_units(row['raw_text']), word_units(row['raw_text'])[1:])]
    summary = {'attempted': len(records), 'successful': len(complete), 'failed': len(records) - len(complete),
               'stop_reason_counts': dict(Counter(r.get('stop_reason', 'failure') for r in records)),
               'original_sources': len(original),
               'teacher_forced_failures': sum(bool(r.get('teacher_forced_failure')) for r in records),
               'corpus_distinct_1': len(set(all_words)) / len(all_words) if all_words else None,
               'corpus_distinct_2': len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else None,
               'diagnostics_by_condition': {},
               'independent_audio_VLM_judge': {'status': 'pending_not_run', 'scores': None},
               'human_review': {'status': 'pending_not_run', 'scores': None},
               'winner': None, 'automatic_quality_ranking': False}
    for condition in sorted({r['condition'] for r in complete}):
        subset = [r for r in complete if r['condition'] == condition]
        summary['diagnostics_by_condition'][condition] = {
            'sources': len(subset),
            'valid_two_window_fraction': sum(r['metrics']['temporal']['matches_fixed_windows'] for r in subset) / len(subset),
            **{f'mean_{key}': sum(r['metrics'][key] for r in subset) / len(subset)
               for key in ('word_count', 'exact_sentence_max_count', 'repeated_sentence_fraction',
                           'repeated_4gram_fraction', 'action_category_count', 'body_category_count')}}
    nll = [r['teacher_forced'] for r in original if r.get('teacher_forced')]
    if nll:
        total = sum(r['nll_sum'] for r in nll)
        summary['reference_likelihood'] = {'sources': len(nll),
            'within_model_nll_per_token': total / sum(r['assistant_tokens'] for r in nll),
            'nll_per_char': total / sum(r['reference_chars'] for r in nll),
            'nll_per_word': total / sum(r['reference_words'] for r in nll)}
    summary['counterfactual'] = counterfactual_metrics(records) if any(r['condition'] != 'original' for r in records) else None
    return summary


def review_request(records):
    return {'status': 'pending_not_run', 'external_upload_performed': False,
            'authorization_required_before_private_audio_upload': True,
            'generator_self_judging_is_ground_truth': False,
            'blind_review_protocol': 'Randomize anonymous model names and A/B positions; retain rationale, unknown and tie.',
            'axes': ['usable continuous choreography', 'musical rhythm/structure plausibility',
                     'repetition and diversity', 'person/image quality and frozen H3 renderer non-regression'],
            'not_sufficient': ['GT semantic match alone (multiple valid choreographies)', 'text length',
                               'audio counterfactual change magnitude', 'keypoints/BAS/TA alone'],
            'video_followup': 'Exactly three fixed cases/config; same raw caption+reference image, frozen pure-video H3, seed/settings.',
            'local_items': [{'id': r['id'], 'condition': r['condition'], 'raw_text': r.get('raw_text'),
                             'source_audio_ref': r['audio_ref'],
                             'required_waveform_transform': 'zero_waveform_preserve_length' if r['condition'] == 'silence' else 'none',
                             'reference_target': r['reference_target'],
                             'failure': r.get('failure')} for r in records]}


def evaluate_one(adapter, source, condition, donor, max_new_tokens, seed=42, teacher_forced=False):
    record = {'id': source['id'], 'source_group_id': source['source_group_id'], 'condition': condition,
              'audio_source_id': donor['id'], 'audio_source_group_id': donor['source_group_id'],
              'audio_sha256': donor['audio_sha256'], 'audio_window_seconds': donor['audio_window_seconds'],
              'audio_transform': 'zero_waveform_preserve_length' if condition == 'silence' else 'none',
              'reference_target': source['target_text'], 'reference_target_windows': source['target_windows'],
              'generation_budget': max_new_tokens, 'raw_text': '', 'failure': None}
    start = time.monotonic()
    try:
        record['audio_ref'] = file_ref(donor['audio_path'])
        if record['audio_ref']['sha256'] != donor['audio_sha256']:
            raise ValueError('Audio content changed from fixed split')
        result = adapter.generate(audio_only_input(source, donor), max_new_tokens=max_new_tokens,
                                  silence=condition == 'silence', seed=seed)
        record['native_output'] = result
        record['raw_text'] = result['raw_text']
        record['new_token_ids'] = result['new_token_ids']
        record['eos_token_ids'] = result['eos_token_ids']
        record['native_stop_reason'] = result.get('stop_reason')
        if not isinstance(record['raw_text'], str):
            raise ValueError('Native adapter must return raw_text without normalization')
        record['stop_reason'] = classify_stop(result['new_token_ids'], result['eos_token_ids'],
                                               max_new_tokens, result.get('native_stop', False))
        record['generated_tokens'] = len(result['new_token_ids'])
        record['metrics'] = text_metrics(record['raw_text'])
        if teacher_forced and condition == 'original':
            try:
                nll = adapter.teacher_forced(source)
                record['teacher_forced'] = normalize_nll(nll['sum'], nll['token_count'], source['target_text'])
            except Exception as exc:
                record['teacher_forced_failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    except Exception as exc:
        record['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        record.setdefault('audio_ref', {'path': donor['audio_path'], 'sha256': donor['audio_sha256']})
    record['elapsed_seconds'] = time.monotonic() - start
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True, choices=['moss', 'music_flamingo', 'omni_captioner'])
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=BASELINE)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--selection', type=Path, help='JSON list of same fixed source IDs for all models')
    parser.add_argument('--mode', choices=['val', 'test'], default='val')
    parser.add_argument('--counterfactual', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto', help='auto selects visible CUDA, otherwise CPU')
    parser.add_argument('--teacher-forced', action='store_true', help='Additional reference likelihood, never replaces AR')
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error('Budgets/limits must be positive')
    if args.out.exists():
        parser.error('Output must be fresh; do not overwrite an earlier evaluation')
    splits, data_summary = validate_data(args.data)
    selection = json.loads(args.selection.read_text()) if args.selection else None
    rows = select_rows(splits[args.mode], args.mode, args.limit, selection, args.seed)
    conditions = make_conditions(rows, args.counterfactual)
    checkpoint_ref = checkpoint_identity(args.checkpoint, args.family, args.data)
    base_ref = verify_base_identity(args.model, checkpoint_ref['training_config']['model'])
    random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Native loading/generation implementation is shared with the training worker.
    from music_planner_adapters import load_planner_adapter
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'music_planner_ar_eval_v1', 'family': args.family, 'model': str(args.model.resolve()),
                'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_identity': checkpoint_ref,
                'base_identity': base_ref, 'device': device, 'mode': args.mode,
                'data_summary': file_ref(args.data / 'summary.json'),
                'split': file_ref(args.data / f'{args.mode}.jsonl'),
                'selection_ref': file_ref(args.selection) if args.selection else None,
                'source_ids': [r['id'] for r in rows], 'code': file_ref(__file__),
                'adapter_code': file_ref(__import__('music_planner_adapters').__file__),
                'decoding': {'temperature': 0.0, 'do_sample': False, 'seed': args.seed,
                             'max_new_tokens': args.max_new_tokens, 'added_penalties': False,
                             'GT_length_truncation': False, 'raw_output_cleaning': False},
                'source_and_exact_audio_isolation_verified': True,
                'song_and_near_duplicate_isolation_verified': data_summary['song_and_near_duplicate_isolation_verified'],
                'test_used_for_checkpoint_selection': False, 'complete': False}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    try:
        adapter = load_planner_adapter(family=args.family, model_path=args.model, checkpoint=args.checkpoint,
                                       device=device, max_length=checkpoint_ref['training_config']['max_length'])
    except Exception as exc:
        manifest['load_failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
        raise
    records = []
    with (args.out / 'raw.jsonl').open('x') as stream:
        for source, condition, donor in conditions:
            record = evaluate_one(adapter, source, condition, donor, args.max_new_tokens, args.seed, args.teacher_forced)
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
    (args.out / 'summary.json').write_text(json.dumps(summarize(records), ensure_ascii=False, indent=2, allow_nan=False))
    (args.out / 'review_request.json').write_text(json.dumps(review_request(records), ensure_ascii=False, indent=2, allow_nan=False))
    manifest['complete'] = not any(r['failure'] for r in records)
    manifest['actual_generated_tokens'] = sum(r.get('generated_tokens', 0) for r in records)
    manifest['attempted'] = len(records)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    if not manifest['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
