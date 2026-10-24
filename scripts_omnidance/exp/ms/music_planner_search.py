"""Finite music-planner search and reviewed RSI data staging, never a reward model.

The fair adapter remains greedy. This module explicitly samples its native prefix,
keeps raw output, and requires external, version-bound reviews before selection.
Importing or validating staging artifacts does not load torch or any model.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

from scripts_omnidance.exp.ms.music_planner_eval import classify_stop, parse_windows, text_metrics

FAMILIES = ('moss', 'music_flamingo', 'omni_captioner')
GROUPS = ('source_group_id', 'song_group_id', 'near_duplicate_group_id')
DIMENSIONS = ('music_fit', 'genre_compatibility', 'transition_coherence')
SHA = re.compile(r'^[0-9a-f]{64}$')


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'{name}: expected exactly {sorted(keys)}')


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _sha(value, name):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError(f'{name} must be a lowercase SHA256')


def _number(value, low, high, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite in [{low},{high}]')


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'Non-finite JSON value: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=invalid)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to replace earlier evidence or reviews accidentally.
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                allow_nan=False) + '\n')


def validate_isolation(manifest, require_verified=False):
    _keys(manifest, ('schema', 'version', 'verification', 'members'), 'isolation manifest')
    if manifest['schema'] != 'music_planner_isolation_v1':
        raise ValueError('Unknown isolation manifest schema')
    _text(manifest['version'], 'isolation version')
    _keys(manifest['verification'], GROUPS, 'group verification')
    for key in GROUPS:
        proof = manifest['verification'][key]
        _keys(proof, ('status', 'evidence_sha256'), f'{key} verification')
        if proof['status'] == 'verified':
            _sha(proof['evidence_sha256'], f'{key} evidence')
        elif proof['status'] == 'unknown' and proof['evidence_sha256'] is None:
            if require_verified:
                raise ValueError(f'{key} grouping must be verified; unknown cannot approve training')
        else:
            raise ValueError('Grouping verification must be verified with evidence or explicitly unknown/null')
    if not isinstance(manifest['members'], list) or not manifest['members']:
        raise ValueError('Isolation requires members from all train/dev/test splits')
    by_id, owners, splits = {}, defaultdict(dict), set()
    for member in manifest['members']:
        _keys(member, ('id', 'split', 'audio_sha256') + GROUPS, 'isolation member')
        sid, split = _text(member['id'], 'source id'), member['split']
        if split not in ('train', 'dev', 'test') or sid in by_id:
            raise ValueError('Invalid split or duplicate source id in isolation manifest')
        _sha(member['audio_sha256'], 'member audio')
        for key in GROUPS + ('audio_sha256',):
            value = member[key]
            if value is None and key in GROUPS and manifest['verification'][key]['status'] == 'unknown':
                continue
            _text(value, key)
            previous = owners[key].setdefault(value, split)
            if previous != split:
                raise ValueError(f'Cross-split leakage through {key}: {value}')
        by_id[sid] = member
        splits.add(split)
    if splits != {'train', 'dev', 'test'}:
        raise ValueError('Complete isolation inventory must include train, dev, and test')
    return by_id


def validate_generator(generator):
    _keys(generator, ('family', 'model_id', 'base_model_id', 'model_sha256', 'version'), 'generator')
    if generator['family'] not in FAMILIES:
        raise ValueError('Unknown native planner family')
    for key in ('model_id', 'base_model_id', 'version'):
        _text(generator[key], f'generator {key}')
    _sha(generator['model_sha256'], 'generator model identity')


def validate_sampling(sampling):
    _keys(sampling, ('temperature', 'top_p', 'top_k', 'max_new_tokens'), 'sampling')
    _number(sampling['temperature'], .000001, 100., 'temperature')
    _number(sampling['top_p'], .000001, 1., 'top_p')
    if type(sampling['top_k']) is not int or sampling['top_k'] < 0:
        raise ValueError('top_k must be a nonnegative integer')
    if type(sampling['max_new_tokens']) is not int or sampling['max_new_tokens'] <= 0:
        raise ValueError('max_new_tokens must be a positive integer')


def sample_native(adapter, row, *, seed, temperature=.9, top_p=.95, top_k=50,
                  max_new_tokens=2048):
    """Stochastic native decoding; never monkey-patches the fair greedy adapter."""
    import torch
    params = dict(temperature=temperature, top_p=top_p, top_k=top_k,
                  max_new_tokens=max_new_tokens)
    validate_sampling(params)
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError('seed must be an integer in [0,2**63)')
    # prefix retains the native family template, SHA check, full waveform length,
    # processor resampling, and context checks. Only audio keys reach it.
    batch = adapter.prefix(row)
    prefix = batch['input_ids']
    if prefix.shape[1] + max_new_tokens > adapter.max_length:
        raise ValueError('Generation budget exceeds context; truncation forbidden')
    adapter.model.eval()
    parameter = next(adapter.model.parameters())
    if parameter.device.type not in ('cpu', 'cuda'):
        raise ValueError('Sampling currently supports CPU/CUDA only')
    devices = [] if parameter.device.type == 'cpu' else [
        parameter.device.index if parameter.device.index is not None else torch.cuda.current_device()]
    decoding = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                    num_beams=1, repetition_penalty=1.0, no_repeat_ngram_size=0,
                    encoder_repetition_penalty=1.0)
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for device in devices:
            torch.cuda.default_generators[device].manual_seed(seed)
        with torch.autocast(parameter.device.type, dtype=torch.bfloat16,
                            enabled=parameter.dtype == torch.bfloat16):
            output = adapter.model.generate(**adapter._move(batch), **decoding,
                                            max_new_tokens=max_new_tokens, use_cache=True)
    sequence = output.sequences if hasattr(output, 'sequences') else output
    if isinstance(sequence, tuple):
        sequence = sequence[0]
    if sequence.ndim != 2 or sequence.shape[0] != 1:
        raise ValueError('Native model returned an invalid generation')
    if not torch.equal(sequence[:, :prefix.shape[1]].cpu(), prefix.cpu()):
        raise ValueError('Native generation changed its input prefix')
    tokens = sequence[0, prefix.shape[1]:].cpu().tolist()
    eos = adapter.eos_token_ids()
    stop = classify_stop(tokens, eos, max_new_tokens)
    return dict(raw_text=adapter.tokenizer.decode(tokens, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False),
                raw_with_special_tokens=adapter.tokenizer.decode(tokens, skip_special_tokens=False,
                                                                 clean_up_tokenization_spaces=False),
                new_token_ids=tokens, eos_token_ids=eos, stop_reason=stop,
                budget_truncated=stop == 'budget_truncation', seed=seed, decoding=decoding,
                input_tokens=prefix.shape[1], prefix_sha256=object_sha(prefix.cpu().tolist()),
                audio_input='full-waveform', silence_applied=False, **adapter._last_audio_info)


def _audio_input(row, member):
    # Ground-truth actions, genre, images and videos never enter generation.
    return dict(id=member['id'], source_group_id=member['source_group_id'],
                audio_path=str(Path(row['audio_path']).expanduser().resolve()),
                audio_sha256=member['audio_sha256'], audio_window_seconds=[0., 5.0625])


def validate_selected_rows(rows, isolation):
    members = validate_isolation(isolation)
    if not isinstance(rows, list) or not rows:
        raise ValueError('No selected source rows; require a JSON list')
    selected, seen = [], set()
    for row in sorted(rows, key=lambda r: r.get('id', '')):
        sid = row.get('id')
        if sid not in members or sid in seen:
            raise ValueError('Unknown or duplicate selected source')
        member = members[sid]
        if member['split'] == 'test':
            raise ValueError('Test is final-only; no candidate search or RSI on test')
        allowed = (member['split'], 'val') if member['split'] == 'dev' else (member['split'],)
        if row.get('split') not in allowed:
            raise ValueError('Selected source split differs from isolation inventory')
        for key in GROUPS + ('audio_sha256',):
            if row.get(key, member[key]) != member[key]:
                raise ValueError(f'Selected source differs from manifest: {key}')
        if row.get('audio_window_seconds', [0., 5.0625]) != [0., 5.0625]:
            raise ValueError('Physical audio window crop forbidden')
        audio = _audio_input(row, member)
        if file_sha(audio['audio_path']) != member['audio_sha256']:
            raise ValueError('Selected audio bytes differ from isolation manifest')
        selected.append((member, audio))
        seen.add(sid)
    return selected


def _diagnostics(generation, budget):
    for key in ('new_token_ids', 'eos_token_ids'):
        ids = generation.get(key)
        if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError(f'Invalid generation {key}')
    if not generation['eos_token_ids']:
        raise ValueError('Missing EOS identity')
    if not isinstance(generation.get('raw_text'), str):
        raise ValueError('Generation must preserve raw_text')
    if not isinstance(generation.get('raw_with_special_tokens'), str):
        raise ValueError('Generation must preserve raw decode with special tokens, including empty output')
    _sha(generation.get('prefix_sha256'), 'native prefix')
    if type(generation.get('seed')) is not int or not 0 <= generation['seed'] < 2 ** 63:
        raise ValueError('Invalid recorded sampling seed')
    stop = classify_stop(generation['new_token_ids'], generation['eos_token_ids'], budget)
    if generation.get('stop_reason') != stop or generation.get('budget_truncated') is not (stop == 'budget_truncation'):
        raise ValueError('Recorded stop reason differs from raw token evidence')
    if generation.get('audio_input') != 'full-waveform' or generation.get('silence_applied') is not False:
        raise ValueError('Search requires the complete original music waveform')
    if generation.get('processor_sample_rate') != 16000 or type(generation.get('n_samples')) is not int:
        raise ValueError('Missing native audio length evidence')
    if abs(generation['n_samples'] - 81000) > 5:
        raise ValueError('Native audio length no longer matches the full source')
    metrics = text_metrics(generation['raw_text'])
    temporal = metrics['temporal']
    eligible = (stop == 'eos' and temporal['matches_fixed_windows'] and temporal['nonempty_window_text']
                and '<|' not in generation['raw_text'] and '<audio>' not in generation['raw_text'])
    return dict(text=metrics, format_eligible=bool(eligible),
                diagnostic_only_not_music_reward=True)


def generate_candidates(rows, isolation, generator, *, count=4, seed=42, sampling=None,
                        sampler=None):
    """Testable finite orchestration; sampler receives strictly audio-only input."""
    selected = validate_selected_rows(rows, isolation)
    validate_generator(generator)
    if type(count) is not int or not 2 <= count <= 32:
        raise ValueError('Require 2..32 explicitly sampled candidates per source')
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError('Invalid base seed')
    if sampler is None:
        raise ValueError('An explicit sampled native adapter is required')
    sampling = sampling or dict(temperature=.9, top_p=.95, top_k=50, max_new_tokens=2048)
    validate_sampling(sampling)
    candidates = []
    for member, audio in selected:
        sid = member['id']
        for index in range(count):
            derived = int(object_sha(dict(seed=seed, source=sid, candidate=index))[:15], 16)
            generation = sampler(audio, seed=derived, **sampling)
            expected = dict(do_sample=True, temperature=sampling['temperature'], top_p=sampling['top_p'],
                            top_k=sampling['top_k'], num_beams=1, repetition_penalty=1.0,
                            no_repeat_ngram_size=0, encoder_repetition_penalty=1.0)
            if generation.get('decoding') != expected or generation.get('seed') != derived:
                raise ValueError('Sampler did not record the requested stochastic decoding/seed')
            item = dict(source=dict(member, audio_path=audio['audio_path'],
                                    audio_window_seconds=[0., 5.0625]),
                        index=index, generator=generator, generation=generation,
                        diagnostics=_diagnostics(generation, sampling['max_new_tokens']))
            item['candidate_sha256'] = object_sha(item)
            candidates.append(item)
    pool = dict(schema='music_planner_candidate_pool_v1', isolation=isolation,
                isolation_sha256=object_sha(isolation), generator=generator, sampling=sampling,
                candidates_per_source=count, candidates=candidates)
    pool['run_sha256'] = object_sha(pool)
    validate_pool(pool)
    return pool


def validate_pool(pool):
    _keys(pool, ('schema', 'isolation', 'isolation_sha256', 'generator', 'sampling',
                 'candidates_per_source', 'candidates', 'run_sha256'), 'candidate pool')
    if pool['schema'] != 'music_planner_candidate_pool_v1':
        raise ValueError('Unknown candidate pool schema')
    if pool['run_sha256'] != object_sha({k: v for k, v in pool.items() if k != 'run_sha256'}):
        raise ValueError('Candidate pool run hash differs')
    members = validate_isolation(pool['isolation'])
    if pool['isolation_sha256'] != object_sha(pool['isolation']):
        raise ValueError('Isolation snapshot hash differs')
    validate_generator(pool['generator'])
    validate_sampling(pool['sampling'])
    count = pool['candidates_per_source']
    if type(count) is not int or not 2 <= count <= 32:
        raise ValueError('Invalid candidates_per_source')
    if not isinstance(pool['candidates'], list) or not pool['candidates']:
        raise ValueError('Empty candidate pool')
    grouped, hashes = defaultdict(list), set()
    for item in pool['candidates']:
        _keys(item, ('source', 'index', 'generator', 'generation', 'diagnostics', 'candidate_sha256'), 'candidate')
        digest = item['candidate_sha256']
        if digest in hashes or digest != object_sha({k: v for k, v in item.items() if k != 'candidate_sha256'}):
            raise ValueError('Duplicate or changed candidate hash')
        source = item['source']
        _keys(source, ('id', 'split', 'audio_sha256', 'audio_path', 'audio_window_seconds') + GROUPS,
              'candidate source')
        expected = members.get(source['id'])
        if not expected or any(source[key] != expected[key] for key in expected):
            raise ValueError('Candidate source differs from isolation manifest')
        if source['split'] == 'test' or source['audio_window_seconds'] != [0., 5.0625]:
            raise ValueError('Test search or physical audio crop forbidden')
        if not Path(source['audio_path']).is_absolute() or file_sha(source['audio_path']) != source['audio_sha256']:
            raise ValueError('Review audio path/bytes differ from the candidate identity')
        if item['generator'] != pool['generator']:
            raise ValueError('Candidate generator identity differs')
        gen, sampling = item['generation'], pool['sampling']
        expected_decoding = dict(do_sample=True, num_beams=1, repetition_penalty=1.0,
                                 no_repeat_ngram_size=0, encoder_repetition_penalty=1.0,
                                 **{key: sampling[key] for key in ('temperature', 'top_p', 'top_k')})
        if gen.get('decoding') != expected_decoding:
            raise ValueError('Candidate was not generated with declared stochastic decoding')
        if item['diagnostics'] != _diagnostics(gen, sampling['max_new_tokens']):
            raise ValueError('Candidate diagnostics changed')
        if type(item['index']) is not int:
            raise ValueError('Candidate index must be an integer')
        grouped[source['id']].append(item['index'])
        hashes.add(digest)
    if any(sorted(indices) != list(range(count)) for indices in grouped.values()):
        raise ValueError('Incomplete candidate group')
    return {item['candidate_sha256']: item for item in pool['candidates']}


def validate_policy(policy, generator):
    _keys(policy, ('schema', 'rubric', 'judge', 'thresholds'), 'review policy')
    if policy['schema'] != 'music_planner_review_policy_v1':
        raise ValueError('Unknown review policy schema')
    rubric = policy['rubric']
    _keys(rubric, ('id', 'version', 'text', 'sha256'), 'rubric')
    for key in ('id', 'version', 'text'):
        _text(rubric[key], f'rubric {key}')
    if hashlib.sha256(rubric['text'].encode('utf-8')).hexdigest() != rubric['sha256']:
        raise ValueError('Rubric text hash differs')
    judge = policy['judge']
    _keys(judge, ('kind', 'id', 'version', 'model_id', 'base_model_id', 'model_sha256'), 'judge')
    _text(judge['id'], 'judge id')
    _text(judge['version'], 'judge version')
    if judge['kind'] == 'human':
        if any(judge[key] is not None for key in ('model_id', 'base_model_id', 'model_sha256')):
            raise ValueError('A human reviewer cannot have a hidden model identity')
    elif judge['kind'] == 'audio_model':
        for key in ('model_id', 'base_model_id'):
            _text(judge[key], f'judge {key}')
            if judge[key].strip().lower() == generator[key].strip().lower():
                raise ValueError('Same-model self-scores are not independent music reviews')
        _sha(judge['model_sha256'], 'judge model identity')
        if judge['model_sha256'] == generator['model_sha256']:
            raise ValueError('Generator and judge model artifacts are identical')
    else:
        raise ValueError('Review requires a human or independent audio_model')
    thresholds = policy['thresholds']
    _keys(thresholds, ('min_dimension_score', 'min_confidence', 'preference_margin'), 'review thresholds')
    _number(thresholds['min_dimension_score'], 0, 5, 'minimum dimension score')
    _number(thresholds['min_confidence'], 0, 1, 'minimum confidence')
    _number(thresholds['preference_margin'], .000001, 5, 'preference margin')


def review_request(pool, policy):
    candidates = validate_pool(pool)
    validate_policy(policy, pool['generator'])
    request = dict(schema='music_planner_review_request_v1', run_sha256=pool['run_sha256'],
                   isolation_sha256=pool['isolation_sha256'], policy=policy,
                   policy_sha256=object_sha(policy), scope='audio_and_text_plan_only',
                   dimensions=list(DIMENSIONS), score_range=[0, 5],
                   selection_status='pending_independent_review',
                   instructions=('Listen to the original audio and assess the proposed plan, not an observed video. '
                                 'Scores must not be based on format, text length, or rare action counts. '
                                 'Mark uncertain when evidence is insufficient. This cannot certify physical execution '
                                 'or frame-accurate synchronization. Do not use any video/reference action labels.'),
                   candidates=[dict(candidate_sha256=digest, source_id=item['source']['id'],
                                    audio_sha256=item['source']['audio_sha256'],
                                    audio_path=item['source']['audio_path'], split=item['source']['split'],
                                    raw_text=item['generation']['raw_text'],
                                    format_eligible=item['diagnostics']['format_eligible'])
                               for digest, item in sorted(candidates.items())])
    request['request_sha256'] = object_sha(request)
    return request


def validate_reviews(pool, policy, reviews):
    request = review_request(pool, policy)
    _keys(reviews, ('schema', 'request_sha256', 'judge', 'rubric', 'reviews'), 'reviews')
    if reviews['schema'] != 'music_planner_reviews_v1' or reviews['request_sha256'] != request['request_sha256']:
        raise ValueError('Review request/run/candidate/policy binding differs or is stale')
    if reviews['judge'] != policy['judge'] or reviews['rubric'] != policy['rubric']:
        raise ValueError('Review judge/rubric identity or version differs')
    if not isinstance(reviews['reviews'], list):
        raise ValueError('reviews must be a list')
    candidates, imported = validate_pool(pool), {}
    for review in reviews['reviews']:
        _keys(review, ('candidate_sha256', 'source_id', 'audio_sha256', 'original_audio_reviewed',
                       'verdict', 'scores', 'confidence', 'reason', 'allow_preference_use'), 'candidate review')
        digest = review['candidate_sha256']
        if digest not in candidates or digest in imported:
            raise ValueError('Unknown or duplicate reviewed candidate')
        source = candidates[digest]['source']
        if review['source_id'] != source['id'] or review['audio_sha256'] != source['audio_sha256']:
            raise ValueError('Review belongs to another source or audio')
        if review['original_audio_reviewed'] is not True:
            raise ValueError('Text-only scoring is not an independent music review')
        if review['verdict'] not in ('approve', 'reject', 'uncertain'):
            raise ValueError('Invalid review verdict')
        _keys(review['scores'], DIMENSIONS, 'review scores')
        for key in DIMENSIONS:
            _number(review['scores'][key], 0, 5, key)
        _number(review['confidence'], 0, 1, 'review confidence')
        _text(review['reason'], 'review evidence/reason')
        if type(review['allow_preference_use']) is not bool:
            raise ValueError('Preference use must be explicitly true/false')
        imported[digest] = review
    return request, imported


def export_reviewed(pool, policy, reviews):
    request, imported = validate_reviews(pool, policy, reviews)
    grouped, pseudo, preferences, decisions = defaultdict(list), [], [], []
    thresholds = policy['thresholds']
    isolation_verified = all(proof['status'] == 'verified'
                             for proof in pool['isolation']['verification'].values())
    for item in pool['candidates']:
        grouped[item['source']['id']].append(item)
    for sid, items in sorted(grouped.items()):
        source = items[0]['source']
        decision = dict(source_id=sid, split=source['split'], status='pending_independent_review',
                        selected_candidate_sha256=None)
        if any(item['candidate_sha256'] not in imported for item in items):
            decisions.append(decision)
            continue
        if source['split'] != 'train':
            decision['status'] = 'dev_review_only_never_training_export'
            decisions.append(decision)
            continue
        if not isolation_verified:
            decision['status'] = 'blocked_unverified_isolation_no_training_export'
            decisions.append(decision)
            continue
        def eligible(item):
            r = imported[item['candidate_sha256']]
            return (item['diagnostics']['format_eligible'] and r['verdict'] == 'approve'
                    and r['confidence'] >= thresholds['min_confidence']
                    and min(r['scores'].values()) >= thresholds['min_dimension_score'])
        def score(item):
            return sum(imported[item['candidate_sha256']]['scores'].values()) / len(DIMENSIONS)
        approved = sorted((item for item in items if eligible(item)),
                          key=lambda item: (-score(item), item['candidate_sha256']))
        if not approved:
            decision['status'] = 'no_explicitly_approved_eligible_candidate'
            decisions.append(decision)
            continue
        top = approved[0]
        if len(approved) > 1 and abs(score(top) - score(approved[1])) < 1e-12:
            decision['status'] = 'tie_no_unique_selection'
            decisions.append(decision)
            continue
        digest = top['candidate_sha256']
        decision.update(status='review_approved_staging_only', selected_candidate_sha256=digest)
        parsed = parse_windows(top['generation']['raw_text'])
        provenance = dict(pool_source='generated_plan', observed_video_ground_truth=False,
                          candidate_sha256=digest, run_sha256=pool['run_sha256'],
                          request_sha256=request['request_sha256'], reviews_sha256=object_sha(reviews),
                          isolation_sha256=pool['isolation_sha256'], policy_sha256=object_sha(policy),
                          judge=policy['judge'], rubric=policy['rubric'],
                          generation=top['generator'], review=imported[digest])
        pseudo.append(dict(schema='music_planner_pseudo_sft_v1', **source,
                           target_text=top['generation']['raw_text'],
                           target_windows=[dict(start=w['start'], end=w['end'], caption=w['text'])
                                           for w in parsed['windows']],
                           review_approved=True, provenance=provenance,
                           fair_baseline_loader_compatible=False))
        for other in sorted(items, key=lambda item: item['candidate_sha256']):
            rejected = imported[other['candidate_sha256']]
            chosen = imported[digest]
            if (other is top or not other['diagnostics']['format_eligible']
                    or other['generation']['raw_text'] == top['generation']['raw_text']
                    or rejected['verdict'] == 'uncertain'
                    or not chosen['allow_preference_use'] or not rejected['allow_preference_use']
                    or rejected['confidence'] < thresholds['min_confidence']
                    or score(top) - score(other) < thresholds['preference_margin']):
                continue
            preferences.append(dict(schema='music_planner_preference_v1', **source,
                                    chosen=top['generation']['raw_text'], rejected=other['generation']['raw_text'],
                                    chosen_candidate_sha256=digest,
                                    rejected_candidate_sha256=other['candidate_sha256'],
                                    preference_basis='independent_audio_text_review_score_margin',
                                    score_margin=score(top) - score(other),
                                    chosen_review=chosen, rejected_review=rejected, provenance=provenance))
        decisions.append(decision)
    result = dict(schema='music_planner_reviewed_staging_v1', run_sha256=pool['run_sha256'],
                  request_sha256=request['request_sha256'], reviews_sha256=object_sha(reviews),
                  isolation_sha256=pool['isolation_sha256'], policy_sha256=object_sha(policy),
                  pseudo_sft=pseudo, preference_pairs=preferences, decisions=decisions,
                  isolation_training_eligible=isolation_verified,
                  automatic_weight_update_implemented=False, test_used=False)
    result['staging_sha256'] = object_sha(result)
    return result


def validate_staging(pool, policy, reviews, staging):
    # Recompute the entire deterministic export: verifies every selection, group,
    # provenance field, approval, score margin, text and source split, not just rows.
    if staging != export_reviewed(pool, policy, reviews):
        raise ValueError('Staging differs from the deterministic reviewed train-only export')
    return dict(valid=True, pseudo_sft_rows=len(staging['pseudo_sft']),
                preference_pairs=len(staging['preference_pairs']),
                staging_sha256=staging['staging_sha256'], test_used=False)


def _local_artifacts(path, recursive=False):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError('Require existing local model/checkpoint artifacts')
    files = sorted(p for p in (path.rglob('*') if recursive else path.iterdir()) if p.is_file())
    if not files:
        raise ValueError('No local model artifacts')
    return [dict(name=str(p.relative_to(path)).replace('\\', '/'), sha256=file_sha(p)) for p in files]


def prepare_fair_inputs(data, split='train', limit=8):
    """Copy existing identities; unknown song/near-duplicate groups stay null."""
    if split not in ('train', 'dev') or type(limit) is not int or limit <= 0:
        raise ValueError('Prepare train/dev with a positive limit; test is final-only')
    members, rows, references = [], [], {}
    for filename, assigned in (('train', 'train'), ('val', 'dev'), ('test', 'test')):
        path = Path(data).resolve() / (filename + '.jsonl')
        references[filename] = file_sha(path)
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('split') != filename:
                raise ValueError('Original fair split file contains another split')
            for key in ('id', 'source_group_id', 'audio_path', 'audio_sha256'):
                _text(row.get(key), f'original row {key}')
            member = dict(id=row['id'], split=assigned, source_group_id=row['source_group_id'],
                          song_group_id=row.get('song_group_id'),
                          near_duplicate_group_id=row.get('near_duplicate_group_id'),
                          audio_sha256=row['audio_sha256'])
            members.append(member)
            if assigned == split:
                audio = Path(row['audio_path']).expanduser()
                if not audio.is_absolute():
                    audio = path.parent / audio
                rows.append(dict(member, audio_path=str(audio.resolve()),
                                 audio_window_seconds=row.get('audio_window_seconds')))
    manifest = dict(schema='music_planner_isolation_v1', version='fair-input-inventory-v1',
                    verification={
                        'source_group_id': dict(status='verified', evidence_sha256=object_sha(references)),
                        'song_group_id': dict(status='unknown', evidence_sha256=None),
                        'near_duplicate_group_id': dict(status='unknown', evidence_sha256=None)},
                    members=sorted(members, key=lambda member: member['id']))
    validate_isolation(manifest)
    rows = sorted(rows, key=lambda row: row['id'])[:limit]
    validate_selected_rows(rows, manifest)
    return rows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Inventory fair train/val/test; unknown music groups remain unknown')
    prepare.add_argument('--data', type=Path, required=True)
    prepare.add_argument('--split', choices=('train', 'dev'), default='train')
    prepare.add_argument('--limit', type=int, default=8)
    prepare.add_argument('--out-dir', type=Path, required=True)
    generate = sub.add_parser('generate', help='Stochastic native decoding; no automatic winner')
    generate.add_argument('--rows', type=Path, required=True, help='JSON list of selected train/dev audio rows')
    generate.add_argument('--isolation', type=Path, required=True)
    generate.add_argument('--family', choices=FAMILIES, required=True)
    generate.add_argument('--model', type=Path, required=True)
    generate.add_argument('--checkpoint', type=Path)
    generate.add_argument('--model-id', required=True)
    generate.add_argument('--base-model-id', required=True)
    generate.add_argument('--model-version', required=True)
    generate.add_argument('--count', type=int, default=4)
    generate.add_argument('--seed', type=int, default=42)
    generate.add_argument('--temperature', type=float, default=.9)
    generate.add_argument('--top-p', type=float, default=.95)
    generate.add_argument('--top-k', type=int, default=50)
    generate.add_argument('--max-new-tokens', type=int, default=2048)
    generate.add_argument('--max-length', type=int, default=8192)
    generate.add_argument('--device', default='cpu', help='Explicit local CPU/CUDA; no scheduler or allocation')
    generate.add_argument('--out', type=Path, required=True)
    request = sub.add_parser('review-request', help='Prepare original-audio external review requests')
    export = sub.add_parser('export', help='Stage explicitly reviewed train-only pseudo SFT/preference pairs')
    check = sub.add_parser('check', help='Validate all immutable bindings and optionally exact staged export')
    for command in (request, export, check):
        command.add_argument('--pool', type=Path, required=True)
        command.add_argument('--policy', type=Path, required=True)
    request.add_argument('--out', type=Path, required=True)
    export.add_argument('--reviews', type=Path, required=True)
    export.add_argument('--out', type=Path, required=True)
    check.add_argument('--reviews', type=Path)
    check.add_argument('--staging', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        rows, isolation = prepare_fair_inputs(args.data, args.split, args.limit)
        args.out_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.out_dir / 'selected_rows.json', rows)
        write_json(args.out_dir / 'isolation.json', isolation)
    elif args.command == 'generate':
        rows, isolation = read_json(args.rows), read_json(args.isolation)
        validate_selected_rows(rows, isolation)
        validate_sampling(dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                               max_new_tokens=args.max_new_tokens))
        if args.out.exists():
            raise ValueError('Output already exists; refusing to load a model and overwrite evidence')
        # Hash actual local weights/config/processor bytes; no size-only identity.
        artifacts = dict(base=_local_artifacts(args.model))
        if args.checkpoint:
            export_path = args.checkpoint / 'adapter'
            artifacts['checkpoint'] = _local_artifacts(export_path if export_path.is_dir()
                                                       else args.checkpoint, recursive=True)
        generator = dict(family=args.family, model_id=args.model_id, base_model_id=args.base_model_id,
                         version=args.model_version, model_sha256=object_sha(artifacts))
        from scripts_omnidance.exp.ms.music_planner_adapters import load_planner_adapter
        adapter = load_planner_adapter(args.family, args.model, checkpoint=args.checkpoint,
                                       device=args.device, max_length=args.max_length)
        sampling = dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        max_new_tokens=args.max_new_tokens)
        result = generate_candidates(rows, isolation, generator, count=args.count, seed=args.seed,
                                     sampling=sampling,
                                     sampler=lambda row, **kw: sample_native(adapter, row, **kw))
        write_json(args.out, result)
    else:
        pool, policy = read_json(args.pool), read_json(args.policy)
        request = review_request(pool, policy)
        if args.command == 'review-request':
            write_json(args.out, request)
        elif args.command == 'export':
            reviews = read_json(args.reviews)
            result = export_reviewed(pool, policy, reviews)
            validate_staging(pool, policy, reviews, result)
            write_json(args.out, result)
        else:
            result = dict(valid=True, request_sha256=request['request_sha256'],
                          selection_status='pending_independent_review')
            if args.staging and not args.reviews:
                raise ValueError('--staging validation requires --reviews')
            if args.reviews:
                reviews = read_json(args.reviews)
                _, imported = validate_reviews(pool, policy, reviews)
                result['reviews_valid'] = True
                result['reviewed_candidates'] = len(imported)
                result['candidate_count'] = len(pool['candidates'])
                result['selection_status'] = 'reviews_imported_selection_requires_export'
                if args.staging:
                    result = validate_staging(pool, policy, reviews, read_json(args.staging))
            print(json.dumps(result, sort_keys=True, ensure_ascii=False))


if __name__ == '__main__':
    main()
