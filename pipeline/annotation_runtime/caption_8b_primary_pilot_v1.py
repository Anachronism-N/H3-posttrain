"""CPU preparation and caller-owned 8B annotation trial; no model/resource imports."""
import argparse
import ast
import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time
from types import SimpleNamespace

RT = Path(__file__).resolve().parent
BAR = RT.parent
CASES = ('VGfurWgIdrVQNyhI', 'HVsstNGHcxHNsWkM', 'VrydLuJOvAnzhiQt')
MODEL = '/opt/models/Qwen3-VL-8B-Instruct-review-extension-v2'
MODEL_SHA = '220d9152229696caec4decebb89a160afc8ac9b6ba91ab1b9d790343a82d733a'
SOURCE_MODEL_SHA = '26722d150f791da5ff1befaf753efd651a105b17849adb74ea07cfe0ba01779c'
STAGE_SHA = '51b9509cb19df000c26fdc7afddafc233a1d2c3a865952aa4c144266a046985d'
FLAGS = dict(semantic_approval=False, human_review_completed=False,
             quality_approved=False, training_eligible=False, production_candidate_counted=False)
SETTINGS = dict(seed=0, temperature=0.0, max_pixels=200704, max_model_len=16384,
                max_num_seqs=8, max_tokens=1800, disable_any_whitespace=True)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def strict(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            need(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def checkpoint(identity):
    weights = sorted(row['sha256'] for name, row in identity['files'].items()
                     if name.endswith(('.safetensors', '.bin', '.pt', '.gguf')))
    need(weights, 'checkpoint must have real weight-content SHA inventory')
    return digest({'weight_file_content_sha256': weights})


def identity_digest(identity):
    value = {'root': identity['root'], 'files': {name: {'path': row['path'], 'sha256': row['sha256']}
             for name, row in identity['files'].items()}}
    return hashlib.sha256((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()).hexdigest()


class Reader:
    """Each bound input is read/hashed once in the prepare closure, never per batch."""
    def __init__(self):
        self.files = {}

    def read(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        if path not in self.files:
            before = path.stat()
            raw = path.read_bytes()
            need(path.stat() == before, 'input changed while reading')
            self.files[path] = (raw, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        raw, ref = self.files[path]
        need(expected is None or ref['sha256'] == expected, 'input SHA changed: ' + str(path))
        return raw, ref

    def json(self, path, expected=None):
        raw, ref = self.read(path, expected)
        return strict(raw), ref

    def checked(self, ref):
        return self.json(ref['path'], ref['sha256'])[0]


def original_filter(source):
    namespace = dict(copy=copy, re=re, json=json, math=math)
    refs = []
    inventories = (
        ('ms_caption_rewrite.py', {'window_bounds', 'neutral_fact_text'}),
        ('ms_visual_observe.py', {'strict_json', 'parse_response'}),
        ('ms_bar_annotation.py', {'PARTS', 'OUTPUT_SCHEMA', 'need', 'check_indices',
          'claimed_part_groups', 'validate_observation', 'excluded_semantic_claim',
          'evidence_only_note', 'filter_supported_observation', 'facts_from_observations'}))
    reader = Reader()
    for filename, names in inventories:
        raw, ref = reader.read(Path(source) / filename)
        nodes = [n for n in ast.parse(raw).body if
                 (isinstance(n, ast.FunctionDef) and n.name in names) or
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
        found = {n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in nodes}
        need(found == names, 'original pure-function inventory changed')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), ref['path'], 'exec'), namespace)
        refs.append(ref)
        if filename == 'ms_caption_rewrite.py':
            namespace['core'] = SimpleNamespace(window_bounds=namespace['window_bounds'],
                                               neutral_fact_text=namespace['neutral_fact_text'])
    namespace['FLAGS'] = FLAGS
    return SimpleNamespace(**namespace, code_refs=refs)


def write_once(path, value, raw=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if raw else encoded(value)
    with path.open('xb') as stream:
        stream.write(data)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest()}


def prepare(output, *, bar=BAR, stage_path=None, expected_stage_sha=STAGE_SHA, source=None):
    bar, output = Path(bar).resolve(), Path(output).resolve()
    need(not output.exists(), 'new independent output root required; no overwrite/resume')
    need(not output.is_relative_to(bar / 'curated222939_collaborative_v3'), 'production root forbidden')
    read = Reader()
    stage, stage_ref = read.json(stage_path or bar / 'runtime/reviewer8b_stage_N7_collab3_v1.json', expected_stage_sha)
    model = stage['model_identity']
    need(stage['complete'] is True and stage['full_copy_SHA_verified'] is True, 'complete 8B stage required')
    need(model['root'] == MODEL and model['sha256'] == MODEL_SHA and identity_digest(model) == MODEL_SHA,
         'declared 8B model identity changed')
    need(len(model['files']) == 15 and sum(n.endswith('.safetensors') for n in model['files']) == 4,
         'full 15-file/four-weight checkpoint required')
    core = original_filter(source or bar / 'runtime/source_curated_consumer_v1')
    records, cases = [], []
    for shard, sid in enumerate(CASES):
        consumer = bar / 'curated222939_collaborative_v3/consumers' / ('shard-%04d' % shard)
        root = consumer / 'sources/train' / sid
        terminal, terminal_ref = read.json(root / 'consumer.terminal.json')
        need(terminal['id'] == sid and terminal['split'] == 'train' and terminal['joint_complete'] is True,
             'exact source terminal required')
        joint = read.checked(terminal['joint_candidate_ref'])
        need(joint['id'] == sid and joint['annotator_model_sha256'] == SOURCE_MODEL_SHA,
             'original 32B source binding changed')
        visual_ref = joint['visual_result_ref']
        need(Path(visual_ref['path']).resolve().is_relative_to(consumer / 'visual'), 'cross-shard visual ref')
        visual = read.checked(visual_ref)
        need(visual['id'] == sid, 'cross-source visual ref')
        old_identity = read.checked(visual['model_identity_ref'])
        need(old_identity['sha256'] == SOURCE_MODEL_SHA and identity_digest(old_identity) == SOURCE_MODEL_SHA,
             'source model identity changed')
        vroot = Path(visual['raw_rgb_manifest']['path']).parent.parent
        prep_receipt, _ = read.json(vroot / 'prepared.receipt.json')
        prep, prep_ref = read.json(vroot / 'prepared.json', prep_receipt['prepared_sha256'])
        rgb = read.checked(visual['raw_rgb_manifest'])
        read.read(joint['source_identity']['files']['video_path']['path'], joint['source_identity']['files']['video_path']['sha256'])
        windows = visual['window_observations']
        need([o['window'] for o in windows] == [w['window'] for w in prep['windows']], 'original ordered windows changed')
        cases.append(dict(id=sid, consumer_terminal_ref=terminal_ref, joint_candidate_ref=terminal['joint_candidate_ref'],
                          visual_result_ref=visual_ref, source_window_input_ref=prep_ref,
                          source_video_ref=joint['source_identity']['files']['video_path'],
                          teacher_model_identity=old_identity, teacher_model_ref=visual['model_identity_ref'],
                          original_annotations_are_comparison_not_ground_truth=True))
        for observation in windows:
            window, attempt = observation['window'], Path(observation['attempt_path']).resolve()
            need(attempt.is_relative_to(vroot / 'windows' / window['window_id']), 'cross-window attempt path')
            manifest, manifest_ref = read.json(attempt.parent / 'manifest.json')
            request, request_ref = read.json(attempt / 'request.json')
            receipt, _ = read.json(attempt / 'receipt.json')
            need(receipt['files_sha256']['request.json'] == request_ref['sha256'], 'original request receipt SHA changed')
            need(manifest['window'] == window and manifest['binding'] == visual['binding'], 'window input binding changed')
            need(request['sampling'] == {'temperature': 0.0, 'seed': 0, 'max_tokens': 1800}, 'original sampling budget changed')
            need(request['model'] == old_identity['root'], 'source request model differs')
            parts = manifest.get('hand_roi_evidence') or {}
            native = []
            images = []
            for frame in manifest['frames']:
                need(frame in rgb['frames'], 'window frame is not bound original RGB')
                i = frame['index']
                need(math.isclose(frame['time'], prep['probe']['times'][i], abs_tol=1e-9) and
                     math.isclose(frame['source_timestamp'], prep['probe']['source_timestamps'][i], abs_tol=1e-9), 'PTS mismatch')
                data, ref = read.read(vroot / 'rgb' / frame['file'], frame['sha256'])
                images.append(data)
                crops = []
                for part in ('left_hand', 'right_hand'):
                    for crop in parts.get(part, {}).get('crops', []):
                        if crop['frame_index'] == i:
                            data, ref = read.read(crop['path'], crop['sha256'])
                            images.append(data)
                            crops.append(crop)
                native.append(dict(index=i, time=frame['time'], source_timestamp=frame['source_timestamp'],
                                   same_source_native_ROI=crops, body_jpeg_ref={'path': str(vroot / 'rgb' / frame['file']), 'sha256': frame['sha256']}))
            urls = [c['image_url']['url'] for c in request['messages'][1]['content'] if c['type'] == 'image_url']
            need([base64.b64decode(u.split(',', 1)[1], validate=True) for u in urls] == images,
                 'prompt body/ROI bytes or ordered images changed')
            new_request = copy.deepcopy(request)
            new_request['model'] = MODEL
            records.append(dict(id=sid, window=window, original_request_ref=request_ref,
                original_window_manifest_ref=manifest_ref, manifest=manifest, request=new_request,
                source_request_without_model_sha256=digest({k:v for k,v in request.items() if k != 'model'}),
                forbidden=prep['forbidden'], native_RGB_audit_correspondence=native,
                source_raw_notes_ref=read.read(attempt / 'response.raw.txt')[1]))
    need(len(records) == 8 and len(cases) == 3, 'exact three-case/eight-window scope required')
    code_ref = read.read(Path(__file__))[1]
    manifest = dict(schema='caption_8b_primary_pilot_v1', case_ids=list(CASES), cases=cases, requests=records,
        declared_checkpoint=checkpoint(model), model_identity=model, model_stage_ref=stage_ref,
        original_filter_code_refs=core.code_refs, pilot_code_ref=code_ref, settings=SETTINGS,
        request_delta=['model'], checkpoint_is_manifest_declaration_not_prompt_change=True,
        budgets=dict(new_8B_window_calls=8, independent_32B_reviews=0, source_style_reruns=0,
                     production_attempt_calls=0, production_repair_budget=0), **FLAGS)
    manifest_ref = write_once(output / 'manifest.json', manifest)
    return SimpleNamespace(manifest=manifest, manifest_ref=manifest_ref, core=core)


def load_prepared(manifest_path, expected_sha):
    read = Reader()
    manifest, ref = read.json(manifest_path, expected_sha)
    source = Path(manifest['original_filter_code_refs'][0]['path']).parent
    core = original_filter(source)
    need(core.code_refs == manifest['original_filter_code_refs'], 'original filter source changed')
    read.read(manifest['pilot_code_ref']['path'], manifest['pilot_code_ref']['sha256'])
    return SimpleNamespace(manifest=manifest, manifest_ref=ref, core=core)


def benchmark(prepared, output, *, runtime=None, admission=None, sampling=None, batch_size=8,
              memory=None):
    """Caller owns real safe admission, model load/warmup, sampling and GPU telemetry.

    admission(manifest) is a context manager retaining the parent's safe lease.
    It must yield admitted/node/gpus/tp/test_only. runtime() returns the existing
    LLM with _ms_bar_model_sha256 plus _ms_8b_pilot_settings attested at load.
    sampling(request) must preserve its schema/seed/budget and disable whitespace.
    No retry exists: claiming an independent run consumes its eight-call budget.
    """
    need(admission is not None and runtime is not None and sampling is not None, 'default deny: actual caller callbacks required')
    need(batch_size in (1, 2, 8), 'batch size must preserve eight unique requests')
    output = Path(output).resolve()
    need(not output.exists(), 'existing/conflicted/failed run is terminal; do not redispatch')
    m = prepared.manifest
    need(digest(m) == prepared.manifest_ref['sha256'], 'prepared manifest changed in memory')
    need(output.is_relative_to(Path(prepared.manifest_ref['path']).parent), 'benchmark must stay inside independent prepare root')
    with admission(m) as approved:
        need(approved.get('admitted') is True, 'runtime admission denied')
        need(approved.get('node') == 'N6' and approved.get('gpus') == [3] and approved.get('tp') == 1,
             'only caller-admitted N6 GPU3 TP1 trial permitted; never N7 reviewer GPUs')
        claim = dict(manifest_ref=prepared.manifest_ref, reserved_unique_window_calls=8,
                     admission=approved, failure_is_terminal_no_redispatch=True)
        write_once(output / 'run.claim.json', claim)
        llm = runtime()
        need(getattr(llm, '_ms_bar_model_sha256', None) == MODEL_SHA, '8B actual-runtime marker mismatch')
        need(getattr(llm, '_ms_8b_pilot_settings', None) == SETTINGS, 'actual runtime settings differ from source protocol')
        before_memory = memory() if memory else None
        observations, times, prompt_tokens, completion_tokens = [], [], 0, 0
        records = m['requests']
        for offset in range(0, len(records), batch_size):
            group = records[offset:offset + batch_size]
            requests = [copy.deepcopy(r['request']) for r in group]
            samples = [sampling(r) for r in requests]
            start = time.perf_counter()
            failure = None
            try:
                outputs = llm.chat([r['messages'] for r in requests], samples, use_tqdm=False)
                need(len(outputs) == len(group) and all(len(o.outputs) == 1 for o in outputs), 'one completion per source request required')
            except Exception as error:
                outputs, failure = [], str(error)
            times.append(time.perf_counter() - start)
            for index, record in enumerate(group):
                root = output / 'inference_results' / record['id'] / record['window']['window_id']
                write_once(root / 'request.json', record['request'])
                result = dict(id=record['id'], window=record['window'], status='failed', error=failure,
                              model_identity_sha256=MODEL_SHA, **FLAGS)
                if failure is None:
                    output_item, completion = outputs[index], outputs[index].outputs[0]
                    raw = completion.text.encode()
                    result['raw_notes_ref'] = write_once(root / 'response.raw.txt', raw, raw=True)
                    p, c = len(output_item.prompt_token_ids or []), len(completion.token_ids or [])
                    prompt_tokens += p
                    completion_tokens += c
                    generation = dict(prompt_tokens=p, completion_tokens=c, finish_reason=completion.finish_reason,
                                      max_tokens=1800, model_call_performed=True, model_identity_sha256=MODEL_SHA)
                    result['generation_ref'] = write_once(root / 'generation.json', generation)
                    try:
                        need(completion.finish_reason == 'stop' and c < 1800, 'truncated/unfinished completion')
                        raw_value = prepared.core.parse_response(json.dumps(raw.decode()).encode())
                        checked = prepared.core.filter_supported_observation(raw_value, record['manifest'], record['forbidden'])
                        result.update(status='abstain' if checked['abstain'] else 'candidate', json=checked, error=None)
                    except (ValueError, KeyError, TypeError) as error:
                        result['error'] = str(error)
                result['observation_ref'] = write_once(root / 'observation.json', result)
                observations.append(result)
        wall = sum(times)
        candidates = []
        for sid in CASES:
            local = [o for o in observations if o['id'] == sid]
            facts = prepared.core.facts_from_observations([o['window'] for o in local], local)
            captions = [dict(window=o['window'], caption=' '.join(f['raw_text'] for f in facts
                         if f['window_id'] == o['window']['window_id']), used_fact_ids=[f['fact_id'] for f in facts
                         if f['window_id'] == o['window']['window_id']]) for o in local]
            candidate = dict(id=sid, facts=facts, captions=captions,
                full_window_coverage=all(o['status'] == 'candidate' and o.get('json', {}).get('events') for o in local),
                full_fact_text_preserved=True, raw_notes=[o.get('raw_notes_ref') for o in local], **FLAGS)
            candidate['candidate_ref'] = write_once(output / 'inference_results' / sid / 'candidate.json', candidate)
            candidates.append(candidate)
        receipt = dict(schema='caption_8b_primary_benchmark_v1', manifest_ref=prepared.manifest_ref,
            admission=approved, batch_size=batch_size, new_window_calls=len(observations), requests_count=8,
            completion_tokens=completion_tokens, prompt_tokens=prompt_tokens, modelchat_wall_seconds=wall,
            completion_tokens_per_modelchat_second=completion_tokens / wall if wall else None,
            memory_before=before_memory, memory_after=memory() if memory else None,
            observations=observations, candidates=candidates,
            measured_real_model=not approved.get('test_only', False), load_warmup_validation_review_excluded=True,
            production_pipeline_speedup_measured=False, independent_review_calls=0, **FLAGS)
        receipt['receipt_ref'] = write_once(output / 'benchmark.receipt.json', receipt)
        return receipt


def quality_audit_plan(prepared, receipt, *, reviewer_identity=None):
    """Return independent pixel audit tasks; original annotations are not GT.

    The reviewer caller must decode each video's union of original frame indices
    ONCE at native RGB dimensions, keeping PTS/body context plus same-frame ROIs.
    Full raw notes and EVERY admitted fact/ten part-support rows go to 32B; a
    matching label string does not pass. This function cannot approve quality.
    """
    if reviewer_identity is not None:
        need(identity_digest(reviewer_identity) == reviewer_identity['sha256'], 'reviewer identity attestation changed')
        need(checkpoint(reviewer_identity) != prepared.manifest['declared_checkpoint'], 'same 8B checkpoint is not an independent reviewer')
        need(reviewer_identity['sha256'] == SOURCE_MODEL_SHA, 'independent original 32B model required')
    tasks = []
    for case in prepared.manifest['cases']:
        sid = case['id']
        records = [r for r in prepared.manifest['requests'] if r['id'] == sid]
        candidate = next(c for c in receipt['candidates'] if c['id'] == sid)
        tasks.append(dict(id=sid, original_video_ref=case['source_video_ref'],
            original_windows=[r['window'] for r in records], candidate_ref=candidate['candidate_ref'],
            raw_notes=candidate['raw_notes'], all_fact_ids=[f['fact_id'] for f in candidate['facts']],
            native_RGB_frames=[r['native_RGB_audit_correspondence'] for r in records],
            decode_policy='one_native_RGB_union_decode_per_case_exact_global_indices_and_PTS_no_rescale',
            check=['main_combined_motion_and_repetition', 'every_fact_text_pixel_supported', 'every_part_own_two_frames',
                   'anatomical_sides', 'fine_gesture_unknown_abstention', 'no_ROI_extra_temporal_support',
                   'all_original_windows_covered', 'contradictions_and_lost_facts_vs_pixels'],
            original_32B_annotation_ref=case['visual_result_ref'], original_32B_is_comparison_not_GT=True,
            native_pixel_decode_and_review_pending=True, **FLAGS))
    return dict(schema='caption_8b_independent_pixel_audit_plan_v1', mode='32B_native_visual_or_web_human',
        reviewer_identity=reviewer_identity, tasks=tasks,
        budget=dict(independent_32B_review_window_calls=8, new_8B_annotation_calls=0),
        same_8B_reviewer_independence_prohibited=True, label_text_agreement_is_not_semantic_validation=True,
        review_result_must_bind_candidate_raw_notes_source_video_window_PTS_and_pixel_evidence=True, **FLAGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prepare_output')
    parser.add_argument('--stage-sha', required=True)
    args = parser.parse_args()
    result = prepare(args.prepare_output, expected_stage_sha=args.stage_sha)
    print(json.dumps(dict(manifest_ref=result.manifest_ref, cases=3, windows=8, GPU_model_loaded=False, **FLAGS)))
