"""Fixed original visual requests only; GPU runtime and resource ownership belong to caller."""
import argparse
import ast
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
BAR = RT.parent
DEFAULT_OUT = BAR / 'visual_batch_fixed3_benchmark_v1'
SOURCES = ('VGfurWgIdrVQNyhI', 'HVsstNGHcxHNsWkM', 'VrydLuJOvAnzhiQt')
MODEL_SHA = '26722d150f791da5ff1befaf753efd651a105b17849adb74ea07cfe0ba01779c'
MODEL_REF_SHA = 'c24c37d47c4d9d67b68b053c3c09db25de4396d02eba39b0308fb0d6f54c1f09'
SCHEMA = 'visual_batch_fixed3_benchmark_v1'
FLAGS = dict(production_counts=0, production_claims=0, semantic_approval=False,
             human_review_completed=False, formal_training=False, training_eligible=False)
EXPECTED_STATE = dict(model_identity_sha256=MODEL_SHA, source_reference_sha256=MODEL_REF_SHA,
    dtype='bfloat16', quantization=None, tensor_parallel_size=2, max_num_seqs=8,
    max_model_len=16384, max_pixels=200704, max_images_per_prompt=16,
    structured_outputs_backend='xgrammar', disable_any_whitespace=True,
    seed=0, model_load_completed=True)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('non-finite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def exclusive(path, raw):
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def append_journal(path, value):
    with Path(path).open('ab') as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


class Boundary:
    def __init__(self):
        self.refs = {}

    def file(self, path, expected=None):
        path = str(Path(path).resolve())
        if path not in self.refs:
            self.refs[path] = dict(path=path, sha256=sha_file(path))
        ref = self.refs[path]
        need(expected is None or ref['sha256'] == expected, 'original file SHA changed: ' + path)
        return ref

    def checked(self, ref):
        self.file(ref['path'], ref['sha256'])
        return strict(Path(ref['path']).read_bytes())

    def optional(self, path):
        path = Path(path).resolve()
        if path.exists():
            return self.file(path)
        self.refs[str(path)] = dict(path=str(path), sha256=None)
        return self.refs[str(path)]


def verify_boundary(refs):
    for ref in refs:
        path = Path(ref['path'])
        actual = sha_file(path) if path.exists() else None
        need(actual == ref['sha256'], 'read-only original boundary changed: ' + str(path))


def pure_bar(source=RT / 'source_curated_consumer_v1'):
    namespace = dict(copy=copy, re=re, math=math, need=need)
    files = [('ms_caption_rewrite.py', {'window_bounds'}),
             ('ms_bar_annotation.py', {'PARTS', 'SYSTEM', 'observation_schema', 'check_indices',
               'claimed_part_groups', 'validate_observation', 'excluded_semantic_claim',
               'evidence_only_note', 'filter_supported_observation'})]
    for filename, names in files:
        path = Path(source) / filename
        tree = ast.parse(path.read_bytes(), filename=str(path))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names or
                 isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
        found = {n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in nodes}
        need(found == names, 'frozen pure function inventory changed')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        if filename == 'ms_caption_rewrite.py':
            namespace['core'] = SimpleNamespace(window_bounds=namespace['window_bounds'])
    return SimpleNamespace(**namespace)


def schema_check(value, schema):
    kind = schema.get('type')
    types = {'object': lambda x: isinstance(x, dict), 'array': lambda x: isinstance(x, list),
             'string': lambda x: isinstance(x, str), 'integer': lambda x: type(x) is int,
             'boolean': lambda x: type(x) is bool}
    need(kind in types and types[kind](value), 'response JSON schema type differs')
    if 'enum' in schema:
        need(value in schema['enum'], 'response JSON schema enum differs')
    if 'const' in schema:
        need(value == schema['const'], 'response JSON schema const differs')
    if kind == 'object':
        properties = schema['properties']
        need(set(schema.get('required', [])) <= set(value), 'response JSON schema required keys missing')
        need(schema.get('additionalProperties', True) or set(value) <= set(properties), 'response extra JSON keys')
        for key, child in value.items():
            schema_check(child, properties[key])
    if kind == 'array':
        need(schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', float('inf')), 'response array bounds')
        prefix = schema.get('prefixItems', [])
        for index, child in enumerate(value):
            sub = prefix[index] if index < len(prefix) else schema.get('items')
            need(isinstance(sub, dict), 'response forbidden additional array items')
            schema_check(child, sub)
    if kind == 'string':
        need(schema.get('minLength', 0) <= len(value) <= schema.get('maxLength', float('inf')), 'response string bounds')


def isolated_out(out):
    out = Path(out).resolve()
    need(out.is_relative_to(BAR.resolve()) and out.parent == BAR.resolve() and
         out.name.startswith('visual_batch_fixed3_benchmark'), 'explicit independent BAR benchmark output required')
    return out


def prepare(out=DEFAULT_OUT, budget_paths=()):
    out = isolated_out(out)
    need(not out.exists(), 'benchmark output already exists; never reset budget or overwrite')
    boundary = Boundary()
    template_path = RT / 'curated_consumer_s0_v1_template.json'
    template = boundary.checked(boundary.file(template_path))
    for path, expected in template['source_sha256'].items():
        if Path(path).parent == RT / 'source_curated_consumer_v1' and Path(path).name in (
                'ms_bar_annotation.py', 'ms_caption_rewrite.py'):
            boundary.file(path, expected)
    model_ref = template['modelref']
    model = boundary.checked(model_ref)
    need(model_ref['sha256'] == MODEL_REF_SHA and model['model_identity']['sha256'] == MODEL_SHA, 'original 32B checkpoint identity differs')
    bar = pure_bar()
    cases, requests = [], []
    for shard, sid in enumerate(SOURCES):
        source_root = BAR / ('curated222939_collaborative_v3/consumers/shard-%04d/sources/train/' % shard) / sid
        terminal_ref = boundary.file(source_root / 'consumer.terminal.json')
        terminal = boundary.checked(terminal_ref)
        need(terminal['id'] == sid and terminal['status'] == 'joint_candidate' and terminal['joint_complete'],
             'fixed original source is not a complete candidate')
        joint = boundary.checked(terminal['joint_candidate_ref'])
        visual = boundary.checked(joint['visual_result_ref'])
        need(joint['id'] == visual['id'] == sid and visual['status'] == 'candidate', 'source candidate mapping differs')
        visual_root = Path(visual['window_observations'][0]['attempt_path']).parents[2]
        prepared_ref = boundary.file(visual_root / 'prepared.json')
        prepared = boundary.checked(prepared_ref)
        prepared_receipt = boundary.checked(boundary.file(visual_root / 'prepared.receipt.json'))
        need(prepared_receipt['prepared_sha256'] == prepared_ref['sha256'] and
             prepared_receipt['binding'] == prepared['binding'] == visual['binding'], 'prepared source binding differs')
        identity = boundary.checked(visual['model_identity_ref'])
        need(identity['sha256'] == MODEL_SHA and identity['source_reference_sha256'] == MODEL_REF_SHA,
             'original model identity/source reference differs')
        for ref in prepared['source_identity']['files'].values():
            if ref:
                boundary.file(ref['path'], ref['sha256'])
        for ref in prepared['source_refs']:
            boundary.file(ref['path'], ref['sha256'])
        if prepared['raw2d_mask'].get('available'):
            boundary.file(prepared['raw2d_mask']['path'], prepared['raw2d_mask']['sha256'])
        need(len(prepared['windows']) == len(visual['window_observations']), 'all original observation windows required')
        for manifest, observation in zip(prepared['windows'], visual['window_observations']):
            wid = manifest['window']['window_id']
            attempt = Path(observation['attempt_path'])
            need(observation['window'] == manifest['window'] and attempt.parent.name == wid,
                 'original window mapping differs')
            receipt = boundary.checked(boundary.file(attempt / 'receipt.json'))
            refs = {name: boundary.file(attempt / name, receipt['files_sha256'][name])
                    for name in ('request.json', 'binding.json', 'generation.json')}
            request = boundary.checked(refs['request.json'])
            binding = boundary.checked(refs['binding.json'])
            original_generation = boundary.checked(refs['generation.json'])
            need(binding['source_binding']['id'] == sid and binding['window'] == manifest['window'] and
                 all(binding[k] == v for k, v in prepared['binding'].items()), 'native binding differs')
            need(request['sampling'] == dict(temperature=0.0, seed=0, max_tokens=1800) and
                 request['model'] == identity['root'], 'original model/sampling/token budget differs')
            indices = [f['index'] for f in manifest['frames']]
            need(len(indices) == 8 and request['json_schema'] == bar.observation_schema(indices), 'original eight-frame schema differs')
            need(request['messages'][0] == dict(role='system', content=bar.SYSTEM), 'original system prompt changed')
            header = strict(request['messages'][1]['content'][0]['text'])
            denied = {p: sorted(set(indices) & set(v)) for p, v in prepared['forbidden'].items() if set(indices) & set(v)}
            need(header['allowed_source_frame_indices'] == indices and header['explicit_raw2d_forbidden_part_frames'] == denied,
                 'original hard-mask request differs')
            expected_images = []
            for frame in manifest['frames']:
                expected_images.append(boundary.file(visual_root / 'rgb' / frame['file'], frame['sha256'])['sha256'])
                for part in ('left_hand', 'right_hand'):
                    for crop in (manifest.get('hand_roi_evidence') or {}).get(part, {}).get('crops', []):
                        if crop['frame_index'] == frame['index']:
                            expected_images.append(boundary.file(crop['path'], crop['sha256'])['sha256'])
            images = [c['image_url']['url'] for c in request['messages'][1]['content'] if c['type'] == 'image_url']
            need(len(images) <= 16 and all(s.startswith('data:image/jpeg;base64,') for s in images), 'native image format/budget differs')
            need([sha_bytes(base64.b64decode(s.split(',', 1)[1], validate=True)) for s in images] == expected_images,
                 'request image bytes differ from original native evidence')
            raw_ref = boundary.file(attempt / 'response.raw.txt', receipt['files_sha256']['response.raw.txt'])
            raw_value = strict(Path(raw_ref['path']).read_bytes())
            schema_check(raw_value, request['json_schema'])
            baseline = bar.filter_supported_observation(raw_value, manifest, {p: set(v) for p, v in prepared['forbidden'].items()})
            need(baseline == observation['json'], 'original support-filter CPU replay differs from frozen candidate')
            requests.append(dict(key=sid + ':' + wid, source_id=sid, window_id=wid, refs=refs,
                original_response_ref=raw_ref, original_support_filter_CPU_replay_equal=True,
                manifest=manifest, forbidden=prepared['forbidden'], original_completion_tokens=original_generation['completion_tokens'],
                original_prompt_tokens=original_generation['prompt_tokens'], image_count=len(images)))
        cases.append(dict(source_id=sid, terminal_ref=terminal_ref, joint_ref=terminal['joint_candidate_ref'],
                          visual_ref=joint['visual_result_ref'], prepared_ref=prepared_ref,
                          window_count=len(prepared['windows'])))
    for path in budget_paths:
        boundary.optional(path)
    manifest = dict(schema=SCHEMA, scope='visual_observation_only_all_original_windows; no boundary/fulltext/style calls',
        cases=cases, requests=requests, request_count=len(requests), planned_model_request_count=2 * len(requests),
        total_completion_token_budget=2 * len(requests) * 1800, per_request_per_variant_planned_calls=1,
        variants=[dict(name='A', batch_size=2), dict(name='B', batch_size=8)], order='AB',
        order_reason='Only 8 existing windows: AB preserves B batch8 and one call/request/variant; ABBA would split B or repeat calls.',
        comparison='batch organization only on SAME resident TP2/max_num_seqs8; NOT original max_num_seqs2 cold baseline',
        repeated_original_requests_explicit=True, no_warmup_calls=True, expected_runtime_state=EXPECTED_STATE,
        model_ref=model_ref, original_boundary=list(boundary.refs.values()), original_retry_budget_paths=[str(Path(p).resolve()) for p in budget_paths],
        original_retry_budget_modified=False, output=str(out), **FLAGS)
    need([c['window_count'] for c in cases] == [2, 3, 3] and len(requests) == 8, 'fixed three-source window count changed')
    out.mkdir()
    exclusive(out / 'manifest.json', encoded(manifest))
    return manifest


class CallerOwnedRuntime:
    """Wrap an already-loaded native LLM; callbacks must inspect live managed ownership/state."""
    def __init__(self, llm, sampling_cls, actual_state, memory_snapshot):
        need(callable(actual_state) and callable(memory_snapshot), 'live state/memory callbacks required')
        self.llm, self.sampling_cls = llm, sampling_cls
        self.actual_state, self.memory_snapshot = actual_state, memory_snapshot

    def state(self):
        return self.actual_state(self.llm)

    def memory(self):
        return self.memory_snapshot(self.llm)

    def chat(self, items):
        from vllm.sampling_params import StructuredOutputsParams
        sampling = [self.sampling_cls(**item['request']['sampling'],
            structured_outputs=StructuredOutputsParams(json=copy.deepcopy(item['request']['json_schema']))) for item in items]
        messages = [item['request']['messages'] for item in items]
        start = time.perf_counter()
        try:
            outputs = self.llm.chat(messages, sampling, use_tqdm=False)
        finally:
            self.last_chat_wall_seconds = time.perf_counter() - start
        need(len(outputs) == len(items), 'missing/extra native output; no resubmit')
        result = []
        seen = set()
        for item, output in zip(items, outputs):
            need(len(output.outputs) == 1 and output.request_id not in seen, 'native output/request_id cardinality differs')
            seen.add(output.request_id)
            completion = output.outputs[0]
            result.append(dict(key=item['key'], source_id=item['source_id'], window_id=item['window_id'],
                native_request_id=output.request_id, raw=completion.text,
                prompt_tokens=len(output.prompt_token_ids or []), completion_tokens=len(completion.token_ids),
                finish_reason=completion.finish_reason, stop_reason=completion.stop_reason,
                last_token_is_eos=bool(completion.token_ids) and completion.token_ids[-1] == self.llm.get_tokenizer().eos_token_id))
        return result


def default_admission(manifest, runtime):
    raise PermissionError('caller-owned managed resource admission is required; benchmark never allocates GPUs')


def run_benchmark(manifest_path, runtime=None, admission=default_admission):
    manifest_path = Path(manifest_path).resolve()
    manifest_raw = manifest_path.read_bytes()
    manifest = strict(manifest_raw)
    out = isolated_out(manifest['output'])
    need(manifest_path == out / 'manifest.json' and manifest['schema'] == SCHEMA, 'independent manifest path/schema differs')
    need(runtime is not None and callable(admission), 'caller-owned runtime/admission required; no model loader exists')
    need(manifest['cases'] and tuple(c['source_id'] for c in manifest['cases']) == SOURCES and
         manifest['request_count'] == len(manifest['requests']) == 8 and
         manifest['planned_model_request_count'] == 16 and manifest['total_completion_token_budget'] == 28800 and
         manifest['variants'] == [dict(name='A', batch_size=2), dict(name='B', batch_size=8)] and
         manifest['expected_runtime_state'] == EXPECTED_STATE, 'immutable fixed scope/budget differs')
    need(not (out / 'started.json').exists(), 'benchmark already started; resume/retry never dispatches again')
    ownership = admission(manifest, runtime)
    need(isinstance(ownership, dict) and ownership.get('admitted') is True and ownership.get('caller_owned') is True,
         'managed ownership admission denied')
    state = runtime.state()
    need(all(k in state and state[k] == v for k, v in EXPECTED_STATE.items()), 'live native runtime state differs')
    verify_boundary(manifest['original_boundary'])
    bar = pure_bar()
    expected_keys = [sid + ':w%05d' % i for sid, count in zip(SOURCES, (2, 3, 3)) for i in range(count)]
    need([i['key'] for i in manifest['requests']] == expected_keys, 'duplicate/reordered/out-of-scope request; no dispatch')
    prepared_by_source = {c['source_id']: strict(Path(c['prepared_ref']['path']).read_bytes()) for c in manifest['cases']}
    items = []
    for item in manifest['requests']:
        request = strict(Path(item['refs']['request.json']['path']).read_bytes())
        binding = strict(Path(item['refs']['binding.json']['path']).read_bytes())
        prepared = prepared_by_source[item['source_id']]
        original_window = next(w for w in prepared['windows'] if w['window']['window_id'] == item['window_id'])
        need(item['manifest'] == original_window and item['forbidden'] == prepared['forbidden'] and
             binding['source_binding']['id'] == item['source_id'] and binding['window'] == original_window['window'],
             'locked original window/source/support mapping differs')
        need(request['sampling'] == dict(temperature=0.0, seed=0, max_tokens=1800) and
             request['json_schema'] == bar.observation_schema([f['index'] for f in original_window['frames']]),
             'locked sampling/schema budget changed')
        items.append({**item, 'request': request})
    exclusive(out / 'started.json', encoded(dict(manifest_sha256=sha_bytes(manifest_raw), ownership=ownership,
                                                runtime_state=state, budget=16, **FLAGS)))
    journal = out / 'calls.jsonl'
    report = dict(schema=SCHEMA, scope=manifest['scope'], status='running', variants={}, manifest_sha256=sha_bytes(manifest_raw),
                  actual_model_request_count=0, repeated_original_requests_explicit=True,
                  quality_is_separate_from_speed=True, original_retry_budget_modified=False, **FLAGS)
    try:
        for variant in manifest['variants']:
            name, size = variant['name'], variant['batch_size']
            records, wall, batches = [], 0.0, []
            memory_before = runtime.memory()
            for offset in range(0, len(items), size):
                group = items[offset:offset + size]
                append_journal(journal, dict(event='dispatch_reserved', variant=name, keys=[i['key'] for i in group]))
                report['actual_model_request_count'] += len(group)
                outputs = runtime.chat(group)
                elapsed = runtime.last_chat_wall_seconds
                need(type(elapsed) in (int, float) and elapsed > 0, 'actual native llm.chat wall timing required')
                wall += elapsed
                batches.append(dict(size=len(group), chat_wall_seconds=elapsed))
                need(len(outputs) == len(group), 'missing/extra output; reserved calls never resubmit')
                for item, response in zip(group, outputs):
                    need(all(response.get(k) == item[k] for k in ('key', 'source_id', 'window_id')), 'response source/window mapping differs')
                    need(response['finish_reason'] == 'stop' and type(response['completion_tokens']) is int and
                         0 < response['completion_tokens'] < 1800 and type(response['prompt_tokens']) is int and
                         response['prompt_tokens'] > 0, 'completion truncated/not normal EOS; never retry')
                    value = strict(response['raw'])
                    schema_check(value, item['request']['json_schema'])
                    checked = bar.filter_supported_observation(value, item['manifest'], {p: set(v) for p, v in item['forbidden'].items()})
                    record = dict(variant=name, key=item['key'], source_id=item['source_id'], window_id=item['window_id'],
                        request_sha256=item['refs']['request.json']['sha256'], binding_sha256=item['refs']['binding.json']['sha256'],
                        prompt_tokens=response['prompt_tokens'], completion_tokens=response['completion_tokens'],
                        finish_reason=response['finish_reason'], raw_response=response['raw'], filtered_observation=checked,
                        window_candidate=bool(checked['events']), admitted_fact_count=len(checked['events']), **FLAGS)
                    generation_root = out / 'generations' / name / item['source_id'] / item['window_id']
                    generation_root.mkdir(parents=True, exist_ok=False)
                    raw_path = generation_root / 'response.raw.txt'
                    filtered_path = generation_root / 'filtered.json'
                    generation_path = generation_root / 'generation.json'
                    exclusive(raw_path, response['raw'].encode())
                    exclusive(filtered_path, encoded(checked))
                    exclusive(generation_path, encoded(dict(variant=name, key=item['key'], source_id=item['source_id'],
                        window_id=item['window_id'], model_call_performed=True, model_identity_sha256=state['model_identity_sha256'],
                        prompt_tokens=response['prompt_tokens'], completion_tokens=response['completion_tokens'],
                        finish_reason=response['finish_reason'], max_tokens=1800, request_sha256=record['request_sha256'],
                        binding_sha256=record['binding_sha256'], raw_response_sha256=sha_file(raw_path), **FLAGS)))
                    record.update(raw_response_ref=dict(path=str(raw_path), sha256=sha_file(raw_path)),
                        generation_ref=dict(path=str(generation_path), sha256=sha_file(generation_path)),
                        filtered_ref=dict(path=str(filtered_path), sha256=sha_file(filtered_path)))
                    records.append(record)
                    append_journal(journal, dict(event='response_validated', **record))
            completion_tokens = sum(r['completion_tokens'] for r in records)
            prompt_tokens = sum(r['prompt_tokens'] for r in records)
            cases = []
            for sid in SOURCES:
                local = [r for r in records if r['source_id'] == sid]
                candidates = sum(r['window_candidate'] for r in local)
                cases.append(dict(source_id=sid, window_count=len(local), window_candidate_count=candidates,
                    all_window_full_fact_coverage=candidates == len(local), admitted_fact_count=sum(r['admitted_fact_count'] for r in local),
                    coverage_is_schema_support_candidate_not_human_truth=True))
            report['variants'][name] = dict(batch_size=size, batches=batches, global_chat_wall_seconds=wall,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, request_count=len(records),
                completion_tokens_per_second=completion_tokens / wall, total_tokens_per_second=(completion_tokens + prompt_tokens) / wall,
                request_throughput_per_second=len(records) / wall, memory_before=memory_before, memory_after=runtime.memory(),
                cases=cases, records=records, timing_excludes_model_load_warmup_preparation_artifact_and_CPU_validation=True)
        a, b = report['variants']['A'], report['variants']['B']
        report.update(status='completed', visual_request_throughput_ratio_B_over_A=b['request_throughput_per_second'] / a['request_throughput_per_second'],
                      text_equality_is_not_quality_gate=True, full_pipeline_speedup_not_measured=True)
    except Exception as error:
        report.update(status='failed_no_resubmit', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        try:
            verify_boundary(manifest['original_boundary'])
            need(manifest_path.read_bytes() == manifest_raw, 'locked benchmark manifest changed during run')
            report['original_boundary_unchanged'] = True
        except Exception as error:
            report.update(status='failed_original_boundary_changed', original_boundary_unchanged=False,
                          boundary_error=type(error).__name__ + ': ' + str(error))
            raise
        finally:
            exclusive(out / 'result.json', encoded(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare'])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--budget-path', action='append', default=[], help='Original readonly tasksource retry ledger; absence is also locked')
    args = parser.parse_args(argv)
    manifest = prepare(args.out, args.budget_path)
    print(json.dumps(dict(manifest=str(Path(manifest['output']) / 'manifest.json'),
                         cases=manifest['cases'], request_count=manifest['request_count'],
                         planned_model_request_count=manifest['planned_model_request_count'], **FLAGS), ensure_ascii=False))


if __name__ == '__main__':
    main()
