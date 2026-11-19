"""Opt-in direct admitted-fact recipe and exactly-three-source CPU evidence pilot."""
import argparse
import copy
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
BAR = RT.parent
REPO = Path('/apdcephfs_gy4/share_302533218/cedricnie/H3-posttrain')
SOURCE = RT / 'source_curated_consumer_v1'
RECIPE = 'caption_direct_native_admitted_facts_boundary_off_v1'
FROZEN = {
    'ms_bar_annotation.py': 'df29139651345ebd4e743657f3553f2e5165c8ab327bfd66ac84f225dc63005e',
    'ms_caption_rewrite.py': '6052fbb8f799c8a88833b8f564a935dad4562e16b8faf63f3f265143c0c43232',
    'curated_pipeline.py': 'd5cd607d4789e3b3cb361b2f32c35c4adc438cda8b9d2f5262dc143e61e952fc',
}
CASES = ((3, 0, 'UccGtwXYFHtdGcPV', 32787),
         (3, 3, 'BrduoqCtBJTMfLie', 32583),
         (4, 0, 'NsKZIogtsMiyLwxT', 62160))
FLAGS = dict(semantic_approval=False, human_review_completed=False, training_eligible=False,
             formal_approval=False, formal_training=False, training_entry_updated=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


@lru_cache(maxsize=1)
def native():
    for name, expected in FROZEN.items():
        if sha(SOURCE / name) != expected:
            raise ValueError('Frozen native validator/helper changed: ' + name)
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    import ms_bar_annotation as bar
    import curated_pipeline as pipeline
    if Path(bar.__file__).resolve() != SOURCE / 'ms_bar_annotation.py':
        raise ValueError('Native module imported from a different source package')
    return bar, pipeline


def direct_caption(windows, facts, boundaries):
    """Preserve every admitted native text and ID; boundaries never imply actions."""
    bar, _ = native()
    window_ids = [w['window_id'] for w in windows]
    ids = [f['fact_id'] for f in facts]
    if len(set(window_ids)) != len(window_ids) or len(set(ids)) != len(ids):
        raise ValueError('Duplicate source window/fact IDs')
    if any(f['window_id'] not in window_ids or not isinstance(f['text'], str)
           or not f['text'].strip() for f in facts):
        raise ValueError('Fact lacks an original window or nonempty native text')
    result = {'windows': [], 'issues': []}
    for window in windows:
        local = [f for f in facts if f['window_id'] == window['window_id']]
        result['windows'].append({'window_id': window['window_id'],
            'caption': ' '.join(f['text'] for f in local),
            'used_fact_ids': [f['fact_id'] for f in local], 'used_boundary_ids': []})
    bar.check_supported_rewrite(result, windows, facts, boundaries)
    return result


def direct_final(windows, facts, boundaries):
    try:
        body = direct_caption(windows, facts, boundaries)
    except ValueError as error:
        return {'status': 'failed', 'error': type(error).__name__ + ': ' + str(error),
                'format_valid': False, 'model_call_performed': False,
                'derivation': RECIPE, **FLAGS}
    return {'status': 'candidate' if facts else 'abstain', 'json': body,
            'format_valid': True, 'model_call_performed': False,
            'model_rewrite_performed': False, 'semantic_validation_completed': False,
            'derivation': RECIPE, **FLAGS}


@contextmanager
def direct_recipe(bar, args):
    """Future new-run hook: unchanged native RGB/support; no boundary/fulltext call."""
    expected_bar, _ = native()
    if bar is not expected_bar or args.boundary_review != 'off':
        raise ValueError('Direct recipe requires frozen native package and explicit boundary-review=off')
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Direct recipe requires a fresh isolated output; never resume an old generation recipe')
    for name in ('curated222939_collaborative_v3', 'curated222939_collaborative_v4',
                 'curated222939_production_v1', 'curated222939_repair_v1'):
        protected = BAR / name
        if out == protected or out.is_relative_to(protected):
            raise ValueError('Direct recipe cannot modify existing production/repair output')
    old_process, old_params = bar.process_source, bar.params
    source = inspect.getsource(old_process)
    begin = '        supplied = {"windows":'
    end = '        final = rewrite_with_reference_repair(item, runtime, args, facts)'
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError('Frozen native final rewrite hook anchors differ')
    start, stop = source.index(begin), source.index(end) + len(end)
    source = source[:start] + '        final = DIRECT_FINAL(windows, facts, boundaries)' + source[stop:]
    old_boundary = '        for spec in prepared["boundaries"]:'
    if source.count(old_boundary) != 1:
        raise ValueError('Frozen native boundary hook anchor differs')
    source = source.replace(old_boundary, '        for spec in []:')
    old_empty = '        final = {"status": "abstain", "json": baseline, "model_call_performed": False, **FLAGS}'
    if source.count(old_empty) != 1:
        raise ValueError('Frozen native empty-fact hook anchor differs')
    source = source.replace(old_empty, '        final = DIRECT_FINAL(windows, facts, boundaries)')
    marker = '    # Stage snapshots are append-only; an enlarged/resumed run has a new content key.'
    if source.count(marker) != 1:
        raise ValueError('Frozen native result metadata hook anchor differs')
    source = source.replace(marker, "    result.update(schema=DIRECT_RECIPE, source=DIRECT_RECIPE, caption_recipe=DIRECT_RECIPE,\n"
                            "                  direct_recipe_execution_ref=DIRECT_EXECUTION, native_fulltext_model_output=False,\n"
                            "                  native_window_observation_request_and_support_filter_unchanged=True,\n"
                            "                  boundary_review='off', fulltext_rewrite='direct_exact_native_fact_text',\n"
                            "                  no_actionclaim_from_unknown_boundaries=True, **DIRECT_FLAGS)\n" + marker)
    execution = reference(__file__)
    namespace = dict(bar.__dict__, DIRECT_FINAL=direct_final, DIRECT_RECIPE=RECIPE,
                     DIRECT_EXECUTION=execution, DIRECT_FLAGS=FLAGS)
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), namespace)

    frozen_configuration = None

    def params(current_args, code):
        nonlocal frozen_configuration
        value = {**old_params(current_args, code), 'caption_recipe': RECIPE,
                'direct_recipe_execution_ref': execution, 'boundary_review': 'off',
                'fulltext_rewrite': 'direct_exact_native_fact_text',
                'native_request_sampling_and_labels_unchanged': False,
                'native_window_observation_request_and_support_filter_unchanged': True,
                'no_actionclaim_from_unknown_boundaries': True,
                **FLAGS}
        if current_args.boundary_review != 'off':
            raise ValueError('Direct recipe boundary setting changed during execution')
        if frozen_configuration is None:
            frozen_configuration = copy.deepcopy(value)
        elif value != frozen_configuration:
            raise ValueError('Direct recipe parameters changed during execution')
        return copy.deepcopy(frozen_configuration)

    bar.process_source, bar.params = namespace['process_source'], params
    try:
        yield {'caption_recipe': RECIPE, 'execution_ref': execution,
               'execution_identity_changed': True, 'boundary_model_calls': 0,
               'fulltext_model_calls': 0, **FLAGS}
    finally:
        bar.process_source, bar.params = old_process, old_params


class Evidence:
    """One content/SHA read per immutable artifact; no population scan or model dispatch."""
    def __init__(self):
        self.cache = {}

    def raw(self, path, expected=None):
        path = Path(path).resolve()
        if path not in self.cache:
            before = path.stat()
            data = path.read_bytes()
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('Original artifact changed during read: ' + str(path))
            self.cache[path] = (data, hashlib.sha256(data).hexdigest(), after)
        data, actual, _ = self.cache[path]
        if expected is not None and actual != expected:
            raise ValueError('Original artifact SHA differs: ' + str(path))
        return data

    def checked(self, ref):
        bar, _ = native()
        return bar.core.strict_json(self.raw(ref['path'], ref['sha256']))

    def load(self, path):
        bar, _ = native()
        return bar.core.strict_json(self.raw(path))

    def ref(self, path):
        path = Path(path).resolve()
        self.raw(path)
        return dict(path=str(path), sha256=self.cache[path][1])

    def verify_stable(self):
        for path, (_, _, before) in self.cache.items():
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('Immutable source evidence changed after replay: ' + str(path))


def replay_observations(evidence, original, prepared, root):
    bar, _ = native()
    windows = [m['window'] for m in prepared['windows']]
    observations = original['window_observations']
    if len(windows) != len(observations):
        raise ValueError('Original observation/window count differs')
    union = evidence.checked(original['raw_rgb_manifest'])
    if union['binding'] != original['binding'] or prepared['binding'] != original['binding']:
        raise ValueError('Original prepared/union/source binding differs')
    forbidden = {p: set(v) for p, v in prepared['forbidden'].items()}
    replays = []
    model = evidence.checked(original['model_identity_ref'])
    for manifest, observation in zip(prepared['windows'], observations):
        wid = manifest['window']['window_id']
        if observation['window'] != manifest['window'] or observation['status'] not in ('candidate', 'abstain'):
            raise ValueError('Nonterminal or mismatched original observation')
        local_manifest = evidence.load(root / 'windows' / wid / 'manifest.json')
        if local_manifest != {**manifest, 'binding': original['binding'],
                'union_manifest_sha256': prepared['union_manifest_sha256'],
                'raw2d_mask': prepared['raw2d_mask'], **bar.FLAGS}:
            raise ValueError('Original window/prepared metadata differs')
        for frame in manifest['frames']:
            if frame not in union['frames']:
                raise ValueError('Window RGB not in original union')
            image = (root / 'rgb' / frame['file']).resolve()
            if not image.is_relative_to(root / 'rgb'):
                raise ValueError('RGB frame escapes original source')
            evidence.raw(image, frame['sha256'])
        for group in manifest.get('hand_roi_evidence', {}).values():
            if isinstance(group, dict):
                for crop in group.get('crops', []):
                    evidence.raw(crop['path'], crop['sha256'])
        attempt = Path(observation['attempt_path']).resolve()
        if attempt.parent != root / 'windows' / wid or not re.fullmatch(r'attempt-[0-9]+', attempt.name):
            raise ValueError('Original observation attempt/source/window path differs')
        receipt = evidence.load(attempt / 'receipt.json')
        required = {'request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'response.json', 'final.json'}
        if not required <= set(receipt['files_sha256']) or receipt['status'] != 'candidate':
            raise ValueError('Original terminal observation receipt differs')
        for name, expected in receipt['files_sha256'].items():
            if Path(name).name != name:
                raise ValueError('Unsafe original receipt artifact name')
            evidence.raw(attempt / name, expected)
        binding = evidence.load(attempt / 'binding.json')
        if binding != receipt['binding'] or binding != {**original['binding'],
                'window': manifest['window'], 'frames_sha256': bar.batch.digest(manifest['frames'])}:
            raise ValueError('Original generation/frame/source binding differs')
        request = evidence.load(attempt / 'request.json')
        generation = bar.core.validate_generation(attempt, request)
        if generation.get('model_call_performed') is not True or generation['model_identity_sha256'] != model['sha256']:
            raise ValueError('Original observation was not an actual bound model generation')
        if Path(request['model']).resolve() != Path(model['root']).resolve():
            raise ValueError('Original observation model path differs')
        raw = bar.core.strict_json(evidence.raw(attempt / 'response.raw.txt'))
        replay = bar.filter_supported_observation(raw, manifest, forbidden)
        response, final = evidence.load(attempt / 'response.json'), evidence.load(attempt / 'final.json')
        if replay != response or response != observation['json'] or final['json'] != response:
            raise ValueError('Full original support-filter replay differs from admitted observation')
        if observation['status'] != ('abstain' if replay['abstain'] else 'candidate'):
            raise ValueError('Original abstention/candidate state differs')
        replays.append({'window': manifest['window'], 'status': observation['status'], 'json': replay})
    facts = bar.facts_from_observations(windows, replays)
    if facts != original['facts']:
        raise ValueError('Original admitted fact text/part/frame/source-clause replay differs')
    return windows, facts


def pilot(report, cpu_tests):
    started = time.perf_counter()
    bar, pipeline = native()
    evidence, cases = Evidence(), []
    budgets = []
    for version, shard, sid, ordinal in CASES:
        production = BAR / ('curated222939_collaborative_v%d' % version)
        consumer = production / 'consumers' / ('shard-%04d' % shard)
        completed = consumer / 'completed.jsonl'
        notice, line_index = None, None
        with completed.open('rb') as stream:
            for index, line in enumerate(stream):
                if not line.endswith(b'\n'):
                    break
                item = bar.core.strict_json(line)
                if item.get('id') == sid:
                    notice, line_index = item, index
                    break
        if notice is None:
            raise ValueError('Fixed case has no actual completed reference: ' + sid)
        joint = evidence.checked(notice['joint_candidate_ref'])
        original = evidence.checked(joint['visual_result_ref'])
        if joint['source_binding'] != original['binding'] or joint['id'] != sid or joint['joint_complete']:
            raise ValueError('Fixed original incomplete/source binding differs')
        if sid != 'BrduoqCtBJTMfLie' and original['final']['status'] != 'failed':
            raise ValueError('Fixed rewrite-failure case is not an actual failed final')
        source_plan = evidence.load(RT / ('curated_collaborative%d_source_plan_v1.json' % version))
        selected_ref = next(ref for member, ref in source_plan['selected_sources'].items()
                            if source_plan['members'][member]['mode'] == 'consumer'
                            and source_plan['members'][member]['shard'] == shard)
        selected = evidence.checked(selected_ref)
        match = next((row for row in selected if row['id'] == sid), None)
        if match != {'id': sid, 'split': 'train', 'batch_id': joint['batch_id'], 'source_ordinal': ordinal}:
            raise ValueError('Actual selected opaque source identity/range differs')
        root = Path(original['raw_rgb_manifest']['path']).resolve().parent.parent
        prepared_receipt = evidence.load(root / 'prepared.receipt.json')
        prepared = evidence.checked({'path': str(root / 'prepared.json'),
                                    'sha256': prepared_receipt['prepared_sha256']})
        plan = evidence.load(root / 'plan.json')
        if (prepared_receipt['binding'] != original['binding'] or bar.batch.digest(plan) != original['binding']['full_source_row_sha256']
                or bar.batch.digest(original['manifest_record']) != original['binding']['manifest_record_sha256']
                or plan['binding'] != original['binding']['source_binding']):
            raise ValueError('Original plan/source/prepared SHA binding differs')
        for path in (BAR / 'curated222939_repair_v1' / 'attempt-budget/train' / (sid + '.json'),
                     consumer / 'attempt-budget/train' / (sid + '.json')):
            budgets.append({'path': str(path), 'before_sha256': sha(path) if path.exists() else None})
        cpu_start = time.perf_counter()
        windows, facts = replay_observations(evidence, original, prepared, root)
        final = direct_final(windows, facts, [])
        full = (final['status'] == 'candidate' and len(final['json']['windows']) == len(windows)
                and all(w['caption'].strip() for w in final['json']['windows']))
        visual = copy.deepcopy(original)
        visual.pop('result_ref', None)
        visual.update(status=final['status'], final=final, original_final=copy.deepcopy(original['final']),
            final_timed_caption=bar.core.timed_text({'windows': windows}, final['json']['windows']) if 'json' in final else '',
            coverage={**original['coverage'], 'full_window_caption': bool(full)},
            rewrite_fulltext_calls_per_attempt=0, caption_recipe=RECIPE, **FLAGS)
        result = pipeline.joint_candidate(joint, visual, copy.deepcopy(joint['style_result']), joint['annotator_model_sha256'])
        result.update(status='direct_cpu_joint_candidate' if result['joint_complete'] else
                      'direct_cpu_partial_candidate' if final['status'] != 'failed' else 'direct_cpu_rejected',
                      caption_recipe=RECIPE, original_joint_ref=notice['joint_candidate_ref'],
                      original_visual_result_ref=joint['visual_result_ref'], **FLAGS)
        if result['style_result'] != joint['style_result']:
            raise ValueError('Original joint Dance style changed')
        cases.append({'id': sid, 'source_ordinal': ordinal, 'source_version': version,
            'original_completed_line': {'path': str(completed), 'line_index': line_index},
            'original_final': copy.deepcopy(original['final']), 'original_joint_complete': False,
            'native_admission_full_filter_replayed': True, 'all_admitted_facts_unchanged': facts == original['facts'],
            'native_final_validator_passed': final['status'] != 'failed', 'window_count': len(windows),
            'admitted_fact_count': len(facts), 'window_statuses': [o['status'] for o in original['window_observations']],
            'admitted_facts': facts, 'direct_final': final, 'joint_candidate': result,
            'boundary_observations_preserved': copy.deepcopy(original['boundary_observations']),
            'historical_boundary_evidence_not_used_as_newly_generated': True,
            'used_boundary_ids': [], 'new_model_calls': 0, 'new_gpu_use': False,
            'cpu_replay_and_direct_seconds': time.perf_counter() - cpu_start})
    for budget in budgets:
        path = Path(budget['path'])
        budget['after_sha256'] = sha(path) if path.exists() else None
        if budget['after_sha256'] != budget['before_sha256']:
            raise ValueError('Original fixed source attempt budget changed')
    evidence.verify_stable()
    for name, expected in FROZEN.items():
        if sha(SOURCE / name) != expected:
            raise ValueError('Frozen native code changed after CPU pilot: ' + name)
    summary = {'schema': 'caption_direct_facts_pilot3_v1', 'caption_recipe': RECIPE,
        'execution_ref': reference(__file__), 'frozen_source_refs': [reference(SOURCE / name) for name in FROZEN],
        'cpu_tests': cpu_tests,
        'cpu_only': True, 'new_model_calls': 0, 'new_gpu_use': False, 'new_source_attempts': 0,
        'new_budget_reservations': 0, 'original_production_completion_counts_changed': False,
        'source_count': 3, 'joint_complete_candidates': sum(c['joint_candidate']['joint_complete'] for c in cases),
        'partial_candidates': sum(c['joint_candidate']['status'] == 'direct_cpu_partial_candidate' for c in cases),
        'rejected_candidates': sum(c['joint_candidate']['status'] == 'direct_cpu_rejected' for c in cases),
        'cases': cases, 'attempt_budget_before_after': budgets,
        'immutable_evidence_refs': [dict(path=str(path), sha256=row[1]) for path, row in evidence.cache.items()],
        'immutable_artifacts_read_once': len(evidence.cache),
        'cpu_total_seconds': time.perf_counter() - started,
        'speedup_or_gpu_throughput_claim': None,
        'independent_review_required': True, 'native_fulltext_model_output': False, **FLAGS}
    report = Path(report).resolve()
    if report != REPO / 'results/caption_direct_facts_pilot3_v1.json':
        raise ValueError('Only the isolated fixed pilot report destination is permitted')
    if report.exists():
        raise ValueError('Pilot report already exists; do not overwrite an earlier experiment')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot3', action='store_true', required=True)
    parser.add_argument('--report', default=str(REPO / 'results/caption_direct_facts_pilot3_v1.json'))
    args = parser.parse_args()
    sys.path.insert(0, str(RT / 'smalltests'))
    from test_caption_direct_facts_v1 import run_cpu_tests
    test_start = time.perf_counter()
    cpu_tests = run_cpu_tests()
    cpu_tests['cpu_seconds'] = time.perf_counter() - test_start
    value = pilot(args.report, cpu_tests)
    print(json.dumps({key: value[key] for key in ('source_count', 'joint_complete_candidates',
        'partial_candidates', 'rejected_candidates', 'new_model_calls', 'cpu_total_seconds')}, sort_keys=True))


if __name__ == '__main__':
    main()
