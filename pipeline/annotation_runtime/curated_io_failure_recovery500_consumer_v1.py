"""Finite three-source IO-failure recovery; original outputs stay immutable."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

RT = Path(__file__).resolve().parent
BASE = RT / 'source_curated_consumer_v1/curated_pipeline.py'
BASE_SHA = 'd5cd607d4789e3b3cb361b2f32c35c4adc438cda8b9d2f5262dc143e61e952fc'
OUT = RT.parent / 'curated222939_io_failure_recovery_v1'
START, END = 500, 503
SELECTED = RT / 'curated_io_failure_recovery500_selected_v1.json'
SELECTED_SHA = '4a3d1cffa04c1f4c0f2a1432c80acef5846fcac7a1a2533e8458095e205b620e'
REVISIONS = RT / 'curated_io_failure_recovery500_revision_requests_v1.json'
REVISIONS_SHA = '0fc3aea2520d29a18ccf8153a8125a8dcd80301abfbeae106b2a20adee59a527'
KEYS = ('id', 'split', 'batch_id', 'source_ordinal')


def load_pipeline():
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
        raise RuntimeError('Original curated consumer snapshot changed')
    spec = importlib.util.spec_from_file_location('curated_consumer_original_v1', BASE)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    return pipeline


def selected_rows(pipeline, manifest):
    if pipeline.sha(SELECTED) != SELECTED_SHA or pipeline.sha(REVISIONS) != REVISIONS_SHA:
        raise ValueError('Recovery selection/revision SHA changed')
    summary = pipeline.load(manifest)
    rows = pipeline.load(SELECTED)
    revisions = pipeline.load(REVISIONS)
    if (summary['total'] != 222939 or len(rows) != 3
            or [row['source_ordinal'] for row in rows] != list(range(START, END))
            or len({row['id'].casefold() for row in rows}) != 3
            or revisions['source_count_unique'] != 3
            or revisions['source_range'] != {'start': START, 'end': END}
            or [{key: request[key] for key in KEYS} for request in revisions['requests']] != rows):
        raise ValueError('Recovery requires exactly the original three unique failures')
    batch = next(batch for batch in summary['batches'] if batch['batch_id'] == 'train-b000001')
    if (batch['first_source_ordinal'] != START or batch['count'] != 500
            or batch['sha256'] != '1e2676e1dfb524bafbcf4ea91952e5b4355e7bdc9942d08dad9c311cdce2af7a'
            or pipeline.sha(batch['path']) != batch['sha256']):
        raise ValueError('Recovery original batch SHA/binding changed')
    with Path(batch['path']).open() as stream:
        original = [json.loads(next(stream)) for _ in rows]
    if [{key: row[key] for key in KEYS} for row in original] != rows:
        raise ValueError('Recovery selected sources differ from exact original batch rows')
    for row, request in zip(rows, revisions['requests']):
        ref = request['old_terminal_ref']
        expected_old = RT.parent / 'curated222939_production_v1/sources' / row['split'] / row['id'] / 'producer.terminal.json'
        terminal = pipeline.load(ref['path'])
        if (Path(ref['path']) != expected_old or pipeline.sha(ref['path']) != ref['sha256']
                or terminal['status'] != 'failed' or {key: terminal[key] for key in KEYS} != row
                or request['revision_request']['new_out'] != str(OUT)
                or request['revision_request']['expected_new_terminal_path'] != str(OUT / 'sources' / row['split'] / row['id'] / 'producer.terminal.json')
                or request['revision_request']['old_terminal_preserved'] is not True):
            raise ValueError('Recovery old failure SHA or independent new output changed')
    return rows


def configuration(pipeline, args):
    bar, _, _ = pipeline.modules(args.repo, args.ms_source)
    args.hand_roi_root = str(Path(args.out).resolve() / 'roi')
    value = bar.params(args, bar.code_identity())
    style = Path(bar.__file__).with_name('ms_dance_style.py')
    value['style_module_sha256'] = pipeline.sha(style) if style.exists() else None
    return value


def install(pipeline):
    source = inspect.getsource(pipeline.run_consumer)
    changes = [
        ("visual_out = out / 'visual' / f'shard-{args.shard:04d}'",
         "visual_out = out / 'visual' / 'io-failure-recovery500-n3g67-v1'"),
        ("'configuration': configuration, 'pipeline_code': ref(__file__), **FLAGS",
         "'configuration': configuration, 'pipeline_code': ref(__file__), 'consumer_wrapper': ref(CONSUMER_WRAPPER), **FLAGS"),
        ("assigned_total = sum(1 for i in range(summary['total']) if i % args.nshard == args.shard)",
         "assigned_total = len(args.ids)"),
        ("assigned_total = sum(1 for _ in iter_rows(summary, args.shard, args.nshard, ids=wanted))",
         "assigned_total = len(wanted)"),
    ]
    old = "                with visual.bundle_lock(directory):\n                    with visual.bundle_lock(root):"
    new = """                with visual.bundle_lock(directory):
                    if consumer_terminal.exists():
                        previous = load(consumer_terminal)
                        if previous.get('params_sha256') != visual.digest(configuration):
                            raise ValueError('Locked consumer parameter fingerprint differs')
                        artifact = previous.get('joint_candidate_ref')
                        if artifact and sha(artifact['path']) != artifact['sha256']:
                            raise ValueError('Locked resumed joint candidate changed')
                        seen.add(hint['id'])
                        counts['resumed_terminal'] += 1
                        continue
                    with visual.bundle_lock(root):"""
    changes.append((old, new))
    changes.append(("                hint = json.loads(line)", """                hint = json.loads(line)
                selected = RECOVERY_ROWS.get(hint['id'])
                if selected and {key: hint[key] for key in RECOVERY_KEYS} != selected:
                    raise ValueError('Recovery ready source differs from exact selected binding')"""))
    changes.append(("                            joint = joint_candidate(row, result, style_result, model_sha)", """                            joint = joint_candidate(row, result, style_result, model_sha)
                            joint['recovery_revision_requests_ref'] = ref(RECOVERY_REVISIONS)
                            joint['old_failed_producer_terminal_ref'] = RECOVERY_OLD_REFS[row['id']]
                            joint['edition'] = 'io-failure-recovery500-v1'"""))
    for old, new in changes:
        if source.count(old) != 1:
            raise RuntimeError('Consumer wrapper patch anchor changed: ' + old)
        source = source.replace(old, new)
    if source.count("queue = out / 'ready.jsonl'") != 1 or "PRIVATE_READY" in source:
        raise RuntimeError('Native public-ready input changed')
    pipeline.RECOVERY_KEYS = KEYS
    pipeline.RECOVERY_REVISIONS = str(REVISIONS)
    pipeline.RECOVERY_ROWS = {row['id']: row for row in pipeline.load(SELECTED)}
    pipeline.RECOVERY_OLD_REFS = {row['id']: row['old_terminal_ref'] for row in pipeline.load(REVISIONS)['requests']}
    joint_source = inspect.getsource(pipeline.joint_candidate)
    old_windows = "    windows = visual_result.get('final', {}).get('json', {}).get('windows', [])"
    if joint_source.count(old_windows) != 1:
        raise RuntimeError('Recovery joint null-safety patch anchor changed')
    joint_source = joint_source.replace(old_windows, "    windows = ((visual_result.get('final') or {}).get('json') or {}).get('windows', [])")
    exec(compile(joint_source, str(Path(__file__).resolve()), 'exec'), pipeline.__dict__)
    pipeline.CONSUMER_WRAPPER = str(Path(__file__).resolve())
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), pipeline.__dict__)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--selected-sources', required=True)
    extra, remaining = parser.parse_known_args(argv)
    pipeline = load_pipeline()
    args = pipeline.parser().parse_args(remaining)
    if (args.mode != 'consumer' or args.shard != 0 or args.nshard != 1 or args.limit != 0
            or args.ids or args.stage != 'run' or not args.watch
            or Path(extra.selected_sources).resolve() != SELECTED
            or Path(args.out).resolve() != OUT):
        raise ValueError('Only finite recovery consumer shard0/nshard1/limit0 watch/run admitted')
    rows = selected_rows(pipeline, args.manifest)
    if pipeline.load(extra.selected_sources) != rows:
        raise ValueError('Explicit selected sources differ from original curated batches')
    args.ids = [row['id'] for row in rows]
    install(pipeline)
    result = pipeline.run_consumer(args)
    export = export_candidates(pipeline, rows)
    return {**result, 'review_required_export': export}


def export_candidates(pipeline, rows):
    if pipeline.sha(REVISIONS) != REVISIONS_SHA or pipeline.sha(SELECTED) != SELECTED_SHA:
        raise ValueError('Recovery export selection/revision SHA changed')
    requests = {request['id']: request for request in pipeline.load(REVISIONS)['requests']}
    exported = []
    for row in rows:
        directory = OUT / 'sources' / row['split'] / row['id']
        terminal_path = directory / 'consumer.terminal.json'
        terminal = pipeline.load(terminal_path)
        joint_ref = terminal.get('joint_candidate_ref')
        joint = {}
        if joint_ref:
            expected = directory / 'joint-results' / (joint_ref['sha256'] + '.json')
            if Path(joint_ref['path']) != expected or pipeline.sha(expected) != joint_ref['sha256']:
                raise ValueError('Recovery joint candidate path/content SHA changed')
            joint = pipeline.load(expected)
        complete = bool(joint.get('joint_complete')) and terminal.get('status') == 'joint_candidate'
        text = joint.get('target_text', '') if complete else ''
        if complete and (not text.startswith('Dance style: ') or not text.strip()):
            raise ValueError('Recovery complete candidate lacks joint target text')
        exported.append({**row, 'edition': 'io-failure-recovery500-v1',
                         'old_failed_producer_terminal_ref': requests[row['id']]['old_terminal_ref'],
                         'consumer_terminal_ref': pipeline.ref(terminal_path),
                         'joint_candidate_ref': joint_ref, 'status': terminal['status'],
                         'joint_complete': complete, 'target_text': text,
                         'review_required': True, 'training_eligible': False, **pipeline.FLAGS})
    payload = {'schema': 'curated_io_failure_recovery_review_required_v1',
               'selected_count': 3, 'total_source_count': 222939,
               'revision_requests_ref': pipeline.ref(REVISIONS),
               'old_outputs_preserved': True, 'candidates': exported, **pipeline.FLAGS}
    path = OUT / 'exports' / ('review-required-' + pipeline.digest(payload) + '.json')
    pipeline.save(path, payload)
    return pipeline.ref(path)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False), flush=True)
