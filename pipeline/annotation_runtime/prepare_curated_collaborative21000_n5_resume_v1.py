"""CPU-only immutable resume builder, gated by actual finite/old resource receipts."""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import prepare_curated_coarse_empty_n5_v1 as finite

RT = Path(__file__).resolve().parent
BAR = RT.parent
MANAGER = RT / 'curated_collaborative21000_n5_resume_resource_manager_v1.py'
BUILDER = Path(__file__).resolve()
FROZEN_PREPARE_SHA = '1190df9feed04400391c9d4b7b0df11e9bb40b2ef642754d3dcaaffcdb733319'
FINITE_LAUNCH_SHA = '9f5e51a9e2461e864c824add7370ff89fcb1b84db1eae13c59a26b81d77f6806'
FINITE_MANAGER_SHA = '22f5dbb1b9acdb36cb8f4bdb894dc2fcf54a023cc38c35a54c6b848e68e24b68'
if finite.sha(finite.__file__) != FROZEN_PREPARE_SHA:
    raise RuntimeError('frozen finite builder source changed')
old = finite.module('n5_resume_original_collab2', RT / 'curated_collaborative21000_resource_manager_v2.py')
base = old.c.base
base.sha = finite.sha
old.c.ALLOWED = {'producer': {'N5': [(0,), (1,)]}}
base.AUTHORIZED_PAIRS = {'N5': ((0,), (1,))}
SAFE_KEYS = ('all_holders_safe', 'locks_released', 'leases_restored', 'managed_parent_wait', 'unknown_jobs_not_signaled')
TERMINAL_STATUSES = ('joint_repair_candidate', 'incomplete_repair_candidate',
                     'failed_preserved_no_redispatch', 'not_dispatched_after_shared_failure')
CHANGED_KEYS = {'phase', 'queue', 'holders', 'holder_capture', 'completed_resource_gate',
                'resource_manager', 'source_sha256', 'resume_original_config'}


def phase(gpu):
    if type(gpu) is not int or gpu not in (0, 1):
        raise RuntimeError('only N5 physical GPU0 or GPU1 producer resume allowed')
    return 'curated_collaborative21000_n5g%d_resume_v1' % gpu


def paths(gpu):
    stem = phase(gpu)
    return {'queue': BAR / 'queues' / stem, 'launch': RT / (stem + '_launch.json'),
            'capture': RT / (stem + '_holder_capture.json'), 'receipt': RT / (stem + '_build_receipt.json')}


def same_identity(a, b):
    return all(a[k] == b[k] for k in base.IDENTITY_KEYS)


def old_config(gpu):
    path, digest = finite.PREVIOUS[gpu]
    return {'path': str(path), 'sha256': digest}, finite.checked({'path': str(path), 'sha256': digest})


def check_admission(cfg, config_ref, admission):
    expected = {'config_sha256': config_ref['sha256'], 'manager_sha256': finite.BASE_SHA,
                'phase': cfg['phase'], 'node': 'N5', 'gpus': cfg['gpus'], 'targetcount': cfg['targetcount'],
                'producer': cfg['producer'], 'plan': cfg['plan'], 'out': cfg['out'],
                'source_sha256': cfg['source_sha256'], 'model_identity_sha256': cfg['modelref']['model_identity_sha256']}
    if any(admission.get(k) != v for k, v in expected.items()):
        raise RuntimeError('actual finite admission config/source/model binding changed')
    argv = ['/opt/venv_eval/bin/python', '-B', '-u', str(finite.MANAGER),
            '--config', config_ref['path'], '--config-sha', config_ref['sha256'], '--manager-sha', finite.BASE_SHA]
    if admission['manager_identity']['argv'] != argv or admission['manager_identity']['uid'] != os.getuid():
        raise RuntimeError('actual finite manager identity/argv mismatch')
    if (admission['budget']['headroom'] < 160 * 2**30 or len(admission['holders']) != 2
            or [h['gpu'] for h in admission['holders']] != [0, 1]):
        raise RuntimeError('actual finite TP2 budget/holder inventory mismatch')
    for holder, owner in zip(admission['holders'], cfg['holders']):
        if holder['uuid'] != owner['uuid'] or not same_identity(holder['identity'], owner):
            raise RuntimeError('actual finite admitted holder differs from frozen capture')


def resource_gate():
    # All paths are finite, fixed receipts or exactly three jobs, never a pool scan.
    prior = finite.prior_safe(base)
    launch_ref = {'path': str(finite.LAUNCH), 'sha256': FINITE_LAUNCH_SHA}
    cfg = finite.checked(launch_ref)
    if (cfg['node'] != 'N5' or cfg['gpus'] != [0, 1] or cfg['phase'] != finite.PHASE
            or cfg['queue'] != str(finite.QUEUE) or cfg['out'] != str(finite.OUT)
            or cfg['targetcount'] != 3 or cfg['input_count'] != 222939
            or cfg['global_revision'] != finite.REVISION
            or cfg['maximum_total_visual_calls'] != 3 or cfg['maximum_text_model_calls'] != 0):
        raise RuntimeError('only current frozen finite-three transaction allowed')
    if finite.checked(cfg['previous_resources_ref']) != prior or finite.sha(finite.MANAGER) != FINITE_MANAGER_SHA:
        raise RuntimeError('finite launch does not bind actual old safe/restored history')
    refs = {launch_ref['path']: launch_ref['sha256']}
    refs.update(finite.references(prior))
    refs.update(finite.references(cfg['previous_resources_ref']))
    refs[str(finite.MANAGER)] = FINITE_MANAGER_SHA
    for gpu, row in enumerate(prior['previous_pairs']):
        old_ref, previous = old_config(gpu)
        transition_path = RT / ('n5_coarse_empty_transition_gpu%d_v1.json' % gpu)
        transition = finite.load(transition_path)
        admission_path = Path(previous['queue']) / 'admission.json'
        admitted = finite.load(admission_path)
        live_path = Path(previous['queue']) / 'live.json'
        live = finite.load(live_path)
        expected = {'schema': 'owned_producer_resource_transition_v1', 'node': 'N5', 'gpus': [gpu],
                    'old_config_ref': old_ref, 'original_queue': previous['queue'], 'signal': 'SIGTERM',
                    'signal_target': 'exact_owned_manager_pidfd',
                    'reason': 'temporary_finite_three_coarse_window_validation',
                    'original_outputs_preserved': True, 'incomplete_sources_remain_pending': True,
                    'old_source_selection_completed': False, 'unknown_jobs_not_signaled': True}
        if (any(transition.get(k) != v for k, v in expected.items())
                or not same_identity(transition['manager_identity'], admitted['manager_identity'])
                or live.get('phase') != previous['phase'] or live.get('argv') != previous['argv']
                or not same_identity(live['manager_identity'], admitted['manager_identity'])
                or live.get('child_identity') is None
                or not same_identity(transition['worker_identity'], live['child_identity'])
                or transition['worker_identity']['argv'] != previous['argv']
                or transition['worker_identity']['ppid'] != admitted['manager_identity']['pid']
                or transition['worker_identity']['uid'] != admitted['manager_identity']['uid']
                or transition['worker_identity']['cgroup'] != admitted['manager_identity']['cgroup']
                or transition['worker_identity']['CVD'] != ['CUDA_VISIBLE_DEVICES=%d' % gpu]
                or type(transition.get('observed_unix_ns')) is not int
                or not admission_path.stat().st_mtime_ns <= transition['observed_unix_ns'] <= (Path(previous['queue']) / 'resources_safe.json').stat().st_mtime_ns):
            raise RuntimeError('old producer safe TERM transition proof mismatch')
        if not same_identity(row['restored']['holder_identity'], cfg['holders'][gpu]) or row['restored']['uuid'] != cfg['holders'][gpu]['uuid']:
            raise RuntimeError('finite captured holder is not restored old producer holder')
        refs.update(finite.references([finite.ref(transition_path), finite.ref(live_path)]))
    admission_path, safe_path = finite.QUEUE / 'admission.json', finite.QUEUE / 'resources_safe.json'
    admission, safe = finite.load(admission_path), finite.load(safe_path)
    check_admission(cfg, launch_ref, admission)
    if safe.get('phase') != cfg['phase'] or any(safe.get(k) is not True for k in SAFE_KEYS):
        raise RuntimeError('finite resources not actually safe/restored')
    if safe_path.stat().st_mtime_ns < admission_path.stat().st_mtime_ns:
        raise RuntimeError('finite safe receipt predates admission')
    restored = []
    for gpu, holder in enumerate(admission['holders']):
        path = finite.QUEUE / ('holder_restored_gpu%d.json' % gpu)
        value = finite.load(path)
        if (value.get('gpu') != gpu or value.get('uuid') != holder['uuid']
                or value.get('restore_argv') != base.RESTORE_ARGV or value.get('managed_parent_wait') is not True
                or value['original_identity'] != holder['identity']
                or not admission_path.stat().st_mtime_ns <= path.stat().st_mtime_ns <= safe_path.stat().st_mtime_ns):
            raise RuntimeError('finite restored physical holder evidence mismatch')
        restored.append(value)
        refs.update(finite.references(finite.ref(path)))
    refs.update(finite.references([finite.ref(admission_path), finite.ref(safe_path)]))
    finished_path, failure_path = finite.QUEUE / 'worker_exit.json', finite.QUEUE / 'failure.json'
    finished = finite.load(finished_path) if finished_path.exists() else None
    failure = finite.load(failure_path) if failure_path.exists() else None
    if finished is None and failure is None:
        raise RuntimeError('finite actual worker exit or actual preserved failure required')
    for path, value in ((finished_path, finished), (failure_path, failure)):
        if value is None:
            continue
        if value.get('phase') != cfg['phase'] or not admission_path.stat().st_mtime_ns <= path.stat().st_mtime_ns <= safe_path.stat().st_mtime_ns:
            raise RuntimeError('finite exit/failure is not current admitted transaction')
        refs.update(finite.references(finite.ref(path)))
    if finished is not None and (type(finished.get('exit_code')) is not int or finished.get('targetcount') != 3
            or finished.get('input_count') != 222939 or finished.get('producer') != cfg['producer']
            or finished.get('plan') != cfg['plan'] or finished.get('source_unchanged') is not True
            or finished.get('semantic_approval') is not False or finished.get('training_entry_updated') is not False):
        raise RuntimeError('finite worker exit provenance changed')
    if failure is not None and any(failure.get(k) is not True for k in ('partial_outputs_preserved', 'unknown_jobs_not_signaled')):
        raise RuntimeError('actual finite failures must be preserved, not relabeled success')
    manifest = finite.checked(cfg['jobs_manifest_ref'])
    if len(manifest.get('jobs', [])) != 3:
        raise RuntimeError('only complete three-source finite evidence admitted')
    refs.update(finite.references(cfg['jobs_manifest_ref']))
    terminal_refs = []
    for hint in manifest['jobs']:
        job = finite.checked(hint['job_ref'])
        path = finite.OUT / 'worker-terminals' / (hint['source_key'] + '.json')
        terminal_ref = finite.ref(path)
        terminal = finite.checked(terminal_ref)
        if (job['source_key'] != hint['source_key'] or terminal.get('source_key') != hint['source_key']
                or terminal.get('original_job_ref') != hint['job_ref'] or terminal.get('global_revision') != finite.REVISION
                or terminal.get('status') not in TERMINAL_STATUSES or terminal.get('semantic_approval') is not False
                or terminal.get('training_eligible') is not False
                or not admission_path.stat().st_mtime_ns <= path.stat().st_mtime_ns <= safe_path.stat().st_mtime_ns):
            raise RuntimeError('finite per-source failure/partial evidence changed')
        if terminal.get('joint_result_ref') and terminal['joint_result_ref']['path'] != str(finite.OUT / 'revisions' / finite.REVISION / hint['source_key'] / 'result.json'):
            raise RuntimeError('finite terminal result outside its current source revision')
        terminal_refs.append(terminal_ref)
        refs.update(finite.references([hint['job_ref'], terminal_ref, terminal]))
    leases_path, worker_path = finite.QUEUE / 'actual_leases_held.json', finite.QUEUE / 'actual_worker_admission.json'
    held, worker_admission = finite.load(leases_path), finite.load(worker_path)
    if (held.get('schema') != 'coarse_empty_n5_actual_leases_held_v1'
            or held.get('config_ref') != launch_ref or held.get('resource_admission_ref') != finite.ref(admission_path)
            or held.get('node') != 'N5' or held.get('gpus') != [0, 1]
            or not same_identity(held['manager_identity'], admission['manager_identity'])
            or worker_admission.get('launch_config_ref') != launch_ref
            or worker_admission.get('resource_admission_ref') != finite.ref(admission_path)
            or worker_admission.get('actual_held_leases_ref') != finite.ref(leases_path)
            or worker_admission.get('source_job_refs') != [h['job_ref'] for h in manifest['jobs']]
            or worker_admission.get('jobs_manifest_ref') != cfg['jobs_manifest_ref']
            or worker_admission.get('global_revision') != finite.REVISION
            or worker_admission.get('gpus') != [0, 1] or worker_admission.get('tensor_parallel_size') != 2
            or worker_admission.get('out') != str(finite.OUT)):
        raise RuntimeError('actual finite direct-parent worker admission/held lease provenance mismatch')
    if len(held.get('leases', [])) != 2 or [v.get('gpu') for v in held['leases']] != [0, 1]:
        raise RuntimeError('both finite actual held GPU leases required')
    for value, holder in zip(held['leases'], admission['holders']):
        finite.verify_lease_value(value['stat'])
        if (value['stat'] is None or value['path'] != '/tmp/od_reserve_%d' % holder['gpu']
                or value['uuid'] != holder['uuid']):
            raise RuntimeError('finite held lease path/UUID/exact integer-ns provenance mismatch')
    for path in (leases_path, worker_path):
        if not admission_path.stat().st_mtime_ns <= path.stat().st_mtime_ns <= safe_path.stat().st_mtime_ns:
            raise RuntimeError('finite actual worker/lease proof outside admitted transaction')
    last_path = finite.QUEUE / 'last_result_proof.json'
    if last_path.exists():
        last = finite.load(last_path)
        if (last.get('terminal_refs') != terminal_refs or last.get('partial_outputs_preserved') is not True
                or type(last.get('actual_model_dispatches')) is not int or not 0 <= last['actual_model_dispatches'] <= 3
                or last.get('text_model_calls') != 0 or last.get('semantic_approval') is not False):
            raise RuntimeError('finite last-result source failure preservation proof changed')
        if not admission_path.stat().st_mtime_ns <= last_path.stat().st_mtime_ns <= safe_path.stat().st_mtime_ns:
            raise RuntimeError('finite last result proof outside current admitted transaction')
        for reference in last.get('evidence_refs', []):
            if not Path(reference['path']).resolve().is_relative_to(finite.OUT.resolve()):
                raise RuntimeError('finite last result evidence outside current output')
        refs.update(finite.references([finite.ref(last_path), last]))
    elif finished is None:
        raise RuntimeError('without worker_exit actual last-result source failures proof required')
    summary_path = finite.OUT / 'finite.stage.summary.json'
    if summary_path.exists():
        summary = finite.load(summary_path)
        if (summary.get('terminals') != terminal_refs or summary.get('jobs_manifest_ref') != cfg['jobs_manifest_ref']
                or summary.get('source_job_refs') != [h['job_ref'] for h in manifest['jobs']]
                or summary.get('source_count') != 3 or summary.get('terminal_count') != 3
                or summary.get('global_revision') != finite.REVISION or summary.get('text_model_calls') != 0
                or type(summary.get('visual_model_calls')) is not int or not 0 <= summary['visual_model_calls'] <= 3
                or summary.get('partial_outputs_preserved') is not True or summary.get('semantic_approval') is not False
                or summary.get('training_eligible') is not False):
            raise RuntimeError('finite current summary source/partial flags mismatch')
        if finished is not None:
            saved = finished.get('summaries', {}).get(str(summary_path), {}).get('actual_finite_result')
            if saved != summary:
                raise RuntimeError('finite actual summary differs from worker_exit summary')
        if not admission_path.stat().st_mtime_ns <= summary_path.stat().st_mtime_ns <= safe_path.stat().st_mtime_ns:
            raise RuntimeError('finite summary outside current admitted transaction')
        for reference in summary.get('evidence_refs', []):
            if not Path(reference['path']).resolve().is_relative_to(finite.OUT.resolve()):
                raise RuntimeError('finite summary evidence outside current output')
        refs.update(finite.references([finite.ref(summary_path), summary]))
    elif finished is not None:
        raise RuntimeError('actual worker_exit requires its finite stage summary')
    for name in ('actual_leases_held.json', 'actual_worker_admission.json', 'live.json'):
        path = finite.QUEUE / name
        if path.exists():
            refs.update(finite.references(finite.ref(path)))
    # Bind referenced partial raw responses/results without modifying or approving them.
    for path, digest in refs.items():
        if finite.sha(path) != digest:
            raise RuntimeError('resume resource/evidence SHA changed: ' + path)
    return {'schema': 'n5_collab2_resume_actual_resource_gate_v1', 'queue': str(finite.QUEUE),
            'launch_config': launch_ref, 'references': [{'path': p, 'sha256': h} for p, h in sorted(refs.items())],
            'restored_holders': restored, 'old_source_selection_completed': False,
            'finite_worker_exit_present': finished is not None,
            'finite_worker_exit_code': finished['exit_code'] if finished is not None else None,
            'finite_failure_present': failure is not None, 'source_failures_preserved': True,
            'semantic_approval': False, 'training_eligible': False}


def verify_resume_config(cfg):
    gpu = cfg['gpus'][0]
    expected_paths = paths(gpu)
    original_ref, previous = old_config(gpu)
    if (cfg['node'] != 'N5' or cfg['mode'] != 'producer' or cfg['gpus'] != [gpu]
            or cfg['phase'] != phase(gpu) or cfg['queue'] != str(expected_paths['queue'])
            or cfg.get('resume_original_config') != original_ref or cfg['resource_manager'] != finite.ref(MANAGER)
            or cfg['holder_capture']['path'] != str(expected_paths['capture'])
            or set(cfg) != set(previous) | {'resume_original_config'}):
        raise RuntimeError('exact independent N5 resume config/phase binding changed')
    if any(cfg[k] != v for k, v in previous.items() if k not in CHANGED_KEYS):
        raise RuntimeError('original producer argv/selection/env/model/out/reuse changed')
    if any(cfg['source_sha256'].get(p) != h for p, h in previous['source_sha256'].items()):
        raise RuntimeError('original frozen producer source inventory changed')
    gate = resource_gate()
    if cfg['completed_resource_gate'] != gate:
        raise RuntimeError('strict actual finite and old transition resource gate changed')
    owner = cfg['holders'][0]
    restored = gate['restored_holders'][gpu]
    if owner['uuid'] != restored['uuid'] or not same_identity(owner, restored['holder_identity']):
        raise RuntimeError('fresh holder is not actually restored finite holder')
    expected = dict(previous['source_sha256'])
    expected.update({str(BUILDER): finite.sha(BUILDER), str(MANAGER): finite.sha(MANAGER),
                     original_ref['path']: original_ref['sha256'],
                     cfg['holder_capture']['path']: cfg['holder_capture']['sha256'],
                     str(finite.__file__): FROZEN_PREPARE_SHA})
    expected.update({r['path']: r['sha256'] for r in gate['references']})
    if cfg['source_sha256'] != expected:
        raise RuntimeError('complete resume sources/config/evidence inventory changed')
    return gate


def build(gpu):
    finite.check_node()
    p = paths(gpu)
    if any(path.exists() for path in p.values()):
        raise RuntimeError('immutable resume launch/capture/receipt/queue already exists; no overwrite')
    gate = resource_gate()  # Must succeed BEFORE any real capture or config creation.
    original_ref, previous = old_config(gpu)
    cfg = copy.deepcopy(previous)
    cfg.update(phase=phase(gpu), queue=str(p['queue']), completed_resource_gate=gate,
               resource_manager=finite.ref(MANAGER), resume_original_config=original_ref)
    old.selected_rows(cfg)
    old.c.check_model(cfg)
    for path, digest in previous['source_sha256'].items():
        if finite.sha(path) != digest:
            raise RuntimeError('original producer source changed before fresh capture')
    old.readonly_lock_inventory([gpu])
    if base.headroom()['headroom'] < base.required_budget(cfg):
        raise RuntimeError('fresh producer headroom below native required budget')
    old.capture_current(argparse.Namespace(node='N5', mode='producer', gpus=[gpu], out=str(p['capture'])))
    captured = finite.load(p['capture'])
    cfg.update(holders=captured['holders'], holder_capture=finite.ref(p['capture']))
    cfg['source_sha256'].update({str(BUILDER): finite.sha(BUILDER), str(MANAGER): finite.sha(MANAGER),
                                original_ref['path']: original_ref['sha256'],
                                str(p['capture']): cfg['holder_capture']['sha256'],
                                str(finite.__file__): FROZEN_PREPARE_SHA})
    cfg['source_sha256'].update({r['path']: r['sha256'] for r in gate['references']})
    verify_resume_config(cfg)
    finite.save_once(p['launch'], cfg)
    receipt = {'schema': 'n5_collab2_resume_build_receipt_v1', 'launch_ref': finite.ref(p['launch']),
               'wrapper_ref': finite.ref(MANAGER), 'builder_ref': finite.ref(BUILDER),
               'original_config_ref': original_ref, 'resource_gate': gate,
               'fresh_holder_capture': cfg['holder_capture'], 'source_selection': cfg['source_selection'],
               'source_range': cfg['source_range'], 'targetcount': 5000, 'out': cfg['out'],
               'reuse_existing_results': previous['reuse_existing_results'], 'original_argv_unchanged': True,
               'signals_sent': False, 'GPU_started': False, 'semantic_approval': False,
               'built_unix_ns': time.time_ns(), 'base_manager_sha256': finite.BASE_SHA,
               'command': ['/opt/venv_eval/bin/python', '-B', '-u', str(MANAGER), '--config', str(p['launch']),
                           '--config-sha', finite.sha(p['launch']), '--manager-sha', finite.BASE_SHA]}
    finite.save_once(p['receipt'], receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--build', action='store_true')
    actions.add_argument('--check-gate', action='store_true')
    parser.add_argument('--gpu', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.build and args.gpu is None:
        parser.error('--build requires --gpu 0 or 1')
    print(json.dumps(build(args.gpu) if args.build else resource_gate(), ensure_ascii=False))
