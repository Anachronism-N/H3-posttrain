"""CPU-only freeze then fresh N6G3 authorization; GPU execution is explicit."""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import time
import uuid

RT = Path(__file__).resolve().parent
SOURCE = RT / 'curated_clause_salvage_review_pilot5_n6g3_source_prepared_v1.json'
SELECTED = RT / 'curated_clause_salvage_review_pilot5_selected_v1.json'
MANAGER = RT / 'curated_clause_salvage_review_pilot5_resource_manager_v1.py'
POINTER = RT / 'curated_clause_salvage_review_pilot5_n6g3_admission_v1.json'
BASE_SHA = '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
MODEL_SHA = '5607668eb5c476308f20ca20b2853c97bc846f5d43182df2a66729b3c9c4385f'
MODEL_IDENTITY_SHA = '220d9152229696caec4decebb89a160afc8ac9b6ba91ab1b9d790343a82d733a'


def module(path):
    spec = importlib.util.spec_from_file_location('clause_pilot5_build_resource', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def lease_value(s):
    return None if s is None else [s.st_dev, s.st_ino, s.st_uid, s.st_mode, s.st_size, s.st_mtime_ns]


def fresh_capture(manager):
    base, worker = manager.base, manager.worker
    if '28.216.19.225' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('CPU review3 waiter must execute on N6')
    lockfd, pidfd = None, None
    try:
        path = Path('/tmp/ms12k_gpu_3.lock')
        if path.exists() or path.is_symlink():
            lockfd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            s = os.fstat(lockfd)
            if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_size:
                raise worker.NotReady('Unknown GPU3 lock shape/owner/content; untouched')
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = base.lease_snapshot(Path('/tmp/od_reserve_3'))
        gpu_uuid = base.uuid_for(3)
        running = base.apps()
        active = [x for x in running if x['uuid'] == gpu_uuid]
        if len(active) != 1:
            raise worker.NotReady('GPU3 does not contain exactly one project holder; unknown active untouched')
        owner = dict(base.identity(active[0]['pid']), gpu=3, uuid=gpu_uuid)
        base.check_holder(owner, base.identity(owner['pid']), 3, gpu_uuid, running)
        pidfd = os.pidfd_open(owner['pid'])
        if select.select([pidfd], [], [], 0)[0]:
            raise worker.NotReady('GPU3 holder already exited')
        budget = base.headroom()
        if budget['headroom'] < base.required_budget({}):
            raise worker.NotReady('N6 headroom below native review admission budget')
        base.check_holder(owner, base.identity(owner['pid']), 3, base.uuid_for(3), base.apps())
        if lease_value(base.lease_snapshot(Path('/tmp/od_reserve_3'))) != lease_value(lease):
            raise worker.NotReady('GPU3 lease changed during read-only capture')
        return dict(node='N6', gpus=[3], holders=[owner], lease=lease_value(lease), headroom=budget,
                    captured_unix_ns=time.time_ns(), signals_sent=False, gpu_started=False, leases_modified=False)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        if isinstance(error, worker.NotReady):
            raise
        raise worker.NotReady('GPU3 state not safe/available; leave unknown active untouched: ' + str(error)) from error
    finally:
        if pidfd is not None:
            os.close(pidfd)
        if lockfd is not None:
            fcntl.flock(lockfd, fcntl.LOCK_UN)
            os.close(lockfd)



def save_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def all_refs(value):
    refs = []
    def visit(item):
        if isinstance(item, dict):
            if isinstance(item.get('path'), str) and isinstance(item.get('sha256'), str):
                refs.append(dict(path=item['path'], sha256=item['sha256']))
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    return list({(r['path'], r['sha256']): r for r in refs}.values())


def freeze(args):
    if SOURCE.exists() or SELECTED.exists():
        raise RuntimeError('Pilot5 fixed selection/config already exists; do not overwrite')
    m = module(MANAGER)
    w = m.worker
    if w.sha(RT / 'resource_manager.py') != BASE_SHA:
        raise RuntimeError('Frozen base manager changed')
    s = w.support()
    manifest = s.Manifest(w.MANIFEST)
    if manifest.sha256 != w.MANIFEST_SHA:
        raise RuntimeError('Original main manifest SHA changed')
    stream = w.CANDIDATES / 'review_required.jsonl'
    if not stream.exists():
        raise w.NotReady('No real main clause-salvage candidates yet; no placeholder authorization created')
    raw = stream.read_bytes()
    lines = raw.splitlines(keepends=True)
    notices = {}
    for line in lines:
        if not line.endswith(b'\n'):
            break
        n = json.loads(line)
        key = n['source_key']
        if key in notices and notices[key] != n:
            raise RuntimeError('Main candidate notice changed inside fixed edition')
        notices[key] = n
    keys = sorted(notices)
    if args.source_key:
        if not 1 <= len(args.source_key) <= 5 or len(set(args.source_key)) != len(args.source_key):
            raise RuntimeError('One-to-five distinct real candidate source keys required')
        keys = args.source_key
    if not keys or any(k not in notices for k in keys[:5]):
        raise w.NotReady('Waiting requested real main candidates; no placeholder selection/config')
    cases = []
    for key in keys[:5]:
        row = w.bind_notice(notices[key], manifest, s)
        references = all_refs(row)
        for r in references:
            if w.sha(r['path']) != r['sha256']:
                raise RuntimeError('Main candidate source SHA changed before freeze')
        cases.append(dict(source_key=key, notice=notices[key],
            label_result_ref=row['review_provenance']['label_result_ref'],
            input_row_sha256=w.canonical_sha(row), frozen_refs=references,
            review_required=row['review_provenance']['review_delta']['review_required'],
            review_reason=row['review_provenance']['review_delta']['reason']))
    required_count = sum(c['review_required'] for c in cases)
    if not required_count:
        raise w.NotReady('No new supported body facts or pure rewrite failure; no VLM authorization created')
    chosen = dict(schema=w.PHASE, global_revision=w.EDITION,
        total_source_count=222939, expected_batch_count=448, selected_source_count=required_count, candidate_source_count=len(cases),
        original_manifest_ref=dict(path=str(w.MANIFEST), sha256=w.MANIFEST_SHA),
        input_stream_path=str(stream), stream_prefix_sha256=hashlib.sha256(raw).hexdigest(),
        cases=cases, selection_rule='explicit_max5_changed_only' if args.source_key else 'sorted_real_main_max5_changed_only',
        fixed_pilot_case_not_population_sample=True, placeholder_authorization=False)
    model = RT / 'reviewer8b_stage_N6_extension_v2.json'
    stage = w.load(model)
    if (w.sha(model) != MODEL_SHA or stage.get('complete') is not True
            or stage.get('full_copy_SHA_verified') is not True
            or stage['model_identity']['sha256'] != MODEL_IDENTITY_SHA
            or stage['model_identity']['root'] != '/opt/models/Qwen3-VL-8B-Instruct-review-extension-v2'):
        raise RuntimeError('Exact completed N6 staged 8B model identity required')
    save_new(SELECTED, chosen)
    rows, references = w.ready_records(SELECTED, w.sha(SELECTED))
    files = list((RT / 'source_curated_review_v2').rglob('*.py'))
    files += [Path(__file__), MANAGER, RT / 'curated_clause_salvage_review_pilot5_v1.py',
        RT / 'resource_manager.py', RT / 'curated_review_extension_resource_manager_v1.py',
        w.SUPPORT, w.MANIFEST, model,
        RT.parent.parent / '32b_replica_stage_receipts/source_reference_v1.json']
    bindings = {str(p.resolve()): w.sha(p) for p in files}
    bindings.update({r['path']: r['sha256'] for r in references})
    for bid in {r['batch_id'] for r in rows}:
        b = manifest.descriptors[bid]
        bindings[b['path']] = b['sha256']
    cfg = dict(schema=w.PHASE, preparation_status='real_fixed5_source_prepared_not_launch_admitted',
        node='N6', gpus=[3], holders=[], holder_capture=None, release_authorized=False,
        phase=w.PHASE, queue=str(m.QUEUE), out=str(w.OUT), cwd=str(RT / 'source_curated_review_v2'),
        producer=w.ref(RT / 'curated_clause_salvage_review_pilot5_v1.py'), plan=w.ref(SELECTED),
        input_count=222939, expected_batch_count=448, targetcount=required_count, candidatecount=len(cases),
        source_selection=dict(shard=0, nshard=1, limit=required_count),
        modelref=dict(path=str(model), sha256=MODEL_SHA, model_identity_sha256=MODEL_IDENTITY_SHA),
        source_sha256=bindings, resource_manager_ref=w.ref(MANAGER), base_manager_sha256=BASE_SHA,
        reuse_existing_results=False, semantic_approval=False, ready_source_refs=references,
        env=dict(CUDA_VISIBLE_DEVICES='3', OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            MS_VISUAL_CPU_SITEPATH='/opt/venv_eval/lib/python3.11/site-packages'),
        argv=['/opt/venv_vllm/bin/python', '-B', '-u', str(RT / 'curated_clause_salvage_review_pilot5_v1.py'),
            '--out', str(w.OUT), '--selection', str(SELECTED), '--selection-sha', w.sha(SELECTED),
            '--model', stage['model_identity']['root'], '--model-reference', str(model),
            '--annotator-model-reference', str(RT.parent.parent / '32b_replica_stage_receipts/source_reference_v1.json'),
            '--ms-source', str(RT / 'source_curated_review_v2')])
    save_new(SOURCE, cfg)
    print(json.dumps(dict(source_prepared=w.ref(SOURCE), selection=w.ref(SELECTED),
        manager=w.ref(MANAGER), builder=w.ref(__file__), frozen_source_keys=[c['source_key'] for c in cases],
        GPU_started=False, release_authorized=False), ensure_ascii=False), flush=True)


def source_config(args):
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if not args.source_prepared_sha or not args.self_sha or not args.manager_sha:
        raise RuntimeError('Explicit source-prepared/builder/manager SHA arguments required')
    if digest(__file__) != args.self_sha or digest(SOURCE) != args.source_prepared_sha:
        raise RuntimeError('Frozen builder/source-prepared SHA changed')
    cfg = json.loads(SOURCE.read_bytes())
    if (cfg['node'] != 'N6' or cfg['gpus'] != [3] or cfg['holders'] != []
            or cfg['release_authorized'] is not False or cfg['holder_capture'] is not None
            or not 1 <= cfg['targetcount'] <= cfg['candidatecount'] <= 5 or cfg['resource_manager_ref'] != dict(path=str(MANAGER), sha256=args.manager_sha)):
        raise RuntimeError('Only real unadmitted exact pilot5 source-prepared config allowed')
    for path, sha in cfg['source_sha256'].items():
        if digest(path) != sha:
            raise RuntimeError('Frozen pilot5 source/label changed: ' + path)
    if digest(MANAGER) != args.manager_sha or digest(RT / 'resource_manager.py') != BASE_SHA:
        raise RuntimeError('Frozen pilot5 manager/base changed')
    m = module(MANAGER)
    m.selected(cfg)
    m.support.check_model(cfg)
    return cfg, m


def authorize(args):
    cfg, m = source_config(args)
    _, references = m.worker.ready_records(cfg['plan']['path'], cfg['plan']['sha256'])
    capture = fresh_capture(m)
    if args.readonly_now:
        print(json.dumps(dict(ready=True, node='N6', gpus=[3], selected_manifest_ref=cfg['plan'],
            selected_source_count=cfg['targetcount'], candidate_source_count=cfg['candidatecount'], holder_pid=capture['holders'][0]['pid'],
            signals_sent=False, GPU_started=False, leases_modified=False), ensure_ascii=False), flush=True)
        return
    if m.QUEUE.exists() or m.worker.OUT.exists() or POINTER.exists():
        raise RuntimeError('Pilot5 queue/output/admission already exists; do not start twice')
    fresh = fresh_capture(m)
    if (any(fresh['holders'][0][k] != capture['holders'][0][k] for k in m.base.IDENTITY_KEYS + ('gpu', 'uuid'))
            or fresh['lease'] != capture['lease']):
        raise m.worker.NotReady('GPU3 holder/lease changed before authorization; no signals')
    token = uuid.uuid4().hex
    captured = RT / ('curated_clause_salvage_review_pilot5_n6g3_fresh_' + token + '.json')
    launch = RT / ('curated_clause_salvage_review_pilot5_n6g3_launch_' + token + '.json')
    save_new(captured, fresh)
    active = copy.deepcopy(cfg)
    active.update(holders=fresh['holders'], holder_capture=m.worker.ref(captured), release_authorized=True,
        ready_source_refs=references, source_prepared_ref=m.worker.ref(SOURCE),
        preparation_status='fresh_N6G3_exact_real_changed_1to5')
    for r in [m.worker.ref(captured), m.worker.ref(SOURCE), m.worker.ref(__file__)]:
        active['source_sha256'][r['path']] = r['sha256']
    save_new(launch, active)
    m.validate(active)
    m.base.check_sources(active, launch, m.base.sha(launch), BASE_SHA)
    m.support.check_model(active)
    save_new(POINTER, dict(config=m.worker.ref(launch), capture=m.worker.ref(captured),
        manager=m.worker.ref(MANAGER), source_prepared_ref=m.worker.ref(SOURCE),
        selected_manifest_ref=cfg['plan'], node='N6', gpus=[3], signals_sent_by_builder=False,
        GPU_started_by_builder=False))
    argv = ['/opt/venv_eval/bin/python', '-B', '-u', str(MANAGER), '--config', str(launch),
        '--config-sha', m.base.sha(launch), '--manager-sha', BASE_SHA]
    print(json.dumps(dict(config=m.worker.ref(launch), argv=argv, GPU_started=False), ensure_ascii=False), flush=True)
    if args.authorize_and_run:
        os.execv(argv[0], argv)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze-main-first5', action='store_true')
    g.add_argument('--readonly-now', action='store_true')
    g.add_argument('--authorize-only', action='store_true')
    g.add_argument('--authorize-and-run', action='store_true')
    p.add_argument('--source-key', action='append')
    p.add_argument('--source-prepared-sha')
    p.add_argument('--manager-sha')
    p.add_argument('--self-sha')
    args = p.parse_args()
    if args.freeze_main_first5:
        freeze(args)
    else:
        authorize(args)


if __name__ == '__main__':
    main()
