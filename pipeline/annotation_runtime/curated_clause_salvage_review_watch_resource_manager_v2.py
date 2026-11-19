"""Fresh N6 GPU3 only; frozen base pidfd/heartbeat/restoration, natural audit yield."""
import importlib.util
import json
import re
from pathlib import Path
import subprocess
import time

RT = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


support = module('salvage_watch_resource_support', RT / 'curated_review_extension_resource_manager_v1.py')
base = support.base
worker = module('salvage_watch_resource_worker', RT / 'curated_clause_salvage_review_watch_strict_resume_v2.py')
PHASE = worker.PHASE
QUEUE_PREFIX = 'curated_clause_salvage_review_watch_n6g3_v2-'


def selected(cfg):
    # Dynamic arrival stream is not a fake finite zero-source population.
    return [], 222939


def check_sources(cfg, config_path, config_sha, manager_sha):
    if base.sha(base.__file__) != manager_sha or base.sha(config_path) != config_sha:
        raise RuntimeError('Frozen base manager/config changed')
    bindings = dict(cfg['source_sha256'])
    for kind in ('producer', 'plan'):
        reference = cfg[kind]
        if bindings.get(reference['path'], reference['sha256']) != reference['sha256']:
            raise RuntimeError('Conflicting frozen dependency reference')
        bindings[reference['path']] = reference['sha256']
    for path, expected in bindings.items():
        if not Path(path).is_absolute() or base.sha(path) != expected:
            raise RuntimeError('Frozen watch source changed: ' + path)
    plan = worker.checked(cfg['plan'])
    proof = worker.checked(plan['resume_state_ref'])
    if cfg.get('resume_state_ref') != plan['resume_state_ref']:
        raise RuntimeError('V2 resume proof not explicitly fixed in config/launch')
    oldstate = worker.checked(proof['original_state_ref'])
    cursor = worker.AppendCursor(worker.STREAM, dict(oldstate['cursor']))
    cursor.next()
    with worker.STREAM.open('rb') as stream:
        prefix = stream.read(proof['consumed_prefix_bytes'])
    import hashlib
    if (hashlib.sha256(prefix).hexdigest() != proof['consumed_prefix_sha256']
            or oldstate['cursor'] != proof['original_cursor']
            or len(oldstate['sources']) != proof['processed_source_count']
            or proof.get('state_migrated') is not False or proof.get('model_calls_performed') != 0):
        raise RuntimeError('Frozen old state/prefix/source count resume proof changed')
    if (plan['schema'] != PHASE or plan['total_source_count'] != 222939
            or plan['expected_batch_count'] != 448 or plan['input_stream'] != str(worker.STREAM)
            or cfg['input_count'] != 222939 or cfg['expected_batch_count'] != 448
            or cfg['targetcount'] != 0 or cfg.get('dynamic_source_watch') is not True):
        raise RuntimeError('Dynamic stream/full222939/448 scope changed')


def validate(cfg):
    if (cfg['node'] != 'N6' or cfg['gpus'] != [3] or cfg.get('release_authorized') is not True
            or cfg['phase'] != PHASE or cfg.get('semantic_approval') is not False):
        raise RuntimeError('Explicit N6 physical GPU3 release authorization required')
    if '28.216.19.225' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('Wrong physical node')
    queue = Path(cfg['queue'])
    if (queue.parent != RT.parent / 'queues' or not queue.name.startswith(QUEUE_PREFIX)
            or not re.fullmatch(r'[0-9]+-[0-9a-f]{12}', queue.name[len(QUEUE_PREFIX):]) or queue.exists()
            or cfg['out'] != str(worker.OUT) or cfg['env'].get('CUDA_VISIBLE_DEVICES') != '3'
            or cfg.get('reuse_existing_results') is not True):
        raise RuntimeError('Fresh independent queue/out/GPU3 state resume required')
    if len(cfg['holders']) != 1 or cfg['holders'][0]['gpu'] != 3:
        raise RuntimeError('Exactly one GPU3 project holder required')
    required = {'--out': worker.OUT, '--manifest': worker.pilot.MANIFEST,
        '--launch': cfg['plan']['path'], '--launch-sha': cfg['plan']['sha256'],
        '--model': worker.MODEL, '--model-reference': worker.MODEL_REF,
        '--annotator-model-reference': RT.parent.parent / '32b_replica_stage_receipts/source_reference_v1.json',
        '--ms-source': RT / 'source_curated_review_v2'}
    for key, expected in required.items():
        if base.argv_option(cfg, key) != str(expected):
            raise RuntimeError('Watch worker path/model differs: ' + key)
    if (cfg['producer']['path'] != str(RT / 'curated_clause_salvage_review_watch_strict_resume_v2.py')
            or cfg['producer']['path'] not in cfg['argv'] or '--watch' not in cfg['argv']
            or cfg['cwd'] != str(RT / 'source_curated_review_v2')
            or cfg['source_selection'] != dict(shard=0, nshard=1, limit=0)):
        raise RuntimeError('One resident dynamic watch runtime required')
    snap = worker.checked(cfg['holder_capture'])
    age = time.time_ns() - snap['captured_unix_ns']
    if (snap['node'] != 'N6' or snap['gpus'] != [3] or snap['holders'] != cfg['holders']
            or snap['signals_sent'] is not False or not 0 <= age <= 1200 * 10**9):
        raise RuntimeError('Capture stale/outside resource scope')
    lease = base.lease_snapshot(Path('/tmp/od_reserve_3'))
    current = None if lease is None else [lease.st_dev, lease.st_ino, lease.st_uid,
        lease.st_mode, lease.st_size, lease.st_mtime_ns]
    if current != snap['lease']:
        raise RuntimeError('GPU3 lease changed since fresh capture')
    holder = cfg['holders'][0]
    base.check_holder(holder, base.identity(holder['pid']), 3, base.uuid_for(3), base.apps())
    if base.required_budget(cfg) != 160 * 2**30:
        raise RuntimeError('Frozen160GiB safety/headroom budget required')


def check_model(cfg):
    if cfg['modelref']['path'] != str(worker.MODEL_REF) or cfg['modelref']['sha256'] != worker.MODEL_REF_SHA:
        raise RuntimeError('Exact staged8B receipt required')
    support.check_model(cfg)
    identity = worker.load(worker.MODEL_REF)['model_identity']
    # Actual complete disk SHA inventory, not merely stage flags or config/index hashes.
    for name, entry in identity['files'].items():
        if base.sha(Path(identity['root']) / name) != entry['sha256']:
            raise RuntimeError('Actual8B checkpoint content changed: ' + name)


def summaries(cfg, out):
    path = out / 'run.summary.json'
    if not path.exists():
        return {}
    value = worker.load(path)
    if (value.get('schema') != PHASE or value['global_revision'] != worker.EDITION
            or value['launch_ref'] != cfg['plan'] or value['total_source_count'] != 222939
            or value['expected_batch_count'] != 448
            or any(value.get(k) is not False for k in ('training_eligible', 'semantic_approval',
                'human_review_completed', 'allreviewed', 'all_batches_reviewed', 'original_batch_review_complete'))):
        raise RuntimeError('Watch summary claims wrong scope or semantic approval')
    yielded, drained = value['reserved_recovery_yield'], value['input_drain']
    if yielded:
        evidence = value.get('recovery_yield_evidence') or {}
        if (drained or value['source_records_consumed'] is not False
                or evidence.get('reason') != 'reserved_recovery_yield'
                or evidence.get('recovery_admission_absent') is not True
                or evidence.get('reserved_queue') != str(worker.RECOVERY_QUEUE)
                or evidence.get('other_tasks_signaled') is not False
                or len(evidence.get('recovery_source_ids', [])) != 3):
            raise RuntimeError('Natural yield lacks reserved recovery evidence')
        rows, ready_refs = worker.recovery.ready_records()
        if (ready_refs != evidence['recovery_ready_refs']
                or [r['id'] for r in rows] != evidence['recovery_source_ids']
                or (worker.RECOVERY_QUEUE / 'admission.json').exists()):
            raise RuntimeError('Natural yield no longer matches actual reserved recovery ready gate')
        for reference in evidence['recovery_ready_refs']:
            if base.sha(reference['path']) != reference['sha256']:
                raise RuntimeError('Recovery ready reference changed')
    elif not drained or value.get('input_drain_scope') != 'current_complete_append_stream_snapshot_only':
        raise RuntimeError('Only evidenced recovery yield or explicit current-input drain is stage-complete')
    if value.get('resume_state_ref') != cfg.get('resume_state_ref') or value.get('persistent_state_schema') != worker.LEGACY_PHASE:
        raise RuntimeError('V2 summary lacks original state proof/schema binding')
    state = worker.checked(value['state_ref'])
    if len(state['sources']) != value['processed_source_count']:
        raise RuntimeError('Incremental source count differs')
    for key, source in state['sources'].items():
        transaction = worker.checked(source['transaction_ref'])
        summary = worker.checked(transaction['summary_ref'])
        if (transaction['identity']['source_key'] != key
                or base.sha(transaction['ledger_ref']['path']) != transaction['ledger_ref']['sha256']
                or summary.get('training_eligible') is not False):
            raise RuntimeError('Source write-once ledger/summary changed')
    return {str(path): value}


base.AUTHORIZED_PAIRS = {'N6': ((3,),)}
base.validate_pair = validate
base.selected_sources = selected
base.check_sources = check_sources
base.check_model = check_model
base.stage_summaries = summaries
if __name__ == '__main__':
    base.main()
