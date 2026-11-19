"""Only N6 physical GPU3; exact frozen pilot5 and base pidfd/lease restoration."""
import importlib.util
import json
from pathlib import Path
import subprocess
import time

RT = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


support = module('clause_pilot5_model_resource', RT / 'curated_review_extension_resource_manager_v1.py')
base = support.base
worker = module('clause_pilot5_worker', RT / 'curated_clause_salvage_review_pilot5_v1.py')
PHASE = worker.PHASE
QUEUE = RT.parent / 'queues/curated_clause_salvage_review_pilot5_n6g3_v1'


def selected(cfg):
    rows, _ = worker.ready_records(cfg['plan']['path'], cfg['plan']['sha256'])
    return [r['id'] for r in rows], 222939


def validate(cfg):
    if cfg['node'] != 'N6' or cfg['gpus'] != [3] or cfg.get('release_authorized') is not True:
        raise RuntimeError('Explicit N6 physical GPU3 release authorization required')
    if '28.216.19.225' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('Pilot5 review executing on wrong node')
    if (cfg['phase'] != PHASE or cfg['input_count'] != 222939 or not 1 <= cfg['targetcount'] <= cfg['candidatecount'] <= 5
            or cfg['expected_batch_count'] != 448 or cfg['semantic_approval'] is not False):
        raise RuntimeError('Exact finite1..5 changed/full222939/448 nonapproval scope required')
    if (cfg['queue'] != str(QUEUE) or cfg['out'] != str(worker.OUT)
            or cfg['env'].get('CUDA_VISIBLE_DEVICES') != '3'
            or cfg['source_selection'] != dict(shard=0, nshard=1, limit=cfg['targetcount'])):
        raise RuntimeError('Only independent pilot5 output/queue/GPU3 allowed')
    if len(cfg['holders']) != 1 or cfg['holders'][0]['gpu'] != 3:
        raise RuntimeError('Exactly one GPU3 holder required')
    required = {'--out': worker.OUT, '--selection': cfg['plan']['path'],
        '--selection-sha': cfg['plan']['sha256'],
        '--model': '/opt/models/Qwen3-VL-8B-Instruct-review-extension-v2',
        '--model-reference': RT / 'reviewer8b_stage_N6_extension_v2.json',
        '--annotator-model-reference': RT.parent.parent / '32b_replica_stage_receipts/source_reference_v1.json',
        '--ms-source': RT / 'source_curated_review_v2'}
    for key, value in required.items():
        if base.argv_option(cfg, key) != str(value):
            raise RuntimeError('Pilot5 worker path/model/selection differs: ' + key)
    if (cfg['producer']['path'] != str(RT / 'curated_clause_salvage_review_pilot5_v1.py')
            or cfg['producer']['path'] not in cfg['argv']
            or cfg['cwd'] != str(RT / 'source_curated_review_v2')
            or cfg.get('reuse_existing_results') is not False):
        raise RuntimeError('Frozen unique finite reviewer required; old review reuse forbidden')
    capture = cfg['holder_capture']
    if base.sha(capture['path']) != capture['sha256']:
        raise RuntimeError('GPU3 fresh capture changed')
    snap = json.loads(Path(capture['path']).read_bytes())
    age = time.time_ns() - snap['captured_unix_ns']
    if (snap['node'] != 'N6' or snap['gpus'] != [3] or snap['holders'] != cfg['holders']
            or snap['signals_sent'] is not False or not 0 <= age <= 1200 * 10**9):
        raise RuntimeError('GPU3 capture stale/outside exact resource scope')
    lease = base.lease_snapshot(Path('/tmp/od_reserve_3'))
    current = None if lease is None else [lease.st_dev, lease.st_ino, lease.st_uid,
                                       lease.st_mode, lease.st_size, lease.st_mtime_ns]
    if current != snap['lease']:
        raise RuntimeError('GPU3 lease differs from fresh capture')
    _, refs = worker.ready_records(cfg['plan']['path'], cfg['plan']['sha256'])
    if refs != cfg['ready_source_refs']:
        raise RuntimeError('Frozen selected main candidate label/SHA lineage changed before admission')
    selected(cfg)


def summaries(cfg, out):
    path = out / 'run.summary.json'
    if not path.exists():
        return {}
    value = json.loads(path.read_bytes())
    ids, _ = selected(cfg)
    if (value.get('schema') != PHASE or value['global_revision'] != worker.EDITION
            or value['selected_source_ids'] != ids or value['processed_source_count'] != cfg['targetcount']
            or value['selected_source_count'] != cfg['targetcount'] or value['candidate_source_count'] != cfg['candidatecount'] or value['total_source_count'] != 222939
            or value['expected_batch_count'] != 448 or value['source_records_consumed'] is not True
            or value['selected_manifest_ref'] != cfg['plan']
            or any(value.get(k) is not False for k in ('training_eligible', 'semantic_approval',
                'human_review_completed', 'original_batch_review_complete', 'all_batches_reviewed'))
            or value['fixed_pilot_case_not_population_sample'] is not True
            or value['original_review_scores_reused'] is not False):
        raise RuntimeError('Pilot5 summary claims changed scope or semantic approval')
    for reference in value['review_record_refs']:
        if base.sha(reference['path']) != reference['sha256']:
            raise RuntimeError('Pilot5 review ledger changed')
    return {str(path): value}


base.AUTHORIZED_PAIRS = {'N6': ((3,),)}
base.validate_pair = validate
base.selected_sources = selected
base.check_model = support.check_model
base.stage_summaries = summaries
if __name__ == '__main__':
    base.main()
