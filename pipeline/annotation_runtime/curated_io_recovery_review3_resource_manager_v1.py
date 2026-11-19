"""Only N6 physical GPU3 for exact 3-source independent recovery review."""
import importlib.util
import json
from pathlib import Path
import subprocess
import time

RT = Path(__file__).resolve().parent


def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


support = module('recovery3_model_resource', RT / 'curated_review_extension_resource_manager_v1.py')
base = support.base
worker = module('recovery3_worker', RT / 'curated_io_recovery_review3_v1.py')
PHASE = 'curated_io_recovery_review3_v1'
QUEUE = RT.parent / 'queues/curated_io_recovery_review3_n6g3_v1'


def selected(cfg):
    if cfg['plan'] != dict(path=str(worker.SELECTED), sha256=worker.SELECTED_SHA):
        raise RuntimeError('Exact recovery3 selection SHA differs')
    return [r['id'] for r, _ in worker.source_rows()], 222939


def validate(cfg):
    if cfg['node'] != 'N6' or cfg['gpus'] != [3] or cfg.get('release_authorized') is not True:
        raise RuntimeError('Explicit N6 physical GPU3 release authorization required')
    if '28.216.19.225' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('Recovery3 review executing on wrong node')
    if cfg['phase'] != PHASE or cfg['input_count'] != 222939 or cfg['targetcount'] != 3:
        raise RuntimeError('Exact selected3/full222939 scope required')
    if cfg['queue'] != str(QUEUE) or cfg['out'] != str(worker.OUT) or cfg['env'].get('CUDA_VISIBLE_DEVICES') != '3':
        raise RuntimeError('Only independent recovery3 output/queue/GPU3 allowed')
    if cfg['source_selection'] != dict(shard=0, nshard=1, limit=3):
        raise RuntimeError('Exact finite three source selection required')
    if len(cfg['holders']) != 1 or cfg['holders'][0]['gpu'] != 3:
        raise RuntimeError('Exactly one GPU3 holder required')
    required = {'--out': worker.OUT, '--model': Path('/opt/models/Qwen3-VL-8B-Instruct-review-extension-v2'),
        '--model-reference': RT / 'reviewer8b_stage_N6_extension_v2.json',
        '--annotator-model-reference': RT.parent.parent / '32b_replica_stage_receipts/source_reference_v1.json',
        '--ms-source': RT / 'source_curated_review_v2'}
    for key, value in required.items():
        if base.argv_option(cfg, key) != str(value):
            raise RuntimeError('Recovery3 worker path/model differs: ' + key)
    if cfg['producer']['path'] != str(RT / 'curated_io_recovery_review3_v1.py') or cfg['producer']['path'] not in cfg['argv']:
        raise RuntimeError('Frozen finite recovery3 reviewer required')
    capture = cfg['holder_capture']
    if base.sha(capture['path']) != capture['sha256']:
        raise RuntimeError('GPU3 fresh capture changed')
    snap = json.loads(Path(capture['path']).read_bytes())
    age = time.time_ns() - snap['captured_unix_ns']
    if (snap['node'] != 'N6' or snap['gpus'] != [3] or snap['holders'] != cfg['holders']
            or snap['signals_sent'] is not False or not 0 <= age <= 1200 * 10**9):
        raise RuntimeError('GPU3 capture stale/outside exact resource scope')
    lease = base.lease_snapshot(Path('/tmp/od_reserve_3'))
    current = None if lease is None else [lease.st_dev, lease.st_ino, lease.st_uid, lease.st_mode, lease.st_size, lease.st_mtime_ns]
    if current != snap['lease']:
        raise RuntimeError('GPU3 lease differs from fresh capture')
    rows, refs = worker.ready_records()
    if refs != cfg['ready_source_refs']:
        raise RuntimeError('All three recovery terminal/oldfailed/input references changed before admission')
    selected(cfg)


def summaries(cfg, out):
    path = out / 'run.summary.json'
    if not path.exists():
        return {}
    value = json.loads(path.read_bytes())
    ids, _ = selected(cfg)
    if (value.get('schema') != PHASE or value['selected_source_ids'] != ids
            or value['processed_source_count'] != 3 or value['selected_terminal_count'] != 3
            or value['total_source_count'] != 222939 or value['original_batch_source_count'] != 500
            or value['source_records_consumed'] is not True or value['training_eligible'] is not False
            or value['semantic_approval'] is not False or value['human_review_completed'] is not False
            or value['original_batch_review_complete'] is not False):
        raise RuntimeError('Recovery3 summary claims changed scope or semantic approval')
    for reference in value['review_record_refs']:
        if base.sha(reference['path']) != reference['sha256']:
            raise RuntimeError('Recovery3 review ledger changed')
    return {str(path): value}


base.AUTHORIZED_PAIRS = {'N6': ((3,),)}
base.validate_pair = validate
base.selected_sources = selected
base.check_model = support.check_model
base.stage_summaries = summaries
if __name__ == '__main__':
    base.main()
