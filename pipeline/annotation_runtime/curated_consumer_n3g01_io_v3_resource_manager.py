"""N3 G0/G1 only; preserve verified pidfd/lease/restore transaction."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import subprocess

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('curated_resource_original_v1', RT / 'curated_resource_manager.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
spec = importlib.util.spec_from_file_location('curated_consumer_n3g01_io_v3', RT / 'curated_consumer_n3g01_io_v3.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
c.ALLOWED = {'consumer': {'N3': [(0, 1)]}}
c.base.AUTHORIZED_PAIRS = {'N3': ((0, 1),)}
original_validate = c.validate


def selected_rows(cfg):
    selection = {key: int(c.base.argv_option(cfg, '--' + key, default))
                 for key, default in (('shard', 0), ('nshard', 1), ('limit', 0))}
    if selection != {'shard': 0, 'nshard': 1, 'limit': 0} or cfg['source_selection'] != selection:
        raise RuntimeError('Finite collaborative source selection changed')
    reference = cfg['selected_sources']
    if (c.base.argv_option(cfg, '--selected-sources') != reference['path']
            or c.base.sha(reference['path']) != reference['sha256']
            or cfg['source_range'] != {'start': 1000, 'end': 2000}
            or cfg['input_count'] != 222939 or cfg['targetcount'] != 1000):
        raise RuntimeError('Explicit 1000:2000 selection binding changed')
    rows = worker.selected_rows(worker.load_pipeline(), cfg['manifest_summary']['path'])
    if json.loads(Path(reference['path']).read_text()) != rows:
        raise RuntimeError('Explicit selected paths differ from original full curated batches')
    return rows


def selected(cfg):
    return [row['id'] for row in selected_rows(cfg)], 222939


def validate(cfg):
    original_validate(cfg)
    if cfg['node'] != 'N3' or cfg['mode'] != 'consumer' or cfg['gpus'] != [0, 1]:
        raise RuntimeError('Only N3 G0/G1 collaborative consumer admitted')
    rows = selected_rows(cfg)
    wanted = {row['id'] for row in rows}
    if Path(cfg['private_ready']) != Path(cfg['out']) / worker.PRIVATE_READY:
        raise RuntimeError('Isolated ready queue binding changed')
    capture = cfg['holder_capture']
    if c.base.sha(capture['path']) != capture['sha256']:
        raise RuntimeError('Fresh holder capture changed')
    captured = json.loads(Path(capture['path']).read_text())
    if (captured['holders'] != cfg['holders'] or captured['node'] != 'N3'
            or captured['gpus'] != [0, 1] or captured['mode'] != 'consumer'
            or captured['signals_sent'] is not False
            or not 0 <= time.time() - captured['captured_unix'] <= 1200):
        raise RuntimeError('Holder capture stale or outside exact authorization')
    for gpu, expected in zip(cfg['gpus'], captured['leases']):
        lease = c.base.lease_snapshot(Path('/tmp/od_reserve_%d' % gpu))
        current = None if lease is None else [lease.st_dev, lease.st_ino, lease.st_uid, lease.st_mode, lease.st_size, lease.st_mtime_ns]
        if current != expected:
            raise RuntimeError('Exact captured GPU lease changed before launch')
    # No old worker may receive a selected hint before its consumer terminal exists.
    queue = Path(cfg['out']) / 'ready.jsonl'
    if queue.exists():
        with queue.open() as stream:
            for line in stream:
                if not line.endswith('\n'):
                    break
                hint = json.loads(line)
                if hint['id'] in wanted and not (Path(cfg['out']) / 'sources' / hint['split'] / hint['id'] / 'consumer.terminal.json').exists():
                    raise RuntimeError('Selected source already public without terminal; unsafe old-worker race')


def summaries(cfg, out):
    rows = selected_rows(cfg)
    results = {}
    for row in rows:
        path = out / 'sources' / row['split'] / row['id'] / 'consumer.terminal.json'
        value = json.loads(path.read_text())
        if (value['id'] != row['id'] or value['split'] != row['split']
                or value['batch_id'] != row['batch_id']
                or value.get('params_sha256') != cfg['consumer_params_sha256']
                or value['status'] not in ('joint_candidate', 'incomplete_candidate', 'failed', 'producer_failed')):
            raise RuntimeError('Selected consumer terminal binding/status changed')
        if value.get('joint_candidate_ref'):
            artifact = value['joint_candidate_ref']
            if c.base.sha(artifact['path']) != artifact['sha256']:
                raise RuntimeError('Selected consumer output changed')
        results[row['id']] = value
    markers = []
    for path in (out / 'runs').glob('v0-*/completed.json'):
        value = json.loads(path.read_text())
        protocol = json.loads(path.with_name('protocol.json').read_text())
        if (value.get('selected_total') == 1000 and value.get('queue_sources_consumed') == 1000
                and protocol.get('consumer_wrapper') == cfg['producer']):
            markers.append(path)
    if len(results) != 1000 or not markers:
        raise RuntimeError('Finite collaborative completion receipt missing')
    return {str(markers[-1]): {'summary': {'stage': 'run', 'total_source_count': 222939,
            'selected_source_count': 1000, 'processed_source_count': 1000,
            'plan_sha256': cfg['plan']['sha256']},
            'selection': {'selected_ids': [row['id'] for row in rows]},
            'source_failures_preserved': True, 'semantic_approval': False}}


c.base.validate_pair = validate
c.base.selected_sources = selected
c.base.stage_summaries = summaries

if __name__ == '__main__':
    if '--capture-current' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--capture-current', action='store_true')
        parser.add_argument('--node', required=True)
        parser.add_argument('--gpus', nargs='+', type=int, required=True)
        parser.add_argument('--out', required=True)
        parser.add_argument('--mode', required=True)
        args = parser.parse_args()
        if args.node != 'N3' or args.gpus != [0, 1] or args.mode != 'consumer':
            raise RuntimeError('Capture restricted to N3 G0/G1 consumer')
        if c.HOSTS['N3'] not in subprocess.check_output(['hostname', '-I'], text=True).split():
            raise RuntimeError('Capture host mismatch')
        owners, leases = [], []
        for gpu in args.gpus:
            uuid = c.base.uuid_for(gpu)
            apps = c.base.apps()
            active = [app for app in apps if app['uuid'] == uuid]
            if len(active) != 1:
                raise RuntimeError('Exactly one authorized holder required')
            owner = c.base.identity(active[0]['pid'])
            owner.update(gpu=gpu, uuid=uuid)
            c.base.check_holder(owner, c.base.identity(owner['pid']), gpu, uuid, c.base.apps())
            lease = c.base.lease_snapshot(Path('/tmp/od_reserve_%d' % gpu))
            leases.append(None if lease is None else [lease.st_dev, lease.st_ino, lease.st_uid, lease.st_mode, lease.st_size, lease.st_mtime_ns])
            owners.append(owner)
        value = {'node': args.node, 'mode': args.mode, 'gpus': args.gpus, 'holders': owners,
                 'leases': leases, 'captured_unix': time.time(), 'headroom': c.base.headroom(), 'signals_sent': False}
        with Path(args.out).open('x') as stream:
            json.dump(value, stream, indent=2)
        print(json.dumps({'path': args.out, 'sha256': c.base.sha(args.out), 'signals_sent': False}), flush=True)
    else:
        c.base.main()
