"""CPU-only two-node probe of the exact frozen append implementation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

RT = Path(__file__).resolve().parent
ROOT = RT / 'curated_collaborative30000_ceph_probe_v1'
BASE = RT / 'source_curated_producer_v1/curated_pipeline.py'
SHA = '800efe36d410bcd951a07deb5b18e1588b34765fcb0aa0c728cb9c6921f3697e'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--writer', choices=['N3', 'N5'])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != SHA:
        raise RuntimeError('Frozen producer changed')
    spec = importlib.util.spec_from_file_location('probe_native', BASE)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    ROOT.mkdir(exist_ok=True)
    stream = ROOT / 'two_node_ready_probe.jsonl'
    if args.writer:
        with (ROOT / (args.writer + '.started')).open('x') as marker:
            marker.write(str(time.time_ns()))
        deadline = time.monotonic() + 60
        while not all((ROOT / (node + '.started')).exists() for node in ['N3', 'N5']):
            if time.monotonic() > deadline:
                raise RuntimeError('Other writer missing; no probe writes')
            time.sleep(.05)
        for i in range(64):
            pipeline.append(stream, {'node': args.writer, 'sequence': i, 'payload': args.writer * (128 * 1024),
                                     'formal_training': False})
        with (ROOT / (args.writer + '.completed')).open('x') as marker:
            marker.write(str(time.time_ns()))
        print(json.dumps({'writer': args.writer, 'records': 64, 'gpu_started': False}), flush=True)
    elif args.check:
        if not all((ROOT / (node + '.completed')).exists() for node in ['N3', 'N5']):
            raise RuntimeError('Probe writers not both completed')
        raw = stream.read_bytes()
        if not raw.endswith(b'\n'):
            raise RuntimeError('Incomplete probe tail')
        rows = [json.loads(line) for line in raw.splitlines()]
        expected = {(n, i) for n in ['N3', 'N5'] for i in range(64)}
        if (len(rows) != 128 or {(r['node'], r['sequence']) for r in rows} != expected
                or any(r['payload'] != r['node'] * (128 * 1024) for r in rows)):
            raise RuntimeError('Malformed/interleaved/duplicate native Ceph append probe')
        result = {'status': 'exact_native_append_probe_passed', 'records': len(rows), 'bytes': len(raw),
                  'stream': {'path': str(stream), 'sha256': hashlib.sha256(raw).hexdigest()},
                  'append_source': {'path': str(BASE), 'sha256': SHA}, 'nodes': ['N3', 'N5'],
                  'large_payload_bytes': 262144, 'general_concurrency_proof': False,
                  'boundary': 'Frozen source lock plus stream flock+flush+fsync; whole ROI directory not atomic',
                  'gpu_started': False, 'formal_training': False}
        with (ROOT / 'receipt.json').open('x') as marker:
            json.dump(result, marker, indent=2)
        print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
