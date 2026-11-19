"""Three existing ROI inputs, serial/parallel durable publication, zero model calls."""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
sys.path.insert(0, str(RT))
import curated_producer_io_v4 as serial
import caption_roi_publish_parallel_v1 as parallel


def load(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    with Path(path).open('xb') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False).encode() + bytes([10]))


def stage(original, local, roi):
    prepared = load(original / 'source_manifest.json')
    manifest = load(original / 'roi_manifest.json')
    if manifest['id'] != prepared['id'] or prepared['id'] != original.name:
        raise ValueError('ROI source ID differs')
    local.mkdir(parents=True, exist_ok=False)
    (local / 'crops').mkdir()
    names = ['source_manifest.json', 'prepare_receipt.json', 'raw_wholebody134.npz']
    names.extend(str(Path(asset['path']).relative_to(original)) for asset in manifest['crops'])
    for name in names:
        source = original / name
        if source.is_symlink() or not source.resolve().is_relative_to(original.resolve()):
            raise ValueError('ROI source/path isolation differs')
        with source.open('rb') as src, (local / name).open('xb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
    mapped = serial.base.remap_paths(manifest, original, local)
    save(local / 'roi_manifest.json', mapped)
    receipt = load(original / 'extract_receipt.json')
    receipt['roi_manifest_sha256'] = roi.sha256(local / 'roi_manifest.json')
    save(local / 'extract_receipt.json', receipt)
    roi.validate_completed(local, prepared)
    return prepared


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--producer-config', required=True)
    p.add_argument('--proof-root', required=True)
    p.add_argument('--parallel-sha', required=True)
    args = p.parse_args()
    proof = Path(args.proof_root)
    if proof.parent != RT or not proof.name.startswith('caption_roi_publish_parallel_v1_proof_'):
        raise ValueError('Independent proof root required')
    if hashlib.sha256(Path(parallel.__file__).read_bytes()).hexdigest() != args.parallel_sha:
        raise ValueError('Frozen parallel publisher differs')
    cfg = load(args.producer_config)
    argv = cfg['native_argv']
    repo = Path(argv[argv.index('--repo') + 1])
    roi_path = repo / 'scripts_omnidance/exp/ms/ms_hand_roi.py'
    spec = importlib.util.spec_from_file_location('benchmark_roi', roi_path)
    roi = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(roi_path.parent))
    spec.loader.exec_module(roi)
    timings = Path(cfg['out']) / 'source_stage_timings.jsonl'
    successful = [v for line in timings.read_bytes().splitlines() if (v := json.loads(line))['status'] == 'ready']
    selected = successful[-3:]
    if len(selected) != 3 or len({v['id'] for v in selected}) != 3:
        raise ValueError('Exactly three existing successful sources required')
    proof.mkdir(exist_ok=False)
    local_root = Path(tempfile.mkdtemp(prefix=proof.name + '-', dir='/tmp'))
    save(proof / 'assignment.json', dict(sources=[dict(id=v['id'], split=v['split']) for v in selected],
        producer_config=args.producer_config, parallel_sha=args.parallel_sha, local_cache=str(local_root),
        new_model_calls=0, production_source_budget_consumed=0, training_eligible=False))
    results = []
    for index, source in enumerate(selected):
        sid = source['id']
        original = Path(cfg['source_root']) / 'roi' / sid
        order = ('serial', 'parallel') if index % 2 == 0 else ('parallel', 'serial')
        case = dict(id=sid, order=order, routes={})
        comparable = {}
        for route in order:
            begin = time.perf_counter()
            local = local_root / route / sid
            prepared = stage(original, local, roi)
            copied = time.perf_counter() - begin
            final = proof / route / sid
            fn = serial.publish_roi if route == 'serial' else parallel.publish_roi
            publishing = time.perf_counter()
            value = fn(roi, local, final, prepared, source['seconds']['roi_extraction'])
            elapsed = time.perf_counter() - begin
            case['routes'][route] = dict(copy_and_local_validation_seconds=copied,
                publication_including_final_validation_seconds=time.perf_counter() - publishing,
                end_to_end_seconds=elapsed, crops=len(value['crops']))
            comparable[route] = serial.base.remap_paths(value, final, Path('/normalized/roi') / sid)
        if comparable['serial'] != comparable['parallel']:
            raise ValueError('Publication changed ROI semantics/identity/asset hashes')
        case['same_roi_semantics_and_asset_hashes'] = True
        results.append(case)
        save(proof / (sid + '.result.json'), case)
        print(json.dumps(case), flush=True)
    serial_seconds = sum(v['routes']['serial']['end_to_end_seconds'] for v in results)
    parallel_seconds = sum(v['routes']['parallel']['end_to_end_seconds'] for v in results)
    report = dict(cases=results, serial_total_seconds=serial_seconds, parallel_total_seconds=parallel_seconds,
        observed_ratio=serial_seconds / parallel_seconds, includes_copy_and_native_final_validation=True,
        order_and_storage_cache_not_fully_isolated=True, new_model_calls=0,
        production_source_budget_consumed=0, training_eligible=False)
    save(proof / 'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
