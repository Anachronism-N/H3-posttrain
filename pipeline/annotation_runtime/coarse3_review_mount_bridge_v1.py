"""Exact content-bound mount-device bridge for the fixed-three prepared review."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

RT = Path(__file__).resolve().parent
WORKER = RT / 'coarse3_independent_review_v1.py'
WORKER_SHA = 'd1a924d5fbac924f37fcb1a149e5e1929c37419009a4b7e73d691a49e5009f1d'
IDS = ('AuuitdcBJyGOnWaH', 'AzPomlvsANsevIFb', 'DxNCjtNsxcazNpGb')


def bridge_references(historical, current):
    if len(historical) != len(current):
        raise ValueError('Mount bridge reference count differs')
    pairs = []
    for old, now in zip(historical, current):
        if set(old) != set(now) or any(old[k] != now[k] for k in old if k != 'stat'):
            raise ValueError('Mount bridge path/content identity differs')
        a, b = old['stat'], now['stat']
        if set(a) != set(b) or any(a[k] != b[k] for k in a if k != 'device'):
            raise ValueError('Mount bridge inode/size/time identity differs')
        if type(a.get('device')) is not int or type(b.get('device')) is not int:
            raise ValueError('Mount-local devices must be explicit integers')
        pairs.append({'historical': old, 'current': now,
                      'same_supplied_path_full_SHA_inode_size_timestamps': True})
    return copy.deepcopy(historical), pairs


def load_worker():
    raw = WORKER.read_bytes()
    if hashlib.sha256(raw).hexdigest() != WORKER_SHA:
        raise ValueError('Frozen fixed-three review source changed')
    spec = importlib.util.spec_from_file_location('mount_bridge_fixed_three_review', WORKER)
    m = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(WORKER), 'exec'), m.__dict__)
    return m


def install(m):
    lib = m.libraries()
    original = lib.audit.source_references
    def source_references(row):
        if row.get('id') not in IDS:
            return original(row)
        root = m.OUT / 'sources' / ('train-' + row['id']) / lib.batch.digest(row)[:24]
        evidence_path = root / 'evidence.json'
        evidence = lib.q.load(evidence_path)
        receipt = lib.q.load(root / 'evidence.receipt.json')
        if lib.q.sha(evidence_path) != receipt['sha256']:
            raise ValueError('Original prepared evidence SHA changed')
        binding = evidence['binding']
        if (binding['source_id'] != row['id'] or binding['split'] != row['split']
                or binding['input_row_sha256'] != lib.batch.digest(row)
                or binding['label_sha256'] != lib.batch.digest(row['label'])):
            raise ValueError('Exact original input/label binding differs')
        # Original function full-hashes each referenced file and checks its
        # expected SHA. Only that freshly checked result may be bridged.
        current = original(row)
        preserved, pairs = bridge_references(binding['source_refs'], current)
        proof = {'schema': 'fixed_three_review_mount_device_bridge_v1', 'source_id': row['id'],
                 'input_row_sha256': binding['input_row_sha256'],
                 'original_evidence_ref': lib.q.identity(evidence_path), 'reference_pairs': pairs,
                 'bridge_code_ref': lib.q.identity(__file__), 'original_evidence_bytes_changed': False,
                 'content_identity_not_inferred_from_stat': True}
        digest = lib.batch.digest(proof)
        lib.q.save_once(m.OUT / 'mount-device-bridges' / (digest + '.json'), proof)
        return preserved
    lib.audit.source_references = source_references
    return lib, original


def prepare_check():
    m = load_worker()
    lib, original = install(m)
    teacher = lib.c.checked({'path': str(RT / 'curated_coarse_empty_n5g01_launch_v1.json'),
        'sha256': '9f5e51a9e2461e864c824add7370ff89fcb1b84db1eae13c59a26b81d77f6806'})
    manifest = lib.c.checked(teacher['jobs_manifest_ref'])
    population = lib.q.identity(m.OUT / 'frozen.population.json')
    results = []
    try:
        for hint in manifest['jobs']:
            candidate, verified = m.validate_candidate(lib, teacher, hint, ffprobe='/opt/ms_tools/ffprobe')
            row = m.review_row(lib, candidate, verified, population)
            root = m.OUT / 'sources' / ('train-' + row['id']) / lib.batch.digest(row)[:24]
            before = lib.q.identity(root / 'evidence.json')
            with lib.review_library():
                evidence = lib.audit.prepare_evidence(row, root, ffprobe='/opt/ms_tools/ffprobe',
                    ffmpeg='/usr/local/bin/ffmpeg', hand_roi_root=row['hand_roi']['path'])
                m.verify_evidence(lib, evidence, row, root, verified['probe'])
            if lib.q.identity(root / 'evidence.json') != before:
                raise ValueError('Mount bridge modified original prepared evidence')
            results.append({'source_id': row['id'], 'exact_binding_and_native_media_verified': True,
                            'body_frames': len(evidence['frames']), 'native_hand_ROIs': len(evidence['hand_rois'])})
    finally:
        lib.audit.source_references = original
    return {'source_count': 3, 'model_calls': 0, 'original_evidence_modified': False, 'sources': results}


if __name__ == '__main__':
    print(json.dumps(prepare_check()), flush=True)
