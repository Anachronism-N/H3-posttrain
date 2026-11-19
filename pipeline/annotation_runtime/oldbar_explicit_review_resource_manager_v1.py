"""Finite independent oldbar3 audit on exact N6 physical GPU3, native base leases."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

RT = Path(__file__).resolve().parent
BASE_SHA = '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
STAGE_SHA = '5607668eb5c476308f20ca20b2853c97bc846f5d43182df2a66729b3c9c4385f'
PHASE = 'oldbar3_explicit_independent_review_v1'
MANIFEST = RT / 'oldbar3_explicit_alternative_candidates_v1.jsonl'
OUT = RT.parent / 'curated_oldbar3_independent_reviews_v1'
QUEUE = RT.parent / 'queues/curated_oldbar3_independent_reviews_n6g3_v1'
MODEL = Path('/opt/models/Qwen3-VL-8B-Instruct-review-extension-v2')
STAGE = RT / 'reviewer8b_stage_N6_extension_v2.json'
ANNOTATOR = RT.parent.parent / '32b_replica_stage_receipts/source_reference_v1.json'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


if digest(RT / 'resource_manager.py') != BASE_SHA:
    raise RuntimeError('Original base resource manager SHA changed; no resource actions')
support = module('oldbar3_extension_model_support', RT / 'curated_review_extension_resource_manager_v1.py')
base = support.base
worker = module('oldbar3_independent_worker', RT / 'oldbar_explicit_review_v1.py')
_cache = {}
_source_cache = worker.References()
_original_sha = base.sha


def cached_sha(path):
    path = Path(path).resolve(strict=True)
    before = worker.batch.snapshot(path)
    key = str(path)
    if key not in _cache or _cache[key][0] != before:
        value = _original_sha(path)
        if worker.batch.snapshot(path) != before:
            raise RuntimeError('File changed while full hashing: ' + key)
        _cache[key] = (before, value)
    return _cache[key][1]


base.sha = cached_sha


def selected(cfg):
    if cfg['plan'] != dict(path=str(MANIFEST), sha256=worker.PILOT_SHA256):
        raise RuntimeError('Exact frozen oldbar3 manifest required')
    rows, _ = worker.load_manifest(MANIFEST)
    return [r['id'] for r in rows], 222939


def source_references():
    rows, manifest = worker.load_manifest(MANIFEST)
    refs = _source_cache
    for row in rows:
        worker.validate_source(row, refs, ffprobe='/opt/ms_tools/ffprobe')
    files = {k: dict(path=k, sha256=v['sha256']) for k, v in refs.files.items()}
    for row in rows:
        for name in ('visual_result_ref', 'raw_rgb_manifest', 'hand_roi'):
            ref = row[name]
            files[ref['path']] = dict(path=ref['path'], sha256=ref['sha256'])
    files[str(MANIFEST)] = dict(path=str(MANIFEST), sha256=manifest['sha256'])
    return rows, [files[k] for k in sorted(files)]


def check_model(cfg):
    if cfg['modelref']['path'] != str(STAGE) or cfg['modelref']['sha256'] != STAGE_SHA:
        raise RuntimeError('Frozen N6 reviewer stage reference differs')
    support.check_model(cfg)
    value = json.loads(STAGE.read_bytes())
    identity = value['model_identity']
    content = {n: {'path': r['path'], 'sha256': r['sha256']} for n, r in identity['files'].items()}
    if worker.batch.digest({'root': identity['root'], 'files': content}) != identity['sha256']:
        raise RuntimeError('N6 complete checkpoint inventory digest mismatch')
    for name, ref in identity['files'].items():
        if cached_sha(MODEL / name) != ref['sha256']:
            raise RuntimeError('N6 reviewer full checkpoint content SHA changed: ' + name)
    # Producer can be remote-attested only through this original full inventory
    # reference. Independent weight-content identities, not deployment names.
    teacher, _ = worker.audit.read_model_reference(ANNOTATOR, permit_attested=True)
    if (worker.audit.checkpoint_identity(teacher) == worker.audit.checkpoint_identity(identity)
            or teacher['sha256'] == identity['sha256']):
        raise RuntimeError('Reviewer checkpoint is not independent of annotator')
    rows, _ = worker.load_manifest(MANIFEST)
    if any(r['annotator_model_sha256'] != teacher['sha256'] for r in rows):
        raise RuntimeError('Three source annotator identity differs from full checkpoint reference')


def lease_value(value):
    return None if value is None else [value.st_dev, value.st_ino, value.st_uid, value.st_mode, value.st_size, value.st_mtime_ns]


def validate(cfg):
    if cfg.get('node') != 'N6' or cfg.get('gpus') != [3] or cfg.get('release_authorized') is not True:
        raise RuntimeError('Only explicitly authorized N6 physical GPU3 release allowed')
    if '28.216.19.225' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('Wrong node; original N6 only')
    if cfg.get('phase') != PHASE or cfg.get('input_count') != 222939 or cfg.get('targetcount') != 3:
        raise RuntimeError('Finite three sources/full curated 222939 scope required')
    if cfg.get('queue') != str(QUEUE) or cfg.get('out') != str(OUT) or cfg['env'].get('CUDA_VISIBLE_DEVICES') != '3':
        raise RuntimeError('Only independent oldbar3 queue/output/CVD3 allowed')
    if cfg.get('source_selection') != dict(shard=0, nshard=1, limit=3):
        raise RuntimeError('Exact finite three source selection required')
    if len(cfg['holders']) != 1 or cfg['holders'][0]['gpu'] != 3:
        raise RuntimeError('Exactly one GPU3 holder required; ignored holders are not free GPUs')
    if cfg.get('semantic_approval') is not False or cfg.get('training_entry_updated') is not False:
        raise RuntimeError('Independent audit does not approve semantic/training entry')
    if cfg.get('resource_manager_ref') != dict(path=str(Path(__file__).resolve()), sha256=cached_sha(__file__)):
        raise RuntimeError('Frozen exact wrapper resource manager identity required')
    required = {'--manifest': MANIFEST, '--out': OUT, '--model': MODEL, '--model-reference': STAGE,
                '--annotator-model-reference': ANNOTATOR, '--ffprobe': Path('/opt/ms_tools/ffprobe'),
                '--ffmpeg': Path('/usr/local/bin/ffmpeg')}
    for option, expected in required.items():
        if base.argv_option(cfg, option) != str(expected):
            raise RuntimeError('Oldbar3 worker path mismatch: ' + option)
    if cfg['argv'][:3] != ['/opt/venv_vllm/bin/python', '-B', '-u'] or cfg['env'].get('PYTHONPATH') != '':
        raise RuntimeError('Original vLLM interpreter and unshadowed GPU dependencies required')
    if '--run' not in cfg['argv'] or '--prepare-only' in cfg['argv'] or '--watch' in cfg['argv']:
        raise RuntimeError('Finite actual --run required, not prepare/watcher')
    producer = RT / 'oldbar_explicit_review_v1.py'
    if cfg['producer']['path'] != str(producer) or str(producer) not in cfg['argv']:
        raise RuntimeError('Frozen oldbar independent reviewer required')
    capture = cfg['holder_capture']
    if cached_sha(capture['path']) != capture['sha256']:
        raise RuntimeError('Fresh capture SHA changed')
    snap = json.loads(Path(capture['path']).read_bytes())
    age = time.time_ns() - snap['captured_unix_ns']
    if (snap['node'] != 'N6' or snap['gpus'] != [3] or snap['holders'] != cfg['holders']
            or snap.get('signals_sent') is not False or snap.get('gpu_started') is not False
            or snap.get('leases_modified') is not False or not 0 <= age <= 1200 * 10**9):
        raise RuntimeError('Capture outside fresh exact single-resource no-action scope')
    if lease_value(base.lease_snapshot(Path('/tmp/od_reserve_3'))) != snap['lease']:
        raise RuntimeError('GPU3 lease changed since fresh capture')
    selected(cfg)
    _, refs = source_references()
    if refs != cfg['ready_source_refs']:
        raise RuntimeError('Frozen original label/generation/input/media/native refs changed')
    for reference in refs:
        if cfg['source_sha256'].get(reference['path']) != reference['sha256']:
            raise RuntimeError('Original source artifact omitted from admission SHA bindings')


def summaries(cfg, out):
    path = out / 'independent.summary.json'
    if not path.exists():
        return {}
    value = worker.audit.read_json(path)
    rows, manifest = worker.load_manifest(MANIFEST)
    expected = {r['id'] for r in rows}
    if (value.get('schema') != worker.SCHEMA or value.get('manifest') != manifest
            or value.get('processed_source_count') != 3 or value.get('audited_source_count') != 3
            or value.get('population_source_count') != 3 or value.get('sampled_reviewed_denominator') != 3
            or value.get('total_curated_source_count') != 222939 or set(value['selected_source_ids']) != expected
            or value.get('original_curated_batch_completed') is not False
            or value.get('original_curated_batches_completed') is not False
            or value.get('independent_sample_review_completed') is not True):
        raise RuntimeError('Oldbar3 processed/audited/full-curated scope mismatch')
    if any(value.get(k) is not False for k in worker.FALSE_FLAGS) or value.get('eligible_for_training') is not False:
        raise RuntimeError('Independent sampled review cannot approve training/human semantics')
    ledger_path = out / 'independent.ledger.jsonl'
    ledger = [worker.audit.strict_json(line) for line in ledger_path.read_bytes().splitlines() if line.strip()]
    if len(ledger) != 3 or {r['source_id'] for r in ledger} != expected:
        raise RuntimeError('Exact three independent review ledger records required')
    reviewer = worker.ManifestReviewer(MODEL, OUT, model_reference=STAGE, annotator_model_reference=ANNOTATOR,
        ffprobe='/opt/ms_tools/ffprobe', ffmpeg='/usr/local/bin/ffmpeg')
    check_model(cfg)  # Cached full checkpoint hashes are valid only at unchanged stats.
    reviewer._checkpoint_content_rehashed = True
    reviewer.identity()
    for candidate in rows:
        row = worker.review_row(candidate, manifest)
        root = out / 'sources' / (candidate['split'] + '-' + candidate['id']) / worker.batch.digest(row)[:24]
        evidence = worker.audit.read_json(root / 'evidence.json')
        probe = worker.bar.probe_actual(row['video_path'], '/opt/ms_tools/ffprobe')
        worker.verify_evidence(evidence, row, root, probe)
        checked, receipt = worker.verify_review_receipt(root, row, evidence, reviewer)
        result = next(r for r in ledger if r['source_id'] == row['id'])
        if (result['status'] != checked['status'] or result['review_receipt'] != receipt
                or result['label_sha256'] != worker.batch.digest(candidate['label'])
                or any(result.get(k) is not False for k in worker.FALSE_FLAGS)):
            raise RuntimeError('Three source independent ledger/receipt binding mismatch')
    return {str(path): value, str(ledger_path): dict(sha256=cached_sha(ledger_path), audited_sources=3,
        training_eligible=False, semantic_approval=False, original_curated_batches_completed=False)}


base.AUTHORIZED_PAIRS = {'N6': ((3,),)}
base.validate_pair = validate
base.selected_sources = selected
base.check_model = check_model
base.stage_summaries = summaries
if __name__ == '__main__':
    base.main()
