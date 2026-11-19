"""Single admitted N5 GPU7 v5 reviewer; frozen pidfd/lease/restore transaction."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

RT = Path(__file__).resolve().parent
ROOT = RT.parent
spec = importlib.util.spec_from_file_location('frozen_review_resource_v5', RT / 'resource_manager.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def validate(cfg):
    if cfg['node'] != 'N5' or cfg['gpus'] != [7] or cfg.get('release_authorized') is not True or cfg.get('source_prepared') is not True:
        raise RuntimeError('Only source-prepared explicitly authorized N5 physical GPU7 v5 allowed')
    if '28.216.19.137' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('Wrong reviewer node')
    if (len(cfg['holders']) != 1 or cfg['holders'][0]['gpu'] != 7 or cfg['env']['CUDA_VISIBLE_DEVICES'] != '7'
            or cfg['producer']['path'] != str(RT / 'main_review_watch_v5.py') or cfg['producer']['path'] not in cfg['argv']
            or cfg['phase'] != 'main_review_watch_v5' or not Path(cfg['queue']).name.startswith('main_review_watch_v5-')
            or cfg['cwd'] != str(RT / 'source_curated_review_v2') or cfg['input_count'] != 222939 or cfg['targetcount'] != 222939
            or cfg['expected_batch_count'] != 448 or cfg['reuse_existing_results'] is not True or cfg['semantic_approval'] is not False):
        raise RuntimeError('Frozen original full-source v5 reviewer identity differs')
    for option, value in (('--results-root', str(ROOT / 'curated222939_production_v1')),
                          ('--out', str(ROOT / 'curated_independent_reviews_v1')),
                          ('--model', '/opt/models/Qwen3-VL-8B-Instruct'),
                          ('--ms-source', str(RT / 'source_curated_review_v2')),
                          ('--additional-results-root', str(ROOT / 'curated222939_collaborative_v2')),
                          ('--additional-out', str(ROOT / 'curated_independent_reviews_collaborative_v2')),
                          ('--additional-manifest', str(RT / 'curated_collaborative21000_auto_review_manifest_v2.json'))):
        if base.argv_option(cfg, option) != value:
            raise RuntimeError('Worker namespace/model/edition option differs: ' + option)
    binding_path = base.argv_option(cfg, '--source-binding')
    if binding_path != cfg['source_binding']['path'] or cfg['source_sha256'].get(binding_path) != cfg['source_binding']['sha256']:
        raise RuntimeError('Complete source-binding SHA required')
    binding = json.loads(Path(binding_path).read_text())
    prefix = binding['main_prefix']
    stream = ROOT / 'curated222939_production_v1/completed.jsonl'
    s = stream.stat()
    if prefix['path'] != str(stream) or [s.st_dev, s.st_ino] != prefix['identity'] or s.st_size < prefix['bytes']:
        raise RuntimeError('Source-prepared original stream identity changed')
    h = hashlib.sha256()
    with stream.open('rb') as f:
        left = prefix['bytes']
        while left:
            block = f.read(min(left, 1048576))
            if not block:
                raise RuntimeError('Original prepared prefix truncated')
            h.update(block)
            left -= len(block)
    if h.hexdigest() != prefix['sha256']:
        raise RuntimeError('Source-prepared original prefix changed')


def check_model(cfg):
    ref = cfg['modelref']
    path = RT / 'reviewer8b_stage_TENCENT64.site.json'
    model = Path('/opt/models/Qwen3-VL-8B-Instruct')
    if ref['path'] != str(path) or base.sha(path) != ref['sha256']:
        raise RuntimeError('Original full-copy model evidence changed')
    value = json.loads(path.read_text())
    if value.get('complete') is not True or value.get('full_copy_SHA_verified') is not True or value['destination'] != str(model):
        raise RuntimeError('Original local model copy not fully attested')
    actual = {str(p.relative_to(model)) for p in model.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(model).parts)}
    if actual != set(value['files']):
        raise RuntimeError('Actual local model inventory differs')
    for name, f in value['files'].items():
        p = model / name
        s = p.stat()
        if f['path'] != str(p) or (s.st_size, s.st_mtime_ns, s.st_ino, s.st_dev) != (f['bytes'], f['mtime_ns'], f['inode'], f['device']):
            raise RuntimeError('Native local model storage changed: ' + name)
    for name in ('config.json', 'model.safetensors.index.json', 'preprocessor_config.json', 'chat_template.json'):
        if base.sha(model / name) != value['files'][name]['sha256']:
            raise RuntimeError('Actual model config/index changed')
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    if set(index['weight_map'].values()) != {n for n in actual if n.endswith('.safetensors')}:
        raise RuntimeError('Actual model index weights differ')


def summaries(cfg, out):
    p = out / 'run.summary.json'
    extra = ROOT / 'curated_independent_reviews_collaborative_v2/run.summary.json'
    if not p.exists() or not extra.exists():
        return {}
    values = {str(p): json.loads(p.read_text()), str(extra): json.loads(extra.read_text())}
    for path, count, batches in ((str(p), 222939, 448), (str(extra), 10000, 20)):
        v = values[path]
        if (v.get('schema') != 'curated_independent_review_watch_v5' or v['total_source_count'] != count
                or v['expected_batch_count'] != batches or v['terminal_source_count'] != count
                or v['all_batches_reviewed'] is not True or v['all_source_labels_completed'] is not True
                or v['pending_stream_record'] or v['source_binding'] != cfg['source_binding']['path']):
            return {}
    return values


base.validate_pair = validate
base.check_model = check_model
base.stage_summaries = summaries
base.AUTHORIZED_PAIRS = {'N5': ((7,),)}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--config-sha', required=True)
    ap.add_argument('--manager-sha', required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    proof = json.loads(Path(cfg['source_only_preflight_receipt']).read_text())
    if (proof['schema'] != 'main_review_v5_source_only_v1' or proof['config'] != {'path': args.config, 'sha256': args.config_sha}
            or proof['wrapper_sha256'] != base.sha(__file__) or proof['base_manager_sha256'] != args.manager_sha
            or proof['holder_release_performed'] is not False or proof['GPU_model_loaded'] is not False):
        raise RuntimeError('Completed real source-only preflight receipt required before release')
    base.main()
