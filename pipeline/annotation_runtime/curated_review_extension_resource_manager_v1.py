"""N6 physical GPU2 only; reuse the frozen pidfd/lease/restoration manager."""
import importlib.util
import json
from pathlib import Path
import subprocess

RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('extension_review_resource', RT / 'resource_manager.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def validate(cfg):
    if cfg['node'] != 'N6' or cfg['gpus'] != [2] or cfg.get('release_authorized') is not True:
        raise RuntimeError('explicit release authorization for N6 physical GPU2 required')
    if '28.216.19.225' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('wrong physical review node')
    if len(cfg['holders']) != 1 or cfg['holders'][0]['gpu'] != 2 or cfg['env']['CUDA_VISIBLE_DEVICES'] != '2':
        raise RuntimeError('exact single holder/CVD mismatch')
    if cfg['input_count'] != 222939 or cfg['targetcount'] != 222939:
        raise RuntimeError('original curated scope must remain 222939')
    expected_out = base.EXPERIMENT / 'curated_independent_reviews_extension_v1'
    if Path(cfg['out']).resolve() != expected_out or base.argv_option(cfg, '--out') != str(expected_out):
        raise RuntimeError('only isolated extension output allowed')
    required = {'--manifest': base.EXPERIMENT / 'curated222939_inputs_v1/manifest.summary.json',
        '--collaborative-root': base.EXPERIMENT / 'curated222939_collaborative_v1',
        '--repair-root': base.EXPERIMENT / 'curated222939_repair_v1',
        '--extra-visual-root': base.EXPERIMENT / 'curated_review_extra_visual_v1',
        '--ms-source': RT / 'source_curated_review_v2',
        '--model-reference': RT / 'reviewer8b_stage_N6_extension_v2.json'}
    for option, expected in required.items():
        if base.argv_option(cfg, option) != str(expected):
            raise RuntimeError('extension worker path mismatch: ' + option)
    if '--watch' not in cfg['argv'] or cfg['producer']['path'] != str(RT / 'curated_review_extension_watch_v1.py'):
        raise RuntimeError('frozen extension resident watcher required')
    if cfg['producer']['path'] not in cfg['argv'] or not Path(cfg['cwd']).is_absolute():
        raise RuntimeError('worker argv/cwd mismatch')
    if cfg['phase'] != 'curated_review_extension_v1' or cfg.get('semantic_approval') is not False:
        raise RuntimeError('extension phase or semantic gate mismatch')
    if base.argv_option(cfg, '--global-revision') != 'repair-20261025-v1':
        raise RuntimeError('explicit current JOINT repair edition required')


def check_model(cfg):
    ref = cfg['modelref']
    if base.sha(ref['path']) != ref['sha256']:
        raise RuntimeError('N6 reviewer model reference changed')
    value = json.loads(Path(ref['path']).read_text())
    if value.get('node') != 'N6' or value.get('complete') is not True or value.get('full_copy_SHA_verified') is not True:
        raise RuntimeError('verified N6 8B staging required')
    identity = value['model_identity']
    if identity['sha256'] != ref['model_identity_sha256'] or identity['root'] != '/opt/models/Qwen3-VL-8B-Instruct-review-extension-v2':
        raise RuntimeError('review model identity mismatch')
    root = Path(identity['root'])
    inventory = {p.name for p in root.iterdir() if p.is_file() and not p.name.startswith('.')}
    if inventory != set(identity['files']) or len([n for n in inventory if n.endswith('.safetensors')]) != 4:
        raise RuntimeError('8B model file inventory mismatch')
    for name, entry in identity['files'].items():
        p = root / name
        s = p.stat()
        now = dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                   device=s.st_dev, inode=s.st_ino)
        if p.is_symlink() or entry['path'] != str(p) or now != entry['stat']:
            raise RuntimeError('8B model storage changed: ' + name)
    for name in ('config.json', 'model.safetensors.index.json'):
        if base.sha(root / name) != identity['files'][name]['sha256']:
            raise RuntimeError('8B config/index content changed')


def summaries(cfg, out):
    # A resident watcher has no false 448-batch "finished" claim on interruption.
    return {}


base.AUTHORIZED_PAIRS = {'N6': ((2,),)}
base.validate_pair = validate
base.check_model = check_model
base.stage_summaries = summaries
if __name__ == '__main__':
    base.main()
