"""N5 physical GPU7 only; retain the frozen pidfd/lease/restoration manager."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

RT = Path(__file__).resolve().parent
ROOT = RT.parent

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

base = module('frozen_main_review_manager', RT/'resource_manager.py')
worker = module('verified_main_review_watch', RT/'curated_review_watch_v4.py')
BASE_SHA = '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
MODEL_SHA = '76ed7d54737f787434ad8153a0910dfc8d024f146533e9b01443d0b3cfe2dd15'
ORIGINAL = ROOT/'curated222939_production_v1'
OUT = ROOT/'curated_independent_reviews_v1'


def validate(cfg):
    if (cfg['node'] != 'N5' or cfg['gpus'] != [7] or cfg.get('release_authorized') is not True
            or cfg.get('source_prepared') is not True or cfg.get('launch_ready') is not True):
        raise RuntimeError('Only explicitly authorized, prepared N5 physical GPU7 review allowed')
    if '28.216.19.137' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('Wrong reviewer node')
    if (len(cfg['holders']) != 1 or cfg['holders'][0]['gpu'] != 7 or cfg['env']['CUDA_VISIBLE_DEVICES'] != '7'
            or cfg['producer']['path'] != str(RT/'curated_review_watch_v4.py')
            or base.argv_option(cfg, '--results-root') != str(ORIGINAL)
            or base.argv_option(cfg, '--out') != str(OUT) or cfg['out'] != str(OUT)
            or cfg['cwd'] != str(RT/'source_curated_review_v2') or cfg['input_count'] != 222939
            or cfg['targetcount'] != 222939 or cfg['expected_batch_count'] != 448
            or cfg['semantic_approval'] is not False or cfg['reuse_existing_results'] is not True):
        raise RuntimeError('Frozen full main batch reviewer identity/coverage differs')
    if (cfg['phase'] != 'curated_review_watch_v4'
            or Path(cfg['queue']).parent != ROOT/'queues'
            or not Path(cfg['queue']).name.startswith('curated_review_watch_v4-')):
        raise RuntimeError('New v4 queue namespace required')
    if base.sha(RT/'resource_manager.py') != BASE_SHA:
        raise RuntimeError('Frozen base manager changed')
    plan_ref = cfg['recovery_plan_ref']
    if (base.argv_option(cfg, '--completed-stream') != cfg['completed_stream']
            or base.argv_option(cfg, '--recovery-plan') != plan_ref['path']
            or base.argv_option(cfg, '--recovery-plan-sha') != plan_ref['sha256']):
        raise RuntimeError('Worker stream/plan differs from admitted lineage')
    worker.prepared_stream(cfg['completed_stream'], ORIGINAL, plan_ref)
    prefix_path = cfg['completed_stream_prefix_binding']
    if prefix_path not in cfg['source_sha256']:
        raise RuntimeError('fresh immutable stream prefix reference required')
    prefix = json.loads(Path(prefix_path).read_text())
    stream = cfg['completed_stream']
    current = Path(stream).stat()
    if prefix['path'] != stream or (current.st_dev, current.st_ino) != (prefix['device'], prefix['inode']) or current.st_size < prefix['bytes']:
        raise RuntimeError('stream changed after fresh snapshot')
    with Path(stream).open('rb') as f:
        data = f.read(prefix['bytes'])
    if hashlib.sha256(data).hexdigest() != prefix['sha256']:
        raise RuntimeError('fresh committed stream prefix changed')


def check_model(cfg):
    ref = cfg['modelref']
    path = RT/'reviewer8b_stage_TENCENT64.site.json'
    model = Path('/opt/models/Qwen3-VL-8B-Instruct')
    if (ref['path'] != str(path) or ref['sha256'] != MODEL_SHA
            or ref['model_identity_sha256'] != MODEL_SHA or base.sha(path) != MODEL_SHA
            or base.argv_option(cfg, '--model') != str(model)):
        raise RuntimeError('Original N5 8B stage identity differs; no restaging allowed')
    value = json.loads(path.read_text())
    if value.get('complete') is not True or value['destination'] != str(model) or len(value['files']) != 15:
        raise RuntimeError('Native N5 model stage incomplete')
    inventory = {str(p.relative_to(model)) for p in model.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(model).parts)}
    if inventory != set(value['files']):
        raise RuntimeError('Native model file inventory differs')
    for name, item in value['files'].items():
        file = model/name
        stat = file.stat()
        if item['path'] != str(file) or (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev) != (item['bytes'], item['mtime_ns'], item['inode'], item['device']):
            raise RuntimeError('Native N5 staged model storage changed: '+name)
    for name in ('config.json', 'model.safetensors.index.json', 'preprocessor_config.json', 'tokenizer_config.json', 'chat_template.json'):
        if base.sha(model/name) != value['files'][name]['sha256']:
            raise RuntimeError('Native N5 model configuration changed: '+name)


def summaries(cfg, out):
    path = out/'run.summary.json'
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if (value.get('schema') != 'curated_independent_review_watch_v4'
            or value.get('total_source_count') != 222939 or value.get('expected_batch_count') != 448
            or value.get('terminal_source_count') != 222939 or value.get('all_batches_reviewed') is not True
            or value.get('all_source_labels_completed') is not True or value.get('pending_stream_record')
            or value.get('recovery_plan_ref') != cfg['recovery_plan_ref']):
        raise RuntimeError('Incomplete real full-batch source/review coverage')
    return {str(path): value}

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
    if (proof['schema'] != 'curated_review_v4_source_only_v1' or proof['config'] != {'path': args.config, 'sha256': args.config_sha}
            or proof['wrapper_sha256'] != base.sha(__file__) or proof['base_manager_sha256'] != args.manager_sha
            or proof['holder_release_performed'] is not False or proof['GPU_model_loaded'] is not False
            or proof['node'] != 'N5' or proof['gpus'] != [7]):
        raise RuntimeError('completed real source-only preflight receipt required before resource launch')
    base.main()
