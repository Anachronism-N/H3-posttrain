"""Record actual N3 local model metadata only; never load model or mutate resources."""
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('next_range_manager', RT / 'curated_collaborative21000_resource_manager_v2.py')
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)
receipt = json.loads((RT / 'curated_collaborative21000_build_receipt_v2.json').read_text())
reference = receipt['configs']['n3g67']
if manager.c.base.sha(reference['path']) != reference['sha256']:
    raise RuntimeError('Next-range consumer config changed')
cfg = json.loads(Path(reference['path']).read_text())
manager.validate(cfg)
manager.c.check_model(cfg)
root = Path('/opt/models/Qwen3-VL-32B-Instruct')
source = json.loads(Path(cfg['modelref']['path']).read_text())['model_identity']
files = {}
for name, expected in source['files'].items():
    path = root / name
    info = path.stat()
    if info.st_size != expected['stat']['size']:
        raise RuntimeError('Actual local model file size changed')
    files[name] = {'size': info.st_size, 'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns,
                   'inode': info.st_ino, 'device': info.st_dev}
indexes = {}
for name in ('config.json', 'model.safetensors.index.json'):
    digest = manager.c.base.sha(root / name)
    if digest != source['files'][name]['sha256']:
        raise RuntimeError('Actual local model config/index SHA changed')
    indexes[name] = digest
report = RT / 'curated_collaborative21000_n3_modelstat_readonly_v2.json'
value = {'config': reference, 'modelref': cfg['modelref'], 'model_identity_sha256': source['sha256'],
         'actual_local_model_root': str(root), 'file_count': len(files), 'actual_stat': files,
         'actual_config_index_sha256': indexes, 'checked_unix_ns': time.time_ns(),
         'tensor_payload_rehashed': False, 'model_loaded': False, 'signals_sent': False,
         'gpu_started': False, 'leases_created': False}
with report.open('x') as stream:
    json.dump(value, stream, indent=2)
print(json.dumps({'report': str(report), 'sha256': manager.c.base.sha(report), 'file_count': len(files),
                  'actual_config_index_sha256': indexes, 'model_loaded': False, 'signals_sent': False}))
