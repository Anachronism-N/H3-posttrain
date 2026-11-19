import importlib.util,json,subprocess
from pathlib import Path
RT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('verified_batch_review_resource',RT/'resource_manager.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
def validate(cfg):
 if cfg['node']!='N5' or cfg['gpus']!=[7] or not cfg['release_authorized']:raise RuntimeError('only N5 physical GPU7 batch reviewer allowed')
 if '28.216.19.137' not in subprocess.check_output(['hostname','-I'],text=True).split():raise RuntimeError('wrong reviewer node')
 if len(cfg['holders'])!=1 or cfg['holders'][0]['gpu']!=7 or cfg['env']['CUDA_VISIBLE_DEVICES']!='7':raise RuntimeError('exact holder/CVD mismatch')
 if cfg['producer']['path'] not in cfg['argv'] or not Path(cfg['cwd']).is_absolute():raise RuntimeError('frozen batch reviewer path required')
 if cfg['input_count']!=222939 or cfg['targetcount']!=222939:raise RuntimeError('complete curated coverage required')
def check_model(cfg):
 r=cfg['modelref'];v=json.loads(Path(r['path']).read_text())
 if base.sha(r['path'])!=r['sha256'] or not v['complete']:raise RuntimeError('review model stage changed')
 for f in v['files'].values():
  s=Path(f['path']).stat()
  if(s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)!=(f['bytes'],f['mtime_ns'],f['inode'],f['device']):raise RuntimeError('review model storage changed')
def summaries(cfg,out):
 p=out/'run.summary.json';v=json.loads(p.read_text())
 if v['total_source_count']!=222939 or v['expected_batch_count']!=448 or not v['all_batches_reviewed']:raise RuntimeError('incomplete real batch review coverage')
 return {str(p):v}
base.validate_pair=validate;base.check_model=check_model;base.stage_summaries=summaries;base.AUTHORIZED_PAIRS={'N5':((7,),)}
if __name__=='__main__':base.main()
