import json,importlib.util,subprocess,sys
from pathlib import Path
RT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('review_resource_base',RT/'resource_manager.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
def validate(cfg):
 if cfg['node']!='N5' or cfg['gpus']!=[7] or cfg.get('release_authorized') is not True:raise RuntimeError('only N5 GPU7 independent review authorized')
 if '28.216.19.137' not in subprocess.check_output(['hostname','-I'],text=True).split():raise RuntimeError('wrong reviewer node')
 if len(cfg['holders'])!=1 or cfg['holders'][0]['gpu']!=7 or cfg['env']['CUDA_VISIBLE_DEVICES']!='7':raise RuntimeError('exact reviewer holder/CVD required')
 if cfg['producer']['path'] not in cfg['argv'] or '--run' not in cfg['argv']:raise RuntimeError('actual reviewer run required')
def selection(cfg):
 rows=[json.loads(l) for l in Path(cfg['plan']['path']).read_text().splitlines() if l.strip()];ids=[r['id'] for r in rows]
 if len(ids)!=25 or len(set(ids))!=25 or cfg['targetcount']!=25:raise RuntimeError('fixed independent pilot25 identity mismatch')
 return ids,len(ids)
def check_model(cfg):
 r=cfg['modelref'];m=json.loads(Path(r['path']).read_text())
 if base.sha(r['path'])!=r['sha256'] or not m['complete']:raise RuntimeError('reviewer stage identity changed')
 for f in m['files'].values():
  p=Path(f['path']);s=p.stat()
  if (s.st_size,s.st_mtime_ns,s.st_ino,s.st_dev)!=(f['bytes'],f['mtime_ns'],f['inode'],f['device']):raise RuntimeError('reviewer model file changed')
def summaries(cfg,out):
 p=out/'run.summary.json';v=json.loads(p.read_text())
 if v['batch_count']!=1 or not v['all_batches_completed']:raise RuntimeError('not all pilot sources actually reviewed')
 ids,_=selection(cfg);s=json.loads(Path(v['batch_summaries'][0]).read_text())
 if set(s['reviewed_source_ids'])!=set(ids):raise RuntimeError('review sample not complete')
 return {str(p):v}
base.validate_pair=validate;base.selected_sources=selection;base.check_model=check_model;base.stage_summaries=summaries
base.AUTHORIZED_PAIRS={'N5':((7,),)}
if __name__=='__main__':base.main()
