"""Explicit curated producer/reviewer assignments with the verified GPU lifecycle."""
import argparse, importlib.util, json, os, socket, subprocess, time
from pathlib import Path
RT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('curated_base_resource',RT/'resource_manager.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
HOSTS={'N1':'28.216.19.69','N3':'28.216.19.213','N5':'28.216.19.137','N6':'28.216.19.225','N8':'28.216.18.136'}
ALLOWED={'producer':{'N8':[(0,),(1,),(2,),(3,)],'N1':[(0,),(1,)],'N5':[(0,),(1,)]},'consumer':{'N8':[(4,5),(6,7)],'N1':[(0,1),(2,3)]},'review':{'N5':[(7,)]}}
def validate(cfg):
 mode=cfg.get('mode');node=cfg['node'];gpus=cfg['gpus'];owners=cfg['holders']
 if tuple(gpus) not in ALLOWED.get(mode,{}).get(node,[]):raise RuntimeError('outside explicit curated GPU assignment')
 addresses=subprocess.check_output(['hostname','-I'],text=True).split()
 if HOSTS[node] not in addresses:raise RuntimeError('manager executing on wrong physical node')
 if cfg.get('release_authorized') is not True:raise RuntimeError('holder release not authorized')
 if len(owners)!=len(gpus) or [o['gpu'] for o in owners]!=gpus or len({o['pid'] for o in owners})!=len(gpus) or len({o['uuid'] for o in owners})!=len(gpus):raise RuntimeError('holder inventory mismatch')
 if cfg['env'].get('CUDA_VISIBLE_DEVICES')!=','.join(map(str,gpus)):raise RuntimeError('worker CVD mismatch')
 if cfg['producer']['path'] not in cfg['argv'] or not Path(cfg['cwd']).is_absolute():raise RuntimeError('frozen worker path/cwd mismatch')
 if not isinstance(cfg['targetcount'],int) or cfg['targetcount']<=0:raise RuntimeError('positive selected source count required')
 if mode in ('producer','consumer') and mode not in cfg['argv']:raise RuntimeError('worker mode mismatch')
 if mode in ('producer','consumer'):
  summary_ref=cfg['manifest_summary']
  if base.argv_option(cfg,'--manifest')!=summary_ref['path'] or cfg['source_sha256'].get(summary_ref['path'])!=summary_ref['sha256']:raise RuntimeError('frozen manifest summary/argv mismatch')
  summary=json.loads(Path(summary_ref['path']).read_text())
  if summary['total']!=222939 or summary['manifest']!=cfg['original_full_manifest']:raise RuntimeError('curated manifest provenance mismatch')
 if any(n in cfg['env'] for n in ('DWPOSE_ALLOW_CPU',)):raise RuntimeError('CPU-only ROI not admitted')
def selected(cfg):
 selection={key:int(base.argv_option(cfg,'--'+key,default)) for key,default in (('shard',0),('nshard',1),('limit',0))}
 if cfg.get('source_selection',selection)!=selection or selection['nshard']<=0 or not 0<=selection['shard']<selection['nshard'] or selection['limit']<0:raise RuntimeError('curated selection mismatch')
 ids=[];seen=set();total=0
 with Path(cfg['plan']['path']).open() as stream:
  for line in stream:
   if not line.strip():continue
   row=json.loads(line);sid=row['id']
   if not isinstance(sid,str) or not sid or sid.casefold() in seen:raise RuntimeError('invalid/duplicate curated source')
   seen.add(sid.casefold())
   if total%selection['nshard']==selection['shard'] and (not selection['limit'] or len(ids)<selection['limit']):ids.append(sid)
   total+=1
 if total!=222939 or cfg.get('input_count')!=total:raise RuntimeError('entire curated pool must remain 222939')
 return ids,total
def check_model(cfg):
 ref=cfg['modelref']
 if base.sha(ref['path'])!=ref['sha256']:raise RuntimeError('model reference changed')
 model=json.loads(Path(ref['path']).read_text())
 if cfg['mode']=='producer':
  if model['sha256']!=ref['model_identity_sha256']:raise RuntimeError('ROI model identity changed')
  for f in model['files'].values():
   if base.sha(f['path'])!=f['sha256']:raise RuntimeError('ROI model/code content changed')
 elif cfg['mode']=='consumer':original_model(cfg)
 else:
  if not model.get('complete'):raise RuntimeError('review model staging incomplete')
  for f in model['files'].values():
   p=Path(f['path']);st=p.stat()
   if st.st_size!=f['bytes'] or st.st_mtime_ns!=f['mtime_ns'] or st.st_ino!=f['inode'] or st.st_dev!=f['device']:raise RuntimeError('reviewer model storage changed')
def summaries(cfg,out):
 if cfg['mode']!='producer':return original_summaries(cfg,out)
 expected,total=selected(cfg);expected_set=set(expected);results={}
 for p in (out/'sources').glob('*/*/producer.terminal.json'):
  v=json.loads(p.read_text());sid=v['id']
  if sid not in expected_set:continue
  if v['status'] not in ('ready','failed'):raise RuntimeError('producer terminal not complete')
  if sid in results:raise RuntimeError('duplicate producer terminal')
  if v['status']=='ready':
   for key in ('plan_ref','roi_ref'):
    r=v[key]
    if base.sha(r['path'])!=r['sha256']:raise RuntimeError('producer artifact changed')
  results[sid]=v
 markers=list((out/'runs').glob('*/completed.json'))
 if set(results)!=set(expected) or not markers:raise RuntimeError('not every assigned source is terminal')
 return {str(markers[-1]):{'summary':{'stage':'run','total_source_count':total,'selected_source_count':len(expected),'processed_source_count':len(results),'plan_sha256':cfg['plan']['sha256']},'selection':{'selected_ids':expected},'source_failures_preserved':True}}
original_selected=base.selected_sources;original_model=base.check_model;original_summaries=base.stage_summaries
base.validate_pair=validate;base.selected_sources=selected;base.check_model=check_model;base.stage_summaries=summaries
base.AUTHORIZED_PAIRS={n:tuple(set(p for scope in ALLOWED.values() for p in scope.get(n,[]))) for n in HOSTS}
def capture(node,gpus,out,mode):
 if tuple(gpus) not in ALLOWED.get(mode,{}).get(node,[]):raise RuntimeError('invalid capture scope')
 if HOSTS[node] not in subprocess.check_output(['hostname','-I'],text=True).split():raise RuntimeError('capture host mismatch')
 owners=[]
 for g in gpus:
  uuid=base.uuid_for(g);apps=[a for a in base.apps() if a['uuid']==uuid]
  if len(apps)!=1:raise RuntimeError('must have exactly one authorized holder')
  owner=base.identity(apps[0]['pid']);owner.update(gpu=g,uuid=uuid)
  base.check_holder(owner,base.identity(owner['pid']),g,uuid,base.apps());base.lease_snapshot(Path('/tmp/od_reserve_%d'%g));owners.append(owner)
 value={'node':node,'mode':mode,'gpus':gpus,'holders':owners,'captured_unix':time.time(),'headroom':base.headroom(),'signals_sent':False}
 with Path(out).open('x') as f:json.dump(value,f,indent=2)
 print(json.dumps({'path':out,'sha256':base.sha(out),'signals_sent':False}),flush=True)
if __name__=='__main__':
 if '--capture-current' in __import__('sys').argv:
  p=argparse.ArgumentParser();p.add_argument('--capture-current',action='store_true');p.add_argument('--node',required=True);p.add_argument('--gpus',nargs='+',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--mode',required=True);a=p.parse_args();capture(a.node,a.gpus,a.out,a.mode)
 else:base.main()
