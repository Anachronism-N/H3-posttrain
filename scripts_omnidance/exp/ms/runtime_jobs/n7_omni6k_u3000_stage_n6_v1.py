import datetime, hashlib, json, os, pathlib, shutil, subprocess
P=pathlib.Path
MS=P('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS')
OUT=MS/'n7_omni6k_u3000_eval_n6_v1'
BASE=P('/opt/omni30b_dance_official_v1')
DEPLOY=P('/opt/omni_n7_u3000_deploy_v1')
SOURCE='/opt/omni_ms_style_full_train_v3'
CK='checkpoint-u0003000-e04-p0000000'
FP='c5e4955f0b0dd90ebf6afdc3556be2c6127cc4443a9eb150c8d8cff4ac57d315'
CORE='4d82caffd4f6ce354fd6fbd240827d0cac69d66414de33e847501369b0e82dd2'
EXPECTED={'adapter/adapter_model.safetensors':'06cf2c1e5882f6a010305cbd7dc55b8ce261c4d9cd5588eabd28bc2881fec75e','adapter/adapter_config.json':'1a62d41b7e2cf39ee02b1382853b53663c0d44850446f82df575bdbbfa1737ae'}
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 with P(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def event(k,**v):
 print(json.dumps(dict(event=k,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),**v)),flush=True)
def copy(src,dst):
 subprocess.run(['scp','-O','-rp','-o','BatchMode=yes','-o','ConnectTimeout=15','28.216.18.144:'+src,str(dst)],check=True)
def main():
 assert os.getuid()==0
 OUT.mkdir(exist_ok=False)
 save(OUT/'stage_plan.json',dict(host='28.216.19.225',source_host='28.216.18.144',source_checkpoint=SOURCE+'/'+CK,deployment_only=True,resume_checkpoint=False,training_state_read_or_copied=False,manager_pid=os.getpid(),base_local=str(BASE),deployment=str(DEPLOY)))
 event('source_stage_started')
 free=shutil.disk_usage('/opt').free
 cg=P('/sys/fs/cgroup/memory'); room=int((cg/'memory.limit_in_bytes').read_text())-int((cg/'memory.usage_in_bytes').read_text())
 assert free>100*2**30 and room>180*2**30
 assert not DEPLOY.exists()
 DEPLOY.mkdir()
 for item in ('processor','provenance.json'):copy(SOURCE+'/'+item,DEPLOY)
 copy(SOURCE+'/'+CK+'/complete.json',DEPLOY/'source_checkpoint_marker.json')
 copy(SOURCE+'/'+CK+'/adapter',DEPLOY)
 # The source manifest binds all small deployment files; optimizer/RNG are never opened.
 remote_code='import pathlib,hashlib,json; p=pathlib.Path('+repr(SOURCE)+'); c=p/'+repr(CK)+'; fs=[c/"complete.json",p/"provenance.json"]+list((p/"processor").iterdir())+list((c/"adapter").iterdir()); print(json.dumps({str(x.relative_to(p)):hashlib.sha256(x.read_bytes()).hexdigest() for x in fs if x.is_file()}))'
 import shlex
 raw=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','28.216.18.144','/opt/venv_h3/bin/python -c '+shlex.quote(remote_code)],capture_output=True,text=True,check=True).stdout
 source=json.loads(raw)
 marker=json.loads((DEPLOY/'source_checkpoint_marker.json').read_text())
 assert marker['schema']=='omni-dance-sft-v1' and (marker['updates'],marker['epoch'],marker['position'])==(3000,4,0) and marker['fingerprint']==FP
 assert 'training_state.pt' in marker['files']
 deployed={}
 for n,expected in source.items():
  local=DEPLOY/('source_checkpoint_marker.json' if n==CK+'/complete.json' else n.removeprefix(CK+'/'))
  actual=sha(local);assert actual==expected,(n,actual,expected);deployed[str(local.relative_to(DEPLOY))]=actual
 for n,expected in EXPECTED.items():assert marker['files'][n]==expected and deployed[n]==expected
 provenance=json.loads((DEPLOY/'provenance.json').read_text())
 assert provenance['source_sha256']==CORE and provenance['model_path']==str(BASE) and provenance['model_id']=='Qwen/Qwen3-Omni-30B-A3B-Instruct@26291f793822fb6be9555850f06dfe95f2d7e695'
 if not BASE.exists():
  event('base_copy_started',source='28.216.18.144:'+str(BASE),disk_free_bytes=free,cgroup_headroom_bytes=room)
  copy(str(BASE),BASE)
 official=json.loads((BASE/'local_staging_complete.json').read_text())
 assert official['complete'] and official['official_content_verified'] and official['destination']==str(BASE) and official['revision']=='26291f793822fb6be9555850f06dfe95f2d7e695'
 expected_base=dict(provenance['model_files'])
 for f in official['files']:
  assert f['official_content_match'] and P(f['name']).name==f['name']
  assert f['name'] not in expected_base or expected_base[f['name']]==f['sha256']
  expected_base[f['name']]=f['sha256']
 actual_base={}
 for n,expected in expected_base.items():
  assert P(n).name==n
  actual=sha(BASE/n);assert actual==expected,(n,actual,expected)
  actual_base[n]=actual;event('base_file_full_sha_verified',name=n,sha256=actual)
 receipt=dict(complete=True,deployment_only=True,resume_checkpoint=False,training_state_read_or_copied=False,source_host='28.216.18.144',source_checkpoint=SOURCE+'/'+CK,source_checkpoint_marker_sha256=deployed['source_checkpoint_marker.json'],source_checkpoint_fingerprint=FP,updates=3000,epoch=4,position=0,full_training_completed=False,model_id=provenance['model_id'],base_path=str(BASE),base_files_sha256=actual_base,deployment_files_sha256=deployed,source_files_sha256=source,core_sha256=CORE)
 save(DEPLOY/'deployment_receipt.json',receipt)
 save(OUT/'stage_complete.json',dict(**receipt,deployment_receipt_sha256=sha(DEPLOY/'deployment_receipt.json')))
 event('source_stage_complete',base_files=len(actual_base),deployment_files=len(deployed))
if __name__=='__main__':main()
