"""Reuse only the idle N2 g23 native-safe slot for a three-case minimal-patch diagnostic."""
import hashlib,importlib.util,json,os,shlex,subprocess,sys,time
from pathlib import Path
RT=Path(__file__).parent

def run():
 root=Path('/root/stage2_motion_revision_final_main_N2_20261028_v1');sys.path.insert(0,str(root/'code'));from stage2_local_publication_v1 import checked,write_once
 binding=json.loads((root/'runtime_bindings/N2-g23-rv3main.json').read_bytes());old=checked(binding['config_ref']);assert old['gpus']==[2,3] and old['evidence_policy']=='window_visibility_boundary_evidence_v1'
 write_once(root/'minimal_patch_diag_before_drain_v2.json',dict(original_config_ref=binding['config_ref'],gpus=[2,3],fixed_sources=3,new_pool_claims=0,no_signals=True,old_outputs_budgets_unchanged=True,observed_unix=time.time()))
 stop=root/'out/FEEDER_STOP_EVIDENCE1'
 if not stop.exists():write_once(stop,dict(reason='Latest task narrows VLM to event patches; old trial feeders must not claim more data',new_pool_claims=0))
 q=Path(old['queue']);until=time.monotonic()+180
 while not (q/'resources_safe.json').exists():assert time.monotonic()<until and not (q/'failure.json').exists();time.sleep(1)
 safe=json.loads((q/'resources_safe.json').read_bytes());assert all(safe.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored')) and json.loads((q/'worker_exit.json').read_bytes())['exit_code']==0
 summary=checked(dict(path=str(root/'worker_state'/old['owner']/old['phase']/'summary.json'),sha256=hashlib.sha256((root/'worker_state'/old['owner']/old['phase']/'summary.json').read_bytes()).hexdigest()));assert summary['processed']==0 and summary['all_outcomes']==[],'No production calls may be silently discarded'
 code=root/'minimal_patch_diag_code_v2';code.mkdir(exist_ok=False);hashes={}
 mapping={'motion_revision_native_manager_minimal_patch_diag_v2.py':'motion_revision_native_manager_minimal_patch_diag_v2.py','motion_revision_model_worker_evidence_v1.py':'motion_revision_model_worker_minimal_patch_v2.py','motion_revision_codec_v1.py':'motion_revision_codec_minimal_patch_v2.py','stage2_local_publication_v1.py':'stage2_local_publication_v1.py','resource_manager.py':'resource_manager.py'}
 for target,source in mapping.items():
  raw=(RT/source).read_bytes()
  if target=='motion_revision_codec_v1.py':raw=raw.replace(b'max_num_seqs=8,max_model_len',b'max_num_seqs=16,max_model_len')
  compile(raw,str(code/target),'exec');p=code/target;fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  hashes[str(p)]=hashlib.sha256(raw).hexdigest()
 path=code/'motion_revision_native_manager_minimal_patch_diag_v2.py';spec=importlib.util.spec_from_file_location('minimal_patch_native',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);holders=[]
 for g in old['gpus']:
  uuid=m.b.uuid_for(g);apps=[x for x in m.b.apps() if x['uuid']==uuid];assert len(apps)==1;h=m.b.identity(apps[0]['pid']);h.update(gpu=g,uuid=uuid);m.b.check_holder(h,m.b.identity(h['pid']),g,uuid,m.b.apps());holders.append(h)
 phase=old['phase']+'_minimal_patch_diag2';cp=root/(phase+'_config.json');producer=dict(path=str(path),sha256=hashes[str(path)]);c=dict(old);c.update(phase=phase,queue=str(root/'queues'/phase),config_path=str(cp),holders=holders,producer=producer,resource_manager_ref=producer,source_sha256=dict(old['source_sha256'],**hashes),argv=['/opt/venv_vllm/bin/python','-B',str(path),'worker','--config',str(cp)],evidence_policy='minimal_event_patch_eligible_pairs_v2',diagnostic_only=True,targetcount=3,evidence_pilot=False)
 cr=write_once(cp,c);m.check_sources(c,cp,cr['sha256'],m.ORIGINAL_BASE_SHA);m.b.check_model(c);m.validate(c)
 write_once(root/'minimal_patch_diagnostic_deployment_v2.json',dict(config_ref=cr,fixed_cases=3,gpus=[2,3],automatic_production_promotion=False,observed_unix=time.time()))
 print(json.dumps(dict(event='minimal_patch_native_ready',config_ref=cr,gpus=[2,3])),flush=True)
 subprocess.run(['/opt/venv_vllm/bin/python','-B',str(path),'--config',str(cp),'--config-sha',cr['sha256'],'--manager-sha',m.ORIGINAL_BASE_SHA],cwd='/tmp',check=True)
 q=Path(c['queue']);s=json.loads((q/'resources_safe.json').read_bytes());assert all(s.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored'))
 print(json.dumps(dict(event='minimal_patch_diagnostic_native_safe',outcomes=str(root/'evidence_fixed_three_minimal_patch_outcomes_v2.json'))),flush=True)

if __name__=='__main__':run()
