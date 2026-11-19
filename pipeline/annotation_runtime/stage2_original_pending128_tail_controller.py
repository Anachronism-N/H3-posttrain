"""First execution of original128 pending sources after all N1 original-claim tail results ACK."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
RT=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime')
ROOT=RT.parent


def put(p,raw):
    p.parent.mkdir(parents=True,exist_ok=True);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    d=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(d);os.close(d)
    return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def run():
    original=Path('/root/stage2_dynamic_native37610_N1_20261028_v1');tail=original/'original_uncalled_recovery_launched.json';deadline=time.monotonic()+86400
    print(json.dumps(dict(event='never_started_original128_waiting_last_N1_recovery',source_count=128,old_config_not_restarted=True)),flush=True)
    while not tail.exists():assert time.monotonic()<deadline;time.sleep(15)
    sys.path.insert(0,str(original/'code'));from stage2_local_publication_v1 import checked,write_once
    launched=json.loads(tail.read_bytes());assert launched['source_count']==8 and launched['original_owners_preserved'] is True;configs=[checked(r) for r in launched['configs']]
    while True:
        ready=True
        for c in configs:
            q=Path(c['queue'])
            if (q/'failure.json').exists():raise RuntimeError('Original claim recovery failed; keep pending128 unclaimed')
            if not (q/'resources_safe.json').exists():ready=False;continue
            s=json.loads((q/'resources_safe.json').read_bytes());assert all(s.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored')) and json.loads((q/'worker_exit.json').read_bytes())['exit_code']==0
        if ready:
            expected={tuple(p.relative_to(original/'out'/'sources').parts[:2]) for p in (original/'out'/'sources').glob('*/*/commit.json')};acked=set()
            for p in (original/'out'/'outbox').iterdir():
                if not (p/'sealed.json').exists() or not (p/'upload_receipt.json').exists():ready=False;break
                seal=json.loads((p/'sealed.json').read_bytes());ack=json.loads((p/'upload_receipt.json').read_bytes());assert ack['batch_id']==seal['batch_id'] and ack['archive_sha256']==seal['archive_sha256'];acked.update(map(tuple,seal['source_keys']))
            ready=ready and expected==acked
        if ready:break
        assert time.monotonic()<deadline;time.sleep(10)
    root=Path('/root/stage2_dynamic_originalpending128_N1_20261028_v1');root.mkdir(exist_ok=False);code=root/'code';code.mkdir();hashes={}
    for n in ('stage2_local_dynamic_native_manager_v4.py','stage2_local_model_worker_v1.py','stage2_local_packet_v1.py','stage2_local_publication_v1.py','stage2_model_codec_v1.py','stage2_cpu_feeder_v3.py','stage2_feeder_media_v1.py','stage2_node_uploader_v1.py','resource_manager.py','stage2_start_staged_v3.py'):
        r=put(code/n,(RT/n).read_bytes());hashes[r['path']]=r['sha256']
    sys.path.insert(0,str(code));import stage2_local_dynamic_native_manager_v4 as m
    plan_path=ROOT/'ms_vlm_n1_fastmeta_inputs_20261028_v1/n1g01_plan128.jsonl';raw=plan_path.read_bytes();assert hashlib.sha256(raw).hexdigest()=='f0f48ba76929c469843e1d16571238186b9b1a949a30eafbc63c2fcbd5644be9';rows=[json.loads(x) for x in raw.splitlines() if x];assert len(rows)==128;legacy_template=json.loads((RT/'ms_fullframe_vlm_postprocess_batch_v8_fastmeta128_n1g01_20261028_template.json').read_bytes());assert not Path(legacy_template['out']).exists() and not Path(legacy_template['queue']).exists(),'Original128 already started elsewhere; do not duplicate'
    plan=put(root/'pool_manifest.jsonl',raw);hashes[plan['path']]=plan['sha256'];base=json.loads((Path('/root/stage2_dynamic_legacy15973_N1_20261028_v1')/'deployment.json').read_bytes());template=checked(next(x['config_ref'] for x in base['configs'] if x['gpus']==[0,1]));modelraw=Path(template['modelref']['path']).read_bytes();model=put(root/'model_reference.json',modelraw);hashes[model['path']]=model['sha256'];model.pop('bytes');model['model_identity_sha256']=template['modelref']['model_identity_sha256'];query="import pathlib,sys;sys.stdout.buffer.write(pathlib.Path('/root/stage2_pool_originalpending128_20261028_v1/token').read_bytes())";r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','28.216.17.70','/opt/venv_eval/bin/python','-B','-c',shlex.quote(query)],capture_output=True,timeout=15);assert r.returncode==0;put(root/'token',r.stdout)
    holders=[]
    for g in (0,1):
        uuid=m.b.uuid_for(g);apps=[x for x in m.b.apps() if x['uuid']==uuid];assert len(apps)==1;holder=m.b.identity(apps[0]['pid']);holder.update(gpu=g,uuid=uuid);m.b.check_holder(holder,m.b.identity(holder['pid']),g,uuid,m.b.apps());holders.append(holder)
    phase='stage2_dynamic_originalpending128_N1_g01';cp=root/(phase+'_config.json');producer=dict(path=str(code/'stage2_local_dynamic_native_manager_v4.py'),sha256=hashes[str(code/'stage2_local_dynamic_native_manager_v4.py')]);c=dict(template);c.update(phase=phase,owner='N1-g01-originalpending128',local_root=str(root),out=str(root/'out'),queue=str(root/'queues'/phase),config_path=str(cp),holders=holders,input_count=128,targetcount=128,plan=dict(path=plan['path'],sha256=plan['sha256']),modelref=model,source_sha256=hashes,producer=producer,resource_manager_ref=producer,worker_state_layout='by_phase',argv=['/opt/venv_vllm/bin/python','-B',producer['path'],'worker','--config',str(cp)]);cr=write_once(cp,c);m.check_sources(c,cp,cr['sha256'],m.ORIGINAL_BASE_SHA);m.b.check_model(c);m.validate(c);feed=write_once(root/(phase+'_feeder.json'),dict(out=c['out'],owner=c['owner'],token_path=str(root/'token'),claim_endpoint='http://28.216.17.70:19003/claim',ffmpeg_path='/usr/local/bin/ffmpeg',prefetch_packets=8));write_once(root/'deployment.json',dict(node='N1',configs=[dict(config_ref=cr,feeder_ref=feed,owner=c['owner'],phase=phase,gpus=[0,1])],global_pool_namespace='originalpending128',global_pool_count=128,source_sha256=hashes,source_claims_created=0,old_failed_fixed_config_not_restarted=True));write_once(root/'previous_tail_safe_and_shared_ACK.json',dict(original_tail_ref=str(tail),source_count=len(expected),all_original_tail_node_results_ACKed=True,new_source_pool_allocation_unchanged=True,observed_unix=time.time()))
    stop=original/'out'/'UPLOAD_RECOVERY_STOP'
    if not stop.exists():write_once(stop,dict(reason='All original recovery ACKed; transfer one node uploader to never-started128',models_called=0))
    def original_uploader_alive():
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:a=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
            except (OSError,UnicodeDecodeError):continue
            if len(a)>2 and a[2]==str(original/'code/stage2_original_recovery_uploader.py'):return True
        return False
    until=time.monotonic()+20
    while original_uploader_alive():assert time.monotonic()<until;time.sleep(.5)
    (root/'out'/'ready'/c['owner']).mkdir(parents=True);logs=root/'logs';logs.mkdir();write_once(logs/'uploader_launch.json',dict(one_uploader_per_node=True,models_called=0));cmd=shlex.join(['/opt/venv_eval/bin/python','-B',str(code/'stage2_node_uploader_v1.py'),'--local-root',c['out'],'--shared-root',str(ROOT/'stage2_dynamic_local_shared_results_20261028_v1'),'--node','N1','--seconds','21600'])+' > '+shlex.quote(str(logs/'uploader.log'))+' 2>&1';subprocess.run(['tmux','-S',str(root/'tmux.sock'),'new-session','-d','-s','main3936','-n','uploader','-c','/tmp',cmd],check=True);import stage2_start_staged_v3 as launcher;launcher.start(root,[0,1]);print(json.dumps(dict(event='original_never_started128_first_execution',old_config_not_restarted=True,source_count=128)),flush=True)

if __name__=='__main__':run()
