"""Add external GPU4-7 TP4 replica using same50000 ledger and one node uploader."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import select
import shlex
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlsplit
from stage2_local_publication_v1 import write_once,require

SELF=Path(__file__).resolve().parent
NODES=tuple('a100-extra-0'+str(i) for i in range(3,7))


def launch_window(root,socket,session,name,argv,first=False):
    log=root/(name+'.log');require(not log.exists(),'Existing launch log must not be overwritten');cmd=shlex.join(argv)+' > '+shlex.quote(str(log))+' 2>&1'
    if first:args=['tmux','-S',str(socket),'new-session','-d','-s',session,'-n',name,'-c','/tmp',cmd]
    else:args=['tmux','-S',str(socket),'new-window','-d','-t',session,'-n',name,'-c','/tmp',cmd]
    subprocess.run(args,check=True)


def relay(a):
    oldroot=Path('/root/stage2_external50000_relay_v1');old=json.loads((oldroot/'relay.json').read_bytes());db=Path(old['database']);require(db==oldroot/'pool.db' and db.is_file(),'Reuse the original durable50000 pool; never initialize another')
    from stage2_dynamic_pool_v1 import DynamicPool
    p=DynamicPool(db)
    try:require(sum(p.progress().values())==50000,'Original source-group scope differs')
    finally:p.close()
    root=Path('/root/stage2_external50000_relay_expand_v1');root.mkdir(exist_ok=False);cfg=dict(old,local_root=str(root),bind=a.bind,port=a.port,allowed_owners=[n+'-g'+pair+'-external50000' for n in NODES for pair in ('0123','4567')]);write_once(root/'relay.json',cfg);write_once(root/'deployment.json',dict(original_database=str(db),original_token_path=old['token_path'],new_source_pool_created=False,extra_owners_added=4,old_relay_not_stopped=True,models_called=0,observed_unix=time.time()));launch_window(root,root/'tmux.sock','expanded-relay','relay',[sys.executable,'-B',str(SELF/'stage2_external_relay.py'),'--config',str(root/'relay.json')],True);print(json.dumps(dict(event='external_expanded_relay_started',port=a.port,same_original_database=True,old_relay_kept_running=True,new_source_claims=0)),flush=True)


def node(a):
    from stage2_external_native_manager import verify,memory,apps
    root=Path('/root/stage2_external50000_'+a.node+'_v1');oldpath=root/'manager.json';old=json.loads(oldpath.read_bytes());require(old['node']==a.node and old['gpus']==[0,1,2,3] and old['expected_hostname']==subprocess.check_output(['hostname'],text=True).strip(),'Original physical node/primary replica must match');ad=json.loads((Path(old['queue'])/'admission.json').read_bytes());require(ad['config_sha256']==hashlib.sha256(oldpath.read_bytes()).hexdigest(),'Original primary config changed');newpath=root/'manager_g4567.json';require(not newpath.exists() and not (root/'queue_g4567').exists(),'Already-used extra replica must not restart');require(memory()['headroom']>=160*2**30 and __import__('shutil').disk_usage('/tmp').free>=8*2**30,'Fresh160GiB effective memory/tmp8GiB required');inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True);uuids={x.split(',')[1].strip() for x in inventory.splitlines() if int(x.split(',')[0]) in (4,5,6,7)};require(len(uuids)==4 and not [x for x in apps() if x['uuid'] in uuids],'Selected external GPUs4-7 must be empty; no unknown process signals')
    token=(root/'token').read_text().strip();u=urlsplit(a.relay_url);require(u.scheme in ('http','https') and u.hostname and not u.username and not u.password and u.path in ('','/') and not u.query and not u.fragment,'Explicit expanded03 relay origin required');owner=a.node+'-g4567-external50000';request=urllib.request.Request(a.relay_url.rstrip('/')+'/progress',data=json.dumps(dict(owner=owner)).encode(),headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=15) as r:progress=json.loads(r.read(8192))
    require(sum(progress['states'].values())==50000 and progress['states'].get('unused',0)>0,'Same original50000 pool with unused sources required')
    cfg=dict(old);cfg.update(owner=owner,gpus=[4,5,6,7],queue=str(root/'queue_g4567'),config_path=str(newpath),env=dict(old['env'],CUDA_VISIBLE_DEVICES='4,5,6,7'),code_sha256={str(SELF/n):hashlib.sha256((SELF/n).read_bytes()).hexdigest() for n in ('stage2_external_native_manager.py','stage2_local_publication_v1.py','stage2_model_codec_v1.py','stage2_local_model_worker_v1.py','stage2_local_packet_v1.py','stage2_external_feeder.py','stage2_external_uploader.py','stage2_feeder_media_v1.py')});verify(cfg,str(newpath));write_once(newpath,cfg);feed=dict(json.loads((root/'feeder.json').read_bytes()),owner=owner,relay_url=a.relay_url);write_once(root/'feeder_g4567.json',feed);upload=dict(json.loads((root/'uploader.json').read_bytes()),relay_url=a.relay_url,required_owners=[old['owner'],owner]);write_once(root/'uploader_expanded.json',upload)
    # Drain only the old CPU uploader; the original model and feeder stay running.
    candidates=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
        except (OSError,UnicodeDecodeError):continue
        if '--config' in args and args[args.index('--config')+1]==str(root/'uploader.json') and any(x.endswith('/stage2_external_uploader.py') for x in args):candidates.append(int(p.name))
    require(len(candidates)==1,'Exactly one original node CPU uploader must be live');pid=candidates[0];fd=os.pidfd_open(pid);write_once(root/'expanded_uploader_handoff.json',dict(old_CPU_uploader_pid=pid,old_primary_model_kept_running=True,new_source_pool_created=False,extra_owner=owner,observed_unix=time.time()));stop=Path(cfg['out'])/'UPLOAD_STOP'
    if not stop.exists():write_once(stop,dict(reason='Replace single CPU uploader with dual-owner compatible uploader; old primary model continues',models_called=0))
    require(bool(select.select([fd],[],[],180)[0]),'Original CPU uploader not yet drained; no extra model or second uploader started');os.close(fd)
    q=Path(cfg['queue']);socket=root/'expanded_tmux.sock';session='expanded50000';launch_window(root,socket,session,'uploader-expanded',[sys.executable,'-B',str(SELF/'stage2_external_uploader.py'),'--config',str(root/'uploader_expanded.json')],True);launch_window(root,socket,session,'manager-g4567',[cfg['python'],'-B',str(SELF/'stage2_external_native_manager.py'),'--config',str(newpath)]);deadline=time.monotonic()+40
    while not (q/'live.json').exists():
        require(not (q/'failure.json').exists(),'Extra native admission failed, no new source claims');require(time.monotonic()<deadline,'Extra native admission pending; feeder not started');time.sleep(.2)
    launch_window(root,socket,session,'feeder-g4567',[sys.executable,'-B',str(SELF/'stage2_external_feeder.py'),'--config',str(root/'feeder_g4567.json')]);print(json.dumps(dict(event='external_second_TP4_admitted',node=a.node,gpus=[4,5,6,7],old_primary_kept_running=True,same_original_ledger=True,single_uploader=True,worker_identity=json.loads((q/'live.json').read_bytes())['child_identity'],model_load_pending=True,quality_approved=False,training_eligible=False)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);r=s.add_parser('relay');r.add_argument('--bind',required=True);r.add_argument('--port',type=int,default=18999);n=s.add_parser('node');n.add_argument('--node',required=True,choices=NODES);n.add_argument('--relay-url',required=True);a=p.parse_args()
    if a.mode=='relay':relay(a)
    else:node(a)
