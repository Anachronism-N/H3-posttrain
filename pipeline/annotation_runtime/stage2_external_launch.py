"""One-time03 pool initialization and persistent external TP4 node deployment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit
import urllib.request
from stage2_local_publication_v1 import write_once,require

SELF=Path(__file__).resolve().parent
MANIFEST_SHA='5a25521c6908ed3852ebba54242b79e718f73a01444ade4260ebfaebf39f2ce1'
KEYS_SHA='9e5dad34ec7c7e10ec40b477b39ee9592b88c52c46b42e6535dab5b402e7a9fc'
NODES=tuple('a100-extra-0'+str(i) for i in range(3,7))


def raw_once(p,raw):
    p.parent.mkdir(parents=True,exist_ok=True);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    d=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(d);os.close(d)


def tmux(root,session,windows):
    require(shutil.which('tmux') is not None,'Install tmux for persistent execution')
    for index,(name,argv) in enumerate(windows):
        log=root/(name+'.log');require(not log.exists(),'Existing launch logs must not be overwritten');cmd=shlex.join(argv)+' > '+shlex.quote(str(log))+' 2>&1'
        if index==0:a=['tmux','-S',str(root/'tmux.sock'),'new-session','-d','-s',session,'-n',name,'-c','/tmp',cmd]
        else:a=['tmux','-S',str(root/'tmux.sock'),'new-window','-d','-t',session,'-n',name,'-c','/tmp',cmd]
        subprocess.run(a,check=True)


def relay(a):
    from stage2_dynamic_pool_v1 import DynamicPool
    root=Path('/root/stage2_external50000_relay_v1');require(not root.exists(),'Existing external central pool cannot be reinitialized');raw=(SELF/'sources/manifest.jsonl').read_bytes();keys=(SELF/'sources/source_keys.jsonl').read_bytes();require(hashlib.sha256(raw).hexdigest()==MANIFEST_SHA and hashlib.sha256(keys).hexdigest()==KEYS_SHA,'Frozen50000 reservation changed');rows=[json.loads(x) for x in raw.splitlines() if x];require(len(rows)==50000 and [[r['split'],r['id']] for r in rows]==[json.loads(x) for x in keys.splitlines() if x],'Exact source list/key mapping required');root.mkdir();raw_once(root/'token',secrets.token_hex(32).encode());raw_once(root/'manifest.jsonl',raw);p=DynamicPool.create(root/'pool.db',rows,[]);p.close();shared=Path(a.shared_root);shared.mkdir(parents=True,exist_ok=True)
    cfg=dict(local_root=str(root),database=str(root/'pool.db'),token_path=str(root/'token'),bind=a.bind,port=a.port,shared_root=str(shared),allowed_owners=[n+'-g0123-external50000' for n in NODES]);write_once(root/'relay.json',cfg);write_once(root/'deployment.json',dict(source_count=50000,manifest_sha256=MANIFEST_SHA,source_keys_sha256=KEYS_SHA,source_claims_created=0,one_central_pool=True,observed_unix=time.time()))
    tmux(root,'external-relay',[('relay',[sys.executable,'-B',str(SELF/'stage2_external_relay.py'),'--config',str(root/'relay.json')])]);print(json.dumps(dict(event='external50000_relay_started',root=str(root),port=a.port,token_file=str(root/'token'),source_count=50000,source_claims_created=0)),flush=True)


def node(a):
    from stage2_external_native_manager import verify,memory,apps
    require(a.node in NODES,'Explicit external03-06 node label required');root=Path('/root/stage2_external50000_'+a.node+'_v1');require(not root.exists(),'Used external node configuration must never restart');out=root/'out';python=Path(a.python) if a.python else Path(sys.executable);require(python.is_file(),'Explicit local vLLM Python required');model=Path(a.model);require(model.is_absolute() and model.resolve()==model,'Explicit complete local original model required');token=Path(a.token_file).read_bytes();require(len(token.strip())>=32,'Copy03 relay token to this node without printing it')
    require(subprocess.run([str(python),'-B','-c','import vllm,torch; assert torch.cuda.is_available()'],capture_output=True).returncode==0,'Use a local Python with working vLLM/CUDA')
    require(memory()['headroom']>=160*2**30 and shutil.disk_usage('/tmp').free>=8*2**30,'Fresh memory160GiB and tmp8GiB required')
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True);uuids={line.split(',')[1].strip() for line in inventory.splitlines() if int(line.split(',')[0]) in (0,1,2,3)};require(len(uuids)==4 and not [x for x in apps() if x['uuid'] in uuids],'External selected four GPUs must be empty')
    u=urlsplit(a.relay_url);require(u.scheme in ('http','https') and u.hostname and not u.username and not u.password and u.path in ('','/') and not u.query and not u.fragment,'Explicit03 relay origin required')
    request=urllib.request.Request(a.relay_url.rstrip('/')+'/progress',data=json.dumps(dict(owner=a.node+'-g0123-external50000')).encode(),headers={'Authorization':'Bearer '+token.decode().strip(),'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=15) as response:state=json.loads(response.read(8192))
    require(sum(state['states'].values())==50000,'Relay is not the exact50000 source group')
    root.mkdir();out.mkdir();raw_once(root/'token',token)
    owner=a.node+'-g0123-external50000';q=root/'queue';cfg=dict(node=a.node,owner=owner,gpus=[0,1,2,3],local_root=str(root),out=str(out),queue=str(q),config_path=str(root/'manager.json'),expected_hostname=subprocess.check_output(['hostname'],text=True).strip(),new_stage_authorized=True,source_count=50000,source_manifest_sha256=MANIFEST_SHA,model_path=str(model),python=str(python),dtype='bfloat16',quantization=None,tp=4,max_num_seqs=32,env=dict(CUDA_VISIBLE_DEVICES='0,1,2,3',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1'),code_sha256={str(SELF/p):hashlib.sha256((SELF/p).read_bytes()).hexdigest() for p in ('stage2_external_native_manager.py','stage2_local_publication_v1.py','stage2_model_codec_v1.py','stage2_local_model_worker_v1.py','stage2_local_packet_v1.py','stage2_external_feeder.py','stage2_external_uploader.py','stage2_feeder_media_v1.py')})
    write_once(root/'manager.json',cfg);verify(cfg,cfg['config_path']);require(memory()['headroom']>=160*2**30 and shutil.disk_usage('/tmp').free>=8*2**30,'Fresh memory160GiB and tmp8GiB required')
    feed=dict(out=str(out),owner=owner,relay_url=a.relay_url,token_path=str(root/'token'),ffmpeg_path=a.ffmpeg);write_once(root/'feeder.json',feed);upload=dict(out=str(out),node=a.node,owner=owner,shared_root=a.shared_root,relay_url=a.relay_url,token_path=str(root/'token'));write_once(root/'uploader.json',upload)
    (out/'ready'/owner).mkdir(parents=True);tmux(root,'external50000',[('manager',[str(python),'-B',str(SELF/'stage2_external_native_manager.py'),'--config',str(root/'manager.json')]),('uploader',[sys.executable,'-B',str(SELF/'stage2_external_uploader.py'),'--config',str(root/'uploader.json')])])
    deadline=time.monotonic()+30
    while not (q/'live.json').exists():
        require(not (q/'failure.json').exists(),'Native admission failed; preserve old queue and do not claim');require(time.monotonic()<deadline,'Native admission still pending; feeder not started');time.sleep(.2)
    cmd=shlex.join([sys.executable,'-B',str(SELF/'stage2_external_feeder.py'),'--config',str(root/'feeder.json')])+' > '+shlex.quote(str(root/'feeder.log'))+' 2>&1';subprocess.run(['tmux','-S',str(root/'tmux.sock'),'new-window','-d','-t','external50000','-n','feeder','-c','/tmp',cmd],check=True);print(json.dumps(dict(event='external50000_node_admitted',node=a.node,root=str(root),worker_identity=json.loads((q/'live.json').read_bytes())['child_identity'],TP=4,model_load_pending=True,quality_approved=False,training_eligible=False)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True);r=sub.add_parser('relay');r.add_argument('--bind',required=True);r.add_argument('--port',type=int,default=18997);r.add_argument('--shared-root',required=True);n=sub.add_parser('node');n.add_argument('--node',required=True,choices=NODES);n.add_argument('--relay-url',required=True);n.add_argument('--token-file',required=True);n.add_argument('--model',required=True);n.add_argument('--python');n.add_argument('--ffmpeg',default='/usr/local/bin/ffmpeg');n.add_argument('--shared-root',required=True);a=p.parse_args()
    if a.mode=='relay':relay(a)
    else:node(a)
