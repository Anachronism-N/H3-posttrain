"""Launch a correctly allocated external node: original TP4 model, one uploader, one revision DB."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit
import urllib.request
from stage2_local_publication_v1 import require,write_once

CODE=Path(__file__).resolve().parent
NODES=tuple('a100-extra-0'+str(i) for i in range(1,7))


def run(a):
    import motion_revision_external_manager_v1 as m
    root=Path('/root/stage2_motion_revision_external_'+a.node+'_v1');phase='g'+a.pair;owner=a.node+'-'+phase+'-rv3main';queue=root/('queue_'+phase);cp=root/('manager_'+phase+'.json');require(not cp.exists() and not queue.exists(),'This pair configuration is already used; never restart model/claim')
    allocation_path=CODE/'allocation/allocation.json';raw=allocation_path.read_bytes();allocation=json.loads(raw);node=next(x for x in allocation['nodes'] if x['node']==a.node);require(node['source_count']==12000 and allocation['revision']=='english_merged_edges_motion_revision_v3','Exact assigned new-revision dataset required')
    u=urlsplit(a.server_origin);require(u.scheme in ('http','https') and u.hostname and not u.username and not u.password and u.path in ('','/') and not u.query and not u.fragment,'Explicit authorized server origin required');token=Path(a.token_file).read_text().strip();require(len(token)>=32,'Securely copy central token to a local file; never print it');python=Path(a.python);require(python.is_file() and Path(a.model).is_absolute(),'Existing local vLLM Python and original model paths required');require(shutil.which('tmux'),'Persistent tmux required')
    request=urllib.request.Request(a.server_origin.rstrip('/')+'/progress',data=json.dumps(dict(owner=owner)).encode(),headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=15) as f:progress=json.loads(f.read(8192))
    require(progress['source_count']==12000 and progress['revision']==allocation['revision'] and sum(progress['states'].values())==12000,'Connected endpoint is not this node allocation')
    require(m.memory()['headroom']>=160*2**30 and shutil.disk_usage('/tmp').free>=8*2**30,'Fresh160GiB effective memory and tmp8GiB required');pair=list(map(int,a.pair));inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True);uuids={line.split(',')[1].strip() for line in inventory.splitlines() if int(line.split(',')[0]) in pair};require(len(uuids)==4 and not [x for x in m.apps() if x['uuid'] in uuids],'Selected4 GPUs must be empty; first safely drain old workloads, no unknown process kill')
    if a.pair=='4567':
        first=root/'worker_state'/(a.node+'-g0123-rv3main')/'load.receipt.json';require(first.exists(),'Start and verify first TP4 before second node load budget')
        require(not (root/'queue_g0123'/'failure.json').exists(),'Primary failed: investigate before expanding')
    require(subprocess.run([str(python),'-B','-c','import torch,vllm;assert torch.cuda.is_available()'],capture_output=True).returncode==0,'Use existing working vLLM Python, not a CPU-only environment')
    root.mkdir(exist_ok=True);out=root/'out';out.mkdir(exist_ok=True);token_path=root/'token'
    if token_path.exists():require(token_path.read_text().strip()==token,'Original node token changed')
    else:
        fd=os.open(token_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'w') as f:f.write(token);f.flush();os.fsync(f.fileno())
    codes={str(CODE/name):hashlib.sha256((CODE/name).read_bytes()).hexdigest() for name in ('motion_revision_external_manager_v1.py','motion_revision_model_worker_v1.py','motion_revision_codec_v1.py','motion_revision_external_feeder_v1.py','motion_revision_external_uploader_v1.py','stage2_local_publication_v1.py')}
    cfg=dict(node=a.node,owner=owner,gpus=pair,local_root=str(root),out=str(out),queue=str(queue),config_path=str(cp),expected_hostname=subprocess.check_output(['hostname'],text=True).strip(),new_stage_authorized=True,source_count=12000,revision=allocation['revision'],allocation_path=str(allocation_path),allocation_sha256=hashlib.sha256(raw).hexdigest(),python=str(python),model_path=str(Path(a.model)),dtype='bfloat16',quantization=None,tp=4,max_num_seqs=8,env=dict(CUDA_VISIBLE_DEVICES=','.join(map(str,pair)),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1'),code_sha256=codes)
    m.verify(cfg,str(cp));write_once(cp,cfg);feeder=write_once(root/('feeder_'+phase+'.json'),dict(out=str(out),owner=owner,token_path=str(token_path),claim_endpoint=a.server_origin.rstrip('/')+'/claim',server_origin=a.server_origin,ffmpeg_path=a.ffmpeg,claim_size=8))
    if not (out/'EXPECTED_OWNERS.json').exists():write_once(out/'EXPECTED_OWNERS.json',dict(owners=node['owners'],reason='node quota shared by two sequentially admitted TP4 workers'))
    logs=root/'logs';logs.mkdir(exist_ok=True);sock=root/'tmux.sock';session='correct-allocated-motion'
    def window(name,args,first=False):
        p=logs/(name+'.log');require(not p.exists(),'Existing launch log cannot be overwritten');cmd=shlex.join(args)+' > '+shlex.quote(str(p))+' 2>&1';argv=['tmux','-S',str(sock),'new-session','-d','-s',session,'-n',name,'-c','/tmp',cmd] if first else ['tmux','-S',str(sock),'new-window','-d','-t',session,'-n',name,'-c','/tmp',cmd];subprocess.run(argv,check=True)
    if a.pair=='0123':
        up=write_once(root/'uploader.json',dict(out=str(out),node=a.node,shared_root=a.shared_root,relay_url=a.server_origin,token_path=str(token_path)));window('uploader',[sys.executable,'-B',str(CODE/'motion_revision_external_uploader_v1.py'),'--config',up['path']],True)
    else:require((root/'uploader.json').exists(),'One node uploader must already exist')
    window('manager_'+phase,[str(python),'-B',str(CODE/'motion_revision_external_manager_v1.py'),'--config',str(cp)]);deadline=time.monotonic()+40
    while not (queue/'live.json').exists():require(time.monotonic()<deadline and not (queue/'failure.json').exists(),'Native admission not complete; no feeder start');time.sleep(.2)
    window('feeder_'+phase,[sys.executable,'-B',str(CODE/'motion_revision_external_feeder_v1.py'),'--config',feeder['path']]);write_once(root/('launch_'+phase+'.json'),dict(node=a.node,owner=owner,gpus=pair,allocation_sha256=cfg['allocation_sha256'],source_count=12000,model_loading=True,one_node_uploader=True,same_central_source_ledger=True,observed_unix=time.time()))
    print(json.dumps(dict(event='correct_revision_external_node_admitted',node=a.node,pair=a.pair,allocated_sources=12000,revision=cfg['revision'],worker_identity=json.loads((queue/'live.json').read_bytes())['child_identity'],model_loading=True,one_node_uploader=True,quality_approved=False,training_eligible=False)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',required=True,choices=NODES);p.add_argument('--pair',required=True,choices=('0123','4567'));p.add_argument('--server-origin',required=True);p.add_argument('--token-file',required=True);p.add_argument('--python',required=True);p.add_argument('--model',required=True);p.add_argument('--ffmpeg',default='/usr/local/bin/ffmpeg');p.add_argument('--shared-root',required=True);run(p.parse_args())
