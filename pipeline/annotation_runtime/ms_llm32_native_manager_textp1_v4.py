"""Local dynamic native manager: original PIDFD/FLOCK/lease lifecycle, all hot I/O local."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

sys.dont_write_bytecode=True
SELF=Path(__file__).resolve()
SPEC=importlib.util.spec_from_file_location('local_native_base',SELF.parent/'resource_manager.py')
b=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(b)
ORIGINAL_BASE_SHA='6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
HOSTS={'N1':'28.216.19.69','N2':'28.216.17.70','N3':'28.216.19.213','N4':'28.216.19.143','N5':'28.216.19.137','N6':'28.216.19.225','N7':'28.216.18.144','N8':'28.216.18.136'}


def need(ok,msg):
    if not ok:raise ValueError(msg)


def local_path(p):
    p=Path(p);need(p.is_absolute() and str(p).startswith('/root/stage2_') and p.resolve()==p,'Stable owned node-local storage required');return p


def headroom():
    p=Path('/sys/fs/cgroup/memory');cur=int((p/'memory.usage_in_bytes').read_text());maximum=int((p/'memory.limit_in_bytes').read_text());available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    return dict(current=cur,maximum=str(maximum),headroom=min(maximum-cur,available))


def check_sources(cfg,path,sha,manager_sha):
    need(b.sha(b.__file__)==ORIGINAL_BASE_SHA and b.sha(path)==sha and manager_sha==ORIGINAL_BASE_SHA,'Original native lifecycle and local config SHA required')
    need(cfg['schema']=='stage2_local_dynamic_native_config_v1' and cfg['producer']==cfg['resource_manager_ref'] and cfg['producer']['path']==str(SELF),'Exact local native producer required')
    for name,digest in cfg['source_sha256'].items():
        p=local_path(name);need(b.sha(p)==digest,'Frozen local code/model reference changed: '+name)
    need(cfg['config_path']==str(path) and cfg['argv']==['/opt/venv_vllm/bin/python','-B',str(SELF),'worker','--config',str(path)],'Exact local worker argv required')
    need(cfg['env']['CUDA_VISIBLE_DEVICES']==','.join(map(str,cfg['gpus'])) and cfg['cwd']=='/tmp','Exact admitted device/cwd required')
    local_path(cfg['out']);local_path(cfg['queue']);local_path(cfg['modelref']['path'])
    local_path(cfg['plan']['path']);need(cfg['source_sha256'].get(cfg['plan']['path'])==cfg['plan']['sha256'] and b.sha(cfg['plan']['path'])==cfg['plan']['sha256'],'Frozen local dynamic pool provenance plan required')
    need(cfg['producer']['path'] in cfg['source_sha256'] and cfg['source_sha256'][cfg['producer']['path']]==cfg['producer']['sha256'],'Frozen local producer binding differs')


def validate(cfg):
    pair=cfg['gpus'];need(cfg['node'] in HOSTS and tuple(pair) in ((0,1),(2,3),(4,5),(6,7)) and all(type(g) is int for g in pair),'Explicit authorized physical pair required')
    need(HOSTS[cfg['node']] in b.subprocess.check_output(['hostname','-I'],text=True).split(),'Wrong physical node')
    need(cfg['release_authorized'] is True and cfg['new_stage_authorized'] is True and type(cfg['targetcount']) is int and 1<=cfg['targetcount']<=500000,'Explicit source and holder release authorization required')
    need(len(cfg['holders'])==2 and [x['gpu'] for x in cfg['holders']]==pair and len({x['pid'] for x in cfg['holders']})==2,'Exact ordered holder identities required')
    need(not local_path(cfg['queue']).exists(),'Fresh exclusive queue required, never restart')
    need(Path(cfg['queue']).parent==Path(cfg['local_root'])/'queues' and Path(cfg['out'])==Path(cfg['local_root'])/'out','Owned local lifecycle paths required')
    need(shutil.disk_usage('/tmp').free>=8*2**30 and headroom()['headroom']>=160*2**30,'Fresh160GiB and tmp8GiB admission required')
    for g,owner in zip(pair,cfg['holders']):b.check_holder(owner,b.identity(owner['pid']),g,b.uuid_for(g),b.apps());b.lease_snapshot(Path('/tmp/od_reserve_%d'%g))


def worker_gate(cfg,before_load=False):
    need(os.getcwd()==cfg['cwd'] and all(os.environ.get(k)==v for k,v in cfg['env'].items()),'Actual local worker env/cwd differs')
    check_sources(cfg,cfg['config_path'],cfg['_config_sha'],ORIGINAL_BASE_SHA)
    q=Path(cfg['queue']);admit=json.loads((q/'admission.json').read_bytes());parent=b.identity(os.getppid());current=b.identity(os.getpid());expected=admit['manager_identity']
    need(parent['pid']==expected['pid'] and all(parent[k]==expected[k] for k in b.IDENTITY_KEYS) and parent['uid']==current['uid']==os.getuid() and parent['cgroup']==current['cgroup'],'Exact native parent identity/UID/cgroup required')
    need(parent['argv']==['/opt/venv_vllm/bin/python','-B',str(SELF),'--config',cfg['config_path'],'--config-sha',cfg['_config_sha'],'--manager-sha',ORIGINAL_BASE_SHA] and current['argv']==cfg['argv'],'Exact manager and worker argv required')
    deadline=time.monotonic()+20
    while not (q/'live.json').exists():need(time.monotonic()<deadline,'Native live receipt unavailable');time.sleep(.05)
    live=json.loads((q/'live.json').read_bytes());need(all(live['child_identity'][k]==current[k] for k in b.IDENTITY_KEYS),'Exact live native worker identity required')
    fds=[]
    for p in (Path('/proc')/str(parent['pid'])/'fd').iterdir():
        try:fds.append((os.readlink(p),p.stat()))
        except FileNotFoundError:pass
    locks=Path('/proc/locks').read_text().splitlines();uuids=[]
    for gpu in cfg['gpus']:
        uuid=b.uuid_for(gpu);uuids.append(uuid);expected_holder=cfg['holders'][cfg['gpus'].index(gpu)];need(uuid==expected_holder['uuid'],'Physical GPU UUID changed')
        p=Path('/tmp/ms12k_gpu_%d.lock'%gpu);st=p.lstat();need(any(name==str(p) and v.st_dev==st.st_dev and v.st_ino==st.st_ino for name,v in fds),'Parent native lock FD missing')
        owner=False
        for line in locks:
            a=line.split()
            if len(a)<6 or a[1]!='FLOCK' or a[4]!=str(parent['pid']):continue
            x,y,z=a[5].split(':')
            if (int(x,16),int(y,16),int(z))==(os.major(st.st_dev),os.minor(st.st_dev),st.st_ino):owner=True
        need(owner,'Exact kernel FLOCK owner missing')
        p=Path('/tmp/od_reserve_%d'%gpu);st=p.lstat();need(st.st_size==0 and st.st_uid==os.getuid() and time.time()-st.st_mtime<30 and any(name==str(p) and v.st_dev==st.st_dev and v.st_ino==st.st_ino for name,v in fds),'Owned lease FD/live heartbeat missing')
    for app in b.apps():
        if app['uuid'] not in uuids:continue
        need(not before_load,'GPU occupied before admitted model load');pid=app['pid']
        for _ in range(20):
            if pid==current['pid']:break
            ancestor=b.identity(pid);need(ancestor['uid']==current['uid'] and ancestor['cgroup']==current['cgroup'],'Unknown GPU app owner');pid=ancestor['ppid']
        need(pid==current['pid'],'GPU app outside this worker ancestry')
    need(headroom()['headroom']>=80*2**30,'Original active safety reserve required')


def worker(path):
    from stage2_local_publication_v1 import checked,write_once
    import ms_llm32_worker_v4 as w
    import ms_llm32_codec_v4 as codec
    cfg=json.loads(Path(path).read_bytes());cfg['_config_sha']=b.sha(path);worker_gate(cfg,True);llm,sampler=codec.runtime()
    for shard in Path(codec.MODEL).glob('*.safetensors'):
        fd=os.open(shard,os.O_RDONLY|os.O_NOFOLLOW)
        try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        finally:os.close(fd)
    worker_gate(cfg);engine=llm.llm_engine.vllm_config
    need(str(engine.model_config.dtype)=='torch.bfloat16' and engine.model_config.quantization is None and engine.parallel_config.tensor_parallel_size==2 and engine.scheduler_config.max_num_seqs==32 and engine.model_config.max_model_len==32768,'Original32B BF16TP2 text-only32 recipe')
    state=Path(cfg['local_root'])/'worker_state'/cfg['owner']/cfg['phase'];write_once(state/'load.receipt.json',dict(model_load_completed=True,model_init_count=1,revision=codec.REVISION,dtype='bfloat16',quantization=None,tp=2,max_num_seqs=32,context_length=32768,image_inputs=0,video_inputs=0,loaded_unix=time.time()));queue=Path(cfg['out'])/'ready'/cfg['owner'];outcomes=[];idle=0.;begin=time.time()
    while True:
        worker_gate(cfg);ready=[p for p in sorted(queue.glob('*.ready.json')) if not (Path(cfg['out'])/'consumed_ready'/cfg['owner']/p.name).exists()];failed=(queue/'FEEDER_FAILED_TEXTP1').exists();drained=(queue/'FEEDER_DRAINED_TEXTP1').exists()
        if not ready and (drained or failed):break
        if not ready or len(ready)<4 and not (drained or failed):time.sleep(.5);idle+=.5;continue
        selected=ready[:4];references=[checked(dict(path=str(p),sha256=b.sha(p)))['packet_ref'] for p in selected];outcomes+=w.run_packets(references,cfg['out'],cfg['owner'],llm,sampler,lambda:worker_gate(cfg))
        for p,reference in zip(selected,references):write_once(Path(cfg['out'])/'consumed_ready'/cfg['owner']/p.name,dict(packet_ref=reference,revision=codec.REVISION,source_outputs_published_locally=True,models_already_called=True))
        write_once(state/('progress%06d.json'%len(outcomes)),dict(revision=codec.REVISION,processed=len(outcomes),outcomes=outcomes,model_requests=len(outcomes),idle_seconds=idle,observed_unix=time.time(),quality_approved=False,training_eligible=False))
    write_once(Path(cfg['out'])/'generation_drained_textp1'/cfg['owner'],dict(owner=cfg['owner'],generation_complete=True,feeder_failed=failed));write_once(state/'summary.json',dict(schema='stage2_local_dynamic_worker_summary_v1',revision=codec.REVISION,status='completed',owner=cfg['owner'],processed=len(outcomes),all_outcomes=outcomes,model_init_count=1,model_requests=len(outcomes),idle_seconds=idle,begin_unix=begin,end_unix=time.time(),quality_approved=False,training_eligible=False))


def summaries(cfg,out):
    from stage2_local_publication_v1 import checked
    p=Path(cfg['local_root'])/'worker_state'/cfg['owner']/cfg['phase']/'summary.json';need(p.is_file(),'Worker exited before summary; see '+str(Path(cfg['queue'])/'worker.runtime.log'));x=json.loads(p.read_bytes());need(x['owner']==cfg['owner'] and x['status']=='completed' and x['model_init_count']==1,'Complete local dynamic worker summary required')
    keys=set()
    for item in x['all_outcomes']:
        key=(item['split'],item['id']);need(key not in keys,'Duplicate source output');keys.add(key);c=checked(item['commit_ref']);r=checked(c['result_ref']);need(c['id']==r['id']==item['id'] and c['split']==r['split']==item['split'] and c['status']==r['status']==item['status'] and c['owner']==cfg['owner'] and c['quality_approved'] is False and c['training_eligible'] is False and c['new_model_calls']==len(r['stages'])<=1,'Local commit/result/budget mapping differs')
    need(x['processed']==len(keys),'Dynamic actual processed count differs');return {str(p):x}


def main():
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        p=argparse.ArgumentParser();p.add_argument('--config',required=True);worker(p.parse_args(sys.argv[2:]).config);return
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);p.add_argument('--manager-sha',required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_bytes());need(a.manager_sha==ORIGINAL_BASE_SHA,'Original lifecycle hash required')
    check_sources(cfg,a.config,a.config_sha,a.manager_sha);validate(cfg)
    b.EXPERIMENT=Path(cfg['local_root']);b.headroom=headroom;b.validate_pair=validate;b.check_sources=check_sources;b.selected_sources=lambda c:([],c['input_count']);b.stage_summaries=summaries
    b.main()

if __name__=='__main__':main()
