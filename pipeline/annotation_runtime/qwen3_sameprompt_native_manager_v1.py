"""Local dynamic native manager: original PIDFD/FLOCK/lease lifecycle, all hot I/O local."""
import argparse
import importlib.util
import json
import hashlib
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


def pure_model_check(cfg):
    ref=cfg['modelref'];need(b.sha(ref['path'])==ref['sha256'],'Pure-LM reference changed');v=json.loads(Path(ref['path']).read_bytes());model=Path(v['root']);need(model==Path('/opt/models/Qwen3-32B') and v['model_identity_sha256']==ref['model_identity_sha256'],'Bound pure-LM identity differs')
    for name,size in v['files'].items():need((model/name).is_file() and (model/name).stat().st_size==size,'Pure-LM resource incomplete: '+name)
    need(b.sha(model/'config.json')==v['config_sha256'] and b.sha(model/'model.safetensors.index.json')==v['index_sha256'],'Pure-LM config/index changed')
    need(json.loads((model/'config.json').read_bytes())['model_type']=='qwen3','Expected pure Qwen3, not VL')


def worker(path):
    from stage2_local_publication_v1 import checked,write_once
    import ms_llm32_codec_v4 as codec
    cfg=json.loads(Path(path).read_bytes());cfg['_config_sha']=b.sha(path);worker_gate(cfg,True);pure_model_check(cfg)
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    import xgrammar
    inputs=checked(cfg['diagnostic_manifest_ref']);need(inputs['source_count']==3,'Exact fixed-three diagnostic only');contexts=[];requests=[];schemas=[]
    for ref in inputs['entries']:
        x=checked(ref);need(x['context']['row']['id']==x['id'],'Diagnostic source differs');request=x['request'];need(hashlib.sha256(json.dumps(request,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==x['prompt_messages_sha256'],'Exact historical prompt changed');need(x['schema']==codec.response_schema(x['context']),'Same historical schema required');xgrammar.Grammar.from_json_schema(json.dumps(x['schema']),strict_mode=True);contexts.append(x);requests.append(request);schemas.append(x['schema'])
    llm=LLM(model='/opt/models/Qwen3-32B',dtype='bfloat16',tensor_parallel_size=2,max_num_seqs=32,max_model_len=32768,gpu_memory_utilization=.88,structured_outputs_config={'backend':'xgrammar','disable_any_whitespace':True},enable_prefix_caching=True,trust_remote_code=False,seed=0,enforce_eager=True)
    for shard in Path('/opt/models/Qwen3-32B').glob('*.safetensors'):
        fd=os.open(shard,os.O_RDONLY|os.O_NOFOLLOW)
        try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        finally:os.close(fd)
    worker_gate(cfg);engine=llm.llm_engine.vllm_config;need(str(engine.model_config.dtype)=='torch.bfloat16' and engine.model_config.quantization is None and engine.parallel_config.tensor_parallel_size==2,'Original BF16TP2 pure-LM comparison')
    root=Path(cfg['local_root']);state=root/'worker_state'/cfg['owner']/cfg['phase'];write_once(state/'load.receipt.json',dict(model_load_completed=True,model_init_count=1,model='/opt/models/Qwen3-32B',dtype='bfloat16',tp=2,max_num_seqs=32,context_length=32768,enable_thinking=False,loaded_unix=time.time(),diagnostic_only=True))
    for x in contexts:write_once(root/'budgets'/x['id']/'Qwen3_sameprompt.claim.json',dict(source=x['id'],stage='Qwen3_sameprompt_three_v1',maximum_calls=1,automatic_retry=False,no_production_pool_claim=True,original_prompt_ref=x['original_request_ref']))
    params=[SamplingParams(temperature=0,max_tokens=4096,structured_outputs=StructuredOutputsParams(json=schema)) for schema in schemas];begin=time.time();wall=time.monotonic();values=llm.chat(requests,sampling_params=params,use_tqdm=False,chat_template_kwargs={'enable_thinking':False});need(len(values)==3,'Response source mapping differs');elapsed=time.monotonic()-wall;rows=[]
    for x,value in zip(contexts,values):
        need(len(value.outputs)==1,'One candidate');v=value.outputs[0];raw=write_once(root/'results'/(x['id']+'.raw.json'),dict(id=x['id'],raw_text=v.text,finish_reason=v.finish_reason,completion_tokens=len(v.token_ids),prompt_sha256=x['prompt_messages_sha256'],model='/opt/models/Qwen3-32B',enable_thinking=False,original_reference=x['original_result_ref'],same_messages=True,same_schema=True));parsed=None;error=None
        try:
            need(v.finish_reason=='stop' and 0<len(v.token_ids)<4096,'Incomplete output, no retry');parsed=json.loads(v.text);codec.validate(parsed,x['context']);parsed=codec.validate(codec.normalize(parsed,x['context']),x['context'])
        except Exception as e:error=repr(e)
        result=write_once(root/'results'/(x['id']+'.json'),dict(id=x['id'],raw_ref=raw,parsed=parsed,error=error,source_original_events=x['context']['draft']['events'],original_caption=x['original_caption'],same_messages=True,same_schema=True,same_temperature=True,enable_thinking=False,max_tokens=4096,quality_approved=False,human_review_completed=False))
        rows.append(dict(id=x['id'],result_ref=result,error=error))
    result=write_once(root/'comparison_results.json',dict(source_count=3,rows=rows,model='Qwen3-32B',reference_model='Qwen3-VL-32B-Instruct language-only',same_prompt=True,same_schema=True,temperature=0,max_tokens=4096,thinking=False,generation_wall_seconds=elapsed,original_pool_claims=0,new_diagnostic_model_calls=3,quality_approved=False,completed_unix=time.time()))
    write_once(state/'summary.json',dict(status='completed',owner=cfg['owner'],processed=0,all_outcomes=[],model_init_count=1,model_requests=3,diagnostic_outcomes_ref=result,diagnostic_case_count=3,begin_unix=begin,end_unix=time.time(),quality_approved=False,training_eligible=False))
    print(json.dumps(dict(event='Qwen3_sameprompt_three_complete',sources=3,wall_seconds=elapsed,errors=[r for r in rows if r['error']],report_ref=result)),flush=True)


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
    b.check_model=pure_model_check
    b.EXPERIMENT=Path(cfg['local_root']);b.headroom=headroom;b.validate_pair=validate;b.check_sources=check_sources;b.selected_sources=lambda c:([],c['input_count']);b.stage_summaries=summaries
    b.main()

if __name__=='__main__':main()
