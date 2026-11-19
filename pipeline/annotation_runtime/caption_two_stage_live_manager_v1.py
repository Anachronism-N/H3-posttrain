"""Native managed GPU lane generating unreviewed drafts with basic input checks."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
sys.dont_write_bytecode=True
RT=Path(__file__).resolve().parent
BAR=RT.parent
SELF=Path(__file__).resolve()
CPU_SITE=os.environ.get('MS_VISUAL_CPU_SITEPATH','/opt/venv_eval/lib/python3.11/site-packages')
if CPU_SITE not in sys.path:sys.path.append(CPU_SITE)
sys.path.insert(0,str(RT))
import caption_two_stage_light_v1 as light
s=importlib.util.spec_from_file_location('light_native_base',RT/'resource_manager.py')
base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
s=importlib.util.spec_from_file_location('light_ownership_helpers',RT/'bench_manager_driver_v2.py')
h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
BASE_SHA='6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
HOSTS={'N4':'28.216.19.143','N2':'28.216.17.70','N7':'28.216.18.144'}
MODEL='/opt/models/Qwen3-VL-32B-Instruct'


def ref(path):return dict(path=str(Path(path).resolve()),sha256=base.sha(path))
def selected(cfg):
    rows=light.load(cfg['plan']['path'])
    light.need(rows and len({(r['split'],r['id']) for r in rows})==len(rows),'Unique remaining source assignment required')
    return [r['id'] for r in rows],len(rows)


def validate(cfg):
    light.need(base.sha(base.__file__)==BASE_SHA and cfg['producer']==ref(SELF),'Original base and separate worker identity required')
    light.need(cfg['node'] in HOSTS and HOSTS[cfg['node']] in h.subprocess.check_output(['hostname','-I'],text=True).split(),
        'Wrong physical execution node')
    light.need(cfg['gpus'] in ([0,1],[2,3],[4,5],[6,7]) and cfg['release_authorized'] is True,
        'Explicit pair release authorization required')
    light.need(cfg['env']['CUDA_VISIBLE_DEVICES']==','.join(map(str,cfg['gpus'])),'GPU assignment differs')
    light.need(cfg['argv']==['/opt/venv_vllm/bin/python','-B',str(SELF),'worker','--config',cfg['config_path']],
        'Worker command differs')
    light.need(cfg['targetcount']==selected(cfg)[1] and Path(cfg['draft_out']).parent==BAR
        and Path(cfg['draft_out']).name.startswith('caption_two_stage_live_'),'Draft assignment/output differs')
    previous=light.load(cfg['previous_safe']['path'])
    light.need(base.sha(cfg['previous_safe']['path'])==cfg['previous_safe']['sha256'] and
        all(previous.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored')),
        'Previous lane resources must be safely restored')
    original=light.load(cfg['previous_config']['path'])
    light.need(original['node']==cfg['node'] and original['gpus']==cfg['gpus'] and original['out']==cfg['old_out'],
        'Exact old lane source/GPU assignment differs')
    old_live=light.load(Path(original['queue'])/'live.json')['child_identity']
    try:current=base.identity(old_live['pid'])
    except FileNotFoundError:current=None
    light.need(current is None or current['start_ticks']!=old_live['start_ticks'],'Old consumer still owns selected sources')
    for gpu,owner in zip(cfg['gpus'],cfg['holders']):
        restored=light.load(Path(original['queue'])/('holder_restored_gpu%d.json'%gpu))
        light.need(restored['uuid']==owner['uuid'] and all(restored['holder_identity'][k]==owner[k] for k in base.IDENTITY_KEYS),
            'Current capture differs from safely restored holder')
    light.need(Path(cfg['old_out'],'completed.jsonl').read_bytes()==Path(cfg['old_completed_prefix']).read_bytes(),
        'Old lane resumed after source handoff')
    for owner,gpu in zip(cfg['holders'],cfg['gpus']):
        base.check_holder(owner,base.identity(owner['pid']),gpu,base.uuid_for(gpu),base.apps())
    h.need(not any(h.lock_snapshot(g)['held'] for g in cfg['gpus']),'GPU pair already locked')


def gate(cfg,before_load=False):
    globals_=dict(h.parent_gate.__globals__,SELF=SELF,base=base,NODE=cfg['node'],GPUS=cfg['gpus'],CVD=cfg['env']['CUDA_VISIBLE_DEVICES'])
    fn=types.FunctionType(h.parent_gate.__code__,globals_,'parent_gate',h.parent_gate.__defaults__)
    return fn(cfg,before_load)


def actual_state(llm):
    vc=llm.llm_engine.vllm_config
    light.need(vc.parallel_config.tensor_parallel_size==2 and vc.scheduler_config.max_num_seqs==8
        and vc.model_config.multimodal_config.mm_processor_kwargs=={'max_pixels':200704}
        and vc.structured_outputs_config.disable_any_whitespace is True,'Loaded model/runtime budget differs')


def worker(config_path):
    cfg=light.load(config_path)
    cfg.update(_config_path=str(Path(config_path).resolve()),_config_sha=base.sha(config_path))
    gate(cfg,True)
    rows=light.load(cfg['plan']['path']);out=Path(cfg['draft_out'])
    light.write_once(out/'assignment.json',dict(recipe=light.RECIPE,source_rows=rows,
        old_out=cfg['old_out'],original_config_resume_forbidden=True,**light.FLAGS))
    from vllm import LLM,SamplingParams
    llm=LLM(model=MODEL,dtype='bfloat16',tensor_parallel_size=2,max_num_seqs=8,max_model_len=16384,
        gpu_memory_utilization=.88,limit_mm_per_prompt={'image':16,'video':0},mm_processor_kwargs={'max_pixels':200704},
        structured_outputs_config={'backend':'xgrammar','disable_any_whitespace':True},enable_prefix_caching=True,
        trust_remote_code=False,seed=0)
    actual_state(llm);llm._ms_bar_model_sha256=cfg['modelref']['model_identity_sha256']
    args=argparse.Namespace(stage='run',batch_size=8,max_num_seqs=8,retry_failed=False,model=MODEL,
        visual_max_tokens=1800,hand_roi_root=str(Path(cfg['source_root'])/'roi'),sample_fps=4.,ffmpeg='/usr/local/bin/ffmpeg')
    counts={};done=set();pending=[];metadata=light.Inputs();start=time.perf_counter()
    for row in rows:
        draft_path=out/'drafts'/row['split']/row['id']/'draft.json'
        if draft_path.exists():
            draft=light.load(draft_path)
            light.need(draft['id']==row['id'] and draft['split']==row['split'] and draft['training_eligible'] is False,
                'Existing draft source/approval differs')
            done.add((row['split'],row['id']));status=draft['status'];counts[status]=counts.get(status,0)+1
    for row in rows:
        sid,split=row['id'],row['split']
        old_terminal=Path(cfg['old_out'])/'sources'/split/sid/'consumer.terminal.json'
        light.need(not old_terminal.exists(),'Old source already processed; do not consume budget again')
    def runtime():return llm,SamplingParams
    def admitted(a,r):return gate(cfg)['admitted'] is True
    def group(entries):
        nonlocal metadata
        result=light.generate_batch(entries,args,runtime,out,admitted)
        for draft in result['drafts']:
            done.add((draft['split'],draft['id']));status=draft['status'];counts[status]=counts.get(status,0)+1
        progress=dict(recipe=light.RECIPE,node=cfg['node'],gpus=cfg['gpus'],processed=len(done),selected=len(rows),
            counts=counts,new_window_calls=result['new_model_calls'],last_chat_batch_sizes=result['chat_batch_sizes'],
            wall_seconds=time.perf_counter()-start,UTC=time.strftime('%FT%TZ',time.gmtime()),**light.FLAGS)
        base.atomic_write(out/'progress.json',light.encoded(progress)) if hasattr(base,'atomic_write') else light.bar_module().batch.atomic_write(out/'progress.json',light.encoded(progress))
        print(json.dumps(progress),flush=True);metadata=light.Inputs()
    while len(done)<len(rows):
        progress=False
        for row in rows:
            identity=(row['split'],row['id'])
            if identity in done or any((x['row']['manifest_record']['split'],x['row']['id'])==identity for x in pending):continue
            producer=Path(cfg['source_root'])/'sources'/row['split']/row['id']/'producer.terminal.json'
            if not producer.exists():continue
            try:
                p=metadata.json(producer)
                if p['status']=='failed':
                    draft=dict(id=row['id'],split=row['split'],status='draft_needs_regeneration',reason=p.get('error','producer_failed'),**light.FLAGS)
                    light.write_once(out/'drafts'/row['split']/row['id']/'draft.json',draft);done.add(identity)
                    counts[draft['status']]=counts.get(draft['status'],0)+1;continue
                entry=light.load_entry(cfg['source_root'],row,out,metadata)
            except (OSError,ValueError,KeyError,TypeError) as error:
                light.write_once(out/'drafts'/row['split']/row['id']/'draft.json',dict(id=row['id'],split=row['split'],
                    status='draft_needs_regeneration',reason=type(error).__name__+': '+str(error),**light.FLAGS))
                done.add(identity);counts['draft_needs_regeneration']=counts.get('draft_needs_regeneration',0)+1;continue
            pending.append(entry);progress=True
            if len(pending)==3:group(pending);pending=[]
        if pending:group(pending);pending=[]
        if not progress and len(done)<len(rows):time.sleep(10)
    report=dict(recipe=light.RECIPE,processed=len(done),selected=len(rows),counts=counts,
        wall_seconds=time.perf_counter()-start,**light.FLAGS)
    light.write_once(Path(cfg['out'])/'summary.json',report)


def summaries(cfg,out):
    report=light.load(out/'summary.json')
    light.need(report['processed']==report['selected']==cfg['targetcount'],'Not every assigned source is terminal')
    return {str(out/'summary.json'):dict(summary=dict(stage='run',total_source_count=cfg['targetcount'],
        selected_source_count=cfg['targetcount'],processed_source_count=report['processed'],plan_sha256=cfg['plan']['sha256']),
        selection=dict(selected_ids=selected(cfg)[0]),draft_counts=report['counts'],semantic_approval=False)}


def main():
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--config',required=True)
        worker(p.parse_args().config);return
    base.validate_pair=validate;base.selected_sources=selected;base.stage_summaries=summaries
    base.main()
if __name__=='__main__':main()
