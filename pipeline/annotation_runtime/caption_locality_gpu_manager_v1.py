"""Same resident TP2, three sources: full local bundle vs Ceph with copy/writeback."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
sys.dont_write_bytecode=True
RT=Path(__file__).resolve().parent;SELF=Path(__file__).resolve();sys.path.append('/opt/venv_eval/lib/python3.11/site-packages');sys.path.insert(0,str(RT))
import caption_two_stage_pilot_manager_v1 as pilot
import caption_local_source_bundle_v1 as local
m=pilot.m;BASE_SHA=m.BASE_SHA
m.SELF=SELF;m.pilot.DEFAULT_OUT=RT.parent/'caption_two_stage_locality_gpu_trial_v1'
m.PRIOR={g:('caption_two_stage_pilot_n6_v1','caption_two_stage_pilot_n6_v1') for g in m.GPUS}
m.parent_gate=types.FunctionType(m.helpers.parent_gate.__code__,dict(m.helpers.parent_gate.__globals__,SELF=SELF,base=m.base),
    'parent_gate',m.helpers.parent_gate.__defaults__)

def restored(gpu,safe_ref):
    q=m.BAR/'queues/caption_two_stage_pilot_n6_v1'
    m.need(Path(safe_ref['path']).resolve()==q/'resources_safe.json','Latest safe predecessor required')
    safe=m.checked(safe_ref);a=m.checked(m.ref(q/'admission.json'));e=m.checked(m.ref(q/'worker_exit.json'));r=m.checked(m.ref(q/('holder_restored_gpu%d.json'%gpu)))
    m.need(a['node']=='N6' and a['gpus']==[2,3] and a['manager_sha256']==BASE_SHA and e['exit_code']==0 and e['source_unchanged'] is True
        and all(safe[k] is True for k in ('all_holders_safe','locks_released','leases_restored')),'Predecessor must be safely returned')
    old=next(x for x in a['holders'] if x['gpu']==gpu)
    m.need(r['uuid']==old['uuid'] and all(r['original_identity'][k]==old['identity'][k] for k in m.base.IDENTITY_KEYS),'Restored GPU identity differs')
    return r
m.restored=restored

def build(args):
    capture_ref=dict(path=str(args.capture.resolve()),sha256=args.capture_sha);capture=m.validate_capture(m.checked(capture_ref))
    q=m.BAR/'queues/caption_two_stage_pilot_n6_v1';old=m.checked(m.ref(RT/'caption_two_stage_pilot_launch_main_v1.json'))
    modelref=old['modelref'];paths=[SELF,Path(local.__file__),Path(local.light.__file__),RT/'caption_two_stage_pilot_manager_v1.py',
        RT/'caption_batch8_direct_full_manager_v1.py',RT/'resource_manager.py',RT/'bench_manager_driver_v2.py',RT/'curated_resource_manager.py',
        Path(m.pilot.__file__),RT/'caption_direct_facts_v1.py',RT/'visual_batch_fixed3_benchmark_v1.py',Path(capture_ref['path'])]
    paths.extend(q/n for n in ('admission.json','worker_exit.json','resources_safe.json','holder_restored_gpu2.json','holder_restored_gpu3.json'))
    cfg=dict(old);cfg.update(phase='caption_two_stage_locality_gpu_n6_v1',holders=capture['holders'],holder_capture=capture_ref,
        out=str(m.BAR/'caption_batch8_direct_full_execution_locality_gpu_v1'),pilot_out=str(m.pilot.DEFAULT_OUT),queue=str(m.BAR/'queues/caption_two_stage_locality_gpu_n6_v1'),
        producer=m.ref(SELF),resource_manager_ref=m.ref(SELF),config_path=str(args.config_out.resolve()),source_sha256={str(p.resolve()):m.base.sha(p) for p in paths})
    cfg['env']['OMP_NUM_THREADS']='1';cfg['argv']=['/opt/venv_vllm/bin/python','-B',str(SELF),'worker','--config-file',cfg['config_path']]
    m.need(not Path(cfg['queue']).exists() and not Path(cfg['out']).exists() and not m.pilot.DEFAULT_OUT.exists(),'Fresh trial roots required')
    m.validate(cfg);local.light.write_once(args.config_out,cfg);return m.ref(args.config_out)

def worker(path):
    cfg=local.light.load(path);cfg.update(_config_path=str(Path(path).resolve()),_config_sha=m.base.sha(path));m.validate(cfg);m.parent_gate(cfg,True)
    out=m.pilot.DEFAULT_OUT;execution=Path(cfg['out'])
    local.light.write_once(execution/'driver.started.json',dict(trial='locality',**local.light.FLAGS))
    local.light.write_once(out/'experiment.claim.json',dict(reserved_window_calls=16,source_count=3,order='local_then_direct',**local.light.FLAGS))
    from vllm import LLM,SamplingParams
    llm=LLM(model=m.MODEL,dtype='bfloat16',tensor_parallel_size=2,max_num_seqs=8,max_model_len=16384,gpu_memory_utilization=.88,
        limit_mm_per_prompt={'image':16,'video':0},mm_processor_kwargs={'max_pixels':200704},
        structured_outputs_config={'backend':'xgrammar','disable_any_whitespace':True},enable_prefix_caching=True,trust_remote_code=False,seed=0)
    m.actual_state(llm);llm._ms_bar_model_sha256=cfg['modelref']['model_identity_sha256']
    source=m.BAR/'curated222939_collaborative_v3';metas=[{'id':s,'split':'train'} for s in m.pilot.SOURCES]
    args=argparse.Namespace(stage='run',batch_size=8,max_num_seqs=8,retry_failed=False,model=m.MODEL,visual_max_tokens=1800,
        hand_roi_root=str(source/'roi'),sample_fps=4.,ffmpeg='/usr/local/bin/ffmpeg')
    calls=[];walls=[]
    class Model:
        def __getattr__(self,n):return getattr(llm,n)
        def chat(self,messages,*a,**kw):
            m.parent_gate(cfg);m.need(sum(calls)+len(messages)<=16,'Trial budget exhausted');calls.append(len(messages));t=time.perf_counter()
            try:return llm.chat(messages,*a,**kw)
            finally:walls.append(time.perf_counter()-t)
    runtime=lambda:(Model(),SamplingParams)
    admitted=lambda a,r:bool(m.parent_gate(cfg))
    local_result=local.run_local_batch(source,metas,args,runtime,out,'local-first','/tmp',admitted)
    local_chat=sum(walls);t=time.perf_counter();direct_out=out/'caption_two_stage_direct';inputs=local.light.Inputs()
    entries=[local.light.load_entry(source,meta,direct_out,inputs) for meta in metas];direct=local.light.generate_batch(entries,args,runtime,direct_out,admitted)
    report=dict(scope='resident_TP2_three_case_end_to_end_copy_decode_GPU_and_writeback',order='local_then_direct',
        local_end_to_end_seconds=local_result['end_to_end_seconds'],direct_end_to_end_seconds=time.perf_counter()-t,
        local_chat_seconds=local_chat,direct_chat_seconds=sum(walls)-local_chat,actual_calls=sum(calls),chat_batch_sizes=calls,
        copy_seconds=local_result['copy_seconds'],writeback_seconds=local_result['writeback_seconds'],
        local_archive=local_result['archive_path'],local_source_count=3,direct_source_count=direct['source_count'],
        cache_order_not_isolated=True,production_counts=0,**local.light.FLAGS)
    local.light.write_once(out/'report.json',report);local.light.write_once(execution/'summary.json',report)

def summaries(cfg,out):
    v=local.light.load(out/'summary.json');m.need(v['actual_calls']==16 and v['direct_source_count']==3,'Actual trial call/source count differs')
    return {str(out/'summary.json'):dict(summary=dict(stage='run',total_source_count=3,selected_source_count=3,processed_source_count=3,plan_sha256=cfg['plan']['sha256']),
        selection=dict(selected_ids=list(m.pilot.SOURCES)),actual_calls=16,semantic_approval=False)}
m.summaries=summaries

def main():
    if len(sys.argv)>1 and sys.argv[1]=='capture':
        p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--out',type=Path,required=True);print(json.dumps(m.capture_current(p.parse_args().out)))
    elif len(sys.argv)>1 and sys.argv[1]=='build':
        p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--capture',type=Path,required=True);p.add_argument('--capture-sha',required=True);p.add_argument('--config-out',type=Path,required=True);print(json.dumps(build(p.parse_args())))
    elif len(sys.argv)>1 and sys.argv[1]=='worker':
        p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--config-file',required=True);worker(p.parse_args().config_file)
    else:m.manager()
if __name__=='__main__':main()
