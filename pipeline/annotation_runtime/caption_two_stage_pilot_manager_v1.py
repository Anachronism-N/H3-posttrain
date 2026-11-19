"""Three-source stage-one draft trial with the original GPU lifecycle."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types
sys.dont_write_bytecode=True
RT=Path(__file__).resolve().parent
SELF=Path(__file__).resolve()
sys.path.insert(0,str(RT))
import caption_two_stage_pipeline_v1 as drafts
spec=importlib.util.spec_from_file_location('two_stage_native',RT/'caption_batch8_direct_full_manager_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.SELF=SELF
m.pilot.DEFAULT_OUT=RT.parent/'caption_two_stage_generated_pilot3_v1'
m.PRIOR={g:('caption_phase_motion_n6_v1','caption_phase_motion_n6_v1') for g in m.GPUS}
m.parent_gate=types.FunctionType(m.helpers.parent_gate.__code__,dict(m.helpers.parent_gate.__globals__,SELF=SELF,base=m.base),
    'parent_gate',m.helpers.parent_gate.__defaults__)
BASE_SHA=m.BASE_SHA

def restored(gpu,safe_ref):
    queue=m.BAR/'queues/caption_phase_motion_n6_v1'
    m.need(Path(safe_ref['path']).resolve()==queue/'resources_safe.json','Exact previous trial safe required')
    safe=m.checked(safe_ref);admission=m.checked(m.ref(queue/'admission.json'));exited=m.checked(m.ref(queue/'worker_exit.json'))
    value=m.checked(m.ref(queue/('holder_restored_gpu%d.json'%gpu)))
    m.need(admission['node']=='N6' and admission['gpus']==[2,3] and admission['manager_sha256']==BASE_SHA
        and admission['model_identity_sha256']==m.MODEL_SHA and exited['exit_code']==0 and exited['source_unchanged'] is True
        and safe['phase']==admission['phase']==exited['phase']=='caption_phase_motion_n6_v1'
        and all(safe[k] is True for k in ('all_holders_safe','locks_released','leases_restored','managed_parent_wait','unknown_jobs_not_signaled')),
        'Predecessor resources are not safely restored')
    original=next(x for x in admission['holders'] if x['gpu']==gpu)
    m.need(value['uuid']==original['uuid'] and value['managed_parent_wait'] is True
        and all(value['original_identity'][k]==original['identity'][k] for k in m.base.IDENTITY_KEYS),'Restored identity differs')
    return value
m.restored=restored

def build(args):
    capture_ref=dict(path=str(args.capture.resolve()),sha256=args.capture_sha)
    capture=m.validate_capture(m.checked(capture_ref))
    modelref=dict(path=str(args.modelref.resolve()),sha256=args.modelref_sha,model_identity_sha256=m.MODEL_SHA)
    previous=m.BAR/'queues/caption_phase_motion_n6_v1'
    paths=[SELF,Path(drafts.__file__),RT/'caption_batch8_direct_full_manager_v1.py',RT/'resource_manager.py',
        RT/'bench_manager_driver_v2.py',RT/'curated_resource_manager.py',RT/'visual_batch_fixed3_benchmark_v1.py',
        RT/'caption_direct_facts_v1.py',Path(m.pilot.__file__),Path(capture_ref['path']),Path(modelref['path'])]
    paths.extend(m.pilot.direct.SOURCE/n for n in m.pilot.direct.FROZEN)
    paths.extend(previous/n for n in ('admission.json','worker_exit.json','resources_safe.json','holder_restored_gpu2.json','holder_restored_gpu3.json'))
    cfg=dict(node='N6',gpus=[2,3],mode='consumer',release_authorized=True,phase=args.phase,
        input_count=3,targetcount=3,holders=capture['holders'],holder_capture=capture_ref,modelref=modelref,
        out=str(args.execution_root.resolve()),pilot_out=str(m.pilot.DEFAULT_OUT),queue=str(args.queue.resolve()),
        config_path=str(args.config_out.resolve()),cwd=str(RT),reuse_existing_results=False,
        resource_manager_ref=m.ref(SELF),producer=m.ref(SELF),plan=m.ref(m.pilot.__file__),
        env=dict(CUDA_VISIBLE_DEVICES='2,3',MS_VISUAL_CPU_SITEPATH='/opt/venv_eval/lib/python3.11/site-packages',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',OMP_NUM_THREADS='2'),
        source_sha256={str(p.resolve()):m.base.sha(p) for p in paths})
    cfg['argv']=[m.PYTHON,'-B',str(SELF),'worker','--config-file',cfg['config_path']]
    m.need(not args.config_out.exists() and not args.queue.exists() and not args.execution_root.exists()
        and not m.pilot.DEFAULT_OUT.exists(),'Fresh draft trial/config/output required')
    m.validate(cfg)
    return m.ref(m.pilot.direct.native()[1].save(args.config_out,cfg))

def worker(path,factory=None):
    bar,pipeline=m.pilot.direct.native()
    cfg=bar.core.strict_json(Path(path).read_bytes())
    cfg.update(_config_path=str(Path(path).resolve()),_config_sha=m.base.sha(path))
    m.validate(cfg);m.parent_gate(cfg,before_load=True)
    execution=Path(cfg['out']);pipeline.save(execution/'driver.started.json',dict(stage='draft_generation',**drafts.FLAGS))
    args=m.pilot.pilot_args(m.MODEL,cfg['modelref']['path'])
    model=bar.model_reference(args)
    if factory is None:
        from vllm import LLM,SamplingParams
    else:LLM,SamplingParams=factory
    llm=LLM(model=m.MODEL,dtype='bfloat16',quantization=None,tensor_parallel_size=2,max_model_len=16384,
        max_num_seqs=8,gpu_memory_utilization=.88,limit_mm_per_prompt={'image':16,'video':0},
        mm_processor_kwargs={'max_pixels':200704},structured_outputs_config={'backend':'xgrammar','disable_any_whitespace':True},
        enable_prefix_caching=True,trust_remote_code=False,seed=0)
    m.actual_state(llm);llm._ms_bar_model_sha256=model['sha256']
    pipeline.save(execution/'model_load.receipt.json',dict(actual_runtime=m.actual_state(llm),model_identity_sha256=model['sha256'],**drafts.FLAGS))
    inputs=m.pilot.Inputs();configuration=bar.params(args,bar.code_identity());configuration.update(caption_recipe=drafts.RECIPE)
    entries,provenance=m.pilot.fixed_sources(inputs,args,configuration)
    styles={e['row']['id']:p['style'] for e,p in zip(entries,provenance)}
    def admission(a,r):
        m.parent_gate(cfg);m.actual_state(llm);return a is args
    report=drafts.generate_batch(entries,args,lambda:(llm,SamplingParams),args.out,admission,styles)
    inputs.verify()
    pipeline.save(execution/'driver.summary.json',dict(stage='draft_generation',report=report,
        source_ids=list(m.pilot.SOURCES),**drafts.FLAGS))
    return report

def summaries(cfg,out):
    v=m.checked(m.ref(out/'driver.summary.json'));r=v['report']
    m.need(r['source_count']==3 and r['window_call_count']==8 and r['synchronous_rewrite_calls']==0
        and r['synchronous_review_calls']==0 and len(r['published'])==3,'Draft trial actual summary differs')
    return {str(out/'driver.summary.json'):dict(summary=dict(stage='run',total_source_count=3,selected_source_count=3,
        processed_source_count=3,plan_sha256=cfg['plan']['sha256']),selection=dict(selected_ids=list(m.pilot.SOURCES)),
        stage='draft_generation',semantic_approval=False)}

m.summaries=summaries

def main():
    if len(sys.argv)>1 and sys.argv[1] in ('capture','build','worker'):
        p=argparse.ArgumentParser();p.add_argument('command')
        if sys.argv[1]=='capture':p.add_argument('--out',type=Path,required=True)
        elif sys.argv[1]=='worker':p.add_argument('--config-file',required=True)
        else:
            for n in ('capture','modelref','config-out','execution-root','queue'):p.add_argument('--'+n,type=Path,required=True)
            for n in ('capture-sha','modelref-sha','phase'):p.add_argument('--'+n,required=True)
        a=p.parse_args();r=m.capture_current(a.out) if a.command=='capture' else build(a) if a.command=='build' else worker(a.config_file)
        if a.command!='worker':print(json.dumps(r))
    else:m.manager()
if __name__=='__main__':main()
