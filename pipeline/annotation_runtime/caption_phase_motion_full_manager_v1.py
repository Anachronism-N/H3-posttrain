"""Fixed fresh motion-grounded generation over the original eight visual windows."""
import argparse
import copy
import importlib.util
from pathlib import Path
import sys

RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
sys.path.insert(0, str(RT))
import caption_phase_motion_request_v1 as grounded
spec = importlib.util.spec_from_file_location('grounded_native_manager', RT/'caption_batch8_direct_full_manager_v3.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
BASE_SHA = m.BASE_SHA
PRIOR_QUEUE = RT.parent/'queues/caption_motion_grounded_full_n6_v2'
PRIOR_CONFIG_SHA = 'dd43ddb5c7d4d4112c735b7254f8930ef8224110d2c7ea685b903e8852dbf574'
m.SELF = SELF
m.PRIOR = {g: (PRIOR_QUEUE.name, PRIOR_QUEUE.name) for g in m.GPUS}
m.parent_gate = m.types.FunctionType(m.helpers.parent_gate.__code__, dict(m.helpers.parent_gate.__globals__, SELF=SELF, base=m.base), 'parent_gate', m.helpers.parent_gate.__defaults__)
m.pilot.DEFAULT_OUT = m.BAR/'caption_phase_motion_full_pilot_v1'
m.pilot.SCHEMA = 'caption_phase_motion_full_pilot_v1'
m.pilot.direct.RECIPE = grounded.RECIPE

def restored(gpu, safe_ref):
    m.need(Path(safe_ref['path']).resolve() == PRIOR_QUEUE/'resources_safe.json', 'Exact completed full-pilot predecessor required')
    safe = m.checked(safe_ref)
    admission = m.checked(m.ref(PRIOR_QUEUE/'admission.json'))
    exited = m.checked(m.ref(PRIOR_QUEUE/'worker_exit.json'))
    holder = m.checked(m.ref(PRIOR_QUEUE/('holder_restored_gpu%d.json'%gpu)))
    m.need(admission['config_sha256'] == PRIOR_CONFIG_SHA and admission['node'] == 'N6' and admission['gpus'] == [2,3]
        and admission['manager_sha256'] == BASE_SHA and admission['model_identity_sha256'] == m.MODEL_SHA
        and safe['phase'] == admission['phase'] == exited['phase'] == PRIOR_QUEUE.name
        and all(safe.get(k) is True for k in ('all_holders_safe','locks_released','leases_restored','managed_parent_wait','unknown_jobs_not_signaled'))
        and exited['exit_code'] == 0 and exited['source_unchanged'] is True and bool(exited['summaries']),
        'Predecessor must be actually completed and safely restored')
    original = next(h for h in admission['holders'] if h['gpu']==gpu)
    m.need(holder['gpu']==gpu and holder['uuid']==original['uuid'] and holder['managed_parent_wait'] is True
        and holder['restore_argv']==m.base.RESTORE_ARGV
        and all(holder['original_identity'][k]==original['identity'][k] for k in m.base.IDENTITY_KEYS),
        'Native restored holder identity differs')
    return holder
m.restored = restored

def build(args):
    capture_ref = dict(path=str(args.capture.resolve()), sha256=args.capture_sha)
    capture = m.validate_capture(m.checked(capture_ref))
    modelref = dict(path=str(args.modelref.resolve()), sha256=args.modelref_sha, model_identity_sha256=m.MODEL_SHA)
    paths = [SELF, RT/'caption_batch8_direct_full_manager_v3.py', Path(m.pilot.__file__), RT/'resource_manager.py',
        RT/'bench_manager_driver_v2.py', RT/'curated_resource_manager.py', RT/'visual_batch_fixed3_benchmark_v1.py',
        RT/'caption_cross_source_stage_v1.py', RT/'caption_direct_facts_v1.py', Path(grounded.__file__), RT/'caption_motion_grounded_request_v1.py',
        RT/'caption_compact_visual_v1.py', Path(capture_ref['path']), Path(modelref['path'])]
    paths.extend(m.pilot.direct.SOURCE/name for name in m.pilot.direct.FROZEN)
    paths.extend(PRIOR_QUEUE/name for name in ('admission.json','worker_exit.json','resources_safe.json','holder_restored_gpu2.json','holder_restored_gpu3.json'))
    cfg = dict(node='N6',gpus=[2,3],mode='consumer',release_authorized=True,phase=args.phase,
        targetcount=3,input_count=3,holders=capture['holders'],holder_capture=capture_ref,modelref=modelref,
        out=str(args.execution_root.resolve()),pilot_out=str(m.pilot.DEFAULT_OUT),queue=str(args.queue.resolve()),
        config_path=str(args.config_out.resolve()),cwd=str(RT),reuse_existing_results=False,
        resource_manager_ref=m.ref(SELF),producer=m.ref(SELF),plan=m.ref(m.pilot.__file__),
        env=dict(CUDA_VISIBLE_DEVICES='2,3',MS_VISUAL_CPU_SITEPATH='/opt/venv_eval/lib/python3.11/site-packages',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',OMP_NUM_THREADS='2'),
        source_sha256={str(p.resolve()):m.base.sha(p) for p in paths},
        generation_recipe=grounded.RECIPE, fact_source_recipe=grounded.RECIPE, grounded_request_ref=m.ref(grounded.__file__))
    cfg['argv']=[m.PYTHON,'-B',str(SELF),'worker','--config-file',cfg['config_path']]
    m.need(not args.queue.exists() and not args.execution_root.exists() and not m.pilot.DEFAULT_OUT.exists(), 'Fresh recipe output/execution/queue required')
    m.validate(cfg)
    return m.ref(m.pilot.direct.native()[1].save(args.config_out,cfg))

class Runtime(m.Runtime):
    def chat(self,messages,*args,**kwargs):
        self.gate()
        m.actual_state(self.llm)
        bar,_=m.pilot.direct.native()
        m.need(self.calls+len(messages)<=8 and all(msg[0]['content']==grounded.SYSTEM for msg in messages),
            'Only eight fresh grounded native visual requests allowed')
        self.calls += len(messages)
        return self.llm.chat(messages,*args,**kwargs)

def install_requests():
    bar,_=m.pilot.direct.native()
    original = bar.visual_request
    def request(args,root,manifest,forbidden):
        return grounded.prepare_request(original(args,root,manifest,forbidden),manifest)
    bar.visual_request=request
    m.Runtime=Runtime

def main():
    if len(sys.argv)>1 and sys.argv[1]=='build':
        parser=argparse.ArgumentParser()
        parser.add_argument('command')
        for name in ('capture','modelref','config-out','execution-root','queue'):
            parser.add_argument('--'+name,type=Path,required=True)
        for name in ('capture-sha','modelref-sha','phase'):
            parser.add_argument('--'+name,required=True)
        print(build(parser.parse_args()))
        return
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        install_requests()
    m.main()

if __name__=='__main__':
    main()
