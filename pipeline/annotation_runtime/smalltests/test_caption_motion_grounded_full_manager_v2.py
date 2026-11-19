"""CPU tests for a fresh motion-grounded native request recipe."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
RT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RT),str(Path(__file__).resolve().parent)]
import caption_motion_grounded_full_manager_v2 as d
from test_caption_batch8_direct_full_pilot_v1 import FullFixture,FakeLLM,bar,pipeline

def config():
    return NS(model_config=NS(model=d.m.MODEL,dtype='torch.bfloat16',quantization=None,trust_remote_code=False,seed=0,max_model_len=16384,
        multimodal_config=NS(mm_processor_kwargs={'max_pixels':200704},get_limit_per_prompt=lambda k:16 if k=='image' else 0)),
        scheduler_config=NS(max_num_seqs=8),parallel_config=NS(tensor_parallel_size=2),
        structured_outputs_config=NS(backend='xgrammar',disable_any_whitespace=True),
        cache_config=NS(enable_prefix_caching=True,gpu_memory_utilization=.88))

class GroundedTests(unittest.TestCase):
    def test_original_base_and_separate_recipe(self):
        self.assertEqual(Path(d.m.base.__file__).resolve(),RT/'resource_manager.py')
        self.assertEqual(d.m.base.sha(d.m.base.__file__),d.BASE_SHA)
        self.assertEqual(d.m.pilot.direct.RECIPE,d.grounded.RECIPE)
        self.assertEqual(d.m.pilot.DEFAULT_OUT.name,'caption_motion_grounded_full_pilot_v1')
    def test_runtime_requires_exact_new_prompt_and_eight_call_budget(self):
        llm=NS(vllm_config=config(),chat=lambda *a,**kw:[])
        runtime=d.Runtime(llm,lambda:None)
        with self.assertRaisesRegex(ValueError,'grounded'):
            runtime.chat([[{'content':bar.SYSTEM}]],[])
        self.assertEqual(runtime.calls,0)
        runtime.chat([[{'content':bar.SYSTEM+d.grounded.CONTRACT}]]*8,[])
        self.assertEqual(runtime.calls,8)
        with self.assertRaisesRegex(ValueError,'grounded'):
            runtime.chat([[{'content':bar.SYSTEM+d.grounded.CONTRACT}]],[])
    def test_actual_full_native_transactions_new_request_identity_and_no_text_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=FullFixture(tmp)
            execution=Path(tmp)/'execution'
            cfgpath=Path(tmp)/'config.json'
            cfg=dict(config_path=str(cfgpath),out=str(execution),modelref={'path':'/cpu/model'},plan={'path':str(d.m.pilot.__file__),'sha256':d.m.PILOT_SHA})
            pipeline.save(cfgpath,cfg)
            loaded=[]
            class LLM(FakeLLM):
                def __init__(self,**kw):
                    super().__init__();self.vllm_config=config();loaded.append(self)
                def chat(self,messages,sampling,use_tqdm=False):
                    import copy
                    original_messages=copy.deepcopy(messages)
                    compatible=copy.deepcopy(messages)
                    for msg in compatible:
                        if msg[0]['content']!=bar.SYSTEM+d.grounded.CONTRACT:
                            raise AssertionError('Expected new grounded prompt')
                        msg[0]['content']=bar.SYSTEM
                    outputs=super().chat(compatible,sampling,use_tqdm)
                    self.calls[-1]=original_messages
                    return outputs
            identity=dict(root=d.m.MODEL,sha256=d.m.MODEL_SHA,files={},cpu_fake=True)
            original=bar.visual_request
            old_runtime=d.m.Runtime
            try:
                d.install_requests()
                with f.context(),patch.object(d.m,'validate'),patch.object(d.m,'parent_gate',return_value={'owned':True}),patch.object(d.m.pilot,'pilot_args',return_value=f.args),patch.object(bar,'model_reference',return_value=identity):
                    report=d.m.worker(cfgpath,factory=(LLM,object))
                self.assertEqual(len(loaded),1)
                self.assertEqual([len(call) for call in loaded[0].calls],[8])
                self.assertEqual(report['joint_complete_candidates'],3)
                self.assertEqual(report['boundary_model_calls'],0)
                self.assertEqual(report['rewrite_model_calls'],0)
                for source in report['results']:
                    self.assertEqual(source['visual']['caption_recipe'],d.grounded.RECIPE)
                    for observation in source['visual']['window_observations']:
                        request=bar.load(Path(observation['attempt_path'])/'request.json')
                        self.assertEqual(request['messages'][0]['content'],bar.SYSTEM+d.grounded.CONTRACT)
                        self.assertEqual(request['generation_recipe'],d.grounded.RECIPE)
                        self.assertEqual(request['fact_source_recipe'],d.grounded.RECIPE)
                        self.assertEqual(request['response_protocol_recipe'],'bar_video_first_weak_v1')
                        self.assertEqual(request['sampling']['max_tokens'],1800)
            finally:
                bar.visual_request=original;d.m.Runtime=old_runtime

if __name__=='__main__':unittest.main()
