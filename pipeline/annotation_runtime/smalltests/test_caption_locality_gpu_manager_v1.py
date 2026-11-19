"""Real two-route generation pipeline, synthetic model; no GPU/resource actions."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace as NS
import unittest
from unittest.mock import patch
RT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(RT),str(Path(__file__).resolve().parent)]
import caption_locality_gpu_manager_v1 as d
import test_caption_two_stage_generation_v2 as fixture
import test_caption_two_stage_light_v1 as models

class GpuLocalityTests(unittest.TestCase):
    def test_exact_two_routes_16_calls_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);f=fixture.LightLaneTests().fixture(tmp);(base/'curated222939_collaborative_v3').symlink_to(f.original,target_is_directory=True)
            out=base/'caption_two_stage_locality_trial';exe=base/'execution';exe.mkdir();cfg=base/'config.json'
            d.local.light.write_once(cfg,dict(out=str(exe),modelref=dict(model_identity_sha256='CPU-model')))
            fake=ModuleType('vllm');loaded=[]
            class LLM(models.Model):
                def __init__(self,**kwargs):super().__init__();loaded.append(self)
            fake.LLM=LLM;fake.SamplingParams=NS
            helper=models.LightTests();decoder=fixture.LightLaneTests().decode
            with (patch.dict(sys.modules,{'vllm':fake}),patch.object(d.m,'validate'),patch.object(d.m,'parent_gate',return_value={'admitted':True}),
                    patch.object(d.m,'actual_state'),patch.object(d.m,'BAR',base),patch.object(d.m.pilot,'DEFAULT_OUT',out),
                    patch.object(d.m.pilot,'SOURCES',('source0','source1','source2')),helper.no_expensive_checks(),
                    patch.object(d.local.light.subprocess,'run',side_effect=decoder)):
                d.worker(cfg)
            report=d.local.light.load(out/'report.json')
            self.assertEqual(len(loaded),1);self.assertEqual(report['actual_calls'],16);self.assertEqual(report['chat_batch_sizes'],[8,8])
            self.assertEqual(report['local_source_count'],3);self.assertEqual(report['direct_source_count'],3)
            self.assertTrue(Path(report['local_archive']).exists());self.assertFalse(report['training_eligible'])
    def test_native_base_and_explicit_new_trial_predecessor(self):
        self.assertEqual(d.m.base.sha(d.m.base.__file__),d.BASE_SHA)
        self.assertEqual(d.m.parent_gate.__globals__['SELF'],d.SELF)

if __name__=='__main__':unittest.main()
