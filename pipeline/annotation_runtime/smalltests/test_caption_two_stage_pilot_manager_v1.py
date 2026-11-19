"""New stage-one caller uses native resources and generates drafts only."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
RT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RT),str(Path(__file__).resolve().parent)]
import caption_two_stage_pilot_manager_v1 as d
from test_caption_batch8_direct_full_pilot_v1 import FullFixture,FakeLLM,bar,pipeline
from test_caption_batch8_direct_full_manager_v1 import config

class PilotCallerTests(unittest.TestCase):
    def test_base_identity_unmodified(self):
        self.assertEqual(Path(d.m.base.__file__).resolve(),RT/'resource_manager.py')
        self.assertEqual(d.m.base.sha(d.m.base.__file__),d.BASE_SHA)
        self.assertEqual(d.m.parent_gate.__globals__['SELF'],d.SELF)
    def test_fake_native_stage1_generates_three_drafts_eight_calls_no_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=FullFixture(tmp);f.out=Path(tmp)/'caption_two_stage_fixture';f.args.out=str(f.out)
            cfgpath=Path(tmp)/'config.json';execution=Path(tmp)/'execution'
            pipeline.save(cfgpath,dict(out=str(execution),modelref=dict(path='/cpu/modelref')))
            loaded=[]
            class LLM(FakeLLM):
                def __init__(self,**kwargs):super().__init__();self.vllm_config=config();loaded.append(self)
            model=dict(root=d.m.MODEL,sha256=d.m.MODEL_SHA,files={},cpu_fake=True)
            with f.context(),patch.object(d.drafts,'BAR',Path(tmp)),patch.object(d.m,'validate'),patch.object(d.m,'parent_gate',return_value={'owned':True}),patch.object(d.m.pilot,'pilot_args',return_value=f.args),patch.object(bar,'model_reference',return_value=model):
                report=d.worker(cfgpath,factory=(LLM,object))
                summary=d.summaries({'plan':{'sha256':'cpu'}},execution)
            self.assertEqual(len(loaded),1)
            self.assertEqual([len(x) for x in loaded[0].calls],[8])
            self.assertEqual(report['source_count'],3)
            self.assertEqual(report['synchronous_review_calls'],0)
            self.assertEqual(report['synchronous_rewrite_calls'],0)
            self.assertEqual(len(report['published']),3)
            self.assertTrue(all(not x['training_eligible'] and x['needs_visual_revision'] for x in report['published']))
            self.assertEqual(next(iter(summary.values()))['stage'],'draft_generation')

if __name__=='__main__':unittest.main()
