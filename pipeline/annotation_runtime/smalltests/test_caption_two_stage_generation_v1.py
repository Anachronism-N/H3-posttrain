"""Finite two-stage draft generation with a real native ownership fixture."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
RT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RT),str(Path(__file__).resolve().parent)]
import caption_two_stage_generation_v1 as d
from test_caption_batch8_production_consumer_v1 import ProductionFixture,FakeLLM,pipeline

class DraftGenerationTests(unittest.TestCase):
    def fixture(self,tmp):
        f=ProductionFixture(tmp)
        f.out=Path(tmp)/'caption_two_stage_new'
        f.args.out=str(f.out)
        handoff=Path(f.args.handoff_ref['path']);h=json.loads(handoff.read_bytes())
        exclusion=Path(h['old_exclusion_ref']['path']);x=json.loads(exclusion.read_bytes());x['new_out']=str(f.out)
        exclusion.write_bytes(pipeline.raw_json(x));h['old_exclusion_ref']=pipeline.ref(exclusion);h['new_out']=str(f.out)
        handoff.write_bytes(pipeline.raw_json(h));f.args.handoff_ref=pipeline.ref(handoff)
        return f
    def test_six_sources_two_native_batches_drafts_no_synchronous_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);llm=FakeLLM()
            with f.context(),patch.object(d.drafts,'BAR',Path(tmp)):
                report=d.run_consumer(f.args,f.runtime(llm),lambda a,r:True)
            self.assertEqual([len(x) for x in llm.calls],[8,8])
            self.assertEqual(report['new_source_count'],6)
            self.assertEqual(report['synchronous_rewrite_calls'],0)
            self.assertEqual(report['synchronous_review_calls'],0)
            self.assertEqual(report['synchronous_style_calls'],0)
            notices=[json.loads(x) for x in (f.out/'revision_queue.jsonl').read_bytes().splitlines()]
            self.assertEqual(len(notices),6);self.assertEqual(len({r['id'] for r in notices}),6)
            self.assertTrue(all(r['needs_visual_revision'] and not r['training_eligible'] for r in notices))
            for r in notices:
                draft=pipeline.load(r['draft_ref']['path'])
                self.assertTrue(any('right foot moves' in w['draft_text'] for w in draft['windows']))
            with f.context(),patch.object(d.drafts,'BAR',Path(tmp)):
                again=d.run_consumer(f.args,f.runtime(llm),lambda a,r:True)
            self.assertEqual(again['new_source_count'],0)
            self.assertEqual(len(llm.calls),2)
    def test_missing_ownership_proof_denied_before_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);llm=FakeLLM();f.args.handoff_ref=None
            with patch.object(d.drafts,'BAR',Path(tmp)),self.assertRaisesRegex(ValueError,'handoff'):
                d.run_consumer(f.args,f.runtime(llm),lambda a,r:True)
            self.assertEqual(llm.calls,[])
    def test_default_resource_admission_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp)
            with self.assertRaisesRegex(ValueError,'resources'):
                d.run_consumer(f.args,f.runtime(FakeLLM()))
            self.assertFalse(f.out.exists())

if __name__=='__main__':unittest.main()
