"""CPU-only two-stage contracts; raw claims are NOT supported training facts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
RT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(RT),str(Path(__file__).resolve().parent)]
import caption_two_stage_pipeline_v1 as d
from test_caption_batch8_direct_full_pilot_v1 import FullFixture,FakeLLM,bar,pipeline

class TwoStageTests(unittest.TestCase):
    def fixture(self,tmp,empty=False):
        f=FullFixture(tmp,empty_window=empty)
        out=Path(tmp)/'caption_two_stage_fixture'
        f.args.out=str(out)
        for e in f.entries:
            e['root']=out/'sources'/e['row']['id']
            e['raw_line']=bar.batch.json_bytes(e['row'])
            e['binding']['full_source_row_sha256']=bar.batch.digest(e['row'])
            e['binding']['manifest_record_sha256']=bar.batch.digest(e['row']['manifest_record'])
            e['binding']['raw_line_sha256']=bar.sha(e['raw_line'])
        return f,out
    def test_generate_three_sources_eight_windows_raw_preserved_no_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            f,out=self.fixture(tmp);llm=FakeLLM()
            with f.context(),patch.object(d,'BAR',Path(tmp)),patch.object(bar,'filter_supported_observation',side_effect=AssertionError('No synchronous semantic filter')):
                result=d.generate_batch(f.entries,f.args,f.runtime(llm),out,lambda a,r:True)
            self.assertEqual([len(x) for x in llm.calls],[8])
            self.assertEqual(result['synchronous_rewrite_calls'],0)
            self.assertEqual(result['synchronous_review_calls'],0)
            self.assertEqual(result['new_style_model_calls'],0)
            self.assertEqual(len(result['published']),3)
            for receipt in result['published']:
                draft=pipeline.load(receipt['draft_ref']['path'])
                self.assertFalse(draft['training_eligible'])
                self.assertTrue(draft['needs_visual_revision'])
                self.assertTrue(draft['style_revision_pending'])
                self.assertTrue(any('right foot moves' in w['draft_text'] for w in draft['windows']))
                self.assertTrue(all(w['raw_observation']['uncertainties'] for w in draft['windows']))
            before=(out/'revision_queue.jsonl').read_bytes()
            with f.context(),patch.object(d,'BAR',Path(tmp)):
                with self.assertRaisesRegex(ValueError,'claimed'):
                    d.generate_batch(f.entries,f.args,f.runtime(llm),out,lambda a,r:True)
            self.assertEqual(before,(out/'revision_queue.jsonl').read_bytes())
            self.assertEqual(len(llm.calls),1)
    def test_empty_window_not_stillness(self):
        with tempfile.TemporaryDirectory() as tmp:
            f,out=self.fixture(tmp,True);llm=FakeLLM()
            with f.context(),patch.object(d,'BAR',Path(tmp)):
                result=d.generate_batch(f.entries,f.args,f.runtime(llm),out,lambda a,r:True)
            drafts=[pipeline.load(r['draft_ref']['path']) for r in result['published']]
            empty=[w for x in drafts for w in x['windows'] if w['status']=='draft_partial']
            self.assertTrue(empty)
            self.assertTrue(all(w['draft_text']=='' for w in empty))
            self.assertEqual(sum(len(x) for x in llm.calls),7)
    def test_default_resource_denied_before_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            f,out=self.fixture(tmp)
            with self.assertRaisesRegex(ValueError,'admission'):
                d.generate_batch(f.entries,f.args,f.runtime(FakeLLM()),out)
            self.assertFalse(out.exists())
    def test_input_change_before_publish_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input.json';p.write_text('{}');reader=d.Reader();reader.raw(p);p.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'publication'):reader.verify()
    def test_schema_support_semantics_deferred_but_frame_enum_preserved(self):
        value=bar.empty_observation('unknown')
        value['events']=[dict(id='e1',description='The right foot moves.',parts=['head'],frame_indices=[0,1])]
        value['abstain']=False
        m=dict(frames=[dict(index=0),dict(index=1)])
        self.assertEqual(d.schema_only(value,m),value)
        value['events'][0]['frame_indices']=[0,100]
        with self.assertRaises(ValueError):d.schema_only(value,m)
    def test_revision_publish_exact_repeat_no_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(d,'BAR',Path(tmp)):
            out=Path(tmp)/'caption_two_stage_test'
            draft=d.assemble('source',dict(split='train',batch_id='b0'),dict(source_binding=dict(id='source')),
                [dict(window=dict(window_id='w0',start=0.,end=1.),status='draft_partial',draft_text='')],None)
            reader=d.Reader();a=d.publish(draft,out,reader);before=Path(a['draft_ref']['path']).read_bytes()
            b=d.publish(draft,out,d.Reader())
            self.assertEqual(a,b);self.assertEqual(before,Path(a['draft_ref']['path']).read_bytes())
            self.assertFalse(draft['training_eligible']);self.assertTrue(draft['style_revision_pending'])

if __name__=='__main__':unittest.main()
