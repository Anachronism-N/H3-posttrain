"""Stage-two request uses pixels, keeps the draft and applies native support gates."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
RT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RT))
import caption_two_stage_revision_v1 as d
bar,pipeline=d.drafts.native.native()

class RevisionTests(unittest.TestCase):
    def test_revision_request_retains_exact_original_pixel_data_and_untrusted_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);request=dict(model='/cpu/model',messages=[dict(role='system',content=bar.SYSTEM),
                dict(role='user',content=[dict(type='text',text='original header'),dict(type='image_url',image_url={'url':'data:image/jpeg;base64,b3JpZ2luYWw='})])],
                sampling=dict(temperature=0.,seed=0,max_tokens=1800),json_schema=bar.observation_schema([0,1]),prompt_sha256='cpu')
            req=pipeline.ref(pipeline.save(p/'request.json',request))
            draft=dict(schema=d.drafts.RECIPE,id='s0',training_eligible=False,needs_visual_revision=True,
                windows=[dict(window=dict(window_id='w0',start=0.,end=1.),original_request_ref=req,
                    raw_observation=dict(events=[]),raw_text='unverified content',status='draft_partial')],
                style_revision_pending=True,source_binding=dict(source_binding=dict(id='s0')))
            receipt=dict(id='s0',draft_ref=pipeline.ref(pipeline.save(p/'draft.json',draft)))
            before=(p/'draft.json').read_bytes()
            plan=d.prepare_source(receipt);new=plan['tasks'][0]['request']
            self.assertEqual(new['messages'][1]['content'][:-1],request['messages'][1]['content'])
            self.assertEqual(new['sampling'],request['sampling']);self.assertEqual(new['json_schema'],request['json_schema'])
            self.assertIn('unverified_proposal',new['messages'][1]['content'][-1]['text'])
            self.assertEqual(before,(p/'draft.json').read_bytes())
            self.assertTrue(plan['requires_pixels_not_text_only']);self.assertFalse(plan['training_eligible'])
    def test_missing_visual_input_deferred_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);draft=dict(schema=d.drafts.RECIPE,id='s0',training_eligible=False,needs_visual_revision=True,
                windows=[dict(window=dict(window_id='w0',start=0.,end=1.),status='draft_partial',reason='no actual frames')],
                style_revision_pending=True,source_binding=dict(source_binding=dict(id='s0')))
            r=dict(id='s0',draft_ref=pipeline.ref(pipeline.save(p/'draft.json',draft)))
            plan=d.prepare_source(r)
            self.assertEqual(plan['tasks'][0]['status'],'needs_new_visual_observation')
            self.assertNotIn('request',plan['tasks'][0])
    def test_native_part_support_gate_still_applies_after_revision(self):
        value=bar.empty_observation('unknown')
        value.update(events=[dict(id='e1',description='The left hand opens.',parts=['left_hand'],frame_indices=[0,1])],abstain=False)
        manifest=dict(window=dict(window_id='w0',start=0.,end=1.),frames=[dict(index=i,time=i*.1) for i in (0,1)])
        result=d.validate_revised_observation(copy.deepcopy(value),manifest,{})
        self.assertEqual(result['events'],[])
        value['part_support'][bar.PARTS.index('left_hand')]['frame_indices']=[0,1]
        result=d.validate_revised_observation(value,manifest,{})
        self.assertTrue(result['events'])

if __name__=='__main__':unittest.main()
