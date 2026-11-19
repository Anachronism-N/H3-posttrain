"""Body-only request contracts, no image rendering or model dispatch."""
from pathlib import Path
import sys
import copy
import hashlib
import json
import unittest
RT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RT))
import caption_body_motion_first_v1 as d

class BodyFirstTests(unittest.TestCase):
    def request(self):
        bar=d.native.native();frames=[dict(index=i,time=i/16,source_timestamp=i/16,file=str(i)+'.jpg',sha256='a'*64) for i in (0,2)]
        manifest=dict(window=dict(window_id='w0',start=0.,end=1.),frames=frames,hand_roi_evidence={'left_hand':{'crops':[]},'right_hand':{'crops':[]}})
        content=[dict(type='text',text=json.dumps({'allowed_source_frame_indices':[0,2],'hand_ROI_support':{'left_hand':[0,2]}}))]
        for f in frames:
            content.extend([dict(type='text',text='Source frame %d, time %.9fs, original PTS %.9fs.'%(f['index'],f['time'],f['source_timestamp'])),dict(type='image_url',image_url={'url':'data:image/jpeg;base64,Qm9keQ=='})])
            content.extend([dict(type='text',text=json.dumps({'same_source_frame_as_body_context':True,'not_another_time_observation':True,'frame_index':f['index']})),dict(type='image_url',image_url={'url':'data:image/jpeg;base64,Um9J'})])
        return dict(model='/cpu/model',messages=[dict(role='system',content=bar.SYSTEM),dict(role='user',content=content)],sampling=dict(temperature=0.,seed=0,max_tokens=1800),json_schema=bar.observation_schema([0,2]),prompt_sha256=hashlib.sha256(bar.SYSTEM.encode()).hexdigest()),manifest
    def test_body_pixels_indices_schema_and_budget_exact(self):
        request,manifest=self.request();before=copy.deepcopy(request)
        value=d.prepare_request(request,manifest)
        self.assertEqual(request,before)
        self.assertEqual(value['sampling'],request['sampling']);self.assertEqual(value['json_schema'],request['json_schema'])
        self.assertEqual(value['model'],request['model'])
        content=value['messages'][1]['content']
        self.assertEqual(len(content),5)
        self.assertEqual([x['image_url']['url'] for x in content if x['type']=='image_url'],['data:image/jpeg;base64,Qm9keQ==']*2)
        self.assertEqual(json.loads(content[0]['text'])['allowed_source_frame_indices'],[0,2])
        self.assertEqual(json.loads(content[0]['text'])['hand_ROI_support'],{})
        self.assertEqual(value['fine_hand_evidence_deferred'],manifest['hand_roi_evidence'])
    def test_arbitrary_pairs_or_reordered_body_rejected(self):
        request,manifest=self.request()
        request['messages'][1]['content'][3]['text']=json.dumps({'same_source_frame_as_body_context':False,'not_another_time_observation':True})
        with self.assertRaises(ValueError):d.prepare_request(request,manifest)
        request,manifest=self.request();manifest['frames'].reverse()
        with self.assertRaises(ValueError):d.prepare_request(request,manifest)
    def test_repeat_recipe_and_budget_change_rejected(self):
        request,manifest=self.request();derived=d.prepare_request(request,manifest)
        with self.assertRaises(ValueError):d.prepare_request(derived,manifest)
        request['sampling']['max_tokens']=800
        with self.assertRaises(ValueError):d.prepare_request(request,manifest)
    def test_changed_body_pixels_produce_new_binding_without_original_mutation(self):
        request,manifest=self.request();first=d.prepare_request(request,manifest)
        request['messages'][1]['content'][2]['image_url']['url']='data:image/jpeg;base64,Y2hhbmdl'
        second=d.prepare_request(request,manifest)
        self.assertNotEqual(first['recipe_identity_sha256'],second['recipe_identity_sha256'])
        self.assertFalse(second['quality_approved']);self.assertFalse(second['training_eligible'])

if __name__=='__main__':unittest.main()
