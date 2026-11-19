"""Prepared packet verification touches only local metadata/images/video."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from stage2_local_publication_v1 import write_once
import stage2_local_packet_v1 as m

class PacketTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='local-packet-cpu-',dir='/tmp');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.owner='N8-g01-model';id_='SyntheticSource';self.video=self.root/(id_+'.mp4');self.video.write_bytes(b'opaque CPU-only video bytes');sha=hashlib.sha256(self.video.read_bytes()).hexdigest()
        logical='/shared/immutable/cases/train/'+id_+'/fullframe_manifest.json';manifest=dict(schema='ms_fullframe_pose2d_manifest_v1',id=id_,split='train',video_path='/shared/videos/'+id_+'.mp4',video_sha256=sha,frame_count=3,times=[0.,.04,.08],source_timestamps=[10.,10.04,10.08],quality_approved=False,semantic_approval=False,training_eligible=False,needs_visual_revision=True)
        binding=dict(schema='ms_fullframe_pose2d_binding_v1',observations_sha256='b'*64);body=dict(schema=binding['schema'],fullframe_manifest=manifest,observations_sha256=binding['observations_sha256']);binding['sha256']=hashlib.sha256(json.dumps(body,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        draft=dict(schema='ms_fullframe_pose2d_rule_draft_v2',id=id_,split='train',ms_text='Visible endpoint pose',source=dict(fullframe_manifest_path=logical,video_path=manifest['video_path'],video_sha256=sha,fullframe_binding=binding),quality_approved=False,semantic_approval=False,training_eligible=False,needs_visual_revision=True)
        self.dr=write_once(self.root/'draft.json',draft);self.mr=write_once(self.root/'manifest.json',manifest);original=dict(id=id_,split='train',video_path=manifest['video_path'],rule_draft_ref=dict(path='/shared/immutable/draft.json',sha256=self.dr['sha256']),fullframe_manifest_ref=dict(path=logical,sha256=self.mr['sha256']))
        images=[]
        for i in range(3):
            p=self.root/('frame%d.jpg'%i);p.write_bytes(bytes([255,216])+b'syntheticJPEG'+bytes([255,217]));images.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        self.packet=dict(schema='stage2_prepared_packet_v1',owner=self.owner,claim=dict(schema='stage2_dynamic_claim_receipt_v1',owner=self.owner,batch_id='claim0001',automatic_retry=False,maximum_model_calls_per_source=2,rows=[original],source_keys=[['train',id_]],source_count=1),sources=[dict(id=id_,split='train',draft_ref=self.dr,manifest_ref=self.mr,selected_frames=[dict(frame_index=i,time=t,source_pts=p) for i,(t,p) in enumerate(zip(manifest['times'],manifest['source_timestamps']))],image_refs=images,local_video_ref=dict(path=str(self.video),sha256=sha),single_source_video_copy=True)])
    def validate(self):return m.validate_packet(self.packet,self.root,self.owner)
    def reject(self):
        with self.assertRaises((ValueError,FileNotFoundError)):self.validate()
    def test_local_packet_preserves_original_logical_source_without_shared_reads(self):
        r=self.validate();self.assertEqual(len(r),1);self.assertEqual(r[0]['original_row']['fullframe_manifest_ref']['path'],self.packet['claim']['rows'][0]['fullframe_manifest_ref']['path'])
    def test_wrong_owner_rejected(self):self.packet['owner']='N1-g01-other';self.reject()
    def test_claim_owner_rejected(self):self.packet['claim']['owner']='N1-g01-other';self.reject()
    def test_claim_retry_forbidden(self):self.packet['claim']['automatic_retry']=True;self.reject()
    def test_source_claim_budget_rejected(self):self.packet['claim']['maximum_model_calls_per_source']=3;self.reject()
    def test_boolean_count_rejected(self):self.packet['claim']['source_count']=True;self.reject()
    def test_wrong_source_id_rejected(self):self.packet['sources'][0]['id']='Other';self.reject()
    def test_wrong_source_split_rejected(self):self.packet['sources'][0]['split']='test';self.reject()
    def test_rewritten_original_sha_rejected(self):self.packet['claim']['rows'][0]['rule_draft_ref']['sha256']='0'*64;self.reject()
    def test_native_frame_order_rejected(self):self.packet['sources'][0]['selected_frames'].reverse();self.reject()
    def test_sampling_reduction_rejected(self):self.packet['sources'][0]['image_refs'].pop();self.reject()
    def test_changed_jpeg_rejected(self):Path(self.packet['sources'][0]['image_refs'][0]['path']).write_bytes(b'bad');self.reject()
    def test_changed_video_rejected(self):self.video.write_bytes(b'changed');self.reject()
    def test_second_source_copy_claim_rejected(self):self.packet['sources'][0]['single_source_video_copy']=False;self.reject()
    def test_foreign_local_root_ref_rejected(self):self.packet['sources'][0]['local_video_ref']['path']='/shared/video.mp4';self.reject()
    def test_missing_video_rejected(self):self.video.unlink();self.reject()
if __name__=='__main__':unittest.main(verbosity=2)
