"""Standalone published external metadata contracts, models/media never executed."""
from contextlib import ExitStack
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import ms_vlm_external_direct_inputs_v1 as m

class ExternalInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='external-caption-cpu-',dir='/tmp');self.addCleanup(self.tmp.cleanup)
        self.p=Path(self.tmp.name)/'batch';self.commit_path=self.p/'commit.json';self.files=[];self.receipts=[];self.assignment=[]
        for i in range(3):
            id_='External%03d'%i;d=self.p/'cases/train'/id_;manifest=d/'fullframe_manifest.json';video=Path(self.tmp.name)/(id_+'.mp4');source=dict(id=id_,split='train',video_path=str(video))
            value=dict(schema='ms_fullframe_pose2d_manifest_v1',id=id_,split='train',video_path=str(video),video_sha256='a'*64,frame_count=2,frame_indices=[0,1],times=[0.,.04],source_timestamps=[10.,10.04],pts_origin=10.,width=704,height=1280,duration_seconds=.08,**dict(list(m.v.FLAGS.items())[:4]))
            mr=m.v.write_once(manifest,value);binding=dict(schema='ms_fullframe_pose2d_binding_v1',observations_sha256='b'*64)
            payload=dict(schema=binding['schema'],fullframe_manifest=value,observations_sha256=binding['observations_sha256'])
            import json
            binding['sha256']=hashlib.sha256(json.dumps(payload,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            draft=dict(schema='ms_fullframe_pose2d_rule_draft_v2',id=id_,split='train',ms_text='Visible arm pose',source=dict(fullframe_manifest_path=str(manifest),video_path=str(video),video_sha256='a'*64,fullframe_binding=binding),**dict(list(m.v.FLAGS.items())[:4]))
            dr=m.v.write_once(d/'draft.json',draft)
            for r in (mr,dr):self.files.append(dict(path=r['path'],sha256=r['sha256'],bytes=Path(r['path']).stat().st_size))
            self.assignment.append(source);self.receipts.append(dict(source_row=dict(source,assignment_index=i),status=m.STATUS,vlm_called=False,needs_visual_revision=True,quality_approved=False,semantic_approval=False,training_eligible=False))
        self.commit=dict(schema='ms_unified_novlm_batch_commit_v1',edition='external_ms_unified_clusters80k_20261027_v1',worker='a100-extra-02',prefetch_drained=True,quality_approved=False,semantic_approval=False,training_eligible=False,package=dict(path=str(self.p/'batch.tar'),sha256='c'*64,bytes=10240),batch_sha256='c'*64,files=self.files,receipts=self.receipts)
        self.model=patch.object(m.v,'runtime',side_effect=AssertionError('No models in caption input selection'));self.model.start();self.addCleanup(self.model.stop)

    def rows(self):return m.published_rows(self.commit_path,self.commit,set(),self.assignment)
    def reject(self):
        with self.assertRaises((ValueError,FileNotFoundError)):self.rows()
    def test_direct_files_pass_without_any_media_archive_read(self):
        r=self.rows();self.assertEqual(len(r),3);self.assertFalse((self.p/'batch.tar').exists());self.assertTrue(all('origin_proof_ref' not in x for x in r))
    def test_existing_consumed_source_excluded(self):
        r=m.published_rows(self.commit_path,self.commit,{'external000'},self.assignment);self.assertEqual([x['id'] for x in r],['External001','External002'])
    def test_failed_receipt_not_retried(self):self.receipts[0]['status']='failed_sealed_no_retry';self.assertEqual(len(self.rows()),2)
    def test_wrong_schema_rejected(self):self.commit['schema']='unknown';self.reject()
    def test_wrong_epoch_rejected(self):self.commit['edition']='foreign';self.reject()
    def test_wrong_worker_rejected(self):self.commit['worker']='a100-extra-03';self.reject()
    def test_training_approved_input_not_silently_changed(self):self.commit['training_eligible']=True;self.reject()
    def test_batch_path_binding_rejected(self):self.commit['package']['path']='/tmp/foreign.tar';self.reject()
    def test_batch_sha_binding_rejected(self):self.commit['package']['sha256']='d'*64;self.reject()
    def test_undeclared_member_rejected(self):self.files.pop(0);self.reject()
    def test_duplicate_file_path_rejected(self):self.files.append(self.files[0]);self.reject()
    def test_duplicate_source_rejected(self):self.receipts.append(copy.deepcopy(self.receipts[0]));self.reject()
    def test_prior_vlm_receipt_not_called_again(self):self.receipts[0]['vlm_called']=True;self.reject()
    def test_boolean_assignment_index_rejected(self):self.receipts[0]['source_row']['assignment_index']=True;self.reject()
    def test_out_of_range_assignment_rejected(self):self.receipts[0]['source_row']['assignment_index']=40000;self.reject()
    def test_real_assignment_row_binding_checked(self):self.receipts[0]['source_row']['video_path']='/tmp/foreign.mp4';self.reject()
    def test_modified_file_sha_rejected(self):Path(self.files[0]['path']).write_bytes(b'{}');self.reject()
    def test_published_length_mismatch_rejected(self):self.files[0]['bytes']+=1;self.reject()

if __name__=='__main__':unittest.main(verbosity=2)
