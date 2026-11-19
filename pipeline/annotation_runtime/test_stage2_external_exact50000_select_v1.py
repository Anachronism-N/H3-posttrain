"""Metadata-only exact external list contracts, no caption/model reads."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import stage2_external_exact50000_select_v1 as m

class ExactListTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='exact50k-cpu-',dir='/tmp');self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)/'batch';self.p.mkdir();self.cp=self.p/'commit.json';self.assignment=[dict(id='Source000',split='train',video_path='/shared/Source000.mp4')];self.ticket=dict(assignment_sha256='a'*64);row=dict(self.assignment[0],assignment_index=0);base=self.p/'cases/train/Source000'
        self.commit=dict(schema='ms_unified_novlm_batch_commit_v1',edition='external_ms_unified_clusters80k_20261027_v1',worker='a100-extra-01',assignment_sha256='a'*64,prefetch_drained=True,semantic_approval=False,quality_approved=False,training_eligible=False,package=dict(path=str(self.p/'batch.tar'),sha256='b'*64,bytes=10240),batch_sha256='b'*64,files=[dict(path=str(base/name),sha256='c'*64,bytes=100) for name in ('draft.json','fullframe_manifest.json')],receipts=[dict(source_row=row,status='extracted_rule_draft_unapproved',vlm_called=False,quality_approved=False,semantic_approval=False,training_eligible=False,needs_visual_revision=True)])
    def run_parse(self):
        self.cp.write_text(json.dumps(self.commit));return m.parse_batch((self.cp,'a100-extra-01',self.ticket,self.assignment))
    def test_metadata_list_no_caption_or_archive_file_required(self):
        r,proof=self.run_parse();self.assertEqual(len(r),1);self.assertFalse((self.p/'batch.tar').exists());self.assertFalse(Path(r[0]['rule_draft_ref']['path']).exists());self.assertEqual(r[0]['id'],'Source000');self.assertEqual(proof['worker'],'a100-extra-01')
    def test_failed_ms_receipt_not_allocated(self):self.commit['receipts'][0]['status']='failed_sealed_no_retry';self.assertEqual(self.run_parse()[0],[])
    def test_source_assignment_mismatch_rejected(self):self.commit['receipts'][0]['source_row']['video_path']='/foreign';self.reject()
    def test_wrong_assignment_sha_rejected(self):self.commit['assignment_sha256']='d'*64;self.reject()
    def test_foreign_epoch_skipped(self):self.commit['edition']='foreign';self.assertEqual(self.run_parse(),([],None))
    def test_duplicate_source_rejected(self):self.commit['receipts']*=2;self.reject()
    def test_missing_manifest_ref_rejected(self):self.commit['files'].pop();self.reject()
    def test_previously_called_vlm_receipt_rejected(self):self.commit['receipts'][0]['vlm_called']=True;self.reject()
    def test_wrong_package_binding_rejected(self):self.commit['package']['sha256']='d'*64;self.reject()
    def test_unapproved_flag_cannot_be_overwritten(self):self.commit['quality_approved']=True;self.reject()
    def reject(self):
        with self.assertRaises((AssertionError,KeyError)):self.run_parse()
if __name__=='__main__':unittest.main(verbosity=2)
