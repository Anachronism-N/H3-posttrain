"""Node uploader only seals durable local commits and records publication recovery."""
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import stage2_node_uploader_v1 as u
from stage2_local_publication_v1 import LocalOutbox,write_once

class UploaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='uploader-cpu-',dir='/tmp');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.local=self.root/'local';self.shared=self.root/'shared';self.box=LocalOutbox(self.local,self.shared,'N8')
        for i in range(3):
            id_='Source%03d'%i;d=self.local/'sources/train'/id_;r=write_once(d/'postprocessed.json',dict(id=id_,split='train',status='failed_sealed_no_retry',quality_approved=False,training_eligible=False,stages=[],diagnostic_only=True));write_once(d/'commit.json',dict(id=id_,split='train',status='failed_sealed_no_retry',result_ref=r,quality_approved=False,training_eligible=False));write_once(self.local/'budgets/train'/id_/'source.claim.json',dict(diagnostic_only=True))
    def test_drained_tail_publishes_and_records_metrics_without_model(self):
        (self.local/'GENERATION_DRAINED').touch()
        def end(_):(self.local/'UPLOAD_STOP').touch()
        with patch.object(u.time,'sleep',side_effect=end):u.run(self.local,self.shared,'N8',60)
        files=list((self.local/'outbox').glob('*/upload_receipt.json'));self.assertEqual(len(files),1);self.assertEqual(len(list((self.local/'upload_metrics').iterdir())),1)
    def test_resume_metrics_and_sealed_sources_do_not_overwrite(self):
        ref=self.box.seal([['train','Source000'],['train','Source001'],['train','Source002']]);self.box.upload(ref);(self.local/'GENERATION_DRAINED').touch();write_once(self.local/'upload_metrics/00000001.json',dict(original=True))
        def end(_):(self.local/'UPLOAD_STOP').touch()
        with patch.object(u.time,'sleep',side_effect=end):u.run(self.local,self.shared,'N8',60)
        self.assertEqual(len(list((self.local/'outbox').iterdir())),1)
    def test_second_uploader_cannot_run_same_node_root(self):
        fd=self.box.uploader_lock()
        try:
            with self.assertRaises(BlockingIOError):u.run(self.local,self.shared,'N8',1)
        finally:__import__('os').close(fd)
    def test_write_failure_keeps_archive_and_does_not_call_models(self):
        (self.local/'GENERATION_DRAINED').touch()
        def end(_):(self.local/'UPLOAD_STOP').touch()
        with patch.object(u.LocalOutbox,'upload',side_effect=OSError('Synthetic Ceph error')),patch.object(u.time,'sleep',side_effect=end):u.run(self.local,self.shared,'N8',60)
        self.assertEqual(len(list((self.local/'outbox').glob('*/batch.tar'))),1);self.assertEqual(len(list((self.local/'outbox').glob('*/upload_receipt.json'))),0)
    def test_invalid_lifetime_does_not_open_loop(self):
        with self.assertRaises(ValueError):u.run(self.local,self.shared,'N8',0)
if __name__=='__main__':unittest.main(verbosity=2)
