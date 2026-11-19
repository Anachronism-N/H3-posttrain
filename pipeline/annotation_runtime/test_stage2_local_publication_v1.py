"""Real local durable publication and no-clobber archive recovery tests."""
import copy
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import stage2_local_publication_v1 as m

class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='stage2-outbox-cpu-',dir='/tmp');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.box=m.LocalOutbox(self.root/'local',self.root/'shared','N2');self.keys=[]
        for i in range(3):
            id_='Source%03d'%i;d=self.box.local/'sources/train'/id_;result=dict(id=id_,split='train',status='postprocessed_unapproved',quality_approved=False,training_eligible=False,stages=[]);ref=m.write_once(d/'postprocessed.json',result);m.write_once(d/'commit.json',dict(id=id_,split='train',status=result['status'],result_ref=ref,quality_approved=False,training_eligible=False));m.write_once(self.box.local/'budgets/train'/id_/'source.claim.json',dict(source=id_,automatic_retry=False));self.keys.append(['train',id_])
    def seal(self):return self.box.seal(self.keys)
    def test_seal_local_before_any_shared_directory_created(self):
        r=self.seal();v=m.checked(r);self.assertEqual(v['source_count'],3);self.assertFalse(self.box.shared.exists())
        with tarfile.open(Path(r['path']).parent/'batch.tar','r:') as t:self.assertIn('batch_manifest.json',t.getnames())
    def test_upload_commits_archive_then_ack_and_keeps_local_cache(self):
        ref=self.seal();r=self.box.upload(ref);self.assertTrue(r['may_cleanup_local']);self.assertEqual(r['models_called'],0);commit=json.loads(Path(r['shared_commit']['path']).read_bytes());self.assertTrue(Path(commit['archive']['path']).exists());self.assertTrue((Path(ref['path']).parent/'batch.tar').exists());self.assertTrue((self.box.local/'sources/train/Source000/postprocessed.json').exists())
    def test_already_committed_batch_only_recovers_local_receipt(self):
        ref=self.seal();ack=self.box.upload(ref);p=Path(ref['path']).parent/'upload_receipt.json';p.unlink()
        with patch.object(m,'rename_noreplace',side_effect=AssertionError('No archive re-upload after reliable commit')):r=self.box.upload(ref)
        self.assertEqual(r['shared_commit'],ack['shared_commit'])
    def test_no_overwrite_different_existing_commit(self):
        ref=self.seal();sealed=m.checked(ref);target=self.box.shared/'N2'/sealed['batch_id'];m.write_once(target/'commit.json',{'foreign':True})
        with self.assertRaises(ValueError):self.box.upload(ref)
        self.assertEqual(json.loads((target/'commit.json').read_bytes()),{'foreign':True})
    def test_real_no_clobber_rename(self):
        source=self.root/'a';target=self.root/'b';source.write_bytes(b'original');target.write_bytes(b'foreign')
        with self.assertRaises(FileExistsError):m.rename_noreplace(source,target)
        self.assertEqual(source.read_bytes(),b'original');self.assertEqual(target.read_bytes(),b'foreign')
    def test_partial_archive_without_commit_is_verified_before_commit(self):
        ref=self.seal();s=m.checked(ref);target=self.box.shared/'N2'/s['batch_id'];target.mkdir(parents=True);(target/'batch.tar').write_bytes((Path(ref['path']).parent/'batch.tar').read_bytes());r=self.box.upload(ref);self.assertTrue(Path(r['shared_commit']['path']).is_file())
    def test_wrong_partial_archive_preserves_local_and_shared_data(self):
        ref=self.seal();s=m.checked(ref);target=self.box.shared/'N2'/s['batch_id'];target.mkdir(parents=True);(target/'batch.tar').write_bytes(b'partial')
        with self.assertRaises(ValueError):self.box.upload(ref)
        self.assertEqual((target/'batch.tar').read_bytes(),b'partial');self.assertFalse((target/'commit.json').exists());self.assertTrue((Path(ref['path']).parent/'batch.tar').exists())
    def test_tampered_sealed_archive_does_not_publish_commit(self):
        ref=self.seal();(Path(ref['path']).parent/'batch.tar').write_bytes(b'tampered')
        with self.assertRaises(ValueError):self.box.upload(ref)
        self.assertFalse(any(self.box.shared.rglob('commit.json')))
    def test_duplicate_sources_refused_before_seal(self):
        with self.assertRaises(ValueError):self.box.seal([self.keys[0],self.keys[0]])
    def test_duplicate_seal_not_overwritten(self):
        self.seal()
        with self.assertRaises(FileExistsError):self.seal()
    def test_single_uploader_real_kernel_flock(self):
        fd=self.box.uploader_lock()
        try:
            with self.assertRaises(BlockingIOError):self.box.uploader_lock()
        finally:os.close(fd)
    def test_missing_commit_cannot_count_as_published(self):
        (self.box.local/'sources/train/Source000/commit.json').unlink()
        with self.assertRaises(FileNotFoundError):self.seal()
    def test_input_symlink_is_not_archived(self):
        d=self.box.local/'sources/train/Source000';(d/'foreign.json').symlink_to(d/'postprocessed.json')
        with self.assertRaises(ValueError):self.seal()
    def test_traversal_source_id_refused(self):
        with self.assertRaises(ValueError):self.box.seal([['train','../foreign']])
    def test_unapproved_result_flags_must_remain_false(self):
        p=self.box.local/'sources/train/Source000/commit.json';x=json.loads(p.read_bytes());x['quality_approved']=True;p.write_bytes(m.encoded(x))
        with self.assertRaises(ValueError):self.seal()
    def test_outside_result_ref_refused(self):
        p=self.box.local/'sources/train/Source000/commit.json';x=json.loads(p.read_bytes());x['result_ref']=m.write_once(self.root/'foreign.json',{'id':'Source000','split':'train'});p.write_bytes(m.encoded(x))
        with self.assertRaises(ValueError):self.seal()
    def test_seal_preserves_batch_request_and_response_evidence(self):
        p=self.box.local/'sources/train/Source000/postprocessed.json';x=json.loads(p.read_bytes());r=m.write_once(self.box.local/'batch_calls/000001/request.json',{'inputs':3});s=m.write_once(self.box.local/'batch_calls/000001/response.json',{'outputs':3});x['stages']=[dict(batch_request_ref=r,batch_response_ref=s)];p.write_bytes(m.encoded(x));commit=self.box.local/'sources/train/Source000/commit.json';c=json.loads(commit.read_bytes());c['result_ref']=dict(path=str(p),sha256=m.hashlib.sha256(p.read_bytes()).hexdigest());commit.write_bytes(m.encoded(c));ref=self.seal()
        with tarfile.open(Path(ref['path']).parent/'batch.tar','r:') as t:self.assertIn('batch_calls/000001/request.json',t.getnames());self.assertIn('batch_calls/000001/response.json',t.getnames())
    def test_ceph_unsupported_rename_uses_no_clobber_link(self):
        from types import SimpleNamespace
        source=self.root/'a';target=self.root/'b';source.write_bytes(b'original')
        def unsupported(*args):return -1
        with patch.object(m.ctypes,'CDLL',return_value=SimpleNamespace(renameat2=unsupported)),patch.object(m.ctypes,'get_errno',return_value=m.errno.EINVAL):m.rename_noreplace(source,target)
        self.assertEqual(target.read_bytes(),b'original');self.assertFalse(source.exists())
    def test_ceph_link_fallback_never_overwrites_existing_target(self):
        from types import SimpleNamespace
        source=self.root/'a';target=self.root/'b';source.write_bytes(b'original');target.write_bytes(b'foreign')
        def unsupported(*args):return -1
        with patch.object(m.ctypes,'CDLL',return_value=SimpleNamespace(renameat2=unsupported)),patch.object(m.ctypes,'get_errno',return_value=m.errno.EINVAL),self.assertRaises(FileExistsError):m.rename_noreplace(source,target)
        self.assertEqual(source.read_bytes(),b'original');self.assertEqual(target.read_bytes(),b'foreign')
if __name__=='__main__':unittest.main(verbosity=2)
