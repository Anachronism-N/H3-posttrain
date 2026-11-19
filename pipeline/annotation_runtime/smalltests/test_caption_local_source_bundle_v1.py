"""End-to-end local input/result/writeback contracts; no model or GPU allocation."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from PIL import Image
RT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(RT),str(Path(__file__).resolve().parent)]
import caption_local_source_bundle_v1 as local
from test_caption_two_stage_generation_v2 import LightLaneTests
from test_caption_two_stage_light_v1 import Model,LightTests

class LocalBundleTests(unittest.TestCase):
    def test_all_inputs_intermediates_and_results_local_before_archive_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);f=LightLaneTests().fixture(tmp);metas=[{'id':r['id'],'split':'train'} for r in f.entries[:3]]
            args=f.args;args.ffmpeg='fake-local-decoder';model=Model();decoded=[]
            def decode(command,**kwargs):
                import re
                decoded.append(command[command.index('-i')+1]);script=Path(command[command.index('-filter_script:v')+1]).read_text()
                indices=re.findall(r'eq\(n\\,(\d+)\)',script)
                for i,_ in enumerate(indices,1):Image.new('RGB',(32,48),(i,0,0)).save(command[-1]%i,'JPEG')
                return NS(returncode=0)
            before={p:p.read_bytes() for p in f.original.rglob('*') if p.is_file()}
            with LightTests().no_expensive_checks(),patch.object(local.light.subprocess,'run',side_effect=decode):
                result=local.run_local_batch(f.original,metas,args,lambda:(model,NS),base/'shared','source-batch',base,lambda a,r:True)
            self.assertEqual([len(x) for x in model.calls],[8])
            self.assertTrue(all(str(base) in p and 'source-batch-' in p for p in decoded))
            self.assertEqual(before,{p:p.read_bytes() for p in f.original.rglob('*') if p.is_file()})
            self.assertTrue(result['reliable_writeback']);self.assertGreater(result['archive_bytes'],0)
            self.assertGreaterEqual(result['end_to_end_seconds'],result['copy_seconds'])
            drafts=list(local.archive_drafts(result['archive_path']))
            self.assertEqual(len(drafts),3);self.assertTrue(all(not d['training_eligible'] for _,d in drafts))
            self.assertTrue(all(d['binding']['local_execution'] is True for _,d in drafts))
            self.assertTrue(all(d['binding']['original_source_record']['video_path']!=decoded[0] for _,d in drafts))
            count=len(model.calls)
            again=local.run_local_batch(f.original,metas,args,lambda:(model,NS),base/'shared','source-batch',base,lambda a,r:True)
            self.assertEqual(again['archive_path'],result['archive_path']);self.assertEqual(len(model.calls),count)
    def test_failed_local_batch_keeps_budget_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);args=NS();root=base/'shared'
            with self.assertRaises(FileNotFoundError):
                local.run_local_batch(base/'missing',[{'id':'s','split':'train'}],args,lambda:None,root,'failed',base,lambda a,r:True)
            self.assertTrue((root/'budgets/failed/failure.json').exists())
            with self.assertRaisesRegex(ValueError,'reserved'):
                local.run_local_batch(base/'missing',[{'id':'s','split':'train'}],args,lambda:None,root,'failed',base,lambda a,r:True)
    def test_writeback_never_overwrites_existing_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);data=p/'local';data.mkdir();(data/'result.json').write_text('{}')
            shared=p/'shared';value=local.writeback(data,shared,'batch')
            before=Path(value['archive_path']).read_bytes()
            with self.assertRaisesRegex(ValueError,'published'):local.writeback(data,shared,'batch')
            self.assertEqual(before,Path(value['archive_path']).read_bytes())

if __name__=='__main__':unittest.main()
