"""Finite light generation uses cached producer timelines, not repeated probes."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
from PIL import Image
RT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(RT),str(Path(__file__).resolve().parent)]
import caption_two_stage_generation_v2 as d
from test_caption_batch8_production_consumer_v1 import ProductionFixture
from test_caption_two_stage_light_v1 import Model
import test_caption_two_stage_light_v1 as light_tests

class LightLaneTests(unittest.TestCase):
    def fixture(self,tmp):
        f=ProductionFixture(tmp);f.out=Path(tmp)/'caption_two_stage_light_lane';f.args.out=str(f.out)
        handoff=Path(f.args.handoff_ref['path']);h=__import__('json').loads(handoff.read_bytes());h['new_out']=str(f.out)
        exclusion=Path(h['old_exclusion_ref']['path']);x=__import__('json').loads(exclusion.read_bytes());x['new_out']=str(f.out)
        from test_caption_batch8_production_consumer_v1 import pipeline
        exclusion.write_bytes(pipeline.raw_json(x));h['old_exclusion_ref']=pipeline.ref(exclusion);handoff.write_bytes(pipeline.raw_json(h));f.args.handoff_ref=pipeline.ref(handoff)
        for row in f.entries:
            roi=Path(f.args.hand_roi_root)/row['id']/'roi_manifest.json'
            value=__import__('json').loads(roi.read_bytes());value['video_sha256']=row['binding']['video_sha256']
            roi.write_bytes(pipeline.raw_json(value))
        return f
    def decode(self,command,**kwargs):
        import re
        script=Path(command[command.index('-filter_script:v')+1]).read_text()
        indices=re.findall(r'eq\(n\\,(\d+)\)',script)
        for offset,_ in enumerate(indices,1):Image.new('RGB',(32,48),(offset,0,0)).save(Path(command[-1]%offset),'JPEG')
        return NS(returncode=0)
    def test_two_groups_cached_timeline_no_native_media_or_history_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);model=Model();bar=d.light.bar_module()
            helper=light_tests.LightTests()
            with helper.no_expensive_checks(),patch.object(d.light.subprocess,'run',side_effect=self.decode) as decode:
                report=d.run_consumer(f.args,lambda:(model,NS),lambda a,r:True)
            self.assertEqual([len(x) for x in model.calls],[8,8])
            self.assertEqual(decode.call_count,6)
            self.assertEqual(report['media_sha_reads'],0);self.assertEqual(report['media_probes'],0)
            self.assertEqual(report['history_review_reads'],0);self.assertEqual(report['new_model_calls'],16)
            queue=(f.out/'revision_queue.jsonl').read_bytes()
            with helper.no_expensive_checks(),patch.object(d.light.subprocess,'run',side_effect=AssertionError('No second decode')):
                again=d.run_consumer(f.args,lambda:(model,NS),lambda a,r:True)
            self.assertEqual(again['new_model_calls'],0)
            self.assertEqual(queue,(f.out/'revision_queue.jsonl').read_bytes())
            self.assertEqual(len(model.calls),2)
    def test_corrupt_JPEG_or_wrong_source_is_failure_not_a_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);entry=d.light.load_entry(f.args.source_root,dict(id='source0',split='train'),f.out,d.light.Inputs())
            entry['binding']['source_binding']=dict(entry['binding']['source_binding'],id='different')
            with self.assertRaisesRegex(ValueError,'ID mismatch'):
                d.light.prepare_entry(entry,f.args,d.light.Inputs())
            p=Path(tmp)/'bad.jpg';p.write_bytes(b'not an image')
            with self.assertRaises(OSError):d.light.Inputs().jpeg(p)
    def test_source_time_and_points_remain_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);entry=d.light.load_entry(f.args.source_root,dict(id='source0',split='train'),f.out,d.light.Inputs())
            entry['actual']=copy.deepcopy(entry['actual']);entry['actual']['video_probe']['times'][1]=-1
            with self.assertRaisesRegex(ValueError,'timeline'):
                d.light.prepare_entry(entry,f.args,d.light.Inputs())
    def test_no_resource_authorization_no_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=self.fixture(tmp);model=Model()
            with self.assertRaisesRegex(ValueError,'safety'):
                d.run_consumer(f.args,lambda:(model,NS))
            self.assertEqual(model.calls,[])

if __name__=='__main__':unittest.main()
