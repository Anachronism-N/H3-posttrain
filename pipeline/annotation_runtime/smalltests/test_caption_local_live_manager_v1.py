"""Real local copy/archive pipeline with CPU model and next-batch overlap proof."""
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace as NS
from threading import Event
import unittest
from unittest.mock import patch
RT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(RT),str(Path(__file__).resolve().parent)]
import caption_local_live_manager_v1 as d
import test_caption_two_stage_generation_v2 as fixtures
import test_caption_two_stage_light_v1 as models

class LocalLiveTests(unittest.TestCase):
    def test_six_sources_two_batches_next_copy_overlaps_current_GPU_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);f=fixtures.LightLaneTests().fixture(tmp);out=base/'caption_two_stage_local_test';exe=base/'execution';exe.mkdir()
            rows=[{'id':r['id'],'split':'train'} for r in f.entries];plan=base/'plan.json';d.local.light.write_once(plan,rows)
            cfgpath=base/'config.json';d.local.light.write_once(cfgpath,dict(plan={'path':str(plan)},draft_out=str(out),
                source_root=str(f.original),out=str(exe),node='N6',gpus=[2,3],source_handoff={'path':'/cpu/handoff'},
                modelref={'model_identity_sha256':'CPU-model'}))
            overlap=Event();model_list=[];original=d.local.stage_source
            def stage(root,meta,path):
                result=original(root,meta,path)
                if meta['id']=='source3':overlap.set()
                return result
            class Model(models.Model):
                def __init__(self,**kwargs):super().__init__();model_list.append(self)
                def chat(self,*a,**kw):
                    if not self.calls:assert overlap.wait(3),'Next batch did not copy during current model phase'
                    return super().chat(*a,**kw)
            fake=ModuleType('vllm');fake.LLM=Model;fake.SamplingParams=NS
            with (patch.dict(sys.modules,{'vllm':fake}),patch.object(d,'gate',return_value={'admitted':True}),
                patch.object(d.local,'stage_source',side_effect=stage),models.LightTests().no_expensive_checks(),
                patch.object(d.local.light.subprocess,'run',side_effect=fixtures.LightLaneTests().decode)):
                d.worker(cfgpath)
            self.assertEqual(len(model_list),1);self.assertEqual([len(x) for x in model_list[0].calls],[8,8])
            progress=d.local.light.load(out/'progress.json');self.assertEqual(progress['processed'],6)
            self.assertTrue(progress['next_batch_copy_overlaps_GPU']);self.assertFalse(progress['training_eligible'])
            archives=list(out.glob('*.tar'));self.assertEqual(len(archives),2)
            drafts=[value for archive in archives for _,value in d.local.archive_drafts(archive)]
            self.assertEqual(len(drafts),6);self.assertEqual(len({x['id'] for x in drafts}),6)
    def test_resource_admission_failure_does_not_load_or_create_drafts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=Path(tmp)/'config.json';d.local.light.write_once(cfg,{})
            with patch.object(d,'gate',side_effect=PermissionError('not owned')):
                with self.assertRaises(PermissionError):d.worker(cfg)
            self.assertEqual([x.name for x in Path(tmp).iterdir()],['config.json'])
    def test_native_resource_base_identity_is_preserved(self):
        self.assertEqual(d.base.sha(d.base.__file__),d.BASE_SHA)

if __name__=='__main__':unittest.main()
