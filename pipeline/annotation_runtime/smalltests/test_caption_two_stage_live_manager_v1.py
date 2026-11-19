"""Native manager startup and real light generation, with CPU synthetic model only."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS,ModuleType
import unittest
from unittest.mock import patch
RT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(RT),str(Path(__file__).resolve().parent)]
import caption_two_stage_live_manager_v1 as m
from test_caption_two_stage_light_v1 import LightTests,Model

class LiveManagerTests(unittest.TestCase):
    def test_original_native_base_and_actual_runtime_configuration(self):
        self.assertEqual(m.base.sha(m.base.__file__),m.BASE_SHA)
        vc=NS(parallel_config=NS(tensor_parallel_size=2),scheduler_config=NS(max_num_seqs=8),
            model_config=NS(multimodal_config=NS(mm_processor_kwargs={'max_pixels':200704})),
            structured_outputs_config=NS(disable_any_whitespace=True))
        llm=NS(llm_engine=NS(vllm_config=vc));m.actual_state(llm)
        vc.scheduler_config.max_num_seqs=2
        with self.assertRaises(ValueError):m.actual_state(llm)
    def test_real_light_worker_three_sources_and_current_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);helper=LightTests();out,args,entries=helper.fixture(base)
            cfgpath=base/'config.json';plan=base/'remaining.json';rows=[{'id':e['row']['id'],'split':'train'} for e in entries]
            m.light.write_once(plan,rows)
            cfg=dict(plan={'path':str(plan)},draft_out=str(out),source_root=str(base/'original'),old_out=str(base/'old'),
                out=str(base/'execution'),node='N4',gpus=[0,1],modelref={'model_identity_sha256':'cpu-model'})
            m.light.write_once(cfgpath,cfg)
            models=[];fake=ModuleType('vllm')
            class Loaded(Model):
                def __init__(self,**kwargs):super().__init__();models.append(self)
            fake.LLM=Loaded;fake.SamplingParams=NS
            def load_entry(root,row,output,inputs):return next(e for e in entries if e['row']['id']==row['id'])
            for row in rows:m.light.write_once(base/'original/sources/train'/row['id']/'producer.terminal.json',dict(status='ready'))
            with (patch.dict(sys.modules,{'vllm':fake}),patch.object(m,'gate',return_value={'admitted':True}),
                 patch.object(m,'actual_state'),patch.object(m.light,'load_entry',side_effect=load_entry),helper.no_expensive_checks()):
                m.worker(cfgpath)
                progress=m.light.load(out/'progress.json');summary=m.light.load(base/'execution/summary.json')
            self.assertEqual([len(x) for x in models[0].calls],[8]);self.assertEqual(summary['processed'],3)
            self.assertEqual(progress['counts'],{'draft_generated':3});self.assertFalse(progress['training_eligible'])
            self.assertEqual(len((out/'revision_queue.jsonl').read_bytes().splitlines()),3)
    def test_parent_admission_rejection_prevents_model_import_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json';m.light.write_once(p,{})
            with patch.object(m,'gate',side_effect=PermissionError('not owned')):
                with self.assertRaises(PermissionError):m.worker(p)
            self.assertEqual([x.name for x in Path(tmp).iterdir()],['config.json'])

if __name__=='__main__':unittest.main()
