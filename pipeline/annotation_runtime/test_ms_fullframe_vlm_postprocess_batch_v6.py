"""CPU regressions for bounded fast-disk prefetch and sixteen active sequences."""
import ast
import json
from pathlib import Path
import threading
from unittest.mock import patch
import unittest
import test_ms_fullframe_vlm_postprocess_batch_v5 as old
import ms_fullframe_vlm_postprocess_batch_v6 as v
old.v=v
old.original.v=v

class PipelineTests(old.BatchV5Tests):
    def setUp(self):
        super().setUp()
        self.fast=patch.object(v,"LOCAL_WORK",self.root/"fastdisk")
        self.fast.start()
        self.addCleanup(self.fast.stop)

    def test_original_native_resource_functions_and_recipe_unchanged(self):
        source=(v.RT/"ms_fullframe_vlm_postprocess_batch_v5.py").read_text().replace("max_num_seqs=8","max_num_seqs=16")
        functions=lambda tree:{x.name:ast.dump(x,include_attributes=False) for x in ast.parse(tree).body if isinstance(x,ast.FunctionDef)}
        before,after=functions(source),functions(Path(v.__file__).read_text())
        for name in ("gate","previous_safe","validate","validate_common","actual_controller_headroom","runtime","actual_model","worker","main","stage_images","request_for","response_schema","audit_check","evidence_policy_violations"):
            self.assertEqual(before[name],after[name],name)
        self.assertEqual(v.base.required_budget({"gpus":[0,1]}),160*2**30)

    def test_next_batch_prepare_overlaps_current_visual_call(self):
        cfg,rows,manifests=self.setup_sources(16,8)
        next_started=threading.Event()
        llm=old.original.FakeLLM()
        def prepare(row,manifest,local,ffmpeg):
            if row["id"]=="Source008":next_started.set()
            return self.images(row,manifest,local,ffmpeg)
        chat=llm.chat
        def checked_chat(*args,**kwargs):
            if not llm.calls:
                self.assertTrue(next_started.wait(5),"Next batch must prefetch while first visual batch runs")
                self.assertTrue((Path(cfg["out"])/"budgets/train/Source008/source.claim.json").is_file())
            return chat(*args,**kwargs)
        llm.chat=checked_chat
        with patch.object(v,"rows_for",return_value=rows),patch.object(v,"gate"),patch.object(v,"stage_images",side_effect=prepare):
            summary=v.run_sources(cfg,llm,lambda x:x)
            v.summaries(cfg,Path(cfg["out"]))
        self.assertEqual([x["sources"] for x in llm.calls],[8,8,8,8])
        self.assertEqual(summary["new_model_calls"],32)
        self.assertEqual(list(v.LOCAL_WORK.iterdir()),[])
        progress=v.load(Path(cfg["out"])/"progress/000008.json")
        self.assertEqual(progress["prefetch_batches"],1)
        self.assertGreaterEqual(progress["preparation_wait_seconds"],0)

    def test_failed_next_prepare_does_not_retry_or_drop_previous(self):
        cfg,rows,llm,summary=self.run_case(count=17,batch=8,fail_prepare="Source008")
        self.assertEqual(summary["processed"],17)
        self.assertEqual(summary["all_outcomes"][8]["status"],"failed_sealed_no_retry")
        self.assertEqual(summary["all_outcomes"][8]["new_model_calls"],0)
        self.assertEqual([x["sources"] for x in llm.calls],[8,8,7,7,1,1])

    def test_fastdisk_symlink_rejected_before_media_copy(self):
        target=self.root/"foreign"
        target.mkdir()
        v.LOCAL_WORK.symlink_to(target,target_is_directory=True)
        cfg,rows,model,summary=self.run_case(count=4,batch=4)
        self.assertEqual(model.calls,[])
        self.assertTrue(all(x["status"]=="failed_sealed_no_retry" for x in summary["all_outcomes"]))
        self.assertEqual(list(target.iterdir()),[])

class PipelineArchiveTests(old.StrictArchiveV5Tests):
    pass

if __name__=="__main__":unittest.main(verbosity=2)
