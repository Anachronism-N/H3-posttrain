"""Asynchronous durable source claims do not block the current inference batch."""
import ast
import stat
import threading
from pathlib import Path
from unittest.mock import patch
import unittest
import test_ms_fullframe_vlm_postprocess_batch_v7 as old
import ms_fullframe_vlm_postprocess_batch_v10 as v
old.v=v
old.old.v=v
old.old.old.v=v
old.old.old.original.v=v

class AsyncClaimTests(old.FastPublicationTests):
    def test_original_native_resource_functions_and_recipe_unchanged(self):
        functions=lambda text:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
        before=functions((v.RT/'ms_fullframe_vlm_postprocess_batch_v7.py').read_text());after=functions(Path(v.__file__).read_text())
        for name in ('gate','previous_safe','validate','validate_common','runtime','actual_model','worker','main','stage_images','summaries','source_summaries'):
            self.assertEqual(before[name],after[name],name)
        self.assertEqual(v.base.required_budget({'gpus':[2,3]}),160*2**30)

    def test_four_source_claims_fsync_in_parallel_before_model_call(self):
        barrier=threading.Barrier(4,timeout=5)
        fsync=v.os.fsync
        seen=set()
        lock=threading.Lock()
        def sync(fd):
            target=Path('/proc/self/fd/'+str(fd)).resolve()
            if target.name=='source.claim.json':
                with lock:seen.add(threading.get_ident())
                barrier.wait()
            return fsync(fd)
        with patch.object(v.os,'fsync',side_effect=sync):
            cfg,rows,model,summary=self.run_case(count=4,batch=4)
        self.assertEqual(len(seen),4)
        self.assertEqual(summary['new_model_calls'],8)

    def test_source_claim_must_finish_before_media_copy(self):
        cfg,rows,manifests=self.setup_sources(4,4)
        def images(row,manifest,local,ffmpeg):
            path=Path(cfg['out'])/'budgets'/row['split']/row['id']/'source.claim.json'
            value=v.load(path);self.assertEqual(value['source'],row)
            return self.images(row,manifest,local,ffmpeg)
        model=old.old.old.original.FakeLLM()
        with patch.object(v,'rows_for',return_value=rows),patch.object(v,'gate'),patch.object(v,'stage_images',side_effect=images):
            summary=v.run_sources(cfg,model,lambda x:x);v.summaries(cfg,Path(cfg['out']))
        self.assertEqual(summary['new_model_calls'],8)

class AsyncClaimArchiveTests(old.FastPublicationArchiveTests):
    pass

if __name__=='__main__':unittest.main(verbosity=2)
