"""Independent batching, immutable-reference cache and bounded parallel preparation tests."""
import copy
import hashlib
from pathlib import Path
import threading
from unittest.mock import patch
import unittest
import test_ms_fullframe_vlm_postprocess_batch_v4 as original
import ms_fullframe_vlm_postprocess_batch_v5 as v
original.v=v

class BatchV5Tests(original.BatchTests):
    def test_batch_size_outside_four_eight_is_rejected(self):
        for size in (True,1,2,64):
            with self.assertRaises(ValueError):v.run_sources({"batch_sources":size},original.FakeLLM(),lambda x:x)

    def test_sixteen_are_two_chat_calls(self):
        cfg,rows,llm,summary=self.run_case(count=16,batch=16)
        self.assertEqual([x["sources"] for x in llm.calls],[16,16])
        self.assertEqual(summary["new_model_calls"],32)

    def test_thirty_two_and_short_tail(self):
        cfg,rows,llm,summary=self.run_case(count=35,batch=32)
        self.assertEqual([x["sources"] for x in llm.calls],[32,32,3,3])
        self.assertEqual(summary["new_model_calls"],70)

    def test_parallel_prepare_has_four_owned_workers(self):
        cfg,rows,manifests=self.setup_sources(8,8)
        barrier=threading.Barrier(4,timeout=10)
        owners=set()
        lock=threading.Lock()
        def images(row,manifest,local,ffmpeg):
            with lock:owners.add(threading.get_ident())
            barrier.wait()
            return self.images(row,manifest,local,ffmpeg)
        model=original.FakeLLM()
        with patch.object(v,"rows_for",return_value=rows),patch.object(v,"gate"),patch.object(v,"stage_images",side_effect=images):
            result=v.run_sources(cfg,model,lambda x:x)
            v.summaries(cfg,Path(cfg["out"]))
        self.assertEqual(len(owners),4)
        self.assertEqual(result["new_model_calls"],16)

    def test_checked_cache_does_not_alias_returned_json(self):
        p=self.root/"cache.json";reference=v.write_once(p,{"items":["original"]})
        first=v.checked(reference);first["items"].append("mutable")
        self.assertEqual(v.checked(reference),{"items":["original"]})

    def test_changed_cached_reference_is_rejected(self):
        p=self.root/"cache.json";reference=v.write_once(p,{"value":"original"})
        v.checked(reference);p.write_bytes(b'{"value":"changed"}')
        with self.assertRaises(ValueError):v.checked(reference)

    def test_cached_reference_payload_is_read_once(self):
        p=self.root/"cache.json";reference=v.write_once(p,{"value":"original"})
        v.checked(reference)
        with patch.object(Path,"read_bytes",side_effect=AssertionError("No repeated payload read")):
            self.assertEqual(v.checked(reference),{"value":"original"})

class StrictArchiveV5Tests(original.StrictArchiveTests):
    pass

if __name__=="__main__":unittest.main(verbosity=2)
