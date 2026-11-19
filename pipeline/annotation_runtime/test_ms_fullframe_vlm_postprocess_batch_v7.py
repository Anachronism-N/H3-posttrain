"""Durable publication retains all bounded fast-disk pipeline regressions."""
import ast
import os
from pathlib import Path
from unittest.mock import patch
import unittest
import test_ms_fullframe_vlm_postprocess_batch_v6 as old
import ms_fullframe_vlm_postprocess_batch_v7 as v
old.v=v
old.old.v=v
old.old.original.v=v

class FastPublicationTests(old.PipelineTests):
    def test_original_native_resource_functions_and_recipe_unchanged(self):
        functions=lambda src:{x.name:ast.dump(x,include_attributes=False) for x in ast.parse(src).body if isinstance(x,ast.FunctionDef)}
        before=functions((v.RT/"ms_fullframe_vlm_postprocess_batch_v6.py").read_text())
        after=functions(Path(v.__file__).read_text())
        for name in ("gate","previous_safe","validate","validate_common","runtime","actual_model","worker","main","stage_images","run_sources","summaries","source_summaries"):
            self.assertEqual(before[name],after[name],name)
        self.assertEqual(v.base.required_budget({"gpus":[0,1]}),160*2**30)

    def test_durable_write_keeps_file_then_directory_fsync(self):
        calls=[]
        fsync=os.fsync
        def capture(fd):
            calls.append(Path('/proc/self/fd/'+str(fd)).resolve())
            return fsync(fd)
        p=self.root/'durable/result.json'
        with patch.object(v.os,'fsync',side_effect=capture):reference=v.write_once(p,{"value":"actual"})
        self.assertEqual(calls[-2:],[p,p.parent])
        self.assertEqual(v.checked(reference),{"value":"actual"})

    def test_publication_references_do_not_rehash_own_unchanged_files(self):
        p=self.root/'refs/value.json';reference=v.write_once(p,{"value":"actual"})
        with patch.object(v.base,'sha',side_effect=AssertionError('No repeated payload hash')):
            self.assertEqual(v.ref(p),reference)
            self.assertEqual(v.checked(reference),{"value":"actual"})

    def test_published_file_change_is_not_served_from_write_cache(self):
        p=self.root/'refs/value.json';reference=v.write_once(p,{"value":"original"})
        p.write_bytes(b'{"value":"changed"}')
        self.assertNotEqual(v.ref(p),reference)
        with self.assertRaises(ValueError):v.checked(reference)

    def test_directory_cache_remains_bounded(self):
        for i in range(140):v.write_once(self.root/('dir%03d'%i)/'value.json',{"i":i})
        self.assertLessEqual(len(v._DIRECTORY_FDS),128)
        self.assertLessEqual(len(v._OWN_CACHE_KEYS),256)

class FastPublicationArchiveTests(old.PipelineArchiveTests):
    pass

if __name__=='__main__':unittest.main(verbosity=2)
