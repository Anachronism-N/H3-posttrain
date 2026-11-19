"""Thirty-two active sequences preserve the full durable-publication contract."""
import ast
from pathlib import Path
import unittest
import test_ms_fullframe_vlm_postprocess_batch_v7 as old
import ms_fullframe_vlm_postprocess_batch_v9 as v
old.v=v
old.old.v=v
old.old.old.v=v
old.old.old.original.v=v

class Concurrent32Tests(old.FastPublicationTests):
    def test_original_native_resource_functions_and_recipe_unchanged(self):
        src=(v.RT/'ms_fullframe_vlm_postprocess_batch_v7.py').read_text().replace('max_num_seqs=16','max_num_seqs=32')
        functions=lambda text:{x.name:ast.dump(x,include_attributes=False) for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)}
        self.assertEqual(functions(src),functions(Path(v.__file__).read_text()))
        self.assertEqual(v.base.required_budget({'gpus':[2,3]}),160*2**30)

class Concurrent32ArchiveTests(old.FastPublicationArchiveTests):
    pass

if __name__=='__main__':unittest.main(verbosity=2)
