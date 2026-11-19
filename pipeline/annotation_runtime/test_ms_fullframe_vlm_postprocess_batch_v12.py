"""N1 32-sequence scope retains all asynchronous-claim and path-continuity contracts."""
import ast
from pathlib import Path
import importlib
import unittest
import test_ms_fullframe_vlm_postprocess_batch_v11 as old
import ms_fullframe_vlm_postprocess_batch_v12 as v
for name in ('test_ms_fullframe_vlm_postprocess_batch_v4','test_ms_fullframe_vlm_postprocess_batch_v5','test_ms_fullframe_vlm_postprocess_batch_v6','test_ms_fullframe_vlm_postprocess_batch_v7','test_ms_fullframe_vlm_postprocess_batch_v10','test_ms_fullframe_vlm_postprocess_batch_v11'):
    importlib.import_module(name).v=v
class N1ConcurrentAsyncTests(old.ParentCacheTests):
    def test_original_native_resource_functions_and_recipe_unchanged(self):
        source=(v.RT/'ms_fullframe_vlm_postprocess_batch_v11.py').read_text().replace('max_num_seqs=16','max_num_seqs=32')
        functions=lambda text:{x.name:ast.dump(x,include_attributes=False) for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)}
        self.assertEqual(functions(source),functions(Path(v.__file__).read_text()))
        self.assertEqual(v.PAIRS['N1'],((0,1),(2,3),(4,5),(6,7)))
        self.assertEqual(v.base.required_budget({'gpus':[0,1]}),160*2**30)
class N1ConcurrentArchiveTests(old.ParentCacheArchiveTests):
    pass
if __name__=='__main__':unittest.main(verbosity=2)
