"""The N1 scope extension retains the complete bounded-pipeline contract."""
import ast
from pathlib import Path
import unittest
import test_ms_fullframe_vlm_postprocess_batch_v6 as old
import ms_fullframe_vlm_postprocess_batch_v8 as v
old.v=v
old.old.v=v
old.old.original.v=v

class N1ScopePipelineTests(old.PipelineTests):
    def test_original_native_resource_functions_and_recipe_unchanged(self):
        functions=lambda text:{x.name:ast.dump(x,include_attributes=False) for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)}
        before=functions((v.RT/'ms_fullframe_vlm_postprocess_batch_v6.py').read_text());after=functions(Path(v.__file__).read_text())
        self.assertEqual(before,after)
        self.assertEqual(v.HOSTS['N1'],'28.216.19.69')
        self.assertEqual(v.PAIRS['N1'],((0,1),(2,3),(4,5),(6,7)))
        self.assertEqual(v.base.required_budget({'gpus':[0,1]}),160*2**30)

class N1ScopeArchiveTests(old.PipelineArchiveTests):
    pass

if __name__=='__main__':unittest.main(verbosity=2)
