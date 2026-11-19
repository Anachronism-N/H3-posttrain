"""Local codec preserves the exact previously verified prompt and model recipe."""
import ast
from pathlib import Path
import sys
import unittest
import stage2_model_codec_v1 as m

class CodecTests(unittest.TestCase):
    def test_all_extracted_functions_are_ast_identical(self):
        old=Path(m.__file__).parent/'ms_fullframe_vlm_postprocess_batch_v12.py'
        f=lambda text:{x.name:ast.dump(x,include_attributes=False) for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)}
        prior=f(old.read_text());current=f(Path(m.__file__).read_text())
        for name,body in current.items():self.assertEqual(prior[name],body,name)
    def test_codec_does_not_load_model_at_import(self):self.assertNotIn('vllm',sys.modules)
    def test_original_unquantized_recipe_is_preserved(self):
        tree=ast.parse(Path(m.__file__).read_text());r=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='runtime');text=ast.get_source_segment(Path(m.__file__).read_text(),r)
        self.assertIn('dtype="bfloat16"',text);self.assertIn('tensor_parallel_size=2',text);self.assertIn('max_num_seqs=32',text);self.assertIn('max_model_len=16384',text);self.assertIn('max_tokens=1800',text);self.assertNotIn('quantization=',text)
if __name__=='__main__':unittest.main(verbosity=2)
