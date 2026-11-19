"""Stable parent-path validation is amortized while leaf identity is checked."""
import ast
from pathlib import Path
from unittest.mock import patch
import unittest
import test_ms_fullframe_vlm_postprocess_batch_v10 as old
import ms_fullframe_vlm_postprocess_batch_v11 as v
old.v=v
old.old.v=v
old.old.old.v=v
old.old.old.old.v=v
old.old.old.old.original.v=v

class ParentCacheTests(old.AsyncClaimTests):
    def test_original_native_resource_functions_and_recipe_unchanged(self):
        functions=lambda text:{x.name:ast.dump(x,include_attributes=False) for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)}
        before=functions((v.RT/'ms_fullframe_vlm_postprocess_batch_v10.py').read_text());after=functions(Path(v.__file__).read_text())
        for name in ('gate','previous_safe','validate','validate_common','runtime','actual_model','worker','main','stage_images','run_sources','summaries','source_summaries'):
            self.assertEqual(before[name],after[name],name)

    def test_parent_recursive_resolve_happens_only_at_first_boundary(self):
        p=self.root/'videos';p.mkdir();v.validate_video_path(p/'First.mp4','First')
        with patch.object(Path,'resolve',side_effect=AssertionError('No repeated parent resolve')):
            v.validate_video_path(p/'Second.mp4','Second')

    def test_parent_inode_change_rejected(self):
        p=self.root/'videos';p.mkdir();v.validate_video_path(p/'First.mp4','First');p.rename(self.root/'oldvideos');p.mkdir()
        with self.assertRaises(ValueError):v.validate_video_path(p/'Second.mp4','Second')

    def test_new_symlink_parent_rejected(self):
        real=self.root/'real';real.mkdir();p=self.root/'alias';p.symlink_to(real,target_is_directory=True)
        with self.assertRaises(ValueError):v.validate_video_path(p/'First.mp4','First')

    def test_video_leaf_symlink_rejected_even_after_parent_cache(self):
        p=self.root/'videos';p.mkdir();v.validate_video_path(p/'First.mp4','First');target=p/'Other.mp4';target.write_bytes(b'video');(p/'Second.mp4').symlink_to(target)
        with self.assertRaises(ValueError):v.validate_video_path(p/'Second.mp4','Second')

class ParentCacheArchiveTests(old.AsyncClaimArchiveTests):
    pass

if __name__=='__main__':unittest.main(verbosity=2)
