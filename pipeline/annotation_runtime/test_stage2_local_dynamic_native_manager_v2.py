"""Native admission includes the locally frozen dynamic pool plan required by base lifecycle."""
import importlib
import unittest
import test_stage2_local_dynamic_native_manager_v1 as old
import stage2_local_dynamic_native_manager_v2 as m
old.m=m
class NativePlanTests(old.NativeTests):
    def test_original_native_lifecycle_remains_exact(self):
        self.assertEqual(m.b.sha(m.b.__file__),m.ORIGINAL_BASE_SHA)
        self.assertIn("cfg['plan']",__import__('pathlib').Path(m.__file__).read_text())
if __name__=='__main__':unittest.main(verbosity=2)
