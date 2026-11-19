"""N1 higher concurrency continuation retains frozen predecessor and once-budget checks."""
import importlib
import unittest
import test_ms_vlm_batch_safe_dispatch_v8 as old
import ms_vlm_batch_safe_dispatch_v9 as d
for name in ('test_ms_vlm_batch_safe_dispatch_v1','test_ms_vlm_batch_safe_dispatch_v2','test_ms_vlm_batch_safe_dispatch_v3','test_ms_vlm_batch_safe_dispatch_v4','test_ms_vlm_batch_safe_dispatch_v5','test_ms_vlm_batch_safe_dispatch_v7','test_ms_vlm_batch_safe_dispatch_v8'):
    m=importlib.import_module(name);m.d=d
    if name=='test_ms_vlm_batch_safe_dispatch_v1':m.v=d.v
class N1ConcurrentContinuationTests(old.N1ContinuationTests):
    pass
if __name__=='__main__':unittest.main(verbosity=2)
