"""N1 stable predecessor continuation retains all native ownership guards."""
import unittest
import test_ms_vlm_batch_safe_dispatch_v7 as old
import ms_vlm_batch_safe_dispatch_v8 as d
old.d=d
old.old.d=d
old.old.old.d=d
old.old.old.original.d=d
old.old.old.original.original.d=d
old.old.old.original.original.original.d=d
old.old.old.original.original.original.v=d.v
class N1ContinuationTests(old.AsyncParentDispatchTests):
    pass
if __name__=='__main__':unittest.main(verbosity=2)
