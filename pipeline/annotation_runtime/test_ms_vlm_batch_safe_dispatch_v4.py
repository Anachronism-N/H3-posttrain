"""Independent safe continuation for the bounded local pipeline caller."""
import unittest
import test_ms_vlm_batch_safe_dispatch_v3 as original
import ms_vlm_batch_safe_dispatch_v4 as d
original.d=d
original.original.d=d
original.original.original.d=d
original.original.original.v=d.v

class FastDiskDispatchTests(original.OptimizedDispatchTests):
    pass

if __name__=="__main__":unittest.main(verbosity=2)
