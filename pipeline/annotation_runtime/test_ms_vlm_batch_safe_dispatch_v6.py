"""Thirty-two active sequence safe continuation, preserving all prior guards."""
import unittest
import test_ms_vlm_batch_safe_dispatch_v5 as old
import ms_vlm_batch_safe_dispatch_v6 as d
old.d=d
old.old.d=d
old.old.original.d=d
old.old.original.original.d=d
old.old.original.original.original.d=d
old.old.original.original.original.v=d.v
class Concurrent32DispatchTests(old.PublicationDispatchTests):
    pass
if __name__=='__main__':unittest.main(verbosity=2)
