"""Exclude the entire completed4000 pool, including not-yet-registered source plans."""
import unittest
from unittest.mock import patch
import test_ms_vlm_new_native_local_inputs_v1 as old
import ms_vlm_new_native_local_inputs_v2 as m
old.m=m

class ReservedPoolSelectionTests(old.NewSupplySelectionTests):
    def setUp(self):
        super().setUp()
        original=m.v.checked.side_effect
        self.reserved=[]
        def checked(ref):
            if ref['path'].endswith('ms_vlm_native_new4000_inputs_20261028_v1/source_selection.json'):
                values=self.reserved+[('UnusedReserved%04d'%i) for i in range(4000-len(self.reserved))]
                return dict(source_count=4000,source_keys=[['train',x] for x in values])
            return original(ref)
        self.stack.enter_context(patch.object(m.v,'checked',side_effect=checked))

    def test_whole_prior4000_reserved_before_registration_is_excluded(self):
        self.reserved=['New0000','New0001']
        batches,keys,refs=m.select(2)
        self.assertEqual(keys,[['train','New0002'],['train','New0003']])
        self.assertTrue(any(r['path'].endswith('ms_vlm_native_new4000_inputs_20261028_v1/source_selection.json') for r in refs))

    def test_overlap_entire_stream_refuses_new_budget(self):
        self.reserved=self.ids
        with self.assertRaises(ValueError):m.select(1)

if __name__=='__main__':unittest.main(verbosity=2)
