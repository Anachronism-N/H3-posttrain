"""Caller integration with real cached Gate and native physical-ownership CPU fixtures."""
import copy
import unittest
from unittest.mock import patch
import ms_unified_native_cached_manager_v1 as caller
import test_ms_cached_native_gate_v1 as fixture

class CallerGateTests(unittest.TestCase):
    def setUp(self):
        self.subject = fixture.CachedGateTests()
        self.subject.setUp()
        self.addCleanup(self.subject.doCleanups)
        s = self.subject
        self.stack = s.stack
        for name, value in (("base", fixture.base), ("h", s.h), ("SELF", s.source),
                            ("_NATIVE_GATE", None), ("_GATE_PROFILE", {"calls": 0, "seconds": 0.0})):
            self.stack.enter_context(patch.object(caller, name, value))
        self.addCleanup(self.close_gate)
    def close_gate(self):
        if caller._NATIVE_GATE is not None:
            caller._NATIVE_GATE.close()
    def test_real_helper_used_once_dynamic_gate_every_stage(self):
        s = self.subject
        proof = caller.gate(s.cfg, True)
        self.assertTrue(proof["fixed_evidence_cache"])
        gate = caller._NATIVE_GATE
        reads = [item.read_calls for item in gate._files]
        for _ in range(4):
            self.assertTrue(caller.gate(s.cfg, False)["caller_owned"])
        self.assertIs(caller._NATIVE_GATE, gate)
        self.assertEqual(reads, [item.read_calls for item in gate._files])
        self.assertEqual(s.apps.call_count, 5)
        self.assertEqual(s.uuid.call_count, 5)
        self.assertEqual(caller._GATE_PROFILE["calls"], 5)
        self.assertEqual(gate.profile["calls"], 5)
    def test_fixed_file_mutation_cannot_be_recached(self):
        s = self.subject
        caller.gate(s.cfg)
        s.source.write_bytes(s.source.read_bytes() + b"# changed")
        with self.assertRaisesRegex(PermissionError, "evidence changed"):
            caller.gate(s.cfg)
        with self.assertRaisesRegex(PermissionError, "closed or failed"):
            caller.gate(s.cfg)
    def test_dynamic_lease_expiry_still_rejected_after_cache_hit(self):
        s = self.subject
        caller.gate(s.cfg)
        s.lease.st_mtime -= 100
        with self.assertRaisesRegex(PermissionError, "heartbeat"):
            caller.gate(s.cfg)
        self.assertTrue(all(item.fd is None for item in caller._NATIVE_GATE._files))
    def test_before_load_occupied_same_worker_not_permitted(self):
        s = self.subject
        s.apps.return_value = [{"uuid": "GPU-fixture", "pid": s.current["pid"]}]
        with self.assertRaisesRegex(PermissionError, "occupied before"):
            caller.gate(s.cfg, True)

if __name__ == "__main__":
    unittest.main()
