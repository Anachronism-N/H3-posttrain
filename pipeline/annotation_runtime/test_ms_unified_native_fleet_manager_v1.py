"""Caller integration with real cached Gate and native physical-ownership CPU fixtures."""
import copy
import unittest
from unittest.mock import patch
import ms_unified_native_fleet_manager_v1 as caller
import test_ms_cached_native_gate_v2 as fixture

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
    def test_N3_restored_GPU0_dynamic_gate(self):
        import os
        s = self.subject
        s.cfg["node"], s.cfg["gpus"] = "N3", [0]
        s.cfg["env"]["CUDA_VISIBLE_DEVICES"] = "0"
        s.cfg["holders"][0]["gpu"] = 0
        s.current["CVD"] = ["CUDA_VISIBLE_DEVICES=0"]
        s.admission.update(node="N3", gpus=[0], holders=copy.deepcopy(s.cfg["holders"]))
        s.live["child_identity"] = copy.deepcopy(s.current)
        s.write_receipts()
        original_lstat, original_readlink = os.lstat, os.readlink
        with patch.object(os, "lstat", side_effect=lambda p, *a, **k:
                          s.lock if os.fspath(p) == "/tmp/ms12k_gpu_0.lock" else
                          s.lease if os.fspath(p) == "/tmp/od_reserve_0" else original_lstat(p, *a, **k)),              patch.object(os, "readlink", side_effect=lambda p, **k:
                          "/tmp/ms12k_gpu_0.lock" if os.fspath(p) == s.fd_dir + "/10" else
                          "/tmp/od_reserve_0" if os.fspath(p) == s.fd_dir + "/11" else original_readlink(p, **k)):
            result = caller.gate(s.cfg, True)
            self.assertTrue(result["caller_owned"])
            self.assertEqual(caller._NATIVE_GATE.cfg["node"], "N3")
            self.assertEqual(s.uuid.call_args.args, (0,))

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
