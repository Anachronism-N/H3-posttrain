"""Live-query and real cached-Gate integration tests, no GPU or process signals."""
import copy
import types
import unittest
from unittest.mock import Mock, patch
import ms_live_nvml_queries_v2 as queries
import ms_unified_native_nvml_fleet_manager_v1 as caller
import test_ms_cached_native_gate_v3 as fixture

class QueryTests(unittest.TestCase):
    def setUp(self):
        self.nv = types.SimpleNamespace(nvmlInit=Mock(), nvmlShutdown=Mock(),
            nvmlDeviceGetHandleByIndex=Mock(side_effect=lambda i: i),
            nvmlDeviceGetUUID=Mock(return_value="GPU-expected"),
            nvmlDeviceGetComputeRunningProcesses=Mock(return_value=[]))
        self.query = queries.LiveQueries(types.SimpleNamespace(identity="native identity"), [2], ["GPU-expected"], self.nv)
        self.addCleanup(self.query.close)
    def test_real_loader_registers_module_before_execution(self):
        import tempfile, pathlib, sys
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "nvml.py"
            path.write_text("import sys\nassert __name__ in sys.modules\ndef nvmlInit():pass\ndef nvmlShutdown():pass\n")
            with patch.object(queries, "NVML_SOURCE", path):
                q = queries.LiveQueries(None, [2], ["GPU-expected"])
                self.assertIs(q.nvml, sys.modules["ms_live_nvml"])
                q.close()

    def test_requery_live_processes_no_ttl(self):
        self.assertEqual(self.query.apps(), [])
        self.nv.nvmlDeviceGetComputeRunningProcesses.return_value = [types.SimpleNamespace(pid=42, usedGpuMemory=2*1024**2)]
        self.assertEqual(self.query.apps(), [dict(uuid="GPU-expected", pid=42, used_MiB="2")])
        self.assertEqual(self.nv.nvmlDeviceGetComputeRunningProcesses.call_count, 2)
        self.assertEqual(self.query.identity, "native identity")
    def test_bytes_uuid_and_scope(self):
        self.nv.nvmlDeviceGetUUID.return_value = b"GPU-expected"
        self.assertEqual(self.query.uuid_for(2), "GPU-expected")
        with self.assertRaises(RuntimeError): self.query.uuid_for(3)
    def test_changed_uuid_rejected_also_during_apps(self):
        self.query.uuid_for(2)
        self.nv.nvmlDeviceGetUUID.return_value = "GPU-changed"
        with self.assertRaisesRegex(RuntimeError, "differs"): self.query.apps()
    def test_query_error_is_not_empty_apps(self):
        self.nv.nvmlDeviceGetComputeRunningProcesses.side_effect = RuntimeError("driver error")
        with self.assertRaisesRegex(RuntimeError, "driver error"): self.query.apps()
    def test_invalid_pid_rejected(self):
        self.nv.nvmlDeviceGetComputeRunningProcesses.return_value = [types.SimpleNamespace(pid=0)]
        with self.assertRaisesRegex(RuntimeError, "process ID"): self.query.apps()
    def test_close_and_invalid_inventory(self):
        self.query.close(); self.query.close()
        self.nv.nvmlShutdown.assert_called_once()
        with self.assertRaises(RuntimeError): self.query.apps()
        with self.assertRaises(RuntimeError): queries.LiveQueries(None, [2,3], ["GPU-expected"], self.nv)

class NativeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.s = fixture.CachedGateTests(); self.s.setUp()
        self.addCleanup(self.s.doCleanups)
        s = self.s
        self.nv = types.SimpleNamespace(nvmlInit=Mock(), nvmlShutdown=Mock(),
            nvmlDeviceGetHandleByIndex=Mock(side_effect=lambda i:i),
            nvmlDeviceGetUUID=Mock(return_value="GPU-fixture"),
            nvmlDeviceGetComputeRunningProcesses=Mock(return_value=[]))
        cls = queries.LiveQueries
        for k,v in [("base", fixture.base),("h",s.h),("SELF",s.source),("_NATIVE_GATE",None),("_LIVE_QUERIES",None),("_GATE_PROFILE",{"calls":0,"seconds":0.0})]:
            s.stack.enter_context(patch.object(caller,k,v))
        s.stack.enter_context(patch.object(caller.live_queries,"LiveQueries",side_effect=lambda b,g,u:cls(b,g,u,self.nv)))
        self.addCleanup(self.close)
    def close(self):
        if caller._NATIVE_GATE: caller._NATIVE_GATE.close()
        if caller._LIVE_QUERIES: caller._LIVE_QUERIES.close()
    def test_native_checks_and_real_time_gpu_query_each_call(self):
        s=self.s
        self.assertTrue(caller.gate(s.cfg,True)["caller_owned"])
        self.assertTrue(caller.gate(s.cfg,False)["caller_owned"])
        self.assertEqual(self.nv.nvmlDeviceGetComputeRunningProcesses.call_count,2)
        self.assertEqual(s.h.kernel_lock_matches.call_count,2)
        self.assertEqual(s.apps.call_count,0)  # no nvidia-smi subprocess via base.apps
        s.lease.st_mtime-=100
        with self.assertRaisesRegex(PermissionError,"heartbeat"): caller.gate(s.cfg)
    def test_external_process_still_denied_after_cache_hit(self):
        s=self.s;caller.gate(s.cfg)
        foreign=copy.deepcopy(s.current);foreign.update(pid=98765,ppid=1)
        s.identities[98765]=foreign
        root=copy.deepcopy(s.current);root.update(pid=1,ppid=0);s.identities[1]=root
        self.nv.nvmlDeviceGetComputeRunningProcesses.return_value=[types.SimpleNamespace(pid=98765,usedGpuMemory=1)]
        with self.assertRaisesRegex((PermissionError,KeyError),"."): caller.gate(s.cfg)
    def test_before_load_even_own_compute_is_denied(self):
        s=self.s
        self.nv.nvmlDeviceGetComputeRunningProcesses.return_value=[types.SimpleNamespace(pid=s.current["pid"],usedGpuMemory=1)]
        with self.assertRaisesRegex(PermissionError,"occupied before"): caller.gate(s.cfg,True)

if __name__=="__main__":unittest.main()
