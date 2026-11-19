"""Live NVML queries for the existing native Gate; no GPU allocation or data TTL."""
import importlib.util
from pathlib import Path

NVML_SOURCE = Path("/usr/local/lib/python3.11/site-packages/pynvml.py")

class LiveQueries:
    def __init__(self, base, gpus, expected_uuids, module=None):
        if module is None:
            spec = importlib.util.spec_from_file_location("ms_live_nvml", NVML_SOURCE)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.base, self.nvml, self.gpus = base, module, tuple(gpus)
        self.expected_uuids = dict(zip(self.gpus, expected_uuids))
        if len(self.expected_uuids) != len(self.gpus) or len(expected_uuids) != len(self.gpus):
            raise RuntimeError("Exact ordered admitted physical UUIDs required")
        self.closed = False
        self.nvml.nvmlInit()
    def __getattr__(self, key):
        return getattr(self.base, key)
    def uuid_for(self, gpu):
        if self.closed or gpu not in self.gpus:
            raise RuntimeError("NVML gate closed or GPU outside admitted physical scope")
        handle = self.nvml.nvmlDeviceGetHandleByIndex(gpu)
        value = self.nvml.nvmlDeviceGetUUID(handle)
        value = value.decode() if isinstance(value, bytes) else value
        if not isinstance(value, str) or not value.startswith("GPU-") or value != self.expected_uuids[gpu]:
            raise RuntimeError("Physical GPU UUID differs from native admission")
        return value
    def apps(self):
        if self.closed:
            raise RuntimeError("NVML gate closed")
        records = []
        for gpu in self.gpus:
            handle = self.nvml.nvmlDeviceGetHandleByIndex(gpu)
            uuid = self.uuid_for(gpu)
            for process in self.nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if type(process.pid) is not int or process.pid <= 0:
                    raise RuntimeError("Invalid live compute process ID")
                used = getattr(process, "usedGpuMemory", None)
                records.append(dict(uuid=uuid, pid=process.pid,
                                    used_MiB=str(used // (1024 * 1024)) if isinstance(used, int) else "unknown"))
        return records
    def close(self):
        if not self.closed:
            self.closed = True
            self.nvml.nvmlShutdown()
