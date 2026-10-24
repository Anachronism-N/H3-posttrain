"""Synthetic CPU tests of archived live supervisor bytes; no real wait/export."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ENTRY=Path(__file__).resolve().parents[1]/"runtime_jobs/ms_visual_full_postverify_supervisor_v1.py"
spec=importlib.util.spec_from_file_location("supervisor_contract_cpu",ENTRY)
supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.jobs=[{"shard":i,"out":str(self.root/f"job{i}"),"resource_queue":str(self.root/f"resource{i}")} for i in range(8)]

    def write(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def terminal(self,job,exit_code=0):
        c=supervisor.controls(job)
        summary={"selected_ids":1500,"processed_ids":1500,"semantic_approval":False}
        receipt={"exit_code":exit_code,"source_unchanged":True,"input_unchanged":True,"summary_path":str(c/"summary.json"),"summary":summary}
        self.write(c/"summary.json",summary)
        self.write(Path(job["resource_queue"])/"worker_exit.json",receipt)
        (c/"index.jsonl").write_text("serialized fixture index, not actual output")
        self.write(c/"run.receipt.json",{"synthetic":True})
        return c,receipt

    def test_waiting_reads_no_source_frames_or_media(self):
        with mock.patch.object(Path,"read_text",side_effect=AssertionError("No file contents needed for missing terminals")):
            pending,proofs=supervisor.terminal_scan(self.jobs)
        self.assertEqual(len(pending),8)
        self.assertEqual(proofs,[])
        self.assertTrue(all(len(r["missing_artifacts"])==4 for r in pending))

    def test_all_actual_terminal_codes_zero_or_one_can_ready(self):
        for i,job in enumerate(self.jobs):self.terminal(job,i%2)
        pending,proofs=supervisor.terminal_scan(self.jobs)
        self.assertEqual(pending,[])
        self.assertEqual(len(proofs),8)
        self.assertTrue(all(len(p["summary_sha256"])==64 and len(p["worker_exit_sha256"])==64 for p in proofs))

    def test_abnormal_or_untyped_resource_exit_not_complete(self):
        c,receipt=self.terminal(self.jobs[0]);p=Path(self.jobs[0]["resource_queue"])/"worker_exit.json"
        for value in (-9,True,False,"0"):
            self.write(p,{**receipt,"exit_code":value})
            with self.assertRaisesRegex(ValueError,"abnormal"):supervisor.terminal_scan(self.jobs[:1])

    def test_changed_source_or_input_claim_not_terminal(self):
        c,receipt=self.terminal(self.jobs[0]);p=Path(self.jobs[0]["resource_queue"])/"worker_exit.json"
        for key,value in (("source_unchanged",False),("input_unchanged",1)):
            self.write(p,{**receipt,key:value})
            with self.assertRaisesRegex(ValueError,"drift"):supervisor.terminal_scan(self.jobs[:1])

    def test_embedded_summary_bool_int_or_path_reseal_rejected(self):
        c,receipt=self.terminal(self.jobs[0]);p=Path(self.jobs[0]["resource_queue"])/"worker_exit.json"
        changed={**receipt,"summary":{**receipt["summary"],"semantic_approval":0}}
        self.write(p,changed)
        with self.assertRaisesRegex(ValueError,"summary mismatch"):supervisor.terminal_scan(self.jobs[:1])
        self.write(p,{**receipt,"summary_path":"/different-output"})
        with self.assertRaisesRegex(ValueError,"summary mismatch"):supervisor.terminal_scan(self.jobs[:1])

    def test_partial_source_count_never_triggers_export(self):
        c,receipt=self.terminal(self.jobs[0]);summary={**receipt["summary"],"processed_ids":1499}
        self.write(c/"summary.json",summary)
        self.write(Path(self.jobs[0]["resource_queue"])/"worker_exit.json",{**receipt,"summary":summary})
        with self.assertRaisesRegex(ValueError,"incomplete"):supervisor.terminal_scan(self.jobs[:1])

    def test_exclusive_launch_claim_cannot_overwrite(self):
        path=self.root/"launch.claim.json"
        supervisor.exclusive(path,{"claimed_once":True})
        before=path.read_bytes()
        with self.assertRaises(FileExistsError):supervisor.exclusive(path,{"claimed_once":False})
        self.assertEqual(path.read_bytes(),before)

    def test_scope_lock_prevents_second_supervisor(self):
        import fcntl
        path=self.root/"scope.lock"
        with path.open("a") as first,path.open("a") as second:
            fcntl.flock(first,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):fcntl.flock(second,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_fixed_identity_drift_is_rejected_without_model_calls(self):
        path=self.root/"source.py";path.write_bytes(b"CPU source fixture")
        expected={str(path):supervisor.sha(path)}
        with mock.patch.object(supervisor,"EXPECTED",expected):
            supervisor.hashes_unchanged()
            path.write_bytes(b"changed CPU fixture")
            with self.assertRaisesRegex(ValueError,"changed"):supervisor.hashes_unchanged()

    def test_archived_live_source_identity_matches(self):
        self.assertEqual(supervisor.sha(ENTRY),"a93ac7c5eb99ce9639cca1e54f2e3cf2fef6029d5b35aea492c094b0d9b3b3f0")
        self.assertEqual(supervisor.EXPECTED[str(supervisor.EXPORTER)],"4a562a5ff7d5bfcc4b898b60f2e4883e628b3f2f8482e43431286de7f558a558")
        self.assertEqual(supervisor.ARGV[-2:],["--ffprobe","/opt/ms_tools/ffprobe"])

    def test_preflight_never_starts_export_or_creates_registry(self):
        plan=self.root/"plan.json";self.write(plan,{"shards":8,"target_ids":12000,"jobs":self.jobs})
        output=self.root/"future-export";control=self.root/"control-only"
        with mock.patch.object(supervisor,"PLAN",plan),mock.patch.object(supervisor,"OUTPUT",output),mock.patch.object(supervisor,"CONTROL",control),mock.patch.object(supervisor,"hashes_unchanged"),mock.patch.object(supervisor,"duplicate_processes",return_value=[]),mock.patch("sys.argv",[str(ENTRY),"--self-sha",supervisor.sha(ENTRY),"--preflight"]),mock.patch.object(supervisor.subprocess,"Popen",side_effect=AssertionError("No actual export in a synthetic preflight")),redirect_stdout(io.StringIO()):
            self.assertEqual(supervisor.main(),0)
        self.assertFalse(output.exists())
        self.assertFalse(control.exists())


if __name__=="__main__":unittest.main()
