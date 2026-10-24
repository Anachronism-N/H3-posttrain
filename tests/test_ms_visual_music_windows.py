"""CPU-only shared-window schema fixtures; no VLM, detector or actual MS run."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

MS=Path(__file__).resolve().parents[1]/"scripts_omnidance/exp/ms"
sys.path.insert(0,str(MS))
import ms_visual_music_windows as visual


class MusicVisualTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.binding={"id":visual.IDS[0],"input_sha256":"a"*64,"video_sha256":"b"*64,"audio_sha256":"c"*64,
            "actual_video_pts_sha256":"d"*64,"media_start":0.0,"media_end":5.0625,"video_pts_offset":0.0,"video_fps":16.0,"decoded_video_frames":81}

    def plan(self,sid,bpm=120):
        binding={**self.binding,"id":sid}
        return visual.music.bpm_duration_windows(binding,bpm,beats_per_chunk=4,min_tail_seconds=1.5,
            provenance={"source":"synthetic CPU fixture, not a BPM estimate"})

    def shared_fixture(self):
        plans=[self.plan(sid) for sid in visual.IDS]
        records=[]
        for p in plans:
            path=self.root/(p["id"]+".windows.json")
            path.write_bytes(visual.batch.json_bytes(p))
            records.append({"id":p["id"],"window_file":path.name,"window_sha256":visual.batch.sha256(path)})
        raw=b"".join((json.dumps(p)+"\n").encode() for p in plans)
        path=self.root/"windows.jsonl";path.write_bytes(raw)
        sha=hashlib.sha256(raw).hexdigest()
        summary={"complete":True,"processed_count":3,"failed_count":0,"errors":[],"records":records,"window_manifest_sha256":sha}
        (self.root/"summary.json").write_bytes(visual.batch.json_bytes(summary))
        (self.root/"summary.receipt.json").write_bytes(visual.batch.json_bytes({"summary_sha256":visual.batch.sha256(self.root/"summary.json"),"window_manifest_sha256":sha}))
        return path,sha,plans

    def test_shared_exact_same_three_ids_and_window_bytes(self):
        path,sha,plans=self.shared_fixture()
        actual,ref=visual.read_shared(path,sha)
        self.assertEqual(actual,plans)
        self.assertEqual(ref["sha256"],sha)
        self.assertEqual([p["id"] for p in actual],visual.IDS)

    def test_changed_jsonl_or_source_window_file_rejected(self):
        path,sha,plans=self.shared_fixture()
        raw=path.read_bytes();path.write_bytes(raw+b"\n")
        with self.assertRaisesRegex(ValueError,"JSONL changed"):
            visual.read_shared(path,sha)
        path.write_bytes(raw)
        (self.root/(plans[0]["id"]+".windows.json")).write_text("{}")
        with self.assertRaisesRegex(ValueError,"source shared window receipt changed"):
            visual.read_shared(path,sha)

    def test_nonterminal_summary_not_treated_as_available_windows(self):
        path,sha,plans=self.shared_fixture()
        summary=json.loads((self.root/"summary.json").read_text());summary["complete"]=False
        (self.root/"summary.json").write_bytes(visual.batch.json_bytes(summary))
        receipt=json.loads((self.root/"summary.receipt.json").read_text());receipt["summary_sha256"]=visual.batch.sha256(self.root/"summary.json")
        (self.root/"summary.receipt.json").write_bytes(visual.batch.json_bytes(receipt))
        with self.assertRaisesRegex(ValueError,"not terminal"):
            visual.read_shared(path,sha)

    def test_wrong_source_order_or_heldout_not_relabelled_authority(self):
        path,sha,plans=self.shared_fixture()
        plans[0],plans[1]=plans[1],plans[0]
        raw=b"".join((json.dumps(p)+"\n").encode() for p in plans);path.write_bytes(raw);newsha=hashlib.sha256(raw).hexdigest()
        summary=json.loads((self.root/"summary.json").read_text());summary["window_manifest_sha256"]=newsha
        (self.root/"summary.json").write_bytes(visual.batch.json_bytes(summary))
        receipt=json.loads((self.root/"summary.receipt.json").read_text());receipt["window_manifest_sha256"]=newsha;receipt["summary_sha256"]=visual.batch.sha256(self.root/"summary.json")
        (self.root/"summary.receipt.json").write_bytes(visual.batch.json_bytes(receipt))
        with self.assertRaisesRegex(ValueError,"same first three authority"):
            visual.read_shared(path,newsha)

    def pts_fixture(self):
        path,sha,plans=self.shared_fixture()
        actual,ref=visual.read_shared(path,sha)
        verifier=Path(visual.exact_pts.__file__)
        records=[{"id":p["id"],"policy":"bpm","plan":{"path":str(self.root/(p["id"]+".windows.json")),"sha256":visual.batch.sha256(self.root/(p["id"]+".windows.json"))},"actual_video_pts_sha256":p["binding"]["actual_video_pts_sha256"],"exact_pts_digest_equal_original":True,"canonical_typed_plan_verified":True} for p in plans]
        value={"schema":"music_window_exact_pts_typed_verification_v1","complete":True,
            "input_manifest_sha256":"input-fixture-sha","actual_video_pts_probed_not_inferred_from_nlf":True,
            "old_reports_and_captions_modified":False,"MS_or_model_inference_repeated":False,"GPU_used":False,
            "source_sha256":visual.batch.sha256(Path(visual.music.__file__)),"verifier_path":str(verifier),"verifier_sha256":visual.batch.sha256(verifier),
            "bundle_counts":{str(self.root):{"summary_sha256":ref["summary_sha256"]}},"records":records}
        output=self.root/"pts-audit.json";output.write_bytes(visual.batch.json_bytes(value))
        return output,visual.batch.sha256(output),[(plans,ref)],value

    def test_independent_real_pts_type_receipt_required_and_bound(self):
        path,sha,strategies,value=self.pts_fixture()
        got=visual.verify_pts_receipt(path,sha,strategies,"input-fixture-sha")
        self.assertEqual(got["sha256"],sha)
        self.assertTrue(got["actual_video_pts_probed_not_inferred_from_nlf"])
        for key,new in (("complete",1),("actual_video_pts_probed_not_inferred_from_nlf",False),("input_manifest_sha256","other")):
            changed=copy.deepcopy(value);changed[key]=new;path.write_bytes(visual.batch.json_bytes(changed))
            with self.assertRaises(ValueError):
                visual.verify_pts_receipt(path,visual.batch.sha256(path),strategies,"input-fixture-sha")

    def test_independent_pts_receipt_rejects_missing_selected_source_or_typed_flag(self):
        path,sha,strategies,value=self.pts_fixture()
        for change in ("missing","typed"):
            changed=copy.deepcopy(value)
            if change=="missing":changed["records"].pop()
            else:changed["records"][0]["canonical_typed_plan_verified"]=1
            path.write_bytes(visual.batch.json_bytes(changed))
            with self.assertRaisesRegex(ValueError,"no independent"):
                visual.verify_pts_receipt(path,visual.batch.sha256(path),strategies,"input-fixture-sha")

    def test_half_open_shared_boundaries_not_rebased_fixed2s(self):
        times=[i/16 for i in range(81)]
        a,status=visual.select_policy(times,{"start":0.,"end":.16})
        b,status2=visual.select_policy(times,{"start":.16,"end":2.16})
        self.assertEqual(a,[0,2])
        self.assertEqual(b[0],3)
        self.assertEqual(b[-1],34)
        self.assertFalse(set(a).intersection(b))
        self.assertEqual((status,status2),("prepared","prepared"))

    def test_short_window_less_two_true_frames_abstain_no_frame_fill(self):
        times=[i/16 for i in range(81)]
        indices,status=visual.select_policy(times,{"start":0.,"end":.01})
        self.assertEqual(indices,[0])
        self.assertEqual(status,"abstain_insufficient_source_frames")
        indices,status=visual.select_policy(times,{"start":.01,"end":.02})
        self.assertEqual(indices,[])
        self.assertEqual(status,"abstain_insufficient_source_frames")

    def test_long_window_frame_budget16_preserves_last_actual_support(self):
        times=[i/16 for i in range(81)]
        indices,status=visual.select_policy(times,{"start":0.,"end":5.0625})
        self.assertEqual(len(indices),16)
        self.assertEqual(indices[0],0);self.assertEqual(indices[-1],80)
        self.assertEqual(status,"prepared")

    def test_source_binding_checks_actual_nlf_grid_not_timestamp_guess(self):
        path=self.root/"fixture.npz"
        import numpy as np
        np.savez(path,pose=np.zeros((81,72)),fps=np.asarray(16.0))
        source={"files":{"nlf_path":{"sha256":"a"*64},"video_path":{"sha256":"b"*64},"audio_path":{"sha256":"c"*64}}}
        probe={"times":[i/16 for i in range(81)],"media_end":5.0625,"time_origin_source_timestamp":0.0}
        binding=visual.actual_binding({"id":visual.IDS[0],"nlf_path":str(path)},source,probe)
        self.assertEqual(binding["decoded_video_frames"],81)
        self.assertEqual(binding["media_end"],5.0625)
        self.assertEqual(binding["actual_video_pts_sha256"],hashlib.sha256(visual.batch.json_bytes(probe["times"])).hexdigest())
        probe["times"][1]=.1
        with self.assertRaisesRegex(ValueError,"time grid"):
            visual.actual_binding({"id":visual.IDS[0],"nlf_path":str(path)},source,probe)

    def test_schema_does_not_promote_music_grid_to_real_bar(self):
        p=self.plan(visual.IDS[0])
        self.assertEqual(p["strategy"],"bpm_duration_grid_not_detected_beats")
        self.assertIs(p["beat_phase_verified"],False)
        self.assertIs(p["downbeat_verified"],False)
        self.assertIs(p["audio_video_alignment_verified"],False)
        changed=copy.deepcopy(p);changed["windows"][0]["end"]+=.1
        with self.assertRaises(ValueError):visual.music.validate_shared_plan(changed,p["binding"])

    def test_summary_model_vs_support_abstain_and_partial_delivery_separate(self):
        records=[{"plan_status":"ready","windows":[{"status":"candidate","caption":"Movement.","model_call_performed":True},
            {"status":"abstain","caption":"","model_call_performed":False,"support_reason":"fewer frames"}]},
            {"plan_status":"insufficient_downbeats","windows":[]}]
        s=visual.summarize(records)
        self.assertEqual(s["window_count"],2)
        self.assertEqual(s["model_executed_window_count"],1)
        self.assertEqual(s["insufficient_frame_abstention_count"],1)
        self.assertEqual(s["pending_insufficient_downbeat_plan_count"],1)
        self.assertEqual(s["usable_complete_strategy_source_count"],0)
        self.assertTrue(s["VLM_shared_manifest_actual_execution"])
        self.assertFalse(s["all_plans_executed"])
        for key in ("semantic_approval","human_review_completed","formal_training","full_pool_updated"):
            self.assertIs(s[key],False)

    def test_prepare_only_never_claims_vlm_shared_actual_execution(self):
        s=visual.summarize([{"plan_status":"ready","windows":[{"status":"prepared","caption":"","model_call_performed":False}]}])
        self.assertFalse(s["VLM_shared_manifest_actual_execution"])
        self.assertFalse(s["all_ready_windows_terminal"])
        self.assertEqual(s["window_status_counts"]["prepared"],1)


if __name__=="__main__":unittest.main()
