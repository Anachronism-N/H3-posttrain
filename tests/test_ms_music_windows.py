"""Synthetic contracts only; these tests do not establish real MS/beat accuracy."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance" / "exp" / "ms"
sys.path.insert(0, str(MS))
import ms_music_windows as W
import ms_chunk_caption as original
import test_ms_chunk_caption as fixtures


def binding(end=5.0625):
    return {"id": "synthetic", "input_sha256": "1"*64, "video_sha256": "2"*64,
            "audio_sha256": "3"*64, "actual_video_pts_sha256": "4"*64, "media_start": 0.,
            "media_end": end, "video_pts_offset": 3., "video_fps": 16., "decoded_video_frames": 81}


def source():
    return {"method": "beat_this.inference.File2Beats", "checkpoint_name": "final0", "dbn": False,
            "device": "cpu", "synthetic": False, "downbeats_origin": "model_downbeat_head_postprocessed",
            "checkpoint_sha256": "a"*64, "audio_sha256": "3"*64,
            "library_source_sha256": "b"*64, "library_version": "fixture-not-real"}


class WindowPolicyTests(unittest.TestCase):
    def test_bpm_actual_formula_grid_and_short_tail_original_mapping(self):
        plan = W.bpm_duration_windows(binding(), 120.)
        self.assertEqual(plan["strategy"], "bpm_duration_grid_not_detected_beats")
        self.assertEqual(plan["params"]["target_seconds"], 4*60/120)
        self.assertEqual([(w["start"], w["end"]) for w in plan["raw_windows"]], [(0,2),(2,4),(4,5.0625)])
        self.assertEqual(plan["windows"], [{"start": 0., "end": 2., "raw_window_indices": [0]},
                                            {"start": 2., "end": 5.0625, "raw_window_indices": [1,2]}])
        self.assertFalse(plan["params"]["first_zero_is_beat_phase"])
        self.assertFalse(plan["beat_phase_verified"])
        self.assertFalse(plan["downbeat_verified"])
        W.validate_shared_plan(plan, binding())

    def test_bpm_low_high_no_detected_beat_limits_or_retiming(self):
        for bpm, target in ((30.,8.), (240.,1.)):
            plan = W.bpm_duration_windows(binding(12.25), bpm)
            self.assertEqual(plan["params"]["target_seconds"], target)
            self.assertEqual(plan["raw_windows"][0]["end"], target)
            self.assertEqual(plan["windows"][-1]["end"],12.25)
            self.assertEqual([i for w in plan["windows"] for i in w["raw_window_indices"]], list(range(len(plan["raw_windows"]))))

    def test_short_clip_and_exact_threshold_tail_and_exact_grid_end(self):
        self.assertEqual(len(W.bpm_duration_windows(binding(.4),30.)["windows"]),1)
        exact = W.bpm_duration_windows(binding(5.5),120.)
        self.assertEqual(len(exact["windows"]),3)
        self.assertEqual(exact["windows"][-1]["raw_window_indices"],[2])
        self.assertEqual(len(W.bpm_duration_windows(binding(4.),120.)["raw_windows"]),2)

    def test_invalid_typed_bpm_or_duration_refused(self):
        for bpm in (True, "120", 0, -1, float("nan"), float("inf")):
            with self.subTest(bpm=bpm), self.assertRaises(ValueError):
                W.bpm_duration_windows(binding(), bpm)
        for key, value in (("media_end", True),("media_start",False),("decoded_video_frames",True),
                           ("video_sha256","oops"),("video_pts_offset","0")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                W.bpm_duration_windows({**binding(), key:value},120.)
        for kwargs in ({"beats_per_chunk":True},{"beats_per_chunk":0},{"min_tail_seconds":-1}):
            with self.assertRaises(ValueError): W.bpm_duration_windows(binding(),120.,**kwargs)

    def test_strict_keeps_every_complete_bar_and_two_edges(self):
        db = [.13,2.07,4.09]
        plan = W.bar_windows(binding(),db,source=source())
        self.assertEqual(plan["downbeats_seconds"],db)
        self.assertEqual([(w["start"],w["end"]) for w in plan["raw_bars"]],[(.13,2.07),(2.07,4.09)])
        self.assertEqual([w["kind"] for w in plan["raw_windows"]],
                         ["prefix_residual","complete_detected_bar","complete_detected_bar","suffix_residual"])
        self.assertEqual([w["raw_window_indices"] for w in plan["windows"]],[[0],[1],[2],[3]])
        self.assertEqual(plan["windows"][0]["start"],0)
        self.assertEqual(plan["windows"][-1]["end"],5.0625)
        self.assertFalse(plan["downbeat_verified"])
        W.validate_shared_plan(plan,binding())

    def test_merge_short_edges_only_preserves_raw_bars_and_indices(self):
        strict = W.bar_windows(binding(),[.13,2.07,4.09],source=source())
        merged = W.bar_windows(binding(),[.13,2.07,4.09],source=source(),policy="merge_short_edges")
        self.assertEqual(strict["raw_bars"],merged["raw_bars"])
        self.assertEqual(strict["raw_windows"],merged["raw_windows"])
        self.assertEqual([w["raw_window_indices"] for w in merged["windows"]],[[0,1],[2,3]])
        self.assertEqual([(w["start"],w["end"]) for w in merged["windows"]],[(0,2.07),(2.07,5.0625)])
        W.validate_shared_plan(merged,binding())

    def test_single_bar_merges_both_edges_with_mapping(self):
        plan = W.bar_windows(binding(3.),[.3,2.7],source=source(),policy="merge_short_edges")
        self.assertEqual(plan["windows"],[{"start":0.,"end":3.,"raw_window_indices":[0,1,2]}])
        self.assertEqual(len(plan["raw_bars"]),1)

    def test_short_interior_bar_and_exact_edge_threshold_not_merged(self):
        plan = W.bar_windows(binding(5.5),[1.5,2.,2.3,4.],source=source(),policy="merge_short_edges")
        self.assertEqual(len(plan["windows"]),5)
        self.assertAlmostEqual(plan["raw_bars"][1]["end"]-plan["raw_bars"][1]["start"],.3)

    def test_no_synthetic_edge_when_downbeat_at_media_boundaries(self):
        plan=W.bar_windows(binding(4.),[0.,2.,4.],source=source())
        self.assertEqual(len(plan["raw_windows"]),2)
        self.assertTrue(all(w["kind"]=="complete_detected_bar" for w in plan["raw_windows"]))

    def test_fewer_than_two_downbeats_is_insufficient_not_fake_bar(self):
        for db in ([],[0.],[.3],[5.0625]):
            for policy in ("strict","merge_short_edges"):
                with self.subTest(db=db,policy=policy):
                    plan=W.bar_windows(binding(),db,source=source(),policy=policy)
                    self.assertEqual(plan["status"],"insufficient_downbeats")
                    self.assertEqual(plan["raw_bars"],[])
                    self.assertEqual(plan["windows"][0]["start"],0)
                    self.assertEqual(plan["windows"][-1]["end"],5.0625)
                    self.assertFalse(plan["params"]["insufficient_downbeats_fallback"])

    def test_invalid_downbeat_timestamps_refused(self):
        for db in ([.2,.2],[2.,.2],[-.1,2.],[.1,6.],[True,2.],[.1,float("nan")],"0,2"):
            with self.subTest(db=db),self.assertRaises(ValueError):
                W.bar_windows(binding(),db,source=source())

    def test_fake_bar_source_and_typed_claims_refused(self):
        cases = [{"method":"audio_onset_peaks"},{"method":"bpm_duration_grid"},
                 {"synthetic":True},{"synthetic":0},{"dbn":0},{"dbn":"false"},
                 {"downbeats_origin":"every_four_beats"},{"checkpoint_name":"guessed"},
                 {"audio_sha256":"9"*64},{"checkpoint_sha256":None},{"library_source_sha256":"oops"}]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                W.bar_windows(binding(),[.1,2.],source={**source(),**change})

    def test_bool_window_bounds_and_bool_mapping_are_not_numbers(self):
        plan=W.bpm_duration_windows(binding(),120.)
        for field in ("bound","mapping"):
            bad=copy.deepcopy(plan)
            if field=="bound": bad["windows"][0]["start"]=False
            else: bad["windows"][0]["raw_window_indices"]=[False]
            with self.subTest(field=field),self.assertRaises(ValueError):W.validate_shared_plan(bad,binding())

    def test_source_sha_pts_and_time_binding_mismatch_refused(self):
        plan = W.bpm_duration_windows(binding(),120.)
        for key, value in (("audio_sha256","9"*64),("actual_video_pts_sha256","9"*64),
                           ("video_sha256","9"*64),("media_end",5.1),("video_pts_offset",0.)):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"binding mismatch"):
                W.validate_shared_plan(plan,{**binding(),key:value})

    def test_altered_grid_mapping_or_promoted_accuracy_refused(self):
        plan = W.bpm_duration_windows(binding(),120.)
        for change in ("start","mapping","verified","interval"):
            bad=copy.deepcopy(plan)
            if change=="start":bad["windows"][0]["start"] = .2
            if change=="mapping":bad["windows"][-1]["raw_window_indices"] = [1]
            if change=="verified":bad["beat_phase_verified"] = True
            if change=="interval":bad["window_interval"] = "[start,end]"
            with self.subTest(change=change),self.assertRaises(ValueError):W.validate_shared_plan(bad,binding())


class ConsumerTests(unittest.TestCase):
    def arrays(self):
        return fixtures.SliceTests().arrays()

    def test_bpm_real_consumer_contract_half_open_support_and_no_reset(self):
        arrays=self.arrays()
        plan=W.bpm_duration_windows(binding(4.1),120.)
        calls=[]
        def captioner(sliced,src,shim):
            calls.append(sliced)
            return fixtures.fake_captions(sliced,src,shim)
        views=W.describe_shared_windows(arrays,plan,binding(4.1),None,None,captioner=captioner)
        self.assertEqual(views[0]["selected_target_indices"],list(range(40)))
        self.assertEqual(views[1]["selected_target_indices"],list(range(40,81)))
        self.assertEqual(calls[1]["time"][0],2.)
        self.assertEqual(calls[1]["translation"][0,0],5.)
        self.assertIs(calls[1]["source_time"],arrays["source_time"])
        self.assertEqual(views[0]["supported_descriptions"][0]["end"],1.95)
        self.assertEqual(views[0]["window_end"],2.)
        self.assertFalse(views[0]["sample_span_is_complete_window_evidence"])
        self.assertIn("A second later",views[0]["original_captions"][0]["ms"][0])

    def test_partial_support_text_retained_but_not_fake_full_caption(self):
        arrays=self.arrays(); arrays["source_joint_valid"][15,4]=False
        views=W.describe_shared_windows(arrays,W.bpm_duration_windows(binding(4.1),120.),binding(4.1),None,None,captioner=fixtures.fake_captions)
        self.assertEqual(views[0]["status"],"abstain")
        self.assertEqual(views[0]["abstain_reason"],"partial_support")
        self.assertEqual(views[0]["cleaned_motion_text"],"")
        self.assertTrue(views[0]["partial_supported_motion_text"])
        self.assertTrue(views[0]["original_captions"])

    def test_strict_short_edge_can_have_genuine_no_output_not_baseline_substitute(self):
        def minimum_samples(sliced,src,shim):
            return [] if len(sliced["time"])<4 else fixtures.fake_captions(sliced,src,shim)
        plan=W.bar_windows(binding(4.1),[.1,2.,4.],source=source())
        views=W.describe_shared_windows(self.arrays(),plan,binding(4.1),None,None,captioner=minimum_samples)
        self.assertEqual(views[0]["status"],"abstain")
        self.assertEqual(views[0]["abstain_reason"],"no_output")
        self.assertEqual(views[0]["original_captions"],[])
        self.assertEqual(views[0]["cleaned_motion_text"],"")
        self.assertEqual(views[1]["status"],"candidate")

    def test_shared_manifest_can_be_used_twice_without_window_changes(self):
        plan=W.bar_windows(binding(4.1),[.1,2.,4.],source=source(),policy="merge_short_edges")
        serialized=json.dumps(plan,sort_keys=True)
        a=W.describe_shared_windows(self.arrays(),plan,binding(4.1),None,None,captioner=fixtures.fake_captions)
        b=W.describe_shared_windows(self.arrays(),json.loads(serialized),binding(4.1),None,None,captioner=fixtures.fake_captions)
        self.assertEqual(a,b)
        self.assertEqual(json.dumps(plan,sort_keys=True),serialized)

    def test_mismatched_consumer_source_rejected_before_caption_call(self):
        captioner=mock.Mock()
        with self.assertRaises(ValueError):
            W.describe_shared_windows(self.arrays(),W.bpm_duration_windows(binding(4.1),120.),
                  {**binding(4.1),"video_sha256":"9"*64},None,None,captioner=captioner)
        captioner.assert_not_called()

    def test_altered_source_report_receipt_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"synthetic.json"
            path.write_text('{"id":"synthetic"}')
            path.with_suffix(".receipt.json").write_text(json.dumps({"report_sha256":"0"*64}))
            with self.assertRaisesRegex(ValueError,"receipt"):
                W.read_verified_report(path)


class ActualBeatThisInputTests(unittest.TestCase):
    def setUp(self):
        import ms_truebar_caption as bar
        self.bar=bar
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        self.put=lambda name,payload:self.write(name,payload)
        checkpoint=self.put("checkpoint",b"synthetic model fixture, never real inference")
        inference=self.put("inference.py",b"class File2Beats: pass")
        preprocessing=self.put("preprocessing.py",b"fixture preprocessing")
        helper=self.put("helper.py",b"fixture helper")
        license_path=self.put("LICENSE",b"fixture MIT")
        metadata=self.put("METADATA",b"fixture version")
        audio=self.put("audio.wav",b"fixture audio")
        self.manifest=self.put("input.jsonl",(json.dumps({"id":"synthetic","audio_path":str(audio)})+"\n").encode())
        src={"schema":"beatthis_checkpoint_source_v1","checkpoint_variant":"final0","official_sha1_match":True,
             "license":"MIT","checkpoint_path":str(checkpoint),"checkpoint_sha256":self.sha(checkpoint),
             "official_response_sha1":hashlib.sha1(checkpoint.read_bytes()).hexdigest(),
             "license_path":str(license_path),"license_sha256":self.sha(license_path),
             "package_metadata_path":str(metadata),"package_metadata_sha256":self.sha(metadata)}
        source_path=self.put("source.json",json.dumps(src).encode())
        params={"device":"cpu","dbn":False,"float16":False,"postprocessor":"minimal","postprocessor_fps":50,"audio_cropping":"none; full audio_path"}
        runtime={"beatthis_inference_path":str(inference),"beatthis_inference_sha256":self.sha(inference),
                 "beatthis_preprocessing_sha256":self.sha(preprocessing),"helper_path":str(helper),
                 "helper_sha256":self.sha(helper),"packages":{"beat-this":"synthetic-not-evidence"}}
        self.row={"schema":"beatthis_audio_detections_v1","id":"synthetic","status":"ok",
             "inference_called":True,"model_prediction_returned":True,
             "source_row":json.loads(self.manifest.read_bytes()),"input_manifest_line_number":1,
             "input_manifest_line_sha256":hashlib.sha256(self.manifest.read_bytes().rstrip(b"\n")).hexdigest(),
             "model_sha256":self.sha(checkpoint),"model_source_manifest_sha256":self.sha(source_path),
             "helper_sha256":runtime["helper_sha256"],"library_versions":runtime["packages"],"params":params,
             "input_manifest_sha256":self.sha(self.manifest),"audio_path":str(audio),"audio_sha256":self.sha(audio),
             "duration_seconds":4.,"sample_rate":10,"audio_frames":40,
             "beats":[.1,1.,2.1,3.],"downbeats":[.1,2.1],"complete_downbeat_interval_count":1}
        self.detections=self.root/"detections.jsonl"
        self.receipt={"schema":"beatthis_audio_detections_v1","status":"completed","exit_code":0,"failed_count":0,
             "output_path":str(self.detections),"input_manifest_path":str(self.manifest),"input_manifest_sha256":self.sha(self.manifest),
             "expected_count":1,"attempted_count":1,"successful_count":1,"params":params,"runtime":runtime,
             "model_runtime":{"parameter_devices":["cpu"],"parameter_dtypes":["torch.float32"]},
             "model":{"checkpoint_path":str(checkpoint),"checkpoint_sha256":self.sha(checkpoint),
                      "source_manifest_path":str(source_path),"source_manifest_sha256":self.sha(source_path),"source":src}}
        self.finalsha=self.sha(checkpoint);self.finalsha1=hashlib.sha1(checkpoint.read_bytes()).hexdigest()
        self.sync()

    def write(self,name,payload):
        path=self.root/name;path.write_bytes(payload);return path

    def sync(self):
        self.detections.write_text(json.dumps(self.row)+"\n")
        self.receipt["output_sha256"]=self.sha(self.detections)
        (self.root/"run_receipt.json").write_text(json.dumps(self.receipt))

    def normalize(self):
        with mock.patch.object(self.bar,"FINAL0_SHA256",self.finalsha),mock.patch.object(self.bar,"FINAL0_SHA1",self.finalsha1):
            return self.bar.normalize_detections(self.detections,self.manifest)

    def test_fixture_preserves_raw_downbeats_and_source_refs_not_real_accuracy(self):
        normalized=self.normalize()[0]
        self.assertEqual(normalized["downbeats_seconds"],self.row["downbeats"])
        self.assertEqual(normalized["source"]["raw_downbeats_seconds"],self.row["downbeats"])
        self.assertEqual(normalized["source"]["raw_detection_line_sha256"],hashlib.sha256(self.detections.read_bytes().rstrip(b"\n")).hexdigest())
        self.assertFalse(normalized["source"]["bar_phase_accuracy_verified"])

    def test_altered_detection_bytes_refused_before_normalization(self):
        self.detections.write_text(self.detections.read_text()+" " )
        with self.assertRaisesRegex(ValueError,"SHA mismatch"):self.normalize()

    def test_faked_actual_inference_flag_refused(self):
        self.row["inference_called"]=False;self.sync()
        with self.assertRaisesRegex(ValueError,"inference success"):self.normalize()

    def test_typed_dbn_and_actual_runtime_refused(self):
        self.receipt["params"]["dbn"]=0;self.sync()
        with self.assertRaisesRegex(ValueError,"typed CPU"):self.normalize()

    def test_altered_audio_sha_refused(self):
        Path(self.row["audio_path"]).write_bytes(b"altered audio")
        with self.assertRaisesRegex(ValueError,"SHA mismatch"):self.normalize()

    def test_fake_bar_timestamp_not_a_model_beat_refused(self):
        self.row["downbeats"]=[.2,2.1];self.sync()
        with self.assertRaisesRegex(ValueError,"beat-snapped"):self.normalize()

    def test_original_out_of_media_downbeat_rejected_without_clipping(self):
        self.row["downbeats"]=[.1,4.01];self.sync()
        with self.assertRaisesRegex(ValueError,"no implicit clipping"):self.normalize()

    def test_source_row_line_identity_mismatch_refused(self):
        self.row["input_manifest_line_sha256"]="0"*64;self.sync()
        with self.assertRaisesRegex(ValueError,"source row/line"):self.normalize()

    def test_nonofficial_checkpoint_not_admitted(self):
        with self.assertRaisesRegex(ValueError,"not verified official"):
            self.bar.normalize_detections(self.detections,self.manifest)

    def test_new_helper_code_mismatch_refused(self):
        Path(self.receipt["runtime"]["helper_path"]).write_bytes(b"altered helper")
        with self.assertRaisesRegex(ValueError,"SHA mismatch"):self.normalize()


if __name__=="__main__":unittest.main()
