"""Detected-beat windows are estimates, not action boundaries or tempo grids."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
MS=Path(__file__).resolve().parents[1]/"scripts_omnidance"/"exp"/"ms"
sys.path.insert(0,str(MS))
import ms_beat_chunk_compare as C
import ms_chunk_caption as original
import test_ms_chunk_caption as fixtures


def estimate(beats=None):
    return {"detected_beats_seconds":list(np.arange(.1,5,.5) if beats is None else beats),
            "rms":.1,"onset_std":1.,"onset_max":5.,"beat_peak_alignment_fraction":.8}


class BeatWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ManifestAndCLITests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.run_cli()
        self.output_temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.output_temp.cleanup)
        self.args=C.parser().parse_args(["--manifest",str(self.fixture.manifest),"--baseline",self.fixture.args.out,
                    "--out",str(Path(self.output_temp.name)/"independent_C"),"--template",str(self.fixture.template),
                    "--upstream-src",str(self.fixture.src)])
        import hashlib
        self.audio={**estimate(),"audio_sha256":hashlib.sha256(self.fixture.audio.read_bytes()).hexdigest(),
                    "duration_seconds":5.0625,"native_sample_rate":16000,"decoded_sample_count":81000,
                    "bpm_estimate":120.,"estimator":C.ESTIMATOR}

    def run_C(self):
        import types,contextlib
        describe=original.describe_windows
        def init(context):C.CONTEXT=context
        with mock.patch.object(C,"initialize",side_effect=init), mock.patch.object(C.subprocess,"run",return_value=types.SimpleNamespace(stdout=json.dumps(self.audio))), mock.patch.object(original,"probe_video",return_value=fixtures.video_info()), mock.patch.object(original,"describe_windows",side_effect=lambda a,w,s,h:describe(a,w,s,h,captioner=fixtures.fake_captions)), contextlib.redirect_stdout(io.StringIO()):
            return C.run(self.args)

    def test_matched_workflow_hash_raw_stats_and_tempo_grid_distinct(self):
        result=self.run_C()
        self.assertTrue(result["complete"])
        self.assertEqual(result["processed_count"],1)
        report=json.loads((Path(self.args.out)/"synthetic.json").read_bytes())
        self.assertEqual(report["audio_sha256"],self.audio["audio_sha256"])
        self.assertEqual(report["A_whole_clip"]["original_captions"][0]["ms"],["She raises her left arm twice. A second later she raises her left arm twice."])
        self.assertFalse(report["tempo_grid_comparison_only"]["used_as_C"])
        self.assertFalse(report["audio_video_alignment_verified"])
        self.assertFalse(report["motions_on_beats_verified"])
        self.assertEqual(result["A_B_C_same_processed_source_stats"]["A"]["source_clip_denominator"],1)
        self.assertEqual(result["A_B_C_same_processed_source_stats"]["C"]["source_clip_denominator"],1)

    def test_wav_duration_mismatch_is_failed_not_fixed_fallback(self):
        self.audio["duration_seconds"]=7.
        result=self.run_C()
        self.assertFalse(result["complete"])
        self.assertEqual(result["failed_count"],1)
        self.assertIn("WAV length",result["errors"][0]["error"])

    def test_source_audio_sha_mismatch_refuses_same_sample_comparison(self):
        self.audio["audio_sha256"]="0"*64
        result=self.run_C()
        self.assertFalse(result["complete"])
        self.assertIn("C source bytes differ",result["errors"][0]["error"])

    def test_existing_out_and_id_order_not_overwritten(self):
        self.run_C()
        before=(Path(self.args.out)/"summary.json").read_bytes()
        with self.assertRaisesRegex(ValueError,"new independent"):
            self.run_C()
        self.assertEqual(before,(Path(self.args.out)/"summary.json").read_bytes())
        self.args.out=str(Path(self.output_temp.name)/"wrong_id_out")
        row={**self.fixture.row,"id":"different"}
        self.fixture.write_manifest([row])
        with self.assertRaisesRegex(ValueError,"same ordered"):
            self.run_C()


class BeatWindowsTests(unittest.TestCase):
    def test_actual_detected_boundaries_not_global_bpm_grid(self):
        detected=[.13,.65,1.11,1.64,2.08,2.62,3.10,3.65,4.13,4.60,5.08]
        plan=C.choose_windows(estimate(detected),5.5)
        self.assertEqual(plan["strategy"],"detected_beats")
        self.assertEqual(plan["windows"][0]["start"],0)
        self.assertEqual(plan["windows"][-1]["end"],5.5)
        for left,right in zip(plan["windows"],plan["windows"][1:]):
            self.assertEqual(left["end"],right["start"])
            self.assertIn(left["end"],detected)
            self.assertEqual(detected[left["detected_boundary_beat_index"]],left["end"])
        self.assertNotEqual(plan["windows"][0]["end"],2.)
        self.assertFalse(plan["window_boundaries_are_action_boundaries"])
        self.assertFalse(plan["bars_or_downbeats_verified"])

    def test_short_tail_never_invents_boundary(self):
        plan=C.choose_windows(estimate(),4.01)
        self.assertEqual(plan["strategy"],"detected_beats")
        self.assertEqual(plan["windows"][-1]["end"],4.01)
        self.assertGreaterEqual(plan["windows"][-1]["end"]-plan["windows"][-1]["start"],1.5)
        for window in plan["windows"][:-1]:self.assertIn(window["end"],estimate()["detected_beats_seconds"])

    def test_optional2_or8_with_1_5_to4_seconds_boundaries(self):
        for count in (2,4,8):
            with self.subTest(count=count):
                plan=C.choose_windows(estimate(),5.0625,beats_per_chunk=count)
                self.assertEqual(plan["strategy"],"detected_beats")
                self.assertEqual(plan["beats_per_chunk_requested"],count)
                for window in plan["windows"]:
                    self.assertGreaterEqual(window["end"]-window["start"],1.5)
                    self.assertLessEqual(window["end"]-window["start"],4.)

    def test_weak_silent_short_coverage_or_irregular_beats_fixed_fallback_explicit(self):
        cases=[{**estimate(),"rms":0.},{**estimate(),"onset_std":0.},estimate([.2,.7]),
               estimate([2,2.5,3,3.5]),estimate([.1,.3,1.3,1.5,2.5,2.7,3.7,3.9]),
               {**estimate(),"beat_peak_alignment_fraction":0.}]
        for value in cases:
            with self.subTest(value=value):
                plan=C.choose_windows(value,5.0625)
                self.assertEqual(plan["strategy"],"fixed_fallback")
                self.assertTrue(plan["fallback_reason"])
                self.assertEqual([(x["start"],x["end"]) for x in plan["windows"]],[(0,2),(2,5.0625)])
                self.assertEqual(plan["detected_internal_boundaries_seconds"],[])

    def test_invalid_beats_and_parameters_fail_not_silently_repaired(self):
        for value in ([.1,.5,float('nan')],[.1,.5,.5],[.5,.1]):
            with self.subTest(value=value),self.assertRaises(ValueError):C.choose_windows(estimate(value),5)
        with self.assertRaises(ValueError):C.choose_windows(estimate(),5,beats_per_chunk=3)
        with self.assertRaises(ValueError):C.choose_windows(estimate(),5,min_seconds=5,max_seconds=4)

    def test_short_clip_complete_media_window_stays_honest(self):
        plan=C.choose_windows(estimate([.1,.3,.5,.7]),.8)
        self.assertEqual([(x["start"],x["end"]) for x in plan["windows"]],[(0,.8)])
        self.assertFalse(plan["bars_or_downbeats_verified"])

    def test_half_open_global_coordinates_not_reset_for_explicit_C(self):
        arrays=fixtures.SliceTests().arrays()
        windows=C.choose_windows(estimate(),4.1)["windows"]
        views=original.describe_windows(arrays,windows,None,None,captioner=fixtures.fake_captions)
        for window,view in zip(windows,views):
            selected=np.flatnonzero((arrays["time"]>=window["start"])&(arrays["time"]<window["end"]))
            self.assertEqual(view["selected_target_indices"],selected.tolist())
            self.assertFalse(view["coordinate_reset_applied"])
            self.assertFalse(view["semantic_approval"])
            sliced,_=original.slice_arrays(arrays,window["start"],window["end"])
            if len(selected):np.testing.assert_array_equal(sliced["translation"][0],arrays["translation"][selected[0]])

    def test_gap_partial_preserved_in_C_not_filled_with_dance_actions(self):
        arrays=fixtures.SliceTests().arrays();arrays["joint_valid"][15,3]=False
        windows=C.choose_windows(estimate(),4.1)["windows"]
        views=original.describe_windows(arrays,windows,None,None,captioner=fixtures.fake_captions)
        self.assertEqual(views[0]["abstain_reason"],"partial_support")
        self.assertEqual(views[0]["cleaned_motion_text"],"")
        self.assertTrue(views[0]["partial_supported_motion_text"])

    def test_word_repeat_counts_do_not_join_windows_or_remove_numbers(self):
        stats=C.word_stats(["left arm moves twice left arm moves twice","left arm moves twice"])
        self.assertEqual(stats["total_words"],12)
        self.assertEqual(stats["extra_repeated_fourgrams_within_caption"],1)
        self.assertEqual(stats["nonempty_count"],2)

    def test_estimator_limits_prior_and_not_calibrated(self):
        self.assertEqual(C.ESTIMATOR["tempo_range_bpm"],[60,180])
        self.assertEqual(C.ESTIMATOR["tempo_prior_bpm"],120)
        self.assertFalse(C.ESTIMATOR["calibrated"])
        self.assertFalse(C.ESTIMATOR["bar_or_downbeat_detector"])

    def test_audio_estimator_finite_native_samples_and_byte_sha(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"synthetic.wav"
            audio=np.sin(np.arange(22050)*2*np.pi*440/22050).astype(np.float32)*.01
            sf.write(path,audio,22050)
            with mock.patch.dict(sys.modules,{"librosa":mock.Mock(__version__="synthetic",onset=mock.Mock(onset_strength=mock.Mock(return_value=np.tile([0.,1.,0.,0.],12))))}):
                result=C.audio_estimate(path)
            self.assertEqual(result["decoded_sample_count"],22050)
            self.assertEqual(result["duration_seconds"],1.)
            self.assertEqual(len(result["audio_sha256"]),64)
            self.assertFalse(result["reliability_is_calibrated_confidence"])
