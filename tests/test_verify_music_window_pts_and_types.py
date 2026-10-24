import copy
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms"
sys.path.insert(0, str(MS))
import ms_music_windows as music
import verify_music_window_pts_and_types as audit
from ms_optimize import json_bytes


class ExactTypedWindowTests(unittest.TestCase):
    def binding(self):
        return dict(id="test", input_sha256="0" * 64, video_sha256="1" * 64, audio_sha256="2" * 64,
            actual_video_pts_sha256="3" * 64, media_start=0., media_end=5.0625,
            video_pts_offset=0., video_fps=16., decoded_video_frames=81)

    def test_valid_bpm_preserves_raw_tail_mapping(self):
        b = self.binding()
        p = music.bpm_duration_windows(b, 120.)
        self.assertIs(audit.validate_plan(p, b), p)
        self.assertEqual(p["windows"][-1]["raw_window_indices"], [1, 2])

    def test_bool_resealed_window_and_raw_grid_index_rejected(self):
        b = self.binding()
        for field in ("start", "grid_index"):
            p = music.bpm_duration_windows(b, 120.)
            if field == "start":
                p["windows"][0]["start"] = False
            else:
                p["raw_windows"][1]["grid_index"] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.validate_plan(p, b)

    def test_bool_binding_or_phase_rejected(self):
        b = self.binding()
        for field in ("media_start", "decoded_video_frames", "beat_phase_verified"):
            p = music.bpm_duration_windows(b, 120.)
            if field == "beat_phase_verified":
                p[field] = True
            else:
                p["binding"][field] = False if field == "media_start" else True
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.validate_plan(p, b)

    def test_actual_video_pts_not_declared_nlf_digest(self):
        b = self.binding()
        time = np.arange(81) / 16.
        b["actual_video_pts_sha256"] = hashlib.sha256(json_bytes(time.tolist())).hexdigest()
        p = music.bpm_duration_windows(b, 120.)
        info = dict(time=time, media_end=5.0625, pts_offset=0., fps=16.)
        self.assertEqual(audit.verify_actual_pts(p, info), b)
        bad = copy.deepcopy(p)
        bad["binding"]["actual_video_pts_sha256"] = "4" * 64
        with self.assertRaises(ValueError):
            audit.verify_actual_pts(bad, info)

    def test_actual_end_or_frame_count_mismatch_rejected(self):
        b = self.binding()
        time = np.arange(81) / 16.
        b["actual_video_pts_sha256"] = hashlib.sha256(json_bytes(time.tolist())).hexdigest()
        p = music.bpm_duration_windows(b, 120.)
        for altered in (dict(time=time, media_end=6., pts_offset=0., fps=16.),
                        dict(time=time[:-1], media_end=5.0625, pts_offset=0., fps=16.)):
            with self.subTest(end=altered["media_end"]), self.assertRaises(ValueError):
                audit.verify_actual_pts(p, altered)

    def test_strict_and_merged_bar_positive_and_typed_negative(self):
        b = self.binding()
        source = dict(method="beat_this.inference.File2Beats", checkpoint_name="final0", dbn=False,
            device="cpu", synthetic=False, downbeats_origin="model_downbeat_head_postprocessed",
            checkpoint_sha256="5" * 64, audio_sha256=b["audio_sha256"],
            library_source_sha256="6" * 64, library_version="1.1.0")
        for policy in ("strict", "merge_short_edges"):
            p = music.bar_windows(b, [.7, 2.7, 4.7], source=source, policy=policy)
            self.assertIs(audit.validate_plan(p, b), p)
            bad = copy.deepcopy(p)
            bad["raw_bars"][0]["bar_index"] = False
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                audit.validate_plan(bad, b)


if __name__ == "__main__":
    unittest.main()
