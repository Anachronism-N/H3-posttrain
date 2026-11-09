"""CPU preparation probes; no audio fixtures, model loading or semantic claims."""
from copy import deepcopy
import hashlib
import json
import unittest

from captionlab.targets import (build_crop_manifest, convert_record, restore_raw,
                               validate_source_records)


CLASSES = ["ballet", "contemporary", "unknown"]


def source(identifier="s1", split="train", *, body=None, times=None, genre="contemporary"):
    body = body if body is not None else (
        "[0.000-2.000s] One side wrist moves left. One side wrist moves left.\n"
        "[2.000-5.000s] The other side elbow angle increases.\n")
    times = times if times is not None else [(0., 2.), (2., 5.)]
    token = hashlib.sha256(identifier.encode()).hexdigest()
    return {"schema": "captionlab.source.v1", "id": identifier, "split": split,
            "source_group_id": "video-" + identifier, "song_id": "song-" + identifier,
            "near_duplicate_group_id": "near-" + identifier,
            "raw_caption": ("Dance style: " + genre + ".\n") + body,
            "genre_header_template": "Dance style: {genre}.\n",
            "audio": {"path": "/audio/" + identifier + ".wav", "sha256": token,
                      "pcm_sha256": hashlib.sha256((identifier + "-pcm").encode()).hexdigest(),
                      "sample_rate": 100, "num_samples": 500,
                      "source_start_seconds": 12.0, "pad_intervals": []},
            "windows": [{"start": start, "end": end, "context_dependency": "unknown"}
                        for start, end in times]}


def convert(row, **kwargs):
    return convert_record(row, genre_classes=CLASSES, **kwargs)


class ConversionTests(unittest.TestCase):
    def test_only_declared_header_and_markers_removed_not_repetition(self):
        row = source()
        before = deepcopy(row)
        result = convert(row)
        self.assertEqual(result["target_text"],
                         " One side wrist moves left. One side wrist moves left.\n"
                         " The other side elbow angle increases.\n")
        self.assertEqual(result["target_text"].count("One side wrist moves left."), 2)
        self.assertEqual(row, before)
        self.assertEqual(restore_raw(result).encode(), row["raw_caption"].encode())
        self.assertEqual(result["jev"]["class_index"], 1)
        self.assertTrue(result["jev"]["known_mask"])
        self.assertEqual(result["genre_classes"], ["ballet", "contemporary"])
        self.assertFalse(result["jev"]["independently_verified"])

    def test_keep_timestamps_is_separate_factor(self):
        row = source()
        result = convert(row, keep_timestamps=True)
        self.assertEqual(result["target_text"], row["raw_caption"].split("\n", 1)[1])
        self.assertEqual([part["kind"] for part in result["source_spans"]],
                         ["removed_genre_header", "retained"])
        self.assertEqual(restore_raw(result), row["raw_caption"])

    def test_unicode_crlf_action_whitespace_byte_mapping_and_wrapping(self):
        row = source(body="[0-2s]  左手移向左侧。\r\n动作继续。\r\n[2-5s]\t右膝弯曲。\r\n")
        row["raw_caption"] = row["raw_caption"].replace("contemporary.\n", "contemporary.\r\n", 1)
        row["genre_header_template"] = "Dance style: {genre}.\r\n"
        result = convert(row)
        self.assertEqual(result["target_text"], "  左手移向左侧。\r\n动作继续。\r\n\t右膝弯曲。\r\n")
        raw_bytes = row["raw_caption"].encode("utf-8")
        target_bytes = result["target_text"].encode("utf-8")
        for part in result["source_spans"]:
            a, b = part["source_byte_span"]
            self.assertEqual(raw_bytes[a:b], part["text"].encode("utf-8"))
            a, b = part["target_byte_span"]
            self.assertEqual(target_bytes[a:b], part["text"].encode("utf-8") if part["kind"] == "retained" else b"")
        self.assertEqual(restore_raw(result), row["raw_caption"])

    def test_unknown_and_absent_genre_are_masked_not_real_class(self):
        result = convert(source(genre="unknown"))
        self.assertFalse(result["jev"]["known_mask"])
        self.assertIsNone(result["jev"]["class_index"])
        self.assertEqual(result["jev"]["label"], "unknown")
        self.assertIn("The other side", result["target_text"])
        self.assertEqual(restore_raw(result), result["source"]["raw_caption"])
        row = source()
        row["raw_caption"] = row["raw_caption"].split("\n", 1)[1]
        row["genre_header_template"] = None
        result = convert(row)
        self.assertFalse(result["jev"]["known_mask"])
        self.assertEqual(result["jev"]["label_source"], "missing")

    def test_no_side_or_context_inference_from_other_then(self):
        row = source(body="[0-2s] The other side moves.\n[2-5s] Then repeat.\n")
        result = convert(row)
        self.assertEqual([w["context_dependency"] for w in result["windows"]], ["unknown", "unknown"])
        self.assertFalse(result["context_dependency_inferred"])
        self.assertNotIn("left", result["target_text"])

    def test_explicit_display_rounding_uses_unmodified_authoritative_window(self):
        row = source(body="[0.000-2.000s] Move.\n[2.000-5.063s] Repeat.\n", times=[(0., 2.), (2., 5.0625)])
        row["audio"].update(sample_rate=16000, num_samples=81000)
        result = convert(row)
        self.assertEqual(result["windows"][1]["end"], 5.0625)
        self.assertEqual(result["windows"][1]["displayed_timestamp"], "[2.000-5.063s]")
        with self.assertRaises(ValueError):
            convert(row, timestamp_tolerance_seconds=0)

    def test_malformed_or_inline_markers_and_metadata_disagreement_rejected(self):
        for body in ("[0-2s] Move. [2-5s] Repeat.",
                     "[-1-2s] Move.\n[2-5s] Repeat.",
                     "[nan-2s] Move.\n[2-5s] Repeat.",
                     "[0-2e0s] Move.\n[2-5s] Repeat.",
                     "[2-0s] Move.\n[2-5s] Repeat.",
                     "[0-2s] Move [aside].\n[2-5s] Repeat.",
                     "[0-2s] \n[2-5s] Repeat.",
                     "[0-1s] Move.\n[2-5s] Repeat.",
                     "[0-2.0009s] Move.\n[1.9992-5s] Repeat."):
            with self.subTest(body=body), self.assertRaises(ValueError):
                convert(source(body=body))

    def test_unknown_style_is_not_silently_mapped_or_inferred(self):
        with self.assertRaises(ValueError):
            convert(source(genre="jazz"))
        row = source()
        row["genre_header_template"] = "Style: {genre}\n"
        with self.assertRaises(ValueError):
            convert(row)

    def test_second_header_is_not_reclassified_as_an_action(self):
        row = source(body="[0-2s] Move.\nDance style: ballet.\n[2-5s] Repeat.\n")
        with self.assertRaisesRegex(ValueError, "Only one"):
            convert(row)

    def test_strict_finite_types_shape_and_temporal_metadata(self):
        changes = [("sample_rate", True), ("num_samples", 0), ("source_start_seconds", float("nan")),
                   ("sha256", "not-a-hash"), ("pad_intervals", [[0, 501]])]
        for name, value in changes:
            row = source()
            row["audio"][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                convert(row)
        for times in ([(0, 2), (1.9, 5)], [(0, 2), (2, 5.001)], [(0, 0), (2, 5)]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                convert(source(times=times))
        row = source()
        row["windows"][0]["context_dependency"] = True
        with self.assertRaises(ValueError):
            convert(row)
        row = source()
        row["extra"] = "not silently retained or discarded"
        with self.assertRaises(ValueError):
            convert(row)
        with self.assertRaises(ValueError):
            convert(source(), keep_timestamps=1)
        with self.assertRaises(ValueError):
            convert(source(), timestamp_tolerance_seconds=10 ** 400)

    def test_modified_target_or_evidence_cannot_restore_or_crop(self):
        original = convert(source())
        for field in ("target_text", "source_sha256"):
            broken = deepcopy(original)
            broken[field] += "tampered"
            with self.subTest(field=field), self.assertRaises(ValueError):
                restore_raw(broken)
        broken = deepcopy(original)
        broken["source_spans"][1]["source_byte_span"][0] += 1
        with self.assertRaises(ValueError):
            build_crop_manifest([broken])


class IsolationTests(unittest.TestCase):
    def test_all_explicit_association_keys_block_cross_split(self):
        for field in ("source_group_id", "song_id", "near_duplicate_group_id",
                      "sha256", "pcm_sha256", "path"):
            a, b = source("one"), source("two", "val")
            if field in a:
                b[field] = a[field]
            else:
                b["audio"][field] = a["audio"][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Cross-split risk"):
                validate_source_records([a, b])
            with self.subTest(crop=field), self.assertRaisesRegex(ValueError, "Cross-split risk"):
                build_crop_manifest([convert(a), convert(b)])

    def test_windows_path_case_and_separator_aliases_rejected_cross_platform(self):
        a, b = source("one"), source("two", "test")
        a["audio"]["path"] = "C:\\Audio\\Music.wav"
        b["audio"]["path"] = "c:/audio/MUSIC.wav"
        with self.assertRaisesRegex(ValueError, "audio_path"):
            validate_source_records([a, b])

    def test_unknown_song_groups_never_become_audited_and_same_split_positive_retained(self):
        a, b = source("one"), source("two")
        a["song_id"] = b["song_id"] = None
        a["near_duplicate_group_id"] = b["near_duplicate_group_id"] = None
        b["audio"]["pcm_sha256"] = a["audio"]["pcm_sha256"]
        report = validate_source_records([a, b])
        self.assertEqual(report["source_records"], 2)
        self.assertFalse(report["song_groups_complete"])
        self.assertFalse(report["content_similarity_audited"])
        self.assertEqual(build_crop_manifest([convert(a), convert(b)])["source_record_count"], 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_source_records([a, a])


class CropManifestTests(unittest.TestCase):
    def test_mixed_ordered_class_vocabularies_cannot_corrupt_jev_indices(self):
        a = convert(source("one"))
        b = convert_record(source("two"), genre_classes=["contemporary", "ballet"])
        with self.assertRaisesRegex(ValueError, "same ordered"):
            build_crop_manifest([a, b])

    def test_existing_complete_windows_only_and_no_audio_file_created(self):
        result = build_crop_manifest([convert(source())])
        self.assertEqual([row["num_samples"] for row in result["windows"]], [200, 300])
        self.assertEqual([row["declared_window_seconds"] for row in result["windows"]], [[0., 2.], [2., 5.]])
        self.assertEqual(result["windows"][0]["target_text"].count("One side wrist moves left."), 2)
        for row in result["windows"]:
            self.assertIsNone(row["output_audio_path"])
            self.assertIsNone(row["output_pcm_sha256"])
            self.assertFalse(row["audio_cropped"])
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["jev_observation_samples"], [0, 500])
        self.assertFalse(result["ready_for_training"])
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_duration_selection_does_not_split_merge_or_hide_exclusions(self):
        row = source(body="[0-0.5s] Short.\n[0.5-4s] Long.\n[4-5s] Valid.\n",
                     times=[(0, .5), (.5, 4), (4, 5)])
        result = build_crop_manifest([convert(row)])
        self.assertEqual(len(result["windows"]), 1)
        self.assertEqual(result["windows"][0]["window_index"], 2)
        self.assertEqual([r["window_index"] for r in result["excluded_windows"]], [0, 1])
        self.assertEqual(result["windows"][0]["target_text"], " Valid.\n")

    def test_half_up_indices_true_clock_padding_and_absolute_time(self):
        row = source(body="[0.05-1.05s] Move.\n", times=[(.05, 1.05)])
        row["audio"].update(sample_rate=10, num_samples=50, source_start_seconds=10.5,
                             pad_intervals=[[0, 4], [10, 15]])
        row["windows"][0]["context_dependency"] = "dependent"
        crop = build_crop_manifest([convert(row)])["windows"][0]
        self.assertEqual((crop["start_sample"], crop["end_sample"], crop["num_samples"]), (1, 11, 10))
        self.assertEqual(crop["absolute_source_seconds"], [10.6, 11.6])
        self.assertEqual(crop["parent_pad_intersections"], [[1, 4], [10, 11]])
        self.assertEqual(crop["crop_local_pad_intersections"], [[0, 3], [9, 10]])
        self.assertEqual(crop["context_dependency"], "dependent")
        self.assertEqual(crop["audio"], row["audio"])

    def test_unknown_padding_origin_and_context_remain_unknown(self):
        row = source()
        row["audio"].update(pad_intervals=None, source_start_seconds=None)
        crop = build_crop_manifest([convert(row)])["windows"][0]
        self.assertIsNone(crop["absolute_source_seconds"])
        self.assertIsNone(crop["parent_pad_intersections"])
        self.assertEqual(crop["context_dependency"], "unknown")

    def test_crop_action_bytes_trace_back_to_unicode_source(self):
        row = source(body="[0-2s] 左侧。\n[2-5s] 重复。\n")
        result = build_crop_manifest([convert(row, keep_timestamps=True)])
        raw = row["raw_caption"].encode()
        for crop in result["windows"]:
            a, b = crop["action_byte_span"]
            self.assertEqual(raw[a:b], crop["target_text"].encode())
            self.assertNotIn("[", crop["target_text"])

    def test_sample_collapsed_window_and_invalid_duration_range_fail(self):
        row = source(body="[0.01-0.02s] Move.\n", times=[(.01, .02)])
        row["audio"].update(sample_rate=10, num_samples=50)
        with self.assertRaisesRegex(ValueError, "collapses"):
            convert(row)
        for kwargs in ({"min_seconds": 0}, {"min_seconds": 3, "max_seconds": 2},
                       {"max_seconds": float("inf")}, {"min_seconds": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_crop_manifest([convert(source())], **kwargs)


if __name__ == "__main__":
    unittest.main()
