"""Independent contract probes; every object here is a test fixture, not data.

Uses only the standard library plus the package under review. The regression
cases intentionally fail until the corresponding boundary is repaired.
"""
import os
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from choreobench import dataset as dataset_module
from choreobench.audio import inspect_audio, make_windows, write_window
from choreobench.dataset import import_structure, load_dataset, release_report, resolve_audio
from choreobench.io import file_sha, read_json, write_json
from choreobench.schema import ValidationError, validate_dataset, validate_plan, validate_song
from choreobench.providers import ChatCompletionsProvider, ReplayProvider


def song(song_id="review_song", *, split="dev", frames=81000, rate=16000):
    return {
        "song_id": song_id,
        "split": split,
        "song_group_id": "sg_" + song_id,
        "artist_group_id": "ag_" + song_id,
        "near_duplicate_group_id": "ng_" + song_id,
        "brief": "",
        "audio": {"path": "audio.wav", "sha256": "0" * 64,
                  "sample_rate": rate, "num_samples": frames},
        "music": {"source": "unavailable", "beats": [], "downbeats": [],
                  "sections": [{"section_id": "s000", "start": 0, "end": frames,
                                "label": "unknown", "repeat_group": None}]},
    }


def manifest(songs, *, synthetic=True):
    return {"schema": "choreobench.dataset.v1", "dataset_id": "review_fixture",
            "synthetic": synthetic, "grouping": {"status": "reviewed", "evidence": "unit-test declaration only"},
            "songs": songs}


def event(event_id="e0", *, start=0, end=81000, track="body"):
    return {"event_id": event_id, "start": start, "end": end, "track": track,
            "action": "Hold still.", "attributes": {"rest": True}, "entry": {}, "exit": {},
            "motif_id": None, "evidence": []}


def plan(events=None):
    return {"schema": "choreobench.plan.v1", "song_id": "review_song", "sample_rate": 16000,
            "num_samples": 81000, "motifs": [], "events": events or [event()], "provenance": {}}


def write_wav(path, *, rate=16000, frames=16):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\0\0" * frames)


class StrictJSONReviewTests(unittest.TestCase):
    def test_numeric_overflow_is_rejected_at_any_nesting_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overflow.json"
            for raw in ('{"value":1e999}', '{"nested":[-1e999]}'):
                with self.subTest(raw=raw):
                    path.write_text(raw, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_json(path)

    def test_duplicate_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"nested":{"value":1,"value":2}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_json(path)


class PortableSchemaReviewTests(unittest.TestCase):
    def test_song_ids_must_not_name_windows_devices_or_alternate_streams(self):
        # These IDs feed plans/<song_id>.json; a portable manifest must not
        # acquire a different filesystem interpretation on Windows.
        for value in ("CON", "nul", "COM1", "Lpt9", "clip:alternate"):
            with self.subTest(song_id=value):
                value_song = song()
                value_song["song_id"] = value
                with self.assertRaises(ValidationError):
                    validate_song(value_song)

    def test_casefold_song_ids_are_rejected_before_file_publication(self):
        first, second = song("SongA"), song("songa")
        with self.assertRaises(ValidationError):
            validate_dataset(manifest([first, second]))

    def test_non_filename_event_ids_keep_the_existing_colon_contract(self):
        value = plan([event("event:0")])
        self.assertIs(validate_plan(value), value)

    def test_unhashable_motif_reference_is_a_validation_error(self):
        value = plan()
        value["events"][0]["motif_id"] = []
        with self.assertRaises(ValidationError):
            validate_plan(value)

    def test_boolean_time_is_not_an_integer_sample(self):
        value = plan()
        value["events"][0]["start"] = False
        with self.assertRaises(ValidationError):
            validate_plan(value)

    def test_half_open_tail_beat_is_outside_audio(self):
        value = song()
        value["music"]["beats"] = [81000]
        with self.assertRaises(ValidationError):
            validate_song(value)

    def test_distinct_limbs_can_be_simultaneous_but_parent_cannot(self):
        value = plan([event("left", track="left_arm"), event("right", track="right_leg")])
        self.assertIs(validate_plan(value), value)
        conflict = plan([event("all", track="body"), event("left", track="left_arm")])
        with self.assertRaises(ValidationError):
            validate_plan(conflict)

    def test_exact_audio_cross_split_leakage_is_rejected(self):
        first, second = song("first", split="train"), song("second", split="test")
        with self.assertRaises(ValidationError):
            validate_dataset(manifest([first, second]))

    def test_shared_group_cross_split_leakage_is_rejected(self):
        for group_field in ("song_group_id", "artist_group_id", "near_duplicate_group_id"):
            with self.subTest(field=group_field):
                first, second = song("first", split="train"), song("second", split="test")
                second["audio"]["sha256"] = "1" * 64
                second[group_field] = first[group_field]
                with self.assertRaises(ValidationError):
                    validate_dataset(manifest([first, second]))

    def test_native_local_release_requires_exact_81000_at_16khz(self):
        exact = release_report(manifest([song()], synthetic=False), track="local")
        self.assertTrue(exact["eligible_by_declared_contract"])
        for frames in (80999, 81001):
            with self.subTest(frames=frames):
                value = manifest([song(frames=frames)], synthetic=False)
                self.assertFalse(release_report(value, track="local")["eligible_by_declared_contract"])


class WaveformIdentityReviewTests(unittest.TestCase):
    def test_relative_audio_is_resolved_against_manifest_not_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = song()
            value["audio"]["path"] = "audio/track.wav"
            expected = (root / "audio" / "track.wav").resolve()
            self.assertEqual(resolve_audio(value, root / "manifest.json"), expected)

    def test_wav_clock_and_hash_cannot_bind_different_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, replacement = root / "audio.wav", root / "replacement.wav"
            write_wav(audio, rate=16000, frames=16)
            write_wav(replacement, rate=32000, frames=32)
            replacement_sha = file_sha(replacement)
            value_song = song(frames=16, rate=16000)
            value_song["audio"]["sha256"] = replacement_sha
            path = root / "manifest.json"
            write_json(path, manifest([value_song]))
            original_hash = dataset_module.file_sha
            switched = False

            def swap_before_hash(path_arg):
                nonlocal switched
                if Path(path_arg) == audio and not switched:
                    os.replace(str(replacement), str(audio))
                    switched = True
                return original_hash(path_arg)

            # The inspected clock is 16/16000; the hashed path then becomes
            # 32/32000. A loader must not accept that mixed-version identity.
            with patch.object(dataset_module, "file_sha", side_effect=swap_before_hash):
                with self.assertRaises(ValidationError):
                    load_dataset(path)

    def test_structure_receipt_binds_the_consumed_json_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            write_wav(audio, rate=16000, frames=16000)
            value_song = song(frames=16000)
            value_song["audio"]["sha256"] = file_sha(audio)
            dataset_path, structure_path = root / "dataset.json", root / "structure.json"
            write_json(dataset_path, manifest([value_song]))
            first = {"beats": [0], "downbeats": [0], "segments": [
                {"start": 0, "end": 1, "label": "first"}]}
            later = {"beats": [0], "downbeats": [0], "segments": [
                {"start": 0, "end": 1, "label": "later"}]}
            write_json(structure_path, first)
            consumed_sha = file_sha(structure_path)
            original_read = dataset_module.read_json
            switched = False

            def replace_after_read(path_arg):
                nonlocal switched
                result = original_read(path_arg)
                if Path(path_arg) == structure_path and not switched:
                    write_json(structure_path, later, overwrite=True)
                    switched = True
                return result

            out = root / "imported.json"
            with patch.object(dataset_module, "read_json", side_effect=replace_after_read):
                try:
                    import_structure(dataset_path, "review_song", structure_path, out,
                                     audio_sha256=value_song["audio"]["sha256"])
                except ValidationError:
                    return  # Rejecting a concurrently changed input is valid.
            receipt = read_json(root / "imported.structure_receipt.json")
            self.assertEqual(receipt["tool_output_sha256"], consumed_sha)

    def test_interrupted_derived_wav_is_not_published_as_final_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source.wav", root / "window.wav"
            write_wav(source, rate=16000, frames=81000)
            audio = inspect_audio(source)
            window = make_windows(81000, 16000)[0]
            original_open = Path.open

            class InterruptedWriter:
                def __init__(self, handle):
                    self.handle = handle

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.handle.close()

                def write(self, value):
                    self.handle.write(value[:10])
                    self.handle.flush()
                    raise OSError("simulated interruption during final-path write")

                def __getattr__(self, name):
                    return getattr(self.handle, name)

            def interrupt_final_path(path, mode="r", *args, **kwargs):
                handle = original_open(path, mode, *args, **kwargs)
                if path == destination and mode in ("xb", "wb"):
                    return InterruptedWriter(handle)
                return handle

            with patch.object(Path, "open", new=interrupt_final_path):
                try:
                    write_window(audio, window, destination)
                except OSError:
                    self.assertFalse(destination.exists(), "A partial final WAV permanently blocks resume")
                else:
                    # Atomic temporary publication never opens the final path
                    # for writing, so this fault does not affect that design.
                    self.assertEqual(inspect_audio(destination)["num_samples"], 81000)


class ProviderBoundaryReviewTests(unittest.TestCase):
    def test_null_usage_is_unknown_cost_not_a_lost_successful_response(self):
        payload = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": None}

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return json.dumps(payload).encode("utf-8")

        provider = ChatCompletionsProvider({"kind": "chat_completions", "model": "unit-test-model",
                                            "base_url": "http://localhost:9/v1", "api_key_env": ""})
        with patch("choreobench.providers.urllib.request.urlopen", return_value=Response()):
            result = provider.generate({"messages": [{"role": "user", "content": "Return JSON."}],
                                        "seed": 42, "max_output_tokens": 32})
        self.assertEqual(result["text"], "{}")
        self.assertEqual(result["stop_reason"], "stop")
        self.assertIsNone(result["costs"]["input_tokens"])
        self.assertIsNone(result["costs"]["output_tokens"])
        self.assertEqual(json.loads(result["raw"]["body"]), payload)

    def test_unbound_replay_cannot_be_labelled_non_synthetic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            write_json(path, {"schema": "choreobench.replay.v1", "synthetic": False, "source": {},
                              "responses": {"song/local/0000/00": {"text": "An unbound fixed caption."}}})
            try:
                provider = ReplayProvider({"kind": "replay", "path": str(path)}, role="local")
                result = provider.generate({"call_id": "song/local/0000/00", "response_kind": "local_caption",
                                            "seed": 42, "max_output_tokens": 64,
                                            "audio": {"sha256": "1" * 64}})
            except ValueError:
                return
            self.assertTrue(result["synthetic"], "A replay with neither request binding nor provenance is a fixture")


if __name__ == "__main__":
    unittest.main()
