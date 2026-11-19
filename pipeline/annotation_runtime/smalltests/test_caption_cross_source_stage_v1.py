"""Real frozen prepare/request/filter/receipt/resume with CPU-only fake media/model."""
import base64
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
import caption_cross_source_stage_v1 as cross
import caption_direct_facts_v1 as direct
bar, _ = direct.native()


class FakeLLM:
    def __init__(self, bad_source=None):
        self.calls = []
        self.samplings = []
        self.bad_source = bad_source

    def chat(self, messages, sampling, use_tqdm=False):
        self.calls.append(copy.deepcopy(messages))
        self.samplings.append(copy.deepcopy(sampling))
        outputs = []
        for msgs in messages:
            if msgs[0]["content"] != bar.SYSTEM:
                raise AssertionError("No boundary/rewrite model dispatch permitted")
            header = json.loads(msgs[1]["content"][0]["text"])
            indices = header["allowed_source_frame_indices"][:2]
            value = bar.empty_observation("CPU fake completion, not semantic truth")
            events = []
            for event_id, part, text in (("e1", "head", "The dancer turns the head."),
                    ("e2", "left_hand", "The dancer opens the left hand. The right foot moves.")):
                supported = [i for i in indices if i not in header["explicit_raw2d_forbidden_part_frames"].get(part, [])]
                if len(supported) >= 2:
                    value["part_support"][bar.PARTS.index(part)]["frame_indices"] = supported
                    events.append(dict(id=event_id, description=text, parts=[part], frame_indices=supported))
            value.update(events=events, abstain=not bool(events))
            images = [base64.b64decode(c["image_url"]["url"].split(",", 1)[1])
                      for c in msgs[1]["content"] if c["type"] == "image_url"]
            invalid = self.bad_source is not None and any(self.bad_source.encode() in image for image in images)
            text = "not-json" if invalid else json.dumps(value, ensure_ascii=False)
            completion = SimpleNamespace(text=text, token_ids=[7, 8], finish_reason="stop", stop_reason=None)
            outputs.append(SimpleNamespace(outputs=[completion], prompt_token_ids=[1, 2, 3]))
        return outputs

    def get_tokenizer(self):
        return SimpleNamespace(eos_token_id=9)


class Fixture:
    """Only external media probe/decode and vLLM sampling constructor are faked."""
    def __init__(self, base, empty_window=False, masked=False):
        self.base = Path(base)
        self.out = self.base / "out"
        self.out.mkdir()
        self.probes = {}
        self.entries = []
        self.decoder_calls = []
        records = []
        for source_index, count in enumerate((3, 3, 2)):
            sid = "source%d" % source_index
            video, audio = self.base / (sid + ".mp4"), self.base / (sid + ".wav")
            video.write_bytes((sid + " synthetic CPU source, not real video").encode())
            audio.write_bytes(b"provenance only, never model input")
            record = dict(id=sid, video_path=str(video), audio_path=str(audio), nlf_path=None,
                          source_group="cpu_contract", split="train")
            records.append(record)
            times = [i / 8. for i in range(count * 4)]
            pts_sha = bar.batch.digest(times)
            probe = dict(times=times, source_timestamps=times, actual_video_pts_sha256=pts_sha,
                         media_start=0., media_end=count / 2., video_fps=8., video_pts_offset=0.)
            self.probes[str(video)] = probe
            binding = dict(id=sid, input_sha256=None, video_sha256=bar.batch.sha256(video),
                           audio_sha256=bar.batch.sha256(audio), actual_video_pts_sha256=pts_sha,
                           decoded_video_frames=len(times), media_start=0., media_end=count / 2., video_fps=8.)
            windows = [dict(start=i / 2., end=(i + 1) / 2., raw_window_indices=[i]) for i in range(count)]
            if empty_window and source_index == 0:
                windows[0]["end"] = .125
            row = dict(schema=bar.SCHEMA, id=sid, manifest_record=record, binding=binding, windows=windows,
                window_manifest=dict(schema="motionscript_music_windows_v1", id=sid, binding=copy.deepcopy(binding),
                    windows=copy.deepcopy(windows), params={"policy": "merge_short"},
                    raw_windows=copy.deepcopy(windows), fallback_metadata={}),
                source_refs={}, plan_status="ready")
            if masked:
                mask_path = self.base / (sid + ".mask.json")
                # Different per-source masks detect late-bound cross-source closures.
                part = "head" if source_index == 0 else "left_hand" if source_index == 1 else None
                mask = dict(schema="bar_raw2d_part_mask_v1", id=sid, video_sha256=binding["video_sha256"],
                    actual_video_pts_sha256=pts_sha, parts={} if part is None else
                    {part: {str(i): False for i in range(len(times))}})
                bar.core.save_json(mask_path, mask)
                row["raw2d_mask"] = dict(path=str(mask_path), sha256=bar.batch.sha256(mask_path))
            self.entries.append(dict(row=row, raw_line=b"", root=self.out / "sources" / sid, binding={}))
        manifest = self.base / "manifest.jsonl"
        manifest.write_bytes(b"".join(json.dumps(r, ensure_ascii=False, sort_keys=True).encode() + b"\n" for r in records))
        for entry in self.entries:
            row = entry["row"]
            row["source_refs"] = {"manifest": dict(path=str(manifest), sha256=bar.batch.sha256(manifest))}
            entry["raw_line"] = bar.batch.json_bytes(row) + b"\n"
            entry["binding"] = dict(source_binding=row["binding"], plan_sha256="a" * 64,
                full_source_row_sha256=bar.batch.digest(row), raw_line_sha256=bar.sha(entry["raw_line"]),
                manifest_record_sha256=bar.batch.digest(row["manifest_record"]), params_sha256="b" * 64)
        self.args = bar.parser().parse_args(["--plan", str(self.base / "unused-plan"),
            "--model", str(self.base / "unused-model"), "--out", str(self.out), "--stage", "run",
            "--boundary-review", "off", "--batch-size", "8", "--max-num-seqs", "8",
            "--ffmpeg", "cpu-fake-ffmpeg", "--ffprobe", "cpu-fake-ffprobe"])
        self.ref_cache = {}

    def decode(self, command, **kwargs):
        if command[0] != "cpu-fake-ffmpeg":
            raise AssertionError("Unexpected external command")
        self.decoder_calls.append(command)
        script = Path(command[command.index("-filter_script:v") + 1]).read_text()
        indices = [int(i) for i in re.findall(r"eq\(n\\,(\d+)\)", script)]
        sid = Path(command[command.index("-i") + 1]).stem
        for offset, index in enumerate(indices, 1):
            Path(command[-1] % offset).write_bytes((sid + " CPU image frame=" + str(index)).encode())
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    def sampling(self, sampling_cls, schema, max_tokens):
        return SimpleNamespace(temperature=0., seed=0, max_tokens=max_tokens, json_schema=copy.deepcopy(schema))

    def context(self):
        stack = ExitStack()
        stack.enter_context(patch.object(bar, "probe_actual", side_effect=lambda path, _: copy.deepcopy(self.probes[path])))
        stack.enter_context(patch.object(bar.subprocess, "run", side_effect=self.decode))
        stack.enter_context(patch.object(bar.core, "structured_sampling", side_effect=self.sampling))
        stack.enter_context(direct.direct_recipe(bar, self.args))
        for entry in self.entries:
            stack.enter_context(bar.batch.bundle_lock(entry["root"]))
        return stack

    def runtime(self, llm):
        return lambda: (llm, object)


def no_runtime():
    raise AssertionError("Completed native receipts must not dispatch/reload")


class CrossSourceContracts(unittest.TestCase):
    def test_three_sources_eight_windows_one_batch_original_requests_and_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp, masked=True)
            llm = FakeLLM()
            original_filter = bar.filter_supported_observation
            expected_filters = {}

            def checked_filter(value, manifest, forbidden):
                if not expected_filters:
                    for entry in f.entries:
                        prepared = bar.load(entry["root"] / "prepared.json")
                        for window in prepared["windows"]:
                            expected_filters[window["frames"][0]["sha256"]] = {p: set(v) for p, v in prepared["forbidden"].items()}
                self.assertEqual(forbidden, expected_filters[manifest["frames"][0]["sha256"]])
                return original_filter(value, manifest, forbidden)

            with patch.object(bar, "prepare_source", wraps=bar.prepare_source) as prepare, \
                    patch.object(bar, "verify_prepared", wraps=bar.verify_prepared) as verify, \
                    patch.object(bar, "filter_supported_observation", side_effect=checked_filter), f.context():
                staged = cross.stage_sources(f.entries, f.args, f.runtime(llm), f.ref_cache)
                self.assertEqual(prepare.call_count, 3)
                self.assertEqual(verify.call_count, 0)
                self.assertEqual([len(c) for c in llm.calls], [8])
                self.assertEqual(len(f.decoder_calls), 3)
                self.assertEqual(len(staged.receipts), 8)
                self.assertFalse(list(f.out.glob("sources/*/results/*.json")))
                self.assertFalse(list(f.out.rglob("evidence_abstention.json")))
                self.assertFalse((f.out / "progress.json").exists())
                self.assertFalse((f.out / "latest.summary.json").exists())
                expected_requests = []
                for entry in f.entries:
                    prepared = bar.load(entry["root"] / "prepared.json")
                    forbidden = {p: set(v) for p, v in prepared["forbidden"].items()}
                    for manifest in prepared["windows"]:
                        root = entry["root"] / "windows" / manifest["window"]["window_id"] / "attempt-0"
                        request = bar.visual_request(f.args, entry["root"] / "rgb", manifest, forbidden)
                        self.assertEqual((root / "request.json").read_bytes(), bar.batch.json_bytes(request))
                        self.assertEqual(bar.load(root / "binding.json"), {**entry["binding"],
                            "window": manifest["window"], "frames_sha256": bar.batch.digest(manifest["frames"])})
                        receipt = bar.load(root / "receipt.json")
                        self.assertEqual(receipt["status"], "candidate", receipt.get("error"))
                        self.assertTrue({"request.json", "binding.json", "response.raw.txt", "generation.json",
                            "response.json", "final.json"} <= set(receipt["files_sha256"]))
                        expected_requests.append(request)
                self.assertEqual(llm.calls[0], [r["messages"] for r in expected_requests])
                for sampling, request in zip(llm.samplings[0], expected_requests):
                    self.assertEqual(vars(sampling), {**request["sampling"], "json_schema": request["json_schema"]})
                results = [cross.resume_source(staged, i, f.args, no_runtime, f.ref_cache) for i in range(3)]
                self.assertEqual(verify.call_count, 3)
                self.assertEqual(prepare.call_count, 3)
                self.assertEqual([len(c) for c in llm.calls], [8])
                self.assertEqual([len(r["facts"]) for r in results], [3, 3, 4])
                self.assertTrue(all(r["status"] == "candidate" and not r["boundary_observations"] and
                    r["rewrite_fulltext_calls_per_attempt"] == 0 and not r["training_eligible"] for r in results))
                self.assertTrue(all("right foot" not in fact["text"] for r in results for fact in r["facts"]))
                replay = cross.stage_sources(f.entries, f.args, no_runtime, f.ref_cache)
                self.assertEqual(verify.call_count, 6)
                self.assertEqual(replay.receipts, staged.receipts)
                again = [cross.resume_source(replay, i, f.args, no_runtime, f.ref_cache) for i in range(3)]
                self.assertEqual(again, results)
                self.assertEqual([len(c) for c in llm.calls], [8])

    def test_batch8_and_native_single_source_batch2_same_output_bytes_and_facts(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp, masked=True)
            llm = FakeLLM()
            with f.context():
                staged = cross.stage_sources(f.entries, f.args, f.runtime(llm), f.ref_cache)
                cross_results = [cross.resume_source(staged, i, f.args, no_runtime, f.ref_cache) for i in range(3)]
                expected = {}
                for root in (entry["root"] for entry in f.entries):
                    for path in root.rglob("*"):
                        if path.is_file() and (path.name in ("request.json", "binding.json", "response.raw.txt",
                                "generation.json", "response.json", "final.json", "receipt.json") or path.parent.name == "results"):
                            expected[path] = path.read_bytes()
                # Reuse the exact paths/prepared inputs; only this temporary CPU fixture is reset.
                for root in (entry["root"] for entry in f.entries):
                    for path in root.glob("windows/*/attempt-0/*"):
                        if path.is_file():
                            path.unlink()
                    for path in root.glob("results/*.json"):
                        path.unlink()
                f.args.batch_size = 2
                single_llm = FakeLLM()
                single = [bar.process_source(e["row"], e["raw_line"], e["root"], f.args, e["binding"],
                    f.runtime(single_llm), f.ref_cache) for e in f.entries]
                self.assertEqual([len(c) for c in single_llm.calls], [2, 1, 2, 1, 2])
                self.assertEqual(single, cross_results)
                for path, data in expected.items():
                    self.assertEqual(path.read_bytes(), data, str(path))

    def test_failed_source_does_not_redispatch_completed_other_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            llm = FakeLLM(bad_source="source1")
            with f.context():
                staged = cross.stage_sources(f.entries, f.args, f.runtime(llm), f.ref_cache)
                results = [cross.resume_source(staged, i, f.args, no_runtime, f.ref_cache) for i in range(3)]
                self.assertEqual([r["status"] for r in results], ["candidate", "failed", "candidate"])
                snapshots = {p: p.read_bytes() for p in f.out.glob("sources/*/windows/*/attempt-0/receipt.json")}
                replay = cross.stage_sources(f.entries, f.args, no_runtime, f.ref_cache)
                again = [cross.resume_source(replay, i, f.args, no_runtime, f.ref_cache) for i in (2, 1, 0)]
                self.assertEqual([r["id"] for r in again], ["source2", "source1", "source0"])
                self.assertEqual([len(c) for c in llm.calls], [8])
                self.assertEqual({p: p.read_bytes() for p in snapshots}, snapshots)
                self.assertFalse(list(f.out.glob("sources/*/windows/*/attempt-1")))

    def test_orphan_raw_is_failed_never_resubmitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            llm = FakeLLM()
            with f.context():
                root = f.entries[0]["root"] / "windows/w00000/attempt-0"
                root.mkdir(parents=True)
                (root / "response.raw.txt").write_bytes(b"{}")
                staged = cross.stage_sources(f.entries, f.args, f.runtime(llm), f.ref_cache)
                self.assertEqual([len(c) for c in llm.calls], [7])
                self.assertEqual(staged.receipts[str(root)]["status"], "failed")
                results = [cross.resume_source(staged, i, f.args, no_runtime, f.ref_cache) for i in range(3)]
                self.assertEqual(results[0]["status"], "failed")
                self.assertEqual(results[0]["window_observations"][0]["status"], "failed")
                self.assertEqual((root / "response.raw.txt").read_bytes(), b"{}")
                cross.stage_sources(f.entries, f.args, no_runtime, f.ref_cache)
                self.assertEqual([len(c) for c in llm.calls], [7])

    def test_empty_window_abstains_only_on_native_resume_and_has_no_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp, empty_window=True)
            llm = FakeLLM()
            with f.context():
                staged = cross.stage_sources(f.entries, f.args, f.runtime(llm), f.ref_cache)
                empty = f.entries[0]["root"] / "windows/w00000"
                self.assertEqual([len(c) for c in llm.calls], [7])
                self.assertFalse((empty / "attempt-0").exists())
                self.assertFalse((empty / "evidence_abstention.json").exists())
                result = cross.resume_source(staged, 0, f.args, no_runtime, f.ref_cache)
                self.assertEqual(result["window_observations"][0]["status"], "abstain")
                self.assertFalse(result["window_observations"][0]["model_call_performed"])
                self.assertEqual(result["final"]["json"]["windows"][0]["caption"], "")
                self.assertFalse(result["coverage"]["full_window_caption"])
                self.assertTrue((empty / "evidence_abstention.json").exists())
                self.assertFalse((empty / "attempt-0").exists())

    def test_preparation_failure_remains_per_source_exception_without_terminal_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            llm = FakeLLM()
            f.entries[1]["row"]["binding"]["video_sha256"] = "f" * 64
            with f.context():
                staged = cross.stage_sources(f.entries, f.args, f.runtime(llm), f.ref_cache)
                self.assertIsInstance(staged.errors[1], ValueError)
                self.assertEqual([len(c) for c in llm.calls], [5])
                self.assertFalse(list(f.entries[1]["root"].glob("failures/*.json")))
                self.assertFalse(list(f.out.glob("sources/*/results/*.json")))
                with self.assertRaisesRegex(ValueError, "source media SHA"):
                    cross.resume_source(staged, 1, f.args, no_runtime, f.ref_cache)
                for index in (0, 2):
                    self.assertEqual(cross.resume_source(staged, index, f.args, no_runtime, f.ref_cache)["status"], "candidate")
                cross.stage_sources(f.entries, f.args, no_runtime, f.ref_cache)
                self.assertEqual([len(c) for c in llm.calls], [5])

    def test_changed_input_row_binding_raw_line_and_configuration_rejected_before_resume(self):
        for key in ("row", "binding", "raw_line", "args"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(tmp)
                with f.context():
                    staged = cross.stage_sources(f.entries, f.args, f.runtime(FakeLLM()), f.ref_cache)
                    if key == "args":
                        f.args.visual_max_tokens += 1
                    elif key == "raw_line":
                        f.entries[1][key] += b" "
                    else:
                        f.entries[1][key]["tampered"] = True
                    with self.assertRaisesRegex(ValueError, "inputs or configuration changed"):
                        cross.resume_source(staged, 0, f.args, no_runtime, f.ref_cache)
                    self.assertFalse(list(f.out.glob("sources/*/results/*.json")))

    def test_changed_source_asset_rejected_by_original_verify_before_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            with f.context():
                staged = cross.stage_sources(f.entries, f.args, f.runtime(FakeLLM()), f.ref_cache)
                Path(f.entries[0]["row"]["manifest_record"]["video_path"]).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "source asset changed"):
                    cross.resume_source(staged, 0, f.args, no_runtime, f.ref_cache)
                self.assertEqual(cross.resume_source(staged, 2, f.args, no_runtime, f.ref_cache)["status"], "candidate")

    def test_tampered_terminal_receipt_rejected_without_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            with f.context():
                staged = cross.stage_sources(f.entries, f.args, f.runtime(FakeLLM()), f.ref_cache)
                root = f.entries[1]["root"] / "windows/w00000/attempt-0"
                (root / "response.raw.txt").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                    cross.resume_source(staged, 1, f.args, no_runtime, f.ref_cache)
                for index in (0, 2):
                    self.assertEqual(cross.resume_source(staged, index, f.args, no_runtime, f.ref_cache)["status"], "candidate")

    def test_requires_explicit_run_no_retry_and_distinct_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            f.args.retry_failed = True
            with self.assertRaisesRegex(ValueError, "retry_failed=False"):
                cross.stage_sources(f.entries, f.args, no_runtime, f.ref_cache)
            f.args.retry_failed = False
            f.args.stage = "export"
            with self.assertRaisesRegex(ValueError, "stage=run"):
                cross.stage_sources(f.entries, f.args, no_runtime, f.ref_cache)
            f.args.stage = "run"
            f.entries[1]["root"] = f.entries[0]["root"]
            with self.assertRaisesRegex(ValueError, "Duplicate caller source roots"):
                cross.stage_sources(f.entries, f.args, no_runtime, f.ref_cache)
            self.assertFalse(list(f.out.glob("sources/*/prepared.json")))


if __name__ == "__main__":
    unittest.main()
