"""CPU serialized evidence/schema fixtures only; never call vLLM or a detector.

Hash fixture bytes are not real videos, JPEGs, NLF assets or model completions.
No fixture candidate is claimed as a real generated or human-reviewed result.
"""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

MS = Path(__file__).resolve().parents[1]/"scripts_omnidance/exp/ms"
sys.path.insert(0,str(MS))
import ms_visual_constrained_retry as retry


class ConstrainedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.old=self.root/"old-bundle"
        self.old.mkdir()
        self.sources=self.root/"synthetic-hash-sources"
        self.sources.mkdir()
        row={"id":"fixture","source_group":"synthetic hash fixture","split":"train","input_line":1,"input_row_sha256":"f"*64}
        for key in ("video_path","audio_path","nlf_path"):
            path=self.sources/key
            path.write_bytes(("CPU hash fixture not real "+key).encode())
            row[key]=str(path)
        source=retry.batch.source_identity(row)
        frames=[]
        for index in (0,4):
            p=self.old/f"frame-{index}.jpg"
            p.write_bytes(f"CPU hash fixture {index}, not JPEG".encode())
            frames.append({"index":index,"time":index/16,"source_timestamp":index/16,"file":p.name,"sha256":retry.batch.sha256(p)})
        manifest={"input":row,"window":{"start":0.0,"end":2.0},"frames":frames}
        raw_original=retry.batch.json_bytes({"motion_caption":"Unreviewed CPU serialization fixture.","events":[{"id":"e1","description":"Fixture.","frame_indices":[0]}],"uncertainties":[],"abstain":False})
        for name,raw in (("manifest.json",retry.batch.json_bytes(manifest)),("manifest.receipt.json",b"{}"),("request.json",retry.batch.json_bytes({"model":"fixture-not-loaded","messages":[]})),("response.raw.txt",raw_original),("generation.json",b"{}"),("receipt.json",b"{}")):
            (self.old/name).write_bytes(raw)
        run_path=self.root/"original-run.receipt.json"
        run_path.write_bytes(b"synthetic serialized hash fixture, not a real VLM run")
        self.case={"bundle":str(self.old),"run_receipt_path":str(run_path),"run_receipt_sha256":retry.batch.sha256(run_path),
            "manifest":manifest,"source":source,"request_bytes":(self.old/"request.json").read_bytes(),
            "original_files_sha256":{name:retry.batch.sha256(self.old/name) for name in ("manifest.json","manifest.receipt.json","request.json","response.raw.txt","generation.json","receipt.json")},
            "original_motion_caption":"Unreviewed CPU serialization fixture.","original_failure_reason":"single-frame CPU fixture rejection",
            "schema":retry.window_schema([0,4])}
        self.destination,self.lineage=retry.prepare_case(self.case,self.root/"new-independent-out",{"sha256":retry.MODEL_SHA},"p"*64)

    def value(self):
        return {"motion_caption":"The arm rises, then lowers.","events":[{"id":"e1","description":"An arm rise followed by lowering.","frame_indices":[0,4]}],"uncertainties":[],"abstain":False}

    def generation(self,finish="stop",tokens=100):
        return {"finish_reason":finish,"completion_tokens":tokens,"max_tokens":1200,
            "eos_token_id":151645,"last_output_token_id":151645,"last_output_token_is_eos":True,
            "synthetic_cpu_fixture_not_actual_model_inference":True}

    def finish(self,value=None,generation=None):
        return retry.finish_response(self.destination,self.case,self.lineage,retry.batch.json_bytes(value if value is not None else self.value()),generation or self.generation())

    def test_pair_enum_uses_real_global_indices_and_no_unsupported_uniqueItems(self):
        indices=[32,36,40,44,48,52,56,60,64,68,72,76,80]
        schema=retry.window_schema(indices)
        frame=schema["oneOf"][0]["properties"]["events"]["items"]["properties"]["frame_indices"]
        self.assertEqual(frame["enum"],[list(p) for p in itertools.combinations(indices,2)])
        self.assertEqual(len(frame["enum"]),78)
        self.assertEqual(frame["items"]["enum"],indices)
        self.assertEqual((frame["minItems"],frame["maxItems"]),(2,2))
        self.assertNotIn("uniqueItems",json.dumps(schema))
        self.assertNotIn("kick",json.dumps(schema))
        self.assertNotIn("genre",json.dumps(schema))
        self.assertNotIn([32,32],frame["enum"])
        self.assertNotIn([0,4],frame["enum"])

    def test_invalid_or_duplicate_input_indices_not_guessed(self):
        for indices in ([0],[4,0],[0,0],[False,4],[0,"4"],list(range(17))):
            with self.assertRaisesRegex(ValueError,"actual ordered distinct"):
                retry.window_schema(indices)

    def test_candidate_and_real_abstain_branch_structure(self):
        candidate,abstain=retry.window_schema([0,4])["oneOf"]
        self.assertEqual(candidate["properties"]["abstain"],{"const":False})
        self.assertEqual(candidate["properties"]["events"]["minItems"],1)
        self.assertEqual(abstain["properties"]["abstain"],{"const":True})
        self.assertEqual(abstain["properties"]["motion_caption"],{"const":""})
        self.assertEqual(abstain["properties"]["events"]["maxItems"],0)

    def test_new_request_exact_old_bytes_and_frames_only_referenced(self):
        self.assertEqual((self.destination/"request.json").read_bytes(),(self.old/"request.json").read_bytes())
        self.assertTrue(self.lineage["same_original_request_bytes"])
        self.assertTrue(self.lineage["new_independent_decoding_protocol"])
        self.assertTrue(self.lineage["not_free_decoding_resume"])
        self.assertFalse(self.lineage["full_pool_updated"])
        self.assertFalse((self.destination/"frames").exists())
        self.assertEqual(self.lineage["original_frames"][0]["original_absolute_path"],str(self.old/"frame-0.jpg"))

    def test_schema_does_not_grant_approval_and_valid_fixture_keeps_original_failed(self):
        old=(self.old/"response.raw.txt").read_bytes()
        receipt=self.finish()
        self.assertEqual(receipt["status"],"candidate")
        self.assertFalse(receipt["formal_training"])
        self.assertFalse(receipt["semantic_approval"])
        self.assertFalse(receipt["human_review_completed"])
        self.assertFalse(receipt["full_pool_updated"])
        self.assertEqual((self.old/"response.raw.txt").read_bytes(),old)
        candidate=retry.read_json(self.destination/"candidate.json")
        self.assertEqual(candidate["decoding_protocol"],retry.SCHEMA)
        self.assertEqual(candidate["events"][0]["frame_indices"],[0,4])
        self.assertFalse(candidate["full_pool_updated"])
        generation=retry.read_json(self.destination/"generation.json")
        self.assertTrue(generation["synthetic_cpu_fixture_not_actual_model_inference"])

    def test_numeric_and_blank_ids_rejected_without_coercion(self):
        for value in (1,""):
            candidate=self.value()
            candidate["events"][0]["id"]=value
            with self.assertRaisesRegex(ValueError,"unique nonempty"):
                retry.batch.validate_candidate(candidate,self.case["manifest"])
        candidate=self.value()
        candidate["events"][0]["id"]=1
        receipt=self.finish(candidate)
        self.assertEqual(receipt["status"],"failed")
        self.assertIn('"id": 1',(self.destination/"response.raw.txt").read_text())
        self.assertFalse((self.destination/"candidate.json").exists())

    def test_original_validator_rejects_single_duplicate_or_foreign_evidence(self):
        for indices in ([0],[0,0],[0,999]):
            value=self.value()
            value["events"][0]["frame_indices"]=indices
            with self.assertRaisesRegex(ValueError,"at least two distinct supplied"):
                retry.batch.validate_candidate(value,self.case["manifest"])
        value=self.value()
        value["events"][0]["frame_indices"]=[0]
        receipt=self.finish(value)
        self.assertEqual(receipt["status"],"failed")
        self.assertEqual(retry.read_json(self.destination/"response.raw.txt")["events"][0]["frame_indices"],[0])
        self.assertFalse((self.destination/"candidate.json").exists())

    def test_unordered_pair_is_rejected_in_new_protocol_not_silently_reordered(self):
        value=self.value()
        value["events"][0]["frame_indices"]=[4,0]
        self.assertTrue(retry.batch.validate_candidate(value,self.case["manifest"])["format_valid"])
        receipt=self.finish(value)
        self.assertEqual(receipt["status"],"failed")
        self.assertIn("exact allowed distinct",receipt["error"])
        self.assertEqual(retry.read_json(self.destination/"response.raw.txt")["events"][0]["frame_indices"],[4,0])

    def test_duplicate_event_id_still_rejected_by_original_validator(self):
        value=self.value()
        value["events"].append({"id":"e1","description":"Other fixture fact.","frame_indices":[0,4]})
        receipt=self.finish(value)
        self.assertEqual(receipt["status"],"failed")
        self.assertIn("unique nonempty",receipt["error"])

    def test_over140_words_not_truncated_or_rewritten_to_pass(self):
        value=self.value()
        value["motion_caption"]="word "*141
        receipt=self.finish(value)
        self.assertEqual(receipt["status"],"failed")
        self.assertIn("140 words",receipt["error"])
        self.assertEqual(len(retry.read_json(self.destination/"response.raw.txt")["motion_caption"].split()),141)

    def test_exact_sentence_repetition_not_cleaned_to_pass(self):
        value=self.value()
        value["motion_caption"]="The arm rises. The arm rises."
        receipt=self.finish(value)
        self.assertEqual(receipt["status"],"failed")
        self.assertIn("identical clause",receipt["error"])

    def test_length_finish_with_valid_json_keeps_raw_failed(self):
        receipt=self.finish(generation=self.generation("length",1200))
        self.assertEqual(receipt["status"],"failed")
        self.assertIn("truncated",receipt["error"])
        self.assertTrue((self.destination/"response.raw.txt").is_file())
        self.assertFalse((self.destination/"candidate.json").exists())

    def test_abstain_empty_not_padded_to_fulfill_goal(self):
        value={"motion_caption":"","events":[],"uncertainties":["No support"],"abstain":True}
        receipt=self.finish(value)
        self.assertEqual(receipt["status"],"abstain")
        candidate=retry.read_json(self.destination/"candidate.json")
        self.assertEqual(candidate["motion_caption"],"")
        self.assertEqual(candidate["events"],[])
        self.assertFalse(candidate["semantic_approval"])

    def test_abstain_with_caption_is_rejected_not_cleared(self):
        value={"motion_caption":"Unsupported fixture fact.","events":[],"uncertainties":[],"abstain":True}
        receipt=self.finish(value)
        self.assertEqual(receipt["status"],"failed")
        self.assertEqual(retry.read_json(self.destination/"response.raw.txt")["motion_caption"],value["motion_caption"])

    def test_changed_original_frame_and_run_receipt_detected(self):
        for path in (self.old/"frame-0.jpg",Path(self.case["run_receipt_path"])):
            original=path.read_bytes()
            path.write_bytes(b"changed hash fixture")
            with self.assertRaisesRegex(ValueError,"changed"):
                retry.verify_original_unchanged(self.case)
            path.write_bytes(original)
        retry.verify_original_unchanged(self.case)

    def test_changed_original_response_cannot_be_attested(self):
        (self.old/"response.raw.txt").write_bytes(b"changed old response")
        receipt=self.finish()
        self.assertEqual(receipt["status"],"failed")
        self.assertIn("changed",receipt["error"])
        self.assertFalse((self.destination/"candidate.json").exists())

    def test_outside_allowed_two_case_selection_rejected_before_vllm(self):
        with self.assertRaisesRegex(ValueError,"only the authorized two-window"):
            retry.main(["--bundle",str(self.old),"--out",str(self.root/"not-created"),
                "--model",str(self.root/"no-model"),"--model-reference",str(self.root/"no-reference"),
                "--original-full-root",str(self.root)])
        self.assertFalse((self.root/"not-created").exists())

    def test_one_trial_only_no_silent_resume_or_overwrite(self):
        self.finish()
        first=(self.destination/"response.raw.txt").read_bytes()
        with self.assertRaisesRegex(ValueError,"one trial only"):
            self.finish()
        self.assertEqual((self.destination/"response.raw.txt").read_bytes(),first)
        with self.assertRaises(FileExistsError):
            retry.prepare_case(self.case,self.root/"new-independent-out",{"sha256":retry.MODEL_SHA},"p"*64)


if __name__ == "__main__":
    unittest.main()
