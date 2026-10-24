"""CPU hash/message fixtures, not real videos or successful model observations."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

MS=Path(__file__).resolve().parents[1]/"scripts_omnidance/exp/ms"
sys.path.insert(0,str(MS))
import ms_visual_failed3_reobserve as observe


class ReobserveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.old=self.root/"old";self.old.mkdir()
        row={"id":"fixture","source_group":"synthetic CPU fixture","split":"train"}
        for key in ("video_path","audio_path","nlf_path"):
            p=self.root/key;p.write_bytes(("hash fixture, not actual "+key).encode());row[key]=str(p)
        frames=[]
        for index in (0,4):
            p=self.old/f"frame-{index}.jpg";p.write_bytes(b"not JPEG hash fixture"+str(index).encode())
            frames.append({"index":index,"time":index/16,"source_timestamp":index/16,"file":p.name,"sha256":observe.batch.sha256(p)})
        self.original={"model":"/fixture/local-model-not-loaded","messages":[{"role":"system","content":observe.batch.SYSTEM},{"role":"user","content":[{"type":"text","text":"Original sequence."},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,Zml4dHVyZQ=="}}]}],"temperature":0.0,"max_tokens":1200}
        for name,value in (("request.json",self.original),("manifest.json",{}),("manifest.receipt.json",{}),("generation.json",{}),("receipt.json",{})):
            (self.old/name).write_bytes(observe.batch.json_bytes(value))
        (self.old/"response.raw.txt").write_bytes(b"SENTINEL OLD CAPTION MUST NOT ENTER THE NEW PROMPT")
        run=self.root/"run.json";run.write_bytes(b"synthetic original run hash fixture")
        self.case={"spec":{"id":"fixture","window":{"start":0.0,"end":2.0},"failure_class":"frame_support","error":"CPU fixture old evidence failure"},
            "bundle":str(self.old),"manifest":{"input":row,"window":{"start":0.0,"end":2.0},"frames":frames},
            "source":observe.batch.source_identity(row),"original_request":self.original,"new_request":observe.new_request(self.original),
            "original_files_sha256":{p.name:observe.batch.sha256(p) for p in self.old.iterdir() if p.suffix!=".jpg"},
            "original_run_receipt_path":str(run),"original_run_receipt_sha256":observe.batch.sha256(run)}
        self.dest,self.lineage=observe.prepare(self.case,self.root/"new", "p"*64)

    def value(self):
        return {"motion_caption":"The arm rises, then lowers.","events":[{"id":"e1","description":"An arm rise followed by lowering.","frame_indices":[0,4]}],"uncertainties":[],"abstain":False}

    def finish(self,value=None,finish="stop",tokens=100):
        stats={"finish_reason":finish,"completion_tokens":tokens,"max_tokens":1200,"synthetic_fixture_not_real_model_call":True}
        return observe.finish(self.dest,self.case,self.lineage,observe.batch.json_bytes(value if value is not None else self.value()),stats)

    def test_three_distinct_failure_classes_not_previous_two_cases(self):
        self.assertEqual([s["failure_class"] for s in observe.SPECS],["frame_support","word_limit","abstention_contract"])
        self.assertNotIn("TKBOLWtIPQzcxGLa",[s["id"] for s in observe.SPECS])
        self.assertNotIn("HtezJbaEumeOaxQT",[s["id"] for s in observe.SPECS])
        for spec in observe.SPECS:
            self.assertEqual(spec["window"],{"start":0.0,"end":2.0})
            for key in ("request_sha256","raw_sha256","generation_sha256"):
                self.assertEqual(len(spec[key]),64)

    def test_new_prompt_only_appends_contract_preserves_old_system_images_and_bytes(self):
        original=copy.deepcopy(self.original)
        new=observe.new_request(original)
        self.assertEqual(original,self.original)
        self.assertEqual(new["messages"][0],original["messages"][0])
        self.assertEqual(new["messages"][1]["content"][:-1],original["messages"][1]["content"])
        self.assertEqual(new["messages"][1]["content"][-1],{"type":"text","text":observe.CLARIFICATION})
        self.assertNotIn("SENTINEL OLD CAPTION",json.dumps(new))
        self.assertNotIn("structured_outputs",new)
        self.assertEqual((self.dest/"request.original.json").read_bytes(),(self.old/"request.json").read_bytes())
        self.assertFalse(self.lineage["old_caption_supplied_to_model"])
        self.assertFalse(self.lineage["structured_pair_enum_used"])

    def test_static_empty_abstain_contract_not_forced_candidate(self):
        self.assertIn("at most 90 English words",observe.CLARIFICATION)
        self.assertIn("hard limit\nremains 140 words",observe.CLARIFICATION)
        self.assertIn('abstain=true, motion_caption="", events=[]',observe.CLARIFICATION)
        self.assertIn("purely static sequence",observe.CLARIFICATION)
        self.assertIn("Never force a movement",observe.CLARIFICATION)
        self.assertIn("two distinct",observe.CLARIFICATION)
        self.assertIn("Merely adding a second index",observe.CLARIFICATION)
        self.assertIn("uncertainties",observe.CLARIFICATION)

    def test_candidate_fixture_no_approval_old_error_or_raw_changed(self):
        before=(self.old/"response.raw.txt").read_bytes();r=self.finish()
        self.assertEqual(r["status"],"candidate")
        for key in ("full_pool_updated","semantic_approval","human_review_completed","formal_training"):
            self.assertIs(r[key],False)
        self.assertEqual((self.old/"response.raw.txt").read_bytes(),before)
        self.assertTrue(r["original_failures_preserved"])
        gen=observe.load(self.dest/"generation.json")
        self.assertIsNone(gen["structured_output_constraint"])
        self.assertTrue(gen["synthetic_fixture_not_real_model_call"])

    def test_target90_is_advisory_hard140_validator_unchanged(self):
        value=self.value();value["motion_caption"]="word "*100
        self.assertEqual(self.finish(value)["status"],"candidate")
        self.assertEqual(len(observe.load(self.dest/"response.raw.txt")["motion_caption"].split()),100)

    def test_over140_not_cut_down_or_rewritten(self):
        value=self.value();value["motion_caption"]="word "*141;r=self.finish(value)
        self.assertEqual(r["status"],"failed");self.assertIn("140 words",r["error"])
        self.assertEqual(len(observe.load(self.dest/"response.raw.txt")["motion_caption"].split()),141)
        self.assertFalse((self.dest/"candidate.json").exists())

    def test_single_duplicate_and_foreign_frame_not_repaired(self):
        for indices in ([0],[0,0],[0,999]):
            value=self.value();value["events"][0]["frame_indices"]=indices
            with self.assertRaises(ValueError):observe.batch.validate_candidate(value,self.case["manifest"])
        value=self.value();value["events"][0]["frame_indices"]=[0];r=self.finish(value)
        self.assertEqual(r["status"],"failed")
        self.assertEqual(observe.load(self.dest/"response.raw.txt")["events"][0]["frame_indices"],[0])

    def test_numeric_id_not_cast_to_string(self):
        value=self.value();value["events"][0]["id"]=1;r=self.finish(value)
        self.assertEqual(r["status"],"failed")
        self.assertEqual(observe.load(self.dest/"response.raw.txt")["events"][0]["id"],1)

    def test_true_empty_abstain_is_valid_but_not_nonempty_caption(self):
        value={"motion_caption":"","events":[],"uncertainties":["No movement supported."],"abstain":True};r=self.finish(value)
        self.assertEqual(r["status"],"abstain")
        self.assertEqual(observe.load(self.dest/"candidate.json")["motion_caption"],"")
        self.assertFalse(r["formal_training"])

    def test_nonempty_abstain_remains_failed_not_cleared(self):
        value={"motion_caption":"Static fixture pose.","events":[],"uncertainties":[],"abstain":True};r=self.finish(value)
        self.assertEqual(r["status"],"failed")
        self.assertIn("empty caption",r["error"])
        self.assertEqual(observe.load(self.dest/"response.raw.txt")["motion_caption"],value["motion_caption"])

    def test_length_valid_json_remains_failed(self):
        r=self.finish(finish="length",tokens=1200)
        self.assertEqual(r["status"],"failed");self.assertIn("truncated",r["error"])
        self.assertTrue((self.dest/"response.raw.txt").exists());self.assertFalse((self.dest/"candidate.json").exists())

    def test_old_frame_and_receipt_modified_cannot_be_attested(self):
        for p in (self.old/"frame-0.jpg",self.old/"receipt.json",Path(self.case["original_run_receipt_path"])):
            before=p.read_bytes();p.write_bytes(b"tampered hash fixture")
            with self.assertRaisesRegex(ValueError,"changed"):observe.unchanged(self.case)
            p.write_bytes(before)
        observe.unchanged(self.case)

    def test_one_real_trial_only_no_automatic_retry_or_overwrite(self):
        self.finish();before=(self.dest/"response.raw.txt").read_bytes()
        with self.assertRaisesRegex(ValueError,"one trial only"):self.finish()
        self.assertEqual((self.dest/"response.raw.txt").read_bytes(),before)
        with self.assertRaises(FileExistsError):observe.prepare(self.case,self.root/"new","p"*64)


if __name__=="__main__":unittest.main()
