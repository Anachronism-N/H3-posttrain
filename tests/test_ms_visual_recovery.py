"""Synthetic CPU-only recovery contracts; no GPU/model/detector loads or calls.

Fixtures represent serialized proof files only. Hash bytes are not JPEGs,
loadable weights or VLM completions; fixture candidates are not real labels.
"""
import argparse
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

MS=Path(__file__).resolve().parents[1]/"scripts_omnidance/exp/ms"
sys.path.insert(0,str(MS))
import ms_visual_recovery as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.old=self.root/"old";self.old.mkdir()
        self.run_output=self.root/"run-output"
        row={"id":"fixture","source_group":"CPU source fixture","split":"train","input_line":1,"input_row_sha256":"i"*64}
        for key in ("video_path","audio_path","nlf_path"):
            path=self.root/key;path.write_bytes(("Hash fixture, not real "+key).encode());row[key]=str(path)
        self.source=recovery.batch.source_identity(row)
        frames=[];content=[{"type":"text","text":recovery.batch.OBSERVE_TEXT+"Allowed Source frame indices for this window: [0, 4]. Use only these actual indices; never fill in missing evidence."}]
        for i in (0,4):
            path=self.old/f"frame-{i}.jpg";path.write_bytes(f"Frame hash fixture {i}; not actual JPEG".encode())
            f={"index":i,"time":i/16,"source_timestamp":i/16,"file":path.name,"sha256":recovery.batch.sha256(path)};frames.append(f)
            content.extend([{"type":"text","text":f"Source frame {i}, time {i/16:.6f} seconds."},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(path.read_bytes()).decode()}}])
        self.request={"model":"/fixture","messages":[{"role":"system","content":recovery.batch.SYSTEM},{"role":"user","content":content}],"temperature":0.0,"max_tokens":1200}
        self.write(self.old/"request.json",self.request)
        binding={"id":"fixture","source_sha256":self.source["sha256"],"model_sha256":"m"*64,"params_sha256":"o"*64,"window":{"start":0.0,"end":2.0}}
        self.manifest={"schema":recovery.batch.VISUAL_SCHEMA,"batch_schema":recovery.batch.SCHEMA,"frames":frames,"input":row,"window":binding["window"],"binding":binding,
            "video_sha256":self.source["files"]["video_path"]["sha256"],"request_sha256":recovery.batch.sha256(self.old/"request.json")}
        self.write(self.old/"manifest.json",self.manifest)
        self.write(self.old/"manifest.receipt.json",{"manifest_sha256":recovery.batch.sha256(self.old/"manifest.json")})
        oldraw=recovery.batch.json_bytes({"motion_caption":"Old unreviewed fixture, not an actual VLM output.","events":[{"id":"e1","description":"Fixture.","frame_indices":[0]}],"uncertainties":[],"abstain":False})
        self.write(self.old/"generation.json",{"model_call_performed":True,"raw_response_sha256":hashlib.sha256(oldraw).hexdigest(),"finish_reason":"stop","completion_tokens":10,"max_tokens":1200,"synthetic_fixture_not_real_model_call":True})
        old_receipt=recovery.batch.finish_response(self.old,binding,self.manifest,oldraw)
        final=self.old/"final.json";self.write(final,{"CPU_original_final_fixture":True})
        original_row={"id":"fixture","source_group":row["source_group"],"split":"train","original_input":row,"source_identity":self.source,"model_instance_identity_sha256":"m"*64,"params_sha256":"o"*64,"final_path":str(final),
            "windows":[{"status":"failed","window":binding["window"],"bundle":str(self.old),"source_frames":frames,"error":old_receipt["error"]}]}
        self.original_row=original_row
        self.target=recovery.targets_from_rows([original_row],str(self.run_output))[0][0]
        self.new=recovery.target_dir(self.run_output,self.target);self.new.mkdir(parents=True)
        self.params_sha="p"*64
        new,schema,lineage=recovery.expected_attempt_artifacts(self.target,self.manifest,self.params_sha)
        for name,value in (("request.json",new),("schema.json",schema),("lineage.json",lineage)):
            self.write(self.new/name,value)

    def write(self,p,v):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(recovery.batch.json_bytes(v))

    def response(self):
        return {"motion_caption":"The arm rises, then lowers.","events":[{"id":"e1","description":"The arm moves upward then lowers.","frame_indices":[0,4]}],"uncertainties":[],"abstain":False}

    def finish(self,v=None,finish="stop",tokens=100):
        recovery.claim_call(self.new,self.target,self.params_sha)
        stats={"finish_reason":finish,"completion_tokens":tokens,"max_tokens":1200,"prompt_tokens":100,
            "stop_reason":None,"eos_token_id":151645,"last_output_token_id":151645,"last_output_token_is_eos":True,
            "versions":{"vllm":"0.31.0","xgrammar":"0.2.7","llguidance":"1.7.6"},"synthetic_fixture_not_actual_model_call":True}
        return recovery.write_completion(self.new,self.target,self.manifest,recovery.batch.json_bytes(v if v is not None else self.response()),stats,self.params_sha,recovery.batch.digest(recovery.load(self.new/"schema.json")))

    def window(self,status,caption=""):
        return {"status":status,"caption":caption,"events":[],"error":None if status!="failed" else "original CPU fixture error","window":{"start":0.,"end":2.},"raw_response_path":"unchanged-old-raw"}

    def test_pending_before_source_traversal_or_export_read(self):
        args=argparse.Namespace(original_plan=str(self.root/"launch.json"),original_export=str(self.root/"not-yet-exported"),out=str(self.root/"plan-out"),run_out=str(self.root/"planned-run"))
        with mock.patch.object(recovery.original,"read_plan",return_value=({"jobs":[]},[])),mock.patch.object(recovery.original,"terminal_snapshot",side_effect=recovery.original.Pending("active original")):
            with self.assertRaises(recovery.original.Pending):recovery.plan_stage(args)
        self.assertFalse(Path(args.out).exists())
        with mock.patch.object(recovery,"plan_stage",side_effect=recovery.original.Pending("8workers active")),redirect_stdout(io.StringIO()):
            code=recovery.main(["plan","--original-plan",args.original_plan,"--original-export",args.original_export,"--out",args.out,"--run-out",args.run_out])
        self.assertEqual(code,2)
        self.assertFalse(Path(args.out).exists())

    def test_plan_only_original_failed_not_good_or_validabstain(self):
        bundle=self.old/"w0";bundle.mkdir()
        for name in ("manifest.json","request.json","response.raw.txt","generation.json","manifest.receipt.json","receipt.json"):(bundle/name).write_bytes(b"CPU hash fixture "+name.encode())
        final=self.old/"final.json";final.write_bytes(b"final CPU fixture")
        frames=[{"index":0,"time":0.,"source_timestamp":0.,"file":"frame0.jpg","sha256":"f"*64}]
        failed={**self.window("failed"),"bundle":str(bundle),"source_frames":frames}
        row={"id":"fixture","status":"failed","source_group":"source","split":"train","original_input":{"id":"fixture"},
            "source_identity":{"sha256":"s"*64},"model_instance_identity_sha256":"m"*64,"params_sha256":"p"*64,
            "final_path":str(final),"windows":[failed,self.window("candidate","Original good.")]}
        targets,blocked=recovery.targets_from_rows([row,{**row,"id":"good","windows":[self.window("candidate","Kept"),self.window("abstain")]}],str(self.run_output))
        self.assertEqual([t["key"] for t in targets],["fixture:w0"])
        self.assertEqual(targets[0]["max_new_observations"],1)
        self.assertEqual(blocked,[])
        self.assertEqual(targets[0]["original_refs"]["response.raw.txt"]["sha256"],recovery.batch.sha256(bundle/"response.raw.txt"))

    def test_plan_cpu_contract_keeps_immutable_selection_and_no_gpu(self):
        launchdir=self.root/"original-launch";launchdir.mkdir();launchpath=launchdir/"plan.json";launchpath.write_bytes(b"CPU launch proof fixture")
        exportdir=self.root/"original-export";exportdir.mkdir()
        (exportdir/"manifest.json").write_bytes(b"CPU verified-export proof fixture")
        (exportdir/"captions.jsonl").write_bytes(b"CPU serialized-caption fixture")
        ref=self.root/"reference.json";ref.write_bytes(b"CPU reference proof fixture")
        row={"id":"good","windows":[self.window("candidate","Original"),self.window("abstain")]}
        args=argparse.Namespace(original_plan=str(launchpath),original_export=str(exportdir),out=str(self.root/"recovery-plan"),run_out=str(self.root/"planned-run"))
        with mock.patch.object(recovery,"verify_original",return_value=({"model_reference_receipt_path":str(ref)},[row],{"jobs":[],"input_manifest_sha256":"i"*64})):
            result=recovery.plan_stage(args)
        self.assertFalse(result["GPU_run_performed"])
        self.assertEqual(result["targets"],0)
        plan=recovery.load(Path(args.out)/"plan.json")
        self.assertEqual(plan["preserved_good_candidate_windows"],1)
        self.assertEqual(plan["preserved_valid_abstain_windows"],1)
        self.assertEqual(plan["max_new_observations_per_target"],1)
        self.assertEqual(plan["run_output_path"],args.run_out)
        self.assertIs(plan["formal_training"],False)
        self.assertEqual((Path(args.out)/"targets.jsonl").read_bytes(),b"")
        with self.assertRaises(ValueError):
            with mock.patch.object(recovery,"verify_original",return_value=({"model_reference_receipt_path":str(ref)},[row],{"jobs":[],"input_manifest_sha256":"i"*64})):
                recovery.plan_stage(args)

    def test_source_error_not_faked_into_two_failed_frame_targets(self):
        row={"id":"missing","status":"failed","windows":[],"source_identity":{"sha256":"s"*64},"expected_windows":[{"start":0.,"end":2.},{"start":2.,"end":5.0625}],"failure_receipt_path":"/source-error","failure_receipt_sha256":"e"*64}
        targets,blocked=recovery.targets_from_rows([row],str(self.run_output))
        self.assertEqual(targets,[]);self.assertEqual(len(blocked),1)
        self.assertTrue(blocked[0]["no_window_or_model_response_fabricated"])
        self.assertEqual(len(blocked[0]["expected_windows"]),2)

    def test_missing_old_generation_is_blocked_not_claimed(self):
        (self.old/"generation.json").unlink()
        row={"id":"fixture","windows":[{**self.window("failed"),"bundle":str(self.old),"receipt_sha256":"r"*64}]}
        targets,blocked=recovery.targets_from_rows([row],str(self.run_output))
        self.assertEqual(targets,[])
        self.assertEqual(blocked[0]["status"],"blocked_missing_original_frame_or_generation_evidence")
        self.assertFalse((self.new/"call.claim.json").exists())

    def test_duplicate_failed_key_not_scheduled_twice(self):
        for name in ("manifest.json","request.json","response.raw.txt","generation.json","manifest.receipt.json","receipt.json"):(self.old/name).write_bytes(b"fixture")
        f=self.old/"final.json";f.write_bytes(b"fixture")
        row={"id":"fixture","source_group":"g","split":"train","original_input":{},"source_identity":{"sha256":"s"*64},"model_instance_identity_sha256":"m","params_sha256":"p","final_path":str(f),"windows":[{**self.window("failed"),"bundle":str(self.old),"source_frames":[]}]}
        with self.assertRaisesRegex(ValueError,"duplicate failed target"):
            recovery.targets_from_rows([row,row],str(self.run_output))

    def test_exclusive_durable_claim_one_call_budget(self):
        self.assertTrue(recovery.claim_call(self.new,self.target,self.params_sha))
        before=(self.new/"call.claim.json").read_bytes()
        self.assertFalse(recovery.claim_call(self.new,self.target,self.params_sha))
        self.assertEqual((self.new/"call.claim.json").read_bytes(),before)
        with self.assertRaisesRegex(ValueError,"different target/protocol"):
            recovery.claim_call(self.new,self.target,"new-protocol-sha")

    def test_crash_after_claim_never_resends_model_call(self):
        recovery.claim_call(self.new,self.target,self.params_sha)
        receipt=recovery.interrupted_claim(self.new,self.target)
        self.assertFalse(recovery.claim_call(self.new,self.target,self.params_sha))
        self.assertIsNone(receipt["model_call_performed"])
        checked=recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)
        self.assertEqual(checked["status"],"failed")
        self.assertIn("no reissue",checked["error"])
        self.assertFalse((self.new/"candidate.json").exists())

    def test_partial_raw_without_generation_stays_unknown_failure(self):
        recovery.claim_call(self.new,self.target,self.params_sha)
        raw=recovery.batch.json_bytes(self.response());(self.new/"response.raw.txt").write_bytes(raw)
        receipt=recovery.interrupted_claim(self.new,self.target)
        self.assertTrue(receipt["raw_response_available"])
        self.assertEqual(recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)["status"],"failed")
        self.assertEqual((self.new/"response.raw.txt").read_bytes(),raw)
        self.assertFalse(recovery.claim_call(self.new,self.target,self.params_sha))

    def test_completed_claim_reuses_verified_receipt_no_second_call(self):
        receipt=self.finish();before=(self.new/"response.raw.txt").read_bytes()
        self.assertFalse(recovery.claim_call(self.new,self.target,self.params_sha))
        self.assertEqual(recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)["status"],"candidate")
        with self.assertRaisesRegex(ValueError,"cannot be regenerated"):
            self.finish()
        self.assertEqual((self.new/"response.raw.txt").read_bytes(),before)
        self.assertTrue(receipt["model_call_performed"])

    def test_recover_crash_after_candidate_write_without_repeat(self):
        original=recovery.exclusive
        def interrupt(path,data):
            if Path(path).name=="receipt.json":raise OSError("fixture crash before receipt commit")
            original(path,data)
        with mock.patch.object(recovery,"exclusive",side_effect=interrupt):
            with self.assertRaises(OSError):self.finish()
        self.assertTrue((self.new/"candidate.json").exists())
        self.assertFalse((self.new/"receipt.json").exists())
        raw=(self.new/"response.raw.txt").read_bytes();g=recovery.load(self.new/"generation.json")
        recovered=recovery.write_completion(self.new,self.target,self.manifest,raw,g,self.params_sha,recovery.batch.digest(recovery.load(self.new/"schema.json")))
        self.assertEqual(recovered["status"],"candidate")
        self.assertFalse(recovery.claim_call(self.new,self.target,self.params_sha))

    def test_shape_only_schema_has_no_pair_forcing_or_uniqueItems_claim(self):
        s=recovery.shape_schema([32,36,40]);e=s["oneOf"][0]["properties"]["events"]["items"]
        self.assertEqual(e["properties"]["frame_indices"]["items"]["enum"],[32,36,40])
        self.assertEqual(e["properties"]["frame_indices"]["minItems"],2)
        self.assertNotIn("enum",e["properties"]["frame_indices"])
        self.assertNotIn("uniqueItems",json.dumps(s))
        self.assertEqual(s["oneOf"][1]["properties"]["motion_caption"],{"const":""})
        self.assertEqual(s["oneOf"][1]["properties"]["events"]["maxItems"],0)
        for bad in ([0],[0,0],[4,0],[False,4]):
            with self.assertRaises(ValueError):recovery.shape_schema(bad)

    def test_new_prompt_never_reads_old_caption_or_patches_frames(self):
        before=copy.deepcopy(self.request);new=recovery.request_for_recovery(before)
        self.assertEqual(before,self.request)
        self.assertEqual(new["messages"][0],before["messages"][0])
        self.assertEqual(new["messages"][1]["content"][:-1],before["messages"][1]["content"])
        self.assertEqual(new["messages"][1]["content"][-1],{"type":"text","text":recovery.CLARIFICATION})
        self.assertIn("hard limit remains 140",recovery.CLARIFICATION)
        self.assertIn('abstain=true, motion_caption="", events=[]',recovery.CLARIFICATION)
        self.assertNotIn("fixture old rejected caption",json.dumps(new))

    def test_nonempty_abstain_failed_raw_not_cleared(self):
        value={"motion_caption":"Static fixture pose.","events":[],"uncertainties":[],"abstain":True}
        r=self.finish(value);self.assertEqual(r["status"],"failed")
        self.assertIn("empty caption",r["error"])
        self.assertEqual(recovery.load(self.new/"response.raw.txt")["motion_caption"],value["motion_caption"])
        self.assertFalse((self.new/"candidate.json").exists())

    def test_legal_abstain_not_nonempty_delivery(self):
        value={"motion_caption":"","events":[],"uncertainties":["No support"],"abstain":True}
        self.assertEqual(self.finish(value)["status"],"abstain")
        result=recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)
        self.assertEqual(result["caption"],"")
        counts=recovery.original.caption_availability([{"windows":[result,self.window("candidate","Good")]}])
        self.assertEqual(counts["usable_full_caption_count"],0)
        self.assertFalse(counts["caption_delivery_complete"])

    def test_duplicate_indices_and_numeric_id_rejected_not_coerced(self):
        for v in ([0,0],[0],[0,999]):
            value=self.response();value["events"][0]["frame_indices"]=v
            with self.assertRaises(ValueError):recovery.batch.validate_candidate(value,self.manifest)
        value=self.response();value["events"][0]["id"]=1
        self.assertEqual(self.finish(value)["status"],"failed")
        self.assertEqual(recovery.load(self.new/"response.raw.txt")["events"][0]["id"],1)

    def test_length_eos_and_word_budget_retained(self):
        self.assertEqual(self.finish(finish="length",tokens=1200)["status"],"failed")
        self.assertEqual(recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)["status"],"failed")
        self.assertFalse((self.new/"candidate.json").exists())

    def test_over140_and_duplicate_sentence_not_cleaned(self):
        for caption in ("word "*141,"The arm rises. The arm rises."):
            v=self.response();v["motion_caption"]=caption
            with self.assertRaises(ValueError):recovery.batch.validate_candidate(v,self.manifest)
        v=self.response();v["motion_caption"]="word "*141
        self.assertEqual(self.finish(v)["status"],"failed")
        self.assertEqual(len(recovery.load(self.new/"response.raw.txt")["motion_caption"].split()),141)

    def test_new_protocol_cannot_mix_original_receipt(self):
        self.finish();r=recovery.load(self.new/"receipt.json");r["schema"]=recovery.batch.SCHEMA;self.write(self.new/"receipt.json",r)
        with self.assertRaisesRegex(ValueError,"exact target/protocol"):
            recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)

    def test_new_raw_generation_hash_or_schema_sha_tamper_rejected(self):
        self.finish()
        for name in ("response.raw.txt","schema.json","generation.json"):
            path=self.new/name;before=path.read_bytes();path.write_bytes(b"tampered CPU proof")
            with self.assertRaises((ValueError,OSError)):
                recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)
            path.write_bytes(before)
        r=recovery.load(self.new/"receipt.json");r["files_sha256"].pop("generation.json");self.write(self.new/"receipt.json",r)
        with self.assertRaisesRegex(ValueError,"lacks generation"):
            recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)

    def test_candidate_bool_int_resealed_still_rejected(self):
        self.finish();c=recovery.load(self.new/"candidate.json");c["format_valid"]=1;self.write(self.new/"candidate.json",c)
        r=recovery.load(self.new/"receipt.json");r["files_sha256"]["candidate.json"]=recovery.batch.sha256(self.new/"candidate.json");self.write(self.new/"receipt.json",r)
        with self.assertRaisesRegex(ValueError,"typed original"):
            recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)

    def test_claim_bool_int_and_attempt_two_resealed_rejected(self):
        recovery.claim_call(self.new,self.target,self.params_sha)
        c=recovery.load(self.new/"call.claim.json")
        for key,value in (("attempt_index",True),("at_most_one_new_model_call",1)):
            self.write(self.new/"call.claim.json",{**c,key:value})
            with self.assertRaisesRegex(ValueError,"different target/protocol"):
                recovery.claim_call(self.new,self.target,self.params_sha)
        self.write(self.new/"call.claim.json",c)
        receipt=recovery.interrupted_claim(self.new,self.target);receipt["attempt_index"]=2;self.write(self.new/"receipt.json",receipt)
        with self.assertRaisesRegex(ValueError,"exact target/protocol"):
            recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)

    def test_merge_keeps_old_good_and_abstain_and_raw_lineage(self):
        good=self.window("candidate","Original good.");abstain=self.window("abstain")
        rows=[{"id":"good","status":"candidate","windows":[good,abstain]},
              {"id":"fixture","status":"failed","windows":[self.window("failed"),good]}]
        target={**self.target,"id":"fixture"}
        replacement={"status":"candidate","caption":"New model phase.","events":[],"error":None,"new_protocol":recovery.SCHEMA}
        before=copy.deepcopy(rows);merged=recovery.merge_rows(rows,[target],{"fixture:w0":replacement})
        self.assertEqual(rows,before)
        self.assertEqual(merged[0]["windows"],before[0]["windows"])
        self.assertEqual(merged[1]["windows"][1],good)
        self.assertEqual(merged[1]["windows"][0]["original_window"],before[1]["windows"][0])
        counts=recovery.original.caption_availability(merged)
        self.assertEqual(counts["usable_full_caption_count"],1);self.assertFalse(counts["caption_delivery_complete"])
        self.assertIs(merged[1]["formal_training"],False)

    def test_merge_cannot_replace_good_or_valid_abstain(self):
        for status,cap in (("candidate","Original good."),("abstain","")):
            rows=[{"id":"fixture","status":status,"windows":[self.window(status,cap)]}]
            with self.assertRaisesRegex(ValueError,"cannot replace an original good"):
                recovery.merge_rows(rows,[self.target],{"fixture:w0":{"status":"candidate","caption":"Not allowed","events":[],"error":None}})

    def test_merge_missing_or_foreign_targets_pending_not_fake_slots(self):
        rows=[{"id":"fixture","status":"failed","windows":[self.window("failed")]}]
        for replacements in ({},{"another:w0":{}}):
            with self.assertRaisesRegex(ValueError,"missing/foreign"):
                recovery.merge_rows(rows,[self.target],replacements)

    def test_all_terminal_failed_ledger_not_complete_caption(self):
        rows=[{"id":"fixture","status":"failed","windows":[self.window("failed"),self.window("candidate","Good")]}]
        result={"status":"abstain","caption":"","events":[],"error":None}
        merged=recovery.merge_rows(rows,[self.target],{"fixture:w0":result})
        self.assertEqual(merged[0]["status"],"candidate")
        counts=recovery.original.caption_availability(merged)
        self.assertEqual(counts["actual_terminal_window_receipt_count"],2)
        self.assertEqual(counts["usable_full_caption_count"],0)
        self.assertFalse(counts["caption_delivery_complete"])
        self.assertEqual(counts["partial_nonempty_caption_count"],1)

    def test_output_and_reference_paths_cannot_escape(self):
        with self.assertRaisesRegex(ValueError,"overlaps"):
            recovery.safe_out(self.old/"sub",[self.old])
        alias=self.root/"alias";alias.symlink_to(self.old,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,"overlaps"):
            recovery.safe_out(alias,[self.old])
        with self.assertRaisesRegex(ValueError,"unsafe"):
            recovery.target_dir(self.run_output,{**self.target,"id":"../escape"})
        alias_output=self.root/"alias-output";alias_output.mkdir()
        (alias_output/"attempt1").symlink_to(self.old,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,"escapes"):
            recovery.target_dir(alias_output,{**self.target,"run_output_path":str(alias_output)})
        bad={**self.target,"original_bundle":str(self.old),"original_refs":{"request":{"path":"relative","sha256":"a"*64}}}
        with self.assertRaisesRegex(ValueError,"outside explicitly allowed"):
            recovery.inspect_target(bad,[self.root/"another"])

    def test_run_with_only_uncertain_previous_claim_never_loads_weights(self):
        plans=self.root/"plan-only";plans.mkdir()
        planpath=plans/"plan.json";planpath.write_bytes(b"serialized plan fixture, not full pool proof")
        refpath=self.root/"model-reference.json"
        self.write(refpath,{"model_identity":{"root":"/fixture/no-weights","sha256":recovery.MODEL_SHA}})
        out=self.root/"run-only"
        plan={"original_export":str(self.root/"original-export"),"original_launch_ref":recovery.reference(planpath),
            "model_reference_ref":recovery.reference(refpath),"blocked_scope":[],"run_output_path":str(out)}
        target={**self.target,"run_output_path":str(out)}
        target["target_sha256"]=recovery.target_digest(target)
        args=argparse.Namespace(plan=str(planpath),out=str(out),model="/fixture/no-weights")
        original_exclusive=recovery.exclusive
        def install_previous_claim(path,data):
            original_exclusive(path,data)
            if Path(path).name=="run.receipt.json":
                params=json.loads(data)["params_sha256"]
                directory=recovery.target_dir(out,target)
                recovery.claim_call(directory,target,params)
                recovery.interrupted_claim(directory,target)
        # Serialized prior claimed-but-missing response, never a fake LLM success.
        with mock.patch.object(recovery,"load_plan",return_value=(plan,[target])),mock.patch.object(recovery,"verify_original",return_value=({},[],{"jobs":[]})),mock.patch.object(recovery,"targets_from_rows",return_value=([target],[])),mock.patch.object(recovery,"inspect_target",return_value=(self.manifest,{},self.request)),mock.patch.object(recovery.batch,"model_identity",side_effect=AssertionError("weights must not be loaded")),mock.patch.object(recovery,"exclusive",side_effect=install_previous_claim):
            result=recovery.run_stage(args)
        self.assertEqual(result["counts"],{"failed":1})
        self.assertFalse((out/"load.receipt.json").exists())

    def test_original_reference_without_generation_raw_sha_is_rejected(self):
        for key in ("raw_response_sha256",):
            g={"model_call_performed":True}
            self.assertNotIn(key,g)
        self.finish();g=recovery.load(self.new/"generation.json");g.pop("raw_response_sha256")
        self.write(self.new/"generation.json",g)
        r=recovery.load(self.new/"receipt.json");r["files_sha256"]["generation.json"]=recovery.batch.sha256(self.new/"generation.json")
        self.write(self.new/"receipt.json",r)
        with self.assertRaises((ValueError,KeyError)):
            recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)

    def test_partial_recovery_export_remains_pending_without_resource_marker(self):
        plan={"original_export":str(self.root/"original-export"),"original_launch_ref":recovery.reference(self.new/"request.json"),"blocked_scope":[],"run_output_path":str(self.root/"no-run")}
        args=argparse.Namespace(plan="fixture",run_out=str(self.root/"no-run"),resource_exit=str(self.root/"no-exit"),out=str(self.root/"not-exported"))
        with mock.patch.object(recovery,"load_plan",return_value=(plan,[])),mock.patch.object(recovery,"verify_original",return_value=({},[],{"jobs":[]})),mock.patch.object(recovery,"targets_from_rows",return_value=([],[])):
            with self.assertRaises(recovery.original.Pending):recovery.export_stage(args)
        self.assertFalse(Path(args.out).exists())

    def test_single_record_prior_resealed_cannot_pass_after_valid_base(self):
        self.finish()
        self.assertEqual(recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)["status"],"candidate")
        new=recovery.load(self.new/"request.json")
        new["messages"][1]["content"].append({"type":"text","text":"Injected old caption as prior."})
        self.write(self.new/"request.json",new)
        receipt=recovery.load(self.new/"receipt.json")
        receipt["files_sha256"]["request.json"]=recovery.batch.sha256(self.new/"request.json")
        self.write(self.new/"receipt.json",receipt)
        with self.assertRaisesRegex(ValueError,"pinned original RGB messages"):
            recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)

    def test_resealed_lineage_or_schema_not_self_attestation(self):
        self.finish()
        self.assertEqual(recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)["status"],"candidate")
        original_receipt=recovery.load(self.new/"receipt.json")
        for name,change,error in (("lineage.json",lambda v:v.update(original_source_sha256="a"*64),"lineage is not anchored"),
                                  ("schema.json",lambda v:v["oneOf"][0]["properties"]["events"]["items"]["properties"]["frame_indices"].update(minItems=1),"exact shape_schema")):
            before=(self.new/name).read_bytes();value=json.loads(before);change(value);self.write(self.new/name,value)
            receipt=copy.deepcopy(original_receipt);receipt["files_sha256"][name]=recovery.batch.sha256(self.new/name);self.write(self.new/"receipt.json",receipt)
            with self.assertRaisesRegex(ValueError,error):
                recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)
            (self.new/name).write_bytes(before);self.write(self.new/"receipt.json",original_receipt)
        self.assertEqual(recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)["status"],"candidate")

    def test_same_plan_target_cannot_claim_in_second_output_tree(self):
        self.assertTrue(recovery.claim_call(self.new,self.target,self.params_sha))
        self.assertFalse(recovery.claim_call(self.new,self.target,self.params_sha))
        another=self.root/"another-run"/"attempt1"/"fixture"/"w0000"
        with self.assertRaisesRegex(ValueError,"unique planned"):
            recovery.claim_call(another,self.target,self.params_sha)
        self.assertFalse(another.exists())
        plan={"run_output_path":str(self.run_output)}
        args=argparse.Namespace(plan="fixture-plan",out=str(self.root/"another-run"),model="/fixture")
        with mock.patch.object(recovery,"load_plan",return_value=(plan,[self.target])),mock.patch.object(recovery,"verify_original",side_effect=AssertionError("no original pool read after output mismatch")):
            with self.assertRaisesRegex(ValueError,"immutable plan run output"):
                recovery.run_stage(args)

    def test_preserved_export_caption_must_equal_rebuilt_producer_proof(self):
        final=self.old/"mock-final.json";self.write(final,{"input":self.manifest["input"],"counts":{"candidate":2,"abstain":0,"failed":0,"prepared":0},"observation_complete_without_failures":True})
        row={"id":"fixture","shard":0,"status":"candidate","original_input":self.manifest["input"],
             "source_identity":self.source,"final_path":str(final),"final_sha256":recovery.batch.sha256(final),
             "windows":[self.window("candidate","Original producer caption.")],"timed_caption":"Original producer caption."}
        rebuilt={k:copy.deepcopy(row[k]) for k in ("id","status","source_identity","final_path","final_sha256","windows","timed_caption")}
        job={"out":str(self.root/"mock-job"),"shard":0,"argv":["py","--ffprobe","unused-ffprobe"]}
        self.write(recovery.original.control_path(job,8)/"run.receipt.json",{})
        launch={"jobs":[job]}
        with mock.patch.object(recovery.batch,"source_identity",return_value=self.source),mock.patch.object(recovery.batch,"probe_video",return_value={}),mock.patch.object(recovery.original,"verify_record",return_value=rebuilt):
            recovery.verify_preserved_rows([row],launch)  # Complete legal matching baseline first.
            mutated=copy.deepcopy(row);mutated["windows"][0]["caption"]="Injected caption after original export reseal."
            mutated["timed_caption"]="Injected caption after original export reseal."
            with self.assertRaisesRegex(ValueError,"differs from rebuilt producer proof"):
                recovery.verify_preserved_rows([mutated],launch)

    def test_complete_eos_stats_are_required_not_filled(self):
        self.finish()
        self.assertEqual(recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)["status"],"candidate")
        base=recovery.load(self.new/"generation.json");r=recovery.load(self.new/"receipt.json")
        for key in ("prompt_tokens","max_tokens","eos_token_id","last_output_token_id","last_output_token_is_eos","stop_reason","versions"):
            changed=copy.deepcopy(base);changed.pop(key);self.write(self.new/"generation.json",changed)
            receipt=copy.deepcopy(r);receipt["files_sha256"]["generation.json"]=recovery.batch.sha256(self.new/"generation.json");self.write(self.new/"receipt.json",receipt)
            with self.assertRaises(ValueError):recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)
        self.write(self.new/"generation.json",base);self.write(self.new/"receipt.json",r)
        self.assertEqual(recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)["status"],"candidate")

    def test_resealed_eos_boolean_or_last_token_mismatch_rejected(self):
        self.finish()
        self.assertEqual(recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)["status"],"candidate")
        base=recovery.load(self.new/"generation.json");receipt=recovery.load(self.new/"receipt.json")
        for key,value in (("last_output_token_is_eos",1),("last_output_token_id",123),("prompt_tokens",True)):
            changed=copy.deepcopy(base);changed[key]=value;self.write(self.new/"generation.json",changed)
            sealed=copy.deepcopy(receipt);sealed["files_sha256"]["generation.json"]=recovery.batch.sha256(self.new/"generation.json");self.write(self.new/"receipt.json",sealed)
            with self.assertRaises(ValueError):recovery.verify_attempt(self.new,self.target,self.manifest,self.params_sha)

    def test_fixed_new_params_preserve_typed_budget_and_unique_output(self):
        path=self.root/"fixed-plan.json";path.write_bytes(b"CPU plan reference fixture")
        plan={"run_output_path":str(self.run_output)}
        params=recovery.recovery_params(path,plan,"/fixture")
        self.assertEqual(params["run_output_path"],str(self.run_output))
        self.assertEqual(params["max_new_observations_per_target"],1)
        self.assertEqual(params["hard_words"],140)
        self.assertEqual(params["structured_backend"],"xgrammar")
        modified={**params,"seed":False}
        self.assertFalse(recovery.original.same_json(params,modified))

    def test_nonempty_production_plan_roundtrip_preserves_target_and_source_proofs(self):
        launch=self.root/"original-launch"/"launch-boundary.json";self.write(launch,{"serialized_full_boundary_fixture":True})
        original_export=self.root/"export-boundary";original_export.mkdir()
        self.write(original_export/"manifest.json",{"serialized_verified_boundary_fixture":True})
        (original_export/"captions.jsonl").write_bytes(b"original export hash fixture\n")
        model_ref=self.root/"model-ref-boundary.json";self.write(model_ref,{"not_actual_model_loaded":True})
        args=argparse.Namespace(original_plan=str(launch),original_export=str(original_export),out=str(self.root/"nonempty-plan"),run_out=str(self.run_output))
        manifest={"model_reference_receipt_path":str(model_ref)}
        layout={"jobs":[{"out":str(self.old),"shard":0}],"input_manifest_sha256":"i"*64}
        # Only the already verified original full-pool boundary is mocked.
        # Actual plan serialization, hashes, loading and all target proofs execute.
        with mock.patch.object(recovery,"verify_original",return_value=(manifest,[self.original_row],layout)):
            result=recovery.plan_stage(args)
        self.assertEqual(result["targets"],1)
        plan,targets=recovery.load_plan(Path(args.out)/"plan.json")
        self.assertEqual(len(targets),1)
        target=targets[0]
        self.assertEqual(target["target_sha256"],self.target["target_sha256"])
        self.assertEqual(recovery.target_digest(target),target["target_sha256"])
        # JSONL sorts nested keys; producer source hash must use pinned original order.
        self.assertNotEqual(list(target["original_input"]),list(self.manifest["input"]))
        request,schema,lineage=recovery.expected_attempt_artifacts(target,self.manifest,self.params_sha)
        self.assertEqual(request,recovery.request_for_recovery(self.request))
        self.assertEqual(schema,recovery.shape_schema([0,4]))
        self.assertEqual(lineage["original_source_sha256"],self.source["sha256"])
        self.assertEqual(recovery.verify_attempt(self.new,target,self.manifest,self.params_sha)["status"] if (self.new/"receipt.json").exists() else "prepared","prepared")
        self.finish()
        self.assertEqual(recovery.verify_attempt(self.new,target,self.manifest,self.params_sha)["status"],"candidate")
        mutated=copy.deepcopy(target);mutated["original_input"]["source_group"]="injected context"
        mutated["target_sha256"]=recovery.target_digest(mutated)
        with self.assertRaisesRegex(ValueError,"original manifest/input/frames"):
            recovery.expected_attempt_artifacts(mutated,self.manifest,self.params_sha)

    def test_preserved_canonical_export_reuses_producer_input_order_for_source_sha(self):
        final=self.old/"producer-final.json"
        self.write(final,{"input":self.manifest["input"],"counts":{"candidate":2,"abstain":0,"failed":0,"prepared":0},"observation_complete_without_failures":True})
        row={"id":"fixture","shard":0,"status":"candidate","original_input":self.manifest["input"],
            "source_identity":self.source,"final_path":str(final),"final_sha256":recovery.batch.sha256(final),
            "windows":[self.window("candidate","Original phase.")],"timed_caption":"Original phase."}
        serialized=json.loads(recovery.payload([row]).decode())
        self.assertNotEqual(list(serialized["original_input"]),list(self.manifest["input"]))
        rebuilt={k:copy.deepcopy(row[k]) for k in ("id","status","source_identity","final_path","final_sha256","windows","timed_caption")}
        job={"out":str(self.root/"job"),"shard":0,"argv":["py","--ffprobe","unused-fixture"]}
        self.write(recovery.original.control_path(job,8)/"run.receipt.json",{})
        def rebuild(indexed,original_input,j,run,probe,source,loaded):
            self.assertEqual(list(original_input),list(self.manifest["input"]))
            self.assertEqual(source["sha256"],self.source["sha256"])
            return rebuilt
        # Actual source hashing executes; only actual media/VLM verification is mocked.
        with mock.patch.object(recovery.batch,"probe_video",return_value={}),mock.patch.object(recovery.original,"verify_record",side_effect=rebuild):
            recovery.verify_preserved_rows([serialized],{"jobs":[job]})

    def test_target_digest_is_order_independent_but_type_and_content_sensitive(self):
        target=self.target
        roundtrip=json.loads(recovery.payload([target]).decode())
        self.assertEqual(recovery.target_digest(target),recovery.target_digest(roundtrip))
        changed=copy.deepcopy(roundtrip);changed["max_new_observations"]=True
        self.assertNotEqual(recovery.target_digest(target),recovery.target_digest(changed))
        changed=copy.deepcopy(roundtrip);changed["original_input"]["source_group"]="other"
        self.assertNotEqual(recovery.target_digest(target),recovery.target_digest(changed))

    def test_exclusive_output_never_overwrites_existing_proof(self):
        p=self.root/"proof";recovery.exclusive(p,b"first")
        with self.assertRaises(FileExistsError):recovery.exclusive(p,b"second")
        self.assertEqual(p.read_bytes(),b"first")


if __name__=="__main__":unittest.main()
