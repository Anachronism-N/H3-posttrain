"""Synthetic CPU hash/schema counterexamples; never call or impersonate a VLM.

Fixtures exercise serialized producer receipts, not actual media observations,
loadable model weights, human review, GPU completion or NLF detector outputs.
"""
import argparse
import base64
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

MS = Path(__file__).resolve().parents[1] / "scripts_omnidance/exp/ms"
sys.path.insert(0,str(MS))
import ms_visual_batch as batch
import ms_visual_export as export


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "shard"
        self.out.mkdir()
        self.model = self.root / "fixture-model-not-loadable"
        self.model.mkdir()
        (self.model/"config.json").write_text("{}")
        (self.model/"weights.bin").write_bytes(b"CPU model identity fixture; not loadable weights")
        self.identity = batch.model_identity(self.model)
        self.code = batch.code_identity()
        self.args = argparse.Namespace(model=str(self.model),tp=2,batch_size=2,max_num_seqs=2,
            max_pixels=200704,gpu_memory_utilization=.88,ffmpeg="/usr/local/bin/ffmpeg",ffprobe="/opt/ms_tools/ffprobe")
        self.params = batch.params_identity(self.args,self.code)
        sources = self.root / "source-fixtures"
        sources.mkdir()
        self.row = {"id":"fixture","source_group":"explicit-fixture-source","split":"train"}
        for key,suffix in (("video_path","video"),("audio_path","audio"),("nlf_path","nlf")):
            path = sources / suffix
            path.write_bytes(("CPU hash fixture, not actual "+suffix).encode())
            self.row[key] = str(path)
        self.row["input_line"] = 1
        self.row["input_row_sha256"] = batch.digest(self.row)
        self.source = batch.source_identity(self.row)
        times = [i/16 for i in range(81)]
        self.probe = {"times":times,"source_timestamps":times,"media_end":5.0625}
        self.job = {"out":str(self.out),"resource_queue":str(self.root/"resource"),"shard":0,"host":"fixture-no-node"}
        self.run = {"model_identity":self.identity,"params_identity":self.params}
        self.clip = self.out / "ids/fixture/fixture-protocol"
        self.clip.mkdir(parents=True)

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        batch.atomic_write(path,batch.json_bytes(value))

    def bundle(self,index=0,status="candidate",finish="stop"):
        window = batch.fixed_windows(0,5.0625,2,1.5)[index]
        bundle = self.clip / f"w{index:04d}"
        bundle.mkdir(exist_ok=True)
        selected = batch.select_frames(self.probe["times"],window["start"],window["end"])
        frames=[]
        for i in selected:
            image = bundle / f"frame-{i}.jpg"
            image.write_bytes(f"CPU frame hash fixture {i}; not a JPEG".encode())
            frames.append({"index":i,"time":i/16,"source_timestamp":i/16,"file":image.name,"sha256":batch.sha256(image)})
        content=[{"type":"text","text":batch.OBSERVE_TEXT+"Allowed Source frame indices for this window: "+json.dumps(selected)+". Use only these actual indices; never fill in missing evidence."}]
        for f in frames:
            content.extend([{"type":"text","text":f"Source frame {f['index']}, time {f['time']:.6f} seconds."},
                {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode((bundle/f["file"]).read_bytes()).decode()}}])
        request={"model":str(self.model),"messages":[{"role":"system","content":batch.SYSTEM},{"role":"user","content":content}],"temperature":0.,"max_tokens":1200}
        self.write(bundle/"request.json",request)
        binding=batch.binding_for(self.row,self.source,self.identity,self.params,window)
        manifest={"schema":batch.VISUAL_SCHEMA,"batch_schema":batch.SCHEMA,"input":self.row,"binding":binding,
            "video_path":self.row["video_path"],"video_sha256":self.source["files"]["video_path"]["sha256"],
            "frames":frames,"request_sha256":batch.sha256(bundle/"request.json"),"media_end":5.0625,
            "source_identity":self.source,"window":window}
        self.write(bundle/"manifest.json",manifest)
        self.write(bundle/"manifest.receipt.json",{"manifest_sha256":batch.sha256(bundle/"manifest.json")})
        value={"motion_caption":"" if status=="abstain" else "The arm rises, then lowers.",
            "events":[] if status=="abstain" else [{"id":"e1","description":"An arm rise followed by lowering.","frame_indices":selected[:2]}],
            "uncertainties":[],"abstain":status=="abstain"}
        raw=batch.json_bytes(value) if status!="invalid-json" else b"not JSON response fixture"
        stats={"finish_reason":finish,"completion_tokens":1200 if finish=="length" else 50,
            "max_tokens":1200,"raw_response_sha256":hashlib.sha256(raw).hexdigest(),
            "model_call_performed":True,"synthetic_cpu_fixture_not_real_inference":True,
            "prompt_tokens":100,"last_output_token_is_eos":finish=="stop"}
        self.write(bundle/"generation.json",stats)
        receipt=batch.finish_response(bundle,binding,manifest,raw)
        return bundle,binding,manifest,receipt

    def final_index(self,statuses=("candidate","candidate")):
        items=[]
        for index,status in enumerate(statuses):
            bundle,binding,manifest,receipt=self.bundle(index,status)
            items.append({"bundle":str(bundle),"binding":binding,"receipt":receipt})
        return batch.final_caption({"row":self.row,"clip":str(self.clip),"source":self.source,"windows":items,
            "media_end":5.0625,"media_end_basis":"CPU fixture, not ffprobe execution"})

    def plan_fixture(self):
        source=self.root/"all-input.jsonl"
        with source.open("w") as f:
            for i in range(12000):
                f.write(json.dumps({k:v for k,v in {**self.row,"id":f"id{i}"}.items() if k not in ("input_line","input_row_sha256")})+"\n")
        jobs=[]
        for shard in range(8):
            out=self.root/f"out{shard}"
            argv=["/existing/python",str(MS/"ms_visual_batch.py")]
            options={"--stage":"run","--manifest":str(source),"--out":str(out),"--model":str(self.model),"--ffmpeg":"/usr/local/bin/ffmpeg","--ffprobe":"/opt/ms_tools/ffprobe","--shard":str(shard),"--nshard":"8","--limit":"0","--tp":"2","--batch-size":"2","--max-num-seqs":"2","--prepare-workers":"2","--max-pixels":"200704","--gpu-memory-utilization":"0.88"}
            for k,v in options.items():
                argv.extend([k,v])
            jobs.append({"out":str(out),"resource_queue":str(self.root/f"queue{shard}"),"node":1,"host":"fixture","gpus":[0,1],"shard":shard,"expected_ids":1500,"argv":argv,"env":{"CUDA_VISIBLE_DEVICES":"0,1"}})
        plan={"schema":export.PLAN_SCHEMA,"target_ids":12000,"shards":8,"jobs":jobs,
            "input_manifest":str(source),"input_manifest_sha256":batch.sha256(source),
            "producer":str(MS/"ms_visual_batch.py"),"producer_sha256":batch.sha256(MS/"ms_visual_batch.py"),
            "weak_supervision_only":True,"formal_approval":False,"semantic_approval":False,"human_review_completed":False}
        path=self.root/"plan.json"
        self.write(path,plan)
        return path,plan

    def test_real_partition_schema_12000_union_and_duplicate_shards(self):
        path,plan=self.plan_fixture()
        parsed,rows=export.read_plan(path)
        self.assertEqual(len(rows),12000)
        self.assertEqual([r["id"] for r in rows[3::8]][:2],["id3","id11"])
        self.assertEqual(len(rows[3::8]),1500)
        plan["jobs"][1]["shard"]=0
        self.write(path,plan)
        with self.assertRaisesRegex(ValueError,"missing or duplicate"):
            export.read_plan(path)

    def test_partition_wrong_limit_changed_input_and_approval_rejected(self):
        path,plan=self.plan_fixture()
        original=json.loads(json.dumps(plan))
        argv=plan["jobs"][0]["argv"]
        argv[argv.index("--limit")+1]="50"
        self.write(path,plan)
        with self.assertRaisesRegex(ValueError,"fixed launch protocol"):
            export.read_plan(path)
        self.write(path,{**original,"formal_approval":True})
        with self.assertRaisesRegex(ValueError,"weak supervision"):
            export.read_plan(path)
        self.write(path,original)
        with Path(original["input_manifest"]).open("ab") as f:
            f.write(b"changed bytes")
        with self.assertRaises((ValueError,TypeError)):
            export.read_plan(path)

    def test_duplicate_missing_wrong_shard_index_rejected_before_source_reads(self):
        path,raw_plan=self.plan_fixture()
        plan,rows=export.read_plan(path)
        job=plan["jobs"][0]
        control=export.control_path(job,8)
        control.mkdir(parents=True)
        expected=[r["id"] for r in rows[0::8]]
        summary={"selected_ids":1500,"processed_ids":1500,"counts":{"candidate":1500,"abstain":0,"failed":0,"prepared":0}}
        self.write(control/"summary.json",summary)
        terminal={"expected_ids":1500,"summary_path":str(control/"summary.json"),"summary":summary}
        broken_orders=[expected[:-1],expected[:-1]+[expected[0]],["id1"]+expected[1:]]
        for broken in broken_orders:
            (control/"index.jsonl").write_text("".join(json.dumps({"id":sid})+"\n" for sid in broken))
            with mock.patch.object(export,"terminal_snapshot"),mock.patch.object(export,"verify_terminal",return_value=terminal),mock.patch.object(export,"verify_run",return_value=(self.run,"hash",True)),mock.patch.object(export,"verify_admission",return_value={}):
                with self.assertRaisesRegex(ValueError,"wrong-shard or reordered"):
                    export.verify_all(plan,rows,self.identity,self.code,"not-called-ffprobe")

    def test_replica_stage_incomplete_or_wrong_file_sha_rejected(self):
        job={**self.job,"node":3,"argv":["py","--model",str(self.model)]}
        stage_dir=self.root/"stages"
        plan={"_verified_plan_sha256":"plan-sha","input_manifest_sha256":"input-sha","producer_sha256":"code-sha","model_stage_receipts_directory":str(stage_dir)}
        admitted={"plan_sha256":"plan-sha","job":job,"input_sha256":"input-sha","producer_sha256":"code-sha","model_identity_sha256":self.identity["sha256"],"selected_ids":["fixture"]}
        self.write(Path(job["resource_queue"])/"admission.json",admitted)
        stage={"complete":True,"node":3,"root":str(self.model),"model_identity_sha256":self.identity["sha256"],"files":{name:{"path":row["path"],"sha256":row["sha256"],"size":row["stat"]["size"]} for name,row in self.identity["files"].items()}}
        self.write(stage_dir/"N3_v1.json",stage)
        self.assertIn("stage_receipt_sha256",export.verify_admission(job,plan,[self.row],self.identity))
        self.write(stage_dir/"N3_v1.json",{**stage,"complete":False})
        with self.assertRaisesRegex(ValueError,"not complete"):
            export.verify_admission(job,plan,[self.row],self.identity)
        stage["files"]["weights.bin"]["sha256"]="a"*64
        self.write(stage_dir/"N3_v1.json",stage)
        with self.assertRaisesRegex(ValueError,"content/path/size mismatch"):
            export.verify_admission(job,plan,[self.row],self.identity)

    def test_admission_exact_job_plan_and_selected_order_required(self):
        job={**self.job,"node":1,"argv":["py","--model",str(self.model)]}
        plan={"_verified_plan_sha256":"plan-sha","input_manifest_sha256":"input-sha","producer_sha256":"code-sha"}
        admitted={"plan_sha256":"plan-sha","job":job,"input_sha256":"input-sha","producer_sha256":"code-sha",
                  "model_identity_sha256":self.identity["sha256"],"selected_ids":["fixture"]}
        self.write(Path(job["resource_queue"])/"admission.json",admitted)
        result=export.verify_admission(job,plan,[self.row],self.identity)
        self.assertIn("admission_sha256",result)
        for key,value in (("selected_ids",["another"]),("plan_sha256","changed"),("producer_sha256","changed")):
            self.write(Path(job["resource_queue"])/"admission.json",{**admitted,key:value})
            with self.assertRaises(ValueError):
                export.verify_admission(job,plan,[self.row],self.identity)

    def test_same_basename_not_identity_but_explicit_equal_content_allowed(self):
        other=self.root/"other"/self.model.name
        other.mkdir(parents=True)
        for p in self.model.iterdir():
            (other/p.name).write_bytes(p.read_bytes())
        identity=batch.model_identity(other)
        self.assertNotEqual(identity["sha256"],self.identity["sha256"])
        self.assertEqual(export.verify_model(identity,self.identity,str(other)),batch.digest(export.content_inventory(self.identity)))
        with self.assertRaisesRegex(ValueError,"explicit model path"):
            export.verify_model(identity,self.identity,str(self.model))
        (other/"weights.bin").write_bytes(b"different hash fixture")
        with self.assertRaisesRegex(ValueError,"content inventory"):
            export.verify_model(batch.model_identity(other),self.identity,str(other))

    def test_cross_node_inode_and_mtime_do_not_change_content_identity(self):
        other=json.loads(json.dumps(self.identity))
        for file in other["files"].values():
            file["stat"]["device"] += 1
            file["stat"]["inode"] += 100
            file["stat"]["mtime_ns"] += 100
        self.assertEqual(export.content_inventory(other),export.content_inventory(self.identity))

    def test_model_inventory_digest_and_path_escape_rejected(self):
        broken=json.loads(json.dumps(self.identity))
        broken["files"]["weights.bin"]["sha256"]="a"*64
        with self.assertRaisesRegex(ValueError,"identity digest"):
            export.content_inventory(broken)
        broken=json.loads(json.dumps(self.identity))
        broken["files"]["../escape"] = broken["files"].pop("weights.bin")
        with self.assertRaisesRegex(ValueError,"unsafe"):
            export.content_inventory(broken)

    def test_pending_never_creates_export_files(self):
        plan={"jobs":[self.job],"shards":8}
        with self.assertRaises(export.Pending):
            export.terminal_snapshot(plan)
        path=self.root/"export"
        with mock.patch.object(export,"read_plan",return_value=(plan,[])),redirect_stdout(io.StringIO()):
            self.assertEqual(export.main(["--plan",str(self.root/"plan"),"--out",str(path)]),2)
        self.assertFalse(path.exists())

    def test_terminal_changed_input_and_cancelled_rejected(self):
        queue=Path(self.job["resource_queue"])
        path=queue/"worker_exit.json"
        value={"exit_code":1,"source_unchanged":True,"input_unchanged":True}
        self.write(path,value)
        self.assertEqual(export.verify_terminal(self.job,{}),value)
        for changed in ({**value,"input_unchanged":False},{**value,"source_unchanged":False},{**value,"cancelled":True},{**value,"exit_code":-9}):
            self.write(path,changed)
            with self.assertRaises(ValueError):
                export.verify_terminal(self.job,{})

    def test_real_schema_fixture_candidate_and_abstain_have_no_approval(self):
        for index,status in enumerate(("candidate","abstain")):
            bundle,binding,_,_=self.bundle(index,status)
            checked=export.verify_window(bundle,binding,self.source,self.row,self.probe,self.params,True)
            self.assertEqual(checked["status"],status)
            self.assertEqual(checked["caption"],"" if status=="abstain" else "The arm rises, then lowers.")

    def test_valid_json_length_remains_failed_no_candidate(self):
        bundle,binding,_,receipt=self.bundle(finish="length")
        self.assertEqual(receipt["status"],"failed")
        checked=export.verify_window(bundle,binding,self.source,self.row,self.probe,self.params,True)
        self.assertEqual(checked["status"],"failed")
        self.assertEqual(checked["generation"]["finish_reason"],"length")
        self.assertFalse((bundle/"candidate.json").exists())

    def test_failed_invalid_schema_can_be_counted_without_fake_facts(self):
        bundle,binding,_,_=self.bundle(status="invalid-json")
        result=export.verify_window(bundle,binding,self.source,self.row,self.probe,self.params,True)
        self.assertEqual(result["status"],"failed")
        self.assertEqual(result["caption"],"")
        self.assertEqual(result["raw_lexical_risk_flags"],{})
        self.assertEqual(export.risk_flags({"motion_caption":"raw","events":[None,{"description":"text","frame_indices":[]}]}),{})

    def test_missing_generation_cannot_resurrect_raw_as_candidate(self):
        bundle,binding,_,_=self.bundle()
        (bundle/"generation.json").unlink()
        with self.assertRaises((ValueError,OSError)):
            export.verify_window(bundle,binding,self.source,self.row,self.probe,self.params,True)

    def test_raw_frame_and_receipt_tamper_rejected(self):
        bundle,binding,_,_=self.bundle()
        for name in ("response.raw.txt","frame-0.jpg","request.json","candidate.json"):
            path=bundle/name
            original=path.read_bytes()
            path.write_bytes(b"tampered CPU fixture")
            with self.assertRaises((ValueError,OSError)):
                export.verify_window(bundle,binding,self.source,self.row,self.probe,self.params,True)
            path.write_bytes(original)

    def test_request_does_not_accept_extra_prior_text_even_rehashed(self):
        bundle,_,manifest,_=self.bundle()
        request=export.load(bundle/"request.json")
        request["messages"][1]["content"].append({"type":"text","text":"MS prior action vocabulary"})
        self.write(bundle/"request.json",request)
        with self.assertRaisesRegex(ValueError,"foreign captions/music"):
            export.verify_request(bundle,manifest,self.params)

    def test_actual_global_indices_and_pts_verified_not_just_manifest_hash(self):
        bundle,binding,_,_=self.bundle()
        probe={**self.probe,"times":[t+.01 for t in self.probe["times"]]}
        with self.assertRaisesRegex(ValueError,"actual PTS|global frame"):
            export.verify_window(bundle,binding,self.source,self.row,probe,self.params,True)

    def test_final_exact_raw_caption_and_abstain_gap(self):
        indexed=self.final_index(("candidate","abstain"))
        result=export.verify_record(indexed,self.row,self.job,self.run,self.probe,self.source,True)
        self.assertEqual(result["status"],"candidate")
        self.assertEqual(result["timed_caption"],"[0.000-2.000s] The arm rises, then lowers.")
        self.assertEqual([w["status"] for w in result["windows"]],["candidate","abstain"])
        final=export.load(indexed["final_path"])
        final["timed_caption"]+=" Filled missing fact."
        self.write(Path(indexed["final_path"]),final)
        indexed["final_sha256"]=batch.sha256(Path(indexed["final_path"]))
        with self.assertRaisesRegex(ValueError,"rewritten or gaps"):
            export.verify_record(indexed,self.row,self.job,self.run,self.probe,self.source,True)

    def test_obsolete_attempt_final_is_not_exported_as_current(self):
        indexed=self.final_index()
        (self.clip/"w0000/attempt-000001").mkdir()
        with self.assertRaisesRegex(ValueError,"obsolete retry attempt"):
            export.verify_record(indexed,self.row,self.job,self.run,self.probe,self.source,True)

    def test_wrong_sid_sourcegroup_split_and_falseapproval_rejected(self):
        indexed=self.final_index()
        final_path=Path(indexed["final_path"])
        original=export.load(final_path)
        for key,value in (("id","another"),("semantic_approval",True)):
            final={**original,key:value}
            self.write(final_path,final)
            changed={**indexed,"final_sha256":batch.sha256(final_path)}
            with self.assertRaises(ValueError):
                export.verify_record(changed,self.row,self.job,self.run,self.probe,self.source,True)
        self.write(final_path,original)
        for key,value in (("source_group","other"),("split","test")):
            with self.assertRaises(ValueError):
                export.verify_record(indexed,{**self.row,key:value},self.job,self.run,self.probe,self.source,True)

    def test_failed_source_keeps_real_error_and_no_fake_two_windows(self):
        failure={"id":self.row["id"],"status":"failed","error":"CPU fixture source processing failure",
            "input":self.row,"model_sha256":self.identity["sha256"],"params_sha256":self.params["sha256"],
            "semantic_approval":False,"human_review_completed":False}
        path=export.control_path(self.job,8)/"failures/fixture.json"
        self.write(path,failure)
        result=export.verify_record({"id":"fixture","status":"failed","error":failure["error"]},self.row,self.job,self.run,self.probe,self.source,False)
        self.assertEqual(result["windows"],[])
        self.assertEqual(len(result["expected_windows"]),2)
        self.assertTrue(result["expected_windows_have_no_terminal_receipt"])

    def test_lexical_flags_do_not_rewrite_caption_or_mark_semanticapproval(self):
        value={"motion_caption":"A person in a store waves the right hand.","events":[
            {"description":"Later arm move","frame_indices":[32,36]},
            {"description":"Earlier arm move","frame_indices":[0,4]}]}
        before=json.dumps(value)
        flags=export.risk_flags(value)
        self.assertIn("scene_or_camera_lexical",flags)
        self.assertEqual(flags["event_first_evidence_not_chronological"],[32,0])
        self.assertEqual(json.dumps(value),before)

    def test_path_escape_and_symlink_never_allowed(self):
        outside=self.root/"outside"
        outside.write_bytes(b"fixture")
        (self.out/"alias").symlink_to(outside)
        for path in (outside,self.out/"alias",self.out/"../outside"):
            with self.assertRaisesRegex(ValueError,"escapes"):
                export.beneath(path,self.out)

    def test_output_idempotence_rebuild_and_tamper(self):
        rows=[{"id":"fixture","status":"abstain","formal_training":False}]
        payload=b"".join((json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n").encode() for r in rows)
        manifest={"status":"terminal_verified_unreviewed","captions_sha256":hashlib.sha256(payload).hexdigest()}
        out=self.root/"export"
        self.assertEqual(export.output_bundle(out,rows,manifest),"created")
        self.assertEqual(export.output_bundle(out,rows,manifest),"verified_existing")
        (out/"captions.jsonl").write_bytes(b"tampered export")
        with self.assertRaisesRegex(ValueError,"SHA mismatch"):
            export.output_bundle(out,rows,manifest)

    def test_typed_false_zero_true_one_saved_manifest_never_compares_equal(self):
        self.assertFalse(export.same_json({"formal_training":False},{"formal_training":0}))
        self.assertFalse(export.same_json({"all_terminal":True},{"all_terminal":1}))
        self.assertTrue(export.same_json({"a":1,"b":False},{"b":False,"a":1}))
        rows=[{"id":"fixture","formal_training":False}]
        payload=b"".join((json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n").encode() for r in rows)
        manifest={"all_planned_workers_terminal":True,"all_expected_ids_accounted_for":True,
            "all_ids_have_two_terminal_windows":True,"caption_delivery_complete":False,
            "formal_training":False,"semantic_approval":False,"human_review_completed":False,"ground_truth":False,
            "captions_sha256":hashlib.sha256(payload).hexdigest()}
        out=self.root/"typed-output"
        export.output_bundle(out,rows,manifest)
        numeric_reseals=[(key,1 if value else 0) for key,value in manifest.items() if isinstance(value,bool)]
        self.assertEqual(len(numeric_reseals),8)
        for key,value in numeric_reseals:
            self.write(out/"manifest.json",{**manifest,key:value})
            with self.assertRaisesRegex(ValueError,"differs from rebuilt"):
                export.output_bundle(out,rows,manifest)

    def test_candidate_typed_format_flag_tamper_rejected_even_if_rehashed(self):
        bundle,binding,_,_=self.bundle()
        candidate=export.load(bundle/"candidate.json")
        candidate["format_valid"]=1
        self.write(bundle/"candidate.json",candidate)
        receipt=export.load(bundle/"receipt.json")
        receipt["files_sha256"]["candidate.json"]=batch.sha256(bundle/"candidate.json")
        self.write(bundle/"receipt.json",receipt)
        with self.assertRaisesRegex(ValueError,"typed original"):
            export.verify_window(bundle,binding,self.source,self.row,self.probe,self.params,True)

    def test_full_caption_count_excludes_candidate_abstain_and_failed_gaps(self):
        def window(status,caption=""):
            return {"status":status,"caption":caption}
        rows=[{"windows":[window("candidate","first"),window("candidate","second")]},
              {"windows":[window("candidate","first"),window("abstain")]},
              {"windows":[window("candidate","first"),window("failed")]},
              {"windows":[window("abstain"),window("abstain")]},
              {"windows":[],"expected_windows":[{"start":0,"end":2},{"start":2,"end":5.0625}]}]
        counts=export.caption_availability(rows)
        self.assertEqual(counts["usable_full_caption_count"],1)
        self.assertEqual(counts["incomplete_caption_count"],4)
        self.assertEqual(counts["empty_caption_count"],2)
        self.assertFalse(counts["caption_delivery_complete"])
        self.assertEqual(counts["clip_denominator"],5)
        self.assertEqual(counts["window_denominator"],10)
        self.assertEqual(counts["partial_nonempty_caption_count"],2)
        self.assertEqual(counts["any_nonempty_caption_count"],3)
        self.assertEqual(counts["sources_without_nonempty_caption"],2)
        self.assertEqual(counts["actual_candidate_window_count"],4)
        self.assertEqual(counts["actual_abstain_window_count"],3)
        self.assertEqual(counts["actual_failed_window_count"],1)
        self.assertEqual(counts["source_error_missing_window_count"],2)
        self.assertFalse(counts["goal_full_caption_coverage_satisfied"])
        self.assertTrue(counts["usable_caption_is_not_semantic_approval"])
        self.assertEqual(counts["max_additional_failed_attempts"],2)

    def test_third_retry_attempt_rejected_not_silently_approved(self):
        indexed=self.final_index()
        root=self.clip/"w0000"
        for i in range(1,4):
            (root/f"attempt-{i:06d}").mkdir()
        final=export.load(indexed["final_path"])
        final["segments"][0]["bundle"]=str(root/"attempt-000003")
        self.write(Path(indexed["final_path"]),final)
        indexed["final_sha256"]=batch.sha256(Path(indexed["final_path"]))
        with self.assertRaisesRegex(ValueError,"bounded two additional"):
            export.verify_record(indexed,self.row,self.job,self.run,self.probe,self.source,True)

    def test_partial_existing_export_no_complete_repair_or_overwrite(self):
        path=self.root/"partial"
        path.mkdir()
        (path/"captions.jsonl").write_bytes(b"interrupted fixture")
        with self.assertRaisesRegex(ValueError,"without complete manifest"):
            export.output_bundle(path,[],{})
        self.assertFalse((path/"manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
