"""Wait for genuine full6k completion; one N4 native AV pipeline, strict CPU audits."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
from music_direct_compare_manager_v3 import identity, snapshot, sha, HOLDER

ROOT = Path("/apdcephfs_gy4/share_302533218/cedricnie")
REPO = ROOT / "H3-posttrain"
MS = ROOT / "diff/logs_omnidance/exp/MS"
TRAIN = MS / "music_direct_token_full6k_n2_v1"
TRAIN_QUEUE = MS / "music_direct_token_full6k_n2_queue_v1"
OUT = MS / "music_direct_full6k_final2250_video_n4_v1"
QUEUE = MS / "music_direct_full6k_final2250_video_n4_queue_v1"
PROTOCOL = "direct_token_full6000_val500_epochs3_v1"
FINAL = TRAIN / "model_step02250.pt"
STATE = TRAIN / "train_state_step02250.pt"
STAGED = Path("/opt/music_direct_full6k_final2250_n4_v1.pt")
UUID = "GPU-75bfe692-7dfc-32b5-3a52-04d4533aeaeb"
IDS = ["wJPptuhclZmKAEWs", "wJPrpFsThWAcBQIw", "wJQTclrlOLEYYfTB"]
EXPECTED_HOLDER = dict(pid=3528489, start_ticks="506932287", cuda="3",
    argv=["/apdcephfs_gy2/share_303214315/cedricnie/miniconda3/envs/longlive/bin/python", HOLDER], uids=[0,0,0,0])


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def small_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def terminal_gate(record, partial=None):
    if partial is not None:
        raise ValueError("Real partial_failure is terminal; never retrain or substitute a checkpoint")
    if not isinstance(record, dict):
        raise ValueError("Training finished record is pending")
    expected = dict(actual_successful_updates=2250, actual_exposures=18000,
                    actual_unique_train=6000, validation_epochs=3, final_format4_state_tensors=467)
    if record.get("complete") is not True or record.get("protocol") != PROTOCOL:
        raise ValueError("Require exact successful full6k protocol")
    for key, value in expected.items():
        if type(record.get(key)) is not int or record[key] != value:
            raise ValueError("Incomplete or wrong-type terminal budget: " + key)
    if record.get("final_checkpoint") != str(FINAL):
        raise ValueError("Only immutable actual final model_step02250.pt is accepted")
    configuration = record.get("configuration", {})
    for key, value in dict(arm="token", steps=2250, epochs=3, train_samples=6000,
                          val_samples=500, local_batch=8, global_batch=8, seed=42).items():
        if type(configuration.get(key)) is not type(value) or configuration[key] != value:
            raise ValueError("Wrong final configuration: " + key)
    return record


def metadata_gate(metadata, tensor_count, trainable_count):
    expected = dict(format_version=4, state_scope="direct_music_to_h3_text",
                    backend="musicfm", context_policy="forbidden", freeze_backbone=False,
                    adapter="linear_1024_5120_bias", output_normalization="none_raw_hidden")
    if not isinstance(metadata, dict):
        raise ValueError("Missing full model metadata")
    for key, value in expected.items():
        if type(metadata.get(key)) is not type(value) or metadata[key] != value:
            raise ValueError("Wrong actual deployment scope: " + key)
    if type(tensor_count) is not int or tensor_count != 467 or type(trainable_count) is not int or trainable_count != 402:
        raise ValueError("Full dense MusicFM + Linear trainable/state scope incomplete")
    if metadata.get("online_training", {}).get("train_backbone") is not True:
        raise ValueError("Full dense train-backbone scope missing")


def cpu_test():
    good = dict(protocol=PROTOCOL, complete=True, actual_successful_updates=2250, actual_exposures=18000,
        actual_unique_train=6000, validation_epochs=3, final_format4_state_tensors=467, final_checkpoint=str(FINAL),
        configuration=dict(arm="token", steps=2250, epochs=3, train_samples=6000, val_samples=500,
                           local_batch=8, global_batch=8, seed=42))
    assert terminal_gate(good) == good
    rejected = []
    for key, value in [("actual_successful_updates",750),("actual_successful_updates",2249),
        ("validation_epochs",2),("complete",False),("complete",1),("actual_successful_updates",True),
        ("final_checkpoint",str(TRAIN / "model_step00500.pt"))]:
        changed = copy.deepcopy(good); changed[key] = value
        try:
            terminal_gate(changed)
        except ValueError:
            rejected.append(key + "=" + str(value))
        else:
            raise AssertionError("Terminal mutation admitted")
    try:
        terminal_gate(good, dict(complete=False, error="fixture actual failure"))
    except ValueError:
        rejected.append("partial_failure")
    else:
        raise AssertionError("Partial failure admitted")
    metadata = dict(format_version=4,state_scope="direct_music_to_h3_text",backend="musicfm",
        context_policy="forbidden",freeze_backbone=False,adapter="linear_1024_5120_bias",
        output_normalization="none_raw_hidden",online_training=dict(train_backbone=True))
    metadata_gate(metadata,467,402)
    for key, value in [("freeze_backbone",0),("freeze_backbone",True),
                       ("format_version",True),("state_scope","music_to_h3_text")]:
        changed = copy.deepcopy(metadata); changed[key] = value
        try:
            metadata_gate(changed,467,402)
        except ValueError:
            rejected.append("metadata." + key + "=" + str(value))
        else:
            raise AssertionError("Metadata mutation admitted")
    return dict(complete=True, actual_gpu_validation=False, fixture_only=True,
                strict_terminal_success_tested=True, rejected=rejected, source_sha256=sha(__file__))


def live(phase, **details):
    value = dict(complete=False, phase=phase, time=time.time(), manager_pid=os.getpid(), **details)
    atomic(QUEUE / "live.json", value)
    print(json.dumps(value), flush=True)


def wait_final():
    while True:
        partial = small_json(TRAIN / "partial_failure.json")
        finished = small_json(TRAIN / "finished.json")
        if partial is not None:
            atomic(QUEUE / "upstream_partial_failure.json", partial)
            terminal_gate(finished, partial)
        if finished is not None:
            terminal_gate(finished)
            pointer = small_json(TRAIN / "latest.json")
            training_exit = small_json(TRAIN_QUEUE / "finished.json")
            if (isinstance(pointer, dict) and type(pointer.get("steps")) is int and pointer["steps"] == 2250
                    and pointer.get("checkpoint_name") == FINAL.name and pointer.get("train_state_name") == STATE.name
                    and FINAL.is_file() and STATE.is_file()
                    and isinstance(training_exit, dict) and training_exit.get("complete") is True
                    and type(training_exit.get("exit_code")) is int and training_exit["exit_code"] == 0):
                return finished, pointer
        live("waiting_final_no_GPU_no_model_coldread", progress=small_json(TRAIN / "progress.json"),
             latest=small_json(TRAIN / "latest.json"), finished_seen=finished is not None)
        time.sleep(60)


def full_cpu_checkpoint(finished, pointer):
    import torch
    from music_text_adapter import load_direct_music_text_checkpoint
    checksum = sha(FINAL)
    if pointer.get("checkpoint_sha256") != checksum:
        raise ValueError("Final checkpoint differs from actual immutable model/state pointer")
    saved = torch.load(STATE, map_location="cpu", weights_only=True)
    if (saved.get("protocol") != PROTOCOL or saved.get("configuration") != finished["configuration"]
            or saved.get("steps") != 2250 or saved.get("remaining_updates") != 0
            or saved.get("checkpoint_name") != FINAL.name or saved.get("checkpoint_sha256") != checksum
            or len(saved.get("history", [])) != 2250 or len(saved.get("evaluations", [])) != 3
            or not isinstance(saved.get("rng"), dict) or not isinstance(saved.get("optimizer"), dict)
            or len(saved["optimizer"].get("state", {})) != 402
            or len(saved.get("batches", [])) != 2250):
        raise ValueError("Actual complete saved Adam/RNG/sampler/final pointer is incomplete")
    for epoch in range(3):
        history = saved["history"][epoch*750:(epoch+1)*750]
        if [event["ids"] for event in history] != saved["batches"][epoch*750:(epoch+1)*750]:
            raise ValueError("Actual optimizer exposure history differs from saved sampler")
        ids = [sid for event in history for sid in event["ids"]]
        if len(ids) != 6000 or len(set(ids)) != 6000:
            raise ValueError("Each actual epoch must expose 6000 unique samples")
    if set(IDS) & {sid for event in saved["history"] for sid in event["ids"]}:
        raise ValueError("Fixed3 extra generation holdouts leaked into training")
    for epoch, evaluation in enumerate(saved["evaluations"], 1):
        if evaluation.get("epoch") != epoch or evaluation.get("validation_samples") != 500:
            raise ValueError("Required actual full val500 epoch missing")
    state_report = dict(complete=True, optimizer_states=402, steps=2250, exposures=18000,
        actual_three_unique_epochs=True, full_saved_Adam_RNG_sampler_pointer=True,
        train_state=str(STATE), checkpoint_sha256=checksum, training_finished=finished)
    del saved
    model = load_direct_music_text_checkpoint(FINAL).eval()
    count = sum(p.requires_grad for p in model.parameters())
    metadata_gate(model.checkpoint_metadata,len(model.state_dict()),count)
    state_report.update(strict_loaded_complete_MusicFM=True, state_tensors=467, trainable_tensors=count,
                        metadata=model.checkpoint_metadata)
    del model
    if STAGED.exists():
        if sha(STAGED) != checksum:
            raise FileExistsError("Unfamiliar existing stage; do not overwrite")
    else:
        with FINAL.open("rb") as source, STAGED.open("xb") as target:
            for block in iter(lambda: source.read(8*1024*1024), b""):
                target.write(block)
            target.flush(); os.fsync(target.fileno())
        if sha(STAGED) != checksum:
            raise ValueError("Actual N4 local checkpoint stage SHA differs")
    state_report.update(local_stage=str(STAGED), local_stage_actual_sha256=checksum)
    atomic(QUEUE / "final_cpu_checkpoint.json",state_report)
    return checksum


def model_processes():
    processes = []
    for base in Path("/proc").iterdir():
        if not base.name.isdigit() or int(base.name) == os.getpid():
            continue
        try:
            item = identity(int(base.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        argv = item["argv"]
        scripts = [Path(arg).name for arg in argv[1:] if arg.endswith(".py")]
        if (any(name in ("sample_h3_ar.py", "h3_video_infer_chunked_lora_v1.py") or name.startswith("eval_anchor100_review_n4")
                or name.startswith("eval_purefm100_review_n4") for name in scripts)
                and "--cpu_test" not in argv):
            processes.append(item)
    return processes


def acquire():
    lease = Path("/tmp/music_direct_full6k_final2250_n4_gpu3.lock").open("a")
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        resource = snapshot(); models = model_processes()
        gpu = next(row for row in resource["gpus"] if row[0] == "3")
        if gpu[1] != UUID:
            raise RuntimeError("Fresh N4 GPU3 UUID mismatch")
        if len(models) >= 2 or min(resource["memavailable_bytes"],resource["cgroup_headroom_bytes"]) < 200*1024**3:
            live("waiting_resource_max2pipelines_180GiBplus20", resource=resource, model_processes=models)
            time.sleep(60); continue
        active = [row for row in resource["apps"] if row[1] == UUID]
        holders = []
        if active:
            if len(active) != 1:
                live("waiting_unknown_GPU3_no_signal",active=active);time.sleep(60);continue
            holder = identity(int(active[0][0]))
            if (any(holder[key] != value for key,value in EXPECTED_HOLDER.items())
                    or holder["cgroup"] != resource["cgroup"] or int(active[0][2]) >= 1200):
                live("waiting_GPU3_identity_changed_no_signal", process=holder);time.sleep(60);continue
            holders.append(holder)
        reserve = Path("/tmp/od_reserve_3")
        if reserve.exists() and (reserve.is_symlink() or not reserve.is_file() or reserve.stat().st_uid != 0 or reserve.stat().st_size != 0):
            raise RuntimeError("Unfamiliar GPU3 reserve; do not touch")
        reserve.touch()
        for holder in holders:
            fd = os.pidfd_open(holder["pid"])
            try:
                if identity(holder["pid"]) != holder:
                    raise RuntimeError("Holder changed after pidfd_open; no signal")
                atomic(QUEUE / "holder_release.json",dict(snapshot=resource,exact_holder=holder,signal="pidfd_SIGTERM"))
                signal.pidfd_send_signal(fd,signal.SIGTERM)
                if not select.select([fd],[],[],30)[0]:
                    raise RuntimeError("Exact holder did not exit; no escalation")
            finally:
                os.close(fd)
        fresh = snapshot()
        if any(row[1] == UUID for row in fresh["apps"]):
            raise RuntimeError("GPU3 acquired an unexpected process; no signal")
        if len(model_processes()) >= 2:
            raise RuntimeError("Another pipeline appeared after admission; do not start third")
        atomic(QUEUE / "fresh_resource.json",fresh)
        return lease


def validate_case(case, checksum):
    from music_direct_video_manager_n4_v1 import environment
    output = OUT / "joint" / (case["sample_id"] + "_" + Path(case["audio_path"]).stem + ".mp4")
    record = small_json(output.with_suffix(".audit.json"))
    if record is None:
        return None
    inputs,audit = record["input"],record["audit"]
    if record.get("complete") is not True or inputs["case"] != case or inputs["checkpoint_sha256"] != checksum:
        raise ValueError("Actual case/checkpoint audit mismatch")
    assert audit["scheduler_updates"] == dict(video=50,audio=50)
    for key in ["full_cached_prefix_preserved","original_cached_prefix_tags_preserved",
                "packed_text_tags_exact","native_audio_vae_unchanged","original_joint_pipeline"]:
        assert audit[key] is True, key
    assert audit["native_audio_condition"] == "real_reference_audio" and audit["music_mode"] == "ref"
    assert audit["reference"]["exists"] and audit["visual_reference"]["exists"]
    assert audit["text_encoder_calls"] == 0 and audit["semantic_text"] is None
    assert audit["prompt_embeds_shape"] == [case["prefix_length"]+254,5120]
    assert inputs["student_reads_target_or_caption"] is False and inputs["teacher_identity"] is False
    assert inputs["student_zero_input"] is False and inputs["output_audio"] == "original_mux"
    assert record["mux_audio_sha256"] == inputs["native_waveform_sha256"]
    assert record["video_sha256"] == sha(output)
    source_identity = {}
    for kind in ["image","audio","prefix"]:
        path = case[kind+"_path"]; actual = sha(path)
        assert actual == case["prefix_file_sha256" if kind == "prefix" else kind+"_sha256"], kind + " source identity"
        source_identity[kind] = dict(path=path,sha256=actual)
    code = 'import sys,json;sys.path.insert(0,sys.argv[1]);from music_generation_validate import probe_video;print(json.dumps(probe_video(__import__("pathlib").Path(sys.argv[2]))))'
    env=environment();env["CUDA_VISIBLE_DEVICES"]=""
    result = subprocess.run(["/opt/venv_eval/bin/python","-B","-c",code,str(Path(__file__).parent),str(output)],
        env=env,text=True,capture_output=True,timeout=600)
    if result.returncode:
        raise RuntimeError("Independent CPU all-frame/audio decode failed, not training failure: " + result.stderr[-6000:])
    probe = json.loads(result.stdout.strip().splitlines()[-1])
    checked = dict(sample_id=case["sample_id"],path=str(output),sha256=record["video_sha256"],
        audit_path=str(output.with_suffix(".audit.json")),source_identity=source_identity,
        full_media_decode_checked=True,human_quality_pending=True,**probe)
    atomic(OUT / (case["sample_id"] + ".validated.json"),checked)
    return checked


def wait_review_http(records):
    import gzip
    import urllib.error
    import urllib.parse
    import urllib.request
    model_id = "MS/music_direct_full6k_final2250_video_n4_v1"
    base = "http://28.216.19.69:8775"
    while True:
        try:
            with urllib.request.urlopen(base + "/manifest",timeout=30) as response:
                status = response.status; data = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
            manifest = json.loads(data)
            models = {item["id"]:item for item in manifest["models"]}
            model = models.get(model_id)
            if status != 200 or model is None or set(model.get("conds",{}).get("joint",{})) != set(IDS):
                live("waiting_unified_publisher_final3_registration",model_id=model_id);time.sleep(60);continue
            assert "MS/music_direct_video_compare_n4_v1/mean" in models
            assert "MS/music_direct_video_compare_n4_v1/token" in models
            assert any(model_id in preset.get("models",[]) and "2250" in preset["name"] for preset in manifest["presets"])
            ranges=[]
            for record in records:
                path=Path(record["path"])
                assert model["conds"]["joint"][record["sample_id"]] == [path.name]
                request=urllib.request.Request(base + "/media?" + urllib.parse.urlencode({"p":str(path)}),
                                              headers={"Range":"bytes=0-1023"})
                with urllib.request.urlopen(request,timeout=30) as response:
                    chunk=response.read(); headers=dict(response.headers); code=response.status
                with path.open("rb") as source:
                    expected=source.read(1024)
                assert code == 206 and chunk == expected and headers.get("Content-Range") == "bytes 0-1023/" + str(path.stat().st_size)
                ranges.append(dict(sample_id=record["sample_id"],status=code,content_range=headers["Content-Range"],bytes_verified=1024))
            checked=dict(complete=True,manifest_status=200,model_id=model_id,video_range_checks=ranges,
                         historical_mean_token_preserved=True,full3_registered=True,human_quality_pending=True)
            atomic(OUT / "review_http.json",checked)
            atomic(QUEUE / "finished.json",dict(complete=True,actual_final_video_count=3,review_http=checked,
                 report=str(OUT / "report.json"),actual_training_updates=2250,human_quality_pending=True))
            live("technical_complete_final3_and_review_HTTP_verified",actual_videos=3,review_http=checked)
            return
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            live("waiting_review_HTTP_retry",error=repr(error));time.sleep(60)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--cpu_test",action="store_true")
    parser.add_argument("--manager_sha");args=parser.parse_args()
    if args.cpu_test:
        print(json.dumps(cpu_test()),flush=True);return
    if os.getuid() != 0 or sha(__file__) != args.manager_sha:
        raise RuntimeError("Require UID0 and exact accepted independent manager source")
    manager_lock=Path("/tmp/music_direct_full6k_final2250_n4_manager.lock").open("a")
    fcntl.flock(manager_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if QUEUE.exists() or OUT.exists():
        raise FileExistsError("Existing same-task output/queue must be inspected, never repeated or overwritten")
    QUEUE.mkdir();atomic(QUEUE / "cpu_gate_tests.json",cpu_test())
    atomic(QUEUE / "plan.json",dict(manager_pid=os.getpid(),identity=identity(os.getpid()),
        manager_sha256=args.manager_sha,train=str(TRAIN),actual_final_only=str(FINAL),node="N4",gpu=3,gpu_uuid=UUID,
        no_retraining=True,no_GPU_until_actual_terminal=True,no_waiting_model_coldread=True,
        expected=3,case_ids=IDS,expected_holder=EXPECTED_HOLDER,out=str(OUT),
        lifecycle="managed_Bash_SSH_foreground",human_quality_pending=True))
    try:
        finished,pointer=wait_final();live("actual_final_CPU_strict_load")
        checksum=full_cpu_checkpoint(finished,pointer)
        import music_direct_video_manager_n4_v1 as previous
        previous.OUT=OUT;previous.ARMS=[("final2250",3,UUID,STAGED,checksum)]
        preflight,manifest=previous.preflight()
        command=previous.command("final2250",STAGED);command[command.index("--out")+1]=str(OUT)
        preflight.update(manager_source_sha256=args.manager_sha,actual_final_checkpoint=str(FINAL),expected_videos=3,
                         source_people=3,command=command)
        OUT.mkdir();(OUT / "joint").mkdir();(OUT / "logs").mkdir()
        atomic(OUT / "cpu_preflight.json",preflight)
        lease=acquire();live("starting_actual_final_native_AV",command=command)
        records=[];seen=set()
        with (OUT / "logs/generation.log").open("x") as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=previous.environment(3))
            atomic(QUEUE / "child.json",dict(pid=process.pid,identity=identity(process.pid),command=command))
            while True:
                for case in manifest["cases"]:
                    if case["sample_id"] not in seen:
                        checked=validate_case(case,checksum)
                        if checked:
                            records.append(checked);seen.add(case["sample_id"])
                            atomic(OUT / "partial_validated.json",dict(complete=False,expected=3,cases=records,
                                full_media_decode_checked=True,human_quality_pending=True))
                            live("partial_video_CPU_verified",actual_videos=len(records),validated=str(OUT / "partial_validated.json"))
                if process.poll() is not None:
                    break
                Path("/tmp/od_reserve_3").touch();time.sleep(20)
        atomic(QUEUE / "generation_exit.json",dict(exit_code=process.returncode))
        if process.returncode != 0:
            raise RuntimeError("Final native sampler failed; preserve all partial logs, no model substitution")
        text=(OUT / "logs/generation.log").read_text()
        assert "[h3ar] done" in text and text.count("[direct native saved] ") == 3
        assert "Traceback (most recent call last)" not in text and len(records) == 3
        assert sha(previous.JOINT) == preflight["manifest_sha256"]
        assert all(sha(path) == value for path,value in preflight["source_files"].items())
        atomic(OUT / "report.json",dict(complete=True,expected=3,cases=records,actual_final_updates=2250,
            actual_final_checkpoint=str(FINAL),checkpoint_sha256=checksum,full_media_decode_checked=True,
            human_quality_pending=True,musical_quality_accepted=False,
            native_audio="real_reference_audio_original_joint_50_video_and_audio_steps_original_mux",
            preflight_sha256=sha(OUT / "cpu_preflight.json")))
        live("three_actual_final_videos_verified_review_registration_pending",report=str(OUT / "report.json"))
        wait_review_http(records)
    except BaseException as error:
        atomic(QUEUE / "partial_failure.json",dict(complete=False,error=repr(error),time=time.time(),
            upstream_partial_failure=small_json(TRAIN / "partial_failure.json"),no_retrain=True,outputs_preserved=True))
        raise


if __name__ == "__main__":
    main()
