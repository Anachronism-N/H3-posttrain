"""Independent fixed-three holdout native generation; never add these IDs to a split."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import struct
import sys
import time
import wave

sys.dont_write_bytecode = True
ROOT = Path("/apdcephfs_gy4/share_302533218/cedricnie")
REPO = ROOT / "H3-posttrain"
MS = ROOT / "diff/logs_omnidance/exp/MS"
EXP = MS / "music_planner_fixed2s_training_20261024_v1"
DATA = MS / "music_planner_fixed2s_baseline_20261025_clean_v1"
HANDOFF = EXP / "eval_handoff_clean_v10"
FROZEN = EXP / "runtime/source_v10"
IDS = ["wJPptuhclZmKAEWs", "wJPrpFsThWAcBQIw", "wJQTclrlOLEYYfTB"]
FAMILIES = ("moss", "music_flamingo", "omni_captioner")
METADATA = MS / "omni_h3_ms_style_eval3_v1/omni_infer3.jsonl"
IMAGES = MS / "omni_h3_ms_style_eval3_v1/h3_gt_caption3.json"
H3 = REPO / "scripts_omnidance/exp/ms/h3_video_infer.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_eval():
    return load_module("fixed3_frozen_eval", FROZEN / "music_planner_eval.py")


def new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def verify_ref(ref, ev):
    actual = ev.file_ref(ref["path"])
    if actual != ref:
        raise ValueError("Frozen input content/size/path changed: " + ref["path"])
    return actual


def media_evidence(audio, image, ev):
    with wave.open(str(audio), "rb") as stream:
        channels, rate, frames = stream.getnchannels(), stream.getframerate(), stream.getnframes()
        if channels != 1 or abs(frames / rate - 5.0625) > 1 / rate:
            raise ValueError("Fixed3 needs original mono, complete 5.0625-second audio")
        if len(stream.readframes(frames)) != frames * channels * stream.getsampwidth():
            raise ValueError("Truncated WAV payload")
    with Path(image).open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("Historical reference is not a PNG with IHDR")
    width, height = struct.unpack(">II", header[16:24])
    if (width, height) != (704, 1280):
        raise ValueError("Historical reference geometry differs")
    return dict(audio_ref=ev.file_ref(audio), image_ref=ev.file_ref(image),
                audio_header=dict(channels=channels, sample_rate=rate, frames=frames, duration_seconds=frames/rate),
                image_header=dict(width=width, height=height), image_full_decode_verified=False)


def holdout_rows(metadata, image_records, splits, ev):
    if [row["id"] for row in metadata] != IDS or [row["id"] for row in image_records] != IDS:
        raise ValueError("Historical fixed3 membership/order differs")
    all_rows = [row for rows in splits.values() for row in rows]
    source_ids = {row["id"] for row in all_rows}
    audio_shas = {row["audio_sha256"] for row in all_rows}
    rows = []
    for meta, image in zip(metadata, image_records):
        if meta["id"] in source_ids or meta.get("audio_window_seconds") != [0, 5.0625]:
            raise ValueError("Fixed3 not an independent full-audio holdout")
        if not isinstance(meta.get("audios"), list) or len(meta["audios"]) != 1:
            raise ValueError("Historical metadata must bind one explicit audio")
        evidence = media_evidence(meta["audios"][0], image["image_path"], ev)
        if evidence["audio_ref"]["sha256"] in audio_shas:
            raise ValueError("Fixed3 exact audio overlaps current train/val/test")
        rows.append(dict(id=meta["id"], audio_path=evidence["audio_ref"]["path"],
                         audio_sha256=evidence["audio_ref"]["sha256"], audio_window_seconds=[0.0, 5.0625],
                         image_path=image["image_path"], historical_source_group=meta.get("source_group"),
                         historical_split_label=meta.get("split"), **evidence))
    if len({row["audio_sha256"] for row in rows}) != 3:
        raise ValueError("Fixed3 exact audio duplicates")
    return rows


def prepare_protocol():
    if HANDOFF.exists():
        raise FileExistsError("Protocol output must be fresh: " + str(HANDOFF))
    ev = frozen_eval()
    templates = EXP / "runtime/training_templates_v10.json"
    bindings = json.loads(templates.read_text())["source_sha256"]
    code_paths = [FROZEN / ("music_planner_" + name + ".py") for name in ("train", "adapters", "eval")]
    code_paths.append(EXP / "runtime/stage_models.py")
    for path in code_paths:
        if ev.file_ref(path)["sha256"] != bindings[str(path)]:
            raise ValueError("Source differs from frozen cleaned training binding")
    splits, summary = ev.validate_data(DATA)
    metadata = [json.loads(line) for line in METADATA.read_text().splitlines() if line.strip()]
    rows = holdout_rows(metadata, json.loads(IMAGES.read_text())["records"], splits, ev)
    val = ev.select_rows(splits["val"], "val", limit=32, seed=42)
    mapping = [{"id": source["id"], "condition": condition, "audio_source_id": donor["id"],
                "source_group_id": source["source_group_id"], "audio_source_group_id": donor["source_group_id"],
                "audio_sha256": donor["audio_sha256"]}
               for source, condition, donor in ev.make_conditions(val, counterfactual=True)]
    protocol = dict(schema="music_planner_fixed3_protocol_v1",
        fixed3_role="independent_generation_holdout_not_current_val_or_test", fixed3=rows,
        fixed3_source_id_and_exact_audio_exclusion_verified=True,
        fixed3_original_video_group_and_song_near_duplicate_isolation_verified=False,
        isolation_scope="Current split IDs and exact audio SHA only; historical group labels are not source-group proof",
        historical_metadata_use="Only explicit id/audio/window/image binding; no historical message or caption input/output",
        historical_refs=[ev.file_ref(METADATA), ev.file_ref(IMAGES)],
        code=[ev.file_ref(path) for path in code_paths + [Path(__file__), H3,
              REPO / "scripts_omnidance/exp/ms/h3_ms_style_train_eval3_followup_v1.py",
              REPO / "notes/29_autonomous_experiment_completion_20261024.md"]],
        training_templates=ev.file_ref(templates),
        data=dict(path=str(DATA), summary=ev.file_ref(DATA / "summary.json"),
                  splits={split: ev.file_ref(DATA / (split + ".jsonl")) for split in splits},
                  preparation_access="Metadata/hash/isolation inspection only, no test generation or likelihood evaluation"),
        val32=dict(seed=42, ordering="SHA256(UTF8(42:<id>)) ascending", ids=[r["id"] for r in val],
                   donor_mapping=mapping, donor_policy="Frozen eval cyclic first different source group and exact audio SHA",
                   song_near_duplicate_isolation_verified=summary["song_and_near_duplicate_isolation_verified"]),
        decoding=dict(seed=42, do_sample=False, max_new_tokens=2048, native_training_prefix_unchanged=True,
                      extra_instruction=False, raw_output_cleaning=False, GT_length_truncation=False, added_penalties=False),
        test=dict(final_evaluations_per_family=1, limit=500, seed=42, selection="frozen eval SHA ordering",
                  requires_all_three_val_best_checkpoint_identities_frozen_before_first_test=True,
                  val_best_checkpoint_freeze_status="pending", consumed_for_model_selection=False,
                  enforcement="Protocol requirement only; existing eval CLI does not enforce a global once-only test ledger"),
        h3=dict(width=704, height=1280, num_frames=124, fps=24, steps=50, seed=42, cfg=1.0, flow_shift=12.0,
                audio_input=False, audio_output=False, original_reference_images=True,
                renderer="frozen pure-video base, no LoRA", new_condition_cache_required=True,
                source=ev.file_ref(H3), raw_caption_passthrough=True),
        approved_for_gpu=False, pretrained_GPU_generation_executed=False, human_quality_accepted=False)
    HANDOFF.mkdir(exist_ok=False)
    new_json(HANDOFF / "val32_selection.json", protocol["val32"]["ids"])
    protocol["val32"]["selection_ref"] = ev.file_ref(HANDOFF / "val32_selection.json")
    new_json(HANDOFF / "protocol.json", protocol)
    print(json.dumps(dict(protocol=str(HANDOFF / "protocol.json"), fixed3_ids=IDS, approved_for_gpu=False)))


def checkpoint_content_identity(path, identity, ev):
    if Path(path).is_symlink():
        raise ValueError("Checkpoint root symlink forbidden")
    path = Path(path).resolve()
    marker = json.loads((path / "trainer_state.json").read_text())
    refs = marker.get("files")
    if not isinstance(refs, list) or not refs:
        raise ValueError("Whole checkpoint file content manifest missing")
    expected = {}
    for ref in refs:
        item = Path(ref["path"])
        if not item.is_absolute() or not item.is_relative_to(path) or item == path / "trainer_state.json":
            raise ValueError("Checkpoint manifest path is outside this checkpoint")
        if str(item) in expected:
            raise ValueError("Duplicate checkpoint manifest path")
        expected[str(item)] = ref
    inventory = list(path.rglob("*"))
    if any(item.is_symlink() for item in inventory) or path.is_symlink():
        raise ValueError("Checkpoint symlinks forbidden")
    actual = {str(item) for item in inventory if item.is_file() and item != path / "trainer_state.json"}
    late_sidecars = {str(path / name) for name in ("validation.json", "val_generation_request.json")}
    additional = actual - set(expected)
    if set(expected) - actual or additional - late_sidecars:
        raise ValueError("Checkpoint inventory differs beyond known late-written sidecars")
    sidecar_refs = [ev.file_ref(name) for name in sorted(additional)]
    world = marker.get("world_size")
    if type(world) is not int or world < 1:
        raise ValueError("Checkpoint world size missing/invalid")
    required = {str(path / name) for name in ("trainable.pt", "scheduler.pt", "adapter/adapter_config.json",
                                             "adapter/adapter_model.safetensors", "adapter/planner_adapter.json")}
    required |= {str(path / f"{kind}-rank{rank}.{suffix}") for rank in range(world)
                 for kind, suffix in (("rng", "pt"), ("writer", "json"))}
    if not required.issubset(expected):
        raise ValueError("Completed checkpoint lacks portable adapter/scheduler/all-rank RNG/writer")
    if marker.get("zero3") is True:
        for kind in ("optim", "model"):
            shards = {name for name in expected if "/distributed/" in name and name.endswith(kind + "_states.pt")}
            required_shards = {str(path / f"distributed/zero3/zero_pp_rank_{rank}_mp_rank_00_{kind}_states.pt")
                               for rank in range(world)}
            if shards != required_shards:
                raise ValueError("Completed ZeRO3 checkpoint has wrong exact rank set: " + kind)
    elif not {str(path / f"optimizer-rank{rank}.pt") for rank in range(world)}.issubset(expected):
        raise ValueError("Completed checkpoint lacks all-rank optimizer files")
    verified = []
    for name, ref in sorted(expected.items()):
        # checkpoint_identity already hashed trainable.pt; avoid a second content pass.
        actual_ref = identity["trainable"] if name == identity["trainable"]["path"] else ev.file_ref(name)
        if actual_ref != ref:
            raise ValueError("Checkpoint file content changed: " + name)
        verified.append(actual_ref)
    verify_ref(identity["trainer_state"], ev)
    return dict(files=verified, whole_manifest_inventory_and_content_SHA_verified=True,
                late_written_sidecars=sidecar_refs, late_sidecars_covered_by_training_marker=False,
                weight_content_hash_passes_this_invocation=1,
                optimizer_tensor_semantics_or_new_process_training_resume_verified=False,
                scope="All marker-listed checkpoint bytes, exact inventory plus separately hashed known late sidecars; not Adam finite/moment/resume semantics")


def checkpoint_source_identity(checkpoint_ref, protocol):
    config = checkpoint_ref["training_config"]
    refs = {ref["path"]: ref["sha256"] for ref in protocol["code"]}
    required = {"trainer_sha256": str(FROZEN / "music_planner_train.py"),
                "adapter_source_sha256": str(FROZEN / "music_planner_adapters.py")}
    for key, path in required.items():
        if config.get(key) != refs[path]:
            raise ValueError("Checkpoint training source differs from fixed3 frozen protocol: " + key)
    return {key: config[key] for key in required}


def checkpoint_selection_identity(path, expected_sha, expected_marker_sha, checkpoint_ref, family, ev):
    if checkpoint_ref["trainer_state"]["sha256"] != expected_marker_sha:
        raise ValueError("Selected checkpoint marker differs from independently frozen marker SHA")
    receipt_ref = ev.file_ref(path)
    if receipt_ref["sha256"] != expected_sha:
        raise ValueError("Checkpoint selection receipt SHA changed")
    receipt = json.loads(Path(path).read_text())
    checkpoint = Path(checkpoint_ref["trainer_state"]["path"]).parent
    if isinstance(receipt.get("checkpoint"), dict):
        if (receipt.get("schema") != "music-planner-outputs-v1" or receipt.get("family") != family
                or receipt.get("checkpoint_complete") is not True
                or receipt.get("test_selection_performed") is not False
                or receipt.get("fingerprint") != checkpoint_ref["fingerprint"]
                or receipt.get("cursor") != checkpoint_ref["cursor"]
                or receipt["checkpoint"] != checkpoint_ref["trainer_state"]
                or receipt.get("run_config") != checkpoint_ref["run_config"]):
            raise ValueError("Training outputs selection does not bind this completed checkpoint")
        policy = "Explicit actual training-output checkpoint, not automatically a val-best choice"
    elif isinstance(receipt.get("checkpoint"), str):
        validation_path = checkpoint / "validation.json"
        if Path(path).resolve() != checkpoint.parent / "best.json" or Path(receipt["checkpoint"]).resolve() != checkpoint:
            raise ValueError("Val-best selection is not this checkpoint parent best.json")
        validation = json.loads(validation_path.read_text())
        if (validation.get("test_used") is not False or receipt.get("test_used") is not False
                or validation.get("comparison_scope") != "within-family-tokenizer-only"
                or receipt.get("comparison_scope") != validation.get("comparison_scope")
                or any(receipt.get(key) != validation.get(key) for key in ("nll", "assistant_tokens", "source_count"))
                or validation.get("source_count") != checkpoint_ref["training_config"]["data"]["val"]["sources"]):
            raise ValueError("Val-best selection differs from actual checkpoint validation")
        policy = "Actual val-best pointer and matching validation, bound to independently frozen checkpoint marker SHA"
    else:
        raise ValueError("Selection must be actual training outputs or parent val-best receipt")
    return dict(receipt=receipt_ref, checkpoint_marker=checkpoint_ref["trainer_state"],
                independently_frozen_checkpoint_marker_sha256=expected_marker_sha, policy=policy,
                validation=ev.file_ref(checkpoint / "validation.json") if (checkpoint / "validation.json").is_file() else None)


def base_stage_identity(model, family, checkpoint_ref, ev):
    trained = checkpoint_ref["training_config"]["model"]
    base = ev.verify_base_identity(model, trained)
    stage = load_module("fixed3_stage", EXP / "runtime/stage_models.py")
    stage_ref = ev.file_ref(Path(str(Path(model).resolve()) + ".stage_receipt.json"))
    if trained.get("weights_manifest") != {"path": stage_ref["path"], "sha256": stage_ref["sha256"]}:
        raise ValueError("Official local stage receipt differs from checkpoint training")
    marker = json.loads(Path(stage_ref["path"]).read_text())
    _, digest, _, _, files = stage.read_download(
        {"path": marker["download_receipt"], "sha256": marker["download_receipt_sha256"]}, family)
    stage.check_staged(marker, digest, family, files, Path(model).resolve())
    if marker.get("full_source_and_local_sha_verified") is not True:
        raise ValueError("Official stage receipt lacks prior full source/local content verification")
    base["official_stage"] = dict(receipt=stage_ref, official_download=ev.file_ref(marker["download_receipt"]),
        revision=marker["revision"], prior_copy_source_and_local_SHA_verified=True,
        current_inventory_and_file_stat_and_metadata_SHA_verified=True,
        full_weight_content_SHA_reverified_this_invocation=False,
        limitation="Stat-only weight check can miss a same-size rewrite with unchanged timestamps; not a current full content audit",
        verifier_code=ev.file_ref(stage.__file__))
    return base


def native_record(adapter, row, ev):
    record = dict(id=row["id"], audio_ref=row["audio_ref"], image_ref=row["image_ref"], failure=None,
                  generation_budget=2048, raw_text="")
    start = time.monotonic()
    try:
        verify_ref(row["audio_ref"], ev)
        verify_ref(row["image_ref"], ev)
        result = adapter.generate(ev.audio_only_input(row, row), max_new_tokens=2048, silence=False, seed=42)
        record["native_output"] = result
        record["raw_text"] = result["raw_text"]
        record["new_token_ids"] = result["new_token_ids"]
        record["eos_token_ids"] = result["eos_token_ids"]
        if not isinstance(record["raw_text"], str) or not record["raw_text"].strip():
            raise ValueError("Native raw output empty/not text")
        if any(type(token) is not int for token in result["new_token_ids"] + result["eos_token_ids"]):
            raise ValueError("Native token IDs invalid")
        record["stop_reason"] = ev.classify_stop(result["new_token_ids"], result["eos_token_ids"], 2048,
                                                  result.get("native_stop", False))
        record["generated_tokens"] = len(result["new_token_ids"])
        record["budget_truncation"] = record["stop_reason"] == "budget_truncation"
        record["ended_with_eos"] = record["stop_reason"] == "eos"
        if record["stop_reason"] not in ("eos", "native_stop"):
            raise ValueError("Native generation incomplete: " + record["stop_reason"])
    except Exception as exc:
        record["failure"] = dict(type=type(exc).__name__, message=str(exc))
    record["elapsed_seconds"] = time.monotonic() - start
    return record


def generate(args):
    if not re.fullmatch(r"cuda(?::[0-9]+)?", args.device):
        raise ValueError("An explicit CUDA device is required; no CPU/auto fallback")
    if args.out.exists() or not args.out.resolve().is_relative_to(HANDOFF.resolve()):
        raise ValueError("Fresh output must be inside eval_handoff_v1")
    ev = frozen_eval()
    protocol_ref = ev.file_ref(HANDOFF / "protocol.json")
    protocol = json.loads(Path(protocol_ref["path"]).read_text())
    if protocol["schema"] != "music_planner_fixed3_protocol_v1" or [r["id"] for r in protocol["fixed3"]] != IDS:
        raise ValueError("Wrong fixed3 protocol")
    for ref in protocol["code"] + protocol["historical_refs"] + list(protocol["data"]["splits"].values()):
        verify_ref(ref, ev)
    verify_ref(protocol["data"]["summary"], ev)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
        raise RuntimeError("Requested CUDA device is unavailable; no CPU fallback")
    checkpoint_ref = ev.checkpoint_identity(args.checkpoint, args.family, DATA)
    checkpoint_sources = checkpoint_source_identity(checkpoint_ref, protocol)
    selection_ref = checkpoint_selection_identity(args.checkpoint_selection, args.checkpoint_selection_sha,
                                                  args.checkpoint_marker_sha, checkpoint_ref, args.family, ev)
    full_checkpoint = checkpoint_content_identity(args.checkpoint, checkpoint_ref, ev)
    metadata = json.loads((args.checkpoint / "adapter/planner_adapter.json").read_text())
    if (metadata.get("family") != args.family or type(metadata.get("max_length")) is not int
            or metadata["max_length"] != checkpoint_ref["training_config"]["max_length"]):
        raise ValueError("Portable adapter family/context differs from training configuration")
    base_ref = base_stage_identity(args.model, args.family, checkpoint_ref, ev)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    sys.path.insert(0, str(FROZEN))
    import music_planner_adapters as adapters
    if Path(adapters.__file__).resolve() != FROZEN / "music_planner_adapters.py":
        raise ValueError("Native adapter import is not frozen source_v7")
    args.out.mkdir(exist_ok=False)
    manifest = dict(schema="music_planner_fixed3_native_eval_v1", family=args.family, protocol=protocol_ref,
        checkpoint_identity=checkpoint_ref, checkpoint_content=full_checkpoint, checkpoint_selection=selection_ref,
        checkpoint_source_identity=checkpoint_sources, base_identity=base_ref,
        source_ids=IDS, device=args.device, decoding=protocol["decoding"], code=ev.file_ref(__file__),
        mode="independent_fixed3_holdout", test_used=False,
        complete=False, h3_handoff_approved=False, pretrained_GPU_generation_executed=False,
        human_quality_accepted=False, approved_for_gpu=False)
    manifest_path = args.out / "manifest.json"
    new_json(manifest_path, manifest)
    try:
        adapter = adapters.load_planner_adapter(args.family, args.model, checkpoint=args.checkpoint,
            device=args.device, max_length=checkpoint_ref["training_config"]["max_length"])
        if any(p.device != device for p in adapter.model.parameters()):
            raise RuntimeError("Model parameters are not all on the requested CUDA device")
        records = []
        with (args.out / "raw.jsonl").open("x", encoding="utf-8") as stream:
            for row in protocol["fixed3"]:
                record = native_record(adapter, row, ev)
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
        manifest["pretrained_GPU_generation_executed"] = any("native_output" in r for r in records)
        manifest["attempted"] = len(records)
        manifest["complete"] = len(records) == 3 and not any(r["failure"] for r in records)
        if all(isinstance(r["raw_text"], str) and r["raw_text"].strip() for r in records):
            version = "music-planner-fixed3-native-v1:" + checkpoint_ref["fingerprint"] + ":" + checkpoint_ref["trainer_state"]["sha256"]
            handoff = dict(records=[dict(id=row["id"], image_path=row["image_path"],
                motion_caption=record["raw_text"], caption_version=version)
                for row, record in zip(protocol["fixed3"], records)])
            new_json(args.out / "h3_records.json", handoff)
            h3 = load_module("fixed3_h3_validation", H3)
            h3.captions(args.out / "h3_records.json", 3)
            manifest["h3_handoff_approved"] = manifest["complete"]
            manifest["h3_records"] = ev.file_ref(args.out / "h3_records.json")
        else:
            manifest["complete"] = False
    except Exception as exc:
        manifest["complete"] = False
        manifest["h3_handoff_approved"] = False
        manifest["failure"] = dict(type=type(exc).__name__, message=str(exc))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not manifest["complete"] or not manifest["h3_handoff_approved"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("prepare", help="CPU-only freeze fixed3 evidence and val32/donor protocol; fresh output only")
    run = sub.add_parser("generate", help="Only run later on a separately admitted CUDA allocation; not a GPU launcher")
    run.add_argument("--family", choices=FAMILIES, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--checkpoint-selection", type=Path, required=True, help="Actual training outputs JSON or checkpoint parent best.json")
    run.add_argument("--checkpoint-selection-sha", required=True, help="Previously frozen SHA256 of that actual selection receipt")
    run.add_argument("--checkpoint-marker-sha", required=True, help="Independently frozen trainer_state.json SHA256 from checkpoint selection time")
    run.add_argument("--device", required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare_protocol() if args.mode == "prepare" else generate(args)


if __name__ == "__main__":
    main()
