"""Opt-in caption/reference video-only H3 SFT; legacy AV training is untouched."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path('/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930')))

import accelerate
import torch
from peft import LoraConfig, get_peft_model

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, ImageCropAndResize
from diffsynth.diffusion import DiffusionTrainingModule, ModelLogger, add_general_config, add_video_size_config
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing, save_training_args
from diffsynth.utils.data.minimax_h3 import MiniMaxH3ReferenceLoader

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
CACHE_FORMAT = "h3-video-only-caption-v1"
LORA_TARGETS = "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2"
FORBIDDEN_INPUTS = {
    "input_audio", "audio", "video_audio", "retake_audio", "audio_input_latents",
    "input_latents_audio", "denoise_mask_audio", "ref_audio_anchor", "audio_path",
}
FORBIDDEN_CONDITIONING = {
    "input_image", "end_image", "keyframes", "keyframe_indices", "keyframe_cond_anchor",
    "retake_video", "input_latents_video", "denoise_mask_video", "first_frame_latents",
}


def validate_video_inputs(values):
    for key in FORBIDDEN_INPUTS | FORBIDDEN_CONDITIONING:
        if key in values and values[key] is not None:
            raise ValueError(f"Video-only caption/reference SFT rejects {key}")
    if "audio_latents" in values:
        value = values["audio_latents"]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (2, 32, 0):
            raise ValueError("Only the length-zero (2,32,0) audio type placeholder is allowed")
    references = values.get("references")
    if references is not None:
        specs = references if isinstance(references, list) else [references]
        for spec in specs:
            if not isinstance(spec, dict) or spec.get("type") not in ("image", "video"):
                raise ValueError("Only explicit image/video reference specs are allowed")
            if "audio" in spec or "input_audio" in spec:
                raise ValueError("Audio is not allowed inside a visual reference")
    packed = values.get("packed")
    if packed is not None:
        if packed["audio_pos"].numel() or (packed["token_tags"] == 2).any():
            raise ValueError("Packed cache contains audio positions or tag2; rebuild the caption cache")
    prompt = values.get("prompt")
    if prompt is not None and (not isinstance(prompt, str) or "<audio" in prompt.lower()):
        raise ValueError("Prompt must be text with no <Audio ...> references")
    if values.get("cfg_scale", 1.0) != 1.0:
        raise ValueError("This video-only SFT recipe fixes CFG at 1")


def validate_caption_cache(inputs, caption_version):
    if not isinstance(inputs, (tuple, list)) or len(inputs) != 3:
        raise ValueError("Expected the new (shared, positive, negative) caption cache")
    shared, positive, negative = inputs
    metadata = shared.get("h3_video_cache", {})
    if metadata.get("format") != CACHE_FORMAT or metadata.get("caption_version") != caption_version:
        raise ValueError("Old/mismatched caption cache is forbidden; re-encode the new teacher caption")
    if not metadata.get("caption_sha256"):
        raise ValueError("Caption cache is missing caption provenance")
    for values in inputs:
        validate_video_inputs(values)
    if negative:
        raise ValueError("CFG1 cache must not contain a negative embedding branch")
    if "input_latents" not in shared or "prompt_embeds" not in positive or "packed" not in positive:
        raise ValueError("Cache lacks the video target or newly encoded caption/packed condition")
    return inputs


def teacher_video_prompt(data):
    from h3_video_only import video_prompt
    caption = data.get("motion_caption")
    if not isinstance(caption, str) or not caption.strip() or "<Picture" in caption or "subject_definitions:" in caption:
        raise ValueError("Require raw motion_caption, not an already wrapped prompt")
    validate_video_inputs({"prompt": caption})
    return video_prompt(caption)


class VisualReferenceLoader(MiniMaxH3ReferenceLoader):
    def load_block(self, spec):
        validate_video_inputs({"references": spec})
        return super().load_block(spec)

    def load_audio(self, *args, **kwargs):
        raise AssertionError("Audio reference loading is forbidden")


class WindowAlignedVideoLoader(LoadVideo):
    def get_num_frames(self, reader):
        frames, fps = int(reader.count_frames()), float(reader.get_meta_data()['fps'])
        if (self.num_frames != 124 or self.frame_rate != 24 or not self.fix_frame_rate or
                frames < 1 or not math.isfinite(fps) or fps <= 0 or
                abs(frames / fps - 5.0625) > 1 / fps):
            raise ValueError('Explicit repeat-last protocol requires the verified 5.0625-second paired source')
        return self.num_frames

    def mapping(self, reader):
        frames, fps = int(reader.count_frames()), float(reader.get_meta_data()['fps'])
        output_frames = self.get_num_frames(reader)
        indices = [self.map_single_frame_id(index, fps, frames) for index in range(output_frames)]
        natural = sum(index / self.frame_rate < frames / fps for index in range(output_frames))
        return dict(protocol='nearest24_repeat_last124_v1', source_frames=frames, source_fps=fps,
                    source_duration_seconds=frames / fps, target_frames=output_frames,
                    target_fps=self.frame_rate, target_duration_seconds=output_frames / self.frame_rate,
                    source_frame_indices=indices, unique_source_frames=len(set(indices)),
                    natural_target_frames=natural, terminal_padding_frames=output_frames - natural,
                    padding_does_not_create_motion_evidence=True)


class VideoCaptionDataset(UnifiedDataset):
    def __init__(self, *args, caption_version, temporal_policy='native_down', **kwargs):
        super().__init__(*args, **kwargs)
        self.caption_version = caption_version
        self.temporal_policy = temporal_policy
        if not len(self):
            raise ValueError("The selected dataset is empty")
        if not self.load_from_cache:
            for row in self.data:
                validate_video_inputs(row)
                if row.get("caption_version") != caption_version:
                    raise ValueError("Every teacher caption row needs the selected caption_version")
                teacher_video_prompt(row)
                if not row.get("references"):
                    raise ValueError("Teacher caption rows require explicit visual references")

    def __getitem__(self, index):
        if self.load_from_cache:
            # Do not enter legacy UnifiedDataset's OD_H3_POSI/NEGA/audio mutation paths.
            value = self.cached_data_operator(self.cached_data[index % len(self.cached_data)])
            return validate_caption_cache(value, self.caption_version)
        if self.temporal_policy == 'nearest24_repeat_last124':
            source = self.data[index % len(self.data)]
            if source.get('video_window_seconds') != [0, 5.0625]:
                raise ValueError('Require explicit source video window [0,5.0625] for repeat-last124')
            path = Path(source['video'])
            path = path if path.is_absolute() else Path(self.base_path) / path
            reader = self.main_data_operator.get_reader(str(path))
            try:
                mapping = self.main_data_operator.mapping(reader)
            finally:
                reader.close()
            data = super().__getitem__(index)
            if len(data['video']) != 124:
                raise ValueError('Repeat-last source did not produce the declared 124 frames')
            data['temporal_source_mapping'] = mapping
            return data
        return super().__getitem__(index)


def video_flow_match_loss(pipe, inputs_shared, inputs_posi, *, anchor_lambda=0.1, timestep_id=None, noise=None):
    """Video-only FM plus a soft frozen-base prediction anchor at sigma >= .5."""
    validate_video_inputs(inputs_shared)
    validate_video_inputs(inputs_posi)
    scheduler = pipe.scheduler
    if not scheduler.training or len(scheduler.timesteps) != 1000:
        raise ValueError("Video scheduler must be initialized for 1000 training timesteps")
    if anchor_lambda < 0:
        raise ValueError("anchor_lambda must be nonnegative")
    if timestep_id is None:
        timestep_id = int(torch.randint(0, len(scheduler.timesteps), (1,)).item())
    target_video = inputs_shared["input_latents"]
    if target_video.ndim != 5:
        raise ValueError("input_latents must be a video latent tensor [B,C,T,H,W]")
    timestep = scheduler.timesteps[timestep_id].reshape(1).to(device=target_video.device, dtype=torch.float32)
    sigma = float(scheduler.sigmas[timestep_id])
    noise = torch.randn_like(target_video) if noise is None else noise
    if noise.shape != target_video.shape:
        raise ValueError("Video noise shape does not match the video target")
    values = {**inputs_shared, **inputs_posi}
    values.pop("h3_video_cache", None)
    values.pop("input_latents", None)
    values["video_latents"] = scheduler.add_noise(target_video, noise, timestep)
    prediction = pipe.model_fn(dit=pipe.dit, timestep_video=timestep, **values)
    if isinstance(prediction, tuple):
        prediction, audio_placeholder = prediction
        if audio_placeholder.numel():
            raise AssertionError("Video-only model_fn returned audio")
    if prediction.shape != target_video.shape:
        raise ValueError("Prediction includes reference rows or has the wrong target shape")
    target = scheduler.training_target(target_video, noise, timestep)
    fm = torch.nn.functional.mse_loss(prediction.float(), target.float())
    anchor = fm.new_zeros(())
    if anchor_lambda and sigma >= 0.5:
        teacher_values = {**values, "use_gradient_checkpointing": False, "use_gradient_checkpointing_offload": False}
        # Public PEFT context restores the student's adapter state before checkpoint recomputation.
        with torch.no_grad(), pipe.dit.disable_adapter():
            teacher = pipe.model_fn(dit=pipe.dit, timestep_video=timestep, **teacher_values)
            if isinstance(teacher, tuple):
                teacher, audio_placeholder = teacher
                if audio_placeholder.numel():
                    raise AssertionError("Frozen video-only teacher returned audio")
        anchor = torch.nn.functional.mse_loss(prediction.float(), teacher.detach().float())
    weight = scheduler.training_weight(timestep).to(device=fm.device, dtype=fm.dtype)
    loss = (fm + anchor_lambda * anchor) * weight
    pipe.video_sft_metrics = {
        "fm": fm.detach(), "anchor": anchor.detach(), "sigma": sigma,
        "anchor_active": bool(anchor_lambda and sigma >= 0.5), "scheduler_weight": float(weight),
    }
    return loss


class H3VideoOnlyTrainingModule(DiffusionTrainingModule):
    def __init__(
        self, weights_root=None, *, caption_version, task="sft:train", device="cpu",
        extra_inputs="references", lora_base_model="dit", lora_target_modules=LORA_TARGETS,
        lora_rank=32, lora_alpha=64, anchor_lambda=0.1, training_cfg_scale=1.0,
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pipe=None,
    ):
        super().__init__()
        if task not in ("sft", "sft:data_process", "sft:train"):
            raise ValueError("Unsupported video-only SFT task")
        if extra_inputs != "references" or lora_base_model != "dit":
            raise ValueError("Only extra_inputs=references and DiT LoRA are supported")
        if training_cfg_scale != 1.0 or anchor_lambda not in (0.0, 0.1, 1.0):
            raise ValueError("Recipe requires CFG1 and anchor lambda 0, 0.1 or 1")
        if not caption_version or lora_rank <= 0:
            raise ValueError("caption_version and a positive LoRA rank are required")
        self.task, self.caption_version = task, caption_version
        self.anchor_lambda = anchor_lambda
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        if pipe is None:
            from h3_video_only import build_video_only_pipeline, video_only_model_fn
            mode = "data_process" if task.endswith(":data_process") else "train" if task.endswith(":train") else "all"
            pipe = build_video_only_pipeline(
                weights_root, device=device, torch_dtype=torch.bfloat16,
                load_text_encoder=mode != "train", load_mode=mode,
                vram_management=mode == "data_process",
            )
            pipe.model_fn = video_only_model_fn
        self.pipe = pipe
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.freeze_except([])
        self.pipe = self.split_pipeline_units(
            task, self.pipe, lora_base_model="dit", remove_unnecessary_params=True,
            loss_required_params=("input_latents", "cfg_scale", "h3_video_cache"),
            force_remove_params_shared=("video_latents",),
            force_remove_params_nega=("prompt_embeds", "text_token_tags", "packed", "prompt", "negative_prompt"),
        )
        if not task.endswith(":data_process"):
            if self.pipe.dit is None:
                raise ValueError("Training requires the original BF16 DiT")
            targets = lora_target_modules.split(",")
            if set(targets) != set(LORA_TARGETS.split(",")):
                raise ValueError(f"Use only the approved attention/MLP LoRA targets: {LORA_TARGETS}")
            self.pipe.dit = get_peft_model(self.pipe.dit, LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha, target_modules=targets, lora_dropout=0.0, bias="none",
            ))
            self.pipe.dit.eval()
            for name, parameter in self.named_parameters():
                if parameter.requires_grad and (not name.startswith("pipe.dit.") or ".lora_" not in name):
                    raise AssertionError(f"Unexpected trainable base/audio/TC/VAE parameter: {name}")
                if parameter.requires_grad:
                    parameter.data = parameter.data.to(self.pipe.torch_dtype)
            print(json.dumps({"event": "video_sft_parameters", "trainable_names": sorted(self.trainable_param_names()),
                              "trainable_numel": sum(p.numel() for p in self.trainable_modules()),
                              "anchor_lambda": anchor_lambda, "cfg_scale": 1.0}), flush=True)

    def get_pipeline_inputs(self, data):
        validate_video_inputs(data)
        if data.get("caption_version") != self.caption_version:
            raise ValueError("Teacher caption version mismatch")
        if not data.get("references"):
            raise ValueError("Explicit image/video references are required; no first-frame fallback")
        prompt = teacher_video_prompt(data)
        shared = {
            "input_video": data["video"], "height": data["video"][0].size[1],
            "width": data["video"][0].size[0], "num_frames": len(data["video"]),
            "references": data["references"], "ref_image_short_edge": 2048,
            "ref_video_short_edge": 768, "ref_video_max_pixels": 768 * 1344,
            "imgvid_cond_noise_aug": self.pipe.imgvid_cond_noise_aug,
            "tiled": True, "tile_size": 256, "tile_overlap": 64,
            "cfg_scale": 1.0, "seed": 42, "rand_device": "cpu",
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "h3_video_cache": {"format": CACHE_FORMAT, "caption_version": self.caption_version,
                               "caption_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                               "sample_id": str(data.get("id", data.get("sample_id", ""))),
                               "temporal_source_mapping": data.get("temporal_source_mapping")},
        }
        return shared, {"prompt": prompt}, {}

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        else:
            validate_caption_cache(inputs, self.caption_version)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        shared, positive, _ = inputs
        inputs = shared, positive, {}
        if self.task.endswith(":data_process"):
            shared.pop("video_latents", None)
            validate_caption_cache(inputs, self.caption_version)
            return inputs
        return video_flow_match_loss(self.pipe, shared, positive, anchor_lambda=self.anchor_lambda)


def export_diffsynth_lora(state, lora_alpha, lora_rank):
    # DiffSynth's native loader has no implicit PEFT alpha/r scale.
    return {key.removeprefix("base_model.model."): value * (lora_alpha / lora_rank)
            if ".lora_B." in key else value for key, value in state.items()}


COMPLETE_MARKER = "complete.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(root, paths):
    root = Path(root)
    return {str(Path(path).relative_to(root)): {"sha256": file_sha256(path), "bytes": Path(path).stat().st_size}
            for path in sorted(map(Path, paths))}


def checkpoint_files(root):
    return [path for path in Path(root).rglob("*") if path.is_file() and path.name != COMPLETE_MARKER]


def write_complete_marker(root, signature, progress=None):
    root = Path(root)
    files = file_manifest(root, checkpoint_files(root))
    if not files:
        raise ValueError("Cannot mark an empty checkpoint complete")
    marker = {"format": "h3-video-complete-v1", "scope": signature.get("data_scope"),
              "signature": signature, "files": files,
              "progress": progress.state_dict() if progress is not None else None}
    # Marker is written only after save_state returns on every rank and all files are hashed.
    with (root / COMPLETE_MARKER).open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    return marker


def verify_complete_marker(root, signature=None):
    root = Path(root)
    marker_path = root / COMPLETE_MARKER
    if not marker_path.is_file():
        raise ValueError("Partial checkpoint/cache has no completion marker; cannot resume")
    marker = json.loads(marker_path.read_text())
    if marker.get("format") != "h3-video-complete-v1":
        raise ValueError("Invalid checkpoint completion protocol")
    if signature is not None and marker["signature"] != signature:
        raise ValueError("Resume source/cache/model/code/recipe signature changed")
    actual = file_manifest(root, checkpoint_files(root))
    if not actual or actual != marker["files"]:
        raise ValueError("Checkpoint/cache files changed or are incomplete")
    return marker


def main_process_result(accelerator, operation):
    from accelerate.utils import broadcast_object_list
    payload = [None]
    if accelerator.is_main_process:
        try:
            payload[0] = {"result": operation()}
        except Exception as error:
            payload[0] = {"error": f"{type(error).__name__}: {error}"}
    broadcast_object_list(payload, from_process=0)
    if "error" in payload[0]:
        raise ValueError(payload[0]["error"])
    return payload[0]["result"]


def formal_training_provenance(metadata_path, caption_version):
    from dance_caption_fusion import (AUDIO_WINDOW_SECONDS, PROTOCOL, approved_review,
                                     load_manifest, read_json, review_record_map,
                                     source_record, validate_target)
    metadata_path = Path(metadata_path).resolve()
    sidecar_path = metadata_path.parent / "source_sidecar.json"
    sidecar = read_json(sidecar_path)
    sources = {}

    def verify(source):
        if not isinstance(source, dict) or set(source) != {"path", "sha256", "size_bytes"}:
            raise ValueError("Formal source requires a complete path/SHA/size record")
        actual = source_record(source["path"], source["sha256"])
        if actual != source:
            raise ValueError("Formal source record changed")
        sources[actual["path"]] = actual
        return actual

    def reject_fixture(value):
        if any(value.get(key, False) is not False for key in ("fixture_only", "formal_reuse_forbidden")):
            raise ValueError("Fixture or formal-reuse-forbidden data cannot enter formal training")

    reject_fixture(sidecar)
    if sidecar.get("protocol") != PROTOCOL or sidecar.get("complete") is not True:
        raise ValueError("Formal training requires a completed reviewed export")
    policy = sidecar.get("h3_metadata_policy", {})
    expected_policy = {"train_only": "h3_video_train.json", "val_only": "h3_video_val.json",
                       "combined": "h3_video.json", "combined_use": "exchange_and_audit_only",
                       "combined_direct_cache_or_training_forbidden": True}
    if any(policy.get(key) != value or type(policy.get(key)) is not type(value)
           for key, value in expected_policy.items()):
        raise ValueError("Formal export must enforce train-only H3 metadata")
    outputs = sidecar.get("outputs", {})
    names = {"train.jsonl", "val.jsonl", "omni.jsonl", "h3_video.json", "h3_video_train.json", "h3_video_val.json"}
    if set(outputs) != names:
        raise ValueError("Formal export outputs are incomplete")
    for name, source in outputs.items():
        if Path(source["path"]).resolve() != sidecar_path.parent / name:
            raise ValueError("Export output path does not match its declared partition")
        verify(source)
    if metadata_path != Path(outputs["h3_video_train.json"]["path"]).resolve():
        raise ValueError("Combined/val metadata cannot produce a formal training cache")
    if (not isinstance(caption_version, str) or not caption_version.startswith(PROTOCOL + ":") or
            not caption_version[len(PROTOCOL) + 1:].strip() or sidecar.get("caption_version") != caption_version):
        raise ValueError("Formal export caption version differs from the frozen rewrite protocol")
    manifest_source, review_source = verify(sidecar["manifest"]), verify(sidecar["review"])
    verify(sidecar["source_code"])
    manifest = load_manifest(manifest_source["path"])
    review = read_json(review_source["path"])
    reject_fixture(manifest)
    reject_fixture(review)
    reviews = review_record_map(review)
    records = {record["id"]: record for record in manifest["records"]}
    exported = sidecar.get("records", [])
    if (not isinstance(exported, list) or len(exported) != len(records) or
            {row["id"] for row in exported} != set(records) or set(reviews) != set(records) or
            review.get("manifest_sha256") != manifest_source["sha256"]):
        raise ValueError("Formal review/export IDs or manifest SHA differ")
    expected_rows = []
    videos = {"train": set(), "val": set()}
    audios = {"train": set(), "val": set()}
    for row in exported:
        record, approval = records[row["id"]], reviews[row["id"]]
        for value in (row, record, approval):
            reject_fixture(value)
        rewrite_source = verify(row["rewrite_source"])
        rewrite = read_json(rewrite_source["path"])
        reject_fixture(rewrite)
        approved_review(approval, rewrite_source["path"], record)
        if (row.get("trained_caption_approved") is not True or row.get("human_review") != approval or
                row.get("source_record_manifest") != manifest_source or row.get("review_source") != review_source or
                row.get("approval_provenance") != dict(manifest=manifest_source, review=review_source, rewrite=rewrite_source) or
                rewrite.get("id") != row["id"] or rewrite.get("split") != record["split"] or
                row.get("split") != record["split"] or row.get("source_group") != record["source_group"] or
                rewrite.get("manifest_sha256") != manifest_source["sha256"] or
                rewrite.get("rewrite_input_sha256") != record["rewrite_input_sha256"] or
                rewrite.get("caption_version") != caption_version or row.get("caption_version") != caption_version):
            raise ValueError("Formal row approval/rewrite provenance differs")
        target = rewrite["target"]
        if not validate_target(target, record)["format_valid"]:
            raise ValueError("Reviewed rewrite fails its actual target contract")
        if any(row.get(key) != target[key] for key in ("motion_caption", "used_facts", "conflicts", "time_scope")):
            raise ValueError("Export caption/facts differ from the reviewed rewrite")
        media = record.get("provenance", {})
        if not {"audio", "image", "paired_video", "meta"} <= set(media):
            raise ValueError("Formal manifest lacks paired media provenance")
        for source in media.values():
            verify(source)
        for key, field in (("audio", "audio_path"), ("image", "image_path"), ("paired_video", "paired_video_path")):
            if (str(Path(record[field]).absolute()) != media[key]["path"] or
                    row.get(key + "_source") != media[key]):
                raise ValueError("Formal media path differs from its reviewed source")
        scope = record["time_scope"]
        for window in (record.get("audio_window_seconds"), scope.get("training_audio_window_seconds"),
                       row.get("audio_window_seconds"), row.get("video_window_seconds")):
            if (not isinstance(window, list) or any(type(value) not in (int, float) for value in window) or
                    window != AUDIO_WINDOW_SECONDS):
                raise ValueError("Formal paired window is missing or invalid")
        meta = read_json(media["meta"]["path"])
        frames, fps = scope.get("paired_normalized_natural_frames"), scope.get("paired_normalized_fps")
        if (type(frames) is not int or frames < 1 or type(fps) not in (int, float) or
                not math.isfinite(fps) or fps <= 0 or frames / fps != AUDIO_WINDOW_SECONDS[1] or
                meta.get("natural_output_frames") != frames or meta.get("target_fps") != fps or
                meta.get("video") != media["paired_video"]["path"] or row.get("source_time_scope") != scope):
            raise ValueError("Formal paired video/window evidence differs")
        videos[record["split"]].add(str(Path(media["paired_video"]["path"]).resolve()))
        audios[record["split"]].add(str(Path(media["audio"]["path"]).resolve()))
        expected_rows.append(dict(id=row["id"], source_group=record["source_group"], split=record["split"],
            motion_caption=target["motion_caption"], caption_version=caption_version,
            video=media["paired_video"]["path"], references=[dict(type="image", image=media["image"]["path"])],
            video_window_seconds=list(AUDIO_WINDOW_SECONDS), trained_caption_approved=True,
            provenance=dict(manifest=manifest_source, review=review_source, rewrite=rewrite_source,
                            image=media["image"], paired_video=media["paired_video"])))
    for partition in ("train", "val"):
        selected = [row for row in expected_rows if row["split"] == partition]
        actual_rows = read_json(outputs["h3_video_" + partition + ".json"]["path"])
        for item in actual_rows:
            reject_fixture(item)
            if item.get("trained_caption_approved") is not True or item.get("split") != partition:
                raise ValueError("Formal metadata requires actual approved rows from its declared split")
        if (not selected or type(sidecar.get("counts", {}).get(partition)) is not int or
                sidecar["counts"][partition] != len(selected) or actual_rows != selected):
            raise ValueError("Formal split metadata/counts differ from the reviewed export")
    combined_rows = read_json(outputs["h3_video.json"]["path"])
    for item in combined_rows:
        reject_fixture(item)
        if item.get("trained_caption_approved") is not True:
            raise ValueError("Combined audit rows lack actual boolean approval")
    if (combined_rows != expected_rows or
            videos["train"] & videos["val"] or audios["train"] & audios["val"]):
        raise ValueError("Formal export contains combined metadata drift or split media leakage")
    return dict(protocol="h3-reviewed-train-provenance-v1", metadata=outputs["h3_video_train.json"],
                sidecar=source_record(sidecar_path), verified_sources=sources,
                validator_source=source_record(Path(__file__).with_name("dance_caption_fusion.py")),
                caption_version=caption_version, h3_metadata_policy=policy,
                train_ids=[row["id"] for row in expected_rows if row["split"] == "train"],
                fixture_only=False, formal_reuse_forbidden=False, approval_chain_verified=True)


def source_label_training_provenance(metadata_path, caption_version):
    """Bind unreviewed H3 targets to the authorized frozen Omni source-label pilot."""
    import dance_caption_fusion as fusion
    import omni_dance_sft as omni
    metadata_path = Path(metadata_path).resolve()
    sidecar_path = metadata_path.parent / "source_label_h3_sidecar.json"
    sources = {}

    def verify(source):
        if not isinstance(source, dict) or not {"path", "sha256", "size_bytes"} <= set(source):
            raise ValueError("Source-label H3 requires a complete path/SHA/size record")
        actual = fusion.source_record(source["path"], source["sha256"])
        if actual != {key: source[key] for key in ("path", "sha256", "size_bytes")}:
            raise ValueError("Source-label H3 source record changed")
        sources[actual["path"]] = actual
        return actual

    def reject_fixture(value):
        if any(value.get(key, False) is not False for key in ("fixture_only", "formal_reuse_forbidden")):
            raise ValueError("Fixture data cannot enter source-label H3 training")

    sidecar_source = fusion.source_record(sidecar_path)
    sources[sidecar_source["path"]] = sidecar_source
    sidecar = fusion.read_json(sidecar_path)
    reject_fixture(sidecar)
    if (sidecar.get("protocol") != "h3-source-label-train-provenance-v1" or
            sidecar.get("complete") is not True or sidecar.get("caption_version") != caption_version or
            sidecar.get("authorization") != omni.SOURCE_LABEL_AUTHORIZATION or
            sidecar.get("label_quality") != "source_labels_unreviewed" or
            any(sidecar.get(key) is not False for key in ("approval_chain_verified", "human_review_completed", "semantic_approval")) or
            sidecar.get("original_source_window_mapping") != "unknown"):
        raise ValueError("Source-label H3 requires authorized unreviewed labels, not semantic/human approval")
    validator_source = verify(sidecar["frozen_omni_validator_source"])
    if validator_source["path"] != str(Path(omni.__file__).resolve()):
        raise ValueError("Source-label H3 must reuse the frozen Omni training validator")
    verify(sidecar["source_code"])
    parent_metadata = verify(sidecar["parent_omni_metadata"])
    parent_sidecar = verify(sidecar["parent_omni_sidecar"])
    if Path(parent_sidecar["path"]) != Path(parent_metadata["path"]).parent / "source_label_sidecar.json":
        raise ValueError("Source-label H3 parent sidecar and metadata directories differ")
    parent_config = argparse.Namespace(metadata=Path(parent_metadata["path"]), data_scope="source_label_pilot",
                                       smoke_steps=0, min_epochs=3, max_epochs=3)
    rows, train, val, parent = omni.validate_data_boundary(parent_config)
    if (parent.get("metadata") != parent_metadata or parent.get("sidecar") != parent_sidecar or
            parent.get("caption_version") != caption_version or parent.get("approval_chain_verified") is not False or
            sidecar.get("parent_caption_version") != caption_version or
            parent.get("manifest") != sidecar.get("parent_manifest")):
        raise ValueError("Source-label H3 parent metadata/version/manifest identity differs")
    manifest_source = verify(sidecar["parent_manifest"])
    manifest = fusion.load_manifest(manifest_source["path"])
    reject_fixture(manifest)
    records = {record["id"]: record for record in manifest["records"]}
    if len(records) != len(rows) or set(records) != {row["id"] for row in rows}:
        raise ValueError("Source-label H3 parent manifest IDs differ")
    outputs = sidecar.get("outputs", {})
    if set(outputs) != {"h3_video_train.json", "h3_video_val.json"}:
        raise ValueError("Source-label H3 requires separate complete train/val outputs")
    policy = dict(train_only="h3_video_train.json", val_only="h3_video_val.json",
                  combined_direct_cache_or_training_forbidden=True)
    if sidecar.get("h3_metadata_policy") != policy:
        raise ValueError("Source-label H3 metadata policy must forbid combined/val training")
    for name, source in outputs.items():
        if Path(source["path"]).resolve() != metadata_path.parent / name:
            raise ValueError("Source-label H3 output partition path differs")
        verify(source)
    if metadata_path != Path(outputs["h3_video_train.json"]["path"]).resolve():
        raise ValueError("Source-label H3 cache/training requires train-only metadata")
    expected_rows = []
    groups = {key: {"train": set(), "val": set()} for key in ("source_group", "image", "paired_video")}
    for row in rows:
        record = records[row["id"]]
        reject_fixture(row)
        reject_fixture(record)
        if (row["source_group"] != record["source_group"] or row["split"] != record["split"] or
                row["caption_version"] != caption_version):
            raise ValueError("Source-label H3 row ID/group/split/version differs from parent")
        media = {key: verify(record["provenance"][key]) for key in ("image", "paired_video", "meta")}
        if (record["paired_video_path"] != media["paired_video"]["path"] or
                record["image_path"] != media["image"]["path"]):
            raise ValueError("Source-label H3 image/video paths differ from parent manifest")
        meta = fusion.read_json(media["meta"]["path"])
        reject_fixture(meta)
        if (type(meta.get("natural_output_frames")) is not int or meta["natural_output_frames"] != 81 or
                type(meta.get("target_fps")) not in (int, float) or meta["target_fps"] != 16 or
                meta.get("target_frames") != 81 or meta.get("tail_padding_frames") != 0 or
                meta.get("video") != media["paired_video"]["path"] or
                meta.get("first_frame") != media["image"]["path"]):
            raise ValueError("Source-label H3 requires actual normalized 81-frame/16fps unpadded media evidence")
        caption = omni.text_content(row["messages"][-1]["content"])
        expected_rows.append(dict(id=row["id"], source_group=row["source_group"], split=row["split"],
            motion_caption=caption, caption_version=caption_version, video=media["paired_video"]["path"],
            references=[dict(type="image", image=media["image"]["path"])], video_window_seconds=[0, 5.0625],
            trained_caption_approved=False, label_quality="source_labels_unreviewed",
            provenance=dict(parent_omni_metadata=parent_metadata, parent_manifest=manifest_source, **media)))
        groups["source_group"][row["split"]].add(row["source_group"])
        for key in ("image", "paired_video"):
            groups[key][row["split"]].add(str(Path(media[key]["path"]).resolve()))
    if any(group["train"] & group["val"] for group in groups.values()):
        raise ValueError("Source-label H3 source group or visual media leaks across train/val")
    if sidecar.get("counts") != dict(train=512, val=128) or (len(train), len(val)) != (512, 128):
        raise ValueError("Source-label H3 requires the actual 512/128 parent pilot")
    for partition in ("train", "val"):
        selected = [row for row in expected_rows if row["split"] == partition]
        actual_rows = fusion.read_json(outputs["h3_video_" + partition + ".json"]["path"])
        for row in actual_rows:
            reject_fixture(row)
            validate_video_inputs(row)
            if row.get("trained_caption_approved") is not False or row.get("label_quality") != "source_labels_unreviewed":
                raise ValueError("Source-label H3 metadata cannot claim caption approval")
        if (type(sidecar["counts"][partition]) is not int or len(selected) != sidecar["counts"][partition] or
                actual_rows != selected):
            raise ValueError("Source-label H3 exact metadata/captions/partitions differ from parent Omni")
    for source in list(sources.values()):
        verify(source)
    return dict(protocol="h3-source-label-train-provenance-v1", metadata=outputs["h3_video_train.json"],
        sidecar=sidecar_source, parent_omni_provenance=parent, verified_sources=sources,
        frozen_omni_validator_source=validator_source, caption_version=caption_version, h3_metadata_policy=policy,
        train_ids=[row["id"] for row in expected_rows if row["split"] == "train"],
        val_ids=[row["id"] for row in expected_rows if row["split"] == "val"],
        authorization=omni.SOURCE_LABEL_AUTHORIZATION, label_quality="source_labels_unreviewed",
        approval_chain_verified=False, human_review_completed=False, semantic_approval=False,
        original_source_window_mapping="unknown", fixture_only=False, formal_reuse_forbidden=False)


def validate_data_boundary(dataset, args):
    scope = getattr(args, "data_scope", "interface") or "interface"
    if scope not in ("interface", "reviewed_train", "source_label_train") or scope == "interface" and args.max_train_steps != 2:
        raise ValueError("Interface data requires explicit max_train_steps=2; it cannot enter 100/300 training")
    if scope == "source_label_train" and getattr(args, "temporal_policy", "native_down") != "nearest24_repeat_last124":
        raise ValueError("Source-label H3 requires explicit nearest24_repeat_last124 temporal policy")
    if dataset.load_from_cache:
        root = Path(args.dataset_base_path)
        marker = verify_complete_marker(root)
        signature = marker["signature"]
        code = {path.name: file_sha256(path) for path in (Path(__file__), Path(__file__).with_name("h3_video_only.py"))}
        if (marker.get("scope") != scope or signature.get("data_scope") != scope or signature.get("caption_version") != args.caption_version or
                signature.get("temporal_policy") != getattr(args, "temporal_policy", "native_down") or
                signature.get("sources", {}).get("code") != code):
            raise ValueError("Cache scope/caption/temporal policy/producer source differs; create a fresh cache")
        cache_files = {str(Path(path).relative_to(root)) for path in dataset.cached_data}
        expected_files = {name for name in marker["files"] if name.endswith(".pth")}
        if (signature.get("dataset_size") != len(dataset) or cache_files != expected_files or
                signature.get("dataset_metadata_path") is None):
            raise ValueError("Cache dataset membership or producer metadata provenance differs")
        provenance = signature.get("data_provenance", {})
        if scope == "reviewed_train":
            actual = formal_training_provenance(provenance.get("metadata", {}).get("path", ""), args.caption_version)
            if (provenance != actual or len(dataset) != len(actual["train_ids"]) or
                    signature["dataset_metadata_path"] != actual["metadata"]["path"] or
                    signature.get("sources", {}).get("metadata_sha256") != actual["metadata"]["sha256"]):
                raise ValueError("Formal cache approval/metadata/media provenance changed")
        elif scope == "source_label_train":
            actual = source_label_training_provenance(provenance.get("metadata", {}).get("path", ""), args.caption_version)
            if (provenance != actual or len(dataset) != len(actual["train_ids"]) or
                    signature["dataset_metadata_path"] != actual["metadata"]["path"] or
                    signature.get("sources", {}).get("metadata_sha256") != actual["metadata"]["sha256"]):
                raise ValueError("Source-label cache metadata or unreviewed source provenance changed")
        elif provenance.get("protocol") != "h3-interface-only-v1" or provenance.get("formal_reuse_forbidden") is not True:
            raise ValueError("Interface cache lacks explicit nonformal provenance")
        provenance_sha = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        expected_captions = {}
        if scope in ("reviewed_train", "source_label_train"):
            expected_captions = {row["id"]: hashlib.sha256(teacher_video_prompt(row).encode()).hexdigest()
                                 for row in json.loads(Path(provenance["metadata"]["path"]).read_text())}
        sample_ids = set()
        for path in dataset.cached_data:
            value = validate_caption_cache(torch.load(path, map_location="cpu", weights_only=False), args.caption_version)
            info = value[0]["h3_video_cache"]
            if info.get("data_scope") != scope or info.get("data_provenance_sha256") != provenance_sha:
                raise ValueError("Cached sample scope or approval binding differs from its producer")
            if scope in ("reviewed_train", "source_label_train"):
                sid = info.get("sample_id")
                if sid in sample_ids or sid not in expected_captions or info["caption_sha256"] != expected_captions[sid]:
                    raise ValueError("Cached sample ID/caption differs from its approved train split")
                sample_ids.add(sid)
        if scope in ("reviewed_train", "source_label_train") and sample_ids != set(expected_captions):
            raise ValueError("Formal cache is missing approved training samples")
    elif scope == "reviewed_train":
        provenance = formal_training_provenance(args.dataset_metadata_path, args.caption_version)
        if (len(dataset) != len(provenance["train_ids"]) or
                dataset.data != json.loads(Path(args.dataset_metadata_path).read_text())):
            raise ValueError("Loaded formal dataset differs from its approved train metadata")
    elif scope == "source_label_train":
        provenance = source_label_training_provenance(args.dataset_metadata_path, args.caption_version)
        if (len(dataset) != len(provenance["train_ids"]) or
                dataset.data != json.loads(Path(args.dataset_metadata_path).read_text())):
            raise ValueError("Loaded source-label dataset differs from its unreviewed train metadata")
    else:
        provenance = dict(protocol="h3-interface-only-v1", metadata_path=str(Path(args.dataset_metadata_path).resolve()),
                          metadata_sha256=file_sha256(args.dataset_metadata_path), formal_reuse_forbidden=True)
    return dict(data_scope=scope, data_provenance=provenance)


def source_fingerprint(dataset, args):
    base = Path(args.dataset_base_path)
    code_paths = [Path(__file__), Path(__file__).with_name("h3_video_only.py")]
    model_root = Path(args.weights_root)
    transformer = model_root / "transformer"
    shards = sorted(transformer.glob("model*.safetensors"))
    if not shards or not (transformer / "config.json").is_file():
        raise ValueError("Original DiT config and actual weight shards are required for provenance")
    model_files = shards + sorted(transformer.glob("*.json")) + [model_root / "model_index.json"]
    result = {"code": {path.name: file_sha256(path) for path in code_paths},
              "original_dit": file_manifest(model_root, model_files)}
    if dataset.load_from_cache:
        marker = verify_complete_marker(base)
        if marker["signature"].get("sources", {}).get("code") != result["code"]:
            raise ValueError("Caption cache was encoded with different wrapper/training source code")
        paths = [Path(path) for path in dataset.cached_data]
        result["cache"] = {str(path.relative_to(base)): marker["files"][str(path.relative_to(base))] for path in paths}
        result["cache_complete_marker"] = file_sha256(base / COMPLETE_MARKER)
        result["cache_source"] = marker["signature"]
        if marker["signature"].get("temporal_policy", "native_down") != getattr(args, "temporal_policy", "native_down"):
            raise ValueError('Training temporal policy differs from the actual cache producer')
    else:
        result["metadata_sha256"] = file_sha256(args.dataset_metadata_path)
        media_paths = set()
        for row in dataset.data:
            media_paths.add(Path(row["video"]))
            for reference in row["references"]:
                media_paths.add(Path(reference[reference["type"]]))
        result["source_media"] = {str(path): file_sha256(path if path.is_absolute() else base / path)
                                  for path in sorted(media_paths)}
        encoder_files = list((model_root / "text_encoder").glob("model*.safetensors"))
        encoder_files += list((model_root / "text_encoder").glob("*.json"))
        encoder_files += list((model_root / "processor").glob("*"))
        encoder_files += list((model_root / "video_vae/source").glob("*"))
        result["frozen_encoders"] = file_manifest(model_root, [path for path in encoder_files if path.is_file()])
    return result


def training_signature(accelerator, dataset, args):
    boundary = main_process_result(accelerator, lambda: validate_data_boundary(dataset, args))
    return {**boundary, "caption_version": args.caption_version, "dataset_base_path": str(Path(args.dataset_base_path).resolve()),
            "dataset_metadata_path": str(Path(args.dataset_metadata_path).resolve()) if args.dataset_metadata_path else None,
            "dataset_size": len(dataset),
            "anchor_lambda": args.anchor_lambda, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha, "weights_root": args.weights_root,
            "world_size": accelerator.num_processes, "accumulation": args.gradient_accumulation_steps,
            "seed": args.seed, "lora_target_modules": getattr(args, "lora_target_modules", LORA_TARGETS),
            "gradient_checkpointing": getattr(args, "use_gradient_checkpointing", True),
            "gradient_checkpointing_offload": getattr(args, "use_gradient_checkpointing_offload", False),
            "temporal_policy": getattr(args, "temporal_policy", "native_down"),
            "sources": main_process_result(accelerator, lambda: source_fingerprint(dataset, args))}


def validate_cache_offload(model):
    for name in ("text_encoder", "video_vae"):
        encoder = getattr(model.pipe, name)
        if not getattr(encoder, "vram_management_enabled", False):
            raise ValueError(f"{name} is not VRAM managed")
        if any(parameter.device.type != "cpu" for parameter in encoder.parameters()):
            raise ValueError(f"{name} must start offloaded on CPU")
        wrappers = [layer for layer in encoder.modules() if hasattr(layer, "offload_device")]
        if not wrappers or any(str(layer.offload_device) != "cpu" for layer in wrappers):
            raise ValueError(f"{name} must contain actual CPU-offload wrappers")


def launch_video_cache(accelerator, dataset, model, args):
    if not model.pipe.vram_management_enabled:
        raise ValueError("Data processing requires CPU-offloaded video VAE/text encoder")
    # Never call model.to or Accelerator.prepare(model): that defeats pipeline-unit offload.
    validate_cache_offload(model)

    signature = training_signature(accelerator, dataset, args)
    model.pipe.device = accelerator.device
    output = Path(args.output_path)
    output.mkdir(exist_ok=True)
    local_ids = range(accelerator.process_index, len(dataset), accelerator.num_processes)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, list(local_ids)), batch_size=1,
                                        collate_fn=lambda batch: batch[0], num_workers=args.dataset_num_workers)
    folder = output / str(accelerator.process_index)
    folder.mkdir(exist_ok=False)
    with torch.no_grad():
        for index, data in enumerate(loader):
            value = model(data)
            value[0]["h3_video_cache"].update(data_scope=signature["data_scope"],
                data_provenance_sha256=hashlib.sha256(json.dumps(signature["data_provenance"], sort_keys=True).encode()).hexdigest())
            value = model.transfer_data_to_device(value, "cpu")
            with (folder / f"{index}.pth").open("xb") as stream:
                torch.save(value, stream)
    model.pipe.load_models_to_device([])
    accelerator.wait_for_everyone()
    boundary = main_process_result(accelerator, lambda: validate_data_boundary(dataset, args))
    if any(signature[key] != value for key, value in boundary.items()):
        raise ValueError("Cache source approval or media changed during encoding")
    main_process_result(accelerator, lambda: write_complete_marker(output, signature))
    if accelerator.is_main_process:
        print(json.dumps({"event": "video_caption_cache_complete", "samples": len(dataset),
                          "whole_model_to_calls": 0, "unit_managed_cpu_offload": True}), flush=True)


class TrainingProgress:
    def __init__(self, signature):
        self.signature = signature
        self.optimizer_updates = 0
        self.epoch = 0
        self.batches_consumed = 0
        self.microbatches = 0

    def state_dict(self):
        return vars(self).copy()

    def load_state_dict(self, state):
        required = {"signature", "optimizer_updates", "epoch", "batches_consumed", "microbatches"}
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError("Incomplete training progress state")
        if state["signature"] != self.signature:
            raise ValueError("Resume requires matching caption/data/recipe/world size/accumulation")
        if any(type(state[key]) is not int or state[key] < 0 for key in required - {"signature"}):
            raise ValueError("Training progress counters must be nonnegative integers")
        if state["optimizer_updates"] > state["microbatches"] or state["batches_consumed"] > state["microbatches"]:
            raise ValueError("Training progress counters are inconsistent")
        self.__dict__.update(state)

    def validate_budget(self, batches_per_epoch, num_epochs, max_updates):
        if (self.epoch > num_epochs or self.batches_consumed > batches_per_epoch or
                self.optimizer_updates > max_updates or
                self.epoch == num_epochs and self.batches_consumed != 0):
            raise ValueError("Resume cursor exceeds the local loader or requested training budget")


def launch_video_training(accelerator, dataset, model, model_logger, args):
    if args.enable_model_cpu_offload or args.enable_optimizer_cpu_offload or args.customized_optimizer:
        raise ValueError("Use Accelerator/ZeRO offload configuration; legacy per-layer/custom optimizer is unsupported")
    signature = training_signature(accelerator, dataset, args)
    resume_marker = None
    if args.resume_training_state:
        resume_marker = main_process_result(accelerator, lambda: verify_complete_marker(args.resume_training_state, signature))
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    generator = torch.Generator()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, generator=generator,
                                             collate_fn=lambda batch: batch[0], num_workers=args.dataset_num_workers)
    model.to(device=accelerator.device)
    model.pipe.device = accelerator.device
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    progress = TrainingProgress(signature)
    accelerator.register_for_checkpointing(progress)
    if args.resume_training_state:
        progress.load_state_dict(resume_marker.get("progress"))
        progress.validate_budget(len(dataloader), args.num_epochs, args.max_train_steps)
        accelerator.load_state(args.resume_training_state)
        if progress.state_dict() != resume_marker["progress"]:
            raise ValueError("Restored custom progress differs from the verified completion marker")
        progress.validate_budget(len(dataloader), args.num_epochs, args.max_train_steps)
    model_logger.num_steps = progress.optimizer_updates
    if accelerator.is_main_process:
        save_training_args(args)
        print(json.dumps({"event": "video_sft_budget", "effective_batch": accelerator.num_processes * args.gradient_accumulation_steps,
                          "optimizer_updates": progress.optimizer_updates, "max_train_steps": args.max_train_steps,
                          "budget_scope": "interface_smoke_only" if signature["data_scope"] == "interface" else
                              "source_label_weak_supervision" if signature["data_scope"] == "source_label_train" else "human_gated_pilot_diagnostic",
                          "checkpoint_scope": "accelerator model+optimizer+scheduler+RNG+epoch/batch progress"}), flush=True)

    def save_state(name):
        destination = Path(args.output_path) / name
        exists = torch.tensor(int(accelerator.is_main_process and destination.exists()), device=accelerator.device)
        if accelerator.reduce(exists, reduction="sum").item():
            raise FileExistsError(f"Refuse checkpoint overwrite: {destination}")
        accelerator.wait_for_everyone()
        accelerator.save_state(str(destination))
        accelerator.wait_for_everyone()
        main_process_result(accelerator, lambda: write_complete_marker(destination, signature, progress))

    stop = progress.optimizer_updates >= args.max_train_steps
    for epoch in range(progress.epoch, args.num_epochs):
        if stop:
            break
        generator.manual_seed(args.seed + epoch)
        dataloader.set_epoch(epoch)
        skip = progress.batches_consumed if epoch == progress.epoch else 0
        epoch_loader = accelerator.skip_first_batches(dataloader, skip) if skip else dataloader
        for data in epoch_loader:
            with accelerator.accumulate(model):
                loss = model({}, inputs=data) if dataset.load_from_cache else model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.batches_consumed += 1
                progress.microbatches += 1
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    progress.optimizer_updates += 1
                    model_logger.on_step_end(accelerator, model, args.save_steps, loss=loss)
                    if accelerator.is_main_process:
                        metrics = accelerator.unwrap_model(model).pipe.video_sft_metrics
                        print(json.dumps({"event": "video_sft_update", "optimizer_updates": progress.optimizer_updates,
                                          "microbatches": progress.microbatches, "epoch": epoch,
                                          "loss": float(loss.detach()),
                                          **{k: float(v) if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}}), flush=True)
                    if args.save_steps and progress.optimizer_updates % args.save_steps == 0:
                        save_state(f"training-state-{progress.optimizer_updates}")
                    stop = progress.optimizer_updates >= args.max_train_steps
            if stop:
                break
        if not stop:
            progress.epoch = epoch + 1
            progress.batches_consumed = 0
        if args.save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch)
        if stop:
            break
    save_state(f"training-state-final-{progress.optimizer_updates}")
    model_logger.on_training_end(accelerator, model, args.save_steps)
    if accelerator.is_main_process:
        print(json.dumps({"event": "video_sft_finished", "optimizer_updates": progress.optimizer_updates,
                          "microbatches": progress.microbatches, "human_quality_verified": False,
                          "convergence_verified": False}), flush=True)


def video_sft_parser():
    parser = add_video_size_config(add_general_config(argparse.ArgumentParser(description=__doc__)))
    parser.add_argument("--weights_root", type=str)
    parser.add_argument("--caption_version", type=str)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--anchor_lambda", type=float, choices=(0.0, 0.1, 1.0), default=0.1)
    parser.add_argument("--training_cfg_scale", type=float, choices=(1.0,), default=1.0)
    parser.add_argument("--initialize_model_on_cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_steps", type=int, choices=(2, 100, 300), default=100,
                        help="2 is an interface/gradient smoke only; 100/300 are explicit reviewed or source-label pilot diagnostics")
    parser.add_argument("--data_scope", choices=("interface", "reviewed_train", "source_label_train"), default="interface",
                        help="Interface permits only two updates; reviewed_train/source_label_train require their separate verified train-only provenance")
    parser.add_argument("--resume_training_state", type=str)
    parser.add_argument("--cpu_self_test", action="store_true")
    parser.add_argument("--temporal_policy", choices=("native_down", "nearest24_repeat_last124"), default="native_down",
                        help="Explicit verified short-window resampling and last-frame padding; never implicit new motion evidence")
    parser.set_defaults(data_file_keys="video,references", extra_inputs="references", task="sft:train",
                        lora_base_model="dit", lora_target_modules=LORA_TARGETS, lora_rank=32,
                        learning_rate=1e-5, num_frames=124, height=1280, width=704,
                        use_gradient_checkpointing=True, num_epochs=3, save_steps=100)
    return parser


def cpu_self_test():
    from diffsynth.diffusion import FlowMatchScheduler
    from diffsynth.diffusion.base_pipeline import BasePipeline

    class TinyDiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.qkv_proj = torch.nn.Linear(2, 2, bias=False)
            self.attn.out_proj = torch.nn.Linear(2, 2, bias=False)
            self.mlp = torch.nn.Module()
            self.mlp.fc1 = torch.nn.Linear(2, 2, bias=False)
            self.mlp.fc2 = torch.nn.Linear(2, 2, bias=False)
            self.audio_patch_proj = torch.nn.Linear(2, 2)
            self.audio_out = torch.nn.Linear(2, 2)

        def forward(self, video_latents):
            value = video_latents.movedim(1, -1)
            value = self.attn.out_proj(self.attn.qkv_proj(value))
            return self.mlp.fc2(self.mlp.fc1(value)).movedim(-1, 1)

    class TinyPipe(BasePipeline):
        def __init__(self):
            super().__init__(device="cpu", torch_dtype=torch.float32)
            self.scheduler = FlowMatchScheduler("MiniMax-H3")
            self.dit = TinyDiT()
            self.text_encoder = torch.nn.Linear(2, 2)
            self.video_vae = torch.nn.Linear(2, 2)
            self.audio_vae = None
            self.units = []
            self.model_fn = lambda dit, video_latents, **kwargs: (dit(video_latents), video_latents.new_empty((2, 32, 0)))

        @property
        def scheduler_audio(self):
            raise AssertionError("Audio scheduler access is forbidden")

    def forbid(*args):
        raise AssertionError("Forbidden frozen/audio forward")

    torch.manual_seed(123)
    pipe = TinyPipe()
    model = H3VideoOnlyTrainingModule(caption_version="cpu-test", task="sft", pipe=pipe, lora_rank=2)
    for module in (pipe.dit.audio_patch_proj, pipe.dit.audio_out, pipe.text_encoder, pipe.video_vae):
        module.register_forward_pre_hook(forbid)
    base_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    # Nonzero student delta makes the anchor check meaningful, not zero-init tautology.
    for name, parameter in pipe.dit.named_parameters():
        if "lora_B" in name:
            torch.nn.init.constant_(parameter, 0.02)
    shared = {"input_latents": torch.randn(1, 2, 2, 2, 2), "audio_latents": torch.empty(2, 32, 0)}
    noise = torch.randn_like(shared["input_latents"])
    loss0 = video_flow_match_loss(pipe, shared, {}, anchor_lambda=0.0, timestep_id=100, noise=noise)
    loss1 = video_flow_match_loss(pipe, shared, {}, anchor_lambda=1.0, timestep_id=100, noise=noise)
    metrics = pipe.video_sft_metrics
    assert metrics["anchor"] > 0 and metrics["anchor_active"]
    torch.testing.assert_close(loss1 - loss0, metrics["anchor"] * metrics["scheduler_weight"])
    loss1.backward()
    grads = [p.grad for p in model.trainable_modules()]
    assert grads and any(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=1e-5)
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in base_before:
            torch.testing.assert_close(parameter, base_before[name], rtol=0, atol=0)
    import copy
    from diffsynth.utils.lora.minimax_h3 import MiniMaxH3LoRALoader
    fused = copy.deepcopy(pipe.dit.get_base_model())
    # Build a plain original base, then verify deployment LoRA has identical alpha/r semantics.
    plain = TinyDiT()
    plain.load_state_dict({name.replace(".base_layer.", "."): value for name, value in fused.state_dict().items()
                           if ".lora_" not in name})
    exported = export_diffsynth_lora(model.export_trainable_state_dict(model.state_dict(), "pipe.dit."), 64, 2)
    MiniMaxH3LoRALoader(torch_dtype=torch.float32).fuse_lora_to_base_model(plain, exported)
    with torch.no_grad():
        torch.testing.assert_close(plain(shared["input_latents"]), pipe.dit(shared["input_latents"]), rtol=1e-5, atol=1e-6)
    low = video_flow_match_loss(pipe, shared, {}, anchor_lambda=1.0, timestep_id=950, noise=noise)
    assert torch.isfinite(low) and not pipe.video_sft_metrics["anchor_active"]
    for invalid in ({"input_audio": "x.wav"}, {"references": {"type": "audio", "audio": "x.wav"}},
                    {"references": {"type": "video_audio", "video": "x.mp4"}},
                    {"input_image": "first-frame.png"}, {"audio_latents": torch.zeros(2, 32, 1)}):
        try:
            validate_video_inputs(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Forbidden audio/first-frame input was accepted")
    try:
        validate_caption_cache((shared, {}, {}), "cpu-test")
    except ValueError:
        pass
    else:
        raise AssertionError("Legacy cache was accepted")
    from h3_video_only import H3VideoOnlyPipeline, VideoOnlyPacked, enable_video_only_dit
    from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT
    actual_pipe = H3VideoOnlyPipeline(torch_dtype=torch.float32)
    actual_pipe.dit = enable_video_only_dit(MiniMaxH3DiT(
        num_layers=2, token_refiner_num_layers=1, hidden_size=48, num_attention_heads=4,
        attention_head_dim=12, ffn_hidden_size=64, time_embed_dim=32, time_embed_hidden_size=48,
        adaln_out_features=864, final_adaln_out_features=96, rope_inv_freq_len=2,
    ).float())
    actual_model = H3VideoOnlyTrainingModule(caption_version="cpu-test", task="sft:train", pipe=actual_pipe, lora_rank=2)
    actual_pipe.dit.audio_patch_proj.register_forward_pre_hook(forbid)
    actual_pipe.dit.final_layer.audio_out.register_forward_pre_hook(forbid)
    for name, parameter in actual_pipe.dit.named_parameters():
        if "lora_B" in name:
            torch.nn.init.constant_(parameter, 0.002)
    actual_target = torch.randn(1, 24, 2, 4, 4)
    actual_text = torch.randn(3, 5120)
    packed = VideoOnlyPacked().process(actual_pipe, prompt_embeds=actual_text, video_latents=actual_target,
        audio_latents=torch.empty(2, 32, 0), text_token_tags=torch.ones(3, dtype=torch.long),
        ref_blocks=[dict(kind="image", latent_t=1, latent_h=4, latent_w=4, ref_audio_t=0)])["packed"]
    actual_loss = video_flow_match_loss(actual_pipe,
        {"input_latents": actual_target, "ref_visual_anchor": torch.randn(4, 96), "use_gradient_checkpointing": True},
        {"prompt_embeds": actual_text, "packed": packed}, anchor_lambda=0.1, timestep_id=100)
    actual_loss.backward()
    assert actual_pipe.video_sft_metrics["anchor"] > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actual_model.trainable_modules())
    assert all(p.grad is None for p in actual_model.parameters() if not p.requires_grad)

    from diffsynth.core.vram.layers import AutoWrappedModule
    from types import SimpleNamespace
    offload_pipe = TinyPipe()
    for name in ("text_encoder", "video_vae"):
        wrapped = AutoWrappedModule(torch.nn.Linear(2, 2), offload_dtype=torch.float32, offload_device="cpu",
            onload_dtype=torch.float32, onload_device="cpu", preparing_dtype=torch.float32,
            preparing_device="cpu", computation_dtype=torch.float32, computation_device="cpu")
        wrapped.vram_management_enabled = True
        setattr(offload_pipe, name, wrapped)
    offload_pipe.vram_management_enabled = True
    validate_cache_offload(SimpleNamespace(pipe=offload_pipe))
    from unittest.mock import patch
    # The upstream CPU pathway lacks torch.cpu.empty_cache; no CUDA operation is mocked.
    with patch.object(torch.cpu, "empty_cache", lambda: None, create=True):
        offload_pipe.load_models_to_device(["video_vae"])
        offload_pipe.video_vae(torch.ones(1, 2))
        assert offload_pipe.video_vae.state == 2 and offload_pipe.text_encoder.state == 0
        offload_pipe.load_models_to_device(["text_encoder"])
        offload_pipe.text_encoder(torch.ones(1, 2))
        assert offload_pipe.video_vae.state == 0 and offload_pipe.text_encoder.state == 2
        offload_pipe.load_models_to_device([])
        assert offload_pipe.video_vae.state == offload_pipe.text_encoder.state == 0
    assert "\n    model.to(" not in __import__("inspect").getsource(launch_video_cache)
    import ast
    cache_tree = ast.parse(__import__("inspect").getsource(launch_video_cache))
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in ("prepare", "to") for node in ast.walk(cache_tree))

    import tempfile

    def synthetic_formal_export(root, train_samples=1):
        from dance_caption_fusion import CpuChecks, PROTOCOL, source_record, object_sha
        root.mkdir(exist_ok=True)
        version = PROTOCOL + ":CPU-boundary-simulation"
        manifest_path, review_path = root / "manifest.json", root / "review.json"
        records, rewrites, reviews, exported, h3_rows = [], [], [], [], []

        def save(path, value):
            path.write_text(json.dumps(value), encoding="utf-8")
            return source_record(path)

        for index, partition in enumerate(["train"] * train_samples + ["val"]):
            sid = "synthetic" + str(index)
            media = {}
            for key in ("audio", "image", "paired_video"):
                path = root / (sid + "." + key)
                path.write_bytes(("Temporary CPU boundary simulation only: " + key + sid).encode())
                media[key] = source_record(path)
            meta = root / (sid + ".meta.json")
            media["meta"] = save(meta, dict(video=media["paired_video"]["path"], natural_output_frames=81, target_fps=16))
            record = CpuChecks().record()
            record["time_scope"].update(paired_normalized_natural_frames=81, paired_normalized_fps=16,
                                       training_audio_window_seconds=[0, 5.0625])
            material = {"temporary_CPU_contract_simulation": sid}
            record.update(id=sid, split=partition, source_group=sid, rewrite_input=material,
                          rewrite_input_sha256=object_sha(material), audio_window_seconds=[0, 5.0625],
                          audio_path=media["audio"]["path"], image_path=media["image"]["path"],
                          paired_video_path=media["paired_video"]["path"], provenance=media)
            records.append(record)
        manifest_source = save(manifest_path, dict(protocol=PROTOCOL, records=records, blocked_ids=[],
                                                   synthetic_cpu_contract_fixture=True))
        for record in records:
            target = CpuChecks().target()
            rewrite = dict(id=record["id"], split=record["split"], manifest_sha256=manifest_source["sha256"],
                           rewrite_input_sha256=record["rewrite_input_sha256"], caption_version=version, target=target)
            rewrite_source = save(root / (record["id"] + ".rewrite.json"), rewrite)
            approval = dict(id=record["id"], facts_approved=True, narrative_approved=True,
                window_verified=True, conflicts_resolved=True, reviewed_audio_window_seconds=[0, 5.0625],
                reviewer="CPU-contract-simulation-NOT-human", window_evidence="Temporary mocked approval only",
                notes="No real target export or human approval claim", rewrite_sha256=rewrite_source["sha256"])
            reviews.append(approval)
            rewrites.append((rewrite, rewrite_source))
        review_source = save(review_path, dict(protocol=PROTOCOL, records=reviews,
                                               manifest_sha256=manifest_source["sha256"], synthetic_cpu_contract_fixture=True))
        for record, approval, (rewrite, rewrite_source) in zip(records, reviews, rewrites):
            media, target = record["provenance"], rewrite["target"]
            provenance = dict(manifest=manifest_source, review=review_source, rewrite=rewrite_source)
            exported.append(dict(id=record["id"], split=record["split"], source_group=record["source_group"],
                rewrite_source=rewrite_source, audio_source=media["audio"], image_source=media["image"],
                paired_video_source=media["paired_video"], source_record_manifest=manifest_source,
                review_source=review_source, human_review=approval, trained_caption_approved=True,
                approval_provenance=provenance, source_time_scope=record["time_scope"],
                caption_version=version, audio_window_seconds=[0, 5.0625], video_window_seconds=[0, 5.0625], **target))
            h3_rows.append(dict(id=record["id"], split=record["split"], source_group=record["source_group"],
                motion_caption=target["motion_caption"], caption_version=version, video=media["paired_video"]["path"],
                references=[dict(type="image", image=media["image"]["path"])], video_window_seconds=[0, 5.0625],
                trained_caption_approved=True, provenance=dict(**provenance, image=media["image"], paired_video=media["paired_video"])))
        outputs = {}
        for name, rows in (("h3_video.json", h3_rows), ("h3_video_train.json", h3_rows[:train_samples]),
                           ("h3_video_val.json", h3_rows[train_samples:])):
            outputs[name] = save(root / name, rows)
        for name in ("train.jsonl", "val.jsonl", "omni.jsonl"):
            outputs[name] = save(root / name, {"synthetic_cpu_contract_fixture": True})
        sidecar = dict(protocol=PROTOCOL, complete=True, fixture_only=False, records=exported,
            manifest=manifest_source, review=review_source, caption_version=version,
            source_code=source_record(Path(__file__).with_name("dance_caption_fusion.py")), counts={"train": train_samples, "val": 1},
            outputs=outputs, h3_metadata_policy=dict(train_only="h3_video_train.json", val_only="h3_video_val.json",
                combined="h3_video.json", combined_use="exchange_and_audit_only", combined_direct_cache_or_training_forbidden=True))
        save(root / "source_sidecar.json", sidecar)
        return root / "h3_video_train.json", version

    formal_negative_cases = ("fixture", "row_fixture", "row_forbidden", "review_fixture", "review_forbidden", "bool_approval", "bool_review",
        "combined", "val", "val_split", "export_sha", "review_sha", "rewrite_sha", "changed_media",
        "changed_meta", "missing_export", "incomplete_export", "wrong_caption", "bool_window", "missing_evidence", "wrong_policy")
    for case in formal_negative_cases:
        with tempfile.TemporaryDirectory(prefix="h3-formal-gate-CPU-only-") as directory:
            root = Path(directory)
            metadata, version = synthetic_formal_export(root)
            sidecar_path = root / "source_sidecar.json"
            sidecar = json.loads(sidecar_path.read_text())
            rows = json.loads(metadata.read_text())
            dataset = VideoCaptionDataset(base_path=str(root), metadata_path=str(metadata), caption_version=version)
            config = SimpleNamespace(dataset_base_path=str(root), dataset_metadata_path=str(metadata),
                                     caption_version=version, max_train_steps=100, data_scope="reviewed_train",
                                     temporal_policy="nearest24_repeat_last124")
            assert validate_data_boundary(dataset, config)["data_provenance"]["approval_chain_verified"]
            if case == "fixture":
                sidecar["fixture_only"] = True
            elif case in ("review_fixture", "review_forbidden", "bool_review", "bool_window", "missing_evidence"):
                review = json.loads((root / "review.json").read_text())
                approval = review["records"][0]
                if case == "review_fixture":
                    review["fixture_only"] = True
                elif case == "review_forbidden":
                    review["formal_reuse_forbidden"] = True
                elif case == "bool_review":
                    approval["facts_approved"] = 1
                elif case == "bool_window":
                    approval["reviewed_audio_window_seconds"] = [False, 5.0625]
                else:
                    approval["window_evidence"] = ""
                (root / "review.json").write_text(json.dumps(review))
                from dance_caption_fusion import source_record
                sidecar["review"] = source_record(root / "review.json")
            elif case in ("bool_approval", "val_split", "row_fixture", "row_forbidden"):
                if case in ("row_fixture", "row_forbidden"):
                    rows[0]["fixture_only" if case == "row_fixture" else "formal_reuse_forbidden"] = True
                else:
                    rows[0]["trained_caption_approved" if case == "bool_approval" else "split"] = 1 if case == "bool_approval" else "val"
                metadata.write_text(json.dumps(rows))
                from dance_caption_fusion import source_record
                sidecar["outputs"][metadata.name] = source_record(metadata)
            elif case in ("combined", "val"):
                config.dataset_metadata_path = str(root / ("h3_video.json" if case == "combined" else "h3_video_val.json"))
            elif case == "export_sha":
                sidecar["outputs"]["h3_video_train.json"]["sha256"] = "wrong"
            elif case == "review_sha":
                sidecar["review"]["sha256"] = "wrong"
            elif case == "rewrite_sha":
                sidecar["records"][0]["rewrite_source"]["sha256"] = "wrong"
            elif case in ("changed_media", "changed_meta"):
                path = root / ("synthetic0.paired_video" if case == "changed_media" else "synthetic0.meta.json")
                path.write_bytes(b"changed temporary media")
            elif case == "missing_export":
                sidecar["outputs"].pop("h3_video_val.json")
            elif case == "incomplete_export":
                sidecar["complete"] = 1
            elif case == "wrong_caption":
                config.caption_version += "-changed"
            elif case == "wrong_policy":
                sidecar["h3_metadata_policy"]["combined_direct_cache_or_training_forbidden"] = False
            sidecar_path.write_text(json.dumps(sidecar))
            argv = ["h3_video_sft.py", "--task", "sft:data_process", "--weights_root", str(root),
                    "--caption_version", config.caption_version, "--dataset_base_path", str(root),
                    "--dataset_metadata_path", config.dataset_metadata_path, "--output_path", str(root / "not-created"),
                    "--max_train_steps", "100", "--data_scope", "reviewed_train", "--temporal_policy", "nearest24_repeat_last124"]
            with patch.object(sys, "argv", argv), patch("accelerate.Accelerator", return_value=accelerate.Accelerator(cpu=True)), \
                    patch(__name__ + ".H3VideoOnlyTrainingModule") as builder:
                try:
                    main()
                except ValueError:
                    pass
                else:
                    raise AssertionError("Formal boundary accepted negative case: " + case)
                assert not builder.called, case
            assert not (root / "not-created").exists()

    with tempfile.TemporaryDirectory(prefix="h3-formal-main-positive-CPU-") as directory:
        root = Path(directory)
        metadata, version = synthetic_formal_export(root)
        for budget in (100, 300):
            argv = ["h3_video_sft.py", "--task", "sft:data_process", "--weights_root", str(root),
                    "--caption_version", version, "--dataset_base_path", str(root),
                    "--dataset_metadata_path", str(metadata), "--output_path", str(root / "not-created"),
                    "--max_train_steps", str(budget), "--data_scope", "reviewed_train", "--temporal_policy", "nearest24_repeat_last124"]
            class ExpectedModelBoundary(Exception):
                pass
            with patch.object(sys, "argv", argv), patch("accelerate.Accelerator", return_value=accelerate.Accelerator(cpu=True)), \
                    patch(__name__ + ".H3VideoOnlyTrainingModule", side_effect=ExpectedModelBoundary) as builder:
                try:
                    main()
                except ExpectedModelBoundary:
                    pass
                else:
                    raise AssertionError("Legal formal source did not reach the model construction boundary")
                assert builder.call_count == 1
            assert not (root / "not-created").exists()
            default_argv = list(argv)
            position = default_argv.index("--data_scope")
            del default_argv[position:position + 2]
            with patch.object(sys, "argv", default_argv), patch("accelerate.Accelerator", return_value=accelerate.Accelerator(cpu=True)), \
                    patch(__name__ + ".H3VideoOnlyTrainingModule") as builder:
                try:
                    main()
                except ValueError:
                    pass
                else:
                    raise AssertionError("Formal raw 100/300 budget accepted implicit default scope")
                assert not builder.called

    class TinyCache(torch.utils.data.Dataset):
        load_from_cache = True

        def __init__(self, root):
            self.cached_data = [str(path) for path in sorted(Path(root).glob("*.pth"))]

        def __len__(self):
            return 8

        def __getitem__(self, index):
            latents = torch.full((1, 2, 2, 2, 2), index / 8.0)
            return ({"input_latents": latents, "audio_latents": torch.empty(2, 32, 0),
                     "h3_video_cache": {"format": CACHE_FORMAT, "caption_version": formal_version, "caption_sha256": "test"}},
                    {"prompt_embeds": torch.ones(1, 2), "packed": {
                        "audio_pos": torch.empty(0, dtype=torch.long), "token_tags": torch.zeros(1, dtype=torch.long)}}, {})

    def run_tiny(output, max_steps, resume=None):
        accelerate.utils.set_seed(123)
        tiny = H3VideoOnlyTrainingModule(caption_version=formal_version, task="sft:train", pipe=TinyPipe(), lora_rank=2)
        accelerator = accelerate.Accelerator(cpu=True, gradient_accumulation_steps=2)
        config = SimpleNamespace(
            enable_model_cpu_offload=False, enable_optimizer_cpu_offload=False, customized_optimizer=None,
            learning_rate=1e-5, weight_decay=0.01, dataset_num_workers=0, caption_version=formal_version,
            dataset_base_path=str(Path(temporary) / "cache"), dataset_metadata_path=None, anchor_lambda=0.1, lora_rank=2, lora_alpha=64,
            weights_root=str(Path(temporary) / "weights"), gradient_accumulation_steps=2, seed=123, resume_training_state=resume,
            output_path=output, max_train_steps=max_steps, num_epochs=2, save_steps=2,
            data_scope="reviewed_train", temporal_policy="nearest24_repeat_last124",
        )
        logger = ModelLogger(output, remove_prefix_in_ckpt="pipe.dit.")
        launch_video_training(accelerator, TinyCache(config.dataset_base_path), tiny, logger, config)
        return {n: p.detach().clone() for n, p in tiny.named_parameters() if p.requires_grad}

    with tempfile.TemporaryDirectory(prefix="h3-video-sft-cpu-") as temporary:
        fixture = Path(temporary)
        cache_root = fixture / "cache"
        cache_root.mkdir()
        formal_metadata, formal_version = synthetic_formal_export(fixture / "formal-export", train_samples=8)
        formal_provenance = formal_training_provenance(formal_metadata, formal_version)
        for index, row in enumerate(json.loads(formal_metadata.read_text())):
            value = TinyCache(cache_root)[index]
            value[0]["h3_video_cache"].update(sample_id=row["id"], data_scope="reviewed_train",
                caption_sha256=hashlib.sha256(teacher_video_prompt(row).encode()).hexdigest(),
                data_provenance_sha256=hashlib.sha256(json.dumps(formal_provenance, sort_keys=True).encode()).hexdigest())
            torch.save(value, cache_root / f"{index}.pth")
        cache_signature = {"caption_version": formal_version, "data_scope": "reviewed_train", "dataset_size": 8,
            "dataset_metadata_path": str(formal_metadata),
            "data_provenance": formal_provenance, "temporal_policy": "nearest24_repeat_last124",
            "sources": {"metadata_sha256": file_sha256(formal_metadata), "code": {
                Path(__file__).name: file_sha256(__file__),
                "h3_video_only.py": file_sha256(Path(__file__).with_name("h3_video_only.py")),
        }}}
        write_complete_marker(cache_root, cache_signature)
        for budget in (100, 300):
            argv = ["h3_video_sft.py", "--task", "sft:train", "--weights_root", str(fixture),
                    "--caption_version", formal_version, "--dataset_base_path", str(cache_root),
                    "--output_path", str(fixture / "positive-not-created"), "--max_train_steps", str(budget),
                    "--data_scope", "reviewed_train", "--temporal_policy", "nearest24_repeat_last124"]
            with patch.object(sys, "argv", argv), patch("accelerate.Accelerator", return_value=accelerate.Accelerator(cpu=True)), \
                    patch(__name__ + ".H3VideoOnlyTrainingModule", side_effect=ExpectedModelBoundary) as builder:
                try:
                    main()
                except ExpectedModelBoundary:
                    pass
                else:
                    raise AssertionError("Legal formal cache did not reach the model construction boundary")
                assert builder.call_count == 1
            default_argv = list(argv)
            position = default_argv.index("--data_scope")
            del default_argv[position:position + 2]
            with patch.object(sys, "argv", default_argv), patch("accelerate.Accelerator", return_value=accelerate.Accelerator(cpu=True)), \
                    patch(__name__ + ".H3VideoOnlyTrainingModule") as builder:
                try:
                    main()
                except ValueError:
                    pass
                else:
                    raise AssertionError("Formal cache 100/300 budget accepted implicit default scope")
                assert not builder.called
        assert not (fixture / "positive-not-created").exists()
        model_root = fixture / "weights"
        (model_root / "transformer").mkdir(parents=True)
        (model_root / "transformer/config.json").write_text('{"fixture": true}')
        (model_root / "transformer/model.safetensors").write_bytes(b"fixture-weights")
        (model_root / "model_index.json").write_text('{"fixture": true}')
        class CacheRunnerFixture(DiffusionTrainingModule):
            def __init__(self):
                super().__init__()
                self.pipe = offload_pipe

            def to(self, *args, **kwargs):
                raise AssertionError("Cache runner must never move the entire model")

            def forward(self, data):
                return data

        class RawCacheFixture(TinyCache):
            load_from_cache = False

            def __init__(self):
                self.data = json.loads(formal_metadata.read_text())

        cache_args = SimpleNamespace(
            dataset_base_path=str(fixture), dataset_metadata_path=str(formal_metadata), weights_root=str(model_root),
            caption_version=formal_version, anchor_lambda=0.1, learning_rate=1e-5, weight_decay=0.01,
            lora_rank=2, lora_alpha=64, gradient_accumulation_steps=1, seed=123,
            dataset_num_workers=0, output_path=str(fixture / "cache-output"), max_train_steps=100,
            data_scope="reviewed_train", temporal_policy="nearest24_repeat_last124",
        )
        with patch.object(torch.cpu, "empty_cache", lambda: None, create=True):
            launch_video_cache(accelerate.Accelerator(cpu=True), RawCacheFixture(), CacheRunnerFixture(), cache_args)
        assert len(list((fixture / "cache-output").rglob("*.pth"))) == 8
        verify_complete_marker(fixture / "cache-output")
        baseline = run_tiny(str(Path(temporary) / "baseline"), 3)
        resumed_output = str(Path(temporary) / "resume")
        run_tiny(resumed_output, 2)
        resumed = run_tiny(resumed_output, 3, str(Path(resumed_output) / "training-state-final-2"))
        for name in baseline:
            torch.testing.assert_close(baseline[name], resumed[name], rtol=0, atol=0)
        checkpoint = Path(resumed_output) / "training-state-final-2"
        for key in ("optimizer.bin", "scheduler.bin", "custom_checkpoint_0.pkl", COMPLETE_MARKER):
            assert (checkpoint / key).is_file(), key
        marker = verify_complete_marker(checkpoint)

        def must_reject(operation):
            try:
                operation()
            except ValueError:
                return
            raise AssertionError("Changed source or partial checkpoint was accepted")

        progress_state = marker["progress"]
        for changes in ({"batches_consumed": -1}, {"batches_consumed": True},
                        {"optimizer_updates": progress_state["microbatches"] + 1}):
            invalid = dict(progress_state, **changes)
            must_reject(lambda invalid=invalid: TrainingProgress(marker["signature"]).load_state_dict(invalid))
        must_reject(lambda: TrainingProgress(marker["signature"]).load_state_dict({"signature": marker["signature"]}))
        loaded_progress = TrainingProgress(marker["signature"])
        loaded_progress.load_state_dict(progress_state)
        loaded_progress.validate_budget(4, 2, 100)
        must_reject(lambda: loaded_progress.validate_budget(0, 2, 100))
        must_reject(lambda: loaded_progress.validate_budget(4, 0, 100))
        must_reject(lambda: loaded_progress.validate_budget(4, 2, 1))
        changed = json.loads(json.dumps(marker["signature"]))
        changed["sources"]["code"][Path(__file__).name] = "changed-source"
        must_reject(lambda: verify_complete_marker(checkpoint, changed))
        partial = fixture / "partial"
        partial.mkdir()
        (partial / "optimizer.bin").write_bytes(b"partial")
        must_reject(lambda: verify_complete_marker(partial))
        original_cache = (cache_root / "0.pth").read_bytes()
        (cache_root / "0.pth").write_bytes(b"modified-cache")
        must_reject(lambda: run_tiny(str(fixture / "reject-cache"), 3, str(checkpoint)))
        (cache_root / "0.pth").write_bytes(original_cache)
        original_optimizer = (checkpoint / "optimizer.bin").read_bytes()
        (checkpoint / "optimizer.bin").write_bytes(b"changed-optimizer")
        must_reject(lambda: verify_complete_marker(checkpoint))
        (checkpoint / "optimizer.bin").write_bytes(original_optimizer)
        (model_root / "transformer/model.safetensors").write_bytes(b"changed-model")
        must_reject(lambda: run_tiny(str(fixture / "reject-model"), 3, str(checkpoint)))
    cached_negative_cases = ("interface_upgrade", "scope_override", "wrong_caption", "wrong_policy",
                             "wrong_dataset_size", "sample_scope", "sample_binding", "val_sample", "source_review_drift")
    for case in cached_negative_cases:
        with tempfile.TemporaryDirectory(prefix="h3-cache-gate-CPU-only-") as directory:
            root = Path(directory)
            metadata, version = synthetic_formal_export(root / "export")
            provenance = formal_training_provenance(metadata, version)
            cache = root / "cache"
            cache.mkdir()
            row = json.loads(metadata.read_text())[0]
            info = dict(format=CACHE_FORMAT, caption_version=version, sample_id=row["id"], data_scope="reviewed_train",
                caption_sha256=hashlib.sha256(teacher_video_prompt(row).encode()).hexdigest(),
                data_provenance_sha256=hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest())
            value = ({"input_latents": torch.zeros(1, 2, 2, 2, 2), "audio_latents": torch.empty(2, 32, 0),
                      "h3_video_cache": info}, {"prompt_embeds": torch.ones(1, 2), "packed": {
                      "audio_pos": torch.empty(0, dtype=torch.long), "token_tags": torch.zeros(1, dtype=torch.long)}}, {})
            signature = dict(data_scope="reviewed_train", data_provenance=provenance, dataset_size=1,
                dataset_metadata_path=str(metadata), caption_version=version, temporal_policy="nearest24_repeat_last124",
                sources=dict(metadata_sha256=file_sha256(metadata), code={path.name: file_sha256(path)
                    for path in (Path(__file__), Path(__file__).with_name("h3_video_only.py"))}))
            budget, scope, policy, caption = 100, "reviewed_train", "nearest24_repeat_last124", version
            if case in ("interface_upgrade", "scope_override"):
                signature["data_scope"] = info["data_scope"] = "interface"
                signature["data_provenance"] = dict(protocol="h3-interface-only-v1", formal_reuse_forbidden=True)
                if case == "scope_override":
                    scope = "interface"
            elif case == "wrong_caption":
                caption += "-changed"
            elif case == "wrong_policy":
                policy = "native_down"
            elif case == "wrong_dataset_size":
                signature["dataset_size"] = 2
            elif case == "sample_scope":
                info["data_scope"] = "interface"
            elif case == "sample_binding":
                info["data_provenance_sha256"] = "wrong"
            elif case == "val_sample":
                info["sample_id"] = "synthetic1"
            elif case == "source_review_drift":
                (root / "export/review.json").write_text("{}")
            torch.save(value, cache / "0.pth")
            write_complete_marker(cache, signature)
            argv = ["h3_video_sft.py", "--task", "sft:train", "--weights_root", str(root),
                    "--caption_version", caption, "--dataset_base_path", str(cache),
                    "--output_path", str(root / "not-created"), "--max_train_steps", str(budget), "--temporal_policy", policy]
            if scope:
                argv += ["--data_scope", scope]
            with patch.object(sys, "argv", argv), patch("accelerate.Accelerator", return_value=accelerate.Accelerator(cpu=True)), \
                    patch(__name__ + ".H3VideoOnlyTrainingModule") as builder:
                try:
                    main()
                except ValueError:
                    pass
                else:
                    raise AssertionError("Cached boundary accepted negative case: " + case)
                assert not builder.called, case
            assert not (root / "not-created").exists()

    with tempfile.TemporaryDirectory(prefix="h3-interface-scope-CPU-only-") as directory:
        root = Path(directory)
        metadata = root / "fixture.json"
        metadata.write_text(json.dumps([dict(id="fixture", video=str(root / "unused.mp4"),
            references=[dict(type="image", image=str(root / "unused.png"))], split="train", fixture_only=True,
            formal_reuse_forbidden=True, caption_version="interface-only", motion_caption="The dancer bends and extends the arms.")]))
        dataset = VideoCaptionDataset(base_path=str(root), metadata_path=str(metadata), caption_version="interface-only")
        args = SimpleNamespace(dataset_base_path=str(root), dataset_metadata_path=str(metadata),
                               caption_version="interface-only", max_train_steps=2, data_scope="interface",
                               temporal_policy="native_down")
        boundary = validate_data_boundary(dataset, args)
        assert boundary["data_scope"] == "interface" and boundary["data_provenance"]["formal_reuse_forbidden"]
        cache = root / "cache"
        cache.mkdir()
        value = ({"input_latents": torch.zeros(1, 2, 2, 2, 2), "audio_latents": torch.empty(2, 32, 0),
                  "h3_video_cache": dict(format=CACHE_FORMAT, caption_version="interface-only", caption_sha256="fixture",
                  data_scope="interface", data_provenance_sha256=hashlib.sha256(json.dumps(boundary["data_provenance"], sort_keys=True).encode()).hexdigest())},
                 {"prompt_embeds": torch.ones(1, 2), "packed": {"audio_pos": torch.empty(0, dtype=torch.long),
                                                               "token_tags": torch.zeros(1, dtype=torch.long)}}, {})
        torch.save(value, cache / "0.pth")
        signature = dict(**boundary, dataset_size=1, dataset_metadata_path=str(metadata), caption_version="interface-only",
            temporal_policy="native_down", sources=dict(code={path.name: file_sha256(path)
                for path in (Path(__file__), Path(__file__).with_name("h3_video_only.py"))}))
        interface_marker = write_complete_marker(cache, signature)
        assert interface_marker["scope"] == "interface"
        cached = VideoCaptionDataset(base_path=str(cache), metadata_path=None, caption_version="interface-only")
        cached_args = SimpleNamespace(**vars(args))
        cached_args.dataset_base_path, cached_args.dataset_metadata_path = str(cache), None
        assert validate_data_boundary(cached, cached_args) == boundary
        for budget in (100, 300):
            cached_args.max_train_steps, cached_args.data_scope = budget, "reviewed_train"
            try:
                validate_data_boundary(cached, cached_args)
            except ValueError:
                pass
            else:
                raise AssertionError("Interface cache was upgraded to formal budget")
            args.max_train_steps = budget
            try:
                validate_data_boundary(dataset, args)
            except ValueError:
                pass
            else:
                raise AssertionError("Interface raw metadata was upgraded to formal budget")

    import dance_caption_fusion as fusion
    import omni_dance_sft as omni
    # These weak-label gate fixtures are synthetic and NOT reviewed/real media.
    with tempfile.TemporaryDirectory(prefix="h3-source-label-synthetic-not-reviewed-CPU-") as directory:
        root = Path(directory)
        parent_metadata = root / "omni.jsonl"
        parent_sidecar = root / "source_label_sidecar.json"
        parent_metadata.write_text("synthetic parent reader is mocked; not real Omni training data")
        parent_sidecar.write_text("{}")
        version = "synthetic-source-label-not-reviewed"
        records, omni_rows, h3_rows = [], [], []
        manifest_path = root / "manifest.json"
        for partition, count in (("train", 512), ("val", 128)):
            image, video, meta_path = (root / (partition + suffix) for suffix in (".png", ".mp4", ".meta.json"))
            image.write_bytes(b"synthetic not reviewed image")
            video.write_bytes(b"synthetic not reviewed video")
            meta_path.write_text(json.dumps(dict(natural_output_frames=81, target_fps=16, target_frames=81,
                tail_padding_frames=0, video=str(video), first_frame=str(image))))
            media = {key: fusion.source_record(path) for key, path in (("image", image), ("paired_video", video), ("meta", meta_path))}
            for index in range(count):
                sid = f"synthetic-{partition}-{index}"
                caption = "The dancer bends and extends the arms."
                records.append(dict(id=sid, source_group=sid, split=partition, provenance=media,
                                    paired_video_path=str(video), image_path=str(image)))
                omni_rows.append(dict(id=sid, source_group=sid, split=partition, caption_version=version,
                                      messages=[dict(role="assistant", content=caption)]))
        manifest_path.write_text(json.dumps(dict(records=records, synthetic_not_reviewed=True)))
        manifest_source = fusion.source_record(manifest_path)
        parent_source = fusion.source_record(parent_metadata)
        for row, record in zip(omni_rows, records):
            media = record["provenance"]
            h3_rows.append(dict(id=row["id"], source_group=row["source_group"], split=row["split"],
                motion_caption=row["messages"][-1]["content"], caption_version=version, video=media["paired_video"]["path"],
                references=[dict(type="image", image=media["image"]["path"])], video_window_seconds=[0, 5.0625],
                trained_caption_approved=False, label_quality="source_labels_unreviewed",
                provenance=dict(parent_omni_metadata=parent_source, parent_manifest=manifest_source, **media)))
        outputs = {}
        for partition in ("train", "val"):
            path = root / ("h3_video_" + partition + ".json")
            path.write_text(json.dumps([row for row in h3_rows if row["split"] == partition]))
            outputs[path.name] = fusion.source_record(path)
        sidecar_path = root / "source_label_h3_sidecar.json"
        sidecar = dict(protocol="h3-source-label-train-provenance-v1", complete=True, caption_version=version,
            authorization=omni.SOURCE_LABEL_AUTHORIZATION, label_quality="source_labels_unreviewed",
            approval_chain_verified=False, human_review_completed=False, semantic_approval=False,
            original_source_window_mapping="unknown", frozen_omni_validator_source=fusion.source_record(omni.__file__),
            source_code=fusion.source_record(__file__), parent_omni_metadata=parent_source,
            parent_omni_sidecar=fusion.source_record(parent_sidecar), parent_caption_version=version,
            parent_manifest=manifest_source, outputs=outputs, counts=dict(train=512, val=128),
            h3_metadata_policy=dict(train_only="h3_video_train.json", val_only="h3_video_val.json",
                                   combined_direct_cache_or_training_forbidden=True))
        sidecar_path.write_text(json.dumps(sidecar))
        train = [row for row in omni_rows if row["split"] == "train"]
        val = [row for row in omni_rows if row["split"] == "val"]
        parent = dict(metadata=parent_source, sidecar=fusion.source_record(parent_sidecar), caption_version=version,
                      manifest=manifest_source, approval_chain_verified=False)
        metadata = root / "h3_video_train.json"
        with patch.object(omni, "validate_data_boundary", return_value=(omni_rows, train, val, parent)) as parent_reader, \
                patch.object(fusion, "load_manifest", side_effect=fusion.read_json):
            provenance = source_label_training_provenance(metadata, version)
            assert provenance["approval_chain_verified"] is False and provenance["label_quality"] == "source_labels_unreviewed"
            config = parent_reader.call_args.args[0]
            assert (config.data_scope, config.smoke_steps, config.min_epochs, config.max_epochs) == ("source_label_pilot", 0, 3, 3)
            for changes in ({"label_quality": "reviewed"}, {"fixture_only": True}, {"formal_reuse_forbidden": True},
                            {"semantic_approval": True}, {"human_review_completed": True}, {"approval_chain_verified": 0}):
                sidecar_path.write_text(json.dumps(dict(sidecar, **changes)))
                must_reject(lambda: source_label_training_provenance(metadata, version))
            sidecar_path.write_text(json.dumps(sidecar))
            must_reject(lambda: source_label_training_provenance(root / "h3_video_val.json", version))
            must_reject(lambda: source_label_training_provenance(root / "h3_video.json", version))
            original_rows = json.loads(metadata.read_text())
            for changes in ({"motion_caption": "Wrong label"}, {"fixture_only": True}, {"trained_caption_approved": True}, {"split": "val"}):
                changed_rows = [dict(original_rows[0], **changes)] + original_rows[1:]
                metadata.write_text(json.dumps(changed_rows))
                changed_sidecar = json.loads(json.dumps(sidecar))
                changed_sidecar["outputs"][metadata.name] = fusion.source_record(metadata)
                sidecar_path.write_text(json.dumps(changed_sidecar))
                must_reject(lambda: source_label_training_provenance(metadata, version))
            metadata.write_text(json.dumps(original_rows))
            sidecar_path.write_text(json.dumps(sidecar))
        # Scope/cache binding is exercised with mocked provenance, never by upgrading a fixture.
        one_row = h3_rows[0]
        one_metadata = root / "one-synthetic.json"
        one_metadata.write_text(json.dumps([one_row]))
        mock_provenance = dict(provenance, metadata=fusion.source_record(one_metadata), train_ids=[one_row["id"]])
        cache = root / "weak-cache"
        cache.mkdir()
        info = dict(format=CACHE_FORMAT, caption_version=version, data_scope="source_label_train", sample_id=one_row["id"],
            caption_sha256=hashlib.sha256(teacher_video_prompt(one_row).encode()).hexdigest(),
            data_provenance_sha256=hashlib.sha256(json.dumps(mock_provenance, sort_keys=True).encode()).hexdigest())
        value = ({"input_latents": torch.zeros(1, 2, 2, 2, 2), "h3_video_cache": info},
                 {"prompt_embeds": torch.ones(1, 2), "packed": {"audio_pos": torch.empty(0, dtype=torch.long),
                 "token_tags": torch.zeros(1, dtype=torch.long)}}, {})
        signature = dict(data_scope="source_label_train", data_provenance=mock_provenance, dataset_size=1,
            dataset_metadata_path=str(one_metadata), caption_version=version, temporal_policy="nearest24_repeat_last124",
            sources=dict(metadata_sha256=file_sha256(one_metadata), code={path.name: file_sha256(path)
                for path in (Path(__file__), Path(__file__).with_name("h3_video_only.py"))}))
        torch.save(value, cache / "0.pth")
        write_complete_marker(cache, signature)
        dataset = VideoCaptionDataset(base_path=str(cache), metadata_path=None, caption_version=version)
        args = SimpleNamespace(dataset_base_path=str(cache), dataset_metadata_path=None, caption_version=version,
                               max_train_steps=100, data_scope="source_label_train", temporal_policy="nearest24_repeat_last124")
        with patch(__name__ + ".source_label_training_provenance", return_value=mock_provenance), \
                patch(__name__ + ".formal_training_provenance", side_effect=AssertionError("Weak labels must not invoke review approval")):
            assert validate_data_boundary(dataset, args)["data_provenance"]["approval_chain_verified"] is False
            for changes in ({"sample_id": "synthetic-val-0"}, {"caption_sha256": "wrong"},
                            {"data_scope": "interface"}, {"data_provenance_sha256": "wrong"}):
                with patch(__name__ + ".verify_complete_marker", return_value=dict(scope="source_label_train", signature=signature, files={"0.pth": {}})), \
                        patch.object(torch, "load", return_value=({**value[0], "h3_video_cache": dict(info, **changes)}, value[1], {})):
                    must_reject(lambda: validate_data_boundary(dataset, args))
            args.data_scope = "interface"
            must_reject(lambda: validate_data_boundary(dataset, args))
            args.max_train_steps = 2
            must_reject(lambda: validate_data_boundary(dataset, args))
        assert video_sft_parser().get_default("data_scope") == "interface"

    print(json.dumps({"cpu_self_test": "passed", "source_label_synthetic_not_reviewed_gate": True,
                      "source_label_bad_label_fixture_val_combined_scope_cache_rejected": True,
                      "formal_metadata_negative_cases": list(formal_negative_cases),
                      "formal_cache_negative_cases": list(cached_negative_cases),
                      "formal_models_loading_boundary_rejections": True, "formal_export_approval_chain_positive": True,
                      "interface_scope_fixed_two_updates": True, "interface_cache_budget_upgrade_rejected": True,
                      "formal_100_300_main_model_boundary_positive": True, "formal_budget_explicit_reviewed_scope_required": True,
                      "cached_sample_scope_caption_approval_binding": True,
                      "actual_lora_backward": True, "frozen_base_unchanged": True,
                      "soft_anchor_relative_check": True, "audio_forbidden_hooks": True,
                      "actual_optimizer_checkpoint_resume_parity": True,
                      "real_tiny_H3_lora_anchor_checkpoint_backward": True,
                      "deployment_lora_alpha_rank_parity": True,
                      "reject_changed_source_cache_model_checkpoint_and_partial": True,
                      "strict_progress_fields_types_local_loader_and_budget": True,
                      "restored_progress_matches_completion_marker": True,
                      "actual_cpu_offload_wrapper_state_switch": True, "cache_runner_whole_model_to_calls": 0,
                      "scope": "CPU fake-pipe resume plus real tiny H3 blocks; no full-weight/GPU/ZeRO claim"}), flush=True)


def main():
    parser = video_sft_parser()
    args = parser.parse_args()
    if args.cpu_self_test:
        cpu_self_test()
        return
    if not args.weights_root or not args.caption_version:
        parser.error("--weights_root and --caption_version are required")
    unsupported = ("model_paths", "model_id_with_origin_paths", "fp8_models", "offload_models", "quant_options",
                   "template_model_id_or_path", "lora_checkpoint", "preset_lora_path", "preset_lora_model",
                   "resume_from_checkpoint", "trainable_models")
    for key in unsupported:
        if getattr(args, key):
            parser.error(f"--{key} is not supported; use original weights and full --resume_training_state")
    if args.task not in ("sft", "sft:data_process", "sft:train"):
        parser.error("Only video-only SFT tasks are allowed")
    if args.extra_inputs != "references" or set(args.data_file_keys.split(",")) != {"video", "references"}:
        parser.error("Use only --extra_inputs references and --data_file_keys video,references")
    if args.num_frames % 17 != 5:
        parser.error("--num_frames must be 17n+5 (default 124)")
    if args.dataset_repeat != 1 or args.num_epochs <= 0 or args.gradient_accumulation_steps <= 0:
        parser.error("Use dataset_repeat=1 and positive epoch/accumulation budgets")
    output = Path(args.output_path)
    if not output.parent.is_dir() or (output.exists() and not args.resume_training_state):
        parser.error("Use a new output directory with an existing parent; existing runs need a full training-state resume")
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    accelerate.utils.set_seed(args.seed)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters),
                        accelerate.utils.InitProcessGroupKwargs(timeout=timedelta(hours=2))],
    )
    video_operator = UnifiedDataset.default_video_operator(
        base_path=args.dataset_base_path, max_pixels=args.max_pixels, height=args.height, width=args.width,
        height_division_factor=32, width_division_factor=32, num_frames=args.num_frames,
        time_division_factor=17, time_division_remainder=5, frame_rate=24, fix_frame_rate=True,
    )
    if args.temporal_policy == 'nearest24_repeat_last124':
        if args.num_frames != 124:
            parser.error('Repeat-last124 fixes --num_frames at124')
        video_operator = WindowAlignedVideoLoader(
            num_frames=124, time_division_factor=17, time_division_remainder=5,
            frame_processor=ImageCropAndResize(args.height, args.width, args.max_pixels, 32, 32),
            frame_rate=24, fix_frame_rate=True,
        )
    dataset = VideoCaptionDataset(
        base_path=args.dataset_base_path, metadata_path=args.dataset_metadata_path, repeat=1,
        data_file_keys=args.data_file_keys.split(","), main_data_operator=video_operator,
        caption_version=args.caption_version, temporal_policy=args.temporal_policy,
        special_operator_map={"references": VisualReferenceLoader(
            base_path=args.dataset_base_path, height=args.height, width=args.width,
            max_pixels=args.max_pixels, num_frames=args.num_frames, frame_rate=24,
        )},
    )
    if (args.task == "sft:train") != dataset.load_from_cache:
        parser.error("sft:train requires new cached inputs; sft/data_process requires teacher-caption metadata")
    main_process_result(accelerator, lambda: validate_data_boundary(dataset, args))
    model = H3VideoOnlyTrainingModule(
        args.weights_root, caption_version=args.caption_version, task=args.task,
        device=accelerator.device if args.task == "sft:data_process" else "cpu" if args.initialize_model_on_cpu else accelerator.device,
        extra_inputs=args.extra_inputs, lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        anchor_lambda=args.anchor_lambda, training_cfg_scale=args.training_cfg_scale,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
    )
    logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
                         state_dict_converter=lambda state: export_diffsynth_lora(state, args.lora_alpha, args.lora_rank),
                         enable_tensorboard_log=args.enable_tensorboard_log, enable_csv_log=args.enable_csv_log,
                         enable_swanlab_log=args.enable_swanlab_log, swanlab_project=args.swanlab_project,
                         enable_wandb_log=args.enable_wandb_log, wandb_project=args.wandb_project)
    if args.task == "sft:data_process":
        launch_video_cache(accelerator, dataset, model, args)
    else:
        launch_video_training(accelerator, dataset, model, logger, args)


if __name__ == "__main__":
    main()
