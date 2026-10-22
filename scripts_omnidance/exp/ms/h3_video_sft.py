"""Opt-in caption/reference video-only H3 SFT; legacy AV training is untouched."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path('/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930')))

import accelerate
import torch
from peft import LoraConfig, get_peft_model

from diffsynth.core import UnifiedDataset
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


class VideoCaptionDataset(UnifiedDataset):
    def __init__(self, *args, caption_version, **kwargs):
        super().__init__(*args, **kwargs)
        self.caption_version = caption_version
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
                               "sample_id": str(data.get("id", data.get("sample_id", "")))},
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
    marker = {"format": "h3-video-complete-v1", "signature": signature, "files": files,
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
    return {"caption_version": args.caption_version, "dataset_base_path": str(Path(args.dataset_base_path).resolve()),
            "dataset_metadata_path": args.dataset_metadata_path, "dataset_size": len(dataset),
            "anchor_lambda": args.anchor_lambda, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha, "weights_root": args.weights_root,
            "world_size": accelerator.num_processes, "accumulation": args.gradient_accumulation_steps,
            "seed": args.seed, "lora_target_modules": getattr(args, "lora_target_modules", LORA_TARGETS),
            "gradient_checkpointing": getattr(args, "use_gradient_checkpointing", True),
            "gradient_checkpointing_offload": getattr(args, "use_gradient_checkpointing_offload", False),
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
            value = model.transfer_data_to_device(value, "cpu")
            with (folder / f"{index}.pth").open("xb") as stream:
                torch.save(value, stream)
    model.pipe.load_models_to_device([])
    accelerator.wait_for_everyone()
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
        if state["signature"] != self.signature:
            raise ValueError("Resume requires matching caption/data/recipe/world size/accumulation")
        self.__dict__.update(state)


def launch_video_training(accelerator, dataset, model, model_logger, args):
    if args.enable_model_cpu_offload or args.enable_optimizer_cpu_offload or args.customized_optimizer:
        raise ValueError("Use Accelerator/ZeRO offload configuration; legacy per-layer/custom optimizer is unsupported")
    signature = training_signature(accelerator, dataset, args)
    if args.resume_training_state:
        main_process_result(accelerator, lambda: verify_complete_marker(args.resume_training_state, signature))
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
        accelerator.load_state(args.resume_training_state)
    model_logger.num_steps = progress.optimizer_updates
    if accelerator.is_main_process:
        save_training_args(args)
        print(json.dumps({"event": "video_sft_budget", "effective_batch": accelerator.num_processes * args.gradient_accumulation_steps,
                          "optimizer_updates": progress.optimizer_updates, "max_train_steps": args.max_train_steps,
                          "budget_scope": "interface_smoke_only" if args.max_train_steps == 2 else "human_gated_pilot_diagnostic",
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
                        help="2 is an interface/gradient smoke only; 100/300 are human-gated pilot diagnostics, not full-data training")
    parser.add_argument("--resume_training_state", type=str)
    parser.add_argument("--cpu_self_test", action="store_true")
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

    class TinyCache(torch.utils.data.Dataset):
        load_from_cache = True

        def __init__(self, root):
            self.cached_data = [str(path) for path in sorted(Path(root).glob("*.pth"))]

        def __len__(self):
            return 8

        def __getitem__(self, index):
            latents = torch.full((1, 2, 2, 2, 2), index / 8.0)
            return ({"input_latents": latents, "audio_latents": torch.empty(2, 32, 0),
                     "h3_video_cache": {"format": CACHE_FORMAT, "caption_version": "cpu-test", "caption_sha256": "test"}},
                    {"prompt_embeds": torch.ones(1, 2), "packed": {
                        "audio_pos": torch.empty(0, dtype=torch.long), "token_tags": torch.zeros(1, dtype=torch.long)}}, {})

    def run_tiny(output, max_steps, resume=None):
        accelerate.utils.set_seed(123)
        tiny = H3VideoOnlyTrainingModule(caption_version="cpu-test", task="sft:train", pipe=TinyPipe(), lora_rank=2)
        accelerator = accelerate.Accelerator(cpu=True, gradient_accumulation_steps=2)
        config = SimpleNamespace(
            enable_model_cpu_offload=False, enable_optimizer_cpu_offload=False, customized_optimizer=None,
            learning_rate=1e-5, weight_decay=0.01, dataset_num_workers=0, caption_version="cpu-test",
            dataset_base_path=str(Path(temporary) / "cache"), dataset_metadata_path=None, anchor_lambda=0.1, lora_rank=2, lora_alpha=64,
            weights_root=str(Path(temporary) / "weights"), gradient_accumulation_steps=2, seed=123, resume_training_state=resume,
            output_path=output, max_train_steps=max_steps, num_epochs=2, save_steps=2,
        )
        logger = ModelLogger(output, remove_prefix_in_ckpt="pipe.dit.")
        launch_video_training(accelerator, TinyCache(config.dataset_base_path), tiny, logger, config)
        return {n: p.detach().clone() for n, p in tiny.named_parameters() if p.requires_grad}

    with tempfile.TemporaryDirectory(prefix="h3-video-sft-cpu-") as temporary:
        fixture = Path(temporary)
        cache_root = fixture / "cache"
        cache_root.mkdir()
        for index in range(8):
            (cache_root / f"{index}.pth").write_bytes(f"fixture-{index}".encode())
        write_complete_marker(cache_root, {"caption": "cpu-test", "sources": {"code": {
            Path(__file__).name: file_sha256(__file__),
            "h3_video_only.py": file_sha256(Path(__file__).with_name("h3_video_only.py")),
        }}})
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

        cache_args = SimpleNamespace(
            dataset_base_path=str(cache_root), dataset_metadata_path=None, weights_root=str(model_root),
            caption_version="cpu-test", anchor_lambda=0.1, learning_rate=1e-5, weight_decay=0.01,
            lora_rank=2, lora_alpha=64, gradient_accumulation_steps=1, seed=123,
            dataset_num_workers=0, output_path=str(fixture / "cache-output"),
        )
        with patch.object(torch.cpu, "empty_cache", lambda: None, create=True):
            launch_video_cache(accelerate.Accelerator(cpu=True), TinyCache(cache_root), CacheRunnerFixture(), cache_args)
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

        changed = json.loads(json.dumps(marker["signature"]))
        changed["sources"]["code"][Path(__file__).name] = "changed-source"
        must_reject(lambda: verify_complete_marker(checkpoint, changed))
        partial = fixture / "partial"
        partial.mkdir()
        (partial / "optimizer.bin").write_bytes(b"partial")
        must_reject(lambda: verify_complete_marker(partial))
        (cache_root / "0.pth").write_bytes(b"modified-cache")
        must_reject(lambda: run_tiny(str(fixture / "reject-cache"), 3, str(checkpoint)))
        (cache_root / "0.pth").write_bytes(b"fixture-0")
        original_optimizer = (checkpoint / "optimizer.bin").read_bytes()
        (checkpoint / "optimizer.bin").write_bytes(b"changed-optimizer")
        must_reject(lambda: verify_complete_marker(checkpoint))
        (checkpoint / "optimizer.bin").write_bytes(original_optimizer)
        (model_root / "transformer/model.safetensors").write_bytes(b"changed-model")
        must_reject(lambda: run_tiny(str(fixture / "reject-model"), 3, str(checkpoint)))
    print(json.dumps({"cpu_self_test": "passed", "actual_lora_backward": True, "frozen_base_unchanged": True,
                      "soft_anchor_relative_check": True, "audio_forbidden_hooks": True,
                      "actual_optimizer_checkpoint_resume_parity": True,
                      "real_tiny_H3_lora_anchor_checkpoint_backward": True,
                      "deployment_lora_alpha_rank_parity": True,
                      "reject_changed_source_cache_model_checkpoint_and_partial": True,
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
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    video_operator = UnifiedDataset.default_video_operator(
        base_path=args.dataset_base_path, max_pixels=args.max_pixels, height=args.height, width=args.width,
        height_division_factor=32, width_division_factor=32, num_frames=args.num_frames,
        time_division_factor=17, time_division_remainder=5, frame_rate=24, fix_frame_rate=True,
    )
    dataset = VideoCaptionDataset(
        base_path=args.dataset_base_path, metadata_path=args.dataset_metadata_path, repeat=1,
        data_file_keys=args.data_file_keys.split(","), main_data_operator=video_operator,
        caption_version=args.caption_version,
        special_operator_map={"references": VisualReferenceLoader(
            base_path=args.dataset_base_path, height=args.height, width=args.width,
            max_pixels=args.max_pixels, num_frames=args.num_frames, frame_rate=24,
        )},
    )
    if (args.task == "sft:train") != dataset.load_from_cache:
        parser.error("sft:train requires new cached inputs; sft/data_process requires teacher-caption metadata")
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
