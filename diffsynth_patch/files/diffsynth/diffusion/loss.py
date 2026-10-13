from .base_pipeline import BasePipeline
import torch


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    if "lora" in inputs:
        # Image-to-LoRA models need to load lora here.
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"]) * inputs.get("noise_scale", 1.0)
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def FlowMatchSFTAudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # audio
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler.add_noise(inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = pipe.scheduler.training_target(inputs["audio_input_latents"], audio_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    if inputs.get("audio_input_latents") is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio
    return loss


def FlowMatchSFTMiniMaxH3AudioVideoLoss(pipe: BasePipeline, training_cfg_scale: float = 1.0, audio_loss_weight: float = 1.0, inputs_nega: dict | None = None, training_cfg_mode: str = "inverse", **inputs):
    if training_cfg_mode not in ("inverse", "direct"):
        raise ValueError("training_cfg_mode must be 'inverse' or 'direct'")
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep_video = pipe.scheduler.timesteps[timestep_id].to(dtype=torch.float32, device=pipe.device)
    timestep_audio = pipe.scheduler_audio.timesteps[timestep_id].to(dtype=torch.float32, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep_video)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep_video)

    import os as _os
    if _os.environ.get("OD_H3_MUSIC") == "1":  # local patch (OmniDance M1a): MERT features for this clip -> h3_music.STATE
        import sys as _sys; _sys.path.insert(0, "/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar")
        import h3_music as _MU
        _MU.set_from_inputs(inputs)
    clean_audio = _os.environ.get("OD_H3_CLEAN_AUDIO", "0") == "1" and "audio_input_latents" in inputs
    if clean_audio:
        # local patch (OmniDance): music is a clean, time-aligned input condition (retake-style mask 0), not generated
        inputs["audio_latents"] = inputs["audio_input_latents"]
        inputs["input_latents_audio"] = inputs["audio_input_latents"]
        inputs["denoise_mask_audio"] = torch.zeros_like(inputs["audio_input_latents"])
    elif "audio_input_latents" in inputs:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler_audio.add_noise(inputs["audio_input_latents"], audio_noise, timestep_audio)
        training_target_audio = pipe.scheduler_audio.training_target(inputs["audio_input_latents"], audio_noise, timestep_audio)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    if _os.environ.get("OD_H3_DPO") == "1":  # local patch (OmniDance W1b): music-contrastive Diffusion-DPO (bidirectional)
        import sys as _sys; _sys.path.insert(0, "/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar")
        import h3_dpo as _DPO
        return _DPO.dpo_loss(pipe, models, inputs, timestep_video, timestep_audio, training_target)
    tf_frames = None
    if _os.environ.get("OD_H3_TF", "0") == "1":
        # local patch (OmniDance H3-AR, Causal Forcing stage 1): teacher forcing with block-causal attention.
        # frames before the target block = clean GT context (mask 0 -> timestep 1.0); loss only on the target block.
        import sys as _sys, random as _random
        _sys.path.insert(0, "/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3ar")
        import h3_causal as _HC
        if not getattr(_HC, "_installed", False):
            _HC.install(); _HC._installed = True
        lat = inputs["input_latents"]; T = lat.shape[2]
        n_rv = 0 if inputs.get("ref_visual_anchor") is None else inputs["ref_visual_anchor"].shape[0]
        n_ra = 0 if inputs.get("ref_audio_anchor") is None else inputs["ref_audio_anchor"].shape[0]
        plan = _HC.make_plan(inputs["packed"], n_rv, n_ra, T, block=int(_os.environ.get("OD_H3_BLOCK", "6")), lookahead=float(_os.environ.get("OD_H3_LOOKAHEAD", "8")))
        _HC.STATE["plan"] = plan  # stays set through backward (checkpoint recompute)
        plan_nega = None
        if training_cfg_scale > 1.0 and inputs_nega and "packed" in inputs_nega:  # local patch: block-causal plan for the uncond branch
            plan_nega = _HC.make_plan(inputs_nega["packed"], n_rv, n_ra, T, block=int(_os.environ.get("OD_H3_BLOCK", "6")), lookahead=float(_os.environ.get("OD_H3_LOOKAHEAD", "8")))
        if _os.environ.get("OD_H3_CD", "0") == "1":  # Causal Forcing++ stage 2 (causal consistency distillation)
            import h3_cd as _CD
            return _CD.cd_loss(pipe, models, inputs, plan)
        f0, f1 = plan["bounds"][_random.randint(0, len(plan["bounds"]) - 1)]
        m = torch.ones_like(lat); m[:, :, :f0] = 0
        inputs["input_latents_video"] = lat
        inputs["denoise_mask_video"] = m
        tf_frames = (f0, f1)
        # history mode (OmniDance W5 on H3): tf = clean context; mix = with prob OD_H3_DF_P all context frames get one
        # sigma_h~U(0,OD_H3_DF_MAX); df = every context block gets its own sigma~U(0,1). Context timestep = 1 - sigma.
        hist = _os.environ.get("OD_H3_HIST", "tf")
        segs = []
        if hist == "mix" and f0 > 0 and _random.random() < float(_os.environ.get("OD_H3_DF_P", "0.3")):
            segs = [(0, f0, _random.uniform(0.0, float(_os.environ.get("OD_H3_DF_MAX", "0.2"))))]
        elif hist == "df":
            segs = [(b0, b1, _random.random()) for b0, b1 in plan["bounds"] if b1 <= f0]
        if segs:
            ctx = lat.clone(); ctx_t = torch.ones(T, dtype=torch.float32, device=lat.device)
            for b0, b1, s in segs:
                ctx[:, :, b0:b1] = (1 - s) * lat[:, :, b0:b1] + s * torch.randn_like(lat[:, :, b0:b1])
                ctx_t[b0:b1] = 1 - s
            inputs["input_latents_video"] = ctx
            inputs["od_ctx_t"] = ctx_t
        if not getattr(_HC, "_logged", False):
            print(f"[OD H3-TF] plan blocks={plan['bounds']} groups={len(plan['groups'])} target={tf_frames}", flush=True); _HC._logged = True
    if training_cfg_scale > 1.0:
        if not inputs_nega:
            raise ValueError(
                "MiniMax-H3 CFG-aware training requires unconditional inputs. "
                "When using split training, rebuild the data cache with the same "
                "--training_cfg_scale value."
            )
        inputs_uncond = {**inputs, **inputs_nega}
        inputs_uncond["use_gradient_checkpointing"] = False
        inputs_uncond["use_gradient_checkpointing_offload"] = False
        _plan_posi = None
        if tf_frames is not None:  # local patch (OmniDance): uncond forward under its own block-causal plan
            assert plan_nega is not None, "TF + training_cfg_scale needs a cache with unconditional packed inputs"
            _plan_posi = _HC.STATE["plan"]; _HC.STATE["plan"] = plan_nega
        with torch.no_grad():
            noise_pred_uncond, noise_pred_audio_uncond = pipe.model_fn(
                **models, **inputs_uncond,
                timestep_video=timestep_video, timestep_audio=timestep_audio,
            )
        if _plan_posi is not None:
            _HC.STATE["plan"] = _plan_posi

    noise_pred, noise_pred_audio = pipe.model_fn(
        **models, **inputs,
        timestep_video=timestep_video, timestep_audio=timestep_audio,
    )

    if training_cfg_scale > 1.0:
        if training_cfg_mode == "direct":
            uncond = noise_pred_uncond.detach().float()
            training_target = uncond + training_cfg_scale * (training_target.float() - uncond)
            if "audio_input_latents" in inputs and not clean_audio:
                uncond_audio = noise_pred_audio_uncond.detach().float()
                training_target_audio = uncond_audio + training_cfg_scale * (training_target_audio.float() - uncond_audio)
        else:
            # The checkpoint's conditional prediction has CFG distilled into it.
            # Rearrange the CFG equation to recover the raw velocity fitted to the
            # standard flow-matching target, using the current model as the teacher.
            noise_pred = (noise_pred + (training_cfg_scale - 1.0) * noise_pred_uncond) / training_cfg_scale
            noise_pred_audio = (noise_pred_audio + (training_cfg_scale - 1.0) * noise_pred_audio_uncond) / training_cfg_scale

    if tf_frames is not None:
        loss = torch.nn.functional.mse_loss(noise_pred[:, :, tf_frames[0]:tf_frames[1]].float(), training_target[:, :, tf_frames[0]:tf_frames[1]].float())
    else:
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep_video)
    if "audio_input_latents" in inputs and not clean_audio:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler_audio.training_weight(timestep_audio) * audio_loss_weight
        loss = loss + loss_audio
    if _os.environ.get("OD_H3_MUSIC") == "1":  # local patch (OmniDance M2): MoE router load-balancing term (0 for M1a)
        loss = loss + _MU.aux_loss()
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            denom = sigma_ - sigma
            denom = torch.sign(denom) * torch.clamp(denom.abs(), min=1e-6)
            target = (latents_ - inputs_shared["latents"]) / denom
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
