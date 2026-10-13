"""Shared pieces for OmniDance DMD distillation (E5).

One frozen S4 DiT (real score) + two switchable LoRA adapters on every Linear of the DiT:
  mode "real" -> base only (teacher), "gen" -> base + gen LoRA (student), "fake" -> base + fake LoRA (critic).
Flow matching convention (DiffSynth FlowMatch): x_t = (1-s) x0 + s eps, v = eps - x0, timestep = s*1000.
"""
import importlib.util
import math
import os
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn

REPO = "/apdcephfs_gy4/share_302533218/cedricnie/diff"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

NEG_PROMPT = None  # filled from inference script default


def load_infer_module():
    spec = importlib.util.spec_from_file_location(
        "od_infer", f"{REPO}/examples/wanvideo/model_inference/Wan2.2-TI2V-5B-Audio.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- dual LoRA
class DualLoRALinear(nn.Module):
    mode = "real"  # class-level switch shared by all instances

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        dev = base.weight.device
        self.lora = nn.ModuleDict()
        for name in ("gen", "fake"):
            a = nn.Linear(base.in_features, rank, bias=False, device=dev, dtype=torch.float32)
            b = nn.Linear(rank, base.out_features, bias=False, device=dev, dtype=torch.float32)
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            nn.init.zeros_(b.weight)
            self.lora[name] = nn.Sequential(a, b)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        y = self.base(x)
        m = DualLoRALinear.mode
        if m == "real":
            return y
        a, b = self.lora[m]
        return y + nn.functional.linear(nn.functional.linear(x, a.weight.to(x.dtype)), b.weight.to(x.dtype)) * self.scale


def inject_dual_lora(dit: nn.Module, rank: int, alpha: float, skip=("head", "patch_embedding", "time_embedding", "time_projection", "text_embedding")):
    targets = []
    for name, mod in dit.named_modules():
        for cname, child in mod.named_children():
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and not any(full.startswith(s) for s in skip):
                targets.append((mod, cname, full))
    for parent, cname, _ in targets:
        setattr(parent, cname, DualLoRALinear(getattr(parent, cname), rank, alpha))
    return [t[2] for t in targets]


def lora_params(dit, which):
    return [p for n, p in dit.named_parameters() if f".lora.{which}." in n]


def lora_state_dict(dit, which):
    return {n: p.detach().cpu() for n, p in dit.named_parameters() if f".lora.{which}." in n}


def load_lora_state(dit, sd):
    own = dict(dit.named_parameters())
    for n, v in sd.items():
        own[n].data.copy_(v.to(own[n].device, own[n].dtype))


# ---------------------------------------------------------------- pipeline
def build_pipe(ckpt_path, device, model_root="/dev/shm/od/Wan-AI/Wan2.2-TI2V-5B", mert_path="/dev/shm/od/MERT-v1-330M"):
    m = load_infer_module()
    global NEG_PROMPT
    NEG_PROMPT = m.DEFAULT_NEGATIVE_PROMPT
    args = Namespace(model_path=model_root, ckpt_path=ckpt_path, lora_path=None, ckpt_type="onlyCKPT",
                     device=device, audio_sink_bias_mode="auto", mert_path=mert_path, low_vram=False,
                     vram_limit_gb=None, lora_alpha=1.0)
    pipe = m.load_pipeline(args)
    pipe.dit.requires_grad_(False)
    pipe.dit.eval()
    return pipe, m


def audio_scale_table(peak=2.0, floor=1.0, hold=5, decay_end=10, n=50, shift=5.0, infer_mod=None):
    """sigma -> audio guidance scale, mirroring the 50-step text_first inference schedule."""
    sig = []
    for i in range(n):
        s = 1.0 - i / n
        sig.append(shift * s / (1 + (shift - 1) * s))
    sc = [infer_mod.compute_text_only_audio_scale(peak, i, n, hold_steps=hold, decay_end_step=decay_end, floor_scale=floor)
          for i in range(n)]
    return torch.tensor(sig), torch.tensor(sc)


def shift_sigma(u, shift=5.0):
    return shift * u / (1 + (shift - 1) * u)


@torch.no_grad()
def prepare_sample(pipe, prompt, image, audio_embeds, height, width, num_frames, seed):
    """Run pipeline units -> (shared, posi, nega). audio_embeds None => no audio condition."""
    pipe.scheduler.set_timesteps(50, denoising_strength=1.0, shift=5.0)
    inputs_posi = {"prompt": prompt, "vap_prompt": " ", "tea_cache_l1_thresh": None, "tea_cache_model_id": "", "num_inference_steps": 50}
    inputs_nega = {"negative_prompt": NEG_PROMPT, "negative_vap_prompt": " ", "tea_cache_l1_thresh": None, "tea_cache_model_id": "", "num_inference_steps": 50}
    inputs_shared = {
        "input_image": image, "end_image": None, "input_video": None, "denoising_strength": 1.0,
        "control_video": None, "reference_image": None, "camera_control_direction": None, "camera_control_speed": 1 / 54,
        "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        "vace_video": None, "vace_video_mask": None, "vace_reference_image": None, "vace_scale": 1.0,
        "seed": seed, "rand_device": "cpu", "height": height, "width": width, "num_frames": num_frames,
        "cfg_scale": 2.0, "cfg_merge": False, "sigma_shift": 5.0, "motion_bucket_id": None, "longcat_video": None,
        "tiled": True, "tile_size": (30, 52), "tile_stride": (15, 26), "sliding_window_size": None, "sliding_window_stride": None,
        "input_audio": None, "audio_embeds": audio_embeds, "audio_sample_rate": 16000,
        "s2v_pose_video": None, "s2v_pose_latents": None, "motion_video": None, "animate_pose_video": None,
        "animate_face_video": None, "animate_inpaint_video": None, "animate_mask_video": None, "vap_video": None,
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)
    return inputs_shared, inputs_posi, inputs_nega


@torch.no_grad()
def prepare_sample_wave(pipe, prompt, image, waveform, sample_rate, height, width, num_frames, seed):
    """Like prepare_sample but encodes raw audio with MERT (waveform None => no audio)."""
    if waveform is None:
        return prepare_sample(pipe, prompt, image, None, height, width, num_frames, seed)
    pipe.scheduler.set_timesteps(50, denoising_strength=1.0, shift=5.0)
    feats = pipe.mert_audio_encoder.get_audio_feats_per_inference(
        waveform, sample_rate, batch_frames=num_frames, fps=16, dtype=pipe.torch_dtype, device=pipe.device)
    ae = feats[0] if isinstance(feats, list) else feats
    return prepare_sample(pipe, prompt, image, ae, height, width, num_frames, seed)


def velocity(pipe, shared, cond, x_t, sigma, mode, grad=False):
    """Flow velocity prediction under the given LoRA mode."""
    DualLoRALinear.mode = mode
    x = x_t.clone()
    x[:, :, 0:1] = shared["first_frame_latents"]
    t = torch.tensor([float(sigma) * 1000.0], device=x.device, dtype=pipe.torch_dtype)
    kw = dict(shared)
    kw.update(cond)
    kw["latents"] = x
    kw["use_gradient_checkpointing"] = grad
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    with torch.set_grad_enabled(grad):
        v = pipe.model_fn(**models, **kw, timestep=t)
    # NOTE: mode is left as-is; callers must re-set DualLoRALinear.mode before backward (checkpoint recompute).
    return v


def cond_views(posi, nega):
    """text_first CFG branches: i (neg text, no audio), ti (text, no audio), tia (text+audio)."""
    tia = dict(posi)
    ti = dict(posi); ti.pop("audio_embeds", None)
    i = dict(nega); i.pop("audio_embeds", None)
    return i, ti, tia


STUDENT_SIGMAS = [shift_sigma(x) for x in (1.0, 0.75, 0.5, 0.25)]


def student_rollout(pipe, shared, posi, noise, exit_k, grad_last=True, gen=None):
    """Run student steps 0..exit_k; last step optionally with grad. Returns x0 prediction at step exit_k."""
    x = noise
    x0 = None
    for k in range(exit_k + 1):
        s = STUDENT_SIGMAS[k]
        with_grad = grad_last and k == exit_k
        v = velocity(pipe, shared, posi, x, s, "gen", grad=with_grad)
        x0 = x - s * v
        if k < exit_k:
            s_next = STUDENT_SIGMAS[k + 1]
            eps = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) if gen is not None else torch.randn_like(x)
            x = ((1 - s_next) * x0 + s_next * eps).detach()
    x0 = x0.clone()
    x0[:, :, 0:1] = shared["first_frame_latents"]
    return x0


@torch.no_grad()
def student_sample(pipe, shared, posi, seed=0, steps=4):
    gen = torch.Generator(device=shared["latents"].device).manual_seed(seed)
    noise = torch.randn(shared["latents"].shape, generator=gen, device=shared["latents"].device, dtype=shared["latents"].dtype)
    return student_rollout(pipe, shared, posi, noise, steps - 1, grad_last=False, gen=gen)


def decode(pipe, latents):
    video = pipe.vae.decode(latents, device=pipe.device, tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
    return pipe.vae_output_to_video(video)
