"""ode_gen.py: E6-style stage 1 for H3-AR (Causal-Forcing / Self-Forcing route, 10-11 after TF failed).
Bidirectional teacher (original H3 DiT + optional LoRA, cfg 1, official shift 12, N teacher steps) integrates the PF-ODE
from noise on cached training conditions (split-training cache .pth: prompt embeds, packed, ref anchors, clean music latents),
and stores the trajectory at the student's sigmas -> causal ODE regression / Self-Forcing DMD later.

python ode_gen.py --list <cache paths txt> --out <dir> [--lora f] [--steps 50] [--student_steps 4] --shard g --nshard N
Output <out>/<cache-id>.pt: {"sigmas": [s_0..s_{K-1}, 0], "x": (K+1, C, T, H, W) bf16  (x[0] = noise, x[K] = teacher x0),
                             "src": cache path, "lora": ...}
"""
import argparse, glob, os, sys, time
import torch

p = argparse.ArgumentParser()
p.add_argument("--list", required=True); p.add_argument("--out", required=True)
p.add_argument("--lora", default=""); p.add_argument("--w", default="/dev/shm/h3/Ref2VA")
p.add_argument("--steps", type=int, default=50); p.add_argument("--student_steps", type=int, default=4)
p.add_argument("--shift", type=float, default=12.0); p.add_argument("--audio_shift", type=float, default=3.0)
p.add_argument("--shard", type=int, default=0); p.add_argument("--nshard", type=int, default=1)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--resident", action="store_true", help="keep the DiT on GPU (no CPU offload streaming)")
a = p.parse_args()
U = "/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930"
sys.path.insert(0, U); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
paths = [l.strip() for l in open(a.list) if l.strip()]
paths = paths[:a.limit] if a.limit else paths
paths = paths[a.shard::a.nshard]
oname = lambda p_: f"{a.out}/{os.path.basename(os.path.dirname(os.path.dirname(os.path.realpath(p_))))}_{os.path.basename(os.path.dirname(os.path.realpath(p_)))}_{os.path.basename(p_)[:-4]}.pt"
todo = [x for x in paths if not os.path.exists(oname(x))]
print(f"[ode] shard {a.shard}: {len(todo)} todo", flush=True)
if not todo:
    sys.exit(0)
os.makedirs(a.out, exist_ok=True)

from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig  # noqa: E402

_od = "cuda" if a.resident else "cpu"
vram = dict(offload_dtype=torch.bfloat16, offload_device=_od, onload_dtype=torch.bfloat16, onload_device=_od,
            preparing_dtype=torch.bfloat16, preparing_device="cuda", computation_dtype=torch.bfloat16, computation_device="cuda")
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[ModelConfig(path=sorted(glob.glob(f"{a.w}/transformer/model*.safetensors")), **vram)],
    processor_config=ModelConfig(path=f"{a.w}/processor/"))
if a.lora:
    from safetensors.torch import load_file
    sd = {k: v for k, v in load_file(a.lora).items() if not k.startswith("od_music.")}
    pipe.load_lora(pipe.dit, state_dict={k: v.to(device=pipe.device, dtype=pipe.torch_dtype) for k, v in sd.items()})
pipe.load_models_to_device(pipe.in_iteration_models)
models = {n: getattr(pipe, n) for n in pipe.in_iteration_models}

pipe.scheduler.set_timesteps(a.steps, shift=a.shift)
pipe.scheduler_audio.set_timesteps(a.steps, shift=a.audio_shift)
T_SIG = [float(s) for s in pipe.scheduler.sigmas]
import copy
_sc = copy.deepcopy(pipe.scheduler); _sc.set_timesteps(a.student_steps, shift=a.shift)
S_SIG = [float(s) for s in _sc.sigmas[:a.student_steps]]
# teacher step index whose *input* sigma is closest to each student sigma (student sigma[0] = 1 -> pure noise, index 0)
S_IDX = [min(range(a.steps), key=lambda k: abs(T_SIG[k] - s)) for s in S_SIG]
print(f"[ode] student sigmas {S_SIG} -> teacher steps {S_IDX} (teacher sigmas {[round(T_SIG[k], 4) for k in S_IDX]})", flush=True)

dev = pipe.device
n = 0; t0 = time.time()
with torch.no_grad():
    for path in todo:
        sh, posi, _ = torch.load(path, map_location="cpu", weights_only=False)
        lat_gt = sh["input_latents"]
        g = torch.Generator(device="cpu").manual_seed(hash(os.path.realpath(path)) % (2 ** 31))
        x = torch.randn(lat_gt.shape, generator=g, dtype=torch.float32).to(dev, torch.bfloat16)
        aud = sh["audio_input_latents"].to(dev, torch.bfloat16)
        kw = dict(audio_latents=aud, input_latents_audio=aud, denoise_mask_audio=torch.zeros_like(aud),
                  ref_visual_anchor=sh["ref_visual_anchor"].to(dev, torch.bfloat16), ref_audio_anchor=sh["ref_audio_anchor"].to(dev, torch.bfloat16),
                  imgvid_cond_noise_aug=sh.get("imgvid_cond_noise_aug", pipe.imgvid_cond_noise_aug),
                  audio_cond_noise_aug=sh.get("audio_cond_noise_aug", pipe.audio_cond_noise_aug),
                  prompt_embeds=posi["prompt_embeds"].to(dev, torch.bfloat16),
                  packed={k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in posi["packed"].items()},
                  use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)
        keep = []
        for k, tv in enumerate(pipe.scheduler.timesteps):
            if k in S_IDX:
                keep.append(x.detach().cpu().clone())
            tv = tv.unsqueeze(0).to(dtype=torch.float32, device=dev)
            ta = pipe.scheduler_audio.timesteps[k].unsqueeze(0).to(dtype=torch.float32, device=dev)
            v, _ = pipe.model_fn(**models, video_latents=x, timestep_video=tv, timestep_audio=ta, **kw)
            x = pipe.step(pipe.scheduler, x, k, noise_pred=v)
        keep.append(x.detach().cpu().clone())
        o = oname(path)
        torch.save({"sigmas": [T_SIG[k] for k in S_IDX] + [0.0], "x": torch.stack(keep).to(torch.bfloat16)[:, 0],
                    "src": os.path.realpath(path), "lora": a.lora, "teacher_steps": a.steps, "shift": a.shift}, o + ".tmp")
        os.replace(o + ".tmp", o)
        n += 1
        print(f"[ode] {n}/{len(todo)} {(time.time() - t0) / n:.0f}s/traj", flush=True)
print(f"[ode] shard {a.shard} done {n}", flush=True)
