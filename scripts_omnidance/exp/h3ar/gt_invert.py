"""gt_invert.py (10-15): ODE inversion of GT training latents with the original H3 (paired-noise reflow on real data).
For each cached GT clip, integrate the base model's conditional velocity field backwards (sigma 0 -> 1, the exact
sigma grid of the 50-step shift-12 sampler, mirror of the sampler's Euler steps) on the same cached conditions as
training (rich-prompt sidecar, clean music mask 0, reference anchors). The result z is the noise from which the base
sampler (approximately) regenerates this GT clip; training FM with x_t = (1-s) x0 + s z keeps every target
deterministic per sample (no conditional mean over noises).
--check N: for the first N items of each worker also re-sample from z and report the latent reconstruction error.
Output <out>/inv/<rel with / -> _>.pt = {"z": (C,T,H,W) bf16, "rel", "src", "std", "recon_rel" (if checked)}
python gt_invert.py --root /opt/h3cache/gt512 --list SG512/sg512.txt --out H3AR/INV512 [--check 1]
"""
import argparse, glob, os, sys, time
import torch

p = argparse.ArgumentParser()
p.add_argument("--root", required=True); p.add_argument("--list", required=True); p.add_argument("--out", required=True)
p.add_argument("--w", default="/dev/shm/h3/Ref2VA"); p.add_argument("--posi_dir", default="/opt/h3cache/posi")
p.add_argument("--steps", type=int, default=50); p.add_argument("--shift", type=float, default=12.0); p.add_argument("--audio_shift", type=float, default=3.0)
p.add_argument("--check", type=int, default=0)
a = p.parse_args()
sys.path.insert(0, "/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930")
rels = [l.strip() for l in open(a.list) if l.strip()]
L, C = f"{a.out}/inv", f"{a.out}/.claims"
os.makedirs(L, exist_ok=True); os.makedirs(C, exist_ok=True)
key = lambda r: r.replace("/", "_")[:-4]


def load_entry(rel):  # identical to sg_gen.py / unified_dataset.py rich-prompt sidecar handling
    path = os.path.join(a.root, rel)
    sh, posi, nega = torch.load(path, map_location="cpu", weights_only=False)
    sc = torch.load(os.path.join(a.posi_dir, os.path.relpath(os.path.realpath(path), "/opt/h3cache"))[:-4] + ".posi.pt",
                    map_location="cpu", weights_only=False)
    pe = posi["prompt_embeds"]
    posi = {**posi, "prompt_embeds": torch.cat([pe[:sc["Lp"]], sc["tail"].to(pe.dtype)], 0), "packed": sc["packed"]}
    return path, sh, posi


from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig  # noqa: E402

vram = dict(offload_dtype=torch.bfloat16, offload_device="cuda", onload_dtype=torch.bfloat16, onload_device="cuda",
            preparing_dtype=torch.bfloat16, preparing_device="cuda", computation_dtype=torch.bfloat16, computation_device="cuda")
pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cuda",
                                         model_configs=[ModelConfig(path=sorted(glob.glob(f"{a.w}/transformer/model*.safetensors")), **vram)],
                                         processor_config=ModelConfig(path=f"{a.w}/processor/"))
pipe.load_models_to_device(pipe.in_iteration_models)
models = {n: getattr(pipe, n) for n in pipe.in_iteration_models}
pipe.scheduler.set_timesteps(a.steps, shift=a.shift)
pipe.scheduler_audio.set_timesteps(a.steps, shift=a.audio_shift)
SIG = [float(s) for s in pipe.scheduler.sigmas[:a.steps]] + [0.0]  # sampler goes SIG[k] -> SIG[k+1]
dev = pipe.device
n = 0; t0 = time.time(); checked = 0
with torch.no_grad():
    for rel in rels:
        o = f"{L}/{key(rel)}.pt"
        if os.path.exists(o):
            continue
        try:
            os.mkdir(f"{C}/{key(rel)}")
        except FileExistsError:
            continue
        path, sh, posi = load_entry(rel)
        x0 = sh["input_latents"].to(dev, torch.bfloat16)
        aud = sh["audio_input_latents"].to(dev, torch.bfloat16)
        kw = dict(audio_latents=aud, input_latents_audio=aud, denoise_mask_audio=torch.zeros_like(aud),
                  ref_visual_anchor=sh["ref_visual_anchor"].to(dev, torch.bfloat16), ref_audio_anchor=sh["ref_audio_anchor"].to(dev, torch.bfloat16),
                  imgvid_cond_noise_aug=sh.get("imgvid_cond_noise_aug", pipe.imgvid_cond_noise_aug),
                  audio_cond_noise_aug=sh.get("audio_cond_noise_aug", pipe.audio_cond_noise_aug),
                  prompt_embeds=posi["prompt_embeds"].to(dev, torch.bfloat16),
                  packed={k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in posi["packed"].items()},
                  use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)

        def vel(x, k):
            tv = pipe.scheduler.timesteps[k].unsqueeze(0).to(dtype=torch.float32, device=dev)
            ta = pipe.scheduler_audio.timesteps[k].unsqueeze(0).to(dtype=torch.float32, device=dev)
            v, _ = pipe.model_fn(**models, video_latents=x, timestep_video=tv, timestep_audio=ta, **kw)
            return v.float()

        x = x0.float()
        for k in range(a.steps - 1, -1, -1):  # inverse of x_{k+1} = x_k + (SIG[k+1]-SIG[k]) v(x_k, k), with v(x_{k+1}, k)
            x = x - (SIG[k + 1] - SIG[k]) * vel(x.to(torch.bfloat16), k)
        z = x
        assert torch.isfinite(z).all(), rel
        rec = {}
        if checked < a.check:
            y = z.clone()
            for k in range(a.steps):
                y = y + (SIG[k + 1] - SIG[k]) * vel(y.to(torch.bfloat16), k)
            rec = {"recon_rel": float((y - x0.float()).pow(2).mean() / x0.float().pow(2).mean())}
            checked += 1
        torch.save({"z": z[0].to("cpu", torch.bfloat16), "rel": rel, "src": os.path.realpath(path), "std": float(z.std()),
                    "mean": float(z.mean()), "steps": a.steps, "shift": a.shift, "w": a.w, **rec}, o + ".tmp")
        os.replace(o + ".tmp", o)
        n += 1
        print(f"[inv] {n} {rel} std={z.std():.3f} {rec} {(time.time() - t0) / n:.0f}s/item", flush=True)
print(f"[inv] worker done {n}", flush=True)
