"""probe_fm.py (10-14): where (which noise level) does the FM fine-tune change H3, and does the distilled base even fit FM targets?
For N training items (same cached condition / rich-prompt sidecar as training, clean music mask 0), fixed noise, sigma grid:
  x_t = (1-s) x0 + s eps ; FM target u = eps - x0 (DiffSynth model_fn returns the velocity in this convention);
  report for each model M in {base, LoRAs}: FM MSE vs target built from GT x0 and from SG x0 (the base model's own sample),
  ||v_M - v_base|| / ||v_base||, and the high-frequency energy ratio of the predicted x0 = x_t - s v (latent space, |FFT| above half Nyquist).
Writes a JSON table; prints a compact summary.
python probe_fm.py --root /opt/h3cache/cfg_rich_cur --list sg512.txt --sg_lat <SG512/lat> --lora name=path ... --out probe.json
"""
import argparse, glob, json, os, sys
import torch

p = argparse.ArgumentParser()
p.add_argument("--root", required=True); p.add_argument("--list", required=True); p.add_argument("--sg_lat", required=True)
p.add_argument("--lora", nargs="*", default=[]); p.add_argument("--n", type=int, default=8); p.add_argument("--out", required=True)
p.add_argument("--w", default="/dev/shm/h3/Ref2VA"); p.add_argument("--posi_dir", default="/opt/h3cache/posi")
p.add_argument("--sigmas", default="0.98,0.9,0.75,0.5,0.3,0.15,0.05")
a = p.parse_args()
sys.path.insert(0, "/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930")
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

vram = dict(offload_dtype=torch.bfloat16, offload_device="cuda", onload_dtype=torch.bfloat16, onload_device="cuda",
            preparing_dtype=torch.bfloat16, preparing_device="cuda", computation_dtype=torch.bfloat16, computation_device="cuda")
pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cuda",
                                         model_configs=[ModelConfig(path=sorted(glob.glob(f"{a.w}/transformer/model*.safetensors")), **vram)],
                                         processor_config=ModelConfig(path=f"{a.w}/processor/"))
pipe.load_models_to_device(pipe.in_iteration_models)
models = {n: getattr(pipe, n) for n in pipe.in_iteration_models}
LORAS = {nm: {k: v.to("cuda", torch.bfloat16) for k, v in load_file(pth).items() if not k.startswith("od_music.")}
         for nm, pth in (x.split("=", 1) for x in a.lora)}
dev = pipe.device
SIG = [float(s) for s in a.sigmas.split(",")]


def hf_ratio(x):  # x: (1,C,T,H,W) -> fraction of spatial spectral energy above half Nyquist
    f = torch.fft.fft2(x.float(), dim=(-2, -1)).abs() ** 2
    H, W = f.shape[-2:]
    fy = torch.fft.fftfreq(H, device=x.device).abs()[:, None]; fx = torch.fft.fftfreq(W, device=x.device).abs()[None, :]
    m = ((fy > 0.25) | (fx > 0.25)).float()
    return float((f * m).sum() / f.sum())


rels = [l.strip() for l in open(a.list) if l.strip()][:a.n]
rows = []
with torch.no_grad():
    for i, rel in enumerate(rels):
        path = os.path.join(a.root, rel)
        sh, posi, _ = torch.load(path, map_location="cpu", weights_only=False)
        sc = torch.load(os.path.join(a.posi_dir, os.path.relpath(os.path.realpath(path), "/opt/h3cache"))[:-4] + ".posi.pt", map_location="cpu", weights_only=False)
        pe = posi["prompt_embeds"]; pe = torch.cat([pe[:sc["Lp"]], sc["tail"].to(pe.dtype)], 0)
        gt = sh["input_latents"].to(dev, torch.bfloat16)
        sg = torch.load(f"{a.sg_lat}/{rel.replace('/', '_')[:-4]}.pt", map_location="cpu", weights_only=False)["x0"].unsqueeze(0).to(dev, torch.bfloat16)
        aud = sh["audio_input_latents"].to(dev, torch.bfloat16)
        kw = dict(audio_latents=aud, input_latents_audio=aud, denoise_mask_audio=torch.zeros_like(aud),
                  ref_visual_anchor=sh["ref_visual_anchor"].to(dev, torch.bfloat16), ref_audio_anchor=sh["ref_audio_anchor"].to(dev, torch.bfloat16),
                  imgvid_cond_noise_aug=sh.get("imgvid_cond_noise_aug", pipe.imgvid_cond_noise_aug), audio_cond_noise_aug=sh.get("audio_cond_noise_aug", pipe.audio_cond_noise_aug),
                  prompt_embeds=pe.to(dev, torch.bfloat16), packed={k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in sc["packed"].items()},
                  use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)
        g = torch.Generator(device="cpu").manual_seed(1234 + i)
        eps = torch.randn(gt.shape, generator=g, dtype=torch.float32).to(dev, torch.bfloat16)
        for s in SIG:
            t = torch.tensor([s * 1000.0], dtype=torch.float32, device=dev)
            for src, x0 in (("gt", gt), ("sg", sg)):
                xt = ((1 - s) * x0.float() + s * eps.float()).to(torch.bfloat16)
                u = (eps.float() - x0.float())
                vb = None
                for nm in ["base"] + list(LORAS):
                    pipe.clear_lora(verbose=0)
                    if nm != "base":
                        pipe.load_lora(pipe.dit, state_dict=LORAS[nm], verbose=0)
                    v, _ = pipe.model_fn(**models, video_latents=xt, timestep_video=t, timestep_audio=t, **kw)
                    v = v.float()
                    if nm == "base":
                        vb = v
                    x0p = xt.float() - s * v
                    rows.append({"item": rel, "sigma": s, "target": src, "model": nm,
                                 "fm_mse": float(((v - u) ** 2).mean()), "dv_rel": float((v - vb).norm() / vb.norm()),
                                 "x0_mse": float(((x0p - x0.float()) ** 2).mean()), "x0_hf": hf_ratio(x0p), "tgt_hf": hf_ratio(x0)})
        pipe.clear_lora(verbose=0)
        print(f"[probe] {i + 1}/{len(rels)} {rel}", flush=True)
        json.dump(rows, open(a.out, "w"))
import collections  # noqa: E402
agg = collections.defaultdict(list)
for r in rows:
    agg[(r["target"], r["model"], r["sigma"])].append(r)
print("target model sigma fm_mse x0_mse dv_rel x0_hf tgt_hf")
for k in sorted(agg):
    v = agg[k]; m = lambda f: sum(x[f] for x in v) / len(v)
    print(*k, *(f"{m(f):.4f}" for f in ("fm_mse", "x0_mse", "dv_rel", "x0_hf", "tgt_hf")))
