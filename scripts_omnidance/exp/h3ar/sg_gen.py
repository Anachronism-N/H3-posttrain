"""sg_gen.py: self-generated training targets for the attribution control (10-13).
Original H3 Ref2VA (no LoRA, cfg 1, official shift 12/3, 50 steps) samples the video latent from noise on the *exact*
cached training conditions (rich-prompt sidecar applied the same way as the training loader, clean music mask 0,
reference anchors). Training on these latents vs on the GT latents of the same IDs (same recipe) separates
"the training data is degraded" from "the fine-tuning process itself blurs".

Workers on several nodes share one list of cache-relative paths and claim items by atomic mkdir on cephfs.
python sg_gen.py --root /opt/h3cache/cfg_rich_cur --list sg512.txt --out <dir> [--decode_first]
Output <out>/lat/<rel with / -> _>.pt = {"x0": (C,T,H,W) bf16, "rel", "src", "sha_src", ...}
"""
import argparse, glob, hashlib, os, sys, time
import torch

p = argparse.ArgumentParser()
p.add_argument("--root", required=True); p.add_argument("--list", required=True); p.add_argument("--out", required=True)
p.add_argument("--w", default="/dev/shm/h3/Ref2VA"); p.add_argument("--posi_dir", default="/opt/h3cache/posi")
p.add_argument("--steps", type=int, default=50); p.add_argument("--shift", type=float, default=12.0); p.add_argument("--audio_shift", type=float, default=3.0)
p.add_argument("--limit", type=int, default=0); p.add_argument("--decode_first", action="store_true")
p.add_argument("--lora", default="", help="10-14 self-reproduction test: same cached condition + same seed as the SG target, with a trained LoRA")
p.add_argument("--decode_all", action="store_true", help="decode every generated item to <out>/<key>_gen.mp4 (and the GT latent to <key>_gt.mp4)")
a = p.parse_args()
U = "/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930"
sys.path.insert(0, U)
rels = [l.strip() for l in open(a.list) if l.strip()]
rels = rels[:a.limit] if a.limit else rels
L, C = f"{a.out}/lat", f"{a.out}/.claims"
os.makedirs(L, exist_ok=True); os.makedirs(C, exist_ok=True)
key = lambda r: r.replace("/", "_")[:-4]


def load_entry(rel):
    """cache tuple with the rich-prompt sidecar applied exactly as unified_dataset.py (OD_H3_POSI_DIR)."""
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
cfgs = [ModelConfig(path=sorted(glob.glob(f"{a.w}/transformer/model*.safetensors")), **vram)]
if a.decode_first or a.decode_all:
    cfgs.append(ModelConfig(path=f"{a.w}/video_vae/source/model.safetensors"))
pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cuda", model_configs=cfgs,
                                         processor_config=ModelConfig(path=f"{a.w}/processor/"))
if a.lora:
    from safetensors.torch import load_file
    pipe.load_lora(pipe.dit, state_dict={k: v.to(device="cuda", dtype=torch.bfloat16) for k, v in load_file(a.lora).items() if not k.startswith("od_music.")})
pipe.load_models_to_device(pipe.in_iteration_models)
models = {n: getattr(pipe, n) for n in pipe.in_iteration_models}
pipe.scheduler.set_timesteps(a.steps, shift=a.shift)
pipe.scheduler_audio.set_timesteps(a.steps, shift=a.audio_shift)
dev = pipe.device
n = 0; t0 = time.time(); decoded = False
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
        seed = int(hashlib.sha256(rel.encode()).hexdigest()[:8], 16)
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(sh["input_latents"].shape, generator=g, dtype=torch.float32).to(dev, torch.bfloat16)
        aud = sh["audio_input_latents"].to(dev, torch.bfloat16)
        kw = dict(audio_latents=aud, input_latents_audio=aud, denoise_mask_audio=torch.zeros_like(aud),
                  ref_visual_anchor=sh["ref_visual_anchor"].to(dev, torch.bfloat16), ref_audio_anchor=sh["ref_audio_anchor"].to(dev, torch.bfloat16),
                  imgvid_cond_noise_aug=sh.get("imgvid_cond_noise_aug", pipe.imgvid_cond_noise_aug),
                  audio_cond_noise_aug=sh.get("audio_cond_noise_aug", pipe.audio_cond_noise_aug),
                  prompt_embeds=posi["prompt_embeds"].to(dev, torch.bfloat16),
                  packed={k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in posi["packed"].items()},
                  use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)
        for k, tv in enumerate(pipe.scheduler.timesteps):
            tv = tv.unsqueeze(0).to(dtype=torch.float32, device=dev)
            ta = pipe.scheduler_audio.timesteps[k].unsqueeze(0).to(dtype=torch.float32, device=dev)
            v, _ = pipe.model_fn(**models, video_latents=x, timestep_video=tv, timestep_audio=ta, **kw)
            x = pipe.step(pipe.scheduler, x, k, noise_pred=v)
        x0 = x[0].detach().to("cpu", torch.bfloat16)
        assert torch.isfinite(x0.float()).all(), rel
        torch.save({"x0": x0, "rel": rel, "src": os.path.realpath(path), "seed": seed, "steps": a.steps,
                    "shift": a.shift, "audio_shift": a.audio_shift, "w": a.w, "lora": a.lora}, o + ".tmp")
        os.replace(o + ".tmp", o)
        n += 1
        print(f"[sg] {n} {rel} {(time.time() - t0) / n:.0f}s/item", flush=True)
        if (a.decode_first and not decoded) or a.decode_all:
            import av  # noqa: E402
            pipe.load_models_to_device(["video_vae"])
            pairs = (("gen", x), ("gt", sh["input_latents"].to(dev, torch.bfloat16))) if a.decode_all else (("sg", x), ("gt", sh["input_latents"].to(dev, torch.bfloat16)))
            for nm, lat in pairs:
                fr = pipe.video_vae.decode_video(lat, dtype=torch.bfloat16, tiled=True, tile_size=256, tile_overlap=64)
                vid = pipe.vae_output_to_video(fr, min_value=0, max_value=1)
                with av.open(f"{a.out}/{key(rel)}_{nm}.mp4" if a.decode_all else f"{a.out}/check_{key(rel)}_{nm}.mp4", "w") as c:
                    s = c.add_stream("libx264", rate=24); s.width, s.height = vid[0].size; s.pix_fmt = "yuv420p"; s.options = {"crf": "12"}
                    for im in vid:
                        for pk in s.encode(av.VideoFrame.from_image(im.convert("RGB"))):
                            c.mux(pk)
                    for pk in s.encode():
                        c.mux(pk)
            pipe.load_models_to_device(pipe.in_iteration_models)
            decoded = True
print(f"[sg] worker done {n}", flush=True)
