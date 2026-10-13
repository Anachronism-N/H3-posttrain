"""text_recache.py: replace the POSITIVE text of an existing H3 split-training cache with a new prompt (10-12: rich VLM prompt
instead of the E10 step-by-step motion caption), without re-encoding video/audio.

Ref2VA text = [<Picture 1>: image tokens][<Audio 1>: ][prompt]; the text encoder is causal, so the prefix hidden states equal the
cached ones. Sidecar <out>/<relpath under /opt/h3cache>.posi.pt = {"Lp", "tail" (= new embeds[Lp:]), "packed" (new)}; the loader
(OD_H3_POSI_DIR) rebuilds prompt_embeds = cat(cached[:Lp], tail).

python text_recache.py --list <tsv: cache path \\t sample id> --meta <cache_meta.json> --prompts <json sid->text> --out <dir> --shard g --nshard N
Checks: every sample's img/audio token counts == cached; if the cache file is present locally, the prefix rows match it."""
import argparse, glob, json, os, sys, time
import torch

p = argparse.ArgumentParser()
p.add_argument("--list", required=True); p.add_argument("--meta", required=True); p.add_argument("--prompts", required=True)
p.add_argument("--out", required=True); p.add_argument("--shard", type=int, default=0); p.add_argument("--nshard", type=int, default=1)
p.add_argument("--ref_dirs", default="/opt/h3raw/ref,/dev/shm/h3data/ref"); p.add_argument("--w", default="/dev/shm/h3/Ref2VA")
p.add_argument("--height", type=int, default=832); p.add_argument("--width", type=int, default=480)
a = p.parse_args()
U = "/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930"
sys.path.insert(0, U)
dst = lambda real: os.path.join(a.out, os.path.relpath(real, "/opt/h3cache"))[:-4] + ".posi.pt"
items = [l.rstrip("\n").split("\t") for l in open(a.list) if l.strip()][a.shard::a.nshard]
items = [it for it in items if not os.path.exists(dst(it[0]))]
print(f"[posi] shard {a.shard}: {len(items)} todo", flush=True)
if not items:
    sys.exit(0)
META = json.load(open(a.meta)); PROMPTS = json.load(open(a.prompts))

from PIL import Image
from diffsynth.pipelines.minimax_h3_audio_video import (MiniMaxH3Pipeline, ModelConfig, MiniMaxH3Unit_ShapeChecker,
    MiniMaxH3Unit_NoiseInitializer, MiniMaxH3Unit_ReferenceEncoder, MiniMaxH3Unit_PromptEmbedder, MiniMaxH3Unit_PackedSequenceBuilder)
from diffsynth.models.minimax_h3_text_encoder import _text_ids

vram = dict(offload_dtype=torch.bfloat16, offload_device="cuda", onload_dtype=torch.bfloat16, onload_device="cuda",
            preparing_dtype=torch.bfloat16, preparing_device="cuda", computation_dtype=torch.bfloat16, computation_device="cuda")
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[ModelConfig(path=sorted(glob.glob(f"{a.w}/text_encoder/model*.safetensors")), **vram),
                   ModelConfig(path=f"{a.w}/video_vae/source/model.safetensors", **vram),
                   ModelConfig(path=f"{a.w}/audio_vae/model.safetensors", **vram)],
    processor_config=ModelConfig(path=f"{a.w}/processor/"))
U_shape, U_noise, U_ref = MiniMaxH3Unit_ShapeChecker(), MiniMaxH3Unit_NoiseInitializer(), MiniMaxH3Unit_ReferenceEncoder()
U_prompt, U_pack = MiniMaxH3Unit_PromptEmbedder(), MiniMaxH3Unit_PackedSequenceBuilder()
RD = a.ref_dirs.split(",")
n = 0; t0 = time.time()
with torch.no_grad():
    for real, sid in items:
        m = META[sid]; P = PROMPTS[sid]
        sh = U_shape.process(pipe, a.height, a.width, 124)
        sh.update(U_noise.process(pipe, seed=42, num_frames=124, height=a.height, width=a.width, rand_device="cpu"))
        img = Image.open(next(f"{d}/{sid}_f0.png" for d in RD if os.path.exists(f"{d}/{sid}_f0.png"))).convert("RGB")
        sh.update(U_ref.process(pipe, references=[{"type": "image", "image": img}], seed=42, video_latents=sh["video_latents"], num_frames=124,
                                ref_image_short_edge=2048, ref_video_short_edge=768, ref_video_max_pixels=768 * 1344))
        sh["ref_blocks"] = list(sh["ref_blocks"]) + [{"kind": "audio", "ref_audio_t": m["n_ra"] // 2}]
        ids, _ = U_prompt.preprocess_ref_blocks(pipe, P, sh["ref_blocks"])[:2]
        Lp = len(ids) - len(_text_ids(pipe.tokenizer, P))
        e = U_prompt.process(pipe, P, ref_blocks=sh["ref_blocks"], height=a.height, width=a.width)
        pk = U_pack.process(pipe, prompt_embeds=e["prompt_embeds"], video_latents=sh["video_latents"], audio_latents=sh["audio_latents"],
                            text_token_tags=e["text_token_tags"], ref_blocks=sh["ref_blocks"])["packed"]
        assert pk["img_pos"].numel() == m["n_img"] and pk["audio_pos"].numel() == m["n_aud"], (sid, pk["img_pos"].numel(), m)
        if os.path.exists(real):  # local cache copy -> prefix check
            pe = torch.load(real, map_location="cpu", weights_only=False, mmap=True)[1]["prompt_embeds"]
            rel = float((e["prompt_embeds"][:Lp].float().cpu() - pe[:Lp].float()).abs().mean() / pe[:Lp].float().abs().mean())
            print(f"[posi] check {sid} Lp={Lp} old={pe.shape[0]} new={e['prompt_embeds'].shape[0]} prefix rel-diff={rel:.4f}", flush=True)
            assert rel < 0.05, "prefix differs (wrong reference image?)"
        o = dst(real); os.makedirs(os.path.dirname(o), exist_ok=True)
        torch.save({"Lp": Lp, "tail": e["prompt_embeds"][Lp:].to(torch.bfloat16).cpu().clone(),
                    "packed": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in pk.items()}}, o + ".tmp")
        os.replace(o + ".tmp", o)
        n += 1
        if n % 100 == 0:
            print(f"[posi] {n}/{len(items)} {(time.time() - t0) / n:.1f}s/it", flush=True)
print(f"[posi] shard {a.shard} done {n}", flush=True)
