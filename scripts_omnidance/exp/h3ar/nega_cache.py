"""nega_cache.py --list <tsv: cache_realpath \t metadata.json \t index> --out <dir> [--shard g --nshard N]
Builds the UNCONDITIONAL (negative prompt " ") inputs for an existing H3 split-training cache, so that
--training_cfg_scale > 1 (inverse-CFG / guidance loss) can be used without re-running data_process.

Ref2VA presentation = [<Picture 1>: image tokens][<Audio 1>: ][prompt]; the text encoder is causal, so the
hidden states of the shared prefix are identical for prompt and " ". We therefore store only
  Lp (shared prefix length), tail = nega_embeds[Lp:], packed (nega packed sequence)
-> sidecar <out>/<cache path relative to /opt/h3cache>.nega.pt ; the loader rebuilds
   nega prompt_embeds = cat(posi_embeds[:Lp], tail).
Checks on the first samples of every shard: (a) rebuilding the POSITIVE packed from the cached embeddings
reproduces the cached packed exactly (proves the cache<->metadata mapping), (b) prefix rows match."""
import argparse, glob, json, os, sys, time
import torch

p = argparse.ArgumentParser()
p.add_argument("--list", required=True); p.add_argument("--out", required=True)
p.add_argument("--shard", type=int, default=0); p.add_argument("--nshard", type=int, default=1)
p.add_argument("--base", default="/dev/shm/h3data"); p.add_argument("--w", default="/dev/shm/h3/Ref2VA")
p.add_argument("--check", type=int, default=3)
p.add_argument("--idmode", action="store_true", help="list = <cache path>\t<sample id>: ref image from --ref_dirs, ref audio length from the cache "
               "(the packed only needs ref_audio_t; the audio content never enters the text encoder) -> no metadata / audio files needed")
p.add_argument("--ref_dirs", default="/opt/h3raw/ref,/dev/shm/h3data/ref")
a = p.parse_args()
U = "/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930"
sys.path.insert(0, U)
items = [l.rstrip("\n").split("\t") for l in open(a.list) if l.strip()][a.shard::a.nshard]
dst = lambda real: os.path.join(a.out, os.path.relpath(real, "/opt/h3cache"))[:-4] + ".nega.pt"
items = [it for it in items if not os.path.exists(dst(it[0]))]
print(f"[nega] shard {a.shard}: {len(items)} todo", flush=True)
if not items:
    sys.exit(0)

from diffsynth.pipelines.minimax_h3_audio_video import (MiniMaxH3Pipeline, ModelConfig, MiniMaxH3Unit_ShapeChecker,
    MiniMaxH3Unit_NoiseInitializer, MiniMaxH3Unit_ReferenceEncoder, MiniMaxH3Unit_PromptEmbedder, MiniMaxH3Unit_PackedSequenceBuilder)
from diffsynth.utils.data.minimax_h3 import MiniMaxH3ReferenceLoader
from diffsynth.models.minimax_h3_text_encoder import _text_ids
from PIL import Image

W = a.w
vram = dict(offload_dtype=torch.bfloat16, offload_device="cuda", onload_dtype=torch.bfloat16, onload_device="cuda",
            preparing_dtype=torch.bfloat16, preparing_device="cuda", computation_dtype=torch.bfloat16, computation_device="cuda")  # resident on GPU (63 GB TE)
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[ModelConfig(path=sorted(glob.glob(f"{W}/text_encoder/model*.safetensors")), **vram),
                   ModelConfig(path=f"{W}/video_vae/source/model.safetensors", **vram),
                   ModelConfig(path=f"{W}/audio_vae/model.safetensors", **vram)],
    processor_config=ModelConfig(path=f"{W}/processor/"))
loader = MiniMaxH3ReferenceLoader(base_path=a.base, height=832, width=480, max_pixels=1024 * 1024, num_frames=124, frame_rate=24)
U_shape, U_noise, U_ref = MiniMaxH3Unit_ShapeChecker(), MiniMaxH3Unit_NoiseInitializer(), MiniMaxH3Unit_ReferenceEncoder()
U_prompt, U_pack = MiniMaxH3Unit_PromptEmbedder(), MiniMaxH3Unit_PackedSequenceBuilder()
METAS = {}


def cpu(pk):
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in pk.items()}


def same_packed(x, y):
    for k in y:
        if torch.is_tensor(y[k]):
            if not torch.equal(x[k].cpu().to(y[k].dtype), y[k].cpu()):
                return k
        elif x[k] != y[k]:
            return k
    return None


n = 0; t0 = time.time()
with torch.no_grad():
    RD = a.ref_dirs.split(",")
    for it in items:
        sh = U_shape.process(pipe, 832, 480, 124)
        sh.update(U_noise.process(pipe, seed=42, num_frames=124, height=832, width=480, rand_device="cpu"))
        if a.idmode:
            real, sid = it[0], it[1]
            c = torch.load(real, map_location="cpu", weights_only=False, mmap=True)
            n_ra = int(c[0]["ref_audio_anchor"].shape[0]); pk_c = c[1]["packed"]; L_c = int(c[1]["prompt_embeds"].shape[0])
            img = Image.open(next(f"{d}/{sid}_f0.png" for d in RD if os.path.exists(f"{d}/{sid}_f0.png"))).convert("RGB")
            sh.update(U_ref.process(pipe, references=[{"type": "image", "image": img}], seed=42, video_latents=sh["video_latents"], num_frames=124,
                                    ref_image_short_edge=2048, ref_video_short_edge=768, ref_video_max_pixels=768 * 1344))
            sh["ref_blocks"] = list(sh["ref_blocks"]) + [{"kind": "audio", "ref_audio_t": n_ra // 2}]
            ids_n, tags_n = U_prompt.preprocess_ref_blocks(pipe, " ", sh["ref_blocks"])[:2]
            Lp = len(ids_n) - len(_text_ids(pipe.tokenizer, " "))
            m = {"video": sid}
        else:
            real, meta, idx = it
            if meta not in METAS:
                METAS[meta] = json.load(open(meta))
            m = METAS[meta][int(idx)]
            refs = loader(m["references"])
            sh.update(U_ref.process(pipe, references=refs, seed=42, video_latents=sh["video_latents"], num_frames=124,
                                    ref_image_short_edge=2048, ref_video_short_edge=768, ref_video_max_pixels=768 * 1344))
            ids_p, tags_p = U_prompt.preprocess_ref_blocks(pipe, m["prompt"], sh["ref_blocks"])[:2]
            ids_n, tags_n = U_prompt.preprocess_ref_blocks(pipe, " ", sh["ref_blocks"])[:2]
            Lp = 0
            while Lp < min(len(ids_p), len(ids_n)) and int(ids_p[Lp]) == int(ids_n[Lp]):
                Lp += 1
        neg = U_prompt.process(pipe, " ", ref_blocks=sh["ref_blocks"], height=832, width=480)
        pk_n = U_pack.process(pipe, prompt_embeds=neg["prompt_embeds"], video_latents=sh["video_latents"], audio_latents=sh["audio_latents"],
                              text_token_tags=neg["text_token_tags"], ref_blocks=sh["ref_blocks"])["packed"]
        if a.idmode:  # every sample: ref/target token counts of the nega packed must match the cached (posi) packed
            for k in ("img_pos", "audio_pos"):
                assert pk_n[k].numel() == pk_c[k].numel(), f"{k} count {pk_n[k].numel()} != cached {pk_c[k].numel()} ({real})"
            assert L_c > Lp, (L_c, Lp)
            if n < a.check:
                pe = c[1]["prompt_embeds"]
                d = (neg["prompt_embeds"][:Lp].float().cpu() - pe[:Lp].float()).abs()
                rel = float(d.mean() / pe[:Lp].float().abs().mean())
                print(f"[nega] check ok {sid} Lp={Lp} posi={L_c} nega={len(ids_n)} prefix rel-diff={rel:.4f}", flush=True)
                assert rel < 0.05, "prefix hidden states differ (wrong reference image?)"
        elif n < a.check:
            c = torch.load(real, map_location="cpu", weights_only=False)
            pe = c[1]["prompt_embeds"]
            assert pe.shape[0] == len(ids_p), f"posi length {pe.shape[0]} != {len(ids_p)} ({real} -> {m['video']})"
            pk_p = U_pack.process(pipe, prompt_embeds=pe.to(pipe.device), video_latents=sh["video_latents"], audio_latents=sh["audio_latents"],
                                  text_token_tags=tags_p.to(pipe.device), ref_blocks=sh["ref_blocks"])["packed"]
            bad = same_packed(pk_p, c[1]["packed"])
            assert bad is None, f"posi packed mismatch on {bad} ({real})"
            d = (neg["prompt_embeds"][:Lp].float().cpu() - pe[:Lp].float()).abs()
            rel = float(d.mean() / pe[:Lp].float().abs().mean())
            print(f"[nega] check ok {os.path.basename(m['video'])} Lp={Lp} posi={len(ids_p)} nega={len(ids_n)} prefix rel-diff={rel:.4f}", flush=True)
            assert rel < 0.05, "prefix hidden states differ: text encoder is not causal over the prefix?"
        o = dst(real); os.makedirs(os.path.dirname(o), exist_ok=True)
        torch.save({"Lp": Lp, "tail": neg["prompt_embeds"][Lp:].to(torch.bfloat16).cpu().clone(), "packed": cpu(pk_n)}, o + ".tmp")
        os.replace(o + ".tmp", o)
        n += 1
        if n % 50 == 0:
            print(f"[nega] {n}/{len(items)} {(time.time() - t0) / n:.1f}s/it", flush=True)
print(f"[nega] shard {a.shard} done {n}", flush=True)
