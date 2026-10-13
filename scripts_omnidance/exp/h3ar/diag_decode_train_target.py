"""Decode cached training targets (input_latents) with the H3 video VAE, to check whether the
video the LoRA is trained to reproduce is itself soft on hands/faces (480p + RIFE + VAE).
Picks the K samples with the largest latent temporal change (fast motion) among the first N of a cache.
Writes <out>/<sid>_target.mp4 (VAE decode of the training latent) and <out>/<sid>_src.mp4 (source clip, first 124 frames, same size).
python diag_decode_train_target.py --cache /opt/h3cache/f1 --list /opt/h3cache/posi_list_f1.tsv --out <dir>
"""
import argparse, glob, os
import av, torch
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--list", required=True)
p.add_argument("--out", required=True)
p.add_argument("--scan", type=int, default=48)
p.add_argument("--k", type=int, default=3)
p.add_argument("--w", default="/dev/shm/h3/Ref2VA")
p.add_argument("--src_dir", default="/apdcephfs_gy4/share_302533218/cedricnie/OmniDance_processed")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

rows = [l.split("\t") for l in open(a.list).read().split("\n") if l][:a.scan]
score = []
for path, sid in rows:
    lat = torch.load(path, map_location="cpu", weights_only=False)[0]["input_latents"].float()
    score.append(((lat[:, :, 1:] - lat[:, :, :-1]).abs().mean().item(), path, sid))
score.sort(reverse=True)
print("[diag] top motion:", [(round(s, 4), sid) for s, _, sid in score[:a.k]], flush=True)

from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig  # noqa: E402
pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cuda",
                                         model_configs=[ModelConfig(path=f"{a.w}/video_vae/source/model.safetensors")],
                                         processor_config=ModelConfig(path=f"{a.w}/processor/"))


def write(frames, dst, fps=24):
    with av.open(dst, "w") as c:
        s = c.add_stream("libx264", rate=fps); s.width, s.height = frames[0].size; s.pix_fmt = "yuv420p"; s.options = {"crf": "12"}
        for im in frames:
            for pk in s.encode(av.VideoFrame.from_image(im.convert("RGB"))):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)


for _, path, sid in score[:a.k]:
    lat = torch.load(path, map_location="cpu", weights_only=False)[0]["input_latents"].to("cuda", torch.bfloat16)
    with torch.no_grad():
        fr = pipe.video_vae.decode_video(lat, dtype=torch.bfloat16, tiled=True, tile_size=256, tile_overlap=64)
    tgt = pipe.vae_output_to_video(fr, min_value=0, max_value=1)
    write(tgt, f"{a.out}/{sid}_target.mp4")
    W, H = tgt[0].size
    src = next(iter(glob.glob(f"{a.src_dir}/part_*/video/{sid}.mp4")), None)
    if src:
        with av.open(src) as c:
            fps = float(c.streams.video[0].average_rate)
            sf = [f.to_image().resize((W, H), Image.BICUBIC) for _, f in zip(range(124), c.decode(video=0))]
        write(sf, f"{a.out}/{sid}_src.mp4", fps=round(fps))
    print(f"[diag] {sid} target {len(tgt)}f {W}x{H} src={src}", flush=True)
