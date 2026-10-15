"""genre_mert.py (10-15, mentor genre-CLIP G1 data): balanced pond5 genre set -> MERT-v1-330M features of one 5.17 s segment.
List: --make_list writes <out>/list.tsv (genre \\t path), N per genre, deterministic (sorted + seeded sample).
Worker: --shard i --nshard n; per item saves <out>/feat/<genre>/<stem>.pt = {"mean": [25,1024] time-mean per layer (bf16),
"seq": [25? no: layers 6,12,18,24 -> 4,1024,T] (bf16)} ; segment = 5.17 s starting at 30% of the track (skip intros).
"""
import argparse, os, random, sys
import torch, torchaudio

p = argparse.ArgumentParser()
p.add_argument("--root", default="/apdcephfs_zwfy7/share_304725743/data/music_clap_data/pond5/audio")
p.add_argument("--genres", default="rock,hiphop,dance,funk,jazz,zen,chinese,orchestral")
p.add_argument("--n", type=int, default=3000); p.add_argument("--out", required=True)
p.add_argument("--make_list", action="store_true"); p.add_argument("--shard", type=int, default=0); p.add_argument("--nshard", type=int, default=1)
p.add_argument("--dur", type=float, default=124 / 24)
a = p.parse_args()
L = f"{a.out}/list.tsv"
if a.make_list:
    os.makedirs(a.out, exist_ok=True)
    with open(L, "w") as f:
        for g in a.genres.split(","):
            fs = sorted(os.listdir(f"{a.root}/{g}"))
            fs = [x for x in fs if x.lower().endswith((".wav", ".mp3", ".flac"))]
            for x in random.Random(g).sample(fs, min(a.n, len(fs))):
                f.write(f"{g}\t{a.root}/{g}/{x}\n")
    print("list", sum(1 for _ in open(L))); sys.exit(0)
sys.path.insert(0, "/apdcephfs_gy4/share_302533218/cedricnie/diff")
from diffsynth.models.mert_audio_encoder import MERTAudioEncoder
items = [l.rstrip("\n").split("\t") for l in open(L)][a.shard::a.nshard]
enc = MERTAudioEncoder("/dev/shm/od/MERT-v1-330M").to(torch.bfloat16).cuda().eval()
n = bad = 0
for g, path in items:
    o = f"{a.out}/feat/{g}/{os.path.basename(path).rsplit('.', 1)[0]}.pt"
    if os.path.exists(o):
        continue
    try:
        info = torchaudio.info(path); sr = info.sample_rate; tot = info.num_frames
        seg = int(a.dur * sr); st = max(0, min(int(0.3 * tot), tot - seg))
        w, sr = torchaudio.load(path, frame_offset=st, num_frames=seg); w = w.mean(0)
        if w.numel() < seg * 0.9 or w.abs().max() < 1e-4:
            bad += 1; continue
        with torch.no_grad():
            f = enc.get_audio_feats_per_inference(w, sr, batch_frames=81, fps=16, dtype=torch.bfloat16, device="cuda")
        f = (f[0] if isinstance(f, list) else f)[0]  # [25,1024,T]
        os.makedirs(os.path.dirname(o), exist_ok=True)
        torch.save({"mean": f.float().mean(-1).to(torch.bfloat16).cpu(), "seq": f[[6, 12, 18, 24]].cpu(), "start": st / sr}, o + ".tmp")
        os.replace(o + ".tmp", o); n += 1
    except Exception as e:  # unreadable file: skip, count
        bad += 1; print("[gm] skip", path, repr(e)[:120], flush=True)
    if n % 200 == 0 and n:
        print(f"[gm] shard {a.shard} {n} done {bad} bad", flush=True)
print(f"[gm] shard {a.shard} finished {n} done {bad} bad", flush=True)
