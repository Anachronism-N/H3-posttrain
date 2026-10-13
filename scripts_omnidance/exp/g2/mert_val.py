"""compute MERT features [1,25,1024,81] for audio files missing from the cache -> G2/mert_val/<stem>.pt"""
import csv, glob, os, sys
import torch, torchaudio
sys.path.insert(0, "/apdcephfs_gy4/share_302533218/cedricnie/diff")
from diffsynth.models.mert_audio_encoder import MERTAudioEncoder
csv.field_size_limit(10 ** 9)
C = "/apdcephfs_gy4/share_302533218/cedricnie/diff/data/omnidance/audio_cache/mert_features"
O = "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/G2/mert_val"; os.makedirs(O, exist_ok=True)
aud = set()
for m in sys.argv[1:]:
    for r in csv.DictReader(open(m)):
        aud.add(r["audio"]); aud.add(r["alt_audio"])
enc = MERTAudioEncoder("/dev/shm/od/MERT-v1-330M").to(torch.bfloat16).cuda().eval()
n = 0
for a in sorted(aud):
    s = os.path.basename(a)[:-4]
    if os.path.exists(f"{C}/{s}.pt") or os.path.exists(f"{O}/{s}.pt"):
        continue
    w, sr = torchaudio.load(a); w = w.mean(0)
    with torch.no_grad():
        f = enc.get_audio_feats_per_inference(w, sr, batch_frames=81, fps=16, dtype=torch.bfloat16, device="cuda")
    f = f[0] if isinstance(f, list) else f
    torch.save(f.cpu(), f"{O}/{s}.pt"); n += 1
print("computed", n, "shape", tuple(f.shape) if n else None)
