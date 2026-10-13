"""M2 (notes/20) extra music experts, per clip:
  --phase beat -> <out>/beat/<id>.npy  (207, 6) float16 on the 40 Hz H3 audio-latent grid (first 207/40 = 5.175 s):
                  onset strength, beat impulse (gaussian, sigma 25 ms), sin/cos beat phase, RMS energy, tempo/200.
                  NO torch import in this phase: librosa beat tracking (numba) segfaults when torch is loaded in the same process.
  --phase clap -> <out>/clap/<id>.npy  (512,) float16 LAION CLAP (htsat-unfused) embedding of the same 5.175 s (genre / mood)
python m2_feats.py --phase beat|clap --list <txt: id<TAB>wav per line> --out <dir> --shard i --nshard n   (/opt/venv_h3)
"""
import argparse, os
import numpy as np

N_T, HZ = 207, 40.0
p = argparse.ArgumentParser()
p.add_argument("--phase", required=True, choices=["beat", "clap"])
p.add_argument("--list", required=True); p.add_argument("--out", required=True)
p.add_argument("--shard", type=int, default=0); p.add_argument("--nshard", type=int, default=1)
p.add_argument("--clap", default="/apdcephfs_gy4/share_302533218/cedricnie/models/clap-htsat-unfused")
a = p.parse_args()


def beat_feats(wav):
    import librosa
    y, sr = librosa.load(wav, sr=22050, mono=True, duration=N_T / HZ + 0.05)
    t = np.arange(N_T) / HZ
    hop = 256
    on = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    ft = librosa.frames_to_time(np.arange(len(on)), sr=sr, hop_length=hop)
    onset = np.interp(t, ft, on / (on.max() + 1e-6))
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms = np.interp(t, librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop), rms / (rms.max() + 1e-6))
    tempo, beats = librosa.beat.beat_track(onset_envelope=on, sr=sr, hop_length=hop, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    if len(beats) < 2:
        per = 60.0 / max(tempo, 60.0); beats = np.arange(0, t[-1] + per, per)
    per = float(np.median(np.diff(beats)))
    bx = np.concatenate([beats[:1] - per * np.arange(8, 0, -1), beats, beats[-1:] + per * np.arange(1, 9)])
    k = np.clip(np.searchsorted(bx, t, side="right") - 1, 0, len(bx) - 2)
    ph = (t - bx[k]) / (bx[k + 1] - bx[k])
    imp = np.exp(-((t[:, None] - beats[None]) ** 2) / (2 * 0.025 ** 2)).max(1)
    return np.stack([onset, imp, np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph), rms, np.full(N_T, tempo / 200.0)], 1).astype(np.float16)


items = [l.rstrip("\n").split("\t") for l in open(a.list) if l.strip()][a.shard::a.nshard]
od = f"{a.out}/{a.phase}"; os.makedirs(od, exist_ok=True)
items = [it for it in items if not os.path.exists(f"{od}/{it[0]}.npy")]
print(f"[m2:{a.phase}] {len(items)} todo", flush=True)
n = bad = 0
if a.phase == "beat":
    for sid, wav in items:
        try:
            np.save(f"{od}/{sid}.npy", beat_feats(wav)); n += 1
        except Exception as e:
            bad += 1; print("err", sid, str(e)[:80], flush=True)
        if n % 200 == 0:
            print(f"[m2:beat] {n}", flush=True)
else:
    import librosa, torch
    from transformers import ClapModel, ClapProcessor
    model = ClapModel.from_pretrained(a.clap).cuda().eval(); proc = ClapProcessor.from_pretrained(a.clap)
    for sid, wav in items:
        try:
            y48, _ = librosa.load(wav, sr=48000, mono=True, duration=N_T / HZ)
            with torch.no_grad():
                x = proc(audio=[y48], sampling_rate=48000, return_tensors="pt")
                e = model.get_audio_features(**{k: v.cuda() for k, v in x.items()})
                e = e if torch.is_tensor(e) else e.pooler_output   # transformers 5 returns a ModelOutput
                e = torch.nn.functional.normalize(e, dim=-1)[0].float().cpu().numpy()
            np.save(f"{od}/{sid}.npy", e.astype(np.float16)); n += 1
        except Exception as e:
            bad += 1; print("err", sid, str(e)[:80], flush=True)
        if n % 200 == 0:
            print(f"[m2:clap] {n}", flush=True)
print(f"[m2:{a.phase}] done {n} bad {bad}", flush=True)
