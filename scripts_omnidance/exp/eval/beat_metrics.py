"""Offline beat metrics from saved keypoints (clean process, librosa OK).
For each row: vs conditioning audio and vs alt audio:
  bas_b  Bailando BAS: mean over motion beats (local minima of joint speed) of exp(-d^2/(2*sigma^2)), d=dist to nearest music beat, sigma=3 frames
  oc     motion-onset correlation: max_{lag in [-2,2] frames} pearson(joint speed, audio onset strength resampled to video fps)
Usage: beat_metrics.py <manifest.csv> <kps_dir> <out.jsonl>
"""
import csv, json, os, sys
import numpy as np, librosa
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
csv.field_size_limit(10 ** 9)
man, kd, out = sys.argv[1:4]
onset_cache, beat_cache = {}, {}

def audio_feats(a, fps, n):
    key = (a, fps, n)
    if key not in onset_cache:
        y, sr = librosa.load(a, sr=22050, mono=True, duration=n / fps + 0.1)
        hop = 512
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
        ft = np.arange(n) / fps
        onset_cache[key] = np.interp(ft, t, env)
        _, bf = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, trim=False)
        beat_cache[key] = librosa.frames_to_time(bf, sr=sr, hop_length=hop)
    return onset_cache[key], beat_cache[key]

def speed(k, h):
    k = k / h
    d = np.linalg.norm(k[1:] - k[:-1], axis=-1)
    d = np.nan_to_num(d, nan=0.0).sum(-1)
    return gaussian_filter1d(d, 1.0)

def scores(sp, fps, a):
    env, bts = audio_feats(a, fps, len(sp) + 1)
    env = env[1:]
    mins = argrelextrema(sp, np.less, order=2)[0]
    mb = mins / fps
    sig = 3 / fps
    bas = float(np.mean([np.exp(-np.min((bts - t) ** 2) / (2 * sig ** 2)) for t in mb])) if len(mb) and len(bts) else float("nan")
    best = -1
    for lag in range(-2, 3):
        x = sp[max(0, lag):len(sp) + min(0, lag)]; y = env[max(0, -lag):len(env) + min(0, -lag)]
        if len(x) > 5 and x.std() > 0 and y.std() > 0:
            best = max(best, float(np.corrcoef(x, y)[0, 1]))
    return bas, best

done = set()
if os.path.exists(out):
    done = {json.loads(l)["key"] for l in open(out)}
fo = open(out, "a")
for r in csv.DictReader(open(man)):
    if r["key"] in done: continue
    f = os.path.join(kd, r["key"].replace("/", "__") + ".npz")
    if not os.path.exists(f): continue
    z = np.load(f); sp = speed(z["kps"], float(z["h"])); fps = float(z["fps"])
    b1, o1 = scores(sp, fps, r["audio"]); b2, o2 = scores(sp, fps, r["alt_audio"])
    fo.write(json.dumps(dict(key=r["key"], tag=r["tag"], cond=r["cond"], bas_b=b1, bas_b_alt=b2, oc=o1, oc_alt=o2)) + "\n")
print("done")
