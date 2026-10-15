"""music_caption.py (10-15, mentor idea: map music into H3's text space instead of fine-tuning the distilled backbone).
For every audio of the first N val rows (joint + aswap), compute per-second dance intensity (onset strength, quantised
to 1..5 with quantiles over all listed clips) and beat times, and write a text body that H3's rich-prompt template
wraps, plus two fixed-amplitude probe bodies (calm / energetic) used to test whether text controls motion amplitude.
Output JSON {"timed": {audio_path: body}, "calm": {"*": body}, "energetic": {"*": body}}.
"""
import argparse, csv, json
import librosa, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--csv_dir", default="/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/E2")
p.add_argument("--rows", type=int, default=3); p.add_argument("--out", required=True)
p.add_argument("--dur", type=float, default=124 / 24)
a = p.parse_args()
csv.field_size_limit(10 ** 9)
auds = []
for c in ("joint", "aswap"):
    auds += [r["audio_path"] for r in list(csv.DictReader(open(f"{a.csv_dir}/full_{c}.csv")))[:a.rows]]
auds = list(dict.fromkeys(auds))
WORDS = {1: "barely moves, swaying gently in place", 2: "makes small, relaxed steps and soft arm movements",
         3: "dances with moderate bouncing steps and clear arm gestures",
         4: "dances energetically with strong arm swings, body rolls and quick footwork",
         5: "explodes into big, powerful moves: wide arm throws, high kicks and jumps"}
feat = {}
for f in auds:
    y, sr = librosa.load(f, sr=22050, mono=True, duration=a.dur)
    env = librosa.onset.onset_strength(y=y, sr=sr)
    t = librosa.times_like(env, sr=sr)
    sec = [float(env[(t >= s) & (t < s + 1)].mean()) for s in range(int(np.ceil(a.dur)))]
    tempo, bt = librosa.beat.beat_track(onset_envelope=env, sr=sr)
    feat[f] = (sec, [float(x) for x in librosa.frames_to_time(bt, sr=sr) if x < a.dur], float(np.atleast_1d(tempo)[0]))
allv = np.concatenate([v[0] for v in feat.values()])
qs = np.quantile(allv, [0.2, 0.4, 0.6, 0.8])
timed = {}
for f, (sec, bt, tempo) in feat.items():
    lv = [int(1 + np.searchsorted(qs, v)) for v in sec]
    parts = [f"from {s} to {min(s + 1, a.dur):.0f} seconds the music intensity is {l} of 5 and <Subject 1> {WORDS[l]}"
             for s, l in enumerate(lv)]
    timed[f] = ("<Subject 1> dances to the music and follows its changing energy second by second: " + "; ".join(parts)
                + f". The tempo is about {tempo:.0f} BPM and the beats fall at " + ", ".join(f"{x:.1f}" for x in bt)
                + " seconds; <Subject 1> places a clear accent on each beat.")
    print(f.split("/")[-1], "levels", lv, f"tempo {tempo:.0f}", flush=True)
out = {"timed": timed,
       "calm": {"*": "<Subject 1> dances very calmly with small, restrained movements: gentle swaying, soft hand gestures, "
                     "feet mostly staying in place, no big arm swings, no kicks and no jumps."},
       "energetic": {"*": "<Subject 1> dances with big, exaggerated, high-energy movements: wide powerful arm swings above the head, "
                          "high kicks, jumps, fast spins and deep body waves, using the whole space with maximum amplitude."}}
json.dump(out, open(a.out, "w"), indent=1)
print("written", a.out)
