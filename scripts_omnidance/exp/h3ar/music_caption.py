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
p.add_argument("--q_rows", type=int, default=0, help="v3: thresholds from the first N val rows (0 = all 47, joint+aswap audios)")
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
# "seg" (10-15 v2): text controls timing at ~2.5 s granularity but not per second -> 3 segments, 3 contrasting levels
# (global terciles over all segments of the listed clips), vivid wording as in the calm / energetic probes.
SEGW = {1: "barely moves, only gentle swaying and soft hand gestures with the feet staying in place",
        2: "dances with moderate bouncing steps and clear arm gestures",
        3: "bursts into big, exaggerated, high-energy moves: wide arm swings above the head, high kicks and jumps"}
nseg = 3; edges = np.linspace(0, a.dur, nseg + 1)
segv = {}
for f in auds:
    y, sr = librosa.load(f, sr=22050, mono=True, duration=a.dur)
    env = librosa.onset.onset_strength(y=y, sr=sr); t = librosa.times_like(env, sr=sr)
    segv[f] = [float(env[(t >= edges[i]) & (t < edges[i + 1])].mean()) for i in range(nseg)]
q3 = np.quantile(np.concatenate(list(segv.values())), [1 / 3, 2 / 3])
seg = {}
for f, v in segv.items():
    lv = [int(1 + np.searchsorted(q3, x)) for x in v]
    seg[f] = " ".join(f"{'From' if i == 0 else 'Then from'} {edges[i]:.1f} to {edges[i + 1]:.1f} seconds, matching the "
                      f"{['quiet', 'medium', 'loud and driving'][l - 1]} music, <Subject 1> {SEGW[l]}." for i, l in enumerate(lv))
    print(f.split("/")[-1], "seg levels", lv, flush=True)
# "seg2" (v3): thresholds from the whole val set (not just the probed clips) + tempo word
qa = []
for c in ("joint", "aswap"):
    rr = list(csv.DictReader(open(f"{a.csv_dir}/full_{c}.csv")))
    qa += [r["audio_path"] for r in (rr[:a.q_rows] if a.q_rows else rr)]
qa = list(dict.fromkeys(qa))
allseg, tempos = {}, {}
for f in qa:
    y, sr = librosa.load(f, sr=22050, mono=True, duration=a.dur)
    env = librosa.onset.onset_strength(y=y, sr=sr); t = librosa.times_like(env, sr=sr)
    allseg[f] = [float(env[(t >= edges[i]) & (t < edges[i + 1])].mean()) for i in range(nseg)]
    tempos[f] = float(np.atleast_1d(librosa.beat.beat_track(onset_envelope=env, sr=sr)[0])[0])
Q = np.quantile(np.concatenate(list(allseg.values())), [1 / 3, 2 / 3])
print("seg2 thresholds", Q, "from", len(qa), "audios", flush=True)
seg2 = {}
for f in auds:
    lv = [int(1 + np.searchsorted(Q, x)) for x in allseg[f]]
    bpm = tempos[f]; tw = "slow, flowing" if bpm < 100 else ("steady" if bpm < 128 else "fast, punchy")
    seg2[f] = (f"The music has a {tw} rhythm at about {bpm:.0f} BPM and <Subject 1> moves at a matching {tw} pace. "
               + " ".join(f"{'From' if i == 0 else 'Then from'} {edges[i]:.1f} to {edges[i + 1]:.1f} seconds, matching the "
                          f"{['quiet', 'medium', 'loud and driving'][l - 1]} music, <Subject 1> {SEGW[l]}." for i, l in enumerate(lv)))
    print(f.split("/")[-1], "seg2 levels", lv, f"bpm {bpm:.0f}", flush=True)
out = {"timed": timed, "seg": seg, "seg2": seg2,
       "calm": {"*": "<Subject 1> dances very calmly with small, restrained movements: gentle swaying, soft hand gestures, "
                     "feet mostly staying in place, no big arm swings, no kicks and no jumps."},
       "energetic": {"*": "<Subject 1> dances with big, exaggerated, high-energy movements: wide powerful arm swings above the head, "
                          "high kicks, jumps, fast spins and deep body waves, using the whole space with maximum amplitude."}}
json.dump(out, open(a.out, "w"), indent=1)
print("written", a.out)
