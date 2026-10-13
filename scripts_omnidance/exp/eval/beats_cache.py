"""Precompute librosa beat times for all audio in manifests -> beats.json (run in a clean process; no torch)."""
import csv, json, os, sys
import librosa
csv.field_size_limit(10 ** 9)
out = sys.argv[1]
cache = json.load(open(out)) if os.path.exists(out) else {}
aud = set()
for m in sys.argv[2:]:
    for r in csv.DictReader(open(m)):
        aud.add(r["audio"]); aud.add(r["alt_audio"])
for a in sorted(aud - set(cache)):
    y, sr = librosa.load(a, sr=22050, mono=True, duration=6.0)
    _, bf = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    cache[a] = librosa.frames_to_time(bf, sr=sr).tolist()
json.dump(cache, open(out, "w"))
print(len(cache), "audios")
