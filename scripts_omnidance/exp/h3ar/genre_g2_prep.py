"""genre_g2_prep.py (10-15, G2): pick held-out pond5 clips the trained audio encoder is confident about (one per requested
genre), cut the exact 5.17 s segment the features came from to a wav, and write a body JSON for sample_h3_ar.py:
{"<key>_enc": {"*": caption chosen by the audio encoder}, "<key>_gen": {"*": generic sentence}} plus a meta JSON with the
encoder's class probabilities. The audio encoder only sees MERT of the clip; it never sees the genre folder name.
"""
import glob, json, os, random, sys
import numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from genre_clip import GENRES, CAP, AudioEnc, mert_vec, logits

O = sys.argv[1]; want = sys.argv[2].split(",")
ck = torch.load(f"{O}/clip/audio_enc.pt", weights_only=False); TT = torch.load(f"{O}/clip/text_targets.pt", weights_only=False)
enc = AudioEnc(); enc.load_state_dict(ck["state"]); enc.eval()
Tm = TT["T"].float().mean(1)
items = [(g, f) for g in GENRES for f in sorted(glob.glob(f"{O}/feat/{g}/*.pt"))]
random.Random(0).shuffle(items); val = items[:len(items) // 10]          # same split as genre_clip.py
src = {os.path.basename(l.split("\t")[1]).rsplit(".", 1)[0]: l.rstrip("\n").split("\t")[1] for l in open(f"{O}/list.tsv")}
os.makedirs(f"{O}/g2", exist_ok=True)
body, meta = {}, {}
for g in want:
    cand = [f for gg, f in val if gg == g]
    X = (torch.stack([mert_vec(f) for f in cand]) - ck["mu"]) / ck["sd"]
    with torch.no_grad():
        P = logits(enc, X, Tm).softmax(1)
    j = int(P[:, GENRES.index(g)].argmax()); f = cand[j]; p = P[j]
    stem = os.path.basename(f)[:-3]; st = float(torch.load(f, weights_only=False)["start"])
    info = torchaudio.info(src[stem]); sr = info.sample_rate
    w, sr = torchaudio.load(src[stem], frame_offset=int(st * sr), num_frames=int(124 / 24 * sr) + sr // 10)
    wav = f"{O}/g2/{g}_{stem}.wav"; sf.write(wav, w.t().numpy(), sr)
    pred = GENRES[int(p.argmax())]
    body[f"{g}_enc"] = {"*": CAP[pred]}
    body[f"{g}_gen"] = {"*": "The dancer performs a freestyle dance to the music, moving in sync with its beat and rhythm."}
    meta[g] = {"wav": wav, "src": src[stem], "start": st, "pred": pred, "p": {k: round(float(v), 3) for k, v in zip(GENRES, p)}}
    print(g, "->", pred, meta[g]["p"], wav, flush=True)
json.dump(body, open(f"{O}/g2/body.json", "w"), indent=1); json.dump(meta, open(f"{O}/g2/meta.json", "w"), indent=1)
