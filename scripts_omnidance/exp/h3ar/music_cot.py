"""music_cot.py (10-16): music chain-of-thought with an audio LLM (Qwen2.5-Omni-7B, thinker only, no speech output).
Step 1 (listen): the model hears the 5.17 s clip and writes a short structured analysis (genre, tempo feel, energy and how it
changes over the three 1.7 s parts, accents/drops, mood). Librosa BPM/beat times/onset energy are given as measured hints so
the timing is grounded. Step 2 (choreograph): from that analysis the same model writes the dance description for H3's
detailed_description slot: one dance style + three time-stamped parts, each naming body parts, amplitude and speed
(MotionScript-style wording), matched to the analysed energy. Output JSON {audio_path: {"analysis", "body", "raw"}}
usable as sample_h3_ar.py --body_json <out> --body_key cot.
python music_cot.py --rows 3 --out cot.json   (val audios of the first N joint + aswap rows)
"""
import argparse, csv, json, os
import numpy as np, librosa, torch

p = argparse.ArgumentParser()
p.add_argument("--csv_dir", default="/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/E2")
p.add_argument("--rows", type=int, default=3); p.add_argument("--out", required=True)
p.add_argument("--extra", default="", help="comma-separated extra wav paths (e.g. pond5 rock/zen clips)")
p.add_argument("--model", default="/apdcephfs_gy4/share_302533218/cedricnie/models/Qwen2.5-Omni-7B")
p.add_argument("--dur", type=float, default=124 / 24)
a = p.parse_args()
csv.field_size_limit(10 ** 9)
auds = []
for c in ("joint", "aswap"):
    auds += [r["audio_path"] for r in list(csv.DictReader(open(f"{a.csv_dir}/full_{c}.csv")))[:a.rows]]
auds += [x for x in a.extra.split(",") if x]
auds = list(dict.fromkeys(auds))
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
proc = Qwen2_5OmniProcessor.from_pretrained(a.model)
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()  # text-only thinker
SYS = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
e = a.dur / 3


def ask(wav, text, max_new=300):
    conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": ([{"type": "audio", "audio": wav}] if wav is not None else []) + [{"type": "text", "text": text}]}]
    t = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=t, audio=[wav] if wav is not None else None, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    out = out[0] if isinstance(out, tuple) else out
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


res = json.load(open(a.out)) if os.path.exists(a.out) else {}
import re as _re


def clean(t):  # cut the model's habit of continuing the chat ("Human: ...", "What do you think ...")
    t = _re.split(r"(Human:|User:|What do you think|Do you have any other|So, what do you think)", t)[0]
    return " ".join(t.split()).strip()


for f in auds:
    if f in res:
        continue
    y, sr = librosa.load(f, sr=16000, mono=True, duration=a.dur)
    env = librosa.onset.onset_strength(y=y, sr=sr); t = librosa.times_like(env, sr=sr)
    tempo, bt = librosa.beat.beat_track(onset_envelope=env, sr=sr)
    bts = [round(float(x), 2) for x in librosa.frames_to_time(bt, sr=sr) if x < a.dur]
    segE = [float(env[(t >= i * e) & (t < (i + 1) * e)].mean()) for i in range(3)]
    rel = [round(x / (np.mean(segE) + 1e-6), 2) for x in segE]
    hint = (f"Measured: tempo about {float(np.atleast_1d(tempo)[0]):.0f} BPM; beats at {bts} s; relative onset energy of the three "
            f"parts (0-{e:.1f}s, {e:.1f}-{2 * e:.1f}s, {2 * e:.1f}-{a.dur:.1f}s) = {rel}.")
    genre = clean(ask(y, "What kind of music is this? Answer in one line: genre, main instruments, and whether it is calm, "
                      "medium or energetic.", 60))  # v2: listen first, no measured hints (v1 just echoed the hints)
    analysis = clean(ask(y, "Listen to this music clip. It was identified as: " + genre + " " + hint + " In 3-4 short lines "
                         "describe how loud/driving each of the three parts is (quiet, medium, loud and driving), any accents, hits "
                         "or drops and when, and the overall mood.", 160))
    body = clean(ask(None, "You are a choreographer. Music: " + genre + "\nAnalysis: " + analysis + "\n" + hint + "\n"
               "Choose the dance style that best fits this music from: hip-hop, K-pop, jazz dance, contemporary, Chinese classical "
               "dance, house, popping, ballet. (Rock, electronic and hip-hop music usually suit hip-hop, house or K-pop; calm piano, "
               "ambient or traditional Chinese music suit Chinese classical dance, contemporary or ballet.) "
               "Write the dance for one dancer called <Subject 1>. Output one paragraph of 80-130 English words, no lists, no "
               "headings, nothing after the paragraph. First sentence: '<Subject 1> performs <dance style> to the music.' Then describe the three parts "
               f"in order, starting each with 'From 0.0 to {e:.1f} seconds', 'From {e:.1f} to {2 * e:.1f} seconds', 'From {2 * e:.1f} "
               f"to {a.dur:.1f} seconds'. For each part name the body parts, the size of the movement (small/medium/big) and the speed "
               "(slow/steady/fast), and make the size and speed follow the music's energy in that part (quiet -> small and slow, "
               "loud and driving -> big and fast). Put clear accents on the measured beats. Do not describe clothing, face, camera "
               "or background.", max_new=260))
    res[f] = {"genre": genre, "analysis": analysis, "body": body, "hint": hint}
    json.dump(res, open(a.out, "w"), indent=1)
    print("==", os.path.basename(f), "\n", analysis[:400], "\n--\n", body[:600], flush=True)
json.dump({"cot": {k: v["body"] for k, v in res.items()}, "_full": res}, open(a.out.replace(".json", "_body.json"), "w"), indent=1)
print("written", a.out)
