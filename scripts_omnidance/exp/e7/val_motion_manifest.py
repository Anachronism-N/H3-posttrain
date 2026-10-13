"""build_manifest + rewrite prompts to E10 motion captions (TA on motion clauses). usage: <out_root> <tag> <manifest.csv>"""
import csv, subprocess, sys
csv.field_size_limit(10 ** 9)
out, tag, man = sys.argv[1:4]
E = "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp"
import os
subprocess.run(["python3", "/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/eval/build_manifest.py", out, tag, man, ",".join(c for c in ["joint", "tswap", "aswap", "audio_only"] if os.path.isdir(f"{out}/{c}"))], check=True)
cap = {r["sample_id"]: r["motion_caption"] for r in csv.DictReader(open(f"{E}/E10/val47_motion_captions.csv"))}
src = {r["sample_id"]: r["swap_src"] for r in csv.DictReader(open(f"{E}/E2/full_tswap.csv"))}
base = [r["sample_id"] for r in csv.DictReader(open(f"{E}/E2/full_joint.csv"))]
other = {b: base[(i + 23) % len(base)] for i, b in enumerate(base)}
rows = list(csv.DictReader(open(man)))
for r in rows:
    sid = r["key"].split("/")[-1].split("_")[0] if r["key"].split("/")[-1] not in cap else r["key"].split("/")[-1]
    if r["cond"] == "tswap":
        r["prompt"], r["alt_prompt"] = cap[src[sid]], cap[sid]
    else:
        r["prompt"], r["alt_prompt"] = cap[sid], cap[other[sid]]
w = csv.DictWriter(open(man, "w", newline=""), fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows), "rows, motion-caption prompts")
