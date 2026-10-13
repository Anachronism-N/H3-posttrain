"""Build eval manifest for an OD-Val-47 run directory: <out_root>/<cond>/*.mp4
Usage: build_manifest.py <out_root> <tag> <manifest.csv> [conds]
Filenames contain '<img_stem>_<audio_stem>.mp4'; sample id = img stem.
"""
import csv, glob, os, sys

csv.field_size_limit(10 ** 9)
out_root, tag, dst = sys.argv[1:4]
conds = sys.argv[4].split(",") if len(sys.argv) > 4 else ["joint", "audio_only", "text_only", "tswap", "aswap"]
E2 = "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/E2"
base = list(csv.DictReader(open(f"{E2}/full_joint.csv")))
n = len(base)
idx = {b["sample_id"]: i for i, b in enumerate(base)}
rows = []
for c in conds:
    for v in sorted(glob.glob(f"{out_root}/{c}/*.mp4")):
        stem = os.path.basename(v)[:-4]
        sid = next((s for s in idx if f"_{s}_" in stem + "_" or stem.endswith(s)), None)
        # image stem is the first sample id occurring in the name
        cands = [(stem.find(s), s) for s in idx if s in stem]
        sid = min(cands)[1]
        i = idx[sid]; b = base[i]
        other_p = base[(i + 23) % n]["prompt"]
        other_a = base[(i + 17) % n]["audio_path"]
        r = dict(key=f"{tag}/{c}/{sid}", video=v, ref_image=b["image_path"], cond=c, tag=tag,
                 prompt=b["prompt"], alt_prompt=other_p, audio=b["audio_path"], alt_audio=other_a)
        if c == "tswap":
            r["prompt"], r["alt_prompt"] = other_p, b["prompt"]
        if c == "aswap":
            r["audio"], r["alt_audio"] = other_a, b["audio_path"]
        rows.append(r)
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rows -> {dst}")
