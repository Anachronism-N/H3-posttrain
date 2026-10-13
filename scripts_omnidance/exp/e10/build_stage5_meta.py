import csv, glob, os
csv.field_size_limit(10 ** 9)
R = "/apdcephfs_gy4/share_302533218/cedricnie/diff"
cap = {}
for f in glob.glob(f"{R}/logs_omnidance/exp/E10/train_caps/cap_*.csv"):
    for r in csv.DictReader(open(f)):
        if r["motion_caption"].count(".") >= 3:
            cap[r["sample_id"]] = r["motion_caption"].strip()
out = []
for r in csv.DictReader(open(f"{R}/data/omnidance/metadata_train.csv")):
    s = os.path.basename(r["video"])[:-4]
    if s in cap:
        out.append(dict(video=r["video"], input_audio=r["input_audio"], prompt=r["prompt"].rstrip() + " Detailed body movements: " + cap[s]))
w = csv.DictWriter(open(f"{R}/data/omnidance/metadata_stage5_motioncap.csv", "w", newline=""), fieldnames=["video", "input_audio", "prompt"])
w.writeheader(); w.writerows(out)
print(len(cap), "captions ->", len(out), "rows")
