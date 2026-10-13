"""build_cur_meta.py <data_dir> <ids.txt> <exclude_ids.txt> <out_prefix> <n_parts>: H3 metadata (f0 ref) for curated, not-yet-encoded clips, split in parts."""
import csv, json, os, sys
D, ids, exc, out, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
csv.field_size_limit(10 ** 9)
src = open("/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3/e3_h3_zeroshot.py").read()
ns = {}; exec(src[src.index("def h3_prompt"):src.index("\n\n\n", src.index("def h3_prompt"))], ns)
cap = {r["sample_id"]: r["caption"] for r in csv.DictReader(open("/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/h3_train36k.csv"))}
keep = set(open(ids).read().split()); done = set(open(exc).read().split())
rows = []
for f in sorted(os.listdir(f"{D}/video")):
    s = f[:-4]
    if not f.endswith(".mp4") or ".tmp" in f or s not in cap or s not in keep or s in done: continue
    if not (os.path.exists(f"{D}/ref/{s}_f0.png") and os.path.exists(f"{D}/av/{s}.mp4")): continue
    rows.append(dict(video=f"av/{s}.mp4", prompt=ns["h3_prompt"](cap[s]), input_audio=f"av/{s}.mp4",
                     references=[dict(type="image", image=f"ref/{s}_f0.png"), dict(type="audio", audio=f"av/{s}.mp4")], frame_rate=24))
for k in range(n):
    part = rows[k::n]; json.dump(part, open(f"{out}_p{k}.json", "w"), ensure_ascii=False); print(k, len(part))
