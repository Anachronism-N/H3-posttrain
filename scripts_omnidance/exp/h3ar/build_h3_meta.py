"""metadata.json for upstream H3 Ref2VA training from /dev/shm/h3data (this node's converted clips).
usage: build_h3_meta.py <data_dir> <ref: f0|mid> [limit]"""
import csv, json, os, sys
sys.path.insert(0, "/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3")
csv.field_size_limit(10 ** 9)
D, ref = sys.argv[1], sys.argv[2]; lim = int(sys.argv[3]) if len(sys.argv) > 3 else 0
src = open("/apdcephfs_gy4/share_302533218/cedricnie/diff/scripts_omnidance/exp/h3/e3_h3_zeroshot.py").read()
ns = {}; exec(src[src.index("def h3_prompt"):src.index("\n\n\n", src.index("def h3_prompt"))], ns)
cap = {r["sample_id"]: r["caption"] for r in csv.DictReader(open("/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/H3AR/h3_train36k.csv"))}
rows = []
for f in sorted(os.listdir(f"{D}/video")):
    if not f.endswith(".mp4") or ".tmp" in f: continue
    s = f[:-4]
    if s not in cap or not os.path.exists(f"{D}/ref/{s}_{ref}.png"): continue
    if not os.path.exists(f"{D}/av/{s}.mp4"): continue
    rows.append(dict(video=f"av/{s}.mp4", prompt=ns["h3_prompt"](cap[s]), input_audio=f"av/{s}.mp4",
                     references=[dict(type="image", image=f"ref/{s}_{ref}.png"), dict(type="audio", audio=f"av/{s}.mp4")], frame_rate=24))
    if lim and len(rows) >= lim: break
out = f"{D}/metadata_{ref}{'_'+str(lim) if lim else ''}.json"
json.dump(rows, open(out, "w"), ensure_ascii=False); print(len(rows), out)
