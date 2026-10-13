"""(env MAP_ID=1: map to sample id instead of the MERT file path)
build_music_map.py <cache_view_dir> <src_metadata.json> <out.json> [nproc=8]
Map every cached sample (realpath of <view>/<r>/<i>.pth, e.g. a curate_cache.py symlink view) to its MERT feature file.
Mapping of the source cache: <src>/<r>/<i>.pth <-> metadata[(i*nproc + r) % len] (curate_cache.py)."""
import glob, json, os, sys
view, meta, out = sys.argv[1:4]; P = int(sys.argv[4]) if len(sys.argv) > 4 else 8
MF = ["/apdcephfs_gy4/share_302533218/cedricnie/diff/data/omnidance/audio_cache/mert_features",
      "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/G2/mert_val"]
M = [os.path.basename(m["video"])[:-4] for m in json.load(open(meta))]
res, miss = {}, []
for f in glob.glob(f"{view}/*/*.pth"):
    real = os.path.realpath(f)
    r, i = int(os.path.basename(os.path.dirname(real))), int(os.path.basename(real)[:-4])
    sid = M[(i * P + r) % len(M)]
    p = next((f"{d}/{sid}.pt" for d in MF if os.path.exists(f"{d}/{sid}.pt")), None)
    if p:
        res[real] = sid if os.environ.get("MAP_ID") else p
    else:
        miss.append(sid)
json.dump(res, open(out, "w"))
print(f"[map] {len(res)} mapped, {len(miss)} without MERT {miss[:5]}")
