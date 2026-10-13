"""curate_cache.py <src_cache> <metadata.json> <dst_cache> <ids.txt> [nproc=8]
Symlink view of an upstream H3 data_process cache keeping only samples whose id is in ids.txt.
Mapping (accelerate non-shuffled sharding, verified 10-10): <src>/<r>/<i>.pth <-> metadata[(i*nproc + r) % len]; wrapped padding is dropped."""
import json, os, sys
src, meta, dst, ids = sys.argv[1:5]; P = int(sys.argv[5]) if len(sys.argv) > 5 else 8
keep = set(open(ids).read().split())
M = [os.path.basename(m["video"])[:-4] for m in json.load(open(meta))]
n = k = 0; kept = []
for r in range(P):
    os.makedirs(f"{dst}/{r}", exist_ok=True)
    for f in os.listdir(f"{src}/{r}"):
        i = int(f[:-4]); j = i * P + r
        if j >= len(M): continue
        n += 1
        if M[j] in keep:
            os.symlink(os.path.realpath(f"{src}/{r}/{f}"), f"{dst}/{r}/{f}"); k += 1; kept.append(M[j])
open(f"{dst}/ids.txt", "w").write("\n".join(sorted(kept)) + "\n")
print(f"[curate] {k}/{n} kept -> {dst}")
