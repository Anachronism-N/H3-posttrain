"""stage_sel.py <src_cache> <metadata.json> <sel.txt> <tag> <stage_dir> [nproc=8]: symlinks <stage>/<sid>.pth for selected (tag,sid) rows."""
import json, os, sys
src, meta, sel, tag, st = sys.argv[1:6]; P = int(sys.argv[6]) if len(sys.argv) > 6 else 8
want = {l.split()[1] for l in open(sel) if l.split()[0] == tag}
M = [os.path.basename(m["video"])[:-4] for m in json.load(open(meta))]
os.makedirs(st, exist_ok=True); k = 0
for r in range(P):
    for f in os.listdir(f"{src}/{r}"):
        j = int(f[:-4]) * P + int(r)
        if j < len(M) and M[j] in want and not os.path.exists(f"{st}/{M[j]}.pth"):
            os.symlink(os.path.realpath(f"{src}/{r}/{f}"), f"{st}/{M[j]}.pth"); k += 1
print(f"[stage] {tag}: {k}/{len(want)}")
