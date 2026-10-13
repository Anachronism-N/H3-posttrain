"""cache compact music features for G2: mean of MERT layers 6..12 -> [81,1024] fp16, 32 threads.
python cache_music.py <csv sample_id,mert_path> <out_dir>"""
import csv, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
csv.field_size_limit(10 ** 9)
rows = list(csv.DictReader(open(sys.argv[1]))); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
if len(sys.argv) > 4: rows = rows[int(sys.argv[3])::int(sys.argv[4])]
def one(r):
    f = f"{out}/{r['sample_id']}.npy"
    if os.path.exists(f): return 0
    try:
        m = torch.load(r["mert_path"], map_location="cpu", weights_only=True)[0, 6:13].float().mean(0)[:, :81].t().numpy()
        np.save(f, m.astype(np.float16)); return 1
    except Exception as e:
        print("err", r["sample_id"], str(e)[:80], flush=True); return 0
with ThreadPoolExecutor(32) as ex:
    n = 0
    for i, k in enumerate(ex.map(one, rows)):
        n += k
        if i % 1000 == 0: print(i, n, flush=True)
print("done", n, flush=True)
