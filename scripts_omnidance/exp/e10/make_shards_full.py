"""shards for E2_run_node from a dir with full_{cond}.csv (e.g. E7/val_full = intended prompts). usage: <dir> <n_nodes> <conds>"""
import csv, os, sys
csv.field_size_limit(10 ** 9)
D, nn, conds = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
os.makedirs(f"{D}/shards", exist_ok=True)
for f in os.listdir(f"{D}/shards"):
    os.remove(f"{D}/shards/{f}")
jobs = [(c, r) for c in conds for r in csv.DictReader(open(f"{D}/full_{c}.csv"))]
G = nn * 8; sh = {}
for j, (c, r) in enumerate(jobs):
    g = j % G; sh.setdefault((g // 8, g % 8, c), []).append(r)
for (nd, gp, c), lst in sh.items():
    with open(f"{D}/shards/n{nd}_g{gp}_{c}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lst[0].keys())); w.writeheader(); w.writerows(lst)
print(len(jobs), "jobs", len(sh), "shards")
