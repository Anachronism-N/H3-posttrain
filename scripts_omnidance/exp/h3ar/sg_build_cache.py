"""sg_build_cache.py: build a node-local training cache for the SG512 / GT512 attribution control (10-13).
For every cache-relative path in the list: take the original split-training tuple (+ its rich-prompt posi and nega
sidecars), optionally replace input_latents by the self-generated latent (sg_gen.py), and write
  /opt/h3cache/<name>/<rel>                  (real file, so realpath-based sidecar lookup stays inside <name>)
  /opt/h3cache/posi/<name>/<rel>.posi.pt     /opt/h3cache/nega/<name>/<rel>.nega.pt
--export <dir>: instead copy the resolved tuple and sidecars to <dir> (to move the cache to a node without it);
a later run with --src_root <dir> --src_flat reads from there.
"""
import argparse, os, shutil
import torch

p = argparse.ArgumentParser()
p.add_argument("--list", required=True); p.add_argument("--src_root", required=True)
p.add_argument("--name", default=""); p.add_argument("--lat_dir", default="")
p.add_argument("--export", default=""); p.add_argument("--src_flat", action="store_true")
a = p.parse_args()
H = "/opt/h3cache"
rels = [l.strip() for l in open(a.list) if l.strip()]


def sources(rel):
    if a.src_flat:
        b = os.path.join(a.src_root, rel)[:-4]
        return b + ".pth", b + ".posi.pt", b + ".nega.pt"
    real = os.path.realpath(os.path.join(a.src_root, rel))
    r = os.path.relpath(real, H)[:-4]
    return real, f"{H}/posi/{r}.posi.pt", f"{H}/nega/{r}.nega.pt"


n = 0
for rel in rels:
    pth, posi, nega = sources(rel)
    for f in (pth, posi, nega):
        assert os.path.exists(f), f
    if a.export:
        b = os.path.join(a.export, rel)[:-4]
        os.makedirs(os.path.dirname(b), exist_ok=True)
        for f, ext in ((pth, ".pth"), (posi, ".posi.pt"), (nega, ".nega.pt")):
            if not os.path.exists(b + ext):
                shutil.copyfile(f, b + ext + ".tmp"); os.replace(b + ext + ".tmp", b + ext)
    else:
        dst = f"{H}/{a.name}/{rel}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if a.lat_dir:
            data = torch.load(pth, map_location="cpu", weights_only=False)
            sg = torch.load(f"{a.lat_dir}/{rel.replace('/', '_')[:-4]}.pt", map_location="cpu", weights_only=False)
            gt = data[0]["input_latents"]
            x0 = sg["x0"].unsqueeze(0).to(gt.dtype)
            assert x0.shape == gt.shape and torch.isfinite(x0.float()).all(), rel
            data[0]["input_latents"] = x0
            torch.save(data, dst + ".tmp"); os.replace(dst + ".tmp", dst)
        else:
            shutil.copyfile(pth, dst + ".tmp"); os.replace(dst + ".tmp", dst)
        r = f"{a.name}/{rel}"[:-4]
        for f, d in ((posi, f"{H}/posi/{r}.posi.pt"), (nega, f"{H}/nega/{r}.nega.pt")):
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copyfile(f, d + ".tmp"); os.replace(d + ".tmp", d)
    n += 1
    if n % 64 == 0:
        print(f"[build] {n}/{len(rels)}", flush=True)
print(f"[build] done {n} -> {a.export or H + '/' + a.name}", flush=True)
