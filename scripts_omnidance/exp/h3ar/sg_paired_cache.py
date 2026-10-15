"""sg_paired_cache.py (10-15): paired-noise reflow control.
Copy the SG512 node cache to /opt/h3cache/<name>, adding to every entry the exact initial noise "od_noise" that sg_gen.py
used to sample its target (seed stored in the SG latent file). loss.py then trains FM on the (noise, x0) line of that
sample instead of fresh noise. Same targets / conditions / sidecars as the SG512 run -> only the noise pairing differs.
python sg_paired_cache.py --list SG512/sg512.txt --lat_dir SG512/lat --src /opt/h3cache/sg512 --name sg512p
"""
import argparse, hashlib, os
import torch

p = argparse.ArgumentParser()
p.add_argument("--list", required=True); p.add_argument("--lat_dir", required=True)
p.add_argument("--src", default="/opt/h3cache/sg512"); p.add_argument("--name", default="sg512p")
p.add_argument("--inv", action="store_true", help="--lat_dir holds gt_invert.py outputs ({'z'}): keep the GT target, od_noise = inverted z")
a = p.parse_args()
H = "/opt/h3cache"; src_name = os.path.relpath(a.src, H)
rels = [l.strip() for l in open(a.list) if l.strip()]
for i, rel in enumerate(rels):
    dst = f"{H}/{a.name}/{rel}"
    if not os.path.exists(dst):
        data = torch.load(f"{a.src}/{rel}", map_location="cpu", weights_only=False)
        sg = torch.load(f"{a.lat_dir}/{rel.replace('/', '_')[:-4]}.pt", map_location="cpu", weights_only=False)
        x0 = data[0]["input_latents"]
        if a.inv:
            assert sg["rel"] == rel and os.path.realpath(f"{a.src}/{rel}") == sg["src"], rel
            z = sg["z"].unsqueeze(0)
            assert z.shape == x0.shape and torch.isfinite(z.float()).all(), rel
            data[0]["od_noise"] = z.to(torch.bfloat16)
        else:
            assert sg["rel"] == rel and sg["seed"] == int(hashlib.sha256(rel.encode()).hexdigest()[:8], 16), rel
            assert torch.equal(sg["x0"].unsqueeze(0).to(x0.dtype), x0), f"cache target != SG latent: {rel}"
            g = torch.Generator(device="cpu").manual_seed(sg["seed"])
            data[0]["od_noise"] = torch.randn(x0.shape, generator=g, dtype=torch.float32).to(torch.bfloat16)  # as sg_gen.py
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        torch.save(data, dst + ".tmp"); os.replace(dst + ".tmp", dst)
    r = rel[:-4]
    for side in ("posi", "nega"):
        s, d = f"{H}/{side}/{src_name}/{r}.{side}.pt", f"{H}/{side}/{a.name}/{r}.{side}.pt"
        if os.path.exists(s) and not os.path.exists(d):
            os.makedirs(os.path.dirname(d), exist_ok=True); os.symlink(s, d)
    if (i + 1) % 64 == 0:
        print(f"[paired] {i + 1}/{len(rels)}", flush=True)
print(f"[paired] done {len(rels)} -> {H}/{a.name}", flush=True)
