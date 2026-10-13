"""Prepare a separate H3 training base: fuse a training LoRA, or reproduce the b25-49 Hybrid.

Adapter bases are for training only; inference uses the original base plus the trained LoRA.
Hybrid bases are retained at both training and inference. Sources are never modified.
"""
import argparse, glob, hashlib, json, os
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

p = argparse.ArgumentParser()
p.add_argument("--w", required=True)
p.add_argument("--adapter")
p.add_argument("--hybrid_ref", help="Ref2VA source; --w must be the original FL2VA BF16 directory")
p.add_argument("--out", required=True)
p.add_argument("--scale", type=float, default=1.0)
p.add_argument("--comfy_qkv", action="store_true")
p.add_argument("--device", default="cuda")
a = p.parse_args()
if bool(a.adapter) == bool(a.hybrid_ref):
    p.error("specify exactly one of --adapter and --hybrid_ref")
if os.path.exists(a.out):
    p.error("output already exists; use a new directory to protect existing weights")
if not os.path.isdir(os.path.dirname(os.path.abspath(a.out))):
    p.error("output parent directory does not exist")

cfg = json.load(open(f"{a.w}/transformer/config.json"))
H, Dh = cfg["num_attention_heads"], cfg["attention_head_dim"]
source_files = sorted(glob.glob(f"{a.w}/transformer/model*.safetensors"))
if not source_files:
    p.error("source has no transformer model shards")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


pairs, ref_keys = {}, {}
if a.hybrid_ref:
    ref_cfg = json.load(open(f"{a.hybrid_ref}/transformer/config.json"))
    if {k: v for k, v in cfg.items() if not k.startswith("_")} != {k: v for k, v in ref_cfg.items() if not k.startswith("_")}:
        p.error("FL2VA and Ref2VA architecture configs differ")
    base_keys = {}
    for path in source_files:
        with safe_open(path, "pt") as f:
            for k in f.keys():
                if k in base_keys:
                    p.error(f"duplicate source tensor {k}")
                base_keys[k] = (f.get_slice(k).get_shape(), f.get_slice(k).get_dtype())
    for path in sorted(glob.glob(f"{a.hybrid_ref}/transformer/model*.safetensors")):
        with safe_open(path, "pt") as f:
            for k in f.keys():
                if k in ref_keys:
                    p.error(f"duplicate reference tensor {k}")
                ref_keys[k] = path
                if k not in base_keys or base_keys[k] != (f.get_slice(k).get_shape(), f.get_slice(k).get_dtype()):
                    p.error(f"source/reference shape or dtype mismatch: {k}")
    if set(ref_keys) != set(base_keys):
        p.error("source/reference tensor key sets differ")
    selected = {f"blocks.{i}.adaln_proj.linear.{suffix}" for i in range(25, 50) for suffix in ("weight", "bias")}
    if not selected <= set(base_keys):
        p.error(f"missing Hybrid tensors: {selected - set(base_keys)}")
    if any(dtype not in ("BF16", "F32") for _, dtype in base_keys.values()) or any(base_keys[k][1] != "BF16" for k in selected):
        p.error("Hybrid sources must be original BF16 weights with their FP32 buffers, not converted INT8 weights")
else:
    lora = load_file(a.adapter)
    for k, v in lora.items():
        if ".lora_A" not in k:
            continue
        base = k.split(".lora_A")[0].removeprefix("diffusion_model.")
        kb = k.replace("lora_A", "lora_B")
        alpha_k = k.split(".lora_A")[0] + ".alpha"
        r = v.shape[0]
        s = a.scale * (float(lora[alpha_k]) / r if alpha_k in lora else 1.0)
        pairs[base + ".weight"] = (v, lora[kb], s)
    if not pairs:
        p.error("adapter has no LoRA A/B pairs")
    selected = set(pairs)
    print(f"[fuse] {len(pairs)} LoRA modules from {a.adapter}", flush=True)

os.makedirs(f"{a.out}/transformer")
for d in os.listdir(a.w):
    if d != "transformer":
        os.symlink(os.path.abspath(f"{a.w}/{d}"), f"{a.out}/{d}")
for src in glob.glob(f"{a.w}/transformer/*"):
    if not src.endswith(".safetensors"):
        os.symlink(os.path.abspath(src), f"{a.out}/transformer/{os.path.basename(src)}")

manifest = {"mode": "hybrid_b25_49" if a.hybrid_ref else "training_adapter", "base": os.path.realpath(a.w),
            "reference": os.path.realpath(a.hybrid_ref) if a.hybrid_ref else None,
            "adapter": os.path.realpath(a.adapter) if a.adapter else None, "scale": a.scale,
            "comfy_qkv": a.comfy_qkv, "sources": {}, "outputs": {}, "selected_keys": sorted(selected)}
if a.hybrid_ref:
    manifest["recipe_code"] = "https://github.com/scottmudge/ComfyUI_MinimaxH3HybridLoader"
    manifest["published_int8_variant"] = "https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models"
else:
    manifest["adapter_sha256"] = digest(a.adapter)
done, ref_hashes = 0, set()
for src in source_files:
    dst = f"{a.out}/transformer/{os.path.basename(src)}"
    sd = load_file(src)
    hit = 0
    for k in set(sd) & selected:
        if a.hybrid_ref:
            path = ref_keys[k]
            with safe_open(path, "pt") as f:
                sd[k] = f.get_tensor(k)
            if path not in ref_hashes:
                manifest["sources"][os.path.realpath(path)] = digest(path)
                ref_hashes.add(path)
        else:
            A, B, s = pairs[k]
            A = A.to(device=a.device, dtype=torch.float32)
            B = B.to(device=a.device, dtype=torch.float32)
            if a.comfy_qkv and k.endswith("qkv_proj.weight"):
                B = B.view(3, H, Dh, -1).permute(1, 0, 2, 3).reshape(3 * H * Dh, -1)
            W = sd[k].to(device=a.device, dtype=torch.float32)
            assert W.shape == (B.shape[0], A.shape[1]), (k, W.shape, A.shape, B.shape)
            sd[k] = (W + s * (B @ A)).to(sd[k].dtype).cpu()
        hit += 1
    manifest["sources"][os.path.realpath(src)] = digest(src)
    save_file(sd, dst + ".tmp", metadata={"format": "pt"})
    os.replace(dst + ".tmp", dst)
    with safe_open(dst, "pt") as out, safe_open(src, "pt") as original:
        for k in out.keys():
            if a.hybrid_ref:
                if k in selected:
                    with safe_open(ref_keys[k], "pt") as ref:
                        assert torch.equal(out.get_tensor(k), ref.get_tensor(k)), k
                else:
                    assert torch.equal(out.get_tensor(k), original.get_tensor(k)), k
            elif k not in selected:
                assert torch.equal(out.get_tensor(k), original.get_tensor(k)), k
    manifest["outputs"][os.path.basename(dst)] = digest(dst)
    done += hit
    print(f"[fuse] {os.path.basename(src)}: {hit} replaced/fused and verified", flush=True)
    del sd
assert done == len(selected), (done, len(selected))
with open(f"{a.out}/merge_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"[fuse] done: {done} modules -> {a.out}", flush=True)
