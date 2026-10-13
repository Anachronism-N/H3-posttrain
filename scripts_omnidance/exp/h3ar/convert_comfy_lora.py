"""convert_comfy_lora.py <in> <out> [--merge_with <our lora>]: ComfyUI-layout H3 LoRA (keys diffusion_model.*, qkv rows [role][head][dim])
-> DiffSynth/PEFT layout used by our LoRAs (keys blocks.*.lora_{A,B}.default.weight, qkv rows [head][role][dim]); alpha folded into B.
--merge_with: concatenate ranks with one of our LoRAs (=sum of the two deltas), so the sampler can load both as one file."""
import sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
H, Dh = 56, 128
src, dst = sys.argv[1], sys.argv[2]
with safe_open(src, "pt") as f:
    L = {k: f.get_tensor(k) for k in f.keys()}
out = {}
for k, A in L.items():
    if ".lora_A" not in k:
        continue
    base = k.split(".lora_A")[0]
    B = L[k.replace("lora_A", "lora_B")].float()
    al = L.get(base + ".alpha"); s = float(al) / A.shape[0] if al is not None else 1.0
    if base.endswith("qkv_proj"):
        B = B.view(3, H, Dh, -1).permute(1, 0, 2, 3).reshape(3 * H * Dh, -1)
    nb = base.removeprefix("diffusion_model.")
    out[nb + ".lora_A.default.weight"] = A.to(torch.bfloat16).contiguous()
    out[nb + ".lora_B.default.weight"] = (B * s).to(torch.bfloat16).contiguous()
if "--merge_with" in sys.argv:
    with safe_open(sys.argv[sys.argv.index("--merge_with") + 1], "pt") as f:
        O = {k: f.get_tensor(k) for k in f.keys()}
    for k in list(O):
        if ".lora_A" in k and k in out:
            kb = k.replace("lora_A", "lora_B")
            out[k] = torch.cat([out[k], O[k].to(torch.bfloat16)], 0)
            out[kb] = torch.cat([out[kb], O[kb].to(torch.bfloat16)], 1)
        elif k not in out:
            out[k] = O[k]
save_file(out, dst); print("saved", dst, len(out))
