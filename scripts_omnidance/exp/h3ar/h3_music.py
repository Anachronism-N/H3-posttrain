"""M1a (notes/20): inject MERT-v1-330M music features into MiniMax-H3 as an extra, zero-initialised condition.

Features: per-clip MERT cache [1, 25, 1024, 81] (16 fps grid, same as the Wan2.2 S4 pipeline).
Injector (dit.od_music): softmax layer weights over the 25 MERT layers -> LayerNorm -> MLP(1024->1024->hidden),
last linear zero-initialised, so at step 0 the model is exactly the base H3. The output is resampled to the 40 Hz audio-latent
grid and ADDED to the embeddings of the target-audio tokens (both stereo channels; rows are channel-major), right after
DiT._embed. Video tokens see it through joint attention (block-causal rules unchanged: still music-time-aligned).

Train: env OD_H3_MUSIC=1 + OD_H3_MUSIC_MAP=<json {cache .pth realpath: mert .pt}>; train.py calls setup(dit) before
DeepSpeed init; loss.py calls set_from_inputs(inputs) (dataset adds inputs['od_cache_path']).
Sample: load_injector(dit, lora_file) loads od_music.* keys; set_feat(mert_tensor, n_ref_audio) before each generation.
"""
import json, os
import torch
import torch.nn as nn
import torch.nn.functional as F

import diffsynth.models.minimax_h3_dit as HD

STATE = {"feat": None, "n_ref_audio": 0, "map": None}
MERT_FPS, AUDIO_HZ = 16.0, 40.0


class MusicInjector(nn.Module):
    def __init__(self, hidden, layers=25, dim=1024, mid=1024):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(layers))
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mid)
        self.fc2 = nn.Linear(mid, hidden)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, feat, n_t):
        # feat (25, 1024, T16) -> (n_t, hidden) on the 40 Hz audio-latent grid (linear interpolation in time)
        w = torch.softmax(self.layer_w.float(), 0).to(feat.dtype)
        x = torch.einsum("l,ldt->td", w, feat)                        # (T16, 1024)
        pos = torch.arange(n_t, device=x.device, dtype=torch.float32) * (MERT_FPS / AUDIO_HZ)
        pos = pos.clamp(max=x.shape[0] - 1)
        lo = pos.floor().long(); hi = (lo + 1).clamp(max=x.shape[0] - 1); a = (pos - lo).unsqueeze(1).to(x.dtype)
        x = x[lo] * (1 - a) + x[hi] * a
        x = x.to(self.fc1.weight.dtype)
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


def _lerp_time(x, n_t, src_hz):
    """x (T, d) on a src_hz grid -> (n_t, d) on the 40 Hz audio-latent grid"""
    pos = (torch.arange(n_t, device=x.device, dtype=torch.float32) * (src_hz / AUDIO_HZ)).clamp(max=x.shape[0] - 1)
    lo = pos.floor().long(); hi = (lo + 1).clamp(max=x.shape[0] - 1); a = (pos - lo).unsqueeze(1).to(x.dtype)
    return x[lo] * (1 - a) + x[hi] * a


class MusicMoE(nn.Module):
    """M2: per-time-step soft router over 5 music experts, fused -> zero-init MLP -> hidden.
    experts: mert_lo (MERT layers 0-12, weighted), mert_hi (13-24), beat (onset/beat impulse/phase/rms/tempo, 40 Hz),
             vae (H3's own audio-VAE latent of the music, 40 Hz), clap (global CLAP embedding, broadcast)."""
    NAMES = ("mert_lo", "mert_hi", "beat", "vae", "clap")

    def __init__(self, hidden, d=1024, mid=1024):
        super().__init__()
        self.w_lo = nn.Parameter(torch.zeros(13)); self.w_hi = nn.Parameter(torch.zeros(12))
        self.p_lo = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d))
        self.p_hi = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, d))
        self.p_beat = nn.Sequential(nn.Linear(6, 256), nn.GELU(), nn.Linear(256, d))
        self.p_vae = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, d))
        self.p_clap = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d))
        self.norm = nn.LayerNorm(d)
        self.router = nn.Linear(5 * d, 5)
        nn.init.zeros_(self.router.weight); nn.init.zeros_(self.router.bias)
        self.fc1 = nn.Linear(d, mid); self.fc2 = nn.Linear(mid, hidden)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, feat, n_t):
        dt = self.fc1.weight.dtype
        m = feat["mert"].to(dt)                                                    # (25, 1024, 81)
        lo = torch.einsum("l,ldt->td", torch.softmax(self.w_lo.float(), 0).to(dt), m[:13])
        hi = torch.einsum("l,ldt->td", torch.softmax(self.w_hi.float(), 0).to(dt), m[13:])
        xs = [self.p_lo(_lerp_time(lo, n_t, MERT_FPS)), self.p_hi(_lerp_time(hi, n_t, MERT_FPS)),
              self.p_beat(_lerp_time(feat["beat"].to(dt), n_t, AUDIO_HZ)), self.p_vae(_lerp_time(feat["vae"].to(dt), n_t, AUDIO_HZ)),
              self.p_clap(feat["clap"].to(dt).reshape(1, -1)).expand(n_t, -1)]  # 2-D input: ZeRO-3 linear backward needs >=2-D
        X = torch.stack([self.norm(x) for x in xs], 1)                            # (n_t, 5, d)
        logits = self.router(X.reshape(n_t, -1)).float()
        if self.training and float(os.environ.get("OD_H3_MOE_DROP", "0.1")) > 0:  # expert dropout (keeps >= 1 expert)
            drop = torch.rand(5, device=logits.device) < float(os.environ.get("OD_H3_MOE_DROP", "0.1"))
            if bool(drop.all()):
                drop[torch.randint(0, 5, (1,))] = False
            logits = logits.masked_fill(drop, float("-inf"))
        g = torch.softmax(logits, -1)                                              # (n_t, 5)
        gm = g.mean(0)
        STATE["aux"] = 5.0 * (gm ** 2).sum()                                       # load balance: 1 when uniform
        STATE["gate"] = gm.detach()
        y = (g.to(dt).unsqueeze(-1) * X).sum(1)
        return self.fc2(F.gelu(self.fc1(y)))


_orig_embed = HD.MiniMaxH3DiT._embed


def _embed(self, **kw):
    emb, t_emb = _orig_embed(self, **kw)
    feat = STATE["feat"]
    if feat is not None and hasattr(self, "od_music"):
        tgt = kw["audio_pos"][STATE["n_ref_audio"]:]
        n_t = tgt.shape[0] // 2                                          # stereo, channel-major rows
        w0 = self.od_music.fc1.weight
        if getattr(w0, "ds_id", None) is None and w0.device != emb.device:  # inference with CPU offload (not under ZeRO-3)
            self.od_music.to(emb.device)
        if isinstance(feat, dict):
            e = self.od_music({k: v.to(emb.device) for k, v in feat.items()}, n_t).to(emb.dtype)
        else:
            e = self.od_music(feat.to(emb.device, self.od_music.fc1.weight.dtype), n_t).to(emb.dtype)
        emb = emb.index_add(0, tgt, torch.cat([e, e], 0))
    return emb, t_emb


def install():
    HD.MiniMaxH3DiT._embed = _embed


def setup(dit):
    """train: attach a trainable zero-init injector before accelerator.prepare"""
    install()
    p = next(dit.parameters())
    cls = MusicMoE if os.environ.get("OD_H3_MUSIC_MODE", "mert") == "moe" else MusicInjector
    dit.od_music = cls(dit.hidden_size).to(device=p.device, dtype=p.dtype)
    for q in dit.od_music.parameters():
        q.requires_grad_(True)
    print(f"[OD H3-MUSIC] {type(dit.od_music).__name__} attached, {sum(q.numel() for q in dit.od_music.parameters()) / 1e6:.2f}M params", flush=True)


def load_mert(path):
    t = torch.load(path, map_location="cpu", weights_only=True)
    return t[0] if t.dim() == 4 else t                                   # (25, 1024, 81)


def set_feat(feat, n_ref_audio):
    STATE["feat"] = feat; STATE["n_ref_audio"] = int(n_ref_audio)


MERT_DIRS = ["/apdcephfs_gy4/share_302533218/cedricnie/diff/data/omnidance/audio_cache/mert_features",
             "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/G2/mert_val"]
M2_DIR = "/apdcephfs_gy4/share_302533218/cedricnie/diff/data/omnidance/audio_cache/m2_feats"


def mert_path(sid):
    return next(f"{d}/{sid}.pt" for d in MERT_DIRS if os.path.exists(f"{d}/{sid}.pt"))


def load_moe_feats(sid, audio_latents):
    """audio_latents: H3 audio-VAE latents of the music (C, 32, T40) -> channel mean (T40, 32)"""
    import numpy as np
    return {"mert": load_mert(mert_path(sid)), "beat": torch.from_numpy(np.load(f"{M2_DIR}/beat/{sid}.npy").astype("float32")),
            "clap": torch.from_numpy(np.load(f"{M2_DIR}/clap/{sid}.npy").astype("float32")), "vae": audio_latents.float().mean(0).t().cpu()}


def aux_loss():
    a = STATE.get("aux")
    STATE["aux"] = None
    return 0.0 if a is None else float(os.environ.get("OD_H3_MOE_BAL", "0.01")) * a


def set_from_inputs(inputs):
    if STATE["map"] is None:
        STATE["map"] = json.load(open(os.environ["OD_H3_MUSIC_MAP"]))
    key = os.path.realpath(inputs["od_cache_path"])
    ra = inputs.get("ref_audio_anchor"); n_ra = 0 if ra is None else ra.shape[0]
    if os.environ.get("OD_H3_MUSIC_MODE", "mert") == "moe":
        STATE["n"] = STATE.get("n", 0) + 1
        if STATE["n"] % 20 == 1 and STATE.get("gate") is not None:
            print("[OD H3-MOE] gate " + " ".join(f"{k}={v:.3f}" for k, v in zip(MusicMoE.NAMES, STATE["gate"].tolist())), flush=True)
        set_feat(load_moe_feats(STATE["map"][key], inputs["audio_input_latents"]), n_ra)
        return
    if os.environ.get("OD_H3_MUSIC_DROP") and torch.rand(()) < float(os.environ["OD_H3_MUSIC_DROP"]):
        feat = torch.zeros(25, 1024, 81)                                 # condition dropout (keeps every rank's graph identical)
    else:
        feat = load_mert(STATE["map"][key])
    ra = inputs.get("ref_audio_anchor")
    set_feat(feat.to(inputs["input_latents"].device), 0 if ra is None else ra.shape[0])


def load_injector(dit, lora_file):
    """sample: split od_music.* keys out of a trained LoRA file; returns the remaining (LoRA) state dict"""
    from safetensors.torch import load_file
    sd = load_file(lora_file)
    mus = {k[len("od_music."):]: v for k, v in sd.items() if k.startswith("od_music.")}
    rest = {k: v for k, v in sd.items() if not k.startswith("od_music.")}
    if mus:
        install()
        p = next(dit.parameters())
        cls = MusicMoE if "router.weight" in mus else MusicInjector
        dit.od_music = cls(dit.hidden_size).to(device=p.device, dtype=torch.bfloat16).eval()
        dit.od_music.load_state_dict({k: v.to(torch.bfloat16) for k, v in mus.items()})
        print(f"[OD H3-MUSIC] {cls.__name__} loaded ({len(mus)} tensors)", flush=True)
    return rest, bool(mus)
