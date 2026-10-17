"""H3-AR sampling (block-causal, multi-step per block) on OD-Val-47 with music as clean target audio + reference audio.
Mode `ar`: block by block; context frames = previously generated latents (mask 0 -> clean), current block denoised
`--steps` steps, future frames ignored (invisible under block-causal attention). Mode `bi`: stock bidirectional.
Output layout <out>/<cond>/<sid>_<audiostem>.mp4 (video + original music), like the other samplers.
python sample_h3_ar.py --out <dir> [--lora <dit lora safetensors>] --mode ar --conds joint,aswap --shard i --nshard n
"""
import argparse, csv, glob, os, sys, time
import librosa, torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "h3"))
p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--lora", default="")
p.add_argument("--mode", default="ar", choices=["ar", "bi"])
p.add_argument("--conds", default="joint,aswap")
p.add_argument("--csv_dir", default="/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/E2")
p.add_argument("--prompt_mode", default="motion", choices=["orig", "motion", "music", "empty", "generic"], help="motion: E10 motion caption only; music: motion caption + BPM/beat times written into the prompt (idea M1b)")
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshard", type=int, default=1)
p.add_argument("--rows", type=int, default=3)
p.add_argument("--steps", type=int, default=20)
p.add_argument("--block", type=int, default=6)
p.add_argument("--lookahead", type=float, default=8)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--cm_steps", type=int, default=0, help=">0: multistep consistency sampler (CF++ stage-2 student), N steps per block, sigmas of an N-step shift-5 schedule")
p.add_argument("--shift", type=float, default=5.0, help="video flow shift (official DiffSynth/ComfyUI default 12)")
p.add_argument("--audio_shift", type=float, default=5.0, help="audio flow shift (official default 3)")
p.add_argument("--cfg", type=float, default=1.0, help=">1: real CFG with negative prompt ' ' (diagnose lost CFG-distillation after fine-tuning)")
p.add_argument("--height", type=int, default=832); p.add_argument("--width", type=int, default=480)
p.add_argument("--audio_override", default="", help="replace the music of every case by this wav (speech/noise/silence probe)")
p.add_argument("--native_audio_ckpt", default="", help="Native audio-side CLIP checkpoint; install native encoding weights, retaining the original audio latent interface")
p.add_argument("--music_encoder_ckpt", default="", help="Trained dedicated music encoder plus adapter replaces reference and target audio encoding; not a residual/text hook")
p.add_argument("--audio_tail_mode", default="legacy", choices=["legacy", "repeat_clean"], help="Controlled audio-padding probe: legacy regenerates padded target audio; repeat_clean keeps the repeated clean latent fixed at AdaLN time=1")
p.add_argument("--body_json", default="", help="music_caption.py output; replaces the per-video description (template kept)")
p.add_argument("--body_key", default="timed", help="timed | calm | energetic")
p.add_argument("--soft_text", default="", help="10-15 G3: genre_clip text_targets.pt; description = K placeholder tokens whose "
               "input embeddings are replaced by sum_c p_c T_c (soft prompt written into H3's text-encoder input layer)")
p.add_argument("--soft_p", default="", help='json {genre: prob}, e.g. {"rock":0.5,"zen":0.5}')
p.add_argument("--soft_tokens", default="", help="10-16 AE: genre_ae.py tokens.pt {key: [K,5120]}; key = audio file stem "
               "(of --audio_override or the row's music); overrides --soft_p per clip")
p.add_argument("--prompt_prefix", default="", help="e.g. a community LoRA trigger word (fal realism: r34l1sm)")
p.add_argument("--w", default="/dev/shm/h3/Ref2VA")
p.add_argument("--music_cfg", type=float, default=0.0, help=">0: music guidance v + w*(v(music) - v(music rolled by 50%%)) (time-misaligned negative, same text/ref)")
p.add_argument("--music_cfg_min_sig", type=float, default=0.0, help="apply music guidance only where sigma >= this")
p.add_argument("--lora_steps", default="", help="a:b -> LoRA active only on denoise step indices [a, b) (bi mode); diagnoses which noise range the fine-tune damages")
a = p.parse_args()
if a.prompt_mode == "empty" and (a.body_json or a.soft_text or a.soft_tokens or a.prompt_prefix):
    p.error("Empty prompt must not combine with captions, soft tokens or a prompt prefix")

csv.field_size_limit(10 ** 9)
E = "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp"
cap = {r["sample_id"]: r["motion_caption"] for r in csv.DictReader(open(f"{E}/E10/val47_motion_captions.csv"))}
jobs = []
for c in a.conds.split(","):
    rows = list(csv.DictReader(open(f"{a.csv_dir}/full_{c}.csv")))
    jobs += [(c, r) for r in (rows[:a.rows] if a.rows else rows)]
jobs = jobs[a.shard::a.nshard]
todo = [(c, r, f"{a.out}/{c}/{r['sample_id']}_{os.path.basename(r['audio_path'])[:-4]}.mp4") for c, r in jobs]
todo = [t for t in todo if not os.path.exists(t[2])]
print(f"[h3ar] {len(todo)} todo", flush=True)
if not todo:
    sys.exit(0)

from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig  # noqa: E402
from diffsynth.utils.data.audio_video import write_video_audio  # noqa: E402
import h3_causal as HC  # noqa: E402
src = open(os.path.join(HERE, "..", "h3", "e3_h3_zeroshot.py")).read()
ns = {}; exec(src[src.index("def h3_prompt"):src.index("\n\n\n", src.index("def h3_prompt"))], ns)
h3_prompt = ns["h3_prompt"]

vram = dict(offload_dtype=torch.bfloat16, offload_device="cpu", onload_dtype=torch.bfloat16, onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device="cuda", computation_dtype=torch.bfloat16, computation_device="cuda")
W = a.w
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        ModelConfig(path=sorted(glob.glob(f"{W}/text_encoder/model*.safetensors")), **vram),
        ModelConfig(path=sorted(glob.glob(f"{W}/transformer/model*.safetensors")), **vram),
        ModelConfig(path=f"{W}/video_vae/source/model.safetensors", **vram),
        ModelConfig(path=f"{W}/audio_vae/model.safetensors", **vram),
    ],
    processor_config=ModelConfig(path=f"{W}/processor/"),
    **({"vram_limit": torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5} if a.height > 832 else {}),  # official-res (as e3/oos scripts)
)
if a.music_encoder_ckpt:
    assert not a.native_audio_ckpt and not a.lora and not a.soft_text, 'Dedicated music replacement must not combine with native checkpoints or old injection hooks'
    sys.path.insert(0, os.path.join(HERE, '..', 'ms'))
    from music_encoder_adapter import MusicEncodingProxy
    pipe.audio_vae.encode_audio = MusicEncodingProxy(a.music_encoder_ckpt)
    print('[h3ar] dedicated music encoder + adapter replaces reference AND retake audio encoding', flush=True)
if a.native_audio_ckpt:
    native = torch.load(a.native_audio_ckpt, map_location='cpu', weights_only=False)
    state = (native['native_audio_encoder'] if 'native_audio_encoder' in native else
             {'mean_proj.' + key: value for key, value in native['native_mean_proj'].items()})
    assert state and all(torch.isfinite(value).all() for value in state.values()), 'Non-finite native audio weights'
    # VRAM management adds module segments around wrapped layers; CPU-offloaded
    # parameters remain the source of future onload/preparing copies.
    from diffsynth.core.vram.layers import AutoWrappedModule
    assert not any(module.disk_offload for module in pipe.audio_vae.modules() if isinstance(module, AutoWrappedModule)), 'Native checkpoint loading expects CPU, not disk, offload'
    current = pipe.audio_vae.state_dict()
    keys = {key.replace('.module.', '.'): key for key in current}
    assert len(keys) == len(current), 'Ambiguous wrapped audio parameter names'
    mapped = {}
    for key, value in state.items():
        assert key.startswith(('encoder.', 'pre_block.', 'mean_proj.')) and key in keys, key
        target = keys[key]
        assert current[target].shape == value.shape, key
        mapped[target] = value
    result = pipe.audio_vae.load_state_dict(mapped, strict=False)
    assert not result.unexpected_keys, result.unexpected_keys
    installed = pipe.audio_vae.state_dict()
    assert all(torch.equal(installed[key].cpu(), value.to(installed[key].dtype)) for key, value in mapped.items()), 'Native checkpoint installation mismatch'
    print(f'[h3ar] native audio encoder loaded: {len(mapped)} tensors; no caption retrieval or text-token hook', flush=True)
MUSIC = False
if a.lora:
    import h3_music as MU  # M1a: trained MERT injector (od_music.* keys) travels inside the LoRA file
    _rest, MUSIC = MU.load_injector(pipe.dit, a.lora)
    LORA_SD = {k: v.to(device=pipe.device, dtype=pipe.torch_dtype) for k, v in _rest.items()}
    pipe.load_lora(pipe.dit, state_dict=LORA_SD)
LS = [int(x) for x in a.lora_steps.split(":")] if a.lora_steps else None
LORA_ON = [bool(a.lora)]


def set_lora(on):  # hot-loaded LoRA (vram management on): toggle per denoise step for --lora_steps
    if on != LORA_ON[0]:
        pipe.clear_lora(verbose=0) if not on else pipe.load_lora(pipe.dit, state_dict=LORA_SD, verbose=0)
        LORA_ON[0] = on
SOFT = None
if a.soft_text:
    import json as _js
    _st = torch.load(a.soft_text, weights_only=False); _pp = _js.loads(a.soft_p) if a.soft_p else {}
    _mix = (sum(float(v) * _st["T"][_st["genres"].index(g)].float() for g, v in _pp.items()) / sum(float(v) for v in _pp.values())
            if _pp else None)
    SOFT_TOK = torch.load(a.soft_tokens, weights_only=False) if a.soft_tokens else None
    _ph = pipe.tokenizer(" move" * _st["K"], add_special_tokens=False)["input_ids"]
    assert len(_ph) == _st["K"] and len(set(_ph)) == 1, "placeholder must be K identical tokens"
    SOFT = {"body": "move" + " move" * (_st["K"] - 1), "id": _ph[0], "K": _st["K"], "mix": _mix, "hits": 0}

    def _soft_hook(mod, inp, out):
        ids = inp[0].reshape(-1); hit = (ids == SOFT["id"]).nonzero().reshape(-1)
        if hit.numel() >= SOFT["K"]:
            s = int(hit[0]); assert bool((ids[s:s + SOFT["K"]] == SOFT["id"]).all()), "placeholder not contiguous"
            out = out.clone(); o = out.reshape(-1, out.shape[-1])
            o[s:s + SOFT["K"]] = SOFT["mix"].to(o.device, o.dtype); SOFT["hits"] += 1
            return o.reshape(out.shape)
        return out
    pipe.text_encoder.model.language_model.embed_tokens.register_forward_hook(_soft_hook)
    print(f"[h3ar] soft prompt {_pp} K={SOFT['K']}", flush=True)
MERT_DIRS = ["/apdcephfs_gy4/share_302533218/cedricnie/diff/data/omnidance/audio_cache/mert_features",
             "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/G2/mert_val"]
if a.mode == "ar":
    HC.install()


@torch.no_grad()
def generate(prompt, img, wav, sr):
    pipe.scheduler.set_timesteps(a.steps, shift=a.shift)
    pipe.scheduler_audio.set_timesteps(a.steps, shift=a.audio_shift)
    inputs_posi = {"prompt": prompt}; inputs_nega = {"negative_prompt": " "}
    sh = {"cfg_scale": a.cfg, "height": a.height, "width": a.width, "num_frames": 124, "seed": a.seed, "rand_device": "cpu",
          "tiled": True, "tile_size": 256, "tile_overlap": 64, "use_gradient_checkpointing": False, "use_gradient_checkpointing_offload": False,
          "keyframes": None, "keyframe_indices": None,
          "references": [{"type": "image", "image": img}, {"type": "audio", "audio": wav, "sample_rate": sr}],
          "ref_image_short_edge": 2048, "ref_video_short_edge": 768, "ref_video_max_pixels": 768 * 1344,
          "control_video": None, "control_scale": 1.0, "retake_video": None, "frame_regions_to_retake": None,
          "retake_audio": (wav, sr), "seconds_regions_to_retake": [],
          "imgvid_cond_noise_aug": pipe.imgvid_cond_noise_aug, "audio_cond_noise_aug": pipe.audio_cond_noise_aug, "text_embedding": None}
    for unit in pipe.units:
        sh, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, sh, inputs_posi, inputs_nega)
    if a.audio_tail_mode == 'repeat_clean':
        assert sh.get('input_latents_audio') is not None and sh.get('denoise_mask_audio') is not None
        # Retake unit already repeats the last clean latent for padding. Here
        # only its regenerate mask is changed; waveform/ref/prompt stay identical.
        mask = sh['denoise_mask_audio'].clone()
        clean_rows = mask[0, 0] == 0
        assert clean_rows.any() and mask.shape[-1] == sh['audio_latents'].shape[-1]
        padded = (~clean_rows).nonzero().reshape(-1)
        assert not padded.numel() or torch.equal(padded, torch.arange(int(padded[0]), mask.shape[-1], device=mask.device)), 'Only contiguous tail padding may be changed'
        mask.zero_()
        sh['denoise_mask_audio'] = mask
        print(f'[h3ar] repeat_clean audio tail: {padded.numel()} rows fixed at effective time=1', flush=True)
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {n: getattr(pipe, n) for n in pipe.in_iteration_models}
    lat0 = sh["video_latents"].clone()
    T = lat0.shape[2]
    if MUSIC:
        stem = os.path.basename(CUR_AUDIO[0])[:-4]
        if isinstance(pipe.dit.od_music, MU.MusicMoE):  # M2: MERT + beat + CLAP + H3 audio-VAE latent of the music
            MU.set_feat(MU.load_moe_feats(stem, sh["input_latents_audio"]), sh["ref_audio_anchor"].shape[0])
        else:
            mp = next(f"{d}/{stem}.pt" for d in MERT_DIRS if os.path.exists(f"{d}/{stem}.pt"))
            MU.set_feat(MU.load_mert(mp).to(pipe.device), sh["ref_audio_anchor"].shape[0])
    PL = {"p": None, "n": None}

    def pred(tv, ta):
        HC.STATE["plan"] = PL["p"]
        v, _ = pipe.model_fn(**models, **sh, **inputs_posi, timestep_video=tv, timestep_audio=ta)
        vp = v  # notes/24: music difference uses the un-guided positive branch (vm=vp -> no change for any w)
        if a.cfg > 1:
            HC.STATE["plan"] = PL["n"]
            vn, _ = pipe.model_fn(**models, **sh, **inputs_nega, timestep_video=tv, timestep_audio=ta)
            HC.STATE["plan"] = PL["p"]
            v = vn + a.cfg * (v - vn)
        if a.music_cfg > 0 and float(tv.reshape(-1)[0]) / 1000.0 >= a.music_cfg_min_sig:
            shn = dict(sh)
            for k in ("audio_latents", "input_latents_audio"):
                if shn.get(k) is not None:
                    shn[k] = torch.roll(shn[k], shn[k].shape[-1] // 2, dims=-1)
            vm, _ = pipe.model_fn(**models, **shn, **inputs_posi, timestep_video=tv, timestep_audio=ta)
            v = v + a.music_cfg * (vp - vm)
        return v

    def denoise(region_mask, context):
        sh["video_latents"] = lat0.clone() if context is None else context * (1 - region_mask) + lat0 * region_mask
        if context is not None:
            sh["input_latents_video"] = context; sh["denoise_mask_video"] = region_mask
        if a.cm_steps > 0:  # x0 = x - s v ; re-noise to the next sigma with fresh noise (only the noisy region)
            g = torch.Generator(device="cpu").manual_seed(a.seed + 1)
            for k, s in enumerate(CM_SIG):
                tv = torch.tensor([s * 1000.0], dtype=torch.float32, device=pipe.device)
                v = pred(tv, tv)
                x0 = sh["video_latents"] - s * v
                if k + 1 < len(CM_SIG):
                    nz = torch.randn(x0.shape, generator=g).to(x0)
                    x0 = (1 - CM_SIG[k + 1]) * x0 + CM_SIG[k + 1] * nz
                sh["video_latents"] = x0 * region_mask + sh["video_latents"] * (1 - region_mask)
            return sh["video_latents"]
        for k, tv in enumerate(pipe.scheduler.timesteps):
            tv = tv.unsqueeze(0).to(dtype=torch.float32, device=pipe.device)
            ta = pipe.scheduler_audio.timesteps[k].unsqueeze(0).to(dtype=torch.float32, device=pipe.device)
            if LS is not None:
                set_lora(LS[0] <= k < LS[1])
            v = pred(tv, ta)
            sh["video_latents"] = pipe.step(pipe.scheduler, sh["video_latents"], k, noise_pred=v,
                                            inpaint_mask=sh.get("denoise_mask_video"), input_latents=sh.get("input_latents_video"))
        return sh["video_latents"]

    if a.mode == "bi":
        out = denoise(torch.ones_like(lat0), None)
    else:
        n_rv = sh["ref_visual_anchor"].shape[0]; n_ra = sh["ref_audio_anchor"].shape[0]
        plan = HC.make_plan(inputs_posi.get("packed", sh.get("packed")), n_rv, n_ra, T, block=a.block, lookahead=a.lookahead)
        HC.STATE["plan"] = plan; PL["p"] = plan
        if a.cfg > 1:
            PL["n"] = HC.make_plan(inputs_nega["packed"], n_rv, n_ra, T, block=a.block, lookahead=a.lookahead)
        ctx = torch.zeros_like(lat0)
        for f0, f1 in plan["bounds"]:
            m = torch.ones_like(lat0); m[:, :, :f0] = 0
            res = denoise(m, ctx)
            ctx[:, :, f0:f1] = res[:, :, f0:f1]
        out = ctx
        HC.STATE["plan"] = None
    pipe.load_models_to_device(["video_vae"])
    frames = pipe.video_vae.decode_video(out, dtype=pipe.torch_dtype, tiled=True, tile_size=256, tile_overlap=64)
    return pipe.vae_output_to_video(frames, min_value=0, max_value=1)


CM_SIG = []
if a.cm_steps > 0:
    import copy as _copy
    _sc = _copy.deepcopy(pipe.scheduler); _sc.set_timesteps(a.cm_steps, shift=a.shift)
    CM_SIG = [float(x) for x in _sc.sigmas[:a.cm_steps]]
    print(f"[h3ar] consistency sampler sigmas {CM_SIG}", flush=True)
dur = 124 / 24
CUR_AUDIO = [None]
import json as _json
BEATS = _json.load(open(f"{E}/beats.json")) if a.prompt_mode == "music" else {}
for c, r, dst in todo:
    t0 = time.time()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img = Image.open(r["image_path"]).convert("RGB")
    sr = pipe.audio_vae.sample_rate
    y, _ = librosa.load(a.audio_override or r["audio_path"], sr=sr, mono=False, duration=dur)
    wav = torch.from_numpy(y if y.ndim == 2 else y[None]).float()
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    body = cap.get(r.get("swap_src") or r["sample_id"], r["prompt"]) if a.prompt_mode in ("motion", "music") else r["prompt"]
    if a.prompt_mode == "music":
        bt = [t for t in BEATS.get(r["audio_path"], []) if t < dur]
        if len(bt) >= 2:
            ibi = sorted(bt[i + 1] - bt[i] for i in range(len(bt) - 1))[len(bt) // 2 - 1]
            body += (f" The music has a steady tempo of about {60.0 / ibi:.0f} BPM; its beats fall at "
                     + ", ".join(f"{t:.1f}" for t in bt) + " seconds. <Subject 1> places a clear accent "
                     "(a sharp arm hit, step or body bounce) exactly on each of these beats.")
    CUR_AUDIO[0] = r["audio_path"]
    if a.body_json:  # 10-15: fixed probe body ("*") or music-derived body keyed by audio path, inside the H3 template
        BJ = _json.load(open(a.body_json))[a.body_key]
        body = BJ.get(r["sample_id"], BJ.get(r["audio_path"], BJ.get("*")))
        assert body, r["audio_path"]
    if SOFT is not None:  # K placeholder tokens (template supplies the leading space); their input embeddings become the soft caption
        body = SOFT["body"]
        if SOFT_TOK is not None:  # per-clip encoder tokens, keyed by the music file stem actually used
            _k = os.path.basename(a.audio_override or r["audio_path"])[:-4]
            SOFT["mix"] = SOFT_TOK[_k].float(); assert SOFT["mix"].shape[0] == SOFT["K"], _k
        assert SOFT["mix"] is not None, "need --soft_p or --soft_tokens"
    if a.prompt_prefix:
        body = a.prompt_prefix + " " + body
    # empty: text prompt " " (same as the CFG negative) -> isolates what image + audio alone drive
    # generic: H3 template (reference/audio bindings) + one generic dance sentence, no per-video description
    prompt = " " if a.prompt_mode == "empty" else h3_prompt(body, generic=(c == "audio_only" or a.prompt_mode == "generic"))
    print('[h3ar] input_conditions ' + _json.dumps(dict(prompt_mode=a.prompt_mode, semantic_text=prompt,
          image_path=r["image_path"], audio_path=a.audio_override or r["audio_path"], seed=a.seed,
          music_encoder_ckpt=a.music_encoder_ckpt or None)), flush=True)
    video = generate(prompt, img, wav, sr)
    write_video_audio(video=video, audio=wav, output_path=dst, fps=24, audio_sample_rate=sr)
    if SOFT is not None:
        assert SOFT["hits"] > 0, "soft prompt hook never fired"
    print(f"[h3ar] {c} {r['sample_id']} {time.time() - t0:.0f}s" + (f" soft_hits={SOFT['hits']}" if SOFT else ""), flush=True)
print("[h3ar] done", flush=True)
