"""E3: MiniMax-H3 Ref2VA zero-shot on OD-Val-47 (E2 protocol subset), one GPU per shard.
Conditions: joint (ref image + music reuse + caption), tswap (swapped caption), aswap (swapped music),
audio_only (generic caption). Output <out>/<cond>/<sid>_<audiostem>.mp4 (video + generated audio track).
Run with /opt/venv_h3 and PYTHONPATH=<DiffSynth upstream>.
"""
import argparse, csv, os, sys, time
import torch
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--conds", default="joint,tswap,aswap,audio_only")
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshard", type=int, default=1)
p.add_argument("--height", type=int, default=832)
p.add_argument("--width", type=int, default=480)
p.add_argument("--num_frames", type=int, default=124)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--w", default="/dev/shm/h3/Ref2VA")
p.add_argument("--music_mode", default="ref", choices=["ref", "retake"], help="retake: also feed music as clean time-aligned target audio (no audio regeneration)")
args = p.parse_args()

E2 = "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/E2"
csv.field_size_limit(10 ** 9)
jobs = []
for c in args.conds.split(","):
    src = "joint" if c == "audio_only" else c
    for r in csv.DictReader(open(f"{E2}/full_{src}.csv")):
        jobs.append((c, r))
jobs = jobs[args.shard::args.nshard]
todo = [(c, r, f"{args.out}/{c}/{r['sample_id']}_{os.path.basename(r['audio_path'])[:-4]}.mp4") for c, r in jobs]
todo = [t for t in todo if not os.path.exists(t[2])]
print(f"[h3] {len(todo)} todo", flush=True)
if not todo:
    sys.exit(0)

from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig  # noqa: E402
from diffsynth.utils.data.audio_video import write_video_audio  # noqa: E402
import librosa  # noqa: E402
import glob  # noqa: E402

vram = dict(offload_dtype=torch.bfloat16, offload_device="cpu", onload_dtype=torch.bfloat16, onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device="cuda", computation_dtype=torch.bfloat16, computation_device="cuda")
W = args.w
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        ModelConfig(path=sorted(glob.glob(f"{W}/text_encoder/model*.safetensors")), **vram),
        ModelConfig(path=sorted(glob.glob(f"{W}/transformer/model*.safetensors")), **vram),
        ModelConfig(path=f"{W}/video_vae/source/model.safetensors", **vram),
        ModelConfig(path=f"{W}/audio_vae/model.safetensors", **vram),
    ],
    processor_config=ModelConfig(path=f"{W}/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)


def h3_prompt(caption, generic=False):
    body = ("The dancer performs a freestyle dance to the music, moving in sync with its beat and rhythm."
            if generic else caption)
    return (
        "subject_definitions:\n"
        "<Subject 1> is the single dancer shown in <Picture 1>, keeping the same face, hairstyle, outfit and body shape.\n"
        "<Picture 1> is the first frame of [Shot 1], showing <Subject 1> in the dance setting.\n"
        "<Audio 1> is the dance music track that is reused 1:1 as the complete soundtrack of the target video; "
        "<Subject 1>'s movements follow its beat and rhythm.\n\n"
        "summary:\n"
        "[keyframe completion + audio reuse] The target video starts from <Picture 1> and shows <Subject 1> dancing "
        "in a single continuous shot to the music of <Audio 1>, which is reused as the full soundtrack.\n\n"
        "retention_analysis:\n"
        "<Subject 1> (appears in [Shot 1]): fully_preserved - identity, outfit and environment are kept.\n"
        "<Picture 1> ([Shot 1] first frame): fully_preserved - the video begins exactly from this frame.\n"
        "<Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.\n\n"
        "detailed_description:\n"
        "The target video is in realistic photographic style.\n"
        f"[Shot 1] The shot begins from <Picture 1>. <Subject 1> dances to <Audio 1>. {body}\n\n"
        "overall_soundscape:\nThe soundscape is exactly the music of <Audio 1>, with no added speech or sound effects.\n\n"
        "non_diegetic_music:\nThe dance music from <Audio 1> is reused as the complete score."
    )


dur = args.num_frames / 24
for c, r, dst in todo:
    t0 = time.time()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img = Image.open(r["image_path"]).convert("RGB")
    sr = pipe.audio_vae.sample_rate
    y, _ = librosa.load(r["audio_path"], sr=sr, mono=False, duration=dur)
    wav = torch.from_numpy(y if y.ndim == 2 else y[None]).float()
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    prompt = h3_prompt(r["prompt"], generic=(c == "audio_only"))
    video, audio = pipe(
        prompt=prompt, height=args.height, width=args.width, num_frames=args.num_frames,
        num_inference_steps=args.steps, seed=args.seed,
        references=[{"type": "image", "image": img}, {"type": "audio", "audio": wav, "sample_rate": sr}],
        **({"retake_audio": wav, "retake_audio_sample_rate": sr, "seconds_regions_to_retake": []} if args.music_mode == "retake" else {}),
    )
    write_video_audio(video=video, audio=audio, output_path=dst, fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
    print(f"[h3] {c} {r['sample_id']} {time.time() - t0:.0f}s", flush=True)
print("[h3] done", flush=True)
