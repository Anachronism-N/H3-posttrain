"""OOS 15 s text-to-audio-video test with the original MiniMax-H3 FL2VA checkpoint (official DiffSynth pipeline defaults:
50 steps, video shift 12 / audio shift 3, cfg 1). Prompts are submitted verbatim; the submitted text is saved next to each mp4.
python oos_t2va.py --prompts oos_prompts.json --out <dir> --shard i --nshard n [--w <FL2VA dir>]
"""
import argparse, glob, json, os, time
import torch

p = argparse.ArgumentParser()
p.add_argument("--prompts", required=True)
p.add_argument("--out", required=True)
p.add_argument("--w", default="/opt/h3w/MiniMax-H3/FL2VA")
p.add_argument("--height", type=int, default=768)
p.add_argument("--width", type=int, default=1344)
p.add_argument("--num_frames", type=int, default=362)  # 17*21+5 -> 15.08 s @ 24 fps
p.add_argument("--steps", type=int, default=50)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshard", type=int, default=1)
a = p.parse_args()

cases = json.load(open(a.prompts))[a.shard::a.nshard]
todo = [c for c in cases if not os.path.exists(f"{a.out}/{c['id']}.mp4")]
print(f"[oos] {len(todo)} todo", flush=True)
if not todo:
    raise SystemExit(0)
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig  # noqa: E402
from diffsynth.utils.data.audio_video import write_video_audio  # noqa: E402

vram = dict(offload_dtype=torch.bfloat16, offload_device="cpu", onload_dtype=torch.bfloat16, onload_device="cpu",
            preparing_dtype=torch.bfloat16, preparing_device="cuda", computation_dtype=torch.bfloat16, computation_device="cuda")
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[
        ModelConfig(path=sorted(glob.glob(f"{a.w}/text_encoder/model*.safetensors")), **vram),
        ModelConfig(path=sorted(glob.glob(f"{a.w}/transformer/model*.safetensors")), **vram),
        ModelConfig(path=f"{a.w}/video_vae/source/model.safetensors", **vram),
        ModelConfig(path=f"{a.w}/audio_vae/model.safetensors", **vram),
    ],
    processor_config=ModelConfig(path=f"{a.w}/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 5,
)
os.makedirs(a.out, exist_ok=True)
for c in todo:
    t0 = time.time()
    with open(f"{a.out}/{c['id']}.prompt.txt", "w") as f:
        f.write(c["prompt"])
    video, audio = pipe(prompt=c["prompt"], height=a.height, width=a.width, num_frames=a.num_frames,
                        num_inference_steps=a.steps, seed=a.seed)
    write_video_audio(video=video, audio=audio, output_path=f"{a.out}/{c['id']}.mp4", fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
    meta = dict(id=c["id"], title=c["title"], seed=a.seed, steps=a.steps, height=a.height, width=a.width, num_frames=len(video),
                seconds=len(video) / 24, flow_shift=12.0, audio_flow_shift=3.0, cfg_scale=1.0, checkpoint=a.w, sec=round(time.time() - t0))
    with open(f"{a.out}/{c['id']}.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"[oos] {c['id']} {meta['num_frames']} frames {meta['sec']}s", flush=True)
print("[oos] done", flush=True)
