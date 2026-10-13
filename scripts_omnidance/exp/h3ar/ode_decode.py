"""ode_decode.py <traj.pt> <out.mp4> [--gt]: decode the teacher x0 of an ODE trajectory (or the cached GT latents) to check it."""
import glob, sys, torch
sys.path.insert(0, "/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930")
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
W = "/dev/shm/h3/Ref2VA"
pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[ModelConfig(path=f"{W}/video_vae/source/model.safetensors"), ModelConfig(path=f"{W}/audio_vae/model.safetensors")],
    processor_config=ModelConfig(path=f"{W}/processor/"))
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
c = torch.load(d["src"], map_location="cpu", weights_only=False)
lat = (c[0]["input_latents"][0] if "--gt" in sys.argv else d["x"][-1]).unsqueeze(0).to("cuda", torch.bfloat16)
fr = pipe.video_vae.decode_video(lat, dtype=torch.bfloat16, tiled=True, tile_size=256, tile_overlap=64)
video = pipe.vae_output_to_video(fr, min_value=0, max_value=1)
aud = None  # visual check only
write_video_audio(video=video, audio=aud.float().cpu() if aud is not None else None, output_path=sys.argv[2], fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
print("ok", sys.argv[2], len(video))
