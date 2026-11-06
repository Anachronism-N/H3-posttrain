import argparse, json, os, sys, shutil
from pathlib import Path
import torch.distributed as dist
sys.path.insert(0, 'gen_video')
from gen_video_global import init_dit_model, get_music_base_feature, gen_video

manifest_path = Path(sys.argv[1]); base = manifest_path.parent
cases = json.loads(manifest_path.read_text())
pipe = init_dit_model(argparse.Namespace())
os.makedirs('outputs/benchmark10', exist_ok=True)
out_dir = os.environ.get('WD_OUT')
if out_dir and dist.get_rank() == 0:
    os.makedirs(out_dir, exist_ok=True)
for case in cases:
    feature = str(Path('outputs/benchmark10') / (case['id'] + '_music.npy'))
    if dist.get_rank() == 0:
        get_music_base_feature(str(base / case['audio']), feature, fps=30)
    dist.barrier(device_ids=[dist.get_rank()])
    out = str(Path('outputs/benchmark10') / (case['id'] + '.mp4'))
    gen_video(pipe, feature, str(base / case['image']), case['prompt'], out,
              seed=0, height=1280, width=720, num_frames=149,
              enable_refimage=True, refimage_path=str(base / case['image']),
              enable_global=True, enable_dynamicfps=True, enable_vae_decode_framewise=True,
              enable_skip_layer=True, enable_unimodel=True, sigma_shift=5,
              num_inference_steps=48, cfg_scale=5)
    dist.barrier(device_ids=[dist.get_rank()])
    if out_dir and dist.get_rank() == 0 and os.path.isfile(out):
        partial = os.path.join(out_dir, os.path.basename(out) + '.partial')
        shutil.copyfile(out, partial)
        os.replace(partial, os.path.join(out_dir, os.path.basename(out)))
