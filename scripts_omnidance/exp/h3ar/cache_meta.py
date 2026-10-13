"""cache_meta.py <cache dir> <out.json> [nproc]: per-sample sizes of a split-training cache (CPU, mmap): n_ra, img/audio token counts, prompt length."""
import glob, json, os, sys
from multiprocessing import Pool
import torch
def one(f):
    c = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
    pk = c[1]["packed"]
    return os.path.basename(f)[:-4], dict(n_ra=int(c[0]["ref_audio_anchor"].shape[0]), n_img=int(pk["img_pos"].numel()),
                                          n_aud=int(pk["audio_pos"].numel()), L=int(c[1]["prompt_embeds"].shape[0]))
if __name__ == "__main__":
    fs = sorted(glob.glob(f"{sys.argv[1]}/*.pth"))
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 16) as p:
        res = dict(p.map(one, fs, chunksize=32))
    json.dump(res, open(sys.argv[2], "w")); print("[meta]", len(res))
