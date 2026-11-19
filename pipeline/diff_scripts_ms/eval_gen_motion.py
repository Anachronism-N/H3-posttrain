"""对生成视频跑 NLF 3D + 运动学统计（从第 1 秒起，首秒被参考帧锚定），用于客观判断 prompt 遵循/可区分性。
用法: python eval_gen_motion.py --dirs DIR1 DIR2 ... --out res.json   (DIR 下 joint/ aswap/ 的 mp4)
"""
import argparse, glob, json, os, sys
import numpy as np, torch, torchvision  # noqa: F401
import decord

ap = argparse.ArgumentParser(); ap.add_argument('--dirs', nargs='+'); ap.add_argument('--out', required=True)
ap.add_argument('--skip_s', type=float, default=1.0)
a = ap.parse_args()
m = torch.jit.load(os.environ.get('NLF_PATH', '/apdcephfs_gy4/share_302533218/cedricnie/models/nlf/nlf_l_multi_0.3.2.torchscript')).cuda().eval()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ms_caption import build_coords, kin_stats, SMPLH  # noqa: E402  (导入会清空 CUDA_VISIBLE_DEVICES，须在 CUDA 初始化之后)
TM = dict(np.load(SMPLH))
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
for d in a.dirs:
    for f in sorted(glob.glob(f'{d}/*/*.mp4')):
        key = f'{os.path.basename(d)}/{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)[:-4]}'
        if key in res: continue
        vr = decord.VideoReader(f); fps = vr.get_avg_fps()
        s0 = int(round(a.skip_s * fps)); idx = list(range(s0, len(vr)))
        fr = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)
        T = fr.shape[0]
        pose = np.full((T, 72), np.nan, np.float32); j3d = np.full((T, 24, 3), np.nan, np.float32); valid = np.zeros(T, bool)
        with torch.inference_mode():
            for s in range(0, T, 24):
                r = m.detect_smpl_batched(fr[s:s + 24].cuda())
                if 'boxes' not in r:
                    if any(joints.numel() for joints in r['joints3d']):
                        raise ValueError(f'NLF joints without boxes: {f}, batch {s}')
                    continue
                for k in range(len(r['boxes'])):
                    b = r['boxes'][k]
                    if b.shape[0] == 0: continue
                    i = int(torch.argmax(b[:, 2] * b[:, 3])); t = s + k
                    pose[t] = r['pose'][k][i].float().cpu().numpy(); j3d[t] = r['joints3d'][k][i].float().cpu().numpy() / 1000.; valid[t] = True
        out, vrat = build_coords(dict(pose=pose, j3d=j3d, valid=valid, fps=fps), TM)
        if out is None:
            res[key] = dict(valid=float(vrat)); continue
        coords, yaw = out
        res[key] = dict(valid=float(vrat), **kin_stats(coords, yaw))
        print(key, {k: round(v, 3) for k, v in res[key].items() if k in ('speed', 'jumps', 'airborne_frac', 'pelvis_h_min', 'turn', 'net_turn', 'arm_speed', 'leg_speed')}, flush=True)
        json.dump(res, open(a.out, 'w'), indent=0)
json.dump(res, open(a.out, 'w'), indent=0)
print('DONE', len(res))
