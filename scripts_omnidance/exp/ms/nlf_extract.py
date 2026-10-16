"""NLF 单目 3D 人体估计：每个 clip 的 81 帧(16fps) → SMPL 参数 + 24 关节 3D。
用法: python nlf_extract.py --ids ids.txt --shard i --nshard n --out DIR
输出 DIR/<sid>.npz: pose[T,72] betas[T,10] trans[T,3] j3d[T,24,3](m) unc[T,24] box[T,5] valid[T] fps
"""
import argparse, os, sys, time
import numpy as np
import torch, torchvision  # noqa: F401  (NLF torchscript 需要 torchvision nms)
import decord

VID = '/apdcephfs_gy4/share_302533218/cedricnie/OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/videos/{}.mp4'
NLF = os.environ.get('NLF_PATH', '/apdcephfs_gy4/share_302533218/cedricnie/models/nlf/nlf_l_multi_0.3.2.torchscript')

ap = argparse.ArgumentParser()
ap.add_argument('--ids', required=True)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshard', type=int, default=1)
ap.add_argument('--out', required=True)
ap.add_argument('--bs', type=int, default=27)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
allids = [l.strip() for l in open(a.ids) if l.strip()]
# 动态领取：所有 worker 遍历全表（交错起点），用 O_EXCL 锁文件抢占，任意 worker 数都能吃满
ids = allids[a.shard::a.nshard] + [x for i, x in enumerate(allids) if i % a.nshard != a.shard]
m = torch.jit.load(NLF).cuda().eval()
t0 = time.time(); done = 0
for sid in ids:
    dst = os.path.join(a.out, sid + '.npz')
    if os.path.exists(dst):
        continue
    lock = dst + '.lock'
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        continue
    try:
        vr = decord.VideoReader(VID.format(sid))
        fps = vr.get_avg_fps()
        fr = torch.from_numpy(vr.get_batch(list(range(len(vr)))).asnumpy()).permute(0, 3, 1, 2)
    except Exception as e:
        print('READFAIL', sid, e, flush=True); os.remove(lock); continue
    T = fr.shape[0]
    pose = np.full((T, 72), np.nan, np.float32); betas = np.full((T, 10), np.nan, np.float32)
    trans = np.full((T, 3), np.nan, np.float32); j3d = np.full((T, 24, 3), np.nan, np.float32)
    unc = np.full((T, 24), np.nan, np.float32); box = np.zeros((T, 5), np.float32); valid = np.zeros(T, bool)
    with torch.inference_mode():
        for s in range(0, T, a.bs):
            r = m.detect_smpl_batched(fr[s:s + a.bs].cuda())
            for k in range(len(r['boxes'])):
                b = r['boxes'][k]
                if b.shape[0] == 0:
                    continue
                area = (b[:, 2] * b[:, 3]) if b.shape[1] >= 4 else b[:, 0]
                # boxes = [x, y, w, h, conf]；多人时取面积最大
                i = int(torch.argmax(area))
                t = s + k
                pose[t] = r['pose'][k][i].float().cpu().numpy(); betas[t] = r['betas'][k][i].float().cpu().numpy()
                trans[t] = r['trans'][k][i].float().cpu().numpy(); j3d[t] = r['joints3d'][k][i].float().cpu().numpy() / 1000.
                unc[t] = r['joint_uncertainties'][k][i].float().cpu().numpy(); box[t] = b[i].float().cpu().numpy(); valid[t] = True
    np.savez(dst + '.tmp.npz', pose=pose, betas=betas, trans=trans, j3d=j3d, unc=unc, box=box, valid=valid, fps=fps)
    os.replace(dst + '.tmp.npz', dst)
    os.remove(lock)
    done += 1
    if done % 10 == 0:
        print(f'{done}/{len(ids)} {(time.time()-t0)/done:.1f}s/clip', flush=True)
print('DONE', done, flush=True)
