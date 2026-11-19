"""NLF npz → SMPL-H(52关节) 前向 → MotionScript 动作描述 + 运动学维度原始数值。
用法: python ms_caption.py --nlf DIR --out DIR [--ids ids.txt] [--workers 64]
输出 DIR/<sid>.json: {ms: [句子...], ms_nonagg: [...], kin: {...}, valid_ratio}
坐标约定（与 MotionScript/HumanML3D 一致）:
  y 朝上、米为单位；body 关节每帧去掉 yaw（身体朝 +z），保留俯仰/侧倾；地面 y=0；
  虚拟关节 orientation = 根节点欧拉角(度, 站立≈0, yaw 相对首帧)；translation = 骨盆世界位移(相对首帧, 首帧朝向对齐 +z)。
"""
import argparse, os, sys, io, json, glob, random, contextlib, traceback
import numpy as np
from multiprocessing import Pool

MS_SRC = '/apdcephfs_gy4/share_302533218/cedricnie/MotionScript/src'
SHIM = '/apdcephfs_gy4/share_302533218/cedricnie/MotionScript/shim'
SMPLH = '/apdcephfs_gy4/share_302533218/cedricnie/models/smplh/smplh_neutral_joints.npz'
FPS_MS = 20.0


def rodrigues(r):  # r [...,3] -> [...,3,3]
    th = np.linalg.norm(r, axis=-1, keepdims=True) + 1e-8
    k = r / th
    K = np.zeros(r.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def rot_x(deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):  # a [T] rad -> [T,3,3]
    c, s = np.cos(a), np.sin(a); z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def fk(rots, J, parents):  # rots [T,52,3,3] (rots[:,0] = 全局朝向), J [52,3] -> 相对骨盆的关节位置 [T,52,3]
    T = rots.shape[0]
    Rg = np.zeros((T, 52, 3, 3)); P = np.zeros((T, 52, 3))
    Rg[:, 0] = rots[:, 0]
    for j in range(1, 52):
        p = parents[j]
        Rg[:, j] = Rg[:, p] @ rots[:, j]
        P[:, j] = P[:, p] + np.einsum('tab,b->ta', Rg[:, p], J[j] - J[p])
    return P


def interp_nan(x, valid):  # x [T,...]
    T = x.shape[0]; t = np.arange(T); v = np.where(valid)[0]
    flat = x.reshape(T, -1).copy()
    for c in range(flat.shape[1]):
        flat[:, c] = np.interp(t, v, flat[v, c])
    return flat.reshape(x.shape)


def smooth(x, sigma=1.0):
    if sigma <= 0: return x
    r = int(3 * sigma); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    xp = np.concatenate([x[:1].repeat(r, 0), x, x[-1:].repeat(r, 0)], 0)
    out = np.zeros_like(x)
    for i, w in enumerate(k):
        out += w * xp[i:i + x.shape[0]]
    return out


def resample(x, src_fps, dst_fps):
    T = x.shape[0]; n = int(round((T - 1) * dst_fps / src_fps)) + 1
    ts = np.linspace(0, T - 1, n); i0 = np.floor(ts).astype(int); i1 = np.minimum(i0 + 1, T - 1); w = (ts - i0)
    w = w.reshape((-1,) + (1,) * (x.ndim - 1))
    return x[i0] * (1 - w) + x[i1] * w


def build_coords(d, tmpl):
    valid = d['valid'].astype(bool)
    if valid.mean() < 0.7:
        return None, valid.mean()
    pose = interp_nan(d['pose'].astype(np.float64), valid)            # [T,72] SMPL (相机坐标)
    pel = interp_nan(d['j3d'][:, 0].astype(np.float64), valid)        # 骨盆相机坐标 (m)
    T = pose.shape[0]
    W = np.diag([1., -1., -1.])                                       # 相机(x右,y下,z前) → y 朝上世界
    rots = np.tile(np.eye(3), (T, 52, 1, 1))
    rots[:, :22] = rodrigues(pose[:, :66].reshape(T, 22, 3))
    Rw = W @ rots[:, 0]
    rots[:, 0] = Rw
    J = tmpl['J_template']
    P = fk(rots, J, tmpl['parents'])                                  # 相对骨盆, y 朝上世界
    fwd = Rw @ np.array([0., 0., 1.])
    yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
    P_local = np.einsum('tab,tjb->tja', rot_y(-yaw), P)              # 去 yaw：身体朝 +z
    pel_w = pel @ W.T
    tr = pel_w - pel_w[:1]
    tr = np.einsum('ab,tb->ta', rot_y(-yaw[:1])[0], tr)              # 首帧朝向对齐 +z
    # 地面：脚(7,8,10,11)最低点的 5 分位
    foot_y = (P[:, [7, 8, 10, 11], 1] + pel_w[:, None, 1]).min(1)
    # 单目竖直漂移：相机俯仰使 y 随深度线性变化 → 先对深度回归去掉；再用 1.5s 滑动低分位作地面基线（跳跃 <0.6s 不会被吃掉）
    A = np.stack([np.ones(T), pel[:, 2]], 1)
    coef = np.linalg.lstsq(A, foot_y, rcond=None)[0]
    fy = foot_y - coef[1] * (pel[:, 2] - pel[:, 2].mean())
    w = max(3, int(round(1.5 * float(d['fps'])))) // 2
    floor = np.array([np.percentile(fy[max(0, i - w):i + w + 1], 10) for i in range(T)])
    floor = smooth(floor[:, None], 2.0)[:, 0] + (foot_y - fy)
    body = P_local.copy(); body[..., 1] += (pel_w[:, None, 1] - floor[:, None])
    # 根欧拉角（MotionScript/Salsa 约定：z 朝上世界、站立为 0，yaw 相对首帧）
    M = rot_x(90.)
    Rp = M @ Rw @ rot_x(-90.)
    tx = np.degrees(np.arctan2(Rp[:, 2, 1], Rp[:, 2, 2]))
    ty = np.degrees(np.arctan2(-Rp[:, 2, 0], np.sqrt(Rp[:, 2, 1] ** 2 + Rp[:, 2, 2] ** 2)))
    tz = np.degrees(np.unwrap(np.arctan2(Rp[:, 1, 0], Rp[:, 0, 0])))
    eul = np.stack([tx, ty, tz - tz[0]], -1)
    eul[:, 2] = (eul[:, 2] + 180) % 360 - 180
    fps = float(d['fps'])
    body = smooth(body, 1.0); tr = smooth(tr, 1.0); eul_s = smooth(eul, 1.0)
    body = resample(body, fps, FPS_MS); tr = resample(tr, fps, FPS_MS); eul_s = resample(eul_s, fps, FPS_MS)
    yaw_r = resample(smooth(np.degrees(yaw)[:, None], 1.0), fps, FPS_MS)[:, 0]
    coords = np.concatenate([body, eul_s[:, None], tr[:, None]], 1)   # [T20, 54, 3]
    return (coords, yaw_r), valid.mean()


def kin_stats(coords, yaw_deg):
    b = coords[:, :52]; tr = coords[:, 53]; T = b.shape[0]; dt = 1. / FPS_MS
    v = np.linalg.norm(np.diff(b[:, :22], axis=0), axis=-1) / dt              # [T-1,22] 局部关节速度
    acc = np.linalg.norm(np.diff(b[:, :22], 2, axis=0), axis=-1) / dt ** 2
    arms = [16, 17, 18, 19, 20, 21]; legs = [4, 5, 7, 8, 10, 11]; torso = [3, 6, 9, 12, 15]
    rel = b[:, :22] - b[:, :1]
    vt = np.linalg.norm(np.diff(rel[:, torso], axis=0), axis=-1).mean() / dt
    va, vl = v[:, arms].mean(), v[:, legs].mean()
    pel_h = b[:, 0, 1]
    foot = b[:, [10, 11], 1]
    airborne = (foot.min(1) > 0.10)
    # 连续 >=2 帧离地才算跳
    runs, c = [], 0
    for x in airborne:
        c = c + 1 if x else (runs.append(c) or 0)
    runs.append(c); jumps = int(sum(1 for r in runs if r >= 2))
    sp = v.mean(1)
    s = sp - sp.mean(); ac = np.correlate(s, s, 'full')[len(s) - 1:]; ac = ac / (ac[0] + 1e-8)
    lo, hi = int(0.4 * FPS_MS), min(int(2.5 * FPS_MS), len(ac) - 1)
    rep = float(ac[lo:hi].max()) if hi > lo else 0.
    lw, rw = v[:, [18, 20]].mean(1), v[:, [19, 21]].mean(1)
    sym = float(np.corrcoef(lw, rw)[0, 1]) if lw.std() > 1e-6 and rw.std() > 1e-6 else 0.
    hand_h = b[:, [20, 21], 1].max(1) - b[:, 15, 1]                        # 手相对头的高度
    return dict(
        speed=float(v.mean()), speed_p90=float(np.percentile(sp, 90)),
        arm_speed=float(va), leg_speed=float(vl), torso_speed=float(vt),
        accel=float(acc.mean()), sharp=float((acc.mean(1) > 2 * np.median(acc.mean(1))).mean()),
        travel=float(np.linalg.norm(np.diff(tr[:, [0, 2]], axis=0), axis=-1).sum()),
        disp=float(np.linalg.norm(tr[-1, [0, 2]] - tr[0, [0, 2]])),
        turn=float(np.abs(np.diff(yaw_deg)).sum()), net_turn=float(yaw_deg[-1] - yaw_deg[0]),
        pelvis_h_med=float(np.median(pel_h)), pelvis_h_min=float(pel_h.min()),
        pelvis_h_range=float(np.percentile(pel_h, 95) - np.percentile(pel_h, 5)),
        jumps=jumps, airborne_frac=float(airborne.mean()),
        hands_above_head_frac=float((hand_h > 0).mean()),
        repetition=rep, symmetry=sym, frames=int(T))


_TMPL = None


def init():
    global _TMPL
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(1)
    sys.path[:0] = [SHIM, MS_SRC]
    _TMPL = dict(np.load(SMPLH))
    with contextlib.redirect_stdout(io.StringIO()):
        import text2pose.posescript.captioning  # noqa: F401  预热导入


def run_one(args):
    f, outdir = args
    sid = os.path.basename(f)[:-4]
    dst = os.path.join(outdir, sid + '.json')
    if os.path.exists(dst):
        saved = json.load(open(dst))
        if saved.get('sid') != sid or 'ms' not in saved:
            raise ValueError(f'Invalid MotionScript cache: {dst}')
        return sid, 'skip'
    try:
        import torch
        import text2pose.posescript.captioning as cap
        d = np.load(f)
        res, vr = build_coords(d, _TMPL)
        if res is None:
            result = dict(sid=sid, valid_ratio=float(vr), ms=None)
            status = 'lowvalid'
        else:
            coords, yaw = res
            kin = kin_stats(coords, yaw)
            random.seed(0); np.random.seed(0); torch.manual_seed(0)
            tmpd = f'/tmp/ms_tmp/{sid}'; os.makedirs(tmpd, exist_ok=True)
            with contextlib.redirect_stdout(io.StringIO()):
                out = cap.main(torch.tensor(coords, dtype=torch.float32), save_dir=tmpd, babel_info=False,
                               simplified_captions=False, apply_transrel_ripple_effect=False,
                               apply_stat_ripple_effect=False, random_skip=False, motion_tracking=True, verbose=False)
            _, _, nonagg, agg = out
            result = dict(sid=sid, valid_ratio=float(vr), ms=list(agg), ms_nonagg=list(nonagg), kin=kin,
                          source='MotionScript original rules; NLF/SMPL-H body-only input adapter',
                          hand_fingers_observed=False)
            status = 'ok'
        tmp = dst + f'.{os.getpid()}.tmp'
        with open(tmp, 'w') as stream:
            json.dump(result, stream, allow_nan=False)
        os.replace(tmp, dst)
        return sid, status
    except Exception as e:
        open(dst + '.err', 'w').write(traceback.format_exc())
        return sid, 'err:' + repr(e)[:200]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nlf', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--ids', default=''); ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshard', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.nlf, '*.npz')))
    if a.ids:
        keep = set(open(a.ids).read().split()); files = [f for f in files if os.path.basename(f)[:-4] in keep]
    files = files[a.shard::a.nshard]
    init()  # Dependency failures must stop the parent before Pool can respawn failed workers.
    stat = {}
    with Pool(a.workers, initializer=init) as p:
        for i, (sid, st) in enumerate(p.imap_unordered(run_one, [(f, a.out) for f in files])):
            k = st.split(':')[0]; stat[k] = stat.get(k, 0) + 1
            if st.startswith('err'): print(sid, st, flush=True)
            if (i + 1) % 200 == 0: print(i + 1, stat, flush=True)
    print('DONE', stat, flush=True)
