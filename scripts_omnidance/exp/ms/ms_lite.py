"""MS-lite：MotionScript 思路（posecode 姿态状态 + motioncode 动作事件 + 时间连接词）的确定性动作描述器，
针对单目 NLF 舞蹈数据重设阈值。原版 MotionScript 阈值按 HumanML3D 动捕调，在我们数据上大多输出空句。
输入: NLF npz 目录。输出 OUT/<sid>.json: {segs:[[事件短语...]x3], kin:{...}, coords20 另存 npz}
坐标(build_coords): y 朝上(m), 身体去 yaw 后朝 +z, +x = 舞者自己的左侧; tr = 骨盆位移(首帧朝向系)。
用法: python ms_lite.py --nlf DIR --out DIR [--workers 96]
"""
import argparse, os, sys, json, glob, traceback
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ms_caption import build_coords, kin_stats, SMPLH, FPS_MS  # noqa: E402

# SMPL 关节
PEL, LHIP, RHIP, SP1, LKNEE, RKNEE, SP2, LANK, RANK, SP3, LFOOT, RFOOT, NECK, LCOL, RCOL, HEAD, LSH, RSH, LELB, RELB, LWR, RWR = range(22)
NSEG = 3


def ang(a, b, c):  # 关节 b 处夹角(度)  a,b,c [T,3]
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def dist_ok(t):
    return np.linalg.norm((t[-1] - t[0])[[0, 2]]) < 0.8


def side_word(s):
    return {'l': 'left', 'r': 'right', 'b': 'both'}[s]


def merge_lr(evs):
    """把左右成对的同一事件合并为 both。evs: list of (key, side, text_fmt, score)"""
    out, used = [], set()
    for i, (k, s, f, sc) in enumerate(evs):
        if i in used: continue
        if s in 'lr':
            for j in range(i + 1, len(evs)):
                if j not in used and evs[j][0] == k and evs[j][1] == ('r' if s == 'l' else 'l'):
                    used.add(j); s = 'b'; sc = max(sc, evs[j][3]); break
        out.append((k, s, f, sc))
    return out


def arm_text(fmt, s):
    if s == 'n':
        return fmt
    if s == 'b':
        return fmt.replace('the {side} arm', 'both arms').replace('the {side} hand', 'both hands') \
                  .replace('the {side} elbow', 'both elbows').replace('{side}', 'both')
    return fmt.replace('{side}', side_word(s))


def seg_events(b, tr, yaw, sl, gq):
    """一个时间段内的动作事件。b [T,52,3] 局部坐标; tr [T,3]; yaw [T] 度; sl slice; gq 全局速度分位(用于快慢副词)"""
    x = b[sl]; t = tr[sl]; y = yaw[sl]; T = x.shape[0]
    if T < 4: return []
    ev = []
    a0, a1 = slice(0, max(2, T // 4)), slice(T - max(2, T // 4), T)   # 段首/段尾窗口
    m0 = lambda q: q[a0].mean(0); m1 = lambda q: q[a1].mean(0)
    head_y = x[:, HEAD, 1]
    for s, sh, el, wr in (('l', LSH, LELB, LWR), ('r', RSH, RELB, RWR)):
        hy = x[:, wr, 1] - x[:, sh, 1]                          # 手相对肩高度
        dh = m1(hy) - m0(hy)
        above = (x[:, wr, 1] > head_y + 0.03)
        if dh > 0.20 and above[a1].mean() > 0.5:
            ev.append(('raise_up', s, 'raises the {side} arm above the head', dh))
        elif dh > 0.20:
            ev.append(('raise', s, 'raises the {side} arm', dh))
        elif dh < -0.20:
            ev.append(('lower', s, 'lowers the {side} arm', -dh))
        lat = np.abs(x[:, wr, 0] - x[:, sh, 0]); fwd = x[:, wr, 2] - x[:, sh, 2]
        if m1(lat) - m0(lat) > 0.18 and m1(lat) > 0.40:
            ev.append(('ext_side', s, 'stretches the {side} arm out to the side', m1(lat) - m0(lat)))
        if m1(fwd) - m0(fwd) > 0.18 and m1(fwd) > 0.35:
            ev.append(('ext_fwd', s, 'reaches the {side} arm forward', m1(fwd) - m0(fwd)))
        e = ang(x[:, sh], x[:, el], x[:, wr]); de = m1(e) - m0(e)
        if de < -35: ev.append(('bend_elb', s, 'bends the {side} elbow', -de / 100))
        elif de > 35: ev.append(('str_elb', s, 'straightens the {side} arm', de / 100))
        sp = np.linalg.norm(np.diff(x[:, wr] - x[:, sh], axis=0), axis=-1).sum()
        if sp > 1.2 and abs(dh) < 0.15:
            ev.append(('swing', s, 'swings the {side} arm around', sp / 4))
    for s, hp, kn, an in (('l', LHIP, LKNEE, LANK), ('r', RHIP, RKNEE, RANK)):
        ky = x[:, kn, 1] - x[:, hp, 1]
        if ky.max() > -0.12 and (ky.max() - ky[0]) > 0.15:
            ev.append(('knee_up', s, 'lifts the {side} knee high', ky.max() - ky[0] + 0.2))
        fz = x[:, an, 2] - x[:, hp, 2]; fx = np.abs(x[:, an, 0] - x[:, hp, 0]); fy = x[:, an, 1] - x[:, PEL, 1]
        if fz.max() > 0.45 and fy.max() > -0.55:
            ev.append(('kick', s, 'kicks the {side} leg forward', fz.max()))
        elif fx.max() > 0.55 and fy.max() > -0.6:
            ev.append(('leg_side', s, 'lifts the {side} leg out to the side', fx.max()))
    # 层次：骨盆高度
    ph = x[:, PEL, 1]; dph = m1(ph) - m0(ph)
    if dph < -0.15: ev.append(('squat', 'n', 'squats down low' if m1(ph) < 0.6 else 'bends the knees and lowers the body', -dph * 2))
    elif dph > 0.15: ev.append(('rise', 'n', 'rises back up', dph * 2))
    elif ph.max() - ph.min() > 0.08 and np.sum(np.abs(np.diff(np.sign(np.diff(ph))))) >= 6:
        ev.append(('bounce', 'n', 'bounces on the knees', 0.3))
    if ph.mean() < 0.55: ev.append(('low', 'n', 'stays low near the floor', 0.6))
    # 跳跃：双脚同时离地 >=2 帧
    air = (x[:, [LFOOT, RFOOT], 1].min(1) > 0.10)
    runs = np.diff(np.concatenate([[0], air.astype(int), [0]])); nj = int(((np.where(runs == -1)[0] - np.where(runs == 1)[0]) >= 2).sum())
    if nj >= 2: ev.append(('jump', 'n', 'jumps up repeatedly', 1.0 + nj * 0.2))
    elif nj == 1: ev.append(('jump', 'n', 'jumps up', 1.0))
    # 躯干倾斜
    tv = x[:, NECK] - x[:, PEL]; tv = tv / (np.linalg.norm(tv, axis=-1, keepdims=True) + 1e-8)
    pitch = np.degrees(np.arcsin(np.clip(tv[:, 2], -1, 1))); roll = np.degrees(np.arcsin(np.clip(tv[:, 0], -1, 1)))
    if m1(pitch) - m0(pitch) > 25 or pitch[a1].mean() > 35:
        ev.append(('lean_f', 'n', 'bends the upper body forward', max(m1(pitch) - m0(pitch), pitch[a1].mean()) / 50))
    elif m1(pitch) - m0(pitch) < -20 or pitch[a1].mean() < -20:
        ev.append(('lean_b', 'n', 'leans back', 0.5))
    if abs(m1(roll) - m0(roll)) > 18:
        ev.append(('lean_s', 'n', f"leans to the {'left' if m1(roll) > m0(roll) else 'right'}", abs(m1(roll) - m0(roll)) / 40))
    # 双手相碰
    hd = np.linalg.norm(x[:, LWR] - x[:, RWR], axis=-1)
    if hd.min() < 0.08 and hd.max() > 0.35: ev.append(('clap', 'n', 'brings the hands together', 0.5))
    # 转身 / 位移
    dyaw = y[-1] - y[0]
    if abs(dyaw) > 300: ev.append(('spin', 'n', f"spins a full circle to the {'left' if dyaw > 0 else 'right'}", 3.0))
    elif abs(dyaw) > 140: ev.append(('turn_back', 'n', 'turns around', 2.0))
    elif abs(dyaw) > 50: ev.append(('turn', 'n', f"turns to the {'left' if dyaw > 0 else 'right'}", abs(dyaw) / 90))
    d = t[-1] - t[0]; dist = np.linalg.norm(d[[0, 2]])
    if dist > 0.35:
        yr = np.radians(y[0]); c, s_ = np.cos(yr), np.sin(yr)          # 转到段首身体朝向系
        dx, dz = c * d[0] - s_ * d[2], s_ * d[0] + c * d[2]
        dirw = ('to the left' if dx > 0 else 'to the right') if abs(dx) > abs(dz) else ('forward' if dz > 0 else 'backward')
        ev.append(('step', 'n', ('travels ' if dist > 0.8 else 'steps ') + dirw, dist))
    # 持续姿态(posecode)：段内大部分时间成立
    for s, sh, wr in (('l', LSH, LWR), ('r', RSH, RWR)):
        if (x[:, wr, 1] > head_y + 0.03).mean() > 0.7:
            ev.append(('hold_up', s, 'keeps the {side} arm raised overhead', 0.35))
        elif (np.abs(x[:, wr, 0] - x[:, sh, 0]) > 0.45).mean() > 0.7:
            ev.append(('hold_side', s, 'holds the {side} arm out to the side', 0.3))
    hc = ((x[:, [LWR, RWR], 1].mean(1) > x[:, SP3, 1] - 0.1) & (x[:, [LWR, RWR], 1].mean(1) < x[:, NECK, 1] + 0.05) &
          (np.linalg.norm(x[:, LWR] - x[:, RWR], axis=-1) < 0.35)).mean()
    if hc > 0.7: ev.append(('hands_chest', 'n', 'keeps the hands close in front of the chest', 0.3))
    # 抬脚换步：左右脚交替离地
    la = x[:, LFOOT, 1] - x[:, LFOOT, 1].min(); ra = x[:, RFOOT, 1] - x[:, RFOOT, 1].min()
    nst = int(((la[1:] > 0.06) & (la[:-1] <= 0.06)).sum() + ((ra[1:] > 0.06) & (ra[:-1] <= 0.06)).sum())
    if nst >= 3 and dist_ok(t):
        ev.append(('steps', 'n', 'steps from foot to foot', 0.25 + 0.05 * nst))
    # 摆胯：骨盆左右往返
    px = t[:, 0] - t[:, 0].mean()
    if px.max() - px.min() > 0.15 and np.sum(np.abs(np.diff(np.sign(np.diff(px))))) >= 4:
        ev.append(('sway', 'n', 'sways the hips from side to side', 0.3))
    ev = merge_lr(ev)
    # 速度副词
    v = np.linalg.norm(np.diff(x[:, :22], axis=0), axis=-1).mean() * FPS_MS
    adv = 'quickly' if v > gq[1] else ('slowly' if v < gq[0] else '')
    ev.sort(key=lambda e: -e[3])
    return [dict(k=k, side=s, text=arm_text(f, s), score=float(sc)) for k, s, f, sc in ev[:6]], adv, float(v)


def run_one(args):
    f, outdir, gq = args
    sid = os.path.basename(f)[:-4]; dst = os.path.join(outdir, sid + '.json')
    if os.path.exists(dst): return sid, 'skip'
    try:
        d = np.load(f)
        res, vr = build_coords(d, TM)
        if res is None:
            json.dump(dict(sid=sid, valid_ratio=float(vr), segs=None), open(dst, 'w')); return sid, 'lowvalid'
        coords, yaw = res
        b, tr = coords[:, :52], coords[:, 53]
        T = b.shape[0]; bounds = np.linspace(0, T, NSEG + 1).astype(int)
        segs = []
        for i in range(NSEG):
            r = seg_events(b, tr, yaw, slice(bounds[i], bounds[i + 1] + 1), gq)
            segs.append(dict(events=r[0], adv=r[1], speed=r[2]) if r else dict(events=[], adv='', speed=0.))
        kin = kin_stats(coords, yaw)
        np.savez_compressed(os.path.join(outdir, sid + '_c20.npz'), body=b[:, :22].astype(np.float16), tr=tr.astype(np.float16), yaw=yaw.astype(np.float16))
        json.dump(dict(sid=sid, valid_ratio=float(vr), segs=segs, kin=kin), open(dst, 'w'))
        return sid, 'ok'
    except Exception:
        open(dst + '.err', 'w').write(traceback.format_exc()); return sid, 'err'


TM = None


def init():
    global TM
    TM = dict(np.load(SMPLH))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nlf', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=96); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--gq', default='0.25,0.55', help='段速度 慢/快 阈值 (m/s)')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.nlf, '*.npz')))
    files = [f for f in files if '.tmp' not in f]
    if a.limit: files = files[:a.limit]
    gq = [float(v) for v in a.gq.split(',')]
    st = {}
    with Pool(a.workers, initializer=init) as p:
        for i, (sid, s) in enumerate(p.imap_unordered(run_one, [(f, a.out, gq) for f in files], chunksize=8)):
            st[s] = st.get(s, 0) + 1
            if (i + 1) % 1000 == 0: print(i + 1, st, flush=True)
    print('DONE', st, flush=True)
