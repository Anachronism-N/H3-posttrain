"""舞蹈配乐音频特征（只用 librosa，单独进程，避免与 torch 同进程段错误）：
BPM、节拍时刻、RMS 能量(整体/三段)、起音密度、频谱质心。输出 jsonl。
用法: python audio_feats.py --ids ids.txt --out audio_feats.jsonl --workers 64
"""
import argparse, json, os
import numpy as np
from multiprocessing import Pool
WAV = '/apdcephfs_gy4/share_302533218/cedricnie/OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/audio/{}.wav'


def beat_dp(oenv, sr, hop=512, tightness=100.):
    """全局速度：onset 自相关在 60-180 BPM 内取峰(对数正态先验 120)；节拍：Ellis(2007) 动态规划。"""
    fps = sr / hop
    o = (oenv - oenv.mean()) / (oenv.std() + 1e-8)
    ac = np.correlate(o, o, 'full')[len(o) - 1:]
    lags = np.arange(len(ac)); bpm = 60 * fps / np.maximum(lags, 1)
    ok = (bpm >= 60) & (bpm <= 180)
    w = np.exp(-0.5 * (np.log2(bpm / 120.) / 1.0) ** 2)
    lag = int(lags[ok][np.argmax((ac * w)[ok])])
    period = float(lag)
    score = o.copy(); back = -np.ones(len(o), int)
    for t in range(len(o)):
        lo, hi = int(t - 2 * period), int(t - period / 2)
        if hi <= 0: continue
        prev = np.arange(max(0, lo), hi)
        pen = -tightness * np.log((t - prev) / period) ** 2
        j = np.argmax(score[prev] + pen); score[t] = o[t] + score[prev][j] + pen[j]; back[t] = prev[j]
    tail = np.arange(max(0, len(o) - int(period)), len(o))
    t = int(tail[np.argmax(score[tail])]); beats = []
    while t >= 0: beats.append(t); t = back[t]
    beats = np.array(beats[::-1]) / fps
    return 60 * fps / period, beats.tolist()


def one(sid):
    import librosa
    try:
        y, sr = librosa.load(WAV.format(sid), sr=22050, mono=True)
        dur = len(y) / sr
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, bt = beat_dp(oenv, sr)  # librosa.beat.beat_track 在 numba gufunc 段错误 → 纯 numpy Ellis DP
        from scipy.signal import find_peaks
        fr = np.lib.stride_tricks.sliding_window_view(y, 2048)[::512]
        rms = np.sqrt((fr ** 2).mean(1))
        seg = [float(x.mean()) for x in np.array_split(rms, 3)]
        on, _ = find_peaks(oenv, height=oenv.mean() + 0.5 * oenv.std(), distance=3)
        mag = np.abs(np.fft.rfft(fr * np.hanning(2048), axis=1)); fq = np.fft.rfftfreq(2048, 1 / sr)
        cen = float(((mag * fq).sum(1) / (mag.sum(1) + 1e-8)).mean())
        return dict(sid=sid, dur=dur, bpm=tempo, beats=[round(b, 3) for b in bt], rms=float(rms.mean()),
                    rms_seg=seg, onset_rate=len(on) / max(dur, 1e-3), centroid=cen)
    except Exception as e:
        return dict(sid=sid, err=repr(e)[:200])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', required=True); ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=64)
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)['sid'] for l in open(a.out)}
    ids = [i for i in ids if i not in done]
    with open(a.out, 'a') as fo, Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(one, ids, chunksize=16)):
            fo.write(json.dumps(r) + '\n')
            if (i + 1) % 200 == 0: print(i + 1, flush=True); fo.flush()
    print('DONE', flush=True)
