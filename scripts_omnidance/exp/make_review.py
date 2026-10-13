"""Assemble side-by-side review videos for human inspection.
python make_review.py --out <dir> --cases 6 --col "LABEL=<dir with cond subdirs>:<cond>" ... [--gt]
Each output: <sid>.mp4 = hstack of columns (scaled to H=640, 16 fps, 81/49 frames padded), with the *original* music.
Also writes index.md with prompt text and the swap sources.
"""
import argparse, csv, glob, os, subprocess
csv.field_size_limit(10 ** 9)
p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--cases", type=int, default=6)
p.add_argument("--ids", default="")
p.add_argument("--col", action="append", default=[])
p.add_argument("--gt", action="store_true")
a = p.parse_args()
E2 = "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/E2"
base = list(csv.DictReader(open(f"{E2}/full_joint.csv")))
n = len(base)
ids = a.ids.split(",") if a.ids else [base[i]["sample_id"] for i in range(0, n, max(1, n // a.cases))][:a.cases]
FF = "/usr/local/bin/ffmpeg"
os.makedirs(a.out, exist_ok=True)
idx = ["# Review set", "", "列顺序：" + " | ".join((["GT"] if a.gt else []) + [c.split("=")[0] for c in a.col]), "音轨：原音乐（aswap 列实际使用的是另一首音乐，见 swap 源）", ""]
for sid in ids:
    i = next(k for k, b in enumerate(base) if b["sample_id"] == sid)
    b = base[i]
    vids, labels = [], []
    if a.gt:
        vids.append(b["video"]); labels.append("GT")
    for c in a.col:
        lab, spec = c.split("=", 1)
        d, cond = spec.rsplit(":", 1)
        m = sorted(glob.glob(f"{d}/{cond}/*{sid}_*.mp4"))
        if not m:
            continue
        vids.append(m[0]); labels.append(lab)
    inputs, filt = [], []
    for k, (v, l) in enumerate(zip(vids, labels)):
        inputs += ["-i", v]
        filt.append(f"[{k}:v]fps=16,trim=end_frame=81,scale=-2:640,setsar=1,tpad=stop_mode=clone:stop_duration=2,trim=end_frame=81,"
                    f"drawtext=text='{l}':x=8:y=8:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.5[v{k}]")
    filt.append("".join(f"[v{k}]" for k in range(len(vids))) + f"hstack=inputs={len(vids)}[out]")
    dst = f"{a.out}/{sid}.mp4"
    cmd = [FF, "-y", "-loglevel", "error"] + inputs + ["-i", b["audio_path"], "-filter_complex", ";".join(filt),
           "-map", "[out]", "-map", f"{len(vids)}:a", "-shortest", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        # retry without drawtext (font may be missing)
        cmd = [x if "drawtext" not in x else ";".join(f.split(",drawtext")[0] + f"[v{k}]" if k < len(vids) else f for k, f in enumerate(filt)) for x in cmd]
        r = subprocess.run(cmd, capture_output=True, text=True)
    idx += [f"## {sid}", f"- 文件：`{sid}.mp4`（{r.returncode == 0 and 'ok' or 'FAILED ' + r.stderr[-200:]}）",
            f"- tswap 使用的 prompt 来自 {base[(i + 23) % n]['sample_id']}；aswap 使用的音乐来自 {base[(i + 17) % n]['sample_id']}",
            f"- 原 prompt：{b['prompt'][:700]}", ""]
open(f"{a.out}/index.md", "w").write("\n".join(idx))
print("\n".join(l for l in idx if l.startswith("- 文件")))
