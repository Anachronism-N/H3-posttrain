#!/usr/bin/env python3
"""Incrementally upload OmniDance fullSFT checkpoints to NZC415/m2d.

Scans all stage output dirs for step-*.safetensors, uploads any that are not
yet recorded in hf_uploaded.txt. Safe to re-run periodically (idempotent).
"""
import os
import time
import glob

from huggingface_hub import HfApi

TOKEN = "<HF_TOKEN_REDACTED>"
REPO_ID = "NZC415/m2d"
MODELS_ROOT = "/apdcephfs_gy4/share_302533218/cedricnie/diff/models/train"
REPO_PREFIX = "OmniDance-fullSFT"
DONE_FILE = "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/hf_uploaded.txt"
LOG = "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/hf_upload.log"

api = HfApi(token=TOKEN)


def log(msg):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_done():
    if not os.path.exists(DONE_FILE):
        return set()
    with open(DONE_FILE) as f:
        return {l.strip() for l in f if l.strip()}


def mark_done(key):
    with open(DONE_FILE, "a") as f:
        f.write(key + "\n")


def main():
    done = load_done()
    import time
    now = time.time()
    ckpts = []
    # 扫描多个产出根目录:三阶段主仓库 + warm2brch(主仓库) + maskFT(worktree)
    scan_roots = [
        f"{MODELS_ROOT}/OSS_fullSFT_stage*",
        f"{MODELS_ROOT}/warm2brch_*",
        f"{MODELS_ROOT}/../diff-wl/models/train/OSS_maskFT*",
    ]
    stage_dirs = []
    for pattern in scan_roots:
        for d in sorted(glob.glob(pattern)):
            if os.path.isdir(d) and d not in stage_dirs:
                stage_dirs.append(d)
    for stage_dir in stage_dirs:
        stage_name = os.path.basename(stage_dir)
        # 配额有限:新 lineage(warm2brch/maskFT)只传里程碑步,全量本地有底
        MILESTONES = {
            "warm2brch_fullSFT_stage1_aud5e5_xattn5e6_warm1500": {8000},
            "warm2brch_fullSFT_stage2_fullDiT_lr1e5_warm500": {6000, 12000},
            "OSS_maskFT_from12k": {1500, 3000},
        }
        allowed = MILESTONES.get(stage_name)
        for path in sorted(glob.glob(f"{stage_dir}/step-*.safetensors")):
            # 跳过 10 分钟内修改过的文件(可能正在写入)
            if now - os.path.getmtime(path) < 600:
                continue
            if allowed is not None:
                step_num = int(os.path.basename(path)[5:].split(".")[0])
                if step_num not in allowed:
                    continue
            ckpts.append((stage_name, path))

    # 每个阶段目录附带 args.json / training_config.yaml(推理脚本依赖 ckpt 旁的
    # args.json 自动恢复 enable_global_audio_modulation / audio_sink_bias_mode)。
    # args.json 会被训练进程覆写(阶段进行中), 以内容 md5 判断是否需要重传。
    import hashlib

    def file_sig(p):
        return hashlib.md5(open(p, "rb").read()).hexdigest()[:10]

    aux_files = []
    for stage_dir in stage_dirs:
        stage_name = os.path.basename(stage_dir)
        for fname in ("args.json", "training_config.yaml"):
            p = os.path.join(stage_dir, fname)
            if os.path.exists(p):
                aux_files.append((stage_name, p))

    pending = [(s, p) for s, p in ckpts if f"{s}/{os.path.basename(p)}" not in done]
    aux_pending = []
    for s, p in aux_files:
        sig_key = f"{s}/{os.path.basename(p)}#sig={file_sig(p)}"
        if sig_key not in done:
            aux_pending.append((s, p, sig_key))
    log(f"扫描到 {len(ckpts)} 个 checkpoint, 待上传 {len(pending)} 个; 辅助文件待传 {len(aux_pending)} 个")
    if not pending and not aux_pending:
        return

    for stage_name, path, sig_key in aux_pending:
        key = f"{stage_name}/{os.path.basename(path)}"
        try:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"{REPO_PREFIX}/{stage_name}/{os.path.basename(path)}",
                repo_id=REPO_ID,
                repo_type="model",
                commit_message=f"upload aux {key}",
            )
            log(f"  成功 {key}")
            mark_done(sig_key)
        except Exception as e:
            log(f"  失败 {key}: {str(e)[:200]}")

    for stage_name, path in pending:
        fname = os.path.basename(path)
        key = f"{stage_name}/{fname}"
        size_gb = os.path.getsize(path) / 1e9
        t0 = time.time()
        log(f"上传 {key} ({size_gb:.1f}GB) ...")
        try:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"{REPO_PREFIX}/{stage_name}/{fname}",
                repo_id=REPO_ID,
                repo_type="model",
                commit_message=f"upload OmniDance fullSFT {key}",
            )
            dt = time.time() - t0
            speed = size_gb / dt * 1000 if dt > 0 else 0
            log(f"  成功 {key} ({dt/60:.1f}min, {speed:.0f}MB/s)")
            mark_done(key)
        except Exception as e:
            log(f"  失败 {key}: {str(e)[:200]}")

    failed = sum(1 for s, p in ckpts if f"{s}/{os.path.basename(p)}" not in load_done())
    log(f"本轮完成, 剩余待传(下轮重试): {failed} 个")


if __name__ == "__main__":
    main()
