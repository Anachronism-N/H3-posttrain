#!/usr/bin/env python3
"""Incrementally upload noShare / global-off ablation checkpoints to NZC415/m2d.

沿用 upload_ckpts_hf.py 的同一套路径约定(仓库 NZC415/m2d, 前缀 OmniDance-fullSFT,
按 stage 目录名分子目录),只是新增扫描 .69(noShare stage2/3)和 .70(global-off
stage2/3)这两条链的产出目录。用独立的 done/log 文件,不影响原脚本的记录。
Safe to re-run periodically (idempotent, 会跳过近10分钟内修改过的文件避免传半成品)。
"""
import os
import time
import glob
import hashlib

from huggingface_hub import HfApi

TOKEN = "<HF_TOKEN_REDACTED>"
REPO_ID = "Minervaaa/AblationDance"
MODELS_ROOT = "/apdcephfs_gy4/share_302533218/cedricnie/diff/models/train"
REPO_PREFIX = "OmniDance-fullSFT"
DONE_FILE = "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/hf_uploaded_ns_go.txt"
LOG = "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/hf_upload_ns_go.log"

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


def file_sig(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:10]


def main():
    done = load_done()
    now = time.time()
    ckpts = []
    scan_roots = [
        f"{MODELS_ROOT}/noShare_fullSFT_stage2_5k_joint45_t45_u10",
        f"{MODELS_ROOT}/noShare_fullSFT_stage3_7k_joint30_a30_t30_u10",
        f"{MODELS_ROOT}/OSS_globalOff_fullSFT_stage2_2k_joint45_t45_u10",
        f"{MODELS_ROOT}/OSS_globalOff_fullSFT_stage3_12k_joint30_a30_t30_u10_lr2x",
        f"{MODELS_ROOT}/OSS_standardAttn_fullSFT_stage2_6k_joint45_t45_u10",
        f"{MODELS_ROOT}/OSS_standardAttn_fullSFT_stage3_7k_joint30_a30_t30_u10",
    ]
    stage_dirs = [d for d in scan_roots if os.path.isdir(d)]

    for stage_dir in stage_dirs:
        stage_name = os.path.basename(stage_dir)
        for path in sorted(glob.glob(f"{stage_dir}/step-*.safetensors")):
            if now - os.path.getmtime(path) < 600:
                continue
            ckpts.append((stage_name, path))

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
                commit_message=f"upload OmniDance ablation {key}",
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
