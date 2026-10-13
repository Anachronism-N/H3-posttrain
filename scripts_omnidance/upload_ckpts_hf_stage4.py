#!/usr/bin/env python3
"""Upload stage4 (OSS_fullSFT_stage4_omni257k_b32_2ep) checkpoints to NZC415/m2d.

Standalone variant of upload_ckpts_hf.py: scans ONLY the stage4 dir with its own
done/log files, so the main script's done-records (out of sync with remote
cleanup) are not touched and nothing else gets re-uploaded.
Idempotent; skips files modified within 10 minutes (may be mid-write).
"""
import os
import time
import glob
import hashlib

from huggingface_hub import HfApi

TOKEN = "<HF_TOKEN_REDACTED>"
REPO_ID = "Minervaaa/AblationDance"
STAGE_DIR = "/apdcephfs_gy4/share_302533218/cedricnie/diff/models/train/OSS_fullSFT_stage4_omni257k_b32_2ep"
STAGE_NAME = os.path.basename(STAGE_DIR)
REPO_PREFIX = "OmniDance-fullSFT"
DONE_FILE = "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/hf_uploaded_stage4_abl.txt"
LOG = "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/hf_upload_stage4_abl.log"

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
    for path in sorted(glob.glob(f"{STAGE_DIR}/step-*.safetensors")):
        if now - os.path.getmtime(path) < 600:
            continue  # possibly mid-write
        ckpts.append(path)

    aux_files = []
    for fname in ("args.json", "training_config.yaml"):
        p = os.path.join(STAGE_DIR, fname)
        if os.path.exists(p):
            aux_files.append(p)

    pending = [p for p in ckpts if f"{STAGE_NAME}/{os.path.basename(p)}" not in done]
    aux_pending = []
    for p in aux_files:
        sig_key = f"{STAGE_NAME}/{os.path.basename(p)}#sig={file_sig(p)}"
        if sig_key not in done:
            aux_pending.append((p, sig_key))
    log(f"扫描到 {len(ckpts)} 个 stage4 checkpoint, 待上传 {len(pending)} 个; 辅助文件待传 {len(aux_pending)} 个")
    if not pending and not aux_pending:
        return

    for path, sig_key in aux_pending:
        key = f"{STAGE_NAME}/{os.path.basename(path)}"
        try:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"{REPO_PREFIX}/{STAGE_NAME}/{os.path.basename(path)}",
                repo_id=REPO_ID,
                repo_type="model",
                commit_message=f"upload aux {key}",
            )
            log(f"  成功 {key}")
            mark_done(sig_key)
        except Exception as e:
            log(f"  失败 {key}: {str(e)[:200]}")

    for path in pending:
        fname = os.path.basename(path)
        key = f"{STAGE_NAME}/{fname}"
        size_gb = os.path.getsize(path) / 1e9
        t0 = time.time()
        log(f"上传 {key} ({size_gb:.1f}GB) ...")
        try:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"{REPO_PREFIX}/{STAGE_NAME}/{fname}",
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
            if "403" in str(e) or "storage" in str(e).lower():
                log("配额不足, 停止后续上传")
                break

    failed = sum(1 for p in ckpts if f"{STAGE_NAME}/{os.path.basename(p)}" not in load_done())
    log(f"本轮完成, 剩余待传(下轮重试): {failed} 个")


if __name__ == "__main__":
    main()
