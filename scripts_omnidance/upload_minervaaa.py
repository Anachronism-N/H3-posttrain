#!/usr/bin/env python3
"""上传里程碑权重到 Minervaaa/m2d(该账号有配额,2026-09-08 验证)。
显式清单上传 + 等待 stage2-12000 出炉(最多等 2h)。
"""
import os
import time

from huggingface_hub import HfApi

TOKEN = "<HF_TOKEN_REDACTED>"
REPO = "Minervaaa/m2d"
PREFIX = "OmniDance-fullSFT"
M1 = "/apdcephfs_gy4/share_302533218/cedricnie/diff/models/train"
M2 = "/apdcephfs_gy4/share_302533218/cedricnie/diff-wl/models/train"

# (本地路径, 仓库相对路径, 是否等待出现)
FILES = [
    (f"{M2}/OSS_maskFT_from12k/step-3000.safetensors", "OSS_maskFT_from12k/step-3000.safetensors", False),
    (f"{M2}/OSS_maskFT_from12k/step-1500.safetensors", "OSS_maskFT_from12k/step-1500.safetensors", False),
    (f"{M2}/OSS_maskFT_from12k/args.json", "OSS_maskFT_from12k/args.json", False),
    (f"{M1}/warm2brch_fullSFT_stage2_fullDiT_lr1e5_warm500/step-6000.safetensors", "warm2brch_fullSFT_stage2_fullDiT_lr1e5_warm500/step-6000.safetensors", False),
    (f"{M1}/warm2brch_fullSFT_stage2_fullDiT_lr1e5_warm500/args.json", "warm2brch_fullSFT_stage2_fullDiT_lr1e5_warm500/args.json", False),
    (f"{M1}/warm2brch_fullSFT_stage1_aud5e5_xattn5e6_warm1500/args.json", "warm2brch_fullSFT_stage1_aud5e5_xattn5e6_warm1500/args.json", False),
    (f"{M1}/OSS_fullSFT_stage3_resume2_from8k/step-4000.safetensors", "OSS_fullSFT_stage3_resume2_from8k/step-4000.safetensors", False),
    (f"{M1}/warm2brch_fullSFT_stage2_fullDiT_lr1e5_warm500/step-12000.safetensors", "warm2brch_fullSFT_stage2_fullDiT_lr1e5_warm500/step-12000.safetensors", True),
]

api = HfApi(token=TOKEN)


def log(msg):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)


for local, rel, wait in FILES:
    # 等待文件出现(且 10 分钟内没被修改过 → 写完了)
    deadline = time.time() + 7200
    while True:
        if os.path.exists(local) and time.time() - os.path.getmtime(local) > 600:
            break
        if not wait:
            log(f"跳过(不存在): {rel}")
            local = None
            break
        if time.time() > deadline:
            log(f"等待超时: {rel}")
            local = None
            break
        log(f"等待文件: {rel}")
        time.sleep(300)
    if local is None:
        continue
    size_gb = os.path.getsize(local) / 1e9
    t0 = time.time()
    log(f"上传 {rel} ({size_gb:.1f}GB) ...")
    try:
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=f"{PREFIX}/{rel}",
            repo_id=REPO,
            commit_message=f"upload {rel}",
        )
        dt = time.time() - t0
        log(f"  成功 {rel} ({dt/60:.1f}min, {size_gb/dt*1000:.0f}MB/s)")
    except Exception as e:
        log(f"  失败 {rel}: {str(e)[:200]}")

log("全部完成")
