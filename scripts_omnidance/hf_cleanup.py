#!/usr/bin/env python3
"""清理 HF NZC415/m2d 的早期中间 checkpoint,腾出配额。
保留策略(本地全有底,可随时重传):
- S1/S2: 只留 step-6000(各删 11 个)
- S3 原始: 只留 step-5000(删 9 个)
- resume_from5k: 只留 step-3000(删 5 个)
- resume2: 全保留(最终基线段)
共删 36 个 ≈ 486GB
"""
import os
import sys
from huggingface_hub import HfApi

TOKEN = "<HF_TOKEN_REDACTED>"
REPO = "NZC415/m2d"
PREFIX = "OmniDance-fullSFT"
LOCAL_ROOT = "/apdcephfs_gy4/share_302533218/cedricnie/diff/models/train"
DONE_FILE = "/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/hf_uploaded.txt"

DELETE = []
for d, steps in [
    ("OSS_fullSFT_stage1_5k_text90_u10", range(500, 6000, 500)),
    ("OSS_fullSFT_stage2_5k_joint45_t45_u10", range(500, 6000, 500)),
    ("OSS_fullSFT_stage3_10k_joint30_a30_t30_u10", range(500, 5000, 500)),
    ("OSS_fullSFT_stage3_resume_from5k", range(500, 3000, 500)),
]:
    for s in steps:
        DELETE.append(f"{d}/step-{s}.safetensors")

api = HfApi(token=TOKEN)

# 安全校验:每个要删的远端文件,本地必须存在
missing = []
for rel in DELETE:
    if not os.path.exists(os.path.join(LOCAL_ROOT, rel)):
        missing.append(rel)
if missing:
    print("FATAL: 本地缺失,拒绝删除:", missing, file=sys.stderr)
    sys.exit(1)
print(f"校验通过: {len(DELETE)} 个文件本地全有底, 开始删除...")

for rel in DELETE:
    repo_path = f"{PREFIX}/{rel}"
    try:
        api.delete_file(repo_path, REPO, commit_message=f"cleanup: remove intermediate {rel}")
        print(f"已删 {repo_path}", flush=True)
    except Exception as e:
        print(f"删除失败 {repo_path}: {str(e)[:150]}", flush=True)

# 同步清理 done 记录,防止守护进程重传
done = [l.strip() for l in open(DONE_FILE) if l.strip()]
deleted_keys = {rel for rel in DELETE}
kept = [l for l in done if l not in deleted_keys]
removed = len(done) - len(kept)
with open(DONE_FILE, "w") as f:
    f.write("\n".join(kept) + ("\n" if kept else ""))
print(f"hf_uploaded.txt 同步清理: 移除 {removed} 条记录")
print("CLEANUP_DONE")
