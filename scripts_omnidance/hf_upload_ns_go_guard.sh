#!/bin/bash
# 每 30 分钟跑一次 upload_ckpts_hf_ns_go.py, 把 noShare/global-off 新产出的
# checkpoint 持续同步到 https://huggingface.co/Minervaaa/AblationDance。
# 幂等(靠 hf_uploaded_ns_go.txt 记录), 可以随时 kill 掉这个循环不影响已上传内容。
cd /apdcephfs_gy4/share_302533218/cedricnie/diff || exit 1
while true; do
  python3 scripts_omnidance/upload_ckpts_hf_ns_go.py
  sleep 1800
done
