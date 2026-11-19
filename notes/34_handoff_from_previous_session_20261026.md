# 草稿产线交接（前一会话 → 接管会话），2026-10-26 11:30Z

ROOT=/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1
R=ROOT/runtime，B=ROOT（下文中 B/... 与 ROOT/... 相同）

## 0. 状态
- 前一会话的 cron a559ef04 已删除，CronList 返回 "No scheduled jobs"。旧 cron e923d14c 之前已删除。
- 从 11:30Z 起，前一会话不再启动、停止、park 或交接任何东西。所有生产进程保持原样。
- 输出全部是**未审核草稿**（training_eligible/semantic_approval/quality_approved=False，needs_visual_revision=True）；stage2 统一 RGB/ROI 修订放在后面做；全量 222,939 条**未完成**。
- 修复数据收集任务 MzRpLm（collector_offset2881725_reconciled_v1.py）停在 B/curated222939_production_v1 源文件的坏行 offset 4579239 处（UnrecoverableLine，cursor 保留在 4579239）。前一会话没有处理，由接管会话决定，不要自动跳过。

## 1. 正在跑的进程（11:2xZ 只读 /proc 快照；"mgr"=原生 resource_manager 父进程，"worker"=它的直接子进程）

### 本草稿产线（前一会话负责的部分）
| 节点 IP | 角色 | mgr PID | worker PID | 入口脚本 | config | GPU |
|---|---|---|---|---|---|---|
| N2 28.216.17.70 | pool2 worker | 4112053 | 4112153 | R/caption_pool_live_manager_v2.py | R/caption_two_stage_local_pool2_n2g23_v1_launch.json | 2,3 |
| N2 | pool2 worker | 4112614 | 4112855 | 同上 | R/caption_two_stage_local_pool2_n2g45_v1_launch.json | 4,5 |
| N2 | pool2 worker | 4113508 | 4113703 | 同上 | R/caption_two_stage_local_pool2_n2g67_v1_launch.json | 6,7 |
| N2 | 缓存 producer（CwMUPh，collab3 shard0/4） | 4050281 | 4050319 | R/caption_producer_cached_manager_v1.py / caption_producer_cached_worker_v1.py | R/caption_producer_cached_n2p0_v1_launch.json | 0 |
| N2 | 原生 collab3 producer n2p1（shard1/4） | 3468653 | 3469128 | R/curated_collaborative3_resource_manager_v1.py / curated_collaborative3_producer_v1.py --member n2p1 | R/curated_collaborative3_n2p1_v1_launch.json | 1 |
| N3 28.216.19.213 | pool1 worker | 2301931 | 2302081 | R/caption_pool_live_manager_v1.py | R/caption_two_stage_local_pool_n3g01_v1_launch.json | 0,1 |
| N4 28.216.19.143 | pool1 worker | 1714278 | 1714566 | R/caption_pool_live_manager_v1.py | R/caption_two_stage_local_pool_n4g23_v1_launch.json | 2,3 |
| N4 | pool1 worker | 1714279 | 1714564 | 同上 | R/caption_two_stage_local_pool_n4g45_v1_launch.json | 4,5 |
| N4 | pool1 worker | 1714273 | 1714565 | 同上 | R/caption_two_stage_local_pool_n4g67_v1_launch.json | 6,7 |
| N4 | n4c3_balanced | 1573551 | 1573656 | R/caption_local_live_manager_v1.py | R/caption_two_stage_local_n4c3_balanced_v1_launch.json | 0,1 |
| N6 28.216.19.225 | n6_balanced | 1627994 | 1628091 | R/caption_local_live_manager_v1.py | R/caption_two_stage_local_n6_balanced_v1_launch.json | 2,3 |
| N5 28.216.19.137 | live_n5c3_v3 | 1364828 | 1364942 | R/caption_two_stage_live_manager_v3.py | R/caption_two_stage_live_n5c3_v3_launch.json | 4,5 |
| N7 28.216.18.144 | 原生 collab3 consumer n7c7（shard7/9，旧32B候选，慢） | 3499874 | 3499975 | R/curated_collaborative3_resource_manager_v1.py / curated_collaborative3_consumer_v1.py | R/curated_collaborative3_n7c7_v1_launch.json | 2,3 |
| N7 | 原生 collab3 consumer n7c8（shard8/9） | 3500413 | 3501473 | 同上 | R/curated_collaborative3_n7c8_v1_launch.json | 4,5 |
| N7 | 原生 collab3 producer n7p2（shard2/4） | 3499869 | 3499960 | 同上 / curated_collaborative3_producer_v1.py | R/curated_collaborative3_n7p2_v1_launch.json | 0 |
| N7 | 原生 collab3 producer n7p3（shard3/4） | 3499880 | 3499949 | 同上 | R/curated_collaborative3_n7p3_v1_launch.json | 1 |

- 所有 mgr 的 argv 形如：python -B [-u] <manager.py> --config <cfg> --config-sha <cfg sha> --manager-sha 6bd8a6c5...（原始 resource_manager.py，160GiB 门）。
- N1 上的监控 caption_two_stage_live_monitor_v1.py（PID 2769928，即 monitor sZxrGE）还在跑，它**没有统计 pool2**。

### 同集群里其他会话/任务的进程（前一会话不负责，没碰过，不要当成本产线的空卡）
- N1：collab4 consumer n1c0/n1c1（GPU4-7），consumer cache 和 prepare-ahead（GPU0-3），以及 review/overlay/salvage 服务。
- N3：collaborative21000 n3g67（GPU6,7），full12k visual s0/s1（GPU2-5），coarse6when_safe_wait。
- N5：collab4 n5c2（GPU2,3）/ n5p0（GPU0），collaborative21000 n5g1 resume（curated_producer_io_v3，GPU1），full12k_roi（GPU6），main_review_watch_v9。N5 GPU7 是 holder。
- N6：full12k visual s2/s3（GPU4-7），curated_coarse_fixed3_primary_restore_fresh 加 curated_repair_watch_v2（GPU0,1）。
- N7：curated_collab3_8b_review_manager n7g6/n7g7。GPU6、7 上只有 hold_gpu（约 975MiB，利用率 100%，这是占卡进程，不是在生产）。
- N8：curated_producer p0-p3（GPU0-3），curated_consumer_s1（GPU6,7），prepare_ahead n8s0（GPU4,5）。

## 2. 动态池

### pool1（4 个 worker：N3 g01、N4 g23/g45/g67）
- pool_root：B/caption_two_stage_dynamic_pool_v1
- manifest：R/caption_pool_transition_v1_manifest.json；plan：R/caption_pool_transition_v1_remaining.json（N=5710）；handoff：R/caption_pool_transition_v1_handoff.json；转移代码：R/caption_pool_handoff_v1.py
- 11:29Z：terminals=940，claims=943，plan 中已 ready 的 943 条已全部 claim，**ready_unclaimed=0**，failed_batches=0。它在**等 producer 供料**，近期约 166–258 条/h。

### pool2（3 个 worker：N2 g23/g45/g67），10:07Z 上线
- pool_root：B/caption_two_stage_dynamic_pool_v2
- manifest：R/caption_pool2_consumer_transition_v1_manifest.json；plan：R/caption_pool2_consumer_transition_v1_remaining.json（N=7944）；handoff：R/caption_pool2_consumer_transition_v1_handoff.json
- 代码：R/caption_pool2_consumer_transition_v1 由 R/caption_pool_consumer_handoff_v1.py（SHA 37f65c65…）生成；manager 是 R/caption_pool_live_manager_v2.py（SHA c7c45fb7…，相对 v1 只在 HOSTS 里加了 N2/N7）；测试 R/test_caption_pool_consumer_handoff_v1.py 17 项通过。
- 来源：N2 上三个旧 collab3 consumer（curated_collaborative3_n2c0_v1、n2c1_recovery_v1、n2c2_v1；shard0/1/2）已原生 park 并 safe。原 union 9999 条，已开始或已终态的 2055 条排除，不重派。旧 candidate、cache、claims 都保留，旧 cfg 禁止恢复。
- 11:29Z：terminals=909，claims=927，plan 中已 ready 2585 条，**ready_unclaimed 约 1658**，failed_batches=0。速度约 564–774 条/h（各 10–20 分钟窗口）。按这个速度积压约 2–3 小时消化完，之后也会等料。
- 进度计数规则：pool 的全局进度只计一次（terminals 目录计数）。worker 自己的 progress.json 是 worker 范围，不要和全局相加。

## 3. 其他产线
- n4c3_balanced：draft_out=B/caption_two_stage_local_n4c3_balanced_v1，progress.json 11:29Z 显示 processed 609/2396（generated 606，partial 3），约 50–96 条/h。
- n6_balanced：draft_out=B/caption_two_stage_local_n6_balanced_v1，329/925，约 18–24 条/h。
- live_n5c3_v3：cfg 里 draft_out=**B/caption_two_stage_live_n5c3_v1**（v1、v3 共用这个目录），out=B/caption_two_stage_live_n5c3_v3_execution，queue=B/queues/caption_two_stage_live_n5c3_v3，targetcount=4586。drafts/<split>/ 目录数 11:29Z 是 374（约 25–36 条/h）。
- N2 缓存 producer CwMUPh（caption_producer_cached_n2p0_v1，collab3 shard0/4）：out=B/caption_producer_cached_n2p0_v1，queue=B/queues/caption_producer_cached_n2p0_v1。它写 B/curated222939_collaborative_v3/ready/producer-0000.jsonl。15 分钟实测 215.6 条/h，旧路径约 273 条/h，**未证明提速，不要推广**。
- N7 shard 7/8（n7c7/n7c8）：原样运行，没有停。两者 queue 里都有 low_memory_warning.json。consumers/shard-0007/completed.jsonl 有 762 行，shard-0008 有 718 行，各约 40 条/h，积压约 670–700 条 ready 未开始。N7 headroom 一直在 155.8–157.1GiB，低于 160GiB，所以没迁移。迁移规则：N7 fresh headroom ≥160GiB 且 /tmp ≥8GiB 时才可另建一个**新 group** 迁移，**不能**复用已 claim 的 pool2 group（caption_pool2_consumer_transition_v1）。
- collab3 ready 供料（B/curated222939_collaborative_v3/ready/producer-000{0..3}.jsonl）11:29Z 共 13,921 行（各 3696/3731/3252/3242）。08:45 约 11,872，**供料约 740–850 条/h**。4 个 producer 各 7500 条，大约走到 45%，还剩约 16k 条。

## 4. 未完成的交接/启动操作（停在安全点）
- **"N2 producer 并行化"只写了代码草稿，没执行任何一步**：
  - 已写文件：R/caption_producer_parallel_handoff_v1.py、R/caption_producer_parallel_worker_v1.py（第二个可能是写到一半时被打断的版本，未验证）。**没有**写 manager（caption_producer_parallel_manager_v1.py 不存在），没有测试，没有运行过 --check-only。
  - **没有**写任何 claim、remaining、launch 或 handoff 产物，B/caption_producer_parallel_n2p1_v1 和对应 queue 都不存在，n2p1 原生 producer 没收到任何信号。
  - 思路：n2p1 的 producer 单进程 CPU 只用 9–16%，每源 13–24 秒，基本在等 Ceph IO。计划把剩余来源按轮转分成 K=3 份，同一张 GPU 上跑 3 个原生 IOv4 子进程，ready 用原生 flock 追加。这个思路**未经测试**，接管会话可以采纳，也可以删除这两个草稿文件。
- 除此之外没有其他半途中的交接。pool2 交接的 claim、parked、prepared、validated 都完整，3 个 worker 已在跑。

## 5. 测速脚本
- 运行 \`/usr/bin/python3 /tmp/p2_rate.py\`（N1 本地 /tmp，不在共享盘；如果丢了，按下面说明重写）。输出一行 JSON：t（unix 秒），pool1/pool2 的 terminals 目录计数，n4c3/n6 的 progress.json processed，live_n5c3_v3 的 drafts 目录数。隔一段时间跑两次，差值除以时间就是每条线的速度。
- \`/usr/bin/python3 /tmp/p2_ready.py\`：每个 pool 的 N、plan 内已 ready 数、claims 数，ready_unclaimed≈ready−claims，以及失败批次数。
- n5c3_v3 路径的问题：脚本已改成读 B/caption_two_stage_live_n5c3_v1/drafts（cfg 的 draft_out），但这个目录里**混有 v1、v2 时期的历史草稿**。目前计数 374 是累计值，不是 v3 本身的产量，只能用差值算速度。v3 的精确产量应看 B/caption_two_stage_live_n5c3_v3_execution 或 queue 日志。live_n5c3_v3 下面没有 progress.json。

## 6. 没写进代码的约束和坑
- 只做三类检查：输入可读、形状和时间有序；ID/split 隔离、失败明确记录；write-once、预算不重派、GPU 锁/租约/资源安全。不要重复对媒体或权重做 SHA、不要重复 probe、不要做全 AST 校验。
- 不 kill 未知进程，不删锁（/tmp/ms12k_gpu_N.lock）、租约（/tmp/od_reserve_N）、旧结果或 claims；不降 160GiB 门；claimed/failed 不重派；旧 cfg 不恢复；每个 group 只能执行一次，失败会写 *_failed.json 封存，不自动重试。
- 原生 park 之后旧 manager 停在 h.wait()（state S、wchan=do_wait）是**正常**的，不要要求它退出。safe 判定只看 resources_safe.json 的三项和 holder_restored_gpuN.json。
- 同一个 Ceph 文件在不同节点的 st_dev 不同，跨节点做 proof 比对要去掉 device 字段（rebalance.check_shared_files）。
- process identity 的键包含 'exe'（resource_manager.IDENTITY_KEYS）。
- 原 resource_manager 的 AUTHORIZED_PAIRS 不含 N2/N4/N7；各新 manager 用 FunctionType 克隆 validate/gate 自带授权，不改原文件。
- pool 读 ready 时偶尔遇到 FUSE 瞬时读到坏 JSON，caption_ready_pool_v1 在异常路径会重读一次；旧 N3 本地 lane 曾因此退出过。
- pool 的 worker 计数和全局计数分开；新草稿 edition、旧 candidate edition 和历史保留的 2433 条终态**不要相加**。
- N2 缓存 producer（215.6/h）和并发 ROI 发布（零模型 3 源 1.40x，结果一致）都已测过，**不推广**，因为当前瓶颈不在这里。
- 当前真正的瓶颈是 **producer 供料**（4 个 collab3 producer 合计约 740–850 条/h，每个单进程串行处理来源，主要在等 IO）。pool1 已在等料，pool2 约 2–3 小时后也会等料。
- 用 Read 工具读含控制字符的 launch/live JSON 会报 "Bad control character"，改用 python json.loads。在 Ceph 大目录上用 Glob 容易超时。
- GitHub 推送 be62c90 仍卡在认证，不要索取或打印凭据。
- 集群的空卡判断：hold_gpu.py（约 1GB 显存、利用率 100%）是占卡进程，不是生产；producer（约 1.4GB、利用率约 0%）是在工作的进程，不是空卡。
