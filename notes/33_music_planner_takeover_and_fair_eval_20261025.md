# 三音乐模型接管、音频边界修复与公平评测（2026-10-25）

## 目标与不可突破的边界

比较 MusicFlamingo、Qwen3-Omni-Captioner、MOSS-Music 的音乐到动作 caption 能力。每族仅使用原分配节点的 8 GPU；不占用完整精选标注、ROI 或独立 VLM 审核任务的其他节点。当前固定窗弱监督基准不等待新标注，不单独给某模型换数据。

冻结合同：train=10720、val=490、test=500；每源一条完整约 5.0625 秒音频，输出一份含 [0,2) 和 [2,5.0625) 两窗口的 caption。不是两个物理 2 秒音频训练样本。2 epoch、global batch32（8 rank×micro1×accum4）、670 optimizer 更新、21440 正式源曝光。初始更新 2 和新进程恢复更新 3 均计入正式预算。

配方保持 LoRA rank32/alpha64、语言 attention/MLP/MoE 与末四音频层 LoRA、音频投影可训，LR 1e-5/5e-5、clip1、seed42；BF16 载入后精确升 FP32 存储，冻结基座，算子 BF16、NLL FP32。DS 原生 torch_autocast，通信 FP32，bf16/fp16 开关关闭；原生 gradient checkpointing use_reentrant=True。Captioner 仍使用 audio-only 原生模板，无新增舞蹈 system/instruction，只有 assistant 正文与 EOS 计算 loss。

## v8 实际故障与证据边界

已通过真实保存更新 2、新进程恢复更新 3 的三阶段入口，不根据旧 v1–v7 失败推断 v8。此前 Flamingo、MOSS 和 Captioner 均进入过正式训练。

Flamingo 和 MOSS 完成更新 97（3104 曝光）后，在构造更新 98 的 rank0 第 2 个 microbatch 时失败。旧 adapter 在重采样前执行绝对 1 帧长度检查。实际源 WAV 为 mono PCM16、24kHz、121505 帧；目标长度对应 121500 帧。多 5 帧即 0.208333 毫秒，原文件 SHA 与训练 manifest 一致。soundfile 解码与 WAV header 一致，四个同更新样本中其他三个为 121500 帧。不是计算超时、短源、重采样舍入、已观察到的字节变动或模型 NaN。具体原始生产阶段尚未追溯，不无证据归因于某个 decoder。

两路已有 failure/exit1/resources_safe 回执，锁及租约恢复、占卡程序实际恢复；资源安全不等于训练完成。最后完整断点是更新 3，更新 97 只有日志，不可伪称可恢复到 97。更新 4–97 的 3008 曝光属于未保存、丢弃的旧实验资源开销。Captioner 的同更新风险由相同源排序/代码推导；每次必须读新日志才能认定实际失败，不能把风险预测当真实退出。

初始串行仅 header 审计因存储 I/O 推进缓慢而停止，未声称全量通过。替换为 max4 并发的只读 WAV header 审计，不 Path.resolve、不全音频内容 SHA、不写原数据。完整审计结果已存放实验 runtime/audio_header_audit_v9.json：全部 train10720/val490 header 已读取，但 +5 帧候选未通过。train 分布：10608 精确、105 条 -1 帧、2 条 +5 帧、1 条 +16 帧、3 条 -16 帧、1 条 -3312 帧。最后一条只有 118188 帧/24kHz=4.9245 秒；val484 精确、6 条 -1 帧。不能将该真短源当微量 overrun 放行。下一步追溯五条额外异常的原音视频，不能 pad/trim/跳样本或偷改旧 manifest。当前 normalized rows 的 original_source_window_mapping=unknown、audio_video_alignment_verified=false，恢复完整时间窗也需要来源证据。新 GPU 重跑尚未批准代码门禁，不以“审计结束”冒称“审计通过”。

## 已批准的新版本方案

用户批准修复后新冻结版本、独立输出，三模型从官方基座重新执行 2 → 新进程恢复 3 → 670。不迁移或篡改旧 fingerprint，不修改运行 source_v7、旧 config、旧 checkpoint 或来源清单。adapter SHA 变化必须形成新 fingerprint。

主源码候选修复只允许已观察到的 24kHz 完整源正向最多 5 帧 overrun；原负向缺帧检查和其他采样率不放宽。不 crop/pad、不重写音频或 SHA、不跳样本，保留完整原波形并按原生策略重采样。全数据 header 审计未完成时，不将这一候选宣称已覆盖全量。21 项三族纯 CPU 正反例已实际通过：原长度、正向 1/5 帧、阈值外、缺帧、非 24kHz；核全波形 resample 数值、输出长度、原 SHA 不变、静音等长、错 SHA 与物理窗口拒绝。训练 venv 无 pytest，因此这次真实回归由同环境独立 Python 正反例执行；pytest 文件已加对应测试，但未把无法启动的 pytest 宣称通过。

CPU 审计/回归不是 GPU 训练或质量验收。新启动里程碑必须三族各自真实完成 GPU 更新 2 保存、另一个进程恢复执行更新 3、完整 checkpoint 文件 SHA 与 finite/nonzero 梯度核验、再另一个进程恢复 3 进入正式训练。用 /proc PID、start ticks、argv 与 GPU rank 集合核身份；不得仅凭后台句柄、live.json 或目录存在认定成功。

结束仍在运行的旧 Captioner 已获得用户授权，但只按精确身份和既有管理器受控退出/占卡恢复流程，等待资源安全与 fresh holder/租约/flock 准入后启动，不强杀未知进程。空闲节点保持安全占卡。

## 公平评测与当前接续状态

每模型内部在更新 335/670 的完整 val490 NLL 选择 checkpoint，不跨 tokenizer 直接用 NLL 排名。先冻结实际选择凭据、选中 marker SHA、源版本与完整性，再一次最终 test500；test 不用于选择或调参。

统一 val32 panel：同 source IDs 与 donor mapping、seed42、greedy、cap2048，原音乐/不同源错配音乐/等长静音，保 raw、token IDs、EOS、truncation、复读和音乐依赖。变化幅度不是同步分数，NLL 下降或保存成功不是编舞通过。train/val 没有显式 Dance style 前缀或独立舞种字段；本轮是动作 caption 与音乐依赖弱基准，不能冒称舞种＋编舞联合任务已验收。

独立 fixed3 入口已实现与 CPU 审核中：三条历史固定音频/参考图不在本轮 split，明确独立 holdout，不塞 val/test，不复用旧 caption cache。保完整音频与 raw 原文，转为 H3 records。原 v7 协议不可静默用于新训练源版本，须另冻结一致协议。独立审核曾复现并要求修复 ZeRO shard 仅计数、best 仅锁路径而不锁选中 marker、源版本未绑定三个门禁；修订测试逐版本区分，不合并自测与独立验收。

正式评测需所用 checkpoint 文件全部内容 SHA 一次核验，以及当前官方基座权重内容 SHA 一次核验。历史 staging copy SHA 加当前 stat/metadata 不保证当前 weight bytes，不能互相替代。合法晚写 validation/request sidecar 单独 SHA，不冒称原训练 marker 已覆盖。

同一个冻结 pure-video H3、无 LoRA 混用、相同参考图/prompt 协议：704×1280、124 帧、24fps、50 步、seed42、CFG1、flow shift12；每模型仅固定 3 case。用户重点清晰、连续、有编排、动作幅度与音乐响应，不能只摆 pose。H3 本地权重与 GPU 准入仍须实查，不能猜路径或占其他会话节点。

## 用户批准排除异常源后的新共同合同

2026-10-25 用户明确允许少量问题数据直接从训练列表移除，不再让这几条源恢复阻挡推进。已创建独立 music_planner_fixed2s_baseline_20261025_clean_v1；原始基准及 WAV 不动。依据完成的严格旧音频门禁 header 审计排除 7 条（+5 帧两条、+16 帧一条、-16 帧三条、真实短源一条），而不是仅排除 +5 帧候选的五条剩余异常。保留严格音频门禁，不需要靠放宽容差接受这些源。原 candidate/source_v9 未运行快照不是新数据正式执行版本。

新 train=10713、val=490、test=500；保留训练行原字节与相对顺序，val/test 文件逐字节不变。每 epoch 的最后一个 global batch 消费 25 个真实源，不补入额外训练曝光。2 epoch 仍为 335×2=670 optimizer 更新，正式源曝光改为 21426。通过独立 CPU 提取真实 epoch_order/update_slices/validate_cursor 的两 epoch 仿真，核所有源每 epoch 恰好一次、尾批 25、最终 cursor(epoch2,position0,update670,exposure21426)。CPU 仿真不是 GPU 通过。

exclusion_receipt.json 保存父基准/审计 SHA、七条排除原因；training_contract.json 保存三模型共用合同。已再次核过滤字节、split ID/source_group/audio SHA 隔离。旧 manager 的10720固定count与旧 eval 基准门禁不适用新清单，必须在后续明确的新冻结执行版本同步更新，不能用旧队列或旧fingerprint硬resume。

暂停阶段仅创建与核验新数据合同并取消遗留巡检。用户随后恢复全目标，三模型 clean_v10 于 UTC 2026-10-24 20:59:19–20:59:20 启动：全部旧 v8 已真实在更新98输入检查失败、占卡安全恢复，未再强杀任务。新管理器严格固定 train10713/val490、source budget21426，完整分片同时要求 optim/model 的 rank0..7、mp00 精确集合；其余 pidfd/flock/租约/资源恢复逻辑保留。新 source_v10、manager 和三份配置独立冻结，不改旧快照或旧断点；严格旧长度门禁恢复，只有报错定位信息增强。

新评测门禁要求清洗排除回执内容SHA、父基准身份、训练行精确有序过滤及val/test SHA不变。现有eval CPU单测17通过、train CPU单测32通过/2跳过/3未运行；在N2真实原七源经三族_audio共21次严格拒绝，正常源完整输入通过。新管理器在各实际授权节点只读校验配置/来源SHA、fresh八卡holder与准入余量，另三种非法合同拒绝。CPU验证不表示GPU保存恢复或质量通过。

UTC21:00:46仍在初始GPU更新2保存。UTC21:16:43重新核验：三族均已有smoke.saved和smoke.approved，完整checkpoint与新进程恢复更新3通过，均进入full_training新进程；/proc start_ticks/argv与8GPU rank集合逐项相符，无新失败。该快照Flamingo85、MOSS53，Captioner正式进程刚启动，尚未打印更新4。UTC21:18:28 Flamingo已完成98，超越旧97失败点，但新训练顺序已随过滤改变，不能以更新数本身断言所有剩余数据都已消费。正式启动里程碑通过不等完整670更新、val选择或质量通过。当前上游源恢复调查不再阻挡实验，被排除条目不替换也不补零。

## 当前结论

未完成三模型新版本 GPU 启动验收、670 更新、最终文本评测或 H3 人评，因此没有最强模型结论。只把已执行的 CPU 诊断/回归与真实 v8 失败作为当前证据。后续真实 queue、日志、完整回执优先于本文快照。GitHub 提交仅包含本实验明确源码、测试及分析，不打包其他会话脏工作区、权重、私有音频/图像或凭据。
