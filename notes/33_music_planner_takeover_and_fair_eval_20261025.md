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

## 训练后单卡评测接续与renderer准备

新增独立 music_planner_eval_manager.py，最终源码SHA58dea033347a84ceda239f6ff67c9a90c692056e2add313c2bb604f2ca1ddd59；主会话独立重跑29项CPU测试并在运行前后核源码及测试SHA一致。未运行的并发修改候选不作为dispatch版本。实际部署准备为独立runtime/planner_eval_manager_v2.py，不改正在运行的source_v10或训练manager。该管理器要求本族真实670更新/21426曝光/2epoch完整回执、资源安全、实际epoch335/670全val490内部NLL最小checkpoint及预冻结best/markerSHA；test还要求三族共同final_selection及永久O_EXCL每族一次claim。GPU0采用精确pidfd/flock/lease与恢复holder流程，其他七卡不操作。正式评测前一次当前官方base内容SHA/LFS/gitblob审计及完整checkpoint文件内容审计；CPU tests不等真实30B单卡fit或GPU输出通过。

val32/test的EOS、budget truncation、unknown stop和空raw输出是保留的模型诊断，不把弱模型复读当技术接口失败，也不因截断偷偷改变cap或重测test；native输出字段不一致、超预算、缺失真实接口证据等仍拒绝技术采集完成。fixed3有效H3 handoff保留严格完整停止及caption schema门槛。所有技术完成回执仍quality_accepted/human_quality_accepted=false。

H3准备原worker恢复后未重新注册，不冒称继续运行；主会话只读核三节点既有相同H3环境版本和Ref2VA的13个DiT、14个text、视频VAE和processor存在。独立h3_renderer_v10冻结同h3_video_infer/h3_video_only、288个当前DiffSynth Python源码SHA及三节点环境版本凭据；N2空CUDA执行tinyH3 wrapper自测、7renderer CPU tests通过。未读取H3大权重内容或执行预训练GPU渲染。实际渲染前仍核模型内容身份、源码/环境未漂移与GPU安全准入；空闲卡由既有占卡保持。

## clean_v10 首轮val后保存新故障

UTC2026-10-24 21:57:11左右 MusicFlamingo 在epoch335验证函数返回后的save_checkpoint→trainable_state GatheredParameters退出时触发DeepSpeed0.17.5 Cannot partition a param in flight。最后打印update334/10688曝光；checkpoint335仅空目录，没有complete marker、validation或best，不拿函数调用顺序替代丢失的val收据。最后完整断点为300。既有manager已恢复8holder并写resources_safe，旧snapshot/config/失败目录保留，不硬resume新trainer指纹。

这是新的验证与ZeRO参数生命周期问题，不是之前音频长度、OOM或模型质量验收。正在读实际DS源码并做最小根因修复；不能手填ds_status、删hooks、关闭checkpoint或绕过断言。UTC22:00:51重新核MOSS仍275、Captioner在100保存（最后打印99），两者实际训练进程存活，尚无新failure；不能将Flamingo失败直接归为其他族已失败。自动完成后评测协调器只完成代码与CPU门禁验证，当前旧clean_v10完整回执缺失，不运行评测调度。

## v11 完整预算验收与评测真实进展

原生ZeRO生命周期修复已经完成，trainer9583342d51c3f9b08a1c07f57958c102313364e24b9ed92620e49901fc20fae3；主会话独立N2真实DS0.17.5 CPU45项全过，源码与测试SHA前后相同，临时pytest工具自动清理且未安装训练env。GPU smoke now includes update2/fullval490/save2以及新进程update3/fullval490/save3，诊断val不加入训练曝光或best选择。v11新源和manager独立冻结，旧v10失败与300断点不改，不迁移旧fingerprint。

2026-10-25 UTC03:44实查：三模型均有full.complete与resources_safe，实际670更新、2epoch、21426曝光、完整670断点及恢复证据。模型内部完整val490选择均为epoch670；NLL仅族内选择，不能跨族排名。MusicFlamingo的val32原音乐/错配/静音共96真实GPU输出已技术完成，均EOS、无接口失败、无截断或空输出，但不能据此宣称编舞或音乐同步效果通过。

MOSS首次面板attempt在worker启动前被内存安全准入拒绝，资源已恢复，协调器正确停止派发，旧failure/claim保留。对结束的自有MOSS完整断点进行只读DONTNEED缓存建议，没有改文件/权重或杀其他任务；未将无明显即时缓存下降宣称修复。后续fresh headroom约217GB超过固定160GiB准入阈值，新独立manual_recovery_v1记录失败诊断与授权范围，同数据/weights/解码协议重新captureGPU0生成新attempt，仍保内存门禁。正式test尚无claim或生成，不重测test。

三模型真实选择在eval_handoff_clean_v11/final_selection.json提前冻结，含各实际best/marker/fingerprint/训练完成与资源安全SHA、统一数据和greedyseed42/cap2048解码协议。新任务Flamingofixed3、Captionervalpanel、MOSSvalpanel已按对应原节点GPU0受控启动，当前审计所用checkpoint全部state文件和本地officialbase完整内容，未以长审计或未创建queue误判失败；其他7卡占卡保持。fixed3完整handoff、test500及H3三case人评尚未完成，没有赢家结论。

## 最终推理兼容修复与严格结果边界

MOSS在恢复准入后首次真实GPU96条均0token/142与141音频mask尺寸错误；根因是Transformers5.18不再维护cache_position，而官方MOSS prepare只靠该字段判定cached decode，未利用实际DynamicCache141，第二步重复处理全部prefix142与audio mask141。独立eval-only补丁仅用真实KV长度恢复原生单token分支，prefill和DeepStack原64音频位置不变，不padding、不禁cache、不改本地官方模型或训练指纹。主会话独立N7官方tiny9CPU回归通过；新v4manager保原29资源门禁并额外核每条native patch ref和独立sidecar。实际新GPU面板已产首条2048预算截断、无接口失败：截断是保留的模型诊断，尚未说明全部96完成或质量通过。

Captioner旧val第一条2048tokens中im_end在第246号位置已有，随后重复595次im_start/im_end/newline。native GenerationConfig EOS=None，而adapter只在诊断时fallback tokenizer EOS151645，未传给实际generate。独立EOS补丁仅在native EOS缺失时给generate显式原训练助手终止ID，首EOS前logits/音频prefix/参数配置不变，既有有效EOS模型不打patch。主会话独立N4官方tiny9CPU回归通过；旧val经精确manager pidfd停止且占卡安全恢复，未裁剪或改写旧raw。新v5manager同29资源门禁与EOS provenance/first-end停止校验通过，新的实际CaptionerGPU面板已派发，仍须真实回执确认修复有效。

以上是推理兼容修复，不是增加训练或用test调参。三个最终checkpoint/数据/greedyseed42/cap2048固定。Flamingo先前已有效EOS，无需补丁；其首次最终test500已开始生成，不重测。MOSS/Captioner尚未开始test；为使用其原节点闲置资源，计划按500固定ID全局顺序分为8个不重叠GPU子集，各source仅一次，同native协议与patch身份，聚合恢复原顺序。模型CPU加载串行、生成并行，八卡pidfd/flock/lease全部准入再释放，不能改变test once ledger或对失败partial test自动重测。当前该并行实现仍准备中，不冒称已启动或通过。

H3单GPU0安全包装已独立7组CPU验证并冻结；必须真实三族fixed3有效raw交接、same参考图/renderer权重全SHA、50步seed42CFG1flow12/704×1280/124f24fps/noLoRA实际视频完成后，才能让用户盲评。当前尚无三族H3视频完成或赢家结论。所有旧失败记录保留，另七卡空闲时由原占卡程序保持。

## 文本优先门槛及MOSS非persistent频率加载根因

用户最新要求先卡文本，后续H3 GPU派发暂缓。基础val门槛按三条件每个32源固定：96完整采集、接口失败和空输出0，正常assistant EOS和正确双时间窗≥95%，同一规范化句子至少重复3次的caption比例≤5%。原音乐完全重复、4gram/句内复读、动作/身体部位覆盖及逐case错配/静音变化另作诊断，不把变化量当音乐同步分数；静音输入相同的重复输出不作为多样性失败。当前没有独立DanceStyle GT，不虚报舞种准确率。这是用户请求的val工程门槛，不是用test500调阈值，也不冒称语义编舞验收。

text_gate对实际冻结donor映射、native token/stop和早EOS继续生成、MOSS每条真实finite-forward证据校验；状态complete还要求原collector eval.complete、manifest/raw SHA一致、exit0与resources_safe。8组纯JSON回归通过。完整Flamingo与修复后MOSS各96均通过基础文本门槛，Captioner仍在生成，不依据partial判断胜者。

MOSS缓存修复后的旧异常输出全部token0，实际N7 GPU进一步诊断发现audio_encoder.embed_positions.inv_timescales640在普通HF5.18 meta加载后有垃圾/NaN；nonpersistentbuffer被empty_like而未重建，官方checkpoint同key报告UNEXPECTED。真实CPU官方HF FP32/BF16均复现坏频率，而真实HfDeepSpeedConfig zero3加载保留native ctor FP32 canonical，支持训练初始化正确、推理meta路径损坏的区别。独立11CPU测试通过；仅640官方BF16小tensor与native FP32 canonical.round(BF16)逐元素完全等价，分别如实记录dtype和SHA。新补丁只恢复native nonpersistent位置频率，不改任何已训练参数、其他buffer、配置、checkpoint或数据；生成作用域安装只读finite logits验证，NaN在argmax前拒绝。

同完整MOSS670 checkpoint实际GPU恢复后prefix151936 logits全finite，3token正常为[58,15,13]，teacher-forced单源诊断NLL0.7198，未用作选择或质量评分；新的完整96面板均EOS、无接口失败/空输出/截断，全部原频率及finiteforward回执核验。这些证明加载/生成接口纠正，不是重训或额外源曝光，也不证明舞种准确。所有旧失败/异常raw保留且不用于模型排名。

MOSS固定3条原生caption已完成EOS与资源安全；首次最终test500八卡已受控启动，global ID按index%8分片、不重复；加载模型串行、GPU生成并行。并行源码独立24CPU门禁通过（22分片/finite/API安全+2lease心跳线程），在长内容审核时持续刷新已拥有租约，cleanup前停止并join防FD重用竞态，不绕任何检查。Flamingo已开始的单卡首次test保持，不重测；Captioner待本族完整文本前提后再首次八卡test。当前都尚未证明最终500全部完成，H3与人评尚未执行，没有赢家。

## 当前结论

未完成三模型新版本 GPU 启动验收、670 更新、最终文本评测或 H3 人评，因此没有最强模型结论。只把已执行的 CPU 诊断/回归与真实 v8 失败作为当前证据。后续真实 queue、日志、完整回执优先于本文快照。GitHub 提交仅包含本实验明确源码、测试及分析，不打包其他会话脏工作区、权重、私有音频/图像或凭据。
