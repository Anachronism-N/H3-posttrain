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

## 完整val文本门槛与匿名语义审核（文本优先阶段）

三族val32各原音乐/错配/静音共96真实输出及collector/exit0/resources_safe已全验收；基础EOS、双时间窗三条件均100%，严重同句至少3次caption0%，接口与空输出0。基础门槛全部通过不等语义编排/音乐同步/舞种准确，DanceStyle独立GT仍缺，未用test选择或调门槛。

原音乐句内重复均值：Flamingo2.07%、MOSS3.73%、Captioner6.72%；4gram重复均值12.59%/9.75%/13.59%。原音乐与错配、静音逐case文字全部有变化，证明基本音频依赖而不是正确音乐同步。MOSS val32有一个整条caption精确复用组、第一窗五case共用组及case18两窗正文相同；这些没有被粗基础门槛伪装成语义多样性通过。

独立LLM在不知道模型名、NLL、test/音视频的条件下完整读匿名A/B/C各32原音乐caption，四轴ordinal意见（连续承接/动态动作/语义动作多样性/窗口可执行细节）为A3/3/2/3、B4/3/3/3、C3/3/2/3；审后解码A=Flamingo、B=Captioner、C=MOSS。main独立核实际payload证实所有精确重复组、跨窗同正文case；没有把缺少过渡自动判为物理矛盾，未以弱GT不同扣分。盲审text-only中低置信意见为Captioner优先，Flamingo与MOSS粗评分接近；这不是校准accuracy、音乐或H3人评，也没有最终winner。收到的实际审阅已保存总结与原匿名材料SHA，不伪称保存未提取成功的原长报告。

同一首次最终test500：Flamingo与MOSS均500unique ID/全EOS/双窗100%/接口错误0/实际资源安全；MOSS8卡111281原生forward全部finite且计数一致、global顺序还原。跨case整段完全重复比例Flamingo10.8%、MOSS2.4%；句内重复2.70%与4.64%，不可仅凭一个重复指标判最强。先前偏MOSS是该重复诊断的暂时倾向，更多完整盲审文本证据后暂优先Captioner；Captioner最终test尚未开始，不能宣布三模型最终排名。

Captioner完整val后，唯一受控继续器已派发fixed3文本任务；还在全checkpoint字节审计，实际/proc可见读取第5个optim shard。空stdout/尚无queue不等任务停止，不重复启动。H3权重N2已做CPU完整scope SHA预审，尚未GPU渲染；待实际3族fixed3文本完成后，再按原一致协议做人评。

## 用户新增F1与参考一致性指标

已补标准词级multiset P/R/F1（ROUGE-1）、连续2gram ROUGE-2及sequence LCS ROUGE-L的P/R/F1，去格式时间标签后以固定词/CJK字符规则计算；按两窗口还计算词级F1、动作类别和身体部位集合P/R/F1 macro与micro。8项算术/空集合/多次词/窗口调换/LCS/宏微平均回归通过，唯一初次浮点测试容差已修正而公式与分数未改。指标是同source的一个弱动作参考的一致程度，不是创作编排唯一正确率，也没有独立舞种GT可报告style-F1。

完整val32原音乐macro：Flamingo/MOSS/Captioner的ROUGE-1 F1=0.501/0.485/0.506，ROUGE-2=0.186/0.167/0.192，ROUGE-L=0.262/0.253/0.264，窗口动作类别F1=0.616/0.609/0.625，窗口身体F1=0.701/0.719/0.725。Captioner数值略高但差距小；32配对source的5000次bootstrap(seed42)对ROUGE-1/L及窗口动作/身体F1差值的95%区间均包括0，不能宣称显著胜出。词匹配/类别集合不能区分所有动作顺序或音乐同步，正反例也证实整段词F1为1时调换两窗口动作仍可能窗口F1为0。

错配/静音仍匹配该case的原source弱参考，分数只作为参考匹配与音频依赖诊断，不用于test调阈值或另选checkpoint。全三族val门槛仍沿原95%/5%工程规则，没有按这些新F1结果追调门槛。F1/ROUGE报告与配对不确定性已保存，Captioner最终test在执行前冻结同原checkpoint与协议，不依据新分数改模型。H3同源权重已CPU完整哈希，真实渲染仍需独立实际GPU及人评。

## 2026-10-25 反事实指标与接续实查

下表取已保存完整val32弱参考报告的窗口动作类别macro F1，三条件始终匹配同case原source参考，未交换错配参考：

| 模型 | 原音乐 | 错配音乐 | 静音 |
|---|---:|---:|---:|
| Flamingo | 0.6161 | 0.6500 | 0.5947 |
| MOSS | 0.6085 | 0.6254 | 0.6155 |
| Captioner | 0.6252 | 0.5831 | 0.5061 |

原音乐条件下Captioner在动作类别、身体部位和ROUGE指标均优于本模型的错配/静音条件；Flamingo/MOSS的动作类别F1没有保持这一规律。这比“换音乐文字变化”更有信息，但单个弱参考和规则类别覆盖仍有局限，不能直接证明节拍同步、因果音乐理解或端到端优胜，也不把该探索性诊断改成新验收阈值。已有跨模型配对95%区间只覆盖ROUGE-1/L及窗口动作/身体F1，均含0；未把没有计算的ROUGE-2差值区间说成已检验。

截至07:38 UTC只读实查，Captioner首次test500的manager和8个GPU worker真实存活，GPU0–7为本任务，尚无complete/exit/resources_safe，不重复派发。三族fixed3原生文本已全部完成并安全，H3 v3 capture核N2 GPU0 holder PID/start_ticks/argv身份吻合。随后唯一派发H3 v3 run（后台Fji8Rn），仅N2 GPU0，每族固定三case，同源权重与采样协议；派发时仍需CPU前置/完整内容审计，不能宣称真实GPU渲染已经完成。运行配置SHA=55f7fd1c75b996a43aacfdbbb275a7ffd0cdf813991680f66c9311b4ceb5a332，manager SHA=bba5d6f6cc42e1a7ac2818d41ed738ce0e53c074d3044d7a73558417042320bf。

F1/ROUGE与配对不确定性提交ae3e73f已推送，并以远端refs核实一致；原主工作区及无关改动保留。后续实际queue、日志、完整回执优先于本时间快照。

## 最终test文本工具与H3真实缓存阶段

首次test500统一汇总工具已独立读码并复跑10项CPU测试，冻结在runtime/final_text_report_v1/：报告脚本SHA=d909ef2bd990a2e33f302fdecb9893f077bc168ef6bf3416c5db2cda0037792d，原F1公式SHA=3cdbba545b861839978d553b00935966163fa20e294b7b388550f806dd546ced，测试SHA=fe7c7f402f87f58305255c558054e145955a898a601f80a6b4a8f830655cf03b。复用冻结原公式和实际v3/v2 collector核验，500源配对bootstrap保留每源两窗；另记录全raw UTF8 SHA精确复用组的全部成员比例及corpus distinct。没有因新诊断调整F1/阈值或重测。Flamingo没有记录逐forward finite hook，报告明确为null，不能补造该证据。

唯一CPU接续器aACKpW已派发，只等待已有Captioner首次test的complete/exit0/resources_safe全部出现后调用冻结汇总脚本一次；未生成新测试样本、未占GPU、尚无三模型最终报告。最近只读快照Captioner已写入101条完整分片记录（15/14/13/13/12/12/11/11），这只是部分进度，不是500验收。

H3 Flamingo三个条件缓存已有真实complete、exit0及三个固定ID；GPU0峰值为62655MiB（约61.2GiB），原生text_encoder与video_vae_encode各调用3次，DiT/decode为0且无音频分支。随后进入Flamingo视频阶段。缓存完成证明真实GPU条件计算，不代表视频已完成；最近快照仍无视频case audit。其他七卡holder身份不变。人评准备选择隔离离线匿名包，不修/重启共享review_web，也不把现网页不完整匿名、共享评分或有缺陷shuffle当成已经具备严格盲评。

## 时间戳、动作评分与训练前后对照新增要求

用户进一步要求核时间戳/动作表现，允许独立LLM打分，并比较训练前后。标签源ms_visual_batch.py:215固定chunk_seconds=2/min_tail=1.5，:398按固定桶分窗，:285明确window_boundaries_are_action_boundaries=False。因此现有双窗100%只能叫固定时间分桶/非空格式合规，窗口动作F1也只是同桶弱参考一致性，不能叫精细动作起止时间准确率、自由边界检测或音乐同步验收。当前fixed3协议无target_text或事件边界GT，不为它编造F1/时间真值。

独立匿名LLM已实际审阅val32每族全部32原音乐caption；序数意见在跨窗承接/动态动作/动作多样性/细节四轴为Flamingo3/3/2/3、Captioner4/3/3/3、MOSS3/3/2/3。此为文本意见，不是校准准确率；未听音频/看视频，不能当音乐匹配、动作边界或H3人评。

同一完整val490在更新2与670的NLL分别为Flamingo1.826934→0.765656，MOSS1.804309→0.735513，Captioner1.693363→0.684887。每族内部参考拟合有进步，但更新2已训练过64源，不能称未训练基座；也不跨tokenizer按NLL排模型，不能单凭下降说创作编排/音乐理解提高。

新增最小同条件基座对照正在独立实现：原官方基座checkpoint=None/lora_rank=None，不attach LoRA/modules_to_save，同原完整音频/原native audio-only prefix/greedy/seed42/cap2048，每族仅同固定3case；训练后已有fixed3原始输出直接复用，不重测test500、不增加训练。MOSS缓存/位置频率和Captioner EOS推理兼容修复须两阶段一致，不能把推理bug当训练收益。仅各指定节点GPU0，N7需fresh holder核验，N2等当前H3完整安全后、N4等首次test完整安全后；未派发基座GPU任务，尚无前后质量结论。之后对六匿名组×3case做独立LLM盲文本比较，报告理由、相近/未知，三case结果只作诊断不外推全测试集显著提升。

离线匿名人评包脚本已主会话独立11/11 CPU+synthetic ffmpeg回归验收并冻结：runtime/h3_review_pack_v1/，脚本SHAd07db03ddb996884ae9c2e79fe8f9e61b957f0e0cda4dfbb2c205edaca1333cb，测试SHAe857a11a640df40bb7c9e02744721f4555808a5f24bc69ca2dbcca7c2669e626。每case另存字节一致完整WAV、视频轨copy无损/AAC音轨有损，bundle_id绑定公开manifest/private key/评分JSON；公开包仅隐藏标签，不声称对抗数据拥有者追踪。唯一CPU接续kdE2Vs已启动，必须现有H3九片实际complete/resources_safe且全部门禁通过才构建；没有启动共享server、没有用户评分/接受结论。追加只读审查调用曾遇网关empty-stream，不能拿该失败调用冒新审查证据；本次主会话实测与已经收到的独立只读复核保留各自证据边界。

## 基座对照冻结与真实首片H3进展（2026-10-25）

新增基座producer/manager已完成独立读码和27/27 CPU回归，前后SHA一致，冻结runtime/base_fixed3_v1/：producer=c6cb903616b216dbf537b5af03027744de3262105390a97664aac78e10abd5e6，manager=622f05b931c981e77ecc7e1ec1914a43112167c04504780cd25e286c074b62e7，测试=1cbb409f8f88912719e263327757180d901081ae3edcb76601760bf0055b29b6。部署前纠正了新诊断器遗漏合法5.0625时间格式、没有核真实worker非零退出、export .jinja资源遗漏及实际EOS/input_tokens对照等问题；这些修补未在旧运行快照上就地修改，也没有把CPU测试当GPU通过。

N7原MOSS首次test500 complete/safe/test.exit0已经绑定并独立核验。GPU0独立capture确认原holder身份且signals_sent=false，冻结moss.run_config_v1.json SHA=c9b53af46255d5315b71be0af8b892308564148a3a9b6176d8cac195c84bc4d6；随后唯一派发qk03LN。新base_fixed3.live的PID3472213/start_ticks537331257对应独立基座worker，官方HF加载报告仅known nonpersistent inv_timescales为UNEXPECTED，按同原生位置buffer修复对照；加载报告不是逐参数恢复证明。最近快照raw仍0，尚无三条基座完成/资源恢复证据。其他七卡仍占卡，不重测已训练输出或test500。

N2基座唯一接续5cDwY8、N4基座唯一接续hOPNx3已启动且stdout明确只等待相关H3或首次test的全部actual complete/safe/exit0，不提前上卡；之后才冻结当前前序ref、重新capture、生成原同三case。每族base独立O_EXCLonce claim，不使用最终test账本、不自动重试失败。六匿名组×3case LLM rubric已经在base输出前冻结，无时间事件GT或音乐/视频真值评分。

H3 Flamingo第一片已有完整case audit、实际MP4 SHA=4375522773b4c0b2b8ae8844b6616483db242d4414ba9c95dc0d1da4ec0646c9，完整124帧704×1280@24fps解码、50次DiT及1次VAE decode、无音频分支调用；这已是真实视频，不再仅cache证明。但仍仅1/9视频，不能当三模型视频评测已完成或画质人评通过。最近Captioner首次test写入266/500，最终报告aACKpW仍待完整与安全回执；离线人评kdE2Vs仍待九片/资源安全。后续queue/proc/文件实查优先于上述快照。

## MOSS基座完成与离线评审工具v2修复

MOSS纯基座同3case已完成真实GPU生成，主会话再次用冻结producer/manager核raw/manifest/实际exit0、608/608 finite forward、3EOS/3双窗、noCheckpoint/noLoRA、同源prefix重放及resources_safe全部通过；GPU0已恢复占卡。其基座已经能输出固定双窗，说明固定格式本身不能拿来证明训练收益；动作质量仍待同预冻rubric匿名前后审阅。独立技术总结存results/music_planner_moss_base_fixed3_technical_20261025_v1.json。

主会话发现离线人评v1的元数据拒绝检查误把ffmpeg Input横幅内匿名目标文件绝对路径也当泄露：目标包位于ROOT下，而ROOT本来在forbidden列表，实际构建必被误拒。先在真实synthetic ffmpeg测试中加入父目录反例，明确复现失败，再仅排除Input #0横幅里的当前匿名目标路径，其他实际媒体metadata仍全检查；真实私有comment泄露新增回归继续拒。11/11 CPU回归通过，且已完成第一片H3视频作CPU临时mux保真验证，124packet/PTS/时长原样、完整WAV未改，AAC有损和356 padding样本被明确记录，不是九片人评包已经完成。旧v1冻结源码不动，旧CPU等待接续kdE2Vs在未写包/未写key前精准停止，GPU任务未动。

修复版冻结runtime/h3_review_pack_v2，脚本SHAd498b175ad9eb22110b30c8efdf6bf19a0bce5be2b37cdf91bc45d88b986c7a5，测试SHAddea71d5cbbee690f7b8cbb8dc7f3a09958b6dce538fda5488cbb6213033cfc5。唯一新版CPU接续O3XtBy等现有H3九片完整且safe后构建独立planner_h3_review_offline_v2，private key在公开目录外，未启动共享网页。训练前后匿名材料唯一接续Dyc3fS等三族基座全部actual complete/safe后导出18条六匿名组，不预填评分；后续才独立LLM盲审。最新快照Captioner首次test已写428/500但仍无complete/safe，H3仍2/9，MOSS基座已complete/safe而其他两族基座等待前序。后续实际文件/回执优先于此时间快照。

## Flamingo完整三片视频阶段独立验收

Flamingo视频阶段已完整3case/实际exit0，主会话在N2授权节点只读复用冻结H3 prerequisites/check_stage独立重验通过。原始完整run_complete SHA=e4d5f0092272524241ca690a91336111dc3d4c2ff424f2b1587a8bc9ce8ef11a，累计DiT150/VAE decode3，无音频生成；同704×1280/124帧/24fps/50步/seed42/CFG1/shift12/noLoRA。仅3/9视频，现有manager自动继续Captioner条件缓存，不重复派发；没有用户视频质量评分。最初在coordinator直接调用prerequisites被authorized-host门禁拒绝，随后在N2只读调用通过，没有绕scope或降低门禁。

预先冻结的训练前后LLM rubric已保留发布副本results/music_planner_prepost_LLM_rubric_20261025_v1.json；这不是实际评分。首次Captioner test最近447/500，8rank仍以原once账本跑；最终三模型文本report和另外两基座仍待各自actual complete/safe。

## 当前结论

三模型clean_v11正式训练均已真实完成670更新、2epoch、21426源曝光及完整保存/新进程恢复。完整val32三条件文本工程门槛均通过。弱参考F1与匿名文本审阅暂倾向Captioner，但相关跨模型配对区间含0，没有显著或端到端最强结论。Flamingo/MOSS首次test500已完整并安全；Captioner首次test500仍实跑，最终三模型test汇总未完成。H3已完成Flamingo三个真实GPU条件缓存并进入视频阶段，实际九片视频及用户人评尚未验收。GitHub只包含本实验明确源码、测试及分析，不上传权重、私有音频/图像或凭据，不改变其他会话任务。
