# 两阶段草稿产线接管与提速方案（2026-10-26）

ROOT=/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1
R=ROOT/runtime

## 1. 现状（约 19:00 CST，来自原会话汇报）
| 产线 | 已完成 | 近10分钟速度 | 状态 |
|---|---|---|---|
| 第二池 pool2（N2 g23/g45/g67） | 477 / 7,944 | ~564/h（启动以来平均 ~640/h） | 仍有 ~1,905 条就绪来源未被领取，GPU 利用率 78–100% |
| 第一池 pool1（n4g67/n3g01/n4g45/n4g23） | 800 / 5,710 | ~166/h | 已有就绪来源全部领完，等待准备环节供料 |
| local_n4c3_balanced_v1 | 554 / 2,396 | ~49/h | 低效 |
| local_n6_balanced_v1 | 311 / 925 | ~18/h | 低效 |
| live_n5c3_v3 | 350 | ~25/h | 低效 |
| 合计 | | ~820/h | 全量 222,939 条未完成；当前短窗口不预测完工日期 |

产出全部是未审核草稿（quality_approved=false、training_eligible=false），8B 审核后置。

## 2. 瓶颈判断
1. 来源分配不均：第一池在等料，第二池有积压，低效固定线占着 GPU 却产量很低。
2. 准备环节（ROI/关键点 + BeatThis + 分段）的供料速度决定吞吐上限。此前实测：10 个 producer 全部开着时约 1,450 条/h；N2 缓存准备约 215.6 条/h。
3. 池的参与 worker 列表固定写在清单里，并用 SHA 绑定（R/caption_ready_pool_v1.py:121 check_manifest、:158 Pool），不能热加 worker。合并池只能走交接：停下 worker → 剩余来源组成新池 → 重新加入。

## 3. 改动方案（按顺序执行）
### 3.1 接管前提
- 原会话停止定时循环（cron a559ef04），并停止一切调度和交接操作，交出在跑进程清单和未完成的交接。
- 同一时间只允许一个会话调度这些产线。

### 3.2 合并来源池（pool3）
- 复用原有交接工具 R/caption_pool_consumer_handoff_v1.py（park → remaining → 新 manifest），不另写一套框架。
- pool1、pool2 的 worker 做完正在处理的批次后再 park，不在批次中途停止。
- 新池来源 = pool1 与 pool2 中未领取、未终态的来源（含尚未就绪的）。已领、已失败、已终态的来源一律不重派、不重置预算。
- 参与者 = 原 7 个 worker 的 GPU 组 + 后面回收的 GPU 组；从一个全局就绪队列领取。
- 模型需重新加载；以真实admission、模型启动与新增终态验证部署，不以代码通过当恢复生产。

### 3.3 回收低效固定线
- 对 n4c3_balanced、n6_balanced、live_n5c3_v3：已领批次做完后再 park，剩余来源并入 pool3，GPU 组作为 pool3 的 worker 重新加入。
- N7 两条旧消费者（shard 7/8）保持不动。N7 可用内存低于 160GiB 门槛时不迁移，也不降低门槛。

### 3.4 提升准备环节供料
- 先实测当前每个 producer 的就绪来源产出速度（只统计新增唯一的 ready 来源，剔除重放）。
- N2 GPU0/1已有准备进程，低显存不代表空卡。增加准备进程必须使用授权的原生资源生命周期、互斥来源分配和确定的进程归属，不能裸启动同卡进程或只凭前沿间隔保证不重叠。此前辅助试运行已STOP暂停，未自动恢复。
- 准备环节优先补给剩余就绪来源少于约 1 小时用量的池。
- 本地暂存加批量写回（并发 ROI 发布实测 1.40x）放到第二步，需要同来源对照后再上线。

### 3.5 修复收集坏行（MzRpLm，offset 4579239）
- 不修改原文件。把该行的字节位置、原始字节 SHA、可解析出的来源 ID 写入隔离文件。
- 此5字节残片没有完整来源ID，关联保持unknown；不猜来源、不凭空重做。另行对账若发现缺源，再单独授权修复。
- 从下一行继续收集，最终结果里注明隔离了 1 行。

### 3.6 监控
- 每个池记录：去重后的新增来源数/h、窗口数/h、就绪未领取数、worker 空等时间占比。
- 就绪未领取数低于约 1 小时用量时发出告警。

## 4. 保持不变的约束
- 只保留三类核心检查：输入合法、来源 ID/split 正确、结果不覆盖并保证资源安全。
- 不停止未知进程，不删除锁、租约或旧结果，已领/已失败来源不重派，不恢复旧配置。
- 跳过的审核一律标为未审核，不伪造通过状态。

## 5. 新增服务器的 40GB A100 协作分工
- 优先补准备环节（同配方关键点、节拍、分段），其次扩容同模型32B草稿。不要更换8B模型冒同配方产量。
- 32B BF16在40GB卡上优先同机4卡TP4；TP2仅在完整图像/长度配置实测显存与3case成功后采用。8卡节点先1个TP4，再验证供料、内存和IO后起第2个；少于4卡可先做准备环节。
- 外部agent先核GPU/PID归属、锁/租约、本地盘、环境、同模型和共享路径；不同服务器不视为自动挂载同一路径。只在节点独立工作目录编辑，不修改主集群runtime/manifest/claims或运行冻结源码。
- 当前caption_union_pool_worker_v3.py是试运行入口，恒true parent_gate和外池owner协议尚需安全适配，不能直接复制上线扩容。使用主会话提供且已核验的原生生命周期入口，真实占卡进程按授权释放/恢复；不能仅凭util=0抢卡。
- 所有生产来源必须中央排他分配。共享盘必须核跨节点领取互斥；不共享则用有限来源票据+本地write-once+回传终态。禁止各节点自行遍历精选清单选源，禁止清claim/失败后自动重领。
- 先固定3个被明确分配的来源验证视频PTS、NPZ shape/mask、生成与本地暂存/打包写回，再扩大。仅标unreviewed，所有approval/training资格False。
- 监控每60s，15min去重成功/失败/窗口计数、GPU归属/显存、复制/计算/发布时长、剩余就绪量；无新增不等于闲卡，不自动杀/重试。
- 先返回资源报告（卡数、有效GPU编号、空闲归属、headroom、本地磁盘、模型/环境、共享路径、可承接角色）；主会话随后给唯一worker身份、有限任务票据与安全启动命令。

## 6. 验收
- 每项改动完成后，按 15 分钟窗口统计去重新增来源，与本文第 1 节的约 820/h 对比。
- 不承诺扩容后吞吐或完工时间；按同窗口去重成功来源、失败、窗口数与发布开销实测。

## 7. 接管实测与实施状态（2026-10-26 22:32 CST）
- N4联合试运行三worker（g01/g23/g67）已确认compute进程与实际progress，22:29合计123个来源终态（完整122、partial1）。这是累计试运行产量，不是全量去重小时率。
- v3存在parent_gate恒true与原池foreign owner恢复风险；不能对外复制上线、不能重新启动原池恢复流程。旧claim/缓存/receipt全部保留。新增安全v4使用原生资源生命周期，N6g23为独立试点；未完成真实GPU验收前不称已部署。
- N3/N5/N8低util卡仍有已加载模型和其他任务，不能当空卡；N7原生headroom约157GiB未达160GiB，不能降门扩容。N6g23曾确认已恢复holder且headroom足够，启动前必须fresh复核。
- 辅助producer短窗口供料1044/h，随后因读取稳定性风险STOP暂停；不承诺倍数，不自动清STOP或重复launch。
- 监控v1启动基线逐terminal的FUSE读取阻塞；重复实例被lock拒绝不是生产故障。v2采用跳历史基线、限定小读取，仅统计observed_since_start新增终态，历史总数unknown，真实心跳/窗口完成后再汇报吞吐。
- 精确5字节残片offset4579239已隔离，源关联unknown、完整来源0，不猜ID、不修改原文件、不假设需要某个已知源重做。原collector游标推进4601499，新增收集32个输入不是新增32个生成成功。旧handle tcImOo随后killed，主会话KxVx1F从已存checkpoint续收一次，无rollback/预算重置；最终尾部回执尚待核实。
- 2D规则入口23项tests通过，固定3个真实来源CPU诊断已write-once，114事件/9窗口；三来源均partial且diagnostic_not_production，不计正式标注。PTS差异确为ROI{times,ticks}和bar相对时间列表不同摘要口径，已核同已读小时间列表，不probe/重hash媒体。观测0..5s，未观测尾部/无效点空洞不外推。


## 6. 新增40GB A100服务器：无VLM规则草稿优先（用户最新决定）

- 新增服务器用于关键点/节拍与规则MotionScript草稿，32B/8B审核均后置，不要求TP4。优先每节点1张卡3-case验证，验证后按实测显存/IO逐步扩展，不默认占满服务器。
- 当前2D134抽帧缓存不能直接输入现有3D NLF规则。不得假设NLF齐全、把2D补零当3D、把前处理ready当动作caption完成。首次需要做一个独立2D粗事件适配器；现有3D缓存只有确认同源/格式后才可选用。
- 新agent只写各自独立部署/输出命名空间，不修改现有runtime冻结源、pool1/pool2/union的claims、terminal、锁、租约，也不使用临时union_worker作为扩容接口。正式任务清单与唯一分片由接管会话统一授权，清单未到位时可准备环境/实现适配器但不自选全量来源。
- 2D事件必须使用joint_layout真实映射、person尺度归一化、时间差分、valid/score掩码；没有证据的细动作记unknown。输出保留id/split、真实观测时间段、动作事实依据、MS草稿文本；禁止推断3D旋转、左右脚承重、掌心方向、音乐风格或画面看不到的手指动作。
- 复用已有同源关键点和节拍，不反复SHA/probe。缺cache只在所分配来源上新增提取；关键点与音频模型可使用GPU，但事件运算单条很小，先CPU实现再决定是否批量CUDA。
- 输出draft/unreviewed，semantic_approval=false、quality_approved=false、training_eligible=false、needs_visual_revision=true。实际输出成功/partial/needs_regeneration分开统计。
- 先同3case证明有非空动作事实和时间对齐；随后比较包含复制、模型、规则、写回的15分钟去重MS草稿吞吐，不把初始化/排队之外的局部速度当端到端。
- 本地缓存当前批和下一批，有界预取与批量可靠写回；成功写回后只回收自己创建的缓存，不删除未知文件。
- 部署前必须核GPU进程身份、FLOCK、租约、内存门。现N4临时union进程未具备完整原生资源生命周期，不再复制推广，安全适配完成前保持其他任务不动。


### 22:48 CST 后续状态
- v4最终独立23测试通过，cfg与原生manager SHA真实N6 check-only通过，fresh资源门通过；主会话GFoEe3一次部署，manager1868566→worker1868685，原生admission/live已落盘。已领首个3来源批次，模型/首批durable结果还需验收，不假报部署已恢复产量。
- 主会话z15Rgr持有三池监控，out=takeover_monitor_v2_main，独立14:39:59Z基线；14:48:01Z观察137新增成功（含2partial）/481.6秒约17.07源/分钟，完整15分钟未形成，无警告；不含N6v4单lane/未监控旧fixed线，历史总量unknown。早期子任务监控Zyphz7被回收，不拼接两个基线。
- extra04/05/06验证票据位于ROOT/external_a100_validation_20261026_v1/a100-extra-0{4,5,6}.json，仅各1来源GPU0真实DWPose+BeatThis和CPU规则诊断；extra03分发，01/02先本机环境报告。均不允许批量选源/旧pool领取，正式产量0，不能把票据当生产pool授权。


### 持续巡检（2026-10-26 23:00 CST）
- 原三池主监控z15Rgr的15min窗口14:43–14:58Z观测247成功（含1partial），约988/h；这是terminal新增观察，不含N6新lane/旧固定线，历史总量unknown。N6同一时刻累计42 generated，尚未形成独立完整15min率。
- 原collector已推进到4902093、joint_complete3644/incomplete_or_failed3392；仅原收集分类，不表示模型新生成或独立质量通过。一次续收handle KxVx1F仍待最终回执。
- 已创建会话巡检cron b9f74ced，每10min核持续任务/心跳/错误，只读并报警；无新增不自动kill/删锁/重领，用户暂停则取消巡检但不停止作业。会话结束后调度失效，最长3天自动到期。


### 巡检与N6独立监控上线（2026-10-26 23:13 CST）
- 本轮z15Rgr、GFoEe3、KxVx1F均TaskOutput running；本机监控/collector与N6远端原生父子start_ticks、父进程GPU锁/lease fd独立核验。未自动停止/恢复/重派任何来源。
- N6独立监控CuzysC已由主会话启动，PID4114673，out=ROOT/takeover_n6_monitor_v1，首个真实心跳baseline_initialized已捕获84条历史作为基线，不计新增速率。此前6项监控测试扩展为8项，独立通过；无缓冲4KiB读、终态256KiB/progress1MiB、baseline实际时刻开始滑窗。未热改生产v4源码或配置。
- 三池当前15min246 generated约984/h，无警告；N684generated累计不直接与三池小时率拼接。collector游标5057924，分类joint3768/incomplete3492；尚未最终尾部回执，不称全量完成。


### 新增A100共享盘准备小批次（2026-10-26 23:32 CST）
- 用户已明确01–06均能访问存储盘，取消常规relay，按原共享路径读入本地有界缓存，最终write-once批量共享写回；实际挂载视图仍在节点首次真实读取处核对，不改系统mount。
- 主会话已回读共享03–06验证package和summary，实际NPZ[81,134,2]、CUDA节点profile存在。01独立模型结果仍为用户反馈，其结果包尚本地未发布。CPU草稿仍消费旧32frame，故不能将当前GPU提取速度当最终新草稿速度。
- 已冻结ROOT/external_a100_fullframe_pilot_20261026_v1/allocation.json及assignments/a100-extra-{01,03,04,05,06}.{jsonl,ticket.json}，每节点100条、合计500个唯一train来源。独立新全帧缓存edition，不覆盖旧ROI/producer终态、不读写主32B claims/预算；不是全量222939生成成功。仅各GPU0单个常驻DWPose+BeatThis进程；首批端到端实测后再批准扩卡。
- 输出case receipt/fullframe manifest必须绑定实际视频probe duration/全原生PTS与实际新NPZ，音频时长不证明视频时长或同步；已验证score可>1、无效点保留。每源一次真实读取/复制边界，不重复profile/weight SHA/模型重载。新规则adapter真正消费全帧cache，尚未发布前先存GPUcache与Beat sidecar，不等adapter停GPU准备。
- 原collector KxVx1F已completed，waiting=tail，offset5231852、processed_source_hints874/new_queued_sources392，sources_seen7510；分类joint3908/incomplete3602。这是一次收集完成，不是修复模型推理完成，任何新未知坏行仍failclosed。


### 全帧规则出口发布（2026-10-26）
- ms_fullframe_pose2d_draft_v1.py已独立16tests通过，源码SHA31407e742df71f5a6b980282505399f66027a9e3665c2b19681064d0942b9771。
- 真实固定3case真正消费03新81frameNPZ（不是旧32frameROI）：事件41/37/68，窗口5/2/11，三来源都partial；source记录fullframe_manifest和新binding，无伪ROI/ref/bar。末帧PTS到video.duration的尾部仍unknown，audio只sidecar不声称对齐。
- report=ROOT/external_fullframe_adapter_diagnostic_v1/diagnostic_report.json；发布元数据=ROOT/external_a100_fullframe_pilot_20261026_v1/fullframe_rules_release_v1.json。新NPZ与manifest都diagnostictrue/四quality flags未批准，已发500源pilot仅独立新缓存+粗规则，不正式可训练caption。
- extra01与03同源tracking/valid不同，原因未知；不可仅凭事件/有效点数量强称同recipe或质量提高。后续需核图像预处理/主体跟踪/版本，保持各worker provenance不混归一。


### 03全帧规则导出完成
- abHz9t真实exit0；CPU CLI处理100来源/100written/0existing/0rejected，72.361s，4674信号事件。独立回读全部100草稿：来源唯一100、411窗口，100partial且100含lastPTS之后的未观测尾部，quality/training/semantic均false。
- 验收ROOT/external_a100_fullframe_pilot_20261026_v1/central_rule_exports/a100-extra-03/acceptance.json；规则出目录同根rule_drafts/train/ID/draft.json。无需03节点再次导出，不增加100条唯一媒体计数。
- 01已完成GPUcache的实际schema为ms_fullframe_pose2d_npz_v1+boxes，source_pts/tbase与decoder[N,2]实际start/end，v2只显式内存适配保留原schema/binding，不修改已发布NPZ/metadata、不重做GPU。01的100CPU导出仍待v2交付与真实3case。


### 01画外点规则边界修正（待验证，不已部署）
- 固定前三个实际01缓存：sZdp有71/10466原valid+tracked点越界（脚踝、脚趾、脚跟），OAXB只有4/10854脚趾点越界；leVi无越界。有效原坐标均finite。局部画外模型预测不证明整源损坏。
- 旧v2严格像素边界拒了2/3来源。新策略仅独立rule_observation_mask排除画外支持点，原xy/valid/score与完整缓存binding不改；不裁剪画外坐标到边框、不以局部足点缺失弃整源。有效raw NaN、低于标记阈值、shape/身份/PTS错仍拒绝。输出需披露被排除点/帧/关节与证据缺口，保持partial/unknown。
- 新策略还在回归及固定三case验证中，01的全100规则输入已中央创建，批量CPU导出尚未启动，不能用旧33test交付冒新策略通过。
- 23:31巡检主/N6/独立监控均running，原生父子startticks与锁leaseFD核一致、lease4.4秒；三池15min97草稿，N6独立15min15条且累计219；无监控警告和10min无新增告警。旧collector此前tail完成不重复启动。


### 01画外掩码v2验收与批量CPU导出
- v2最终SHA3e41a9461173117e612703d86a9fd6eeb49ea46ea0f5f159e31637674958594e；独立39项回归通过。v1冻结未修改。
- 新同源3case真实结果：sZdp28事件/71画外点排除，leVi61事件/无画外排除，OAXB26事件/4画外点排除；3written/0rejected/3partial。原xy/score/valid及原schema/boxes/binding保持，仅派生规则支持mask。有效raw NaN或低分valid矛盾仍拒绝，局部画外数据不得贡献事件，未知尾部保留。
- 主会话已独立回读3draft核ID/source/flags/排除点计数；唯一一次100来源CPU导出handle emWnJo，out=ROOT/external_a100_fullframe_pilot_20261026_v1/central_rule_exports/a100-extra-01/rule_drafts_v2，不重跑GPU/清预算/改原缓存。
- 新release ROOT/external_a100_fullframe_pilot_20261026_v1/fullframe_rules_release_v2.json。03此前100规则已完成，不再次导出；01的100尚待最终回执和批次验收，不能用3case写出冒全100成功或质量approval。


### 01的100来源CPU规则导出完成
- emWnJo实际completed/exit0，100attempted/100written/0existing/0rejected，100partial、4673信号事件，CPU CLI84.852s。主会话逐条回读100draft验证原ID/split、原schema、未批准flags和事件支持不含记录的画外点。
- 45来源局部空间证据受限，合计5207原valid+tracked画外点被规则支持mask排除；原NPZ/valid/坐标/分数不改，不凭局部脚点缺失丢弃整源，tailunknown保持。
- 输出ROOT/external_a100_fullframe_pilot_20261026_v1/central_rule_exports/a100-extra-01/rule_drafts_v2，验收acceptance_v2.json；514窗口。
- 实际01与03两批200draft sourceID/split交集0，共200唯一来源，不是新增200媒体或可训练完整caption。两批全部partial/unreviewed，GPU提取成功和CPU导出不得冒质量审批。


### 用户要求统一无VLM流水线（2026-10-27 CST）
- 最新指令：先不运行VLM，只MotionScript；主集群和extraA100使用同一pipeline。目标为视频→全原生帧BGR DWPose→BeatThis→2D规则草稿→本地小批打包共享写回。32B生成/8B审核后置，不把CPU规则质量当VLM完成。
- N6nativev4已精确核PID/argv/startticks后写STOP请求排空，不发杀进程信号；已落drained_not_complete summary：processed=reserved=248、uncommitted_owned=0。尚待resources_safe/holder恢复回执，不把summary当资源释放完成。
- 其它本会话VLM（pool1/pool2/N4v3/N5live）正在核安全停止边界；无关会话任务仍不碰，未自行停群。旧claim/results保留不恢复旧cfg。
- 新入口ms_unified_novlm_v1.py/test新建实施中，复用正式BGR颜色（不是01旧RGB直入）、v2新全frame schema与局部画外规则支持mask，模型常驻只加载一次、无RGB/crops/32B/VLM。未通过新统一3case和真实GPU运行前不能宣称统一已上线。
- 巡检旧cron b9f74ced已替换2c1bc7e6，每10min只读新无VLM目标。预期暂停VLM不重启/不清STOP，不将暂停无新增当故障；主/N6监控保留，已完成collector不重复启。


### 统一无VLM实际三case已验收（2026-10-27 CST）
- 统一核心ms_unified_novlm_v1.py SHA85ea6781…，24独立CPU回归通过；资源包装采用ms_unified_native_manager_v3.py，17独立资源回归通过。核心负责算法，原生caller适配真实cgroup/controller/固定eval环境，160GiB/锁lease/pidfd恢复门不改。
- N6原生GPU2三case Mu5T8w，phase ms_unified_native_v1_diag3c_20261027，worker_exit0/resources_safe全部true；243原生帧、3新NPZ/新规则草稿、0流程失败、3partial。VLM调用0。档案只有15个必要文件，无RGB/ROI裁剪/原视频重复归档。
- 总core23.2087秒，models load once4.6241秒，打包发布8.2159秒；各81帧decode_pose1.9752/1.3774/1.3410秒，rules0.0944/0.0817/0.0872秒。不能按3条外推全量小时率，仍需统一批量端到端验证。
- 旧N6VLM已drained248/248且真实safe；N3g01与N2g23/g45原VLM直属worker及预核6rank经精确pidfdTERM退出，三组已真实safe并复核恢复holder，未信号未知core/holder/群组。N2g67/N4oldpoolg45/N4directv3三组/N5live仍待安全停止，不称本会话VLM全停。
- 初始diag3/diag3b试跑分别在controller命名空间读取及isolatedTorch导入前失败；原failure/claim状态保存，真实pose sourcecalls0，不恢复旧ticket。修正后用独立diag3c票据验证成功。验收R/ms_unified_novlm_real_diag3_acceptance_20261027_v1.json。
- N6无网络，pipOCOhzO失败；已从N2正式依赖离线复用到N6独立/opt/ms_unified_deps_copied_v1，仅BeatThis1.1.0/rotary0.8.9/匹配Torch2.8cu129的torchaudio/torio，未升级其他环境。已核evalNoUsersite需要显式固定/usr/local/lib64和lib installedruntime根，caller只允许这些已验证根，不混旧32BvllmTorch2.13cu130。


### 额外01–06先行运行任务（2026-10-27）
- 已生成ROOT/external_ms_unified_run_20261027_v1/allocation.json与assignments/a100-extra-0{1..6}.{jsonl,ticket.json}，六节点各1000、共6000唯一train ID/split和6000视频路径，排除旧外部500、主N6统一100及固定诊断3case，交集0。
- 授权仅新无VLM全帧BGR DWPose+BeatThis+同v2规则；每节点先GPU0自己的3来源确认新通道/输出全链，保留模型继续不重跑这3条，通过实际独立小模型资源门后允许GPU1/2/3，共至多4进程。每GPU领取本节点1000行索引mod4==physicalGPUindex共250条，未启GPU的分片保持pending不交给其他进程。
- 原生N6资源caller不可冒充额外节点资源授权。额外节点复用实际已验收小模型资源wrapper，统一core load_models/copy_source/extract_and_rule及v2算法；默认拒绝CLI不改成恒true/native_parent伪证。只改本地启动胶水，完整冻结code_refs按ticket核一次，不共享热改源。
- 存储挂载路径若不同只做本地前缀映射，无法读取则03有界relay，不能把ENOENT当全盘无访问或停所有节点等待。每GPU本地8GiB/2来源预取/1copythread，按20来源batch发布、不每源重复媒体/权重SHA或初始模型加载，不保存RGB/crops/VLM。
- 六节点实际执行需外部agent回传，分片文件落地不等于6000已运行/完成；02真实模型尚待本机首3验收，不能沿用01结果。缓存历史同源不同stage不算新增全局媒体量，所有规则草稿未审核不可训练。
- 主集群统一N6GPU2 batch100 J1UmgI已实际运行，最新日志有音频推理且无failure/summary，尚未验收完成。旧VLM安全停止剩余边界研究恢复alive teammate，不重复spawn，不因暂停lane无新增误报运行失败。


### 单归档发布优化已部署（2026-10-27）
- 原统一100来源实际验收完成：100unique/8100帧/100partial/0流程失败，core wall845.9445s、pack_publish248.7604s，只有18.5MB却逐500文件重复fsync/link/dirfsync。实际archive/commit/500payload和safe回执已核。
- 新ms_unified_novlm_v2.py SHA c2f6d404…仅改变输出存储，算法/原生PTS/BGR/跟踪/规则不变，source manifest/draft/receipt均显式archive_member locator，批次仅batch.tar+commit.json两个共享文件；不能以虚拟npz_path谎称散文件存在。失败档案只有receipt，旧claim不重派；真实fsync+exclusive link+dirfsync仍保留。
- 核心33和新原生caller37项独立回归通过，包含100source模型只load一次、archive成员/receipt==commit、原算法事件和新NPZ字节同fixture。
- 零模型同18.5MB发布诊断：归档写回1.1986s、receipt0.2873s，旧dual248.76s；排除packing且非隔离条件，不称总体200倍。诊断新生产source0/noGPU。
- 新N6GPU2实际100来源优化作业x8pnM0已启动，phase ms_unified_native_v1_archive100_n6g2_20261027，ord1100–1199，与原100/外部6000/旧500/3诊断来源交集0；原nativePID1991551→worker1991586持GPU锁lease，真实DWPose+Beat日志/显存1.38GB采样已核，首13来源已claim。最终100可靠提交与端到端速率尚待回执，不修改运行源或自动重试。
- 新caller计数实际动态resourcegate_calls/seconds写最终summary，用于下步区分门控等待，不删除资源门。release R/ms_unified_archive_release_20261027_v2.json。


### 发布优化实际100来源完成（2026-10-27）
- 单归档x8pnM0已真实100unique/8100frames/100partial/0流程错误完成，workerexit0和resources_safe全部true。主会话独立m.summaries核来源索引/档案SHA/size/recipe、逐100receipt==commit、draft归档locator、exact500members和batch只有2共享文件，验收R/ms_unified_archive100_acceptance_20261027_v1.json。
- 新corewhole554.8713s、publish1.3194s；旧邻近100sourcewhole845.9445s、publish248.7604s。观测单卡426→649源/h约1.52倍，非同源隔离性能试验，不能断言只有存储改动贡献差异。VLM0，不把全部partial/unknown当已审训练标签。
- 新实测runtime_gate_profile=107次/248.2605s（约45%墙钟），资源门成为下一明确开销。已派只读组件计时与最小优化设计，不删除PID/start/内核FLOCK/lease/app子树动态门、不热改源，不为快读伪造安全回执。
- 发布优化完成不意味着全群已切换或外部6000已完成。原6000分配及旧claims保留，不重新运行已有100case；新增pipeline保持同算法和模型，归档消费者需按locator读取member，不期待旧散文件路径存在。


### 两外部集群各8万条持续任务（用户新增要求，2026-10-27 CST）
- 集群分组按此前反馈：01–02一组、03–06一组。已实际冻结ROOT/external_ms_unified_clusters80k_20261027_v1/allocation.json及verification.json，两组各新增80000、总160000个唯一train ID/split与160000不同视频路径，和以前全部已分配ID交集0。仅ID/路径去重，不谎称内容相似去重。
- 01/02各40000，03/04/05/06各20000；每节点GPU0–3按自身manifest零基行号mod4各10000/5000，原source_ordinal不能替代分片行号。20来源可靠小批持续执行，单core调用不得把40000一次塞入有≤100限制的run_ticket；复用已验证本地常驻调度，算法同冻结BGR/fullframe/BeatThis/v2规则，无VLM，缓存8GiB/2来源预取/1拷贝线程每GPU。
- 票据路径ROOT/external_ms_unified_clusters80k_20261027_v1/assignments/a100-extra-0N.ticket.json；清单同目录a100-extra-0N.jsonl；队列root/a100-extra-0N.queue.json。01/02队列优先旧正在执行1000，再预分配5000，再新40000；03–06优先旧1000，再新20000。已消费旧票据不重新启动，旧未完成/失败sealed来源保留不重提，工作模型允许已授权下一批自动续接，无须每1000再求许可。
- 原01账目显示999已发布和一个未完成记录，保持原产物/claims；原账目里的02未启动为历史状态，用户最新报告02运行，不能据历史账目强停02。新的路径只是授权和供料，没有主会话启动外部节点的实际证据。
- 8万来源是新追加排他任务，不替换旧票据/预算，不将历史稀疏缓存和新全帧规则的两stage相加当新媒体。票据/队列SHA及16万实际manifest回读唯一性已核，无媒体权重重校/新GPU模型调用。
- 当前巡检主z15Rgr/暂停N6monitor CuzysC存活、心跳新，暂停VLM无新增expected_paused_vlm，旧collector正常tail完成不重启；主集群迭代不要求外部集群停机等待。


### 2026-10-27 实时NVML与八卡持续打标上线
- 实时NVML四卡各100来源已封存提交、来源交集0，400unique/0流程错误/nativesafe：N3g0/g1 core309/317s、N6g2/g3 core321/323s，单卡观测1115–1165源/h；四批从最早admission到最后commit326s，观测约4417源/h（有限批次，不是长期质量批准吞吐）。每100动态门107次只约5.0–5.8s；设备UUID和compute PID逐次查询，无TTL，没有删PID/内核FLOCK/lease/app子树检查。
- 第一轮NVML模块未注册sys.modules在claim/model之前失败，四组均原生safe；失败文件保留，新v2真实隔离加载+29安全测试通过后独立namespace修正运行。不能以原manager任务exit0冒成功，均读取actualworker回执。
- 新连续core ms_unified_novlm_v3.py复用v2相同BGR/全部原生PTS/DWPose/BeatThis/v2规则，models仅一次，20来源每batch独立archive+commit、progress心跳，1..10000有限清单和STOP批次排空；核心12与nativecaller66项独立测试通过，既有监控8项也独立通过。
- 主集群本会话已经恢复且授权的8个slot实际开始持续1000源/slot：N2g2/3/4/5、N3g0/1、N6g2/3，总8000来源，ID与视频路径互斥且排除外部16万及所有旧分配。分配ROOT/ms_unified_continuous_allocation_20261027_v1/allocation.json，configs/plans在R/ms_unified_native_v1_stream1000_*；这不是整个64卡集群独占，其他会话及未安全归还槽位不碰。
- 实际managed handles N2 kcP86b/H937In/ze8k0j/2RD8FX，N3 M9ihNE/Pn7RsJ，N6 u20XtL/3LbD8p；原生真实内存160GiB+GPUholder/PID/锁lease恢复门仍在。当前metadata已核八worker存活、models_init_count1、各20/40以上commit且0失败、VLMfalse。1000有限任务结束后须下一票据，不恢复旧claim。
- 旧代理后续回合上游空响应，已有代码由主会话独立测试；没有凭旧任务state假定仍实现。连续监控原代理无运行产物，主会话独立ms_continuous_patrol_v1.py+7tests通过，managed xC0HaZ已真实采到八phase committed580作为基线（不统计新增），无warn，后续60s增量各batch严格source/parentticket/index/hash校验，global(split,id)去重，只metadata未独立读档案语义。monitor out=ROOT/ms_continuous_patrol_v1，注册表R/ms_novlm_continuous_jobs_registry_v1.json。
- 实际持续commit partial是generatedpipeline的子集，不二次累加；未审不可训练。旧N6VLM停止monitor无新增expected_paused_vlm，collector已tail完成不重启。当前外部01–06各自授权8万group票据独立，不因主集群优化停机。


### 八卡持续监控的实际后续（2026-10-27）
- 主会话xC0HaZ已采到第二轮以上真实增量，无警告；基线580来源排除不算新增，后来260个唯一来源commit已观测。最近metadata snapshot八卡合计840来源：N2 320/N3 280/N6 240，model_init_count八个全部1。这只是最新提交快照，不是8000已完成或人评质量通过。
- 实际原生父进程八个PID/startticks均核，持对应本机GPU FLOCK/leaseFD，租约年龄0–4秒；没有failure.json。新连续文件和资源包装被冻结，不热更新，不重启/重领旧预算。
- 常驻连续核心与caller分别12/66独立CPU测试通过，NVML2加载fix与安全门29测试通过；独立monitor7测试及首真实心跳/增量已核。原子source ID/globalallocation互斥，本机监控基于实际commit源列表和SHA而非sourcefiles数量。
- 10min巡检cron已更新92f707a8，读取当前八卡注册表并核xC0HaZ心跳；source剩余低于100提醒补票据，异常仅只读核并报警，不kill/重派/自动放宽内存。会话结束巡检失效，最长3天自动到期，不能承诺跨会话永久运行。
- 后续目标为维持连续实际产出并提前分配未消费来源，而不是反复有限100测试或停机等优化。外部两集群8万票据不需主集群新版本批准再处理，原batch边界保留现冻结执行版本。


### 八卡提前补料与自动接续
- 用户要求继续推进并核另外两个集群总量。新增主集群refill8000来源已冻结，ms_unified_continuous_refill_20261027_v1/allocation.json，每slot1000；与所有以前external分片/主集群task ID和视频路径交集0。旧任务进行中不修改旧cfg/模型参数/claims。
- 主会话ms_native_refill_dispatch_v1.py+5项CPU安全回归通过；八个受管waiters只在旧有限任务正常完成+实际workerexit0+全prefetchdrained0+native三resource_safe+freshholder/PID/UUID/锁预算通过后构造新cfg并exec原同continuousmanager。旧failure时拒绝新启动，不删除或重派预算。
- waiters R/ms_novlm_refill_waiters_v1.json记录真实handle：N2sNdtbe/qpUGKQ/HQ20vJ/SIU1v1，N3SzPLYH/gZnYr3，N6 661RNd/z0TyNm。新registry同slot仅在安全转换时原子替换，任务handle已真实登记，monitor自动读新phase。
- N3原g0/g1两个1000来源已正常完成safe，分别21:01:16/24Z后自动21:01:40Z新admission，无手工kill；已核新PID2887803/2887804和startticks，newrefill两卡各40已提交、modelsinit1/noerror。N6原g3剩80，waiter仍等正常结束，不停当前运行来抢供料。
- 小文件合并已落实结果单batch档案+commit，不散写5source文件；后续输入/元数据合并优化必须新版本实测，不删source互斥或热改在跑源。external总进度正在只读实际commit去重，160k分配不作产量，run/rollover/new80k不同edition也必须全局source去重。


### 当前总进度的统计边界
- 主集群最新主会话生成main_MS_unique_progress_latest.json，21:06:10Z actual sourceID去重7380（含此前900有限批次、首期和续期continuous，不含3诊断重复），其中3规则证据不足，其余partial。三15min监控observed1480来源≈5920h，无新警告，仅metadata提交观察非质量批准。
- 外部两个集群读一次commit审计JyCLgw受管仍运行，不能把卡慢当0产量。代理已核run版01/02合1999，03–06合4000，rollover01/02可靠追加26/4，新80k03至少已验收80；所有最新未读批次排除。因此全三群当前数只能给已核下界，不把80k授权加完成或重复cache阶段。
- 主N3首两个1000已经cleanfinishedsafe并自动接续下一各1000，new真实PID/提交已核；主N6下一源等待已部署且0kill，不因迭代停模型批次。接续旧失败或资源gate未满足则拒绝新启动；没有clear旧budget。


### 三集群总体进度实际核对（21:06Z快照）
- 主集群统一无VLM来源键7380（此前finite900/当前和已接续来源去重；重复诊断3源不计）；外部01–02统一草稿7190（旧run1999+rollover5191），外部03–06统一草稿4920（旧run4000+新80k920）。实际源键集合三者交集0，共19490个唯一提交来源，约222939分母8.74%，非质量或可训练完成率。
- 外部原完整扫描源12610含500pilot全帧GPcache，已剔除不当规则caption；失败3收据另列排除，partial12102/其它未审8。早期中央200CPUpilot仅诊断不重复加。三群真实源键快照与交集报告R/three_cluster_unified_progress_20261027_v1.json；共享写入中采样非全局原子，最新追加未知，因此为已确认下界。
- 外部current进程活体未独立核，其最新metadata提交不是nativePID证明，不据未见root强称停止或全80k完成。原01/02storage interruption需单独处理旧sealed条目不重提，新rollover继续不等待主优化。
- 03/05/06的新80k已有880/20/20实际提交，04此次扫描尚未见该editioncommit，仅观测下界0，不等于证明节点没运行。每集群80k是排他来源授权，不加入done数。
- 结果小文件→每20源archive+commit已上线，保留fsync/O_EXCL来源互斥；后续输入包/元数据整合需新冻结实测，不热改在跑源。N6g3也已自动接续，旧1000safe后new21:08:53Zadmission/live，未临时kill或清预算。


### 三集群当前活体核验
- 主集群8slot实际PID/startticks、父GPU锁leaseFD与kernelowner均核；N3两卡和N6两卡已按旧批正常结束及资源safe自动接续，不以旧handlecompleted当产线停机。
- 外部01已知IP/配置端口但缺已验真RSA主机密钥，主会话StrictHostKeyChecking=yes实际拒连接，不降低检查；02–06缺可路由endpoint/认证配置。容器hostname、旧lease和近期共享commit不是当前GPU进程独立核验。
- runtime/ms_external_live_snapshot_v1.py可由用户侧agent只读采两轮间隔60s的GPU UUID/compute PID/startticks/UID/cgroup/argv与进程树，不启动模型/信号/领取。节点标签调用者声明、主机实采，需总控核归属与同期提交增长，不能冒SSH独立授权。
- AskUserQuestion在此客户端UI超时不可达，必要外部访问方式需普通文本协调：外部agent回共享live证据，或已验真SSH配置位置，不索取密码/私钥。主集群持续推进不被此阻塞。

### 21:44Z 主集群接续与媒体启动等待实证
- 21:41:21Z核八slot实际worker PID/startticks/PPID/UID/cgroup/exe、compute PID及GPU UUID归属、父FLOCK/lease FD与内核owner均通过；最新20来源commit与分配段对应。N2g5从700增至760并恢复；21:44:58Z八slot均持续新提交、监控age1.08s、无告警。完整15min1719个唯一成功规则来源约6876/h，非质量批准。
- N2g3/g4旧各1000自然完成（worker_exit0、resources_safe三flag真），自动切到新refill票；真实新worker分别711237/start550776618、711940/start550778075，21:44:58Z各已有20新来源。没有signal、clearclaim或热改旧票。
- 只读现场捕获本地ffprobe输入/tmp、本地/opt工具却在FUSE openat等待；继承LD_LIBRARY_PATH末尾空项，openat实际相对路径为glibc-hwcaps/x86-64-v2或v3的标准系统库。动态loader因空搜索项进入共享cwd，这是启动等待的实证，不是Ceph读取本地MP4。
- 零GPU/零claim对照Md9gbs：固定同一本地视频只读FD，1.27MB/81帧；原继承首轮1.929s，仅去LD空项0.173s，仅localcwd0.173s，两者0.175s，另加threads1仍0.175s。六次所有原始frame/stream JSON完全相等，inode/size/mtime未变；原继承复测也0.177s，说明目录/库查找缓存效应，不将首轮比值冒称稳态11倍或threads收益。
- 下一独立版本仅给媒体子进程过滤LD_LIBRARY_PATH空项和明确本地cwd；父环境、全帧PTS/BGR、160GiB原生门/锁租约、模型/规则/归档不改。当前所有冻结core/caller/运行及补料票保持原样，新版尚未部署。
- 外部01–06预定currentlive文件仍缺。可信连接/回传缺口未解决，不能用主八卡、分配16万或历史外部产量宣称全资源活体验收完成。

### 21:57Z 新媒体版本冻结与后续供料
- 独立新ms_media_local_v1.py SHA a7682c431d0d7a401c8d25c13d28d17cc98a1568e41885a64cc0937c703c88e3；continuousmanager v2 SHA 456d05be3bb9657c3eef0d58a3d14260e93f636115727915883c6ccbd0fc8d48。CORE仍v3，不复制整套提取/规则；新增依赖和实际工具闭包，ticket_for提前要求两媒体绝对工具路径。已有v1/core/票据/registry均未改。
- 新旧caller+helper联合148项CPU回归由代理及主会话KWaq0N独立通过；N6实际PYTHONNOUSERSITE隔离eval解释器仅导入新caller/helper及核SHA成功，0模型/0GPU/0claims。N2借用1个已复制本地视频只读FD，实际新helper输出81帧与原始frame/stream JSON完全一致，0.161s，父环境/inode/size/mtime不变；只1个可用现场来源，不冒称3case或GPU生产已验收。
- 同精选train剩余范围194300..200567为6268来源，14尾shards独立逐行核全train、ID/casefold/video/audio各6268唯一；178000..178299已有三份N2有限100票，不能回收。主后续拟8way分配4slot×784+4slot×783；外部16万和主既有16000不变，不取val/test。中央新有限allocation/actual前后ID/path交集验收正在推进，尚未启动新tail生产。
- 原92f707a8巡检已由5e022581替换，每10min只读，增加六节点两轮currentlive/同期commit证据检查；未读到回传标blocked-awaiting-external-evidence，不伪称停机或充分利用完成。不更改生产作业。

### 2026-10-27 02:58Z 最新真实状态与未完成边界
- 主尾段6268来源排他分配、独立actual ID/path交集及早期0..1999有界superset交集均验收0；新caller v2实际在八slot自然安全接续。最后尾段成功6266、失败2，三轮连续累计22265成功/3封存失败；加此前finite900，主实际成功来源键23165。失败均BeatThis输出合法性拒绝，未放宽规则或重试。
- 02:09Z三节点现场核八个tail summary完整来源顺序、worker_exit0、prefetch排空、resources_safe三flag、最后commit SHA/来源段/档案存在尺寸及恢复holder实际PID/start/UID/exe/cgroup/argv通过。原打标worker均已退出，仍running的原manager只do_wait，当前GPU仅恢复holder，不是继续生产。尾段最后一槽23:41:23Z完成，已授权主票全部耗尽，无新中央续票。
- 新版probe完整尾段平均0.16–0.18秒/来源；此前N2约4–5秒为历史不同来源对比，不能冒同源隔离总流水线倍数。
- 用户已于10:16CST回报外部01/02合30651可靠未审记录（含17 needsreg，失败27另报），03–06旧4000加新10079草稿/1失败；与主23165合mixed snapshot67895约30.45%。外部总量来自用户不同观察时点报告，不称中央全量新审计、不重复加pilot或旧1000/5000，不加失败为成功。
- 六份external_live_status/a100-extra-0N.json已经到齐，不再沿用missing旧状态。02:26–02:28UTC本地两轮实采间隔60.2–60.7秒，24GPU各一个compute PID，PID/startticks/PPID/UID/cgroup/exe一致且argv worker标签对应；这是该时点活体，不是无限期当前心跳、也不是新资源释放授权。
- 外部真实publication协议不同：01/02平铺batches/gpuN-run-bNNNN，package/owner_pid/commit_unix/source_row.assignment_index；03–06分层batches/gpuN/run-batchNNN，publications/receipt.assignment_index，无顶层owner。首observer kSdTtO误用分层路径导致01/02没有覆盖，不将其0结果当0产量。修正NOFMYW 02:37:33–02:38:34UTC 62.10秒新5commit100个唯一成功来源，元数据票据范围/归档存在尺寸均核，owner PID与live sample相符。
- uG2HVW 02:40:42–02:50:43UTC 602.37秒原验280成功，05g3 batch025一条BeatThis失败收据缺vlm_called字段引发measurement_incomplete。精确补核该批19成功收据均明确false且scope/index/归档存在尺寸合法，合299成功下界+1失败；缺字段不等于实际调用VLM，不改冻结产物，不给精确吞吐。
- 新外部持续只读observer GukBHF每15分钟运行，02:55:15UTC首baseline24lane已取，已知失败字段缺口继续标incomplete，未当监控进程存活证明主卡已恢复。主xC0HaZ PID597151/start550147663当前实核alive，02:53:19UTC心跳19.8秒且完整15min0新来源为预算排空。
- 主8卡复产需额外新源或获批跨集群排他转移未领来源。8000来源从03–06转主的授权已向用户询问一次，截至本节尚未获得；不能把Stop hook当批准。审批后仍须当地领取屏障/排空在飞/封存未领集合SHA及永久exclude旧自动队列，主中央新票+fresh原生160GiB及PID/GPU/锁租约核验才可生产。未要求外部暂停，未抢领/热改任何有效票，未使用val/test或重试旧sealed。

### 2026-10-27 03:28Z 用户新范围与八卡真实恢复
- 用户新明确推进原始全集排除外部两个集群160000，全集222939减160000为62939；不是转移外部预算。此前跨群8000方案取消且从未执行，新范围包含原val/test仅打标，保持split、training_eligible=false，不作为训练补料。
- 只核allocated会错误遗漏未消费预算。24份main stream/refill/tail summary实际已消费22268（8000+8000+6268），成功22265/3sealed；之前finite900实际1000..1899成功，另外旧N2g2..5各100的ticket和config存在但真实out/双editionclaims/queue根均不存在，400原train确未消费。旧holder config已过期不得直启；需保原ticketid/edition/sourceclaimnamespace的独立v3转换与fresh安全前驱证明，正在准备后续，不重新编号到第二份claim域。early500..999实际统一消费链尚未核全，不能把early保守superset当完成。
- 新原split allocation ms_unified_original_remainder_20261027_v1 已write-once冻结22371：val11110/test11261，共46实际shards一次rawSHA读取（不是45）。八way计划3slot2797、5slot2796；复用frozen fullmanifest全局exactID唯一与旧所有train范围、跨split媒体ID无重叠证明，不重扫160k/22w；诚实不冒新casefold全旧actualset核验。
- 8个新dispatch job真实发布，使用现frozen continuousmanager v2/corev3和原native生命周期，每20源archivecommit、模型一次常驻。03:28Z现场八worker identity/PPID/UID/exe/cgroup/argv/GPUUUID唯一compute应用、父kernelFLOCK/leaseFD及lease心跳全部通过，各提交40条新val草稿共320；latestcommitSHA/来源段/归档存在尺寸真，0流程失败，0VLM、无质量训练批准。N2 worker1004771/1004764/1004773/1004772；N3 worker3124653/3124649；N6 worker2257694/2257693。
- 当前8managed：N2g2 FSZuZZ/g3 6O3Ft8/g4 w3clm7/g5 z6YdWf；N3g0 1SOXZl/g1 qtY8eP；N6g2 apDmPY/g3 2ob3TU。dynamicregistry实际已切remaining phase，monitor03:28:15UTC age54.9s无警告/告警；15min含旧空等和启动期320/h换算1280不当新稳态。正式启动仍原160GiB门，N3预核198GiB为共享node余量不能按两卡重复算。
- 外部16万授权来源/已运行24卡不改、不暂停、不重领。新的main22371真实运行不是监控holder；原400续接和early预算核验继续，目标不是全62939已完成。

### 2026-10-27 主尾段重平衡实际优化
- 用户明确继续优化与充分使用现有资源。06:19Z现场仅N6g2还剩1276来源、其余7slot已完成safe，完整15min120来源≈480/h是单卡尾段，不能把历史8卡速率当当前。
- 规则慢点：两次只读/proc实采FUSE lookup/newfstatat等待，15秒process CPU只增3.06秒且quota throttle计数不增。两批20source/81帧rules约122–124秒，probe约3.6、decode约28秒；源数/帧数稳定。原规则调用在未存在的virtual archive member path上累计14次resolve，画外点重验可20次，与每源5/7秒档位吻合；这支持共享元数据lookup为首要原因，不能仅凭两snapshot断言全部rules时间都属FUSE。无SciPy/媒体/NPZ再读，不盲加CPU/GPU线程。
- 原running ticket自带STOP在owned PID/start核实后请求，未signal。原生提交完整当前batch并排空prefetch，Nhhyte实际exit0、summary drained_not_complete、processed1560/unprocessed1236，resources_safe三flag真。原STOP永久保留，不clear任何claims/锁租约；已领prefix、失败和成功不重发。
- release_receipt.json与allocation.json在ms_unified_unclaimed_tail_rebalance_20261027_v1新独立目录writeonce；核ownclaims/sources仅此命名空间浅层，exact1560 filenames==processedprefix，suffix1236与claims交集0。7way新plans合1236，source字段及原split不变，外部16万完全不动。
- 新独立ms_native_rebalance_dispatch_v1.py SHA4434e82c237cf8e818f9e93c394b4d8768fb3bd831fd85676b5f2056f27fb011，在任何newwrite/资源操作前核release及原六proof SHA/STOP/prefix/suffix/ownclaims；保原160GiB/native锁租约/已完成资源前驱/registryCAS及writeonce。agent21CPU与主48联合回归均通过，不能用测试冒部署完成。clone前驱票必须删除原400 source_budget旧provenance，仅绑定当前source_transfer_ref，防错误归属。
- 7newmanaged N2g2 UP5iT5/g3 7T8Ac7/g4 ug98lQ/g5 VKHJuK；N3g0 khPZed/g1 aDTpWV；N6g3 bgSXV8。真实worker N2=1205367/1205369/1205368/1205337，N3=3275862/3275849，N6g3=2369031。现场PID/start/UID/exe/cgroup/argv、GPU UUID唯一compute、parentkernelFLOCK与leaseFD及latestcommitSHA/source段/archive存在尺寸真；7slot各120可靠新成功共840/1236、0失败、模型初始化1。当前20source批约40–50秒、rules2.1–3.0秒，已恢复多卡有效生产，不冒完全同源隔离倍数或整个62939已完成。
- early来源再次按实际ID查纠正：当前train500/501在外部pilot05/06明确foreignclaims，588在native diag3c sourceclaim已消费，所以整500..999不能新发。497候选502..587+589..999与已核56nativeconfig和5pilotmanifest无交集，但全owner证明未闭合，不据此抢领；不把保守earlysuperset当实际done。

### 2026-10-27 07:36Z 早期497实际补齐及外部监控修复
- 早期预算再次用当前真实63份registered native config selected IDs闭合（含4budgetreuse、新remaining/rebalance）、三张登记表共20引用均被mapping覆盖，497精确ID交集0。复用14外部SHA-bound票据run/rollover/160k范围、5pilotactualassignment、full原manifest全局exactID唯一证明。旧ROI producer覆盖原source只是不同stage，不重领其ROI/VLM预算，也不应永久阻挡未做的统一noVLM新stage。明确排除500/501foreign和588已消费，qualified范围train502..587+589..999共497。
- ms_unified_early497_20261027_v1 新allocation/verification+7plan排他writeonce，七路各71。独立ms_native_fresh_source_dispatch_v1仅复制现tail资源安全逻辑并清除前驱source_budget/source_transfer_ref陈旧标签，原160GiB/holder/PID/FLOCK/lease/sourceSHA/registryCAS保留；28CPU测试通过。未修改任何旧预算/claims/STOP/生产源码，外部16万不变。
- 实际七路均71/71完成，497generated、0sealedfailure（496partial+1needsreg仍未审核）。07:28–07:30UTC现场逐slot核全source顺序、prefetchdrained/exit0/native-safe五flag、latestcommitSHA+来源段+档案存在尺寸、当前restoredholderPID/start/argv/UID/exe/cgroup及GPU onlyholder通过；worker已退出，不能把manager仍running冒生产。单路whole154–169秒为实际有限批次，不预测日期。
- 外部新ms_external_incremental_patrol_v2.py只读cursor监控已独立21CPU通过。旧01/02flat目录每GPU重复enum且>1024拒绝，newflat每node一次enum、node有界12000，首次读每lane最近2metadata，后续只新名/缺失commit有界12每lane；保持真实票SHA/index/IDpath/approval核验，archive仅existence/size/SHA声明，不读tar/media。失败receipt缺vlm字段明确evidencegap、不补false、不吞同批19success。
- 实际--once IqZLGs基线完整成功：6nodes/24lanes/3263名字（跨节点）/18次目录枚举/48小commit，0GPU/信号/claim/归档payload，无measurementgap，baseline新增0不加产量。首次冷基线wall281.9秒只记首次准备成本，不当处理吞吐。持续new7DDVvg在ms_external_incremental_patrol_v2_main_20261027_v1已经07:36:15UTC首sample0完成、无errors；选择config明确6个queue-bound160kedition，不用旧报告推currentGPUalive。pointer runtime/ms_external_monitor_registry_v2.json；legacyGukBHF仍保留但已有directory-bound缺口，不再作为primaryrate凭据。
- 原val/test22371在processedprefix加rebalance1236实际集合交集0后已完全覆盖，22368成功/3sealed；400旧预算成功400。加此前main23165和本次497，main阶段成功条目计46430（需以冻结sourcekeys更新全群总量，不将不同stage/诊断/旧failure加完成）。62939范围内既有pilot缓存/诊断/外部run/rollover须按stage分别审计，不宣称整原62939皆获可训练质量批准。

### 2026-10-27 10:16Z MotionScript两项会计缺口与实际资源使用
- 旧rollover两个epoch各5000来源，单次511commit元数据核账：01发布4987/failedreceipt7/独立sealed4/unknown2；02发布4987/failedreceipt2/sealed11/unknown0。状态来源键互斥、跨worker交集0；archive只存在/尺寸/声明SHA验真，不称payload内容或视觉质量通过。原62939范围62903未审规则结果、31失败封存、3诊断覆盖单列、2产物终态未明。
- 主按原ticket claims_root只读两精确sourceclaim，DTxRgnSZZqFSijgW(index3952/ordinal15904)及KYBwQcXqARLIoECt(index3956/ordinal15912)都实际已由外部01/GPU0/旧PID776882领取，原row/assignmentSHA一致且automatic_retry=false；不能把它们算未领或重新上GPU。主四既有恢复/会计文件无两source的完整产物定位，需要外部01本机精确补旧run日志/现成完整缓存的locator和终态；无产物证据不执行publication-only。
- 新已授权VLM v2后处理三pair实际运行：N2g23 jKcXsu、N2g45 8cVgQy、N6g23 wfPAAJ，共6卡、每pair60不同来源，32B一次常驻模型。主在首N2pair加载后再fresh核g45的共享160GiB资源门而非复制预算。首8→17→23→50可靠source outcomes连续增加，10:16:34UTC检查点共100实际chatcalls/28结构候选22需视觉复核/0sealed流程失败，所有approval/trainingfalse，不属于MS新媒体/新source提取产量。
- 六GPUrank进程祖先归属actualworker、parent kernelFLOCK与lock/leaseFD曾实际核真；每分钟只读监控DYCnLh/PID2422191，output ms_postprocess_batch180_patrol_v1/latest_summary.json，含workeralive、结果进度、600秒无commit告警及最终native-safe。主rule xC0HaZ继续观察终态不误报仍生产。sourcepostprocess claim与旧MS claim分开且不复用原失败budget。
- 当前新MS可执行预算为0；主必须得到新的来源域或旧两source已存在完整产物的可验证证据才能再执行MS，不用监控活体/次要阶段输出伪称主MS持续生产已满足。外部160000原队列不动，旧STOP/claims/失败全部保留。

### 2026-10-27 10:42Z 外部epoch互斥交接与下一版源码接管
- 01/02交接索引 external_epoch_status/epoch_handoff_index_1793093283478066282.json，SHA57d7a61c43aa66faa4ab596a6acc4541d3f3e738034bb4871225d386acfb934c。只读核24份source-key原字节SHA/长度/计数，4assignment/ticket、2summary；按[split,id]精确匹配，各epoch六分类互斥、union=分母且跨epoch交集0，整体90000不同键。8lane原commit SHA/receipt绑定与archive仅stat尺寸通过，未读媒体/NPZ/tar payload，不冒内容/视觉质量审核。
- 17:19CST非原子快照：01 rollover=4982partial+5needs+11failure+2unknown；02 rollover=4985partial+2needs+13failure。01 clusters=26961partial+14needs+5failure+64claimed+12956notclaimed；02 clusters=26195partial+13needs+12failure+73claimed+13707notclaimed。published63157=partial63123+needs34，failure41=26失败commit+15存储封存sidecar，claimed137、notclaimed26663、unknown2，合90000。新clusters80000仅published53183；rollover9974已含原62939口径，不能跨群重复加。
- 用户当地补核两个旧rollover source：原领取worker退出，未找到完整本地结果，仍unknown而非neverclaimed；automatic_retry=false，不重新GPU打标、不无证据publication-only，原claims/STOP/失败不清。索引旧身份仅证明17:19CST快照，不证明当前活体。
- 03–06用户09:36UTC回报新队列可靠提交23620/80000、待深验40、未提交56380、15min新增460、原16模型PID继续。属于用户异时回报，未中央全量source-key/成功失败细分复核；不能全称规则草稿，也不直接与增量monitor相加。
- 03交接目录 external_ms_unified_clusters80k_20261027_v1/results/a100-extra-03/integration_handoff_20261027T093320Z_7dbb8474/；frozen_code.tar 348160bytes、SHA5eb7839dbbcdd887fd0fd9fa6e8503a1f4d5dc499022019c2e734677cd6c813a，11代码成员SHA/长度、manifest/commit与NEXT_VERSION_REQUIREMENTS只读通过。runner95eff3b6…/relayc8d36e48…；实验e794重定位未重跑，c5a重建匹配历史SHA但不是恢复的原历史文件，4源发布实验不冒生产加速。
- 唯一writer仅新 external_ms_batchrpc_next_20261027_v1/ 独立runner/transport/tests：20来源有界RPC、成功/receipt-only失败/已有claim混合批、原生产schema绑定、commit-last完整回读、publication-only复用同archive和原commitbytes不模型重跑。保原5算法SHA，新runner审计wrapper保存BeatThis拒绝数组与predicate、独立有界诊断发布，不放宽校验。尚在实施，未完成冻结验收、未部署。
- 原24外部worker不停、不热更新、不追加第三票、不启动GPU4–7；09:33空闲卡报告是历史证据，不是fresh准入/启动授权。新版本还需独立安全epoch drain/commit/精确PID退出/新冻结票边界，测试不等部署。
- 主3×60后处理已180可靠终态、360实际模型调用，116结构候选+64需视觉复核、0流程封存失败，每pair模型load1；10:42:17UTC既有DYCnLh监控三worker退出/resources_safe后自然exit0。10:44:07UTC独立全180 request/raw/normalized/commit回放、exit.summaries全值比对，以及6实际restoredholder完整identity/UUID唯一compute、FLOCK释放/FD无残留、原lease mtime恢复均通过；三worker真实退出、exit0，pending0。完整验收不是仅monitor通知。运行登记已标worker completed_native_safe，旧parent wait不冒生产。所有approval/trainingfalse、MS_new_unique_sources=0。后续300现成草稿排除186既有来源后，三路各38共114已实际接续（Q5zCsP/7j7195/SJOgZZ），排他allocation SHA50a5a6c9477ba079426a5e483f9ca337635d60a346ffc49b9af116cf8baeab09。11:03:36UTC全worker/parent identity、GPU祖先、FLOCK owner+FD/leaseFD与最近commit/result SHA独立核真，当时36outcomes/72calls；11:21:16UTC有限监控XsEr4n自然exit0，三路38/38共114终态/228calls，81结构候选+33需视觉修订、0流程失败，worker退出/native_safe字段真。11:24UTC另派独立全114回放及每GPU最新restoredholder验收，尚未完成不称fullreplay通过；主运行登记已改completed_pending_final_replay。无MS新来源产量。RPC旧候选虽有29项测试，独立审查仍发现原core实际加载SHA偏离、sparse漏拒、partial全量重发、prepared恢复前GPU查询、audioSHA与claim body绑定漏项六阻断，未接受冻结或部署。已停自己的旧实施subagent（未碰生产）并交独立v2目录唯一writer修复；旧文件保留。两unknown人工新重做预算未获授权，UI询问超时不算授权，继续unknown/不重领。








### 2026-10-27 12:03Z RPC修正独立验收与v3不可变候选封存
- 主独立真实20source生产schema实测发现旧RPC把plan放argv：20success101 publications/20receipts command160695bytes，当前Linux实际errno7/E2BIG；20mixed120679bytes不代表上界安全。旧45tests主a8zwL2虽OK，但最终pre/postcodeSHA变更断言失败，未接受冻结验收。
- 修复者误在已冻结v2修改runner/transport/tests并覆盖旧manifest/CPUreceipt（五文件）；历史45项SHA只保留为事故记录，不能宣称历史writeonce完好。旧源码与回执未回滚、未删除遮掩；自有writer已停止，不碰生产。v3 source_provenance与独立验收收据明确history_integrity=false，后续不覆盖任何v3冻结证据。
- v3目录 external_ms_batchrpc_next_20261027_v3 是corrected v2文件名/字节的独立快照，保11py原basename避免code_signatures/auditrefs额外变动，不等于生产v2部署。13snapshot成员O_EXCL/NOFOLLOW、0444与fsync/字节回读完成；非OS WORM承诺。manifest SHA04e9f7b204c4fe59c7b1eee945b5279d370261ac6aa7f8b7b1e73638cb623151，provenance SHAf5d4b4cbd7396f0cc13596cf9153a709d854c255ef38c3dc3074ea409ad95a8e。冻结manifest pending字段不热改，后续认证以新独立收据为准。
- 修正代码保原5算法SHA，实际加载原core85ea6781…；透明BeatThis tap原样返回输出/metadata，拒绝仍由原core执行。独立诊断强绑原row/index/audioSHA/model/policy/dtype/shape，有界2MiB NPZ/64KiB JSON；超界/object明确evidence_gap，不能称所有原数组全覆盖。支持20mixed/receipt-only失败/已有claim无产物，普通PAX通过/sparse拒绝，partial只补缺项，prepared先核commit无GPU/claim/generation。
- 全部plan-bearing RPC（inspect/publish/verify）改4byte length+有界JSON stdin header；固定command35209bytes，20success header94496bytes/mixed64484bytes，<=4MiB。selectors非阻塞stdin/stdout/stderr同时有界消费，whole-transfer monotonicdeadline，ACK4MiB/stderr8192bytes，fake本地child早拒/断流/超界/堵塞timeout通过。不真实SSH，不触真实媒体/权重/模型/预算。
- 主9DmjDX在修正v2路径完整52独立测试通过且pre/post11code与两sourceJSON不变；再从真实v3路径fX7DON完整52独立测试全部通过，0failure/error/skip且11code/manifest/provenance前后字节不变。两次都禁止gpu/model/claim/copy/extract回调=0；第二次确为v3入口，不以字节相等冒新测试。独立新收据 independent_acceptance_20261027T1203Z.json SHA b981a8594d6976b2e12ae469db8785df4937110ec6ab389683ab1cc182baa9bd，主回读收据/所有源码/两proof SHA匹配。旧4sources实验4.28x未作为生产吞吐证据。
- 当前implemented_not_deployed/deploy_ready=false，CLI仍拒绝生产；新safeepoch/drain/真实前驱退出/排他来源预算/fresh资源准入与部署均未验收。没有新授权预算不能重新上MS。两旧unknown已向用户询问仅两条新人工尝试一次，尚未答复；原claim/31失败/外部16万不重领不转移。
- 114后处理独立No54rh在11:34:39UTC完整回放与exit.summaries全值一致，38/38各76calls，总114/228calls，81结构候选+33待视觉修订、0流程失败；6GPU最新holder/FLOCK/lease真安全，workerexit0。已从pending标completed_native_safe。未人审不可训练，MS_new_unique_sources=0。
- 11:55:53UTC主八槽再次可信SSH实采均MS worker退出、current GPU onlyexactholder、原managerwait/锁释放/leaseFD无残留，近15minMS成功0。12:03UTC主monitor实际活体/heartbeat<180s；外sample17最近窗口1560strictsuccess/0failure，历史stickygap仍rate=null、current GPUlivenessfalse。六外部sample仍02:28/02:55UTC旧证据，age9.1–9.6h；不得据此证明此刻24GPU活体或停机。无新部署、signal/抢卡/清claims/STOP/锁租约，不扫22万/媒体tar/权重。

### 2026-10-27 12:34Z 纯CPU安全epoch预检独立验收
- 仅新增 external_ms_epoch_preflight_20261027_v1/epoch_preflight.py 与 test_epoch_preflight.py，纯内存接口 validate_transition_contract(contract,evidence_blobs,trusted_expected,now_unix)，无CLI/启动或切换入口。旧RPC/v3/生产票据/队列/claims/锁租约不改。先前73项候选漏检：casefold合并身份、freshGPU遮旧claimclose、CPU冻结proof变动、九代码scope/预算下限等；主直接修正，首轮旧测试不计最终验收。
- 主固定源完整84tests通过，失败/错误/跳过0，源码前后SHA不变。独立只读review又跑84suite，加89案例（6只结构正例、83拒绝），共178次真实函数调用在文件/进程/网络/clock patch与audit守卫内，forbidden events空、无新增GPU/model模块；输入未变且重复调用确定性一致。Python启动/导入和源码Read发生守卫外，不冒整个shell零I/O。
- 当前sourceSHA：epoch_preflight.py ebe8a7c0222a82af3052912af8dd2da4ce1278dfdcf3e0b4dd2beb106fd71888，test_epoch_preflight.py eebf51896afe21627f1999d6908c64c68066bfb8f0bb5a7b7039f47a46820ac0。新增独立收据 independent_cpu_acceptance_v1.json SHA01951147ea7065b6031e853ed6b71657d1fa2eed35ad2f7be11fb4594fd70555，主回读收据与两源码SHA匹配，不覆盖任何旧证据。
- 核[split,id]精确身份、original index%4/完整有界lane projection/exactsuffix、positive unique ordinal、原sourceclaim namespace保持、九代码与v3 manifest/独立report绑定，report冻结不变与五禁止callback严格int0、PID与claimclose同样<=120s；不能新GPU快照遮旧claims。外部门32GiB有效memory底线、本地盘/8GiBowncache、disk=8GiB+max(0,32GiB-retained4lanecache)、节点共享预算一次计，不代替主160GiB门。
- 原四full assignment只stat各11189750bytes；预检允许这类独立锚定大manifest元数据ref，不读载荷、不降低ref.size到投影大小；实际blob仍单8MiB/合32MiB。主也用实际v3独立RPC收据与九code refs验证candidate stage schema兼容，不假称验证真实transition/审批/资源。
- 结果恒deploy_ready=false、may_create_claims=false、full_original_assignment_verified=false、actual_process_exit_verified=false、actual_physical_lease_verified=false、full_commit_claim_artifacts_verified=false。结构正例只structurally_valid_not_admitted；缺三类独立本次授权、staleclosure/错误source/claimed/unknown混入、namespace重置、GPU4-7/非空GPU/不足预算、代码篡改均blocked。SHA与声明字段不替代签名或真实用户许可，必须当地在既有锁FD下再次核精确退场、claimquiet、UUID/proc/resource。
- 本项离线技术准备完成，但没有实际部署、安全切换或新MS预算。两旧unknown人工新尝试授权此前只问一次仍未答；外部活体文件仍旧，不能本控绕SSH或猜权限。整体打标goal和只读监控继续，不把84/89测试或既有后处理计作MS新增。

### 2026-10-27 01–02追加来源接口核查结论
- 只读核查已结束。01/02真实队列位于 external_ms_unified_clusters80k_20261027_v1/a100-extra-01.queue.json 与 a100-extra-02.queue.json，schema=ms_external_authorized_rollover_queue_v1，原1000→rollover5000→clusters40000三票SHA绑定；request_more_when_remaining_below=1000仅请求补料，不授予无限预算。补料应称第四票或独立新epoch，不套03–06的第三票限制。
- 当前可读交接中只有五算法code_refs，无01/02实际consumer冻结源码；启动时queueSHA/序列固化、耗尽后重读、STOP/drain及新epoch入口均未验证。当地路径在控制容器不可见不等于系统不存在；未猜SSH路由或绕hostkey。现有三票继续无需新指令，不热改任何队列。
- 中央native dispatcher仅主N2/N3/N6原生合同；已验RPCv3仅03–06两票/exact20000且CLI禁部署，均不能直接作为01/02第四票接续入口。
- 用户新反馈异时去重：01/02 rollover+clusters草稿79847减原rollover9974得新80000草稿69873，封存新失败27，未闭合10100含在途；03–06新草稿30312、封存8、未闭合49680。与原62939的62903草稿合163088/222939约73.15%，不是原子当前总数或动作质量批准，不加监控增量和后处理。
- 01/02各5000仅是待确认建议，不授予来源转移或停供出worker。若取03–06当前80k，须仅选中供出槽位正式关闭领取、排空在途、精确PID/start退出和持久no-resume后fresh核真实未领suffix，再保持sourceclaim防重域冻结release及新epoch。不可直接裁仍运行manifest，也不能保证24原worker全不停同时转移。两旧unknown人工重做和主复工新source预算仍未批准。
