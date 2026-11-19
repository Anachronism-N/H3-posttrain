# 第二阶段资源停靠台账（2026-10-28）

## 用户明确停止全部打标/重写并释放相关资源（2026-10-29）

最新授权覆盖停止当前与遗留标注任务、释放其资源；此前N4不kill与N5标注保留卡改为此轮全释放范围。只处理已核标注脚本/manager/引擎与项目hold_gpu.py，未知无关进程不杀；不删除任何数据、claims、预算、结果或模型。已在停止前并行核N1–N8真实PID/argv/start_ticks/config/queue/gpus。

顺序：全局未领取来源置stopped_unclaimed（assigned/终态不变），所有原生feeder按实际代际STOP停止领取，完成当前原微领取；原生manager自然消费当前已领ready，完成可可靠保存的尾部。遗留空等旧worker/manager先按pidfd TERM精确身份结束，保留原输出与未提交缓存；已核引擎/TP存活且不再有生产parent时TERM，仅本任务祖先或恢复阻塞凭证内已核旧引擎，绝不KILL或进程组杀。native-safe三项真后停止所有项目占卡holder让manager结束，锁/租约由原生cleanup恢复。不再启动任何模型、重试或新source claim。保留结果上传器直到当前节点结果可靠ACK、然后停止相关CPU服务；已有人工查看server可保留供访问。外部六节点中央新领取暂停，已有assigned上传继续；物理释放需独立核远端，不假称外部48卡已全部空。


## 同prompt Qwen3-32B三例对照资源台账

用户授权同prompt试纯语言Qwen3-32B。实际卡安排N2g01两卡做固定3来源；现18生产无独立满足160GiB的卡对，N7g67只有153GiB虽safe不能绕门。N2原feeder代际STOP为node级，先自然完成原已领取ready并等4manager native-safe/全部旧ACK，再新phase恢复g23/g45/g67三生产副本，其他14生产全不动。g01按独立原生manager运行纯LM三例（原完整request/schema/temperature0/max4096冻结，thinking关闭），完成自动safe恢复holder，不重复领取3来源，旧生产budget不清。模型从N1已完成官方下载文件一次复制到N2，不重下载，型号改变仅diag代码modelref采用独立Qwen3完整文件绑定，冻结resource_manager.py不改。N4、N5保留卡不动。


## 当前集群新增N2g01与批64有限试验（2026-10-29）

用户要求评估当前集群更多副本/batch提速并检查结果。既有N4不可kill、N5保留卡不动、原残留可释放授权保持。N2g0旧manager已有native-safe三项真；g1原MS生产manager3468653已worker_exit0，但恢复被单个项目hold_gpu.py PID1154900挡住，锁/租约由原manager保留。GPU无模型。拟先核该holder精确pid/start_ticks/CVD/UID/argv/UUID和TERM未屏蔽，pidfd TERM仅这个已核项目holder，让原manager自行恢复原holder及三项safe；不修改锁/lease、未知进程不信号。随后N2g01新独立code/phase同v4数据库，64调度常驻模型按32/64交替处理有限192–384个首次领取来源，原source O_EXCL一次预算不变、其他17副本继续。三case内容验收独立且未批准全部正文。


本文件是停止前只读盘点。本轮未发任何信号，不证明已停止或已释放。
- 先完成本地输出、单上传器与动态领取，再分A/B/C扩容。
- 停止前重新核manager PID/start/UID/exe/argv/cgroup，只pidfd TERM并等resources_safe三flag全真。
- 不删锁、租约、旧结果、claims和预算；已领、failed、unknown不重派。
- 未核数量不是0，已领/完成/未领exact集合及剩余位置必须在停靠前补齐。
- N1小JSON两读约1.3ms，但仍6个git D/FUSE等待；不证明Ceph全部稳定。
- N8两读约0.92s/1.7ms，只读样本不证明持续写盘稳定。
- N2GPU1 manager3468653持锁，restore_blocked因外来holder1154900占GPU；原expected holder3311219；无safe，不准入g01。
- N7 headroom约152.78GiB<160GiB，不启动。
- systemd offline；tmux安装但无已有server，持久启动接口待验证，不回退会话后台SSH。

## 2026-10-28 当前新版上线与停止前补账

- 当前已真实上线12个原版32B BF16 TP2、32会话副本：N1四、N2二、N3一、N6一、N8四。全部GPU请求、原始回复、预算和结果先落节点/root；每节点单上传器，可靠ACK之后由独立CPU服务闭合ledger。
- 已有3936来源自然耗尽，节点在原生safe三flag＋全部本地来源共享ACK后自动接续新的25965来源池；不是恢复旧配置、重新claim或重新推理。
- 原预留给外部03–06的50000来源仍保持独立排他，没有被主群接续池领取。
- 实时只读命令：python3 runtime/stage2_progress.py --watch。新共享commit与旧后处理summary来源键统一去重；历史草稿175276只作为非原子估算分母，不能称当前全库exact。
- 原30分钟墙钟窗口保留起点和逐样本日志，采集扩至N8没有清零。A→B→C扩容仍以至少30分钟真实窗口为准。
- 以下是此次停止前的实际补账快照，至此未向B/C发送信号。source_sets.json保存exact集合；快照后有新增时以安全退出后的补账为准，不能把旧unclaimed快照重新分配。

|旧产线|已领|已完成|未领|统计口径|停止前证据|
|---|---:|---:|---:|---|---|
|N2 caption_two_stage_local_pool2_n2g67_v1|7944|5304|0|共享池全参与者；本worker已领=已提交=3072|N2:/root/stage2_stop_ledger_N2_20261028_v2/caption_two_stage_local_pool2_n2g67_v1_snapshot.json；SHA657ff3f00dc1f5921f2958db696b90b42a37ad7f2f96f163ea8ac4acf38fa7ef|
|N4 caption_two_stage_local_pool_n4g45_v1|5709|3842|1|共享池全参与者；本worker已领=已提交=2463|N4:/root/stage2_stop_ledger_N4_20261028_v2/caption_two_stage_local_pool_n4g45_v1_snapshot.json；SHA94a00fc2d18aefad774f0265f0853f0f76f01c77e668e1a99c989528507b5066|
|N3 full12k_visual_s0_v1|2930|2929|70|本shard source目录与completed文件名|N3:/root/stage2_stop_ledger_N3_20261028_v2/full12k_visual_s0_v1_snapshot.json；SHAbed28a77609bfc50be3509d044f4e4967b543a9aec73dbf69b623248502a1e99|
|N3 full12k_visual_s1_v1|2929|2928|71|本shard|N3:/root/stage2_stop_ledger_N3_20261028_v2/full12k_visual_s1_v1_snapshot.json；SHA138095bcb42e635f541477e66e111036498deaffbde1842f5d562e2aef2d02b3|
|N6 full12k_visual_s2_v1|2928|2927|72|本shard|N6:/root/stage2_stop_ledger_N6_20261028_v2/full12k_visual_s2_v1_snapshot.json；SHA2ca0d357e74e4f2c76ffd104b22a8d27bb02577bf97e60d97a23e09a218638d3|
|N6 full12k_visual_s3_v1|2929|2928|71|本shard|N6:/root/stage2_stop_ledger_N6_20261028_v2/full12k_visual_s3_v1_snapshot.json；SHAae305f7464da14bc603c4412912a1921c40db324f26c3361119312d7abae3924|

本轮补账不解析视频、caption或原tar，不修改sourceclaim/预算。原schema不同的旧产线结果不并入当前VLM后处理完成数。停止时必须再次核admission中完整PID/start/PPID/UID/exe/argv/cgroup及config SHA，只向原生manager发pidfd TERM；三flag真之前不复用卡。

## 2026-10-28 A阶段30分钟实测与B阶段清退阻塞

- N2本地首个新全局窗口报告：/root/stage2_progress_service_20261028_v1/first_30min_report.json，SHA d0f0e58907d91416cf8d725f82f30e094a4012dd7bca2a9038b7e34cc7838ccd。
- 起点1793161068.8812354，终点1793162893.2907948，墙钟1824.409559s；VLM来源可靠提交从8002到13988，新增5986，11811.821467条/h，包含窗口内加载、空等、旧3936到新25965接续与全部停顿；errors=[]，不是API请求数或GPU驻留数。
- 该时点12副本实际存活已加载，准备696来源；本地已完成待上传218；上传耗时>=5秒7次，最长6.51644秒，不把它声称全部Ceph停顿。
- A阶段已达到用户>=5000/h首个窗口门槛，不代表17w+全量完成或后续所有窗口持续达标。
- 此后严格按已核snapshot/notes台账，只向N2 manager4113508及N4 manager1714279发送一次pidfd TERM；未向其余/未知进程发信号。
- 两原主worker已退出；原manager仍存活、保FLOCK/租约心跳，resources_safe未出。N2 GPU6/7仍被PID4113920/4113921占用，N4 GPU4/5仍被PID1715760/1715761占用。原生restore_blocked明确保锁，不可复用这4卡。
- 四GPU worker的stdout FD inode与各自旧worker.runtime.log一致，原日志有对应PID，UID/cgroup匹配；当前/proc CVD字段为空，不能假称CVD验证通过。引擎父PID分别4113906、1715729仍在；尚未额外发送信号。
- 额外清理旧vLLM引擎/worker超出此前“只TERM原生manager”规则，已询问一次授权。交互UI超时，已在会话请求直接回复“允许清退残留子进程”；当前未获确认。其余12个新版副本继续处理，不能因这4卡受阻停止整条产线。
- B等待脚本5sM5fs/9VKaug仅因180s内无safe返回失败；不是原manager或GPU worker已终态。不得重启原B配置、重发旧claim或删租约排障。
- 已另外冻结主群37610原生未消费草稿索引（train15242/val11109/test11259），进入新原生取料准备；索引不是VLM完成。

## 原生37610供料v3切换前台账

- 当前原生JSON批包供料v2存在跨进程复制竞争；N1g67/N8g67供料失败，原source claims与预算不清理，不恢复这两个旧配置，不重派其失败/未知来源。
- v3只新增同一输入批包的跨进程kernel FLOCK；实际8个Python进程竞争同一tar CPU测试全通过（仅1份local tar）。原GPU模型/prompt/12帧/输出协议不变。
- 来源全集与现状在N2一次SQL快照冻结：/root/stage2_feeder_v3_handoff_20261028_v1/source_sets_before_stop.json，SHA7511ebe28c49e8622bcced3058b47467cc4335c4f89d0769361188c24f20567e。37610总来源，unused32930；已领完整split/id/state/owner/batch存原JSON，不能把快照未领重新发到独立池。
- 各旧owner已领/终态/assigned：N2g23=480/398/82；N1g01=480/384/96；N3g01=472/350/122；N6g23=480/400/80；N8g01=480/396/84；N2g45=472/352/120；N1g23=448/368/80；N8g23=320/288/32；N1g45=424/352/72；N8g45=416/352/64；N1g67=112/96/16；N8g67=96/64/32。此处assigned包含待上传或已领未执行，不据此自动重试。
- 旧各节点local_root=/root/stage2_dynamic_native37610_<N1/N2/N3/N6/N8>_20261028_v1，deployment.json绑定config_ref；config中queue=<root>/queues/<phase>、out=<root>/out。各owner original feeder config同root，旧程序与code_sha256冻结不热改。
- 正常旧槽位只写自身FEEDER_STOP请求，准备线程完成已领packet并FEEDER_DRAINED，原GPU自然消费所有ready后退出，原manager保完整恢复逻辑。失败槽位先核原manager safe三flag；旧已领失败不重试。
- 新group=native37610v3，owner标签独立，领取API18998与原native37610 API18996共享同一/root/stage2_pool_native37610_20261028_v1/pool.db，只取unused，不重建pool或重新分配旧来源。外部50000始终不同source group。
- 新配置每节点/root/stage2_dynamic_native37610v3_<node>_20261028_v1。必须原manager safe三flag、oldworker已退出、所有已有source commit共享ACK且fresh160GiB/tmp8GiB后才启动；失败留下的未完成claims/预算永久保留。
- 至本节写入时尚未写正常槽位FEEDER_STOP。以上是停止前记录，不证明已停；后续以transition_complete和fresh live为准。

## 外部03–06可执行包已交付（未冒称当地上卡）

- 新完整启动包：ROOT/external_stage2_vlm_50000_20261028_v1/stage2_external50000_runtime_v1.tar，50176000bytes，SHA69adf517f166ffd1c19728003892e897d57e138dfb5861a23d88c9129f9ec7a1。旧core_bundle保留历史，不应继续用它请求当地开发。
- 新包内stage2_external_launch.py包含relay/node可执行CLI；外部节点owner标签、同一个SQLite50000来源池、03输入relay、单节点可靠批上传、空卡TP4原生manager已齐备。
- 精度/模型原版32B未量化BF16，唯一model runtime变化为TP2→TP4适应40GB，32会话/12帧/16384长度/1800输出/prompt原样。
- 隔离CPU实际验证真实HTTP batch upload→sharedcommit→ACK→同池终态、重复claim拒绝、鉴权拒绝、ACK重复上传不模型；真实kernel FLOCK与pidfd自己子进程清退及safe三flag测试通过。没有把这些测试当外部GPU生产证据。
- 外部GPU生产尚未核实；部署meta明确production_started=false、external_gpu_model_verified=false，等当地模型load.receipt和真实sharedcommit回报。
- 可信外部03 relay入口来源为旧冻结transport的RELAY_HOST；本端以BatchMode/StrictHostKeyChecking尝试，因现有known-hosts没有相应hostkey而拒绝。交接指定/tmp/ms_pilot_relay_known_hosts本端不可读；未ssh-keyscan采纳未知密钥、未关闭hostkey验证、未猜外部地址。已在会话说明需要已有可信known-hosts路径，或者由当地代理执行新包。
- 主群供料v3已在N1四/N2二/N3一/N6一/N8四共12副本真实加载产出；上一快照可靠VLM来源42955、30分钟10148.08/h，仍未全量完成。新v3跨进程批包锁已防止原copy竞争，旧失败不重试。

## /proc/locks真实根因与原claim恢复（2026-10-28）

- 用户报告10个native37610v3槽位退出。实际worker.runtime.log不是文件丢失，而是安全门无条件将/proc/locks第6列按dev:inode解析；带“->”的kernel等待者记录列移位导致ValueError。manager随后尝试不存在的summary才二次报FileNotFoundError。
- 修复入口stage2_local_dynamic_native_manager_v3.py仅将等待者/非本manager FLOCK记录跳过，真正持锁PID+dev/inode仍严格验证。真实4个kernel FLOCK阻塞进程的CPU回归复现旧崩溃、新解析持锁者匹配通过。
- 原N1四/N2二/N8四槽位resources_safe三flag已真、旧worker已退。新的phase为旧phase追加_kernel_lockfix1；不重启原配置、不删原failure.json，不发新sourceclaim或换原owner。
- /root/stage2_dynamic_native37610v3_<node>_20261028_v1/deployment.json保存原config/queue/out；新config同root但queue独立新phase，runtime_bindings/<owner>.json指新config。新worker的load/progress/summary按phase分目录，原状态不覆盖。
- 原准备包逐包核原owner、持久consumed_ready与source.claim.json：绝大多数原ready来源从未有模型预算；新入口仅消费这些零model-budget包，已consumed跳过，不可能重新推理已完成来源。
- N8g23另32来源已有visual_audit结果，在caption_fusion准入门之前中断；原视觉结果/原预算保留，已metadata-only写failed_sealed_no_retry结果与commit并标记consumed，恢复模型调用0。不重做已完成的视觉阶段，不重试融合预算。
- 原供料进程（v3包复制锁已修）复用，不启动第二个feeder；它继续从同一原native37610 pool排他领取unused。新manager有fresh160GiB/tmp8GiB、准确restoredholder、原生FLOCK/租约与只对已授权holder的pidfd生命周期。
- N3/N6两槽位仍原票继续；本次不停止正常运行的模型。修复完成以新live/load和后续真实local/sharedcommit为准，不以CPU回归或计划冒称已恢复。

## kernel锁解析修复已实际恢复生产

- 十个受影响槽位已由独立新queue/config恢复，原owner/准备包/预算/结果均不改；N3/N6正常模型未停。新phase追加_kernel_lockfix1，runtime_bindings指向新活配置，不把原failure.json覆盖成成功。
- 实际十个新worker分别为N1:1089722/1091297/1093503/1096201，N2:1944802/1945758，N8:4190353/4191273/4192450/4194203。首次逐槽位复核全部活体/load真、各新增32–96来源、无新failure；之后继续增量。总12副本保持在跑。
- 实时进度已默认仅显示每对卡最新活配置；--all仍可查完整历史失败/终态。历史worker_state与新按phase分目录状态的来源键合并去重，速度仍包含修复停顿，不清零窗口。
- 32条视觉已完成、融合前中断的结果已保原视觉证据，只metadata封存failed_sealed_no_retry，新增模型调用0。失败封存总数从2增到34，不把这些当完整清洗成功。
- ACK闭合的CPU服务另修复require漏导入；只对已有sharedcommit/ACK补ledger终态，没有重跑模型。已实见多批50来源ack_ledger_closed，models_called=0。
- 外部包相同锁解析修复版为ROOT/external_stage2_vlm_50000_20261028_v1/stage2_external50000_runtime_v2.tar，SHA233e5695576643ac19b6e0a3418602ca53d58bf577e27c493d534b6757dde82d。不得首次部署旧runtime_v1。
- 后续34803来源已全部ready（0取料失败），manifest SHA725314d0f84e2a6ce2001ff22b4b6c7b4e2287fc9d90543e5d2889f31f47da79；N2同池/root/stage2_pool_externalrem34803_20261028_v1/pool.db，端口19000，claim0。五节点持久controller提前等待当前最新活绑定对应native-safe和全部ACK，再用修复后的manager_v3切入，当前模型不停止。
- 另外15973条旧run/rollover未消费VLM草稿已排他登记，source_index SHA850767b0f20135f12b639084c34763e5315dec32b7b05608e380f608c073fd27；N5正在一次取料，为后续接续准备。不是重做原MotionScript。
- 全量仍未完成，外部03–06 GPU启动仍待当地执行/可信SSH钥匙路径；N2/N4旧残留引擎清退仍等待前述额外授权，不拿持续目标越过只TERM原manager的限制。

## 当前全量流水线已持续接续（仍未全量完成）

- 当前主群12个修复后模型继续原生37610同池处理；其余队列按顺序native37610→externalrem34803→legacy15973，不提前给同一source创建新claim。
- externalrem34803全部ready、0失败，N2中央pool/db与API19000持久存活；五节点/root/stage2_externalrem_rollover_service_<node>_20261028_v1已真实启动，核最新runtime_binding而不是旧已失败配置，等待前驱原生safe+已完成source ACK后自动启动。
- legacy15973全部ready、0失败，manifest SHA850767b0f20135f12b639084c34763e5315dec32b7b05608e380f608c073fd27；N2/root/stage2_pool_legacy15973_20261028_v1、API19001已持久启动，claim0；五节点/root/stage2_legacy_rollover_service_<node>_20261028_v1提前等待34803deployment并自然完成，当前没有模型停止信号。
- 两后续池分别/root/stage2_ack_externalrem34803_code_20261028_v1和/root/stage2_ack_legacy15973_code_20261028_v1有独立CPU ACK服务，models_called=0。当前native37610原owner及v3owner共用同一DB，并只补已有sharedcommit的终态。
- 实时进度v5已经包含五个实际动态池，保留旧各版本样本/原30分钟起点；输出默认当前每对卡最新配置，不把旧failure列成新故障；--all查询历史。
- 2026-10-28当前其他节点只读资源复核：N7有效余量158.15GiB仍小于160，不准入；N5的0/1/6/7按用户保留准备任务，不在这些卡上新增32B；N4三pair被union_pool3实际模型占用，旧safe文件不代表当前空卡，未盲目释放holder或未知引擎。
- 外部50000实启动仍受本端已验证hostkey缺失阻塞；已直接在会话询问一次可信known_hosts路径，或者当地代理执行runtime_v2。等待不影响主群12副本全量流水线；不将外部production_models_started=0冒称资源已充分利用。
- 全量17w+尚未完成，所有输出未人审不可训练。后续unused是实际排他预分配，不是已经VLM完成；它们会在当前队列自然耗尽后接续。

## 全量执行期间遗留来源区分

- native37610同池旧owner中仍有80条assigned，不应误判为当前新owner产线在途。N1旧g67=16；N8旧g23=32、旧g67=32。
- 其中24条有明确CPU_prepare失败回执，models_called=0：N1=8、N8=16。原claim和prepare_failure保留；已metadata-only写failed_sealed_no_retry本地result/commit，新增模型调用0，未上传前不算可靠共享终态。不能自动重试这些已失败来源。
- 其余56条已有完整本地prepared packet且source模型预算不存在：N1=8、N8=48。只冻结原claim恢复清单，没有标失败、没有换owner或重新领取。清单分别N1:/root/stage2_dynamic_native37610_N1_20261028_v1/uncalled_prepared_original_claims.json SHAaa3d9a802938aa75e73dc8d1312f7cc52c9f668d820784e90b5c15d78131f87c；N8同目录节点名N8，SHAd205c1ab449cf74f2aaec57cac9aaa1e085896f8951c2c6a3ec62bb137c0a59a。后续只能按原owner/原packet和零预算规则消费，不重派已经调用模型的32条融合前中断来源。
- 当前12副本持续处理未领来源，没有因这80条中断。当前本地root/tmp快盘各节点仍有约7.6–9.0TiB剩余，暂不需要以删缓存缓解I/O；用户授权的临时cache回收仍只在可靠ACK后执行。
- 6分钟只读连续观察任务hUhTxq真实运行，已观察可靠processed58785→59235、active12且errors=[]；这只是持续运行证据，不是全量完成。
- 外部可信入口及旧N2/N4残留引擎精确pidfd TERM授权已经直接在会话请求一次；等待用户补充期间继续独立全量流水线，不重复索取或绕过限制。

## 外部03–06已实际发布，统一统计纳入分项（2026-10-28）

- 用户反馈外部产物在ROOT/stage2_dynamic_local_shared_results_20261028_v1/a100-extra-03～06/<batch_id>/{batch.tar,commit.json}；不是external_stage2_vlm_50000输入目录。最终结果成员sources/<split>/<id>/postprocessed.json，唯一融合caption字段fused_caption，原视觉审核/修正/限制仍保留。
- 本会已直接读取四节点真实shared commit、核schema/node/batch/source_count/status_counts与归档存在长度，核来源键全部属于冻结50000清单、跨外部节点以及主群交集0。没有重读video或tar payload，也未冒称独立核了当地ACK或GPU活体。
- 起先实读外部5700来源；随后统一进度v6实读5900/50000：03=1500、04=1600、05=1500、06=1300。各节点最新提交有持续增量，不再沿用“外部生产0”的旧结论。
- 原进度服务本就扫描SHARED全部node目录，外部结果已经包含在全局完成数；此次只是加入external_shared_results分项、原五万source范围校验与loaded_active_replicas_scope。不能把5900再加到全局61985之上。
- 同一v6快照 observed_unix1793178016.8142083：全局VLM处理后可靠提交61985、失败封存34、外部共享终态5900、主群共享终态56117（含32失败）、外部主群交集0，30分钟24050.12条/h，errors=[]。历史分母175276仍为估算，不当当前exact。
- loaded_active_replicas=12明确仅N1/N2/N3/N6/N8主群活体范围，不能直接加4宣称16已独立核实。外部live_model_verified=false、local_upload_ack_independently_verified=false；用户当地ACK事实作为外部反馈，主群基于shared提交计数。
- v6默认命令仍python3 runtime/stage2_progress.py --watch；会同时显示03–06共享完成分项。模型运行与上传不因统计升级停止；原所有墙钟样本保留不重置。
- 外部实际sharedcommit证明已有生产结果，不需要再自行初始化第二份50000 pool，也不可重复启动当地原模型。后续只接续正在跑的当地任务，可信hostkey缺口不阻碍共享完成统计。

## 遗留来源闭合与最终结果取用已推进

- 24条明确CPU_prepare失败（N1=8/N8=16）已经本地failed_sealed_no_retry后由各节点唯一现有上传器代投，未增加第二上传器、未调用模型。共享提交与ACK已实际核实；ACK回填原claim outbox。代投证据在当前out/imported_failure_outboxes保存，不混入当前source-group的rollover集合。失败封存总数由34增至58，不把失败算完整清洗。
- 56条旧prepared但模型source预算不存在的原claim已安排最终一次消费。N1/N8各有/root/stage2_original_claim_tail_service_<node>_20261028_v1持久controller；严格等待native37610→externalrem34803→legacy15973全部该节点正常来源完成、原生safe三flag、所有source ACK后，在原physical pair使用原owner、原packet、原out，原模型配方不变。没有重新领取、没有重推理旧visual已调用32条，当前GPU模型不停。
- 56条原claim消费逻辑的CPU夹具实验证明原owner/原claim不变、消费后第二次在模型初始化前拒绝；实际生产尚未执行，不提前标完成。
- 新最终结果目录ROOT/stage2_final_caption_export_20261028_v1，shards/part-<序号>/{captions.jsonl,commit.json}。每分片最多1000来源，含唯一fused_caption、视觉审核、修正/限制、原result定位和未批准/不可训练标志。原tar和原结果保留。
- N5 CPU导出服务/root/stage2_final_export_service_N5_20261028_v1持续增量，只读已提交最终结果，不重读视频/不重跑模型。旧小文件前置读取限为64/轮并行8，原批tar一次复制root快盘后解析，避免Ceph大量seek小读。
- 导出一次中断曾留下part-000000/commit.json零字节；两个fresh stat/read一致确认，原零文件保存在commit.interrupted-zero-<inode>.json，captions.jsonl完整不改。新版采用完整临时commit fsync/readback后无覆盖发布，已有SQLite结果保留；没有覆盖旧非空证据。
- 最终JSONL导出已实际共享发布第一片1000来源，随后已读1715继续处理中，blocked_inputs=0。这只是结果整理进度，不是全量完成；full_corpus_complete仍false。
- 全量事件监控8tgmxI正在运行（30分钟deadline），仅来源组切换、活副本变化、万条进展或异常时通知；当前实见可靠processed72585、主群12活模型，external四节点仍持续sharedcommit。避免无意义重复短审计，阶段边界仍核native-safe/ACK。

## N1内存页缓存安全门事件（未降低门槛）

- 2026-10-28事件监控检出N1g23从12→11副本，实际worker栈为headroom<80GiB主动拒绝融合前准入，不是锁解析回归。原manager已safe三flag、锁租约正常恢复；其余11副本继续。
- N1 cgroup limit约997.83GiB、usage约899.4GiB，其中cache约791GiB、RSS约99GiB；不是GPU显存耗尽。没有修改有效内存计算、160GiB新准入或80GiB运行安全门。
- 仅对本节点已加载完的原模型safetensors以及自有已完成prepared视频/JPEG执行POSIX_FADV_DONTNEED释放磁盘读缓存；没有drop_caches全局操作、没有删文件、没有GPU信号。实际cgroup headroom97.82→159.68GiB，再自有输入回收→169.46GiB。
- 32条visual_audit已完成但caption_fusion调用前被内存门拒绝的原结果，保原raw/normalized和预算，metadata-only封存failed_sealed_no_retry，新增模型调用0。未重试已消耗的阶段预算。
- 新修复manager_v4仅在resident模型加载完后对原权重fd给DONTNEED读缓存建议，保原prompt/12帧/BF16TP2/32会话和所有内存阈值。新配置独立phase_memory_cachefix1，保旧binding在runtime_binding_history后原子更新活pointer；原failure/预算不覆盖。
- 仅恢复该g23到原owner未调用模型的ready包；原feeder持续复用，sourceclaim重新领取0。fresh160GiB/tmp8GiB与exact restoredholder在启动边界再核，达不到就不启动。

## N3/N6自然归还四对卡追加生产（不信号停旧产线）

- 实读N3 full12k_s0/s1与N6 s2/s3各processed_source_count=3000，原manager safe三个标志全真，GPU仅原生恢复holder。没有向这些旧产线发停止信号。
- 拟追加N3 GPU2/3、4/5和N6 GPU4/5、6/7，共8卡、4个原版32B BF16TP2副本。配置以原node local root /root/stage2_dynamic_native37610v3_<node>_20261028_v1生成extra_deployments/<owner>.json；queue新phase、out复用原out，单节点上传器仍只有原一个。
- 新owner分别N3-g23/N3-g45/N6-g45/N6-g67-native37610v3，元数据API19100和原API共享同一native37610 DB，只领unused，没有新增独立来源池或重复claim。
- 后续externalrem/legacy API19101/19102同原各自DB允许16个owner。N3/N6原自有CPU等待器精确pidfd TERM后替换expanded版本，保旧台账，不信号停止GPU模型；expanded安全门包含extra_deployments全部pair，等原生safe和全部ACK才交接单上传器。
- N3首次有效余量158.61GiB尚未达160，必须仅释放已加载模型读缓存或等待，不能按空卡强准入；N6有效余量280.64GiB。新manager_v4只在模型加载完给原权重POSIX_FADV_DONTNEED读缓存建议，所有80/160GiB阈值不变。
- 起始追加配置已准备但models_started=0；实际准入、loaded和shared提交后才算上线，不拿holder100%利用率当生产。

## 新增自然归还资源已实际上线与原池452缺口补齐

- N3-g23/g45与N6-g45/g67共4新增模型已经actual load真、各新增64–96来源、无failure，主群16活模型；原full12k四pair均3000完成与safe三flag真，没有停止原产线信号。仍每node一个上传器，旧N2/N4残留引擎资源未越权复用。
- N3有效余量首次157.52GiB，仅对已加载原权重读缓存fadvise后219.80GiB才准入；新manager_v4模型载入结束释放自己的权重磁盘read-cache，80/160GiB门不变，文件不删除。
- 新extra_deployments位于原native37610v3 node root，producer为本地manager_v4，原out共享但owner/pair新；API19100只增加owner whitelist，DB仍native37610唯一ledger，sourceclaim不重复。
- N3/N6的expanded后续CPU控制器实际存活，安全门覆盖追加全部pair。externalrem34803与legacy15973分别API19101/19102共用原DB并允许16owner；后续采用manager_v4与精确producer动态启动，旧configs/结果/claims不改。
- 原3000 pool只看registry顶层会误报1452未领；完整旧config+pending排除后实际452尚未登记，另1000已有原stage2分配，不重派。新452独立排他reservation为ROOT/stage2_archive_gap452_inputs_20261028_v1，allocation SHAa40c2fa15eacaf94dc461d828a2948c1e5ad4b000f374c88e5c6e1f56c7ca44e。
- 452原JSON已24个归档批次一次取料全部ready，0失败；manifest SHAcd185b9d6bad053f9956a5c67366998e52ccdc2ac1ecf9a22d5fd3acf61a8daa；未恢复旧420失败程序/未忽略旧st_dev迁移proof。N2/root/stage2_pool_archivegap452_20261028_v1/API19002只允许N2g23尾部owner，模型消费尚未开始。
- N2/root/stage2_archivegap452_tail_service_N2_20261028_v1提前等待legacy15973两pair全safe+ACK，然后仅g23处理452；不与N1/N8的56原claim尾部恢复争卡。原56控制器仍等待正常队列结束后保owner一次消费。
- 最终caption导出改为4线程读批tar/root一次copy、单SQLite写者、1000来源可靠分片；原自有CPU旧版本精确替换，已有数据库保留。实见共享export34000来源、blocked0；不把导出数量当新增GPU产量。
- 一份旧导出本地tar copy被CPU服务切换留下50331648bytes、原提交expected66754560；已保留partial到interrupted-inode文件，归档源和sharedcommit不改，只对已有错误记录恢复只读导出，模型调用0。
- 全量阶段监控由1QC40b继续，最终导出监控SWuSFh继续；只事件唤醒不反复全库审计。主群现有normal尚未耗尽，external50k仍当地持续提交，全部目标未完成。

## 第一原生池真实耗尽与第二池接续验收

- 事件监控1QC40b已经实见native37610 unused=0，以及主群模型从原池退出/新池加载。加载期active下降不自动等于故障，实时窗口保留全部此空窗。
- N2/N6/N8已经actual deployment/live并开始externalrem34803，N2/N8首副本实产32来源。后续池不再一直unused34803，必须读现SQLite最新状态。
- N1/N3的旧未来CPU控制器在早前失败回执尚未被修复新binding替代时安全拒绝并已退出；没有假称一直存活。独立safe_resume控制器随后读取当前修复binding、原生safe+全node ACK后才继续既有已staged未来源组，没有重启旧失败controller或重建来源pool。N1已actual新stage首副本准入加载中。
- N3原g01此前一直原v2（未停正常模型）后来在新并发锁等待记录下复现已知解析ValueError。原manager已safe，32个阶段预算中断来源只metadata封存、新模型调用0；剩余32个无模型预算原包用新manager_v4与原owner一次消费，sourceclaim重新领取0。原feeder已durably drained，未重复启动feeder。runtime_binding指向新repair queue；这为该node后续整节点safe/ACK补齐，不影响新增g23/g45有效模型。
- 原native37610旧56准备未执行来源仍按最后原owner尾部controller恢复，不能把assigned当当前pipeline故障；旧真正失败保回执，不能自动重派。原pool全局终态需待ACK、56尾部和最新32封存闭合，不仅看unused0。
- 当前全量结果共享导出已actual 43000来源、43个1000JSONL分片，blocked0，CPU服务2658657存活；原输入与GPU模型不因导出重复处理。

## 34803第二池实际16副本生产与外部额外TP4扩容包

- 主群所有5节点已从原native37610自然safe/ACK完成切入externalrem34803。最新实读主群16已加载活模型，externalrem池unused29651、assigned1202、已回写终态3950；不是仅等待器或计划。N1/N3早前安全拒绝等待器保旧失败，独立repaired_binding新回执后接续，未重复初始化pool。
- 同一快照observed_unix1793182934.4682288：全局可靠清洗95687、失败封存122、外部03–06共享终态16450、30分钟25758.09/h、errors=[]。外部已包含全局，不重复累加；全量仍未完成。
- 用户问能否再部署更多。已完成外部每节点第二个TP4扩容包：ROOT/external_stage2_vlm_50000_20261028_v1/stage2_external50000_expand_tp4_v1.tar，112640bytes，SHA8f36182dd296b724c24f535c5caf94332d4fc8f594020952b5136b814bd43458。
- 外部每节点GPU4–7新副本，原GPU0–3模型和feeder不停止。扩容新relay18999和旧relay共享原50000 SQLite DB/token，绝不再初始化50k池；双owner0123/4567相互防重只领取unused。
- 每外部节点仍只有一个上传器：原CPU uploader按自身STOP自然结束已有上传后，兼容两个owner的新CPU uploader接替；单批可混两owner但来源归属仍原claim，最终等两个GPUworker都generation_drained才结束。旧STOP/ACK/claim保留不清。
- 真实CPU验证两个TP4 owner同池claim互斥、混合owner HTTP封包可靠sharedcommit/ledger终态、单上传器等两worker排空；模型配方保持原32B BF16TP4、不量化、12帧/16k/1800/32会话不改。没有宣称CPU测试代表外部额外4副本已经上卡。
- 外部当地需执行新包relay/node CLI并fresh核GPU4–7为空、有效memory160GiB/tmp8GiB，local existing model/Python复用。中央本端可信SSH限制仍不绕过，额外实际models_started目前0，等当地实际load/live/sharedcommit回报。

## 全量来源末尾128条未启动预留已明确安排首次消费

- 旧固定stage ms_fullframe_vlm_postprocess_batch_v8_fastmeta128_n1g01_20261028是唯一未闭合历史固定票：实际config不存在，模板out/queue/model source预算均不存在，IxgX6E只资源准入拒绝没有打标。其128来源仍N1原预留，没有在其他新池领取。
- 原128plan SHA f0f48ba76929c469843e1d16571238186b9b1a949a30eafbc63c2fcbd5644be9，元数据在N1/root原fastcache可用，保fullframe原logical origin。没有重新生成MotionScript或修改原预留来源。
- 新池/root/stage2_pool_originalpending128_20261028_v1在N2，API19003只允许N1-g01-originalpending128，sourceclaim0。原固定失败资源配置永不重启；后续为这128条从未有model预算的来源第一次执行。
- N1/root/stage2_originalpending128_tail_service_N1_20261028_v1持久controller已启动，只等native37610→34803→15973所有正常工作完成、原56tail中N1的8条原claim恢复safe+ACK之后才占N1g01。前一原tail uploader通过自有STOP自然停后，仍保持一node一uploader，不与当前生产争卡。
- 原128独立pipeline完成后N2对应ACK服务只补sharedcommit终态，models_called=0；当前未首次消费，不标完成。
- 当前9分钟实测继续新增5600可靠来源（102237→107837），16副本全程活，errors=[]；第二池unused21771、external50k共享20800，全量仍未完成。短窗口仅持续运行证据，最终要看所有source终态和可靠导出。

## 2026-10-28 新协议全量重跑授权与旧版本排空

用户确认短边缘合并策略并授权重新启动所有来源审计。新协议：rich仅舞种、全英文、保原计划合并边缘窗口、原events/windows为动作来源、视频修订与跨窗连贯、禁止服装背景、幅度在窗口正文中。旧fused_caption按错误协议产生，不作为本版本合格训练正文，原draft/events/video和过程证据保留。

停靠前现场：16个主群worker位于/root/stage2_dynamic_externalrem34803_<N1/N2/N3/N6/N8>_20261028_v1，deployment.json/config_ref记录queue/out/PID；总源34803，最近SQL unused6635、assigned1118、终态27050，各owner当时processed约1568–1792。精确集合由原pool.db的sources(state/owner/batch)保留，不从旧未领快照重派。当前registry预留总范围不改，重跑使用新revision防重域，与旧预算不覆盖。

先停止本会话自有future CPU等待器，避免旧legacy/452/原56/128在排空后自动接续错误协议；仅对核完整PID/start/argv的自有等待器pidfd TERM，保存每node local停止回执。对当前GPU产线只写本地out/FEEDER_STOP，CPU完成已领packet→FEEDER_DRAINED→模型自然完成ready→原manager恢复holder→safe三flag。每node唯一uploader继续到全部ACK，旧结果/锁/租约/claims/预算不删除。外部03–06必须停领旧协议，但本端无可信SSH入口，不猜/绕过hostkey；需当地执行停领与保ACK。

此节为停止前记录，不代表safe已取得。新源分配不读取旧fused_caption作为事实输入，原窗口与rich舞种来源需先核真正完整计划；不把原draft observed_time窗口冒充merge_short_edges，也不从全局BPM强造小节相位。

## 原生产线（历史快照：当时全部未停止）

### N3 full12k_visual_s0_v1

- manager PID/start：928744 / 532323696
- 观察unix：1793154947.6241941
- GPU：[2, 3]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/full12k_visual_s0_resume_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/full12k_visual_s0_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/full12000_stable_roi_v1/shard0
- 当前安全/失败回执：{}
- 配置targetcount（不是已领）：3000
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N3 full12k_visual_s1_v1

- manager PID/start：930839 / 532359182
- 观察unix：1793154947.6241941
- GPU：[4, 5]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/full12k_visual_s1_resume_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/full12k_visual_s1_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/full12000_stable_roi_v1/shard1
- 当前安全/失败回执：{}
- 配置targetcount（不是已领）：3000
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N3 curated_collaborative21000_n3g67_v2

- manager PID/start：1225306 / 535729781
- 观察unix：1793154947.6241941
- GPU：[6, 7]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/curated_collaborative21000_n3g67_launch_v2.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/curated_collaborative21000_n3g67_v2
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/curated222939_collaborative_v2
- 当前安全/失败回执：{}
- 配置targetcount（不是已领）：10000
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N4 caption_two_stage_local_n4c3_balanced_v1

- manager PID/start：1573551 / 545310258
- 观察unix：1793154948.1474943
- GPU：[0, 1]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/caption_two_stage_local_n4c3_balanced_v1_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/caption_two_stage_local_n4c3_balanced_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/caption_two_stage_local_n4c3_balanced_v1_execution
- 当前安全/失败回执：{"failure.json": {"error": "FileNotFoundError(2, 'No such file or directory')", "phase": "caption_two_stage_local_n4c3_balanced_v1"}, "resources_safe.json": {"phase": "caption_two_stage_local_n4c3_balanced_v1", "all_holders_safe": true, "locks_released": true, "leases_restored": true}}
- 配置targetcount（不是已领）：2396
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N4 caption_two_stage_local_pool_n4g23_v1

- manager PID/start：1714278 / 545693352
- 观察unix：1793154948.1474943
- GPU：[2, 3]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/caption_two_stage_local_pool_n4g23_v1_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/caption_two_stage_local_pool_n4g23_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/caption_two_stage_local_pool_n4g23_v1_execution
- 当前安全/失败回执：{"failure.json": {"error": "FileNotFoundError(2, 'No such file or directory')", "phase": "caption_two_stage_local_pool_n4g23_v1"}, "resources_safe.json": {"phase": "caption_two_stage_local_pool_n4g23_v1", "all_holders_safe": true, "locks_released": true, "leases_restored": true}}
- 配置targetcount（不是已领）：5710
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N4 caption_two_stage_local_pool_n4g45_v1

- manager PID/start：1714279 / 545693352
- 观察unix：1793154948.1474943
- GPU：[4, 5]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/caption_two_stage_local_pool_n4g45_v1_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/caption_two_stage_local_pool_n4g45_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/caption_two_stage_local_pool_n4g45_v1_execution
- 当前安全/失败回执：{}
- 配置targetcount（不是已领）：5710
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N4 caption_two_stage_local_pool_n4g67_v1

- manager PID/start：1714273 / 545693338
- 观察unix：1793154948.1474943
- GPU：[6, 7]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/caption_two_stage_local_pool_n4g67_v1_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/caption_two_stage_local_pool_n4g67_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/caption_two_stage_local_pool_n4g67_v1_execution
- 当前安全/失败回执：{"failure.json": {"error": "FileNotFoundError(2, 'No such file or directory')", "phase": "caption_two_stage_local_pool_n4g67_v1"}, "resources_safe.json": {"phase": "caption_two_stage_local_pool_n4g67_v1", "all_holders_safe": true, "locks_released": true, "leases_restored": true}}
- 配置targetcount（不是已领）：5710
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N5 curated_collaborative4_n5p0_v1

- manager PID/start：1088078 / 540912073
- 观察unix：1793154949.4618518
- GPU：[0]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/curated_collaborative4_n5p0_v1_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/curated_collaborative4_n5p0_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/curated222939_collaborative_v4
- 当前安全/失败回执：{}
- 配置targetcount（不是已领）：20000
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N5 curated_collaborative4_n5c2_v1

- manager PID/start：1088085 / 540912142
- 观察unix：1793154949.4618518
- GPU：[2, 3]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/curated_collaborative4_n5c2_v1_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/curated_collaborative4_n5c2_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/curated222939_collaborative_v4/consumers/shard-0002
- 当前安全/失败回执：{}
- 配置targetcount（不是已领）：5000
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N5 caption_two_stage_live_n5c3_v3

- manager PID/start：1364828 / 544548446
- 观察unix：1793154949.4618518
- GPU：[4, 5]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/caption_two_stage_live_n5c3_v3_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/caption_two_stage_live_n5c3_v3
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/caption_two_stage_live_n5c3_v3_execution
- 当前安全/失败回执：{}
- 配置targetcount（不是已领）：4586
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N5 full12k_roi_v1

- manager PID/start：821928 / 532037057
- 观察unix：1793154949.4618518
- GPU：[6]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/full12k_roi_v1_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/full12k_roi_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/hand_roi_full12000_v1
- 当前安全/失败回执：{}
- 配置targetcount（不是已领）：12000
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N5 main_review_watch_v9

- manager PID/start：1061720 / 540240604
- 观察unix：1793154949.4618518
- GPU：[7]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/main_review_watch_v9-20261025_main_old4exact_v9_1792945680560248991.launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/main_review_watch_v9-20261025_main_old4exact_v9_1792945680560248991
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/curated_independent_reviews_v1
- 当前安全/失败回执：{"failure.json": {"error": "InterruptedError('manager received signal 15')", "phase": "main_review_watch_v9"}, "resources_safe.json": {"phase": "main_review_watch_v9", "all_holders_safe": true, "locks_released": true, "leases_restored": true}}
- 配置targetcount（不是已领）：222939
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N6 curated_repair_resident_v2

- manager PID/start：1291123 / 540829448
- 观察unix：1793154948.2585704
- GPU：[0, 1]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/curated_coarse_fixed3_primary_restore_fresh_launch_v1.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/curated_coarse_fixed3_primary_restore_fresh_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/curated222939_repair_v1
- 当前安全/失败回执：{}
- 配置targetcount（不是已领）：222939
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N6 full12k_visual_s2_v1

- manager PID/start：599171 / 532323791
- 观察unix：1793154948.2585704
- GPU：[4, 5]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/full12k_visual_s2_resume_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/full12k_visual_s2_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/full12000_stable_roi_v1/shard2
- 当前安全/失败回执：{}
- 配置targetcount（不是已领）：3000
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N6 full12k_visual_s3_v1

- manager PID/start：600814 / 532359277
- 观察unix：1793154948.2585704
- GPU：[6, 7]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/full12k_visual_s3_resume_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/full12k_visual_s3_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/full12000_stable_roi_v1/shard3
- 当前安全/失败回执：{}
- 配置targetcount（不是已领）：3000
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N7 curated_collaborative3_n7c7_v1

- manager PID/start：3499874 / 540751116
- 观察unix：1793154949.0168197
- GPU：[2, 3]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/curated_collaborative3_n7c7_v1_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/curated_collaborative3_n7c7_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/curated222939_collaborative_v3/consumers/shard-0007
- 当前安全/失败回执：{}
- 配置targetcount（不是已领）：3334
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N7 curated_collaborative3_n7c8_v1

- manager PID/start：3500413 / 540757507
- 观察unix：1793154949.0168197
- GPU：[4, 5]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/curated_collaborative3_n7c8_v1_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/curated_collaborative3_n7c8_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/curated222939_collaborative_v3/consumers/shard-0008
- 当前安全/失败回执：{}
- 配置targetcount（不是已领）：3333
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N7 curated_collab3_review_n7g6_v1

- manager PID/start：3513805 / 540936538
- 观察unix：1793154949.0168197
- GPU：[6]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/curated_collab3_review_n7g6_launch_v1.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/curated_collab3_review_n7g6_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/curated_collab3_independent_8b_v1/gpu6
- 当前安全/失败回执：{"failure.json": {"error": "RuntimeError('stage worker exit 1')", "phase": "curated_collab3_review_n7g6_v1"}, "worker_exit.json": {"exit_code": 1, "phase": "curated_collab3_review_n7g6_v1", "targetcount": 15000}, "resources_safe.json": {"phase": "curated_collab3_review_n7g6_v1", "all_holders_safe": true, "locks_released": true, "leases_restored": true}}
- 配置targetcount（不是已领）：15000
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N7 curated_collab3_review_n7g7_v1

- manager PID/start：3513811 / 540936563
- 观察unix：1793154949.0168197
- GPU：[7]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/curated_collab3_review_n7g7_launch_v1.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/curated_collab3_review_n7g7_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/curated_collab3_independent_8b_v1/gpu7
- 当前安全/失败回执：{"failure.json": {"error": "InterruptedError('manager received signal 15')", "phase": "curated_collab3_review_n7g7_v1"}, "resources_safe.json": {"phase": "curated_collab3_review_n7g7_v1", "all_holders_safe": true, "locks_released": true, "leases_restored": true}}
- 配置targetcount（不是已领）：15000
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N2 caption_producer_cached_n2p0_v1

- manager PID/start：4050281 / 545974430
- 观察unix：1793154947.430635
- GPU：[0]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/caption_producer_cached_n2p0_v1_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/caption_producer_cached_n2p0_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/caption_producer_cached_n2p0_v1
- 当前安全/失败回执：{"failure.json": {"error": "RuntimeError('Native run not completed with exclusively unstarted sources')", "phase": "caption_producer_cached_n2p0_v1"}, "resources_safe.json": {"phase": "caption_producer_cached_n2p0_v1", "all_holders_safe": true, "locks_released": true, "leases_restored": true}}
- 配置targetcount（不是已领）：4457
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。

### N2 caption_two_stage_local_pool2_n2g67_v1

- manager PID/start：4113508 / 546605005
- 观察unix：1793154947.430635
- GPU：[6, 7]
- config：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/runtime/caption_two_stage_local_pool2_n2g67_v1_launch.json
- queue：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/queues/caption_two_stage_local_pool2_n2g67_v1
- out：/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/caption_two_stage_local_pool2_n2g67_v1_execution
- 当前安全/失败回执：{}
- 配置targetcount（不是已领）：7944
- 已领exact集合与数量：未闭合，停靠前必须补。
- 已完成：未定位权威摘要，停靠前必须补。
- 剩余位置：待核config中plan/ticket/claims及未领suffix，不据未提交数重派。
- 停止／实际safe验收：尚未执行。



## 内容证据纠错接续（原生排空前台账，2026-10-29）

用户最新授权快速修正不可见部位否定动作、侧别/承重和跨窗衔接证据问题；不以 batch=16/32 作为质量问题原因。已独立新增 window_visibility_boundary_evidence_v1，正文仍保留英文、rich仅舞种与原 merge_short_edges 时间窗。原 v3 源码、claim、预算、结果均不覆盖；每来源仍最多两阶段、不自动重试。

拟先在 N2 当前两个 batch16 phase 自然停止供料、消费完原 ready，等待原 manager 三项 native-safe 与全部可靠 ACK，然后进入独立 evidence1 phase。仅使用既有 N2 g23/g45，不触及 g01 或旧残留 g67；不发送任何进程信号。N2 g45 新 phase 首次领取仅3条未领取来源，模型完成后驻留等待真实输出检查；其余新供料等待该检查，未通过不自动扩大。冻结原生 resource_manager.py 不改，160GiB/80GiB/tmp8GiB 门槛不变。

CPU回归已通过：32条独立请求两阶段映射、旧预算禁止重放、重复微领取拒绝、shared commit/ACK；8项证据负例拒绝。此证据不代表实际VLM语义批准。主群盘点无完全空节点；占卡但持锁、残留引擎、内存不足卡不按可用卡计算，保留资源不抢占。


### 最新用户范围收缩：VLM只做局部事件补丁

第一轮同源三例视觉可见性约束诊断已完成：wToy被解剖侧别检查拒绝，pImh被无证据掌面检查拒绝；Xi通过规则但仍把并脚描述成全窗宽站，因此不能批准整窗VLM重写。纯MS＋LLM没有新增这些脚步/承重句，但还较机械。显式 corrections 在原三例10窗中全为空（0条可核查纠错记录），不能将重写量称为纠错收益。

已按用户最新要求独立新增 minimal_event_patch_fewshot_v1：默认原events原样保留、只允许有限方向/部位/动作词替换；明确错误时才删除整个事件；禁止新增动作和整窗caption，要求两帧位于原事件范围；LLM只读应用补丁后的原事件。few-shot策略示范6条。

接续仅复用N2 g23空等槽位：先让当前自有 evidence1 feeder自然结束，核g23模型0生产来源、正常退出及native-safe；用新冻结代码对同3例单独诊断，独立minimal_*阶段预算，不重新领取、不覆盖任何旧结果，不自动扩大生产；诊断结束由原生manager恢复holder。N2 g45先前诊断模型仍驻留待检查，不称为生产活副本，也不触碰N2 g67旧残留。


### 用户最新授权非N4残留释放；N4明确不kill

非N4当前最适合诊断为N2：内存余量354.8GiB，g23最新三例已正常退出并native-safe；g45旧诊断结果已落盘、零生产ready，空等验收；g67旧引擎TP残留0利用率、原manager保锁。用户明确旧残留可释放，N4不可kill。拟仅精确pidfd TERM N2 g45自有诊断manager（由原生cleanup停止直接child），再清已核旧引擎和必要TP残留；g67仅清旧引擎/TP，让原manager自行恢复holder与租约。先保存本地完整身份/config/queue/诊断输出引用/算卡归属，核身份start_ticks、UID、argv、TP ancestry和GPU UUID。不得KILL、不删claims预算、不动N2 g01未核活锁、N5保留卡或N4任何进程。


### N2残留释放完成；仅g23三例修复验证

N2 g45已完成诊断manager、g67旧孤儿EngineCore按新授权精确pidfd TERM，TP自动退出无额外信号，原manager自行恢复holder，g45/g67 native-safe三项全真。g23先前已正常native-safe。目前三对6卡均仅975MiB已恢复holder，无模型；内存余量约370GiB。g01活锁未动、N4全未动、N5保留卡未动。第一次脚本在任何信号前因历史226个已消费ready误判而退出，修正为只看unconsumed后才执行；历史ready/预算不删。完整before/TERM/after本地证据保留。

候选minimum patch v1真实三例没有获批准的VLM纠错；其中wToy/pImh补丁本身为空，拒绝来自中文引用/元数据自由文本，不能称动作错误；Xi提出补丁但有事件外帧与重复事件，不应用。原事件中位时长0.25/0.25/0.3125s，每窗8帧最多仅覆盖24/62、17/42、34/46事件的事件内两帧。

新增minimal_event_patch_eligible_pairs_v2：CPU先选唯一归属窗和事件内相邻帧组合，schema固定候选event id/合法帧对；不可审事件默认保留，限原字词补丁，不写整窗动作；证据限制改enum代码，不允许自由场景/原中文引用。LLM多事件few-shot改为简短主序列，最多4完整句/650字符，保舞种候选混合与原时间窗。不重领旧来源，原三例独立minimal_v2_*诊断预算且不自动重试。此轮仅N2 g23两卡，g45/g67四卡不再加载模型。若局部VLM仍无独立核实收益，优先退回MS＋LLM轻润色；不以“schema通过”批准事实质量，也不把三例外推全库。


## 用户批准32B纯文本主流程接续（2026-10-29）

最新明确授权：推进原MS＋较大LLM轻微润色，8B太小，快速持续全量推进。采用已有32B原模型BF16纯语言模式，图片/视频输入=0，不下载/替换精度、不读旧VLM正文。独立revision english_merged_edges_MS_LLM32B_v4，同原221887输入catalog；每来源仅caption_polish一次、预算O_EXCL永久防重，不清旧v3预算。此前已验证同3例纯文本结果以出处复用，无重推理；外部原72000来源配额保持同node映射，新文本stage预算独立。

N2先使用已原生安全释放的g23/g45/g67六卡，三个TP2副本batch32；g01不动，N4明确不kill且不使用，N5保留卡不动。中央19116新服务负责领取、元数据与可靠HTTP上传/terminalACK。后续N1/N3/N6/N8仅停止原feeder，自然完成既有ready、原生safe三项＋全部旧结果ACK，再启动独立root/phase；不kill任何活在产模型，不丢原claim。停止前各node本地before_old_drain记录完整原config/owner/queue/code绑定与N4排除范围。单node旧uploader全部闭合自然停止后才接新uploader。未知残留导致safe阻塞时不抢卡、不删锁。

新source_count=221887，blocked原引用20仍单列不隐瞒，不让它阻挡既有全部已准备来源。新结果为弱监督未人审不可训练，不把格式通过视作动作真值。实时进度随后改为新文本stage唯一入口，旧VLM产量不得计入新正文。
