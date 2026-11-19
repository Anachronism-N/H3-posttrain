# 全量动作正文规范：当前已批准原MS＋32B纯文本轻润色

## 当前状态：用户已暂停所有标注/重写，主群资源已全部释放

最新用户明确停止当前进程、保留已打标数据、释放全部相关资源；此前自动扩量/补推理/纠错任务都暂停，不得自行恢复。执行停止前notes36和runtime/user_stop_before_inventory_v1.json留真实身份/config/queue/GPU证据。N1–N8所有相关模型、遗留标注worker和项目占卡holder已停，独立现时nvidia-smi所有64卡memory0/util0/compute apps空，原生manager safe三项全真且全部结束。N4亦按本次最新全释放授权处理；无关任务未信号。进度及旧CPU轮询、单node uploader也已停；仅保留数据查看server与N2不占GPU的旧结果接收服务，以接收外部遗留上传。

用户停止后的原文本生产终态：published_unapproved62428、failed_sealed_no_retry11975、stopped_unclaimed_by_user147484（共221887，assigned0）。已完成local source commits74403全部可靠shared上传/ACK闭合，各node未ACK0，不再处理剩余来源。8128条零生成schema拒绝与普通failed均原样保留，预算不清，不再补生成。所有旧VLM/MS/BeatThis、诊断和已下载权重也不删除；最终停止快照ROOT/user_stop_preserve_all_20261029_v1/final_preserved_state.json，node before/TERM/gpu_release_report/CPU_services_stopped记录齐全。

原MS共享完整副本ROOT/original_MS_221887_shared_v1（221887来源/5849包，完成凭证已核），已有纯文本输出ROOT/MS_LLM32_english_merged_edges_v4/production_results，旧版本在原目录保留。当前数据全部未人审，不称质量批准。

外部01–06中央新领取已停止，原V3仅剩72assigned（02为64、03为8）来源保留，不回收重派。此代理无六节点独立GPU实时访问，不能冒称外部48卡全部已空；需要外部执行者按最新停止指令关闭其相关模型/holder并保输出，旧结果接收服务为此暂留。所有历史生产任务均已pending暂停，release任务80在可控主群范围completed；无任何新生成目标，等待用户明确新授权。


### 同prompt Qwen3-32B实际三例对照：不全群替换

用户要求使用相同prompt比较纯语言Qwen3-32B。固定既有3例rlZPIwvAIKanATbE/CgXLeEEFhqpmXvEj/XvRLLnLIHSreZMMf；N6原生产caption_polish.request.json完整system/user消息与schema冻结、哈希相同，temperature0/max_tokens4096/seed0、原窗口不变。Qwen3官方tokenizer模板enable_thinking=False，模型BF16TP2/context32768，3例单次独立诊断预算，不重新领取来源、不改旧预算或结果。模板/分词器必然随模型不同，不声称原始token序列一致；新3例20.25秒不是与旧生产batch32的等条件吞吐对照。

本机N1官方下载权重一次复制至N2/opt/models/Qwen3-32B，17分片/25文件。本轮实卡N2g01；N7g67虽然safe但153GiB不能绕160门。由于现node级供料STOP，先让N2原4生产消费完ready、native-safe/全部ACK，再新phase恢复g23/g45/g67，另外14生产不动。g01诊断正常safe后又恢复原VL32B文字模式生产；最新18副本都活/加载，无manager故障。冻结resource_manager.py不改；独立diag producer仅绑定纯LMmodelref做完整资源核。旧CPU控制器的Popen等待会被原生manager安全恢复后的h.wait阻塞，诊断尚未启动时已精确交接CPU，新controller以native-safe和results判断；未signal GPU模型、未重跑来源。

Qwen3三例3/3格式通过、输出tokens328/245/204，但整体未证明更好：复杂例将各窗动作概括为两膝持续交替，窗3原仅B膝bend/extend无A膝事件，属于扩大；简单例tracking技术句消失，但窗1原A腕下移、A肘弯、A腕后来左移/下移及A膝弯共5信号episode漏述；下肢受限例旧同臂绑定/as同步句消失，但两腕下移、窗边B腕左移和最后A肘弯也消失，不能算完整纠错。正文短不作好/坏依据，问题是漏掉事实、主体泛化与原顺序覆盖不足。详细same-prompt输入/原结果/Qwen3raw/paired_content_assessment.json在新root/Qwen3_sameprompt_three_v1。

当前结论：不据这一组对照换整群。纯LM并不会自动解决现prompt保真问题；后续需单独修A/B主体、真实重叠、tracking元数据边界和覆盖，不能把限制1–4句作为质量目标，也不把3例外推全量。任务78同prompt对照completed；全量主任务66/零生成闭合69/侧别修复74仍未完成。最新可靠正文51142/221887、完整30分钟26864条/h、18活副本36卡，N4不动。


### 同3例逐项实际改动与纯语言权重已备好

用户要求正文不含tracking诊断、逐case说明实际修改内容及合理性，同时下载Qwen3-32B。已复核同3例原121events（40肘角、17膝角、64腕移动）与9窗48句：原stored draft/eventid/text/start/end与plan零改，所有9个英文正文窗均经翻译合句重组，不是仅字词替换。case_change_audit_v1.json详细保存对应原event：至少1tracking诊断新增句、1手臂主体混绑句、2原事件不具正时长重叠的同步关系、3复杂case A侧膝伸信号episode未明确保留（e0007/e0024/e0043）。这些为源对齐观察，不当全库事实错误率，也不把3个信号episode说成3个完整舞步循环。长文本本身不判问题；合理翻译、重叠合句与boundary continuation举例均附原event。旧生产结果未改，全量sidefaithful修复仍待3例真实验证。

官方下载纯语言Qwen/Qwen3-32B已完成：N2/N6没有公网，零文件时失败，改由可联网本机N1下载；本机/opt/models/Qwen3-32B，17safetensors分片与分词器/config/index等25文件，65540285565bytes，官方revision9216db5781bf21249d130ec9da846c4624c16137，model_typeqwen3。大小/index核完整、无partial，GPU使用0，当前18个VL32B语言模式副本未替换。下载凭证本机/root/stage2_Qwen3_32B_download_v1/DOWNLOAD_COMPLETE.json。任务77下载completed，任务76case复核completed；任务74局部绑定/技术诊断修复仍in_progress。

最新可靠纯文本正文30785/221887（13.87%），18副本36卡，完整30分钟可靠吞吐27499条/h；封存10702包含8128条历史生成前schema拒绝，不能全部归为动作错误。N4仍不动。


### 用户最新内容判断与改动量核实

正文长本身不是问题，不以句数/编辑距离判质量差或强制删动作；仅判断事实、两侧归属、时序和技术信息泄漏。现模型为复用本地已备Qwen3-VL32B的纯文字输入路径，并非只能用VL；纯语言模型可以另部署，当前并未证明VL是最佳文本润色选择。

固定3例逐字比较原共享tar与N6实际prepared输入：73＋25＋23＝121个原events，完整draft、原事件ID/text/start/end和plan均相同，原事件实际修改0、VLM补丁应用0。英文caption在9/9窗重新组织生成，不是仅替换几个字。跨语言原ms_text去重摘要与正文字符比不能代表事实编辑率；实际新增/漏掉/错绑动作比例尚未全量核实，不报0。

tracking具体泄漏为正文句“These movements occur after a period of missing tracking data.”，是输入观测诊断应留metadata。绑错为e0020一侧肘伸、e0021另一侧腕右移被合到同一个other arm；正确应保持两个不同arm主体。e0021终4.375/e0022始4.375顺接又被as写成同步，属于附加时间关系错误。原结果不覆盖，仅保存局部更正候选，任务74不再以压缩长度为目标。


### batch64真实验收：不推广，恢复32持续生产

N2g01新增副本已完成288来源交替6组（32/64/32/64/32/64），同模型max_num_seqs64、同协议不同独立来源；非语义质量配对。32合计96来源/204.289秒/92自动接受：生成1691.7/h、有效正文1621.2/h；64合计192来源/410.915秒/181自动接受：生成1682.1/h、有效正文1585.7/h。平均输出tokens488.46/490.27，峰显存73361MiB。64无吞吐收益（生成-0.57%，有效-2.19%），不能声称质量变差或必然最优。

实测报告N2本地batch64_trial_metrics.json，原试验source/result/budget保持。已核活PID/start_ticks后写batch64_measured_decision.json选batch_size32，新增常驻副本无需重载继续生产，原17副本未动。任务73已completed，内容修复任务74仍in_progress，不把吞吐验收当全文语义批准。


## 当前集群扩容与实际正文复核（2026-10-29，追加）

已核N2g1原MSmanager3468653有worker_exit0，唯一阻塞是项目holder1154900；新授权原残留可释放范围内，仅核UID/argv/CVD/UUID/start_ticks/TERM后pidfd TERM该holder，原manager自行恢复g1native-safe三项真，g0旧manager原safe也为真。停前台账在notes36和N2本地N2g01_legacy_holder_release_before.json。N4和N5保留卡未动。

N2g01新增第四个32B BF16 TP2副本，当前主群18副本/36卡，其他17副本不停止。max_num_seqs64，按32/64交替6组处理首次领取来源，共288条，独立微领取仍8，source budget一次不变。CPU64独立请求、重复/重试拒绝、50+14可靠封包回归通过；GPU试验尚未完成，不能将可运行称已提速。新增副本按六组结束后驻留等batch64_measured_decision.json，需读batch64_trial_metrics.json判有效正文/生成墙钟及tokens，非同源质量配对。监控任务d45uEY核活PID/start_ticks，不以超时当完成。

固定当前3例内容核实：rlZPIwvAIKanATbE复杂73events，4窗30句/2379字符，仍近低层事件罗列；CgXLeEEFhqpmXvEj简单25events，正文混入missing tracking data诊断句；XvRLLnLIHSreZMMf下肢受限23events，未加入脚静止旧幻觉，但窗1把e0020一侧肘伸与e0021另一侧腕右移绑到同臂，且把e0021终4.375/e0022始4.375的顺接写成as同步。英文、rich舞种及原时间窗保持，不代表全文批准。review目录在新输出root/current_three_content_review_v1，原窗起中末图已核，两例局部修正候选另存，不覆盖生产、不新增模型调用。

已新增sidefaithful候选：原事件明确中性A/B，技术tracking原因不送正文，while/as仅正时长重叠，少量多臂few-shot防混绑；CPU回归通过但尚未新模型三例验证，不全群盲推、不以改写量作质量收益。任务72内容复核completed，任务73批64实测in_progress，任务74侧别/元信息修复in_progress。原MS共享导出已完成221887源/5849包/17395251200bytes（EXPORT_COMPLETE实核），任务68completed。


## 最新接续：32B文本主线已实跑（2026-10-29）

用户已批准回退原MotionScript＋较大LLM轻微润色。实际使用已有Qwen3-VL-32B-Instruct的语言模式，BF16、未量化、零图像/视频输入，不读旧VLM正文、不加动作。新revision为english_merged_edges_MS_LLM32B_v4，原输入已准备221887，待修原引用20另列。

主群已独立核实17个加载活副本、34卡：N1全8卡4TP2；N2g23/g45/g67六卡3TP2；N3g01/g23/g45六卡3TP2；N6g23/g45/g67六卡3TP2；N8全8卡4TP2。batch32，单来源仅一次文本生成。N4明确不kill；N5保留卡和N7不动。最新快照可靠新正文5328、封存9709，全量未完成。

两项助手实现错误已定位：早期句数/650字符硬限制误封存；textp1新增句末正则不被vLLM/xgrammar接受，生成前即拒绝。textp2移除该正则并通过实际xgrammar编译，已恢复正常领取；首批64来源61通过文本规则，非人审。8128条（N2 8064、N1 64）由原owner/batch和空responses＋对应参数错误证明为生成前拒绝，已登记zero_generation_schema_failures表和共享jsonl。原claims/budgets/failed全保留、不重领；显式补首次生成和合流尚未闭合，任务69继续，不把它们称动作错误。

N3模型初始化另外遇Torch编译图NoneType.size，原64条ready包全零model budget；原生safe后按原owner接续，未重领；N3以enforce_eager绕过图编译，模型/dtype/batch不变。对自有已完成ACK的视频/图片/请求归档、失败初始化已读权重做POSIX_FADV_DONTNEED；未降160/80GiB门，未全局drop_caches或删文件。

输出ROOT/MS_LLM32_english_merged_edges_v4/production_results；中央API19116（ms_llm32_server_schemafix_v4.py）。runtime/stage2_progress.py --watch现在只统计新文本stage及实际runtime_bindings；旧进度CPU轮询已请求停止，旧VLM正文不计入。外部六node原72000配额保持同node映射；92KiB新包在同输出根MS_LLM32_external_runtime_v4.tar，不含token，复用节点旧token/32B权重；尚未上线，不报外部48GPU活用。

原MS访问：ROOT/original_MS_221887_shared_v1/source_manifest.jsonl已完整写出221887来源；元数据包正在单次复制，EXPORT_COMPLETE.json才表示全部共享包就绪。runtime/ms_read_original.py支持按id读draft/events/plan/all；立即访问任意原MS可在N2加--master-local，无需等共享复制。清单video_path为原视频共享路径，draft保原结构化事件，plan保merge_short_edges时间计划，不是VLM正文。单条共享读取已实核48原events。

任务66全量推进仍in_progress，67主群部署completed，68共享原MS导出in_progress，69零生成来源闭合in_progress。未人审不可训练，不以脚本启动或通过率说全量完成。

---

# 历史VLM规范（不覆盖当前已批准纯文本主线）

## 1. 目标与旧版本边界

目标是音乐驱动舞蹈规划/生成的编舞动作监督，不是场景caption或姿态摘要。正式文本为一行舞种标签，加按既定时间窗口组织的英文动作正文。

旧第二阶段“全片最多12帧→最多8个端点动作→无时间窗fused_caption”与本目标不符，不能作为合格新版本正文。原视频、原DWPose/MS事件、原BeatThis缓存、原rich及过程证据保留；不清旧预算或覆盖旧结果。新版本使用独立revision来源防重域。

## 2. 输入职责

- **rich：只提供舞种。** CPU只提取舞种候选/已有标签；完整rich正文、服装外貌背景、rich动作和表达意图不送VLM。舞种缺失/冲突在内部记录；不能用音乐类型或默认freestyle冒充已知舞种。
- **MotionScript：原structured events/windows，而不是去重ms_text。** 保留原事件ID、时间、部位、方向、重复与连续关系作为弱监督线索。二维夹角/坐标变化不是最终自然语言动作，更不是未经视觉确认的真值。
- **视频：每个窗口都有对应有序native PTS帧，覆盖起、中、末及运动过程。** 不能用少量全片抽帧取代所有窗口；只依据原RGB可见证据纠正/补充动作。画外/遮挡不等于静止。
- **节拍：复用原BeatThis实际beats/downbeats缓存或既定plan。** 不从BPM、音量峰值或每四拍发明小节相位。

## 3. 唯一时间策略

使用原计划的merge_short_edges策略：保原raw downbeat/bars、首尾残段和raw_window_indices映射。首尾残段小于原1.5秒参数且有相邻完整检测小节时并入邻窗；不合并短内部小节、不改变既定窗口边界。已有合并窗口计划优先直接复用。

当只有raw detection缓存但未生成全量合并plan时，使用原ms_music_windows.bar_windows相同规则确定一次plan，冻结后MS/VLM/正文/审核共用。检测首拍不足时按原策略保留实际残段或整段并标insufficient_downbeats；不能谎称完整小节或换成固定/BPM窗而不记录。

音频零时刻与归一视频零时刻的原假设保留为未验证元数据。按小节分窗不证明踩拍；窗口起止是描述观察范围，不是精确动作起止。

## 4. VLM审核动作

对每个原窗口：
1. 查原时间事件与视频中可见动作，修正明显规则噪声、补漏的主要动作。
2. 将可信的关节规则整合为自然动作短语，不逐条翻译“二维夹角增大/相对肩坐标移动”，不编造动作来替代未知。
3. 保留有意义的动作先后与真实重复舞步；重叠小信号可合成同一运动，不把真实重复去重。
4. 描述脚步/腿、手臂、手、躯干等可靠部分；不要求每窗固定写全部部位，不做Feet/Arms/Torso机械检查清单。
5. 部位看不清则记录evidence_limits；保其他可靠动作，不抹空全窗。
6. 不推断不可见接触、承重、掌面或手指细节。解剖左右只有视觉能确定才用，否则用image-left/image-right或one arm/other arm，不由2D模型索引擅自指认左右。

## 5. 跨窗口连贯

审核时看到相邻窗的原动作、前后边界帧。判断前窗动作是否继续、结束、返回或确实重复；每窗只写自己的时间部分。

允许continues/returns/then等有证据衔接；禁止为了连贯编出新动作、次数、因果或把前后动作移入当前窗。窗口边界完全不变。跨窗连续关系保内部continuity_notes。

## 6. 正式正文

全部英文，格式如下（格式示例，不作为事实数据）：

```text
Dance style: <style>
[<start>–<end>s] <coherent body-motion sentences for this exact window>
[<start>–<end>s] <next-window motion with supported continuity>
```

每窗通常2–6完整自然句，优先主要动作、方向、顺序、重复与幅度；不硬凑句数，不追求字符/词数变化。时间只在行首envelope，正文不插逐事件数值时间或坐标日志。

幅度写在相应运动短语里（wide arm sweep/small side step/deeper bend），不新增必填global段，不混入音乐能量、情绪或“整体energetic”评分。

衣服/颜色/发型/妆容/人物外貌/表情/背景/道具/灯光/镜头/美观评价禁止进入动作正文。身体部位如head/chest/hips可以定位动作，但不能转成外貌描述。

来源、修正理由、置信度、未知和证据限制留内部JSON；只有真正缺少全部动作证据时用明确的Insufficient visible motion evidence代替虚构正文。英文句子必须完整，EOS/字符上限不代表没有截句。

## 7. 验收与规模化

先固定3个真实case检查原计划→视频窗口帧→audit→最终正文；看格式、时间保持、英文、舞种、外观排除、完整句、动作保留与跨窗衔接，不以schema/token/EOS代替内容。

本轮首个3case候选已经发现技术枚举和截句，拒绝全量放行。修正版另发现coordinate检测误匹配coordinated，必须修检测器而非让正确自然动作被拒绝。所有候选原件保留。

3case达成规范后才部署常驻批模型；原视频一次拷本地、CPU准备与GPU重叠、结果本地批封包、单节点上传器、共享commit+ACK可靠计数。新版本预算按source/revision排他，失败/未知不自动重试、旧结果不算新合格正文。

## 8. 全量来源接入

已定位176771条原稿，但01–02最后一票仍在生成；不提前宣称这就是全集。01–02继续原MS提取，不因本轮VLM停靠而停止。

新增可靠batch commit中的source row/draft/manifest/BeatThis缓存按原ID+split增量加入同一新revision输入目录和唯一模型池。已加入ID不重复；取原数据，不取旧fused_caption。使用最终assignment/完成回执核缺失、失败及未生成来源，不能用API调用数或文件数冒充全量来源数。

如果最后一票使用不同edition、目录或归档格式，必须按实际receipt适配，不能猜路径/制造hash/改原source绑定。只做一次输入边界完整性检查，不反复全库媒体审计。

所有结果未人审，不可训练；协议格式通过不等于人工语义批准。

## 9. 六节点A10040GB的新规范分配

用户明确提供01–02共2节点16卡、03–06共4节点32卡，合计6节点48张A10040GB。全部只执行最终english_merged_edges_motion_revision_v3，不再执行旧无时间窗/无舞种摘要协议。

已实际排他分配第一批72000条原始来源，每节点12000条；每node两TP4组（0–3、4–7）共享该node配额。source IDs六节点之间不重叠。原主群16副本已领来源不变。

来源位于原唯一最终model_pool.db，新增external_allocations节点映射；状态external_reserved对主群只取unused的claim不可见。外部专用API只取本node映射中的external_reserved→原子assigned，未知/失败/已有模型预算不重派。没有新建6份完整模型来源池。后续新增输入仍增量加入原数据库，外部先跑此明确配额，再根据实际吞吐补分，不自行跨node选源。

分配共享目录为ROOT/motion_revision_english_merged_edges_final_20261028_v1/external_six_nodes_allocation_v1。每node有a100-extra-0N.manifest.jsonl及source_keys.jsonl，allocation.json SHA218e0d3fd1d704798ad4f75650aa4713d6d83ab91dd995bcd740d4c7f39d3629。

可执行包motion_revision_six_nodes_runtime_v1.tar，SHA9893335c3404e8d4bd79998cfa8421e38ac2634ccef3ab1fe2326fc345b2c24a。包不含token，包含本规范、六node精确清单和新分窗业务逻辑。N2本地中央API19114已真实HTTP验证六节点各12000预留，外部modelclaim0。生产模型尚待各node本地启动，不把包/CPU测试算48卡已执行。

每node先启动一个原版32B BF16TP4（8conversation、32768context、逐窗native采样、6144output预算），实际load及新结果正常后再开第二个TP4。第二组准入重新核160GiB有效内存与/tmp8GiB，运行80GiB门不变；不能照搬旧50000/32conversation摘要启动包。待旧MS或旧错误协议工作安全排空、GPU空后才使用相应卡，不杀未知进程或删除锁租约。

每node一个CPU上传器，两个owner均在EXPECTED_OWNERS中；每50来源/5min批封包，可靠shared commit和localACK之后才允许回收自己的缓存。共享上传仍进入最终V3 production_results，源postprocessed.json保fused_caption、dance_style、windows、visual_audit、window_plan和continuity_notes；服务一次核source/revision/window证据，不做重复全库媒体审计。

真实隔离HTTP回归验证：主群无法领取external_reserved、外部node不能领取其他node、foreign video query被拒绝、本node结果上传ACK与原同池终态一致。没有生产来源claim或GPU模型调用。实时统计已增加外部六node分项，主群已核16活模型与外部尚未独立验证的活体范围分开。

最新六node分配后的快照：新正确正文7160、失败封存590、16主群已加载活副本、30min6477.37来源/h；原输入prepared127473，01–02最后一票added32023/45185，输入缺口9记录待修。旧错误133k/旧50k产量不计入新规范完成。

## 10. 保持正文协议的并发提速试验

用户要求优化V3吞吐。N2实际8来源视觉API均值40–42秒，融合约21秒，非API中位0.73–0.76秒，idle0；主要瓶颈是推理，不以丢掉舞种/分窗/动作连贯恢复旧摘要速度。

只试N2两pair共4卡，其余14主群模型继续。原32B BF16TP2、prompt/原窗口/原帧/6144输出上限全部不变；两个8源原claim合成16会话，max_num_seqs8→16。每条原owner/pool_claim/原packet_ref与永久预算保持，两次模型调用/source上限不变，不缩批重试。真实16源CPU合批/防重/sharedACK集成通过，实际GPU效果待核。

停止前N2台账/root/stage2_motion_revision_final_main_N2_20261028_v1/batch16_before_drain_ledger.json SHAc12be25f7270ec5a6fe94cb4d31df8ff5a7907f0bec61c6d826dd31d7ba2fa82，含config/queue/out和原来源split/id/state/owner/batch。旧feeder只写自有FEEDER_STOP完成已领包，原模型自然完成ready→manager恢复holder→safe三flag；不信号killGPU、不删结果/claims/锁/租约。全ACK后自然交接唯一node uploader。新batch16代码独立，state另phase、活配置pointer保历史，旧源码不热改。

必须测显存/KV预占与可靠来源墙钟提升，schema/模型驻留或token速不代表收益。外部03–06旧协议已安全排空，新V3只待token可信交付；不公开token、不绕过hostkey。

### batch16实际有限样本与逐节点扩展

N2两副本实际各64来源（128合计），各58合格正文，计算到finalcommit墙钟约390–392秒；合格532.44/535.06条/h、终态587.51/590.41条/h。显存峰74681/74493MiB，原日志无preemption，原claim/packet/<=2预算映射真实核过。证据/root/stage2_motion_revision_final_main_N2_20261028_v1/batch16_live128_acceptance.json SHA b56dd352bbe7dba0445ee835937d4e7614ca309d0dd5d59766bb47e3afdecc4f。

该样本不含模型加载且非同source配对，不能当全局30min或质量提升。与batch8同规范样本每8来源视觉40–42sec/融合21sec相比，合格/h约改善25%，不是恢复旧无时间窗摘要的3–4倍。N2两个新模型继续运行，逐节点仅先升级N3三模型，其他13个模型继续。

N3停止前台账将写原node root，包含当前config/queue/out与原source状态；只FEEDER_STOP自然完成原已领packet、原GPU到safe三flag、全node sourceACK后唯一uploader交接。原源码/原结果/原永久预算不改，新phaseBatch16另state和runtime binding，未调用source才可新revision原pool领取，外部72000预留不碰。

外部03–06已由用户反馈旧协议安全排空、新包部署/中央可达，仅缺token。本端按用户明确授权直接交付真实token并给03本地0600写入命令；没有把token值写notes/记忆或公共盘，不再以hostkey传输阻挡外部启动。等当地actualV3 load/sharedcommit回报，不能把token交付算48卡已启动。

### 单副本32条合批验证（未宣称收益）

用户询问32条是否可合批，已准备4份原8来源microclaim合成32会话、max_num_seqs32候选。模型、prompt、原图像数与原merge_short_edges窗口、英文正文和6144上限全部不变；每source原owner/claim/packet/永久stage预算映射保持。真实CPU32来源合批两次chat调用与sharedACK、防重回归通过，生产GPU32未验收。

仅N3g45计划试32，N3g01/g23保持16；其他13现运行模型不修改。N3当前旧8批供料自然排空后，旧manager safe三flag、全node ACK、唯一uploader交接，再用新16/32候选；原候选启动等待器pidfd TERM后以独立新等待回执替代，未发GPU信号或重派source。

32提交会话不是32真Running；原32k上下文、多窗口图像可能触发KV预占。必须记录实际GPU峰、preemption/耗时/合格正文/h，与16样本比较，不以合批数字或GPU100%冒收益。不因为32OOM缩批重试已领source，不减少窗口/帧以换速度。

### 单副本32实GPU有限样本已完成

N3g45已实际运行32会话，原4份8source microclaim映射/预算和原prompt/window/frame保持。前64来源中53正文通过当前协议检查、11封存失败，计算开始→本地finalcommit339.615秒（不含模型加载）；正文561.81/h、终态678.41/h，GPU峰74555MiB，无KVpreemption日志。N3同时段两个16会话各64source正文515.87/h与467.01/h；32有限吞吐高约9%/20%，没有2倍收益，来源不配对不可认定质量更高。

证据/root/stage2_motion_revision_final_main_N3_20261028_v1/batch16_vs32_actual_first64_per_pair.json SHA15010b3f196c66f2477818e302ac3487aa62096f87f3a96dd99d28a01c3528bc。原预算与来源界保留，失败不縮批重试；接下来现有32副本持续测至少30min可靠sharedACK来源速，未经稳定证据不全群推32。

混合试验启动器的上传node曾默认N2，实际N3已在未生成任何N2批前纠正为N3；旧源码保留，只stop自有CPU uploader并安全启动另名新single uploader，GPU模型不停止，错误节点批次创建0；回执batch16_uploader_node_correction.json SHA19a9f116ccdc05e844c32aba466c72382a230baee16769481713e6994c904e0f。后续部署源已去除固定N2参数，保未来正确标识。

### 32合批30分钟可靠提交测量完成与内容边界

N3真实稳态窗口1800.042秒，sharedcommit加本地ACK来源去重，无采集缺口，包含空闲/暂停/上传，不含窗口之前模型加载。16会话两副本新当前协议接受来源248/266，495.99/531.99条/h，失败24/22；32会话新接受311，621.99条/h，失败29，idle0。32比两16快17–25%；来源未配对，不能认定语义质量更好或所有批都线性收益。正式报告/root/stage2_motion_revision_final_main_N3_20261028_v1/batch32_steady30min_20261028_v1/report.json已存在。

随后三样本原帧review发现：脚部不可见却正文声称无footwork/腿stationary，specific解剖左右/承重超证据，continuity_notes截句与泛化连接词。review保存于最终目录batch32_review_samples_20261028_v1/review.json SHA012e5cdae3dc844211dec9785a861af723e5d0590900de751401c4565f9e0df4。当前协议接受来源是格式/关键词门通过，不是动作正确率或用户验收通过。16/32并发稳定不代替内容质量；未修改在跑prompt、未擅自全量再重跑、不全群推广32，待与用户确认最小内容约束修正。


## 最新局部纠错诊断结论与回退建议（2026-10-29）

用户要求VLM只轻微修改原事件字词，不进行整窗重写；若MS错误少，考虑仅LLM轻微润色。固定同3例分别62/42/46原事件，典型事件时长0.25/0.25/0.3125s，每窗8帧只有24/17/34事件可获得事件内两帧。稀疏抽帧不能与全native-frame MS等价用于逐事件纠错。

最初10窗显式corrections全空，是无可核查纠错记录，不证明MS零错误。两轮受约束视觉诊断及词级few-shot仍出现不可见脚部当静止、侧别/承重细化与大量改向提议。最新minimal v2已固定可审事件和合法帧对，词级提议为21/62、16/42、33/46事件；不得将70提议称70纠错。所有视觉补丁本轮都未应用，真实净纠错收益未独立核实。

同时定位并修复助手校验器两项误判：动作增大/减小被标成direction而非motion，属于内部reason分类不一致，不自动等于越界动作；原有relative-to-shoulder定位信息不能一概视为机械技术日志。词级检查改为反义方向/动作词对，不靠reason细分误拒；自然语句允许原支持的相对部位定位，仅拒完整投影角/数值/坐标日志。对既有模型raw重新CPU规范化，零新模型调用、原失败记录不覆盖。

最新pure MS＋LLM多事件few-shot同3例均通过修正后的语言/格式/证据边界检查；正文较前版简短，pImh没有添加脚静止/承重，wToy保留contemporary / urban混合舞种；这不等于人审或语义全部正确。三例结果及原raw provenance保存于final结果目录内 fixed_three_MS_LLM_wording_v2（不计生产来源完成，未人审不可训练）。裸左右方向仍必须按原image-plane含义解释，不能当解剖侧别。

建议主流程回退到原events/windows＋rich舞种＋LLM轻微润色：不读取VLM重写正文、不加入未验证补丁；只做英文自然表达、局部顺序/真实重复连接和必要字词修订。VLM如保留，应仅作为疑难定向诊断，不作全量自由重写。在用户确认全量路线前，不擅自切换整群、不重置来源/阶段预算、不删除旧结果，也不称三例证明全库无需视觉纠错。

资源：用户明确先不kill N4，N4未动。非N4最可用为N2；g45旧诊断空等和g67旧孤儿引擎已授权精确清理，三native-safe全真；g23最新诊断结束后同样原生恢复holder。只用g23两卡做同3例，g45/g67四卡未重新加载。
