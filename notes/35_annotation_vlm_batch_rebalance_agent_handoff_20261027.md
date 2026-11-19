# 标注后处理、批量提速与外部来源重分配：新 Code Agent 接管文档

> 本文用于新会话接管，不是生产完成证明。最后读取现场：**2026-10-27T16:16:06Z（2026-10-28 00:16 CST）**。
> 当前工作从原“MS优先持续打标”转为用户明确授权的 **已有数据 VLM 视觉审核、LLM 融合重写、批量提速**，并行推进外部剩余来源重分配。
> 所有计数、PID、handle和资源证据都可能过期，接管后先读动态登记与实际执行节点，不依据本文硬编码启动或释放资源。

## 1. 用户最新要求与授权边界

用户明确要求：
1. 暂停了旧goal：原“持续推进MS、VLM其次”的自动Stop循环不应恢复。
2. 主集群对目前能够访问的已有标注数据部署VLM/LLM，做审核、纠正、整理和**一份融合caption**。
3. 已要求真正批量处理；希望达到**至少5,000条来源/小时**，可一次处理更多数据，不希望靠牺牲质量提速。
4. 主集群8节点64GPU，可以多部署模型。但目前实采发现大量GPU属于既有任务；尚未获得明确暂停哪些旧caption/full12k任务的答复，不能把64卡总数当空闲64卡。
5. 外部剩余正常来源约三分之二交给01–02，并告知03–06目标缩减。**这个重分配范围已经明确授权，不需重复确认。** 但必须有真实未领排他集合与安全接续合同，不能热改运行票据、抢已领来源。
6. 结果与交接材料应放可共同访问共享盘；本地模型、临时视频和工作缓存保留快盘，不重复整库搬运。
7. 继续实际推进，不以配置存在、manager running、模型驻留、CPU测试或不断巡检替代生产。

仍未授权：
- 两条旧rollover unknown来源新人工重做尝试（曾问一次未答）。
- 自动重试任何已成功、partial、needs_regeneration、封存失败或已消费stage预算。
- 强杀未知/其他会话进程，删除锁、claims、lease或STOP；暂停具体旧任务以回收更多GPU也未明确获批。

## 2. 权威路径与工具注意事项

~~~text
REPO=/apdcephfs_gy4/share_302533218/cedricnie/H3-posttrain
ROOT=/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1
RT=$ROOT/runtime
历史接管记录=$REPO/notes/34_caption_draft_pipeline_takeover_optimization_20261026.md
当前文档=$REPO/notes/35_annotation_vlm_batch_rebalance_agent_handoff_20261027.md
~~~

优先读取：
- RT/ms_vlm_postprocess_jobs_registry_v1.json：所有已分配/已消费后处理来源计划、当前phase/config/handle。
- RT/ms_postprocess_patrol_registry_v1.json：当前后处理监控指针。
- ROOT/ms_user_existing_postprocess_patrol_20261027_v2/latest_summary.json：当前四路可靠进度。
- RT/ms_novlm_continuous_jobs_registry_v1.json：旧MS动态登记，原worker已结束，不用其中expected_running旧值当生产。
- RT/ms_external_monitor_registry_v2.json：外部增量提交监控指针，非GPU活体证明。

工具/现场坑：
- 专用文件Read可能对大单行JSON截断或返回Error reading file；可用Python读原bytes并只输出必要字段。不要据Read显示错误直接改文件。
- Ceph递归Glob/Grep常超时；优先exact路径、一次浅层目录或小元数据，不扫22万视频/全部tar/权重。
- Bash超过前台timeout可能自动后台化，使用真实TaskOutput，不重复执行。
- 任务编号会随team切换碰撞；TaskUpdate前核Subject，切勿把旧训练任务当新标注任务。
- 后台handle可能在新会话不可见；实际queue receipts与执行节点/proc是权威。TaskOutput unknown不等于worker停机。
- 本session多次daemon低内存/网络错误导致subagent无法继续，有时只有源码/计划完成而物化没有完成，不能照代理“完成”直接报生产完成。

## 3. 当前实际运行：四个32B副本、8张GPU

模型：Qwen3-VL-32B-Instruct，本地 /opt/models/Qwen3-VL-32B-Instruct。
解释器：/opt/venv_vllm/bin/python；BF16、TP2、max_num_seqs=8、max_model_len=16384、gpu_memory_utilization=0.88、prefixcache=true、trust_remote_code=false、seed0。
每来源：本地单次视频复制且核videoSHA → 12张native采样JPEG → visual_audit → caption_fusion → normalized/raw/result/commit可靠发布。
同一个VL模型也用于text-only融合重写；不是额外独立LLM副本。目前实际生产仍是**串行逐来源v3**，batch v4尚未上GPU。

最后监控快照（约16:15:31Z，主在16:16:06Z读取，age34.8s）：

| 节点/GPU | 当前phase | 当前handle | worker PID/startticks | 已提交/250 | 来源请求调用 |
|---|---|---|---|---:|---:|
| N3 0–1 | ms_fullframe_vlm_postprocess_v3_archive250_n3g01_20261027 | 5GbOv3 | 3536109 / 556954998 | 119 | 238 |
| N2 2–3 | ms_fullframe_vlm_postprocess_v3_archive250_n2g23_20261027 | 3Skt3k | 1388888 / 557009175 | 101 | 202 |
| N2 4–5 | ms_fullframe_vlm_postprocess_v3_archive250_n2g45_20261027 | wQUdN6 | 1391610 / 557049467 | 92 | 184 |
| N6 2–3 | ms_fullframe_vlm_postprocess_v3_archive250_n6g23_20261027 | P3JFbk | 2569876 / 557057101 | 89 | 178 |
| 合计 | 4副本、8GPU | | | **401/1000** | **802** |

四路worker_alive/model_loaded=true；无进程failure/alarms，尚未completed/native_safe。
各config路径均RT/<phase>_config.json；输出ROOT/<phase>；资源queue ROOT/queues/<phase>。
监控：**TLo47h**，实际PID3243607；ROOT/ms_user_existing_postprocess_patrol_20261027_v2/latest_summary.json，每60s读取动态登记。该监控持续循环，不应因一个有限批完成就误认为全部供料结束。

已核当前config SHA：
- N3：81cc02812a3c3a7e87df93b480753fdf5ef65231d8627f587c3204d2f604131c
- N2g23：50b70d813338246031035730632bd645f1ae7767c46772d9afeb1d4271e74944
- N2g45：d5264697399739de023c87b4bb2e8d304e4caf2bb1ecc576d412fe01b5571d64
- N6g23：25377df5e1517fcae2df990c6b9b9d24d1aa2f4ba6b9224dbe18ed60afb29ee4

最新resources_safe/holder_restored必须来自当前实际前驱，不能复制过期holder身份。主群原生准入 **80GiB安全+80GiB加载=160GiB** 不降；同node按pair加载后重新核共享headroom，不按GPU复制节点预算。

### 已完成先行批次，不要重领

- 旧04/05/06 pilot后处理：diag3+diag3+180+114，共300来源预算已消费。180/114完整重放与资源终验已完成；初始6diag包含封存/风险，不能重试。
- 新用户user200：01/03 pilot各100，合200；三路67/67/66均summary完成、worker_exit0/native-safe。handle MXMx4D、eX6b0D、peoLhq，登记completed_native_safe_pending_full_replay。需要最终完整summaries回放，不把字段safe等同新完整验收。
- N3归档first20：HKRPyJ，20/20、40请求、exit0/native-safe；5GbOv3接续程序已独立summaries与worker_exit.summaries全值比对后启动下一250。
- 当前四250全部属于同3000master pool的前1000，彼此排他，不能新batch方案再次使用当前plan剩余行，除非旧票明确退休且证明这些行本阶段还未领取并完成释放。
- 重复启动4FJEcN、jvjbIa被resource_manager.py:455 queue.mkdir(exist_ok=False)拦截，发生在资源锁/租约/holder释放前；未重复模型工作。不要清queue绕过。
- EcnCNS是早期配置生成前失败的旧N6handle，现正确handle是P3JFbk。

## 4. 已观察到实际视频复制，不只是模型驻留

2026-10-27约15:31Z并行双采样，间隔15s：
- N2g23提交39→40；g45 28→29；N3 56→57。
- N6提交数当时27未增，但 /tmp/ms-fullframe-udDrlvBhObywNvZn-.../udDrlvBhObywNvZn.mp4 从0→1,251,571bytes，与原src stat尺寸一致，随后12JPEG出现；worker读写量增长。
- N2也观察新来源完整本地mp4和12JPEG；下一来源0字节是复制开始，不要立刻认坏文件。
- 完整commit记录single_source_video_copy=true、model_revision_performed=true，visual_audit/caption_fusion actualcall=true且finish_reason=stop；result/commitSHA和来源匹配。
- 临时副本可靠commit后回收，目录消失是正常；输入视频源路径仍共享，临时视频/JPEG在各节点/tmp。
- 上述probe仅stat文件/proc/io，没有额外读媒体payload或调用模型。

## 5. 当前速度与批量提速状态（接管最优先）

15:20:45–15:35:45Z四路近同步实测：
N2g23 +21、N2g45 +22、N3 +22、N6 +21 = **86来源/15min，约344来源/h**，该窗口流程封存0。
最近10commit间隔均值约39.6–42.7s/副本。
最近3来源request→raw落盘窗口：visual12–20s、fusion10–14s，含资源gate与落盘，**没有精确纯推理计时字段**，不能称全部时间只推理。
visual约432–737 tokens，fusion368–507 tokens；其余开销复制/抽帧/多次native gate/共享metadata和durable发布。
现有max_num_seqs8没有真正用批输入，旧call_stage逐source调用llm.chat且两stage顺序。

### 新batch v4：已经编写，尚未GPU部署

~~~text
RT/ms_fullframe_vlm_postprocess_batch_v4.py
RT/test_ms_fullframe_vlm_postprocess_batch_v4.py
~~~
最后读取（16:16Z）源码：
- caller 57961bytes，SHA **fd05e1b8742e52082682223b810d0f14f70cb3cbcccb77010c6b0483380c37ab**
- tests 14038bytes，SHA **4a215bd56beef23c73ec0256061c569eaf43d60971189e46dbaf9a30817aaa83**

准确关键入口：
- caller:112 input_pair；:171 validate_origin；:251 validate_common；:322 gate；:336 stage_images。
- caller:489 call_stage_batch；:567 run_sources；:659 source_summaries；:737 summaries；:833 main。

实现：cfg.batch_sources只4/8；多conversation一次llm.chat(requests, sampling_params=listparams)，视觉小批一次→各row normalize→合法行fusion小批一次。每来源独立schema/ID/split及最多2请求；另batch_chat_invocations计Python调用数，8来源两stage正常=16来源请求、2chat调用，不混口径。短尾、单行prepare/schema/length错误只封对应行，业务批调用异常封所有affected stage不自动缩批重试；预算在调用前O_EXCL。
保12采样帧、200704pixels、16384context、1800token、原风险/FLAGS，不靠少帧/短caption换速度。
新v4也补强归档必须3locator/5members/原logicalNPZ/存储绑定（旧v3存在字段可选缺口）。

已完成 **18项CPU测试**，J3nVgV当时completed OK：4/8真实mock batch API、短尾、错ID/重排、length、异常预算不重试、已claim跳过、空证据、rawresponse篡改、原资源函数AST/recipe保持。未调用真实模型。
后续尝试增加strictarchive/no-repeat测试时Edit未命中单引号main（实际文件双引号），因此**新增测试未成功写入**；不要冒称已额外覆盖。
当前还需：
1. 读取并审查当前v4完整代码/tests（上面SHA为核查值，不是不可变CPU冻结证明）。
2. 补strictarchive正负例、batch mapping与截断/异常failed记录重放验收；对只含错误record的部分raw证据作核验。
3. 固定代码后完整CPU suite并前后SHA一致、写独立验收，不覆盖已冻结v2/v3/原proof。
4. **用未分配/未领取本后处理stage的新来源**做真实GPU batch4，再独立新source测batch8，证明vLLM JSON grammar、显存、吞吐、真实视频输出。不能从当前四250plan直接另phase重领。
5. 实测稳定窗口后再扩大来源/部署；5000/h当前只是用户目标，约比344/h高14.5倍，未达到，不能承诺靠batch8必达到。

已安装API实读：N2 /opt/venv_vllm/lib/python3.11/site-packages/vllm/entrypoints/llm.py:615，LLM.chat支持Sequence[list[message]]与Sequence[SamplingParams]，返回顺序对应输入；所以批输入不是猜API。
注意daemon低内存导致vlm-batch8-implementation代理2s失败，**批量代码由主会写**，不要等待该代理恢复或重新多人同时写同文件。

## 6. 归档草稿供料、原始绑定与入库故障

冻结串行v3：
- RT/ms_fullframe_vlm_postprocess_v2.py SHA c8492639ab8155b24250db722ffeae7973fb28b6d41f8ad3e2b98442ab60388d。
- RT/ms_fullframe_vlm_postprocess_v3.py SHA **87a612991b7781aed1b7fbaffc48f11062a853e19a2a80a71f85f4a9994073df**。
- RT/ms_vlm_archive_inputs_v1.py SHA **21e908102b20ba91533953c1e6d46af36f32e1622e8b54cf808870bbff1dbf75**。
- RT/test_ms_vlm_archive_inputs_v1.py SHA 9e372f2b413d5b39c905cabd7b33d2b3580e0627e69e8b9f76836672ea840d9d。
- 原resource_manager SHA **6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6**。

SOURCEPOOL=ROOT/ms_fullframe_vlm_archive_inputs_20261027_v1
- frozen_code_cpu_proof.json SHA feab63539c7754522dddaee14a011aeeb03565bc26198a96cec4b20e334fc88b；28CPUtests，已冻结不改。
- source_selection_proof.json SHA d5edc46b0782a3c44837e72efcc555c3c46b05f882ecb05ba9011d904dca3d56。
- source_keys_sha256 1c1ab4e5a96cdbdaf0f7375b8f81934b7a437e1a27efdaebad5ed265837b0404。
- 三个明确stream1000 n2g2/n2g3/n2g4各1000train，150commit，实际old500 overlap0；不是全22w扫描，也没有把val/test改train。
- 首first250_manifest.jsonl SHA **4bb68a92755e6d32353b46d50fb3337b8426d3909127310fb9e11b1a565dfe63**。
- first250_allocation.json SHA **8d098e90442cff62e4e14ecb207e28e8f7eab51a9fb8114e88ffb616c5644229**。
- next750_allocation.json SHA **c491c4605b0787a7373d2b645ca5e2d7b6f894915c9511be39df8b25290bab56**，三份250与首250互斥。
- n2g23_next250_plan.jsonl SHA b8b9e3ce31630d835bba9942b924bd1f22364ebdac1329dacda55d7489c9b4fc。
- n2g45_next250_plan.jsonl SHA 02dadbc3523f2bbab836da240c4492fee496224825a7663e53b82e05c429419d。
- n6g23_next250_plan.jsonl SHA d22cbd724356f2847a5da4aaef4d12f412b1669b0ac0b13130686dca479c9a89。

**全3000 manifest.jsonl/allocation.json截至16:03Z最后检查仍不存在**，不能称全部已入库。已有约1082目录/55batchproof，很多完整，但目录存在不等完整proof。

中断细节：
- 原CPU任务35zeFY被daemon回收/killed。代理archive-existing-vlm-input-writer低内存失败。
- 主恢复CPU任务zcfelI读到1080完整来源后失败JSONDecodeError：CfHpXYxGWnDlXKsx的origin_proof.json为0bytes（断写），两原draft/manifest有效。
- 保留该0bytes原件，主在 ROOT/ms_fullframe_vlm_archive_inputs_recovery_20261027_v1/ 独立恢复原rawJSON/proof，iYD1dx exit0，未再次hash该已验tar、未调模型、未改原claims。
- recovery_receipt.json SHA **b8de72cd8b7d7cb1ffbf26901a8439987ce458bb408fff9dcbf061888ff51649**。
- 恢复originproof SHA db2704eb99130364bbec77273549d03df8596dbfd90d6ff370ef13a46441851f；draftSHA 7dba0e870eafdb6767b7730be6c611281eee0eed0ce88a253c363babb3f7548b；manifestSHA db19da04f759f884fa8e6e91c3bf701560d7ccc1b5c05bf2e3488ce3f9036b0c。
- **整3000恢复尚未继续完成**；接管应仅补missing metadata和恢复路径，保完整原件，复用已验证archive stat+SHAproof，不能反复从头wholetarhash/新建stage预算。恢复程序要基于冻结selection而非当前registry固定len8（现已追加新jobs）。

桥接原理：原MS是batch.tar内部draft/manifest，source.fullframe_manifest_path为不存在的virtual archive member逻辑路径。新目录复制两JSON**原rawbytes不重编码**，plan带fullframe_origin_path与origin_proof_ref；v3验证physical ref与原logical path分離但原canonical fullframe binding digest/observationsSHA不改。不能简单改draft.source路径冒原bytes。
whole archive SHA只在入库边界一次，明确读到了封包NPZ字节但不解析NPZ，不读视频源；热gate只小JSON/proof/commit，不重复tarSHA。
冻结v3 locator原条件可选的普适缺口已发现，不hotpatch运行程序；目前四250输入已逐源外层strict核3locator/5member/storage/virtualNPZ/commitpath及SHA，不只是首20。新batchv4在自身validate_origin补必需字段。
主当前动态MS登记七early497只有497train，N6旧remaining成功1559全val/test；不能从这几个“最新票”凑3000train。

## 7. 全8节点实际资源盘点与扩容限制

可信SSH映射（来源RT/curated_resource_manager.py及已有配置；BatchMode=yes/default known-hosts，不关闭hostkey）：
N1=28.216.19.69；N2=28.216.17.70；N3=28.216.19.213；N4=28.216.19.143；N5=28.216.19.137；N6=28.216.19.225；N7=28.216.18.144；N8=28.216.18.136。

14:07Z实采八节点，没有完全空GPU；本任务能合法使用原MS可证明holder的只有当前四pair。其它任务范围：
- N1全部curated cache/prepareahead/collab4 VLLM，held锁/activelease。
- N2g0 cached producer holder，g1旧conda无当前可信proof且held；g67 caption_two_stage_local_pool2，manager4113508；当前本任务g23/g45。
- N3g23/g45 full12k_visual_s0/s1，g67 curatedcollab21000；本任务g01。
- N4全caption_union_pool3/two_stage，部分同GPU旧holder与VLLM并存，不许挑holder释放。
- N5g0producer，g1旧condaheld，g23collab4，g45two_stage，g6full12k_roi，g7独立review holder；无常规可用pair。
- N6g01repair、g45/g67 full12k_visual_s2/s3；本任务g23。
- N7g01旧conda+producer锁，g23/g45collab3，g67独立review holder；14:07共享headroom156.97GiB<160不准入，不永久禁止以后fresh复查。
- N8g0–3producer，g45prepareahead，g67consumer全occupied。

旧headroomGiB快照：N1 300.565,N2 481.625,N3 196.846,N4 433.213,N5 373.508,N6 313.501,N7 156.971,N8 346.092；当前load后必再核，不拿旧数值并行许诺多个pair。
8节点都发现14safetensors总66,714,912,872bytes和config/index/venv可执行；完整25inventory=14weights+11metadata，原manager仍check25sizes/config/index小SHA和node stageReceipt，不重hash66.7GB权重。
**N2 XML minor_number与raw nvidia-smi index不同**，曾盘点误报后纠正：raw0→minor2,1→3,2→0,3→1,4→6,5→7,6→4,7→5。锁dev按hex整数major/minor/inode归一化，裸字符串zero padding会错误认为空锁。

要更多模型：用户已表达希望扩容，曾明确问是否允许安全暂停具体旧caption/full12k任务让出资源，**尚未答复**；不应再次泛问是否继续，但不能signal未知任务或绕活lease。可先将batch4/8真正上当前8卡做提速验证，再对明确可交还的任务边界扩新冻结wrapper/pairs。

## 8. 外部来源进度、三分之二重分配（授权已获，转移未做）

全集222939；原主范围=62939（排除外部160000）。主已发布62903规则/31封存/3diag/2unknown，无可新领MS预算；用户现阶段已要求本地做后处理，原暂停goal不应不断Stop空转。

14:00Z外部快照：01–02正常剩余=6604未领+169在途=6773；03–06 14:02Z remaining48020；正常剩余参考54793。排除4unknown，不重新分配它们。
两群正常剩余2/3给01–02：ceil(54793*2/3)=36529；01–02原有6773，所以参考转入29756，03–06参考保留18264。
**运行继续期间这些数已变，不是freshmanifest，也不是实际转移数量。**

请求：ROOT/external_remaining_rebalance_20261027_v1/request.json
SHA **54f397b3c51c939b5c65ebcf18e61b180e7de6dade438bf8316df31e0b5f259b**。
status=authorized_scope_pending_fresh_source_release；source_rows_allocated0/source_transfer_startedfalse/production_switch_readyfalse。用户授权已明确，不再次问是否按2/3分；只能在实现/验收边界后行动。

known unknown：
- 01 rollover train/DTxRgnSZZqFSijgW、train/KYBwQcXqARLIoECt：原领取worker退出，无完整本地结果，automatic_retry=false。
- 02 clusters train/kaemOaybVvUjgJam、train/yuMvSKCNpkqKymqh：未明状态，排除不重试。

后续最新用户反馈：
- 03–06 14:31Z reliable32780、深验32740、未提交47220（不等未领），原16GPU继续；没有转出/退休。
- 01最终44975发布=44950partial+25needs，23failed、2unknown；clusters39988published+12failure=40000，rollover4987+11failed+2unknown=5000。
- 02当时40887发布=40867partial+20needs，33failures，继续4GPU；4080未闭合不能全当未领。
- 当前总体总数用最新各epoch精确来源集合，不将rollover与主62939重复加，不把终态failed当草稿/有动作成功。

## 9. 外部当前缺什么，已取得什么

### 03–06真实没有已验证在线drain

共享报告：
ROOT/external_ms_unified_clusters80k_20261027_v1/results/a100-extra-03/transfer_drain_blocked_20261027T1432Z.json
SHA **95a328e5d0dffd1533a5b1f56bbaa312303bf5e52b772b4eff038c21b9bf5e57**。
实际runnerSHA **95eff3b6af6104d090b318647bb6442aa95f6b536c6beb4349ff33f8138af1b9**。
- :698 run持续按原分片enqueue，无drain。
- :524 stop Event只心跳，不控制新领取。
- :675 refreshqueue只原票绑定，修改queue失败不是安全排空。
- :555先O_EXCL claim，:556才next++，backlog不证明没领取。
- CLI无停领/导出/退休参数。
不许为“尽快”kill、热更新、删claim、制造挂载断连或修改queue让其报错。

源码包exact：
ROOT/external_ms_unified_clusters80k_20261027_v1/results/a100-extra-03/integration_handoff_20261027T093320Z_7dbb8474/frozen_code.tar
348160bytes，SHA **5eb7839dbbcdd887fd0fd9fa6e8503a1f4d5dc499022019c2e734677cd6c813a**。
成员production/unified_lane_rollover_v1.py、production/unified_access_transport_rollover_v1.py；relaySHA c8d36e48e930b5dbd235bb21541fe05adc4b6307060bf2bb67658f5f7ff35406。

### 01–02消费者已共享，不能再报“缺源码”

ROOT/external_epoch_status/consumer_delivery/consumer_delivery_7ud5sC/consumer_handoff.json
SHA **5b011b3fc64e621b7c7d6f18e9294e713d3938517cdc94eca95ee788c75ace2d**。
node.source_files[].shared_source_snapshot指同目录sources/<sha>.py，5uniquefiles已主逐bytes/SHA核：
consumer e8a9e310253788d778e26451165331bede32f28176f79fedac24817a53dff36d（01）、c1740b0f58fd471b4d3815b91808390c3e2e209abfa584feb1a48f2748742335（02g0）、8f350bb74877cf2ee8f4423c8de06ca5ea992bb906a4ab633c3bbce650b7d783（02g1–3）。
storage be030ea4025aa3443299cdead1de72d4c738d6346ce41a9d35f80325e2160ab1 / 74f0e6682fbe85286f9a5743c924cf489930678c6d09672b30a5ded8edb45439。
它们启动一次读固定三票run1000→rollover5000→clusters40000，跳已消费old后固定2refs（02g0resume1ref），validatecount5000/40000；**没有通用第四票入口，不会自动接收append**。
Worker.run:334 → validate_assignment:165 → segment:309已实际跨原rollover→clusters，模型复用，但不等能动态新增任意epoch。
01/02资源门外部32GiB headroom、RSS<=24GiB、GPU<=8192MiB、UUID lock+experimentlock、8GiBcache，不把主160GiB改成外部门。

01最终报告：ROOT/external_epoch_status/a100-extra-01/epoch_handoff_y6e6dc1g/{summary.json,audit_completion.json}。
主已核audit绑定summary SHA=e956c39ee7fd7fdd37b18360c0a929e5aa74d25cf2a5228405336a3e2b04a5b0。summary本身旧nonblocking backlog71，而**后续audit epochcounts unprocessed0/pending0**，必须分观察时点，不直接把旧summarybacklog否定最终audit。

### 尚未实施的可选reservation转移设想（不能当上线合同）

主只做40次/tmp CPU原03 enqueue函数竞争测试：原canonical claims/split/id.json两个O_EXCL竞争只有一赢家；reservation赢则oldconsumer FileExists跳过且不copy/模型，oldclaim赢则不覆盖。但这**只验证原子基础**，不是全链路授权/验收，生产reservation=0。
- oldconsumer :557–559 existing_claim_not_reassigned跳过；:655 unique_sources_attempted排除此类，commit覆盖索引但不算实际模型attempt。
- receiver仍需**新独立消费者**支持明确reservationtoken/originalrefs/targetepoch、全局once消费标记且crash永久封存不重试。旧01consumer直接O_EXCL原路径会skip，不能直接拿它处理reservation。
- 多集群共同canonicalclaim namespace、reservation完整fsync/readback/partials保持、原runningworker可读FileExists跳过、全局finalsource对账，都尚未验收。
- 真正strict同一时刻2/3比例需要稳定切点，旧worker无drain不能承诺精确快照；不准仅用14:00remaining派manifest。
- 研究代理live-source-transfer-protocol-review反复漏读已多次给的exacttarpath已停止，不等待它继续；主已读原函数和40race，只完成研究基础。

下一agent可据当前共享实际代码设计最小独立协议/CPU进程竞争测试，再明确当地运行方式；若无法证明兼容，诚实保持不可执行，不发假drain入口。

## 10. 为什么01–02更快：已实读双方，不是RGB/BGR

两群同五算法SHA与**当前实际BGR24**，01旧RGBpilot是历史不同epoch，不能解释现在速率。
01/02consumer decoder:232调用core.decode_bgr，process同extract，没有额外RGB反转。
03也是同core；算法没有简化证据。
实际差异：
- 01/02直接共享FS copy_source、1copy线程/2pending；03直接挂载模式同copy_source，04及relay模式video/audio经SSH stdout复制并stat/合同验证，03–06内部也不完全相同。
- **双方都逐散文件+tar发布**，不能说01只发1tar。01/02storage直接FS临时文件/fsync/readback/link/dirsync；03relay每file SSH/python RPC且每次重读ticket+assignment+queue、目标SHA读回。
- 01/02segment:309–332在当前publication之前预取下一批前2来源，可跨batch重叠；03同步publish后refreshqueue，不跨批publication预取。
- 模型都是驻留，不每source重新load。

非配对但同20源19success+1failed1539frames的小profile：
| 样本 | copy逐源合计 | probe合计 | decode_pose合计 | rules合计 | publish |
|01|10.058s|3.905s|33.736s|31.373s|34.548s|
|03|20.081s|4.038s|34.373s|0.841s|187.060s|
|04|294.573s|3.707s|33.756s|0.257s|402.685s|
逐源合计≠并行batch wall，不可直接相加或算稳定倍数。pose接近，copy/publish差异明显。01whole_e2e60334s源码明确为worker从begin累计，不是batch耗时。
用户03–06新20混合batch发布412.0s/99RPC→66.9s/2RPC，约6.2倍**只publication实验，未原pipeline部署**；原旧4source4.28倍实验别混淆。

## 11. 旧RPC与纯CPU预检已验收，但不是排空/执行许可

这是MotionScript发布优化，不是本VLMbatchv4。目录ROOT/external_ms_batchrpc_next_20261027_v3/。
- 52独立CPUtests从实际v3目录fX7DON全pass，原5core不变；包含20success/mixed stdinheader/fulltimeout/fallback。
- oldargv160695bytes实际LinuxE2BIG；修正固定command35209bytes、bounded4byte stdin JSON header，20success94496bytes、mixed64484。
- manifestSHA04e9f7b204c4fe59c7b1eee945b5279d370261ac6aa7f8b7b1e73638cb623151；provenanceSHAF5d4b4cbd7396f0cc13596cf9153a709d854c255ef38c3dc3074ea409ad95a8e（SHA比较请转小写）。
- independent_acceptance_20261027T1203Z.json SHA b981a8594d6976b2e12ae469db8785df4937110ec6ab389683ab1cc182baa9bd。
- 原v2三源码/两冻结JSON被误覆盖的历史事故明确history integrity=false；v3独立snapshot保basenamev2、原bytes，不谎称历史writeonce完好，不覆盖它们。
- candidate只03–06 exact2票/20000，CLI仍unconditional deny、deploy_readyfalse，不适配01三票第四epoch或在线drain。

ROOT/external_ms_epoch_preflight_20261027_v1/：epoch_preflight.py SHA ebe8a7c0222a82af3052912af8dd2da4ce1278dfdcf3e0b4dd2beb106fd71888，tests eebf51896afe21627f1999d6908c64c68066bfb8f0bb5a7b7039f47a46820ac0。
84suite+89独立追加case（178真实函数calls）全pass，no-I/O守卫；独立receipt SHA01951147ea7065b6031e853ed6b71657d1fa2eed35ad2f7be11fb4594fd70555。
纯内存validate_transition_contract不读现场，不认证签名，输出deploy_ready/may_create_claims/actualPIDexit/actualphysicallease/fullassignment/fullclaim-artifacts verified全false。只是独立锚定摘要自洽，不借positive fixture自动造实际票或宣称资源准入。

## 12. 暂停goal/监控状态

用户local-command已明确 Goal paused，原10min巡检cron5e022581已删除；最后CronList显示No scheduled jobs。不要重启旧Stop hook goal，也不要将新会话的任务默认解释回旧MS优先。
旧worker生产没有被停：主MS全completed/drained；外部生产仍自行运行。旧主MS监控xC0HaZ、外增量7DDVvg是已有读者，不是当前VLM生产；暂停cron不等停止生产。当前新后处理monitor用TLo47h，pointer正确。
六份external_live_status/a100-extra-0N.json旧02:28/02:55UTC快照过期，不能长期证明24GPU活体；但后续用户交接报告是新的当地PID/source证据，须按其观察时刻与实际字段判断，不说“完全没新证据”。外部progressmonitor每900s小metadata，无VLM/信号/claims动作；stickygap rate=null，不把累计新增直接加到用户异时epoch表。

## 13. 新agent首轮按此顺序接续

1. 先读本文、notes/34最新、两后处理registry和latest_summary；在可信节点核当前worker PID/start/PPID/argv/UID/exe/cgroup/CVD、rawUUID/GPUapps祖先、parent真实FLOCK+leasewriteFD与heartbeat、latestresult/commit来源SHA。不要硬编码本文旧handle去重复launch。
2. **继续四个正在跑的250票，不热改它们**。批量v4由唯一新writer接手，先修测试/输入合同+mapping并固定完整suite；已有18pass只是CPU基线，尚没有batch GPU实测。
3. 完成3000剩余元数据供料：source_selection冻结，前1000已在四plan分配/运行；在独立恢复目录解决零字节originproof，保旧原件与已验archiveproof，不从头重hash全部/重调模型。选择未分配的新来源用于batch4/8正式GPU试验。
4. 批量部署仍需要资源安全边界：若四serial票都有余量且模型驻留，不能直接暂停或另claim剩余行；先设计已有输出/claim闭合的独立retirement/drain，或用经过明确交还的别任务pair。没有safe接口就不虚构。对额外旧任务回收询问已经提出一次未答，别泛泛反复问。
5. MS重分配保持用户授权，当前转出0。不能直接修改03/01queue或claimnamespace；基于共享实际consumer设计并验证兼容转移方案，发布exactsourcekeys/SHA与接收新epoch，再告知当地agent执行。若排空机制不存在明确告知，不拿CPUpreflight当执行器。
6. 后处理结果始终未人审不可训练。吞吐按可靠source outcomes统计（两请求是一来源），不将batch_chat_invocations、规则/关键点/诊断/后处理stage相加。也不把needsvisual当流程失败；两者分栏。
7. 每个新冻结代码/manifest/proof O_EXCL且不可覆写原证据；生产成功后才回收自己临时目录；不rm其他任务文件/锁，不绕门、不unknownsignals。

## 14. 可以直接输入新agent的接管指令

> 请先完整读取 notes/35_annotation_vlm_batch_rebalance_agent_handoff_20261027.md 与 notes/34 最新追加记录。接管当前已有数据后处理工作，不恢复已暂停旧MS goal/cron。先实际核四个250后处理phase及TLo47h监控，保正在运行的冻结票与所有已消费预算。优先完成真正batch4/8的独立v4代码、CPU完整映射与异常回归，再用未分配新来源在真实安全资源边界验证GPU吞吐和输出，不以少帧/短文本或重领旧来源达到5000/h。并补齐中断的3000元数据供料，保零字节旧文件，复用已验archiveSHA。外部剩余2/3给01–02已授权，但当前转移0且旧03无drain、01无通用第四票入口；设计并验收新排他合同后才能实施，不热改或kill活跃worker。用户希望更多模型，但额外旧任务回收未明确批准，不能抢其锁租约。所有结果未人审不可训练。全部回复中文，只有实质变化/阻塞/真实验收时简报。

## 15. 2026-10-27 17:08Z 新会话首轮实际接管

- 已完整读本文及notes/34最新追加；没有恢复旧goal/cron、没有重复launch原四票、没有signal/清claims/锁/lease/STOP。
- 16:58:11Z可信SSH核四worker与manager的PID/start/PPID/argv/UID/exe/cgroup/CVD、8 rawGPU UUID/apps完整祖先、parent实际kernel FLOCK与lease可写FD；8秒两采lease heartbeat全前进/inode不变。17:04:40Z四票可靠outcomes为N3 193、N2g23 173、N2g45 164、N6g23 162，共692/1000、1384来源请求；470结构候选、221需视觉修订、1空证据、流程失败/skip0。仍没有summary/resources_safe，全部继续保原票，不是完成。
- 当前TLo47h实际PID3243607/start556982510，实际60s只读动态登记，17:04:04Z监控688是先于节点692的快照，非矛盾。没有重启监控。
- 四个抽样result/commit和原输入JSON/proof SHA、logicalNPZ/3locator/5members/storage全通过；原archive五字段stat以同Python原整数精确比较4/4真。审计中曾把纳秒int经JS JSON roundtrip产生低位误差，重读原bytes后撤回假告警，无proof修改或复hash。此项不是全692或最终full replay。
- 主会唯一batchv4 writer补原18项之外27项，共45CPUtests。补严格归档正负例、成功/failed原nativeframes/grammar/来源映射、failed caption normalized视觉链与stop/tokens、source/stage no-retry预算回放、partial错误response候选计数，以及第二次run零模型/不覆sealed产物。保持12帧/200704pixels/16384context/1800tokens与原生160GiB、安全函数和模型recipe不变。
- 最终固定CPU suite neXCHC exit0，45 tests/0errors/0failures/0skip，源码与冻结v3/base前后SHA相同。caller SHA 2f12067fdb11dcbed6061fc363b35acb0d0b73136676ec9d55aedadc5c06aeb5；tests SHA fee3a5b19035e10f39c7917c22e7ef2ed4cbfc96fb8de522875b456aa53cc530。独立O_EXCL证据ROOT/ms_fullframe_vlm_batch_v4_cpu_acceptance_20261027_v1/final_cpu_acceptance.json SHA092c2dcd12a1a39324a63f4e182ac3e0de3ce167ef3d6fa15404aab7741cf8af。尚无GPUbatch实测，独立review继续；不冒CPU通过为5000/h或质量验收。
- 3000供料恢复独立writer实核完整1081、零proof1、缺失1918；独立ms_vlm_archive_inputs_recovery_v2.py SHA88af11490d88bedfda7ec87682cc23b671efc3217c4b17ea04f286c5b3be7946，10CPU tests通过。managed nFTj1r在ROOT/ms_fullframe_vlm_archive_inputs_recovery_20261028_v2实际物化，复用原55归档proof而非复hash，缺proof归档各自首次边界校验。恢复者已回报前26batch共520来源strict通过、复hash0；整3000仍以最终回执为准，先交未分配新4/8候选供GPU票。
- 原四pair都活占卡，无已验证在线drain，不暂停抢剩余plan。安全接续仅准备独立新wrapper等旧票自然完成+完整producer summaries与worker_exit.summaries全值一致+实际worker退出+当前safe/restored/fresh160GiB，再新source独立batch4。旧manager在resources_safe后h.wait()仍Running正常，不要求旧managed task完成，不signal旧manager。新wrapper实现中，未部署。
- 外部2/3授权不变；没有转移新来源、修改queue/namespace或伪造drain。所有后处理仍未人审不可训练。

### 2026-10-27 18:36Z 原监控退出告警补记

- 延迟通知已实际读取：外部增量监控7DDVvg exit1，写latest_summary.json.tmp失败ENOENT，异常处理再写heartbeat.json.tmp也ENOENT；旧takeover监控z15Rgr、v4XBvN分别写各自.summary_latest.json.tmp失败ENOENT后exit1。三者都是监控发布路径错误，不能据此判生产worker退出或重做标注。
- 18:36:32Z本控制容器精确stat显示ROOT、runtime、三个监控输出目录、当前后处理监控目录及本接管文档均可访问。报错时路径不可达原因尚未确定，可能共享视图暂态等；没有证据确认目录被删除、存储损坏或他人改动。未mkdir重建目录、未清锁、未重启任何旧监控/生产。
- 当前后处理监控小摘要现场读取显示四个原250票processed各250，共1000终态，worker_alive=false、completed/native_safe=true、failure/alarms空。这是当前监控快照，不替代独立完整重放与现场资源终验，也不表示新会话后继批处理已完成或不存在。
- 此节只补告警；新会话已经追加第15节并修改批量代码/恢复入口，旧会话不覆盖其代码、注册表或正在接续任务。接管者不要再把7DDVvg或两个旧takeover句柄当运行中读者；保留其日志，必要恢复由当前负责会话根据现fresh状态决定，不重复启动。


### 外部用户补报（观察时间2026-10-28 01:02 CST，即10-27 17:02Z；非原子）

- 01当前rollover+clusters结果44975，02结果43372，另初始1000两节点结果1999；共90346个partial/needs_regeneration结果，41份共享失败回执另列。已给容量摘要ROOT/external_epoch_status/shared_publication_capacity/capacity_fast_1793120557873/summary.json，主只读2645bytes并核SHA473f60bedc9336dcb98d3415c4d30514f72aad34e8bdb0f8eba3789eb140d3b7吻合；unique_committed_sources=90387含失败41，不冒作90387成功结果。4539个实际canonical batch.tar总16.10GiB，加commit16.39GiB，不含散文件副本/视频/权重/pilot/验收包，也不等目录总占用，未重读包全量SHA。
- 90000来源两轮：已发布88347、未领1538、在途55、失败56、unknown4，合90000；01已终态、02原4worker继续。此活体/15min新增320来自用户当地报告，不冒中央当前GPU独立验收。
- 03–06本期80000旧提交34000、深验33900仍为历史快照，最新全量对账待交；旧4000另计。04–06原12卡继续且新提交，03原4卡空闲但未正常封口，80条已生成未提交产物仅核现成残留，不重推理。独立恢复入口仅未claim原分配，尚不能代替安全转移合同。
- 已建议当地交每类exact source-key/SHA、03旧进程禁止再领取与资源封口证据、新恢复票与未来转入01–02集合排他；不热改04–06、不清claims、不把空GPU当safe。01–02新消费者需独立epoch入口，不直接append第四票。此为协调要求，实际转移仍0。

## 16. 2026-10-27 18:55Z 真实batch GPU与长票扩容接续

- 18:34:58Z可信SSH三节点核原四archive250均250/250，1000可靠终态、2000来源请求；四worker均/proc不存在、当前safe5flag全true、无failure。676结构候选，324其他视觉修订/空证据等终态，不把summary.skippedfailed混当流程失败。原四票没有重启。
- 新batch4 N3g01独立phase=ms_fullframe_vlm_postprocess_batch_v4_trial4_n3g01_20261028，受管dJSSGR/manager3651557、worker3653148，config SHAd94e0d845c94322f1aa4061e9ad63f9542a351608e533615805cc1b4de712ded。新4来源排他，不是重领旧250剩余行；actual160GiB准入。已4/4、8来源请求、2实际chat，visual19.9576s/fusion16.3002s，无调用异常；3结构候选+1需视觉修订。完整v4.summaries==worker_exit.summaries、exit0/sourceunchanged/safe5flag独立通过。peak每卡73471MiB，load.receipt→summary全流程91.3146s，有限观测157.7源/h/副本，不称纯推理或稳定5000/h。独立GPU验收ROOT/ms_vlm_batch_safe_dispatch_acceptance_20261028_v1/batch4_gpu_full_replay_acceptance.json SHA4be177c163eceb4faeaffecc57c5d09983b1ba69456f449e5463d0469febe138。
- 新batch8 N2g23独立phase=ms_fullframe_vlm_postprocess_batch_v4_trial8_n2g23_20261028，managed5oHdpN，worker1487247；按候选8全新来源登记。18:54Z实采已8/8、16来源请求、2chat，visual22.9010s/fusion17.7594s，batchresponse8→8无异常，worker_exit0/safe真；独立完整v4回放仍待记录。没有重启任何旧phase，未知TaskOutput不冒作worker停机。
- v1安全dispatcher当前SHA ea563e13947f0706926b28901568acb6751689db4734e826767631ee3f8fc0a8，已补自身ref、合法futurepending无config、前驱exit/summary/safe closure首尾冻结及物理节点前置。64wrapper＋45caller联合109CPU全pass、pre/post SHA一致。proof ROOT/ms_vlm_batch_safe_dispatch_acceptance_20261028_v1/final_joint_cpu_acceptance.json SHAa240ab8cad11ad7303557db5e325560fc42e9f08b46b21fb0972e9a5e466e601。旧72b/38或45只历史基线，不冒最终验收。
- 用户新增授权扩大批量数量、充分利用显存、多启实例。独立新ms_vlm_batch_safe_dispatch_v2.py仅解除trial source_count=batch_sources，支持1..10000 finite sources、实际batch4/8，保v1在跑原件/原160GiB/锁lease/once/closure；67CPU allpass incl128来源batch8与shorttail，主未改v4模型recipe。v2 SHA f5962d02fbcb0d0d7d4b111ffa9bc1fa6fbf44a49269ac8a1178559df10b7225，tests SHA046bfa29de815c2b50534572c1438e782702a9c376113a7f047fa7e07ed36407；proof ROOT/ms_vlm_batch_safe_dispatch_acceptance_20261028_v2/production_cpu_acceptance.json SHA24a48c112842ceeb8ebf57e82957f6e0745d1c96efe457e71523f825698ad997。
- 新512来源四份128/batch8供料准备唯一managed2T6MYF，来自recovery_v2已有strict batchreceipt，排除全部已登记来源，逐source v4.input_pair边界验收；不再whole tar SHA、未sourceclaim。out ROOT/ms_vlm_batch_production_inputs_20261028_v1，各replica0..3_batch8_plan128.jsonl，finalproof尚未读取不冒512已完成。将用于原四pair安全长票多副本，不能预先抢其它卡。
- BPtRAW实际source键去重（在新4/8前）：旧本轮1520预算终态=1518完成后处理+2失败封存，完整status1019结构候选/497需修订/2空证据。新4/8额外。与主62903、外部新clusters78373、03–06旧34000非原子报告估算已有175276草稿，本轮未处理约17.4万；非全库最新exact census，不含从未出草稿或旧caption/VLM其他协议，不重加原rollover/initial来源。
- 3000恢复：nFTj1r已失败，129完整batch2580来源可靠，末21batch420未闭合；132archiveproof已生成=55原+77v2，只有18无proof需首次SHA。原failedtask不重启。独立v3 f3Zkzp strict5stat首批失败：当前controller共享挂载dev0:300→0:159，inode/size/mtime_ns/ctime_ns原整数exact相等，mountinfo实证旧device不在当前namespace。这不是JS精度误报，不改旧proof/不默默降stat门、不重复tarSHA。允许独立迁移attestation仅metadata continuity（同Ceph后端需真实证明），不能冒当前payloadSHA重新验证。恢复下一独立边界尚未上线，完整3000manifest仍不存在。
- 全过程不恢复旧goal/cron、不清旧claims/STOP/锁租约、无外部source转移。所有结果仍未人审不可训练。

## 17. 19:32Z 新长票发布与可访问性边界

- batch8独立完整v4.summaries回放==worker_exit.summaries，8/8、16requests/2chat，6结构候选+2需修订，exit0/sourceunchanged/safe5flags真。peak74045MiB/卡；load.receipt→summary156.3883s，有限观测184.16源/h/副本。两次API共40.6604s，其余115.7s含供料/资源gate/落盘，不能全称纯推理或把全部余时归因单一项。proof ROOT/ms_vlm_batch_safe_dispatch_acceptance_20261028_v1/batch8_gpu_full_replay_acceptance.json SHA1f22adc26e5727bc6fb3a10916240c495a8aed3a38e8b00b6488b722bf101a88。
- 主另写独立ms_fullframe_vlm_postprocess_batch_v5.py候选，不改已冻结/在跑v4：不可变JSON/proof经首次SHA＋完整5stat签名核验后缓存raw/parsed、后续签名变化即拒；返回deepcopy防共享对象变异；媒体供料仅4个CPU线程，source预算仍主线程O_EXCL，模型chat仍主线程；支持4/8/16/32批conversation，GPU调度max_num_seqs仍8/原BF16-TP2-16384/12帧/1800token/.88与所有live资源门不变。51CPU全pass（缓存payload只读一次/变更拒/alias隔离/4owned prep线程/16/32＋原归档和异常回放），v5 SHA daa07690887802743b4aafba5e566877b01a72d46f8037f5bea5d4a6d50aa42d；tests fed3388245e61677a1aeeeb477eec362452057dcab66cb3afd1bb3faa98730af。CPUproof ROOT/ms_vlm_batch_v5_cpu_acceptance_20261028_v1/cpu_acceptance.json SHA8cb60a5968d7912cafdc701d40bdbae09970fc5adaacf01c4cd30f15e6c754f0。**v5未GPU部署，不称16/32并发GPU验收或速度提升已证实；原v4票不热改。**
- 唯一512source输入2T6MYF已经exit0，四独立128/batch8计划完整v4 input_pair全部验收，复用原已成功strictreceipt且无oldtarSHA/no claim。有限发布器ySsXKj也exit0，四票ready已按同registry锁排他冻结。新v4-prod128：N3g01 UZG7z9，N2g23 Qifk7j，N2g45 0jvJBy，N6g23 ntUviW；四bootstrap已受管，只经native前驱fullreplay/currentholder/锁lease/160GiB后exec。N2g45 bootstrap额外等N2g23实际load.receipt真后才重新fresh准入，不重复节点预算。不能把ready存在当已提交512或四模型已产出；需实际live/progress核。
- 19:29Z八节点新实采：N1八卡GPUapps空且memory.used=0，headroom385.95GiB；这与旧全occupied历史不同，但还未核其全部kernel lock/lease/正在加载CPU任务及资源交还，**不宣称八卡可直接抢占，不新建holder绕旧锁**。其它节点除了本任务恢复holder外仍有原VLLM/producer；本任务四pair已恢复holder，N3 headroom193.97GiB、N2 479.88、N6 311.37，均为各node一次快照。没有暂停其它任务。
- 当前本轮完成后处理1530=旧1518+新4+新8，另2原失败封存不重试。主共享草稿62903＋01–02去rollover后新增78373=141276，减1530得139746待本轮VLM；这是已核共享发布账的非原子范围，不冒当前逐条可读性全量扫描。原全集222939中未获本轮完整结果221409，含尚未产草稿/封存/unknown，不是全部立即可执行caption。
- 03–06每节点各一个已commit batch.tar与包内draft.json实际可读，四caption文本长度259/243/197/228，四原video_path均is_file=true。只读取tar header和draft member，无全archiveSHA、无视频/NPZ payload；证明共享访问路径可用，不证明旧34000每条已独立read或最新总量。样本任务wxWE4L exit0（冷metadata壁时约6m53s），无需再向外部回传这些已共享包。按旧34000异时报告合并估算待本轮约173746（扣2 sealed可执行预算约173744），仍非全库exact fresh census。
- 主要当前慢点有真实CPU证据：512输入准备Python PID3791988在FUSE request_wait_answer/newfstatat，非GPU模型计算；没有重启该任务或把无stdout当停机。阶段内不重扫全部JSON/归档；v5候选针对这项缓存与并行供料，但未替代真实速度验收。

## 18. 2026-10-27 21:44Z 快盘流水线真实验收与持续优化

- 用户21:19前已暂停新goal，之后明确“请持续推进和优化／请持续优化”；只按当前请求继续，不恢复paused goal/cron，不重启任何旧running或failedtask。中断后先核真实/proc和队列。
- v6模型实际max_num_seqs=16、batch_sources16，BF16/TP2/12帧/200704pixels/16384context/1800token/.88及原80+80GiB门不变；元数据先原字节复制/root，媒体只拷贝一次/root/ms_vlm_fastcache，4线程current/next批预取，可靠commit后仅清自身媒体临时目录，持久inputrefs不提前删。54CPU+69safeCPU证明已固定。
- 四v6票已真实完成448/448、896来源请求，291结构候选+157需视觉修订、流程失败0；四worker退出、exit0/sourceunchanged/safe5flags。21:42–21:43在执行节点独立完整v6.summaries==worker_exit.summaries回放成功。有限post-first-batch来源速度：N3g01 419.13/h，N2g23 389.01/h，N2g45 411.62/h，N6g23 394.12/h。各观测窗口不同，不冒共同长稳态；每副本比旧batch8约181–190/h有约2倍真实改善。预取等待首批27–38s，后续约3e-5s，I/O供料已重叠；仍有较多共享结果发布往返。
- 独立回放凭据ROOT/ms_vlm_v6_full_replay_acceptance_20261028_v1/：N3文件SHA d11b9958e2050c1a20ae2a8134c96cc15f824ae96f012bfa6e013c949d1516c4；N2g23 323305e1e07e3d9c67ab77787977b3adc6195362e01bbc25ce5ba0d41947d4f3；N2g45 39551a6f5584fccff1674c047cd254988fc68f5406f62ab5c9d1db16efcbd592；N6 d00196f7348d5227e3639a83472ab01cea5ad22bb67a913709c3544b4ac9d17d。都是<phase>_acceptance.json。
- v7为独立可靠发布优化，不hotpatch旧v6：缓存128个publication dirFD、O_EXCL/NOFOLLOW，仍filefsync+directoryfsync，使用同fileFD readback后验path/stat，自己的新JSON refs绑定rawSHA不重复hash，ownrawcache仅256条。58CPU回归保持source预算、请求/raw/normalized/commit独立回放；SHA58b384fb208790b9638ff7e2f4b833ab9c5755f68110c12b8a1d569ad5d3764a；tests0809c4d09b871e5d3b933e216456db44e0eddb2b557bde99ef208b18b2bcd022；CPUproof ROOT/ms_vlm_batch_v7_cpu_acceptance_20261028_v1/cpu_acceptance.json SHA01c1254915d26f52db5c5b8515b89458767aa0bf5aca3dc9129d1b066ea25425。
- 新v5安全dispatcher指v7caller，兼容保留v4–v8合法futurepending，普通missingcfg不放行；70CPU通过，源码a90893bafce19622ae348941d36690b04547cc5d26734d331559fa6efc2aeaa7，proof同v7CPU目录/safe_dispatch_cpu_acceptance.json SHA403a1a0f9f16526d81e71779d1e1526272d1c16b952c7284cf6437eb5c8e8bfb。旧wrappers不改。
- 原2580可靠metadata中的候选512补料Ae2nWr因实际不足512在任何新write/claim前assert拒绝，未重启。独立按实际可用量选择448，ebm1Ya exit0，ROOT/ms_vlm_remaining_eligible_inputs_20261028_v1/manifest.jsonl SHAa85d99ca80a4d1311c900a8b2c5941f82df61371525dee3e848fdf27a55c0f2c，selection_proof SHA856a170a8d957837e28f6ae84b907792d060db6fbc7e5bbab0f00b0dca491c71；两半224互斥，已复制到N2/N6/root并独立v7输入验收。
- 两个新v7-prod224实际部署：N2g23 phase=ms_fullframe_vlm_postprocess_batch_v7_pubfast224_n2g23_20261028，handleKkkla9/worker1540945；N6g23对应后缀n6g23，handlekfcSJL/worker2705136。DJQrHs排他登记和ready发布exit0，21:42现场两新worker真实存在、queue/config存在、无failure、模型仍加载，尚未读取可靠progress，不报v7速度收益或完成448。原v6来源不重领。
- N1新扩容曾准备独立fresh-empty-pair准入+caller(v8只扩N1范围、函数与v6AST全相同)，73CPU与工厂11CPU通过；N1/root128新输入已有效，但N1实际后来出现8个非本会话旧conda hold_gpu进程（PPID1），不是本会创建的fresh holder。21:33 IxgX6E真实因fresh资源初始化失败退出，未起VLM/未sourceclaim，不重启该失败任务/不清pending登记。询问允许释放这些占卡进程一次；当前未获答复、不signal。不要再拿20:09 empty快照冒当前空闲。
- 为避免有限票反复断料，独立新ms_vlm_new_native_local_inputs_v1.py从另外五条原MS完成stream（n2g5/n3g0/n3g1/n6g2/n6g3）精确cfg/ticket/plan/summary/commit定位新train，排除现全部VLM登记和旧3000pool，terminal/failed/split强核；13CPU＋执行节点pre/postSHA固定通过。唯一managed739zWy在N2/root/ms_vlm_fastcache/native_new4000_inputs_20261028_v1物化，共享selection/manifests在ROOT/ms_vlm_native_new4000_inputs_20261028_v1；原bridge源码SHA保持21e908…，仅新归档首次入库SHA，不重复旧3000归档、不读视频/不解NPZ/不调模型。当前只看到CPUproof与进程running，尚未全4000产物，不冒完成。
- 所有结果仍未人审不可训练；新清洗继续，原外部转移仍0，paused goal/旧cron均未恢复。

## 19. 2026-10-27 22:04Z 发布优化持续产出与新瓶颈

- v7两224票真正增产：21:58实采N2g23 112、N6g23 96；22:05实采160/144，共304/448可靠终态、608来源请求，无failure。当前窗口单pair约446–480源/h，仍为不同来源/短窗口，不冒全群长期速率或全局质量验收。所有源保持12帧、32B BF16 TP2、16active seq、原长caption/16384context/1800tokens/.88；旧v6不hotpatch。
- 新4000原MS供料739zWy真实开始物化，N2/root83个完整batchreceipt共1660来源已见，完整4000manifest/proof尚未出；这1660是输入准备、不是VLM新完成条数。原5条stream cfg/ticket/summary/commit强核、全部旧3000pool和现VLM登记排除。CPUproof ROOT/ms_vlm_native_new4000_inputs_20261028_v1_cpu_acceptance.json SHA79ce40817be91bf8ce2705b729a639d4b36f281af1f1b251b02a360e1ea683ed。未重启任何旧失败恢复任务或反复hash旧已验归档。
- 已从其完整回执提前接首500新源至N2g45，输入HxG7kZ/排他发布iVvwpg均exit0，newphase ms_fullframe_vlm_postprocess_batch_v7_native500_n2g45_20261028，managedzBtjZM、actualworker1544647；加载真但暂未batch_call。21:59及22:06/proc同一source.claim.json fsync(74)等FUSE，FD72具体ROOT/<phase>/budgets/train/AcJnbzBNRihtAIYi/source.claim.json，claimeddirs19、call0、GPU0%。不是模型报错，预算保留，不kill/retry/热改。当前下一批prefetch source预算同步创建会阻塞当前已准备批的GPU调用，这是新确认的串行空窗。
- 独立v10候选解决此项：source预算仍O_EXCL+filefsync+dirfsync，但由4个供料线程先可靠领取再copy/frames；main GPU线程不等下一批预算。publication dirFD在短锁中dup，IO/fsync不持全局锁，FD/cache有界且原目录inode/owner/非symlink检查；API/风险策略/原资源门不变，仍16active seq以隔离收益。60CPU全部pass，额外真实thread-barrier证明4source fsync能重叠、媒体仅在源预算完成后开始。caller bfb48972582e3151f4c1e1d5eb0b50b8b9e458b46ef51e893f53fd8349a8a8b7，tests523b25d4b3ee24ec38d2b1ab1bffec93c6af9063d1437caedc10d51167689ab8，CPUproof ROOT/ms_vlm_batch_v10_cpu_acceptance_20261028_v1/cpu_acceptance.json SHA654e62fe57cce662ed313f57cda8c23e722548124fc86c7368007478f4b9f5d4。**v10尚无GPU实测/部署，不冒已解决这次特定Ceph卡住。**
- N3另一500新源从new4000 selection[500:1000]制metadata-only bundle（包含draft/manifest/origin/verification，不含视频/NPZ），evhh1m已exit0共享发布；4Jo1Hl在N3一次读bundle、校完整SHA和每member长度/SHA/regular/nonsparse/nonduplicate，再/root原bytes重定位物理refs，原logicalNPZ/binding不变。完整N3-inputproof待取，managedp2RSW1和publisherdCSdtN已准备，必须真实proof/原v6 fullreplay/holder160GiB后才新GPU启动，不因bundle存在冒500清洗已完成。
- 下一张32active-seq新票也独立准备：v9仅将v7 max_num_seqs16→32，原12帧/风险/caption/资源安全不变，58CPU固定pass；caller407c6350650092728db0e87e88e0d071df4e777abbc407891f256a6c964a9685，CPUproof ROOT/ms_vlm_batch_v9_cpu_acceptance_20261028_v1/cpu_acceptance.json SHA47968175d4571c329333367356e1100b65e9267a9702ca551ef201eb3ab1639a。新v6dispatcher支持已冻结版本并保original安全，70CPUpass，源码d6774fd7bc60f59e0361f7289f39013b2235abaf6f35ec42e7d118ac89ad6754，proof同v9CPU目录/safe_dispatch_cpu_acceptance.json SHA90c85eb7950536ce5f552625f3b7ffc14b34cd46ad4f549e58808bdd09498701。
- source selection新256取new4000[1000:1256]，唯一6Q1zlR输入准备，F2hWjp受管bootstrap和uhxT9B排他发布器待输入proof；phase ms_fullframe_vlm_postprocess_batch_v9_concurrent32_256_n2g23_20261028，仅在现N2g23 v7-224自然exit0/fullreplay/safe+fresh预算后接续。不能当32seq显存/速度已通过，不重领现224来源或新500范围。
- N1八个旧conda standaloneholder属于非本会话，申请释放授权一次尚未答，保持其进程/锁/lease不动。自己的IxgX6E因fresh资源初始化失败终止，不恢复/重启；新128来源仍pending保留、模型实际0call，不覆盖或重派到别phase。
- 继续推进当前真实清洗与必要独立优化；不恢复任何paused goal/cron，不用manager running或CPUtests替代生产完成。所有结果仍未人审不可训练。

### 22:23Z 本轮完成边界与下一票

- 两v7-224票已448/448、896来源请求，295结构候选+152需修订+1空证据、流程失败0；现场读summary/worker_exit确认exit0/sourceunchanged/native summaries存在、5safe flags全true。完整post-firstcommit窗口单pairN2 479.93源/h、N6 460.09源/h；不同来源窗口，不冒共同长稳态或人评通过。独立新v7全源回放本轮未重复，原生退出内已有全summaries。
- 新4000本地输入在N2已156个完整receipt/3120条，尚未full4000 proof，不加到VLM完成数。前500N2g45长票已真加载但因上文source.claim fsync未完成首批；第二500 N3单metadata bundle已发布，4Jo1Hl本地验收/物化在途，dCSdtN/p2RSW1保持唯一安全接续；不把等待当完成或重复启动。
- 6Q1zlR新256输入准备已exit0，32active新票F2hWjp/uhxT9B按旧N2g23 v7-224自然完成fullreplay/native-safe后续接。当前尚无32seq GPU吞吐/显存结果，CPU58+70只是冻结准入基线。
- 独立v11进一步复用原归档桥接同样的父目录路径策略：首次canonical parent resolve后保dev/inode连续，每leaf检查id/stem/非symlink，媒体copy仍核完整原videoSHA；避免全票重复递归视频父路径查找。叠加v10异步source-budget/fsync，资源函数/16active模型recipe/原风险/12帧/longcaption/原160GiB不变。64CPUtests pass inclparent-inode替换/新symlink-parent/leaf-symlink拒、首次边界后不重resolve；源码0de75099dd39fd33f93ad35e8cc3580b37b3398d29261011357480c532fb639a，tests d2fe0b8f8e95a5f4c20926fd583d328e890d87aacfbe58374bd49ff946dc8427；CPUproof ROOT/ms_vlm_batch_v11_cpu_acceptance_20261028_v1/cpu_acceptance.json SHA822d03a449e78a2eea18ac3794bd5eeb1d9653e86e7be9387c7d4a134b131b1a。v11/v10均未GPU部署，不能宣称已修复当前特定挂起的fsync。
- 当前首要阻塞仍为原生共享FS个别文件fsync/metadata等待，所有在途预算、旧原件保留，无kill/restart/clear-claim/放宽资源门。N1释放占卡授权尚未答，不重复泛问、其它节点任务继续。





### 旧监控 TLo47h 节点白名单退出（延迟通知）

- 已TaskOutput实际核验：TLo47h退出码1，错误为ValueError: unapproved node，不是共享盘ENOENT。旧inline监控仅允许N2/N3/N6；新会话继续扩部署后，若动态登记出现其他节点，旧监控会主动拒绝。当前节点范围以动态登记现场为准。
- 最后有效摘要包含新会话batch v9/v11等phase，不能将此旧监控结束解读为模型worker结束。监控失败本身不证明生产失败；本会话未重新上线旧监控、修改白名单或触碰新会话源码/任务。
- 后继监控需由当前负责会话依据已批准实际节点与可信路由配置，不通过放宽所有节点绕过检查。保留原日志与既有生产，停止使用TLo47h摘要作为当前活体证明。

## 20. 2026-10-28 02:22Z 实测版本验收与N1八卡授权扩容

- 最新真实全回放凭据ROOT/ms_vlm_recent_full_replay_acceptance_20261028_v1/<phase>_acceptance.json已生成：N2v9-concurrent32-256完整256/256、512requests、exit0/safe/sourceunchanged/replay一致，post-firstbatch578.30源/h、load-to-complete550.00源/h（约1675.6s），每卡peak75749MiB。N3v7-native500完整500/500，约480.70源/h、load-to-complete475.10；N2g45v7-native500也完整500/500，post-firstbatch500.19源/h但**包含首批前Ceph长fsync阻塞的全流程只有164.02源/h**，约10974.5s，不只报好窗口。N6v11-async128完整128/128，post-firstbatch501.64、load-to-complete479.87，peak74735/74733MiB。所有结果未人审不可训练。
- 原4000供料739zWy已完整产出：ROOT/ms_vlm_native_new4000_inputs_20261028_v1/input_proof.json SHA12cb7443f612a92e7717ae33e45a5a734085999e058c2f54995d3ad5cdc52580；manifest SHA3af77908d2da5fdbf9878a1776cc3253f91de6a005b1bfb492cba4aa2cd93741。来源JSON在N2/root快盘，首次入库/原字节绑定，输入数量不算VLM完成数。
- 中断延迟返回期间另一现有接续已登记N2 v9-refill1000_n2g23（handleuKD0bQ、worker1625750）取4000pool[1384:2384]，N2 v11-refill1000_n2g45取[2384:3384]，N3/N6 remaining308分[3384:3692]/[3692:4000]。本会先行654方案Sh9mVX在新票写入/源预算前CAS/排重拒绝，未覆盖这些已有登记；已取消自己的无ready/无GPU五个metadata等待器apBMqn/77h2Ty/bZc0XH/NLytq9/eQvOWS，只保独立输入原件，不重派或抢当前来源。
- 当前4个已有副本已在真实生产：N2refill1000两worker1625750/1628499，N3remaining308 worker3848942，N6remaining308 worker2784779。02:12Z实采合192可靠新source、0failure。此数字不同于原阶段完成数，不跨stage/异时重复相加。
- 为新N1扩大来源，独立new_native_local_inputs_v2只新增排除**完整先前4000 selection**（包括未登记来源），不再从这些5stream重选既有预算；15CPU全pass。唯一hASfpI完成下一900输入：ROOT/ms_vlm_native_next900_inputs_20261028_v1/input_proof.json SHA6b288eb2ba4204463c74fe98121e8877a88eebe7825e329d3b515e910cfcac88，manifest SHA3c9a9ed09e5ddda59f8671c085259a2c5e734af88d74d3f617c763bafd0b642c。它们为全新来源，旧N1失败128仍pending保留，不重试。
- 用户在本次AskUserQuestion明确回复“所有卡都可以进行释放，都可以用于启动进程来打标”，本次执行范围是所问**N1八张独立占卡holder**；不是授权signal未知其他节点生产。N1先核8原PID/start/argv/CVD/UUID唯一GPU应用、kernel锁无人持有、leases stale、共享预算>=160GiB；一次取得8kernel锁和lease，PIDFD再次核identity后释放精确holder，等PID退出和NVML apps空，恢复原leases并释放锁。TOGsjK exit0，release_receipt ROOT/ms_vlm_user_authorized_n1_release8_20261028_v1/release_receipt.json SHAa747544b8ee1d5aa0b2f9ffb6f57cf9b5b0a0598a21d55fd995576879f976356。无其他未知进程信号、无sourceclaim。
- N1下一900新来源原字节metadata打包gCV0Rq、N1一次bundleSHA+逐memberSHA/大小/regular/no-duplicate核验并/root四线程落盘16TstG都exit0，分为4份225，ROOT/ms_vlm_n1_new900_verified_inputs_20261028_v1/n1g<01|23|45|67>_manifest.jsonl和_input_proof.json。v8 caller原54CPU/73freshCPU/11factoryCPU不变，不伪造历史restoredholder：fresh-empty核锁lease后创建自己holder，原生main释放自己授权holder起模型。
- N1四新phase=ms_fullframe_vlm_postprocess_batch_v8_new225_n1g<01|23|45|67>_20261028；handles依次6knhF5/aZHlXK/dp41hH/C9o2UE，XqRvKS逐项锁内排他登记/readyexit0。每后pairbootstrap等前一pair实际model_load_completed再重新fresh160GiB核budget，未复用旧headroom预算。02:20Z四个模型都真loaded/live，前三pair可靠提交48/32/16、第四刚load尚0，合96新来源；无failure。旧IxgX6E/128未重启。N1实际worker316139/324738/331213/335787，model32B BF16TP2、16active、12帧/16384context/1800token/.88。尚无N1长期吞吐或整900完成验收。
- N1正常续票的新独立ms_vlm_batch_safe_dispatch_v8.py仅指向已验N1callerv8，70CPU核全部原生安全链；SHA b3953d010521898d5e91ea212be94a346c8e93e9afc8c5751d45e669800229f1，proof ROOT/ms_vlm_n1_continuation_cpu_acceptance_20261028_v1/cpu_acceptance.json SHA6c54a94769671ccf27ef860790309e9d7c2d83d18335628f5e4ac8674146d35a。后续在原225自然闭合后可新排他长票接续，不改在跑票。
- 针对17.4万剩余供料，实读02共享commit v1发现**已可靠发布散文件就在原draft.source逻辑manifest路径**，可直接用原物理ref/逐fileSHA，不需把v1冒充native-v3 archive或补假locator。K1OreM实际20条v11.input_pair全pass（0tar/video/NPZpayload/0model），同时原assignment真实行/index/SHA核验。
- 新external_direct_inputs_v1只按明确01–02 clusters40000原票据，首次读完整assignment40000绑定SHA，逐commit核v1schema/package/memberfiles/receipt success/原source indexrow，排除现所有VLM来源；原JSON不重编码、不改逻辑路径，failed/unknown预算不重试。18CPU拒绝测试全pass、执行节点pre/postSHA一致，CPUproof ROOT/ms_vlm_external_direct_inputs_20261028_v1_cpu_acceptance.json SHA5a8484adaaf73ed2b4a46692791a41cb64725eb10ced4c2f9eaaca0e7cb08a2f。唯一CfFcUE正在选4000现成caption input，仍未读取finalproof，不冒外部4000清洗完成。它只读standaloneJSON，不重新tarSHA/NPZ/video，不调模型。
- 旧TLo47h因N1白名单退出不重启。新独立6小时只读monitor cF7MQi，RT/ms_vlm_live_cleaning_patrol_20261028_v3.py、OUT ROOT/ms_vlm_live_cleaning_patrol_20261028_v3，白名单仅可信N1/N2/N3/N6，每60s远端实际PID/start、summary/progress、latestcommitSHA。首snapshot真实JSON可读、errors空，当前8active阶段合704去重outcomes/1408source requests（不是全库done）。不创建claims、不signal/模型、不改生产registry。
- 本次按当前用户“持续优化”和新明确N1释放授权推进；不恢复旧pausedgoal/cron、不覆盖任何分配/失败原件。持续复用已实测v9/v11/v8而不以新版本数量代替产出。


## 21. 2026-10-28 03:37Z 用户确认保留当前原版模型与清洗进度

- 用户原提“不考虑BF16，使用正常模型”，经本轮AskUserQuestion澄清后明确回复：**当前运行的是什么就保留什么**。实际四类load.receipt全部model_root=/opt/models/Qwen3-VL-32B-Instruct、dtype=bfloat16、quantization=None、TP2；因此保留未量化原版当前精度，不擅自切FP16/FP32、不上W8A16、不重启现有票。
- 03:36:50Z执行节点按全登记summary/最新progress来源键独立去重：**7766个可靠完成VLM+融合重写来源**，其中6838来自完整summary，另928来自N2g45当前1000票可靠commit；2失败封存另列。不是全17w质量批准，所有approval/training仍false。
- N1四225票已900/900，四worker退出/native safe真，原版未量化模型，post-firstcommit各约479–484源/h（窗口约26min）；N2g23 32路1000已1000/1000，post-firstcommit738.64源/h（约78.6min），worker退场但采样时resources_safe未出，不可复用pair；N2g45当前16路1000已928/1000，worker在产，近期626.76源/h（约79.6min）；N3/N6各308已完整，有限速率645.73/621.84源/h，safe真。当前不是八个全活副本，不把有限票完成后驻留/manager wait当全群持续吞吐。
- 节点持久tmux服务上的30分钟全局墙钟baseline：N2/root/stage2_measurement_service_20261028_v1/window_30min，03:30观察18个实际snapshot、report未出，故尚不汇报>=5000/h或30min全局合格。只read统计不影响生产。
- 03–06新50000第二阶段授权已记录ROOT/external_stage2_vlm_50000_20261028_v1/request.json，当前allocated0/claim0、pending exact unusedmanifest和可部署本地dynamic入口；不伪称五万已转移，不同于01–02剩4w+MS。
- 本轮继续实现新架构，未hotpatch旧产线：stage2_model_codec_v1抽取原v12提示/schema/runtime AST原样，保持BF16原版未量化；stage2_local_model_worker_v1只读/rootprepared packets，每微claim1..8（尾批允许），最多4微claim合32会话，两stage模型预算本地O_EXCL，各source保original claimref/batch，错误/截断封存不重试。结果、request/raw/normalized/batchcall/commit都只本地写，再由每node单uploader封50条/5min tar可靠Ceph发布，ACK前cache保留。
- packet→codec→mock模型→本地预算→resultcommit→seal→upload已实际13集成CPUtests通过，另packet16、codec3、publication19、dynamic pool18、uploader5，联合**74CPUtests全pass**。修正caption文本ms_text边界漏检后重跑成功。没有在GPU上启动这些新components，尚缺真实native manager本地准入/gate、feeder+centralpool服务联通与30minGPU整体验收，不能把74CPU当新动态架构部署完毕。
- 用户新停靠约束继续遵守：未向任何B/C原生产线signal、未删锁/lease/旧sourcebudgets；停止台账notes/36已21原线只读登记但计数/剩余exact集合未闭合，故不能据此停靠。N2GPU1restore_blocked和N1git_D等仍为扩容限制，不忽略原160GiB/tmp8GiB門。
