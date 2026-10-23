# 当前全部实验：有限完成标准、最新证据与代码归档

日期：2026-10-24；基础快照01:05 CST，六视频完成更新至01:49 CST（其它在训数字保留各自观察时点）。这是一份**运行中的有限验收清单**，不是“全部实验完成”声明。后续必须以新 completed marker / audit / evaluation JSON 更新本页，不用任务启动或旧代理报告冒充成功。

## 1. 目标与结论边界

- 主线：音乐 → 官方 Omni30B 的动作 caption → H3 原生图文条件 → 真纯视频。另保留已授权的音乐 hidden 映射对照，以及 H3 原 MS＋图像 pureFM / soft-anchor 动作学习对照。
- 用户已授权自动完成当前实验、修错、评估、发布评审并推送 GitHub；这里不新增未经定义的训练规模、BPM 分支、6case 或无限重训。
- 原始 MS 动作按用户假设可信；rich＋MS 融成一份正文，不分别重写，不用逐条原动作语义审核阻挡当前实验。来源绑定、拆分、完整性仍检查。
- N1 本轮 fusion 不注入 BPM；已实现音乐/动作维度 sidecar 不等于这些字段已进入当前 caption 或训练。
- 原始 learned raw caption 即使重复也保留，不临时剪句、改动作、加重复惩罚后冒称原模型效果。输出原窗口对应仍 unknown；格式通过不等于已证实音乐到动作泛化。
- 用户前序反馈：userdirect100 所看几个 case 全部摆 pose；原 MS → 冻结 H3 真纯视频有舞蹈。前者使用 native AV joint/直接 hidden，后者无内部音频、用原生 TC，**不是单变量对照**，不能把差异只归因于音乐 encoder、caption 或数据量。

路径别名（仅历史证据定位，未来运行应显式提供配置）

~~~text
R    = /apdcephfs_gy4/share_302533218/cedricnie
REPO = R/H3-posttrain
MS   = R/diff/logs_omnidance/exp/MS
~~~

## 2. 当前状态：训练、推理、发布分开记

| 项目 | 最新实际证据 | 当前判定 |
|---|---|---|
| N7 原 MS＋genre 官方30B 6000/500 | `finished.json` exit0/completed_sft=true；3750更新、5完整epoch、30000曝光；best NLL在e03，后续plateau | 训练有限预算和独立完整checkpoint审计完成；caption/video目标未验收 |
| N7 最终u3750 fixed3 caption | `omni_ms_style_full_eval3_queue_v2/finished.json` complete=true/exit0，3固定ID，无GT user、无新增训练 | 最新真实推理完成；不混同N6 u3000，不冒称最终三视频已产 |
| N2 token-full6000/500 | `music_direct_token_full6k_n2_v1/progress.json` update=750、actual_exposures=6000、unique_train=6000、complete=false | 3epoch/2250更新进行中；不可把最初500 checkpoint或CPU2250 fixture当正式完成 |
| N5 pureFM@100 | 训练exit0、final100 marker；旧local_v6 queue finished=false来自后验scheduler断言 | 100实际更新已完成；视频另验。旧失败记录保留，不重训已正确pureFM |
| N5 soft-anchor0.1@100 | 同事最新观察update=33；尚无finished或anchor01_verified | 运行中；从原底座独立起步，不是接pureFM权重 |
| N4 pureFM100三视频 | `review_h3_purefm100_n4_chunked_v2` finished/run_complete均true、video_exit0、3/3 completed audit | 整组三视频技术完成；分块LoRA运行marker存在，质量不自动判优 |
| N6 u3000三caption＋视频 | 3原始caption完成，两个512cap重复；视频finished/run_complete均true、video_exit0、3/3 completed audit | u3000三视频技术完成；非最终u3750，human_quality_accepted仍false |
| N1 rich＋MS fusion | 已转三分片v3生成＋v4源可信导出；基础v2计数仅历史，启动3026旧＋2974新 | 尚未full6000 complete；以v4真实导出marker为准，不逐次追加运行计数 |
| 自动评审发布 | publisher从完整audit逐条更新原focused manifest，保留原baseline/presets/scores | 发布不等于人工评分；HTTP可访问验收另以实际结果记 |

N7 NLL五轮：0.778760 / 0.767208 / 0.763533 / 0.773399 / 0.807810。停止原因是 validation_plateau_not_human_quality_acceptance；human_quality_accepted=false。

已读取仓库小报告 `results/ms_20261020/omni30b_ms_style_full_complete_audit_v1.json`：审计任务vr7ERJ exit0，4文件SHA、628个Adam state全部step3750、FP32 moments finite、8rank RNG、终态sampler e05/p0/permutation=null。**这是对真实completed审计报告的复核，不是本CPU检查重新加载15GB Adam，也未额外做GPU续训或下一步恢复。**

N7最终三raw caption已保存 `MS/omni_ms_style_full_eval3_queue_v2/full_predictions.json`，词数307/192/277。其JSON没有EOS/生成token统计，不据词数推断是否撞cap；无新增人工效果评价。

## 3. 固定三个source ID与对照矩阵

共同纯视频协议：704×1280、124帧24fps、50步、seed42、CFG1、video flow shift12。H3内部无audio VAE/patch/head/scheduler；网页音乐仅外评播放。每case必须有complete audit、124帧/尺寸/fps/no-audio/full-decode报告以及50真实video scheduler updates。这里读取既有audit，未重复解码视频或媒体hash。

| 配置 | wJPptuhclZmKAEWs（楼梯） | wJPrpFsThWAcBQIw（红衣） | wJQTclrlOLEYYfTB（商店） | 解释 |
|---|---|---|---|---|
| 原MS＋同图 → 冻结H3纯视频 | complete audit | complete audit | complete audit | 3/3；用户已有舞蹈正反馈，不代填逐case分数 |
| 原MS＋同图 → pureFM100 H3 | complete audit | complete audit | complete audit | 3/3技术完成；LoRA分块保持纯视频协议，未自动判质量通过 |
| 原MS＋同图 → soft-anchor0.1 H3 | 待训练验收 | 待训练验收 | 待训练验收 | 先完成100更新与完整状态验证，再推理 |
| 音乐→Omni u3000 raw→冻结H3 | complete audit | complete audit | complete audit | 3/3技术完成；caption未清洗，非最终u3750 |
| 音乐→Omni最终u3750 raw | caption已产 | caption已产 | caption已产 | 暂无本配置三视频完成证据 |
| mean/token direct@100 | 历史已交付 | 历史已交付 | 历史已交付 | AV joint/native音频＋noTC hidden；用户pose负评，不并入纯视频单变量排名 |
| N2 token-full2250 | 未到最终预算 | 未到最终预算 | 未到最终预算 | 500val检索/hidden工程验收和fixed3视频质量是不同层级 |

对照证据根目录

- 原MS：`MS/h3_ms_style_eval3_gt_videos_v1/run_complete.json`，三个 `<ID>.audit.json`。
- pureFM：`MS/review_h3_purefm100_n4_chunked_v2/videos/<ID>.audit.json`。
- u3000：`MS/review_omni6k_u3000_n6_video_v2/videos/<ID>.audit.json`；caption来源 `MS/n7_omni6k_u3000_eval_n6_v1/omni_prediction3_raw.json`。
- anchor新评测：`MS/review_h3_anchor01_s100_n4_v2`；当前尚无complete视频。旧v1是取消的历史等待器，不是训练/推理失败。
- 旧direct：`MS/music_direct_video_compare_n4_v1`。生成协议见 `scripts_omnidance/exp/ms/music_direct_video_manager_n4_v1.py:47`，并非真纯视频。

u3000 caption真实生成边界

| ID | token数 | EOS | 撞512cap | 重复4gram excess / 最大重复 |
|---|---:|---|---|---:|
| wJPptuhclZmKAEWs | 512 | 无 | 是 | 384 / 32 |
| wJPrpFsThWAcBQIw | 224 | 有 | 否 | 132 / 12 |
| wJQTclrlOLEYYfTB | 512 | 无 | 是 | 370 / 28 |

这些重复统计只说明解码输出存在循环，不审核或否定原MS动作真值，也不证明唯一失败根因。

## 4. 修错与独立CPU反例

1. **N5训练后错误scheduler断言**：local_v6要求DeepSpeed model shard的lr_scheduler为真；本配方scheduler由Accelerate的scheduler.bin保存，DS lr_scheduler=None合法。新anchor_v8检查scheduler.bin以及原8rank模型/Adam/RNG，不篡改原训练SOURCE_SHA。历史错误断言见 `scripts_omnidance/exp/ms/runtime_jobs/h3_ms_style_train_manager_n5_local_v6.py:218`；修正验收见 `scripts_omnidance/exp/ms/runtime_jobs/h3_ms_style_train_manager_n5_anchor_v8.py:225`。prior summary明确full_checkpoint_content_reverified_this_run=false，不能把简要修正验收扩写为本次重新审计全部pureFM内容。
2. **pureFM推理LoRA临时激活OOM**：分块两低秩matmul的临时激活，不改权重、完整条件或sampler。实现 `scripts_omnidance/exp/ms/h3_video_infer_chunked_lora_v1.py:18`；CPU FP32/BF16四shape通过，最大绝对差1.19e-7；GPU真实三case completed audit/run_complete通过。CPUtoy不等于整模型GPU逐值证明。
3. **N6 projector dtype**：加载modules_to_save前先把audio proj1/proj2转成保存dtype，逐tensor严格核实际加载值与saved值，不把BF16损失误当已完整加载。见 `scripts_omnidance/exp/ms/runtime_jobs/n7_omni6k_u3000_infer_n6_v1.py:74`。原失败queue保留，接续视频复用已产真实raw caption，不再重生成。
4. **N7最终评测内存准入**：旧v1在模型加载前因cgroup headroom不足失败。已完成审计报告记录只对本人完成的training_state.pt作POSIX_FADV_DONTNEED，保留180GiB门禁，不全局drop_caches、不删文件、不杀其他进程；最新v2三caption确已complete。
5. **评审manifest缓存原子替换竞态（本次独立发现并复测）**：读旧打开文件后stat新路径，会把旧gzip绑到新mtime，下一GET继续旧body。temp CPU fixture真实失败；主助手改用打开文件fstat后同反例通过。源码 `scripts_omnidance/exp/review_web/server.py:469`。这是本地非HTTP回归，不宣称已重启或远端HTTP验证。
6. **token“整序列排列不变”宣称过强（本次独立反例）**：score_samples按linspace取最多64点，>64序列随机排列会换所取集合。seed42/B2/T128/D5120 logits最大差0.00458858，严格全排列不变assert失败。见 `scripts_omnidance/exp/ms/music_text_contrastive.py:57`、`scripts_omnidance/exp/ms/music_direct_token_compare.py:67`。正确表述是“等距64点集合内无显式时序对应的soft-Chamfer聚合”；不是全token逐项监督，也非完整序列严格排列不变。**不在正在训练时改loss、源码SHA或历史CPU报告。**

## 5. 有限完成验收清单

- [x] N7授权弱监督full6000/500训练达到实际plateau终点；独立完整checkpoint审计已完成。
- [x] N7最终u3750固定三ID music-only raw caption推理实际完成；原输出保留。
- [ ] N1恰好6000样本processed、run_complete、实际failed/included统计；稳定全量导出与lineage。原动作可信，不加额外语义/BPM门禁。
- [ ] N2实际2250成功更新、3epoch、18000曝光、6000unique、500val结果及完整model/Adam/RNG/sampler；固定三case推理，不用检索提升替代视频。
- [ ] N5 anchor0.1独立100更新、八rank完整状态验收；不重新训练已完成pureFM、不额外插入anchor1/300。
- [x] pureFM100与u3000各固定3真实视频完整audit/run_complete及video_exit0（共6条）；不代表人工质量通过。
- [ ] anchor100同固定3case完成。
- [ ] 若将最终u3750接到H3，独立命名配置并固定3视频；不得以u3000三视频替代最终模型验收。
- [ ] 已完成视频逐条发布、保留旧baseline/presets/scores；真实manifest/媒体range HTTP验证。自动可播放不等于人工效果通过。
- [ ] 汇报训练工程完成、效果通过/失败/待人评分离；目标未达成时记录有限负结果，不无限追加训练。
- [ ] 最终Git仅显式纳入必要源码、方案和小结果；diff/CPU/安全检查后由主助手commit/push并核远端。本文作者未commit/push。

资源边界：当前N5八卡训练、N2单卡映射、N1三卡分片融合、N4/N6有限推理按实际admission使用；anchor评测队列最多两条H3 pipeline/节点。未用节点/GPU应释放给用户其它任务，不把八节点全群默认独占。归档脚本资源快照仅历史证据，不能据此声称当前仍占用相同资源。

## 6. 必要Git源码与历史CPU provenance

新的 `scripts_omnidance/exp/ms/runtime_jobs/` 保存**原MS运行源码原文**，未迁移live文件、未修改历史SOURCE_SHA。17文件逐字节与MS原文件相同，AST通过，未发现api_key/password/bearer/private-key字面量。完整SHA/来源路径在 `results/autonomous_experiment_cpu_verification_20261024_v1.json`，不是仅凭文件名宣称同版本。

| 原MS文件名（原执行路径为 MS/文件名） | 归档SHA前12位 |
|---|---|
| `dance_caption_fusion6k_manager_v2.py` | `3840aa1df7c5…` |
| `dance_caption_observed32_30b_manager_v1.py` | `5645a9438ec2…` |
| `eval_anchor100_review_n4_v1.py` | `df2dbbfa0ecc…` |
| `eval_purefm_chunked_review_v2.py` | `9e75abced005…` |
| `eval_ready_models_review_v1.py` | `8c3bc4dab8d6…` |
| `h3_ms_style_cache_manager_n5_v2.py` | `337c6221c8b1…` |
| `h3_ms_style_train_manager_n5_anchor_v8.py` | `b7b3dc169434…` |
| `h3_ms_style_train_manager_n5_local_v6.py` | `df6b26bf1fcd…` |
| `h3_video_dimensions_case1_manager_v8.py` | `5331eb12067c…` |
| `n7_omni6k_u3000_infer_n6_v1.py` | `af14eab04c18…` |
| `n7_omni6k_u3000_manager_n6_v1.py` | `0e3cf082c9f3…` |
| `n7_omni6k_u3000_stage_n6_v1.py` | `44d6c3875ade…` |
| `omni_ms_style_full_eval3_followup_v1.py` | `7f7ac79533f9…` |
| `omni_ms_style_full_manager_v2.py` | `159d6b71728d…` |
| `omni_ms_style_pilot_manager_v1.py` | `2379dd917e1c…` |
| `publish_ready_eval_review_v2.py` | `0efd0c695a7d…` |
| `stage_h3_cache_n5_local_v1.py` | `9012a522cc0d…` |

只保存当前主线必要manager/stage/infer/publisher及其实际admission依赖；local_v6保留scheduler误判来源，u3000旧manager保留真实失败/接续区别。实际N6接续使用eval_ready_models_review_v1，而非把旧u3000 manager重新启动成功冒称当前成功。

**这些copy是历史证据，不是可直接运行的便携launcher。** 内含历史绝对路径、私有host、GPU UUID、UID/cgroup、旧执行目录和SHA门禁，不能作为未来必须使用的配置。未来执行需明确授权的节点/目录/依赖配置；不要盲目运行、不要用python -O关闭assert门禁。copies仅做CPU AST＋字节一致性，**复制后没有再次GPUverified**。权重、媒体、原始caption/私人数据、Adam、凭证、整历史日志没有上传。

必要repo入口（已完整读取，不把新CPU版本冒充原GPU执行版本）

- N7历史真实训练：`scripts_omnidance/exp/ms/omni_ms_style_full_train_v2.py:110`，SHA eaef7443…；调用冻结`omni_dance_sft.py` 4d82caff…。
- 新独立全量入口：`scripts_omnidance/exp/ms/omni_dance_full_sft.py:1703`，SHA a0214ca8…；本次CPU随机真实架构＋full-scope mock gate通过，未GPU训练。
- N2：`scripts_omnidance/exp/ms/music_direct_token_full6k.py:329`、采样`:104`；2250/750预算`:26`，原100对照不覆盖。
- H3训练：`scripts_omnidance/exp/ms/h3_video_sft.py:161`。soft-anchor是在sigma≥0.5加base预测正则，低sigma只FM，不是旧hi硬替GT目标；当前alpha64/rank32、lr1e-5、100预算。
- H3原生图文推理：`scripts_omnidance/exp/ms/h3_video_infer.py:465`；cache来源绑定`:296`，已有媒体audit实现`:387`。
- N1融合system与全6000终态：`scripts_omnidance/exp/ms/runtime_jobs/dance_caption_fusion6k_manager_v2.py:29`、`:288`。
- 用户原动作可信导出：`scripts_omnidance/exp/ms/export_fusion_source_assumed_v1.py:21`，仅确定性代词/列出的synthetic timing cleanup，不改方向/角度/幅度/重复次数。

## 7. 本次独立检查范围

结果简报：`results/autonomous_experiment_cpu_verification_20261024_v1.json`。

- 68个repo相关Python AST初检零错误；17runtime原文copy AST＋SHA一致。
- sidecar 7tests、fusion export cleanup、LoRA chunk FP32/BF16四shape、H3完整CPU selftest（tiny真实blocks/LoRA/anchor/backward/Adam-RNG resume）、Omni两family随机真实架构各2更新、full-data enabled scope正/负边界通过。
- H3推理6项CPU测试通过，**主动排除真实视频全帧解码test**。
- token目标CPU测试限定执行到完整format4模型/Adam/RNG下一步恢复断言之后，止于private init/manifest fingerprint之前；mean/token loss、梯度、mask、false-negative B×B CE通过。不是当前6500媒体重复准入审计。
- 原H3 CPU CLI少传必需dataset_base_path第一次argparse退出；补正确参数后通过，不把调用错误写成训练bug。
- 真实失败反例：manifest竞态已修复并复测通过；>64整序列排列不变宣称不成立，保留结果并收紧解释。
- CPU环境显式CUDA_VISIBLE_DEVICES为空、HF/TRANSFORMERS offline，不联网、不加载pretrained全量权重、不新增GPU、不杀进程、不安装/下载。现存full-decode字段来自此前真实audit，不是本次重解码。

此快照仍有在训/待视频/待人评/待推送项，overall_experiment_goal_complete=false。

## 8. 01:10后续运营入口核查（不改01:05实验快照）

- anchor当前等待/续评入口是 `MS/eval_anchor100_review_n4_v2.py`，SHA `3e01f5489c11809e03d1d3adb9cf5e2485415fbb97fb16e516540135139e9829`，输出 `MS/review_h3_anchor01_s100_n4_v2`。旧v1等待器因所用/usr/bin环境缺safetensors在推理/抢卡前精确停止，保留源码与旧输出；上文v1路径是历史快照，不是当前运行入口。
- publisher当前是 `MS/publish_ready_eval_review_v4.py`，SHA `2500df1a159660e1a973cb703fa74afb050e3aca9fb4fa96e4b218d68628007c`。它纳入anchor-v2，并将单组exit失败/finishedfalse隔离，继续发布其它独立完成组；旧v2源码留作历史。两最新原文已另存runtime_jobs，累计19份，不改live版本。
- N1新入口 `scripts_omnidance/exp/ms/fusion_complete_parallel_n1_v3.py`，SHA `8917ef1d90094d1c392f0dd3097d546371ce62d81ec5fca2b5cfbcb23ed1e682`。复用冻结legacy manager `3840aa1d…`（原文已归档）与helper `fc387286…`，新输出 `MS/fusion_parallel_rewrite_v3`，计划导出 `MS/fusion_source_assumed_full6000_v3`。同配方缺失ID按i%3三分片，原v2产物不覆写；本轮仍train6000，val500 deferred。脚本进入plan或启动不等于6000完成。
- 本次完整读新入口并独立CPU fixture通过：publisher一组失败、两其它组后来3/3发布且旧model/scores保留；N1 synthetic6000行材料化保留1500原字节、三shard各1500且互不重叠/原序号分配/taskSHA/拒绝覆盖；anchor-v2无finished等待及failed/99step/RNG7/wrongmarker/缺shard拒绝。safetensors在实际H3 venv可导入。
- 这些是隔离fixture，不调用live manifest、不联网、不解码真实视频、不读取原6000source_records或模型/Adam、不启动GPU/stop_old，不宣称N1全量导出或anchor已训满。所有冻结源码SHA未改，仍需后续actual completed audit/evaluation更新状态。

## 9. N1源可信v4导出修正（01:13快照）

实际最终导出目标改为 `MS/fusion_source_assumed_full6000_v4`；入口 `scripts_omnidance/exp/ms/fusion_source_assumed_full_n1_v4.py:86`，SHA `30cd043149375920c185e5195767f38b5f929e57d826f73cd32ccb09b6f73795`。生成仍由原三分片v3继续，不改被冻结模型/system/worker，不因导出规则误判重启生成。

旧v3 finalizer把生成的source_refs误指metadata/MS句子等结构标记当硬门禁，与用户已授权的source-assumed正文导出范围不符。v4只要求真实SID/split/group、冻结rich＋MS源和实际输入完整绑定、input hash、非空完整已生成motion body、生成未截断；生成refs/关键词旧flags只保留lineage，不阻断可信原动作正文，也不伪造语义/人工批准。raw JSON若仅尾随文字，可恢复第一个**完整**已输出JSON对象，尾随原文和SHA留存；截断/空正文/重复键必须拒绝。

本次独立temp CPU fixture通过：refs错误/旧flags非阻断；wrong source/group/hash/semantic批准伪造/撞cap/token数量不符拒绝；尾随JSON恢复，截断/空/重复键拒绝；重复结果拒绝；synthetic6000全不完整时不建立EXPORT或complete marker。未执行live export/wait、未读取私人6000source records或模型/视频，不用fixture冒实际全量完成。

2026-10-24T01:13:30.049354+08:00只读plan/目录事实：3026旧记录复用、2974待新生成；shard0/1/2已保存17/12/10个16字符ID结果文件。v4 complete不存在，v3 run_complete不存在；这是**文件数**不是逐条独立eligible计数。仍须最终included=6000/unique=6000/unavailable=0与两导出文件SHA、完整来源lineage后才标全量完成；val500仍deferred。v3自身finalizer可能继续留下refs unavailable失败；不得拿其失败否定已完成的生成，也不得以独立v4已启动冒全量成功。

## 10. 已读取真实启动里程碑与N2最终等待入口

N1真实启动报告 `MS/fusion_parallel_rewrite_v3/parallel_startup_verified_v4.json`，SHA `d4d7d39e58a6cf0fae7a845dcad09f1a86159548a716e74e4af989f3d7f93811` 已独立核SHA并完整读取：milestone为three_shards_loaded_and_first3_real_forwards；复用3026、待新2974，分片分配987/992/995；首9条有路径/记录SHA/parsed_body=true，生成tokens均<2048。报告自带batch6000_complete=false，是**真实启动历史**而非全量完成；首9记录内容不重复读取或全文语义审核，GPU/worker currentidentity仍按该报告时点，不假装本次实时重新查询。小结果JSON已保存必要milestone字段与首9路径SHA，不上传caption正文或整worker/cgroup历史。

N2最终等待入口 `scripts_omnidance/exp/ms/music_direct_full6k_final_video_n4_v1.py:44`，SHA `b1069ea048b6f694713b7871851760eabcff06b1e0a92323180cb12a77910458` 已完整读取并独立执行CPU自测：严格final2250/18000曝光/6000unique/3valepoch/467format4状态、配置scope正例；750/2249/2epoch/false或数值complete/布尔更新/500替代checkpoint/partialfailure/错误部署metadata共12负例全部拒绝。仅fixture，不coldread最终模型/Adam、不采GPU、不联网。

实际小JSON phase仍 `waiting_final_no_GPU_no_model_coldread`；最终 `MS/music_direct_full6k_final2250_video_n4_v1/report.json` 与 `review_http.json` 均不存在，不写成已完。后续严格完成训练状态审核后才在N4 GPU3单pipeline生成fixed3；使用原native AV joint、video/audio各50步、原音乐mux，必须与历史mean/token对照，不和源caption纯视频当单变量。统一publisher还须注册本modelID/fixed3/preset及保留旧mean/token；等待manager看到真实HTTPmanifest＋3range验证后才出技术完成。

## 11. 最终2250统一publisher v5的独立CPU验收

新 `MS/publish_ready_eval_review_v5.py`，SHA `89f72e724e48aecb8d3123839e61065f26dbe24efd95250c563b2e7df159a40a` 已完整读取、AST和原文SHA核验，另存runtime_jobs（累计20份）。原v4文件不改；v5现为当前运行入口，托管任务KLT9Wr已独立核为running，`MS/publish_ready_eval_review_v5.log`真实首事件为pureFM/u3000各3、anchor/final2250各0。旧v4仅精确取代publisher，不是模型训练或推理失败。

独立temp fixture验证：缺最终full receipt不给发布、2249拒、2250/18000/402Adam/467state/full RNG sampler＋真实nativejoint身份记录正例收；纯视频anchor故意audio_streams=1拒绝，但其它三组后续3/3仍发布；精确basename、nativeAV audio_input/output=true与purevideo=false明确区分，oldmean/token/preset/scores保留。仅fake媒体/audit/receipt，无HTTP/GPU/媒体hash或decode。只读真实fixed3 manifest核音频stem恰好等于各SID，当前direct文件名 `<SID>_<SID>.mp4` 与实际case一致。

N2仍须真实finished与final_cpu_checkpoint、逐SID validated/audit再进入v5；CPU fixture2250不是训练2250，发布尚未验证的新模型不得预填槽位。既有pureFM/u3000两组六视频技术完成保持；anchor及最终N2/最终u3750仍按各自actual completion验收。

## 12. fstat修复已实际加载到8775（原本仅CPU回归的后续闭环）

运营只精确重载8775同argv/同focused manifest/同评分，不改source或其他端口。旧PID2974656启动16:24:20Z早于fstat修复mtime17:03:42Z，故旧进程确未加载该修复；新PID3105683/start523476104晚于修复。已独立读取实际托管任务MJNBOR状态running、其启动命令与 `MS/review_8775_fstat_loaded_v2.log` 的serve8775日志，当前fstat为**实际重载已加载修复**，不再仅是磁盘source/本地CPU通过。

同事重载后检查6视频HTTP206、api/manifest200、reviews200且oldmean/token/base保留；具体已保存HTTP证据JSON路径仍在索取，未读取之前按同事实际检查报告标注，不伪造本代理另发HTTP验证。publisher v5 KLT9Wr继续运行，现pureFM/u3000各3、anchor/final2250各0；不把重载成功当全部实验完成。此前§4的本地非HTTP回归描述保留为修复历史。

## 13. 02:06 实际进度与最终身份验收修补

- N2主会话实际读取progress.json为1873/2250、epoch=3、曝光14984、unique_train6000、complete=false；随后最终评测消费者报告实际1903，latest仍1500。两个时点不矛盾。第三轮仍在运行，最终等待器不占GPU、不提前冷读模型，不把1500或检索指标当最终结果。
- N5 anchor实际39/100，尚无100完整训练终态，评测继续等待。本轮pureFM100和u3000各三视频已技术完成，用户人评另见results/h3_purefm100_and_omni_u3000_user_review_20261024.json。
- N1最新worker实测3586/6000，仍只有三个生成worker。已授权独立v5按剩余SID最多八GPU错峰加载；每次fresh cgroup预算准入、保旧完整正文（包括refs-only）、保SYSTEM和模型配方。这里是扩容实施中，不声明八worker已开始forward或6000已完成。
- 最终u3750：主会话读取cache_exit.json=0；video child3962681、parent3962061实际Rl，GPU显存78134MiB，argv使用最终caption及其condition cache。三case audit与finished仍未产出。原sample_cached没有逐步stdout，故无采样日志不证明死锁；有进程/显存也不证明视频成功。已要求独立publisher v6增加final3750第五组，不能以中间u3000替代最终。
- 完成checker独立CPU反例已证明可选SHA门禁会接受错误anchor LoRA或Omni caption，bool可混作exit_code/anchor_lambda，部分native完整状态字段不足也会误过。已授权修补一次性验收代码、强制实际来源绑定及typed budgets，并补交换u3000/final3750等负例；此时修复尚未交付，不把发现漏洞写成已修好。运行中的训练/推理源码保持冻结。
- GitHub仍是阶段200cad1；完整目标当前未达成，最终发布须等全部有限训练、完整6000导出、五组15视频和真实HTTP通过后再次push并核HEAD。


## 14. 最终五角色/15case验收材料的独立核查（仍非全部完成）

当前五角色配置 `MS/goal_completion_video_roles_v1.json`：purefm100、anchor01_100、omni_u3000、direct_full2250、omni_final3750共5个不同modelID；每配置固定3个sourceID，合计15case。四个caption-H3配置是pure_video，direct2250保留原nativeAV，roles/protocol核查通过。02:06 CST实际completed audit仅pureFM3＋u3000 3＝6；anchor/direct2250/final3750尚无完整终态，融合最终v4 complete也不存在。

严格checker实际运行 `MS/goal_actual_completion_audit_1792778114928.json` complete=false/pending，退出2是正确行为，不放宽。该环境Git lsremote超时只说明远端连通待核，**不能据此认定之前push失败**；当前HEAD200cad1仅阶段提交，最终GitHub核由主会话可连环境完成。

独立temp CPU负测发现旧checker身份门禁缺口：source-assumed布尔False可被忽略；N7 exit_code=False与0相等；pureFM anchor_lambda=False与0相等且anchor full-state范围未约束；anchor loraSHA及两Omni captionSHA可选时错误身份仍通过；direct receipt state467/exposure18000/uniqueepoch及native音频/学生未读teacher的条件不足。现有selftest通过不能消除这些反例；已向主助手报告并协调独占checker修正，**不修改任何运行中训练/推理源**。当前video_config也缺anchor loraSHA与两Omni captionSHA，必须补明确身份，不能因为训练尚在跑而删除gate。

小报告 `results/final_packaging_gate_verification_20261024_v1.json` 包含roles、6/15当前事实、真实负例与源码行号、融合v4最终路径、待核Git边界。最终3750运营源码 `scripts_omnidance/exp/ms/runtime_jobs/eval_omni_final3750_review_v1.py` 原文已归档SHA a21f4e278e…、字节一致＋AST通过，复制未再GPU验证（累计21份runtime）。

发布材料只纳必要源码/小报告，不stage来源未知的notes26修改或agent6结果；不上传weights/media/原caption私有数据/Adam/凭证。本文作者不stage、不commit/push。
