# 当前全部实验：有限完成标准、最新证据与代码归档

日期：2026-10-24；只读快照：2026-10-24T01:05:03.330987+08:00。这是一份**运行中的有限验收清单**，不是“全部实验完成”声明。后续必须以新 completed marker / audit / evaluation JSON 更新本页，不用任务启动或旧代理报告冒充成功。

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
| N5 soft-anchor0.1@100 | anchor_v8最新日志update=15；尚无finished或anchor01_verified | 运行中；从原底座独立起步，不是接pureFM权重 |
| N4 pureFM100三视频 | `review_h3_purefm100_n4_chunked_v2/videos` 当前1/3 completed audit，无run_complete | 首case工程通过，其余生成中；非整组成功 |
| N6 u3000三caption＋视频 | 3原始caption完成，两个512cap重复；`review_omni6k_u3000_n6_video_v2/videos` 当前1/3 completed audit | u3000部署与caption推理完成；三视频仍待全部产物、人评 |
| N1 rich＋MS fusion v2 | 当前保存3062/6000个样本JSON，无run_complete | 尚未完；这是文件数，不虚构全部format-valid或全量导出 |
| 自动评审发布 | publisher从完整audit逐条更新原focused manifest，保留原baseline/presets/scores | 发布不等于人工评分；HTTP可访问验收另以实际结果记 |

N7 NLL五轮：0.778760 / 0.767208 / 0.763533 / 0.773399 / 0.807810。停止原因是 validation_plateau_not_human_quality_acceptance；human_quality_accepted=false。

已读取仓库小报告 `results/ms_20261020/omni30b_ms_style_full_complete_audit_v1.json`：审计任务vr7ERJ exit0，4文件SHA、628个Adam state全部step3750、FP32 moments finite、8rank RNG、终态sampler e05/p0/permutation=null。**这是对真实completed审计报告的复核，不是本CPU检查重新加载15GB Adam，也未额外做GPU续训或下一步恢复。**

N7最终三raw caption已保存 `MS/omni_ms_style_full_eval3_queue_v2/full_predictions.json`，词数307/192/277。其JSON没有EOS/生成token统计，不据词数推断是否撞cap；无新增人工效果评价。

## 3. 固定三个source ID与对照矩阵

共同纯视频协议：704×1280、124帧24fps、50步、seed42、CFG1、video flow shift12。H3内部无audio VAE/patch/head/scheduler；网页音乐仅外评播放。每case必须有complete audit、124帧/尺寸/fps/no-audio/full-decode报告以及50真实video scheduler updates。这里读取既有audit，未重复解码视频或媒体hash。

| 配置 | wJPptuhclZmKAEWs（楼梯） | wJPrpFsThWAcBQIw（红衣） | wJQTclrlOLEYYfTB（商店） | 解释 |
|---|---|---|---|---|
| 原MS＋同图 → 冻结H3纯视频 | complete audit | complete audit | complete audit | 3/3；用户已有舞蹈正反馈，不代填逐case分数 |
| 原MS＋同图 → pureFM100 H3 | complete audit | 待产 | 待产 | 同纯视频协议，推理LoRA激活分块；质量待人评 |
| 原MS＋同图 → soft-anchor0.1 H3 | 待训练验收 | 待训练验收 | 待训练验收 | 先完成100更新与完整状态验证，再推理 |
| 音乐→Omni u3000 raw→冻结H3 | complete audit | 待产 | 待产 | caption未清洗；并非最终u3750 |
| 音乐→Omni最终u3750 raw | caption已产 | caption已产 | caption已产 | 暂无本配置三视频完成证据 |
| mean/token direct@100 | 历史已交付 | 历史已交付 | 历史已交付 | AV joint/native音频＋noTC hidden；用户pose负评，不并入纯视频单变量排名 |
| N2 token-full2250 | 未到最终预算 | 未到最终预算 | 未到最终预算 | 500val检索/hidden工程验收和fixed3视频质量是不同层级 |

对照证据根目录

- 原MS：`MS/h3_ms_style_eval3_gt_videos_v1/run_complete.json`，三个 `<ID>.audit.json`。
- pureFM：`MS/review_h3_purefm100_n4_chunked_v2/videos/<ID>.audit.json`。
- u3000：`MS/review_omni6k_u3000_n6_video_v2/videos/<ID>.audit.json`；caption来源 `MS/n7_omni6k_u3000_eval_n6_v1/omni_prediction3_raw.json`。
- anchor新评测：`MS/review_h3_anchor01_s100_n4_v1`；当前尚无complete视频。
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
2. **pureFM推理LoRA临时激活OOM**：分块两低秩matmul的临时激活，不改权重、完整条件或sampler。实现 `scripts_omnidance/exp/ms/h3_video_infer_chunked_lora_v1.py:18`；CPU FP32/BF16四shape通过，最大绝对差1.19e-7；GPU真实首case completed audit通过。CPUtoy不等于整模型GPU逐值证明。
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
- [ ] pureFM100与u3000各固定3真实视频完整audit/run_complete；anchor100同固定3case完成。
- [ ] 若将最终u3750接到H3，独立命名配置并固定3视频；不得以u3000三视频替代最终模型验收。
- [ ] 已完成视频逐条发布、保留旧baseline/presets/scores；真实manifest/媒体range HTTP验证。自动可播放不等于人工效果通过。
- [ ] 汇报训练工程完成、效果通过/失败/待人评分离；目标未达成时记录有限负结果，不无限追加训练。
- [ ] 最终Git仅显式纳入必要源码、方案和小结果；diff/CPU/安全检查后由主助手commit/push并核远端。本文作者未commit/push。

资源边界：当前N5八卡训练、N2单卡映射、N1单卡融合、N4/N6有限推理按实际admission使用；anchor评测队列最多两条H3 pipeline/节点。未用节点/GPU应释放给用户其它任务，不把八节点全群默认独占。归档脚本资源快照仅历史证据，不能据此声称当前仍占用相同资源。

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
