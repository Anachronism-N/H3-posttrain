# OmniDance/H3 实验历史、音乐替换文本失败分析与当前执行状态

整理日期：2026-10-22；推进状态更新至2026-10-23（北京时间）。本文是用户要求的详细归档，不是新的实验启动命令。

## 1. 阅读规则与当前结论

本文将先前会话记忆与当前仓库文档、实际结果报告交叉核验；记忆只用于寻找线索，不作为训练完成或效果成功的独立证据。历史日志的“当前”“待人评”“已证实根因”均按当时语境理解，不能覆盖后来的人评、反例和新授权。

结论分为四类：

- **工程事实**：实际运行、梯度、成功更新、序列化、来源/媒体校验可核实。
- **真实人评**：用户连续观看后的原话或明确反馈；聊天反馈有效，不要求重复填写网页分数。
- **机制证据**：代码/数学性质及受控对照支持的解释；不自动升级为唯一因果。
- **计划**：已批准但未执行或未验收的内容，不写成已完成结果。

截至本次归档：

1. 旧 MusicFM 直接替换 H3 文本 hidden 的 6000/500 正式实验已完整结束，用户反馈仍近摆 pose，目标未达成。
2. 当前主线改为 **音乐 → Omni 离散融合 caption → H3 原生图文 TC → 真纯视频生成**。
3. Omni7B/官方30B真实两更新及保存/重载接口已通过；H3新124帧缓存、两更新、真实八卡恢复也通过。它们全部是独立 fixture 接口验证，不是正式 SFT。
4. 原始 H3 真纯视频 rich/MS 两条同 case 均获可用正评；融合 v7 获“更好且可用”；新增维度 A/B 的最新反馈为“几乎完全一致”。
5. 首32已完成真实AI证据审核：19条保留官方30B正文、13条明确由助手编辑修正，32条assistant_delegated批准；真实导出24train/8val已过Omni/H3来源链。不是人工评分，也不是正式512/128训练。640配对媒体已全部准备完成，正式caption目标仍待实际视觉生成/审核；pilot/full和端到端验收未完成。

## 2. 统一路径、数据与评审口径

```text
R    = /apdcephfs_gy4/share_302533218/cedricnie
REPO = R/H3-posttrain
EXP  = R/diff/logs_omnidance/exp
MS   = EXP/MS
```

仓库源码与旧执行树可能存在同名不同版本，复现必须核实际脚本 SHA。模型、Adam、原人物视频/音频存实验目录，不上传 GitHub；仓库保存代码、方案、小结果 JSON、来源哈希及结论。

### 2.1 不可混淆的数据规模

| 名称 | 实际用途与边界 |
|---|---|
| 旧36k列表 | 早期训练名单，不等同后来VLM精选集；不能将目录名或数量当质量/来源隔离证明 |
| full12k rich候选 | 可恢复原rich正文11992条；原生成模型/指令仍未知，不冒称本轮VLM生成 |
| 旧2k | 多个阶段使用1800train/200val；来源去重口径随阶段改进，不能与新6000集合混同 |
| 专用audio接口扩量 | 嵌套1800/6000训练、固定200val，输出32D原生音频latent代理 |
| 最新旧direct6000 | 6000train/500val、6500唯一来源组，图/P/y/audio完整哈希审计；输出5120文本条件 |
| 当前融合preview32 | 24train/8val、16genre，仅目标质量预览，不是正式512/128pilot |
| 当前接口fixture | Omni8train/2val、H3八条，固定接口caption；明确禁正式复用 |

后续数据默认走已核来源隔离与合格候选，排除固定评测case及关联歌曲/源片。MotionScript、rich、VLM维度、配对音频同ID，不自动代表同观察时间窗。

详细打标与字段说明见 [notes/27](27_dance_annotation_motionscript_vlm_and_dimensions_20261020.md)。

### 2.2 判断质量不能仅依赖指标

旧TextAlign会受到外观变化影响；全图DINO不是独立人脸身份指标；Pose-BAS不同实现存在峰/谷与时间口径差异；历史H3视频24fps与音乐/姿态16Hz按帧截取存在风险。GT校准不足的F1或Omni意见不能选择音乐winner。

固定三个case是诊断协议，不是完整泛化评价；同case多配置也不是独立样本。换音乐“动作改变”不等于“更符合新音乐”。详见 [notes/23](23_h3_posttrain_issue_review_20261014.md)。

## 3. 实验历史总览

| 阶段 | 核心问题 | 实际结果 | 当前处理 |
|---|---|---|---|
| 09-30至10-05 Wan2.2 | SFT、DMD、AR、长视频、reward | 自动指标若干改善，后续画质被人评否决 | 方法/历史参考，不继续默认重训 |
| 10-10至10-13 H3 AV/FM/CFG | 微调后手脸模糊、清单prompt、蒸馏保护 | rich恢复底座表现，但多种微调仍糊；SG/GT对照均退化 | 不把CFG或数据单独当唯一根因 |
| 10-14至10-15 画质归因 | 高噪声、分辨率、base锚定 | 高分辨率底座更好；锚定保底但变化小 | 清晰度与动作学习分开验收 |
| 10-16 原生audio CLIP | 原audio-VAE表征是否更懂舞蹈 | 表示/梯度有学习，生成增量肉眼弱 | 不将head检索提升等同编码器生成收益 |
| 10-17 专用encoder→32D | MuQ/MusicFM替原音频编码 | 48视频及扩量36视频完成；画质保留，未建立音乐winner | 停止无界同配方扩量 |
| 10-18至10-20 音乐→rawH | 文字控制能否由音乐hidden承担 | MSE/NCE、在线联训、六更新行为监督仍pose | 检查目标、空间与泛化 |
| 10-20 noTC/空间 | W/LN/PCA、移除TC、NCE消融 | PCA教师部分舞蹈但伪影；音乐学生NCE0/1仍pose | 不把teacher投影当音乐学生 |
| 10-21 纯文字教师1800 | 去图上下文、单配对监督 | 原曲部分舞蹈但弱跟拍，换曲近pose | 保留部分正结果，不绝对否定hidden路线 |
| 10-21至10-22 direct6000 | 保P/native音乐，直接1024→5120，CLIP/矩 | 28轮/2632更新/18视频；人评仍近pose | 技术收尾，效果失败 |
| 10-22 ABC教师 | 去P、图文y/纯文y | 单case差异小，B/C近似 | 仅教师诊断，不泛化为全部无P无损 |
| 10-22 当前两段主线 | 离散caption＋真纯视频H3 | rich/MS/v7正评；A/B近似；真实两更新接口通过 | 继续正式监督门禁与后续训练 |

## 4. Wan2.2到H3：为何更换主线

早期Wan2.2实验在47条验证上的自动数字包括S4-final TextAlign0.725、DMD b5@3000 TextAlign0.630/身份0.855；AR/长视频也有接缝和效率收益。但2026-10-10用户人工否决Wan画质，计算转向H3。

这些早期数字说明当时的指标或工程改善，不说明用户可用画质。notes/18中的“暂不换底座”是10-05时决策，不能覆盖10-10后的实际转线。也不能从Wan某种回归萎缩直接推出全部H3目标必然失败。

依据：[旧八节点总结](18_experiment_summary_8node_20261005.md)，[10-10转H3记录](19_h3_ar_plan_20261008.md)。

## 5. H3画质退化：CFG、prompt、数据与锚定

### 5.1 不同问题分别处理

- 旧H-A676加真CFG3曾改善手部/动作，这是特定历史配置的正结果。
- de-distill适配器与guidance loss是不同处理，不能只凭名称说某任务同时用了两者。
- 清单动作caption使H3僵硬；原rich prompt能恢复接近E3表现。
- H-A4转rich、full12k/750步后仍有手脸模糊，因此prompt不是所有训练退化的唯一解释。
- 同2091样本/global8/300更新的DS/OST/Hybrid、FM/direct/inverse多种配方仍被判糊。

### 5.2 GT512与SG512对照

SG512使用原始H3自己生成的视频latent作目标，GT512使用真实目标；共同原底座、r64/lr5e-5、repeat5/global8，阶段比较100/300步。SG100也“很多模糊”，说明真实数据降级不是出现退化的必要条件。低lr、r16、高/低噪声段的当时对照均未解决问题。

数值自复现中原底座可逐位复现SG；SG微调后相对轨迹距离约0.41/0.47，高频有所下降。高噪声预测高频下降及LoRA噪声段切换支持把FM微调链路列为重点嫌疑。

严谨结论是**当前训练过程与原发布模型行为存在不匹配的证据**。“蒸馏引导场被FM拉向均值”是支持较强的机制解释，不是已排除所有因素的定理。

### 5.3 锚定与分辨率

旧hi锚定在高噪声直接改为base预测目标；add锚定是FM外加预测正则。两者不能混同。

GT512普通FM300手部置信度joint/aswap0.564/0.567；ANCHI05约0.653/0.669，接近原底座0.672/0.677。但用户随后认为锚定与音乐引导组动作相似、无明显增量；原生704×1280/768×1344比480p好很多，仅手部轻微糊。

因此“像底座/不继续变糊”只是保底，不是音乐或动作能力学好。当前新纯视频方案采用**软base-anchor正则**，并不是复用旧hi硬替目标。

依据：[notes/22 §§3–10](22_h3_bidir_blur_attribution_20261013.md)，[机制与实现风险区分](23_h3_posttrain_issue_review_20261014.md)。

## 6. 原生audio CLIP与专用encoder替换

### 6.1 原生audio-VAE编码端

旧阶段使用1800/200，测试末端投影、全编码侧5轮/20轮、固定对比head等配置。原生模型不是直接使用5120文字token：音频经VAE编码为双声道32通道时序latent，进入独立audio patch与联合AV DiT。

CLIP读取有效音频latent的全局统计，经训练head与文字池化表示对齐；生成不使用这个head。固定head消融支持部分检索收益可能被head吸收。全编码侧权重/梯度和部署已实际核验，不能说“没训/没加载”。

用户连续观看后认为同音乐下各模型动作基本一致，换音乐只有少量差异、各模型仍相近；没有可感知训练增量。head捷径是候选机制，不是唯一因果证明。

### 6.2 MuQ/MusicFM→原生32D接口

真实5.0625秒双声道输入，MuQ/MusicFM原dense均为[2,127,1024]；MERT参考为[2,379,1024]。训练后共享encoder＋时序adapter将25Hz特征转到原40Hz、输出[2,32,203]，替换reference/retake两音频编码位置，保原decoder/DiT。

比较bridge-only、随机adapter pureCLIP、bridge初始化＋CLIP(lambda1)和同初始化lambda0。bridge20轮验证误差/teacher energy约0.68、预测能量约teacher0.32，不能当原生坐标恢复成功。共8配置48视频做完整技术验收。

MusicFM桥接初始化组相对随机pureCLIP曾获较好卡点反馈；并未证明优于原生。MuQ多配置与基线编排相近，画质总体可用。

### 6.3 嵌套1800/6000扩量

共同global16，560/1875更新；六组固定200val，36视频（每组joint3/aswap3）完整验收。

| 配置 | 双向平均R@10 |
|---|---:|
| 1800/lambda0/560 | 12.50% |
| 6000/lambda0/560 | 16.25% |
| 6000/lambda0/1875 | 20.50% |
| 1800/lambda1/560 | 14.50% |
| 6000/lambda1/560 | 14.50% |
| 6000/lambda1/1875 | 18.75% |

检索确实可改善，但原decoder声学重建未一致恢复，自动身体/节拍分数在GT校准不足。用户反馈两预设“肉眼上已经区分不太开了”“画质都没有退化”，未建立数据扩量、加更新或lambda优胜。

### 6.4 去语义文本验证

同权重仅音乐＋图的18视频已做，不额外训练。用户认为动作显著收缩、case1几乎没有舞蹈、其它也小。相对generic的body速度比约原生0.577、lambda0 0.503、lambda1 0.464；它们是运动变化量，不是音乐匹配准确率。

参考图是reference视觉anchor，不是目标首帧硬锁；空文本也移除了人物/任务自然语言绑定，因此该结果不能单独归因于音乐encoder。

依据：[notes/22 §§27–28](22_h3_bidir_blur_attribution_20261013.md)，[专用encoder最终验收](../results/ms_20261016/music_replacement_final_acceptance.json)，[扩量最终验收](../results/ms_20261017/music_scale_final_acceptance_v4.json)。

## 7. 音乐替换文本hidden：R0–R7逐轮协议、结果与转向理由

### 7.1 共用符号、参数与不变条件

以下编号是本归档的顺序索引，不是原训练目录名称。`TC`指H3的Qwen-VL视觉编码＋50层语言模型；`P`是图/引用结构的最终hidden前缀，`Y`是动作prompt的最终hidden后缀；它们均不同于词表input embedding。音乐学生的部署条件记为`S`，不把训练loss读出向量冒称实际完整条件。

`L_H`为教师有效token上的raw MSE除以train-only教师energy，`L_len`为预测log词长对真log词长的MSE；旧`L_seq`为每样本正确/两负例的序列InfoNCE，最新`L_batch`才是全实际batch的双向mean-CLIP。教师词长作为loss标签不自动等于部署GT词长泄漏，R0/R1训练query_count与部署不同的历史偏差另列。

| 轮次/日期 | 实际训练规模与预算 | 可训范围 | 主要loss/初始化 | 教师/生成输入关键区别 |
|---|---|---|---|---|
| R0 10-18 | 1800/200；1875步，batch8，15000曝光≈8.33遍 | adapter4,966,657参数；MusicFM冻结 | raw MSE＋词长；cos权重0 | 图文Y，保TC图P/context；native固定静音 |
| R1 10-18 | 四配方各1800/200、1875步、15000曝光 | 同R0完整adapter | 同R0末权重；MSE/seq-NCE/词长对照 | 同R0；获新授权后原候选直接9视频 |
| R2 10-18–19 | A/B/C各100和300步；800/2400曝光；54视频 | A只adapter；B/C再训316,343,296 dense参数 | 同R0末权重独立启动；seq-NCE，词长1/.01 | 全qmax＋自己词长；保P/context；native静音 |
| R3 10-19 | 3train×2遍、6步、batch1；200表示val＋1行为val；18视频 | dense＋完整42张量adapter | C300；有/无冻结H3预测MSE | 同R2部署；sigma.9/noise42；不是GT FM |
| R4 10-20 | 0音乐更新；200val数值；9新＋3复用教师视频 | 无 | C300的W256/LN255包络与train-PCA256 | 真教师投影，同词长/P/图/native静音 |
| R5 10-20 | W/PCA×NCE1/0；各8train/8val、100步、800曝光 | dense＋3,655,937参数/41张量adapter | C300各自新Adam；固定后投影 | noTC、P0、仅音乐学生；图VAE在，native静音 |
| R6 10-21 | 两起点各1800/200、1200步、9600曝光≈5.33遍；18视频 | dense＋public10,713,601/39张量或control3,655,937/41张量 | public512Identity/control256LN；单配对MSE | 纯文Y、无P/noTC；图VAE在，native静音 |
| R7 10-21–22 | 两组6000/500、global64、28轮、2632步、168000曝光；18视频 | dense＋Linear5,248,000/2张量 | 同公开MusicFM＋直接1024→5120；batch CLIP/矩 | 图文Y＋cached P；原生真音乐、AV joint各50步 |

R0/R1已核adapter lr3e-4、batch8；当前所读首轮结果未独立列出初始随机seed/完整Adam状态，不能拿后轮seed42倒填。R2/R5/R6训练seed42、batch8/micro2/accum4、dense lr1e-5/adapter1e-4、AdamW decay.01/global clip1；R3 decay0；R7 seed42、4rank×16/global64、AdamW decay.01/clip1。未单独执行历史Adam下一步复现的轮次不写“无缝恢复已证明”。

R0–R6学生正式视频共同704×1280、124@24fps、seed42、CFG1、双向video50步、video shift12/audio shift3；独立图像VAE一直存在。固定静音是原audio-VAE编码5.0625秒stereo波形所得非零latent，reference/retake均保留、尾repeat_clean/mask0；只更新video、最后mux音乐，不能称真纯视频。R7恢复真实native reference音乐并实际video/audio各50步，生成音轨保留，不再沿用上述静音/只step视频协议。

### 7.2 R0：冻结音乐骨干，逐token rawH回归首轮

**问题与真实输入。** 希望音乐承担原动作文字控制，不输入人物ID/动作caption给学生，不先加DiT/video loss。2000图文教师由同图结构＋`Dance style`＋完整原MS编码，中性代词逐条留痕；训练平均431.154词、最大1423，val不超qmax。教师raw Y不池化、不L2、不插值压缩；公开MusicFM特征缓存和targets全量SHA/finite核验。

**计算图与开关。** 冻结MusicFM特征→256宽时序memory/stereo/位置编码→图结构最后一行context＋位置queries→两cross-attention/FFN→末端LN/Linear5120＋长度头；完整4,966,657参数更新和部署。TC只产图P/context，H3 DiT/图及音频VAE均冻结。最初训练使用GT词长决定query_count和teacher mask；部署全qmax1423再按自己预测词长截取，不把这一差别写成已完全train/deploy一致。

**优化与技术状态。** raw MSE/train energy＋词长loss，cos权重0，lr3e-4/batch8/1875更新。此前smallfit300长度/幅值不过；rawMSE＋length/cos0/1000步小集修复长度MAE0、能量比.4056，多个因素一起改，不能单独归因cos。正式best/last507个state tensor完整strict/SHA/finite；权重SHA `cbb79c6338c2673d0c2f650a14da815234894055ffc14f0a56f1ec1afd1750fc`。cached/online相对RMS2.386e-7、词长259相同，数值非逐值完全相等。

| 全200val反事实，tokenweighted MSE | 正音乐 | 循环错配音乐 | 真静音音乐 | train逐位置均值 |
|---|---:|---:|---:|---:|
| raw Y误差 | 5.381604 | 5.381605 | 5.382138 | 4.756748 |

长度MAE137.71、energy ratio.454974；正/错输出RMS宏平均.002778。错配几乎不增误差且均值baseline更好，支持平均模板/弱音乐依赖线索，不等于已证明完全坍缩。这里tokenweighted基线4.756748不能与后轮macro4.815058不加口径直接比较。

**视频与判别。** 新raw模型原/换各3＋zero3共9，完整媒体、empty prompt/模型/末投影/native静音审计完成。采样所见小手势/后勾腿、脸头附近抬臂、原地单臂，并有楼梯服装漂移；像素差与预测词长差同时存在，不声称音乐内容因果。旧完整MusicFM→32D adapter→pool→CLIPhead→L2→单5120 token还做了免训练6条诊断，train-only norm272.45535，不是恢复原raw序列，也不能当R0重训结果。

**结论→下一轮。** 工程闭环通过、动作文本替代未建立；错误归因为“adapter没加载”已不合事实。改用真实音乐反事实对比，试图迫使原生raw输出保留音乐区分，而不是先增加数据或把CLIP全局向量当原词序列。

证据：[首轮训练](../results/ms_20261017/music_text_train1800_raw_neutral_v2.json)、[完整权重审核](../results/ms_20261017/music_text_train1800_raw_checkpoint_audit_v2.json)、[九视频](../results/ms_20261017/music_text_trained_all9_integrity_v2.json)、[notes/22 §28.20](22_h3_bidir_blur_attribution_20261013.md#2820-文本侧共享adapter首轮完整结果2026-10-18)。

### 7.3 R1：四配方，原生H序列对比与双真实负例

**同起点与可训范围。** 四组都复用R0的cbb79c…完整模型及同1800/200、lr3e-4/batch8/1875更新/15000曝光；公开MusicFM/TC/DiT/VAE仍冻结，完整adapter＋最后投影/词长头部署。不是音乐dense全量CLIP微调，也没有新增可丢训练head。

**loss细节。** train-only mean/std和L2只计算score；每侧最多64均匀有效token，逐token cosine矩阵双向soft-Chamfer（平滑温度.1），再三候选CE温度.07。标准目标`L_H + λ_len L_len + λ_seq L_seq`。三分支同图context，teacher词长/mask仅监督；R1尚沿旧query_count规则，不能提前写成R2的全qmax共享学生前向。

| 配方 | val tokenweighted MSE | 正音乐胜第一真实wrong | 正−wrong score | 词长MAE |
|---|---:|---:|---:|---:|
| MSE续训，NCE0 | 5.04644 | 52.0% | −.000077 | 138.975 |
| NCE1，wrong＋静音 | 5.49469 | 51.0% | .000050 | 149.000 |
| NCE1，两真实wrong | 5.35961 | 62.0% | .006339 | 146.060 |
| 两真实wrong，词长权重2 | 5.39910 | 58.5% | .007541 | 151.330 |

**不能混淆的数字。** 双真实模型报告的score诊断macro MSE5.491725与回归tokenweighted5.359610不是相互矛盾；R2重算又用自然学生有效区间。含静音组98%胜zero但仅51%胜真wrong，不用三选一chance1/3包装音乐语义成功。小8train300步可100%辨训练音乐而8valmargin负，也不是泛化证据。

**推理授权纠正。** 原门禁四组false、双真实长度146.06超init137.71的5%范围，最初因此未新增视频；用户随后明确要求直接推理而非先校准词长。原候选SHA `895868746f3fdec2b58acdae7fe1345d9c5670195e1850aef714a6930c301f45`保持、diag=true/quality=false，生成原/换/zero9条，不重训、不改gate。native仍固定同静音，学生不读caption/GT词长。

**真实人评。** “还是完全不像跳舞,动作和之前差不多,而且更换歌曲的差别也不大.” 新候选有手势变化、62%有限表示信号也不能推翻用户负评；原/换/zero词长及packed同时变化，不能用RGB差当卡点准确率。

**结论→下一轮。** 双真实负例比识别静音更合理，但未达到可靠动作文字控制。用户接着纠正“为什么不训练encoder的骨干?就是要训练才对啊”，并要求完全对齐训练和生成，触发R2在线dense＋全qmax自身条件修正。

证据：[双真实1875](../results/ms_20261017/music_text_Hclip_realneg_d1800_s1875_v2.json)、[词长权重2](../results/ms_20261017/music_text_Hclip_realneg_len2_d1800_s1875_v3.json)、[未校准直接九视频](../results/ms_20261017/music_text_Hclip_diagnostic_all9_audit_v1.json)、[notes/22 §§28.21–28.23](22_h3_bidir_blur_attribution_20261013.md)。

### 7.4 R2：在线A/B/C，100/300独立预算与54视频

**唯一学生条件。** 每分支真实波形→公开MusicFM dense→原完整adapter，全qmax1423前向，再按自身logL→round/裁剪词长/valid mask/tail；teacher长度只loss。train/eval/full reload/独立worker共用，loss标准化不改变部署raw坐标。仍读同图空正文TC图context并保P，图VAE/native静音在；不把条件协议一致扩大为已学会原文本生成行为。

**参数与初始化。** A冻bone，只adapter4,966,657/42张量；B/C联训实际dense316,343,296/400张量及全部adapter。骨干eval固定BN/dropout但参数有梯度，不是no_grad；FP32，不走冻结features/FP16往返。六组从同cbb79c…init独立新AdamW，100→300不是接续100步moments。

**优化。** seed42，batch8/micro2/accum4，bone1e-5/adapter1e-4、decay.01/clip1。共同`L_H + λ_len L_len + min(step/100,1)L_seq`；A/B λ_len1，C .01。每样本正确/第一wrong/第二wrong三音乐前向；负例不同ID及caption hash，不冒称完整distinct source-song审计。没有DiT/video/FM/节拍loss。

| 全200val，macro口径 | A100 | B100 | C100 | A300 | B300 | C300 |
|---|---:|---:|---:|---:|---:|---:|
| correct>第一wrong | 52.5% | 54.5% | 55.5% | 47.5% | 45.5% | 51.5% |
| tie-aware三候选top1 | 40.50% | 41.25% | 39.50% | 36.25% | 30.50% | 37.50% |
| raw H MSE | 5.45948 | 5.46769 | 5.46963 | 5.37288 | 5.40003 | 5.33456 |
| 词长MAE | 147.415 | 155.445 | 155.885 | 141.070 | 143.010 | 142.340 |

train位置均值macro MSE4.81506，六组逐例0/200胜该基线。八样本train probe比val音乐margin大只是有限probe分化，不称全train拟合或模型完全坍缩。

**54视频与真实边界。** 每组原3/换3/zero3，100步A/B18＋C9、300步ABC27；124帧逐帧/音轨全解码、完整权重/最后投影/empty语义/无GT词长/原packed/native静音hash通过。初始300队列等待PID误匹配及zero恢复保留失败日志，不重跑已正确视频。采样仍是小手势/后勾腿/头脸动作与衣着漂移；后续用户“摆pose”评价与这些目标未达成一致，但不能伪造六组逐case评分。

**词长只读消融。** A/B/C300在每样本四音乐输出共同可见query交集评分，0更新/0新视频，第一wrong胜率仍47.5/45.5/51.5%。原margin与词长接近teacher的相关(.625/.478/.745)不构成因果；共同区间排除了“评分取样词长差是弱判别的全部解释”，没有固定H3布局。

**结论→下一轮。** dense真实训练和完整部署已经核验，不能继续用骨干未训练/忘投影解释失败；100/300预算完成不等于收敛。减词长权重未显著修复，转向有明确新增授权的冻结H3行为监督，而不是擅自加FM或再挡住视频。

证据：[B300完整protocol](../results/ms_20261017/music_text_online_B_s300_summary_v1.json)、[100步AB18](../results/ms_20261017/music_text_online_AB_s100_all18_audit_v1.json)、[100步C9](../results/ms_20261017/music_text_online_C_s100_all9_audit_v1.json)、[300步27](../results/ms_20261017/music_text_online_ABC_s300_all27_audit_v2.json)、[共同区间](../results/ms_20261017/music_text_online_s300_common_interval_diagnostic_v2.json)、[独立重算](../results/ms_20261020/music_h_diagnosis_recomputed_v1.json)。

### 7.5 R3：六更新精确VJP，有/无冻结H3预测一致性

**新增授权及范围。** 用户授权“小规模实验”，只训MusicFM实际dense＋完整adapter，冻结TC/DiT/全部VAE；没有GT FM velocity目标、未恢复学生动作prompt。control/prediction从C300 SHA189a0a74…同起点，各新AdamW，3train按固定顺序轮转两遍、batch1/6更新/6曝光；bone1e-5/adapter1e-4、decay0/clip1。

**行为目标。** 两组共同`L_H + .01 L_len + L_seq`，prediction额外权重1的`mean[(F_student(xσ,I,A)-stopgrad F_teacher(xσ,I,A))²]`。`xσ=.1 z_GT + .9 noise(seed42)`，37个video latent帧、sigma.9；GT latent只是共同noisy state，不是velocity标签。教师图文Y/P仅监督，学生自词长，各自合法packed，双方同参考/静音/native mask。

**精确跨环境梯度。** 音乐环境export FP32tail→H3环境leaf/BF16cast/真实DiT返dL/dtail→音乐环境同checkpoint/wave/context重算torch.equal tail后回传VJP。存盘边界不等于detach代理训练。离散词长无行为梯度，400bone＋38内容adapter有限非零；4词长张量由原L_len训练，共42完整adapter仍更新。

**显存与恢复。** 短2latent smoke通过不能替全长：37latent单卡MLP重算OOM；token分块BF16 parity不过，拒用于训练；改原blocks后半跨第二GPU，不换算子/量化，全长峰50.03/48.33GiB通过。六步每步保存并恢复匹配Adam，独立核442参数状态全部finite、step6，不是逐步重新新建optimizer。H3参数无grad/前后hash不变。

| 固定1留出、同sigma/noise | init C300 | control6 | prediction6 |
|---|---:|---:|---:|
| teacher预测MSE | .02921945 | .02999355 | .02893900 |
| 学生自己词长 | 262 | 389 | 393 |
| 全200val第一wrong胜率 | 51.5% | 50.5% | 50.5% |
| raw macro MSE/词长MAE | 5.33456/142.34 | 5.33036/150.31 | 5.33050/151.12 |

prediction相对control −3.516%、相对init −.960%，严格只有1样本/1噪声/1时间，不是稳定泛化。两个末模型全200三候选top1均34.5%。

**视频与后续机制。** 两组18新视频全部审核，仍小动作/服装漂移；用户后来复审教师上界时确认两六步学生都是摆pose。三个历史监督点的分项梯度中bone NCE/prediction norm比约1320/470/36600，内容adapter并非总可忽略且cos近0；不是全参数强方向冲突，也没虚拟Adam更新来量化真实更新贡献。

**结论→下一轮。** 证明了全长真实输入梯度链，不证明6步行为方案有效或永久无效；仅3固定教师点各访问两次，未覆盖多sigma/noise/学生自由轨迹。下一轮先隔离输出空间及教师上界，避免用小loss标量猜唯一根因或任意放大权重。

证据：[六步summary](../results/ms_20261017/music_text_behavior_prediction_s6_v1_summary.json)、[六次VJP审计](../results/ms_20261017/music_text_behavior_s6_v1_all_training_vjp_audit.json)、[留出比较](../results/ms_20261017/music_text_behavior_heldout_s6_v1_compare.json)、[18媒体](../results/ms_20261017/music_text_behavior_s6_v1_all18_audit.json)、[分项梯度](../results/ms_20261020/music_text_prediction_components_C300_v1.json)。实现：[music_text_behavior.py](../scripts_omnidance/exp/ms/music_text_behavior.py)。

### 7.6 R4：原教师、W/LN/PCA投影，排除音乐预测误差

**零训练诊断。** 从固定C300末投影得到W256包络和更准确LN255仿射包络，把真实teacher Y最优投影进去；无MusicFM预测、无optimizer、教师词长228/199/323及同图P/tags/原packed固定。三空间各3joint＝9新，原neutral教师3复用＝12审核；native仍相同静音、图VAE和video50/mux同历史协议。

**空间如何拟合。** W float64 SVD实际rank256；LN中心`b+Wβ`、基`Wdiag(γ)1⊥`实际rank255，但都仍是放松包络，不限制LN半径/上游可达性。PCA用1800train全部有效token精确FP64中心，每项最多16等距token、共28800，按长度/抽样数加权，低秩q288/niter4/seed42近似取256并正交化；无val/prefix/padding/评测case参与，不称全量精确最优PCA。

| 空间 | 200val投影macro MSE | 楼梯/红衣/商店投影MSE |
|---|---:|---|
| 当前W256 | 4.84153 | 5.24636 / 4.97505 / 4.83517 |
| 当前LN255 | 4.84251 | 5.24740 / 4.97603 / 4.83594 |
| train-only PCA256 | .72084 | .86075 / .71350 / .71105 |

**真实人评。** “val_musicreplace_text_oracle_neutral_v1是舞蹈,其他几个在【先复审教师舞蹈上界】中的配置都是摆pose. 教师空间·pca256是唯一像舞蹈的,但是存在一些肢体反转等的问题,其他的教师空间·w256和教师空间·ln256还是类似摆pose”。用户“ln256”即实际rank255组，不虚增新模型。PCA还有液化/模糊/反转的质量问题，不称通过。

**结论→下一轮。** 真teacher先投影仍失败，给当前已学空间整体提供较强瓶颈证据，排除了该诊断中的音乐预测误差/词长变化。PCA中心和方向都改，不纯旋转单变量；同秩部分跳舞不证明任意256充分/不足，更不等于PCA音乐学生成功。下一轮在noTC学生比较固定W/PCA与NCE开关，并另外检查具体LN头幅度范围。

证据：[投影数值](../results/ms_20261020/music_text_projection_C300_v2_numeric.json)、[12媒体](../results/ms_20261020/music_text_projection_C300_v2_all12_audit.json)、[独立来源审计](../results/ms_20261020/music_text_projection_C300_v2_independent_audit.json)、[人评原话及分析](26_music_h_diagnosis_and_solution_plan_20261020.md#2026-10-20-用户连续评审教师能跳当前输出空间与学生丢失舞蹈)。

### 7.7 R5：noTC固定W/PCA × NCE1/0，8样本100更新

**教师与学生不对称。** 教师仍旧图＋舞种＋中性MS的图文Y；学生仅音乐，彻底不加载TC/processor/tokenizer、P0、prompt/negative=None，图只下游视觉VAE。C300的context weight移除、bias改为可训global_context_bias，None输入与旧zero-context输出exact；不是凭空恢复同图teacher所有信息。

**四实际模型。** 每组从C300完整SHA189a0a74…独立转换、新AdamW，W/PCA固定center/basis不同导致初始条件不同；同空间NCE0/1同init/顺序/预算，不是NCE0从NCE1续训。dense316,343,296/400张量＋剩余adapter3,655,937/41张量全部更新；width256/heads8、两cross-attention/FFN4width、qmax1423。未运行512/1024新宽度训练。

**实际头与loss。** 音乐→LN→可训Linear256→5120得u，再固定`S=μ+BBᵀ(u−μ)`；不是自由PCA系数头、没有删LN或把幅度重新初始化成teacher。8train/8val取旧split首8，C300已在原1800训过，不称从零/新测试集。全部1800train统计/PCA仍被利用；每步8ID、micro2×4，100步每ID曝光100次。

`L=L_H+.01L_len+λ_NCE min(step/100,1)L_seq`，NCE权重1/0；NCE0仍跑三音乐分支计算score，仅系数0，不等于R6单分支。seed42、bone1e-5/adapter1e-4、AdamW decay.01/clip1、无scheduler、FP32/eval-BN有梯度；不加载H3训练、无FM/行为/pose/beat/DPO loss。

| 同8train/8val | W NCE1 | W NCE0 | PCA NCE1 | PCA NCE0 |
|---|---:|---:|---:|---:|
| train raw macro MSE | 5.45656 | 5.33725 | 5.27885 | 5.16737 |
| val raw macro MSE | 5.24840 | 5.18988 | 5.14308 | 5.13436 |
| val词长MAE | 110.625 | 177.625 | 107.625 | 125.375 |
| train三候选top1 | 87.5% | 75% | 100% | 100% |

**生成与判决。** 四模型各原/换/zero9＝36视频，NCE1/NCE0各18独立全解码，noTC、原视觉VAE、native同静音审计通过。用户对新NCE0反事实原话：“所有配置都像是在摆pose而非舞蹈”；NCE1和NCE0所评预设均失败，没有逐模型/逐case排名。换曲仍改内容＋词长＋packed，非固定布局语义干预。

**LN闭球下界。** 固定当前头，把LN内部零均值z放松至`||z||≤sqrt256=16`，FP64 SVD/trust-region/KKT算最近输出，0更新。裸PCA teacher投影三case MSE .71–.86，当前PCA NCE1具体头下界5.18790/4.94552/4.79114，NCE0为5.10820/4.85327/4.71674；无范围约束包络约.72–.89。方向好不等于当前W/γ/中心幅度覆盖；这是固定快照的放松下界，不证明LN永久不能学到。

**结论→下一轮。** NCE0略降MSE，未恢复舞蹈；训练集拟合也差，只有8样本限制结论但不能简单解释为数据少。移除TC不等于移除图VAE；teacher含图上下文和具体头范围都是嫌疑。R6改纯文字teacher、自由输出/尺度初始化与全部1800配对，整体候选配方变化须明示。

证据：[W100完整protocol](../results/ms_20261020/music_text_no_tc_W_fit8_s100_v2_summary.json)、[NCE1媒体18](../results/ms_20261020/music_text_no_tc_fit8_all18_audit_v1.json)、[NCE0媒体18](../results/ms_20261020/music_text_no_tc_nce0_fit8_all18_audit_v1.json)、[LN头范围](../results/ms_20261020/music_text_no_tc_head_range_v1.json)、[notes/26详细协议](26_music_h_diagnosis_and_solution_plan_20261020.md#nce0最终人工负评与本轮详细执行协议)。

### 7.8 R6：纯文字教师1800，public512Identity与control256LN

**目标与真实缓存。** 仅`Dance style＋中性原MS`经冻结50层LM/final norm Identity，add_special_tokens=False、无图/presentation/音频引用，保存完整逐tokenBF16 Y。准确沿旧1800/200 split，从6200源提取exact2000，不偷换另2k；train最大1423/val1113，overflow0。六shard2000PT、860146总token、776078train token来源/token/SHA审核；CephFS镜像只换介质，同字节SHA，唯一消费v2 bundle，误重复v1保留不使用。

**纯teacher上界。** 新纯教师3条无TC/noP＋独立图VAE/native静音视频技术完整；用户确认三个教师配置均跳舞，新pure与原neutral动作几乎一致，但幅度小/跟拍弱。此结论只限该静音/MS/noTC协议，不是否定早期复杂prompt＋真native原始H3。

**两个起点/参数。** public用公开MusicFM＋新width512/末端Identity/globalbias adapter10,713,601参数39张量；control用C300派生width256/LN/globalbias adapter3,655,937/41张量；共同dense316,343,296/400张量。无固定W/PCA后投影。public用train-only全部teacher mean/energy＋前32train音乐query做头均值/全局RMS初始化，scale149.94115、RMS.071831→10.770478，未匹配协方差或动作语义。起点/宽度/LN/初始化同时不同，不把结果只归因Identity。

**训练与恢复真实性。** 各10步真实smoke后从原init fresh正式训练，不继承smoke模型或moments。各1800train/200val、1200更新/batch8/micro2/accum4/9600曝光、seed42/bone1e-5/adapter1e-4、AdamW decay.01/clip1。`L=L_H+.01L_len`，NCE0且完全取消wrong分支/NCE计算，正音乐仅一次前向；反事实只验证/视频。最后模型checkpoint不保存Adam状态，后来若从last加步数需标optimizer重置，不能冒称无缝续训。

| 完整200val | public512Identity | control256LN |
|---|---:|---:|
| raw tokenweighted MSE | 90.70603 | 117.64113 |
| raw macro MSE | 128.17189 | 167.78935 |
| 音乐三候选top1，仅评估 | .325 | .360 |
| 词长MAE | 171.22 | 157.44 |
| train位置均值macro MSE | 5.83669 | 5.83669 |

public初始tokenweighted227.25325→90.70603，control119.20485→117.64113；训练仅初末同集合val＋随机batch日志，没有多中间固定val判断收敛。8条train probe不是全train；约5.33遍完成预算，不能以1200步称充分训练。输出尺度初始化成功不等于实际逐token教师接近。

**正式18视频与音轨。** 同两个末权重各原/换/zero9，noTC/P0，图只独立VAE，真实音乐仅hidden分支、native两路静音。18/18媒体/完整checkpoint/词长/图VAE/nativehash通过；另AAC packet重编码比较18/18相同，独立确认原/换/zero控制配乐来源（zero仍mux原曲）。5.0625s音频与124@24=5.16667s视频的.10417s差单列，未证明跟拍/时间戳一致。

**必须保留的真实正/负评。** “新纯监督·正式1800的几个视频都在跳舞,但是动作幅度分case,case2不是很大,case1和3还行,但是都不太跟拍;【新纯监督·换音乐】还是比较接近摆pose”。这是预设级反馈，未单列public/control/zero胜负；不能说hidden路线从未有舞蹈，也不能称原曲舞蹈已证明稳健音乐编排/身份通过。

**结论→下一轮。** 部分舞蹈恢复、换曲泛化与跟拍失败。用户明确只替换文本向量、不要额外改原条件，词长query不是H3必需、1800/1200或不足。于是R7恢复同图cached P与原native真音乐，取消query/LN/词长头，扩量6000/500并完整多轮验证；不能把R6隔离协议包装成“原生只替文字”的单变量实验。

证据：[真实训练独立审计](../results/ms_20261020/music_text_pure_training_independent_audit_v2.json)、[纯teacher人评](../results/ms_20261020/music_text_pure_teacher_human_review_20261021.json)、[学生18媒体](../results/ms_20261020/music_text_pure_students_all18_audit_v3.json)、[AAC来源审核](../results/ms_20261020/music_text_pure_students_audio_mux_audit_v1.json)、[学生正负人评](../results/ms_20261020/music_text_pure_students_human_review_20261021.json)。

### 7.9 R7：direct6000，cachedP/真native/global64/28轮

**教师与部署边界。** 教师按用户批准仍为`Dance style: {genre}.`＋完整中性MS（genre舞蹈风格），不混无法追溯生成器的rich。每图离线TC得到完整[P;y]，y是整个用户prompt后缀；P含Qwen视觉/引用hidden及原tags。正式部署不加载TC/tokenizer/processor，学生不读P/context/caption/教师词数，图先离线准备P，不跨图复用。

```text
waveform → 在线MusicFM [B,2,T,1024]
         → 声道优先展开[B,2T,1024]
         → 单Linear1024→5120的raw S
cached P/Ptags + mask gather的有效S → 原冻结AV DiT
独立图VAE + 真实native reference音乐/audioVAE → 同原生joint采样
```

通常127 feature时刻×stereo＝254有效音乐token；mask holes按有效行gather，不能截前L。无qmax、PCA、32D桥、query、length head、额外CLIP head。dense316,343,296/400张量＋Linear5,248,000/2张量全部联训，FP32/eval固定BN有梯度；H3全冻结、不进表示训练/无FM/beat/pose/behavior loss。

**数据与完整来源。** 新6500唯一来源组、6000train/500val，排旧200val及固定3case关联源/曲、exact/近似音频/中性caption组。选择v1因错误把source duration当自然81@16窗口而不足6500，保失败报告；修字段语义不降pose/质量门槛后v2从11908pool得到10979独立代表，再选6500。六shard1084/1084/1083/1083/1083/1083，每train1000；全部19500 P/y/audio文件完整字节SHA，实际LM/caption-independent P/raw拆拼，2498122train有效teacher token。fullmanifest SHA40df4ff5…；缓存成功与来源隔离不等于所有动作事实/时序监督质量通过。

**优化与loss。** 两组同零更新init SHAfa7e5ee1…，seed42、4rank×local16/global64，最后尾批48（每rank12）、94更新/完整epoch、不drop或重复；每epoch覆盖6000，28轮168000正曝光。骨干lr1e-5/Linear1e-4、AdamW decay.01/clip1、temperature.07；真实global可微gather和source-group假负例mask，累积不扩大负例池。

令`μ_i=mean_valid(S_i)`，`r_i=sqrt(mean_valid,channel[(S_i−μ_i)²])`，教师对应`μY_i,rY_i`；固定train-only mean/std后L2得`u_i,v_i`。`L_batch=.5(CE(u vᵀ/.07)+CE(v uᵀ/.07))`；A仅此，B另加`mean_i[mean_d((μ_i−μY_i)/std_d)² + log((r_i+ε)/(rY_i+ε))²]`，ε1e-6，两矩权重各1。不是逐token回归、逐channel logstd、全batch RMS或soft-Chamfer；部署S不标准化。

**日志失败与Adam实证。** 首正式epoch完成94步/val500后rank0重复event键崩溃，发生模型/Adam保存前，无可resume状态；修日志在新restart_v3目录从同init完整重训，不伪装接续那94步。最终每epoch存last/best及402 Adam状态、4rank Python/Torch/CUDA RNG/sampler/scheduler/history，独立读取齐全。真实训练可恢复状态与CPU生命周期下一步复现分别报告，不把CPUfixture称实际GPU下一步bit-exact。

**停止规则及实际结果。** 最少20/cap50、相对改善.005/patience8；第一平台LR×.1，第二平台且有限/非塌缩、R@1不低于锚值−.01才数值停。两组实际epoch20骨干/Linear降至1e-6/1e-5，epoch28均2632步正常exit0，reason=`converged_representation_not_video_quality`。低R@1时绝对−.01护栏很弱，不称泛化通过。

| 项目 | A纯batch CLIP | B＋mean/RMS |
|---|---:|---:|
| epoch28 train CLIP | .118141 | .290612 |
| epoch28 val500 CLIP | 7.201042 | 7.327626 |
| epoch28 val R@1 | .007 | .008 |
| 固定train probe64 R@1 | 1.0 | 1.0 |
| epoch28 last rawRMS/teacher | .223623 | .990551 |
| 保存/部署best epoch | 3 | 2 |
| best val CLIP | 5.761932 | 5.893288 |
| 部署best rawRMS/teacher | .187144 | .974624 |

B第3轮实际最低5.880318但改善不足.5%，按既定规则best仍第2轮，不写严格全历史argmin。A部署SHA d4760eb6…、B014dbe05…均best，不是last；不能用last尺度直接归因best视频。A train3.47866→.118141且固定val5.76193→7.20104，降LR后8轮未恢复，支持train/val分化，不支持盲目原配方无限续训。

**18视频、反事实与人评。** 两模型×joint/aswap/zeroM×固定3case，全124@24/704×1280、seed42/CFG1/shifts12/3、图P7437/tags、原视觉与音频VAE、实际AV各50scheduler更新、生成音轨均审计。aswap同时换学生波形和native reference真音乐，不是M-only因果；zeroM准确置0学生输入但native仍原真音乐。ffmpeg/decord完整所选音视频流解码，ffprobe缺失下不冒精确声明stream总数已核。

用户：“还是和之前一样,接近摆pose”。这是R7新6000学生整体定性负评，未单独指定A/B/joint/aswap/zeroM或18条排名，未单列节拍/人物保真；聊天反馈已构成人评完成，不因网页0数字行再等待。B尺度校正成立，但舞蹈目标未达成；full缓存、full联训、完整部署、保P/真native均成立，不能回退成“忘加载/丢P/native静音”的工程故事。

**结论→下一轮。** 技术和研究负结果完整交付，现global表示监督到冻结生成器的缺口仍在，不永久否定MusicFM。先0更新ABC教师诊断澄清P/teacher选项，再转可读离散caption与原生H3文本接口；新caption正式成功须独立验收。

证据：[6500缓存](../results/ms_20261020/music_direct_full6500_native_cache_v1.json)、[完整manifest](../results/ms_20261020/music_direct_training_manifest_complete_v2.json)、[首epoch失败/fresh恢复](../results/ms_20261020/music_direct_epoch_log_failure_and_recovery_v3.json)、[cached教师恒等GPU](../results/ms_20261020/music_direct_cached_teacher_identity_gpu_all3_v1.json)、[18技术/来源/人评总报告](../results/ms_20261020/music_direct6k_v2_final18_technical_v1.json)。原始全审计位于`MS/music_direct6k_v2_final18_queue_v3/finished.json`及`MS/music_direct6k_v2_cross_configuration_audit_v1.json`；仓库总报告绑定其SHA。实现：[music_text_contrastive.py](../scripts_omnidance/exp/ms/music_text_contrastive.py)、[music_text_adapter.py](../scripts_omnidance/exp/ms/music_text_adapter.py)、[music_text_infer.py](../scripts_omnidance/exp/ms/music_text_infer.py)。

### 7.10 ABC教师单case：P移除与纯文/图文Y，不是新学生训练

2026-10-22仅授权旧固定`wJPptuhclZmKAEWs`一个case；A复用P+y图文教师恒等，B新同图文y但无P，C新同prompt纯文y无P，2新＋1复用，optimizer0。三个同caption SHA274e7a7a…、228 source token IDs；A P7437/总7665，B/C228，去P改变packed起点，不称原位置逐值不变。

同原图/真native音乐、独立图与audioVAE、seed42/CFG1/shift12/3、124@24/704×1280、AV各50/保生成音轨/noTC；B-only准入曾误计CPU manager为GPU worker，恢复仅修准入计数，C不重跑。三媒体/真实来源/HTTP Range审核通过。

图文y raw RMS3.82674，纯文14.92790，差RMS14.30449/相对3.7380，token平均cos.95935；它们并不数值等价。用户：“看上去不同的H3语言条件好像结果差别没有很大,尤其是b和c几乎一致”。单case视觉近似，不证明所有prompt无P无损、舞蹈/节拍/身份通过或soft-input训练成功。

**作用与边界。** 仅给教师条件选择线索，不能从两编码视频相近证明音乐预测同分布。LM输入soft embedding路线需同一冻结LM可微前向、正确输入层级/位置，cached最终P不能当input embedding或每层KV；当时只是设计。当前离散caption路线另用新case/真纯视频协议，不能将其正评当本ABC或R7的单变量修复。

证据：[ABC技术/真实反馈](../results/ms_20261020/teacher_prefix_abc_case1_v1.json)，实现：[music_teacher_prefix_ablation.py](../scripts_omnidance/exp/ms/music_teacher_prefix_ablation.py)。

## 8. P、y、embedding与生成接口的澄清

参考图有两条独立路径：

1. 图像VAE→ref_visual_anchor→DiT视觉参考。
2. Qwen-VL视觉网络＋文字embedding→语言模型→最终[P;y]。

P不是图像VAE latent；y虽对应prompt位置，编码时可读取此前图像上下文。tokenizer输出整数IDs[L]；embedding查表才是[L,5120]输入；LM输出hidden是另一层级。

旧direct替换的是LM输出y，音乐rawhidden不再经过LM。将音乐soft embedding拼到LM输入是另一套训练/部署计算图，不能把最终cachedP当原输入embedding或每层KV，也不能冻结LM后no_grad切断音乐梯度。

ABC仅一个旧固定case：A原[P;y]复用，B同图文y无P，C同文字prompt纯文y无P。同图/native音乐/seed/AV50。用户认为差异不大，B/C几乎一致；仅说明该case视觉近似，不证明所有prompt无P无损、两个hidden逐值相同或新soft-input已训练。

当前新融合case FqlHJVDlFGLKCUJS不是旧三人物case。不同case、caption和audio协议不混作单变量因果证据。

## 9. 失败分析：数学可证性质、实测支持与尚未识别的因果

### 9.1 可证的监督盲区：全局pooling不识别最终token排列

R7 DiT接收完整`cached P＋S`，但目标只看每sample有效S的5120维均值及中心化标量RMS。设有效token数n、通道d=5120、行向量S_t，`μ=Σ_t S_t/n`，`r²=Σ_t||S_t−μ||²/(nd)`；对任意有效行置换π，`μ(πS)=μ(S)`、`r(πS)=r(S)`，故A的batch CLIP与B的全部mean/RMS目标均不变。交换有效行而保mask定义即可，不需要交换不同长度/padding。

更精确地，`S_t=W f_t+b`共享Linear满足`μ(S)=W μ(f)+b`。固定n/mask下，若loss仅通过μ，`∂L/∂S_t=(1/n)∂L/∂μ`：每行接到相同的全局读出梯度；经末Linear给f_t的梯度也同一方向`Wᵀg/n`。这不代表bone内部完全不学时序（f_t已上下文化、不同位置Jacobian不同），而是没有直接规定“第t行承担哪一动作/转场/拍点”。

B的尺度项梯度另沿各行中心化偏差方向，主要缩放/拉伸当前分布；只看`trace(Cov(S))/d`的r不能辨协方差方向/跨token关系。同r、同均值可有大量不同条件，匹配尺度不等于恢复语言hidden分布。A的L2 cosine还有正尺度不敏感性：`normalize(αx)=normalize(x)`（α>0），在标准化/中心坐标下可允许尺度偏小；实测A best尺度小、B近teacher，但并未隔离尺度对视频的因果。

**严格结论。** 已证明R7目标对最终有效行排列盲，不是证明教师LM的上下文化mean绝无顺序信息；也不是证明H3对每一种行置换都敏感，更不是pose唯一原因。文本refiner自身的置换性质及后续packed位置需要一起看；“loss没时序监督”与“生成器必因词序置换失败”不能画等号。

源码位置：[music_text_contrastive.py:859](../scripts_omnidance/exp/ms/music_text_contrastive.py) `_direct_sample_statistics`、同文件:870–896矩/CLIP；[music_text_adapter.py:403](../scripts_omnidance/exp/ms/music_text_adapter.py) direct wrapper。当前源码可定位，不把磁盘最新版SHA冒充每轮历史执行版本。

### 9.2 旧soft-Chamfer不是逐词/逐拍一一对应

R1–R5旧序列score先取最多64均匀有效tokens，以标准化/L2后的pairwise cosine矩阵c_ij作双向soft-Chamfer：每行`τ(logsumexp_j(c_ij/τ)−log nY)`，每列反向同式，两侧平均，τ=.1；再候选CE温度.07。它用logsumexp平滑最大值，不是softmax概率直接平均。固定所选token集合时，任一侧重排行只重排矩阵行/列，不改变聚合；不约束词顺序、没有token唯一匹配或时间单调性。若先均匀抽样再置换全序列，所选集合可变，因此只能将“匹配函数对输入集合无序”写准确，不把抽样实现本身称任意全序列置换都严格不变。

很多学生行可匹配同一教师关键词，覆盖不足也可有较高score；teacher词位置包含LM上下文，仍不能按索引硬对25Hz声学时间。最新R7换成mean batch-CLIP并非已经解决这些问题，而是进一步只约束全局读出；R6实际NCE0，不应写“高CLIP训练成绩却没舞蹈”，其音乐score仅评估。

若用OT/Sinkhorn改局部覆盖，双边边缘质量约束可减少所有行匹配一个词的退化，但无序OT本身仍不是时序GT；没有真实窗口/动作事件时间，禁止对角或伪单调词序标签。该统一方案曾讨论，未执行正式新hidden学生训练，不拿讨论代替证据。

### 9.3 raw MSE、mean MSE与inverse-length：能说明什么

对固定teacher长度n及相同mask，`MSE(S,Y)=||μS−μY||²/d + Σ_t||(S_t−μS)−(Y_t−μY)||²/(nd)`。均值误差只是完整误差的一部分；mean MSE很低可同时有很大中心化结构误差，所以mean拟合/尺度不能证明逐token语言条件已恢复。R7 mean与RMS目标也没有最小化后一项；teacher/student词数不同，不能直接套同n逐位置分解当已有对齐。

R0–R6的loss/macro各sample通常含1/n归一化，单token梯度对应`2(S_t−Y_t)/(nd)`；长caption某个决定性动作词的局部误差可被平均稀释，这是inverse-length权重的数学性质，不是观测到所有长caption失败的证明。tokenweighted指标则按有效token总数加权，长caption占比更大，与macro不等价。归档必须注明口径：R0 tokenweighted基线4.756748、R2 macro4.81506、R6纯教师macro5.83669不是同一个目标或同一种平均。

条件分布若同音频对应多个有效caption，对平方损失无限容量风险最小预测是条件均值`E[Y_j|music,可见context]`（需先定义词位置/长度随机变量）；这个均值不保证落在H3的可用条件分布上。词序/措辞可变造成的不可辨目标、图上下文不可见部分、有限模型空间与优化不足均可产生均值化，不能仅凭baseline胜学生就证明某一个唯一原因。

**GT词长泄漏与长度混杂分开。** 教师有效mask/真logL作为loss标签是监督，不是部署输入泄漏；R2/R5/R6共用student_condition无caption/GTlength，R7完全没有词长头/只音乐mask。R0/R1曾用teacher长度设训练query_count，属于明确train/deploy计算差别；不能否认这段历史，但后轮修正后仍pose，不支持“现在GTlength偷偷用于生成”解释。

共同可见query评分0更新结果让A/B/C300第一wrong胜率不变，排除评分取样长度差足以解释弱判别；没有排除视频packed布局因素。自然生成换/零音乐同时改S与自己词长，不能把RGB差当纯内容因果。R7固定5.0625s输出通常254行，取消预测词长，却仍相对teacher词数228/199/323改变布局；接口合法不等于完整条件一致。

证据：[共同区间](../results/ms_20261017/music_text_online_s300_common_interval_diagnostic_v2.json)、[R6实际输入/词长/全200审计](../results/ms_20261020/music_text_pure_training_independent_audit_v2.json)。

### 9.4 具体可达空间：旧256/LN证据不能偷换成新1024必败

对固定末端`Linear(k,5120)`的W/b，每token位于`b+col(W)`，rank≤k；骨干再非线性不解除这个固定末端仿射包络。旧width256有rank≤256，R6 public512有rank≤512，R7直接1024有rank≤1024；5120输出shape不能当自由5120空间，rank限制也不等于token数限制。

旧LN头更窄：记内部归一化z∈1⊥、`||z||<sqrt(k)`（ε使边界受限），`h=b+Wβ+Wdiag(γ)z`。其仿射方向rank≤k−1；当前旧实测LN255。加固定PCA投影后`h=μ+P[b+Wβ−μ+Wdiag(γ)z]`，好basis仍受实际W/γ/b/μ幅度中心约束。放松成闭球给下界，不把包络最优值说成音乐上游可达到值。

R4用真teacher投影排除音乐预测误差、保持teacher长度/P/packed，却旧W/LN仍pose，PCA同秩部分舞蹈且反转；R5 PCA具体头closed-ball误差4.72–5.19，而裸teacher PCA投影.71–.86，是**这些固定checkpoint的坏空间/范围证据**。W/PCA中心与方向同时改，不只旋转；输出weight/γ可训，不能推出LN永远不可学或256永远不足。

R7换1024共享Linear未做同样“真实teacher投影到当前1024空间＋同布局视频”诊断。因此“rank≤1024”是可证结构限制，“该1024空间导致pose”仍假设；不能把旧256的原因迁移成最新唯一cause。R6Identity/global尺度校准也改变多个因素，原曲有舞蹈但换曲弱，只支持候选配方整体的部分结果。

证据：[投影](../results/ms_20261020/music_text_projection_C300_v2_numeric.json)、[LN闭球](../results/ms_20261020/music_text_no_tc_head_range_v1.json)。

### 9.5 强实测：训练/验证分化与“预算完成≠泛化收敛”

R7 A epoch3/16/20/28固定val500 CLIP为5.76193/7.18086/7.17594/7.20104，epoch train为3.47866/.324705/.193995/.118141。epoch20降LR×.1后八完整epoch未恢复，B亦末val7.32763而早期best5.89329。固定probe64R@1末1.0、val500约.007/.008支持有限probe/留出分化，不能称全6000train全中，也不能把不同候选数R@1直接当独立显著性检验。

数值patience规则按完整val平台结束，reason已有`not_video_quality`；它是操作性停止，不是有效语义泛化。-.01绝对检索护栏在R@1<.01时近乎不限制，必须保留这个弱点，不因converged=true忽略负证据。验证unique-caption检索弱也非数学上等价所有音乐语义消失，最终视频失败来自真实人评而非检索替代。

R2最多2400曝光/R3六曝光/R5八样本/R6 9600曝光约5.33遍，不足以证明各方案永久失效或已经充分收敛；R6public初末MSE下降且原曲舞蹈反例必须保留。相比之下R7 28轮同集退化不支持自动原配方无限加步；若换更合适监督/来源数据，必须新协议/预算，不把“停止旧同配方”写成“任何长训无效”。

**checkpoint因果边界。** R7视频来自A epoch3/B epoch2的best；B best/.974624与last/.990551尺度不同，A best/.187144与last/.223623也不同。B第3轮微改善未过.5%保存门槛，best不是严格argmin；用户没有A/B赢家评价，禁止按last数字给best视频排名。

证据：[最终训练/媒体/人评](../results/ms_20261020/music_direct6k_v2_final18_technical_v1.json)。

### 9.6 行为监督覆盖/梯度：已知缺口，不是永久否定

R3原37latent单次预测覆盖完整视频网格，但仅3train、固定sigma.9/noise42各访问两次，不是50步自由生成轨迹或37次更新。一个val点prediction相对control −3.516%、相对init −.960%仅微弱局部信号；学生长度/packed也变，不能称稳健泛化。

三个已发生更新前监督点分项梯度中bone NCE比prediction norm大约1320/470/36600，content adapter行为梯度有一定幅度且与NCE近正交；不同点checkpoint不同，不冒称同模型全train统计。分项求和相对误差均≤1e-4，真实总梯度对应原训练；global clip统一缩总梯度、不独立旋转/只压behavior，norm比也不等于Adam实际更新份额。没有复制原moments作虚拟Adam一步，不能说已证明NCE“吃掉”全部行为更新。

全长两卡精确VJP/bone400/content38有梯度、length4仅L_len梯度是工程证据；旧六步pose不永久否定multi-sigma/多噪声/更适合轨迹行为loss。最新R7没有使用该行为目标；讨论中的真实native新behavior/local/motion接口也未正式落地，不能把旧静音VJP当新真native方案已验。

证据：[分项梯度1](../results/ms_20261020/music_text_prediction_components_C300_v1.json)、[2](../results/ms_20261020/music_text_prediction_components_step2_v1.json)、[3](../results/ms_20261020/music_text_prediction_components_step3_v1.json)、[六次全长VJP](../results/ms_20261017/music_text_behavior_s6_v1_all_training_vjp_audit.json)。

### 9.7 条件协议变化与教师上界：noTC不等于noVAE

必须把各轮生成器的实际条件写在原因前。R0/R1保TC图P/context，但native真音乐被固定静音替代；R5/R6关闭整个TC/Qwen视觉LM与P、保独立图像VAE，又仍原audioVAE编码固定静音；R7按图cached P、原图/audioVAE和真native joint恢复。`noTC＋fixed P`（离线cached P）不是`noVAE`，`fixed_native_silence`也不是删除audio rows/音频分支。

早期E3富复杂prompt有身份/衣着/场景/音频引用，真native音乐、完整AV采样，上界获过用户认可；旧MS/noTC/静音teacher跟拍弱不能泛化成“H3天生不会跟拍”。R4neutral和R6纯teacher都获舞蹈正评，R6学生原曲也获舞蹈正评；H3根本不会跳/hidden替换原则不成立均不合事实。若完整数值条件/长度/tags/packed/图native/noise/采样一致，冻结模型确定性输出应同，identity oracle实际支持此接口原则。

图文y含上下文，学生仅音乐可能无法推出图/措辞；noTC删图语义路径可能影响衣着，但服装漂移也发生在保P的旧学生，不能单独归因删TC。ABC同case去P/纯文y视觉近似仅一条受限反例，不证明全数据P无用、hidden相同或同维/高cos/RMS等价。音轨可播放/全解码只是媒体与来源，不是同步/动作准确。

### 9.8 待验证：多解配对、事实噪声、局部节拍监督缺项

音乐可支持多编舞，不能唯一指定左右肘/手靠哪肩/词序/措辞；同曲多舞、跨曲相似编排可使唯一配对NCE含语义假负例，rawMSE趋平均。R7同来源组mask处理可靠同源，不能自动消除所有跨曲语义近似；旧不同ID/caption hash负例也不是完整歌曲隔离。重复用val200选多配方有选择偏差，不能把8组同200val叫1600独立样本。

rich原生成器/指令未知、MS句序/相对秒数未核，VLM energy/turn与kin可冲突；来源/SHA/窗口自然81帧通过不等于所有动作事实真实。必须保留unknown/conflict并真实审核，不通过删不方便的字段伪装监督质量已通过。

音乐25Hz/stereo的2T与语言词数L不同是合法条件接口，不是维度错误；词索引并非动作时刻。现raw MSE/soft-Chamfer/mean CLIP没有显式beat/onset→动作强调/过渡约束，teacher也未保证强跟拍。即使完美复现弱跟拍teacher，只能期待相似teacher行为，不能自动超过上界。禁止把MS“随后/先后”或均分词号当真实秒GT；beat检测和可微motion目标若要加需独立数据、链路与资源验收。

### 9.9 因果排除表与为何转离散caption

| 过强归因 | 已核反例/当前允许结论 |
|---|---|
| 音乐bone没训/adapter最后层没部署 | R2 B/C、R3/R5/R6/R7完整dense更新和部署有实证；不能解释最新失败 |
| 所有阶段丢P/native静音 | R7保同图P/Ptags、真native/双VAE/AV各50；旧隔离阶段与最新分开 |
| 所有hidden路线从未舞蹈/H3不会跳 | R4neutral、R6纯teacher与1800学生原曲有真实舞蹈正评 |
| 256或1024必然不足/尺度唯一原因 | 旧具体坏空间成立；同秩PCA部分舞蹈且新1024未投影因果；B校尺度仍未达目标 |
| 六步证明任何behavior无效 | 只有3固定训练点/1val点，覆盖及实际Adam贡献未定 |
| 最新A/B某组更好、last解释best | 用户仅整体pose负评；媒体用earlybest，不能拼接last统计因果 |
| 最新纯视频caption正评证明R7已单变量修好 | 新case/文字正文/无audio物理分支均变，不能与旧case/真native/音乐hidden混比较 |

新路线的研究动机不是“已证明音乐语义完全不可能”：把音频→可读离散动作caption与caption→H3分开，可直接查事实/措辞/叙事/窗口，避免只能用global相似度猜冻结DiT是否懂raw S；第二段恢复H3原生图文TC消费真实语言，第一段保留音乐条件可解释输出，并独立学习音频到文本的多解关系。

当前新case原始H3真纯视频rich/MS两条可用、融合v7“更好且可用”，给可读caption生成提供正向上界；它不是音乐学生泛化证明。追加维度A/B“几乎完全一致”是该预设弱增量结论，不支持无限堆字段。用户最新选择正式只微调官方30B，且授权助手以AI review直接推进；这改变后续评审执行者/流程，不把AI review冒称用户连续人评，也不抹掉unknown窗口/冲突/SHA来源要求。30B/H3两更新仅接口fixture，正式pilot/full及GT/predcaption端到端仍须按新授权实际落盘、验收，不能写已正式成功。

## 10. 当前两段主线及实际验收

### 10.1 音乐→离散caption

首版仅音频，动作计划与外观分开；Omni保未来图像能力但视觉tower冻结且禁止调用。Thinker文字输出，不创建Talker/Code2Wav。

最初接口候选为Qwen2.5-Omni7B和官方Qwen3-Omni30B-A3B-Instruct。用户最新明确**正式pilot/full只微调官方30B，不再做7B微调对照**。30B官方revision为`26291f793822fb6be9555850f06dfe95f2d7e695`，25文件内容身份已核。历史7B接口和冻结重写结果保留，不再安排7B学生训练。

训练：LLM attention＋MLP LoRA r32/alpha64，audio末4blocks LoRA、proj全训（7B proj，30B proj1/proj2）；30B专家3DParameter用PEFT target_parameters。不把attention-only冒称专家MLP已训。LoRA lr5e-5/proj1e-5，BF16基底/FP32可训与Adam。user/audio/pad label=-100，仅assistant caption＋EOS CE；训练/部署共用processor边界。

**7B接口已通过**：八train/两val fixture，2updates、完整adapter/Adam/RNG/sampler/processor，非正式SFT。

**30B接口已通过**：fresh v4 2updates/16examples/336assistant tokens，NLL4.10893354→3.35096491；第二步LM576/audio48/proj4非零，192专家LoRA张量全部非零。u0/u1/u2内容SHA、628Adamstep2、8rankRNG、sampler/history、生产重载及processor prefixexact、真实posttrain1条文字完成。另原入口epochval1条，不私加预算。生成用了固定接口标签，caption的措辞/动作/音乐质量尚未人评，不能用于正式训练起点或宣称已学编舞。

### 10.2 30B两次OOM及修复

1. 首forward全专家fold384MiBOOM：from_pretrained base默认eval，新PEFT wrapper.training=True但decoder.training=False；GC flag虽true实际不运行。加wrapped.train()。
2. 首Adam成功/完整u1后第二backward768MiBOOM：原Traceback定位grouped_mm_fallback_backward的grad_weight=zeros_like(weight)。新Adam moments常驻＋DDPgrad/bucket双份使峰值不足。
3. 最小DDP gradient_as_bucket_view=True与slot结束释放output/NLL等，保专家/audio/FP32Adam与BF16折叠数学；真实两rankCPU628键两update/保存重载下一步bit-exact，然后官方8GPU v4全部成功。

旧失败partial和完整u1保留；新v4同seed/officialbase fresh，不跨source假resume。CephFS隐藏.checkpoint先写adapter再写10GB训练state的慢写经/proc核实，不能静态无日志就称通信死锁。

### 10.3 caption＋参考图→真纯视频H3

保原图文TC与图像VAE参考，TC/VAE冻结，原BF16 DiT只训LoRA。真正无audioVAE、audio noise/input/reference、tag2/audio_pos、audio patch/head、audio scheduler/decode；零长度(2,32,0)只类型占位，不是有限长度静音。文件1video/0audio，评审外置wav不进入H3。

新124协议：源81@16=5.0625s，原loader下取107只覆盖72源帧，不能冒124监督；显式nearest24_repeat_last124保全81源帧，122自然采样位置＋2尾pad，总124@24=5.166667s。保存完整mapping，padding不新增动作事实。

八条新cache latent[1,24,37,80,44]已真TC/VAE/fullsource decode/SHA审核；8GPU ZeRO3真2updates、softanchor0.1活跃。training/final各28文件marker/8Adamstep2/8RNG/model/scheduler/custom已核。

实际8GPU load_state→0额外更新→独立resave，26项8model＋8Adam＋8RNG＋scheduler＋custom state/tensor精确相等。**next_update_verified=false**，不包装GPU恢复后第三步已验证。源码9f5de5对应历史接口资产；后续新正式gate源码不能重标这些旧cache。

### 10.4 冻结底座prompt正向结果与A/B最新反馈

同case FqlHJVDlFGLKCUJS、同图/seed42/CFG1/shift12/50步/704×1280/124@24，无H3音频。

| 配置 | 文本处理 | 真实反馈 |
|---|---|---|
| rich-motion-only | rich前三章动作/叙事，排外观/镜头 | 与MS“两条都可用” |
| MS-motion-only | 原MS五条动作，保历史未核伪时间边界 | 与rich“两条都可用” |
| 融合v7 | 助手编辑173词，rich质感＋MS具体动作；不是Qwen产物/训练GT | “融合更好且可用” |
| v8-A | v7主体逐字＋arms主导/graceful dreamy | A/B“几乎完全一致” |
| v8-B | v7主体逐字＋upright/in-place/no jumps | A/B“几乎完全一致” |

A/B维度来自原VLM/kin，energy low/medium、turns none/MS转身冲突未加，quality已在v7不重复。两条新媒体各全124帧解码、同模型SHA与图、50DiT/VAEdecode1/TC0/audiohooks0审核。

“几乎一致”限A/B视觉观感，不推pixel相同、A/B与v7也相同或所有维度无效。最新已收到此人评，不再重复请用户填写同项目分数；保v7已接受基线，不自动叠加相同弱增量。

[8774对照入口](https://launcher-ql3qjy14j2-codeserver.ide.taiji.woa.com/proxy/8774/)保旧rich/MS/v7及新v7/A/B预设。

## 11. 目前具体在做什么，什么尚未启动

### 11.1 已结束、不再称“正在跑”的任务

- 官方30B下载/内容验证/本地staging、旧PGLuNL错误rewrite以及修复v6全32生成。
- 7B与30B两更新接口；H3新124cache/两更新/完整GPU恢复接口。
- rich/MS/v7/v8-A/B冻结底座推理和媒体审核。
- 旧6000formal/18学生/ABC教师消融。

### 11.2 本次正在收尾的工程工作

- 更新A/B人评、30B完整接口结果与详细历史文档。
- H3正式100/300来源门禁：在模型加载前核真实fusion export、train-only metadata/sidecar/manifest/review/rewrite/media SHA、严格四项审批及window。
- interface只能2updates，正式cache逐样本绑scope/provenance/train ID/caption SHA，拒翻marker把fixture升级正式；combined/val不得进正式训练cache。
- 当前门禁CPU正反例已实现，正式CLI必须显式`--data_scope reviewed_train`；默认interface仅允许max_train_steps=2，即使合法审批链也不能漏显式scope直接100/300。合成TemporaryDirectory模拟审批链不是真实人评，不实际导出32或上卡。
- 最终正式gate源码SHA为`16c81225b62d77627b739522040128b957fddd4cabfc7f88417a421cf4086b40`；旧接口9f5与初版门禁71284883不是该最终版本。当前新代码只经过CPU模型加载边界/恢复测试，没有再做GPU训练，因此不把旧9f5真实GPU成功迁移成新版本GPU验收。

### 11.3 正式实验仍被真实监督门禁阻断

历史快照：首32在新validator下27ready/5invalid/0人工approved，所有rich/MS源窗仍unknown。**用户最新委托助手执行审核并允许直接推进，改为独立assistant_delegated AI证据审核，不再等待用户逐条POST。** 助手实际观察配对片段、核正文动作覆盖/新增无据事件/冲突，保存证据路径SHA与AI身份，不能把AI判断冒成人工评分、不能把unknown原源窗改成已证实。视频可用评价与目标事实审核仍分开。

获审导出：Omni train/val＋combined omni.jsonl；H3 train-only/val-only分文件，combined仅交换/审计。只有非fixture、统一版本、媒体和来源无变化才能导出。

正式计划（已批准，尚未完成）：

1. Omni512/128，三个完整epoch；再6000/500最少3最多10，每轮固定val500，自回归输出，patience3/delta0.005。
2. H3同init三组pureFM/softanchor0.1/1各100成功更新，固定3cases人评合格再300，再winner全6000；完整阶段3–10epoch的预算/验证停止入口还需落实，不能说现CLI已支持并跑完full。
3. GT/predcaption × frozen/finetuned H3 ×固定3case，最多12视频，分别判断caption、生成、人物清晰与音乐匹配。

[8773真实32审核入口](https://launcher-ql3qjy14j2-codeserver.ide.taiji.woa.com/proxy/8773/)GET不创建评分，四项默认false；助手不代POST。

## 12. 计算资源：实际使用、快照与后续安排

### 12.1 刚结束阶段真正使用的资源

| 任务 | 实际节点/GPU | 完成情况 |
|---|---|---|
| 7B接口 | N2节点内8卡 | 两更新/保存重载完成 |
| 官方30B v2/v3/v4 | N7节点内8卡，失败任务结束后独立fresh | 最终v4完整通过，旧失败保留 |
| H3新124cache | N3 GPU0 | 八条缓存/完整映射审核完成 |
| H3新124训练/严格恢复 | N3节点内8卡 | 两更新＋0额外更新GPU恢复完成 |
| v8两caption推理 | N4 GPU0/1 | 各1case新cache→50步视频，均完成 |
| 目标/门禁/文档 | CPU | 不需要占GPU |

这些是阶段实际工作，不表示同一瞬间所有任务并行，也不能将占卡matmul算训练。

### 12.2 八节点最新只读快照

实际采样时间：**2026-10-22 23:17:39 北京时间**，八节点并发只读SSH。硬件均8×A800-SXM4-80GB，每卡81920MiB。

| 节点 | 地址 | 真正实验GPU | holder GPU | 真空闲GPU | 每卡显存/利用率 | host可用GiB | cgroup当前/剩余GiB |
|---|---|---:|---:|---:|---|---:|---|
| N1 | 28.216.19.69 | 0 | 8运行 | 0 | 975MiB / 100% | 1532.54 | 455.54 / 542.29 |
| N2 | 28.216.17.70 | 0 | 8运行 | 0 | 975MiB / 100% | 605.85 | 409.66 / 588.18 |
| N3 | 28.216.19.213 | 0 | 0 | 8 | 0MiB / 0% | 1662.05 | 494.92 / 502.92 |
| N4 | 28.216.19.143 | 0 | 8运行 | 0 | 975MiB / 100% | 1688.38 | 307.20 / 690.63 |
| N5 | 28.216.19.137 | 0 | 8运行 | 0 | 975MiB / 100% | 1660.33 | 467.55 / 530.28 |
| N6 | 28.216.19.225 | 0 | 8运行 | 0 | 975MiB / 100% | 1653.76 | 489.95 / 507.88 |
| N7 | 28.216.18.144 | 0 | 0 | 8 | 0MiB / 0% | 1573.20 | 649.76 / 348.08 |
| N8 | 28.216.18.136 | 0 | 8停止态 | 0 | 975MiB / 0% | 1566.97 | 555.20 / 442.64 |

总计：**真正模型实验0/64卡；holder48卡（40运行、8停止）；真空闲16卡；未知compute0卡**。N8停止态T仍占CUDA上下文，不算空闲也不算训练。所有计算进程逐PID/UID/完整argv/CUDA→UUID核实为项目hold_gpu.py矩阵乘；100%利用率不是正在训练。

cgroup limit各约997.83GiB；host可用内存可能远大于容器可用，不能用host数字单独准入。N3/N7实际已结束训练/恢复并空闲；本快照没有GPU推理或SFT，不把之前用过的节点仍说成在运行。

64个reserve文件都存在但保护均已过1200秒，active保护0；N1现有watchdog PID4018413，周期600秒，可重新补holder，因此此快照不保证未来空闲。未操作holder、watchdog、reserve或未知任务。

CPU H3自测PID曾23:16:59存在，采样时已退出；独立受管理C2GQhv实际exit0才是成功依据，不用进程消失推断成功。当前继续的内容是文档/发布/正式监督准备，不是活跃GPU训练。

完整逐卡UUID/PID/UID/argv/reserve年龄及cgroup采样保存于`MS/cluster_resource_snapshot_20261022_document_v1.json`；仓库的[资源汇总快照](../results/ms_20261020/cluster_resource_utilization_summary_20261022.json)供复核。

### 12.3 为什么暂不铺满64卡

历史快照中瓶颈已从“接口不能训练”转为“正式监督未审核”。用户最新明确将审核委托助手并允许直接推进，因此**不再等待用户逐条人工确认**；现在由助手以实际媒体与来源证据完成AI审核，再产正式目标并自主营造30B pilot。多上GPU仍不能替代事实核对，但不存在未授予审核权限的阻断。

完成任务的节点应可供用户其它任务，不能以holder100%伪装有效利用。历史watchdog会每600秒检查空闲并补holder；reserve只保护mtime<1200秒。原holder仅保持占卡，不是有用实验；是否解除占卡/预留需明确操作范围，资源快照不会主动kill未知任务或替用户占空卡。

### 12.4 获审后的资源Wave计划，不是当前占用

- 音乐caption学生只训官方30B：优先使用已经验过完整模型/8GPU链路的N7，或按资源准入调整；**取消N1的7B微调对照**。正式启动重新核host/cgroup/UUID/未知进程，不以历史空闲快照准入。N1可作CPU目标审核/数据组织，不保留无意义GPU训练占用。
- N3融合扩写/验证，N4新caption缓存与评测。
- N5 pureFM100、N6 anchor0.1、N7 anchor1，N8固定case评测。
- 若某组需16卡先核NCCL和显存，不默认全64rank一锅启动。未用卡让出，保既有review服务。

## 13. 可复核结果与复现入口

| 内容 | 仓库报告/代码 |
|---|---|
| 历史MusicFM扩量人评 | [music_scale_final_acceptance_v4](../results/ms_20261017/music_scale_final_acceptance_v4.json) |
| rawH在线/行为重算 | [music_h_diagnosis_recomputed_v1](../results/ms_20261020/music_h_diagnosis_recomputed_v1.json) |
| 纯文1800部分正评 | [music_text_pure_students_human_review](../results/ms_20261020/music_text_pure_students_human_review_20261021.json) |
| direct6000最终18技术与人评 | [music_direct6k_v2_final18_technical](../results/ms_20261020/music_direct6k_v2_final18_technical_v1.json) |
| 融合v7技术及正评 | [h3_video_editorial_fusion_case1_audit](../results/ms_20261020/h3_video_editorial_fusion_case1_audit_v7.json) |
| A/B维度技术及近似人评 | [h3_video_dimensions_case1_audit_v8](../results/ms_20261020/h3_video_dimensions_case1_audit_v8.json) |
| 角度误报修复/32原raw未变 | [dance_caption_angle_validation_recheck](../results/ms_20261020/dance_caption_angle_validation_recheck_v7.json) |
| 两阶段获审导出CPU | [dance_caption_two_stage_export_cpu](../results/ms_20261020/dance_caption_two_stage_export_cpu_v8.json) |
| 官方30B完整接口 | [omni30b_official_pretrained_smoke2_verified](../results/ms_20261020/omni30b_official_pretrained_smoke2_verified_v4.json) |
| H3新124训练与真实恢复 | [h3_video_repeatlast124_pretrained_smoke_and_restore](../results/ms_20261020/h3_video_repeatlast124_pretrained_smoke_and_restore_v2.json) |
| H3正式100/300来源门禁CPU | [h3_video_reviewed_training_gate_cpu](../results/ms_20261020/h3_video_reviewed_training_gate_cpu_v3.json) |
| 32实际AI批准/24+8导出及640媒体齐备 | [dance_caption_approved32_and_pilot640_media](../results/ms_20261020/dance_caption_approved32_and_pilot640_media_v3.json) |
| 30B MS＋风格已实际开训、首轮重复负结果 | [omni30b_ms_style_pilot_actual_training](../results/ms_20261020/omni30b_ms_style_pilot_actual_training_v1.json) |
| 融合提取/重写/获审导出 | [dance_caption_fusion.py](../scripts_omnidance/exp/ms/dance_caption_fusion.py) |
| Omni训练/部署/保存恢复 | [omni_dance_sft.py](../scripts_omnidance/exp/ms/omni_dance_sft.py) |
| H3真纯视频执行 | [h3_video_only.py](../scripts_omnidance/exp/ms/h3_video_only.py) |
| H3训练/124窗口/正式gate | [h3_video_sft.py](../scripts_omnidance/exp/ms/h3_video_sft.py) |
| H3新caption缓存/生成/媒体审计 | [h3_video_infer.py](../scripts_omnidance/exp/ms/h3_video_infer.py) |

已完成历史资产和源码指纹：Omni最终接口7c463626…；H3真实124GPU接口9f5de5e2…；fusion导出b53aece3…。新增H3正式门禁修改后需新source/newcache，不能把旧9f5状态的成功重标为新版本真实GPU成功。

## 14. 30B正式实验距离与最新审核授权

用户最新决策：“直接调30b即可,不需要调7b了”；随后明确“我希望你来进行审核,此外,也可以直接进行推进”。后续正式学生只官方30B，审核改由助手执行透明AI证据审核，不冒人工评分；不再等待用户逐条批准，也不重复已通过的7B/30B接口smoke。

### 14.1 已完成的启动前提

- 官方30B完整权重/processor身份已核，N7本地模型已暂存；8GPU真实2update/AdamRNG保存重载/部署文字生成已完整通过。
- 真纯视频H3底座已得到rich/MS/fusion单case正评；H3八条124帧cache/2update/真实GPU恢复接口已通过。
- 30B pilot的**512train/128val来源候选与640条配对媒体**已齐备。首版13条音频不足完整窗口，严格拒绝后仅同split替换13条，627已有帧图复用；有效清单`MS/dance_caption_pilot640_30b_only_manifest_v2.json`，SHA`2f1c4d2ddc5b299f0a86161a59635c719b3f5b84646c49d9e78a484f4481573e`，标准visual index SHA`06d4f36a6116cde5cfc5c4ccfbbd42144409f244718ca3c254a752531ed00d68`。仍caption_targets_generated=false：媒体齐备不是640条正式标签。
- 首32真实AI审核与导出已完成：19条保留原30B正文、13条透明助手编辑；32批准review SHA`7de27bda7bd2c39e1e03b79cf16f444b7cf31200620a6881f1d6471c5051c446`，导出24train/8val，实际Omni reader/H3正式来源链通过。AI批准不冒human，不把unknown原rich/MS源窗改verified。
- Omni三epochpilot预算min_epochs=max_epochs=3、smoke_steps=0，沿正式数据/官方base fresh，不从接口fixture LoRA接训。2026-10-23发现并补齐旧Omni reader的真实审批链/fixture/精确规模缺口：最终源码`4186fddf…`显式`--data-scope reviewed_pilot`，模型加载/out创建前核512/128、统一版本、真实审批链和完整导出内容，审批身份进入resume fingerprint。37项CPU数据契约和原架构完整恢复回归已独立通过；真实24/8导出可读但正式pilot正确拒。不宣称已获640正式审批或新gate已GPU训练。

### 14.2 还缺哪些实际交付

| 剩余交付 | 为何需要 | 完成后的动作 |
|---|---|---|
| 32条实际媒体AI观察 | **已完成**32片段/384帧图，89条保守OBS事实 | 旧正文1粗粒度可留、26需重写、5拒当前目标；媒体无缺失，原rich/MS源窗仍unknown |
| 首32实际正文修正与目标审批 | **已完成**独立AI核帧/正文、13编辑/19保留、32透明AI批准、真实24/8导出/来源链 | 不再等待用户逐条POST；原raw、错误事实和编辑来源分别保存，不能升级为512formal |
| 官方30B图像依据候选生成 | **已结束**32严格JSON/实际32visualcalls/0audio，但旧80–180词标准0valid，三case另有真实动作顺序/幻觉错误 | 保失败原稿，视觉重试暂停，不再阻挡现有MS＋风格弱监督训练 |
| 512/128正式caption目标及逐项来源/质量检查 | 640媒体/索引已齐，caption尚未生成和逐条审批 | 实际核图/动作順序/正文冲突，透明派生兼容target，导出统一版本omni.jsonl及H3 train-only/val-only |
| Omni正式来源门禁 | **已完成**显式scope/数量/预算、复用真实审批链、逐字段导出核对、审批身份恢复签名；37项CPU契约与原架构恢复独立通过 | 真实24/8helper链通过但formalpilot拒；新gate尚未正式GPU训练，不迁移旧smoke结论 |
| 最后8GPU/host-cgroup准入与数据签名冻结 | 旧资源快照可能被watchdog补占卡 | 直接启动官方30B三完整epochpilot，不再另加smoke |

不是“还需要模型下载、再验证7B、再验证音频关闭、再重训旧6000”。当前关键路径是**把候选转成真实可训练监督**，这项不能以一个批准布尔或格式通过代替。

### 14.3 最新推进方式：先用简单MS＋风格训练

2026-10-23用户明确严格融合数据处理持续过久不可接受，并授权“可以快速选用简单的motionscript+风格数据先开始实验”。当前优先**官方30B音乐→原MotionScript＋原风格的512/128三epoch弱监督基线**，不再将rich融合/视觉候选逐条审批作为该独立实验前置。新scope `source_label_pilot`明确label_quality=source_labels_unreviewed、human_review_completed=false、semantic_approval=false、原窗口映射unknown；核原MS/genre/音频与配对素材SHA、固定split、去人物代词/合成秒词和实际预算，不假审过标签，也不把fixture两更新升级为正式训练。

该简化基线已实际fresh启动N7八卡：manager3198080/torchrun3198407、2026-10-23 07:55:54北京时间，生产源码4d82caff…冻结、数据SHAab2550fd…、smoke0，未复用接口权重。已80次更新且第一轮固定val NLL0.883553；三条自回归输出均长重复，明确不能以NLL宣布可用。保原三epoch继续，原6000/500标签转换同时进行，不要求512质量成功才准备全量，也不忽略明显输出错误机械扩量。不另请许可、不重下载、不做新smoke。严格融合的`reviewed_pilot`仍保原审批门禁，但已暂停视觉重试，不能继续阻挡用户最新授权的MS风格基线。训练完成后固定3case接冻结真纯视频H3；其视频效果决定下一步，不用表示指标或文档发布替代。

30B首个正式训练产物必须记录`smoke_steps=0`、实际512/128、三完整epoch、每轮valNLL/自回归输出、完整Adam/RNG/sampler；正式训练后仍要固定3case视频验证，不能把captionNLL当舞蹈成功。

## 15. 下一步与停止边界

- 不重跑旧6500cache、direct6000、18学生或ABC；A/B反馈已收，不自动重复评分或弱维度扫描。
- 首32真实审核链保留；用户新授权优先现成MS＋风格弱监督，暂停视觉教师重试。N7实际30B源标签512/128三epoch已启动；原6000/500标签转换和N3 H3弱标签缓存并行，不继续做无关全视频重复哈希或接口smoke。原窗unknown、语义审核false不伪装GT。
- 获审后新版本caption/cache、正式pilot/full与端到端三个case验收按批准预算推进。
- 所有阶段训练正确结束、完整来源/媒体审核、实际人评及最终GitHub核验之后，才宣告整体完成并取消本轮跟进。
- 当前整体目标未完成；本详细归档、接口通过和单case正评都不会缩减正式目标。
