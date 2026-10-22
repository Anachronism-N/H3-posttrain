# OmniDance/H3 实验历史、音乐替换文本失败分析与当前执行状态

整理日期：2026-10-22（北京时间）。本文是用户要求的详细归档，不是新的实验启动命令。

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
5. 首32候选27格式过、5格式拒，真实事实/叙事/时间窗/冲突审批仍0，正式 pilot/full 和端到端验收尚未执行完成。

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

## 7. 音乐替换文本hidden的三代方案

### 7.1 第一代：图上下文rawH、query与词长

教师用图结构＋Dance style＋原MS，经冻结H3 TC得到上下文化rawH。学生先用冻结MusicFM特征训练adapter，后改为实际dense骨干＋完整adapter联训；query位置、cross-attention、末端投影和预测词长均保部署。

早期1800/200、1875更新，验证rawMSE约5.382、长度MAE137.71；双真实音乐负例NCE候选正音乐胜循环错配约62%，但视频仍“不像跳舞”。旧soft-Chamfer评分不是最新6000的全batch mean-CLIP；两者不可合写一个loss。

后来A/B/C各100/300步六组在线训练54视频：A冻骨干，B/C联训且词长loss权重不同。六组验证rawMSE均差于位置均值基线4.81506，0/200胜基线；这是拟合不足/平均模板线索，不证明彻底坍缩或均值视频更好。

### 7.2 六更新冻结H3行为监督

control/prediction从C300同起点，3配对样本各访问两次，共6更新，固定sigma0.9/noise42；另18视频。

一个留出点prediction误差相对control低3.516%，相对init低0.960%；仅一个样本/噪声/时间，不是泛化证明。用户反馈仍pose，不能据此永久否定全部行为监督。

### 7.3 输出空间与noTC诊断

原neutral教师有舞蹈。真实教师H经过旧W256/LN255最优空间投影仍pose；train-PCA256部分像舞蹈但液化/模糊/反转。teacher投影排除了音乐预测误差，因此旧已学空间确有损失证据；不推出任意256维必败或PCA音乐学生成功。

noTC W/PCA同8train/8val、100更新，分别NCE1/NCE0；四组valMSE约5.13–5.25，用户两预设均pose。对当前固定LN头的放松闭球下界约4.72–5.19，而PCA teacher投影约0.71–0.86：好的basis不等于这个实际头的尺度/中心已覆盖教师。

### 7.4 第二代：纯文字教师＋无TC学生1800

teacher仅舞种＋中性MS，不输入图/presentation，保存逐tokenBF16H；学生完全不运行TC，图只走独立视觉VAE，native音乐固定静音、配乐仅mux。

两起点public512/末端Identity与control256/LN，各1800train/200val、global8、1200更新，18学生视频技术完成。用户原曲预设认为有舞蹈，case1/3幅度尚可、case2小，但不太跟拍；换曲仍近pose。

这是部分正结果，必须保留：不能把整个hidden路线说成从未有舞蹈，也不能把原曲舞蹈当稳健音乐编排成功。该阶段约5.33遍曝光，不以1200更新自动称收敛。

### 7.5 第三代：direct6000＋cachedP＋原native音乐

这是最新已结束的旧主线，特征如下：

```text
教师：参考图＋Dance style＋完整中性MS → TC → [P;y]
学生：真实music waveform → 在线MusicFM [B,2,T,1024]
                         → 声道优先展开[B,2T,1024]
                         → 单个Linear1024→5120
部署：同图cached P＋学生254有效音乐tokens → 冻结原AV DiT
```

MusicFM实际dense316,343,296参数/400张量＋Linear5,248,000参数/2张量都联训。无qmax、词长头、PCA、32D桥或部署丢弃的CLIPhead。保同图P/Ptags、独立图/audio VAE、原生真实音乐与AV各50步；正式生成不加载TC。

A优化对称batch CLIP；B加逐样本均值/RMS。两组同init、6000/500、global64、lr骨干1e-5/adapter1e-4、epoch20降LR、epoch28数值停止，共2632更新。18视频按保存best而非last生成。

| 项目 | A纯CLIP | B＋mean/RMS |
|---|---:|---:|
| 最终trainCLIP | 0.118141 | 0.290612 |
| 最终val500CLIP | 7.201042 | 7.327626 |
| 最终valR@1 | 0.007 | 0.008 |
| train probe64R@1 | 1.0 | 1.0 |
| last28 rawRMS/teacher | 0.223623 | 0.990551 |
| 部署best epoch | 3 | 2 |
| best rawRMS/teacher | 0.187144 | 0.974624 |

用户本轮原话：“还是和之前一样,接近摆pose”。这是整体定性负评，不捏造A/B或18条逐例排名。B校尺度未建立舞蹈成功；不能用last尺度直接归因best视频。

aswap同时换M和native音乐，不是M-only因果；zeroM保native真实音乐；18输入/缓存/权重/音轨/媒体来源均完成审计。数值平台与工程流程完成，连续舞蹈目标未达成。

依据：[notes/25](25_music_h_training_clip_dpo_and_generation_20261019.md)、[notes/26](26_music_h_diagnosis_and_solution_plan_20261020.md)、[纯文1800人评](../results/ms_20261020/music_text_pure_students_human_review_20261021.json)、[direct6000终验](../results/ms_20261020/music_direct6k_v2_final18_technical_v1.json)。

## 8. P、y、embedding与生成接口的澄清

参考图有两条独立路径：

1. 图像VAE→ref_visual_anchor→DiT视觉参考。
2. Qwen-VL视觉网络＋文字embedding→语言模型→最终[P;y]。

P不是图像VAE latent；y虽对应prompt位置，编码时可读取此前图像上下文。tokenizer输出整数IDs[L]；embedding查表才是[L,5120]输入；LM输出hidden是另一层级。

旧direct替换的是LM输出y，音乐rawhidden不再经过LM。将音乐soft embedding拼到LM输入是另一套训练/部署计算图，不能把最终cachedP当原输入embedding或每层KV，也不能冻结LM后no_grad切断音乐梯度。

ABC仅一个旧固定case：A原[P;y]复用，B同图文y无P，C同文字prompt纯文y无P。同图/native音乐/seed/AV50。用户认为差异不大，B/C几乎一致；仅说明该case视觉近似，不证明所有prompt无P无损、两个hidden逐值相同或新soft-input已训练。

当前新融合case FqlHJVDlFGLKCUJS不是旧三人物case。不同case、caption和audio协议不混作单变量因果证据。

## 9. 失败原因：按证据强度排序

### 9.1 最强：监督与实际生成条件不充分匹配

最新direct损失消费每sample有效token均值及中心化标量RMS，而DiT读取全序列。对有效输出行置换，mean/RMS/CLIP都不变。

共享Linear又满足mean(Wf+b)=Wmean(f)+b：全局NCE只对这一读出施约束，没有直接规定每行承担哪个动作、转场或拍点。RMS只管总体能量，不能识别协方差方向。

**可证的是损失不识别学生最终token排列；不能说教师上下文化mean绝无顺序信息，也未证明H3对任意置换都敏感，更不是唯一pose根因。**

代码：[music_text_contrastive.py](../scripts_omnidance/exp/ms/music_text_contrastive.py) 的_direct_sample_statistics/_direct_losses；[music_text_adapter.py](../scripts_omnidance/exp/ms/music_text_adapter.py) 的DirectMusicTokenAdapter。

### 9.2 强证据：泛化分化，更多同配方更新非优先

A epoch3→28 trainCLIP3.479→0.118，固定val5.762→7.201；降LR后八完整epoch未恢复。probe64全中而val弱，是有限probe与留出集的分化，不称全train全中，也不将不同候选数的R@1当直接统计检验。

充分训练和泛化训练是不同概念。原1800预算可能不足，不能拿它证明任何路线已收敛；最新6000多轮退化则不支持无限同配方长训。

### 9.3 较强候选：可达空间/条件分布

固定W后每音乐token在b+col(W)，秩≤1024；外形5120不等于自由5120语言hidden。骨干非线性不解除末端仿射限制。旧W/LN teacher投影为具体坏空间提供证据，但不能证明最新1024空间必然不足。

教师H是上下文化语言/图文结构载体。学生可有合理global语义仍偏离冻结DiT习惯的序列分布，因此更高检索/更低MSE/匹配RMS不保证可生成编排。

### 9.4 待验证：音频到动作文字一对多、监督噪声

音乐可支持多种舞蹈；音频无法唯一确定左右肘、手靠哪肩和精确词序。同曲多编舞/跨曲相似动作可能使MSE平均化或唯一配对NCE含语义假负例。已有同曲来源mask不自动解决全部跨曲语义近似。

rich生成器未知、MS句序/相对秒数未核、VLM energy/turn与kin可能冲突；这些需要真实观察，而非自动删除不方便字段后当监督通过。

### 9.5 确定缺项：局部时序/节拍不是词索引

2T≠L不是接口错误，但音乐时间网格与语言词序含义不同。原损失不显式约束动作强调/过渡/拍点，教师也不必强跟拍。不能均分词索引或使用MS伪秒数制造时间GT。

### 9.6 不支持的过强归因

- 不支持最新6000“音乐骨干没训练/末端映射没部署/P全丢/原音频被静音”。
- 不支持“所有问题由CFG造成”“H3根本不会舞蹈”“MotionScript所有形式不可用”。
- 不支持“只要同维度或RMS就相同”“所有256维不可能”“任何长训都无效”。
- 不支持“旧六更新行为loss已永久否定行为监督”。

## 10. 当前两段主线及实际验收

### 10.1 音乐→离散caption

首版仅音频，动作计划与外观分开；Omni保未来图像能力但视觉tower冻结且禁止调用。Thinker文字输出，不创建Talker/Code2Wav。

候选：Qwen2.5-Omni7B和官方Qwen3-Omni30B-A3B-Instruct，30B官方revision26291f793822fb6be9555850f06dfe95f2d7e695，25文件内容身份已核。

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

首32在新validator下27ready/5invalid/0approved；所有rich/MS源窗仍unknown。人审需真实事实/叙事/窗口/冲突、reviewer/window_evidence/notes，精确SHA绑定。video可用不等于这32条事实窗获批。

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

瓶颈已从“接口不能训练”转为“正式监督尚未人工确认”。在0审批时启动512/128或6000/500，会违反已批准门禁，浪费计算还可能训练错误时间窗/不真正融合的目标。多上GPU不能替代事实核对。

完成任务的节点应可供用户其它任务，不能以holder100%伪装有效利用。历史watchdog会每600秒检查空闲并补holder；reserve只保护mtime<1200秒。原holder仅保持占卡，不是有用实验；是否解除占卡/预留需明确操作范围，资源快照不会主动kill未知任务或替用户占空卡。

### 12.4 获审后的资源Wave计划，不是当前占用

- N1 7B SFT、N2 30B SFT；按实际依赖可调整，正式启动先核host/cgroup/UUID/未知进程。
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
| 融合提取/重写/获审导出 | [dance_caption_fusion.py](../scripts_omnidance/exp/ms/dance_caption_fusion.py) |
| Omni训练/部署/保存恢复 | [omni_dance_sft.py](../scripts_omnidance/exp/ms/omni_dance_sft.py) |
| H3真纯视频执行 | [h3_video_only.py](../scripts_omnidance/exp/ms/h3_video_only.py) |
| H3训练/124窗口/正式gate | [h3_video_sft.py](../scripts_omnidance/exp/ms/h3_video_sft.py) |
| H3新caption缓存/生成/媒体审计 | [h3_video_infer.py](../scripts_omnidance/exp/ms/h3_video_infer.py) |

已完成历史资产和源码指纹：Omni最终接口7c463626…；H3真实124GPU接口9f5de5e2…；fusion导出b53aece3…。新增H3正式门禁修改后需新source/newcache，不能把旧9f5状态的成功重标为新版本真实GPU成功。

## 14. 下一步与停止边界

- 不重跑旧6500cache、direct6000、18学生或ABC；A/B反馈已收，不自动重复评分或弱维度扫描。
- 先完成正式来源gate的代码/CPU验收与发布，再等待必要真实32审批；未获批准期间继续独立工程与文档，不偷铺训练。
- 获审后新版本caption/cache、正式pilot/full与端到端三个case验收按批准预算推进。
- 所有阶段训练正确结束、完整来源/媒体审核、实际人评及最终GitHub核验之后，才宣告整体完成并取消本轮跟进。
- 当前整体目标未完成；本详细归档、接口通过和单case正评都不会缩减正式目标。
