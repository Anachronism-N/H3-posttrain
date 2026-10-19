# H3 音乐条件问题分析与解决方案

更新日期：2026-10-20，北京时间。代码与原始结果基准：[5e93a581e6540e6163e994d5f669a964a6822aef](https://github.com/Anachronism-N/H3-posttrain/commit/5e93a581e6540e6163e994d5f669a964a6822aef)，提交时间为 2026-10-19 22:48。最新文档与人工反馈基准：[1448e5662d7541cc37d5c0f84bdc0457f2c9d951](https://github.com/Anachronism-N/H3-posttrain/commit/1448e5662d7541cc37d5c0f84bdc0457f2c9d951)，提交时间为 2026-10-20 00:45；该提交只更新文档，未改变代码或数值结果。时间均为北京时间。

**当前结论：在线骨干训练、完整 adapter 部署和学生条件共享已经实现；现有配方仍未建立可用的音乐编舞能力。** 下一阶段优先检查教师目标、跨图换歌正向监督、行为监督覆盖和梯度作用，再依据投影实验决定是否扩大 adapter。

本次完成精确源码核验与已提交报告重算，未重新运行 GPU 训练或连续播放视频。下文的解决方案是待实施实验设计，不是已经修复模型效果的声明。已有 72 条视频的 visual 报告仍是当时的采样帧评审；最新文档另记录人工反馈“这一轮还是类似摆pose而不是真正的舞蹈”。这确认最新一轮连续舞蹈目标未达成，但没有逐条 control/prediction 评分，不能扩大成全部历史配置或全部 72 条的逐例否决。[新增反馈及方案](https://github.com/Anachronism-N/H3-posttrain/blob/1448e5662d7541cc37d5c0f84bdc0457f2c9d951/notes/25_music_h_training_clip_dpo_and_generation_20261019.md#L443)

## 目标与已经落实的工程改动

目标保持为：音乐和参考图输入学生，训练 MusicFM 实际 dense 骨干及完整 adapter，输出 H3 原生文本编码器输出侧的 raw H，替代动作文字条件。H3 文本编码器、DiT 和 VAE 冻结，生成时不读取动作描述或真实 token 长度，也不添加人物身份文字或硬锁首帧。

```text
教师：当前参考图结构 ＋ 动作描述 → 冻结文本编码器 → 教师 raw H

学生：音乐波形 → 在线预处理 → 可训练 MusicFM dense 骨干
                                           ↓
当前参考图结构 context ─────────────→ 完整可训练 adapter
                                           ↓
                            全 qmax raw H ＋ 自身预测长度
                                           ↓
                      自身有效 tail ＋ 原图 prefix/tags/packed
                                           ↓
                                 冻结 H3 DiT 和 VAE
                                           ↓
                                          视频
```

真实音乐通过学生 H 分支进入生成，原生两路音频条件保持相同固定静音；生成后再混入配乐。音乐 H 不重新经过文本编码器的 50 层。

| 项目 | 当前实现与证据 | 不能据此推出的结论 |
|---|---|---|
| 在线音乐训练 | B/C 更新 316,343,296 个 dense 参数、400 个 tensor；完整 adapter 为 4,966,657 个参数、42 个 tensor | 骨干更新本身不等于音乐控制成功 |
| 学生条件共享 | 全 qmax、预测长度、mask、tail 共用可微构造；V2 完整重载与独立部署比较通过 | 所检查样本一致不等于所有输入都获得文字同等控制 |
| H3 条件接口 | 选定参考的空语义 context、prefix、tags 和原 packed 已核验 | 接口 oracle 不等于教师舞蹈质量已验收 |
| 行为梯度 | 冻结 DiT 对学生 tail 求 VJP，再精确重放音乐学生回传 | 存盘不代表断梯度，也不代表已经训练整个采样轨迹 |
| 完整视频 | 六组在线模型 54 条、行为对照两组 18 条均有技术完整审计 | 解码通过不能替代舞蹈、音乐同步和人物保持评审 |

依据：[在线与行为实验最终状态](https://github.com/Anachronism-N/H3-posttrain/blob/5e93a581e6540e6163e994d5f669a964a6822aef/notes/25_music_h_training_clip_dpo_and_generation_20261019.md#L312)、[在线优化器和训练循环](https://github.com/Anachronism-N/H3-posttrain/blob/5e93a581e6540e6163e994d5f669a964a6822aef/scripts_omnidance/exp/ms/music_text_contrastive.py#L379-L456)。

当前在线表示损失为 raw MSE / train teacher energy、对数长度 MSE 和同图音乐反事实序列 InfoNCE。行为实验在此基础上增加冻结 H3 的视频预测一致性 MSE。两者都不是 DPO：没有同输入下的输出偏好对或策略相对参考策略的概率比。[CLIP 与 DPO 说明](25_music_h_training_clip_dpo_and_generation_20261019.md)

## 最新结果与独立重算

### 六组在线训练

A 冻结音乐骨干、只训 adapter；B 联合训练且长度权重为 1；C 联合训练且长度权重为 0.01。六组使用同一 1800/200 划分；100 步为 800 次正样本曝光，300 步为 2400 次，后者从同初始化独立重跑，不是接续 100 步的 optimizer。

| 配方 | 正确音乐胜过第一条真实错配音乐 | 三候选 tie-aware top1 | raw H MSE 样本均值 | 长度 MAE | 逐例胜过训练位置均值 |
|---|---:|---:|---:|---:|---:|
| A 100 步 | 52.5% | 40.50% | 5.45948 | 147.415 | 0/200 |
| B 100 步 | 54.5% | 41.25% | 5.46769 | 155.445 | 0/200 |
| C 100 步 | 55.5% | 39.50% | 5.46963 | 155.885 | 0/200 |
| A 300 步 | 47.5% | 36.25% | 5.37288 | 141.070 | 0/200 |
| B 300 步 | 45.5% | 30.50% | 5.40003 | 143.010 | 0/200 |
| C 300 步 | 51.5% | 37.50% | 5.33456 | 142.340 | 0/200 |

训练集逐位置平均 H 在同一教师有效 mask 下的样本均值 MSE 为 **4.81506**。本次逐行重算确认，六组和下面两组行为模型均为 0/200 胜过该基线。八组共享同一批 200 个验证 ID，不是 1600 个独立样本。此结论限定为当前 raw H MSE，不证明均值 H 的视频更好，也不直接证明所有学生输出已经完全坍缩。

A/B/C300 的共同可见 query 区间诊断保留了 47.5% / 45.5% / 51.5% 的胜率。评分长度差不足以解释弱判别；这不是固定生成 packed 布局的视频实验，长度对实际视频的影响仍未单独隔离。[共同区间报告](../results/ms_20261017/music_text_online_s300_common_interval_diagnostic_v2.json)

### 冻结 H3 预测监督

control 与 prediction 均从 C300 初始化，3 个真实配对训练样本轮转两遍，共 6 次更新。六次 VJP 都使用 sigma 0.9、noise seed 42、37 个视频 latent 帧；同一训练 ID 第二次访问的教师预测 SHA 不变。因此准确说法是三个固定教师监督点各访问两次，学生条件、长度和预测仍会随更新变化。

| 模型 | 一个留出样本的教师预测 MSE | 该样本学生长度 | 全 200 验证第一错配胜率 | raw H MSE 样本均值 | 长度 MAE |
|---|---:|---:|---:|---:|---:|
| C300 初始化 | 0.02921945 | 262 | 51.5% | 5.33456 | 142.34 |
| control 6 更新 | 0.02999355 | 389 | 50.5% | 5.33036 | 150.31 |
| prediction 6 更新 | 0.02893900 | 393 | 50.5% | 5.33050 | 151.12 |

预测组在这个留出点相对 control 误差低 **3.516%**，相对初始化低 **0.960%**；只有一个样本、一个时间和一个噪声，不能称为稳定泛化收益。两组全 200 验证三候选 top1 都是 34.5%。[留出比较](../results/ms_20261017/music_text_behavior_heldout_s6_v1_compare.json)

54＋18 条视频的采样帧记录仍主要为小手势、后勾腿、手到嘴或头后、原地单臂动作，并存在楼梯人物衣着漂移。它们未建立歌曲对应的完整舞步增量；后续“摆pose”人工负评与目标未达成一致，但不是量化的逐例音乐同步或画质测量。[行为视频评审范围](../results/ms_20261017/music_text_behavior_s6_v1_visual.json)

完整重算数据与源码核验范围见 [独立重算报告](../results/ms_20261020/music_h_diagnosis_recomputed_v1.json)。

## 问题与解决方案

### 教师动作目标的正确性与可学性

教师 raw H 来自配对舞蹈描述，包含具体动作顺序、文字表达和长短。音乐通常不能唯一确定这些细节。回归这些目标可能形成平均表示；这是一项待检验解释，不能仅凭 MSE 认定原因已经确定。

描述本身也有已知待核对内容。原记录中，楼梯 MotionScript 的 360 度转圈与另一份描述及所查 GT 帧存在冲突；原男性代词会影响人物和衣着，中性化后有所恢复。[动作与描述核对](https://github.com/Anachronism-N/H3-posttrain/blob/5e93a581e6540e6163e994d5f669a964a6822aef/notes/22_h3_bidir_blur_attribution_20261013.md#L1104-L1105)、[接口 oracle](https://github.com/Anachronism-N/H3-posttrain/blob/5e93a581e6540e6163e994d5f669a964a6822aef/notes/22_h3_bidir_blur_attribution_20261013.md#L1123)

解决步骤：

1. 优先连续复核已生成的 6 条 oracle 和 3 条中性代词 oracle，逐条记录舞蹈完整性、描述中的动作事件、人物衣着及动作幅度。无需重做已经通过的接口拆拼检查。
2. 核对教师文字与原视频，删除或修正错误动作、人物属性和无法验证的细节；保留原文、修订版及变更理由。统一为较简洁的动作事件表达，先在少量样本上验证，再决定是否替换正式教师缓存。
3. 与已有同图 empty、学生视频比较；需要区分目标问题时补训练位置均值 H 条件。保持参考图、seed、静音及采样参数一致，记录各自长度，不能把不同布局当完全同条件。
4. 教师视频若仍不能提供所需舞蹈与人物保持，先修教师条件。教师视频合格只是必要条件，还需判断它表达的动作是否合理地受音乐支持，不能把任意好看的文字视频当作音乐控制成功。

### 跨图换歌组合缺少正向目标

当前 `_online_conditions` 在同图 I_i 下计算 S(m_i,I_i)、S(m_j,I_i)、S(m_k,I_i)，三个输出只对当前教师 H(I_i,c_i) 比分。后两条只作为负例，只有正配对有 raw H 和长度回归。行为损失也只作用于正分支。[实际分支](https://github.com/Anachronism-N/H3-posttrain/blob/5e93a581e6540e6163e994d5f669a964a6822aef/scripts_omnidance/exp/ms/music_text_contrastive.py#L210-L223)、[损失计算](https://github.com/Anachronism-N/H3-posttrain/blob/5e93a581e6540e6163e994d5f669a964a6822aef/scripts_omnidance/exp/ms/music_text_contrastive.py#L433-L442)

这足以表达“换歌后不要匹配描述 i”，却没有明确表达“当前人物搭配音乐 j 应当产生什么条件”。它可能鼓励配对辨别而未建立组合生成能力；现有负例并非必然错误，问题是正向监督不完整。

建议构造少量交叉正例：

$$
T_{ij}=TC(I_i,c_j),\qquad S_{ij}=S(m_j,I_i).
$$

在当前参考图 I_i 下重新用冻结 TC 编码可迁移的动作描述 c_j，对 S_ij 提供 raw H、长度及配对对比监督。**不能直接复用 H(I_j,c_j)**，因为教师 H 已经受到 donor 参考图上下文影响。描述需去除 donor 服饰、场景和隐含人物属性，并检查动作是否适配当前图。

首轮建议选择 4 张训练参考图和各自配对的 4 条音乐形成 16 格，预先留出 4 个非对角组合，只训练其余 12 格，保留 4 个原始配对。例如以 0 至 3 编号，留出 (i,(i+1) mod 4)，每张图和每条音乐仍出现在训练格中。留出组合也不能作为训练负分支出现。这只检查小量拟合和组合监督，另需保留未见音乐及人物验证，不能用全训练的 16 格证明泛化。

对照从相同完整 checkpoint 开始，manifest 明确每个训练格在正分支、负分支的曝光数、更新数和唯一教师目标数，固定优化器、基础权重和种子。可在相同的 12 个可见组合上比较有无交叉教师正监督，记录额外教师编码量和学生前向量；同更新数不等于相同独立信息量，结论应归于新增组合监督整体。语义相近描述、同源或近似歌曲允许多正例或过滤，不把它们硬设为负例。

**首轮交叉样本只用于 H 表示监督。** 通常没有“参考图 i＋音乐 j”的真实配对视频，不能默认把原视频 i 的低噪声状态当作动作 j 的行为训练状态。若以后加入行为损失，须明确状态来自哪个教师条件和哪条轨迹。

### 预测监督的覆盖不足与梯度作用未分离

当前行为目标在同一 noisy video、参考和静音条件下比较：

$$
L_{pred}=\mathrm{mean}\left[(F_W(x_\sigma,I,H_{student})-\mathrm{stopgrad}(F_W(x_\sigma,I,H_{teacher})))^2\right].
$$

W 冻结，梯度经学生 tail 传入音乐模型。代码以 x_sigma=(1−sigma)z_GT＋sigma noise 构造状态，没有以 noise−z_GT 为 GT velocity 标签。完整 37 latent 帧表示单次预测覆盖整段，不表示 37 次更新，也不表示已经覆盖 50 步自由生成。

`prepare` 固定前 3 train＋1 val，`update` 限制训练 index 小于 3、行为权重只能为 0 或 1，并固定若干学习率和基础损失系数。它目前是受控小实验入口，需要改造成 manifest 驱动后才能用于通用行为蒸馏。[样本选择](https://github.com/Anachronism-N/H3-posttrain/blob/5e93a581e6540e6163e994d5f669a964a6822aef/scripts_omnidance/exp/ms/music_text_behavior.py#L49-L58)、[更新入口](https://github.com/Anachronism-N/H3-posttrain/blob/5e93a581e6540e6163e994d5f669a964a6822aef/scripts_omnidance/exp/ms/music_text_behavior.py#L315-L373)

六次预测组总梯度 norm 为 1.437 至 3.116，均触发 global clip 1。预测 loss 标量约为基础 loss 的 0.93% 至 1.51%，但量纲不同，不能据此判断预测梯度被压制。当前没有同参数状态下的分项梯度和方向报告。

解决步骤：

1. 在同 checkpoint、样本、sigma/noise 和学生 tail 下，分别取得 raw H、length、NCE、prediction 的未加权及实际加权梯度，按骨干、内容 adapter、长度头统计 norm、dot/cos，检查分项之和等于总梯度。长度离散截取不对预测 loss 可导，长度头依靠既有长度监督；其零 prediction 梯度的 cos 记为 null，不当作方向冲突或断链。
2. 保存全局裁剪系数。global norm clipping 统一缩放总梯度，不单独旋转其方向。若判断 Adam 实际更新贡献，须复制同一 moments、step、组学习率、epsilon、decay 和裁剪流程，比较基础损失与基础加预测损失的虚拟一步。正式 checkpoint/optimizer 更新数为零，两个副本各一步单列；状态不可得时只报告梯度竞争。
3. 改用 manifest 指定真实配对样本、当前图、音乐、教师描述、GT latent 来源、sigma/noise、模型和缓存 SHA。保留现有精确 tail replay、VJP 来源绑定及 Adam 状态恢复。
4. 先选择实际 50 步 shift12 日程中的纯噪声起点、一个中 sigma 和一个低 sigma，使用运行时导出的 sigma 数组、索引、dtype 和哈希，不能把采样步数、timestep 和 sigma 混用。每个点使用两个不同噪声种子，control/prediction 完全共用状态序列。
5. 初轮可选 4 至 8 个真实配对训练样本，每样本 3 个 sigma×2 个噪声，形成 24 至 48 次更新的覆盖检查。另预先固定至少 4 个留出真实配对样本和未见噪声，报告每个 sigma 的配对误差、分布和反向样本。这个预算用于判别信号，不代表收敛。

预测评估只需前向；分项梯度诊断只选少量点，避免对所有评估点重复做 VJP。N 个留出样本、3 个 sigma、2 个噪声比较 init/control/prediction，需要 18N 次学生前向和 6N 次教师前向；教师只计算并缓存一次。记录 noise 和 x_sigma 的实际哈希，不能只凭 seed 编号认定输入相同。冻结教师预测按完整状态键缓存；学生 VJP 不能跨参数更新复用。现有 full37 两卡方案已经实际通过，峰值约 50.03/48.33 GiB；该测量不保证新输入耗时或显存相同。

若多 sigma 的真实配对状态上出现可重复收益，再考虑教师生成轨迹及少量学生实际访问状态，检验自由生成分布。交叉参考条件使用哪条状态轨迹需单独定义。

### adapter 公共输出子空间的限制

当前 width 固定为 256，最终 `raw_output` 是共享 Linear(256,5120)。对于固定 checkpoint，所有输出 token 都属于维数不超过 256 的公共仿射子空间；这不等于只能输出 256 个 token。无 query self-attention 也不意味着完全无法表达时序，因为绝对 query 和音乐时序 memory 仍存在。[结构实现](https://github.com/Anachronism-N/H3-posttrain/blob/5e93a581e6540e6163e994d5f669a964a6822aef/scripts_omnidance/exp/ms/music_text_adapter.py#L104-L170)

先做三种教师条件对照，固定原图 prefix、教师长度、seed 和布局；教师长度只用于该 oracle 诊断，不进入学生部署：

| 条件 | 要判断的问题 |
|---|---|
| 原始教师 H | 当前目标的行为上限 |
| H 到当前 W 列空间的最优投影 | 当前输出空间本身能否承载目标 |
| H 到训练集拟合 PCA256 的投影，必要时 PCA512 | 更好的同秩空间是否足够，以及更高秩是否有必要 |

若 raw_output 权重为 W、偏置为 b，Q 为 W 列空间的正交基，对列向量 h 的最优仿射投影为 b＋QQᵀ(h−b)。这个下界对应当前输出空间，不等于模型在实际音乐输入下能够达到它。PCA 及中心仅用训练集拟合；近似算法应保存算法、采样和残差，不能把近似解声称为精确最优。

若当前投影教师仍能有效控制视频而学生不行，优先修音乐映射和监督；若当前空间差、PCA256 好，优先修空间学习；若同秩投影普遍损害控制且更高秩恢复，才支持扩宽或增加更合适的序列结构。raw H 重构改善仍需行为验证，不能独立证明视频收益。

### 长度与音乐内容需要分开诊断

共同区间评分已经表明：仅调整 score 的取样长度不会修复当前音乐判别。视频原/换/零音乐时，学生预测长度和 packed 仍同时变化，动作变化并不能独立归为音乐语义。

建议使用学生全 qmax 输出做两类纯诊断：固定一个由学生预测范围确定的共同长度，只交换 H 内容；再固定全 qmax token 数值，改变截取长度。前者检查同长度内容影响，后者检查有效 token 数量、可见内容截取与布局的联合影响，不能称为纯布局隔离。保留原生预测长度版本作为主线，不把教师真实长度作为部署输入。

已有四样本长度梯度较大的记录是梯度竞争线索，C 将长度权重降为 0.01 后没有建立明确收益，不能认定长度 loss 是唯一根因。若分项梯度支持继续隔离，可单独比较 length head 输入 stop-gradient，使长度头训练而长度梯度不更新共享音乐内容表示；这与再次降低权重是不同干预，需要独立对照。

## 数据与评估需要保留的边界

原始配乐与某个动作描述配对，不代表该配乐唯一要求该舞蹈。应以原始歌曲或可靠近似音频组管理 split 和负例，记录动作语义相近的多正例。旧 2k 划分曾发现跨 split 高度重叠配乐，后续 scale_v4 已另做候选分组；不能据旧划分宣布当前实验已经泄漏，也不能把候选分组称为完整歌曲隔离。

当前训练上下文的空语义一致性在选定 4 个参考上数值通过，不应描述为 2000 条全部逐值检查。新增教师或交叉样本需保存当前图 context 来源，并做对应等价核验。

8 组报告反复使用相同 200 验证样本，继续选配方会产生选择偏差。新的行为留出样本、噪声及组合应预先固定，报告全部错配对照，避免只挑第一或第二 wrong 中更好的成绩。三候选 top1、两两胜率、H MSE、预测 MSE 和舞蹈质量分别报告。

连续音画评审需单列舞蹈完整性、音乐响应、人物衣着、画质及异常，保留同图同 seed 的正确/错配/零音乐，盲化模型标签。像素差、单帧大动作、预测 MSE 降低或媒体解码成功，均不足以单独证明音乐编舞。已有自动节拍或模型评分未建立可靠校准时，不用其替代连续评审。

## 实施顺序与判断标准

| 顺序 | 具体工作 | 可以支持的判断 | 结果不足时如何处理 |
|---|---|---|---|
| 1 | 复核已有 oracle 和描述，建立逐例目标清单 | 当前协议确有可蒸馏的舞蹈与人物保持能力 | 先修教师与数据，不把接口通过当质量上限 |
| 2 | 同参数零更新分项梯度；同时做教师输出子空间投影 | 分离梯度竞争、输出空间限制和映射误差 | 按证据决定系数或结构，不直接扫描更多模型 |
| 3 | 4×4 少量交叉正例，12 组合训练、4 组合留出 | 当前图换歌的正向条件可拟合，初步支持组合监督 | 若训练格也无音乐依赖，回查目标、容量和优化；不能把训练格成绩称泛化 |
| 4 | 小预算多 sigma/noise 行为对照，另有真实配对留出 | 多样本、多状态误差改善可重复且依赖正确音乐 | 单点改善或所有音乐同样改善，只支持有限拟合信号 |
| 5 | 固定小量人物与歌曲连续盲评，保留自然长度及长度控制 | 表示改善转化为可感知舞蹈、音乐响应和人物保持 | 明确记录无收益或不确定，不能以技术完成宣布任务成功 |

这些是研究判断标准，不是禁止用户已授权诊断生成的准入门槛。每个对照只改变对应因素；改变教师、组合数据、状态覆盖、损失权重和模型宽度时分别记录，避免把多项同时改动的结果归给一个因素。

## 对应代码改动位置与复现

| 文件 | 下一实现应修改的内容 |
|---|---|
| `scripts_omnidance/exp/ms/music_text_cache.py` | 当前图与 donor 描述的交叉教师缓存、caption 版本及图 context 来源 |
| `scripts_omnidance/exp/ms/music_text_pipeline.py` | 输入 manifest、组合划分、源歌曲组及数值 bundle 的 SHA 绑定 |
| `scripts_omnidance/exp/ms/music_text_contrastive.py` | 交叉正向 H/长度目标、多正例 mask、同预算对照和内容/长度诊断 |
| `scripts_omnidance/exp/ms/music_text_behavior.py` | 从固定 3＋1 改为 manifest，调度多 sigma/noise、分项梯度和完整 optimizer 恢复 |
| `scripts_omnidance/exp/ms/music_text_diagnostic.py` | 追加当前输出空间与 train-only PCA 投影诊断，保留共同区间评分 |
| `scripts_omnidance/exp/ms/music_text_adapter.py` | 先保留共享学生条件；只有投影证据支持时再独立比较宽度或长度分支隔离 |

以下是现有在线 B 配方的命令模板，可复现旧对照，不包含尚未实现的交叉正例或通用行为实验参数。路径须替换为具有已核验缓存和模型的 Linux 环境；输出必须是新目录，父目录已存在。

```bash
H3_REPO=/path/H3-posttrain

/opt/venv_music/bin/python \
  "$H3_REPO/scripts_omnidance/exp/ms/music_text_contrastive.py" \
  --online \
  --online_bundle /path/verified_online_inputs.pt \
  --init /path/original_raw_h_checkpoint.pt \
  --split /path/split.json \
  --out /path/new_online_B_run \
  --backbone_train \
  --steps 300 --batch_size 8 --microbatch 2 \
  --backbone_lr 0.00001 --adapter_lr 0.0001 \
  --clip_weight 1 --length_weight 1
```

去掉 `--backbone_train` 对应 A；保持联合训练并改 `--length_weight 0.01` 对应 C。行为实验的实际阶段命令与哈希见 [已有复现记录](../results/ms_20261017/music_text_behavior_s6_v1_reproduction_commands.json)，它仍是固定小实验，不应伪造不存在的 manifest 或 sigma 扫描参数。

视频继续使用 `scripts_omnidance/exp/h3ar/sample_h3_ar.py` 的 `--music_text_ckpt` raw H 路径，保留 empty、bi、CFG1、固定 native 静音和 repeat_clean；具体 CSV 和命令见 [生成流程](25_music_h_training_clip_dpo_and_generation_20261019.md#当前代码如何生成视频)。现有方案不需要改为 DPO，也不把新增音乐目标等同于直接训练 DiT。

独立重算只依赖 Python 标准库，可在仓库根目录检查各组逐例基线胜负：

```python
import json
from pathlib import Path

root = Path("results/ms_20261017")
names = [f"music_text_online_{group}_s{step}_val200_v1"
         for step in (100, 300) for group in ("A", "B", "C_len001")]
names += [f"music_text_behavior_{group}_s6_v1_val200"
          for group in ("control", "prediction")]
for name in names:
    rows = json.loads((root / f"{name}.json").read_text())["cases"]
    wins = sum(row["raw_mse"] < row["position_mean_mse"] for row in rows)
    print(name, wins, "/", len(rows))
```

本轮交付为问题分析、实验方案、独立重算和现状文档修正。上述新增教师、交叉训练、梯度干预、多状态行为实验与模型扩宽均未在本轮实施，不把方案推送等同于音乐替代动作文字目标已经完成。
