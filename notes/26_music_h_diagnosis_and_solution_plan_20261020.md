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

## 2026-10-20 自动执行接续：输出空间与分项梯度零更新诊断

用户随后明确要求按本方案自动推进、修复错误、跟进实验并推送全部结果。此节为后续实际执行状态，不抹除上文当时“未实施”。保留音乐＋参考图学生、冻结H3、完整映射、三生成case；没有启动后置首帧／身份文字／单独长度校准，也没有直接扩大相同loss训练预算。

### 当前执行范围及先验检查

1. 复用已存在neutral教师三joint的原始视频及缓存；它们的完整解码／接口等价已通过，但连续可用舞蹈的人评尚待确认，不重新生成相同原H基线。
2. 同C300固定checkpoint零更新计算两种输出仿射包络与训练集PCA256投影，再直接进行同原教师长度／图prefix／tags／packed的三case生成诊断。它是**teacher oracle空间诊断，不是音乐预测或学生推理模式变更**，不以GT长度提供给学生主线。
3. 复用旧小实验已验全37latent VJP，在同一参数状态下分解raw/length/NCE/prediction梯度；不再次装载冻结H3，不更新任何参数/buffer或正式optimizer。

源码保持旧oracle identity和预测等价门禁；投影走独立`ProjectedTeacherTailEmbedder`／`--text_hidden_projection`，只要求投影的原长度和原packed/前缀一致，允许预测改变作为诊断结果。旧oracle缓存**没有完整prefix**，仅context/tags/ids/tail；新路径运行原空prompt编码当前参考图，逐值比较context/ids/tags后再拼接。不能把缓存context一行冒称全prefix缓存。原/投影共享同次实际prefix，原packed由相同参考shape和原教师tail重新构造并逐值比较。

### 实际数值投影结果

完整checkpoint C300 SHA仍为`189a0a74ea1789b9a4bc1b8d1f202264c354f418b0881339eca10baf0673f605`，waveform/teacher bundle与原训练SHA一致。数值不修改原checkpoint或teacher缓存。

- W宽松仿射包络：中心b，float64 SVD按`max(shape)*eps64*smax`数值rank选基，实际rank256，Gram最大误差约1.2e-13。
- LN仿射包络：中心`b+Wβ`，对`Wdiag(γ)Z`求基，Z是零均值空间1⊥的255维正交基，实际rank255。它比W包络更准确，但仍不约束LayerNorm方差／可达半径，**仍是放松下界而非可达到的音乐输出**。
- train-PCA256：精确FP64计算全部1800train有效tail token的中心；每项最多16个确定性等距token（总28800），采样按原长度/采样数加权，`torch.pca_lowrank(q=288,niter=4,seed42)`近似主空间再正交化。没有prefix/padding/val/三生成case参与拟合，不把近似协方差或随机SVD宣称全量精确最优PCA。空间中心与W包络不同，因此差异应归于当前已学仿射空间整体（中心和方向），不能只归为旋转方向。

| 条件空间 | 同200val原raw H投影MSE | 三独立生成case投影MSE（楼梯／红衣／商店） |
|---|---:|---|
| 当前W256包络 | 4.84153 | 5.24636／4.97505／4.83517 |
| 当前LN255包络 | 4.84251 | 5.24740／4.97603／4.83594 |
| train-only近似PCA256 | 0.72084 | 0.86075／0.71350／0.71105 |

三个oracle原教师token数228／199／323全部保持；投影幂等最大误差<6e-12，实际BF16cast误差单列。另有最初v1以采样均值为中心的探索产物保留，**实际生成采用v2全train中心及加权采样结果**，不能混用两个PCA数值。

数值提示当前已学公共输出空间与教师空间存在较大偏差；同秩PCA重构较好尚不证明视频／舞蹈更好，也不证明256维必然足够。完整元报告为`results/ms_20261020/music_text_projection_C300_v2_numeric.json`，数值basis与tails PT仅保存在实验目录，不公开权重或媒体。

CPU验证覆盖秩亏SVD、投影幂等、正交残差、空间内identity；实际9份payload及mock prefix拼接通过（最初fixture缺tokenizer对象已修正，仅fixture错误，未启动GPU）。真实空prefix/context及packed必须在每条实际采样日志再次验收，不能用CPUfixture代替。

### 三个实际监督点的梯度分项结果

复用旧prediction小实验第1／2／3次更新前的各自checkpoint和对应wave/context/full37 VJP；每个点内部所有loss参数状态与噪声state相同，**三个点之间checkpoint不同**，是三个已发生监督点的局部诊断，不冒称同一模型的全train统计。使用实际权重`raw/Etrain、length*.01、NCE*1、prediction*1`，所有参数/buffers原样，optimizer更新0。

| 监督点 | bone：NCE norm | bone：prediction norm | content adapter：NCE／prediction norm | bone NCE/prediction cosine | content NCE/prediction cosine |
|---|---:|---:|---|---:|---:|
| 第1点，AAca… | 1.62339 | 0.00123263 | 0.52392／0.13578 | 0.80600 | -0.00829 |
| 第2点，ACYT… | 1.93829 | 0.00412092 | 0.47310／0.36064 | -0.00104 | 0.00605 |
| 第3点，ACjm… | 2.82799 | 0.00007722 | 0.64572／0.04886 | -0.10660 | -0.00972 |

实际分项和与单次combined derivative数值比较relative-L2误差6.55e-5／5.37e-5／6.75e-6（FP32计算，预定容差1e-4），总norm1.73476／2.04070／3.11576，global clip系数.57645／.49003／.32095。总norm与旧实际训练对应点吻合；没有用surrogate dot scalar冒称prediction loss值。lengthhead预测梯度为零，方向cos=null，保留原长度监督。

**目前支持的局部判断：** bone的行为梯度norm在这三点比NCE小约1320／470／36600倍；内容adapter的行为梯度并非总可忽略，夹角接近正交。不是证据支持全参数强反向冲突，也不能说global clip单独压制prediction（它统一缩放总梯度），更不能由norm比例断言Adam真实更新份额。本轮没有复制moments做虚拟Adam一步，`optimizer_virtual_steps_performed=0`保留；后续只在准确moment/step/参数快照和预算明确时补测，不因当前norm小直接任意乘100。

完整逐组未加权／实际加权norm、dot/cos与求和误差见`music_text_prediction_components_C300_v1.json`、`music_text_prediction_components_step2_v1.json`、`music_text_prediction_components_step3_v1.json`。结果补上原文“尚无分项梯度报告”的缺口，但尚未证明连续舞蹈失败的唯一原因。

### 视频队列与后续决策

三种空间各固定3joint共9新视频已用八节点各一张卡启动，每节点一个H3冷加载；PCA两节点的第三case在同节点前任务完成后串行，旧neutral3视频复用。实际日志已核实投影identity/source SHA、空语义输入、原teacher长度、同参考prefix ids/tags/context、原packed exact、native静音无真实音乐旁路。**截至本节记录，视频仍在采样，完整解码与连续人评未宣称完成。** 独立读码审计未发现实质bug，但不替代视频结果。

完成后仅新增投影诊断清单和“先复审教师上界”预设，保留原34模型／旧预设和评分。若teacher本身不合格，先修描述事实与目标，不因输出投影MSE改善直接开学生大训练；若当前包络视频差但PCA同秩可用，优先研究仿射空间/语义映射学习；若同秩都损害控制再按同train中心嵌套PCA512考察，不能只据raw MSE扩宽架构。

后续交叉正例12+4的留出必须按(图ID,音乐源ID)屏蔽所有正负分支并审计init历史曝光，过滤语义近歌后不足负例不得回填留出；当前单sid代码不可直接复用组合ID。当前还未做交叉正例训练、多sigma/noise行为训练、stop-gradient长度分支或扩宽，均按教师上界和上述机制诊断结果决定具体单变量协议，不同时更改全部因素。

## 输出空间诊断实际生成完成：同长度投影改变控制，PCA部分接近教师

本节更新前述采样中的当前状态。W256/LN255/PCA256每个3joint共9新视频全部采样exit0；neutral教师3joint原视频复用，没有新增同原H基线。`results/ms_20261020/music_text_projection_C300_v2_all12_audit.json`为expected=passed12（9新＋3复用），704×1280/24fps/124帧decord逐帧＋ffmpeg视频/选定音轨完整解码通过。实际教师长度228/199/323固定；三投影同case的实际prefix SHA一致，source/context/ids/tags、原packed逐值一致，四组native静音hash一致。没有用预测等价门槛要求投影等于原teacher、没有关闭旧oracle identity检查、没有更新音乐模型。独立`music_text_projection_C300_v2_independent_audit.json`重查12份媒体/日志SHA及当前布局证据；源basis/tail文件仅本地使用，不公开数值PT或媒体。

### 实际采样帧观察与证据边界

主线程Read四组reference＋0/1/2/3/4/5.125秒图，`music_text_projection_C300_v2_visual.json`记录，**仍不是连续听看或新增用户舞蹈评分**：
- 原neutral教师：楼梯保持浅黄衣/灰裤，采样可见步移、转身、双臂展；红衣和商店有变化的双臂／肘部、站姿与重心。只能确认所见动作事件，不能据此认证持续编舞、所有文字事实或卡点。
- W256：楼梯变深蓝运动衣裤和小包，以小手势/后勾腿为主；红衣转为手到嘴／脑后／单臂抬起；商店低运动、原地有限臂部动作。
- LN255：与W控制母题接近，楼梯运动衣裤漂移、红衣脸/头部手势和商店低幅变化仍有；不据采样声称两者统计等价。
- train-PCA256：红衣和商店采样臂部／站姿序列比W/LN更接近原neutral教师；楼梯仍有较大转身、步移和展臂，但变白蓝运动衣裤/小包，人物外观未保持。PCA没有在本轮被训练成音乐学生，只是投影teacher条件。

teacher对投影的未对齐RGB MAE分别：楼梯W.14578/LN.12753/PCA.12092；红衣.06460/.06485/.03020；商店.05023/.05057/.01386。它们不是pose/舞种/节拍accuracy，楼梯服饰/位置变动可占大量像素，不用最小RGB差给编舞打分。

**目前可以支持：** 即使固定原teacher长度与当前图prefix/原packed，投影到当前W/LN包络也改变了教师可见动作与参考外观；不能把这些变化完全归为学生预测长度或漏加载最后投影。相同秩train-PCA256在两case的动作序列更像teacher，是当前已学仿射空间整体可能不足的具体诊断证据。**不能支持：** 256维必然足够、必须扩宽、PCA音乐encoder已训练成功、完整LayerNorm输出可达性已排除、或teacher连续舞蹈已被人工验收。PCA与W中心/方向都不同，仍不只是一项旋转变化；楼梯外观漂移未解决。先明确可用teacher，随后可优先研究原raw输出空间的更合适train-only初始化/公共分量处理与组合监督，而不是直接加大rank或无限增更新。

### 评审入口、保留与下一阶段门控

最终8766清单`manifest_music_text_projection_C300_v2.json`为37模型/3人物/17预设，原34模型、全部旧预设与评分映射保留。首预设【教师输出空间·零训练同长度】原neutral教师／W256／LN255／train-PCA256，仅joint、清楚标“不是音乐预测”；第二预设【先复审教师舞蹈上界】原neutral教师／C300／六步control／六步prediction，复用已有视频。

`music_text_projection_C300_v2_review_access.json`核验清单200、N2跨节点200、84个已验唯一视频Range206；计数是此前72音乐模型视频＋本轮9投影视频＋3复用neutral教师视频，不是重新生成12条，也不声称37个模型所有历史视频总数只有84。CPU pure-memory清单fixture验证保留原teacher/三case/旧预设及错误同布局标志拒绝；技术审核不代替连续人评。

本轮零更新空间/梯度诊断与九视频技术生成已经完成。**截至该次发布，下一阶段仍缺teacher连续舞蹈的定性验收信息**，不能把用户对学生“摆pose”负评自动套给原teacher，也不能把助手六帧采样当P0通过。按最新方案先请用户复审已有neutral三个case：若teacher也主要摆pose则先修事实/监督；若teacher在对应case可用再落实单变量组合正例／噪声覆盖或有证据的loss平衡。此门控不会阻止已经授权并已完成的诊断推理，但不能凭未知teacher上界开始扩大同目标训练。首帧、身份文字、单独length校准保持后置。

## 2026-10-20 用户连续评审：教师能跳，当前输出空间与学生丢失舞蹈

用户在暂停自动实验后，对两个现有预设明确反馈：

> val_musicreplace_text_oracle_neutral_v1是舞蹈,其他几个在【先复审教师舞蹈上界】中的配置都是摆pose. 教师空间·pca256是唯一像舞蹈的,但是存在一些肢体反转等的问题,其他的教师空间·w256和教师空间·ln256还是类似摆pose

| 配置 | 本次用户定性结论 |
|---|---|
| 原neutral教师 | 是舞蹈 |
| C300音乐学生、六步control、六步prediction | 都是摆pose |
| train-PCA256投影教师 | 投影组中唯一像舞蹈，但存在肢体反转等问题 |
| 当前W256及LN255投影教师 | 仍类似摆pose |

用户所称“ln256”对应当前清单的LN投影组；实现及数值报告的实际仿射rank为255，仍记为LN255，不改成新实验或256维结果。本次为两个预设的定性连续评审，没有提供逐case评分、异常时间点、音乐同步或完整外观评分；不补造这些信息。此前“教师人评pending”是历史发布状态，现已收到明确教师舞蹈能力反馈，不再以缺教师上界信息重复等待。教师是舞蹈不等于任意描述事实正确、音乐控制或整体质量均验收通过。

### 更新后的机制判断与优先级

1. **不优先归因为H3或当前教师完全不会跳舞。** 原neutral教师已被用户确认有舞蹈，当前失败发生在条件的压缩、空间学习和音乐预测链路中；其他样本的教师事实审计仍保留，不把三个固定case推广为全训练教师合格。
2. **已学仿射输出空间是优先处理的具体瓶颈证据。** W/LN实验已经提供真实教师H的最优仿射投影，排除了音乐预测误差，保持原教师长度、图prefix/tags/packed；仍摆pose。因此对该已学空间而言，即使学生更准确地预测包络内系数，也不能保证恢复原教师舞蹈。PCA的中心和方向均不同，不能只归为方向旋转，更不能把放松包络内投影当学生可达输出。
3. **不把256维判为必然不足或足够。** 同秩PCA保留“像舞蹈”的能力，表明空间学在哪里比单纯秩大小更值得先隔离；PCA存在肢体反转，说明该压缩尚不是质量通过。反转可能来自投影丢失条件信息或生成敏感性，尚不能认定必须扩宽、纯时间顺序错误或PCA坐标正负号导致。
4. **优先的受控后续候选是train-only PCA仿射输出空间。** 恢复实验后可先比较现有空间与PCA空间的音乐学生，保持音乐dense骨干和内容adapter共同训练、仅音乐/参考图输入、H3冻结、自身长度和完整部署。若使用固定PCA输出basis/center，必须明确这是输出层参数化变化（basis/center固定，音乐骨干及可训练adapter仍联合优化），并先验证系数可达范围和LayerNorm约束，不能仅替换W/b就宣称实现。需要共同初始化/预算且单列空间改变带来的初始条件差异，不同时混入交叉正例、多sigma或新loss权重。
5. **音乐映射仍未解决。** PCA教师投影不是音乐学生；即使它能跳也不能证明音乐能预测正确条件。新学生必须生成固定三case并保留原/换/零音乐对照。若需要隔离PCA残余肢体异常，可再比较相同train中心的嵌套PCA256/512教师投影；这是备选诊断，不是已批准或已执行的扩宽训练。

该次反馈记录时用户已暂停自动goal，未恢复训练、未进行虚拟Adam步骤，也未自动提交或推送本次本地记录。

## 2026-10-20 宽度与真实入口实验接续计划

用户随后明确要求继续推进并充分利用计算资源，询问256维能否调整、如何减少adapter损害，以及当前哪些实验在运行。本节为正在实现的有限对照计划，不能冒充已完成训练或生成结果。

### 256维是什么，以及改变它不意味着什么

5120是H3原生文本hidden接口维度；256是当前实现者选择的adapter内部宽度，贯穿音乐memory/context投影、query、两层cross attention、FFN、输出LayerNorm、raw_output及长度头。8头只是整除约束，当前每头32维，并没有H3强制要求width256。实现将允许256/512/1024及合法head配置，FFN为4width，checkpoint显式记录并严格按宽度构造。旧256权重不使用strict=False塞入512模型。增加宽度可以扩大潜在输出仿射秩，但会增加参数/显存，不能证明音乐映射或时序监督随之成功。

### 三条并行工作线（运行状态以日志／产物为准）

1. **真实入口分阶段对照，0更新。** 固定三个review case，音乐学生使用完整C300及自身预测长度，原/换/零音乐分别导出。当前空语义参考prefix仍由原TC生成，音乐tail不回TC。比较实际raw5120、冻结condition_proj5376及原两层token_refiner后，分开prefix与tail统计形态、尺度、位置误差及方向。不同长度的自然部署与共同可见区间分别报告，不以训练—部署exact冒充学生—教师exact。优先仅加载TC和真实冻结refiner权重，不更改算子。
2. **同中心嵌套PCA256/512教师诊断，6新视频。** 一次q544/niter4/seed42的FP32加权采样分解，原全train有效tail的FP64中心不变；一次512列QR后取前256/前512，先验检查嵌套与512残差不大于256。两空间各固定三个case原教师长度、prefix/tags/packed和采样协议，旧neutral教师复用。新PCA256来自新的嵌套分解，不能拿旧独立q288结果当作此次严格嵌套比较。它仅判断提高投影秩是否改善肢体异常，不是音乐模型或宽度训练结果。
3. **固定输出仿射空间的联合学生对照，各100更新。** 两组均以同完整C300为源，共同保留全部可训练音乐dense及完整adapter，raw5120输出之后应用固定空间投影。W组固定当前W列空间／原bias，PCA组固定train-PCA256空间／train中心；相同结构、原数据顺序、基础loss、length0.01、lr1e-5/1e-4及新optimizer，100步各800正配对曝光。中心/方向整体是处理变量，初始H不同单列。此参数化不删除LN、不新增无约束旁路，仍有每个固定参数状态width256的LN可达限制，不能声称音乐学生可达到任意PCA oracle点。先做CPU宽度／完整重载／空间固定及真实dense梯度审计，再完整200val及两组各三case原/换/零共18视频，不用H指标阻止诊断生成。

初始资源核验：八节点真实SSH采用已存在known-host并核对RSA指纹，保持StrictHostKeyChecking=yes；均有H3本地权重，仅N5仍有原在线waveform/teacher bundle。容器cgroup挂载根是`/sys/fs/cgroup/memory`，初次按/proc完整层级拼路径失败已修正，八节点主机／cgroup均满足180GiB单冷加载准入。占GPU的原进程已核实为hold_gpu.py，不计实验利用率。计划N5分GPU做PCA数值、学生导出和两组训练，其余节点分发教师采样与入口对照；启动前再次逐GPU准入，只释放准确核验的holder，所有运行使用工具管理任务，不用nohup/setsid脱离管理。

不在本轮同时改变组合正例、多sigma/noise、生成loss权重、首帧或独立length校准；不直接扫三档宽度训练。源码、数值报告及实际运行／失败状态随后同步GitHub，媒体与权重不上传。

## 用户进一步明确noTC学生接口：连空白文本也不编码

用户随后明确：

> “实际代码先编码空白文字＋参考图”这个可能是有问题的，我们理论上应该完全不编码文本，空白的也不要输入，直接关掉textencoder，直接用音频token替代原先的文本token。

此要求取代前阶段学生保留空语义TC参考prefix/context的协议。旧学生的音乐tail没有重复经过TC，但整条旧学生路径仍运行TC来构造参考条件；不能将两者混为“TC已经完全关闭”。该设计是否导致了摆pose尚未证明，先按用户新接口执行，不以“原prefix必需”阻止验证。

### 新接口与先验检查

- H3加载列表完全省略text_encoder权重，processor_config=None；断言pipe.text_encoder/tokenizer/processor均None。
- 正／负prompt都为None，拒绝空白或任何文字／soft hook；新embedder不调用原PromptEmbedder.process或preprocess_ref_blocks。
- 音乐raw tail独占原文本条件行，全部tags=1，不再拼任何TCprefix；原PackedSequenceBuilder仍把独立视觉VAE参考、固定静音参考音频及目标音画打包。去掉prefix同时改变序列长度／位置，是接口整体干预，不声称只改变文字语义。
- 参考图保留原ReferenceEncoder视觉VAE条件，记录视觉参考rows/hash；不读取任何TC缓存context给学生。

先生成固定三个case的两组joint：①原neutral教师缓存tail直接输入、无TCprefix；②旧C300完整音乐骨干／adapter直接输出、无TCprefix。第二组为适配旧checkpoint签名暂使用固定zeros5120向量，绝非空白文本编码、不是参考context，也**不是已按noTC训练的新模型**。两组共6视频，保持原图、seed、50步、704×1280、bi CFG1与原生两路固定静音。①是teacher-only接口对照；②是已有模型改变条件接口的免训练诊断。它们不证明新接口训练效果或旧／新协议公平同分布。

上述要求出现时，旧接口W/PCA100步训练与嵌套教师视频尚未启动，已暂停；保留宽度与空间CPU代码但不在旧条件上继续新训练。此前旧C300的三个case原／换／零音乐token导出已真实退出0（N5 GPU6，0更新，不装TC但用旧cachedcontext），仅作历史入口分析；长度分别楼梯413/377/249，红衣243/302/171，商店290/261/160。旧prefix重编码的entry_compare不会自动运行，新学生不能复用它作为noTC验收。

CPU noTC采样器门禁7个夹具通过（两合法／五错误组合）；首次AST夹具截取误含sys.path导致NameError，修正截取范围后通过，是测试夹具问题，未进入GPU。后续须核验真实日志的TC未加载／调用0、semantic_text=null、prefix0、raw tail长度与tags、视觉参考存在、native静音hash及完整124帧解码，再决定共享noTC训练参数化。

### noTC真实入口计算已完成，六视频正在接续验收

六节点真实启动noTC教师tail-only三joint与旧C300 zero-context音乐-only三joint，管理句柄J2gEJo／Q8N3Pt。真实日志已记录省略TC权重、processor/tokenizer均None、semantic_text=null、prefix0、独立视觉VAE参考7424行及hash；原native两路静音与repeat_clean一致。曾在运行中检查六sampler均R态、每张实际实验GPU约76GiB且100%利用率，不拿holder算实验利用。当前teacher三条已经采样exit0；其他任务以最终jobs及全124帧音画审核为准，不把启动或exit0当质量验收。冻结采样源码与发布源hash保存`music_text_no_tc_v1_release.json`；审计必须匹配真实plan/shardheader及该release，不得事后添加运行时不存在的infer字段来冒充启动证据。

同时N5导出、N7冻结原生条件入口比较两阶段均真实exit0，报告`music_text_entry_no_tc_C300_v1_export.json`及`music_text_entry_no_tc_C300_v1_compare.json`。只装载condition_proj与原两层token_refiner的19个真实BF16权重，保留原attention/refiner算子及cu=[0,L,L]，没有加载TC／processor／VAE／50个DiT主blocks；前后module state逐tensorhash不变，参数更新0。教师tail只作比较，不提供给音乐学生。此独立条件入口计算不证明完整H3 packed或视频行为等价。

| case | teacher／原音乐／换音乐／零音乐tail长度 | 教师raw token去token均值后RMS | 原音乐学生同统计 | 教师→学生relative-RMS：raw／condition_proj／refiner |
|---|---|---:|---:|---|
| 楼梯 | 228／278／257／157 | 2.31742 | 0.00143366 | 0.64319／0.65621／0.65710 |
| 红衣 | 199／226／281／157 | 2.27246 | 0.00046252 | 0.61791／0.62972／0.65978 |
| 商店 | 323／294／265／157 | 2.25266 | 0.00093100 | 0.63246／0.64423／0.62932 |

relative-RMS比较四条件共同可见前157行，各条件仍先按自然长度完成refiner前向；不是固定长度／samepacked干预，也不把文字token序号当视频时间。原音乐学生各token的raw RMS约2.685，去token均值后的变化却极小：在这三个旧checkpoint zero-context诊断样本中，学生接近重复同一个向量，而教师token变化显著。原／换音乐raw共同区间RMS差仅0.00622／0.01430／0.01378。该差异在进入共同refiner之前已经存在；refiner后的相对误差仍在0.629至0.660，并非重复textencoder造成的证据。跨模块绝对MSE因尺度／归一化不同，不直接比较以宣称误差被大幅放大。

限制：这是旧C300从cachedcontext训练分布移到zeros5120的免训练诊断，zeros经过旧context_projection仍保留其bias；不能直接推广为旧原条件下全部样本坍缩，也不能否定新noTC训练。后续正在实现`music_only_global_bias`学生协议：移除context_projection.weight与外部context输入，把旧bias作为可训练global bias并保留所有共享参数，初始化应逐值等于旧zero-context学生。训练／验证／独立部署都必须context=None，不读取TC cache；图仅走下游视觉VAE。该新学生尚未训练完成，旧W/PCA100步仍暂停，先以同新noTC协议受控拟合再选择输出空间／宽度。

CPU宽度256/512/1024、旧默认行为exact与完整checkpoint重载，以及无TC访问／严格30错误日志反例已实际验证并保存JSON。这些是实现门禁，不是实际MusicFM新接口训练或舞蹈质量通过。`entry-diagnostic-dev`、`no-tc-audit-dev`及此前其他代理曾出现模型网关Empty stream／499失败；只记录真实已返回代码、工具测试及GPU产物，不把失败调用当独立审计成功。

### 无TC六视频已完整技术验收，旧学生仍主要小动作

teacher三条J2gEJo与旧学生三条Q8N3Pt均实际exit0。主线程发现审计代理误要求运行时不存在的jobs/header infer_sha等字段，停止其写入并直接改为冻结release＋实际plan/header samplerSHA绑定，不事后篡改生成日志。实际N8六视频审核MTnMZW exit0，`music_text_no_tc_all6_audit_v1.json` expected=passed6：decord124帧和ffmpeg选定视频／音轨完整解码、704×1280/24fps、媒体／日志／图／音频／teacher tail来源SHA、TC组件均未加载／调用0、semantic_text=null／prefix0全部通过。同case两组原视觉VAE参考rows与hash一致，native固定静音四hash全组一致，原C300完整模型SHA不变。技术完成不等于新学生音乐控制成功。

主线程实际Read两组reference＋六时刻mosaic，`music_text_no_tc_v1_visual.json`记录：teacher-tail-only楼梯有步移、抬臂与转背，红衣／商店有更多不同臂部和站姿；旧C300 zero-context主要小手势、后勾腿、手到脸／伸手手势和原地小臂动作。楼梯两组均由参考浅黄衣／灰裤漂移成深色外套／裤子，旧学生另有小包。此观察只限采样帧，不冒连续听看、没有新增用户评分或肢体异常逐帧验收。

新noTC学生保留`global_context_bias`仅作为可训练全局query／长度先验，不是文本／图片context、不来自TC；完整音乐骨干与剩余全部adapter参数共同训练。旧模型转换只删除context_projection.weight并重命名其bias，全共享权重和初始zero-context预测逐值保持；新的训练、验证、推理接口只收waveform、context=None，禁止提供空白文字或缓存prefix。

真实三更新smoke FIMJ4f已exit0：400个dense骨干tensor和41个adapter tensor实际更新，冻结参数／buffer不变，GPU峰值约8.55GiB；训练／eval／完整checkpoint以及真正独立waveform-only worker的tail／自身长度逐值一致，缓存TCcontext没有读取。训练阶段实测约1.8秒，先前资源核查时GPU0%与文件系统等待不是训练仍在计算；它随后正常存盘／部署结束，未证明存储存在持续死锁。报告为`music_text_no_tc_smoke_v1_summary.json`和`music_text_no_tc_smoke_v1_independent_deployment.json`。仅3更新／6次曝光，不当收敛、泛化或视频成功。

随后尝试同C300源、同noTC globalbias条件的小集8train／8val固定W与train-PCA256对照，各100步batch8/micro2、lr1e-5/1e-4、length0.01、clip1和新optimizer，各计划800曝光。AuGFI9／e5ibIZ在来源门禁阶段均exit1、未更新模型：旧数值投影finished缺当前消费者要求的父checkpoint finished SHA。保留失败log，正在用同train／checkpoint重新产生带完整provenance的W/LN/PCA数值输入（3cPcTa，0音乐参数更新／没有运行TC）。不把缺字段简单忽略或偷偷给历史日志添字段。后续接续用新输出目录并生成固定三case原／换／零共18视频；小量拟合只判断当前目标是否能学，不冒全200val或未知音乐泛化结论。

当前评审服务已追加6新视频，清单39模型／3人物／19预设，旧37模型／旧预设／评分保留；清单本机200与N2跨节点200，六新视频Range206实测通过。84此前已验视频保留旧审核，本轮并未重新逐一检查全部90链接。入口与实际访问见`music_text_no_tc_v1_review_access.json`。

### noTC固定空间两组100更新已完成，18视频生成接续

完整来源数值重建3cPcTa exit0，父C300 finished SHA已实际绑定，结果见`music_text_projection_no_tc_training_inputs_v1.json`。重启新目录v2的W／PCA两组utguvt／0s5qBX均exit0、各100更新／800曝光：400dense骨干tensor与41adapter tensor实际改变，固定输出basis/center及冻结参数／buffers不变，context=None全共享；完整checkpoint与独立worker的raw tail／自身长度／policy逐值一致。W checkpoint SHA为`600b91897c8eb8d84ebb0e8498d2f6a363ca5797d119d43a8929f3d7533af379`，PCA为`7ba5a8b7ed2d09ef5645e3c9c8f69d903d2a0a6fc026a6114ee2a983ebb7e0c5`。

| 两组同8train／8val小量对照 | W空间 | train-PCA256空间 |
|---|---:|---:|
| train三候选top1 | 87.5% | 100% |
| train raw H MSE（样本均值） | 5.45656 | 5.27885 |
| val raw H MSE初始→最终 | 5.29712→5.24840 | 5.26367→5.14308 |
| val训练位置均值H的MSE | 4.81861 | 4.81861 |
| val长度MAE | 110.625 | 107.625 |
| val三候选top1 | 12.5% | 25% |

小集配对辨别能学习，rawtoken却仍未接近教师，且两组val均未优于位置均值baseline。PCA的指标相对好不等于可用舞蹈；只有8val，不称歌曲／人物泛化或全200val。初始化空间中心／方向不同，结论归于固定仿射空间整体，不只方向旋转；width256/LN的可达限制仍在。

两组各固定三case原／换／零共18视频已交四个受管理队列：W正常oD4UQB、PCA正常Wws0fZ、W零wM7D5G、PCA零gtGY28，计划跨八节点18张卡（N1/N2各3，其余各2）。每GPU只跑一个sampler、每节点最多3，仍按主机/cgroup180GiB及准确holder PID准入。**截至这次记录，队列在启动源检查阶段；实际核查四manager于新checkpoint ZIP尾部的CephFS读取D态等待，尚不能宣称18张卡已经开始GPU采样。** 正在写入节点fsync并逐SHA核验；不会以等待超时当完成，不杀未知进程、不放宽来源检查。manager已按checkpointpolicy区分新noTC模型与旧zero-context诊断，独立holder_path显式绑定真实占卡脚本，旧六视频的冻结diff源不改。

共享CephFS读取挂起没有因写入节点fsync立即解除。已停止四个受管理启动器，核实八节点没有留下sampler，四个只含空logs/cache的中止目录改名保存，没有删除／覆盖媒体。随后从写入节点经可信SSH/SCP把完整模型缓存到八节点`/tmp/<run>/last.pt`，每节点两个文件的SHA均与原训练checkpoint一致；不重训、不改权重、不绕过strict known-host。manager新增checkpoint_source，将本地推理缓存及其完整SHA严格绑定canonical finished；审核也支持显式checkpoint_cache_root，不因临时路径改变就丢失模型身份核验。缓存分发实测全部通过，报告`music_text_no_tc_cache_distribution_v1.json`。

缓存方式四队列L6Ukl4／viuxk8／gDywn6／Yq3CH5已实际启动：八节点查到18个sampler，N1 GPU5/6/7、N2 GPU4/6/7，N3至N6 GPU6/7，N7/N8 GPU4/5。它们处于装载／采样阶段，不把CPU活进程等同所有GPU已100%计算；余46张卡仍仅holder待用，不计实验利用率。固定三个case未扩大，为两模型原／换／零音乐的18必要对照。

截至上一阶段发布，18视频后续完整decode／noTC无污染／原图VAE参考／来源及采样视觉／review与GitHub发布仍待完成；当时可审阅的是之前六条免训练noTC诊断，不把100更新权重存在当视频效果成功。

## 18条新noTC学生视频完整技术验收与入口复核完成

四组采样L6Ukl4／viuxk8／gDywn6／Yq3CH5均exit0，实际18条。N8完整审核XzOOPJ exit0，`music_text_no_tc_fit8_all18_audit_v1.json` expected=passed18：全124帧decord＋ffmpeg选定视频／音轨解码、704×1280/24fps、实际model SHA／自身长度／全部raw映射、无TC／空白／prefix或外部context、全部native静音四hash和同case视觉VAE参考rows/hash一致。临时模型cache与canonical训练来源严格绑定，没有把复制权重当重新训练。原／换／零音乐长度仍同时变化，不以像素差单独证明音乐语义影响。

主线程实际Read六套reference＋六时刻mosaic，`music_text_no_tc_fit8_s100_v2_visual.json`记录：W/PCA原／换／零三case仍主要后勾腿、手到脸／嘴、交叉臂与有限原地手势；楼梯深色套装／包明显不同于参考浅黄衣／灰裤。采样帧没有建立连续编舞或新音乐控制收益，PCA略低表示MSE没有直接转为可认证舞蹈。这不是新增用户逐视频评分，也不据采样宣称两个模型统计等价。

补充对**新训练模型本身**的三case实际raw／condition_proj／token_refiner比较（W pzrzUL、PCA 3B9Jw7两流水均exit0）：新checkpoint严格context=None而非旧zeros5120兼容，19个原生BF16入口权重不变／0更新，teacher仅比较。两模型report分别`music_text_entry_no_tc_W_fit8_v2_compare.json`与`music_text_entry_no_tc_PCA_fit8_v2_compare.json`。

| 原音乐生成case | teacher raw序列去均值RMS | W学生同统计 | PCA学生同统计 | W／PCA refiner后teacher-student relative-RMS |
|---|---:|---:|---:|---|
| 楼梯 | 2.31742 | 0.00689603 | 0.00160624 | 0.65443／0.63944 |
| 红衣 | 2.27246 | 0.03735854 | 0.00828067 | 0.64696／0.64089 |
| 商店 | 2.25266 | 0.00385155 | 0.00161285 | 0.60662／0.59184 |

原／换音乐共同区间raw RMS差，W为0.38355／0.37766／0.09966，PCA为0.03723／0.07257／0.01589。歌曲改变已能产生数值变化，但学生内部token变化仍远低于teacher，且共同refiner没有使学生条件接近teacher；不能把文字token位置当真实音乐时间，natural前向后common-visible比较不是同packed布局干预。

截至该次记录，评审清单已追加四个NCE1训练配置，43模型／3人物／21预设，旧39模型／旧评分全部保留。第一预设为【无TC联合训练·空间单变量】，第二为【无TC联合训练·音乐反事实】；不是旧PCA教师投影。当时连续人评尚待反馈；**用户随后明确该音乐反事实预设所有配置都只在摆pose，现记为NCE1预设级负评，不再称无人评。** 未展示NCE0不属于该反馈的自动外推范围。

### 有限下一对照：不加数据／宽度，只隔离NCE目标

8train上NCE1已能辨别配对（W87.5%、PCA100% top1），但raw H MSE仍约5、生成case token近重复且视频未建立可用舞蹈，因此不先凭“数据不足”直接扩量。追加同起点／同空间／同数据顺序／同100更新800曝光／同lr与length0.01的NCE0两组，**唯一改clip_weight1→0**，补齐输出空间W/PCA×NCE0/1的有限2×2。代码不加motion loss、不训H3、不改变学生接口；只检验当前token回归信号是否被配对目标影响，0/1 loss标量／梯度也不能直接当Adam真实贡献。两新训练GUyKxx／aqathS在N5 GPU4/5启动后于C300共享文件801MB读取位置D态等待，未开始更新；已停止等待并从已读节点拷贝同SHA C300及其byte-identical finished到N5本地/tmp缓存，改用新目录v2接续，不放宽来源检查。最终模型、固定3case原／换／零及入口比较尚未完成／不得提前判优。完成这组后不自动继续无限预算扫描；宽度或teacher改动须基于结果另外隔离。

用户进一步要求实验前告知资源需求：若不需要全部节点，用户会自行启动其他任务。后续这组只保留N5（28.216.19.137），先用GPU4/5训练，随后同节点排队生成；不再自动占用N1/N2/N3/N4/N6/N7/N8的GPU。N1只保留评审网页CPU服务。占卡holder不算实验工作，也不因此默认把空闲节点保留；如后续确需额外节点，启动前先告知协调。

用户随后澄清独立推理可以多节点并行，资源说明应区分“最低可运行”与“加速建议”。主线程明确告知新NCE0两组18条推理适合八节点18卡，并询问刚让出的节点是否已有其他任务；用户明确选择“还没安排，可用全部”。因此**本轮推理恢复八节点并行**，仍先同SHA完整cache及主机/cgroup准入，不等于未来阶段默认长期占用全群。单节点多GPU安全冷加载代码保留为资源紧张时可选，不据此人为串行本轮可并行任务。

## 最新人评纠正、NCE0真正新一轮结果与当前展示

用户再次明确：原`val_musicreplace_text_oracle_neutral_v1`有舞蹈动作；旧`教师空间·pca256`虽有一些舞蹈动作，但很多肢体液化／模糊／反转问题；W256／LN255／PCA256教师空间在上一轮就完成，不能当新实验。用户对当前【无TC联合训练·音乐反事实】所有已显示NCE1配置给出“都只是在摆pose”的预设级负评。没有逐case数值，也没有对当时尚未显示的NCE0配置给出验收；不扩大反馈范围。

主线程承认前述汇报混用了旧教师诊断与新学生结果，最新NCE0已经采样技术完成但网页未上线，导致用户无法找到新实验。现在已修复展示与轮次标签：原47模型清单的两个新增首预设明确“新一轮·NCE0”，保留旧43模型／21预设与samples/评分映射；旧PCA教师投影仍是历史对照，既不重生成也不称最新模型。

| 阶段 | 明确模型目录/区别 | 当前验收 |
|---|---|---|
| 上一轮教师投影 | `text_projection_C300_w256_v2/ln255_v2/pca256_v2` | 已完成；PCA虽有动作但液化／模糊／反转，不是质量通过；非音乐学生 |
| 已显示无TCNCE1学生 | `text_no_tc_W_fit8_s100_v2`与`text_no_tc_PCA_fit8_s100_v2`及各zero组 | 各8train/8val、100更新；18视频解码通过，但用户明确反事实预设全部摆pose |
| 真正最新NCE0学生 | `text_no_tc_W_nce0_fit8_s100_v2`与`text_no_tc_PCA_nce0_fit8_s100_v2`及各zero组 | 同初始化/数据/宽度/空间/100步，仅移除NCE；另18视频已完整解码、现上线，人评尚未提供 |

所有视频目录位于`diff/logs_omnidance/exp/H3AR/val_musicreplace_<上述tag>`；新NCE0不是只把同一权重换名字。C300完整起点／globalbias转换／MusicFM dense＋完整剩余adapter／自身长度／无TC无prefix／独立视觉VAE／native静音相同。NCE0 W/PCA完整模型SHA分别为`8e392256ef8cd17b198aa31bec3d3c441f1f5681b3672213640530b9ae8905da`和`e12e1d704be263ba1fbf43727cd8ecc2bd344fb2eb8744208d1098567c7e3f01`，不同于NCE1。

NCE0两组bFRWCk／gor86B均exit0、各100更新／800曝光、400dense／41adapter实际改变及完整waveform-only独立worker一致。采样jYyAcq／X1biPT／zNgUIr／Lu1FOB四队列均exit0，在用户本轮明确允许八节点后并行；后续CPU完整审核BU7u5x expected=passed18，全124帧视频和音轨decode/noTC条件及同图VAE参考与静音hash通过。报告`music_text_no_tc_nce0_fit8_all18_audit_v1.json`。没有新的训练预算或再生成旧W/LN/PCA教师视频。

| 同8train/8val | W NCE1 | W NCE0 | PCA NCE1 | PCA NCE0 |
|---|---:|---:|---:|---:|
| train raw MSE | 5.45656 | 5.33725 | 5.27885 | 5.16737 |
| val raw MSE | 5.24840 | 5.18988 | 5.14308 | 5.13436 |
| val长度MAE | 110.625 | 177.625 | 107.625 | 125.375 |
| train三候选top1 | 87.5% | 75% | 100% | 100% |

移除NCE略改善回归，但raw H仍未接近teacher，长度val误差更大；仅8val，不能用PCA NCE0某项较高top1宣称泛化或舞蹈改善。主线程实际Read六套NCE0原/换/零mosaic，仍可见勾腿、手到脸／前伸手／交叉臂与原地小动作，楼梯仍深色服装漂移；`music_text_no_tc_nce0_fit8_s100_v2_visual.json`只记录采样观察，不替代最新连续人评，也不宣称全配置统计等价。

### 输出头尺度／可达范围的新数值证据（不是新视频）

`music_text_no_tc_head_range_v1.json`为四个实际新noTC checkpoint的0更新诊断，P2Hkbr exit0。计算当前完整输出投影的有效affine中心/255秩映射，并把LayerNorm归一化、零均值内部向量放宽到半径sqrt(256)=16闭球，FP64 SVD与trust-region根求最近输出。CPU各向同性解析解／各向异性KKT和半径边界通过；实际KKT残差约1e-13。closed ball是放松下界，忽略FP32roundoff及所有上游音乐/query可达性，不当精确音乐学生最优值。

PCA固定空间教师MSE楼梯／红衣／商店约0.86075／0.71350／0.71105；当前PCA NCE1输出头的LN闭球下界却为5.18790／4.94552／4.79114，NCE0为5.10820／4.85327／4.71674。无范围约束有效affine包络仍约0.72–0.89。**更好的PCA方向并不代表这个具体输出头的幅度/参数化已经能覆盖teacher；只做投影后维持旧头小幅训练没有解决可达范围。** 这并非LayerNorm独立根因，也不说明任意256维模型不可能；输出weight/gamma可训练，当前固定参数快照的尺度、中心及优化需一起核实。后续应先设计有明确幅度覆盖的输出初始化/参数化对照，而不是在同坏范围上无限加数据或反复汇报旧教师投影。

最新服务同8766：`manifest_music_text_no_tc_nce0_fit8_s100_v2.json`，47模型／3人物／23预设，首预设【新一轮·NCE0对照·非旧教师投影】按同空间NCE1/NCE0排列，次预设【新一轮·NCE0音乐反事实】。HTTP清单200、最新18视频Range206及旧43模型/samples/21presets逐值保留已核验，报告`music_text_no_tc_nce0_fit8_s100_v2_review_access.json`；不称全部126历史链接本轮重新检查。用户无需再次review旧教师投影。截至该次上线，最新NCE0反馈尚待提供；后续用户已明确该新预设全配置也摆pose，见下节，不再等待此预设评审。

## NCE0最终人工负评与本轮详细执行协议

用户对已经上线的【新一轮·NCE0音乐反事实】明确反馈：“所有配置都像是在摆pose而非舞蹈”。这是该预设的连续定性负评，不是逐case数字；现NCE0与NCE1两个预设均未建立可用舞蹈，不再等待此评审。模型更新、原条件接口与音画技术审核完成，不等于音乐编舞目标完成。

### 实际实验矩阵及起点

本轮四配置为固定W256/PCA256空间×NCE权重1/0，各从同一完整C300 checkpoint SHA189a0a74…开始，并非NCE0恢复NCE1最后权重或optimizer。所有共享音乐骨干和adapter权重相同，context_projection.weight移除、其bias重命名为可训练global_context_bias，转换前后新None输入预测与旧zero-context逐值相等。四组均新建AdamW，不恢复C300 moments；W/PCA中心与方向处理不同会改变训练前的输出，初始表示指标分别报告，不能称四组初始条件token逐值相同。NCE0/1同空间的初始化、数据顺序、更新预算一致。

当前MusicFM实际dense可训练参数316343296（400tensor）；新完整剩余adapter3655937参数（41tensor），比旧adapter少context_projection.weight的1310720参数。每组width256、8attention heads、两cross-attention+FFN4width，qmax1423（来自原1800train最大teacher长度，不是固定有效长度）。512/1024宽度只是实现/CPU验证过，本轮没有这些宽度的训练或视频结果。

### 数据、教师与音频

训练使用原1800train前8条、验证原200val前8条，完整ID在各plan.json；生成楼梯/红衣/商店3case均不在该1800/200 split。C300起点已按原1800数据训练过，本轮8条不是从零训练的新数据实验；验证8条也不是全新测试集。teacher raw H由当前参考图及舞种＋中性原版MotionScript经过离线冻结TC得到，正文labels包含词序/措辞和当前图上下文；学生不读文字、图TCcontext或GT长度，仅音乐波形。图只走下游视觉VAE；这使学生拟合的teacher仍包含其输入不能唯一推断的部分，需明确是一项潜在目标失配而非已确定的唯一根因。

共享loader读取前5.0625秒音乐、mono复制为双声道、重采样32kHz、截取/补齐[2,162000] FP32，不做波形音量归一化；音乐模型再按相同方法重采样24kHz，mel固定mean/std→conv→Conformer，约25Hz×1024通道特征，双声道保留。训练始终直接走真实可微dense音乐骨干，不用冻结特征缓存。完整teacher/statistics bundle仍来自1800/200原数据，PCA/mean/std/energy/qmax只用1800train拟合，不把本轮8train说成全部信息源。

PCA中心为原1800train的776078个有效tail token的FP64均值；每item最多16个等距token，共28800，按长度/抽样数加权，FP32随机低秩q288/niter4/seed42取256方向、FP64正交化。无val、图prefix、padding或3生成case参与空间拟合。W组来自固定C300 raw_output列空间，中心其bias，不随此阶段训练重新求基。

### 训练前向、mask、loss与优化

音乐features→256宽memory/stereo/位置编码→1423query＋globalbias→两cross-attention/FFN→LayerNorm→可训练Linear256到5120，产生u；之后固定空间变换H=mu+((u-mu)B)B^T。basis B及中心mu持久buffer不训练，完整raw_output及LN等剩余adapter与MusicFM dense仍共同训练。**此实现是对可训练raw输出施加固定投影，不是直接预测自由PCA系数，也没有重新初始化输出头的方向/幅度到PCA教师统计。** 没有删LN、无新旁路、无query self-attention。

每步包含全部8个训练ID（随机排列），4次microbatch2累积梯度后clip/optimizer一次；每ID正分支100次，共800曝光，不是100×2条。三个音乐分支为正确/第一wrong/第二wrong，各完整音乐前向；teacher正文与长度只回归正确分支。wrong选择不同ID及caption hash，不保证完整源歌曲隔离或语义不同。NCE是每样本三个候选，不是batch8×8对比矩阵；NCE0依然前向并计算NCE以保留原代码，只把其loss系数设0。

原loss=raw MSE/Etrain + .01*log-length MSE + lambda_NCE*min(step/100,1)*NCE(scores/.07)。lambda0的新模型正是去除最后项，不改变其他权重。raw MSE按教师有效token mask对全1423query输出比较，跨microbatch按有效token数加权；length/NCE按样本数加权。teacher长度是监督label，绝不决定student有效tail。NCE只对loss做train mean/std及L2归一化，学生部署H保留raw尺度；双方分别最多均匀64个有效token，双向soft-Chamfer相似度聚合不显式要求逐位置一一对应。正文token索引不是音乐时刻。

学生length_head预测logL，round(exp(clamp(logL)))裁到1..1423，然后取H[:L]及mask；train/validation/deployment共享。离散截取不对length可导，长度头仍依赖其loss；raw MSE覆盖全部teacher有效位置，即使student实际tail短，仍是监督规则而不是把teacher长度喂生成。不能把teacher MSE监督mask完全等同自然部署有效mask。

优化器：MusicFM dense lr1e-5、完整adapter lr1e-4、AdamW weight_decay.01、默认betas(.9,.999)/eps1e-8、global gradient norm clip1，无lr scheduler。NCE1有100步ramp，NCE0为0；FP32参数/前向，禁止cachedfeature FP16roundtrip；音乐骨干eval固定BN统计/随机层但dense参数仍求梯度，adapter train且attention dropout0。每组最后核验400dense/41adapter真实改变、固定空间及冻结buffer不变。完整V3 checkpoint保存音乐骨干＋所有adapter＋basis/center/globalbias；独立Python worker仅输入waveform，train/eval/fullreload/worker自身tail与length逐值一致。

**本轮训练不加载H3 DiT/VAE/TC，未用GT视频latent，不加FM、预测一致性、motion、pose或偏好DPO loss。** 之前的冻结H3预测监督是另一个6更新历史实验，本轮只做raw H/length/NCE表示监督，不能把旧行为监督当此次每步生成端loss。

### 实际生成配置、参考图与静音旁路

生成是音乐→同完整checkpoint→自己预测H和长度→BF16cast→全1文本模态tags，与原视觉VAE图reference及两路固定静音条件交原PackedSequenceBuilder→冻结DiT condition_proj5120到5376→原两层token_refiner→50个DiT主blocks→视频VAE解码。音乐H不经过TC，TC/processor/tokenizer权重与对象都不加载；prompt/negative均None，prefix0，不是空白字符串。两层refiner是文本教师与音乐hidden共同的DiT入口，不是textencoder第二遍。

参考图标准输出704×1280，图reference短边2048、实际prepared2048×3712、视觉VAE7424行；图不是学生adapter输入，只是生成器视觉条件。CFG1、bi双向、50步、video shift12/audio shift3、seed42、124帧24fps、输出704×1280。原生reference/retake音频均5.0625秒静音经冻结audioVAE，effective203、目标207、尾repeat_clean、mask全0；不是直接填0latent。真实音乐只经学生H分支影响生成，最终再mux原/换曲。音轨能播放不证明音乐控制成功。

每模型原3/换3/零3共9；NCE0新18另NCE1前18。原与换保持同图同seed，零仅学生音乐置0，最终音轨仍原曲；输出长度原换零分别：W楼梯420/297/130、红衣248/470/130、商店528/386/130；PCA楼梯430/333/218、红衣314/429/218、商店461/407/218。换歌同时改H内容、长度和packed/RoPE位置，所以视频变化不是纯音乐语义贡献的隔离实验。实际都是小动作的人工负评优先于像素差。

### 实测与为什么不把失败简单归为数据少

四组train raw MSE约5.17–5.46，val5.13–5.25，均值位置baseline val4.81861；NCE0只是略降MSE，没有建立连续舞蹈。只有8train/8val使结论受限，但训练集本身仍拟合差，不能仅靠加数据推定修复，更不能称100步已收敛。meanMSE/正歌辨别好不代表teacher语义或有效舞蹈H已复现。

已完成的新NCE1原生入口0更新比较是三review case，raw5120/proj5376/refiner均保存指标；三个case student去token均值变化远低teacher，相对差异约.59-.65，经refiner后没有接近。**NCE0尚未完成同样的新模型分阶段入口forward，不将NCE1数值冒作NCE0实测。** 对四组已有head范围诊断才包含NCE0，且其对象仅输出层放松下界而不是实际音乐预测。

在固定当前PCA模型参数下，裸PCA space投影teacher MSE约.71-.86，无限制head affine envelope约.72-.89；但包含LN零均值norm<=16闭球放松范围，下界约4.72-5.19。因为实际形态H=mu+P[b+W beta-mu+W diag(gamma)z]，z零均值且norm<sqrt(256)，PCA投影改变空间却没有让已学W/gamma/中心覆盖所需幅度。这里不是gamma总norm固定（它可训练），也不是说LN永久无法输出teacher；它只证明当前checkpoint具体头的范围存在障碍。输出weight/gamma可移动，后续该先核验更合适头初始化/尺度/参数化，再独立检验音乐预测及行为，而不是继续把旧PCA教师有动作当新学生已学到舞蹈。

最新结论：原teacher有舞蹈；PCA teacher有一些动作但液化/模糊/反转，不可作为完整质量成功；无TC的NCE1与NCE0新学生反事实均被用户判摆pose。本轮受控实验流程完成且得到负结果，整个仅音乐＋图可用舞蹈目标尚未完成；后续新修复预算与资源先明确告知，不无限重跑已失败的相同配方。

## 新阶段：纯文本逐token教师、单配对分支和正式数据规模

用户在六项质疑后明确允许推进，但排除28.216.18.136和28.216.18.144。本轮只调度28.216.19.69、28.216.17.70、28.216.19.213、28.216.19.143、28.216.19.137、28.216.19.225，旧被排除节点的缓存不用于新计算。

本节为实现／准备计划，不将旧投影或8条训练模型当新结果。三条并行线：

1. 纯文本教师：仅舞种＋中性原版MotionScript正文，经冻结H3文本LM（final norm Identity），add_special_tokens=False、不用presentation模板、不输入图或原生音频结构，保存有效逐token BF16 raw H，不mean-pool、不保存prefix_context。先三case新的真实教师H直入无TC原管线＋独立视觉VAE图参考检验，再六分片覆盖原1800train/200val。新的教师表示不同于旧图上下文tail，必须实际验证，不能仅shape相同就宣称可用。
2. 单配对训练：clip_weight0仅正音乐一次前向，取消两个wrong分支及NCE计算；反事实只留验证／视频。纯teacher bundle仅targets＋原共享波形及train-only统计，学生contextNone／无TC，无固定W/PCA后投影，不新增生成loss。qmax来自纯文本train实际长度；val超长截取监督及无法预测长度明确报告，不让val扩大query上限。
3. 起点对照：旧C300派生noTC、width256末端LN只作为控制；新模型从公开MusicFM预训练骨干＋新width512、末端Identity的完整adapter开始，raw_output5120自由可训。用全部train教师mean／centered energy及预定前32train音乐query probe做输出头均值与全局RMS初始化校准，不固定PCA方向、不宣称协方差／全部音乐目标已匹配。它是候选配方整体（起点／宽度／末端归一化及初始化不同），不能把效果差异只归于某一个因素。

资源布局：六节点各GPU6编码一个纯teacher分片，N4GPU7先编码三case；随后N1–3GPU7各生成一个纯teachercase，避免重复原图上下文教师。N5GPU4/5准备／并行训练两起点，原1800train/200val正式预算各1200更新、batch8/micro2、9600正配对曝光约5.33遍，lr骨干1e-5/adapter1e-4、length0.01、AdamW/clip沿用。小量10更新smoke只检验真实梯度和部署，非8条数据效果主实验。训练正式样本不再限8条，原/换/零推理仍每配置固定三个case。

启动前核验源SHA/严格checkpoint／GPU实际进程与180GiB主机/cgroup准入，不用holder100%伪装实验利用率；阶段有依赖时明确正在准备／GPU是否已启动。图参考只下游VAE，学生推理完全不加载TC/tokenizer/processor，也不读取teachercaption/长度。后续全200val、完整token/长度/源模型部署一致及3case视频和人评分别验收；纯textteacher无法给出目标舞蹈时先报告边界，不靠扩大音乐训练掩盖接口问题。

### 2026-10-21 实际阶段证据与消费侧修复

- 原1800/200 split SHA `44b07b8b9219b5cbced271ac36ce7e1d26204fe6f47983a113a9dd8585720060` 与旧目标使用的6200条caption源核对，不能改用不一致的另一个2k集合。从真实6200源提取exact2000，纯caption JSON SHA `8d9cbaa84327b677f7838483456d08c7cf0821a12018c91f3d822fd2e9c05b82`。真实tokenizer全量preflight得到train最大1423、val最大1113；qmax只由train决定。
- 六个纯教师分片的采样任务均exit0，合计2000，位于实验MS目录的`music_text_pure_text_targets_v1_s0..s5`，队列报告为`music_text_pure_text_cache_jobs_v1.json`。原生H3只加载保留50层LM所需551个BF16 tensor，final norm Identity，实际加载tensor content SHA `c2ed2ea979f58106b393d2514b15d1f237dadb49a1a44d03f0c20d0a2be541cb`。这里没有图、presentation或mean-pool；队列exit0不能替代消费者对全部ID/token/tail与来源的最终审核。
- 新三case纯教师H真实生成完毕，长度228/199/323，输出`H3AR/val_musicreplace_text_pure_text_teacher_v1/joint`。它不是上一轮图上下文教师或W/PCA投影。独立报告`MS/music_text_pure_teacher_three_video_decode_v1.json`已验证三条均704×1280、24fps、124帧完整解码及音频完整解码；ffprobe不可用，精确声明stream数量未独立验证。来源/noTC/视觉参考审计与连续舞蹈人评另行验收，**不能据此宣称纯教师舞蹈质量通过**。
- 第一版纯数据包消费者曾重复定义loader，且假定cache含`caps_path/cache_source_sha256/text_embed_source_sha256`等实际不存在字段。修复消费者，使用外部冻结`music_text_pure_text_teacher_release_v1.json`绑定真实caps/split/cache/text_embed路径与SHA，不修改已运行cache添加字段。release中的`cache_complete:false`是编码前快照，保留原记录；后续完成情况用新的真实任务报告证明。
- 纯bundle不绑定某一个init，因为初始化依赖其train-only统计；外置init manifest严格绑定bundle/checkpoint/同名JSON报告SHA、split/qmax/music-only/free-head协议，避免循环来源依赖。纯bundle包含明确`artifacts`来源记录，是9个顶层键；旧bundle保持8键严格协议。训练消费侧单独兼容pure9，拒绝额外student features或篡改来源。
- 初次CPU assemble在`28.216.19.137`尝试构建`/tmp/music_text_pure_text_inputs_v1.pt`，只复用旧本地bundle的已审计原始波形，不沿用旧带图targets/statistics/init。该初次尝试因下述CephFS读取阻塞终止；后续本地镜像构建v2、初始化和正式训练均已完成，详见后面的终态记录。日志`MS/music_text_pure_text_inputs_v1.log`保留初次真实IP和冻结源SHA，不把这一旧准备日志称作当前训练进度。
- 开发代理出现网关Empty stream与429限流，失败调用不作为新的代码、报告或审核完成证据；已存在的真实缓存/视频及独立解码报告不因代理失败被改称失败或重新生成。联调修复及验证继续进行，完整训练、200val、新学生视频、人评和GitHub发布仍为后续停止条件。
- 首次CPU assemble实际进程在CephFS的`s0/hwikIbHdfqCnONEW.pt`读取处进入D态、`generic_file_buffered_read`，尚未写出纯bundle，不能称训练中。已停止工具管理任务并对核验cmdline的本任务remote PID发送SIGTERM，保留日志。改从六个分片原写入节点并行传输PT到`28.216.19.137:/tmp/music_text_pure_target_mirror_v1`，逐文件要求字节SHA等canonical finished记录；只更换读取介质，不修改teacher或降低来源门槛。首传输命令因Path拼接TypeError在复制前退出，修正括号后重启。实际新传输已全部exit0且2000个PT逐文件SHA等原finished，报告`MS/music_text_pure_target_mirror_transport_v1.json`；各分片plan/finished也逐字节保持。消费侧`--target_read_root`独立CPU fixture拒绝本地plan/finished/PT篡改、缺失/额外PT和canonical路径改绑；fixture不是实际LM或2000目标验收。后续实际本地assemble已完成`/tmp/music_text_pure_text_inputs_v2.pt`，冻结pipeline SHA `282c9e56093272bf3916290591a5b3cc5b015d3a50b980336ea9947fa9f5b2b2`，独立SHA与完整报告已验收，详见下述v2唯一消费选择；误重复v1也保留但不使用。
- 新纯教师三视频实际完整来源审计`MS/music_text_pure_teacher_all3_audit_v1.json`已complete/3passed。noTC/prefix0、纯teacher来源及图独立视觉VAE分别核验，完整媒体解码复用同SHA报告。官方上游pipeline未有生成前冻结SHA，报告明确`official_pipeline_pre_generation_hash_bound:false`，当前SHA只事后核实，不冒历史来源证明。连续舞蹈人评pending，不把采样mosaic当人评。
- 新评审清单`review_web/manifest_music_text_pure_teacher_v1.json`已实际上线8766并HTTP200验证48模型/24预设，旧47模型/23预设逐对象保留、三人物完全相同、未写评分。首预设【新纯文本教师·无图上下文】展示新pure／原neutral／旧无TC教师三人物joint，标明教师诊断非音乐学生、人评待定。新清单SHA `6055d172d0fb2c69f3f9670dd023f0290c6330ea02a24c05c426a39194b71368`。
- 已独立核实当前训练源SHA `63ec0cbd334a2078d43208b5e58fc2de49122c933bd627c91d1567b6351b405c`与`results/ms_20261020/single_pair_main_pure9_cpu_audit_20261021.json`一致：真实AST的pure9/legacy8通过，28项额外键或来源篡改拒绝；完整Identity checkpoint/None输入worker仅CPU fixture通过，不是实际MusicFM/H3/GPU验收。
- 读取恢复新增`load_pure_targets(..., read_root=None)`与`--target_read_root`。canonical计划、完成记录、cache元数据和源路径不变；镜像plan/finished必须同SHA，每个实际读取PT必须等canonical finished的file SHA，额外/缺失PT拒绝。只记录读取介质，不重新生成或修改冻结teacher缓存。源文件语法与diff检查通过，镜像专门CPU回归及actual2000 bundle另行验收。
- 当前组装/来源审核不需要GPU。下一阶段两起点各用一张卡训练，其余资源可供其他任务；之后三个固定case的独立生成再跨允许节点并行。节点holder的100%利用率不是本轮训练进度，正式训练开始必须以实际梯度、step日志与checkpoint为证。
- 纯教师三条的实际来源/无TC/token IDs/BF16/独立视觉VAE参考审核完成：`MS/music_text_pure_teacher_all3_audit_v1.json`为3/3，marker与六时点mosaic在新生成目录。官方pipeline源码只有posthoc内容SHA核验，缺生成前冻结记录，明确不称预冻结；原曲mux相符尚未独立核验，连续舞蹈人评待定。
- 新评审manifest `review_web/manifest_music_text_pure_teacher_v1.json`实际构建并启用8766，48模型/24预设，原47/23与样本记录逐值保持。新预设为【新纯文本教师·无图上下文】，对照原neutral教师与旧无TC教师；网页manifest实际HTTP200已核验。本次只新生成3条纯教师诊断，不是已训练音乐学生视频。
- 六分片2000个PT的同字节SHA镜像传输全部实际exit0，补齐原plan/finished同字节副本。`MS/music_text_pure_target_mirror_transport_v1.json`证明transport，非训练；镜像读取CPU fixture回归通过。主线程误启动v1/v2两次同源assemble，两者均exit0；保留产物不删除，后续唯一使用v2。实际v2 bundle SHA `f0a0b9e62ad2d882fde7405641cff08c4fd8cdbb9db12d018df7b48318a11672`，2000/1800/200、qmax1423、teacher energy127.65884048888763、val overflow0，独立全文件SHA核验完成。
- 两实际GPU初始化已exit0：19.137 GPU4的public512Identity SHA `103cf3a894e086d5d8221d8b1ab16c710aa85e5708dda4af0db8426fdd95d28c`；GPU5的C300control256LN SHA `3bc609d0253699a7ebee8975bef5a4991fc0f8ed26676442f4e724b9fa9c7149`。public32train waveform实际MusicFM前向校准scale149.94115，centeredRMS从0.07183137到10.77047834，对teacher10.77047842，meanerror1.62087e-6、RMSrelativeerror6.59475e-9；仅头W/b改变，没有拟合协方差/方向或训练更新。完整权重buffer序列化检查与外置manifest绑定已通过，两者optimizer updates均0。
- `BhhOeU`的两真实10步smoke已完成，各80配对曝光，400dense tensors真实更新，public39／control41 adapter tensors改变，冻结buffer不变、完整重载与独立waveform-only Noneworker逐值一致。随后从各原init重新启动正式1800train／200val、各1200更新，不继承smoke权重或optimizer；真实step日志、完整plan、首步gradient与两livePID194349/194355已核验。main误重复启动的`WM1uGf`在准入即被这些正在计算的正式进程正确拒绝，零新增更新；没有放宽准入或终止正常训练。两配置随后均正式1200更新结束，全200val终验和完整checkpoint/独立worker已完成；18新学生视频及连续人评仍待完成，不能用训练完成代替质量验收。
- 独立旧源CPU只读全2000审核`MS/music_text_pure_bundle_cpu_validator_v1.json`已自然exit0并complete，SHA `a0a8bc5d07c7df8ca4252cf3ae3ae257b111bbcfd898aede598307dd900274e5`。全860146 tokens、train统计776078 tokens，实际full tokenization／source／BF16／split六shard门禁通过。报告明确执行源2ae2cd与后续磁盘282c镜像源不同，不冒称此报告测试mirror分支；mirror独立fixture和实际v2assemble分别验收。

### 2026-10-21 纯教师连续人评与下一阶段边界

用户明确反馈：【新纯文本教师·无图上下文】三个配置均在跳舞；旧noTC·教师tail-only·未训练新接口动作略小且重复，另外两个（新纯教师／原neutral）动作几乎一致；三者运动均偏小且未很好跟随音乐节拍。此为预设级连续定性人评，不是逐case量化。此前immutable技术报告中的人评pending是审核时快照，此处记录后续实际人评，不修改报告历史内容。

该实验验证“无需图上下文的文本raw hidden仍可作为舞蹈教师，生成端完全关闭TC且参考图仅走视觉VAE”在三个case可用，不证明音乐学生已学会，不证明强运动幅度或节拍同步。教师生成实际native音频固定静音、原曲仅mux，因此本次上界没有提供真实音乐影响生成的证据。

MusicFM输出约25Hz×1024声学特征，H3入口要求可变长×5120原文本LM语义hidden；需要可训练接口映射，但两cross-attention、qmax1423和内部width512是当前方案选择，不是H3规定。文字token位置不是音乐时间位置，逐tokenMSE也不是拍点／动作时间监督；同音乐配多种合理舞蹈可使配对回归平均化。这些是任务错配的可能机制，不能把本次教师跟拍弱简单归因adapter。

下一步优先完成已有两配置1200更新／full200val，按固定三个case生成每配置原曲／换曲／零音乐共18条，分别验收连续舞蹈、运动幅度、音乐变化与跟拍，不把H MSE改善当目标完成。如学生恢复舞蹈但不跟拍，下一项问题才是音乐时间／节拍到动作强调的显式对齐；新增生成行为或节拍loss需另明确方案与预算，不在当前训练中偷偷加入。

控制组C300control256已实际1200更新／9600曝光完成，全200验证与完整独立worker一致性通过，checkpoint SHA `39dc560334355812f2984f727c9b05f46122c60652a70ab8bcd166f1df47fe89`；tokenweighted rawMSE117.6411，position baseline macroMSE5.83669，音乐判别top1 .36，未建立音乐收益，必须继续视频诊断。public512随后也实际1200更新／9600曝光完成，全200验证和完整独立worker一致性通过，checkpoint SHA `8c3f1133d64d194b7aeb89850a3137afcff23558a9b13ab6756d1f1c68d85fa2`。public tokenweighted rawMSE90.7060、macro128.1719（control117.6411／167.7893），正确对wrong分数差约3.89e-5，top1 .325（control .36），长度绝对误差171.22（control157.44）。纯teacher position_mean baseline macro5.83669优于两学生，但不是生成质量证据。两组音乐条件区分增益未建立，不能将新输出头尺度校准成功说成语义拟合成功；两组最终视频质量均未验收。

两完整checkpoint已归档至canonical MS目录，并在允许节点保持同SHA本地副本。首次MEN82P从19.137运行生成manager，其本地knownhosts缺其他节点可信RSA记录，strict检查拒绝全部采样（零视频）；保留v2失败目录与准入日志，不关闭hostkey检查、不抓未验证key替代。恢复为有原已验证knownhosts的19.69主机本地调度，生成目录使用v3，模型仍同v2。public三节点19.69/17.70/19.213，control三节点19.143/19.137/19.225，各GPU7，原曲／换曲／零输入共18，未使用18.136或18.144。

### 2026-10-21 正式学生生成、技术验收与评审边界

`EN81cc`的两配置9条队列均实际exit0，18视频完整生成，不再称仍训练或生成未结束。`uHJqvj`复用原`music_text_diagnostic.py --online --no_text_encoder --checkpoint_cache_root /tmp`完成独立18/18音视频全解码、sameSHA完整checkpoint、实际noTC日志、自身长度、全1text tags条件shape、四路native静音hash、同图独立视觉VAEhash核验；报告`MS/music_text_pure_students_all18_audit_v3.json` SHA `3c61412f373ac0e9cd75e72baebf7402735c3e2d4eaaa4d8a03ec134ca909528`。technical passed不代表连续舞蹈或跟拍通过。

学生预测长度（原／换／零）：public楼梯304/307/309、红衣304/307/309、商店307/306/309；control楼梯391/411/108、红衣440/323/108、商店409/625/108。模型/换歌/零条件同时改变hidden与长度，packed/RoPE位移仍是混杂因素；像素差不能解释音乐语义收益。

参考图＋六时点mosaic已实际查看：public/control楼梯生成均变深色外套/黑裤/肩包，相比浅黄上衣灰裤参考存在显著服装漂移；红衣有抬臂／手部姿态变化，商店躯干与脚部位移较小。此仅采样帧视觉观察，不冒作连续音画人评，也不能据此独立宣称所有学生均摆pose或通过舞蹈。学生连续舞蹈、运动幅度、跟拍与原/换/零效果目前待用户评审。

`music_text_review.py --pure_students`实际使用18/18报告构建`review_web/manifest_music_text_pure_students_v3.json`，52模型/26预设，原48模型/24预设与3samples逐值保持、不改历史评分。新两个预设【新纯监督·正式1800】和【新纯监督·换音乐】分别显示原/零对照和换音乐，对模型清楚标明非旧8样本投影。已存在教师技术报告的人评pending仍是历史快照，最新教师人评见上节；学生人评不能由教师结论外推。

实际review CLI正常输入与直接函数联调通过；`eNS4Sa`额外在内存改写已完成报告的10个顶层字段（complete/expected/passed/noTC/native哈希声明/GT输入/完整decode/quality声明/重复models/缺rows），全部拒绝且不产出新manifest。不是重解码或全量故障注入，不用未返回的代理fixture报告替代。新服务8766实际HTTP200、52/26与全部18视频Range206已核验。所有本轮新GPU计算已结束；当前只剩CPU报告/评审服务与发布，不默认继续占用六节点做新训练。

后续补充独立音轨审核`results/ms_20261020/music_text_pure_students_audio_mux_audit_v1.json`（SHA `a00665603ad8711403c6541da69c107e1f6249493e103f3746096e34b5a3130f`）。CPU任务`sOQKSv`真实exit0：使用各视频声明的原曲／换曲音频，经相同共享波形loader及实际PyAV `_prepare_audio_stream/_write_audio`在内存重编码预期AAC；与现有MP4逐个非空AAC packet payload及codec元数据比较，18/18完全相同，每条160个非空packet。由此独立确认6条原曲、6条换曲、6条零音乐学生输入控制的实际配乐来源正确（零输入控制仍mux原曲），不是仅检查日志或文件名。没有重新生成视频、没有用GPU、没有修改旧技术报告。

该检查只证明音轨内容与来源，不检查音视频动作跟拍或AAC packet时间戳／容器编辑列表。送编码器的波形长度162000samples／32kHz=5.0625秒，视频124帧／24fps=5.16667秒，内容长度相差约0.10417秒；不把AAC padding视为真实音乐，不据此判定动作时间对齐。当时学生连续人评尚未取得，保留该审核快照；用户随后提供实际学生人评，最新结论见本文最后的“正式学生最终人评与接口澄清”。重复stop hook不是人评，不授权扩训练或伪造质量结论。

### 原生H3文本、音频分支及历史音乐encoder接口的详细对照

以下以当前保留的DiffSynth实现为依据，不冒称复现H3预训练配方。`sample_h3_ar.py`是我们的采样器，与上游完整视频＋音频联合采样分别说明。

#### 原生文本分支：多模态presentation，不只是动作文字

1. Ref2VA的输入先被组织为引用块。文本presentation典型形式是`<Picture 1>: <vision_start>...image_pad...<vision_end> <Audio 1>: 动作prompt`；带声reference视频还包括时间戳和video视觉tokens。音频在该presentation中只有引用名称，**没有波形或声学特征送进TC**。实际内容进入音频VAE分支。见上游`diffsynth/models/minimax_h3_text_encoder.py:172`及`diffsynth/pipelines/minimax_h3_audio_video.py:260`。
2. TC为Qwen3-VL视觉模块＋H3保留的50层language model，hidden width5120，LM末端norm为Identity。参考图通过vision encoder成为LM视觉embedding，所以同一张图会在TC中提供语义信息；LM在整段presentation上运行，动作正文hidden已经条件于前面的图与结构。返回完整`last_hidden_state[0]`，shape `[L_TC,5120]`，不是文本词表ID、不是输入embedding，也不是池化向量。见`minimax_h3_text_encoder.py:16,73,106`。
3. 整个TC输出（含vision block的hidden）占据`text_pos`，全部走`condition_proj:5120→5376`，再经两层token_refiner和其末端RMSNorm。vision block的tag0与普通文本tag1影响主DiT模态调制选择，不把TC视觉hidden转去video_patch_proj。refiner只对这一条件序列做attention／MLP，无RoPE、无timestep调制，不是再次经过TC。见`diffsynth/models/minimax_h3_dit.py:186,287,301`。
4. 同一参考图另经video VAE输出24通道视觉latent、patchify为96维row，作为独立`ref_visual_anchor`输入DiT。这与TC图上下文是两条独立表示；去掉TC不等于删掉视觉VAE参考，也不等于把目标首帧硬固定。见`minimax_h3_audio_video.py:494`。

#### 原生音频分支：40Hz／32维声学latent，reference与retake是两组token

1. 管线将音频重采样32kHz，mono复制为stereo，保留前两声道；VAE本身不负责识别外部采样率。encoder总hop为800，因此每声道40 latent/s。波形 `[2,N]`经encoder→pre_block→mean_proj得到 `[2,32,ceil(N/800)]`，再按原32通道常量均值／std归一化；编码使用posterior mean，不采样。见`minimax_h3_audio_vae.py:414,447,458`。
2. **Reference audio**：经`_encode_audio_ref`压缩并pack，成为固定`ref_audio_anchor`。stereo按声道分别展开，`[2,32,T]→[2T,32]`，不是64维row。见`minimax_h3_audio_video.py:509`。
3. **Retake audio**：同一配乐另调用原音频encoder，提供目标输出音轨的`input_latents_audio`与`denoise_mask_audio`。mask0是保留输入clean位置，mask1是重新生成的位置；短音频重复最后latent补齐，并在上游默认将补齐尾标成regenerate。它与reference不是同一组tokens。见`minimax_h3_audio_video.py:401`。
4. 124帧24fps对应目标音频207 latent位置。我们的固定5.0625秒32k音乐为162000samples，编码后203位置，因此会有4个补齐位置；`repeat_clean`把mask尾部也设0，使这些位置使用重复clean尾，不是人工填零latent。见`sample_h3_ar.py:258`。
5. reference音频row与目标音频row都走`audio_patch_proj:32→5376`；visualrow走`video_patch_proj:96→5376`。三模态输入拼成同一DiT序列，Ref2VA排列为`[TC条件 | 引用图／音频／视频块 | target_audio | target_video | padding]`。音频有40Hz时间坐标及左右声道位置，视频有自身时空RoPE坐标，不是丢掉时间只用一个音乐向量。见`minimax_h3_audio_video.py:700`及`minimax_h3_dit.py:301`。
6. 原生双向主DiT的50个block使用联合self-attention，视频Q可以读取音频K/V，音频也可以读取图和文本。主块AdaLN的shift/scale/gate由**timestep embedding**产生，按token模态video/text/audio及每位置noise时间选择；不是由mean-pool音乐向量直接产生。reference／retake clean位置的时间t=1，生成视频按当前noise时间更新。见`minimax_h3_dit.py:124,166,214,370`及`minimax_h3_audio_video.py:968`。

#### 官方完整采样与我们的实际采样区别

上游`MiniMaxH3Pipeline.__call__`每步同时step视频和音频latent，最后分别解码video/audio（`minimax_h3_audio_video.py:180`）。我们的`sample_h3_ar.py:343-403`取DiT视频预测，只step视频，最终mux输入原曲；audio预测被丢弃。因此音乐不被重新生成，但clean reference／retake音频tokens仍通过联合attention影响视频。`bi`描述注意力／视频采样模式，不等于完整上游joint audio-video采样；本轮没有AR块因果限制。

#### 历史阶段A：训练H3原生音频VAE编码侧

训练路径是`音乐→原生encoder/pre_block/mean_proj→normalized [2,32,T]→时间mean/std池化128维→训练head→5120维L2 embedding`。文字侧是冻结H3文本LM的动作caption均值hidden，经train-only mean/std和L2作为CLIP目标。优化音频编码侧及训练head，冻结文本、DiT、video VAE与音频decoder；不是对动作视频latent做FM。对称InfoNCE监督全局音乐／动作caption对应关系。推理只将训练过的native_audio_encoder权重装回原VAE，**不执行5120训练head**，仍输出32维时序音频latent供原管线reference与retake两处使用。见`ms/native_audio_full_clip.py:64,88,94`及`sample_h3_ar.py:176`。后续“冻结训练head继续encoder”是检验收益被训练head吸收的另一个控制，不改变生成接口。

#### 历史阶段B：专用MusicFM／MuQ替换原音频latent

音乐32k→24k→专用dense特征约25Hz／1024维；双声道分别编码。`TemporalLatentAdapter`用LN1024→Linear256→GELU、局部Conv1d、按绝对秒把25Hz特征对齐到40Hz、最后Linear256→32，输出 `[2,32,T]`。**这里32是H3音频latent通道，256是adapter内部宽度，不是32个文本tokens。** 见`ms/music_encoder_adapter.py:158,189`。

训练同时保留32维时序latent与一个仅loss使用的CLIP读出：latent各声道时间mean/std→128维→audio_head5120→L2。bridge-only冻结公开音乐骨干和audio_head，只训练adapter拟合原音频VAE同音乐latent；pureCLIP训练实际dense骨干＋adapter＋head；bridge＋CLIP同时做InfoNCE和原音频latent MSE；bridgeinit-lambda0只保留桥接adapter起点，后续MSE系数0。见`music_clip_bridge.py:390,518`。冻结dense特征缓存只用于adapter桥接暖启动，不能称为骨干联训。

推理实际替换的是`pipe.audio_vae.encode_audio=MusicEncodingProxy(...)`，**reference与retake两处都换成同训练音乐骨干＋TemporalLatentAdapter**，输出32维latent；训练侧audio_head不进入生成，原文本prompt／TC和独立图VAE仍在。这时音乐确实经音频rows进入联合attention，但没有替代动作文本控制。见`sample_h3_ar.py:170`、`music_encoder_adapter.py:215`。

#### 历史阶段C：旧完整CLIP链的单文本token诊断

另一次明确免训练诊断执行完整旧MusicFM→32维adapter→时间mean/std→训练CLIPhead→L2，然后用train-only全局尺度放大作为**一个5120维文本条件token**。保留旧TC图prefix，native reference／retake固定静音。这个实验特意不丢CLIPhead，但没有inverse-L2或还原逐文本token，也不是当前native hidden回归学生。其失败不能直接否定逐token文本hidden路线。

#### 历史阶段D到本轮：音乐直接预测文本LM输出hidden

旧目标是图＋动作正文经冻结TC后的正文tail；学生用MusicFM＋raw-text adapter，并读取空白文字＋图的TC prefix最后一行context，生成时保留TC prefix并替换正文tail。后按用户要求完全关闭TC，改为music-only global bias／contextNone、prefix0；这段内部bias是可训练模型参数，不是空白文字embedding。旧带图教师tail仍可能含学生无法从音乐推出的图信息。

本轮教师因此改为纯文字→冻结50层LM→原逐tokenBF16 `[L,5120]`，不输入图，不pool。学生波形→在线实际MusicFM dense→memory1024到内部512（control256）→两cross-attention/FFN query→自由raw输出5120＋长度头；末端public Identity，control LN，均无固定W/PCA投影。训练真实dense骨干和**完整adapter／最后5120投影／长度头**，目标为rawMSE/train_energy＋.01log长度MSE，仅一个配对音乐分支；wrong／second／zero仅验证／视频，不是当前训练三分支。H3模型全冻结、不加载进此表示训练，不加FM或节拍loss。见`music_text_adapter.py:180,286,358`、`music_text_contrastive.py:229,548`。

生成用同一完整checkpoint和自身预测长度，经BF16cast作为全部prompt_embeds直接入原condition_proj/refiner；TC/tokenizer/processor不加载、prompt/negative为None、图只经独立视觉VAE；原音频VAE仍执行，但reference／retake输入固定静音，真实音乐只经新hidden分支影响视频。最终配乐mux与音乐条件作用分别验收。**当前“音乐tokens”按内容来自音频，但在H3中走的是文本模态入口，不是原audio_patch_proj分支。**

文字token索引不是音乐时刻：只回归文本hidden不能自动提供40Hz拍点／动作强调时序监督。adapter需要维度／序列／分布转换，但1024→512→5120结构是我们的方案选择，仍有有限宽度输出包络，5120维shape相同不等于已对齐语义或跟拍。学生舞蹈、幅度和音乐控制必须通过本轮完整视频独立判断，最新实际人评如下。

### 2026-10-21 正式学生最终人评与接口澄清

用户原话：“新纯监督·正式1800的几个视频都在跳舞,但是动作幅度分case,case2不是很大,case1和3还行,但是都不太跟拍;【新纯监督·换音乐】还是比较接近摆pose”。本次是新学生两个预设级连续定性人评，不是教师评价、全18逐条数值评分或public512/control模型排名；正式1800预设也包含零输入控制，但用户未单列原曲对零输入的增量，不能编造该差异或人物保真已通过。独立记录`results/ms_20261020/music_text_pure_students_human_review_20261021.json`。

**本轮结果：部分正向——正式1800预设所看视频恢复舞蹈，case1/3动作幅度尚可、case2偏小；明确负向——所有所评配置跟拍弱，换曲预设仍近pose。** 因此本轮授权实验的训练／验证／生成／技术审核／人评流程完成并有真实正负结论，但整体“仅音乐＋图的可靠连续且跟拍舞蹈”未达成。不能继续说学生人评尚未到达，也不能把“人评完成”写成“音乐质量通过”。先前不可变报告中的pending仅表示其产生时的状态，不重写历史媒体审核或评分。

1. 当前MusicFM dense原生特征是 `[N,T_music,1024]`，N为送入骨干的单声道batch；我们stereo wrapper将左右声道当独立batch编码后变为 `[B,2,T_music,1024]`，25Hz，固定5.0625秒片段通常127个feature时刻（精确值由mel/卷积边界决定）。它不是 `[2,32,T]`，不经过旧TemporalLatentAdapter作为本轮学生的中间32维瓶颈。当前链为`MusicFM1024→music_text_adapter内部512(控制256)→逐token5120→H3 condition_proj/refiner`，不是替换audio_patch_proj。
2. H3要求条件每行5120维，但不要求音乐输出L与某个未知caption的词数完全相等。学生本轮输出qmax1423候选再自行截取，教师L只作监督；每音乐feature时刻不是一个caption词。让音乐时序tokens直接经1024→5120投影进入可变长text_pos也是可设计接口，H3没有强制两层cross-attention／word-length预测；它还需验证分布和语义，不是仅改shape即成功。
3. 关闭整套Qwen3-VL TC会同时去掉其视觉encoder＋LM中的图语义路径、引用presentation前缀，以及相应序列长度/位置。独立video VAE参考仍照常运行且本轮同图条件hash一致，所以“VAE图分支未被删除”成立，“整套图像条件完全无影响”不成立。少掉一条条件信息路径可能影响身份／衣着／动作与参考关系，但本轮学生服装漂移不能由这一事实单独归因TC移除，因为教师tokens与学生tokens也不同。
4. 两条图路径共享准备过的参考图pixels，不共享VAE latent：`图pixels→Qwen vision patch encoder→5120视觉embedding→LM→TC hidden`，另为`图pixels→video VAE encoder→24通道latent→96维patch row→video_patch_proj`。TC用pixel_values和grid信息，不拿video VAE编码结果填image_pad。两者最终在同一DiT宽5376序列交互，但入口投影/表示/空间位置不同。
5. 当前不是物理删除H3音频分支：固定静音波形仍通过原audio VAE，两路reference/retake audio tokens、位置与mask保留；只有真实音乐内容不从这个入口进入H3。静音latent并非零，保持原结构不等于保持真实音频信息。删光音频rows则是另一种布局干预，本轮没有做。原生40Hz声学时序信息现在必须由music→text hidden路径承载；文字逐token回归无显式beat／动作强调时间loss，可能不足以迁移该职责，不能直接声称这是唯一不跟拍根因。

**后续决策，不自动追加预算：** 这轮已排除“原曲输入下永远没有舞蹈”和“必须teacher包含参考图才能舞蹈”的绝对判断，但没有建立音乐节拍控制或换曲泛化。下一方案应把音乐时序／拍点到动作强调的对齐与图语义／人物保真分别诊断，先明确要验证的单一假设和所需资源；保留无外部动作文字／无TC要求，不偷偷加原生真音乐旁路或恢复TC，不无限延长同配对MSE配方。

#### 为什么模块M输出与教师CLIP相似仍不保证生成一致

用户新增问题的强前提成立：若M输出逐token完整数值等于原TC输出，长度／tags／packed位置及其他图／音频／seed／采样条件都相同，固定H3实际接收同一条件，确定性推理应相同（浮点非确定性另计）。此前identity oracle确有原hidden拆拼、packed及首步教师/学生DiT预测对照，见`sample_h3_ar.py:295-318`；不能将训练失败说成原生hidden不能替换。

实际CLIP弱得多：(a)旧原生音频/专用32D路线对齐的是normalized temporal latent→mean/std pooling128→训练head5120→L2与caption pooled5120目标，而生成只消费32D序列，训练head不部署；相似的两个全局向量不是同一个5120D逐token接口。(b)本轮formal1800已clip_weight0，只rawMSE+.01length；没有实际“训练CLIP所以高相似”前提。旧rawH序列对比的soft-Chamfer／均匀token抽样不保证完整逐位置顺序数值一致。(c)即使每token cosine很高，cos(H,αH)=1不约束幅度；全局共同均值／方向也可能掩盖决定动作差别的中心化变化，H3带bias投影/refiner/jointattention不保证对该改变不敏感。decoder真正关心的条件差异不能用一个相似度标量完全代替。

(d)只匹配动作tail但去掉TC的图vision prefix，或者学生长度不同导致reference/target RoPE起点变化，都改变完整输入；不能声称仍是“只把相同hidden回填”。(e)本轮实测macro rawMSE public128.1719/control167.7893，position_mean baseline5.83669；音乐候选top1 .325/.36接近三候选随机1/3，并没有证据证明实际送H3的完整rawhidden已接近教师。初始化mean/RMS匹配不等于训练后的完整语义匹配。(f)文本caption不唯一由音乐决定，词位置不是音乐时间，配对MSE可平均掉不同合理动作表述。即使精确复现不跟拍教师，也只能期待复现该教师行为，不能从相似度推出超出教师的拍点控制。

最有诊断意义的比较是在完全同图/native条件/长度布局/采样轨迹下，分层核rawhidden、condition_proj、refiner和冻结DiT预测的误差，再与视频行为对应；形状相同、CLIP检索接近或训练loss降低都不能省略这些前提。此处只澄清已有失败机制，不增加未授权实验。

### 最新范围纠正：只替换文本向量，不将隔离协议冒作原生单变量替换

用户进一步明确：理论上MusicFM `[B,2,T_music,1024]`经adapter直接得到 `[B,L,5120]`即可，认为全qmax＋预测caption词长再截取不必需；实际只希望替换文本向量，其余不改变；最早复杂prompt结果不错，质疑“教师不会跟拍”被泛化；1800训练与1200更新也可能不足。以下纠正不启动新GPU实验、不改变已发布checkpoint或历史记录。

**长度预测不是H3文本接口的必需条件。** 可以按音乐时间长度输出（左右声道融合得到L=T_music，或保留声道序列L=2T_music），经可训练线性/MLP/时序adapter逐行映射到5120，不预测caption词数；也可使用事先规定的固定K条件slots。前者是直接音乐时序表示替换，后者是固定长度表示，均不同于本轮full1423候选再length截断。qmax原设计服务逐词教师MSE，不是H3要求；实现更复杂不能作为必要性的证据。若教师词数可变，训练监督的序列对齐方案需要明确，不能将任意重采样教师称作原生完整tokens；只音乐部署也不自然提供某个未知caption的词数。条件L改变导致packed后续起点改变，应明示这项文本接口本身的布局变化。

**“教师不会跟拍”之前只能指具体静音native、MotionScript/noTC的三教师预设，不能指原始H3或最早复杂prompt基线。** 原E3脚本`diff/scripts_omnidance/exp/h3/e3_h3_zeroshot.py:59-100`有身份/衣着/场景/音频引用等复杂模板，原配实际音乐进入reference，默认music_mode=ref；独立retake模式另外保留实际音乐clean target。调用完整原pipeline，video/audio都step，decode生成音轨（不同于最近只step视频＋mux）。脚本默认480×832，不凭默认值推断所有历史命令；最近固定704×1280已是另一个采样设置。用户早期认可该复杂prompt效果，不能用修改caption、去TCvision与静音原生音频后的人评否定那个上界。

**之前的实验没有严格做到以早期完整原生基线为基准只换文本向量。** 专用32D替换改了音频接口；旧raw-tail路线保留TC图/引用prefix，但原生两路音乐仍被静音、prompt目标也不同且长度由学生预测；最新noTC路线额外移除了TC的Qwen视觉/图文上下文与presentation prefix。只有identity oracle把原hidden拆拼回填与packed／首步预测做一致性验证，说明接口替换原则可行；它不是已训练纯音乐M按原生全条件生成的成功实验。

最新目标应以原生基线实际参考图、音频reference/retake策略、DiT/VAE/采样参数为准，只更改动作文字条件入口，不再默认fixed_native_silence即部署要求。TC输出是包含Qwen视觉和普通文字的多模态条件序列；“只换动作文字部分”应保留原TC视觉/引用部分，而不是把它和整套TC一起删除。此前“完全不加载整个TC”与现在“原TC视觉条件也完整保留”不能同时严格满足，因为该视觉/LM前缀由TC模块产生；若仍坚持整套TC不加载，需另定义如何保留这部分信息，不能声称独立video VAE已经数值等价补回。训练教师采用原先可用复杂prompt，不强制退回纯MotionScript；教师可离线编码完整文字，学生生成仍不接收动作prompt。

**训练不足可能成立，不能以1200更新声称收敛。** 本轮每配置batch8／1200更新／9600曝光，1800train约5.33遍；这是完成既定预算，不是充分训练证明。重新读取实际记录：public验证tokenweighted rawMSE227.25325→90.70603；control119.20485→117.64113。最后8条train-probe为90.12681／117.24107，val为90.70603／117.64113；全200教师位置均值baseline的macro5.83669远低于学生macro128.17189／167.78935。这说明当前拟合/音乐区分仍不足，但8条probe不能证明全train不拟合或排除过拟合。训练仅保存初末val与随机batch每10步日志，没有中间多个同集合验证checkpoint来判收敛；不能拿不同batch loss末几步下降证明长训必有收益。

更多干净配乐/动作数据、保留原复杂prompt教师、较充分更新均值得受控验证，但要先固定正确替换范围，不能在已改动多项原条件的路线无限追加step来回答单变量问题。后续更长训练应报告实际独立样本数/总曝光/完整train与val拟合、音乐反事实及固定三case视频，区分只改善平均token回归与真正音乐控舞。当前trainer的last.pt只保存模型，没有optimizer state；从last权重加步数需注明optimizer重置，不能冒称Adam状态无缝续训。具体新预算/节点需在启动前另行明确。

### 2026-10-21 新批准阶段：6000条、按图缓存P、直接音乐tokens与batch CLIP

下述是用户随后批准的新阶段，不重写上面的旧1800实验。用户确定教师prompt仍采用 `Dance style: {genre}. ` 加完整中性化原版MotionScript；genre是**舞蹈风格**，不是音乐风格。用户曾询问rich来源与MS融合，实际只能追溯到OmniDance原始文本及E2/E3包装链，不能确定rich生成模型/指令；本次不混入rich。旧阶段结束不意味着新阶段完成。

#### 固定替换边界与部署协议

- 原生TC针对每张参考图和引用配置离线得到 `[P;y]`。P保留完整Qwen视觉/引用hidden及原tags，不只是固定system文字，不跨图复用。y是**整个用户prompt x**的hidden后缀，不再拆A/D/Z。
- 离线缓存P和教师y；正式生成不加载/执行TC、tokenizer、processor，不输入空白动作文字。新图仍需在素材准备阶段先处理一次。
- 学生M只接音乐波形，不接P、图context、教师词数或caption。实际在线MusicFM dense `[B,2,T,1024]` 按声道优先展平为 `[B,2T,1024]`，单个可训练 `Linear(1024,5120,bias=True)` 直接输出raw5120。无query、qmax、length head、PCA、32D瓶颈或可丢弃CLIP head。
- 有效音乐feature mask决定长度；左右声道padding可能形成holes，部署按mask gather有效行，不能简单截前L行。实际固定5.0625秒探针输出254行，不要求与教师词数相同。
- 下游只拼 `cached_P + M(music)`，P保原tags，音乐后缀text-tag为1。L改变带来的packed后续起点改变属于此次文本条件替换的明确变化，不能声称未知教师caption的全部RoPE逐值不变。
- 保留原生真实reference音乐、独立图/音频VAE和联合音视频采样。默认 `music_mode=ref`、输出实际生成音轨；retake和原曲mux必须显式另标，不能偷偷继续旧静音/只step视频的sampler。零M/换曲只作诊断，不删除原生音乐入口。

#### 数据、loss与收敛计划

- 从合格MS全集重新选6000条多样train、500条来源组独立val。排除旧val200及固定三case原/换曲相关组，合并原视频、exact/疑似音频重复、相同中性caption来源组；按genre、speed、travel、turn、arm/leg focus等现有轻量指标分层。近重复组不等于已证明独立歌曲，缺失来源必须单列。不足6500不重复凑数、不静默降门槛。
- 同公开MusicFM/Linear初始化，两组均真实训练316,343,296个dense参数及5,248,000个Linear参数；FP32，骨干eval固定BN/dropout但保梯度。冻结H3，不加DiT/FM/pose/节拍loss。
- 每个音乐配对每步只做一次M前向。actual raw有效tokens masked mean与教师完整y mean使用固定train-only mean/std、L2作为loss读出；部署raw不归一化。全实际global batch双向InfoNCE、temperature0.07，同来源组假负例mask。梯度累积不扩大CLIP负例池。
- A：纯batch CLIP。B：同CLIP加每样本5120均值误差/固定train std，以及每样本中心化全token/channel标量RMS的log比例平方，两项初始权重各1。不是全batch RMS、逐channel log-std或固定池RMS目标。这些约束仍不保证协方差、token顺序和节拍对齐。
- 预期4rank×16真实global64，每epoch完整6000尾批48条，各rank12，不drop/重复。GPU不足时两组共同降低真实batch并明确实际负例数；不伪称global64。
- 每epoch固定500候选验证CLIP、双向R@1/5/10、raw moments和非塌缩；固定64条train probe只能诊断子集，不能冒充全train评估。
- 最少20完整epoch，固定验证CLIP连续8次相对改善不足0.5%后LR×0.1；第二平台还须R@1不低于第一平台锚值减0.01且finite/noncollapsed。初始cap50，未满足就标`convergence_unconfirmed`。操作性表示收敛不等于视频成功。当前scheduler各配置独立判断平台，实际LR/停止epoch必须记录；不能预先宣称两组一定同epoch收敛或同步降LR。
- 保存完整模型、Adam、scheduler、各rank Python/Torch/CUDA RNG、epoch/cursor、sampler、history与源指纹，可恢复。与旧1800只存模型checkpoint的协议不同。

#### 本阶段已验证事实（截至本节写入）

1. 正式CPU证据仅使用 `results/ms_20261020/direct_music_tokens_adapter_cpu_v1.json`、`direct_music_tokens_infer_cpu_v1.json`、`direct_music_clip_training_cpu_v1.json`。根results中的同名副本不删除、不作为正式证据。adapter/infer报告为CPU fixture，不包含真实MusicFM权重/H3视频验收。
2. trainer CPU gloo global64/7/1的可微gather+DDP梯度最大误差不超过1.9e-9；真实行/通信padding隔离、source组mask、每sample矩不抵消、完整Adam/RNG恢复及6k尾批覆盖通过。第一平台R@1=.20、第二平台.18拒收敛、.19边界接受、塌缩拒绝。训练源码SHA为 `4e11b5daa6102994b7bbd43ac597873600f66ec36af4f22fedf4e40ea6dd4143`。
3. 真实GPU公开MusicFM+Linear检查已完成：batch16，400个dense张量/316,343,296参数、2个Linear张量/5,248,000参数都有finite梯度；peak allocated 16,469,304,832 bytes（约15.34GiB）。完整format4零更新初始化重载输出逐值一致。报告 `results/ms_20261020/music_direct_public_gpu_probe_v1.json`；该检查只做squared-output scope/memory probe的一次Adam更新，**不是CLIP能力训练**，保存初始化在更新前。
4. 三固定case真实原生TC缓存已完成并独立校验：`wJPptuhclZmKAEWs`、`wJPrpFsThWAcBQIw`、`wJQTclrlOLEYYfTB`。三条P长度均7437，教师y长度分别228/199/323；同实际TC权重内容SHA `3f187f9508c62907301064c8ddc3c9abe74a962e723c25e95d5e5c662549e9b9`。每条实际LM前向、同shape替换整prompt标点的P独立性、P/y拆拼raw/tags一致、文件/张量SHA及prepared图验证通过。路径为 `diff/logs_omnidance/exp/MS/music_direct_eval3_cache_v1_s{0,1,2}`。不重复GPU前向，不把缓存成功当原生DiT预测/视频质量通过。
5. 真实独立GPU音乐worker验收已通过：完整format4重载、exact波形单输入与六字段输出、FP32 `[254,5120]` 与主进程逐值一致，无context/caption。报告 `results/ms_20261020/music_direct_worker_gpu_probe_v1.json`，不是CLIP训练或H3视频验收。两正式训练节点19.69/17.70已得到同SHA的本地 `/tmp/music_direct_public_init_v1.pt`，严格hostkey校验及内容SHA验证通过；共享目录的已有同名文件仍在独立校验，不覆盖不凭大小信任。
6. 6000+500唯一v1选择实际结束为失败：11908pool、263350已知source IDs，仅4715联合独立代表，小于6500；7015条被`invalid_paired_window`拒绝，18条pose有效率低于.8（拒绝原因可重叠），与历史holdout相关218条排除。v1仅发布 `music_direct6k_v1/selection_report.json`，无caps/split/references，不能启动教师缓存。窄查三条被拒实际样本发现源`duration_seconds=5.187988`，但自然窗口仍81帧/16fps/无padding；用源媒体时长上界检查5.0625秒片段是字段语义错误。修复只移除该错误source duration比较，继续锁source_fps16、natural窗口81、sourceframe足够、零padding、704×1280及原质量/pose/组隔离；保v1失败记录，新v2必须独立通过实际选择才可上缓存，不能预称已有6000。
7. 四rank×16真实CLIP/NCCL准入v3已完成，用已有三条真实配对重复组成64计算行，只检验通信/显存/梯度/Adam，两loss配置各同零更新init。报告 `results/ms_20261020/music_direct_ddp_gpu_probe_v3.json`：全部400dense与2Linear参数张量非零finite梯度、BN/buffer固定、adapter前64权重跨rank逐值一致、peak allocated 17,663,881,728 bytes（约16.45GiB）；两loss四rank各自一致。仅3独立配对、同来源负例mask、统计只来自三教师，**不是64独立负例、正式6000训练或能力证明**。v1检查命令把字符串传给只收Path的SHA API失败，v2重组shell引号只执行guard，exit0没有GPU报告，不作为完成证据；保两日志，v3修正实际命令并独立读取报告通过。该报告不证明6500教师缓存、两配置正式CLIP、收敛或原生视频能力。代理general-purpose-2/3曾因上游empty-stream中断，已落盘代码/报告分别核验；不以中断代理消息替代实际完成证据。
8. 真实三case缓存的producer/consumer联调已完成，不再只依赖自拟fixture：`diff/logs_omnidance/exp/MS/music_direct_eval3_cache_audit_v1.json` SHA `6141f6b02230852b04e58ea56baca7e1d083eef2bfb38a50af88fca073721360`。消费者核实际全target/prefix文件SHA、BF16 y、含dtype/shape/bytes的tensor SHA、真实本地tokenizer生成的source IDs及nested plan/finished绑定；不反序列化P，train=[]/val3、statistics=null、trainable=false。正式入口仍要求6000/500/六完整shard，并显式传`--tokenizer_root`；不要求旧producer拥有selection_report/tokenizer_root。pipeline SHA `ffa66b2c12e3a0cc404a2b95cadc71048f7926cd46f68b26e98f7ef59177b662`。主线程之前确实在producer增加了optional selection_report和tokenizer_root metadata，当前cache源码SHA `c097078ef17fe86d79eaf1fc9f61ea8e05e2bcaeefa13426f097c0c0865b2a38`；旧三case产物保留其原source SHA `3ba733...`，没有改写或重新缓存。旧SHA断言失败是快照变更，不回退代码或篡改历史证据。
9. 原生direct生成已修复真实音乐加载为`librosa.load(sr=pipe.audio_vae.sample_rate, mono=False, duration=num_frames/24)`，与E3原生32k路径一致；学生独立调用`load_audio(stem,audio_root=parent)`取waveform。只加载DiT和两个VAE、不加载TC/tokenizer/processor，原native joint pipe保留。限定及整体`git diff --check`、源码SHA/AST与已有joint generation manifest SHA均已由主线程独立通过。最新manager SHA `96abfa40cb20055f481f97651cf78bd149ec41aac07430997266eefe6ddae0c0`；`direct_cached_prefix_native_cpu_v3.json` SHA `871605977d5c655ac4e29da38a4b38e42cbe21c7e3efad132e1f59d61c94e1e4`只是fake-pipe/真实loader CPU回归，不代表实际H3 GPU通过。随后三case真实GPU原生恒等生成`4GSClB`已实际exit0，输出`H3AR/val_music_direct_cached_teacher_identity_v1`：三条raw/tags/packed以及首步video/audio预测逐值exact；两scheduler各50更新，实际生成音轨保留，P7437和两个独立VAE reference存在。三条124帧704×1280/24fps由decord逐帧及ffmpeg -xerror完整音视频解码通过。全审核报告`MS/music_direct_cached_teacher_identity_all3_audit_v1.json` SHA `af699c255b7c183fb2b75a4bba8f8b95c8de50d9dd795acabec6b204fc3dc74c`，可提交摘要`results/ms_20261020/music_direct_cached_teacher_identity_gpu_all3_v1.json`。首次真实审核`BqehT4`因source路径含`h3ar/../ms`与plan字符串不同失败；独立证明canonical路径和SHA一致后，仅消费者统一`Path.resolve(strict=True)`，不改生成源码/缓存/视频、不重跑GPU，`iatbWG`审核通过；生产比较的错误SHA负回归`J40gcW`仍拒绝。ffprobe不可用，报告明确未独立检查声明stream总数；已验证所选视频/音频全解码。此项只证明离线cached full teacher与P+y的恒等接口及技术完整性，不证明live-TC对照或训练音乐M的舞蹈/跟拍成功。新评审清单`review_web/manifest_music_direct_native_teacher_v1.json`追加这三条并保持历史样本/模型/评分；独立8769服务不替换原8766学生页面，预设为“【原生真实音乐·人评待定】原生P+y教师identity；原生joint生成音轨保留”。
10. 修正窗口语义后的唯一选择v2任务`M0QUFr`已实际exit0，`music_direct6k_v2`发布完整6000train/500val：11908pool→10979联合独立代表；6500组唯一、train/val隔离、caps/refs/split IDs一致，6500实际图/音频/meta/MS资产核验完成。主线程独立核四输出SHA：caps `7b1c9d88b8f01c921ab3c03226ccec67eb3373bf414b7dd1a5634d15dddf2b42`；split `a3a9cd8211b8dcdac8e688ea845bcf0f5e07262c48c663a918f97b880ef5c2c6`；references `42055258822f7ab0554b0d57fe1c94ca2851ca4653d5a312467638c9633291e2`；selection_report `03c96415c3d3692d2c09d1032ac0ed420767f2b705a22c33c232719a2b95caca`。无重复凑数或降低质量/pose门槛，保留v1失败报告。代理general-purpose-1完成产物交付后也发生上游empty-stream；已独立核验产物保留，后续由主线程接管。
11. 完整六分片缓存现已安排，分片数量1084/1084/1083/1083/1083/1083、每片train1000；前三节点GPU5避开仍运行的GPU6教师恒等验收，第四节点GPU0、其余两节点GPU6。首轮`AGYrjW`六节点准入通过，但错误使用`/opt/venv_music`，全部在导入缺少imageio时exit1、未创建cache输出；失败管理日志`MS/music_direct6k_v2_cache_launch_v1`完整保留。已CPU实测`/opt/venv_h3`可加载实际缓存入口，改为该现有环境恢复，handle `sF0F73`、新管理日志`MS/music_direct6k_v2_cache_launch_v2`；缓存输出仍为未创建过的`MS/music_direct6k_v2_cache_s{0..5}`，不覆盖任何产物、不安装依赖、不改cache冻结源码。准入只允许明确授权节点/精确自身holder，实际退出和完整finished文件才算缓存完成。按三case实际P文件尺寸及全caption分词，预计P约461.41GiB、y约25.72GiB，总约487.13GiB加开销，几何/token长度差异会改变实际大小；共享文件系统约216TiB空闲不等于已确认用户quota。6500教师缓存、两配置正式CLIP、实际收敛及新学生视频/人评仍未完成；实现/选择/既有CPU与GPU证据已由新commit `21e1bd662b458e4c8488684bcf5b00cbc5b875ee`推送GitHub main，并实核远端同SHA。该commit不是最终训练/质量结论，后续真实GPU三视频审核另行发布。

#### 缓存完成后的事件接续

已将完整缓存→正式manifest→两组正式训练接为单一管理队列，日志`MS/music_direct6k_v2_training_queue_v1`、任务`XATcau`。等待绑定的实际缓存管理进程完成事件（pidfd），不sleep、不重复启动缓存或轮询凑完成；缓存六片complete/exit0/6500、冻结源码SHA通过后，实际调用正式builder构建`MS/music_direct6k_v2_training_manifest_v1.json`，仍核完整target/P字节SHA和真实source IDs，不反序列化P。builder失败或源变更立即停止，不启动训练。

manifest成功后自动在19.69/17.70各0–3四GPU启动相同初始化的A/B，local16/global64、lr1e-5/1e-4、seed42、最少20/cap50/平台8次/相对0.005；A moment_weight0、B1。每卡严格自身holder准入、未知进程拒绝，训练前再次核本地init SHA和实际manifest SHA。输出`MS/music_direct6k_v2_clip_a`、`MS/music_direct6k_v2_clip_moments_b`；其他四节点在缓存结束后不再使用。该队列处于等待只表示自动接续已部署，不表示正式训练已开始；训练完成也不自动代表达到收敛或学生视频/人评通过。10分钟会话定时跟进只用于发现报错、推进验收与后续视频，不另开同名任务。

#### 两组训练完成后的学生视频接续

用户再次要求“并启动训练和后续实验进程”后，已部署第二条唯一完成事件队列`ra7oDl`，管理目录`MS/music_direct6k_v2_final18_queue_v1`。它绑定实际训练队列进程pidfd，当前等待两组正式训练完成，不生成未训练学生；只有两组formal/complete/exit0、至少20完整epoch、94更新/epoch合计至少1880、完整val500、last文件SHA与best/train_state存在及冻结推理源码全部通过后，部署各自`best.pt`。训练cap结束但收敛未确认时保留`converged=false/reason`，视频用于诊断，不篡改为收敛成功。

六配置分别为A/B×joint/aswap/zeroM，每配置只3case、共18条，按六允许节点各GPU6并行执行；使用实际原native生成管理器，未知GPU进程或host/cgroup内存不足拒绝准入，不抢占用户其他任务。每条保缓存P/tags、独立两个VAE、native reference真实音乐、原joint音视频各50步及生成音轨。aswap清单`MS/music_direct_eval3_generation_aswap_v1.json` SHA `1aa66e5dcb7676640ccf52d3e7d9f09d72da667c27d90ef108ff9d270be37f98`已CPU实际构建/独立核验，同图和P/tags，换M波形同时换native reference音频；不是仅M单变量归因。zeroM复用joint清单，只将M波形置零，保原native真实音乐。joint清单SHA `6d08214d9b5e0b5912374880e9b3d464c10941d74d766f99bb5317cd35c48409`不改。来源准备已发布`results/ms_20261020/music_direct_final18_source_binding_v1.json`（commit `6f90936a6795c0856aac968f6cfa1fec5b7f80eb`远端核验），不是学生视频完成证据。

六配置生成全部exit0后，队列实际执行每配置独立`--direct_native_audit`，decord逐帧和ffmpeg -xerror全音视频解码、checkpoint/source/P/native/scheduler/音轨SHA一致性审核。失败保留partial与日志，不覆盖/跳过；全部技术审核通过仍写`musical_quality_accepted=false/human_review_required=true`。真实人评、模型归因与正负分析由后续独立验收完成；队列不能代替用户判断连续舞蹈、幅度、跟拍、换曲和身份保真。

#### 2026-10-22 完整缓存验收与串行I/O修复

6500教师缓存已实际全部exit0，六分片1084/1084/1083/1083/1083/1083完整finished；主线程`4l2VQy`核全部plan/finished SHA、唯一6500 IDs、train6000/val500、全target/P文件ID集合、actual LM/P caption独立性/原raw拆拼。报告`results/ms_20261020/music_direct_full6500_native_cache_v1.json`明确这是缓存完成，不是正式训练或质量成功。

实际19.69读验4个P文件304,881,620 bytes用了36.27秒（单个7.16–11.87秒，`E9u7ZV`）。为避免全量缓存验证和rank0启动串行I/O阻塞，`music_text_align.py`新增受限16线程**完整文件字节SHA**，并核dev/ino/size/mtime_ns/ctime_ns读验前后及消费前后不变；`music_text_pipeline.py`每shard并行核P/y/audio后仍逐项比较全部finished/reference SHA，P不torchload，FP64 per-example mean/RMS及train-only统计不变；trainer同样完整并行核全部源字节、源指纹加入hash helper。NCCL等待timeout显式4小时，以容纳rank0全源完整核验，不关闭内容验证或改变loss/batch/优化器/收敛规则。

实际3cache消费者回归`KWYCiK`得到新报告`MS/music_direct_eval3_cache_parallel_audit_v2.json` SHA `0858840dd1bac625ab155f1b70be1869f1d1f88ce932f40dc7d5f2b8eddc9fd1`；主线程`TWMyJw`证明旧新rows（实际P/y/audio SHA、逐样本teacher均值/RMS）逐值相同。四真实P的完整SHA与finished一致，文件version负例拒绝、错误SHA仍不匹配（`JgmWWd`）；该测试是热cache且不同host，不将0.1秒和远端36秒冒作吞吐加速证明。独立只读审查未发现新阻塞，未另开测试或上卡。

原`XATcau`/`ra7oDl`接续队列已在正式训练前安全暂停，保存其v1 plan和全部日志，未改变/重跑cache。唯一全量builder现为`Ryz64z`，新输出`MS/music_direct6k_v2_training_manifest_v2.json`，仍严格6000/500/六完整shard、全部源字节校验；唯一新训练接续`EjktZa`（`MS/music_direct6k_v2_training_queue_v2`）等待实际builder完成，之后同原A/B配置启动。唯一后续18视频接续`K0aRDS`（`MS/music_direct6k_v2_final18_queue_v2`）等待新训练队列；视频/质量状态不变。旧manifest和队列不删除、不覆盖。当前全量manifest尚未完成，正式GPU训练尚未开始。

缓存结束后已执行资源释放`uDYIoQ`：在19.213/19.143/19.137/19.225仅按本项目精确`diff/scripts_omnidance/exp/hold_gpu.py`路径和所属用户终止32个占卡holder，pidfd确认退出，四节点剩余本项目holder为0；没有停止任何其他计算任务。报告`MS/music_direct6k_unused_four_nodes_holder_release_v1.json`。训练仍仅使用19.69/17.70；后续六节点推理需重新按实际空闲和未知进程拒绝规则准入，不以已释放为未来资源占用保证。

#### 2026-10-22 正式manifest完成、首epoch日志错误与同起点恢复

`Ryz64z`已实际exit0，正式`MS/music_direct6k_v2_training_manifest_v2.json`完整SHA `40df4ff51bb020a9296ad2c54e9c9c069ef9b1e9710ad2c234d3d7782ea2c6e6`，6000train/500val、6500统计行、全部train有效teacher tokens共2,498,122。主线程`wJVH2C`独立核文件SHA、counts和train-onlytoken sum；全P/y/audio字节SHA、真实tokenizer source IDs、FP64统计已由实际builder完成，P不反序列化。构建实际耗时6h28m，不能将16线程理论并行写为已解决所有CephFS I/O耗时。

`EjktZa`随后实际自动启动A/B四rank/global64训练，全部19500源文件字节SHA验证通过；两配置完成首轮94更新并计算完整val500和64train probe后，rank0在`music_text_contrastive.py:1311`日志`dict(event='direct_clip_evaluation', **report)`因report也有event而TypeError。错误发生在模型/Adam保存前，两失败目录只有plan/gradient_step1，无last/best/train_state，不存在可恢复训练状态。日志可见A第1/90更新CLIP4.44149/3.81567；B同处CLIP4.44149/3.92384、矩2.48691/0.15047。这是不同batch训练值，不作同集合验证、模型排名、收敛或能力结论。

仅修日志为`dict(report, event='direct_clip_evaluation', scheduler_event=event)`，保历史report的event不变；生产AST实际语句测试`cITVfL`覆盖None/降LR/收敛3场景，错误SHA和原训练损失/数据/超参未改。新trainer SHA `575cac1ad9c40ed19b53b6cc9c70b5d51b92721e49487d15b3f1429caef37103`。两旧训练目录及`MS/music_direct6k_v2_training_queue_v2`全部失败日志保留；不存在Adamstate因此从同零更新init在两个**新**目录完整重训，不冒充resume。

唯一重启管理任务`hOHxrY`、目录`MS/music_direct6k_v2_training_queue_v3`已实际launch且八卡准入通过。A/B分别输出`MS/music_direct6k_v2_clip_a_restart_v3`、`MS/music_direct6k_v2_clip_moments_b_restart_v3`，使用原6000/500manifest、同init/global64/lr/seed/20min50cap。唯一后续学生任务`xLC6yO`、目录`MS/music_direct6k_v2_final18_queue_v3`重新绑定实际重训managerPID2999182，新模型best.pt用于原18配置，旧K0aRDS失败不算学生视频完成。当前重启仅launch，不表示完成首epoch、收敛或人评。

进一步直接执行生产`_direct_training`的CPU生命周期fixture（`HjjBTt`）：实际epoch日志→模型last/best→Adam/RNG/sampler train_state→finished→从state恢复第二epoch全部通过，step1→2、epoch2→3、history两条保留scheduler event。仅GPU/device/数据边界和实际MusicFM骨干使用明确fake CPU替代，不算实际四rankGPU或正式val证据；测试结果`results/ms_20261020/music_direct_training_lifecycle_cpu_v1.json`。首次fixture`X2JqC7`因仅mock CUDA availability未mock optimizer CUDA graph检查失败，是测试环境问题，不改生产源码；补test boundary后通过。

实际人评接续已部署`zjrb2Z`（`MS/music_direct6k_v2_review_queue_v1`），等待实际`xLC6yO`的18视频/六独立媒体审核完成；仅追加新模型和三固定case，不改变历史评分。之后生成独立focused manifest并启动8771人评页面，8770是用户其他实验服务不触碰，8766旧学生/8769教师服务保留。准备页面不代表已经有18视频或人评，ready.json和真实HTTP/media Range还需在视频完成后验证。

#### 2026-10-22 正式训练阶段性负向证据（A20/B12）

实际重训已连续推进且A/B均跨过首epoch原日志崩溃点、每轮完整val500并保存last/best/full Adam402entries与4rank RNG/sampler状态，独立读取验证通过，不再说正式训练未开始。稳定快照`TNW4yI`：A20完整epoch/1880更新、B12完整epoch/1128更新；新学生视频尚未生成，人评尚未完成。完整快照见`results/ms_20261020/music_direct_training_interim_epoch8_a20_b12_v1.json`，记录的是当时状态，不重写旧失败报告。

同预算epoch8（各752更新）比较：A训练CLIP1.66097、验证CLIP6.58782、验证R@1 .010、rawRMSratio .192；B训练CLIP2.50412、验证CLIP6.35457、R@1 .005、rawRMSratio .960。矩约束确实保住输出尺度，但检索指标不一致，不能宣布B语义泛化/舞蹈质量胜出。

A验证CLIP最好在epoch3=5.76193，epoch20=7.17594；训练CLIP降至.193995，固定64train-probe R@1升至1.0，但val500 R@1仅.007，rawRMSratio .226。B最低观测验证CLIP在epoch3=5.88032（0.5%相对改善规则下保存best仍epoch2=5.89329），epoch12=6.88505、R@1 .001、rawRMSratio .989、64probe R@1 .5703。两组存在明显拟合/验证分化，扩大数据和增加曝光尚未建立验证泛化；64probe不是全train检索，这些负向事实也不能代替最终视频诊断。

A在epoch20达到既定第一平台门槛后实际LR×.1（骨干1e-6/adapter1e-5），仍未宣布收敛；B仍等待自己的平台阶段。保原min20/cap50和已批准两objective，不在运行中改loss/LR规则；后续若convergedflag成立也仅表示现协议数值平台。当前验证R@1低于.01，绝对-.01的检索护栏较弱，不能以通过该护栏称泛化/音乐语义/跟拍成功。最终部署已有best权重、固定18视频与真实人评仍必须完成。

#### 2026-10-22 早期完整断点与事件监控补证（A16/B8历史快照）

早期稳定快照 `4NRg9c` 来自正式 `training_queue_v3/training_0.log`、`training_1.log`，随后已推进至上节A20/B12；下表仅保留补证时的历史状态，不是当前进度。两组每完整epoch均覆盖6000训练样本，验证使用同一完整500候选集；不是同更新数排名，也不是最终学生质量结论：

| 配置 | 完整epoch / 更新 | epoch训练CLIP | 完整val500 CLIP | val R@1 | 固定train-probe64 R@1 | raw RMS / teacher |
| --- | --- | --- | --- | --- | --- | --- |
| A：纯batch CLIP | 16 / 1504 | 0.324705 | 7.180860 | 0.006 | 0.984375 | 0.218189 |
| B：CLIP＋逐样本均值/RMS | 8 / 752 | 2.504122 | 6.354565 | 0.005 | 0.1953125 | 0.960029 |

A最低实际验证CLIP为第3epoch的5.761932，后续训练loss大降且训练probe检索提高，但完整val明显退化，是本配方当前的训练/验证分化证据；不能将训练检索接近满分报告为音乐语义泛化。B最低实际验证CLIP为第3epoch的5.880318，但`best.pt`保存沿用0.5%相对改善门槛，调度器best仍为第2epoch的5.893288。不能把`best.pt`称严格全历史最小值，也不在运行中更改门槛/损失/学习率来混淆原定对照。B输出RMS接近教师仅说明幅值校准，不证明编舞或跟拍提升；A幅值偏小也不能只凭该数值预判视频必败。

`SA75OX`与`bP3lOr`独立读取真实A14epoch/1316更新、B7epoch/658更新的`train_state.pt`：全部历史条目均为完整epoch、6000训练覆盖、val500；每个断点均保存402个Adam参数状态、四rank RNG、四rank sampler（下一epoch且cursor0），`best.pt`存在。随后`kFBsyO`实际核验A15epoch/1410更新的完整Adam/RNG/sampler及last/best；`RLGij5`核验B7epoch/658更新。以上不是CPUfixture代替正式GPU保存。两组当时`converged=false`且尚未降LR，按原协议继续，不能据中间快照声称收敛。

2026-10-22主线程`6zP8U3`/`muszs7`直接核GPU进程：A四rank PID2999558–2999561、B四rank PID2661407–2661410均存在并占各GPU0–3约23724MiB，B四卡瞬时利用率100%，A检查时rank0暂待、其余三卡100%；不能将保存/同步期间单次0%判断为已停。GPU4–7已有其他占用，未终止这些进程。

平台规则仍是完整epoch至少20后第一次平台降LR×0.1，再满足patience8的平台；max50未确认必须如实报告`convergence_unconfirmed`。只读复核发现R@1保护使用绝对容差0.01，在当前低检索率下保护很弱；即使后续出现`converged_representation_not_video_quality`，也只能记录为现协议的数值停止条件满足，不称低验证检索已经修复或得到可用学生。保持运行中trainer SHA不变，后续视频不被表示指标阻挡。

主线程`HYGiLM`直接核验GitHub远端HEAD与本地一致，均为`83e34533d3fcc1c1d49e85c2829dd8b4a9b29b32`，包含前一日志修复提交`26cb803`及生产生命周期/评审接续记录；这只是阶段发布证据，不是最终实验发布。独立事件监控绑定实际训练manager PID2999182，用inotify＋pidfd通知降LR/关键完整epoch/结束/错误，不`sleep`轮询；原10分钟跟进调度保留，直到全部实验、人评和最终发布结束。

截至本快照两正式训练、18学生生成及人评均未完成。唯一`final18_queue_v3`等待真实训练manager结束后使用两组`best.pt`；`review_queue_v1`再等18条完整媒体审核，实际HTTP/Range和连续人评仍需完成。最终推理要重新使用六个获准节点各GPU6，总6卡，启动时再次准入，未知占用拒绝，其他节点和服务不动。原曲/换曲/zeroM各固定3case，aswap同时换M与native音乐，zeroM保native真音乐；不追加case、不把教师恒等结果当学生人评。

#### 2026-10-22 两组正式训练正确退出与最终状态验收（视频/人评尚待）

`hOHxrY`实际exit0，`music_direct6k_v2_training_queue_v3/jobs.json`中两配置均complete、exit_code0。`XRErMI`/`75f4l4`分别绑定真实A/B torchrun PID，等待实际退出后加载最终`train_state.pt`独立验收：steps/converged与finished一致，402个Adam参数状态、四rank RNG和sampler齐全，每条history都是完整epoch、6000训练覆盖和val500，last/best存在。两配置均在epoch20按原规则降LR×0.1，epoch28满足原数值停止条件；不是cap50到期、controller停止、异常退出或仅launch。

| 配置 | 正式完整epoch / 更新 | 最终epoch训练CLIP | 最终val500 CLIP | 最终val R@1 | probe64 R@1 | 最终raw RMS / teacher | 保存best epoch / val CLIP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A：纯batch CLIP | 28 / 2632 | 0.118141 | 7.201042 | 0.007 | 1.0 | 0.223623 | 3 / 5.761932 |
| B：CLIP＋均值/RMS | 28 / 2632 | 0.290612 | 7.327626 | 0.008 | 1.0 | 0.990551 | 2 / 5.893288 |

两组`converged=true`/reason=`converged_representation_not_video_quality`严格只指原定数值平台停止条件满足。完整验证检索弱、早期best后退化、train/val分化仍存在；不能宣布音乐语义泛化、舞蹈或跟拍成功，也不能将每曲唯一caption检索指标直接等同生成失败。B均值/RMS约束校准幅值的数值效果成立，但未建立语义或视频优胜。B实际最低val CLIP在epoch3=5.880318，小改善未达0.5%门槛，保存best仍epoch2；最终18视频按既定best.pt，不偷偷改为last或重训。

A最终last SHA=`fda86016041ad0c9eb3f9922ae3f27f05e5770dec49ccfe13ab5f2e2f6d7e7ca`，B最终last SHA=`2bf65b75e1b345914d7c627be7ca4909d8f9b30460eb2ceb0e3472c012028006`；两组source/init/manifest SHA保持原协议。完整日志、权重、Adam与所有历史失败目录仍在实验存储，不上传GitHub。

阶段补证实际新commit `8a388492853c2b65629bdbbbf76f5c07a38039ff` 已推送，`i2hkaV`直接核remote main与本地一致。旧VSCode Git IPC失效导致一次push认证失败，实际使用当前会话已授权、connectable IPC后成功；未读取/暴露凭据、修改git config、amend或绕过hooks。此处是训练完成报告，尚不是全部实验最终发布。

最终六GPU6只读准入快照已核：六host均只有同uid指定项目holder（CUDA_VISIBLE_DEVICES=6），无未知compute/既有H3 sampler，各GPU6空闲约80178MiB，宿主与cgroup可用均≥180GiB；未提前释放任何holder或启动额外生成。已告知总六卡资源，唯一`xLC6yO`在真实训练manager完成后做checkpoint/source验证、逐shard重新准入再生成18；所有partial保留。其完整媒体审核、跨配置原/换/zeroM来源核对、人评页面HTTP/Range、用户实际连续反馈和最终分析GitHub仍待完成，原跟进调度不得提前取消。

#### 2026-10-22 验证弱为何不等于舞蹈失败、为何暂不原配方无限续训

用户询问A验证弱是否应该更多训练。实际A epoch3/16/20/28的固定val500 CLIP依次5.76193/7.18086/7.17594/7.20104，epoch训练CLIP依次3.47866/.324705/.193995/.118141；epoch20已降LR×.1后再完整训练8epoch仍未恢复验证。当前证据优先支持train/val分化而非仅更新不足，但不能证明任意更长预算永远无效，也不能把旧1800/1200步的不足结论外推为本次28轮早已泛化成功。若后续独立指标持续改善则可明确提出有预算的续训，不自动无限沿同配方重跑。

检索loss/验证实际对有效raw后缀token做逐样本mean，再使用train-only逐维统计标准化、L2归一化、温度.07的对称batch NCE；同歌曲的其他行不作负例。它验的是配对文字表示区分能力，不是逐token时间顺序、连续舞蹈或节拍评分（`music_text_contrastive.py:859–896`）。不同曲目也可能适合相似编舞、同曲可有多个合理编舞，因此唯一配对caption检索弱不能直接解释为所有音乐语义都不存在，更不能预判视频必败；相似caption假负例/目标不充分仅是待验证假设，尚无根因证据。

“不能称音乐语义或舞蹈成功”应严格读作当前尚未建立成功证据，而不是已证明不会跳舞。原native音乐旁路按用户目标保留，视频能跳舞仍需joint/zeroM区分替代文字的音乐映射增量；aswap同时换M和native，只能说明联合换曲响应，不作M-only归因。当前已有两组best正式18视频正在执行，不拿表示指标挡推理，先完成媒体与连续人评，再决定是否续训/改变监督/改验收指标。训练CLI平台满足、完整媒体可解码和真实人评效果分别报告。

独立跨配置来源核对入口为`scripts_omnidance/exp/ms/music_direct_cross_audit.py`：只在真实18与六媒体报告完成后比较原/换/zeroM实际波形重加载SHA、缓存P/Ptags、视觉reference哈希、原native音频保留、两个scheduler各50更新及生成音轨来源；不重跑训练/缓存、不调用TC或做额外生成。`hhfl1Z`显式synthetic18正例与六类坏输入拒绝回归已通过，CPU结果`results/ms_20261020/music_direct_cross_audit_cpu_v1.json`不冒充真实媒体。首CLI自测缺upstream PYTHONPATH失败已诊断，真实入口`1V5MC2`正确环境通过。当前代码与CPU测试仅技术准备，最终真实跨配置报告仍等待18完成事件。

#### 2026-10-22 正式18学生生成、全媒体与反事实来源审核完成（真实人评待定）

唯一`xLC6yO`实际exit0，`MS/music_direct6k_v2_final18_queue_v3/finished.json`为complete/expected18，六config各三个固定人物，全部sampler job exit0。六份`*_all3_audit_v1.json`实际各passed3/complete，主线程独立再次读六report核每条124帧、704×1280、24fps完整decord和ffmpeg选定音轨解码通过；checkpoint/cache/code/日志/视频SHA绑定和noTC、P/Ptags、原视觉/音频VAE、native真实音乐与音视频各50scheduler更新均保留。没有重训或追加case，所有旧失败目录保留。无ffprobe环境下选定音视频流完整解码已核，精确声明stream总数未独立核，报告明确保留该边界，不将完整性当舞蹈人评。

`7KQMTP`在真实video manager退出后实际运行独立cross-audit，exit0；`MS/music_direct6k_v2_cross_configuration_audit_v1.json`为complete/expected18。两模型×三人物六项检查均通过：实际学生波形重加载内容SHA一致、zeroM精确零输入但native音乐与joint相同；aswap同时改变M和native音乐；同人物图/P/Ptags、visual参考及采样一致；生成音轨未被原曲mux替换。这里只证明反事实输入和来源正确，不声称M-only换曲因果或学习映射效果。

部署使用A保存best epoch3 SHA `d4760eb62db72f4327bbb054b5c3d1d947d2458b6a37b85a3f9bd7ece6e21a4b`，B保存best epoch2 SHA `014dbe054c731a7f44e10a447b6660a33773c423ed8b02fa8c1e4948f4e06dee`。这不是epoch28 last生成；两个best选择按同一0.5%改善规则，B第3轮微改善未达门槛的历史事实仍保留。

`zjrb2Z`实际在18全部审核后追加六组且保持旧models/samples/评分，生成`review_web/manifest_music_direct6k_v2_students_review_v3.json`并实际启动独立8771服务；旧8766、教师8769和用户8770不动。`okC7jK`核首页HTTP200、served manifest与文件精确一致、新六模型共18条；实际前端`api/manifest`、`api/reviews`均HTTP200。`27pSI8`对全部18视频逐条核HTTP206 Range、Content-Range和1024字节与源文件一致。原曲预设包括A/B与zeroM，换曲预设包括A/B aswap；zeroM在joint路径下，标签明确，不漏视频。人评需选择视频内音轨，避免外置原曲模式改变听到的声音。

公开技术摘要：`results/ms_20261020/music_direct6k_v2_final18_technical_v1.json`，technical_complete=true但experiment_including_human_review_complete=false/human_review_received=false。视频完成报告SHA `bdd4ddb5d4ad20c15aea402821137bb36db1daaa971aca1b1ebe365e4dbb06d4`；跨配置报告SHA `e7651eb36045f072591fcc511f6e1e639df928505046b1e7423010556c75068f`；focused manifest SHA `596bd9b445db7d051f1a0707b72e1e934bbf24ea72b2ef06ee2bc2b8d5a2eb77`。权重/Adam/视频均不上传GitHub。前序代码与验证解释已实际推送并核remote main=`0ece2491a612371c8b83270a08f8a88d1bb559cf`。

已向用户提供现实际VSCODE代理的8771页面，请对新6000 joint/aswap预设连续听看，反馈真正舞蹈/幅度/跟拍/换曲/人物保真。发布时仅本轮六新model IDs筛选实际评分jsonl，记录0条；不能迁移旧1800、teacher或noTC实验人评，也不能编造效果结论。下一步需真实用户反馈后记录正负结论、最终GitHub发布并核远端；当前保留跟进调度，不再用等待人评为理由追加预算/训练/推理。原6GPU6生成进程已结束，其他占用只读核查且不终止。

#### 2026-10-22 本轮真实连续人评：仍接近摆pose（最终负结果）

在实际两formal数值平台、18学生视频及完整媒体/来源审核、8771新评审页面和GitHub技术报告全部完成后，用户通过本轮实际人评问题明确反馈：**“还是和之前一样,接近摆pose”**。这是本轮新6000直接MusicFM tokens＋按图cached P学生的定性效果评价，不是旧1800、教师identity或自动指标推断。聊天反馈即为真实人评，不再因网页评分jsonl仍0条继续称“未收到人评”。

**最终结论：实验流程与技术验收已完成，但目标效果未达成——仍接近静态姿态变化，未建立可用的连续舞蹈。** 用户未单独指定A/B、joint/aswap/zeroM或逐case，也未单列幅度、跟拍和人物保真，不能生成逐模型排名、全部18逐条负评或虚构节拍/身份判断。既有媒体报告的musical_quality_accepted=false保持，不改成质量通过；既有技术报告补human_review_received=true、experiment_including_human_review_complete=true，并明确负向效果与观察范围。

本轮已验证的正向工程事实仍成立：6500完整cache与6000/500manifest、MusicFM骨干与直接5120映射联训、完整Adam/RNG/sampler、两个objective各28轮2632更新及既定LR平台、保原生P/Ptags/双VAE/真实音乐而部署不调用TC、best权重完整映射进入生成、18视频和反事实输入正确。B矩约束确实校准raw RMS，但没有因此建立可用舞蹈；完成训练不等于任务成功。

当前更倾向暴露**现CLIP配对表示监督与目标连续编舞之间的缺口**，而不是存在已观察到的crash、未保存、未加载完整映射、丢P或原native静音等工程错误。具体根因仍未证明：时间token顺序不受mean-pooling CLIP约束、跨曲相似动作可能形成假负例、音乐到动作关系本身不唯一、输出空间/条件分布偏差等都只能保留为待验假设，不能把本轮负评当作永久否定MusicFM路线或证明单一因素。A训练probe高而val退化、降LR后的8轮未恢复验证，不支持盲目再沿相同配方无限延长；也不能据此断言任意后续预算/监督都无效。

按用户已授权范围收尾本轮：保留全部权重、完整断点、18视频、审计与失败历史，更新代码/方案/分析/真实正负结论并以新commit推送核远端。不自动添加训练、生成行为loss、rich融合、骨干扫描或新case。最终发布核验后取消**仅本轮**跟进调度和反馈监控，保留8771新评审、8766/8769及用户8770服务，不停止未知GPU任务。后续研究需基于这一负结果独立设计受控方案，不能冒充已获续训授权。

#### 2026-10-22 用户新授权：单case原生P／教师编码消融与统一训练设计

原6000实验已最终人评近pose并发布`206c71fe2c8727a8d48edaa5bd814cce833bc602`、取消旧跟进调度。用户随后仅授权三种条件快速诊断各一个case，另要求统一训练方案解释；这不是授权重新训练。当前固定`wJPptuhclZmKAEWs`、同原图/真音乐/seed42/原生双VAE/50joint/CFG1/shift12与3/124帧/704×1280：A复用已经完整审核的`P+y_multimodal`教师identity视频；新B仅同`y_multimodal`无P；新C仅同prompt的`y_text`无P。A/B/C都是教师条件，不混入不同音乐学生权重；A复用不是冒充新生成。B/C各1条，拟用19.69和17.70各GPU6，总2卡，其他节点不占用。

主线程`hSPjJr`实际加载P/图文y/纯文字y缓存：同caption SHA `274e7a7a2d65be52e59e3974672536fb8242e89080b4175670d2d8f3532490c0`和相同228个source token IDs；图文y与纯文y张量不同，P7437。复用完成缓存，不重跑builder/LM/cache；纯文字缓存虽旧partition=train仅历史标签，本轮只做教师推理、不进入6000训练。独立新脚本`music_teacher_prefix_ablation.py`使用原pipeline已有`text_embedding`入口，去P不改原生成/教师源码SHA。B/C直接检查raw exact、P已无、text tags全1、packed text_pos228、原native音频与图VAE哈希与A一致、两个scheduler各50、输出实际生成音轨与完整decord/ffmpeg审核。去P会改变packed布局/位置，不伪称只改视觉信息而其他位置逐值不变。

初次管理任务`IE75J8`准备阶段重核已有三case manifest遇到CephFS P读等待但未新缓存；C实准入成功开始采样。B在启动GPU前准入失败，原错误报告保存。只读`yb6vSc`定位为同host CPU-only manager（CUDA_VISIBLE_DEVICES为空）被exact脚本名误计sampler；GPU6仅此前核过同uid项目holder。独立B-only恢复`ltnUeU`仅将sampler计数限定`--worker`，保主机/cgroup/GPU未知占用检查，实准入exit0；不改运行中脚本/source，不重跑C，不覆盖旧admission/partial。当前B/C尚未完成，结果及人工观感须后续真实产物报告。

`cGYaJp`仅同case表示诊断：图文y raw RMS3.82674，纯文y14.92790，差异RMS14.30449、相对3.7380，逐token平均cosine.95935；这是实际raw表示/幅值差异，不能只看高cosine声称两教师等价，也不能代替B/C视频或确定图信息为唯一根因。

单case最终技术结果已完成：C actualexit0，B-only恢复`ltnUeU`actualexit0，A复用原同case视频；`V37JDX`事件收集后实际保存`MS/music_teacher_prefix_case1_queue_v1/finished_recovered_v1.json`（SHA `16c1364cbd3e68ae58f2f4727716b05cd388732c1098126e3bae22b30d3c7a70`）。三条均完整124帧/音轨decord+ffmpeg通过；B/C各raw228×5120无P，图VAE SHA `9a20a47b2200ceb55381800ed1a625419d41834bdce1648f2589734f1fa40996`、音频VAE SHA `e7fe7b64e08f9eda36ce89b1663694b498f9cb381bf7f0f1aaf340577b48abf4`与native波形SHA `e4123d5d31c72346fe8a1ade737a7498341a817279e905916e70352d3a8b404a`均与A逐值一致，音视频各50更新，保实际生成音轨，noTC。A条件长度7665，B/C228，去P的packed位置变化明确；B/C同词ID但两编码历史/raw幅值不同，不把它们说成只改变单个“图信息”标量。

独立评审`review_web/manifest_teacher_prefix_abc_case1_v1.json`已实际启动8772，只有这一个人物和三教师配置；主线程实际API200及3视频Range206字节核通过，8771旧学生服务保留。公开精简结果为`results/ms_20261020/teacher_prefix_abc_case1_v1.json`，技术complete但人评未定。既有58829f9代码/方案发布不是结果已完的证明，恢复结果本节单独记录。用户未提供这三条实际观感前不推断A/B/C舞蹈或身份优胜，不拿之前6000学生近pose人评迁移为本轮教师结论。

#### 建议统一框架（设计，未启动新训练）

训练和生成必须共用唯一学生`S=M_theta(music)`和一个选定的condition构造器（A/B/C诊断后再决定保P、去P及教师纯文/图文）。学生生成不读caption、教师词数或teacher tokens，H3/TC/VAE权重不训。音乐特征在原25Hz双声道时间坐标下输出raw5120，N=2T；文字长度L仍可不同，不机械插值词索引到音乐时刻、不用GT词长截学生。全局、局部与行为loss都读取同一实际部署S，禁止仅训练投影头吸收收益。

总loss可写`lambda_g L_global + lambda_l L_local + lambda_b L_behavior + lambda_delta L_temporal_behavior + lambda_m L_moments`。全局项保固定train-only标准化后masked mean及双向CLIP；对确有同舞蹈描述的正例组可以另设计multi-positive监督，不能把所有不配对曲目自动当错误动作，当前6000单代表数据不等于全面语义假负例消除。moments仅弱尺度锚定，不保证编舞。

局部项以S与Y每token距离构造N×L cost、对两侧有效tokens求有边缘质量约束的熵正则OT/Sinkhorn软匹配，并加局部对比负例，避免全token匹配一个教师词。使用固定统计和部署raw、不添加可丢训练head；OT不要求N=L，但无序OT仍不提供动作先后。没有真实时间标注时不加虚假对角/单调音乐→词索引约束；若取得真正对齐音乐片段的动作事件描述，可对每真实时间窗口重复全局/局部对齐。原MS叙述里的相对时词不是已核绝对时间标签。

行为项冻结H3，在相同参考图/真实native音频、同VAE目标latent/噪声/联合scheduler时刻下比较teacher与student预测。两分支分别建自己的合法packed位置，只比较同语义目标视频latent网格（排除参考/P/padding），不拿整packed tensor直接相减；teacher no_grad，student保输入梯度，经S回传MusicFM＋adapter。训练不能直接调用整段`@torch.no_grad()`采样接口，需要底层model_fn与可微packing。噪声时间从原联合采样scheduler的多档实际(t_video,t_audio)抽取，不只sigma.9。对预测clean latent可同时比较相邻视频latent时间差、若有足够高质量教师/真实动作再做窗口运动幅度目标；不能对光流推断或文字位置赋伪时间保证。简单时间差会偏好教师轨迹，不保证pose变舞蹈或音乐跟拍。

实现可以一进程自动微分，也可复用已实测leaf VJP机制：算`gS=grad(L_behavior,S)`，把梯度送回在线MusicFM并对`(S*gS.detach()).sum()`反传，严格绑定本步checkpoint/RNG/波形/output SHA，不用旧预测cache梯度。当前已有旧VJP是静音/旧adapter小实验，必须新测真实native/当前部署链，不冒称已落地统一框架。每个更新S只前向一次；cheap global/local用实际批，昂贵behavior可按预定比例选同batch子集，完整梯度同一次optimizer.step，明确batch大小/真实负例池和loss分项梯度，不能用累积伪称大CLIP候选池。初始保持global-only/moments历史对照，先小批真实链路smoke，再逐项加local/behavior的单变量诊断，不一次混所有项。

统一框架必须分别验收：全局留出语义判别；局部token覆盖/时间窗口对应（如有真标签）；H3多噪声段heldout行为误差与运动趋势；最终原/换/zero反事实连续视频和真实人评。跟拍若教师不跟拍，需要独立音频时间锚（beat/onset与真实动作事件/可微motion表示），不能以matching文字或teacher velocity宣称已解决；该项数据与可微链路未实现，需单独资源方案与授权。当前只执行上述一个case教师消融，不开统一正式训练。

#### 2026-10-22 单case真实反馈与LM输入／输出替换位置澄清

用户实际听看ABC教师单case后反馈：**“看上去不同的H3语言条件好像结果差别没有很大,尤其是b和c几乎一致”**。这支持该case在保原VAE参考与真实音乐下，删P或改为纯文字教师没有明显可见差异；不是逐像素等价、所有prompt都无需P、舞蹈/跟拍/人物保真验收通过。已收到真实聊天反馈，不继续等待网页评分或追加case。

用户进一步提出是否只需对齐“经过tokenizer的[L_text,5120]”。实际tokenizer只返回整数IDs [L]；冻结LM词表embedding查表才产生输入X=[L,5120]；随后多模态语言模型处理图文统一序列产生上下文化hidden H=[L_total,5120]，H3模块虽称text encoder，内部是Qwen-VL视觉网络＋因果language_model（非另一个DiT）。原6000训练对齐的是LM**输出**H的prompt后缀y（实际loss对raw y做mean/RMS读出），学生raw输出直接进冻结DiT的condition_proj/token_refiner，生成没有再走VL语言模型。

对齐LM**输入**词embedding是另一种可设计的soft-prompt路线：音乐→S_in [N,5120]，通过冻结LM的inputs_embeds接口得到H_out，再进入H3。纯文版本只运行language_model、不加载VL视觉分支；保图文版本需要正确的VL输入图像features/引用词embedding、多模态位置与deepstack等，不能把最终缓存P输出拼在LM输入前当原图像输入，因为缓存P已被编码，空间相同但层级不同。若要缓存VL前缀计算以节省显存，应研究同层级输入features或每层KV/位置协议，不把P hidden当KV。

输入embedding对齐不必等于输出hidden对齐，也不保证更容易或能逐词复现；N与文字词数可不同，mean-pool监督仍可能丢顺序。可联合LM输入侧配对/局部目标与LM输出侧表示、最终冻结H3行为loss，但学生训练与生成都必须包含同一冻结LM前向。LM参数冻结不意味着对学生输入使用no_grad，必须让输入梯度通过LM回到音乐encoder/adapter；现纯文字缓存helper用inference_mode只适用于离线教师，不能原样作为可微学生分支。新路线引入LM推理/反传资源、无TC部署目标相应变化，当前仅解释设计，未获正式新训练授权，不自动启动。B/C视觉近似只给教师条件选择线索，不能证明音乐soft embeddings训练已对齐或这是原近pose根因。

#### 资源与最终验收

仅允许 `28.216.19.69`、`28.216.17.70`、`28.216.19.213`、`28.216.19.143`、`28.216.19.137`、`28.216.19.225`，禁止18.136/18.144；不能用172.19地址或尾号匹配扩大授权。6500缓存预定六节点各一GPU，完成后A用19.69四GPU、B用17.70四GPU，其他四整节点明确释放。P约7437×5120×2字节/图，6500条可能约461GiB（另加teacher/metadata），需先检查磁盘，训练只读统计表和波形，不能把P全部load进内存。

下一验收依次为：真实cache producer/consumer字段一致，完整4卡forward/backward/Adam与实际global64，原生教师同y回填raw/tags/packed/首步音视频预测，保native音乐的两学生原/换/零M固定三case共18条，完整音画及来源审核与连续舞蹈人评分开。最后发布代码、方案、真实结果和正负结论并核远端commit；不上传权重、Adam大checkpoint、视频或秘密。

### 2026-10-22 新主线正式获批：音乐到离散caption，再caption＋参考图到真纯视频H3

本节取代以上旧6000／单case阶段的资源和后续任务限制，不改其历史协议：用户本轮明确授权全部8节点64张A800-80GB，要求先同步计划；完整计划已通过计划审批，随后要求自动执行、修错、持续跟进直到两个新实验正确结束，并将方案／代码／分析／正负结论推送GitHub。已确认首版music-only，**动作编排与外观分开**；参考图保留，但**不新增目标首帧硬锁**。旧直接MusicFM hidden实验和ABC教师诊断已经结束，不默认重训或追加case。

#### 两项任务与严格数据定义

1. 音乐→可读英文舞蹈编排caption：本地Qwen2.5-Omni-7B Thinker快速对照，官方Qwen3-Omni-30B-A3B-Instruct Thinker主候选。仅文字输出，不创建Talker／Code2Wav；保留未来图像能力但首版不输入图像。官方30B固定revision `26291f793822fb6be9555850f06dfe95f2d7e695`，25文件约70.53GB；下载仍在进行，不把本地既有audio任务全参数captioner当官方未训练起点。官方Captioner仅单音频、无指令／图片，不作主线；Music Flamingo非图像且许可受限，只作备选说明，不默认上卡。
2. 期望caption＋参考图→视频：沿现DiffSynth原BF16 Ref2VA独立video-only入口，保原Qwen-VL图文TC与视觉VAE，两者及baseDiT冻结，只训DiT LoRA。绝无audio VAE加载／编码／解码、target audio noise、非空audio latent／reference、tag2／audio_pos、audio patch／head／scheduler；仅允许(2,32,0)长度零tensor作类型占位。生成文件仅视频轨；评审外部同步音乐不等于H3输入音乐。

rich全文11992可恢复且覆盖6000/500；原txt与旧AV包装内五章正文吻合，最初生成模型／指令未知。新目标用前三章叙事／动作与原MotionScript事实融合重写，后两章外观／场景及旧音景包装排除，所有raw／排除／冲突侧车保留。原MS `kin.frames=101`但无独立fps／绝对时间映射；归一化媒体约5秒不自动证明rich/MS同窗。首32实际来源24train/8val、16genre，音频5.0625秒、源start0/fps16、显式window=[0,5.0625]，**32条rich/MS scope仍unknown**。允许冻结7B text-only生成32预览，禁止伪造人工窗口／事实／冲突确认后导出训练，禁止均分词句制造秒级标签。

输出是“适合音乐的可能编排”，不是宣称音乐能唯一恢复原视频动作。`motion_caption`为原正文，`caption_version`逐条必需；唯一`video_prompt`包装<Subject 1>/<Picture 1>及固定参考外观／环境／单镜头。Omni metadata明确id/source_group/split/messages/audios/audio_window_seconds：user只有固定instruction＋音频，GT仅assistant，不泄漏rich/MS全文至输入。新caption必须新编码，旧y／旧AV packed不可复用；视频latent仅同窗口且内容验证后复用。

#### 预算与门禁（批准配置，不等于已运行）

- Omni：LLM attention/MLP LoRA rank32/alpha64，最后4audio block LoRA、audio projector全训；7B投影proj，30B投影proj1/proj2。30B专家MLP gate_up_proj/down_proj为3D Parameter，PEFT target_parameters必需，不以attention-only冒称MLP已训。视觉冻结并禁止调用，assistant caption＋EOS自回归CE，user/audio/padding labels=-100。起始LoRA lr5e-5／投影lr1e-5、BF16＋FP32可训参数。先8样本2实际更新／完整恢复接口smoke；fixture标签只接口，不作正式质量证据。获审目标后512/128三完整epoch，再6000/500 min3/max10、每轮固定val assistant-token NLL＋自回归输出、patience3／delta0.005，cap结束未证平台如实标未确认。
- H3：原BF16、CFG1、1280高×704宽、124帧／24fps、50视频步；先真无音频pretrained forward/backward与冻结rich/MS/fused各1case上界。DiT attention qkv/out＋MLP fc1/fc2 LoRA32/64、lr1e-5、BF16／activation checkpoint／节点内8卡ZeRO3。pureFM和sigma≥0.5 soft base-prediction anchor λ0.1／1同初始化先100成功更新；anchor仅额外正则，不硬替FM目标。固定3case真实人评合格后300，再获验收winner完整6000预算。像底座不等于动作学习成功，自动指标不代替人评。训练／推理LoRA alpha/r缩放必须显式保持一致。
- 端到端：固定3case，GT／音乐预测caption × frozen／finetuned H3最多12视频，可复用严格同设置基线。H3始终不读音乐；同caption换外部音乐不得改变H3。GT也差定位H3，GT好预测差定位caption阶段，不盲目同时加预算。

#### 八节点Wave安排与资源安全

权威顺序N1=28.216.19.69、N2=28.216.17.70、N3=28.216.19.213、N4=28.216.19.143、N5=28.216.19.137、N6=28.216.19.225、N7=28.216.18.144、N8=28.216.18.136。本轮已批准最后两节点，不拿旧阶段六host限制挡新主线；上卡仍逐次核同uid／精确argv／CUDA映射／GPUUUID及宿主cgroup内存，只释放确认项目holder，不kill未知任务。

Wave0：N1首32冻结重写（目前单卡即可），N2两Omni真实processor／梯度／恢复，N3纯视频H3接口，N4三caption冻结单case，N5–6新caption／视频VAE资产绑定，N7–8本地资产校验与必要的两节点NCCL实测。Wave1：N1 7B SFT、N2 30B SFT、N3扩量重写、N4 cache／caption验证、N5 pureFM100、N6 anchor0.1、N7 anchor1、N8固定case评测。依赖与显存/NCCL真实通过后才扩16卡，不先64rank整体启动；未用卡让用户其它任务，不把占卡matmul当实验利用率。

#### 当前实际实现与验证边界

新增独立 `dance_caption_fusion.py`、`omni_dance_sft.py`、`h3_video_only.py`、`h3_video_sft.py`，不原地破历史AV源码／缓存／权重／视频SHA。主线程已独立跑融合8项CPU边界／导出测试、两个真实随机小Omni各两更新及LoRA/projector/Adam/RNG/sampler重载下一更新一致、真实小H3音频禁hook／前后向、PEFT软anchor／完整CPU恢复／native LoRA导出预测一致。纯视频还新增完整unit-chain两步采样＋假videoVAE解码测试、仅一视频scheduler初始化；测试mock初次递归错误已修，真实复测exit0。**以上均不是完整预训练GPU链路、SFT收敛或画质验收**。

最终首32source清单 `MS/dance_caption_preview32_v3.json` SHA=`bc7d36e820ef274089e6d29e3533de1b5aa8914381f3643d3dba79ab1a0eabbb`，fusion源码SHA=`111b1baa7f180d25044018a586edb0af7b11e75dd6353b7027880652e4ed06f4`。真实本地7B text-only processor返回仅input_ids/attention_mask [1,1134]，明确audio/images/videos=None；未据此声明模型forward通过。已启动唯一N1 GPU0冻结32重写manager，产物未审核前不声明caption生成或人评通过。Omni pretrained梯度smoke、H3 pretrained纯视频／GPU恢复、首32窗口与事实审核、两阶段pilot/full／视频／最终人评尚待完成。

保存恢复必须绑定源码／model revision和完整权重内容／caption及media window／cache SHA、实际worldsize与sampler。只有完整Adam/RNG/progress且完成marker/文件SHA通过的state才resume，partial保留，不覆写、不冒同init重跑为resume。H3 cache TC＋VAE需实际驻留预算／offload验证，不能用initialize-on-CPU假称GPU阶段显存已解决。

所有阶段方案、代码、实际测试与正负分析安全新commit/push直接核远端；不上传权重／Adam／视频／秘密，不amend／改git config／绕hooks。新两项训练、完整视频审核、真实人评与最终GitHub远端核验全部正确结束后才取消仅本轮跟进，保旧8770/8771/8772和未知任务。

#### 2026-10-22 Wave0真实启动、训练部署互通与新发现（仍非新实验完成）

核心方案和三入口实际新commit `7b997ffd325429979a722ff3907d8fbcb86332ad` 已push，`VIHZVi`直接核remote main同SHA。新H3训练入口 `h3_video_sft.py` SHA=`417d5640fcd63b9c745a4fa81ecf0505477e7739069150600e53c3545f735867` 已补：cache阶段TC/VAE真实CPUoffload、专用runner不调用whole-model.to／Accelerator.prepare；主rank实际hash并广播source/model/media/cache；accelerator完整state全部rank成功后写文件SHA completion marker、缺marker/改文件/改source/model/cache拒恢复。主线程`SuU6aD`独立CPU复测通过。这仍没有完成GPU/ZeRO3恢复验收。

独立新 `h3_video_infer.py` 最终SHA=`77bafd9f6165f80328d4e272c901b21fbf45da3898161bcc039a572e1c86a2f8`，主线程`CuVqyY`独立7CPU测试通过：真tinyH3＋PIL参考完整unit-chain→新TC条件缓存→weights_only回读→视频采样；CFG1原unitrunner把正输出同对象镜像至negative，不能把字典镜像误称额外negative编码，已clear并核真实TC一次。native scaled LoRA safetensors按alpha1加载，白名单同时允许blocks和token_refiner.blocks，A/B双向完整检查；两位置融合weight和完整预测与独立base＋B@A相同，避免重复乘alpha/r。真实静态fixture124帧704×1280/24fps已ffmpeg全部input streaminfo核仅1video/0audio，-xerror全RGB解码335216640字节及每帧showinfo PTS/size正确；含audio、错帧数/fps/shape反例拒绝。此静态fixture不是生成画质通过，不上传fixture视频。

当前实际队列：N1单GPU0 `PGLuNL`/`MS/dance_caption_wave0_queue_v1` 冻结32重写，准入仅release同uid精确项目holder3136724，pre-model-content-SHA冷读中；不能无输出当停机重启。N2 `miDbLg`/`MS/omni7b_interface_smoke_wave0_queue_v1`已pidfd绑定同uid现有7B本地staging `11Dfs6` 的ssh退出，之后核全部local分片内容SHA／完整staging marker才重新8GPU准入；不重复stage或训练。Omni接口metadata8train/2val SHA=`9240e0ec3be227ff82652d7be3e2af86c1fdc12980fd7c188f6829e2401df953`，sidecarSHA=`3c1140c320f160f8d06da215c7580072b3723481f59c4885d7838389f91f1a2b`；真音频＋固定明确fixturecaption，只两更新检查梯度/完整重载，不正式SFT／语义人评。实际env没有torchrun脚本，使用python -m torch.distributed.run；Gloo2rank／ZeRO1／tail零loss同步／完整checkpoint已CPU实测，但NCCL不据此宣称通过。

N3单GPU0实际 `beaVgA`/`MS/h3_video_interface_wave0_queue_v1` 正在新8条caption＋参考图cache；实准入release精确项目holder4110957，原TC/VAE已加载，无DiT/audioVAE加载。H3接口metadataSHA=`5290c08e019ebe43662f7dda8e552bc367bd0b18e3e6b45ef3d4f8dc73ec9d0d`；sidecar最初SHA3d5147...，补真实loader观测后实际SHA=`16d96437145766eba72d58b9544632c7b17e2cfa1caa6f344ae87e20ea872817`，不混称原sidecar未变；metadata/source未漂移，运行不读sidecar。唯一fresh训练manager `56fe2B` 已pidfd绑cachemanager271905，cache真实完成且8条completion marker核验后再8GPU/主机内存准入，2成功更新，明确不使用resume。N4 `g96Dms`/`MS/h3_video_frozen_baseline_case1_queue_v1` 两GPU0/1，rich-motion-only和MS-motion-only各一个同case `FqlHJVDlFGLKCUJS`，先stage cache再DiT＋VAE生成；融合第三caption等待32重写，不擅自增加人物。

**重要时间窗口新发现：** 原视频81@16fps=5.0625秒；imageio duration约5.06，24fps可用floor121，原loader短源按17n＋5向下取**107帧**而非124、不补末帧。八条真实CPU完整decode均107帧、704×1280，只覆盖原index0..71（72唯一源帧），107/24=4.458333秒，和music window5.0625秒差0.604167秒。当前N3仅接口fixture，不能称124帧训练／音视频窗一致；生成仍124@24=5.166667秒。正式caption-video监督必须明确合法同窗或显式尾部padding协议再核，不静默用旧loader全片rich。独立观测`MS/h3_video_interface_loader_observation_v1.json`与diagnostic保存，旧文件不重标时间。

**已发现但未修改的恢复语义缺口：** 当前TrainingProgress.load_state_dict只验signature，对缺progress字段／负batches_consumed实际接受；marker/hash验证只证明序列化文件完整，不证明cursor语义。独立拟议strict requiredfields/int非bool非负／local-loadercursor及epoch预算上界／marker.progress与加载state逐值一致的7反例已CPU测试，但生产417d5640尚未修。运行中source需冻结以免cache指纹失效；本次fresh两更新不使用resume，不据此宣布正式恢复已严格通过。训练恢复前必须新source修正并正确说明历史cacheproducerSHA，不隐瞒更改或冒称resume可用。

首32所有scope仍unknown；正式512/128与6000/500目标导出仍须真实facts/narrative/window/conflicts四项审核与rewrite内容SHA。现旧data_review页面是旧2k/MS评分schema，不满足新人工门禁，正在新增独立8773 opt-in reviewer（不覆盖旧服务/评分，不代理POST实际人评）。新GPU更新、完整生成、人评与最终质量均尚未验收，跟进Cron `e06d1c4f`仍运行。

#### 2026-10-22 Wave0真实两模型训练更新、纯视频生成与EOS根因修复

`75d2b05e201ea9a2d0553eaaf22d26e5569338b4` 已实际push并核remote main同SHA，包含隔离纯视频训练／部署入口及前述边界。后续实际产物均与代码版本独立绑定，不把发布或CPU测试当作实验成功。

**7B预训练接口实测已完成：** N2本地5权重分片实际内容SHA核验／暂存结束，唯一`miDbLg`8GPU/ZeRO1完成2真实updates，16example曝光、336assistant tokens、effective batch8；训练NLL4.302760→3.994139，首轮val42tokens NLL4.018852。真实梯度LM196/audioLoRA24/projector2非零（是非零梯度参数tensor数不是层数），音频末4block及LLM/投影均进入同一优化。完整adapter/projector/Adam/每rankRNG/sampler/processor保存重载验证通过，checkpoint `checkpoint-u0000002-e01-p0000008`，trained adapter SHA=`9ea9dc1fae00bf3b0967958c8369f8587b10abd6c75bf6028c301151e2f780b8`。这只验证8train2val固定fixturecaption的两更新链路，不是正式SFT、泛化或语义成功；不复用该fixture权重做正式起点。报告`results/ms_20261020/omni7b_pretrained_smoke2_and_eos_v1.json`。

**H3预训练真纯视频接口已完成：** `beaVgA`实际cache8结束，主线程`EQWfB6`逐条完整SHA/weights_only/新captionversion/目标latent `[1,24,32,80,44]`、零audio positions/tag2、无目标首帧锁核通过；输入107帧与时间窗问题如前节。唯一`56fe2B`随后8GPU/ZeRO3实际2更新（fresh，无resume），FM/softanchor forward/backward及Adam成功。step1 loss0.177884683、sigma0.0704247；step2 loss0.214736864、FM0.127068192、anchor0.000118441807、sigma0.5155094，softanchor确实生效。实际416LoRA A/B tensor，其中208B非零、含token_refiner；native导出约310MB。完整training-state-2/final-2均保存完成marker，8Adam shard与8RNG文件；final marker SHA=`c80a06e9a7874fc424fdc49c43f8b814871aef93aecaad69b71ff4740a0ae217`。为了完整恢复，DS state包含约66GB冻结base分片，CephFS保存／SHA冷读占主要等待，不重复启动、删文件或把读取延迟当crash。只验证两更新与完整保存，尚无真实GPU restore+下一更新验收、不声称收敛或质量。报告`h3_video_only_pretrained_cache8_v1.json`与`h3_video_pretrained_smoke2_v1.json`。

**冻结纯视频上界已有两条真实视频：** `g96Dms`N4 rich-motion-only/MS-motion-only各1同case `FqlHJVDlFGLKCUJS`完成，主线程`2rAon1`独立再次全部RGB解码核124帧／704×1280／24fps、全input stream枚举仅1video/0audio，两个模式真实DiT50次／VAEdecode1／TC0（新条件已独立cache），audio patch/head/referenceencode全部0，audioVAE/audioScheduler永久None；图来源相同。视频SHA rich=`d05bff7cd6381155653e312405c0333cfdf5b8b8170f484e360bb7a2576abfde`；MS见独立报告。此时没有用户观感，不能宣布连续舞蹈／人物清晰／跟拍通过。独立8774 focused评审实际首页200／API gzip正确解压200／两视频Range206字节同源核通过，本地code-server `/proxy/8774/` 首页/API/Range也核通过；外置wav仅用于听看，音乐没有进入H3。复用两条不重生成。融合第三caption初次NqrPjO在GPU0busy准入拒绝且无产物，查明是watchdog新项目holder；只精确核uid/argv/env/UUID后release该holder，独立恢复`a9mnBI`，旧failed队列保留。

**EOS工程根因与受控重写修复：** 首版冻结32重写实际产生9条全invalid，JSON后继续生成虚构Human问答，旧partial及stdout原样保留。实际7B磁盘generation_config未设EOS，Thinkerconfig/tokenizer eos151645，text_config EOS亦null；直接Thinker.generate读取磁盘空EOS造成无法及时结束。主线程确认唯一冻结rewrite PID234177/uid/argv/output后仅终止该错误任务，未停训练/旧review/未知compute。修为全部文字生成显式tokenizer EOS/PAD，CPU AST核所有generate调用，保新source，不从旧Adam“跨源码resume”。纯部署`P78JvF`复用已完成同2update checkpoint，实际输出一段正常caption无Human尾缀，不新增训练；原有错误生成保留。

冻结融合修复只小批诊断，不扩正式训练：v4显式EOS结束正确但两个JSON仍有code fence／conflicts类型／性别代词错误；v5真实processor验证assistant `{` prefill（不是截输出），两个精确JSON中1格式过，首case仅rich遗漏MS、第二性别代词；v6仅输入事实性别代词中性化并保raw manifest，保存actual_rewrite_material及SHA，强制融合具体MS事实和真实fact IDs，两个均精确JSON/无codefence/无Human/无身份词通过。v6两个used_facts均含rich及MS，但lexical action omission仍flag，**格式通过不等于事实覆盖／窗审核**。原所有失败版本保留，不偷偷清洗输出或把软件自动选择当人评。实际同case融合caption已用于仅1条冻结H3推理上界，不作正式SFT目标。批准范围首32按已修v6开始新目录 `MS/dance_caption_rewrite32_normalized_v6`，唯一manager `XJO7T3`N2GPU0；全部32格式／事实检查完成前不导出512/128。报告`dance_caption_preview_failures_and_repairs_v6.json`、`dance_caption_rewrite2_repair_v6.json`。

**恢复cursor语义已在新source修：** H3两更新运行完整结束后，修改TrainingProgress严格requiredfields/int非bool非负、updates≤microbatches、cursor≤local prepared loader、epoch/updates预算，并在实际accelerator.load_state前后核marker.progress与custom state精确一致。主线程`GFAc1j`生产真实tinyH3/fakepipe三更新／完整Adam恢复下一步bitwise一致以及缺字段／负数／bool／cursor/epoch/update超预算等拒绝回归通过。该修正新源码SHA=`8d88f84963f7d7c8f7ba18b4e556656683f300b4d939015bcf0e57670ac90319`；历史cache和DS状态仍producer417d5640，不重标老source，不伪真实GPUresume已通过。后续正式captioncache以新代码生成；GPU恢复需要同版本完整state另实际验收。

**独立caption审核页：** 新 `dance_caption_review.py`标准库HTTP，12CPU synthetic HTTP测试通过（正例POST只TemporaryDirectory fixture，非真实评分）。显示raw rich/MS全文、事实ID、配对图／短片／5.0625秒音频、候选／usedfacts/conflicts/unknownwindow；原sourcevideo无manifest明确SHA则只显示未核，不扩大白名单。四checkbox默认false、实际reviewer/window_evidence/notes/窗口显式提交，server绑定真实manifest/rewrite SHA＋sessionnonce/CSRF/origin；32全四true且目标formatvalid才preview32_approved。GET不建评分文件，不迁移旧reviewjsonl、不代用户POST。修相对API/media routes及显式固定public-origin支持code-server proxy，未信任任意Forwarded头。首旧v3页GET/Range核通过，但外部代理URL连接timeout；新版8773已改绑v6候选与独立reviewout，本地codeserver GET可用性继续核。停止旧8773仅为切换同任务候选版本，保旧8770/8771/8772/8774。正式窗口/事实/冲突人工审核仍未完成，不能因裁剪音频或H3完整cache宣布审核通过。

官方30B下载nppvRl已actualexit0，15分片总70,523,299,202bytes、25官方文件齐；HfApi固定revision/LFS内容SHA/Gitblob取得，N7唯一 `SrdYyD`开始完整内容核验＋local immutable staging，已核的头两分片匹配官方SHA，其余进行中。不得拿文件名／下载退出当全部内容核验或30B梯度成功。本轮整体仍在Wave0/真实人审前，未启动formal512/128或6000/500，不取消`e06d1c4f`跟进。

#### 2026-10-22 真实人评：两个纯视频基线均可用，进一步融合正文

用户观看8774里原始H3真纯视频rich-motion-only与MS-motion-only同case两个已完成视频后明确反馈：**“两条都可用,你可以试着把这两条融合一下”**。本次两条基线的人评已收到，记录quality_accepted=true，不能继续称pending或将旧音乐hidden近pose负评迁移过来；可用结论仅当前两条，不自动推出节拍／身份各维度已单独通过或所有32caption窗口／事实已核。

按用户要求融合的是**caption内容**，不是视频拼接或权重混合。保rich的芭蕾／现代舞质感、轻盈连贯与表达，加入MS的左／右肘弯伸、逆／顺时针转向、手靠近再远离膝盖、左右手离对侧肩、双肘收拢与手靠肩／彼此靠近等五项具体事实。助手直接编辑融合173词，保原事实侧车和未核全局顺序／窗口冲突，去性别代词与伪秒数，不冒称Qwen产物或人工训练目标；跨动作连接只是合理编排提案。

新输入 `MS/h3_video_baseline_fused_case1_humanrequest_v7.json` SHA=`b367f18b1406092f6328f3b2c0e9d294b9e3d8eaa27f0f66e2373f394bebd143`。原始两条可用基线不重跑；新增仅一个同case `FqlHJVDlFGLKCUJS`、同图／seed42／CFG1／50video／704×1280／124@24，没有H3音频输入与输出，不加硬首帧。前一自动v6名为fused的正文近rich-only，即使列出MS IDs也不代表融合覆盖；保为历史诊断，不用其结果替这次具体融合。计划N4 GPU0单slot，逐UUID／uid／argv／主机内存准入后实际生成，完整媒体核验后加入独立8774与两条原视频对比；未生成完不预判效果。

该融合v7已实际完成，唯一生成5SbRq3 exit0，独立dkkewI完整媒体／同图／源SHA核验通过：视频SHA=`98c68983c04196593a96aa05cd3a2d43b3d5818a9029a43f89adf2b959e8153c`，124帧／704×1280／24fps，真实50video更新、VAEdecode1、audio patch/head/referenceencode全部0，无音轨、无audioVAE/scheduler，参考图与两个已获可用反馈的基线同SHA。结果`results/ms_20261020/h3_video_editorial_fusion_case1_audit_v7.json`，**此时融合本身尚未有人评；随后用户实际反馈见下一段，不把该历史pending状态当当前结论**。自动评审接续manager误匹配自身导致pidfd自等待，主线程只结束这一个已知CPUmanager，不重复生成；实际构建独立三配置manifest并替换仅8774旧双配置server，保8770/8771/8772/8773/未知任务。现在仍同1个case：rich、MotionScript、融合v7，外置wav只供评审同步，不进入H3。

#### 2026-10-22 融合v7获真实正评，追加动作维度快速迭代

用户最新原话：**“融合更好且可用,此外,之前你还预设了很多其他的维度,也可以考虑加入prompt中进行尝试.我们需要快速进行迭代,时间来不及了”**。该同case融合视频已获可用且相对改善的真实定性人评，报告已更新；不是逐项分数、32caption事实/窗口批准或正式训练验收。

复用已接受的v7视频与caption正文，优先从现有多维标注核实真实动作属性，冻结两组最小增量、每组仅一个同case视频，保持同参考图、原始H3真纯视频、seed42/CFG1/50步/704×1280/124@24。不猜BPM/秒级动作，不新增外观或镜头标签、不拼接视频。实际准入、生成、完整媒体/来源审核后再请求快速人评，不能在运行前宣称增量更好。30B的PEFT专家baddbmm失败另行准确诊断修复，整体正式pilot/full与端到端目标仍未完成。

两个新增配置已按真实来源冻结：A仅追加“手臂主导＋优雅梦幻表达”，B仅追加“直立＋主要原地＋无跳跃”。同case旧VLM `MS/vlm/labels_s1.jsonl:261`直接支持上述值，preview32_v3的kin显示arm_speed0.4533高于leg0.1834/torso0.0888，pelvis_h_min0.9271、jumps0。保v7逐字主体；energy low/medium和turns around/none冲突不加入，quality smooth/flowing已在正文不重复。都是推理编排诊断，原事实窗仍unknown，不能当人审GT。

新124帧显式协议源码SHA `9f5de5e200098e482313ac01938b22ef7333037b726fec22ccbf4eea6f2e9505`：opt-in `nearest24_repeat_last124`，metadata必须显式video_window_seconds=[0,5.0625]。实际CPU同casevideo/reference加载与逐帧映射已核，全81源帧、122自然位置＋2尾部repeat-last，124@24输出5.166667秒；padding不增加动作事实。报告 `h3_video_repeatlast124_dataset_cpu_v1.json`。本次独立CPU回归RB4b7X实际exit0，真实tinyH3 LoRA/anchor、完整Adam/RNG下一更新bitwise恢复、strictcursor/marker坏输入拒绝均通过；第一次PY7rs8仅因未传CLI必需dataset_base_path exit2，补足后通过，不是算法失败。新GPU缓存/训练/恢复另按同版本执行，原107帧producer417d5640的旧缓存/state保留不重标。

官方30B真实失败已独立读rank0完整Traceback（recovery_queue_v2/training.log:1089–1113）：`torch.baddbmm`申请384MiB时GPU0仅107.81MiB可用，PyTorch allocated77.42GiB，实为CUDA OOM，不是dtype/shape。官方下载与25文件内容暂存已经完成不重下。失败updates0，initial checkpoint存在但没有成功Adam更新；恢复与fresh以完整状态和源码版本严格判定，不能称30B梯度通过。同版本CPU已确认预训练eval模式使checkpoint开关未实际进入：新PEFT wrapper.training=True但base与decoder.training=False，decoder.gradient_checkpointing=True，原生产入口未调用train()；实际Traceback走普通forward分支。最小修复为DDP构造后`wrapped.train()`，保原nonreentrant checkpoint、全部专家/audio范围与FP32可训参数。独立3层真实tinyOmni CPU对照保存tensor唯一bytes246171→31375、checkpoint调用0→1、专家大tensor保存17→0；两更新LM/audioLoRA/projector均非零，adapter变化而base不变。新增自测从eval base开始并spy真实checkpoint调用，正式GPU能否足够仍需实际验收，不因CPU降峰直接宣布30B修好。

实际新链路已启动：N4双GPU0/1新维度cache manager `KpvJSZ` / remote3773121，cache子进程3773194/3773195，完成后同队列进入两独立video；N3新124cache `NFFNKn` / manager31060/producer31067；N7修复train模式后官方30B fresh `uyggZM` / remote3161478/torchrun3161607。都绑定新源码与新输出，尚未以launch声明成功。阶段提交 `05c759179bd998309108c543bca622c868ebe9d5` 已push并直接核远端main一致。

**融合校验角度误报修复：** 历史v6实际生成32、当时格式26过6拒的报告保留。`at 90 degrees`是角度而非时间，最小调整时间regex只豁免明确degree/°单位，不放开裸数、秒数、时间区间/帧数。新增CPU正/负回归第一次发现句末小数点guard误豁免裸数，修为只拒decimal续数字后，Cm5qV0全部9项过。对原32 raw_generation重新严格JSON解析和当前验证ca8PRY exit0，27格式过/5拒，32原文件SHA全部未变；唯一变化oEyYaJWyGGtdtgWp仍有窗口/动作遗漏flags，不称语义获批。人工批准仍0，没有清洗模型输出或训练导出。当前validator SHA8e3ddf0e44d8e6b5c4084fd00401dd86fc6820794715a2f2144b8b0a8d5fb0e5，独立报告dance_caption_angle_validation_recheck_v7.json。源码guard要求重启审核服务，主线程核UID/完整argv后仅pidfd替换旧8773 PID393189，保原manifest/32raw与同评分路径；新BXYBuj服务的本地code-server代理GTt8Gm实际首页/API200、媒体Range206/1024bytes与原PNG一致，27ready/5invalid/0approved，GET未创建真实评分文件。初探未带session cookie被正常403拒绝，补page-issued cookie；另探误用counts.actual_approved改读实际counts.approved，未改production schema或放松防护。保其它review服务，正式监督门禁仍未完成。

#### 2026-10-22 动作维度v8两视频完成并上线；30B真实首更新后再OOM

N4 `KpvJSZ`真实exit0，两新caption各cache/采样完成，主线程`4iwolz`再次全RGB解码核124帧/704×1280/24fps/1video0audio/PTS，模型全部来源SHA与v7相同；各50DiT/50scheduler/VAEdecode1/TC0、audio三类hook0、audioVAE/schedulerNone。A videoSHA5c9ca3d95cc088917d70ae17f54e9f9571c0c880a5c0225376c47e530132ea28；B 24709d3847210fc8bd74b6e1a7d3681c9314054ca992e4f9681fba5102fba6a6。旧v7不重生，正文逐字仅独立追加，未混权重/视频。完整技术报告h3_video_dimensions_case1_audit_v8.json。两阶段manager记录freshUID/argv/CUDA/UUID/GPUmemory，未记录host/cgroup180GiB门禁，不补称做过；实际两视频成功且只释放确认ownholder。

独立新manifest_h3_video_dimensions_case1_v8.json复用三旧可用视频，首预设为v7/A/B，仅1case；原rich/MS/v7预设保留。精确替换已核8774 PID629441，新服务MMK8sK；vdQkNM实际code-server代理首页/API200、三视频Range206各1024bytes与源一致。初探错误media?path改用网页实际media?p，不改production服务。新视频质量仍待真实人评；外置wav不进H3，不能因完整技术审核宣称维度效果更好。

官方30B新trainmode_v3首次真实Adam更新成功：NLL4.10893，8examples/168assistant tokens，LM288/audioLoRA24/projector4梯度tensor非零，192expertLoRA中96非零（A初始零正常）。第一轮完整val NLL3.262013/42tokens，已有1条epochvalcaption（保原入口语义，非正式质量），随后u1完整检查点已提交。CPU实际marker全fileSHA、628Adamstates step1、8rankRNG/sampler/history及processor逐tensor重载通过。保存期间隐藏.checkpoint先adapter后training_state FUSE写入经/proc实际核实，不是consolidate死锁。第二窗口在step前再CUDA OOM申请768MiB，free257.81MiB/allocated75.64GiB/reservedunused1.70GiB；partial updates1/epoch1/position0/boundarysafe true，未到u2/run_complete。具体forward/backward/report缺原Traceback，不能把768MiB直接当专家op；正在精确清理已失败workers并诊断首次Adam moments常驻与DDPbucket/grad双份峰值，不降expert/audio scope。旧u1状态保持不重标，跨源码不得假resume。两更新尚未通过，正式SFT仍未完成。

后续真实NCCL timeout与torchrun自然exit1后取得原Traceback：v3/training.log:898–921明确第二更新`.backward()`进入transformers `integrations/moe.py:236 _grouped_mm_fallback_backward`，创建`grad_weight=torch.zeros_like(weight)`申请768MiB OOM。不是forward/保存故障。全部worker自然退出，无人工kill；完整u1仍保留。下一候选仅DDP gradient_as_bucket_view与slot完成后释放局部output等，先CPU真实两rank同scope/AdamRNG下一步等价，再GPU验；不删专家/audio或改变PEFT折叠舍入。

正式获审目标的导出工程缺口已补：fusion源码b53aece351adf3f94d84f29c0caf5494699132bd85abbeb0a9594377bf9d7f75，新omni.jsonl可供实际reader取train/val；每行顶层caption_version，GT只assistant。新H3独立h3_video_train.json/h3_video_val.json保同motion_caption、图/video真实SHA、显式[0,5.0625]与真实审批来源；combined h3_video.json仅交换/审计不能直接cache/train，因为H3 reader不筛split。export拒fixture/禁formal、混version、缺四项真实审批、bool假秒数、源SHA/路径改变或trainval泄漏，所有source导出前两轮校验；unknown历史factscope不伪改verified。主线程HcvvxA独立10tests/31负例通过，全部只TemporaryDirectory合成审批模拟，不导出真实32。新8773R1zGa0绑定冻结源码，snk71D实际代理首页/API200、27ready/0批准。全量训练门禁仍未过。

已完成下一最小显存布局修复CPU验证：AdLoia真实2rank Gloo、48层tiny Qwen3保216linear+96expert/628可训键、BF16base/FP32Adam，bucketview False/True ×局部释放两组，两更新及完整state重载后第三更新的gradient/adapter/Adamhash全逐位一致。step2 False有628个独立gradstorage，True全部grad._base有效并共享1bucket；slot释放使下一forward旧logits弱引用已死。仅CPU共享storage与数学证据，不当官方GPU峰值节省。主线程最小DDP gradient_as_bucket_view=True＋slotbackward累积detach NLL后del局部引用，源码7c4636262726addbef7735ecc1e9717b66c922ed2a2ef20f264d08ddd22f33bb，nWrp0q两真实tiny架构及完整恢复CPU自测exit0。保全部专家/audio/rank32/alpha64与FP32Adam，不改BF16专家折叠舍入。下轮N7同officialbase/seed42新v4fresh2，旧u1不覆/不重标/不跨源码假resume；再次GPU通过前不称30B错误全部修好。

N7共享梯度新v4已实际启动RueYFY，remote manager3163785/torchrun3163792，fresh非u1恢复，原专家/audio/FP32Adam范围未缩。新124帧H3缓存8条真实审计已独立记录h3_video_repeatlast124_pretrained_cache8_v2.json：全部BF16latent[1,24,37,80,44]、81源帧/122自然/2末pad、packed无audio与tag2；marker0551d8628686e57f3705a99f66de2e62f779627f14ddb34d7c49261b084b25b8。N3LO83n0已真实exit0两更新，training-state-2/final两完整28文件/8Adamstep2/8RNG/模型分片/scheduler/custom审计通过。strict同源码GPUload/resave ykvYup正在独立output进行，只恢复已满2budget不再训练，不称下一更新GPU恢复已经验证。正式目标0批准、未pilot/full，所有当前smoke均fixture。

#### 2026-10-22 用户要求：实验继续时分析旧音乐替换文本失败的可能原因

当前新主线实际进度：v8两个维度视频完整审核并8774上线；N7 bucketview v4已真实两Adam更新，NLL4.1089335→3.3509649，第二更新192专家LoRA张量全非零，正在最终保存/恢复/processor/infer门禁；N3新124帧两更新和完整保存审计已完，实际GPUload_state已返回，零额外更新resave后的严格等价比对尚待。新正式pilot/full与首32真实事实/窗口审核仍未完成，不能用上述接口smoke替代。

**优先判断：旧路线更像监督目标与冻结生成器所需条件不匹配，并伴随泛化不足，不只是训练没跑、骨干没训或shape不一致。** 最新旧6000两组实际28完整epoch/2632updates，400dense＋2Linear联训、完整映射部署、cached P/原生图音VAE/真实音乐/联合50步均已核，18学生视频人评仍近pose。既有旧1800/noTC/静音/预测长度协议的混杂保留为历史，不迁移为已恢复原条件的6000阶段工程归因。

1. **确定的监督欠约束；主因仍须消融。** 实际direct loss仅对有效token均值的标准化L2向量做对称InfoNCE，B再约束每样本均值与全token/channel centered标量RMS（music_text_contrastive.py:859–896）。对token置换不变；可在不改mean/RMS的情况下大幅改变局部方向、顺序、token协方差与语义组合，而DiT读取整序列。单Linear又满足mean(Wf+b)=Wmean(f)+b，CLIP不能充分识别时序输出质量。不是“模型一定没有任何时间信息”，而是loss不能保证那些信息转成H3可读的编排。
2. **输出看似5120维，仍受可达空间与条件分布限制（强假设）。** music_text_adapter.py:260–281共享Linear1024→5120使训练后每token落在同一b+col(W)，秩≤1024；骨干非线性不解除最后线性子空间限制。这不是旧32/256维瓶颈，但也不是自由5120条件。教师y是冻结Qwen-VL上下文化LM输出，含词语/位置/图上下文结构；学生直接交DiT而不再经过该LM，mean/RMS近似不等于进入有效语言hidden序列分布。旧oracle W/LN空间投影仍pose、PCA256部分舞蹈但液化，支持“方向/结构比误差总量重要”，不证明本次1024空间不足；ABC单caseB/C近似也不能证明多模态绑定是主因。
3. **最新正式训练明显train/val分化（有量化证据）。** A epoch3→28 trainCLIP3.47866→0.11814，val500 5.76193→7.20104；B final val7.32763。最终R@1 A.007/B.008，训练probe64均1.0。epoch20降LR后8完整epoch未恢复留出验证，因此更多同配方更新不是优先修复。数值平台满足只表示既定停止条件；不能证明任意预算无效。实际18视频用best Aepoch3/Bepoch2，不能拿final raw RMS直接归因视频；B final尺度校准约.99055对A.22362不代表完整条件语义校准，且用户未给A/B逐模型排名。
4. **音乐→唯一配对动作文字本身一对多（待核数据假设）。** 相同音乐可支持不同舞步、故事与措辞，不同曲也可共享编舞，音乐无法确定具体左手靠哪只肩。旧逐tokenMSE可能平均化；最新CLIP不等于MSE，但按唯一配对caption做对比仍可能偏向样本识别/假负例而非合适编排。已有同曲source组mask不能消除跨曲动作语义近似。需真实语义正例/音乐窗口动作证据才能判，不能把低检索率当该假设已被证明。
5. **文字词位置不是音乐时间；无显式编排/跟拍监督（确定缺项）。** 254音乐token对可变词数不构接口错误，但音乐声道优先时间序列与语言序列含义不同；CLIP均值/RMS不约束动作先后、过渡或拍点。教师caption的伪秒数未核，不能插值词索引制造时序标签；可恢复教师舞蹈也不保证额外跟拍。当前离散caption路线利用Omni自回归语言先验生成一套连贯可能编排，再由原TC生成合法H3条件，针对结构瓶颈而非承诺音乐能唯一恢复GT。H3微调再单独对动作遵循/连续性负责；两阶段还需正式数据与联调人评才能证实改善。

排除过强结论：骨干确实训练，部署并未丢完整映射；5120相同不是语义等价、2T≠L不是接口必错；原P/原音频保留后仍近pose使“单纯没P/静音”不足以解释最新6000负结果；新rich/MS/fusion单case纯视频可用是新协议正证，不是旧18学生根因的单变量证明。最优先关注完整可生成序列的监督/分布与泛化，再是明确音乐时间约束；不无限延长同mean-CLIP配方，也不宣布已锁定唯一根因。旧纯文1800原曲预设曾获“有舞蹈、幅度分case”反馈，但不跟拍且换曲近pose，是反对“hidden替换绝对无效”的重要部分正证。旧6000实际上屏best的RMS比A.187144/B.974624，与last28的.223623/.990551分开；用户只给整体近pose，不伪造B单独排名。

#### 2026-10-22 新124帧H3接口与真实GPU恢复全部验收完成

N3严格接口链路已实际完成：NFFNKn cache8/Z8X1cc全媒体映射通过；LO83n0 fresh8GPU真实2updates、TcCK4m完整training/final各28文件marker/8Adamstep2/8RNG/model/scheduler/custom通过；ykvYup真实8GPU load_state后独立resave actualexit0，0ObpBn对26项（8model＋8Adam＋8RNG＋scheduler＋custom）state/tensor逐值严格相等、signature/progress一致。restore_finished与restore_audit主线程已独立读取核实。源9f5de5/wrapper89439全程冻结，原107/v1资产未覆或重标。

GPU恢复未做额外更新（actual_additional_optimizer_updates=0，next_update_verified=false），不得包装为“恢复后下一更新已验证”。本次仍8条固定fixture/两更新接口，非正式SFT/收敛/画质，也不使首32事实窗自动获批。公开结果h3_video_repeatlast124_pretrained_smoke_and_restore_v2.json。

#### 2026-10-22 官方30B完整接口验收完成，维度A/B已获实际近似反馈

N7 bucketview freshv4 RueYFY真实exit0，Aw7Gjh独立完整验收exit0，verification.json verified=true；实际2updates、16examples、336assistant tokens，NLL4.10893354→3.35096491。第二步628tensor梯度齐，LM576/audioLoRA48/projector4非零，其中全部192expertLoRA非零；rank32/alpha64/FP32Adam及末4audio＋proj1/2范围未减。u0/u1/u2 marker全部文件SHA、628Adamstates step2/8rankRNG/sampler/history完整，生产load_checkpoint及processor训练部署prefix逐tensorexact，真实posttraincaption1条完成（另原入口epochval1条）。freshu0/u1adapter与v3同seed旧SHA一致，不是跨source恢复；原两次OOM分别为checkpointing未生效的forward与moments分配后的grouped_mm反传峰，当前完整接口确已越过，但不代表正式SFT或caption语义通过。公开报告omni30b_official_pretrained_smoke2_verified_v4.json。

用户最新实际反馈：**“A「手臂主导＋优雅梦幻」、B「直立＋原地＋不跳跃」几乎完全一致”**。两个v8同case追加维度未建立可辨的配置间差异，不再等待此项人评；不推逐像素相等、不猜与v7也相同、不自动判各自质量通过或所有维度无效。v7仍是用户此前明确“更好且可用”的基线；32真实事实/窗四项审批仍0，不把视频正评代作训练标签审核。

#### 2026-10-22 H3正式来源门禁与用户要求的完整归档

用户要求将历史实验记忆及相关分析整理详细文档，并详细解释当前实验与资源利用。新独立归档见notes/28_experiment_history_failure_analysis_and_current_status_20261022.md，按阶段区分Wan/H3画质、原生audio/专用32D、三代rawH、ABC及新Omni+纯视频；不是新训练。

H3正式100/300数据来源门禁已实现并主线程C2GQhv独立CPU全部通过，最终源码16c81225b62d77627b739522040128b957fddd4cabfc7f88417a421cf4086b40（不是先交付71284883）。默认data_scope=interface，仅明确max_train_steps2可做接口；正式100/300必须显式--data_scope reviewed_train。模型加载前独立验证fusion export complete/六outputSHA、train-only分区/统一版本、manifest/review/rewriteSHA/ID/四项审批/evidence/window/target和全媒体/meta SHA；缓存producer/scope/provenance/temporalpolicy再核，逐sample绑定trainID/captionSHA及完整成员，禁止翻marker将fixture升级正式。21metadata/9cache/四个漏显式scope拒例、合法100/300到model边界正例与原tiny恢复下一步bitexact全部通过。仅CPU合成审批链，不真实导出/上卡；旧9f5缓存/训练/GPU恢复封存，不重标新source已GPU通过。full3–10epoch完整预算/val平台入口仍待后续必要实施，不在现CPU门禁任务谎称full已支持或运行。

本次八节点资源只读快照2026-10-22 23:17:39：模型实验0/64、holder48（N1/N2/N4/N5/N6运行40，N8停止T态8）、N3/N7真空闲16、未知compute0。holder每卡975MiB，运行者100%仅矩阵乘，不算训练。每节点实际cgroup limit约997.83GiB、headroom348–691GiB；64reserve全部已过1200秒保护，N1原watchdog可补占卡，因此不保证未来空闲。未launch/kill/刷新reserve。详细节点表及当前真实任务/已完成验收/正式监督0审批阻断均写notes28与cluster_resource_utilization_summary_20261022.json；不再把已结束30B/H3/GPU推理称运行中。

### 2026-10-22 最新正式推进决策：只30B、委托AI证据审核

用户要求更详细记录音乐替换文本的逐轮实验/结论/分析，取消后续7B微调对照，尽快正式推进；在澄清目标审核后明确回复“我希望你来进行审核,此外,也可以直接进行推进”。这取代需用户逐条人工POST的当前前置：允许透明assistant_delegated AI证据审核真实配对片段和来源事实，不冒human分数，不把unknown原rich/MS窗口伪verified。

已实际完成新30B-only pilot来源prepare ZyNQ3t，512train/128val，共640记录，`MS/dance_caption_pilot640_30b_only_manifest_v1.json` SHA624408c0bba8926ed21d0756e12c9161ca12dc917569a6e8be5317c0376ab729；所有目标仍未生成/未批准，不是formalSFT。当前实际推进32媒体AI观察、30B冻结teacher重写兼容与透明审核契约；文档notes28的音乐替换文本R0–R7逐轮章节同步补详。距离正式pilot剩真实审核目标→640监督→导出/数据签名→当次8卡准入，技术接口不再重复验、7B不再训练，不等待用户再逐条审批。完成这条关键路径后直接官方30B fresh三epoch/smoke0，未完成前不假称已经正式训练或给未经证据的日历时间。

#### 2026-10-23 实际委托AI审核与官方30B目标重写启动

32配对normalized视频实际CPU全部解码81@16，逐条Read12帧图共384帧，审核旧正文为1粗粒度可留/26需重写/5拒当前target（不是拒媒体）。已保存透明AI观察与89条保守window OBS事实；original rich/MS源窗仍unknown，未冒human评分/全部facts自动True。观察报告在生成manifest冻结后仅修订旧producer26/6与当前validator27/5的说明，source SHA从f59b…改581645…；派生89facts只同步父引用由58a…改c9e2…。当前运行manifest内嵌旧实际观测并保父f59指纹，不重标旧SHA，公开报告保完整lineage。

冻结teacher入口从硬编码7B改为官方30B config/processor/ThinkerConfig（也保历史7B兼容），完整weights loading missing/mismatch/error拒，speech-output unexpected白名单，不创建Talker/Code2Wav；模型全冻结text-only、assistant {预填充与EOS/PAD保持。新审核kind assistant_delegated、明确用户delegation、实际本条video和帧图path/SHA/size、reviewer/window/evidence及四项检查必需；human历史缺kind兼容，不用AI冒人工。最终fusion SHA c74f5e1694f827a180e40e425730e13cdd660775e6756acc7fbbb0c8a29b6f3d，主线程Two3xS18CPUtests实际exit0；H3审批调用传实际record的单行适配后SHA68199706bb58248ff685f929510a563b3a8e9da01629108cb32eab82a06247bc，LMlkRm原正式门禁/恢复CPU全部通过。

新32素材manifest fd2f6a9e78b2ddc232c9c534f128991a10b572ff9ad70525a245ab665b544119 保raw rich/MS与原sources，增加actual normalized OBS观察、冲突与优先限制，不谎原时间mappingverified。N7GPU0经freshUID/argv/CUDA/UUID/host/cgroup准入，真实manager3179081/child3179110，唯一LekoXm实际启动官方30B冻结32重写；只是正式监督准备，**不是30B学生SFT**。已有实际输出，首两条仍有正文保留冲突动作的问题（如承认近膝不可见却正文仍写近膝；承认无下移却正文仍下移），不能以JSON格式通过自动批准，须按OBS保守事实进一步修正后再扩512/128。旧完整7B/30B两更新不重跑。

针对实际正文/冲突相互矛盾，已构建只以89条保守normalized OBS动作作admissible facts的新32素材manifest v2 SHA1209833dae4f39219a4e06bf2fbccc6fe570b8399a2909657ea4f1fa394ec5e9；原rich/MS全文和关联部分支持/排除理由只保侧车，不再把未支持原动作句同等输入让teacher重复。v2仍保原rich/MS源窗unknown、不自动批准。唯一修正队列sXNtIg精确pidfd等待既有v1 manager结束后再freshGPU准入，避免同卡并发/覆partial；学生正式pilot未启动。通过实际target/OBS事实校验后再扩512/128目标与正式30B三epoch。主文档notes28已扩702行，其中音乐替换文本R0–R7＋ABC逐轮244行、数学/因果边界102行，更详版c906dd3已push核远端；AI审核契约与实际官方30B重写启动另a710ea3已核远端。

#### 2026-10-23 首32实际AI批准、640媒体齐备及视觉监督真实读取状态

官方30B text-only重写v1/v2均实际产出32并结束；v2只输入89条保守OBS，但仍有动作反序、无据路线、强度修辞及一条OBS自身站姿误判。独立AI审核A/B实际读取各16目标及16帧图，最终共19条原30B正文可保、13条需要明确编辑。新派生manifest `dance_caption_editorial32_approved_manifest_v3.json` SHA `ea8f85e447ce407e1cb2578041b9154d75680795a1e88648b866a986fc1e1aff`，保全部原generation与source指纹；13条修改明确author=assistant evidence editor，不冒新30B输出。错误站姿事实只在新manifest窄化，不覆盖原OBS。统一caption_version `dance_caption_rich_ms_v1:69b85f4ef18e0e02adaa86e16e9fb49a450926414c3f9b4e30cf9d3832212c19`。

真实32审核 `dance_caption_editorial32_assistant_review_v3.json` SHA `7de27bda7bd2c39e1e03b79cf16f444b7cf31200620a6881f1d6471c5051c446`，review_kind=assistant_delegated；每条绑定本条normalized视频/实际读取帧图及独立审核证据。批准仅当前证据支持的normalized定性动作目标，human_review_completed=false、original_source_window_mapping_verified=false；12采样帧不证明全片动作细节、精确速度/整圈或音乐同步。真实export `dance_caption_editorial32_verified_export_v3` 实际完成，Omni reader24train/8val、H3正式来源链24train通过。该32是目标质量小批，不是正式512/128训练。

640来源媒体首版实际627成功/13短音频失败：11train、2val，11条少1sample、2条少16samples@24k。保持严格[0,5.0625]，不补音/缩窗/放松门禁；13条同split替换，保原失败清单与627已有证据。新candidate v2 `dance_caption_pilot640_30b_only_manifest_v2.json` SHA `2f1c4d2ddc5b299f0a86161a59635c719b3f5b84646c49d9e78a484f4481573e`，media index `pilot_media_inputs640_v2.json` SHA `43276c9b26cc140bcf3d912b587115fae3bbadc81e720121ed5f9085a7df11d0`，640全部prepared/0failed、512train/128val。标准visual index SHA `06d4f36a6116cde5cfc5c4ccfbbd42144409f244718ca3c254a752531ed00d68`，冻结视觉入口CPU load_inputs640实际通过；仍caption_targets_generated=false、visual_review_complete=false，不把文件齐备算真实逐条审核。

实际图像依据的官方30B冻结入口 `omni30b_visual_caption_review.py` SHA `17cc49ef31f89445ab9e7397263ac1e5badef073fca17790585ac6d271164357`：图像processor CPU实测925image tokens、pixel_values[3700,1536]、grid[1,74,50]；10CPU契约通过，候选四项批准固定false。真实视觉32唯一队列 `omni30b_visual_caption32_queue_v1`，N7 GPU0，manager3181390/child3181399，已实际启动，未声称已完成。2026-10-23 02:00北京时间只读/proc证据：UID全0、argv匹配、State=D、wchan=generic_file_buffered_read、fd3指向共享official第7/15权重分片、read_bytes约31.23GB、GPU0显存0MiB。此时是模型内容SHA冷读，不是已知OOM/崩溃或实际GPU视觉前向；保唯一任务，不重启、不覆盖partial、不运行中改visual/fusion源码。

当前剩余工程缺口已只读核实：视觉六字段OBS候选不是fusion四字段获审target，须经实际AI核图/正文后制作透明派生兼容manifest/target，保原raw与unknown映射；现格式/词覆盖不验证动作顺序或正文/conflicts矛盾，不能自动批准。Omni train原reader未接真实export审批链、数量及fixture门禁，需在模型加载前显式scope/preflight，复用H3已实现严格来源链并绑定恢复fingerprint。该必要门禁修正正在独立单文件实现/CPU测试，不重跑已通过官方30B两更新。正式学生SFT仍未启动，整体Cron继续。
