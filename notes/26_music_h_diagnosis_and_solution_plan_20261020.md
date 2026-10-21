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

该检查只证明音轨内容与来源，不检查音视频动作跟拍或AAC packet时间戳／容器编辑列表。送编码器的波形长度162000samples／32kHz=5.0625秒，视频124帧／24fps=5.16667秒，内容长度相差约0.10417秒；不把AAC padding视为真实音乐，不据此判定动作时间对齐。学生连续人评仍未取得，任务179保持未完成；重复stop hook不是人评，也不授权为达到停止条件扩训练或伪造质量结论。

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

文字token索引不是音乐时刻：只回归文本hidden不能自动提供40Hz拍点／动作强调时序监督。adapter需要维度／序列／分布转换，但1024→512→5120结构是我们的方案选择，仍有有限宽度输出包络，5120维shape相同不等于已对齐语义或跟拍。学生舞蹈、幅度和音乐控制仍必须用本轮原／换／零完整视频独立判断。
