# H3 音乐控制后训练方案与验证计划

更新日期：2026-10-14。代码与结果基准：[15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a](https://github.com/Anachronism-N/H3-posttrain/commit/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a)。本文补充 [问题审查](23_h3_posttrain_issue_review_20261014.md)，将最新实验、代码行为与外部技术讨论转化为下一阶段的验证计划。

项目正在探索让 MiniMax H3 根据人物参考和音乐生成舞蹈视频，并逐步扩展音乐条件与长视频能力。目前最值得继续验证的是：**先确认音乐引导能在保留画质的同时改善音乐同步，再把同一教师速度场蒸馏到任务 LoRA。** 现有锚定训练已经缓解部分普通 FM 后训练的质量下降，但尚未证明学到了更好的音乐控制。域内 DeCFG 差分训练可作为一次低预算备选。

## 最新结果支持什么

### 普通 FM 出现退化，锚定缓解退化

以下数字来自 [阶段二手部摘要](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/results/20261014/QUAL_anchor__summary_stage2.txt)。该表声明使用同 3 个案例；2091 是训练样本数，300 是已评估的 checkpoint 步数，括号中的 900 run 表示长训练计划，不能当作 900 步验证结果。

| 配置 | joint hand_conf | aswap hand_conf |
|---|---:|---:|
| 原始 H3 480p | 0.669 | 0.707 |
| 普通 FM GT512 300 步 | 0.564 | 0.567 |
| 高噪声锚定 σ≥0.5，512 样本，300 步 | 0.653 | 0.669 |
| 高噪声锚定 σ≥0.5，2091 样本，300 步 | 0.658 | 0.651 |
| FM 加 5 倍锚定，2091 样本，300 步 | 0.656 | 0.647 |
| 高噪声锚定加 DPO β1000，100 步 | 0.644 | 0.643 |
| 高噪声锚定加 DPO β5000，100 步 | 0.630 | 0.639 |

锚定组的手部检测置信度高于普通 FM 组，但不足以断言已完全恢复原模型画质。`hand_conf` 反映手部关键点可检测性，也受动作幅度、遮挡、人物大小和背景影响；它不直接度量手指结构、纹理、身份或动作自然度。当前 [质量脚本](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/scripts_omnidance/exp/eval/quality_metrics.py#L64) 还只采样前 5 秒的 16 帧，需要配合视频盲评。

**基线需要统一重算。** [早期摘要](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/results/20261014/QUAL_anchor__summary.txt) 中原始 H3 为 n=6、0.672/0.677；[实验笔记后段](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/notes/22_h3_bidir_blur_attribution_20261013.md#L193) 写为 0.669/0.673，阶段二表又为 0.669/0.707。差异原因尚未确认。后续比较必须保存并匹配 case、seed、条件、抽帧列表和评估版本，不能混用这些基线。

### 保留音乐敏感度与改善同步是两个目标

以下是 [音乐敏感度原表](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/results/20261014/QUAL_anchor__music_sensitivity.txt) 的 3 案例均值。指标为相同人物、prompt 和 seed 下 joint 与换音乐 aswap 视频的平均像素差，使用第 24 至 123 帧。

| 配置 | joint 与 aswap 的平均像素差 |
|---|---:|
| 原始 H3 | 0.0421 |
| 普通 FM GT512 300 步 | 0.0186 |
| 高噪声锚定 2091 样本，300 步 | 0.0407 |
| FM 加 5 倍锚定，300 步 | 0.0377 |
| 旧 W1b DPO | 0.0465 |
| 锚定加 DPO β1000，100 步 | 0.0340 |
| 锚定加 DPO β5000，100 步 | 0.0299 |
| MERT M1abR708 | 0.0301 |
| MoE M2bR886 | 0.0221 |

这些结果支持“该小样本中普通 FM 减弱了音乐改变画面的能力，锚定大体保留了这种能力”。像素差增加也可能来自更大的动作、背景变化或不稳定性，不能直接推出节拍更准确。

[音乐引导记录](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/notes/22_h3_bidir_blur_attribution_20261013.md#L215) 提供了一个值得复验的方向：原始 H3 的敏感度约 0.042，加引导 w=1 后约 0.071，w=2 后约 0.063；锚定模型从约 0.041 到加 w=1 后约 0.056。但 w=1 的原始 H3 结果只有 n=2，其余条目未附逐案例原始表。这些是探索记录，不能据此给 w 排名或声称同步显著提升。笔记中的动作增大、手脸清晰观察也需要独立案例复核。该版本已实现高噪声音乐引导锚定目标，但记录的是后续蒸馏任务计划，尚不能将上述推理结果当作学生蒸馏成功的证据。

### 旧 E4 提供线索，不能替代新方案验收

[E4 摘要](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/results/20261014/E4_20261014__summary.md) 中，W1b 的内部音频配对差值为 +0.089±0.032，底座为 +0.033±0.028。前者为正不等于统计上优于底座；应在共同案例上计算两模型配对差值之差。W1b 的 joint motion 从底座 0.181 降至 0.116，约下降 36%，说明音乐指标必须与运动表现一起看。M1a 的 aswap 只有 43 条，也需报告缺失与失败。

此外，[评估调度脚本](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/scripts_omnidance/exp/h3ar/eval_sched.sh#L45) 对 biorig 显式使用 orig 文本，而普通 bi 分支保留默认 motion 文本。旧 HA2、M1、W1b 的 bi 标签不能直接认定与 biorig 的富文本条件相同，应核对实际生成命令。

## 代码中需要先处理的问题

### 训练开关不能任意组合

[loss.py](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/diffsynth_patch/files/diffsynth/diffusion/loss.py#L99) 在 DPO 分支提前返回；[anchor 分支](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/diffsynth_patch/files/diffsynth/diffusion/loss.py#L203) 也在 inverse/direct CFG、TF 区间裁剪和后续辅助损失之前返回。因此，给当前 anchor 或 DPO 任务加 `training_cfg_scale>1`，并不能得到预期的联合目标。anchor 情况下甚至可能已计算无条件前向，却未将它用于最终 loss。

当前 cfg=1、无额外音乐模块的双向锚定实验不受这一组合问题影响。最小修复应先对不支持的组合显式报错，并记录最终生效的 loss 模式；需要联合训练时，再定义损失的组合顺序、视频区域与音频分量。`OD_H3_ANCHOR_MCFG>0` 还依赖 `OD_H3_ANCHOR>0`，否则当前实现不会进入教师分支。不能只增加环境开关。

### 推理的文字 CFG 与音乐引导存在耦合

[sample_h3_ar.py](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/scripts_omnidance/exp/h3ar/sample_h3_ar.py#L124) 先计算文字 CFG，然后将它与未经文字 CFG 的负音乐预测相减。令 `vp` 为正文字正音乐预测，`vu` 为负文字正音乐预测，`vm` 为正文字负音乐预测，文字权重为 g，音乐权重为 w：

```text
v_text = vu + g * (vp - vu)
当前 v = v_text + w * (v_text - vm)
```

如果正负音乐完全相同，即 `vm=vp`，音乐增量仍为 `w*(g-1)*(vp-vu)`。标量例子 `vp=vm=5, vu=2, g=3, w=1` 中，文字 CFG 为 11，当前结果却变为 17。

首轮教师验证固定 `cfg=1`，现有音乐公式在这一条件下成立。若支持联合 CFG，可选择并明确记录以下一种定义：

- 三次前向：`v_text + w*(vp-vm)`，保留原始正音乐预测计算音乐差。
- 两个音乐分支都使用相同文字 CFG，再计算分支差；通常需要增加负文字负音乐的第四次前向。

验收不变量是：正负音乐相同时，增加 w 不改变输出；w=0 恢复文字 CFG；g=1 恢复当前纯音乐引导公式。

### 教师与学生的噪声区间和条件必须一致

当前推理 `music_cfg_min_sig` 默认 0，音乐引导覆盖全部噪声区间；当前 `OD_H3_ANCHOR_MCFG` 教师目标只在 anchor 的高噪声分支参与损失，低噪声仍使用普通 FM 目标。**全区间引导教师与高区间教师加低区间 FM 的学生，是两个不同方案。** 训练前应显式记录 `w(σ)` 与低噪声目标，并以同一规则验收教师和学生。[推理参数](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/scripts_omnidance/exp/h3ar/sample_h3_ar.py#L36)、[训练目标](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/diffsynth_patch/files/diffsynth/diffusion/loss.py#L167)

当前负音乐前向只滚动目标音频 latent，保留参考音频。准确含义是“保留全局音乐参考，扰动目标音乐的时间对齐”，并非替换全部音乐条件。这可以是有意设计；教师与学生应使用相同定义。未来启用 MERT、beat 或 MoE 时，所有随时间对齐的特征必须按同一物理时间偏移，不能各自简单滚动一半数组；不同采样率和截取长度可能造成不同秒数的偏移。

当前锚定运行禁用额外音乐模块、底座冻结，关闭任务 LoRA 后得到原始 W，不能把它写成正在漂移的教师。但未来若加载可训练的音乐注入器，关闭 PEFT LoRA 不会自动关闭注入器；若把 DeCFG 增量 D 融入底座，关闭 LoRA 后得到的是 W+D。教师应明确标识权重、模块与条件，不能仅用“禁用 adapter”代替教师定义。

### 锚定 DPO 的偏好项仍使用 FM 误差

[h3_dpo.py](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/scripts_omnidance/exp/h3ar/h3_dpo.py#L50) 的正负偏好误差仍对比普通 FM 目标，anchor 只替换正分支的 SFT 项。它可能与保留教师质量的目标产生竞争，但现有结果不能证明梯度已经抵消。需要按 σ 分桶记录偏好项和锚定项的梯度范数与夹角。

当前 DPO loss 除以 β，偏好误差的梯度系数涉及 `sigmoid(β*diff)`；将 β1000 改为 β5000 改变饱和程度，不能理解成梯度直接放大五倍。若后续确有必要调节偏好强度，应引入独立的 `lambda_dpo`，并与 β 分开解释。

## 外部讨论与 TrainingAdapter 的有效信息

外部讨论中的“先退化再减去退化”“直接 merge”“adapter 加 inverse”可转化为两类不同操作：权重空间的差分训练，以及输出空间的 inverse-CFG 损失。**inverse 的具体含义尚未确认**，不能据此认定已有实验完全复现了对方配方，也不能把其效果描述作为稳定性保证。

[MiniMax H3 TrainingAdapter 模型卡](https://www.modelscope.cn/models/DiffSynth-Studio/MiniMax-H3-TrainingAdapter) 的核心流程是：先用固定 adapter 将 CFG 蒸馏模型拉向适合任务训练的状态，在此基础上训练任务 LoRA，推理时移除固定 adapter。令 W 为原始 H3 权重，D 为固定 adapter 的有效权重增量，L 为任务 LoRA 增量：

```text
训练有效权重：W + D + L
部署有效权重：W + L = (W + D + L) - D
```

如果部署时已经加载 W+L，就不要再减一次 D。D 的方向由其训练方式决定，不能因为名称包含 DeCFG 就自动取负号。离线 merge 与在线固定 adapter 在有效 scale、alpha/r、模块和 QKV 布局一致时具有相同的加性权重含义；数值精度、量化或 dropout 仍可能影响实际输出。仓库 [fuse_adapter.py](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/scripts_omnidance/exp/h3ar/fuse_adapter.py) 可作为检查入口。

模型卡公开的 adapter 构建流程本身使用自生成样本和普通 FM；任务 LoRA 示例使用 preset adapter，并未要求额外开启 inverse-CFG。因此，已有 SG512 普通 FM LoRA 单独推理变差，并不能否定两阶段差分方案，它可能是一个候选中间增量 D。但参数相减也不保证非线性模型的 CFG 能力必然恢复。[DiffSynth 差分 LoRA 文档](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/zh/Training/Differential_LoRA.md) 明确指出该方案不再端到端优化，效果存在不确定性。

仓库 [q2 配方](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/scripts_omnidance/exp/h3ar/q2_20261010.sh#L21) 已区分 DS 加 FM、DS 加 direct CFG，以及原始底座的 Ref2VA inverse；没有明确记录 DS 加 inverse 的独立结果。补做该组合之前，应理解 direct 与 inverse 的尺度关系。对同一次前向得到并停止梯度的无条件预测 u、模型预测 p、FM 目标 y 和 CFG 权重 g：

```text
direct 目标：u + g*(y-u)
inverse 预测：(p+(g-1)*u)/g
L_direct = g^2 * L_inverse
```

二者在这些条件下有相同最优解，原始 loss/梯度相差 g²；g=3.5 时为 12.25。应先统一归一化，不能把不同更新幅度误判为不同机制，Adam 下也不能简单换算成学习率乘 12.25。是否补做 DS 加 inverse 应放在主要教师验证之后，并关闭 anchor/DPO，确保实际执行该损失。

## 主线方案 音乐引导教师与速度蒸馏

### 先验证教师

固定原始 H3 为教师 T，文字 cfg=1，先使用现有原生音频条件；已有锚定 checkpoint 加引导可作为候选对照。若改选锚定模型为教师，需要单独冻结它的任务 adapter 或使用独立教师 checkpoint；当前关闭全部 LoRA 的逻辑会返回原始 H3，不能直接实现该教师。统一 rich prompt、图像、分辨率、seed、音频时段、scheduler 和 steps，不增加 MERT/MoE 或 DPO。

```text
v_star(x_sigma, sigma, m)
  = T(x_sigma, sigma, m)
    + w(sigma) * [T(x_sigma, sigma, m) - T(x_sigma, sigma, m_negative)]
```

优先复验已有 w=1。若采用 σ≥0.5 的门控，就先评估同样门控的教师；若全区间教师表现更好，学生也应拟合同一全区间目标。沿用现有推理记录减少额外计算，但增加未经反复挑选的验证案例及新 seed。

教师验收应同时看：正确音乐相对时间错位/换音乐的配对同步改善；手脸、身份与服饰保持；动作幅度、自然度及长时间稳定性。音乐相位平移应考察动作事件是否相应移动，节奏变化应考察节拍响应。对照音频扰动时保留人物和文本，避免将文本变化误当音乐控制。像素差只作为诊断。

如果教师只有像素差增大，同步没有改善，或动作和画质明显变差，应先修改负音乐定义或 w(σ)，不进入学生蒸馏。

### 学生拟合同一教师速度场

教师通过后，可先对 8 至 16 条代表性片段保存每条 4 至 8 个真实去噪轨迹状态，复用教师采样时的前向，控制缓存规模。按 σ 覆盖高、中、低噪声区间，避免按 step 等间隔抽取后集中在高噪声。缓存至少包含 `x_sigma`、实际 σ/时间步、`v_star`、音频与参考条件、mask、prompt、seed、scheduler，以及教师权重和引导配置。纯视频蒸馏可仅保存所需视频分量，音频条件必须保持可复现。

```text
L_distill = MSE(student(x_sigma, sigma, conditions), stop_gradient(v_star))
```

首轮在全部噪声区间使用教师目标。即使低噪声 w(σ)=0，低噪声目标也应是原始教师速度，而非自动切回普通 FM，以便单独验证教师速度蒸馏能否兼顾质量与音乐响应。普通 FM 混合项应作为后续明确的消融，不要默认混入。

只用 SG 视频终点重新构造线性加噪状态，不等于保存教师轨迹。首轮轨迹缓存用于验证方案，随后还需要用更多案例和学生实际访问的状态检验误差累积。学生在训练缓存上拟合成功，不等于推理时已经复现教师。

先训练 fresh LoRA 约 100 步做小规模可行性验证，比较底座、教师和无推理音乐引导的学生。通过后再扩展到 300 步及独立评估集；不要同时改变 rank、分辨率、音乐模块和偏好损失。

在线教师实现需要完全冻结教师模块、停止目标梯度、恢复原有 adapter 状态，并保证 ZeRO 各 rank 的教师前向次数一致。不能在部分 rank 按随机 σ 跳过教师前向。离线缓存也需校验视频/音频时间轴与特征版本。

这条路线放大并压缩教师已有的音乐条件能力；它是否足以学习新的舞蹈编排，仍取决于教师本身和后续训练数据。

## 备选方案 域内 DeCFG 差分训练

一种待验证的解释是：公开 adapter 的构建条件与当前 clean-audio、retake 式视频训练不同。公开训练示例中目标音频随时间步加噪，而当前配方保持目标音乐 clean、音频 mask 为 0。联合 DiT 面对的音频状态与可见信息不同，通用 D 可能无法准确补偿当前任务的 FM 漂移。**这只是条件分布不匹配假说，公开模型卡没有证明通用 adapter 在该条件下失效。**

可以复用已有 SG512 100 或 300 步普通 FM LoRA 作为候选域内 D，冻结它，在 W+D 上训练新的任务 L，最后在 W 上加载 L。无需先重新生成大规模 SG 数据。SG512 D 也可能包含舞蹈过拟合、文本/音乐偏置和一般优化漂移，不能直接称其为“纯去 CFG 方向”。

最小验证顺序：

1. 验证 D 的 merge scale、alpha/r 和权重布局；确认底座和 D 冻结、fresh L 初始不改变 W+D，checkpoint 只保存任务 L。
2. 在相同 clean-audio 输入、噪声和时间步下，比较 W、W+官方 D、W+域内 D 的速度变化，按 σ 分桶，检查差异是否与当前任务有关。
3. 只做一次约 100 至 200 步任务训练，分别评估 W+D+L 和部署形态 W+L；与同条件官方 D 配方和已有结果对照。
4. 若移除 D 后仍糊，或音乐控制没有改善，停止扩展该候选。D 单独变糊不能作为否定理由，因为它可能正是两阶段方法的中间状态。

## 实施顺序与验收记录

| 优先级 | 工作 | 通过条件 |
|---|---|---|
| P0 | loss 组合检查，音乐引导公式不变量，统一教师身份与 w(σ) | 配置不再静默失效；正负音乐相同则音乐增量为零 |
| P0 | 统一生成条件与共同案例，重算质量基线 | prompt、seed、帧率、音频时间段和失败数可复核 |
| P1 | 扩充音乐引导教师的配对评估 | 同步改善且手脸、身份与动作自然度没有系统性下降 |
| P1 | 小规模全区间教师速度蒸馏 | 无推理音乐引导的学生保留教师的同步收益与质量 |
| P2 | 一次域内 D 加 fresh L 的差分试验 | 部署 W+L 优于同条件对照 |
| P3 | 教师/学生通过后再加入 MERT、MoE、DPO、AR 或高分辨率 | 每次新增能力都保留已通过的质量和同步验收 |

评估以案例为配对单位，避免把同一视频的多帧当作独立样本。保存逐案例同步、手部检测、身份、运动、盲评及失败状态，报告共同成功集合和全量失败率。与底座比较时计算配对差值之差和置信区间，并预先固定允许的质量下降范围及停止条件。

当前 E4 的 [BAS 路径](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/scripts_omnidance/exp/eval/eval_metrics.py#L156) 已传入真实 fps，不能将 G2 的时间轴问题套用到全部 E4。另行使用 [G2 判别评分](https://github.com/Anachronism-N/H3-posttrain/blob/15fbdbf877ed04dda2d35ba4c465ed5aff4bd85a/scripts_omnidance/exp/g2/score_disc.py) 时，应修复固定 81 帧与 16 Hz 音乐特征之间的物理时间对齐，并校准评分对节拍相位扰动的响应。

实验记录还应包含代码 commit、底座与 adapter 文件内容 hash、完整生效配置、缓存版本和生成命令。`train_ha2.sh` 已新增环境记录，但目录列表 hash 不等于权重内容 hash；调度器已改善非零退出时释放 claim 的行为，仍需处理被杀任务的陈旧 claim 与部分样本失败。

## 归因与历史问题的更新

“CFG 蒸馏模型在普通 FM 后训练中容易退化”与本项目多个诊断结果一致，但“FM 的条件均值必然导致最终视频模糊”不是已证实结论。`probe_fm.py` 的单样本残差不等于真实条件期望估计；用 SG 终点和新噪声构造的线性状态也不等于原始教师轨迹。现有自重现探针只有少量案例，不能排除所有缓存、mask、条件或训练分布问题。

同样，某噪声区间占大多数 scheduler 步数，不代表它占同等比例的积分贡献；应检查实际 Δσ 与速度误差，而非只数步数。低噪声单独启用 LoRA 已有小样本质量改善，可作为定位工具，但启用索引依赖 scheduler、shift 和总步数，且需要确认音乐收益是否同时被移除。

相较此前问题清单，当前 inverse-CFG 代数已改为 FP32；环境记录和评估 claim 处理也已有改进。这些不应继续被写成完全未修复的问题。当前最关键的缺口是教师目标是否有效、训练与推理目标是否一致，以及配对评估是否足以同时证明质量与音乐控制。
