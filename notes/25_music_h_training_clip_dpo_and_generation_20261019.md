# H3 音乐条件训练 CLIP 与 DPO 区别及生成流程

更新日期：2026-10-20，北京时间。当前代码与结果基准：[5e93a581e6540e6163e994d5f669a964a6822aef](https://github.com/Anachronism-N/H3-posttrain/commit/5e93a581e6540e6163e994d5f669a964a6822aef)。最新人工反馈与解决路线基准：[1448e56](https://github.com/Anachronism-N/H3-posttrain/commit/1448e5662d7541cc37d5c0f84bdc0457f2c9d951)。前段早期设计基于 [5fa2be6](https://github.com/Anachronism-N/H3-posttrain/commit/5fa2be623511eb232db60793de3f142c948258c7)，以下已标注历史范围。

目标是训练 MusicFM 的实际 dense 骨干和完整 adapter，让音乐产生能够替代动作文字的 H3 原生条件，最终仅凭音乐和参考图生成舞蹈。H3 文本编码器、DiT 和 VAE 保持冻结。

**当前状态：音乐实际 dense 骨干与完整 adapter 的在线联合训练、全量 raw H 和自身长度的共享学生条件已经实现并运行。** A/B/C 各 100 与 300 步六组训练及 54 条视频完成；随后冻结 H3 预测一致性小实验完成 3 个训练样本、6 次更新和 18 条视频。工程链路通过，但尚未建立可用音乐编舞；预测监督只有单留出点微小信号，不能当作泛化成功。当前仍不是 DPO。10 月 20 日新增定性人工反馈为“还是类似摆pose而不是真正的舞蹈”；这是最新一轮的总体负评，不是全部 72 条视频的逐例评分。

最新问题分析、独立重算与分阶段解决方案见 [H3 音乐条件问题分析与解决方案](26_music_h_diagnosis_and_solution_plan_20261020.md)。下文早期冻结骨干流程和“执行中”条目保留为历史，不能覆盖已经完成的最终状态；历史“需另行授权”只描述当时范围，不是撤销后来已记录的小实验授权。

本文对应 [实验笔记第 28.21 至 28.23 节](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/notes/22_h3_bidir_blur_attribution_20261013.md#L1143)。[10 月 14 日方案](24_h3_music_guidance_distillation_plan_20261014.md) 保留为历史阶段记录，其音乐引导方向不能代替当前 H 条件主线。

## 任务目标与条件接口

音乐不需要先转成一句动作文字。学生直接预测 H3 文本编码器输出侧的动作 hidden states，简称 raw H；每个 token 为 5120 维。它们进入 DiT 原有的 `condition_proj/token_refiner`，不重新经过文本编码器的 50 层。

```text
教师监督
参考图结构 prefix ＋ 舞种和动作描述
    → 冻结 H3 文本编码器
    → 教师 raw H 动作序列和 token 长度

学生目标链路
音乐波形 → 重采样和音乐预处理 → 可训练 MusicFM dense 骨干
                                             ↓
无动作文字的参考图结构 context ───────→ 可训练完整 adapter
                                             ↓
                                全量 raw H ＋ 预测长度
                                             ↓
                                  学生自己的有效 token
                                             ↓
                              原图 prefix ＋ tags 和 packed
                                             ↓
                                    冻结 H3 DiT 和 VAE
                                             ↓
                                            视频
```

参考图 context 是原生结构 prefix 的上下文向量；参考图不进入 MusicFM 音乐骨干。`prefix` 保留参考条件，`tags` 和 `packed` 描述原接口中的 token 标记、位置及序列打包关系。

预测长度是动作条件 token 的数量，不是音乐时长、视频帧数或动作个数。参考图也不等于被硬性锁定的首帧。

动作描述和真实 token 长度只作训练标签。生成时不提供动作描述、教师 H 或真实长度。原生 reference 和 retake 音频条件均保持相同的固定静音，真实音乐通过学生 H 分支影响生成，并在生成后作为配乐混入视频。

## 早期冻结骨干训练流程

一条样本包含音乐、参考图和对应的动作描述。已有正式划分为 1800 条训练、200 条验证。

1. 冻结 H3 文本编码器，以参考图结构和 `Dance style: {genre}. {motion_original}` 产生教师 raw H 动作 tail、长度和结构 prefix 上下文。中性化选项只调整性别代词，保留动作内容。
2. 冻结公开预训练 MusicFM，预计算音乐 dense 特征。当前正式特征缓存是每条 `[2,127,1024]` 的 FP16 张量，表示双声道、时间特征和特征维度。
3. adapter 接收音乐特征和同参考图 context，输出 raw H 序列和对数长度。
4. 对正确歌曲输出计算 H 回归与长度损失；对同图的错配歌曲输出增加对比损失。
5. 反向传播只更新 adapter。骨干出现梯度会触发当前 trainer 的检查错误。

依据：[教师缓存](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/scripts_omnidance/exp/ms/music_text_cache.py#L30)、[冻结骨干和特征缓存要求](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/scripts_omnidance/exp/ms/music_text_contrastive.py#L189-L206)、[优化器与反向传播](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/scripts_omnidance/exp/ms/music_text_contrastive.py#L225-L270)。

批量教师缓存保存 caption 编码中的结构 prefix 最后一行，部署则从空语义 prompt 的结构 prefix 取对应 context。复用缓存需要核验这两种 prefix/context 的等价性，不能仅凭形状相同认定一致，也不能未经验证就推断存在动作文本泄漏。

## 早期损失与 CLIP 的含义

### 两个训练入口的损失

`music_text_align.py` 的回归目标为：

$$
L_{align}=\frac{L_{rawMSE}}{E_{train}}+\lambda_{cos}L_{cos}+\lambda_{len}L_{loglen}.
$$

`music_text_contrastive.py` 的 H 对比目标为：

$$
L_{HCLIP}=\frac{L_{rawMSE}}{E_{train}}+\lambda_{len}L_{loglen}+\lambda_{clip}\min(1,t/100)L_{NCE}.
$$

这里 `E_train` 是训练集教师 raw H 平方值的固定均值，用于平衡损失数值；不是每个 batch 的方差。`L_loglen` 是预测对数长度与真实长度对数的 MSE。

对比 trainer 虽然调用会计算 cosine 的辅助函数，实际只取其中的 `raw_mse`，没有再把 `cosine_loss` 加入总损失。不能把两个入口的公式合并描述。[回归目标](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/scripts_omnidance/exp/ms/music_text_align.py#L258-L262)、[对比目标](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/scripts_omnidance/exp/ms/music_text_contrastive.py#L240-L259)。

### H 序列对比如何计算

固定参考图 I 和教师目标 H_A，将音乐 A、B、C 分别输入同一个学生，得到三个 H 预测。A 是与当前描述配对的音乐；B、C 是错配音乐。希望 A 的预测比 B、C 的预测更接近 H_A。

当前评分从教师有效位置最多均匀抽取 64 个 token。对学生和教师 token 使用训练集逐维 mean/std 标准化，再作 L2 归一化，计算两两余弦相似度，用温度 0.1 的双向 soft-Chamfer 聚合成标量分数。三个候选分数再以温度 0.07 计算交叉熵：

$$
L_{NCE}=-\log\frac{\exp(s_A/0.07)}{\exp(s_A/0.07)+\exp(s_B/0.07)+\exp(s_C/0.07)}.
$$

开启 `--real_music_negatives_only` 时，B、C 是两个不同 ID 的真实音乐样本，且各自的 caption hash 与当前正例不同；代码未验证它们是否来自不同的源歌曲。否则使用一条错配音乐样本和静音。三条分支共享当前参考图 context。[评分与交叉熵](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/scripts_omnidance/exp/ms/music_text_contrastive.py#L38-L65)。

这是项目自定义的 H 序列 InfoNCE。它借用 CLIP 的配对对比思路，没有调用外部 CLIP 编码器，也不是原始 CLIP 的全 batch 双向图文分类目标。mean/std、L2 和 token 抽样只用于损失计算，部署仍使用原始 raw H。这一分支本身没有额外的可训练对比 head。[CLIP 原论文](https://proceedings.mlr.press/v139/radford21a.html)

### 为什么保留回归和长度监督

对比分数主要约束正确配对相对错配的排序。当前评分经过归一化，仅提高分数不足以保证 raw H 的原生数值尺度、逐位置结构和有效长度适合 DiT。H 回归约束原生表示，长度标签约束实际输出范围，对比项增加音乐依赖。

MSE 也可能让模型趋向多个动作描述的平均表示，需要通过对照实验判断权重。其他歌曲只是未与当前描述配对，仍可能适合相似动作，因此负例存在语义歧义。soft-Chamfer 对所抽样 token 集合的顺序不敏感，不提供逐拍动作监督；逐位置 MSE 仍保留位置约束。上述问题是待验证因素，不能直接认定其中某项是唯一失败原因。

## 与 DPO 的区别

| 项目 | 当前 H 对比训练 | 标准 DPO |
|---|---|---|
| 训练数据 | 音乐与动作描述的配对，以及错配音乐 | 同一输入下偏好的输出和不偏好的输出 |
| 优化对象 | 学生 H 与教师 H 的回归和匹配分数 | 当前策略相对冻结参考策略的输出概率比 |
| 当前是否比较视频好坏 | 训练中没有生成或比较视频 | 用于视频时需构造相应偏好数据与适合生成模型的目标 |
| 冻结模型的作用 | 文本编码器提供 H 标签 | 参考策略提供输出的参考概率 |

当前是在问“音乐 A 预测的 H 是否比音乐 B 预测的 H 更接近描述 A”。如果做视频偏好学习，问题会变成“同一首音乐 A、同一张参考图下，更喜欢视频 1 还是视频 2”。

标准 DPO 的关键是偏好对和策略相对参考策略的 log-prob 比值；当前没有这些目标。教师 H 不等于 DPO 参考策略，错配音乐也不表示该歌曲或其生成结果本身不好。是否冻结 DiT、是否包含正负样本，均不能单独决定一种方法是否属于 DPO。[DPO 原论文第 4 节](https://arxiv.org/html/2305.18290v3#S4)

仓库存在其他历史 DPO 实验分支，本节判断针对当前音乐到 raw H 的训练入口。

## 已有结果及其含义

同初始化、1800/200 划分、batch 8、1875 次更新的对比如下：

| 配方 | 正确音乐优于一个真实错配音乐的比例 | 验证 token MSE | 长度 MAE |
|---|---:|---:|---:|
| MSE 续训 | 52.0% | 5.04644 | 138.975 |
| 对比监督含静音负例 | 51.0% | 5.49469 | 149.000 |
| 对比监督用两条真实错配音乐 | 62.0% | 5.35961 | 146.060 |
| 两条真实负例且长度权重为 2 | 58.5% | 5.39910 | 151.330 |

含静音配置能在 98% 的验证样本中让正确音乐胜过静音，但真实歌曲间区分约为随机水平。62% 来自固定 200 条验证样本和一个循环错配方式，并经历多配方选择；它是表示诊断，不是舞蹈准确率，也不是独立泛化结论。[四组记录](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/notes/22_h3_bidir_blur_attribution_20261013.md#L1143-L1179)

双真实负例候选已经实际生成 9 条诊断视频，涵盖 3 人物的原音乐、换音乐和零输入。用户反馈为“还是完全不像跳舞，动作和之前差不多，而且更换歌曲的差别也不大”。视频技术完整不等于音乐编舞成功。现有结果未达到可用的动作文本替代目标，不能再把问题全部归为漏加载完整 adapter，也不能声称最新候选尚未生成视频。[生成与反馈](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/notes/22_h3_bidir_blur_attribution_20261013.md#L1181-L1201)

## 早期联合训练设计

以下保留 5fa2be6 时的联合训练设计。在线入口现已实现并完成六组训练，见后文 2026-10-19 执行与最终验收；本段的原待实现条目不能作为当前缺失判断。

### 数据与学生前向

复用已核验的教师 raw H、长度、结构 context 和 1800/200 划分。音乐改为原始波形在线前向：读取与部署一致的波形，重采样到 MusicFM 的 24 kHz，执行实际 dense 骨干和完整 adapter。

教师标签可以缓存；波形可以缓存。训练骨干时不能继续用旧 dense 特征缓存、预计算静音特征或 detach 子进程作为学生前向。现有类已经具有 `freeze_backbone=False` 的可微分支，但两个实际 trainer 仍使用固定特征；仅改变这个布尔值不足以完成联合训练。[现有可微分支](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/scripts_omnidance/exp/ms/music_text_adapter.py#L206-L239)

三条音乐分支均通过当前学生在线编码，保持同一参考 context。保留 raw H、对数长度和双真实负例 InfoNCE。对比评分需要支持学生预测有效区间和教师有效区间长度不同。

### 训练范围与保存

优化器分组包含实际参与 dense 前向的 MusicFM 参数，以及完整 adapter、最终 5120 维投影和长度头。可将骨干 `1e-5`、adapter `1e-4` 作为初始试验值，实际有效 batch、精度和学习率需在在线显存与梯度自测后固定；不能沿用旧缓存训练的耗时估计。

建议先固定骨干的 BN running statistics、关闭 dropout 等随机行为，同时允许实际 dense 参数及 BN affine 参数更新。`eval()` 不等于关闭梯度，需在每次 wrapper 切换 train 后恢复该策略。未被 dense 前向使用的 SSL head、分类或量化模块不应计入“实际已训练参数”。

保存完整骨干、adapter、最后投影、长度头、必要 buffers 和新的训练协议。旧 checkpoint 元数据描述冻结骨干和 FP16 特征往返，需要更新协议和严格重载校验；加载器不是简单地一律拒绝 `freeze_backbone=False`。

### 有效 token 与生成一致性

当前训练以 batch 内教师最大长度设置 `query_count`，回归和对比评分使用教师有效 mask。部署则输出全 `qmax`，再按 `round(exp(log_length))` 截取学生自己的有效序列；已有模型 `qmax=1423`。[训练 mask](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/scripts_omnidance/exp/ms/music_text_align.py#L163-L171)、[部署截取](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/scripts_omnidance/exp/ms/music_text_adapter.py#L286-L301)

新学生前向应统一输出全量 raw H 和自身预测长度。教师 mask 继续用于回归监督；实际部署序列和对比评分中的学生范围取自学生自己的长度。整数截取本身不可导，长度头先继续使用直接的 log 长度监督，不假设梯度能穿过取整。

adapter 没有 query self-attention，使用绝对 query 位置，相同输入下共同前缀不依赖 query 总数。因此不能直接把 query_count 的差异判为已经证实的失败根因。应实际检查 raw H、预测长度、tail、prefix、tags 和 packed 的一致性，而非只检查参数名和维度。

### 最小对照和验收

先在 2 至 4 条训练样本上做少量更新，检查实际骨干与 adapter 的梯度有限、非零且权重发生变化，H3 权重与固定 BN running statistics 不变。训练前向保留计算图，不能调用带 `no_grad` 的 `generate()`。

随后用相同初始化、波形、样本顺序、损失、学生长度规则和精度做两组对照：A 在线冻结骨干、训练 adapter；B 在线训练 dense 骨干与完整 adapter。B 是目标主线，A 用于判断骨干更新的贡献。先以 100 至 300 步观察学习和视频变化，再决定是否继续原预算。

每组保存后重新加载完整模型，比较学生前向及实际条件打包结果，并生成固定人物的原音乐和换音乐视频，必要时补零输入。分别评价舞蹈性、歌曲对应动作变化、节奏、人物衣着和画质。研究表示分数用于诊断，不应阻止用户已授权的直接视频验证。

这一轮表示训练不包含 DiT/video/生成预测损失。若表示改善仍不能带来视频控制，再单独设计更直接的生成行为监督实验；本方案不承诺仅解冻骨干即可解决所有问题。[联合训练与共享条件方案](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/notes/22_h3_bidir_blur_attribution_20261013.md#L1203-L1255)

## 冻结骨干阶段如何复现训练

以下命令复现已有冻结 MusicFM 的 H 对比阶段，不能训练上述新联合方案。假设已具备经审计、split 和协议一致的教师缓存、音乐特征及完整 raw H 初始 checkpoint。`--targets` 接收含 `finished.json` 和 `plan.json` 的缓存目录；输出目录必须尚不存在，父目录需要已经创建。

```bash
H3_REPO=/path/H3-posttrain

/opt/venv_music/bin/python \
  "$H3_REPO/scripts_omnidance/exp/ms/music_text_contrastive.py" \
  --init /path/raw_h_initial_checkpoint.pt \
  --split /path/split.json \
  --targets /path/teacher_targets \
  --features /path/music_features.pt \
  --out /path/new_hclip_run \
  --real_music_negatives_only \
  --steps 1875 --batch_size 8 \
  --lr 0.0003 --clip_weight 1 --length_weight 1
```

教师缓存入口为 `scripts_omnidance/exp/ms/music_text_cache.py`；冻结音乐特征和回归预训练入口分别是 `music_text_align.py --stage features`、`music_text_align.py --stage train`。这些历史入口不是骨干联合训练替代品。当前在线联合入口为 `music_text_contrastive.py --online --backbone_train`，使用 waveform bundle、分组优化器、共享学生条件和 V2 checkpoint；具体旧对照复现及下一步改动见 [最新解决方案](26_music_h_diagnosis_and_solution_plan_20261020.md#对应代码改动位置与复现)。

## 当前代码如何生成视频

### 环境和输入

实际视频入口是 `scripts_omnidance/exp/h3ar/sample_h3_ar.py`。`music_text_infer.py` 单独运行只输出 H tail，不生成视频。需要已应用项目补丁的 DiffSynth upstream、Ref2VA 权重、MusicFM 源码和本地权重、ffmpeg，以及 H3 和 MusicFM 两套依赖环境。现有代码保留集群路径，换机器需同步调整这些路径；仅克隆仓库不具备完整运行环境。

现有音乐 proxy 默认解释器为 `/opt/venv_music/bin/python`，sampler 没有 `--music_text_python` 参数。当前加载器已支持在线联合训练的完整 V2 checkpoint，相关模型已通过独立 tail 和原生布局核验；后续若改变模型或协议，还需相应重载与部署验证。

在 `/path/cases/full_joint.csv` 写入：

```csv
sample_id,image_path,audio_path,prompt
person_a,/absolute/path/person_a.png,/absolute/path/song_a.wav,
```

`prompt` 列需要存在并留空，当前 sampler 即便选择 empty 也会读取该列。`sample_id` 是样本和输出标识，不是人物语义提示。准备 `full_aswap.csv` 时保持同一个 sample_id 和参考图，把 audio_path 改为另一首歌；aswap 不会自动挑选替换歌曲。

输入音乐应为实际 `.wav` 文件。共享 loader 转为双声道、32 kHz，并裁剪或补零至 162000 samples，即 5.0625 秒，音乐骨干随后重采样到 24 kHz。当前协议输出 124 帧、24 fps，不能把这一命令描述为任意长度整曲生成。[共享音频读取](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/scripts_omnidance/exp/ms/music_clip_bridge.py#L24-L46)

### 视频命令

以下为 Linux 服务器命令模板，替换实际路径；`MUSIC_CKPT` 指加载器支持的完整 raw H 模型。已有候选可以复现已有结果，但尚未证明能够可靠生成音乐驱动舞蹈。

```bash
H3_REPO=/path/H3-posttrain
H3_UPSTREAM=/path/patched-DiffSynth-Studio
H3_WEIGHTS=/path/Ref2VA
MUSIC_CKPT=/path/full_music_text_checkpoint.pt
CASES=/path/cases
OUTPUT=/path/new_output

env CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$H3_UPSTREAM" \
  /opt/venv_h3/bin/python \
  "$H3_REPO/scripts_omnidance/exp/h3ar/sample_h3_ar.py" \
  --w "$H3_WEIGHTS" \
  --music_text_ckpt "$MUSIC_CKPT" \
  --csv_dir "$CASES" \
  --conds joint,aswap --rows 1 \
  --out "$OUTPUT" \
  --mode bi --prompt_mode empty --cfg 1 \
  --fixed_native_silence \
  --audio_tail_mode repeat_clean \
  --steps 50 --shift 12 --audio_shift 3 \
  --height 1280 --width 704 --seed 42
```

只有 `full_joint.csv` 时使用 `--conds joint`。输出为 `OUTPUT/joint/person_a_song_a.mp4`，换音乐结果在 `OUTPUT/aswap/`。sampler 会按文件存在跳过任务，每次新实验使用新输出目录。

零输入诊断在新的输出目录中增加 `--music_text_zero_input`，并可只使用 `--conds joint`；它只清零学生音乐输入，最终配乐仍是原输入音乐。固定 native 静音不等于整个学生没有接收音乐。

raw H 路径要求 `empty`、`bi`、`cfg=1`、固定 native 静音和 `repeat_clean`，不同时启用旧音频 checkpoint、音乐 guidance、LoRA 或教师 oracle。`--music_text_legacy_clip` 是旧单 token 诊断路径，不是本方案的 raw H 序列部署。[实际参数与组合检查](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/scripts_omnidance/exp/h3ar/sample_h3_ar.py#L15-L71)

生成过程只运行音乐学生和冻结 H3，不计算 CLIP 损失。推理不需要动作描述；对比学习的作用已经保存在训练后的学生权重中。

## 2026-10-19 执行接续：在线联合入口已实现，真实验收进行中

本节更新上文“待实现”的状态，不删除旧代码基准和历史负结果。拉取文档提交`013f236`后，已实现以下代码；截至本次记录，真实A/B小集验收已启动，正式100步和视频尚未完成。

- `music_text_adapter.py`：新增`configure_online_training(bool)`与可微`student_condition(waveform, context, waveform_lengths=None)`。学生全qmax输出并自行产生length/mask/tails；generate共用该构造。B只训练实际执行的`model.conv`、`model.conformer.layers`、`model.conformer.layer_norm`；A全部音乐骨干冻结。两组完整adapter可训，骨干保持eval，固定BN running统计并关闭随机层；B的dense BN affine可训。未调用的pos_conv_embed/SSL/cls/quantizer不计为训练参数。
- 新A/B协议统一FP32，禁autocast和历史冻结特征FP16往返。V2完整checkpoint保存scope、BN/随机层/精度、有效token规则、必要buffers及未注册的MusicFM mel mean/std；加载时恢复scope再核对requires_grad参数统计，严格检查state/dtype/shape/finite。旧v1历史checkpoint继续按原协议加载、保持已有输出行为。
- `music_text_contrastive.py --online`：在线waveform三音乐分支共同反传，教师mask仅回归、学生预测有效mask与teacher mask分别取最多64token计算序列score。raw MSE按整有效batch的teacher有效tokens累计，length/NCE按样本累计，microbatch不改变目标。优化器分组默认bone1e-5/adapter1e-4；未增加DPO、DiT或video loss。
- `music_text_pipeline.py --prepare_online_bundle`：校验旧数值bundle/split/init SHA，仅复用冻结教师和train统计，明确移除旧音乐features；读取共享loader的真实波形，缓存`[2,162000]`FP32输入及源文件/波形hash。缓存波形不是缓存可训练骨干的输出。
- `music_text_cache.py --audit_student_context`：重跑空语义TC，对旧教师缓存末行context直接比较，不以形状或两次新前向替代。3train+1val实际检查均exact、RMS0，prefix IDs/tags一致；这是4个选定参考的数值验证，不冒称2000条全比较。
- `music_text_infer.py --layout_audit`：将可微学生的保存probe与独立实际music proxy输出比较，再调用同一`replace_tail`和原生`PackedSequenceBuilder`比较完整prefix/tags/位置布局。无caption/GTlength进入学生；该审计不计算视频损失或宣称生成质量。

CPU独立掩码评分、无效token不影响score、有限非零梯度与微批loss/gradient等价通过。共享模块CPU fake测试覆盖A/Bscope/BN、train/eval、FP32、完整V2重载/异常拒绝；直接由旧模块生成的9类v1测试也通过。但这些不代替真实MusicFM骨干更新和H3部署验收。

当前真实验收在一基N5 `28.216.19.137` GPU6/7并行：A/B各4个训练样本、3次更新、有效batch4/micro2，独立新输出`music_text_online_smoke_A_v1`和`music_text_online_smoke_B_v1`，完整日志保留；前述context审计在N2 GPU7已正确结束。波形/教师bundle在N5本地`/tmp/music_text_online_inputs_v1.pt`，不上传数值PT。两组同源初始化为原raw H回归完整checkpoint SHA `cbb79c6338c2673d0c2f650a14da815234894055ffc14f0a56f1ec1afd1750fc`，不是旧32D音频adapter，也不是用不同骨干缓存作对照。

真实验收后首阶段拟定A/B各100步、有效batch8/micro2、同样本顺序和1800/200、上述学习率/损失/学生长度规则，完成200val和完整重载，再直接生成各固定3人物joint/aswap。先看实际视频再决定继续300步/原预算，不用表示指标阻止用户授权诊断、不自动扩人物或扫描损失。技术失败先修复并保存原失败，正式训练和视频结果按实际完成状态接续发布。

### 实际小集梯度与部署验收完成

A/B各3更新、12曝光均exit0。B实际dense骨干为316,343,296参数/400个tensor，首步400个tensor梯度均有限且非零，组梯度norm1.606665；完整adapter4,966,657参数/42tensor，梯度norm25.184543。B骨干400tensor与adapter42tensor实际权重变化；A骨干0tensor变化、adapter42变化。两组冻结参数和所有persistent buffers（含BN running统计）均未改变。

两组可微训练前向与eval的raw H/log_length/自身length/mask逐值相同，完整V2重载逐值相同，独立音乐环境部署tail逐值相同。小集的表示/长度诊断仍不能当作泛化或舞蹈效果。实测优化区间A1.151s、B2.439s，不包含输入读入、前后验证、checkpoint/部署审计；任务总时长不能冒充GPU训练时长。B峰值11,881,811,968bytes约11.07GiB，无AMP、未以显存为理由冻结骨干。

完整报告位于`results/ms_20261017/music_text_online_smoke_{A,B}_v1.json`及`music_text_online_smoke_{A,B}_weight_audit_v1.json`，context报告为`music_text_online_context_audit_v1.json`。waveform/teacher本地bundle SHA`ced68f1b921a0e529bded7de7d8f2f1d5a68778a1935668994470f852397ce8f`，完整源音轨/输入波形hash元数据另存`music_text_online_waveform_bundle_v1.json`，不发布数值PT。

随后已启动每GPU原生H3 prefix/tags/packed审计→100步的工具管理串行接续；只有技术布局检查通过才进入正式训练，布局检查不是研究质量门槛。正式新目录`music_text_online_A_s100_v1`/`music_text_online_B_s100_v1`；截至本条小集记录，100步和新视频尚未验收，后续以真实产物更新。

### 三组100步完整结果与针对性长度对照

A/B均完成100更新、800实际曝光、全200val、完整checkpoint重载与独立音乐部署；正式预测tail和真实空图prefix/tags/原PackedSequenceBuilder逐值验收通过。A骨干0tensor变化、B400tensor变化，adapter均42tensor变化，冻结权重/buffers不变。完整报告为`music_text_online_{A,B}_s100_{summary,val200,train_probe,weights,layout}_v1.json`。

为进一步判别“骨干更新是否主要用于预测caption长度”，对B step100在同一3train+1val计算未裁剪的单样本分项梯度，不作optimizer更新，全部权重/buffers哈希不变。骨干rawMSE/Etrain梯度norm0.000271–0.000934，长度loss0.7310–4.6334，NCE0.04838–0.26853；长度约为NCE的8–77倍。长度head自身norm8.345–125.503也会参与全模型norm裁剪。该诊断提示梯度竞争，但没有测全训练过程、batch梯度抵消/方向和Adam历史，不能把norm比例说成最终更新贡献或唯一根因。详见`music_text_online_B_s100_gradient_components_v1.json`。

因此增加**唯一单参数C对照**：同B完整骨干/adapter、同init/split/order/FP32/microbatch/100步/lr/锚定/NCE，把length_weight从1降为0.01，不改结构或学生条件、不添加生成loss。协议除该参数外逐字段一致。C也完成800曝光和200val、骨干400tensor更新、完整部署逐值验收；原始报告为`music_text_online_C_len001_s100_{summary,val200,train_probe,weights}_v1.json`。`len001`在本轮明确表示权重0.01，不是0.001。

| 配置（同200val） | correct > 一条真实wrong | correct−wrong score | raw MSE（样本macro） | 长度MAE | correct/wrong输出RMS |
|---|---:|---:|---:|---:|---:|
| A骨干冻结，长度权重1 | 52.5% | -0.00000810 | 5.45948 | 147.415 | 0.002919 |
| B骨干联训，长度权重1 | 54.5% | 0.00001273 | 5.46769 | 155.445 | 0.003503 |
| C骨干联训，长度权重0.01 | 55.5% | 0.00002451 | 5.46963 | 155.885 | 0.004720 |

相同teacher mask下train逐位置平均H的macro MSE为4.81506，仍优于三组。三候选tie-aware top1分别40.5%/41.25%/39.5%，不可把pairwise55.5%当三选一准确率或舞蹈准确率。C只有微小表示变化，没有修复长度或建立明确音乐增益；不能据四样本梯度就宣称降低长度权重成功。反复用同val200选配置仍有选择偏差。优化区间实测A66.514s、B142.618s、C143.304s，不含IO/前后val/checkpoint/部署；B/C峰值约11.09GiB。

A/B各6个joint/aswap和各3个zero joint的18视频任务已启动，C另9条按已验证B生成manager的pidfd结束事件接续同三节点，避免同时开两个H3冷加载。实际B三个joint已成功、全124帧及选定音轨完整解码，并已Read同人物三种配置的六采样帧对照：楼梯仍白蓝衣裤/小包替代参考浅黄衣灰裤，保留小手势/后勾腿母题；红衣仍嘴/脸附近单臂动作、侧头与抬臂；商店仍原地单臂和小重心变化。B局部姿态有变化，但这三条未显示可用编舞增量。此为采样帧视觉诊断，不冒称连续听看、卡点评分或新用户否决；swap/zero和C视频仍需完成。

A与零音乐队列曾在控制节点读取A完整权重时处于`generic_file_buffered_read`，不误判GPU训练停死、不重复启动；B是真实3卡采样而非holder100%。写入节点fsync及原SHA核实已完成，没有改权重。该同步操作不证明IO问题已完全消失，正常队列和日志保留。传输JSON遇默认SFTP subsystem不可用，使用节点支持的legacy SCP完成，仅传元数据，无模型/媒体公开上传。

下一步仍为完成所有已启动视频、完整输入/音画和原/换/零反事实审计，保留新旧模型/评分并追加清楚的在线A/B/C预设。100步尚非收敛结论；是否进入300/原预算按完整视频和表示结果判别，不因当前弱指标阻止推理，也不在本阶段偷偷增加DiT预测loss。

### 同起点300步预算检查完成，视频仍在接续

100步三组表示依赖弱、B三个joint仍小动作母题，因此进行已提出的100–300步范围内预算检查，不直接扩大到1875步。A/B/C均从同一原raw H回归init**独立重跑300步**、2400曝光，保持相同样本顺序/seed/有效batch/精度与各自长度权重，不能称从100恢复optimizer的连续训练。三组训练和全200val退出正确；A0bone、B/C400bone及各42adapter权重变化，固定buffers不变，完整V2重载/独立tail及原生prefix/tags/packed通过。

| 300步配置 | correct > 一条真实wrong | correct−wrong score | 三候选tie-aware top1 | rawMSE macro | 长度MAE |
|---|---:|---:|---:|---:|---:|
| A骨干冻结，length1 | 47.5% | -0.00024221 | 36.25% | 5.37288 | 141.070 |
| B骨干联训，length1 | 45.5% | -0.00026869 | 30.50% | 5.40003 | 143.010 |
| C骨干联训，length0.01 | 51.5% | 0.00036640 | 37.50% | 5.33456 | 142.340 |

B300正确−静音score -0.005302，C300为+0.002735；这是各自预测长度有效区间的对比分数，不能当音画正确率。C相对B只改变长度loss权重、两wrong平均margin为正，但第一wrong胜率51.5%及位置均值MSE更好，仍无明确语义学习/泛化成功；不把第二wrong58.5%挑作唯一成绩，也不以三个配方反复固定val选优推独立显著性。

优化区间A198.787s、B433.236s、C437.327s，不含输入/验证/重载；三组最大显存保持约11.09GiB，无新增H3训练loss。完整各200行、实际权重逐tensor SHA和独立部署/layout报告为`music_text_online_{A,B,C_len001}_s300_{summary,val200,train_probe,weights,layout}_v1.json`。

100步A/B normal各6、A zero3已完成所有采样；B zero及C normal后半/zero按现有节点串行继续，A300已在空出的三节点启动，其余300队列等待既有manager实际pidfd结束。300各配置仍固定三人物，joint/aswap各3、zero joint3，共27新视频，不扩源人物。实际case采样日志的1481/1506s是端到端采样区间，不当训练耗时；占卡程序100%与真正H3 PID/GPU计算分开。

评审追加器独立CPU纯内存测试发现并修复marker/final与展示case/视频路径错绑、scope/zero反转、计数错误和真值字符串误当验收成功；新增分支要求固定案例键集一致、expected/passed及实际row总数一致、旗标严格True和variant布尔一致。仅builder脚本修改，未改活跃采样/骨干源码。最终网页仍须依赖实际音画审计，不以CPU fixture通过当视频完成。

接续期间再次核实任务归属，发现同目标的额外300布局启动被独占日志拒绝，没有覆盖已运行任务；额外300生成等待器的A分支也被已存在正常目录拒绝。保留两次失败日志，只停止工具管理的重复等待器，不停止中心300正常manager及其正在采样子进程。另发现一个PPID1的重复A/B100审计进程，按准确cmdline/out/start_ticks/pidfd终止该孤立重复，保留工具管理的原审计。原PID匹配把动态拼出的完整报告路径要求为字面子串导致等待断言失败，修正为唯一任务token＋报告basename核验再pidfd事件等待，未再启动模型或改产物。三个问题属于调度/事件边界，不是训练或视频模型失败。

目前所有100步A/B原/换/零18视频采样均exit0；对应音画审计正在读取已核实的完整checkpoint和视频。C100另9、300另27均继续按单节点一个H3冷加载接续；不以预定队列标记当完成。新评审builder支持A/B18、C9和300 A/B/C27严格绑定，各模式保留旧模型、三人物及评分。176项纯内存fixture（6正确/170错误）通过，主线程另独立验证ABC新增6模型/2预设及错误路径拒绝；这些仅验证manifest逻辑，不替代实际视频完整审计。

## 2026-10-19 最终验收：六组训练和54视频正确结束，音乐编舞目标仍未达成

本节是本轮最终状态，取代前述“执行中”条目的当前状态，不删除历史过程。A/B/C各100步与300步六组训练、全200val、严格完整V2重载、独立音乐部署及原PackedSequenceBuilder核验全部完成；100步每配置800曝光，300步2400曝光，同一已审核1800/200划分、三人物独立生成集。B/C实际训练316,343,296个dense骨干参数、400tensor，并连同4,966,657参数/42tensor adapter更新；A骨干权重不变。冻结H3不进入优化器/表示训练计算图，未新增DiT/video/生成预测loss。

### 全54视频的技术与视觉证据

每配置固定3个人物joint/aswap各3、zero joint3，共六配置54个唯一视频。完整技术报告：
- `music_text_online_AB_s100_all18_audit_v1.json`：100步A/B18条。
- `music_text_online_C_s100_all9_audit_v1.json`：100步C9条。
- `music_text_online_ABC_s300_all27_audit_v2.json`：300步A/B/C27条。

以上均complete=true、passed=expected，124帧/24fps/704×1280逐帧decord及选定视频/音轨ffmpeg完整解码通过；checkpoint实际内容SHA、所有54日志SHA、空白语义prompt、无teacher/GTlength、raw最后投影、当前完整音乐骨干及两路native固定静音核实。每组native hash相同，跨两预算54条也相同。mux源音轨逐值相等和精确声明流数量未独立检查，原false标记不改；完整解码不等于音乐质量。

主线程实际Read全部100与300组参考＋六时刻采样图，保存`music_text_online_AB_s100_visual_v1.json`、`music_text_online_C_s100_visual_v1.json`和`music_text_online_ABC_s300_visual_v2.json`。这些是采样帧视觉评审，不冒称连续听看、新用户评分或准确率：
- 楼梯：A/B/C、原/换/零仍主要小手势及后勾腿，浅黄衣/灰裤仍被白蓝运动衣裤和小包替代。300步B原/换规划母题相近，zero局部位置/手势改变。
- 红衣：多数仍手到嘴边、手臂放头后和小幅上肢动作；A300换音乐、C300零音乐后段出现更大双臂动作，不能把零音乐也有的大动作当音乐编舞成功。
- 商店：三组原/换/零大多原地单臂、小重心变化，未见建立明显歌曲对应的丰富舞步序列。
- 检查帧没有明显灾难性坏帧，但没有量化画质等价或完整连续清晰度验收；动作差别存在，不称音乐完全无作用。

原/换RGB MAE的300步三case：A(.03833/.04274/.01717)、B(.03112/.02168/.02694)、C(.04928/.01353/.02111)，只是未对齐像素代理。相应原/换/零预测长度：A479/437/289、232/221/177、300/316/187；B467/439/232、284/210/159、292/308/154；C413/377/249、243/302/171、290/261/160。token内容和长度/packed布局都变，不能把这些RGB差当语义控制增益或因果隔离。

### 本轮判别结论与下一步边界

1. **工程偏差已修正**：确实在线训练音乐dense骨干和完整adapter，学生可微train/eval/重载/独立部署共用全qmax、自身length/mask/tail；教师length只loss，loss标准化不改变部署raw坐标。实际图prefix/context、tags和原打包已检查。但表示训练并未每步穿过DiT；不能把条件协议一致说成已经获得文字同等生成行为。
2. **这次不再能以“骨干没训练”或“adapter丢失”解释失败**：B/C骨干400tensor更新，完整链进文本条件分支，仍没有建立可用编舞。结论只限本数据/初始化/损失/100–300步配方，不否定整个idea或宣称收敛。
3. **长度梯度竞争只是部分假说**：四样本length梯度更大；C降至0.01后300步rawMSE略低、两wrong margin平均正，但第一wrong胜率51.5%、位置均值H误差仍更低、视频未显示可靠舞蹈增量。因此没有证明长度loss是唯一根因，也没有证明减权重解决问题。
4. **主要残留证据指向条件表示不足**：300步训练8样本probe的correct/wrong margin较val大，val依赖弱；MSE平均模板、序列score不保词序、歌曲与动作caption语义歧义及图条件先验仍待分离。不能把8样本probe当全train准确率、不能把同val反复比较作独立泛化检验。
5. 不自动继续更多数据/步数/模型扫描，也不偷偷加动作prompt、身份文字或后置首帧。需要更直接的冻结H3生成行为监督时，必须另行明确授权；本轮没有加入。下一阶段可讨论更有语义/时序针对性的音乐侧目标和生成条件监督，但本次阶段实验及负结果已完整交付。

### 调度恢复、资源和发布完整性

中心300零音乐wait用宽泛substring匹配把真正生成PID和新审计waiter同时命中，future失败；A/B正常6+6仍正确结束，旧manager随后exit1。没有因manager异常重跑正常任务。另开只针对未创建零音乐组的恢复队列，A/B zero各3正确结束；按A/B原manager准确pidfd结束接续C300原/换6与zero3。最终恢复报告`music_text_online_s300_generation_recovery_v2.json`保留原错误，日志/媒体不覆盖。额外等待队列遇已存在C目录被拒绝，未创建重复视频。PPID1重复审计和无用waiter已精确终止；原审计继续完成。重复评审服务启动报端口占用，保留原正确服务不重启正常生成。上述是实际发生的调度错误及恢复，不能写成全程从未重复尝试或用原exit1否定完成的模型产物。

最终8766清单`manifest_music_text_online300_ABC_final_v3.json`为30模型/3人物/13预设，完整保留原18模型、人物映射和旧预设；54新视频Range206，主页/清单200，N2跨节点清单200。验收文件`music_text_online_final_review_access_v3.json`。新首预设【在线骨干预算·300步】A冻结/B联训length1/C联训length0.01，切joint/aswap；第二预设比较B/C原与零音乐，仅joint。100步预设也保留。旧评分继续使用同评分文件，不发布音视频或权重。

八节点最终只读核查无本轮训练/采样/推理/审计进程，GPU7各975MiB/100%为占卡程序，不计实验利用率；保留工具管理的8766评审服务。实际六组优化GPU时间合计约23.70分钟（A10066.514+B100142.618+C100143.304+A300198.787+B300433.236+C300437.327秒），不含IO/验证/存权重/生成，不能以holder100%或启动器总时长替代。生成主要利用八节点各一张卡、每节点一个H3冷加载。

最终独立复核代理Explore-1因守护进程内存不足未启动、没有产出；主线程随后直接检查六组protocol、200val、实际更新scope、完整部署、54唯一视频日志SHA及解码记录、最终清单原模型/预设保留，检查通过。不能冒称失败代理提供了独立分析。

## 2026-10-19 后续只读消融：同长度评分没有改变弱音乐判别结论

用户要求持续推进后，先复用已完成A/B/C300完整权重和N5本地waveform/teacher bundle，**零optimizer更新、不校准长度头、不改生成条件、不新增视频**。初步保存结果中score margin与“正确预测长度比wrong更接近teacher长度”的Pearson相关分别0.62516/0.47793/0.74488；但相关性不是长度捷径的因果证据，且自然预测长度相同的样本仅2/4/1条，不能用该小子集判别。

实际消融对每个200val样本的正确/两真实wrong/zero四个输出，先按原学生自身长度重算旧score，再使用四者预测长度最小值的共同可见query交集作为**评分mask**。teacher原有效区间仍仅用于监督，交集长度也不馈入学生；原生学生生成的raw H、log长度、截断tail完全不改。该实验只消除同一次候选评分中的长度选择差异，**不是固定H3 packed布局的视频实验**，也没有引入teacher长度部署。

N5 GPU6完成三模型各200val两次独立执行：最初临时命令与正式`music_text_diagnostic.py --common_interval`入口600条case和统计逐值相同；原score最大绝对误差0、原预测长度全部exact，完整checkpoint SHA/bundle SHA及所有state参数/buffers与原weight audit一致，运行前后不变。报告`results/ms_20261017/music_text_online_s300_common_interval_diagnostic_v2.json`。

| 300步模型 | 原correct>wrong | 共同区间correct>wrong | 共同区间margin wrong | 共同teacher交集MSE正确更优 |
|---|---:|---:|---:|---:|
| A冻结 | 47.5% | 47.5% | -0.00024197 | 50.0% |
| B骨干联训length1 | 45.5% | 45.5% | -0.00026837 | 47.5% |
| C骨干联训length0.01 | 51.5% | 51.5% | +0.00036668 | 52.0% |

两真实wrong的共同区间胜率分别A47.5%/57.5%、B45.5%/45.5%、C51.5%/58.5%；共同区间correct>zero为53.0%/21.0%/74.5%。zero可分不能替代真实音乐语义判别。固定query交集后score变化很小，三个模型对第一真实wrong胜率均不变，**不能把原score弱判别或C的轻微正margin单独归咎于评分长度差**。该结果也不排除长度在原H3视频布局中的影响、不证明内容条件足够、不判定唯一根因；视频反事实混淆仍未在同布局干预下测过。

复现入口（在已核验N5环境执行，输出必须是新路径）：
```sh
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/venv_music/bin/python scripts_omnidance/exp/ms/music_text_diagnostic.py --common_interval --exp_root /apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp --bundle /tmp/music_text_online_inputs_v1.pt --runs music_text_online_A_s300_v1 music_text_online_B_s300_v1 music_text_online_C_len001_s300_v1 --out /apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/music_text_online_s300_common_interval_reproduction.json
```

下一步不再自动扫描相同CLIP/MSE更多步数或校准length。若进一步采用冻结H3的去噪预测一致性监督，只优化MusicFM和完整adapter，需要单独明确授权，因为此前限定不加DiT/video/生成预测loss；本诊断没有越过该边界。只读探索代理本次再次上游错误退出，没有可用结论，以上由主线程代码读取和实际结果核验完成。

## 2026-10-19 新授权的冻结H3预测监督：真实梯度通过，受控训练执行中

用户通过明确选项**“授权小规模实验”**批准：新增冻结H3去噪预测一致性loss，仅优化MusicFM实际dense骨干和完整adapter，先验梯度/显存，再有无loss小实验及固定3case视频；生成仍仅音乐+参考图，教师动作文字只监督。H3文本encoder/DiT/VAE不训练，不恢复动作prompt，不采用GT FM velocity target。此授权仅适用本小实验，历史“未授权新生成loss”不能再当当前阻塞，也不代表允许任意其他loss/扩量。

### 精确VJP边界与真实输入

两套不同Transformers环境不混装，使用`music_text_behavior.py`的prepare/export/h3_vjp/music_vjp/update阶段：音乐环境生成学生全qmax、自身length和FP32tail；H3环境把同一FP32tail置leaf，按原BF16 cast/replace_tail/自己的packed进入原冻结DiT，与仅teacher监督条件的同noisy video/audio/time预测比较，返回dL/dtail；音乐环境按同checkpoint/wave/context重算，要求torch.equal完整tail后明确反传VJP。不是使用detach推理代理训练：数据边界虽存盘，精确链式梯度显式返回，重算失败不更新。离散length round/long不传此loss梯度，保留原loglength监督使全部42adapter仍共同训练。

主线程在N5真实读取原hr索引，1800train/200val全部匹配；选择原3train `AAcaGYtiTozEttaF/ACYTcnkJDuOfXBiA/ACjmBvCtqPhuyLRS`、1val `QcXTBHnhvaKIOlgH`，三生成人物继续独立。原hr `input_latents [1,24,37,80,44]`用于构造两路相同noisy state，不当velocity标签。原参考latent7424×96、实际prepared image2048×3712、空prefix7437×5120与原cache前缀逐值exact；不借用caption-conditioned prefix或teacher长度决定学生。

两路native音频换为同162000样本32k静音经冻结原audio-VAE编码、retake尾部repeat_clean及全零mask；不是填零latent。实际BF16reference/retake哈希与既有部署完全相同。缓存encode_audio返回FP32，初次model_fn index_copy dtype报错；在H3使用边界显式转BF16，与生成单位处理相同。mask缓存原为广播展开[2,32,207]，实际前向取[2,1,207]全零使形状也与部署一致；两种全零广播在模型中相同，但不把不同形状tensor hash冒称exact。数值PT和optimizer不公开。

### 短时长及全124帧实际梯度／显存

- 2个video latent帧smoke：teacher length509、学生自身length352，各自packed17984/17856；prediction MSE0.04043576，FP32 tail grad norm0.08267116。DiT全state hash前后不变、无参数grad；VJP重算精确到MusicFM400个dense tensor＋38个内容adapter tensor，全部有限非零。lengthhead4tensor没有此loss梯度，是预期离散length边界，后续保留length loss。没有optimizer更新。
- 原37latent/704p/124帧、单GPU79.25GiB：teacher/student前向可运行，反向checkpoint重算MLP申请1.30GiB时OOM；不能以短测试通过说全长可训。
- 尝试MLP逐token1024分块，BF16输出/梯度parity预设门槛未通过，拒绝用于训练，没有放宽门槛或修改原H3实现。该失败没有产生新VJP权重更新。
- 改为**原DiT后半blocks跨第二GPU**，只移动设备、不分块算子/换attention/量化；float device copy保留autograd，原block non-reentrant activation checkpoint＋CPU保存激活。全37latent成功：prediction MSE0.02112872、tail grad norm0.01231660，teacher/student各自packed48768/48640，DiT全参数buffer hash不变、参数grad absent。两GPU峰值50.03/48.33GiB，prediction区间289.005s，不含模型加载/模型hash，也非未来耗时预测。使用torch attention。此前2latent损失不可与37latent损失当改进比较。

公开元报告`music_text_behavior_smoke_v1_cache.json`、`music_text_behavior_smoke_v2_h3_vjp.json`、`music_text_behavior_smoke_v2_music_vjp.json`、`music_text_behavior_full37_twogpu_v3_h3_vjp.json`。只读探索代理已完成静态接口核查；真正CUDA梯度/显存结论来自上述实际工具产物，不来自代理推断。启动器一次Python3.11 fstring嵌套引号语法错误，未启动GPU，修正并保留失败日志。

### 有无loss受控小实验（尚未宣称完成）

两组从已完整验收C300同一checkpoint SHA`189a0a74...f605`起点，独立AdamW零初始optimizer；同3个train，固定顺序轮转两遍，**6更新/6曝光、batch1**，bone1e-5/adapter1e-4、weight decay0、global gradclip1。每次存完整音乐checkpoint与匹配optimizer state，后续重载恢复Adam历史，不把逐步重新初始化optimizer当连续训练。

共同原表示loss=`rawMSE/Etrain + 0.01*loglength_MSE + sequence_NCE(two_real_negatives)/temperature0.07`，使用全原train池确定同wrong/second，NCE权重1（从C300延续，不重设warmup）。唯一改动为prediction组额外`MSE(v_student,v_teacher)`权重1；control权重0。训练样本同GT noisy state `x=.1*z+.9*noise(seed42)`，teacher仅监督，双方native静音、参考和时间相同，各自文本尾长/原packed。第一预测VJP复用同起点已核验full37产物，须export内容SHA一致才可消费。其后更新均重算当前完整学生VJP。

最终保存完整V2模型、实际400bone/42adapter更新审计、全200表示val和独立部署；另比较同1val的原始/控制/预测三权重对同frozen teacher的预测误差，不将3train拟合当泛化。随后每配置仍独立固定3源人物joint/aswap/zerojoint，共18新视频，完整音画/input/部署审计及实际评审，旧C3009视频复用对照。不因新loss梯度大或small train下降宣称编舞增量；没有完成前不关闭本轮任务。

## 冻结H3预测监督小实验最终结果（2026-10-19）

两组均正确完成6更新/6曝光，同3train轮转两遍。实际校验除behavior_weight0/1外所有protocol字段相同、样本/负例顺序相同、首步原base loss逐值相等；独立读取两组optimizer.pt：442参数tensor均有有限Adam状态，全部step=6、checkpoint SHA/init SHA/history绑定，证明保存恢复后不是每步重置Adam。两组均400个dense骨干tensor与42adapter tensor更新，冻结参数/buffers不变；全200val、完整V2重载/独立尾部、真实空prefix/原packed均通过。

终点完整权重SHA：control `85f2bb23d792773a8ac15ff0edd820b34cdd8078ecc1bce1c8ce21f4fc111816`；prediction `c148a981c7319bb5c01335efd8b5d78c709b7a43a65dbb56e968ae9212e4e7c1`。数值权重/optimizer不公开；元审计为`music_text_behavior_{control,prediction}_s6_v1_{summary,weights,val200,layout}.json`及`music_text_behavior_s6_v1_training_jobs.json`。每一次实际full37 H3 VJP均无H3参数梯度/哈希变化、与当次train ID/loss/history一致，六次完整元记录为`music_text_behavior_s6_v1_all_training_vjp_audit.json`；29个实际阶段命令及日志SHA为`music_text_behavior_s6_v1_reproduction_commands.json`。复现必须选择新输出路径，不覆盖本轮产物。

### 留出行为误差：有微小信号，不能称泛化成功

用固定1val `QcXTBHnhvaKIOlgH`、同sigma0.9/noise_seed42、同完整124帧noisy state/native静音/reference/fullprefix评估初始C300/control/prediction；三路teacher_prediction SHA完全相同，H3固定参数前后不变。各student仍用自身预测长度，各自原packed。

| 模型 | 同留出teacher预测MSE | student自身长度 | 全200val correct>第一wrong | raw H MSE | 长度MAE |
|---|---:|---:|---:|---:|---:|
| 初始化C300 | 0.02921945 | 262 | 51.5% | 5.33456 | 142.34 |
| 原表示loss control，6更新 | 0.02999355 | 389 | 50.5% | 5.33036 | 150.31 |
| 加预测loss1，6更新 | 0.02893900 | 393 | 50.5% | 5.33050 | 151.12 |

prediction相比control该**单样本单噪声单时间**MSE低3.516%，相对init仅低0.960%；control反而略高于init。此为小规模行为信号，非泛化显著性、卡点准确率或可用舞蹈结论。全200val三候选tie-aware top1 control/prediction均34.5%，第一wrong胜率均50.5%，第二wrong52.0%/52.5%，没有清楚语义判优。训练3个ID第二次经过时prediction loss相对第一次：.02112872→.02018517、.01396171→.01433093、.01772999→.01766812；不同ID损失不能作为统一单调下降曲线，且这不是同固定最终模型对全train的独立评估。

详见`music_text_behavior_heldout_s6_v1_compare.json`及三路`music_text_behavior_heldout_s6_v1_{init,control,prediction}_vjp.json`。该留出没有optimizer更新，不将teacher tail/length输入学生；查询长度和packed仍变，不能用误差下降孤立证明学到了歌曲语义。

### 18新视频正确结束，采样视觉未建立编舞增量

每组固定三参考人物原/换各3、zero joint3，共18新视频exit0。`music_text_behavior_s6_v1_all18_audit.json` complete=true、expected=passed=18，704×1280/24fps/124帧全部逐帧与选定音轨完整解码、empty student/native固定静音/完整当前音乐checkpoint及预测length验证通过。主线程另核对18个真实媒体SHA、日志SHA、无oracle/teacher文字和最后adapter投影，以及两组真实训练协议；`music_text_behavior_s6_v1_independent_audit.json`。未独立核实mux逐值相等、声明流数量，false不改。

实际Read六套reference＋0/1/2/3/4/5.125秒采样图，记录`music_text_behavior_s6_v1_visual.json`，不是连续听看或新用户评分：
- 楼梯原/换/零两组仍转身、小手势、后勾腿；prediction局部腿动作时间与control不同，但浅黄衣灰裤仍替为白蓝运动衣裤并增加小包；未见丰富且歌曲特定的编排增量。
- 红衣两组仍手到嘴边／头后／单臂抬起母题，zero后段姿态有局部变化，不能将变化本身当正确歌曲控制。
- 商店主要原地单臂和小重心变化，prediction原曲某采样帧双臂幅度较大，但换/零与整体舞步没有构成可靠编舞收益证据。
- 没有检查帧层面的灾难性坏帧，不冒称量化清晰度等价或完整连续画质通过。

原/换RGB MAE：control .02377/.01608/.02628、prediction .02084/.01789/.02480；原/换/零长度control493/471/371、305/416/248、400/352/226，prediction495/474/373、307/419/250、401/354/228。两模型同variant长度相近，但没有硬固定布局；RGB差非pose/卡点accuracy，不能作为正效果或统计等价。

### 交付与当前判断

**本次已证实**全124帧冻结H3预测loss可经BF16cast和精确VJP进入MusicFM真实dense骨干及完整内容adapter；原length监督训练其余4个lengthhead参数。单卡MLP反向OOM后两卡split-block正确完成，不用量化/放宽chunk parity或训练H3。**未证实**6更新配方带来可用编舞或可靠泛化增益，不能据此否定预测监督更大方案，也不把仅3train/6exposure当收敛。

8766新清单`manifest_music_text_behavior_s6_final_v1.json`为34模型/3人物/15预设，原30模型/全部旧预设/评分映射保留；新首预设【冻结H3预测监督·6更新】原C300/原表示loss对照/预测loss1，先joint后aswap；第二预设比较两组原与零音乐，仅joint。总72视频Range206、主页/清单200、N2跨节点200，文件`music_text_behavior_s6_v1_review_access.json`。保留旧54视频，不发布任何模型/音视频/optimizer。

八节点最终实际只读核查没有本轮训练、行为VJP、学生infer或采样/解码进程，仅保留工具管理评审服务。主线程完成上述所有检查；额外只读VJP审计代理因上游错误未提供结论，不伪称其独立验收通过。方案/复现代码/所有已完成报告及受限分析公开，任务收尾不等于音乐替代动作文字的总体目标成功。未启动后置首帧、身份文字、单独长度校准、更多数据/更新扫描或其他生成loss。

## 2026-10-20 人工负评：仍像摆pose，而不是连续舞蹈——问题记录与解决路线

**用户最新反馈：**“这一轮还是类似摆pose而不是真正的舞蹈”。这是最新生成的定性人工负评，取代“尚无新用户反馈”的当前状态；上节该措辞仅反映当时采样帧检查。用户未提供18条逐视频评分、未逐个指明control/prediction的等级，也没有评价全部历史配置，不能扩大成所有模型逐条均被否决或编造卡点/画质数字。

**当前判定：** 本轮实现、六更新训练和视频技术验收完成，连续舞蹈效果仍不达目标。此前的单留出MSE微降、非零VJP、媒体解码或局部姿态变化，都不是可用编舞证据。以下内容是问题归因与后续方案；本次只更新文档，没有启动新推理、修改训练loss或扩大训练预算。

### 1. 已证实的不足与尚未证明的原因

| 问题／假说 | 已有证据 | 不能据此推断 |
|---|---|---|
| 学生条件尚未承担动作文字的连续编舞作用 | 用户摆pose负评；正确/错配/零音乐多保留小手势母题；control/prediction200val第一wrong胜率均50.5% | 不能说音乐完全没有作用，或MusicFM→H侧idea不可行 |
| 最新训练覆盖和噪声状态非常有限 | C300起点、3train轮转两遍、6更新/6曝光；固定sigma0.9与seed42，损失在GT加噪latent上计算 | 既不能称收敛，也不能承诺加步数必然解决；起点已训练不等于这次新目标充分训练 |
| 音乐→逐词hidden回归存在多解及平均化风险 | train逐位置平均H的val MSE4.81506仍低于学生约5.33；三组共同可见query评分未改善真实wrong判别 | 不等于已经证明学生逐值等于均值；更不能用相关性认定长度是唯一捷径 |
| 当前对比目标不足以规定动作顺序／拍点 | `score_samples`把最多64token两两余弦矩阵作双向soft-Chamfer，所选token集合顺序不敏感；adapter虽有时间/位置编码，但没有显式步法、衔接或逐拍目标 | 不能称模型无时序能力；逐位置MSE和DiT本身仍有位置与时序结构 |
| 全latent预测MSE可能被静态/背景和低运动分量主导 | 新loss为所有video latent位置的均方差，未按人体/动态区域重权，也未直接验证连续舞蹈、步法或重心 | 这只是待分解的loss贡献假说，未测静态/动态份额，不能先宣布背景主导 |
| 教师上界可能不足 | teacher为原MotionScript＋舞种的H3 hidden；语句多关节位置/转动事实，描述动作不等于冻结H3能生成流畅舞蹈 | 教师连续舞蹈质量尚未逐case确认，不能先说teacher失败、MotionScript无效或富文本必然更好 |
| 训练与生成状态分布不同 | 训练同GT加噪state单步预测，生成随机噪声起点50步、模型自身轨迹；条件构造一致不等于轨迹一致 | 没测各噪声段／student rollout误差，不能称已证明曝光偏差是根因 |
| 长度/packed与图条件先验仍可能混淆 | 所有音乐变体用自己预测长度；reference-only下衣着/小包和姿态母题漂移；共同区间评分不改变47.5/45.5/51.5%胜率 | 评分mask消融不是固定H3布局视频消融；不能用图保真变化单独证明图context捷径 |

已基本排除的工程原因：骨干未训练、最后adapter丢失、预测loss未反传、生成读取teacher caption、真实音乐从native audio旁路进入。实际400dense＋42adapter更新及严格部署证据仍成立。它们不再是继续重复验证的主线。

### 2. 纠正下一步优先级：教师oracle已有，先复审，不重复生成

**已有产物，不是待从零生成：**
- 原MotionScript教师oracle：`H3AR/val_musicreplace_text_oracle_v1`，三joint＋三aswap共6条。
- 中性代词教师oracle：`H3AR/val_musicreplace_text_oracle_neutral_v1`，同三参考人物joint3条。
- 元证据：`results/ms_20261017/music_text_oracle_all6_integrity_v1.json`、`music_text_oracle_neutral_joint_integrity_v1.json`、`music_text_oracle_caps_v1.json`；notes/22 §28.20记录接口等价和楼梯人物保真差异。

此前已验证teacher hidden拆拼／原packed／首步预测等价，但**没有证明这些teacher视频是用户认可的连续舞蹈上界**。中性teacher曾恢复楼梯女性浅黄衣/灰裤，只是参考保真证据，不是编舞成功。原teacher aswap缺原MS时使用明确VLM fallback，与joint不是同种teacher来源，不能作为纯MotionScript／纯格式单变量对照。

因此上一条“先生成teacher对照”的建议应落实为：**先复用已存在neutral三joint，与同case的C300、control、prediction连续复审。** 仅缺失或配置/teacher来源实质不一致的组，才另列必要新对照；不再冷加载生成一套相同视频。teacher拥有真实动作描述，仅作为监督上界，绝不包装成音乐预测分数，学生部署仍只音乐＋参考图。

判别分支：
1. **teacher也主要摆pose**：优先修教师监督/表达，而不是学生加步数模仿。逐case核对原片段动作事实、teacher措辞/顺序和生成；必要时仅teacher侧做事实保持的叙事化改写或动作段整理，与原MS同事实对照，不能凭空加大动作/步法或把新文本送给学生生成。新teacher目标必须在同三case连续舞蹈和参考保真上验收后才重缓存训练；旧目标/视频保留。
2. **teacher能舞蹈、学生仍摆pose**：才进入条件学习和噪声轨迹诊断，证明当前任务存在可达上界，不盲目换teacher或恢复学生prompt。
3. **teacher仅部分case可用**：分别记录，不宣称全上界已通过；先定位失败case的表述/遮挡/参考条件，再选对应问题处理。

### 3. 最小诊断与对应修复（分阶段，不同时扫所有配方）

**P0：定义并复审连续舞蹈效果。** 每条记录连续性、腿部步法、骨盆/重心转移、上下肢协调、动作衔接、节奏强调和参考人物保真；裁切/遮挡的腿部指标弃权。现有三人物固定；用户可复审完整视频，助手不能把六帧采样当连续听看。动作速度、姿态跨度或RGB差只作辅助，不以“越大越好”鼓励随机挥臂/抖动。此前音乐不是唯一编舞真值，因此不要求同曲逐像素复现原片段。

**P1：在不更新模型的条件下区分问题。** 复用已完整权重，先测后决定改哪一项：
- **状态/噪声段诊断**：从实际50步、shift12的采样sigma序列预先固定高/中/低噪声代表点及noise seeds，记录同GT加噪state下teacher/学生预测误差，并在stop-gradient的student自身生成轨迹的同一个state查询teacher；不能比较各自不同state预测再归因，跨GT/rollout不同state误差也不直接作改善百分比。不同sigma的velocity尺度不同，报告归一化残差与分时间段趋势，不只挑最有利时间。未测结果不得承诺哪段能修复。
- **loss信号诊断**：同参数位置记录raw回归、length、NCE、预测loss的分组梯度norm及夹角/冲突，结合既有grad clipping/Adam历史解释；不能仅比较loss数值大小。对video residual按合法人体/背景区域或时间差分分解贡献先只作诊断，未知遮挡/人框误差必须披露。新预测loss只有正确音乐分支，是否同样接受错配音乐条件也能得到相近预测，应作为反事实证据测量，而不是只看correct MSE下降。
- **内容结构诊断**：对共享query区间的token跨音乐方差、均值模板距离、标准化后的有效秩/时间敏感性作分析，区分全量raw MSE被公共分量覆盖与真正语义差异；同长度评分消融已经做完，不重复拿它当新发现。若验证score不保顺序，只在固定的最多64个抽样token集合内置换；任意重排全长序列会改变等距抽样子集，不能误称score必然不变。上下文化teacher H本身仍含词序信息，不把评分聚合性质升级为学生完全没有时序。不把attention权重或有效秩单独当可解释性证明。

**P2：只对P0/P1支持的机制做一个受控修复。**
- 若teacher可用且误差主要在中/低噪声或student轨迹：优先把已有预测一致性监督从固定sigma/GTstate改为覆盖少量预定噪声段，必要时在student同state上teacher查询；H3冻结、teacher仅监督、学生依旧完整音乐/图条件。保持同init、样本和总optimizer更新数，明确state来源和teacher forward次数，留出sample与训练sample隔离。不能把独立两条rollout每帧差当可比监督。
- 若梯度证据支持NCE/length盖住或与behavior冲突：固定其余项做一次权重/梯度尺度平衡对照，量化行为梯度实际贡献，而非因预测loss约0.02就任意加100倍。原length0.01对照已完成且没有生成收益，不反复无依据调length。raw回归保留/移除须独立变量，不同时改初始权重、teacher和时序模块。
- 若teacher可用但content明显平均化：在原H部署接口上比较更接近动作语义的条件目标，例如剔除train-only公共分量的**loss侧**监督、保顺序的对齐或同舞种可区分动作负例；原输出raw坐标不变、完整adapter仍部署，不能把loss标准化向量直接灌入H3。音乐与caption一对多时按曲目重复/动作等价检查假负例，可考虑多正样本或风格层次目标，但不能给每首歌曲硬赋唯一动作。
- 只有残差贡献和实际人评支持“动态部分不足”，才另提人体/时间差分预测监督或motion感知loss；不要先增加可训练pose网络/新语义head、放大运动量或改变DiT。架构如query时序自注意力、分时间音乐token需保留原adapter对照，不能只凭当前摆pose就宣布256维宽度或无query self-attention为根因。

P2上述修改都是**候选方案、未执行且未判优**。此前明确授权的是已完成的冻结H3预测一致性小实验；新的预算、teacher改写、轨迹监督范围以及额外motion/temporal loss，应在具体单变量协议明确后获得授权，不把“思考解决方案／写文档”当自动开训命令。首帧固定、身份文字、单独长度校准继续后置。

### 4. 验收与继续／停止规则

- 优先级：**复用teacher上界复审 → 无训练机制诊断 → 一个有证据的修复 → 固定三case实际生成 → 连续人评与发布**。不能逆序先无限扩数据/步数，也不重复已排除的checkpoint/最后投影问题。
- 每次同init/split/三参考/seed/704p/50步/native静音/完整学生条件，记录唯一改变量；训练样本不混入三生成case。若只换训练state分布，保持更新数和teacher协议一致并另报计算量。
- 效果验收不是MSE阈值：teacher至少需具备所评case的连续可用舞蹈；学生相对同预算control在可观察的动作衔接／步法／协调上有真实增量、换音乐响应合理且明显区别于zero，而非仅静态姿态/衣着变化，画质/参考保真不退化。三case只作为小集人评，不能宣布总体统计显著；未检验时记“待验收”，不编造已达标。
- teacher不通过则先停学生训练修目标；机制诊断无明确方向则发布不确定性，不把多训当唯一补救；修复只改善表示/单步MSE但连续人评仍摆pose时，保持负结论，重新核对目标与state分布，而非宣称工程完成即音乐编舞完成。

本次文档记录保留所有已生成72视频、历史评分与旧负结果，新增用户负评和待验证方案，不修改已发布visual JSON中当时“无新用户评分”的历史记录。
