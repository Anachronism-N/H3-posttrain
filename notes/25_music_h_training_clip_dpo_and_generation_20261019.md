# H3 音乐条件训练 CLIP 与 DPO 区别及生成流程

更新日期：2026-10-19，北京时间。代码基准：[5fa2be623511eb232db60793de3f142c948258c7](https://github.com/Anachronism-N/H3-posttrain/commit/5fa2be623511eb232db60793de3f142c948258c7)。

目标是训练 MusicFM 的实际 dense 骨干和完整 adapter，让音乐产生能够替代动作文字的 H3 原生条件，最终仅凭音乐和参考图生成舞蹈。H3 文本编码器、DiT 和 VAE 保持冻结。

当前 H 序列代码已经使用 CLIP 式对比损失，但仍冻结 MusicFM，只训练 adapter 和长度头。它属于教师隐状态监督与对比学习，没有采用 DPO。下一步应保留对比目标，实现音乐骨干和 adapter 的在线联合训练，并统一学生条件在训练和生成时的构造方式。联合训练方案尚未实现和运行，不能把现有负结果解释为该方案已经失败。

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

## 当前已实现的训练流程

一条样本包含音乐、参考图和对应的动作描述。已有正式划分为 1800 条训练、200 条验证。

1. 冻结 H3 文本编码器，以参考图结构和 `Dance style: {genre}. {motion_original}` 产生教师 raw H 动作 tail、长度和结构 prefix 上下文。中性化选项只调整性别代词，保留动作内容。
2. 冻结公开预训练 MusicFM，预计算音乐 dense 特征。当前正式特征缓存是每条 `[2,127,1024]` 的 FP16 张量，表示双声道、时间特征和特征维度。
3. adapter 接收音乐特征和同参考图 context，输出 raw H 序列和对数长度。
4. 对正确歌曲输出计算 H 回归与长度损失；对同图的错配歌曲输出增加对比损失。
5. 反向传播只更新 adapter。骨干出现梯度会触发当前 trainer 的检查错误。

依据：[教师缓存](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/scripts_omnidance/exp/ms/music_text_cache.py#L30)、[冻结骨干和特征缓存要求](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/scripts_omnidance/exp/ms/music_text_contrastive.py#L189-L206)、[优化器与反向传播](https://github.com/Anachronism-N/H3-posttrain/blob/5fa2be623511eb232db60793de3f142c948258c7/scripts_omnidance/exp/ms/music_text_contrastive.py#L225-L270)。

批量教师缓存保存 caption 编码中的结构 prefix 最后一行，部署则从空语义 prompt 的结构 prefix 取对应 context。复用缓存需要核验这两种 prefix/context 的等价性，不能仅凭形状相同认定一致，也不能未经验证就推断存在动作文本泄漏。

## 当前损失与 CLIP 的含义

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

## 下一步联合训练方案

以下为待实现方案，不是当前可直接启动的新训练命令。

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

## 当前代码如何复现训练

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

教师缓存入口为 `scripts_omnidance/exp/ms/music_text_cache.py`；冻结音乐特征和回归预训练入口分别是 `music_text_align.py --stage features`、`music_text_align.py --stage train`。这些入口不是骨干联合训练替代品。新增联合入口还需实现在线数据、优化器分组、共享学生条件和新 checkpoint 协议，当前没有一个现成的 `--unfreeze_backbone` 参数完成全部工作。

## 当前代码如何生成视频

### 环境和输入

实际视频入口是 `scripts_omnidance/exp/h3ar/sample_h3_ar.py`。`music_text_infer.py` 单独运行只输出 H tail，不生成视频。需要已应用项目补丁的 DiffSynth upstream、Ref2VA 权重、MusicFM 源码和本地权重、ffmpeg，以及 H3 和 MusicFM 两套依赖环境。现有代码保留集群路径，换机器需同步调整这些路径；仅克隆仓库不具备完整运行环境。

现有音乐 proxy 默认解释器为 `/opt/venv_music/bin/python`，sampler 没有 `--music_text_python` 参数。新联合 checkpoint 需要同步适配加载协议后再使用本视频入口。

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
