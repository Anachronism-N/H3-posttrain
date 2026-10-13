# H3 后训练潜在问题与修复计划

日期：2026-10-14（北京时间）  
审查版本：[df99a0a](https://github.com/Anachronism-N/H3-posttrain/commit/df99a0ac2ba1bea3bc32a3aa78da47745d61899d)  
适用范围：双向舞蹈微调、音乐控制评测、AR 与蒸馏代码，以及实验调度。

当前优先任务是解释并控制双向微调后的画质退化，同时修复会误导音乐评估和实验归因的实现问题。双向老师通过画质验收后，再推进音乐增强、AR 和少步蒸馏。

问题按三类记录：**实现问题**表示在审查版本中可直接定位的行为；**实验风险**表示需要检查运行配置或数据才能判断实际影响；**机制假说**表示现有结果支持的候选解释。代码缺陷成立，不等于它已被证明是所有模糊的原因。画质观察来自项目人评记录；以下数值探针不构成 GPU 训练复现。

## 1 最新证据和当前判断

项目以参考人物图、音乐和编舞文本生成舞蹈视频，进一步探索长视频与少步 AR。H3 音乐条件在 DiT 中是独立音频 token，文本编码器处理引用标签和文字。检查重点应是 packed 时间位置、音频有效区间和条件加载的一致性，而不能把全部问题归为音频借用了文本隐空间。

10 月 13 日新增实验改变了早期归因：

- 清单动作 caption 会使原始 H3 动作僵硬，并使微调 LoRA 训偏。富文本修复已经进入 H-A4 和后续实验。
- 同数据 CFG、助手和 Hybrid 对照仍有退化。继续扩大这些变体的依据不足。
- SG512 使用原始 H3 自生成 latent 作为目标，与 GT512 保持相同 ID 和训练配方。记录中的 SG512100 也出现模糊，因此真实数据降级不是解释该现象的必要条件。
- 降学习率、降低 rank 和限制噪声时间段的当前小样本人评也未解决模糊。现有记录支持优先检查当前 FM 微调链路，但尚不能唯一确定是哪一项机制导致退化。
- 去蒸馏底座与推理真实 CFG 的一致方案已安排诊断，应先评审这些结果，再决定新的训练目标。

来源：[富文本修复与早期实验](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/notes/19_h3_ar_plan_20261008.md#L321)；[10 月 13 日对照和最新人评](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/notes/22_h3_bidir_blur_attribution_20261013.md)。

SG 端点回归与教师速度场蒸馏是不同的训练目标。SG 数据来自底座自身，并不自动保证重新训练的线性 FM 速度场等于底座发布时的速度场。“CFG 蒸馏场被 FM 拉回平均场”仍应保留为机制假说，而不是已证明的定理。

## 2 问题优先级

P0 应在下一轮关键决策或依赖该路径的任务之前处理；P1 应在对应实验恢复前处理；P2 属于质量稳定后的工程工作。

| 编号 | 优先级 | 问题 | 证据状态 |
|---|---|---|---|
| E01 | P0 | H3 姿态与 G2 音乐特征缺少真实时间对齐 | 实现问题，历史分数受影响范围待核验 |
| Q01 | P0 | 双向 FM 微调画质退化，训练目标与发布模型行为可能不一致 | 已有人评现象，具体机制待验证 |
| N01 | P0 | direct 和 inverse CFG 混入梯度尺度及 BF16 精度差异 | 代数和数值问题已确认 |
| C01 | P0 | ODE 路径绕过富文本侧车并漏加载音乐注入器 | 实现问题，仅在复用该路径时触发 |
| O01 | P0 | 评测失败后 claim 不释放，错误记录阻止恢复 | 实现问题 |
| A01 | P1 | 因果 plan 不匹配时静默退回双向注意力 | 实现风险，有明确触发条件 |
| A02 | P1 | AR 音频窗口采用错误的固定 latent 跨度 | 时间计算问题已确认 |
| M01 | P1 | 音频探针没有同步替换 MERT 和 MoE 特征 | 实现问题，仅音乐注入模型受影响 |
| E02 | P1 | 指标口径、判别器划分和小样本评审限制结论强度 | 测量与实验风险 |
| X01 | P1 | 部分旧对照的样本量不同，环境配置未完整保存 | 默认配置下的混杂和复现风险 |
| A03 | P2 | lookahead 不代表严格在线音乐因果，采样缺少 KV cache | 任务定义和工程边界 |

## 3 评估与归因问题

### E01 H3 音画评估时间轴不一致

H3 视频输出为 24 fps。eval_metrics.py 保存实际视频 fps，但 score_disc.py 直接取前 81 个关键点帧；MERT 验证特征则固定为 81 个采样点、16 Hz。motion_factors.py 也按帧索引截取，不读取 fps。

按媒体时长 N/fps 计算，81 帧分别对应 3.375 秒和 5.0625 秒；按采样时间戳 i/fps 计算，首尾跨度分别是 3.333 秒和 5 秒。两种口径都说明时间轴不一致。历史 H3 结果是否经过额外转换，需要核对集群脚本和实际关键点文件，不能直接宣布全部历史指标无效。

**修复：** 依据真实时间戳，将姿态、MERT 与节拍转换到相同评估频率和共同有效区间。保留原始 fps、重采样频率、时长、音频有效长度和有效关键点率。身体速度按真实时间差计算，不使用跨 fps 的每帧位移直接比较。

**验收：** 同一条平滑姿态轨迹分别以 16 Hz 和 24 Hz 采样，转换后得到一致的 G2 输入和姿态距离；同一物理速度的 motion 指标一致。随后重算原始 H3 和关键 aswap 对照。

来源：[视频帧读取](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/eval/eval_metrics.py#L51)；[G2 姿态输入](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/g2/score_disc.py#L19)；[MERT 16 Hz 特征](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/g2/mert_val.py#L21)；[姿态距离截取](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/eval/motion_factors.py#L22)。

### E02 指标不足以单独判定可控性与画质

需要分别处理以下限制：

1. **节拍指标口径不同。** eval_metrics.py 的 Pose-BAS 计算音乐 beat 被速度极大值覆盖的比例；beat_metrics.py 使用速度极小值到 beat 的高斯分数。名称、峰定义和以秒表示的容差必须随结果保存。
2. **检测失败影响运动分数。** NaN 置零可能把关键点检测失败当作低运动。应同时报告有效关键点率，并规定可比较的样本集合。
3. **全图 DINO 不是独立的人脸身份指标。** 当前身份分数包含背景、服装及首帧。建议排除首帧，增加人物裁剪和可用人脸的独立评估；手脸伪影仍需人评。
4. **G2 内部划分存在预处理泄漏。** train_disc.py 先对全部加载数据计算标准化，再随机划分训练和验证。应先划分，再只用训练集统计量。歌曲、舞者或同源片段的实际跨集合重叠尚未核验，应检查后再判断。
5. **换音乐可能与文本冲突。** prompt 若描述原视频逐拍编舞，aswap 负分不能全部归因于音频分支。使用不固定逐拍动作的富文本测试纯音乐控制，再评估音乐与动作文本的组合。
6. **固定前三个 case 适合诊断，不能代表整体效果。** 关键结论应在完整验证集上做同种子配对、多个种子复验和盲评，报告失败率与配对置信区间。

G2 在 GT 上的 AUC 0.906 是项目记录中的初步证据。它不能替代对同歌曲错时间片段、相位偏移、变速及相近 BPM 错配的校准，也不足以直接放行 RL。

**验收：** 指标输出带定义和时间基准；按歌曲与同源视频分组的独立集合完成校准；音乐敏感度和符合新音乐的同步质量分别报告；小样本结果不替代完整验证集结论。

来源：[运动和节拍指标](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/eval/eval_metrics.py#L82)；[另一节拍定义](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/eval/beat_metrics.py)；[全图身份评分](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/eval/eval_metrics.py#L160)；[标准化和随机划分](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/g2/train_disc.py#L77)；[G2 校准记录](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/notes/18_experiment_summary_8node_20261005.md#L109)。

## 4 双向画质和训练数值问题

### Q01 FM 微调后的画质退化需要检查目标与推理的一致性

SG512 与 GT512 的现有观察支持把当前微调过程列为优先嫌疑。SG 目标更容易拟合、训练 loss 更低，并不代表生成画质更好。助手、CFG、低学习率等当前配方都未通过人评，也不能反推所有可能的训练目标必然失败。

**优先诊断：** 先评审已安排的去蒸馏底座无 LoRA 加真实 CFG、保留去蒸馏底座的已训 LoRA 加 CFG、原底座 FM LoRA 加 CFG，以及 SG512 LoRA 加 CFG。保持 case、富文本、种子、分辨率与推理日程一致。先确认原始基线和老师可用，不生成大批蒸馏数据。

若真实 CFG 能稳定恢复质量，可以评估训练和推理均使用去蒸馏底座并在推理施加 CFG 的方案，同时记录额外推理开销。若仍无法恢复，候选方向是冻结老师速度场蒸馏或双向分布匹配；先做小规模画质验证，再迁入 AR。

可以考虑在任务目标旁加入冻结原始老师的速度约束，或在原条件回放样本上保持老师行为。约束必须使用相同噪声状态、时间步和条件，并测量领域适配与原始质量之间的取舍。不能把自生成端点 FM 回归等同于此约束。

**验收：** 固定验证集盲评确认手脸、肢体、身份和运动质量；比较早期与后期 checkpoint；判定依据独立于训练 loss。检查源片、预处理片、VAE 重建和模型生成时，必须对齐真实时间戳。

来源：[SG 与 GT 对照及 CFG 推理诊断](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/notes/22_h3_bidir_blur_attribution_20261013.md)；[已安排的诊断入口](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/diag_cfg_after_train.sh)。

### N01 direct 与 inverse CFG 的尺度和精度不同

设条件预测为 p，无梯度无条件预测为 u，FM 目标为 y，引导尺度为 g。direct 目标 q = u + g(y - u)；inverse 残差为 (p + (g - 1)u)/g - y = (p - q)/g。

在相同输入、预测和 detached u 下，精确算术满足：

    L_direct = g² × L_inverse

g = 3.5 时，loss 和原始参数梯度相差 12.25 倍。这不是两个不同的引导机制。Adam 的归一化会影响实际更新，因此也不能直接称为学习率相差 12.25 倍。

另外，inverse 分支在 BF16 做加法和除法，之后才为 MSE 转 FP32；direct 的目标构造已经使用 FP32。纯 Python BF16 舍入探针取 p = 1.0078125、u = y = 1、g = 3.5，当前 inverse 中间值可能舍入成零残差；FP32 inverse loss 约为 4.98246e-6，direct loss 约为 6.10352e-5。这个例子证明数值行为不同，不能证明它就是实际涂抹的根因。

**修复与验收：** 两种 CFG 代数运算均先转 FP32；比较实现时统一损失归一化，使用同一输入验证 loss 和梯度一致。保留原始、归一化 loss 以及梯度范数，避免由不可比的 loss 推断质量。

来源：[CFG 两种分支](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/diffsynth_patch/files/diffsynth/diffusion/loss.py#L172)。

## 5 条件加载和音乐模块问题

### C01 ODE 与 CD 尚未完整复用富文本和音乐条件

ode_gen.py 直接 torch.load 原缓存，绕过 UnifiedDataset 的富文本侧车处理；加载 LoRA 时还剔除 od_music 权重且未安装对应特征。复用该脚本时，可能用旧清单 prompt，或者把音乐增强 checkpoint 变成缺少注入器的老师。

新 sg_gen.py 已明确复用富文本侧车，并用 SHA256 生成稳定种子。这是 SG 路径的已有改进；旧 ODE 路径没有同步修复。ODE 使用 Python hash 派生种子还会依赖进程哈希设置，应改为稳定哈希或显式种子表。

CD 的 teacher、student、EMA 切换只覆盖其 LoRA 设计，音乐注入器没有完整的冻结老师和 EMA 生命周期。纯 H-A 老师不触发这个音乐组合问题；使用 M1 或 M2 老师前需要补齐。

**修复：** 提取训练、SG、ODE、CD 共用的条件加载函数，统一正负侧车、packed、参考锚点、音乐特征和模块权重。为教师与 EMA 明确冻结或独立复制音乐模块。

**验收：** 同一缓存条目经各入口加载后，prompt、packed 和音乐条件一致；同 seed 的老师预测可复现；教师输出不随学生更新变化；音乐 checkpoint 不会被静默剥离。

来源：[ODE 权重与缓存加载](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/ode_gen.py#L42)；[富文本侧车入口](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/diffsynth_patch/files/diffsynth/core/data/unified_dataset.py#L96)；[SG 侧车加载](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/sg_gen.py#L30)；[CD adapter 生命周期](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/h3_cd.py)。

### M01 音频探针与额外音乐特征可能互相矛盾

sample_h3_ar.py 的 audio_override 替换波形，但 CUR_AUDIO 仍来自原 CSV 路径。MERT、beat 和 CLAP 继续按原音乐加载。如果用 M1 或 M2 checkpoint 测试静音、语音或白噪声，模型仍接收原音乐的额外特征。当前不带音乐注入器的原始 H3 或 H-A4 探针不受这一问题影响。

**修复与验收：** 覆盖音频时同步重新计算全部特征，或明确禁用所有额外音乐特征。记录实际波形与各特征来源；静音探针中不能残留原音乐特征。

音乐模型的比较也应使用相同数据、条件和预算。MoE 均衡项主动鼓励平均 gate 接近均匀，gate 约为 0.2 不能单独证明没有学习。应联合查看梯度、注入幅度、特征消融和校准后的音乐匹配结果。

来源：[额外特征加载](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/sample_h3_ar.py#L103)；[音频覆盖与路径](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/sample_h3_ar.py#L180)；[MoE 均衡项](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/h3_music.py#L89)。

## 6 AR 注意力和任务边界

### A01 因果 plan 不匹配时静默退回双向注意力

h3_causal.py 在设置了 plan 但序列长度或 cu_seqlens 不匹配时，调用原始注意力。缓存、文本长度或 CFG 分支改变后，这可能使名义 AR 使用双向计算。现有稠密 mask 对照只能覆盖当时的输入，不能替代完整流程的因果性检查。

**修复与验收：** 无 plan 才允许使用原始注意力；有 plan 时结构不匹配必须报错。固定前缀、音乐和当前视频块，任意扰动未来视频 latent，当前块预测应保持一致。覆盖正负 CFG 分支、富文本和不同块布局。

来源：[注意力回退条件](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/h3_causal.py#L71)。

### A02 音频窗口少算部分视频 latent 的跨度

视频 VAE 的时间跨度是 (1,4,4,4,4) × 5/3，单位为音频 RoPE 时间坐标，40 个单位对应 1 秒。make_plan 却用 vt[1] - vt[0] 作为所有 latent 的跨度。

对 bounds = (0,5)，真实块尾为 17/24 秒。加 lookahead = 8 后，正确截止时间约为 0.908333 秒，当前计算约为 0.783333 秒，相差 125 ms。完整的 5-latent 对齐块会出现这一偏差；音频末端还会受到有效长度裁剪。

**修复与验收：** 内部块使用下一 latent 起点 vt[f1] 求块尾；最后一块按真实跨度推算，再裁到有效音频范围。验证首块、完整块、合并尾块和不同 lookahead，确保 8 个音频坐标单位确实表示额外 200 ms。

来源：[当前块尾计算](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/h3_causal.py#L59)；[VAE 时间映射](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/diffsynth_patch/files/diffsynth/pipelines/minimax_h3_audio_video.py#L590)。

### A03 lookahead 和 AR 推理速度的适用范围

目标音频 queries 可见整段目标音频，参考音频也在全局前缀。视频虽然只选择局部音频 keys，深层音频表示可能已包含未来音乐。因此当前设计适合整段音乐已知、视频因果的离线任务；lookahead 消融不等于完整系统实际可见的未来音乐长度。

若研究在线音乐输入，还必须约束参考音频、音频内部注意力和 MERT、CLAP 的未来上下文，并以未来音频扰动测试验收。

当前 AR 采样每块仍计算整段序列并丢弃未来结果，最后整体解码，缺少可复用 KV cache 和流式 VAE 输出。质量通过后，再缓存固定文本、参考、音乐和历史 K/V，测量首块延迟、持续生成速度、显存以及缓存前后的一致性。

旧 TF 实验同时使用已被否决的清单 caption。它们不能单独证明 teacher forcing 必然失败。将来恢复 AR 时，先从可用双向老师出发，固定富文本和配方，再区分干净 GT 历史与自生成历史下的质量。

来源：[音频可见范围](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/h3_causal.py#L61)；[逐块重算与最终解码](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/sample_h3_ar.py#L122)；[旧 TF 归因](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/notes/19_h3_ar_plan_20261008.md#L305)。

## 7 调度和实验复现

### O01 失败任务可能永久跳过

eval_sched.sh 在采样前建立 claim，采样后不检查退出码、不释放失败 claim。sg_gen.py 也有认领后异常未清理的路径；SG512 已完成，不应把这个恢复风险误写成该批次实际失败。

eval_metrics.py 恢复时把 error key 也视为 done；summarize.py 丢弃错误记录且不报告失败数。如果传入同 key 的互补指标文件，last-wins 会覆盖而不是合并，不能据此断言已有汇总全部损坏。

**修复与验收：** 分开记录 running、done、failed，保存任务 owner 和开始时间；仅在退出成功且结果完整后标记 done。失败任务释放或续租 claim。指标错误允许重试，汇总报告预期数、成功数、失败数和每项有效数量；同 key 的冲突字段显式报错。模拟 OOM、中断、单样本错误后，任务能够恢复且不会并发重复运行。

来源：[评测 claim](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/eval_sched.sh#L29)；[SG claim](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/sg_gen.py#L65)；[错误记录恢复](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/eval/eval_metrics.py#L29)；[指标去重与错误过滤](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/eval/summarize.py#L17)。

### X01 部分旧对照预算不同且关键配置未保存

默认配置下，H-A4 是 16 卡，H-A5hr 与 H-A6dup 是 8 卡。相同更新步数的样本曝光量不同。最新 notes/22 已明确把同数据 CFG 和 SG512、GT512 对照设为 global batch 8，这是改进；仍应核对实际启动环境，不能把所有新对照继续标为相同的旧问题。

runner.py 仅保存 CLI args；clean 音频、TF、音乐模式、分块、侧车目录、噪声时间段等还依赖环境变量。TRAIN_GPUS 在 train_ha2.sh 中只控制 hold.sh，没有显式设置训练 CUDA 可见设备；选择非默认卡且外部未设置 CUDA_VISIBLE_DEVICES 时，会释放与训练使用不同的卡。

采样器默认仍是 motion prompt 和 shift 5/5，评测 worker 已通过参数覆盖为新设置。直接调用入口时仍可能复用旧条件，应统一入口或要求显式选择配置。

**修复与验收：** 保存经过允许列表过滤的实际环境、global batch、累计样本数、数据 ID、prompt 与特征版本、底座和 adapter 哈希、Git 与上游版本。不要保存凭据。launcher 显式设置并校验 GPU visibility。对照先固定预算；若研究 batch，则把 batch 列为实验变量。

来源：[旧双向和数据变体配置](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/q2_20261010.sh#L155)；[新同数据对照](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/notes/22_h3_bidir_blur_attribution_20261013.md)；[参数保存](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/diffsynth_patch/files/diffsynth/diffusion/runner.py#L20)；[GPU 与累积配置](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/train_ha2.sh#L7)；[采样入口默认值](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/sample_h3_ar.py#L20)。

## 8 不应继续列为已确认根因的事项

- 数据普遍模糊的早期结论已经撤回。SG512 的结果进一步削弱了把真实数据降级作为唯一主因的解释；局部数据质量仍需按时间对齐检查。
- AR 全部涂抹是实验现象，TF 方法本身不可行尚未被排除混杂后的实验证明。
- MERT 或 MoE 无效、均匀 gate 代表没学，均不能由现有观察直接推出。
- Hybrid 在推理中被丢弃不是当前实现问题。biorighyb 明确保留 Hybrid 底座；DeCFG 默认使用原 Ref2VA 加新 LoRA，符合其现有方案。实际部署仍需核对权重标识。
- notes/22 中音乐约 5.06 秒、视频约 5.17 秒的尾部 mask 差异，与 E01 的 G2 时间轴错配是两个独立问题。尾部行为也影响原始基线，不能单独解释微调后才出现的退化；下轮统一处理有效长度时应单独核验。

来源：[数据结论修订](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/notes/19_h3_ar_plan_20261008.md#L363)；[最新归因及音频尾部差异](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/notes/22_h3_bidir_blur_attribution_20261013.md)；[Hybrid 推理入口](https://github.com/Anachronism-N/H3-posttrain/blob/df99a0ac2ba1bea3bc32a3aa78da47745d61899d/scripts_omnidance/exp/h3ar/eval_sched.sh#L44)。

## 9 建议执行顺序和完成条件

| 顺序 | 工作 | 完成条件 |
|---|---|---|
| 1 | 修复评估时间轴、CFG 数值与失败恢复，记录实际配置 | 小型不变量检查通过，失败任务可恢复，核心对照可追溯 |
| 2 | 评审已安排的 CFG 推理诊断，确认双向退化出现在哪个环节 | 原始基线可用，明确老师、底座和推理设置，完整验证集质量通过 |
| 3 | 根据诊断决定继续一致 CFG 方案或小规模教师速度及分布匹配 | 保住手脸、身份与运动质量，不以训练 loss 替代画质 |
| 4 | 同数据验证原生音频与简单音乐注入，再评估 MoE | 校准后的音乐同步和最小干预有效，画质无明显退化 |
| 5 | 恢复 AR 前修复 plan 和时间窗口，贯通条件加载 | 未来视频扰动不影响当前块，自生成历史质量可接受 |
| 6 | 推进少步蒸馏、KV cache 与长视频 | 学生质量达到预设标准，缓存计算一致，长视频漂移和实际延迟可测 |

每项完成后记录修复 commit、实际配置、验证结果和是否需要重算历史指标。优先复用已经产生的 checkpoint 和诊断结果，避免重复启动已被当前证据否决的大批 FM、助手或 Hybrid 变体。
