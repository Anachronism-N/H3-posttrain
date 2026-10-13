# 指标说明报告：2D-MM Align、AV-Align、Wav2CLIP-CS

## 1. 报告目的

本文说明 3 个常用于音乐驱动视频或舞蹈视频评测的指标：

- `2D-MM Align`：动作与音乐节拍对齐程度
- `AV-Align`：音频瞬态与视频运动峰值同步程度
- `Wav2CLIP-CS`：音频语义与视频语义一致性

这 3 个指标关注点不同：

- `2D-MM Align` 更关注“是否踩 beat”
- `AV-Align` 更关注“音视频事件是否同步”
- `Wav2CLIP-CS` 更关注“语义/风格是否匹配”

因此，它们是互补关系，不应互相替代。

---

## 2. 总体结论

### 2.1 指标性质

| 指标 | 本质 | 趋势 | 典型范围 | 是否越大越好 |
|---|---|---|---|---|
| `2D-MM Align` | 动作峰到最近音乐 beat 的高斯对齐分数 | 节拍对齐 | `[0, 1]` | 是 |
| `AV-Align` | 音频 onset 与视频运动峰的命中率/匹配率 | 事件同步 | `[0, 1]` | 是 |
| `Wav2CLIP-CS` | Wav2CLIP 音频嵌入与 CLIP 视频嵌入余弦相似度 | 语义一致性 | `[-1, 1]` | 通常是 |

### 2.2 使用建议

- 如果任务是“音乐驱动舞蹈/动作生成”，首看 `2D-MM Align`。
- 如果任务是“音视频同步生成”，补看 `AV-Align`。
- 如果任务还要求“音乐风格与画面语义一致”，再看 `Wav2CLIP-CS`。

### 2.3 与论文实现一致性结论

基于对 DabFusion 论文与当前工具实现的严格逐项比对，可得：

- `2D-MM Align`：当前实现与 DabFusion 论文给出的评分公式一致；但论文未完整公开 2D motion beat 的提取细节，因此更准确的说法是“评分公式一致、峰值提取流程相容”。
- `AV-Align`：当前工具默认实现为 `iou`，并已按 TempoTokens 公开 `av_align.py` 的口径实现；这也是 DabFusion 引用的 AV-Align 来源。
- `Wav2CLIP-CS / CS`：核心定义一致，都是“视频 CLIP 表征与音乐 Wav2CLIP 表征的余弦相似度”；但论文未公开足够细节来严格证明当前聚合方式与作者实现完全相同。

### 2.4 指标参照来源

| 指标 | 当前文档中的含义 | 直接参照来源 | 说明 |
|---|---|---|---|
| `2D-MM Align` | 2D 视频动作峰与音乐 beat 的高斯对齐分数 | DabFusion（WACV 2025） | DabFusion 在文中正式定义该 2D 版本，并说明其灵感来自 AIST++ 的 Beat Alignment Score |
| `AV-Align` | 音频 onset 与视频运动峰的同步分数 | TempoTokens / Yariv et al. 2024 | 当前默认实现对齐公开 `av_align.py` 的 IoU 版本 |
| `Wav2CLIP-CS` | Wav2CLIP 音频向量与 CLIP 视频向量的余弦相似度 | DabFusion 中的 `CLIP-Score (CS)` | 其中音频编码器参照 Wav2CLIP，视频编码器参照 CLIP；当前聚合方式由本工具实现明确化 |

---

## 3. 预处理与公共符号

为理解 3 个指标，先统一定义几个时间序列。

### 3.1 音乐 beat

对音频 `y` 使用 `librosa.beat.beat_track` 提取节拍帧，再转成时间戳：

\[
B^{m} = \{t^{m}_{j}\}_{j=1}^{M}
\]

其中 `B^m` 表示音乐 beat 集合，单位为秒。

在当前工具实现中，对应 [audio_video_align.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/audio_video_align.py:14)。

### 3.2 音频 onset

对音频先计算 onset strength，再用 `librosa.onset.onset_detect` 找瞬态峰：

\[
O^{a} = \{t^{a}_{i}\}_{i=1}^{A}
\]

其中 `O^a` 表示音频 onset 时间集合，单位为秒。

在当前工具实现中，对应 [audio_video_align.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/audio_video_align.py:23)。

### 3.3 视频运动峰

当前代码中，“视频运动峰”并不是所有指标共用同一套提取方式，而是分成两种口径。

第一种用于 `2D-MM Align`，以及 `AV-Align` 的旧版 `symmetric_hit_rate`。对相邻视频帧计算 Farneback 光流，取光流幅值均值作为每一帧间的运动能量：

\[
e_k = \frac{1}{|\Omega|}\sum_{p \in \Omega}\left\|\mathbf{f}_k(p)\right\|_2
\]

其中：

- `k` 是帧间索引
- `\Omega` 是图像像素集合
- `\mathbf{f}_k(p)` 是第 `k` 个帧间在像素 `p` 处的光流向量

然后对能量曲线做滑动平均平滑，再用 `find_peaks` 找局部峰值，得到：

\[
P^{v} = \{t^{v}_{i}\}_{i=1}^{V}
\]

其中 `P^v` 表示视频运动峰时间集合，单位为秒。

第二种用于当前默认的 `AV-Align` 官方口径。代码会先构造 TempoTokens 风格的光流轨迹 `q_i`，再把满足局部极大且幅值不小于 `0.1` 的位置记为峰值：

\[
V^{peak} = \left\{\frac{i}{\mathrm{fps}} \,\middle|\, q_{i-1} < q_i > q_{i+1},\ q_i \ge 0.1 \right\}
\]

这一版本不使用平滑核与 `prominence` 阈值，而是直接复现 TempoTokens 公开脚本中的局部极大值逻辑。

在当前工具实现中，对应：

- 光流能量：[audio_video_align.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/audio_video_align.py:38)
- 平滑：[audio_video_align.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/audio_video_align.py:66)
- 峰值提取：[audio_video_align.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/audio_video_align.py:73)
- 官方 AV 峰值提取：[audio_video_align.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/audio_video_align.py:86)

---

## 4. 2D-MM Align

### 4.1 指标定义

`2D-MM Align` 衡量每个“视频动作峰”与其“最近音乐 beat”的时间距离，并用高斯函数把距离映射为 `[0, 1]` 分数，再对所有动作峰求平均。

设：

- `P^v = {t_i^v}` 为视频运动峰
- `B^m = {t_j^m}` 为音乐 beat
- `\sigma > 0` 为时间容忍参数，单位秒

则分数定义为：

\[
\mathrm{MMAlign}
=
\frac{1}{|P^v|}
\sum_{t_i^v \in P^v}
\exp\left(
-\frac{
\min_{t_j^m \in B^m}|t_i^v - t_j^m|^2
}{2\sigma^2}
\right)
\]

这与 DabFusion 论文中的定义一致。更进一步地，它与 AIST++ 系列工作中常见的 Beat Alignment Score 是同型指标，只是这里的“动作峰”来自 2D 视频光流峰值，而不是 3D 骨架动力学 beat，因此命名为 `2D-MM Align` 更准确。

当前工具实现见 [audio_video_align.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/audio_video_align.py:94)。

### 4.2 公式直觉

对于任意一个动作峰 `t_i^v`：

- 如果它与最近音乐 beat 完全重合，距离 `d=0`，该项得分为 `1`
- 如果偏差越大，得分越接近 `0`

等价地，可写成：

\[
d_i = \min_{t_j^m \in B^m}|t_i^v - t_j^m|
\]

\[
s_i = \exp\left(-\frac{d_i^2}{2\sigma^2}\right)
\]

\[
\mathrm{MMAlign} = \frac{1}{|P^v|}\sum_i s_i
\]

### 4.3 取值范围与解释

- 理论范围：`[0, 1]`
- `1` 表示所有动作峰都精确落在音乐 beat 上
- 越接近 `0` 表示动作峰与音乐 beat 越不相关

经验解读可参考：

- `0.80` 以上：踩点较强
- `0.60` 到 `0.80`：有一定踩点能力
- `0.40` 到 `0.60`：中等，存在明显偏移
- `0.40` 以下：对齐较弱

这只是经验区间，不同数据集、节奏密度、镜头运动强弱都会影响绝对值。

### 4.4 `sigma` 的作用

`\sigma` 是时间容忍度，越大越宽松。

- `\sigma` 小：稍微错拍就被明显惩罚
- `\sigma` 大：分数更“宽容”，模型之间差距会被压缩

当前工具默认按“秒”设置 `sigma`，并在结果中记录 `2d_mm_sigma`，实现见 [audio_video_align.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/audio_video_align.py:112)。

因此：

- 横向比较不同模型时，必须固定 `sigma`
- 报告结果时，建议同时写明 `sigma` 数值

### 4.5 如何解读高分与低分

高分通常说明：

- 视频主要动作变化与音乐拍点较一致
- 大动作切换常发生在音乐节拍附近

低分可能来自：

- 动作根本没有跟音乐走
- 动作跟了音乐，但峰值检测不稳定
- 镜头运动、背景变化干扰了光流峰
- 音乐 beat 提取对弱节拍或非强节奏音乐不稳定

### 4.6 局限性

- 它衡量的是“动作峰是否靠近 beat”，不衡量动作是否好看。
- 它不区分“动作类型是否与音乐风格匹配”。
- 它对相机抖动、镜头推进、背景运动比较敏感，因为动作峰来自全帧光流。

---

## 5. AV-Align

### 5.1 指标定义

`AV-Align` 衡量音频事件峰值与视频运动峰值是否在一个容忍窗口内同步出现。

在当前工具中：

- 音频侧使用 `onset`，不是 `beat`
- 默认视频侧使用 TempoTokens 官方口径的光流峰值

因此它衡量的是“事件同步”，而不是严格意义上的“踩拍”。

工具文档说明见 [使用说明.md](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/使用说明.md:9) 与 [使用说明.md](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/使用说明.md:347)。

### 5.2 当前实现与论文参照关系

这里要特别说明：`AV-Align` 这个名称本身并不像 `FVD` 那样是高度标准化的统一 benchmark 名称。对当前仓库而言，它是一个有明确代码口径、并且默认对齐 TempoTokens 官方实现的可复现指标。论文中更严谨的写法是：

> AV-Align（implemented following the public TempoTokens metric）

如果目标是“严格复现 DabFusion 论文中引用的 AV-Align”，当前工具应使用默认 `iou` 版本，这也是现已设置的默认行为。

### 5.3 默认 `iou` 版本

设：

- `O^a = {t_i^a}` 为音频 onset 集合
- `V^{peak} = {t_j^v}` 为视频峰集合
- `\tau` 为时间容忍半径，单位秒
- `M` 为在容忍窗口内完成“一一贪心匹配”后的匹配数

则默认分数定义为：

\[
\mathrm{AVAlign}_{IoU}
=
\frac{M}{|O^a| + |V^{peak}| - M}
\]

其中匹配规则与当前代码完全一致：每个视频峰最多被匹配一次，且只有当

\[
t_j^v - \tau < t_i^a < t_j^v + \tau
\]

时，音频峰 `t_i^a` 才能与视频峰 `t_j^v` 视为命中。

实现见 [audio_video_align.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/audio_video_align.py:139)。

### 5.4 兼容保留的 `symmetric_hit_rate` 版本

工具中仍保留旧版对称命中率实现，便于兼容此前实验。设：

- `O^a = {t_i^a}` 为音频 onset 集合
- `P^v = {t_j^v}` 为视频运动峰集合
- `\tau` 为容忍窗口半径，单位秒

音频到视频命中率定义为：

\[
H_{a\rightarrow v}
=
\frac{1}{|O^a|}
\sum_{t_i^a \in O^a}
\mathbf{1}
\left[
\min_{t_j^v \in P^v}|t_i^a - t_j^v| \le \tau
\right]
\]

视频到音频命中率定义为：

\[
H_{v\rightarrow a}
=
\frac{1}{|P^v|}
\sum_{t_j^v \in P^v}
\mathbf{1}
\left[
\min_{t_i^a \in O^a}|t_j^v - t_i^a| \le \tau
\right]
\]

该兼容版本的分数为两者平均：

\[
\mathrm{AVAlign}_{sym}
=
\frac{1}{2}
\left(
H_{a\rightarrow v}
+
H_{v\rightarrow a}
\right)
\]

当前实现见：

- `_hit_rate`：[audio_video_align.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/audio_video_align.py:129)
- `compute_av_align`：[audio_video_align.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/audio_video_align.py:159)

### 5.5 与论文公式的关系

DabFusion 论文引用的 AV-Align 原始定义来自 Yariv et al. 2024，其公式写为：

\[
\mathrm{AV\mbox{-}Align}
=
\frac{1}{2|A \cup V|}
\left(
\sum_{a \in A}\mathbf{1}[a \in V]
+
\sum_{v \in V}\mathbf{1}[v \in A]
\right)
\]

其中在另一模态的 `3` 帧窗口内存在峰值即可视为命中。

这个公式见原论文：

- Diverse and Aligned Audio-to-Video Generation via Text-to-Video Model Adaptation
- OpenReview PDF: https://openreview.net/pdf?id=R37J4Zeh5o

从当前代码看，有两点需要明确：

1. `symmetric_hit_rate`

\[
\frac{1}{2}\left(H_{a\rightarrow v}+H_{v\rightarrow a}\right)
\]

这与论文公式 **不相同**，因为其分母按各自集合大小分别归一化，而不是按 `2|A \cup V|` 归一化。

2. `iou`

\[
\frac{M}{|A|+|V|-M}
\]

这与公开代码的实现口径一致。进一步地，TempoTokens 官方脚本采用音频峰到视频峰的贪心一一匹配，并在默认 `window_frames = 3` 时使用 `\pm 1` 帧容忍区间。因此，当前工具默认的 `iou` 版本可以视为已按论文公开代码实现对齐。

### 5.6 容忍窗口

当前代码并不是直接把 `window_frames` 除以 `fps`，而是先把总窗口换算成“半径帧数”：

\[
\text{radius\_frames} = \left\lfloor \frac{\text{window\_frames}}{2} \right\rfloor
\]

\[
\tau = \frac{\text{radius\_frames}}{\text{fps}}
\]

默认 `window_frames = 3` 时，有 `\text{radius\_frames}=1`，因此实际容忍区间是 `\tau = 1/\mathrm{fps}`，也就是“以峰值为中心前后各 `1` 帧”，总窗口共 `3` 帧。实现见 [audio_video_align.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/audio_video_align.py:172)。

这意味着：

- 视频帧率越高，同样 3 帧对应的时间窗口越小
- 不同 fps 的视频比较时，必须确认窗口换算后的时间容忍度是否一致

### 5.7 取值范围与解释

- 理论范围：`[0, 1]`
- 越接近 `1`，表示音频瞬态与视频运动峰越同步
- 越接近 `0`，表示两者同步性越弱

解读时要注意：

- 高 `AV-Align` 不一定意味着“踩拍好”，因为 onset 比 beat 更密、更细粒度
- 一个视频可能 `AV-Align` 高，但 `2D-MM Align` 一般
- 反之，一个视频可能踩拍不错，但并没有对齐大量细碎 onset

### 5.8 如何看两个子项

工具还输出：

- `audio_to_video_hit_rate`
- `video_to_audio_hit_rate`

它们很有解释力：

- `audio_to_video_hit_rate` 高、`video_to_audio_hit_rate` 低：说明大部分音频事件都能在视频里找到响应，但视频里还有许多“多余动作峰”
- `audio_to_video_hit_rate` 低、`video_to_audio_hit_rate` 高：说明视频动作峰数量少，只覆盖了部分音频事件
- 两者都高：同步较好
- 两者都低：同步较差

这两个子项在 `iou` 默认版本下也会同时输出，但它们只是辅助解释量，真正写入主指标的 `av_align` 分数仍是前述 IoU。

### 5.9 局限性

- onset 检测对打击乐、鼓点清晰的音乐更友好。
- 对抒情、长音、弱节奏音乐，onset 本身可能不稳定。
- 视频峰来自光流，因此仍会受镜头运动与背景干扰。
- 它主要衡量“同步事件密度”，不直接评价动作语义是否正确。

---

## 6. Wav2CLIP-CS

### 6.1 指标定义

`Wav2CLIP-CS` 是音频嵌入与视频嵌入的余弦相似度。

其中：

- 音频嵌入来自 `Wav2CLIP`
- 视频嵌入来自 `CLIP` 图像编码器对若干采样帧的特征平均

因此它不是时序同步指标，而是跨模态语义一致性指标。

当前实现见 [wav2clip_score.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/wav2clip_score.py:147)。

从 DabFusion 论文原文表述看：

> “we use the CLIP-Score (CS) to measure the cosine similarity between CLIP embeddings of the dance video and Wav2CLIP embeddings of the music”

因此在“余弦相似度”这一核心定义上，当前实现与论文描述一致。更准确地说，当前文档中的 `Wav2CLIP-CS` 对应的是 DabFusion 中写作 `CLIP-Score (CS)` 的那一项，只是这里把“音频端使用 Wav2CLIP、视频端使用 CLIP”的实现细节明确展开了。

### 6.2 视频嵌入计算

设视频中均匀采样 `N` 帧，记第 `k` 帧图像为 `I_k`，CLIP 图像编码器为 `f_{clip}`。当前代码默认 `N=8`。

先对每帧编码并归一化：

\[
\mathbf{z}_k
=
\frac{f_{clip}(I_k)}{\|f_{clip}(I_k)\|_2}
\]

再对 `N` 帧平均并再次归一化，得到视频向量：

\[
\mathbf{v}
=
\frac{
\frac{1}{N}\sum_{k=1}^{N}\mathbf{z}_k
}{
\left\|
\frac{1}{N}\sum_{k=1}^{N}\mathbf{z}_k
\right\|_2
}
\]

实现见 [wav2clip_score.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/wav2clip_score.py:73)。

### 6.3 音频嵌入计算

设音频波形为 `x`，Wav2CLIP 编码器为 `f_{w2c}`。

音频嵌入写作：

\[
\mathbf{a} = f_{w2c}(x)
\]

若模型返回多段或多帧嵌入，则先做平均，再归一化：

\[
\mathbf{a}'
=
\frac{
\frac{1}{T}\sum_{t=1}^{T}\mathbf{a}_t
}{
\left\|
\frac{1}{T}\sum_{t=1}^{T}\mathbf{a}_t
\right\|_2
}
\]

当前工具中的实现逻辑是：

- 将音频重采样为 `16 kHz`
- 调用 `wav2clip.embed_audio`
- 若返回二维张量，则沿时间/片段维平均
- 最终做 `L2` 归一化

实现见 [wav2clip_score.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/wav2clip_score.py:101)。

需要注意的是，DabFusion 论文没有进一步公开以下评测细节：

- 视频到底使用全部帧还是均匀采样帧
- 视频级 CLIP 表征是“先逐帧相似度再平均”，还是“先平均视频特征再算相似度”
- 评测时使用原始 Wav2CLIP 还是论文中 fine-tuned 的 Wav2CLIP

因此，当前实现与论文属于“定义一致、细节未完全可证”的关系，而不是“已证实逐实现一致”。

### 6.4 余弦相似度公式

最终指标为：

\[
\mathrm{Wav2CLIP\mbox{-}CS}
=
\cos(\mathbf{v}, \mathbf{a}')
=
\frac{\mathbf{v}^{\top}\mathbf{a}'}{\|\mathbf{v}\|_2\|\mathbf{a}'\|_2}
\]

由于实现里两侧已归一化，因此数值上等价于点积：

\[
\mathrm{Wav2CLIP\mbox{-}CS} = \mathbf{v}^{\top}\mathbf{a}'
\]

实现返回见 [wav2clip_score.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/wav2clip_score.py:147)。

### 6.5 取值范围与解释

- 理论范围：`[-1, 1]`
- 越大表示音频与视频在共享语义空间中越接近
- 接近 `0` 表示语义相关性弱
- 小于 `0` 表示语义方向明显不一致

在实际生成任务中，这个值常用于相对比较，不建议脱离数据集单独解释绝对值。

### 6.6 高分与低分分别意味着什么

高分通常说明：

- 音乐情绪、风格、能量感与视频内容比较匹配
- 例如激烈音乐配高动感画面，舒缓音乐配平稳画面

低分可能说明：

- 音频和视频语义不搭
- 视频主体和音乐风格差异大
- 采样帧太少，没有覆盖主要画面内容

### 6.7 为什么不能替代对齐指标

`Wav2CLIP-CS` 只描述“跨模态语义是否相近”，不描述“时刻是否同步”。

所以：

- 一个视频可能 `Wav2CLIP-CS` 很高，但动作根本没踩点
- 也可能 `2D-MM Align` 很高，但视频只是机械踩拍，语义不丰富

这也是工具文档中强调“Wav2CLIP-CS 不能替代踩点指标”的原因，见 [使用说明.md](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/使用说明.md:348)。

### 6.8 局限性

- 它受预训练模型偏好影响较大。
- 它对“抽象音乐情绪”往往比对“精确动作事件”更敏感。
- 视频只抽若干帧做平均，弱化了时序信息。

---

## 7. 三个指标之间的区别

| 维度 | `2D-MM Align` | `AV-Align` | `Wav2CLIP-CS` |
|---|---|---|---|
| 比较对象 | 视频动作峰 vs 音乐 beat | 视频动作峰 vs 音频 onset | 视频语义向量 vs 音频语义向量 |
| 关注重点 | 踩拍 | 事件同步 | 语义匹配 |
| 是否含时间容忍窗口 | 是 | 是 | 否 |
| 是否依赖光流峰值 | 是 | 是 | 否 |
| 是否能解释“语义搭不搭” | 否 | 否 | 是 |
| 是否能解释“是否踩点” | 是 | 部分可以 | 否 |

简单说：

- `2D-MM Align` 看“大拍点”
- `AV-Align` 看“事件级同步”
- `Wav2CLIP-CS` 看“语义风格一致性”

---

## 8. 报告中建议如何写结果解释

### 8.1 当 `2D-MM Align` 高、`AV-Align` 低

可解释为：

> 模型能在主要节拍上完成动作响应，但对更细粒度的音频瞬态同步能力有限。

### 8.2 当 `2D-MM Align` 低、`AV-Align` 高

可解释为：

> 视频动作能响应局部音频事件，但整体节拍结构把握较弱，存在“局部同步、全局不踩拍”的现象。

### 8.3 当 `Wav2CLIP-CS` 高、前两者低

可解释为：

> 视频与音乐在风格或语义层面较匹配，但时序同步和动作节拍对齐不足。

### 8.4 当三者都高

可解释为：

> 模型同时具备较好的节拍对齐、事件同步和跨模态语义一致性。

---

## 9. 实际使用中的注意事项

### 9.1 固定参数

横向比较模型时，至少需要固定：

- `2D-MM Align` 的 `sigma`
- `AV-Align` 的 `window_frames` 或等价时间窗口
- 视频 fps
- 视频抽帧数
- CLIP 模型版本

### 9.2 结合可视化解释

建议同时看：

- `beat_motion_timeline.png`
- `alignment_error_hist.png`
- `flow_energy_motion_peaks.png`

当前工具文档中对这些可视化的解释见 [使用说明.md](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/使用说明.md:200)。

其中，对齐误差定义为：

\[
\mathrm{error} = t^{v}_{peak} - t^{m}_{nearest}
\]

若误差分布：

- 集中在 `0` 附近：踩点好
- 偏正：动作滞后音乐
- 偏负：动作提前音乐

相关说明见 [使用说明.md](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/使用说明.md:227)。

### 9.3 谨慎解释绝对值

这些指标非常依赖：

- 音频类型
- 舞蹈密度
- 镜头运动强弱
- 数据集分布
- 预处理参数

因此建议：

- 以同一数据集内的相对比较为主
- 不要跨论文直接比较未对齐配置的绝对值

---

## 10. 推荐的论文/汇报写法

可以直接使用下面这段概述：

> `2D-MM Align` 用于衡量视频动作峰与音乐节拍的对齐程度。该指标对每个动作峰计算其到最近音乐 beat 的时间距离，并通过高斯核映射为 `[0,1]` 范围内的局部分数，最终取平均；当前定义参照 DabFusion，并继承 AIST++ Beat Alignment Score 的基本思想。`AV-Align` 用于衡量音频 onset 与视频运动峰的同步性，本文默认采用与 TempoTokens 公开实现一致的 IoU 版本。`Wav2CLIP-CS` 用于衡量音频与视频在共享语义嵌入空间中的一致性，对应 DabFusion 中的 `CLIP-Score (CS)`，具体实现为 Wav2CLIP 音频向量与 CLIP 视频向量之间的余弦相似度。三者分别从节拍对齐、事件同步和语义匹配三个维度刻画生成结果质量。 

---

## 11. 参考来源

### 11.1 本项目实现

- 指标总说明：[使用说明.md](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/使用说明.md:5)
- `2D-MM Align` / `AV-Align` 实现：[audio_video_align.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/audio_video_align.py:14)
- `Wav2CLIP-CS` 实现：[wav2clip_score.py](/home/wangm/DiffSynth-Studio/eval_vis_toolkit/s2v_eval/wav2clip_score.py:73)

### 11.2 外部文献与项目

- DabFusion（直接定义 `2D-MM Align`，并在实验中使用 `CLIP-Score / AV-Align`）:
  - WACV 2025 paper: https://openaccess.thecvf.com/content/WACV2025/papers/Wang_Dance_Any_Beat_Blending_Beats_with_Visuals_in_Dance_Video_WACV_2025_paper.pdf
- AIST++ / Beat Alignment Score（`2D-MM Align` 的 3D 先验来源）:
  - Paper page: https://huggingface.co/papers/2101.08779
  - arXiv entry: https://arxiv.org/abs/2101.08779
- TempoTokens / AV-Align:
  - OpenReview paper: https://openreview.net/pdf?id=R37J4Zeh5o
  - Official code: https://github.com/guyyariv/TempoTokens/blob/master/av_align.py
- CLIP:
  - Paper: https://arxiv.org/abs/2103.00020
- Wav2CLIP:
  - Official GitHub: https://github.com/descriptinc/lyrebird-wav2clip
  - Paper page: https://huggingface.co/papers/2110.11499
  - ICASSP 2022 entry: https://doi.org/10.1109/ICASSP43922.2022.9747669

---

## 12. 最终建议

如果只保留一句话总结：

- `2D-MM Align` 看“踩没踩拍”
- `AV-Align` 看“音视频事件同步没同步”
- `Wav2CLIP-CS` 看“音乐和画面语义搭不搭”

三者联合报告，比单独报告任何一个指标都更完整。
