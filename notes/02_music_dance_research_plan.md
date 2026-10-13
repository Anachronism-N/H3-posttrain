# 音乐→舞蹈视频：音频编码器改造研究计划

> 目标：将 Wan2.2-S2V-14B 的音频编码器从语音模型（wav2vec2-xlsr）替换为音乐感知编码器，
> 用于**音乐驱动舞蹈视频生成**（dance-music alignment）
>
> 硬件约束：单张 H100 80GB
> 最后更新：2026-03-12

---

## 一、问题诊断：为什么 wav2vec2-xlsr 不适合

| 维度 | wav2vec2-large-xlsr-53 | 需求 |
|------|----------------------|------|
| 训练数据 | 56k h 多语言**语音** | 音乐（节拍/旋律/节奏） |
| 内部表征 | 音素边界（~10-50ms 精度） | 节拍时序（~500ms@120BPM） |
| 对节奏的敏感性 | 弱（CTC 对齐优化） | 强（舞蹈以节拍为单位） |
| 对音色/和声的编码 | 无（语音任务不需要） | 需要（不同乐器驱动不同动作） |

**核心问题**：用语音模型提取音乐特征，等于用色盲来判断颜色——维度对但信息错位。

---

## 二、编码器选型

### 推荐方案 A：MERT-v1-330M（Phase 1 首选）

```
模型：m-a-p/MERT-v1-330M
架构：HuBERT-Large，24 transformer 层
hidden_size：1024  ← 与 audio_dim=1024 完全匹配 ✓
输出层数：25（1 input + 24 transformer）← 与 wav2vec2-large 完全匹配 ✓
训练数据：160k 小时音乐（MERT 论文）
输出帧率：~75fps（24kHz/320stride）← 需修改 linear_interpolation 的 input_fps
```

**代码改动量**：极小。仅需修改 `wav2vec.py` 的模型加载和 `input_fps` 参数。

### 增强方案 B：MERT + 显式节拍特征（Phase 2 叠加）

在 Phase 1 验证通过后，叠加 `BeatEncoder`：
- 用 `madmom`（精确）或 `librosa.beat_track`（轻量）提取 beat onset 时序
- 编码为 per-frame 节拍强度向量（30fps，与插值后音频帧对齐）
- 小 MLP 投影到 128 维，与 MERT 加权特征拼接后投回 1024 维
- 节拍信息作为"时钟信号"显式告知模型当前帧是否为节拍点

### 备选方案 C：music2vec（facebook/music2vec-v1）

- Data2vec 结构，12 层，hidden_size=768（需额外投影到 1024）
- 参数量更小，但深度不足，不推荐作为主方案

---

## 三、需要重点阅读/修改的文件

### 必须修改

| 文件 | 修改内容 | 优先级 |
|------|---------|--------|
| `diffsynth/models/wav2vec.py` | 替换 Wav2Vec2ForCTC → MERTModel；修改 input_fps=75 | ★★★ |
| `diffsynth/utils/state_dict_converters/wans2v_audio_encoder.py` | 增加 MERT 权重转换器 | ★★★ |
| `diffsynth/configs/model_configs.py` | 注册新的 MERT 音频编码器 config | ★★ |

### 必须深读（不修改）

| 文件 | 学习要点 |
|------|---------|
| `diffsynth/models/wan_video_dit_s2v.py` | CausalAudioEncoder 输入格式；after_transformer_block 注入逻辑 |
| `diffsynth/models/wav2vec.py` | extract_audio_feat / get_audio_embed_bucket_fps 的完整流程 |
| `diffsynth/pipelines/wan_video.py` | WanVideoUnit_S2V：音频如何在推理时被预处理和传入模型 |
| `examples/wanvideo/model_training/train.py` | WanTrainingModule：trainable_modules 如何配置 |

### Phase 2 新增文件

| 文件 | 内容 |
|------|------|
| `diffsynth/models/beat_encoder.py` | BeatEncoder：onset 检测 → 帧级特征 → MLP 投影 |
| `diffsynth/models/music_audio_encoder.py` | 组合 MERT + BeatEncoder 的统一接口 |

---

## 四、实施步骤（分阶段）

### Phase 0：环境与基线（1-2天）

- [ ] 安装 `transformers>=4.39`（MERT 支持版本）
- [ ] 下载 `m-a-p/MERT-v1-330M` 权重
- [ ] 跑通 Wan2.2-S2V-14B 原版推理 `examples/wanvideo/model_inference/Wan2.2-S2V-14B_multi_clips.py`
- [ ] 记录基线推理速度和显存占用

### Phase 1：编码器替换（3-5天）

**Step 1.1：理解现有接口**
- 仔细读 `WanS2VAudioEncoder.extract_audio_feat()` 的输入输出格式
- 特别注意：输出 shape `(25, T, 1024)` 和 `input_fps=50` 这两个参数

**Step 1.2：实现 MERTAudioEncoder**

在 `diffsynth/models/wav2vec.py` 中新增类（保留原类不删）：

```python
class MERTMusicEncoder(torch.nn.Module):
    """Drop-in replacement for WanS2VAudioEncoder using MERT-v1-330M.

    关键差异：
    - 输入采样率：24kHz（MERT）vs 16kHz（wav2vec2）
    - 输出帧率：~75fps（24kHz/320stride）vs ~50fps（16kHz/320stride）
    - hidden_size=1024，num_layers=24 → 25 hidden states（完全相同）
    """
    def __init__(self):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        self.video_rate = 30
        self.audio_fps = 75  # 24000 / 320

    def extract_audio_feat(self, input_audio, sample_rate, processor, ...):
        # 重采样到 24kHz（MERT 要求）
        # 提取所有 hidden states（25层）
        # linear_interpolation(feat, input_fps=75, output_fps=self.video_rate)
        ...
```

**Step 1.3：验证接口兼容性**
- 写单元测试：输出 shape 必须与原 `WanS2VAudioEncoder` 完全一致
- 测试 `get_audio_embed_bucket_fps` 对新编码器输出的处理

**Step 1.4：state_dict 转换器**
- 读 `wans2v_audio_encoder.py` 了解现有格式
- 为 MERT 权重写新的 converter

**Step 1.5：端到端验证**
- 加载替换后的编码器，跑推理
- 目标：模型能正常生成（不崩溃），后续通过训练提升质量

### Phase 2：数据准备（并行进行）

**数据集需求**
- 音乐视频（含舞蹈）：至少 1000 小时以获得好效果，100h 可作验证
- 格式：视频+配套音频，已同步对齐
- 推荐数据集：AIST++、FineDance、TikTok-Dance

**数据预处理管道**
- 提取视频帧（16fps，分辨率 480p）
- 提取音频（单声道，24kHz for MERT）
- 用 VAE 预计算 latents 并缓存（节省训练时显存）
- JSON 格式索引：`{"video_path": ..., "audio_path": ..., "duration": ...}`

### Phase 3：训练（1-2周）

**训练配置（H100 80GB）**

```yaml
# 推荐的训练策略（第一轮）
trainable_modules:
  - wans2v_audio_encoder  # MERT，全量微调
  - casual_audio_encoder  # CausalAudioEncoder，全量
  - audio_injector        # AudioInjector_WAN，全量

freeze_modules:
  - transformer_blocks    # DiT backbone ~14B，冻结

batch_size: 1
gradient_accumulation: 4
learning_rate: 1e-4 (audio encoder), 1e-5 (injector)
dtype: bf16
gradient_checkpointing: true
```

**显存估算**
| 组件 | 显存 |
|------|------|
| 冻结 DiT (14B, BF16) | ~28 GB |
| 可训练参数 (~3.5B, BF16) | ~7 GB |
| 优化器状态 (Adam, FP32) | ~14 GB |
| 激活值 (gradient_checkpointing) | ~8 GB |
| VAE + T5 + 批次数据 | ~8 GB |
| **合计** | **~65 GB** |

→ H100 80GB 可行，有约 15GB 余量可增大 batch_size 或分辨率。

### Phase 4：节拍增强（可选，Phase 1 验证后）

- 实现 `BeatEncoder`
- 修改 `CausalAudioEncoder` 接受可选 beat_features 参数
- ablation study：MERT-only vs MERT+Beat

---

## 五、评估指标

| 指标 | 工具 | 说明 |
|------|------|------|
| Beat Alignment Score (BAS) | 自实现 | 生成视频动作峰值与音频节拍的时序相关性 |
| FrechetVideo Distance (FVD) | 开源实现 | 生成视频质量 |
| 定性评估 | 人工标注 | 节拍是否对齐，动作是否自然 |

---

## 六、关键风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| MERT 帧率不是 75fps | 中 | 实测：`output = model(input); print(len(output.hidden_states[0][0]))` 对已知时长音频验证 |
| CausalAudioEncoder 对新特征分布不适应 | 高 | 训练时 MERT 也要微调，不能冻结 |
| 音乐-舞蹈数据量不足 | 高 | 用数据增强：变速（0.8-1.2x）、pitch shift |
| AudioInjector 训练不稳定 | 中 | 降低 lr，warm-up 1000 steps，zero 初始化残差 |
