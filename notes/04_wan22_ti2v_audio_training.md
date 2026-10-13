# Wan2.2-TI2V-5B Audio 模型结构与训练流程

> 适用脚本：`examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-Audio.sh`
> 最后更新：2026-03-22

---

## 整体架构图

```
Input
├── video frames (480×832, 49帧)  → VAE encode → latents (16, 13, 60, 104)
├── prompt text                   → UM T5-XXL  → context (512, 768)
└── audio (wav, 16kHz)            → MERT-v1-330M → (25层, T_30fps, 1024)
                                                       ↓
                                              CausalAudioEncoder
                                       ┌──────────────┴──────────────┐
                                  audio_emb_global              audio_emb
                                  (T, 1, 2048) AdaIN用       (T, 4, 2048) 交叉注意力用
                                                    ↓
                              WanAudioModel (28-block DiT)
                                   ╔═══════════════╗
                              [0] ─╢ audio inject  ╟─ block 0
                                   ╠═══════════════╣
                              [4] ─╢ audio inject  ╟─ block 4
                                   ╠═══════════════╣
                              [8] ─╢ audio inject  ╟─ block 8
                                   ╠═══════════════╣
                             [12] ─╢ audio inject  ╟─ block 12
                                   ╠═══════════════╣
                             [16] ─╢ audio inject  ╟─ block 16
                                   ╠═══════════════╣
                             [20] ─╢ audio inject  ╟─ block 20
                                   ╠═══════════════╣
                             [24] ─╢ audio inject  ╟─ block 24
                                   ╠═══════════════╣
                             [27] ─╢ audio inject  ╟─ block 27
                                   ╚═══════════════╝
                                          ↓
                                     noise pred → Flow Matching Loss
```

---

## 一、模型组件详解

### 1.1 WanAudioModel（核心 DiT）

继承自 `WanModel`，共 28 个 `DiTBlock`，每块包含：
- **SelfAttention** + 3D RoPE（时空位置编码）
- **CrossAttention**（文本条件）
- **FFN** + 时间步调制（AdaLN-Zero）

在 8 个注入点（blocks 0, 4, 8, 12, 16, 20, 24, 27）额外插入音频旁路：

```
每个注入点结构：
hidden_states (B, F×H×W, C)
    ↓ rearrange → (B×T, H×W, C)         # 拆成逐帧
    ↓ AdaLayerNorm(audio_emb_global[:, t]) # 音频 AdaIN 调制
    ↓ CrossAttention(Q=视频特征, K/V=audio_emb[:, t])  # 4个音频token
    ↓ residual +
    ↓ rearrange → (B, T×H×W, C)         # 合回去
```

### 1.2 CausalAudioEncoder（音频压缩）

```
(25, T, 1024)   # MERT 25层输出
    ↓ 可学习权重加权平均
(1, T, 1024)
    ↓ MotionEncoder_tc（3级因果 Conv1d，步长2）
audio_emb_global: (T, 1, 2048)    # 全局，给 AdaIN
audio_emb:        (T, 4, 2048)    # 逐帧，给 CrossAttn（4个token/帧）
```

### 1.3 MERT 音频编码器

- 模型：`MERT-v1-330M`（音乐感知音频编码器，训练于 160k 小时音乐）
- 输入：24kHz 单声道音频
- 输出：25层隐状态 × T_30fps × 1024
- 帧率转换：75fps → 30fps（线性插值）
- 训练时**完全冻结**，仅提取特征

### 1.4 其他组件

| 组件 | 规格 | 状态 |
|------|------|------|
| 文本编码器 | UM T5-XXL | 冻结 |
| VAE | WanVideoVAE，压缩比 8×（空间）/ 4×（时间） | 冻结 |
| MERT | MERT-v1-330M | 冻结 |

---

## 二、训练流程

### 2.1 数据流

```
metadata.csv
├── video 列  → LoadVideo → [PIL.Image × 49帧]
└── input_audio 列 → LoadAudio(sr=16000) → np.ndarray
```

**元数据格式**：`metadata.csv` 需含 `video,input_audio` 两列（对应 `--data_file_keys`）。

### 2.2 Forward Pass（单步）

```python
# 阶段1：特征提取
audio → MERT.extract_audio_feat()  → (25, T_30fps, 1024)   # frozen
     → CausalAudioEncoder()         → audio_emb, audio_emb_global  # trainable

video → VAE.encode()               → latents_clean  # frozen

prompt → T5.encode()               → context        # frozen

# 阶段2：扩散噪声
timestep = Uniform(min_t, max_t)
latents_noisy = scheduler.add_noise(latents_clean, noise, timestep)
target        = scheduler.training_target(latents_clean, noise, timestep)

# 阶段3：DiT 前向
noise_pred = WanAudioModel(
    latents_noisy,
    context=context,
    audio_input=audio_emb,
    timestep=timestep
)

# 阶段4：损失
loss = MSE(noise_pred, target) * weight(timestep)  # FlowMatchSFTLoss
```

### 2.3 可训练 vs 冻结模块

| 模块 | 状态 | 说明 |
|------|------|------|
| UM T5-XXL | **冻结** | 不参与反传 |
| WanVideoVAE | **冻结** | 只做编解码 |
| MERT-v1-330M | **冻结** | 只提取特征 |
| DiT blocks q/k/v/o | **LoRA 可训练** | rank=32，含所有块 |
| AudioInjector CrossAttention | **可训练** | 新增的音频旁路层 |
| CausalAudioEncoder | **可训练** | 含可学习层加权权重 |

### 2.4 LoRA 配置

```
lora_rank=32
lora_target_modules=q,k,v,o   # 所有注意力投影
lora_base_model=dit
```

LoRA 加到 DiT 全部 28 块的 `q/k/v/o`，**包括 AudioInjector 内的 CrossAttention 层**。

---

## 三、关键维度

```
视频 latent 各阶段形状：
  输入视频:      (1, 3, 49, 480, 832)
  VAE 编码后:    (1, 16, 13, 60, 104)
  Patchify 后:   (1, ~81k, 2048)

音频 embedding：
  MERT 原始:     (25, ~250, 1024)
  加权合并:      (1, ~250, 1024)
  CausalAudioEnc: audio_emb (T, 4, 2048) / audio_emb_global (T, 1, 2048)

文本 context:   (1, 512, 768) → embedding 后 (1, 512, 2048)
时间调制:       每块 (1, 6, 2048)
```

---

## 四、音频注入位置设计

注入位置均匀分布于 28 层 [0, 4, 8, 12, 16, 20, 24, 27]：

- **浅层（0, 4）**：影响全局结构和运动节奏
- **中层（8, 12, 16, 20）**：影响内容与音频的对应关系
- **深层（24, 27）**：影响细节纹理与音频的精细对齐

每帧用 **4个 token** 表示音频（类似 IP-Adapter），比可变长序列更稳定，参数更紧凑。

---

## 五、关键文件索引

| 文件 | 作用 |
|------|------|
| `examples/wanvideo/model_training/train.py` | 训练入口，`WanTrainingModule` |
| `diffsynth/models/wan_video_dit_audio.py` | `WanAudioModel`，音频注入定义 |
| `diffsynth/models/wan_video_dit.py` | 基础 `WanModel`，`DiTBlock` |
| `diffsynth/models/mert_audio_encoder.py` | MERT 特征提取 |
| `diffsynth/pipelines/wan_video.py` | `WanVideoUnit_Audio` 处理单元 |
| `diffsynth/diffusion/loss.py` | `FlowMatchSFTLoss` |
| `diffsynth/trainers/training_module.py` | LoRA 添加与训练工具 |
