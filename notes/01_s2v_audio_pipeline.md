# Wan S2V 音频管道深度解析

> 适用模型：Wan2.2-S2V-14B
> 最后更新：2026-03-12

---

## 1. 整体数据流

```
原始音频 (16kHz PCM)
    │
    ▼
WanS2VAudioEncoder（wav2vec.py）
    ├─ Wav2Vec2ForCTC 提取 25 层 hidden states
    ├─ 每层 shape: (1, T_audio, 1024)，帧率约 50fps
    └─ linear_interpolation: 50fps → 30fps
    输出: (25, T_30fps, 1024)
    │
    ▼
get_audio_embed_bucket_fps(fps=16)
    ├─ 将 30fps 音频对齐到 16fps 视频帧
    ├─ 每帧对应 scale=30/16≈1.875 个音频帧
    └─ 支持 temporal context window m（默认 m=0）
    输出: (25, N_video_frames, 1024)  →  送入模型时 reshape 为 batch
    │
    ▼
CausalAudioEncoder（wan_video_dit_s2v.py）
    ├─ 可学习层权重 self.weights: (1, 25, 1, 1)，初始化 0.01
    ├─ 加权平均 25 层 → (batch, N_frames, 1024)
    └─ MotionEncoder_tc（3 层因果卷积，stride 1→2→2）
         ├─ local 分支: (batch, N_frames, 4, dim=5120)  ← 每帧 4 个 token
         └─ global 分支（need_global=True 时）: (batch, N_frames, 1, dim)
    输出: audio_emb (b, T, 4, 5120), audio_emb_global (b, T, 1, 5120)
    │
    ▼
cal_audio_emb()  在 WanS2VModel.forward() 中调用
    ├─ 在序列头部 prepend 73 帧（填充第 0 帧）用于 motion warm-up
    └─ 裁掉前 19 帧后输出对齐后的 audio_emb
    │
    ▼
after_transformer_block()  在 40 个 DiTBlock 的 12 个位置被调用
    注入层: [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39]
    ├─ 对视频特征做 AdaLayerNorm（用 audio_emb_global[:, 0] 调制）
    ├─ CrossAttention: Q=视频特征, KV=音频 token（4个/帧）
    └─ 残差连接加回原视频特征
```

---

## 2. 关键类说明

### WanS2VAudioEncoder（wav2vec.py）

```python
model = Wav2Vec2ForCTC(config)  # facebook/wav2vec2-large-xlsr-53
# hidden_size=1024, num_hidden_layers=24 → 25 hidden states
# Conv strides=[5,2,2,2,2,2,2] → 累积 stride=320 → 16000/320=50fps
```

### CausalAudioEncoder（wan_video_dit_s2v.py:321）

- 输入：`(batch, num_layers=25, dim=1024, video_length)` （注意 time 维在最后）
- `self.weights`：shape `(1, 25, 1, 1)`，对 25 层做 softmax-like 加权
- `MotionEncoder_tc`：3 层因果卷积，stride 总共 ×4 降采样，产生 multi-scale 特征
- 输出：`(batch, T_frames, num_heads=4, out_dim=5120)`

### AudioInjector_WAN（wan_video_dit_s2v.py:281）

- 在指定 12 层每层放置一个 `CrossAttention` 模块（维度 5120）
- `AdaLayerNorm`：用 global 音频 token 对视频 query 做自适应归一化
- **所有注入层共享相同接口，但参数独立**

---

## 3. 维度速查表

| 变量 | Shape | 说明 |
|------|-------|------|
| wav2vec2 hidden states | (25, T_audio@50fps, 1024) | 25 层输出 |
| 插值后 | (25, T_audio@30fps, 1024) | 对齐视频帧率 |
| bucket 采样后 | (25, N_video_frames, 1024) | 每视频帧一条音频特征 |
| CausalAudioEncoder 输入 | (B, 25, 1024, N_frames) | permute 后 |
| 层加权后 | (B, N_frames, 1024) | 25 层融合 |
| MotionEncoder 输出 | (B, N_frames/4, 4, 5120) | 时序下采样 4x |
| audio_emb（注入用） | (B, N_frames/4, 4, 5120) | KV for cross-attn |
| audio_emb_global | (B, N_frames/4, 1, 5120) | AdaLN 调制信号 |

---

## 4. 训练时可训练的参数范围

冻结 DiT backbone（~14B），**仅训练以下模块**：

| 模块 | 参数量估算 | 说明 |
|------|-----------|------|
| WanS2VAudioEncoder（MERT） | ~330M | 替换后的音乐编码器 |
| CausalAudioEncoder.weights | ~25个标量 | 层加权系数 |
| MotionEncoder_tc | ~5M | 因果卷积模块 |
| AudioInjector_WAN（12层 CrossAttn） | ~3.2B | dim=5120 的 12 个注意力 |
| **合计** | **~3.5B** | **H100 80GB 完全可行** |

---

## 5. 调试要点

- `cal_audio_emb` 中 `motion_frames=[73, 19]`：prepend 73 帧是为了让因果卷积有足够 context，裁掉 19 帧是补偿 MotionEncoder_tc 的 stride=4 延迟
- `m=0`（默认）：每帧只看当前帧音频；可增大 m 引入前后文，但增加计算量
- AudioInjector 的 `injected_block_id` 是 dict 而非 list，通过 DFS 扫描 blocks 确定注入位置
