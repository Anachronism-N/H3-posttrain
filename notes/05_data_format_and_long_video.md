# 训练数据格式要求 & 长视频生成

> 适用模型：Wan2.2-TI2V-5B-Audio
> 最后更新：2026-03-22

---

## 一、训练数据格式要求

### 1.1 目录结构

```
data/dance_dataset/
├── metadata.csv          # 必须文件，索引所有样本
├── videos/
│   ├── dance_001.mp4
│   ├── dance_002.mp4
│   └── ...
└── audios/
    ├── dance_001.wav
    ├── dance_002.wav
    └── ...
```

### 1.2 metadata.csv 格式

**必须包含列**（对应 `--data_file_keys "video,input_audio"`）：

```csv
video,input_audio,text
videos/dance_001.mp4,audios/dance_001.wav,a person dancing to music
videos/dance_002.mp4,audios/dance_002.wav,a person dancing energetically
```

| 列名 | 类型 | 说明 |
|------|------|------|
| `video` | 相对路径 | 相对于 `--dataset_base_path` |
| `input_audio` | 相对路径 | 相对于 `--dataset_base_path` |
| `text` 或 `prompt` | 字符串 | 文本描述 |

- 路径相对于 `--dataset_base_path`（即 `data/dance_dataset`）
- 也支持 `.json` / `.jsonl` 格式

### 1.3 视频要求

| 项目 | 要求 | 说明 |
|------|------|------|
| 格式 | MP4 / AVI / MOV / MKV 等 | imageio 支持的格式 |
| 分辨率 | 任意（会自动裁剪缩放） | 训练目标：480×832 |
| 帧率 | 任意（会自动采样） | 实测源视频 60fps 可用 |
| 帧数 | ≥ 49 帧（建议≥训练帧数） | 不足会自动截断 |
| 时长 | 与音频对齐（秒数相同） | dance_001 实测：11.99s |

**帧数约束**（VAE 压缩比决定）：

```
num_frames % 4 == 1
合法值：1, 5, 9, 13, ..., 49, 53, ..., 81, 121, ...
```

训练用 `num_frames=49`（即 12×4+1），对应 VAE 时间维度压缩后 13 帧。

### 1.4 音频要求

| 项目 | 要求 | 说明 |
|------|------|------|
| 格式 | WAV / MP3 / FLAC 等 | librosa 支持的格式 |
| 声道 | 单声道（librosa 自动转换） | 立体声会被混合 |
| 采样率 | 任意（LoadAudio 重采样至 16kHz，MERT 内部再升至 24kHz） | 原始 24kHz 最佳 |
| 时长 | 与视频对齐 | 不对齐不报错，但训练效果差 |

**音频预处理链**：

```
wav 文件
  → librosa.load(sr=16000)    # 单声道 float32 数组，(T,)
  → MERT 内部 resample 至 24kHz
  → MERT-v1-330M 提取特征      # (25层, T_30fps, 1024)
  → CausalAudioEncoder 压缩    # (T, 4, 2048) + (T, 1, 2048)
```

### 1.5 数据加载流程（代码路径）

```python
# diffsynth/core/data/operators.py
LoadVideo(num_frames=49, time_division_factor=4, time_division_remainder=1)
  → imageio 打开视频
  → 均匀采样 49 帧索引（按帧率换算）
  → ImageCropAndResize(height=480, width=832) 中心裁剪+缩放
  → 返回 List[PIL.Image]

LoadAudio(sr=16000)
  → librosa.load(path, sr=16000)
  → 返回 np.ndarray shape=(T,)
```

---

## 二、推理支持生成多长的视频

### 2.1 默认与可选帧数

| 场景 | num_frames | 时长（@15fps）| VAE latent 帧 |
|------|-----------|-------------|--------------|
| 训练验证 | 49 | ~3.3s | 13 |
| 标准推理 | 81 | ~5.4s | 21 |
| 最长标准 | **121** | **~8s** | **31** |

合法 num_frames 公式：`k × 4 + 1`（k=0,1,2,...）

### 2.2 实际限制来源

没有硬编码的帧数上限，限制来自两方面：

**① 显存（主要瓶颈）**：

DiT self-attention 复杂度 O(n²)，n = latent_frames × (H/8) × (W/8)

| num_frames | latent 序列长度 (480×832) | 估算显存（单卡）|
|-----------|------------------------|--------------|
| 49 | 13 × 60 × 104 = 81,120 | ~30 GB |
| 121 | 31 × 60 × 104 = 193,440 | ~55 GB |
| 241 | 61 × 60 × 104 = 380,640 | >80 GB（OOM）|

**② RoPE 位置编码预计算范围**：模型权重训练时使用固定长度，超出训练分布会降低质量。

---

## 三、长视频生成可能性

### 3.1 TemporalTiler（已实现，推荐方案）

**位置**：`diffsynth/pipelines/wan_video.py`，`TemporalTiler_BCTHW`

**原理**：滑动窗口 + 重叠区余弦加权混合（与音频 overlap-add 同理）

```
窗口大小 = 31 latent帧（对应 121 video帧）
步长    = 15 latent帧（50% 重叠）

时间轴示意：
[==窗口1==|---重叠---|
               |---重叠---|==窗口2==|---重叠---|
                                        |---重叠---|==窗口3==]

重叠区混合权重：
  窗口1淡出: [1.00, 0.97, ..., 0.03]
  窗口2淡入: [0.03, 0.06, ..., 1.00]
  混合后无缝过渡
```

**调用方式**（推理时传入参数）：

```python
pipe(
    ...,
    num_frames=241,               # 总帧数
    sliding_window_size=31,       # 每窗口 latent 帧数
    sliding_window_stride=15,     # 步长
)
```

**理论上可生成任意长度视频**，显存恒定（每次只处理一个窗口）。

**已知问题**：
- 若音频与视频的帧对齐在窗口切换处不连续，可能产生节奏感丢失
- 建议 `sliding_window_stride` ≤ `sliding_window_size` 的 60%

### 3.2 音频同步对长视频的额外挑战

当前 `WanVideoUnit_Audio` 将整段音频一次性处理：

```python
audio_embeds = pipe.mert_audio_encoder.get_audio_feats_per_inference(
    input_audio, audio_sample_rate,
    batch_frames=num_frames - 1,
    fps=16
)
```

长视频时需要：
1. 将音频按窗口切分（与 `sliding_window_stride` 对齐）
2. 每个窗口单独提取对应时段的 `audio_embeds`
3. 保证窗口边界处音频帧索引连续

**目前代码尚未实现音频感知的 TemporalTiler**，如果要生成长视频+音频同步，需要手动处理音频分段。

### 3.3 方案对比

| 方案 | 显存 | 视频质量 | 音频同步 | 实现难度 |
|------|------|---------|---------|---------|
| 直接增大 num_frames | 随帧数增大 | 超过训练分布后变差 | 原生支持 | 无需改动 |
| TemporalTiler（无音频）| 恒定 | 窗口内质量好，边界可能有轻微抖动 | 不支持 | 直接用 |
| TemporalTiler + 音频分段 | 恒定 | 好 | 需对齐 | 需改造 Pipeline |
| FLF2V（首尾帧控制）| 中等 | 运动连续性好 | 不适用此模型 | 不适用 |

### 3.4 实用建议

- **≤121帧（~8秒）**：直接推理，效果最好
- **121~481帧（~8~32秒）**：使用 `TemporalTiler`，sliding_window_size=31, stride=15
- **>481帧（>32秒）**：先生成音乐节拍点，按节拍切段再拼接，效果优于纯滑窗

---

## 四、关键代码位置

| 功能 | 文件 | 行号参考 |
|------|------|---------|
| LoadVideo | `diffsynth/core/data/operators.py` | ~149 |
| LoadAudio | `diffsynth/core/data/operators.py` | ~248 |
| TemporalTiler | `diffsynth/pipelines/wan_video.py` | ~1138 |
| WanVideoUnit_Audio（音频分段处理）| `diffsynth/pipelines/wan_video.py` | ~814 |
| 帧数约束检查 | `diffsynth/pipelines/wan_video.py` | check_resize_height_width |
