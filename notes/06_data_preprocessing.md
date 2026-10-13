# 训练数据预处理流程

> 适用模型：Wan2.2-TI2V-5B-Audio（及其他 WanVideo 模型）
> 最后更新：2026-03-22

---

## 零、分辨率模式选择与横竖屏混合数据集

### 选择依据

| 情况 | 推荐模式 |
|------|---------|
| 数据集全是同一宽高比 | **固定分辨率** - 简单稳定，收敛快 |
| 横竖屏混合，追求画面完整性 | **动态分辨率** - 保留各自宽高比 |
| 横竖屏混合，追求最佳效果 | **分开训练两套 LoRA** |

### 固定 480×832 对竖屏的影响（不推荐）

竖屏 1080×1920 被强制裁到 480×832 的过程：

```
等比放大使宽度够 832:
  scale = 832/1080 = 0.77
  → 640×1482

中心裁剪到 480×832:
  裁掉上下各 501px
  → 只保留中间 32% 的高度   ← 舞者头/脚都会被裁掉
```

**结论：固定横屏分辨率会严重损失竖屏内容，不可用。**

### 方案一：动态分辨率（推荐，最省事）

去掉 `--height`/`--width`，只设像素上限：

```bash
accelerate launch ... train.py \
  --max_pixels 399360 \    # 480 × 832，控制总像素上限
  # 不传 --height --width
  ...
```

实际输出尺寸（以 max_pixels=399360 为例）：

```
横屏 1920×1080  →  缩放后 832×464（保留横屏比例）
竖屏 1080×1920  →  缩放后 464×832（保留竖屏比例）
```

因为 batch_size=1，两种尺寸在同一训练过程中完全兼容，无需对齐。

**优点**：一套脚本、一套权重，画面无损失
**缺点**：LoRA 同时拟合两种宽高比，若两类数据量悬殊可能偏向多数类

### 方案二：分开训练两套 LoRA（效果最优）

```bash
# LoRA-横屏：景深宽广的舞蹈视频
accelerate launch ... train.py \
  --dataset_metadata_path data/landscape_dataset/metadata.csv \
  --height 480 --width 832 \
  --output_path ./models/train/Audio_lora_landscape \
  ...

# LoRA-竖屏：TikTok 竖屏舞蹈视频
accelerate launch ... train.py \
  --dataset_metadata_path data/portrait_dataset/metadata.csv \
  --height 832 --width 480 \    # 注意：height/width 对调
  --output_path ./models/train/Audio_lora_portrait \
  ...
```

推理时根据目标输出比例选择对应权重。

**优点**：各自在最优分辨率上收敛，效果最好
**缺点**：需维护两套权重

### 合并 metadata 的写法（方案一用）

将两个数据集合并进同一个 `metadata.csv`，直接混合训练：

```csv
video,input_audio,text
landscape/dance_001.mp4,landscape/audio_001.wav,a person dancing
landscape/dance_002.mp4,landscape/audio_002.wav,a person dancing
portrait/tiktok_001.mp4,portrait/audio_001.wav,a person dancing
portrait/tiktok_002.mp4,portrait/audio_002.wav,a person dancing
```

```bash
--dataset_base_path data/mixed_dataset \
--dataset_metadata_path data/mixed_dataset/metadata.csv \
```

### 实际建议

- 竖屏数量 < 横屏 30%：动态分辨率混合训练，省事
- 竖屏数量接近横屏：分开训练两套 LoRA，各自更优
- 快速验证阶段：先动态分辨率混合跑几个 epoch，观察 loss 是否稳定

---

## 一、预处理工具

### 1.1 音频预处理脚本（已提供）

**位置**：`scripts/preprocess_audio.py`

**功能**：将任意格式音频批量转为 MERT 所需的 24kHz 单声道 WAV。

```bash
python scripts/preprocess_audio.py \
  --input_dir /path/to/raw_audio \
  --output_dir data/dance_dataset/audios \
  --extensions ".mp3,.wav,.flac,.m4a"
```

**处理逻辑**：
1. 多声道 → 均值合并为单声道（`waveform.mean(dim=0)`）
2. 任意采样率 → 重采样至 24kHz（`torchaudio.functional.resample`）
3. 保留原始相对目录结构，输出 `.wav`

### 1.2 视频预处理（无独立脚本）

视频预处理**内嵌在数据加载管道**中，训练时实时完成，无需离线处理：
- 格式：MP4 / AVI / MOV / MKV / FLV / WebM / GIF 均支持
- 缩放裁剪：由 `ImageCropAndResize` operator 实时处理

---

## 二、49帧是固定要求吗？

**不是**。49帧是 `Wan2.2-TI2V-5B-Audio.sh` 的训练配置，可通过 `--num_frames` 自由修改。

**核心约束（由 VAE 时间压缩比决定）**：

```
num_frames % 4 == 1
合法值：1, 5, 9, 13, ..., 49, ..., 81, 121, ...
```

**短视频自动适配**（`FrameSamplerByRateMixin.get_num_frames`）：

```python
def get_num_frames(self, reader):
    num_frames = self.num_frames          # 目标帧数，如 49
    total_frames = 视频实际总帧数
    if total_frames < num_frames:
        num_frames = total_frames
        while num_frames > 1 and num_frames % 4 != 1:
            num_frames -= 1               # 自动向下取最近合法值
    return num_frames
```

| 视频实际帧数 | 目标 num_frames=49 | 实际采样帧数 |
|------------|------------------|------------|
| ≥ 49 | 49 | 49 |
| 45 | 49 | 45 → 自动降至 **45**（45%4=1 ✓）|
| 40 | 49 | 40 → 自动降至 **37**（37%4=1 ✓）|
| 10 | 49 | 10 → 自动降至 **9**（9%4=1 ✓）|

**不同时长视频混合训练**：完全支持，每条样本独立采样，互不干扰。

---

## 三、不同尺寸视频怎么处理

### 3.1 两种模式

**模式 A：固定分辨率**（当前训练脚本用法）

```bash
--height 480 --width 832
```

所有视频统一缩放裁剪至 480×832，算法：**等比放大 + 中心裁剪**

```
原始: 1920×1080 (16:9)
目标: 480×832

step1: 计算放大比例 = max(832/1920, 480/1080) = max(0.43, 0.44) = 0.44
step2: 等比放大 → 856×480
step3: 中心裁剪 → 480×832（裁掉两侧各 12px）
```

不会拉伸，不会填充黑边，会裁掉超出目标宽高比的部分。

**模式 B：动态分辨率**（不传 `--height`/`--width`）

```bash
--max_pixels 524288    # 默认 512×1024 = 524288
# 不传 --height --width
```

```python
# ImageCropAndResize.get_height_width()
if 视频像素数 > max_pixels:
    scale = sqrt(像素数 / max_pixels)
    height, width = int(h/scale), int(w/scale)   # 等比缩小
height = height // 16 * 16   # 对齐到 16 的倍数（模型要求）
width  = width  // 16 * 16
```

不同视频保留各自宽高比，只保证不超过像素上限且对齐 16px。

### 3.2 分辨率对训练的影响

**无分辨率分桶（Bucketing）**：数据加载器 `collate_fn=lambda x: x[0]`，每步只处理 1 条样本，不需要同批次对齐尺寸。

| 训练模式 | 每步 batch size | 分辨率要求 |
|---------|--------------|---------|
| 固定分辨率 | 1 | 统一 height×width |
| 动态分辨率 | 1 | 各样本独立，只需整除 16 |

---

## 四、完整数据处理管道

```
metadata.csv 中的一行样本
│
├── video 列（相对路径）
│   ↓ ToAbsolutePath(dataset_base_path)
│   ↓ RouteByExtensionName
│   │   ├── [jpg/png/webp] → LoadImage → ImageCropAndResize → ToList
│   │   ├── [gif]          → LoadGIF(frame_processor=ImageCropAndResize)
│   │   └── [mp4/avi/...]  → LoadVideo(frame_processor=ImageCropAndResize)
│   └── → List[PIL.Image]，形状对齐到 height×width
│
└── input_audio 列（相对路径）
    ↓ ToAbsolutePath(dataset_base_path)
    ↓ LoadAudio(sr=16000)
    └── → np.ndarray, shape=(T,)，16kHz 单声道

↓ 进入训练 Forward
video  → VAE.encode()        → latents
audio  → MERT + CausalEncoder → audio_embeds
prompt → T5                  → context
↓
Flow Matching Loss
```

### 帧采样模式

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `fix_frame_rate=False` | 默认 | 按帧索引均匀采样（不考虑时间戳）|
| `fix_frame_rate=True` | 可选 | 按时间戳采样（保证 24fps 均匀时间间隔）|

推荐对舞蹈视频使用 `fix_frame_rate=True`，确保音视频时间轴对齐。

---

## 五、数据准备检查清单

在开始训练前，确认以下项：

- [ ] 音频已通过 `scripts/preprocess_audio.py` 转为 24kHz 单声道 WAV
- [ ] `metadata.csv` 包含 `video`, `input_audio`, `text` 三列
- [ ] 路径为相对于 `--dataset_base_path` 的相对路径
- [ ] 视频时长与音频时长匹配（秒级对齐）
- [ ] 视频帧数 ≥ `num_frames`（不足会自动降帧，但建议 ≥ 49）
- [ ] 固定分辨率模式下，视频原始分辨率 ≥ 训练分辨率（480×832）

---

## 六、关键代码位置

| 功能 | 文件 | 类/函数 |
|------|------|--------|
| 中心裁剪+缩放 | `diffsynth/core/data/operators.py:69` | `ImageCropAndResize` |
| 帧率适配采样 | `diffsynth/core/data/operators.py:110` | `FrameSamplerByRateMixin` |
| 视频加载 | `diffsynth/core/data/operators.py:149` | `LoadVideo` |
| 音频加载 | `diffsynth/core/data/operators.py:248` | `LoadAudio` |
| 数据管道组装 | `examples/wanvideo/model_training/train.py:144` | `default_video_operator` |
| 音频格式转换 | `scripts/preprocess_audio.py` | `convert_to_24khz_mono` |
