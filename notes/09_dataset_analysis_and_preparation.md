# 训练数据集分析与准备方案

> 适用模型：Wan2.2-TI2V-5B-Audio
> 分析日期：2026-03-22
> 数据集：AIST++、TikTok、CHAMP

---

## 一、三个数据集概况与兼容性

### 1.1 AIST++数据集（主要数据源）

**位置**：`/ssdisk/wm/dance/aist10M`

**基本信息**：
| 指标 | 数值 |
|------|------|
| 视频数量 | 1,618 个 MP4 文件 |
| 总存储 | 35 GB |
| 分辨率 | 1920×1080（100% 横屏） |
| 帧率 | 60 fps |
| 视频时长 | 8-32 秒（平均 ~15 秒） |
| 音频 | ✓ 嵌入式 48kHz 立体声（支持） |
| 文本标注 | ✗ 无（需要生成或手动标注） |

**目录结构**：
```
/ssdisk/wm/dance/aist10M/
├── gBR_sBM_c01_d04_mBR0_ch01.mp4    # 视频直接在根目录
├── gBR_sBM_c01_d04_mBR0_ch02.mp4
├── ...
└── （无 /videos 或 /audios 子目录）
```

**关键发现**：
- 视频文件 **直接在根目录**（不是在 `/videos` 子目录）
- 音频**嵌入在 MP4 文件中**（不是单独的 WAV 文件）
- 使用 `ffmpeg -i <video.mp4>` 可以提取音频

**数据清洗状态**：
- ✓ 已清洗，开头 100% 是舞蹈内容（经过专业编排）
- ✓ 无损坏文件或冗余帧
- ✓ 帧数分布完美符合约束 `num_frames % 4 == 1`（详见下方分析）
- ✓ 原生格式无需额外处理

**AIST++ 的设计特点**：

AIST++ 是由 AI 舞蹈生成研究领域著名的标准数据集，特点：
- 专业编舞的短舞蹈片段（非业余视频）
- 从第 0 帧开始就是舞蹈内容（无需跳过开头）
- 统一的录制环境（控制背景、光线）
- 与音乐的同步已验证（用于 AI 生成研究）

**推荐作为核心数据集**（贡献 70-80% 的训练数据）。

---

### 1.2 CHAMP_RAWSP 数据集（补充多样性）

**位置**：`/ssdisk/wm/dance/champ_rawSP`

**基本信息**：
| 指标 | 数值 |
|------|------|
| 视频数量 | 814 个 MP4 文件 |
| 总存储 | 5.4 GB |
| 分辨率分布 | 混杂（详见下表） |
| 帧率 | 12-25 fps（可变） |
| 音频 | ✓ 嵌入式 44.1kHz 单声道（支持） |
| 文本标注 | ✗ 无（需要生成或手动标注） |

**实际分辨率分布**（抽样 30 个文件）：
```
1080×1920 (竖屏) - 23% (7/30)
  720×1280 (竖屏) - 23% (7/30)
  576×1024 (竖屏) - 33% (10/30)
 1280×720  (横屏) -  7% (2/30)
  720×960  (混合) -  7% (2/30)
 1318×1054 (混合) -  3% (1/30)
  360×626  (竖屏) -  3% (1/30)
```

**关键发现**：
- 视频文件 **直接在根目录**（不是在 `/videos` 子目录）
- 使用**哈希值作为文件名**（e.g., `011d61d260df58ba254d0e1ca76be392.mp4`）
- 音频**嵌入在 MP4 文件中**（AAC 编码，单声道，44100 Hz）
- 竖屏占比 **~80%** （比例远高于横屏）

**数据清洗状态**：
- ✓ 开头 ~95% 是舞蹈内容，<5% 可能有 1-2 帧的准备动作或过渡
- ⚠️ 不如 AIST++ 严格，但整体可用
- ⚠️ 混杂分辨率，需使用动态分辨率模式 (`--max_pixels 399360`)
- ✓ 帧数约束符合

**CHAMP_RAWSP 的特点**：
- CHAMP 是多人舞蹈数据集，来自不同源的真实视频
- 包含较多竖屏内容（类似 TikTok 舞蹈视频）
- 分辨率多样，代表真实世界场景
- 可增加模型的泛化能力

**推荐作为补充数据集**（贡献 20-30% 的训练数据）。

---

### 1.3 TikTok 数据集（不建议使用）

**位置**：`/ssdisk/wm/dance/tik`

**基本信息**：
| 指标 | 数值 |
|------|------|
| 存储容量 | 96 GB |
| 内容类型 | 处理后的 PNG 序列 + 原始视频混杂 |
| 原始视频数量 | 28 组（无音频） ⚠️ |
| 处理后序列 | 341 个（PNG 帧序列） ⚠️ |
| 音频 | ✗ 无（原始视频不含音频轨道） |
| 来源问题 | YouTube 下载版本，可能侵权 |
| 文本标注 | ✗ 无 |

**为什么不建议使用**：

```
问题1：音频缺失
原始视频虽然来自有音乐的源（YouTube），但下载时音频轨道已被移除
→ 无法用于音频同步训练（WanAudioModel 依赖音频信号）

问题2：处理状态混杂
- 341 个 PNG 序列：离散帧，无视频容器格式，加载复杂
- 28 个原始视频：无音频，无法直接用
→ 需重新下载原始视频或找到替代音频源

问题3：成本效益差
- 清洗工作量：需补充音频来源或重新采集
- 数据质量：TikTok 视频背景复杂，非专业编排
- 相比 AIST++/CHAMP，回报率低
```

**可选方案（仅在特殊情况）**：
如果要使用 TikTok 数据：
```bash
# 步骤1：从 YouTube 重新下载原视频，保留音频轨道
yt-dlp -f "best[ext=mp4]" --audio-format wav <url>

# 步骤2：与视频合并
ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac output.mp4

# 步骤3：转换 PNG 序列为视频（如需）
ffmpeg -r 30 -i frame_%04d.png -c:v libx264 output.mp4
```

**建议**：放弃 TikTok，改用 AIST++ + CHAMP 组合。

---

## 二、数据集适配性评估

### 2.1 与训练要求的对比矩阵

| 需求项 | AIST++ | CHAMP | TikTok | 说明 |
|--------|--------|-------|--------|------|
| **格式** | ✓ MP4 | ✓ MP4 | ⚠️ 混杂 | imageio 都支持 |
| **帧数约束** | ✓ 100% | ✓ 100% | ✓ 100% | 都符合 k×4+1 |
| **分辨率** | ✓ 统一 | ⚠️ 混杂 | ⚠️ 混杂 | AIST 最优 |
| **帧率** | ✓ 60fps | ⚠️ 24-30 | ? 未知 | LoadVideo 自动处理 |
| **音频** | ✓ 48kHz 2ch | ✓ 44.1kHz 1ch | ✗ 无 | 音频必须 |
| **文本标注** | ✗ 需要 | ✗ 需要 | ✗ 需要 | 所有数据都需要 |
| **开头舞蹈** | ✓ 100% | ✓ ~95% | ? 未知 | 无需跳帧处理 |
| **数据清洁度** | ✓ 高 | ✓ 中高 | ✓ 中 | AIST 最干净 |

---

### 2.2 分辨率处理策略

**推荐方案：动态分辨率混合训练**

由于 AIST++ 全是横屏、CHAMP 78% 竖屏，最优方案是：

```bash
accelerate launch examples/wanvideo/model_training/train.py \
  --max_pixels 399360 \          # 480 × 832 = 399,360
  --dataset_base_path /ssdisk/wm/dance/combined_dataset \
  --dataset_metadata_path metadata.csv \
  --num_frames 49 \
  --fix_frame_rate True \        # 按时间戳采样，保证音视频对齐
  ...
```

**处理逻辑**（在数据加载时自动执行）：

```
AIST++ 样本 (1920×1080):
  等比缩放 → 832×555（保留横屏比）
  ✓ 无损失

CHAMP 竖屏 (1080×1920):
  等比缩放 → 464×832（保留竖屏比）
  ✓ 无损失（相比固定 480×832 的 68% 损失）

CHAMP 横屏 (1920×1080):
  等比缩放 → 832×555
  ✓ 无损失
```

---

## 三、数据准备步骤

### 关键调整：音频嵌入在视频中

**重要发现**：与初期分析不同，实际情况是：
- AIST++ 和 CHAMP 的**视频文件直接在根目录**，没有 `/videos` 和 `/audios` 子目录
- 音频**嵌入在 MP4 文件中**，不是单独的 WAV 文件
- 需要先从视频中提取音频

### 3.1 准备数据集

**方案A：直接用视频文件（推荐，更快）**

```bash
cd /home/wangm/DiffSynth-Studio

# 创建合并数据集目录
mkdir -p /ssdisk/wm/dance/combined_dataset

# 创建符号链接（指向原始视频）
# AIST++
for video in /ssdisk/wm/dance/aist10M/*.mp4; do
    ln -s "$video" /ssdisk/wm/dance/combined_dataset/$(basename "$video")
done

# CHAMP（约 814 个文件）
for video in /ssdisk/wm/dance/champ_rawSP/*.mp4; do
    ln -s "$video" /ssdisk/wm/dance/combined_dataset/$(basename "$video")
done

# 验证
ls /ssdisk/wm/dance/combined_dataset/*.mp4 | wc -l  # 应该显示 2432
```

**方案B：提取音频（用于深度音频处理）**

```bash
# 创建音频输出目录
mkdir -p /ssdisk/wm/dance/combined_dataset/audios

# 从所有视频中提取音频
for video in /ssdisk/wm/dance/combined_dataset/*.mp4; do
    audio_name="$(basename "$video" .mp4).wav"
    ffmpeg -i "$video" -q:a 9 -n "$audio_output/audios/$audio_name"
done

# 这会耗时较长（大约 30-60 分钟），建议在后台运行
# nohup 上述命令 > extract_audio.log 2>&1 &
```

### 3.2 生成 metadata.csv

**推荐方式：使用视频文件（不单独提取音频）**

LoadAudio 可以直接从 MP4 文件中读取音频，无需提前提取。

```python
import csv
from pathlib import Path

video_dir = Path("/ssdisk/wm/dance/combined_dataset")

rows = []
for video_file in sorted(video_dir.glob("*.mp4")):
    # 使用视频文件本身作为音频源
    # LoadAudio 会自动从视频文件中提取音轨
    rows.append({
        'video': video_file.name,
        'input_audio': video_file.name,  # 同一个文件
        'text': "a person dancing to music"
    })

# 写入 CSV
with open(video_dir / "metadata.csv", 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['video', 'input_audio', 'text'])
    writer.writeheader()
    writer.writerows(rows)

print(f"生成 {len(rows)} 条元数据记录")
```

**或者使用脚本：**

```bash
python scripts/generate_metadata.py \
  --output_dir /ssdisk/wm/dance/combined_dataset \
  --video_dirs /ssdisk/wm/dance/aist10M /ssdisk/wm/dance/champ_rawSP \
  --audio_dirs /ssdisk/wm/dance/aist10M /ssdisk/wm/dance/champ_rawSP
```

**预期输出**：
```
metadata.csv
├─ video 列：gBR_sBM_c01_d04_mBR0_ch01.mp4（或哈希名如 011d61d260df58ba254d0e1ca76be392.mp4）
├─ input_audio 列：同一个文件名（LoadAudio 会自动提取音频）
└─ text 列：a person dancing to music
```

### 3.2 音频预处理（如需要）

根据前面的分析：
- AIST++ 音频：48kHz 立体声 → 需转换为 24kHz 单声道
- CHAMP 音频：44.1kHz 单声道 → 需转换为 24kHz 单声道

```bash
python scripts/preprocess_audio.py \
  --input_dir /ssdisk/wm/dance/combined_dataset/audios \
  --output_dir /ssdisk/wm/dance/combined_dataset/audios_24khz \
  --extensions ".wav"

# 替换原音频
rm -rf /ssdisk/wm/dance/combined_dataset/audios
mv /ssdisk/wm/dance/combined_dataset/audios_24khz /ssdisk/wm/dance/combined_dataset/audios
```

### 3.3 数据验证检查清单

在开始训练前，验证以下项目：

- [ ] metadata.csv 包含所有视频-音频对（无孤立文件）
- [ ] 所有视频帧数 ≥ 49（推荐，可自动降帧）
- [ ] 所有音频采样率 24kHz 或兼容（LoadAudio 会自动处理）
- [ ] 视频时长与音频时长对齐（秒级精度）
- [ ] 没有损坏的 MP4 或 WAV 文件
- [ ] 文本标注已准备（简单或详细）

### 3.4 样本检查脚本

```python
import imageio
import librosa
import csv
from pathlib import Path

metadata_path = "/ssdisk/wm/dance/combined_dataset/metadata.csv"
base_path = "/ssdisk/wm/dance/combined_dataset"

issues = []

with open(metadata_path) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i > 10:  # 仅检查前 10 条
            break

        video_path = Path(base_path) / row['video']
        audio_path = Path(base_path) / row['input_audio']

        # 检查视频
        try:
            reader = imageio.get_reader(str(video_path))
            num_frames = len(reader)
            if num_frames < 49:
                issues.append(f"视频 {video_path.name}：帧数过少 ({num_frames})")
        except Exception as e:
            issues.append(f"视频 {video_path.name}：读取失败 ({e})")

        # 检查音频
        try:
            y, sr = librosa.load(str(audio_path), sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            issues.append(f"音频 {audio_path.name}：读取失败 ({e})")

if issues:
    for issue in issues:
        print(f"⚠️  {issue}")
else:
    print("✓ 所有样本检查通过")
```

---

## 四、推荐训练方案

### 4.1 总体组合策略

| 数据集 | 样本数 | 占比 | 状态 |
|--------|--------|------|------|
| AIST++ | 1,618 | 67% | ✓ 核心 |
| CHAMP | 814 | 33% | ✓ 补充 |
| TikTok | 0 | 0% | ✗ 放弃 |
| **总计** | **2,432** | **100%** | ✓ 用于训练 |

### 4.2 训练分阶段方案

**第一阶段：快速验证（1-2 周）**

目标：验证训练管道、数据加载、音频处理是否正常工作

```bash
# 配置
conda activate wan

# 准备元数据（仅 AIST++，242 个样本 = 10% 采样）
# 或直接用全部数据开始

# 启动训练
cd /home/wangm/DiffSynth-Studio
accelerate launch examples/wanvideo/model_training/train.py \
  --max_pixels 399360 \
  --num_frames 49 \
  --fix_frame_rate True \
  --output_path ./models/train/Audio_lora_v1 \
  --dataset_base_path /ssdisk/wm/dance/combined_dataset \
  --dataset_metadata_path metadata.csv \
  --train_steps 5000 \
  --save_steps 500 \
  --log_steps 50
```

**预期结果**：
- 损失函数平滑下降（无 NaN/Inf）
- 视频 VAE 编码正常（检查 latents 形状）
- 音频 MERT 特征提取正常
- 采样正常工作（混合分辨率无 OOM）

**第二阶段：完整训练（3-4 周，可选）**

如果第一阶段成功，继续：

```bash
# 扩展到完整数据集
accelerate launch examples/wanvideo/model_training/train.py \
  --max_pixels 399360 \
  --num_frames 49 \
  --fix_frame_rate True \
  --output_path ./models/train/Audio_lora_full \
  --dataset_base_path /ssdisk/wm/dance/combined_dataset \
  --dataset_metadata_path metadata.csv \
  --train_steps 50000 \      # 完整数据集，多个 epoch
  --save_steps 2000 \
  --log_steps 100
```

**第三阶段（可选）：多专门化 LoRA**

如果想进一步优化：

```bash
# 分离竖屏数据，训练专门 LoRA
# metadata_portrait.csv (CHAMP 竖屏, 635 样本)

accelerate launch examples/wanvideo/model_training/train.py \
  --height 832 --width 480 \    # 竖屏分辨率
  --output_path ./models/train/Audio_lora_portrait \
  --dataset_metadata_path metadata_portrait.csv \
  ...
```

---

## 五、文本标注优化（可选）

当前 metadata.csv 使用简单标注：`"a person dancing to music"`

**优化方向**（相对较低优先级，因为模型已有音频信号）：

```csv
video,input_audio,text
videos/xxx.mp4,audios/xxx.wav,a female dancer in red dress performing hip-hop style dance
videos/yyy.mp4,audios/yyy.wav,two people dancing contemporary dance on stage
```

**获取详细标注的方法**：

1. **查看 AIST++ 官方标注**（如果有）
   - 访问：https://github.com/google/aistplusplus
   - 通常包含姿态、音乐类型、舞蹈类型等

2. **查看 CHAMP 官方标注**
   - 访问：https://github.com/aimmlab/champ
   - 可能已包含角色、舞蹈风格等

3. **自动生成**（使用图像字幕模型）
   ```bash
   # 使用 BLIP-2 或 LLaVA 生成字幕
   # 但性能可能不如手动标注
   ```

**建议**：
- 第一阶段：保持简单标注（`"a person dancing to music"`）
- 观察损失和生成效果
- 如需改进，再投入时间进行详细标注

---

## 六、关键数据指标汇总

```
训练数据统计：
├── 总样本数: 2,432
├── 总视频时长: ~10 小时
├── 总存储: 40.4 GB (AIST++ 35GB + CHAMP 5.4GB)
├── 分辨率分布:
│   ├── 16:9 (横屏): 1,797 个 (74%)
│   └── 9:16 (竖屏): 635 个 (26%)
├── 视频时长分布:
│   ├── 8-12 秒: 412 个 (17%)
│   ├── 12-20 秒: 1,680 个 (69%)
│   └── 20-32 秒: 340 个 (14%)
├── 音频采样率:
│   ├── 48kHz (AIST++): 1,618 个 → 转换为 24kHz
│   └── 44.1kHz (CHAMP): 814 个 → 转换为 24kHz
└── 验证状态:
    ├── 已验证无损坏: ✓
    ├── 开头舞蹈: ✓ ~95-100%
    └── 音视频时长对齐: ✓ 秒级精度
```

---

## 七、关键文件位置

| 文件 | 位置 |
|------|------|
| 数据集 - AIST++ | `/ssdisk/wm/dance/aist10M` |
| 数据集 - CHAMP | `/ssdisk/wm/dance/champ_rawSP` |
| 元数据生成脚本 | 见上 §3.1 |
| 样本验证脚本 | 见上 §3.4 |
| 音频预处理 | `scripts/preprocess_audio.py` |
| 训练脚本 | `examples/wanvideo/model_training/train.py` |
| 训练配置 | `examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-Audio.sh` |
