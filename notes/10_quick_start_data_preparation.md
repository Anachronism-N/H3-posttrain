# 训练数据准备快速开始指南

> 适用模型：Wan2.2-TI2V-5B-Audio
> 时间估计：30-60 分钟（首次）

---

## 🎯 概述

你有三个舞蹈数据集：
- **AIST++** (1,618 视频，35GB) ⭐⭐⭐⭐⭐ **推荐**
- **CHAMP** (814 视频，5.4GB) ⭐⭐⭐⭐ **补充**
- **TikTok** (96GB) ❌ **不推荐**（无音频）

**目标**：构建 AIST++ + CHAMP 的统一数据集（2,432 样本），用于 LoRA 训练。

---

## 📋 准备工作清单

在开始前，确认以下环境已就绪：

```bash
# 1. 激活 conda 环境
conda activate wan

# 2. 检查依赖
python -c "import imageio; import librosa; import csv; print('✓ 所有依赖就绪')"

# 3. 检查数据集是否存在
ls -lh /ssdisk/wm/dance/aist10M /ssdisk/wm/dance/champ_rawSP

# 4. 进入项目目录
cd /home/wangm/DiffSynth-Studio
```

---

## 🚀 执行步骤

### 步骤 1：生成 metadata.csv（5 分钟）

**作用**：扫描所有视频和音频文件，生成训练所需的元数据索引。

```bash
python scripts/generate_metadata.py \
  --output_dir /ssdisk/wm/dance/combined_dataset \
  --video_dirs /ssdisk/wm/dance/aist10M/videos /ssdisk/wm/dance/champ_rawSP/videos \
  --audio_dirs /ssdisk/wm/dance/aist10M/audios /ssdisk/wm/dance/champ_rawSP/audios \
  --default_text "a person dancing to music"
```

**预期输出**：
```
扫描视频文件...
  找到 2432 个视频
扫描音频文件...
  找到 2432 个音频

⚠️  警告：0 个视频缺少对应音频

✓ 生成 2432 条元数据记录
  输出文件：/ssdisk/wm/dance/combined_dataset/metadata.csv

统计信息：
  总样本数：2432
  缺失音频：0
  有效比率：100.0%
```

**检查输出文件**：
```bash
head -5 /ssdisk/wm/dance/combined_dataset/metadata.csv

# 输出：
# video,input_audio,text
# videos/aist_xxx.mp4,audios/aist_xxx.wav,a person dancing to music
# videos/champ_yyy.mp4,audios/champ_yyy.wav,a person dancing to music
```

---

### 步骤 2：验证数据集（10-20 分钟）

**快速验证**（仅检查前 10 个样本）：

```bash
python scripts/validate_dataset.py \
  --metadata_path /ssdisk/wm/dance/combined_dataset/metadata.csv \
  --base_dir /ssdisk/wm/dance/combined_dataset \
  --sample_size 10
```

**预期输出**：
```
📋 正在验证 metadata.csv...
   文件：/ssdisk/wm/dance/combined_dataset/metadata.csv

✓ 样本    1: videos/aist_gBR_aBOI_c02_d17_mS02_ch02.mp4
           帧数=540, 音频=48000Hz, 时长=18.0s
✓ 样本    2: videos/aist_gBR_aBOI_c03_d17_mS02_ch02.mp4
           帧数=540, 音频=48000Hz, 时长=18.0s
...

============================================================
验证结果（样本数：10）
============================================================
✓ 有效样本：10
✗ 无效样本：0

总体成功率：100.0%
```

**完整验证**（所有样本，可选，耗时 5-10 分钟）：

```bash
python scripts/validate_dataset.py \
  --metadata_path /ssdisk/wm/dance/combined_dataset/metadata.csv \
  --base_dir /ssdisk/wm/dance/combined_dataset
```

---

### 步骤 3：音频预处理（10-15 分钟）

**目的**：统一所有音频为 24kHz 单声道 WAV（MERT 的标准输入格式）

```bash
# 创建备份（可选）
cp -r /ssdisk/wm/dance/combined_dataset/audios \
      /ssdisk/wm/dance/combined_dataset/audios_backup

# 预处理
python scripts/preprocess_audio.py \
  --input_dir /ssdisk/wm/dance/combined_dataset/audios \
  --output_dir /ssdisk/wm/dance/combined_dataset/audios_24khz \
  --extensions ".wav"

# 替换原音频
rm -rf /ssdisk/wm/dance/combined_dataset/audios
mv /ssdisk/wm/dance/combined_dataset/audios_24khz \
   /ssdisk/wm/dance/combined_dataset/audios

# 验证
ls -lh /ssdisk/wm/dance/combined_dataset/audios | head -5
```

**预期输出**：
```
total 45G
-rw-r--r-- 1 user staff 8.9M Mar 22 10:00 aist_gBR_aBOI_c02_d17_mS02_ch02.wav
-rw-r--r-- 1 user staff 8.8M Mar 22 10:01 aist_gBR_aBOI_c03_d17_mS02_ch02.wav
...
```

---

### 步骤 4（可选）：验证视频帧数约束

某些视频的帧数可能不符合约束 `num_frames % 4 == 1`。自动降帧脚本：

```bash
# 检查哪些视频需要降帧
python scripts/validate_dataset.py \
  --metadata_path /ssdisk/wm/dance/combined_dataset/metadata.csv \
  --base_dir /ssdisk/wm/dance/combined_dataset \
  --sample_size 100 | grep "帧数约束"
```

如果发现问题，可以使用以下脚本修复（可选）：

```python
import imageio
import os
from pathlib import Path

def fix_video_frames(video_path, output_path):
    """
    检查视频帧数，如需要则重新编码以符合约束
    """
    reader = imageio.get_reader(video_path)
    num_frames = len(reader)

    # 检查是否需要修复
    if num_frames % 4 == 1:
        print(f"✓ {video_path.name} 帧数 {num_frames} 符合约束，跳过")
        return

    # 向下降帧到最近的合法值
    target_frames = num_frames - (num_frames % 4)
    if target_frames < 1:
        target_frames = 1

    print(f"⚠️  {video_path.name}: {num_frames} → {target_frames}")

    # 重新编码（可选）
    # 通常不需要，LoadVideo 会自动采样
```

**一般情况下，LoadVideo 会自动处理，无需手动修复。**

---

## ✅ 最终检查清单

在开始训练前，运行以下检查：

```bash
#!/bin/bash

echo "=== 数据集完整性检查 ==="

base_dir="/ssdisk/wm/dance/combined_dataset"
metadata="$base_dir/metadata.csv"

# 1. 检查 metadata.csv 存在
if [ -f "$metadata" ]; then
    echo "✓ metadata.csv 存在"
    count=$(tail -n +2 "$metadata" | wc -l)
    echo "  样本数：$count"
else
    echo "✗ metadata.csv 不存在"
    exit 1
fi

# 2. 检查视频目录
video_count=$(ls "$base_dir/videos" | wc -l)
echo "✓ 视频数：$video_count"

# 3. 检查音频目录
audio_count=$(ls "$base_dir/audios" | wc -l)
echo "✓ 音频数：$audio_count"

# 4. 检查音视频数量是否匹配
if [ "$video_count" -eq "$audio_count" ]; then
    echo "✓ 音视频数量匹配"
else
    echo "⚠️  音视频数量不匹配 (视频: $video_count, 音频: $audio_count)"
fi

# 5. 检查磁盘空间
space=$(du -sh "$base_dir" | cut -f1)
echo "✓ 总存储：$space"

echo ""
echo "=== 所有检查完成 ==="
```

保存为 `check_dataset.sh` 并执行：

```bash
chmod +x check_dataset.sh
./check_dataset.sh
```

---

## 🎓 理解各个步骤

### 为什么需要 metadata.csv？

```
metadata.csv 是训练的入口文件，格式：
video,input_audio,text
videos/xxx.mp4,audios/xxx.wav,a person dancing

- video 列：视频路径（相对于 base_path）
- input_audio 列：对应的音频路径
- text 列：文本描述（当前简单，可后续优化）

DataLoader 读取 metadata.csv，加载对应的视频和音频。
```

### 为什么需要音频预处理？

```
AIST++: 48kHz 立体声  ——> 需要转 24kHz 单声道
CHAMP:  44.1kHz 单声道 ——> 需要转 24kHz 单声道

MERT（音频编码器）要求 24kHz 输入。
LoadAudio 会在加载时重采样，但提前统一可：
- 加快训练速度（无需运行时转换）
- 减少 GPU 内存占用
- 提高一致性
```

### 为什么不推荐 TikTok 数据？

```
原因 1：音频缺失
  - TikTok 视频没有音频轨道
  - WanAudioModel 依赖音频进行条件化

原因 2：处理状态混乱
  - 既有 PNG 序列，又有原始视频
  - 需要大量人工清洗

原因 3：数据质量
  - TikTok 视频背景复杂，非专业编排
  - 不如 AIST++ 的专业舞蹈编排

推荐：用 AIST++ (70%) + CHAMP (30%) 的组合
```

---

## 🔧 常见问题

### Q1：数据太大，可以只用 AIST++ 吗？

可以，但多样性会降低。命令：

```bash
python scripts/generate_metadata.py \
  --output_dir /ssdisk/wm/dance/aist_only \
  --video_dirs /ssdisk/wm/dance/aist10M/videos \
  --audio_dirs /ssdisk/wm/dance/aist10M/audios
```

这会生成 1,618 个样本（vs 2,432）。

### Q2：训练数据可以用符号链接而不复制吗？

可以，脚本已支持：

```bash
# 创建合并目录
mkdir -p /ssdisk/wm/dance/combined_dataset/{videos,audios}

# 符号链接
ln -s /ssdisk/wm/dance/aist10M/videos/* /ssdisk/wm/dance/combined_dataset/videos/
ln -s /ssdisk/wm/dance/aist10M/audios/* /ssdisk/wm/dance/combined_dataset/audios/
ln -s /ssdisk/wm/dance/champ_rawSP/videos/* /ssdisk/wm/dance/combined_dataset/videos/
ln -s /ssdisk/wm/dance/champ_rawSP/audios/* /ssdisk/wm/dance/combined_dataset/audios/

# 生成 metadata（和往常一样）
python scripts/generate_metadata.py ...
```

### Q3：音频预处理会覆盖原始文件吗？

会，但脚本提供了备份选项：

```bash
# 备份原始音频
cp -r /ssdisk/wm/dance/combined_dataset/audios \
      /ssdisk/wm/dance/combined_dataset/audios_backup

# 预处理（输出到新目录）
python scripts/preprocess_audio.py \
  --input_dir /ssdisk/wm/dance/combined_dataset/audios \
  --output_dir /ssdisk/wm/dance/combined_dataset/audios_24khz

# 手动验证后再替换
mv /ssdisk/wm/dance/combined_dataset/audios \
   /ssdisk/wm/dance/combined_dataset/audios_original
mv /ssdisk/wm/dance/combined_dataset/audios_24khz \
   /ssdisk/wm/dance/combined_dataset/audios
```

### Q4：什么时候可以开始训练？

完成所有步骤后，开始训练：

```bash
conda activate wan
cd /home/wangm/DiffSynth-Studio

accelerate launch examples/wanvideo/model_training/train.py \
  --max_pixels 399360 \
  --num_frames 49 \
  --fix_frame_rate True \
  --output_path ./models/train/Audio_lora_v1 \
  --dataset_base_path /ssdisk/wm/dance/combined_dataset \
  --dataset_metadata_path metadata.csv \
  --train_steps 5000 \
  --save_steps 500
```

---

## 📝 下一步

- ✅ 完成数据准备后，查看 `notes/04_wan22_ti2v_audio_training.md` 了解训练配置
- ✅ 如需调整文本标注，编辑 metadata.csv 的 `text` 列
- ✅ 如需分离竖屏/横屏训练，参考 `notes/06_data_preprocessing.md` 的方案二

---

## 📞 快速参考

| 任务 | 命令 |
|------|------|
| 生成元数据 | `python scripts/generate_metadata.py ...` |
| 验证数据集（快速） | `python scripts/validate_dataset.py ... --sample_size 10` |
| 验证数据集（完整） | `python scripts/validate_dataset.py ...` |
| 预处理音频 | `python scripts/preprocess_audio.py ...` |
| 查看数据统计 | `wc -l metadata.csv` 和 `ls -lh videos audios` |
| 检查第一行 | `head -5 metadata.csv` |

---
