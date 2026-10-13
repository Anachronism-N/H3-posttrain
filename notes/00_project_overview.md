# DiffSynth-Studio 项目总览

> 最后更新：2026-03-12

## 项目定位

DiffSynth-Studio 是一个**视频/图像扩散模型推理与训练框架**，支持多种架构（FLUX, Wan, Qwen-Image, Z-Image 等）。核心设计理念是将模型结构、训练逻辑和数据管道解耦，便于快速扩展新模型变体。

---

## 顶层目录结构

```
DiffSynth-Studio/
├── diffsynth/                  # 核心库
│   ├── models/                 # 模型定义（DiT、VAE、编码器等）
│   ├── pipelines/              # 推理 Pipeline（组合模型）
│   ├── diffusion/              # 训练基础设施（module、runner、loss）
│   ├── core/                   # 数据加载、模型 loader
│   ├── configs/                # 模型配置注册表
│   └── utils/                  # LoRA、state_dict 转换器
├── examples/                   # 各模型推理/训练示例
│   ├── wanvideo/
│   ├── flux/
│   ├── qwen_image/
│   └── z_image/
├── docs/                       # 英/中文文档
├── notes/                      # 学习笔记（本目录）
└── pyproject.toml
```

---

## 核心模型系列（当前活跃）

| 系列 | 基座 | 特点 |
|------|------|------|
| **Wan2.1** | T2V-14B, I2V-14B, VACE-14B | 文本/图像→视频，分辨率多样 |
| **Wan2.2** | T2V/I2V-14B, S2V-14B, Animate-14B, TI2V-5B | 新增 S2V（语音→视频）、动画控制 |
| **Qwen-Image** | DiT-based | 图像生成+编辑，含 EliGen、ControlNet |
| **Z-Image** | 自研 | 高速推理，含 ControlNet Union |

---

## 核心模块关系图

```
Pipeline（wan_video.py）
    ├── 调用 TextEncoder（T5）
    ├── 调用 ImageEncoder（CLIP，可选）
    ├── 调用 AudioEncoder（wav2vec2，S2V only）
    ├── 调用 VAE.encode（输入潜变量）
    ├── 循环调用 DiT.forward（去噪）
    │     └── 内部：自注意力 + 文本交叉注意力 + 音频交叉注意力（S2V）
    └── 调用 VAE.decode（生成帧）
```

---

## 关键文件索引

### 模型层
| 文件 | 说明 |
|------|------|
| `diffsynth/models/wan_video_dit.py` | Wan 基础 DiT（WanModel），含 DiTBlock、CrossAttention |
| `diffsynth/models/wan_video_dit_s2v.py` | S2V 变体（WanS2VModel），含 AudioInjector、CausalAudioEncoder |
| `diffsynth/models/wav2vec.py` | 音频编码器封装（WanS2VAudioEncoder，基于 wav2vec2-large-xlsr） |
| `diffsynth/models/wan_video_vae.py` | 视频 VAE（编解码潜变量） |
| `diffsynth/models/wan_video_text_encoder.py` | 文本编码器（UMT5-xxl） |

### 训练层
| 文件 | 说明 |
|------|------|
| `diffsynth/diffusion/training_module.py` | DiffusionTrainingModule 基类 |
| `diffsynth/diffusion/runner.py` | launch_training_task() 训练入口 |
| `examples/wanvideo/model_training/train.py` | WanTrainingModule，支持 SFT/蒸馏/LoRA |

### 配置层
| 文件 | 说明 |
|------|------|
| `diffsynth/configs/model_configs.py` | 所有模型的 hash/class/extra_kwargs 注册表 |

### 工具层
| 文件 | 说明 |
|------|------|
| `diffsynth/utils/lora/general.py` | 通用 LoRA 注入/融合（基于 PEFT） |
| `diffsynth/utils/state_dict_converters/` | 各模型权重格式转换器 |
| `diffsynth/core/data/unified_dataset.py` | 统一数据集加载（JSON/JSONL/CSV） |

---

## 学习路径建议

1. **[基础]** 理解 DiT 架构 → 读 `wan_video_dit.py`
2. **[进阶]** 理解音频注入 → 读 `wan_video_dit_s2v.py` + `wav2vec.py`
3. **[实战]** 理解训练流程 → 读 `examples/wanvideo/model_training/train.py`
4. **[扩展]** 修改音频编码器 → 见 `notes/03_audio_encoder_research_plan.md`
