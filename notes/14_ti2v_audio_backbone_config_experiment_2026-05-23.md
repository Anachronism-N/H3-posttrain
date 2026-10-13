# Wan2.2 TI2V Audio Backbone/Config 实验日志

Date: 2026-05-23

## 范围

本记录只覆盖当前工作区中与 `Wan2.2-TI2V-5B-Audio` 训练链路直接相关的改动与实验：

- `diffsynth/pipelines/wan_video.py`
- `examples/wanvideo/model_training/train.py`
- `examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-Audio.sh`
- `examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-Audio-formal.yaml`
- `examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-Audio-overfit.yaml`
- `examples/wanvideo/model_inference/Wan2.2-TI2V-5B-Audio.py`
- `examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py`
- `examples/wanvideo/model_training/validate_full/Wan2.2-TI2V-5B.py`
- `examples/wanvideo/model_training/validate_lora/Wan2.2-TI2V-5B.py`
- `data/dance_dataset/metadata.csv`

以下改动当前也在工作区中，但不纳入本实验日志：

- `README.md`
- `bad_data.txt`
- `HuMo/`
- `SCAIL/`
- `outputs/`

## 本轮代码整理

### 1. 显式区分 TI2V base backbone 与 audio backbone

背景问题：

- `Wan2.2-TI2V-5B` 与 `WanAudioModel` 共用同一套基础权重文件
- 之前依赖 `fetch_model(...)` 的自动命中顺序，训练/推理脚本不容易保证到底拿到的是 `wan_video_dit` 还是 `wan_video_dit_audio`

本次改动：

- `WanVideoPipeline.from_pretrained(...)` 新增 `dit_variant`
- 训练脚本、推理脚本、验证脚本分别显式指定：
  - `audio` 用于 `Wan2.2-TI2V-5B-Audio`
  - `base` 用于普通 `Wan2.2-TI2V-5B`

预期收益：

- 避免 base/audio 主干混用
- 让训练配置、推理配置和实验记录可以一一对应

### 2. 抽取共享 backbone 输入准备逻辑

文件：`diffsynth/pipelines/wan_video.py`

本次新增 `prepare_wan_video_backbone_inputs(...)`，将以下通用流程收敛到一个 helper：

- timestep 构造
- text embedding
- image embedding 拼接
- patchify
- `(f, h, w)` grid 信息回传

整理后：

- `WanAudioModel` 不再走一条完全独立的前向捷径
- 音频分支复用普通 TI2V backbone 的输入准备逻辑
- 真正的音频注入点保留在 `after_transformer_block(...)`

这让后续比较 `base` 与 `audio` 的差异时，边界更清楚。

### 3. 训练入口支持 YAML 配置与条件 dropout

文件：`examples/wanvideo/model_training/train.py`

新增能力：

- `--config` 支持加载一个或多个 YAML 配置文件
- 启动时打印关键训练参数，便于留档
- 新增 `text_drop_prob`
- 新增 `audio_drop_prob`

当前保留两套配置：

- `Wan2.2-TI2V-5B-Audio-overfit.yaml`
- `Wan2.2-TI2V-5B-Audio-formal.yaml`

对应整理：

- 启动脚本 `Wan2.2-TI2V-5B-Audio.sh` 不再内嵌 Python 动态写临时配置
- 改为直接读取仓库内 YAML
- 默认读取 `formal.yaml`
- 如需切换到 overfit，可设置：

```bash
TRAIN_CONFIG=examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-Audio-overfit.yaml \
bash examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-Audio.sh
```

### 4. 小样本 metadata prompt 从通用描述改为细节描述

文件：`data/dance_dataset/metadata.csv`

`best480` 小样本中的 `prompt` 从统一的 `a girl dancing to music` 改为逐样本描述，目的是：

- 提高 overfit 样本的文本条件信息量
- 让小样本 sanity check 更接近真实 caption 训练

## 当前配置归档

### `Wan2.2-TI2V-5B-Audio-overfit.yaml`

用途：

- 本地 `best480` 小样本过拟合检查

关键参数：

- `dataset_base_path: data/dance_dataset`
- `num_frames: 81`
- `dataset_repeat: 100`
- `num_epochs: 30`
- `use_gradient_checkpointing: true`
- `text_drop_prob: 0.0`
- `audio_drop_prob: 0.0`
- `output_path: ./models/train/Wan2.2-TI2V-5B-Audio_lora-best480-overfit`

### `Wan2.2-TI2V-5B-Audio-formal.yaml`

用途：

- 大数据正式训练配置

关键参数：

- `dataset_base_path: /ssdisk/wm/dance/combined_dataset/.../train_ready_16fps81_24kmono_nearfit`
- `num_frames: 81`
- `dataset_repeat: 10`
- `num_epochs: 10`
- `use_gradient_checkpointing: false`
- `text_drop_prob: 0.0`
- `audio_drop_prob: 0.1`
- `output_path: ./models/train/Wan2.2-TI2V-5B-Audio_1410mace0_dropCFG0.1_timestep`

说明：

- 当前工作区中的 `formal.yaml` 是“0.1 音频 dropout + timestep 标签”的较新版本
- 本地已有 W&B 记录里还保留了更早的 `dropCFG` 版本，因此日志中会同时出现新旧命名

## 已有实验记录

### 实验 1: `best480` 过拟合检查

时间：

- 启动时间：2026-05-19 12:43:56 UTC

来源：

- `logs/wan22_ti2v5b_audio_best480_overfit_gpu2_20260519_124521.log`
- `wandb/run-20260519_124356-apm41phf/`

对应配置：

- 与当前 `Wan2.2-TI2V-5B-Audio-overfit.yaml` 基本一致

W&B 摘要：

- `run_id`: `apm41phf`
- `wandb_run_name`: `lora_TIA2V_best480_overfit`
- `train/step`: `9000`
- `train/loss`: `0.4241489768`
- `train/gnorm`: `0.0193834630`
- `train/step_time`: `2.7362133469`

结论：

- 训练链路可以完整跑通到 W&B 收尾
- `WanAudioModel + MERT + LoRA` 的最小闭环是通的
- 仅凭训练 loss 还不足以判断动作质量和音画一致性，后续需要统一脚本做可视化抽样

### 实验 2: 正式训练早期控制实验

时间：

- 启动时间：2026-05-20 18:25:27 UTC

来源：

- `wandb/run-20260521_022527-1o42c7fg/`

对应配置：

- `formal.yaml` 的早期版本
- `audio_drop_prob: 1.0`
- metadata 中保留了 `--use_gradient_checkpointing`

W&B 摘要：

- `run_id`: `1o42c7fg`
- `wandb_run_name`: `lora_TIA2V_1410mace0_dropCFG`
- `train/step`: `17896`
- `train/loss`: `0.0291575994`
- `train/gnorm`: `0.0058982028`
- `train/step_time`: `7.1021101680`

备注：

- 该实验更像“强控制”或退化上界测试，因为音频条件被 100% dropout
- 低 loss 不能直接说明音频驱动效果更好，反而可能意味着模型主要在学习无音频条件下的重建

### 实验 3: 正式训练当前主线

时间：

- 启动时间：2026-05-21 11:51:05 UTC

来源：

- `wandb/run-20260521_195105-t6c2jpwu/`

对应配置：

- 接近当前 `Wan2.2-TI2V-5B-Audio-formal.yaml`
- `audio_drop_prob: 0.1`
- `text_drop_prob: 0.0`
- 本次 metadata 中未看到 `--use_gradient_checkpointing`

W&B 摘要：

- `run_id`: `t6c2jpwu`
- `wandb_run_name`: `lora_TIA2V_1410mace0_dropCFG`
- `train/step`: `7550`
- `train/loss`: `0.4617058337`
- `train/gnorm`: `0.0164095020`
- `train/step_time`: `2.7250910942`

备注：

- 相比实验 2，step time 明显更短
- 但该差异同时受到 `audio_drop_prob`、是否启用 gradient checkpointing、run 时长不同等因素影响，暂时不能单独归因

### 其他本地 run 痕迹

以下 run 目录目前只有 metadata，尚未看到完整 summary：

- `wandb/run-20260521_192423-thd6i6ex/`
- `wandb/run-20260522_031909-2ke6m0v4/`

暂按“中断或未完整同步”处理，不纳入当前结果判断。

## 当前判断

### 已经确认的事情

- `dit_variant` 显式化是必要的，能够减少 base/audio 主干误用
- YAML 配置已经足够承载 overfit 与 formal 两条训练分支
- `audio_drop_prob` 已经进入实验设计，不再只是代码层面的预埋开关

### 还不能下结论的事情

- `audio_drop_prob=0.1` 是否优于 `1.0` 或 `0.0`
- 不开 gradient checkpointing 是否只影响速度，还是会间接影响稳定性
- 新 prompt 是否实质改善 `best480` 小样本过拟合质量

## 建议的下一步

1. 固定一组 prompt/image/audio 样例，对 `best480-overfit`、`dropCFG`、`dropCFG0.1_timestep` 三条权重做同脚本可视化对比。
2. 如果继续推进正式训练，优先把 run name 与 YAML 文件名统一，避免 `dropCFG` 与 `dropCFG0.1_timestep` 混名。
3. 若要比较速度，单独控制 `audio_drop_prob`，只切换 `use_gradient_checkpointing`，重新做一次短程测速。

## 2026-05-28 补记：分批提交与 forward 重构

### 已拆出的前置提交

截至 2026-05-28，本轮修改已先拆出两次提交：

- `9b97c06` `feat: add YAML-driven Wan audio training configs and dropout controls`
- `abd4797` `feat: make Wan TI2V backbone selection explicit`

这样处理后，`wan_video.py` 中仍然保留了一批尚未提交的改动，它们不再承担“选哪个 backbone”的职责，而是专注于 `WanAudioModel` 的 forward 结构整理。

### 本次待提交范围

本次补记对应的剩余代码范围只包含：

- `diffsynth/pipelines/wan_video.py`
- 本实验日志文件

不包含：

- `data/dance_dataset/metadata.csv`
- `examples/wanvideo/model_inference/Wan2.2-Animate-14B_pose_only.py`
- `README.md`
- `bad_data.txt`
- `logs/`、`outputs/`、`SCAIL/`、`HuMo/` 等本地实验目录

### forward 重构的直接目的

剩余 `wan_video.py` 改动主要解决三个问题：

1. 让 TI2V 普通 backbone 与 `WanAudioModel` 共享同一套输入准备流程，而不是维护两套近似但逐步分叉的逻辑。
2. 把音频注入放回 block 级别的 `after_transformer_block(...)`，使音频 cross-attention 更贴近 `WanAudioModel` 自身的设计边界。
3. 避免“只要有 `audio_embeds` 就直接走 S2V 分支”的隐式行为，把 `WanAudioModel` 与 `WanS2V` 的路由条件区分清楚。

### 本次 `wan_video.py` 具体变化

#### 1. 抽出共享输入准备 helper

新增 `prepare_wan_video_backbone_inputs(...)`，统一封装：

- timestep 预处理
- text embedding
- image / VAE embedding 拼接
- patchify
- `(f, h, w)` grid 信息返回

这一步的核心价值不是“减少代码行数”，而是给普通 TI2V 与音频 TI2V 提供一致的 backbone 输入约定。

#### 2. 调整 `audio_embeds` 路由逻辑

在 `model_fn_wan_video(...)` 中：

- 当 `dit` 不是 `WanAudioModel` 且存在 `audio_embeds` 时，仍走 `model_fn_wans2v(...)`
- 当 `dit` 是 `WanAudioModel` 时，不再提前 `return dit(...)`
- 改为继续留在统一的 TI2V 主流程中，后续在 transformer block 后做音频注入

这避免了普通 TI2V / S2V / TI2V-Audio 三条路径之间的隐式耦合。

#### 3. 把音频注入收束到 block 后处理

新增：

- `audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)`
- `x = dit.after_transformer_block(...)`

也就是说，音频特征不再通过 `WanAudioModel.forward(...)` 的独立捷径直接吃掉，而是作为统一主干中的一个后处理阶段插入。

### 当前理解

这次 forward 重构更像“结构归一化”，而不是新功能扩张：

- 没有新增新的训练参数
- 没有修改 YAML 配置语义
- 没有引入新的数据格式

它的作用是把前面已经提交的 `dit_variant` 显式分流真正落到执行路径上，让 base/audio backbone 不只是“加载时分开”，而是“前向语义也更清楚地分开”。

## 2026-05-28 补记：fulldit 配置与推理 ckpt 直载

### 本次记录范围

这次补记对应的是一组“配置与推理支持”变更，不对应新的完整训练 summary，因此不写入 `logs/train_run_summary.csv`：

- 新增 `examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-Audio-fulldit.yaml`
- 更新 `examples/wanvideo/model_inference/Wan2.2-TI2V-5B-Audio.py`

### fulldit 配置

`Wan2.2-TI2V-5B-Audio-fulldit.yaml` 基于当前 `vl2836` 这条 480x832 / 81 帧训练线，保留：

- `dataset_repeat: 10`
- `num_epochs: 10`
- `audio_drop_prob: 0.1`
- `use_gradient_checkpointing: true`

与 `formal.yaml` 的关键区别是：

- `trainable_models: dit`
- `lora_base_model: null`
- 输出目录改为 `..._fulldit`

另外，本次顺手修正了一个记录层面的命名问题：原文件里的 `wandb_run_name` / `wandb_tags` 仍沿用了 `audioInj`，容易和只训练 `dit.audio_injector.injector` 的实验串台；现已改为 `fulldit_*` 命名。

### 推理脚本支持

`Wan2.2-TI2V-5B-Audio.py` 新增了 `--ckpt_path`，允许在推理时按以下顺序叠加权重：

1. base model
2. optional training checkpoint
3. optional LoRA

这让以下两类验证路径都更直接：

- 直接评估 full/partial fine-tune checkpoint
- 先加载训练 checkpoint，再叠加 LoRA 做组合验证

脚本当前会打印 checkpoint 加载后的 `missing_keys` / `unexpected_keys` 统计，方便判断权重是否只覆盖了预期模块。

### 当前状态

- 这是配置和推理支持的补记，不代表已经确认存在新的稳定收敛 run
- 当前工作区里未检出可直接回填到 `train_run_summary.csv` 的 `wandb-summary.json`
- 后续若实际启动 `fulldit` 训练并产生完整 W&B summary，再追加到总表更稳妥
