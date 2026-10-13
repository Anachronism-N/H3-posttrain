# 音频 Cross-Attention 实现变更记录

> 目标：为 Wan2.2-TI2V-5B 添加音乐感知音频 cross-attention，支持音乐驱动舞蹈视频生成
> 基础模型：Wan2.2-TI2V-5B（dim=3072, num_heads=24, num_layers=30）
> 音频编码器：MERT-v1-330M（替换 wav2vec2-large-xlsr-53）
> 最后更新：2026-03-19

---

## 模块一：音频编码器

### 新增文件：`diffsynth/models/mert_audio_encoder.py`

**提交历史：**
| 提交 | 内容 |
|------|------|
| `f8886b6` | 初始实现 MERTAudioEncoder |
| `f3aac83` | 修复接口兼容性（添加 `processor=None` 参数）和 hidden state 堆叠方式（`torch.cat` → `torch.stack`） |
| `8760952` | 修复设备处理（`.detach().cpu().numpy()`）、模型/输入设备对齐（`next(self.model.parameters()).device`）、添加 `audio_fps=75` 推导注释 |

**核心类：`MERTAudioEncoder(nn.Module)`**

**为什么选 MERT 而不是 wav2vec2：**
- wav2vec2-xlsr 训练于 56k 小时语音数据，内部表征围绕音素边界组织，对音乐节拍不敏感
- MERT-v1-330M 训练于 160k 小时音乐数据，对节拍/节奏/旋律敏感
- 关键"巧合"：两者架构深度完全相同（24层→25 hidden states，hidden_size=1024），改动极小

**关键设计决策：**

1. **`audio_fps = 75`**：MERT CNN stride=320，采样率 24kHz → 24000/320=75fps 精确值。与 wav2vec2 的 50fps（16kHz/320）不同，需修改插值参数。

2. **`processor=None` 接口兼容**：`WanS2VAudioEncoder.get_audio_feats_per_inference` 要求 `processor` 参数，MERT 内部管理自己的 processor，保留该参数但忽略，确保调用方代码无需修改。

3. **`torch.stack` 而非 `torch.cat`**：`torch.cat(hidden_states, dim=0)` 在 hidden states 为 `(1,T,D)` 时碰巧正确，但 `torch.stack` 语义更明确，对 `(T,D)` 形状也安全。

4. **设备对齐**：`input_values` 移动到 `next(self.model.parameters()).device`，而非硬编码 `device` 参数，确保模型被外部移动到 GPU 后输入自动跟随。

**输出格式（与 WanS2VAudioEncoder 完全相同）：**
```
extract_audio_feat() → (25, T_30fps, 1024)
get_audio_feats_per_inference() → list of (1, 25, 1024*(2m+1), batch_frames)
```

---

## 模块二：WanAudioModel

### 新增文件：`diffsynth/models/wan_video_dit_audio.py`

**提交历史：**
| 提交 | 内容 |
|------|------|
| `3f2980a` | 初始实现 WanAudioModel，含 CausalAudioEncoder + AudioInjector_WAN + forward |
| `121df40` | 修复：`CausalAudioEncoder` 添加 `num_layers=25`；`audio_inject_layers` 改为 `None` 默认值（避免可变默认参数反模式） |
| `1d3f7c1` | 修复：`after_transformer_block` 添加 `enable_adain=False` guard；添加帧对齐断言；添加梯度检查点注释 |
| `7591ab6` | 验证提交：实例化测试通过，CausalAudioEncoder 输出 shape 确认 |

**核心类：`WanAudioModel(WanModel)`**

**为什么继承 WanModel 而不是 WanS2VModel：**
- TI2V-5B 使用 `WanModel`（标准 DiTBlock），S2V-14B 使用 `WanS2VModel`（WanS2VDiTBlock，含双流时间步调制）
- `AudioInjector_WAN` 通过 `torch_dfs` 扫描 `isinstance(mod, DiTBlock)` 注入，标准 DiTBlock 完全兼容
- 音频注入发生在 block forward 之外（`after_transformer_block`），不需要修改 block 内部签名

**关键设计决策：**

1. **注入层选择 `[0, 4, 8, 12, 16, 20, 24, 27]`（8层/30层）**：S2V-14B 在 40 层中注入 12 层，比例约 30%。TI2V-5B 30 层中注入 8 层，比例相近，均匀分布覆盖浅层（语义）和深层（细节）。

2. **`audio_input=None` 可选注入**：`forward` 中 `if audio_emb is not None` 判断使同一模型类既可作为普通 TI2V-5B 使用，也可作为音频驱动模型使用，无需两个类。

3. **`enable_adain=False` guard**：当 `need_global=False` 时，`CausalAudioEncoder` 不返回 global token，`audio_emb_global` 为 None，必须跳过 AdaLayerNorm 调用，否则 crash。

4. **帧对齐断言**：`original_seq_len % num_frames == 0` 确保 rearrange 维度正确，在 shape 不匹配时给出清晰错误而非静默产生错误结果。

5. **音频注入不做梯度检查点**：AudioInjector 是轻量级 cross-attention（相比 DiTBlock 参数量小得多），不值得增加检查点复杂度。

**实测输出 shape（TI2V-5B 配置）：**
```python
audio_input = torch.randn(1, 25, 1024, 80)  # (B, num_layers, dim, T)
audio_emb_global.shape  # (1, 20, 1, 3072)  ← T/4 因 MotionEncoder_tc stride=4
audio_emb.shape         # (1, 20, 5, 3072)  ← num_audio_token+1=5（含 padding_token）
```

**注意**：`audio_emb.shape[2] == 5`（不是 4）是正确的。`MotionEncoder_tc` 在 local 分支的 `num_heads=4` 个 token 后追加了一个可学习的 `padding_tokens`，与 S2V-14B 行为完全一致。

---

## 模块三：Pipeline 集成

### 修改文件：`diffsynth/pipelines/wan_video.py`

**提交历史：**
| 提交 | 内容 |
|------|------|
| `46a8dde` | 添加 `WanAudioModel` import、`WanVideoUnit_Audio` 类、`mert_audio_encoder` 属性、`model_fn_wan_video` 分支 |
| `1ada4f3` | 修复：`latents=latents` → `x=latents`（参数名不匹配）；`mert_audio_encoder is None` 改为 `RuntimeError`（静默失败→明确报错） |

**新增内容：**

**1. `self.mert_audio_encoder = None`（`WanVideoPipeline.__init__`，约 line 54）**
- 与 `self.audio_encoder`（wav2vec2）并列，用于 MERT 编码器
- 需要外部手动赋值：`pipe.mert_audio_encoder = MERTAudioEncoder(...)`

**2. `WanVideoUnit_Audio` 类（约 line 814）**
- 在 `WanVideoUnit_S2V` 之前注册，优先处理
- 仅在 `isinstance(pipe.dit, WanAudioModel)` 时激活，不影响其他模型
- 从 `inputs_shared` pop `input_audio`，调用 `mert_audio_encoder.get_audio_feats_per_inference`
- 将结果写入 `inputs_posi["audio_embeds"]`，负样本为零张量

**3. `model_fn_wan_video` 分支（约 line 1242）**
```python
if isinstance(dit, WanAudioModel):
    return dit(
        x=latents,          # 注意：WanAudioModel.forward 第一参数名为 x，不是 latents
        timestep=timestep,
        context=context,
        audio_input=audio_embeds,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
```
- 位于 S2V `if audio_embeds is not None` 分支之前，优先匹配
- **关键 bug 修复**：`latents=latents` → `x=latents`，参数名必须与 `WanAudioModel.forward(self, x, ...)` 一致

---

## 模块四：模型注册

### 修改文件：`diffsynth/configs/model_configs.py`

**提交：`3f2980a`**

新增条目（与 TI2V-5B 共享相同 model_hash）：
```python
{
    "model_hash": "1f5ab7703c6fc803fdded85ff040c316",  # 与 TI2V-5B 相同权重
    "model_name": "wan_video_dit_audio",
    "model_class": "diffsynth.models.wan_video_dit_audio.WanAudioModel",
    "extra_kwargs": {
        'dim': 3072, 'num_heads': 24, 'num_layers': 30,
        'audio_dim': 1024, 'num_audio_token': 4,
        'audio_inject_layers': [0, 4, 8, 12, 16, 20, 24, 27],
        'enable_adain': True,
        # ... 其余与 TI2V-5B 相同
    },
},
```

**为什么共享 model_hash**：`WanAudioModel` 继承 `WanModel`，加载相同的 TI2V-5B checkpoint 权重，音频模块（`casual_audio_encoder`、`audio_injector`）从随机初始化开始训练。

---

## 模块五：训练脚本

### 新增文件：`examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-Audio.sh`

**提交：`46a8dde`（创建），`1ada4f3`（添加 MERT 注释）**

**关键参数说明：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `--lora_target_modules` | `"q,k,v,o"` | CrossAttention 的实际属性名（实测确认） |
| `--lora_rank` | `32` | 平衡表达能力与参数量 |
| `--extra_inputs` | `"input_audio"` | 触发 `parse_extra_inputs` 中的音频加载路径 |
| `--trainable_models` | `"dit"` | 冻结 T5/VAE，只训练 DiT（含音频模块） |
| `--data_file_keys` | `"video,input_audio,text"` | 数据集 CSV 列名 |

**LoRA 注入位置**：`audio_injector.injector.*.q/k/v/o`（8 个注入层 × 4 个矩阵 = 32 个 LoRA 模块）

**显存估算（H100 80GB）：**
- 冻结 DiT backbone（5B, BF16）：~10 GB
- 可训练音频模块（~1.5B）：~3 GB
- 优化器状态（Adam, FP32）：~6 GB
- 激活值（gradient_checkpointing）：~8 GB
- VAE + T5 + 数据：~8 GB
- **合计：~35 GB**（H100 余量充足）

---

## 模块六：数据集与工具

### 新增文件：`scripts/preprocess_audio.py`

**提交：`aabc193`**

将任意格式音频转换为 MERT 所需的 24kHz 单声道 WAV：
```bash
python scripts/preprocess_audio.py \
    --input_dir /path/to/raw_audio \
    --output_dir data/dance_dataset/audios
```

### 新增文件：`data/dance_dataset/metadata.csv`

数据集索引格式（与 `train.py` 的 `UnifiedDataset` 兼容）：
```csv
video,input_audio,text
videos/dance_001.mp4,audios/dance_001.wav,a person dancing to music
```

---

## 模块七：推理脚本

### 新增文件：`examples/wanvideo/model_inference/Wan2.2-TI2V-5B-Audio.py`

**提交：`a386d04`**

端到端推理流程：
1. 加载 TI2V-5B 权重到 `WanAudioModel`（通过 `model_name="wan_video_dit_audio"`）
2. 手动赋值 `pipe.mert_audio_encoder = MERTAudioEncoder(...)`
3. 加载音频 → `torchaudio.load` → 转单声道
4. 调用 `pipe(prompt=..., input_audio=..., audio_sample_rate=...)`

**注意**：VAE 文件名为 `Wan2.2_VAE.pth`（不是 `Wan2.1_VAE.pth`），与 `model_configs.py` 中 TI2V-5B 的注册一致。

---

## 架构对比：WanAudioModel（5B）vs WanS2VModel（14B）

### 总览

| 维度 | WanS2VModel（14B，原版） | WanAudioModel（5B，本实现） |
|------|------------------------|--------------------------|
| 设计哲学 | **原生音频模型**，从头为音频驱动设计 | **后天注入**，在现有 TI2V-5B 上 retrofit |
| 基础 DiT 块 | `WanS2VDiTBlock`（双流时间步调制） | 标准 `DiTBlock`（单流） |
| 模型维度 | dim=5120, num_heads=40, num_layers=40 | dim=3072, num_heads=24, num_layers=30 |
| 音频注入层数 | 12/40 层（30%） | 8/30 层（27%，比例相近） |
| 音频编码器 | wav2vec2-large-xlsr（语音，16kHz） | MERT-v1-330M（音乐，24kHz） |
| 参考帧机制 | ✅ 有（`cond_encoder` + pose_cond） | ❌ 无（TI2V-5B 不含参考帧编码器） |
| 运动帧机制 | ✅ 有（`FramePackMotioner`，多尺度运动 token） | ❌ 无 |
| 音频预热填充 | ✅ 有（prepend 73 帧，裁掉 19 帧） | ❌ 无（直接送入 CausalAudioEncoder） |
| 训练显存（冻结 backbone） | ~65 GB（H100 勉强） | ~35 GB（H100 充裕） |

---

### 差异一：DiTBlock 类型

**S2V-14B** 使用 `WanS2VDiTBlock`，其 `forward` 接受额外的 `seq_len_x` 参数：

```python
# WanS2VDiTBlock.forward
def forward(self, x, context, t_mod, seq_len_x, freqs):
    # 将时间步调制分为两组：
    # t_mod[:,:,0] → 主内容（视频 latent）
    # t_mod[:,:,1] → 辅助内容（参考帧、运动帧）
    t_mod = [
        torch.cat([element[:,:,0].expand(1, seq_len_x, ..),
                   element[:,:,1].expand(1, x.shape[1]-seq_len_x, ..)], dim=1)
        ...
    ]
```

这是因为 S2V 的序列中混合了视频 latent + 参考帧 token + 运动帧 token，需要对不同 token 施加不同的时间步调制。

**WanAudioModel** 使用标准 `DiTBlock`，序列中只有视频 latent（无参考帧/运动帧混合），时间步调制统一，无需 `seq_len_x`。

---

### 差异二：运动帧机制（S2V 独有）

S2V-14B 有完整的 `FramePackMotioner`，将历史运动帧以多尺度（1x/2x/4x 下采样）编码为 token，拼接到序列末尾：

```
序列结构（S2V）：[视频 latent tokens] + [参考帧 tokens] + [运动帧 tokens（多尺度）]
序列结构（5B）：[视频 latent tokens]（仅此）
```

运动帧的作用是让模型"看到"之前生成的帧，实现时序连贯性。TI2V-5B 没有这个机制，长视频生成时连贯性依赖 VAE 的时序建模。

---

### 差异三：音频预热填充（S2V 独有）

S2V 的 `cal_audio_emb` 在序列头部 prepend 73 帧：

```python
def cal_audio_emb(self, audio_input, motion_frames=[73, 19]):
    # prepend 73 帧（复制第 0 帧）
    audio_input = torch.cat([audio_input[..., 0:1].repeat(1,1,1,73), audio_input], dim=-1)
    audio_emb_global, audio_emb = self.casual_audio_encoder(audio_input)
    # 裁掉前 19 帧（补偿因果卷积延迟）
    audio_emb_global = audio_emb_global[:, 19:].clone()
    merged_audio_emb = audio_emb[:, 19:, :]
```

**为什么需要这个**：`MotionEncoder_tc` 有 3 层因果卷积（stride=1,2,2），总下采样 4x。因果卷积在序列开头没有历史上下文，会产生"冷启动"偏差。prepend 73 帧提供足够的预热，裁掉 19 帧（≈73/4）补偿延迟。

**WanAudioModel 没有这个**：直接送入 `CausalAudioEncoder`，序列开头的音频特征质量略低。如果训练后发现视频开头动作不准，可以考虑补充这个机制。

---

### 差异四：参考帧编码器（S2V 独有）

S2V 有 `self.cond_encoder = nn.Conv3d(cond_dim, dim, ...)` 用于编码 pose 条件（骨骼/姿态视频）：

```python
# S2V forward
x, (f,h,w) = self.patchify(self.patch_embedding(x) + self.cond_encoder(pose_cond))
```

这使 S2V 能同时接受音频 + 姿态两种条件，实现"给定音乐和参考姿态生成舞蹈"。

**WanAudioModel** 只有音频条件，无姿态控制。如需添加姿态控制，可参考 S2V 的 `cond_encoder` 设计。

---

### 差异五：音频编码器（核心改进）

| | S2V-14B | WanAudioModel（5B） |
|--|---------|-------------------|
| 模型 | wav2vec2-large-xlsr-53 | MERT-v1-330M |
| 训练数据 | 56k h 多语言语音 | 160k h 音乐 |
| 采样率 | 16kHz | 24kHz |
| 输出帧率 | ~50fps → 插值到 30fps | ~75fps → 插值到 30fps |
| 节拍感知 | 弱（音素级） | 强（音乐结构级） |
| 参数量 | ~317M | ~330M（相近） |

这是本实现相对于 S2V-14B 的**核心改进点**：用音乐专用编码器替换语音编码器，使模型能感知节拍、旋律、节奏等音乐特征。

---

### 何时选择哪个方案

| 场景 | 推荐 |
|------|------|
| 语音驱动口型/表情 | S2V-14B（原版，语音编码器更合适） |
| 音乐驱动舞蹈（有姿态参考） | S2V-14B + 替换音频编码器为 MERT |
| 音乐驱动舞蹈（无姿态参考，单卡训练） | **WanAudioModel（本实现）** |
| 快速验证音频注入架构 | WanAudioModel（显存需求低，迭代快） |

---

## 已知限制与后续工作

| 限制 | 说明 | 后续方案 |
|------|------|---------|
| MERT 需手动加载 | `pipe.mert_audio_encoder` 需外部赋值，未集成到 `from_pretrained` | 在 `model_configs.py` 注册 MERT，添加 `mert_audio_encoder` 到 `from_pretrained` 加载逻辑 |
| LoRA 只覆盖 CrossAttention | `casual_audio_encoder` 和 `CausalAudioEncoder` 未加 LoRA | 全量微调音频模块，或对 MotionEncoder_tc 也加 LoRA |
| 音频注入不做梯度检查点 | 训练时显存略高于理论值 | 如显存不足，可对 `after_transformer_block` 也加 checkpoint |
| Jukebox 特征适配未实现 | 计划中的 `JukeboxFeatureAdapter`（见 plan 文件）尚未开发 | 见 `notes/02_music_dance_research_plan.md` Phase 1 |

「梯度检查点」在计算图中手动指定「需要保存的断点」，只有断点处的结果会被保留，其余中间结果反向时重新计算

---

## 2026-05-13 未暂存实验改动整理

> 目标：把当前工作区里和训练、推理、数据清洗、SCAIL 批处理相关的未暂存更改按模块记清楚，便于后续继续实验或拆分提交。

### 一、训练监控与异步验证链路

#### 1. `diffsynth/diffusion/logger.py`

本次把原来偏 `wandb-only` 的记录方式补成了「`wandb + 本地文件`」双通道：

- 新增 `train.log` 文件日志，按 step 记录 `epoch / loss / lr / grad_norm / step_time / data_time`
- 新增 `metrics.jsonl`，把训练指标、验证指标、验证失败事件统一落成结构化 JSON 行
- 新增 `on_validation_end(...)` 与 `on_validation_failed(...)`
- 在保存 checkpoint、结束 epoch、训练结束时额外写入日志事件

这部分改动的意义是：即使 `wandb` 不方便回看，实验目录本身也已经带有可追溯的训练过程记录。

#### 2. `diffsynth/diffusion/runner.py`

训练主循环现在支持**异步验证**：

- 新增 `compute_grad_norm(...)`，把梯度范数直接纳入训练日志
- 新增 `is_async_validation_enabled(...)`，由 `val_every_steps + val_dataset_metadata_path + val_num_batches` 共同决定是否启用
- 新增 `validation/snapshots/`、`validation/results/`、`validation/logs/` 三类输出路径
- 在训练过程中轮询异步验证任务，避免验证阻塞主训练
- 训练结束前显式等待所有验证子进程完成，防止最后几次验证丢失

当前实现的运行方式是：

1. 训练到指定 step
2. 若该 step 没碰到正常 `save_steps`，就额外保存一个验证快照
3. 用子进程拉起 `validate_loss.py`
4. 子进程算完后把 `val_loss` 写回 `result.json`
5. 主进程轮询并把结果收进 `metrics.jsonl` / `train.log`

#### 3. `examples/wanvideo/model_training/train.py`

训练入口脚本做了三类结构化调整：

- 新增 `build_wan_dataset(...)`、`build_wan_model(...)`、`build_wandb_config(...)`，让训练与验证共享同一套构建逻辑
- 新增命令行参数：`val_dataset_metadata_path`、`val_every_steps`、`val_num_batches`、`val_script_path`、`val_cuda_visible_devices`
- 训练开始时会把当前参数写入 `output_path/args.json`

其中 `args.json` 是异步验证链路的关键，因为验证子进程靠它复原训练配置。

#### 4. `examples/wanvideo/model_training/validate_loss.py`

这是这次新增的验证 worker：

- 读取训练目录里的 `args.json`
- 覆盖验证集路径和待评估 LoRA checkpoint
- 复用 `build_wan_dataset(...)` 和 `build_wan_model(...)`
- 用 `accelerate` 跑有限个 batch 的前向损失
- 输出 `step / val_loss / val_batches / val_time / lora_checkpoint`

它的定位非常明确：只负责快速验证损失，不承担正式推理评测。

#### 5. `train.sh` 与 `examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-Audio.sh`

训练启动脚本这次主要是为了把实验参数显式化：

- 顶层 `train.sh` 新增统一入口变量：`GPU_ID / NUM_EPOCHS / SAVE_STEPS / VAL_EVERY_STEPS / VAL_NUM_BATCHES / LEARNING_RATE / GRADIENT_ACCUMULATION_STEPS / WANDB_RUN_NAME / WANDB_TAGS`
- 启动时会把完整 launch command 打印到日志，方便复现
- 音频训练脚本把训练数据目录切到仓库内 `data/dance_dataset`
- `dataset_repeat` 从 `1` 调到 `100`
- `num_epochs`、`save_steps`、`learning_rate`、`gradient_accumulation_steps` 都改成环境变量可覆盖
- 新增验证集参数：`--val_dataset_metadata_path /ssdisk/wm/dance/combined_dataset/metaval.csv`
- `extra_inputs` 从只喂 `input_audio` 改成 `input_image,input_audio`
- `wandb_run_name` 和 `wandb_tags` 改成可注入变量

这一组修改的目标是把训练任务从「一次性命令」改成「可回放的实验配置」。

### 二、推理入口与音频路径稳定性

#### 1. `examples/wanvideo/model_inference/Wan2.2-TI2V-5B-Audio.py`

音频版 TI2V 推理脚本从 demo 形式改成了更完整的 CLI：

- 支持 `--input_image + --audio_path + --prompt` 的完整输入组合
- 支持 `--lora_path` 和 `--lora_alpha`
- 模型加载改为显式使用本地 `ModelConfig(path=...)`
- 自动解析本地 MERT 缓存目录
- 新增 `--save_with_audio`，可以把原音频重新 mux 回生成视频
- 新增 `sanitize_proxy_env(...)`，默认去掉 socks 代理环境变量，避免本地加载模型时走歪代理

这意味着它现在更适合作为正式实验推理入口，而不是只跑一个硬编码样例。

#### 2. `examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py`

无音频版 TI2V 推理脚本也做了同样方向的整理：

- 从硬编码示例改成命令行驱动
- 支持纯文生视频，也支持传 `--input_image`
- 支持 `--save_with_audio` 仅做封装回写
- 同样改成本地模型路径装配和代理环境清理

这个改动的作用是让基础 TI2V 与音频版 TI2V 的使用方式更一致，便于做横向对照实验。

#### 3. `diffsynth/models/mert_audio_encoder.py`

MERT 编码器补了本地路径解析逻辑：

- 如果用户给的是完整本地目录，直接使用
- 如果给的是不完整目录，会明确报出缺失的 `config.json / preprocessor_config.json`
- 如果给的是 repo id，会尝试从 `~/.cache/modelscope/...` 和 `~/.cache/huggingface/...` 里找本地缓存

这能减少“模型明明下载过，但脚本还是去联网找”的情况。

#### 4. `diffsynth/pipelines/wan_video.py`

这里的修改很小，但对兼容性关键：

- `patchify(...)` 现在兼容两种返回值
- 如果实现返回 `(tokens, (f, h, w))`，就直接走 tuple 分支
- 如果还是旧版 BCHW latent，就走回退逻辑并手动 `rearrange`

它主要是在兼容当前 `WanAudioModel` / `Animate` 侧不同 patchify 输出风格，避免推理脚本在新旧实现间直接炸掉。

#### 5. `examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py` 与 `examples/wanvideo/model_inference/Wan2.2-Animate-14B_pose_only.py`

这两份脚本本轮改动主要偏“实验入口校正”：

- `Wan2.1-VACE-1.3B.py` 改成本地文件路径装配，关闭 `redirect_common_files`
- `Wan2.2-Animate-14B_pose_only.py` 把默认样例路径切到当前 `eval/scail` 相关数据上，便于直接验证 pose-only 结果

它们不属于主训练链路，但属于同一批推理验证工具的整理。

### 三、SCAIL 批处理与 pose 渲染流水线

#### 1. `examples/wanvideo/model_inference/run_manifest_pose_inference_v2.py` 与 `run_manifest_pose_inference.sh`

这部分已经单独写进 `notes/11_script_logic_reference.md`。当前未暂存状态里，它对应的是：

- 以 `manifest.jsonl` 为入口的批量姿态驱动推理
- 按模型类型自动切分 `Fun / Animate / S2V`
- 新增一个 `.sh` 入口脚本，固定用 `wan_pose` 环境启动

它属于“批量对照不同 Wan 模型”的实验工具链。

#### 2. `batch_extract_scail_pose_and_generate.py`

新增单环境批处理脚本，串联：

- DWPose 检测
- NLF 3D pose 提取
- pose 视频渲染
- SCAIL 最终生成
- `manifest.jsonl` 状态记录

适合依赖都装在一个环境里时快速整批处理。

#### 3. `batch_scail_cross_env.py`

新增跨环境 orchestrator，核心变化有三点：

- 把 `dwpose-meta / nlf-only / render-only / generate-only` 拆成独立 mode
- 通过 `conda run -n ...` 把不同阶段放到不同环境
- 中间缓存拆成 `dwpose_meta / pose_pkl / rendered_pose / scail_generated`

另外还补了：

- `--max-frames`：便于先截短视频快速调通
- `--pose-render-mode body_only`：便于对照不同 pose 渲染形式
- `manifest.jsonl` 按样本记录成功、失败和 traceback

#### 4. `render_scail_pose_cached.py`

新增缓存消费式渲染脚本，用来：

- 只读取已有 `poses.pkl / nlf.pkl / meta.json`
- 按参考图重新估计对齐相机
- 分块渲染 pose 视频
- 在 `full` 与 `body_only` 两种显示模式间切换

它的价值在于把“重渲染 pose 可视化”从完整抽取流程里剥出来。

### 四、数据索引与坏样本记录

#### 1. `data/dance_dataset/metadata.csv`

数据索引从原来的：

```csv
video,input_audio,text
```

调整成：

```csv
image_path,input_audio,video,prompt
```

同时当前示例行也切到了：

- 首帧图：`imgs/dance_001_first_frame.jpg`
- 裁剪视频：`videos/dance_001_832x480_scale_center_crop.mp4`

这说明当前实验已经从“只有视频+音频”转向“显式提供参考图+音频+视频+文本”的组织方式。

#### 2. `bad_data.txt`

这次往 `bad_data.txt` 里追加了大量坏样本记录，类型主要有两类：

- `ffmpeg` 读元数据/转码失败，例如部分 `HE-AAC` 音频视频在抽帧或解码阶段报错
- 数据索引里有路径，但仓库当前数据目录下找不到对应文件

这份文件现在更像是一个**训练前排雷清单**：

- 哪些样本在数据加载阶段就会炸
- 哪些路径已经失效
- 后续筛数据时应该优先排除哪一批文件

### 五、当前这批未暂存改动可以怎么理解

如果按实验目的来分，这批更改大致可以拆成四条线：

1. 训练侧：把 LoRA 训练流程补成“可记录、可异步验证、可复现”
2. 推理侧：把 TI2V / TI2V-Audio / Animate / VACE 脚本整理成可直接跑本地模型的实验入口
3. SCAIL 侧：补齐单环境与跨环境两套 pose 提取到生成的流水线
4. 数据侧：把参考图显式纳入 metadata，并持续沉淀坏样本清单

如果后面要拆 commit，建议也按这四条线拆，而不要把训练、推理、数据清洗和批处理脚本混在一个提交里。

---

## 2026-05-20 已提交落地记录

> 说明：本节只记录已经进入 git 历史的代码改动与关键配置更新，用于和上面的“未暂存整理”区分。对应当前主线提交：
>
> - `0562225` `chore: ignore generated media artifacts`
> - `4682462` `feat: add async validation to Wan audio training`
> - `8b43e35` `feat: adapt Wan TI2V data flow for MERT audio inputs`
> - `f672300` `feat: convert Wan TI2V inference scripts into CLIs`
> - `872c22f` `feat: add SCAIL batch pose generation tooling`
> - `2e83131` `docs: add align debug and evaluation notes`

### 一、仓库边界与产物清理

#### 1. `.gitignore`

新增常见生成媒体与压缩产物忽略规则：

- `*.mp4`
- `*.wav`
- `*.png`
- `*.jpg`
- `*.zip`

这一步的目的不是修改功能，而是防止后续实验输出继续污染版本历史。

#### 2. 顶层示例视频产物删除

从版本库中删除了 3 个已被跟踪的推理输出视频：

- `video_1_Wan2.2-S2V-14B-1.mp4`
- `video_1_Wan2.2-S2V-14B.mp4`
- `video_1_Wan2.2-TI2V-5B.mp4`

这表示后续默认不再把生成结果作为源码的一部分维护。

### 二、Wan 音频训练链路

#### 1. `diffsynth/diffusion/logger.py`

训练日志从单纯依赖 `wandb` 扩展为“`wandb + 本地文件`”双通道：

- 新增 `train.log`
- 新增 `metrics.jsonl`
- step 级别记录：
  - `loss`
  - `lr`
  - `grad_norm`
  - `step_time`
  - `data_time`
  - `epoch`
- 新增验证结束与验证失败事件记录

这使训练目录本身就能承载追踪信息，不再完全依赖在线面板。

#### 2. `diffsynth/diffusion/runner.py`

训练循环新增异步验证能力：

- `compute_grad_norm(...)`
- `is_async_validation_enabled(...)`
- 统一验证路径：
  - `validation/snapshots/`
  - `validation/results/`
  - `validation/logs/`
- 训练过程中可拉起验证子进程
- 训练结束前会等待未完成验证任务

这相当于把“边训练边快速看验证损失”正式接进了训练主流程。

#### 3. `examples/wanvideo/model_training/train.py`

训练入口做了结构化重组：

- 抽出 `build_wan_dataset(...)`
- 抽出 `build_wan_model(...)`
- 抽出 `build_wandb_config(...)`
- 训练启动时把参数落盘到 `args.json`

同时新增验证相关参数：

- `val_dataset_metadata_path`
- `val_every_steps`
- `val_num_batches`
- `val_script_path`
- `val_cuda_visible_devices`

#### 4. `examples/wanvideo/model_training/validate_loss.py`

新增独立验证 worker，用于：

- 从训练目录读取 `args.json`
- 指定 LoRA checkpoint 做快速验证
- 跑有限 batch 的 `val_loss`
- 把结果写回结构化 JSON

它的定位是“训练期间快速损失验证”，不是完整评测脚本。

#### 5. 训练脚本关键配置更新

`train.sh` 与 `examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-Audio.sh` 已提交的关键配置包括：

- 训练超参数改为环境变量可覆盖：
  - `GPU_ID`
  - `NUM_EPOCHS`
  - `SAVE_STEPS`
  - `VAL_EVERY_STEPS`
  - `VAL_NUM_BATCHES`
  - `LEARNING_RATE`
  - `GRADIENT_ACCUMULATION_STEPS`
  - `WANDB_RUN_NAME`
  - `WANDB_TAGS`
- 启动时显式打印 launch command
- 音频 LoRA 训练脚本默认数据切到仓库内 `data/dance_dataset`
- 验证集入口显式加入：
  - `--val_dataset_metadata_path /ssdisk/wm/dance/combined_dataset/metaval.csv`
- `extra_inputs` 改为：
  - `input_image,input_audio`

这批配置更新的核心作用是让训练任务更像“可回放实验配置”而不是一次性命令。

### 三、数据接口与 MERT 适配

#### 1. `data/dance_dataset/metadata.csv`

数据格式已正式切换为：

```csv
image_path,input_audio,video,prompt
```

相较旧格式：

```csv
video,input_audio,text
```

当前数据组织方式已经显式引入参考图，说明训练/推理假设从“纯视频+音频”转向“参考图+音频+视频+文本”。

#### 2. `diffsynth/core/data/operators.py`

`CachedLoadAudio` 新增 `return_sample_rate` 支持：

- 旧接口：`(audio, cache_key)`
- 新接口：`(audio, sample_rate, cache_key)`

这样下游可以显式拿到真实波形采样率，而不是完全依赖默认值。

#### 3. `diffsynth/models/mert_audio_encoder.py`

MERT 路径解析能力增强：

- 若传入完整本地目录，直接使用
- 若目录不完整，明确提示缺失文件
- 若传入 repo id，则尝试从本地缓存中定位：
  - `~/.cache/modelscope/...`
  - `~/.cache/huggingface/...`

这主要是为“模型已下载但脚本仍想联网”的问题兜底。

#### 4. `diffsynth/pipelines/wan_video.py`

已提交的兼容性修复有两点：

- `WanVideoUnit_Audio` 支持显式 sample rate tuple 输入
- `patchify(...)` 兼容：
  - `(tokens, (f, h, w))`
  - 旧版 BCHW latent 返回风格

前者保证 MERT 能拿到正确采样率，后者保证新旧 patchify 实现混用时不直接崩。

#### 5. MERT 关键配置更新

本轮提交后，MERT 音频链路的关键假设更明确了：

- 训练/推理样本允许显式传入参考图
- 音频 loader 可以把真实采样率继续传下游
- 本地 MERT 模型优先从缓存目录解析

另外，`notes/12_ti2v_audio_24khz_mert_fix_2026-05-18.md` 已作为专项修复说明纳入版本库，用于单独追踪 24kHz 路径修复背景。

### 四、Wan 推理脚本 CLI 化

#### 1. `examples/wanvideo/model_inference/Wan2.2-TI2V-5B-Audio.py`

这份脚本已经从 demo 整理为完整 CLI：

- 支持 `--input_image`
- 支持 `--audio_path`
- 支持 `--prompt`
- 支持 `--lora_path`
- 支持 `--lora_alpha`
- 支持 `--save_with_audio`
- 支持低显存和 tiled 参数
- 默认清理 socks 代理环境变量

它现在更适合作为正式实验推理入口，而不是只跑硬编码样例。

#### 2. `examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py`

无音频版 TI2V 脚本也已完成 CLI 化：

- 从硬编码示例改成命令行驱动
- 支持 text-to-video
- 支持 image-to-video
- 支持可选音频回写
- 显式检查本地模型目录与 diffusion shard

#### 3. `examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py`

该脚本改为显式本地路径装配：

- 使用本地 `ModelConfig(path=...)`
- 关闭 `redirect_common_files`

这保证它优先走本地模型文件，不再依赖隐式公共文件重定向。

### 五、SCAIL 批处理与 pose 工具链

#### 1. `batch_extract_scail_pose_and_generate.py`

新增单环境批处理脚本，串联：

- DWPose 检测
- NLF 3D pose 提取
- pose 视频渲染
- SCAIL 最终生成
- `manifest.jsonl` 状态记录

#### 2. `batch_scail_cross_env.py`

新增跨环境 orchestrator，支持：

- `pipeline`
- `dwpose-meta`
- `nlf-only`
- `render-only`
- `generate-only`
- `dry-run`

同时将中间缓存拆分为：

- `dwpose_meta`
- `pose_pkl`
- `rendered_pose`
- `scail_generated`

并增加：

- `--max-frames`
- `--pose-render-mode body_only`
- `manifest.jsonl` 成功/失败/traceback 记录

#### 3. `render_scail_pose_cached.py`

新增缓存消费式 pose 渲染脚本：

- 只读取已有 `poses.pkl / nlf.pkl / meta.json`
- 按参考图重估对齐相机
- 支持 `full` 与 `body_only`

它把“重渲染 pose 可视化”从完整提取流程中独立了出来。

#### 4. `examples/wanvideo/model_inference/run_manifest_pose_inference_v2.py`

新增 manifest 驱动的批量推理入口：

- 以 `manifest.jsonl` 为输入
- 自动识别模型类型：
  - `Fun`
  - `Animate`
  - `S2V`
- 自动读取参考图尺寸和 pose 视频元信息
- 统一批量保存输出

`run_manifest_pose_inference.sh` 则提供了固定 `wan_pose` 环境的 shell 包装入口。

### 六、文档与研究支撑材料

#### 1. `notes/13_commit_breakdown_2026-05-20.md`

新增分批提交说明文档，记录：

- 本轮 6 个 commit 的拆分理由
- 每个 commit 的主题边界
- 适合复用到 PR 的中文说明

它的作用是把“代码历史”转成“可读的研发记录”。

#### 2. 对齐调试与评测文档

已提交的文档补充还包括：

- `docs/plans/2026-04-18-align-debug-design.md`
- `docs/plans/2026-04-18-align-debug-implementation.md`
- `notes/chapter5_experimental_evaluation_and_analysis.md`
- `notes/metric_report_2d_mm_align_av_align_wav2clip.md`

这部分不直接改变运行逻辑，但为后续：

- `align debug` 功能实现
- 指标解释
- 论文/报告写作

提供了统一文字基础。
