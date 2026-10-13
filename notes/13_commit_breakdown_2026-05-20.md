# 2026-05-20 分批提交说明

> 目的：对 2026-05-20 整理出的 6 个 commit 做结构化中文说明，便于后续撰写 PR 描述、回顾开发脉络、判断是否需要继续拆分。
>
> 范围：仅覆盖已经提交的 6 个 commit，不包含当前工作区中仍未提交的 `README.md`、`bad_data.txt`、`Wan2.2-Animate-14B_pose_only.py`、`notes/03_implementation_changelog.md` 以及本地实验目录。

---

## 总览

本轮提交共拆分为 6 个 commit：

| 顺序 | Commit | 标题 | 主题 |
|---|---|---|---|
| 1 | `0562225` | `chore: ignore generated media artifacts` | 清理生成媒体产物 |
| 2 | `4682462` | `feat: add async validation to Wan audio training` | Wan 音频训练异步验证与日志增强 |
| 3 | `8b43e35` | `feat: adapt Wan TI2V data flow for MERT audio inputs` | 数据接口与 MERT 适配 |
| 4 | `f672300` | `feat: convert Wan TI2V inference scripts into CLIs` | Wan TI2V 推理脚本 CLI 化 |
| 5 | `872c22f` | `feat: add SCAIL batch pose generation tooling` | SCAIL 批处理、pose 渲染与 manifest 推理工具链 |
| 6 | `2e83131` | `docs: add align debug and evaluation notes` | 对齐调试设计文档与评测说明文档 |

这 6 个 commit 的拆分原则是：

- 先把产物清理与忽略规则独立出去，避免后续功能 commit 混入二进制垃圾
- 将训练主链路作为独立功能提交
- 将数据与 MERT 适配、推理脚本 CLI 化进一步拆开
- 将 SCAIL 工具链集中整理
- 将纯文档内容单独提交，避免与代码逻辑耦合

---

## Commit 1

### `0562225` `chore: ignore generated media artifacts`

### 提交目的

将仓库中的“生成结果”和“中间媒体产物”从版本控制中剥离，减少后续实验过程对工作区的污染。

### 涉及文件

- `.gitignore`
- `video_1_Wan2.2-S2V-14B-1.mp4`
- `video_1_Wan2.2-S2V-14B.mp4`
- `video_1_Wan2.2-TI2V-5B.mp4`

### 主要改动

- 在 `.gitignore` 中新增以下忽略规则：
  - `*.mp4`
  - `*.wav`
  - `*.png`
  - `*.jpg`
  - `*.zip`
- 删除 3 个已被跟踪的推理输出视频

### 设计意图

该提交不改变任何训练、推理或数据处理逻辑，只负责清理仓库边界。其作用是：

- 防止后续推理输出再次进入 `git status`
- 保持代码提交与实验结果解耦
- 为后续功能提交提供更干净的 diff 基线

### 适用的 PR 说明

本提交用于清理仓库中误跟踪的推理结果文件，并补充常见媒体产物的忽略规则，避免实验输出继续污染版本历史。

---

## Commit 2

### `4682462` `feat: add async validation to Wan audio training`

### 提交目的

为 `Wan2.2-TI2V-5B-Audio` 训练链路补齐“异步验证、结构化日志、参数留档、可回放脚本入口”能力，使训练任务不再只是一次性命令，而是可追踪、可复现实验流程。

### 涉及文件

- `diffsynth/diffusion/logger.py`
- `diffsynth/diffusion/runner.py`
- `examples/wanvideo/model_training/train.py`
- `examples/wanvideo/model_training/validate_loss.py`
- `examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-Audio.sh`
- `train.sh`

### 主要改动

#### 1. 训练日志增强

`diffsynth/diffusion/logger.py` 从以 `wandb` 为主的记录方式扩展为“`wandb + 本地文件`”双通道：

- 新增 `train.log`
- 新增 `metrics.jsonl`
- 在 step 级别记录：
  - `loss`
  - `lr`
  - `grad_norm`
  - `step_time`
  - `data_time`
  - `epoch`
- 新增验证完成/验证失败事件记录
- 在保存 checkpoint、结束 epoch、训练结束时写入明确日志事件

#### 2. 异步验证机制

`diffsynth/diffusion/runner.py` 增加训练期间的异步验证支持：

- 新增 `compute_grad_norm(...)`
- 新增 `is_async_validation_enabled(...)`
- 统一构造验证路径：
  - `validation/snapshots/`
  - `validation/results/`
  - `validation/logs/`
- 在训练过程中以子进程方式启动验证 worker
- 训练结束前等待未完成验证任务，避免尾部验证结果丢失

#### 3. 训练入口结构化

`examples/wanvideo/model_training/train.py` 进行了结构整理：

- 抽出：
  - `build_wan_dataset(...)`
  - `build_wan_model(...)`
  - `build_wandb_config(...)`
- 新增验证参数：
  - `val_dataset_metadata_path`
  - `val_every_steps`
  - `val_num_batches`
  - `val_script_path`
  - `val_cuda_visible_devices`
- 训练开始时写出 `args.json`
- 对 `prompt` / `text` 做统一解析，避免数据列不一致导致 prompt 为空

#### 4. 新增验证 worker

新增 `examples/wanvideo/model_training/validate_loss.py`：

- 从 `args.json` 恢复训练配置
- 加载指定 LoRA checkpoint
- 复用训练侧数据集与模型构建逻辑
- 在有限 batch 上计算 `val_loss`
- 输出结构化验证结果 JSON

#### 5. 启动脚本参数化

`Wan2.2-TI2V-5B-Audio.sh` 与 `train.sh` 做了入口整理：

- 训练超参数支持环境变量覆盖
- 显式打印关键配置
- 显式打印 launch command
- 将验证相关参数纳入脚本入口

### 设计意图

本提交的核心是把训练流程从“直接启动”提升为“实验系统”：

- 可以回看训练过程
- 可以在训练期间插入快速验证
- 可以在训练目录中保留完整实验配置
- 可以在脚本层面对多次运行保持一致的参数管理方式

### 边界说明

本提交主要聚焦训练主链路，不包含推理脚本改造，也不处理外部评测逻辑。

### 适用的 PR 说明

本提交为 Wan 音频训练流程新增异步验证与本地结构化日志能力，同时整理训练入口脚本与参数留档逻辑，使实验具备更好的可追踪性和可复现性。

---

## Commit 3

### `8b43e35` `feat: adapt Wan TI2V data flow for MERT audio inputs`

### 提交目的

将 TI2V 音频训练/推理链路中的数据接口与 MERT 输入约束对齐，确保参考图、显式采样率和本地 MERT 缓存路径能够稳定进入模型数据流。

### 涉及文件

- `data/dance_dataset/metadata.csv`
- `diffsynth/core/data/operators.py`
- `diffsynth/models/mert_audio_encoder.py`
- `diffsynth/pipelines/wan_video.py`
- `notes/12_ti2v_audio_24khz_mert_fix_2026-05-18.md`

### 主要改动

#### 1. 数据接口更新

`data/dance_dataset/metadata.csv` 从旧格式：

```csv
video,input_audio,text
```

更新为新格式：

```csv
image_path,input_audio,video,prompt
```

这表明当前训练/推理样本组织方式已转向显式参考图输入。

#### 2. 音频缓存加载器增强

`diffsynth/core/data/operators.py` 中的 `CachedLoadAudio` 新增 `return_sample_rate`：

- 兼容原先的 `(audio, cache_key)` 返回值
- 支持新的 `(audio, sample_rate, cache_key)` 返回值

该改动为 MERT 路径显式传递真实采样率提供了接口基础。

#### 3. MERT 本地路径解析增强

`diffsynth/models/mert_audio_encoder.py` 增加模型路径解析逻辑：

- 若传入完整本地目录，直接使用
- 若传入不完整目录，明确报出缺失文件
- 若传入 repo id，则尝试从：
  - `~/.cache/modelscope/...`
  - `~/.cache/huggingface/...`
  中查找本地缓存

目标是减少“本地已有模型但脚本仍尝试联网”的问题。

#### 4. Pipeline 输入兼容性修复

`diffsynth/pipelines/wan_video.py` 的改动包括：

- `WanVideoUnit_Audio` 支持显式 sample rate 的 tuple 输入
- `patchify(...)` 兼容两种返回风格：
  - 新实现直接返回 `(tokens, (f, h, w))`
  - 旧实现仍返回 BCHW latent

这提高了新旧实现混用时的稳健性，并使 MERT 路径能够拿到真实波形采样率。

#### 5. 修复说明文档

新增 `notes/12_ti2v_audio_24khz_mert_fix_2026-05-18.md`，用于记录：

- 24kHz MERT 输入的背景
- 音频缓存目录调整
- 显式 sample rate 透传的必要性
- 修复后的预期行为

### 设计意图

本提交本质上解决三类问题：

- 数据格式从“视频+音频”转向“参考图+音频+视频+文本”
- 本地模型与本地缓存路径解析更稳
- 数据流中显式保留 MERT 所需采样率，避免静默退化

### 边界说明

本提交不包含推理脚本 CLI 化改造，只聚焦“数据与 MERT 适配”这一条链路。

### 适用的 PR 说明

本提交对齐了 Wan TI2V 音频链路中的数据格式、音频缓存返回值与 MERT 本地加载路径，使 24kHz MERT 输入假设能够在训练/推理数据流中被显式保留。

---

## Commit 4

### `f672300` `feat: convert Wan TI2V inference scripts into CLIs`

### 提交目的

将 Wan TI2V 系列推理脚本从硬编码示例整理为可复用 CLI，统一本地模型装配方式、输入参数接口和保存行为，使其更适合作为正式实验入口。

### 涉及文件

- `examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py`
- `examples/wanvideo/model_inference/Wan2.2-TI2V-5B-Audio.py`
- `examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py`

### 主要改动

#### 1. TI2V-Audio 推理脚本 CLI 化

`examples/wanvideo/model_inference/Wan2.2-TI2V-5B-Audio.py` 从 demo 形式升级为完整 CLI：

- 支持 `--input_image`
- 支持 `--audio_path`
- 支持 `--prompt`
- 支持 `--lora_path`
- 支持 `--lora_alpha`
- 支持 `--save_with_audio`
- 支持低显存参数与 tiled 开关
- 清理 socks 代理环境变量，减少本地加载异常

#### 2. TI2V 推理脚本 CLI 化

`examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py` 也做了同方向整理：

- 从硬编码样例改为 CLI 入口
- 兼容纯 prompt 和 image-to-video
- 可选 mux 回原音频
- 显式检查本地模型目录与权重文件

#### 3. VACE 本地模型装配

`examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py` 改为显式本地文件装配，并关闭 `redirect_common_files`，避免默认回退到远程或共享路径解析。

### 设计意图

本提交主要解决的是“推理入口工程化”问题：

- 让脚本不再依赖硬编码样例路径
- 让本地模型装配方式更统一
- 让 TI2V 和 TI2V-Audio 的使用方式更接近

### 边界说明

本提交只覆盖推理脚本 CLI 化，不再包含数据列格式调整或 MERT 数据流适配逻辑。

### 适用的 PR 说明

本提交将 Wan TI2V、TI2V-Audio 和 VACE 推理脚本整理为可复用 CLI，并统一本地模型加载、输入参数组织和输出保存方式，使其更适合作为正式实验入口。

---

## Commit 5

### `872c22f` `feat: add SCAIL batch pose generation tooling`

### 提交目的

为 SCAIL 相关实验补齐批处理工具链，包括单环境批处理、跨环境 orchestrator、缓存 pose 重渲染，以及基于 manifest 的 Wan 模型批量推理入口。

### 涉及文件

- `batch_extract_scail_pose_and_generate.py`
- `batch_scail_cross_env.py`
- `render_scail_pose_cached.py`
- `examples/wanvideo/model_inference/run_manifest_pose_inference_v2.py`
- `run_manifest_pose_inference.sh`
- `notes/11_script_logic_reference.md`

### 主要改动

#### 1. 单环境批处理流水线

新增 `batch_extract_scail_pose_and_generate.py`，串联以下流程：

- DWPose 检测
- NLF 3D pose 提取
- pose 视频渲染
- SCAIL 最终生成

适用于依赖可放在同一环境中的快速批处理场景。

#### 2. 跨环境 orchestrator

新增 `batch_scail_cross_env.py`，支持在不同 conda 环境间拆阶段执行：

- `pipeline`
- `dwpose-meta`
- `nlf-only`
- `render-only`
- `generate-only`
- `dry-run`

同时对中间缓存进行显式分层：

- `dwpose_meta`
- `pose_pkl`
- `rendered_pose`
- `scail_generated`

并通过 `manifest.jsonl` 持久化记录状态、错误与 traceback。

#### 3. Pose 缓存重渲染

新增 `render_scail_pose_cached.py`，使已有缓存可以独立重渲染：

- 读取已有 `poses.pkl / nlf.pkl / meta.json`
- 重新估计对齐相机
- 输出 pose 视频
- 支持 `full` 与 `body_only`

这样可以避免为了改 pose 可视化而重复跑完整提取链。

#### 4. 基于 manifest 的推理入口

新增 `examples/wanvideo/model_inference/run_manifest_pose_inference_v2.py`：

- 以 `manifest.jsonl` 为输入
- 自动识别模型类型：
  - `Fun`
  - `Animate`
  - `S2V`
- 自动读取参考图尺寸与 pose 视频元信息
- 统一批量推理保存输出

`run_manifest_pose_inference.sh` 则提供固定 `wan_pose` 环境的 shell 入口。

#### 5. 工具链说明文档

新增 `notes/11_script_logic_reference.md`，用于记录脚本职责边界、主流程和分支逻辑，降低后续维护成本。

### 设计意图

该提交的核心价值是把原本分散的实验脚本整理成可复用工具链：

- 可以批量处理样本
- 可以在多环境下稳定分阶段运行
- 可以复用中间缓存
- 可以统一不同 Wan pose 模型的批量推理方式

### 边界说明

这是本轮提交中体量最大的一个 commit，全部为新增文件，适合内部实验整理；若未来需要对外发 PR，可考虑再拆为：

- manifest 推理工具
- SCAIL pose 批处理工具
- 工具链文档

### 适用的 PR 说明

本提交新增了一组围绕 SCAIL pose 提取、缓存重渲染和 manifest 批量推理的实验工具，用于提升跨环境批处理和多模型对照实验的执行效率。

---

## Commit 6

### `2e83131` `docs: add align debug and evaluation notes`

### 提交目的

补充对齐调试设计文档与实验评测说明文档，为后续实现 `align debug` 能力和撰写论文/报告提供文字基础。

### 涉及文件

- `docs/plans/2026-04-18-align-debug-design.md`
- `docs/plans/2026-04-18-align-debug-implementation.md`
- `notes/chapter5_experimental_evaluation_and_analysis.md`
- `notes/metric_report_2d_mm_align_av_align_wav2clip.md`

### 主要改动

#### 1. Align Debug 设计文档

`docs/plans/2026-04-18-align-debug-design.md` 说明：

- 何时启用 debug 输出
- 输出哪些中间文件
- 指标 JSON 包含哪些字段
- 控制台打印哪些关键信息

其目标是为 `SCAIL/scail_pose` 中的 `align` 推理路径增加可视化与可诊断能力。

#### 2. Align Debug 实现计划

`docs/plans/2026-04-18-align-debug-implementation.md` 给出更细的实现拆解：

- helper 职责
- 测试优先策略
- 分阶段任务拆分
- 计划中的文件路径与测试覆盖点

#### 3. 实验评估章节草稿

`notes/chapter5_experimental_evaluation_and_analysis.md` 为论文或报告中的实验章节草稿，覆盖：

- 实验目标
- 数据与对比对象
- 视觉保真度指标
- 音乐对齐与 GT 参照分析
- 初步结论表述

#### 4. 指标说明文档

`notes/metric_report_2d_mm_align_av_align_wav2clip.md` 对以下指标进行系统说明：

- `2D-MM Align`
- `AV-Align`
- `Wav2CLIP-CS`

内容包括：

- 指标本质
- 趋势方向
- 典型值域
- 与论文实现的一致性说明
- 使用建议

### 设计意图

该提交不引入任何运行逻辑变化，而是补足研究与实现之间的文档桥梁：

- 为后续编码提供计划依据
- 为实验结果撰写提供统一表述
- 为指标解释提供对外口径

### 适用的 PR 说明

本提交新增对齐调试的设计与实现计划文档，并补充实验评估章节草稿和指标说明文档，为后续实现与论文写作提供统一参考。

---

## 结论

从提交质量看，本轮 6 个 commit 可以分成三类：

### 1. 边界清晰、可直接保留

- `0562225`
- `4682462`
- `8b43e35`
- `f672300`
- `2e83131`

这五者主题边界明确，后续无论继续开发还是对外同步，都比较适合直接保留。

### 2. 主题成立，但未来可视需要再拆

- `872c22f`

其中：

- `872c22f` 体量较大，包含多个新增脚本和一份说明文档

它在当前内部研发节奏下是合理的，但若以后要做更精细的 PR 审查，可以进一步细分。

### 3. 当前仍未纳入本轮提交的内容

以下改动刻意未纳入本说明对应的 6 个 commit：

- `README.md`
- `bad_data.txt`
- `examples/wanvideo/model_inference/Wan2.2-Animate-14B_pose_only.py`
- `notes/03_implementation_changelog.md`
- `.codex`
- `HuMo/`
- `SCAIL/`
- `SF/`
- `check_link_down.sh`
- `extract_first_frame.sh`
- `logs/`
- `outputs/`

保留这些内容在工作区的原因是：

- 一部分属于本地实验路径或运行产物
- 一部分属于外部仓库目录
- 一部分改动体量过大或主题边界不清，适合后续单独判断
