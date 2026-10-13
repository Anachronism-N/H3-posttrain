# 脚本逻辑与作用速记

> 适用范围：仓库内用于姿态驱动推理、SCAIL 处理流水线、训练辅助、数据预处理、数据集统计的常用脚本
> 最后更新：2026-05-13

---

## 一、`run_manifest_pose_inference_v2.py`

**位置**：`examples/wanvideo/model_inference/run_manifest_pose_inference_v2.py`

### 1.1 脚本作用

这是一个**批量推理脚本**。它从 `manifest.jsonl` 读取样本，取每条样本中的：

- `rendered_pose_path`：已经渲染好的 pose 控制视频
- `ref_image` 或 `c`：参考人物图
- `video`：原始视频（仅 S2V 分支需要，用来抽音频）

然后根据 `--model_path` 自动判断模型类型，批量生成对应输出视频。

支持三类模型：

- `Wan2.2-Fun-5B-Control`
- `Wan2.2-Animate-14B`
- `Wan2.2-S2V-14B`

模型类型判断见脚本中的 `detect_model_kind(...)`。

### 1.2 输入输出

**输入**：

- 一个 `manifest.jsonl`
- 本地模型目录
- 每条样本对应的参考图、pose 视频，必要时还有原视频

**输出**：

- 输出目录默认是 `outputs/wan_manifest_pose_infer/<模型目录名>/`
- 每条样本输出一个 `stem.mp4`

对应主循环在 `main()` 和 `run_single_row(...)` 中完成。

### 1.3 主流程

整体流程可以概括为：

1. 解析命令行参数，定位 manifest、模型目录、输出目录
2. 从 `manifest.jsonl` 逐行读取样本，只保留 `status=success` 的行
3. 根据模型名选择 `Fun / Animate / S2V` 三个推理分支
4. 对每条样本读取参考图尺寸、pose 视频元信息、帧数
5. 运行对应 pipeline
6. 保存生成视频

其中有两个通用辅助逻辑：

- `get_video_meta(...)`：读取 `fps / frame_count / width / height`
- `align_num_frames(...)`：把帧数对齐到 Wan 推理要求的 `1 + 4k`

### 1.4 三个模型分支分别做什么

#### A. Fun-5B-Control 分支

这一支是 `v2` 相比旧版最重要的变化。

它不再直接走 DiffSynth 里的常规 Wan pipeline，而是：

- 先定位本地 `VideoX-Fun` 仓库根目录
- 从 `VideoX-Fun` 读取 `wan_civitai_5b.yaml`
- 用 `VideoX-Fun` 自己的 transformer / VAE / tokenizer / scheduler 组装 pipeline
- 可选启用 `teacache`、`cfg_skip`、`sequential_cpu_offload`

然后推理时：

- 把参考图转成 latent
- 把 pose 视频转成 `control_video`
- 构造空的 `inpaint_video` 和 `mask_video`
- 调用 `pipe.pipeline(...)` 生成结果

这一支更像是“用 manifest 批量驱动 VideoX-Fun 的 Fun 控制推理”。

#### B. Animate-14B 分支

这一支的作用是做**pose-only animate 推理**。

处理方式是：

- 读取参考图作为 `input_image`
- 读取 pose 视频前 `num_frames - 4` 帧作为 `animate_pose_video`
- 构造一个全黑的假 `face video`

脚本里还对 animate adapter 做了一个 patch，使它只吃 pose 条件，不走 face 条件的真实输入。这一点通过 `patch_pose_only_animate_adapter(...)` 完成。

#### C. S2V-14B 分支

这一支除了参考图和 pose 视频，还会额外使用音频：

- 从 `row["video"]` 指向的原视频中提取 wav
- 用 `librosa` 读成波形
- 作为 `input_audio`
- 与 `s2v_pose_video` 一起送入 S2V pipeline

最后保存为**带音频的视频**。

### 1.5 `v2` 的特点

相对旧版思路，`v2` 的关键特征是：

- `Fun-5B-Control` 改成依赖本地 `VideoX-Fun` 仓库
- 脚本会自动尝试在 `~/videoFun`、`~/VideoFun`、`~/VideoX-Fun` 下寻找该仓库
- 更贴近 `VideoX-Fun` 原生推理栈，而不是完全走 DiffSynth 封装

因此，如果你主要跑 `Fun-5B-Control`，这个脚本本质上是一个**批量推理桥接器**。

### 1.6 适用场景

适合：

- 已经有 `manifest.jsonl`
- 已经有 `rendered_pose_path`
- 想批量比较不同 Wan 模型在同一批 pose 条件上的结果

不适合：

- 从零生成 pose
- 做训练数据预处理
- 统计整个数据集的视频分布

### 1.7 补充入口脚本

项目根目录里的 `run_manifest_pose_inference.sh` 只是一个**薄包装脚本**：

- 它会定位 `examples/wanvideo/model_inference/run_manifest_pose_inference_v2.py`
- 默认用 `conda run -n wan_pose python ...` 启动
- 主要作用是减少环境切换时手动输长命令

因此，真正的业务逻辑在 `run_manifest_pose_inference_v2.py`，而 `.sh` 只是启动入口

---

## 二、`video_pre.py`

**位置**：`video_pre.py`

### 2.1 脚本作用

这是一个**离线预处理脚本**，用于把：

- 参考图 `eval/img_896-512/*.png`
- 动作 `eval/pkl/*.pkl`
- 音频或同名视频 `eval/wav/*.wav / *.mp4`

整理成可以后续评测或批量推理使用的**定长分段视频数据**。

它不是模型推理脚本，而是数据准备脚本。

### 2.2 输入输出

**输入目录**：

- `eval/img_896-512`
- `eval/pkl`
- `eval/wav`

**输出目录**：

- `eval/scail/pkl_aligned/`
- `eval/scail/wav_16fps/`
- `eval/scail/pkl_aligned_113/`
- `eval/scail/segments.csv`

### 2.3 处理逻辑

脚本按排序后的文件名，把参考图和 pkl 一一配对。

对每一对 `(ref_img, pkl)`，依次做下面几步：

1. 检查 `eval/wav/<name>.wav` 是否存在
2. 如果没有 wav，就从 `eval/wav/<name>.mp4` 里抽出单声道 16k wav
3. 调 `SCAIL/scail_pose/pkl_align.py`
4. 生成与参考图对齐的人体动作视频 `pkl_aligned/*.mp4`
5. 把该视频和音频重新 mux 成 `16fps` 视频
6. 用 `ffprobe` 统计总帧数
7. 每 `113` 帧切一段，保存到 `pkl_aligned_113/`
8. 把 `ref` 和 `segment_path` 记入 `segments.csv`

### 2.4 核心参数含义

脚本顶部有几组固定配置：

- `pkl_fps = 30`：`pkl_align.py` 输出动作视频时使用的 fps
- `fps = 16`：后续统一重采样的目标视频 fps
- `seg_frames = 113`：每个切分片段固定帧数
- `seg_duration = seg_frames / fps`：每段时长，约 `7.0625s`

所以这个脚本最终产出的不是原始整段视频，而是：

- **16fps**
- **每段 113 帧**
- **带音频**

的一批切片。

### 2.5 这个脚本解决的问题

它主要解决三个问题：

- 把 `pkl` 动作序列变成可视化视频
- 把音频和动作视频重新对齐、统一采样率/帧率
- 把长视频切成固定长度的小段，方便后续统一处理

### 2.6 局限

需要注意：

- 它假设 `ref image` 和 `pkl` 按排序后一一对应
- 不做全排列匹配
- 目录路径是写死的，不是通用 CLI 工具
- 只面向当前 `eval/` 目录结构

因此它更像是**项目内的专项预处理流水线**，不是通用脚本模板。

---

## 三、`check_video.py`

**位置**：`check_video.py`

### 3.1 脚本作用

这是一个**数据集视频统计脚本**，用于快速统计某个视频目录里的：

- 分辨率
- 帧率
- 每种 `(宽, 高, fps)` 组合出现了多少次

它适合做一个“数据集概览”，不负责生成新文件。

### 3.2 处理逻辑

脚本会：

1. 遍历 `VIDEO_DIR` 目录下所有 `.mp4`
2. 用 OpenCV 打开视频
3. 读取 `width / height / fps`
4. 把 `fps` 四舍五入到 2 位小数
5. 用 `(width, height, fps)` 作为 key 统计数量
6. 最后按表格形式打印结果

### 3.3 输出长什么样

它输出的是汇总表，不是逐文件明细。类似：

```text
分辨率      帧率     数量
1920x1080   59.94    120
1080x1920   29.97    45
```

### 3.4 适用场景

适合：

- 快速看一个目录中视频分辨率是否统一
- 判断帧率是否混杂
- 为训练前的数据分析做粗统计

不适合：

- 查询单个视频的完整元信息
- 输出时长、总帧数、编码格式
- 生成 CSV 或更详细的报告

### 3.5 局限

当前脚本有几个明显限制：

- `VIDEO_DIR` 是写死路径
- 只统计 `.mp4`
- 不输出单个文件名
- 不输出时长和总帧数

所以它定位上是一个**快速检查脚本**，不是完整的元信息分析工具。

---

## 四、`batch_extract_scail_pose_and_generate.py`

**位置**：`batch_extract_scail_pose_and_generate.py`

### 4.1 脚本作用

这是一个**单环境的一条龙 SCAIL 批处理脚本**。它把一批原始驱动视频和同名参考图，直接处理成：

- `pose_pkl/*.pkl`
- `rendered_pose/*_rendered.mp4`
- `scail_generated/*.mp4`
- `manifest.jsonl`

换句话说，它把「提 3D pose -> 渲染 pose 视频 -> 调 SCAIL 生成」串成了一条顺序流水线。

### 4.2 输入输出

**输入**：

- 一个视频目录 `--input-dir`
- 一个参考图目录 `--ref-dir`
- SCAIL / NLF 相关模型路径

**输出**：

- `output-root/pose_pkl/`
- `output-root/rendered_pose/`
- `output-root/scail_generated/`
- `output-root/manifest.jsonl`

### 4.3 主流程

主循环在 `main()` 里完成，每个样本大致经历：

1. 根据 stem 找到驱动视频和参考图
2. 用 `DWposeDetector` 跑逐帧 2D pose
3. 用 `process_video_nlf(...)` 提取 3D NLF pose，并保存成 `pkl`
4. 结合参考图做相机对齐与 pose 重渲染
5. 调 `SCAIL/generate.py` 生成最终视频
6. 把成功或失败状态写入 `manifest.jsonl`

### 4.4 关键开关

- `--only-extract-render`：只做 pose 提取和渲染，不跑生成
- `--only-generate`：直接消费已有 `rendered_pose`，跳过前处理
- `--skip-existing`：如果最终视频已存在就跳过
- `--no-align`：渲染 pose 时不做参考图对齐
- `--dry-run`：只打印输入输出映射，不做 GPU 计算

### 4.5 适用场景

适合：

- 在**同一个环境**里快速跑通 SCAIL 全流程
- 想顺手产出 `manifest.jsonl` 方便后续复用
- 需要对每个样本保留 `pkl / pose video / generated video`

不适合：

- DWPose / NLF / 生成依赖分散在多个 conda 环境时
- 想把各阶段缓存拆开复用时

---

## 五、`batch_scail_cross_env.py`

**位置**：`batch_scail_cross_env.py`

### 5.1 脚本作用

这是一个**跨环境版 SCAIL 编排脚本**。它把单环境脚本拆成多个阶段，分别放到不同 conda 环境执行，用来解决：

- `DWpose`、`NLF`、`SCAIL generate` 依赖不在同一环境
- 某些阶段想单独重跑
- 已经缓存了中间结果，不想每次全流程重来

### 5.2 支持的模式

脚本支持这些 `--mode`：

- `pipeline`：按 `dwpose-meta -> nlf-only -> render-only -> generate-only` 全流程串起来
- `dwpose-meta`：只提 2D pose 和 bbox 元数据
- `nlf-only`：只根据缓存 bbox 跑 NLF 3D pose
- `render-only`：只根据缓存 pose 渲染视频
- `generate-only`：只调用 SCAIL 生成
- `dry-run`：只打印样本与路径

### 5.3 它和单环境版的主要区别

与 `batch_extract_scail_pose_and_generate.py` 相比，它的关键特征是：

- 把 `DWpose`、`NLF`、`generate` 分别丢到 `--dwpose-env`、`--nlf-env`、`--generate-env`
- 额外缓存 `dwpose_meta/*.json`、`*_poses.pkl`、`*_bboxes.pkl`
- 渲染阶段默认调用 `render_scail_pose_cached.py`，避免重复做前两步
- 支持 `--max-frames` 和 `--pose-render-mode body_only`

### 5.4 输出组织

它会把中间结果拆得更细：

- `dwpose_meta/`
- `pose_pkl/`
- `rendered_pose/`
- `scail_generated/`
- `manifest.jsonl`

这意味着它更像一个**可断点续跑的批处理 orchestrator**，而不是单个推理脚本。

### 5.5 适用场景

适合：

- 机器上依赖环境分裂明显
- 需要多次重跑某个阶段
- 想保留 `dwpose` 与 `nlf` 缓存做后续对照实验

不适合：

- 只是想最快速跑通一次 demo

---

## 六、`render_scail_pose_cached.py`

**位置**：`render_scail_pose_cached.py`

### 6.1 脚本作用

这是一个**只负责渲染 pose 视频的缓存消费脚本**。它不再做 DWPose 检测或 NLF 提取，而是直接读取已有缓存：

- `dwpose_meta/<stem>.json`
- `dwpose_meta/<stem>_poses.pkl`
- `pose_pkl/<stem>.pkl`

然后把它们渲染成 `rendered_pose/*.mp4`。

### 6.2 处理逻辑

脚本大致做下面几件事：

1. 根据 `stem` 找到元数据、2D pose 和 3D pose 缓存
2. 读取参考图，并推导目标分辨率
3. 如果不开 `--no-align`，就用参考图重新估计相机参数
4. 用 `render_nlf_as_images(...)` 分块渲染
5. 通过 OpenCV `VideoWriter` 写成最终 mp4

### 6.3 为什么单独拆这个脚本

拆出来的直接好处是：

- 调相机对齐策略时，只需要重跑渲染
- 换 `pose_render_mode=body_only` 时，不用重新提 3D pose
- 可以让 `batch_scail_cross_env.py` 在 `wan_pose` 环境里专门调用它

### 6.4 特点

- 支持 `--chunk-size` 分块渲染，避免一次性堆太多帧
- 支持 `--pose-render-mode full/body_only`
- 支持 `--no-align`
- 已有输出且没传 `--overwrite` 时会直接跳过

---

## 七、`validate_loss.py`

**位置**：`examples/wanvideo/model_training/validate_loss.py`

### 7.1 脚本作用

这是一个**异步验证损失脚本**。它的职责不是训练，而是：

- 读取训练时保存下来的 `args.json`
- 加载某个 LoRA checkpoint
- 构建验证集 dataloader
- 跑若干个 batch 的 `val_loss`
- 把结果写到一个 `result.json`

### 7.2 它在整体实验里的位置

这个脚本通常不是手动单独使用，而是被训练过程中的异步验证逻辑拉起。也就是说：

- 训练主进程负责正常反向传播
- 到达 `val_every_steps` 时，额外启动这个脚本
- 这个脚本读 checkpoint，单独计算验证损失
- 训练结束前再统一等待这些验证任务收尾

### 7.3 输入输出

**输入**：

- `--train_args_path`
- `--lora_checkpoint`
- `--val_dataset_metadata_path`
- `--step`

**输出**：

- `--result_path` 指向的 JSON，里面至少有 `step / val_loss / val_batches / val_time`

### 7.4 适用场景

适合：

- 训练过程中想观察验证集损失趋势
- 不想把验证直接塞进主训练 loop 里拖慢吞吐

不适合：

- 做完整评测指标
- 代替正式推理评测脚本

---

## 八、怎么选用

如果目标是：

- **批量跑姿态驱动生成**：用 `run_manifest_pose_inference_v2.py`
- **不想手动切环境，直接把 SCAIL 全流程跑完**：用 `batch_extract_scail_pose_and_generate.py`
- **需要把 DWPose / NLF / 渲染 / 生成拆开，并跨 conda 环境执行**：用 `batch_scail_cross_env.py`
- **已经有 `poses.pkl` 和 `nlf.pkl`，只想重渲染 pose 视频**：用 `render_scail_pose_cached.py`
- **训练期间异步看验证损失**：用 `validate_loss.py`
- **把 pkl/音频/参考图整理成分段视频数据**：用 `video_pre.py`
- **先看视频目录的分辨率和帧率分布**：用 `check_video.py`

一句话总结：

- `run_manifest_pose_inference_v2.py` 是“批量生成”
- `batch_extract_scail_pose_and_generate.py` 是“单环境全流程”
- `batch_scail_cross_env.py` 是“跨环境编排”
- `render_scail_pose_cached.py` 是“只重渲染 pose”
- `validate_loss.py` 是“训练期异步验证”
- `video_pre.py` 是“离线整理数据”
- `check_video.py` 是“快速统计数据集”
