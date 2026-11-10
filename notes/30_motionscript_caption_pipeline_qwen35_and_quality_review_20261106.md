# MotionScript 全自动打标 Pipeline（Qwen3.5-397B-A17B）与质量复盘

本文档记录 `ms_original_planner_dataset_v1` 全量打标（22 万条）的工程过程与**事后质量复盘**：
服务部署 → 批处理打标 → 生产问题与优化 → 缺口补跑 → 最终产物 → **质量缺陷归因与筛检方案**。

> **重要结论先行**：覆盖率 100% 只证明"每条都有输出"，**不证明润色到位**。
> 事后复查发现 **约 97% 的记录实际上是关节句直译，几乎没有自然语言润色**。
> 第 9、10 节是本文最重要的部分，说明了验收标准为何失效、缺陷如何量化、以及如何快速筛出可用样本。

---

## 1. 目标与约束

### 1.1 任务

利用已部署的 Qwen3.5-397B-A17B 多模态推理实例，对原版 MotionScript（按时间窗切分的关节级动作描述）做：

1. **自然语言表达改善**：把"一侧手腕相对肩部向画面左侧移动"式的关节流水句改写成连贯的动作描述；
2. **有视觉证据的动作补充**：从视频中补充躯干转向、步法、身体位移、完整手臂轨迹、姿势间过渡等源文本遗漏的信息；
3. **有证据的纠正**：只有在视频证据明确时才修正源文本错误。

### 1.2 硬约束

| 约束 | 说明 |
|---|---|
| 事件保真 | 源事件的部位、侧别代称、画面方向、肩部相对关系、幅度、时序、同时性、反转、重复动作必须保留 |
| 窗口不变 | 保留源窗口 id / 顺序 / start / end，不得静默裁剪或重分配事件 |
| 不臆造 | 不编造命名舞步、次数、手型、接触、支撑腿、精确踢腿高度、画面外动作；遮挡部位记为未知而非静止 |
| 纯动作 | 正文不得包含服装、背景、镜头、音乐、关键点、置信度、源 id、帧号、编辑评论 |
| 格式 | `Dance style: <style>\n[start-end s] <caption>\n...` |
| 输出 | 仅 JSON，无 markdown、无思维链 |

### 1.3 输入数据

`ms_original_planner_dataset_v1/{train,val,test}.jsonl`，每条记录：

```json
{
  "id": "...",
  "source_group_id": "...",
  "split": "train",
  "target_text": "Dance style: contemporary\n[0.000-2.480s] 一侧手腕...",
  "target_windows": [{"start": 0.0, "end": 2.48, "caption": "一侧手腕..."}],
  "caption_version": "ms-original-style-v1",
  "video_path": "...",
  "style_metadata_not_in_target": {"dance_style": "contemporary", "style_source": "..."}
}
```

视频以 **53 个 tar 分片**管理（`part_0000.tar` … `part_0052.tar`，每片约 5,000 个 `<sample_id>.mp4`，单条约 1.2 MB）。打标侧只依赖 tar。

---

## 2. 服务部署（星云 + vLLM）

### 2.1 环境事实

| 项 | 值 |
|---|---|
| 集群 / 队列 | `asi_ap-southeast-1_lj_01` / `future_l20a_videope`（项目 `futurelab_vlm_caption`） |
| 节点 | L20A（Blackwell SM100），**aarch64**，4 GPU/节点，单卡显存 185–198 GB |
| 调度粒度 | `--worker_gpu` 以 1/100 卡为单位：`400` = 4 卡，`800` = 8 卡 |
| 服务镜像 | `mdl0.6:user_cu128_gcc13_py3.10.13_..._arm64` |

### 2.2 依赖组合（踩坑后的稳定版本）

```
torch==2.13.0+cu130 (download.pytorch.org/whl/cu130)
transformers==5.17.0
vllm==0.31.0 (--no-deps)
flashinfer-python==0.7.0.post1, flashinfer-cubin==0.6.13
quack-kernels==0.6.5, nvidia-cutlass-dsl==4.7.1
compressed-tensors==0.17.0, xgrammar==0.2.7, outlines_core==0.2.14
llguidance>=1.7,<1.8, model-hosting-container-standards==0.1.16
nvidia-cuda-nvcc==13.0.88
```

关键环境变量：

```bash
VLLM_USE_DEEP_GEMM=0
VLLM_MOE_USE_DEEP_GEMM=0
VLLM_USE_FLASHINFER_SAMPLER=0     # 规避 sampler JIT
FLASHINFER_DISABLE_VERSION_CHECK=1
CUDA_HOME=<site-packages>/nvidia/cu13   # 并把它加入 PATH 最前、剔除系统 /usr/local/cuda
```

### 2.3 启动参数

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /checkpoint/qwen35_model_extract/model \
  --served-model-name Qwen3.5-397B-A17B \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --max-model-len 16384 \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 \
  --dtype auto \
  --mm-encoder-attn-backend TORCH_SDPA \
  --attention-backend TRITON_ATTN \
  --moe-backend triton
```

要点：

- Qwen3.5 使用 **GDN 线性注意力 + 标准注意力**（非 MLA），GDN prefill 依赖 flashinfer 的 `gdn_prefill`，必须 0.7+；
- DeepGEMM 需要 NVCC ≥ 12.9，镜像不满足 → 全关，MoE 走 triton；
- FP8 KV cache 是 32 并发不 OOM 的关键；
- 上下文 16384 同时约束输入（视频 token）与输出（`max_tokens`）。

### 2.4 模型分发

- FP8 权重约 406 GB，打包为单 tar：`oss://futurelab-cognwm-sg/niezichao.nzc/qwen35_fp8_model.tar`；
- 容器启动先下载 tar → 解压到 `/checkpoint/qwen35_model_extract/model`；
- 早期按 107 个对象并发下载（16 线程），受带宽限制约 45 MB/s（≈1 小时）；改为单 tar 后显著更快，并在多副本同时 staging 时**主动停掉多余副本集中带宽**。

### 2.5 凭证

星云会注入 STS 环境变量，但不具备目标 bucket 权限。因此启动脚本**优先读取随代码打包的 `oss.properties`**，读不到才回退环境变量。**任何凭证都不写入代码、日志或本文档。**

---

## 3. 打标主流程

### 3.1 单条样本的处理（`m2d/annotate_pipeline.py`）

```
planner 记录
   │  build_user_prompt(record)
   ▼
Qwen3.5-397B-A17B（system=SYSTEM_PROMPT，user=prompt + video_url data URI）
   │  response_format={"type":"json_object"}，temperature=0.2，max_tokens=8192
   ▼
parse_model_json  →  validate_output  →  assemble_motion_caption
   ▼
{motion_caption, windows, validation_errors, ...}
```

**User prompt 结构**：

```
sample_id / split / source_group_id / provided_dance_style / clip_duration_seconds

Frozen windows and source events:
Window w0: [0.000s - 2.480s]
  Event w0e0: 一侧手腕相对肩部在画面中下移
  Event w0e1: 一侧肘部的二维投影夹角减小
Window w1: ...

Refine the supplied choreography under the system rules. Do not change window boundaries.
```

视觉输入用 `video_url`（`data:video/mp4;base64,...`），模型侧按内部采样得到有序帧证据。

### 3.2 输出 schema

```json
{
  "sample_id": "",
  "dance_style": "",
  "windows": [{
    "window_id": "", "start": "", "end": "",
    "caption": "自然、完整的英文动作描述",
    "source_updates": [{"source_id": "", "action": "keep|uncertain|correct",
                        "replacement": null, "evidence_frame_ids": [], "reason": ""}],
    "additions": [{"addition_id": "", "description": "",
                   "evidence_frame_ids": [], "evidence_is_exact_onset_offset": false}],
    "coverage": [{"source_id": "", "caption_quotes": ["caption 中的精确子串"]}],
    "limitations": []
  }]
}
```

设计意图：

- `coverage` 强制模型自证"每个源事件在正文中被表达"；
- `additions` / `source_updates` 必须带证据帧 id（通常 ≥2 帧）；
- `limitations` 显式记录无法忠实表达的事实；
- `motion_caption` **由 pipeline 组装**（不是模型直接拼格式）。

**事后看，`coverage` 的"逐事件覆盖"要求正是 97% 直译的成因之一**（见 9.2）。

### 3.3 校验（不足以判断质量）

`validate_output` 只检查：

- `sample_id` 一致；
- 窗口数量一致；
- 每个窗口 start/end 与源误差 < 1e-6。

**这些都是结构校验，不涉及语言质量。** 这是本次验收失效的直接原因。

### 3.4 批处理与并发

`batch_annotate.py`：

- 输入：`--tar-start/--tar-end/--reverse/--limit/--concurrency/--service`；
- 每个 tar：下载 → 解出 planner 匹配视频 → 线程池并发请求 → 写 `part_XXXX.jsonl`；
- 每 300 条增量写 `partial_part_XXXX_N.jsonl`，可恢复；
- 服务侧用"最少在途请求"选择 endpoint。

生产配置：每批任务 1 机 4 卡（仅作 HTTP 客户端）+ 32 并发，对应 1 个 vLLM 副本。
**18 个批任务并行，峰值吞吐约 27,000 条/小时**；单批稳态约 1,200–1,500 条/小时。

---

## 4. 生产过程中的问题与修复

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | `Unsupported architecture Qwen3_5MoeForConditionalGeneration` | vLLM 版本过旧 | 升到 0.31.0 |
| 2 | aarch64 内网 torch 2.10 为 CPU-only | 内网 PyPI 无 CUDA 包 | `download.pytorch.org/whl/cu130` 装 2.13.0 |
| 3 | DeepGEMM 编译要求 NVCC ≥ 12.9 | 镜像 CUDA 版本低 | 关闭 DeepGEMM / MoE DeepGEMM |
| 4 | flash-attn ABI 冲突 | 与 vLLM 预编译算子不兼容 | 卸载 flash-attn / flash-attn-cute |
| 5 | `flashinfer.gdn_prefill` 缺失 | flashinfer < 0.7 | 升级 flashinfer 0.7.0.post1 |
| 6 | cubin 版本不匹配 | 版本号不一致 | `FLASHINFER_DISABLE_VERSION_CHECK=1` |
| 7 | JIT 报 PTX 9.4 vs 9.0 | 系统 ptxas 被优先使用 | `CUDA_HOME` 指向 nvidia/cu13 并调整 PATH |
| 8 | sampler JIT 失败 | flashinfer sampler | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| 9 | 32 并发 OOM | KV cache 显存 | `--kv-cache-dtype fp8` |
| 10 | HTTP 400 `max_tokens` 超上下文 | 输入+输出 > 16384 | `max_tokens` 8192 |
| 11 | 集群读 `futurelab-base-model-sg` 403 | bucket 策略 | 换 `futurelab-cognwm-sg` |
| 12 | 容器内 OSS 鉴权失败 | 星云注入 STS 覆盖 AK/SK | 优先读随包 `oss.properties` |
| 13 | staging 慢（406GB@45MB/s） | 分对象并发 + 多副本抢带宽 | 单 tar 分发；必要时停副本 |
| 14 | 提交报 "queue and project not bound" | 项目与队列绑定 | 固定 `-j futurelab_vlm_caption` |
| 15 | 补跑全部 `ConnectionError` | 服务副本重启后**节点 IP 变化** | 每次从 `task get` 读实时 Host |
| 16 | `JSONDecodeError`（8192 处截断） | 极端长输出 | 重试递增 `max_tokens` 至 14336；仍失败走纯文本兜底 |
| 17 | 少量 `sample_id mismatch` | 模型回显 id 不一致 | 降级为告警，正文有效即接受 |
| 18 | 补跑卡在 tar 下载 | oss2 无读超时 | `signal.alarm` 包裹 + 重试 |

### 4.1 帧输入 vs 视频输入

早期尝试"抽帧 + 全局帧 id + 时间戳表"输入。3 条动作最丰富样本的对比显示**帧输入相对整段视频没有显著提升**，且 prompt 更长、证据对齐更复杂。最终统一用整段视频输入。

> 复盘：这个结论本身没问题，但"没有显著提升"的判断也是基于**粗粒度人工看感**，没有量化指标，属于同类验收缺陷。

### 4.2 Prompt 迭代

- v1：仅"自然语言改写"，实测过于保守、句间重复多；
- v2（当前 `SYSTEM_PROMPT`）：加入事件级保真硬规则、视觉补充白名单、禁止泛化词、coverage 自证、"零补充/零纠正也合法"。

**v2 的副作用（事后发现）**：为满足"每个源事件都要有 disposition 与 coverage 引用"，模型倾向把每条事件单独成句直译，**抑制了合并表达**，反而放大了模板化。

---

## 5. 缺口识别与补跑（96.1% → 100%）

### 5.1 第一轮汇总

53 个 tar 全部产出后合并去重：**190,564 / 198,315（96.1%）**。缺口 7,751 条，分布极不均匀。

### 5.2 id → tar 索引

本地持有全部 53 个 tar（约 310 GB）。用 `tarfile` **只读 header（seek 跳过数据）**并行扫描，构建 `sample_id → tar` 映射（263,350 个 mp4 片段），几分钟完成。

### 5.3 两阶段补跑

1. **整 tar 补跑**（`retry_missing.py`）：按 tar 区间扫描 todo id，只解出缺失视频。8 任务并行，7,751 → 1,368。
2. **小包补跑**（`retry_fill.py`）：按 `id → tar` 从本地 tar 只抽取缺失记录视频，打成 3 个均衡小 tar（1.0/0.36/0.36 GB，替代 26 GB 整包），内建 3 次重试、`max_tokens` 递增、纯文本兜底。

尾部：1,368 → 103 → 22 → 1 → 0。

### 5.4 资源治理

发现 20+ 个历史任务共用仅存的 3 个 vLLM 副本（大量在做已覆盖 tar 的全量重跑），单任务吞吐被压到 ~88 条/小时。确认主体结果落盘后，**停掉 19 个冗余重跑任务（76 卡）**，只留 3 个补跑 + 3 个服务副本，吞吐恢复到 ~1,500 条/小时/副本。

---

## 6. 最终产物

### 6.1 HuggingFace

`https://huggingface.co/datasets/avagelion/dance-data` → `qwen35_captions/`

| 文件 | 说明 |
|---|---|
| `train_captions.jsonl` | 198,315 条（2,685,551,373 B） |
| `valtest_captions.jsonl` | 22,352 条（304,568,971 B） |
| `all_captions.jsonl` | 220,667 条（2,990,120,344 B） |
| `summary.json` / `README.md` | 指标与字段说明 |

> 本机无法直连 huggingface.co，上传经 `hf-mirror.com` 代理完成（token 需含 `repo.write`），上传后按字节数与本地逐一对齐。

### 6.2 OSS 备份

`oss://futurelab-cognwm-sg/niezichao.nzc/m2d_output/`：`_merged_captions.jsonl`、`_captions_valtest.jsonl`、`_captions_all.jsonl`、`_summary*.json`，以及分片原始输出 `part_XXXX.jsonl` / `retry_part_XXXX.jsonl` / `fill_*.jsonl`。

### 6.3 记录格式

```json
{"id": "...", "source_group_id": "...", "split": "train",
 "dance_style": "contemporary", "video_path": "...",
 "motion_caption": "Dance style: contemporary\n[0.000-2.480s] ...",
 "windows": [{"start": 0.0, "end": 2.48, "caption": "..."}],
 "caption_version": "qwen397b_visual_refine_v1"}
```

---

## 7. 覆盖率与结构性指标

| 指标 | train | val+test |
|---|---|---|
| 覆盖 | 198,315 / 198,315（100%） | 22,352 / 22,352（100%） |
| 平均长度 | 1,833.9 字符 | 1,844.4 字符 |
| 格式异常 | 0 | 0 |
| 泛化词命中 | 0 条 | 0 条 |
| 窗口数/源窗口数一致 | 100% | 100% |
| 超长（>10k 字符） | 1 条 | 1 条 |

风格分布（train，Top）：`unknown 111,332`、`contemporary 72,658`、`k-pop 6,101`、`jazz funk / heels 2,363`、`chinese classical 1,648`、`pop / casual 1,432`、`other ethnic / folk 975`、`hip-hop 737`。

> `unknown` 占比高源于源数据 `style_source=unknown_no_vlm`，按规则不做音乐流派推断。

**⚠ 以上全部为结构性指标，与"是否真正润色"无关。见第 9、10 节。**

---

## 8. 复现步骤

```bash
# 1) 部署（1 机 4 卡）
nebula-cli run mdl -j futurelab_vlm_caption \
  --user_params="" --queue=future_l20a_videope \
  --worker_count=1 --worker_cpu=48 --worker_gpu=400 --worker_memory=200000 \
  --algo_name=pytorch280 --entry=serve_qwen.py --job_name=qwen35_serve

# 2) 取实时服务 IP（节点会漂移，务必每次查询）
nebula-cli task get <serve_task_id> | grep worker

# 3) 批处理
nebula-cli run mdl -j futurelab_vlm_caption \
  --user_params="--tar-start=0 --tar-end=6 --concurrency=32 --service=http://<ip>:8000" \
  --queue=future_l20a_videope --worker_count=1 --worker_cpu=32 --worker_gpu=400 \
  --worker_memory=96000 --algo_name=pytorch280 --entry=batch_annotate.py --job_name=qwen35_batch

# 4) 缺口补跑（小包模式）
nebula-cli run mdl -j futurelab_vlm_caption \
  --user_params="--tar-key=<oss_key_of_fill_tar> --tag=fill0 --concurrency=32 --service=http://<ip>:8000" \
  --queue=future_l20a_videope --worker_count=1 --worker_cpu=32 --worker_gpu=400 \
  --worker_memory=96000 --algo_name=pytorch280 --entry=retry_fill.py --job_name=qwen35_fill_0

# 5) 合并对账
python3 final_merge.py
```

---

## 9. 质量缺陷复盘（核心）

### 9.1 事实：大部分记录没有真正润色

在 198,315 条 train 输出上做重复度统计（句级重复率、8-gram 重复率）：

| 指标 | 结果 |
|---|---|
| 句级重复率均值 | **0.366**（p50 0.389，p90 0.600，p99 0.762） |
| 句子重复率 ≥ 0.4 | **48.2%** 的记录 |
| 8-gram 重复率均值 | **0.385** |

典型输出（`JsgrYgJJApXZsIlh`）：

```
One side wrist moves right relative to shoulder; elbow angle increases.
The other side wrist moves down relative to shoulder; elbow angle increases.
One side wrist moves left relative to shoulder; elbow angle increases.
...
```

源文本是"另一侧手腕相对肩部在画面中下移。一侧手腕相对肩部向画面右侧移动。…"的逐事件流，输出**逐条一对一翻译**，只是把中文换成了英文模板句式，没有合并、没有抽象、没有时序组织。

### 9.2 归因

| 因素 | 影响 |
|---|---|
| **源文本形态** | 原始 MotionScript 就是逐关节事件流（每窗口几十条），直译天然模板化 |
| **coverage 逐事件约束** | v2 prompt 要求"每个源事件都要有 disposition 与正文引用"，模型选择最省力的做法：一事件一句 |
| **"不得省略"规则** | 为防止信息丢失而强调不得缩短，进一步抑制了合并概括 |
| **缺少压缩/概括指令** | prompt 从未要求"同向连续移动合并为 continues""反转写成 reverses""多次重复写 repeats" |
| **验收标准缺失** | 只校验结构（窗口/格式/id），没有任何语言质量指标 |
| **验收方式** | 早期只人工看了 3 条"动作最丰富"的样本，且关注点是"是否漏信息"，不是"是否润色" |

**注意：这不是模型能力问题。** 同一模型的 A 类输出（见 10.3）证明它能写出合并句、能补充躯干/步法信息。是**任务定义与验收标准**把输出压向了直译。

### 9.3 我的判断错误

- 把"覆盖率 100% + 格式 0 异常 + 泛化词 0"当成了质量结论，但这些只说明"没有坏格式"，不说明"有润色"；
- 抽样量太小（3 条），且抽样标准偏向"信息量最大"而非"典型"；
- 没有建立可量化的语言质量基线，导致 22 万条批量产出后才发现系统性问题。

---

## 10. 快速筛检方案（已验证）

### 10.1 三个可自动计算的信号

| 信号 | 含义 | 计算 |
|---|---|---|
| `tmpl` 模板句占比 | 句子是否以 `one side / the other side + 部位` 开头 | 正则计数 / 句数 |
| `dup` 句级重复率 | 同一句在正文中重复的比例 | 1 − 唯一句数/句数 |
| `rep8` 8-gram 重复率 | 词级连续片段重复程度 | 重复 8-gram 多余次数 / 总 8-gram |
| `narr` 叙事密度 | 是否出现 step/turn/travel/reach/extend 等叙事动词 | 每千词命中数 |

`/tmp/caption_tier.py` 即按此实现，全量 198,315 条扫描耗时约 20 分钟（单进程）。

### 10.2 分层结果

| 层级 | 判定 | 数量 | 占比 |
|---|---|---:|---:|
| **A_narrative**（可直接用） | `tmpl<0.75` 且 `dup<0.25` 且 `rep8<0.35`，或叙事密度达标 | **264** | **0.1%** |
| **B_partly_rewritten**（部分润色） | 模板句 40–90%，重复中等 | **5,755** | **2.9%** |
| **C_raw_template**（直译，需重跑） | 模板句 ≥75% 或 ≥90% | **192,296** | **97.0%** |

各层指标：

```
A_narrative      tmpl p50=0.429  dup p50=0.095  rep8 p50=0.070  平均 20.5 句 / 156 词
B_partly         tmpl p50=0.609  dup p50=0.296  rep8 p50=0.212  平均 35.4 句 / 236 词
C_raw_template   tmpl p50=1.000  dup p50=0.455  rep8 p50=0.431  平均 40.1 句 / 293 词
```

按窗口数分组，8-gram 重复率稳定在 0.375–0.406，说明**重复与视频长短无关，是表达方式问题**。

### 10.3 A 类样本（相对正常）

```
=== PvDxnNeAFwMiodrK
[0.600-1.220s] One wrist moves left then up; the other wrist moves right then left. Both elbows straighten outward.
[1.220-1.860s] One wrist moves down then right.
...
[3.120-5.062s] The other wrist moves up then left, then up again then left again, then down. ...
  → 有 then/again 的时序组织、有双侧合并（"Both elbows straighten outward"）

=== vsQFcnbmakiTUkHE
[0.000-2.360s] One side wrist moves right relative to shoulder; one side knee angle decreases.
              Torso turns right; arms lift and extend outward with fans.
[2.360-5.062s] One side wrist moves right relative to shoulder.
              Torso continues turning right; arms sweep upward and outward with fans.
  → 补充了躯干转向、手臂轨迹、道具（fans）等视觉信息，并用了 continues
```

这两个例子说明**目标是可达的**，只是当前 pipeline 让它变成了少数。

### 10.4 推荐处置

1. **不要整体重跑 19 万条**，先改 prompt 在 B/C 混合子集上做 200–500 条 A/B 验证；
2. prompt 增加的压缩规则（建议）：
   - 同部位连续同向移动合并为 `continues moving ...`；
   - 方向反转写 `reverses direction`，不要重列一遍；
   - 连续 ≥3 次同向写 `repeats N times`；
   - 同一时刻的多部位合并为一句；
   - **coverage 由"逐事件"改为"逐事件组"**：允许一个源事件组整体对应一句话；
   - 目标：模板句占比 < 30%，句级重复率 < 15%；
3. 验收改为**双轨**：结构校验 + 上述四项语言指标，任一层不达标不进入交付；
4. 抽样从"动作最丰富"改为**分层随机**（按窗口数分层，每层抽 20 条）。

### 10.5 已产出的可用子集（可立即使用）

- A 类 264 条 + B 类 5,755 条 = **6,019 条**（3.0%）可以直接用于训练/评测；
- C 类 192,296 条建议标记为 `caption_version=qwen397b_visual_refine_v1_raw`，仅作**动作事实对齐**用（其窗口与事件覆盖是准确的），不要当叙事文本用。

---

## 11. 经验总结

1. **覆盖率 ≠ 质量**：结构指标全绿时，语言质量可能完全没改善。必须建立语言质量基线。
2. **约束越严，模型越倾向直译**："不得省略 + 逐条覆盖"会把改写任务退化成翻译任务。要在约束中加入"压缩/概括"的正向要求。
3. **打标质量靠"约束 + 自证 + 可量化验收"**，缺一不可。
4. **格式由 pipeline 保证**：模型只输出结构化 payload，正文由代码组装，避免格式漂移。
5. **补跑瓶颈在 I/O**：稀疏缺口时按 id 抽小包（1.7 GB）替代整包下载（26–310 GB）。
6. **服务 IP 会漂移**：客户端应从任务元数据实时取地址。
7. **先确认结果落盘，再动资源**：停任务前必须确认分片输出已写 OSS。
8. **吞吐由"副本数 × 单副本有效并发"决定**：集中资源比堆任务有效。
