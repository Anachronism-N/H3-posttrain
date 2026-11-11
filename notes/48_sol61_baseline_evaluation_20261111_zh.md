# Captioner 训练期间的 baseline 与评测执行方案

先完成 **通用音频模型的局部编舞实验 E2，以及评价指标校准 E5**。两者都不依赖训练中的 captioner，也不需要新增训练。数据从 `avagelion/dance-data/sol61_captions` 构造；当前机器完成数据审计、代码和 CPU 验证，GPU 生成由另一集群执行。按本轮约定，不将既有训练使用情况设为准备实验的前置条件。

本页保留已实现的 Qwen3-Omni runner 与评审命令。正式比较现已扩充为六个音频＋prompt 模型，MOSS-Music/Music Flamingo 优先新增，详见 [多模型与 agent 执行表](49_expanded_baselines_and_agent_experiments_20261111_zh.md)。以下单模型数量仅是联调示例，不是当前正式实验规模。

## 1 现在做哪些实验

| 次序 | 实验 | 具体运行 | 交付物 |
|---|---|---|---|
| 1 | 数据与协议冻结 | 审计全部 train/val/test；确定 val 12 条开发样本、test 96 条正式样本；绑定原 WAV | 清单、文件 SHA、私有参考、音频身份 |
| 2 | 开源 baseline 开发运行 | Qwen3-Omni-30B-A3B-Instruct，音乐＋固定编舞 prompt，每例生成一次 | 12 个原始响应、失败/截断记录、成本 |
| 3 | 指标开发校准 | 两人独立听音盲评；文字 judge 评文本维度；分析不同意见 | 冻结量表、评审说明、judge 配置 |
| 4 | baseline 正式运行 | 用冻结配置生成 96 条 test 输出，不按好坏重抽 | baseline 主表原始数据 |
| 5 | E5 正式验证 | 对同一批输出独立人工评分，再与自动文字维度比较 | 人评一致性、自动与人工相关性、覆盖率 |
| 后续 | 加入训练后的模型 | 在相同 96 段音频上使用该检查点实际训练协议生成；用相同量表评价 | SFT 与 baseline 配对质量比较 |

先用一个 baseline 联调运行与评价链路，再按扩展执行表接入全部六个主比较模型。使用各模型正确的原生音频接口，不能用文字描述音乐代替原始音频，也不能把当前 native loader 的模型路径随意换成另一种架构。

E1 后置，视频事实评价仍需对应视频。E3/E4 的代码和工具测试可以继续，但正式整曲编舞与编辑需要完整歌曲。不能把本集合不相关的 5 秒片段拼成一首歌来代替整曲实验。Agent 关键机制消融仍为选做。

## 2 数据核查与抽样

实际目录为 **`sol61_captions`**，不是 `sol61_captioner`。固定版本：

`1ca58b4ebc7f9457fb7dc7db587c05f3e79b767a`

| 划分 | Caption 数 | 声明的来源组数 | 自动 gate 通过 | 自动 gate 未通过 |
|---|---:|---:|---:|---:|
| train | 198,315 | 197,677 | 197,578 | 737 |
| val | 11,098 | 11,097 | 11,044 | 54 |
| test | 11,254 | 11,253 | 11,203 | 51 |

扫描全部 220,667 条记录：跨划分样本 ID 重叠为 0，跨划分 `source_group_id` 重叠为 0。原始媒体清单中的该组 ID 对应 `video_sha256`，不能把它直接解释成独立歌曲 ID；同歌和近重复隔离尚未建立。公开聚合审计与源文件 SHA 在 [审计 JSON](../captionlab/configs/sol61_source_audit_20261111.json)。

抽样规则固定为：对 `seed:split:id` 求 SHA256 排序，每个声明的视频来源组最多选一例，排除已知跨划分 ID/来源组；seed＝42。开发集取 val 12 例，正式集取 test 96 例。默认保留 gate 未通过样本，避免用标签生产者的自检分数筛选出容易例；若另外报告 gate 子集，须单列。当前没有可靠 BPM/歌曲 ID，不伪造速度分层。风格字段只作抽样后组成统计，不输入生成模型，不用于替代听音评分。

这两个样本集合已在真实源文件上完成确定性选取；[抽样回执](../captionlab/configs/sol61_selection_20261111.json) 保存数量、来源组数和有序 ID 列表的 SHA，供集群重建后核对。音频绑定状态仍为 false，不代表已经生成模型结果。

### 2.1 这些 GT 可以评价什么

数据 README 表明，这批 caption 是对原始 MotionScript 的 **纯文本改写**：

- 原始动作日志主要覆盖 16 类腕部位移、肘/膝投影角度变化事件；左/右指画面方向。
- 正文无动作时间戳；`gestures` 中的 `wNeM` 是源事件编号，不能当成秒数。
- 不能据此建立躯干、头、髋、脚步、转身与空间位移的完整动作真值。
- `gate/coverage/evidence` 是标签生成流程的自动自检，不是独立的人类正确性评分。

因此正式编舞质量评价允许与 GT 不同但合理的动作。参考一致性只回答“是否复现这份参考已经表达的事实”，且应在参考支持的部位与坐标定义内判断；它不能替代音乐适配，也不能证明视频事实正确。主质量盲评不显示 GT，参考一致性另开评审阶段。

来源：[固定版本目录](https://huggingface.co/datasets/avagelion/dance-data/tree/1ca58b4ebc7f9457fb7dc7db587c05f3e79b767a/sol61_captions)、[标签说明](https://huggingface.co/datasets/avagelion/dance-data/blob/1ca58b4ebc7f9457fb7dc7db587c05f3e79b767a/sol61_captions/README.md)。

## 3 baseline 的执行合同

第一项对照使用 **Qwen3-Omni-30B-A3B-Instruct 的 Thinker 文本输出**，不加载我们训练的 checkpoint。原始音乐和明确编舞任务位于同一个 user 消息中；模型获得一个完整 5.0625 秒音频窗，输出一个英文动作段落。

固定 prompt 使用 `choreobench.music_segments.UNTIMED_PROMPT`，已复制到配置文件：描述一个可能的单人编舞乐句、动作先后与并发、力度及过渡；不输出舞种标签、服装场景或数值时间戳。允许中等粒度全身动作。不能因 GT 覆盖范围较窄，就把合理的额外动作直接判成错误。

| 设置 | 首轮固定值 |
|---|---|
| 音频 | 按 ID 关联原 WAV，核对原媒体 SHA、单声道 PCM16、采样数及 5.0625 秒时长 |
| 模型输入 | 音频＋固定任务 prompt；不含 GT、GT 舞种或 gestures |
| 候选数 | 每例 1 个；不做 best-of-N |
| 随机性 | seed＝42；temperature＝0.9，top_p＝0.95，top_k＝50 |
| 输出预算 | 1,024 tokens；context 上限 8,192；开发集检验是否够用后一次冻结 |
| 输出处理 | 保存自由正文，绕过旧双时间窗 caption 解析器；原文不润色、不修复 |
| 失败处理 | 空输出、截断、HTTP/服务错误及未知调用均保留；不因质量重抽 |
| 恢复 | `--resume` 复用已封存响应；已发送但响应不确定的请求不自动重发 |

模型官方接口说明：[模型卡](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/README.md)。不同模型不必共享 tokenizer，但应固定相同任务范围、输入音频与预算政策，并分别记录实际输出 tokens/耗时。训练后的 captioner 须按其真实训练输入格式接入，不能为了“统一 prompt”破坏其训练合同。

## 4 首版指标如何计算

### 4.1 主质量维度

每项采用 1–5 分，2/4 分取相邻锚点之间；证据不足记 unknown。主表分别报告三项，不先相加成未经验证的总分。

| 维度 | 1 分 | 3 分 | 5 分 | 评价输入 |
|---|---|---|---|---|
| 动作清晰度 | 部位、动作或关系大量不明/矛盾 | 主要动作可懂，仍有含混 | 动作与关系明确、表达一致 | 输出文本 |
| 时间组织 | 先后并发混乱，或给定时长内严重堆积转换 | 主线清楚，局部节奏/过渡不清 | 发展、并发与过渡清楚，文字动作负担合理 | 输出文本＋片段时长 |
| 音乐适配 | 与可听速度、能量、重音或发展明显冲突 | 大致兼容，回应较笼统 | 动作安排与可听动态、发展有明确合理对应 | 原音频＋输出文本 |

此前计划中的“时长负担”并入时间组织，首版不再单独重复计分。这里评价文字方案的合理性，不声称验证了人体动力学或精确踩拍。没有动作时间戳与执行轨迹，就不计算伪精确的 beat alignment。

### 4.2 自动与人工各负责什么

- **自动机械检查：**空正文、停止原因、截断、字数、逐字重复句、显式时间窗是否合法。重复统计是诊断，主题复现不自动算错误。
- **自动文字 judge：**动作清晰度、时间组织；可选单独参考一致性。使用固定量表、模型身份、JSON 合同与原文证据引文；伪造引文、无效 JSON、截断均记录为未成功评分。文字 judge 的音乐适配始终为 NA。
- **两位人工评审：**独立看匿名文本并听完整片段，填写三项主质量评分及理由。音频 SHA 与听音范围绑定到评审记录。声明听过是审计记录，不是独立认证。
- **参考一致性：**单独的参考评审包展示去舞种头的 GT 与候选动作文本，按 1/3/5 分锚点评价有限事实的一致程度；不并入主质量分。

这实现的是 **可执行的 v1 评价协议**，目前不能直接称为已经验证的新指标。E5 才用于判断这些自动维度是否可靠；自动文字质量与音乐理解能力必须分别报告。

### 4.3 统计与 E5

1. 先用 val 12 例校准量表、judge 提示及人工耗时；之后冻结，不根据 test 表现修改规则。
2. 对 test 输出保留所有注册槽位。质量均分仅基于实际可评分项，同时报告生成成功率、截断率、缺评/unknown/NA 比例，不将缺失补 0 分。
3. 同一方法先在同一 case 内平均评审分，再按声明来源组平均；方法间用相同 case、相同评审的配对分差，按来源组 bootstrap 2,000 次，报告 95% 区间。来源组不是已确认独立歌曲，应如实标注。
4. 人工一致性：每项报告两名评审的 quadratic weighted Cohen’s κ；少于两对评分、或预期分歧为零时返回未定义，不造出满分一致性。Spearman 在任一评分向量为常数时也未定义。
5. 自动有效性：同一输出上，文字 judge 分数与两名人工分数均值计算 Spearman 相关，并以来源组重采样区间；缺一名人工评分则该对不进入相关计算，同时报告配对覆盖率。文字 judge 的音乐维度没有分数，不能计算音乐相关性。
6. 真实输出不足以单独证明指标的错误识别能力。按 E5 原计划另做受控样例：动作关系矛盾、顺序破坏、过量重复、无依据加入 GT 事实；配套同义改写与合理主题复现作为不应降分的对照。人工确认扰动含义后再冻结。它们不是当前 CPU 单测的替代名称。

第一轮可以先完成前五项。受控样例集、与普通整体 judge 的比较及独立音乐 judge 校准仍需补齐后，才能扩大论文中的指标有效性主张。当前代码不把规则测试通过当成真实语义指标验证通过。

## 5 集群上如何执行

以下从仓库根目录运行，路径为集群示例，替换 `/data/dance/audio` 与 `/models/...`。数据和结果写入已被 Git 忽略的目录。已有 CUDA 模型环境沿用，不安装 CPU Torch 覆盖它。

### 5.1 下载清单、冻结样本、绑定音频

```bash
python -m pip install -r captionlab/requirements-cpu.txt

python -m captionlab.sol61 fetch \
  --out-dir .captionlab-data/sol61-v1 --include-media

python -m captionlab.sol61 freeze \
  --source-dir .captionlab-data/sol61-v1/sol61_captions \
  --split val --limit 12 --seed 42 \
  --out-dir .captionlab-runs/sol61-val12

python -m captionlab.sol61 freeze \
  --source-dir .captionlab-data/sol61-v1/sol61_captions \
  --split test --limit 96 --seed 42 \
  --out-dir .captionlab-runs/sol61-test96

python -m captionlab.sol61 bind \
  --manifest .captionlab-runs/sol61-val12/manifest.pending.json \
  --media-manifest .captionlab-data/sol61-v1/ms_original_planner_dataset_v1/val.jsonl \
  --audio-root /data/dance/audio \
  --out .captionlab-runs/sol61-val12/manifest.public.json
```

`fetch` 默认固定上述 HF commit；下载三个 caption 文件及可选原始 val/test 媒体清单，约 0.46 GB，不下载重复的 `all_captions.jsonl`、1.15 GB 的原始 train 清单或全部媒体包。公开数据不需要 token；需要时仅从 `HF_TOKEN` 环境变量读取。

原 WAV 必须在 `audio-root/<id>.wav`。目前媒体在 HF 的 `audio_tars/part_*.tar` 中；本命令不声称已经建立了按 ID 选择性下载的 tar 索引。优先复用数据预处理集群的原始 WAV。如果集群只有 caption、没有原 WAV，需先搬运选中 ID 的媒体，不能将空文件、静音或重新编码后未知身份的音频当成原音频通过绑定。

正式运行时将 bind 的目录和媒体清单换为 test96/test.jsonl。`manifest.pending.json` 中音频未绑定，runner 会在创建模型之前拒绝执行。`references.private.json` 保留原 GT，仅交给评分阶段。下载 receipt 绑定版本和 SHA；手工复制且无 receipt 的源文件会明确标为 `declared_revision_only`。

### 5.2 先在同一 GPU 进程直接跑 baseline

复制并修改 [native 配置](../captionlab/configs/sol61_qwen3_omni_instruct.example.json) 中的模型目录到本机配置，保持 prompt 与采样参数冻结：

```bash
python -m captionlab.baseline_eval \
  --manifest .captionlab-runs/sol61-val12/manifest.public.json \
  --provider-config /data/eval-config/omni-instruct.json \
  --method-id qwen3-omni-instruct \
  --out-dir .captionlab-runs/val12-qwen3 \
  --seed 42 --max-output-tokens 1024
```

开发验收后用 test96 清单与新结果目录执行一次相同命令。`results.json`、`records.jsonl`、调用日志及原始响应一起保留。运行中断时在原命令后加 `--resume`；不要删除失败记录来改变分母。

模型也可常驻：

```bash
python -m choreobench serve-local \
  --provider /data/eval-config/omni-instruct.json \
  --audio-root /data/dance/audio \
  --host 127.0.0.1 --port 8765 \
  --api-key-env CHOREOBENCH_NATIVE_TOKEN
```

先在服务与客户端环境设置同一个 `CHOREOBENCH_NATIVE_TOKEN`。将启动日志或 `/health` 返回的 `identity_sha256` 填入 [service 配置](../captionlab/configs/sol61_native_service.example.json)，再将 baseline 命令的 provider-config 指向该配置。客户端和服务需要访问相同绝对 WAV 路径；服务不上传音频。health 身份声明不替代服务端模型加载验收。

### 5.3 生成听音盲评包

先按下一节的配置示例填写实际文字 judge 配置，再登记两位人评与该 judge 的固定身份；`roster` 只生成配置，不调用模型：

```bash
python -m captionlab.quality_judge roster \
  --provider-config /data/eval-config/text-judge.json \
  --reviewer-id text-judge-v1 --human h1 --human h2 \
  --out .captionlab-runs/reviewers.json

python -m captionlab.review_bundle collect \
  --result .captionlab-runs/val12-qwen3/results.json \
  --reviewers .captionlab-runs/reviewers.json \
  --out-dir .captionlab-runs/val12-review
```

加入第二种方法时追加一个 `--result .../results.json`。方法必须有相同 case 库存和音频身份；失败槽位仍保留。只把包内 `public/` 交给评审，私有映射留在负责人处。用浏览器打开 `public/index.html` 听音和查看匿名候选，通过页面中的本人链接下载 JSON 模板，**另存到 bundle 之外**再填写，如 `/data/eval-reviews/h1.json` 和 `h2.json`。原包文件保持不变，导入时会核验其 SHA。不要填写别人的表或人工冒填 judge 模板。先评价主质量，再另开参考阶段，避免 GT 锚定主质量判断。

人评需要将 `review_status` 改为 `completed`，在三项 `ratings` 中填写 `rated`、整数分数及理由；已查看证据但仍无法判断的项保留 `unknown/null`。在 `evidence` 中如实填写看过文本、精确原文引文、已听音频的 SHA 和完整听音范围 `[[0,5.0625]]`；未听音频时 `music_fit` 必须为 `not_applicable/null`。主质量阶段 `reference_seen=false`。生成失败的槽位保留 pending，不强行评分。模板字段定义与锚点在 `rubric.json`。

### 5.4 运行自动文字 judge

使用已有的文字推理服务，按 [judge 配置](../captionlab/configs/sol61_text_judge.example.json) 指定实际模型 ID、兼容的 chat-completions 接口和环境变量。服务需要支持请求中的严格 `json_schema`；不支持时会保留错误，不静默改用自由文本。不要将音频模型配置误当文字 judge 配置。

```bash
python -m captionlab.quality_judge run \
  --items .captionlab-runs/val12-review/public/items.json \
  --provider-config /data/eval-config/text-judge.json \
  --reviewer-id text-judge-v1 \
  --out-dir .captionlab-runs/val12-text-judge
```

judge 不接收真实方法身份，不听音，不能给音乐适配打分。调用与原始响应一起保存。句子证据必须是候选原文的精确子串；该检查能拒绝假引文，但不会证明引用后的语义判断正确，因此仍需 E5 人工验证。

### 5.5 汇总与指标校准

负责人导入两份填好的人工文件及 judge 的 `reviews.json`。绑定不匹配、重复评审和身份漂移会被拒绝。解除盲法会恢复方法身份、输出统计并保留缺评分母；只在负责人目录保存此结果。

```bash
python -m captionlab.review_bundle unblind \
  --bundle .captionlab-runs/val12-review \
  --review /data/eval-reviews/h1.json \
  --review /data/eval-reviews/h2.json \
  --review .captionlab-runs/val12-text-judge/reviews.json \
  --resamples 2000 --seed 42 \
  --out .captionlab-runs/val12-unblinded.private.json

python -m captionlab.metric_validation \
  --items .captionlab-runs/val12-unblinded.private.json \
  --reviews .captionlab-runs/val12-unblinded.private.json \
  --human h1 --human h2 --judge text-judge-v1 \
  --n-resamples 2000 --seed 42 \
  --out .captionlab-runs/metric-validity.json
```

有两种方法时，给 `unblind` 追加 `--method-a` 和 `--method-b`，取值为生成时的真实 method-id。人类评分和模型评分分开汇总，禁止混成同一组评审。预登记 roster 使完全缺评的人仍占有槽位。只有一名 judge 的评分、或者两名人工评分都完全相同时，部分相关性可能未定义；保持该结果，不为得到数值制造差异。

可选参考阶段：另建新 bundle，在 `collect` 添加 `--phase reference --references .captionlab-runs/sol61-val12/references.private.json`。材料位于 `private/reference/`，不进入主质量公开包；judge 的 `--items` 指向该处，使用新输出目录。该阶段只评参考一致性，主质量三项必须为 NA，分别导入与汇总。

## 6 资源与验收

| 资源 | 首轮要求 |
|---|---|
| 数据准备 | CPU、约 0.5 GB 源 JSON；108 个选中 WAV；不需要全量视频或 H3 |
| baseline | 已下载的 Qwen3-Omni Instruct 权重、匹配的 CUDA Torch/Transformers 环境、numpy/soundfile/scipy |
| 显存 | 当前 native loader 为 BF16 Thinker、单设备加载；优先在 80 GB 级 GPU 试运行。模型与 KV/激活实际峰值须测量，本实现没有暴露多卡 TP/device_map，不能保证任意显卡可用 |
| 文字 judge | 已部署兼容 chat-completions 的文字模型；可常驻、与音频模型分时运行，无训练需求 |
| 人评 | 2 名评审；开发集先测每例耗时，再冻结正式预算 |

仅 1 个 baseline 时：开发阶段 12 次生成、12 次主质量文字 judge、24 份人工评分；正式阶段 96 次生成、96 次文字 judge、192 份人工评分。参考一致性单独评价时增加相应 judge/人评任务。后续加入 SFT 后，正式主质量为 192 份输出、384 份人工评分；若每份耗时 1–2 分钟，总计约 6.4–12.8 人时，须以开发集实测替换此预算估计。

集群验收顺序：先确认原 WAV 全部通过身份绑定，再对开发集前几例确认模型确实接收音频与编舞 prompt、没有被旧格式解析器丢弃，随后跑完 12 例检查截断与耗时；完成量表校准后才运行 test96。测试集质量差也不换样本、不重写 prompt 后继续沿用同一份正式结果。

本轮本地验证覆盖源数据扫描、确定性抽样、音频合同、调用恢复、GT 隔离、评审导入与统计。**尚未在目标集群加载大模型、生成真实 baseline 或完成独立听音评分。** 这些是上述集群步骤的验收目标，不是 CPU 单测已经证明的结论。

2026-11-11 本地执行 `python -m pytest -c captionlab/pytest.ini -q`：396 passed、5 skipped。跳过项为当前 Python 环境未安装 Torch 的已有运行时/数值检查。新增主质量及参考阶段的联测使用明确标记 synthetic 的 WAV、生成响应和评分，验证跨模块合同，不计入实验成绩。
