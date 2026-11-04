# ChoreoBench：音乐到文本编舞的 benchmark 框架与参考 baseline

本目录将任务定义、方法实现和评估分开。核心产物是可编辑的自然语言编舞时间线；不要求 H3、3D 姿态或视频生成。名称为工作名称，当前交付是 **benchmark 软件、协议与参考 baseline**，尚不包含正式整曲数据集、真实大模型结果或人工评测结论。

新增代码独立于原有 H3/5 秒训练器；原训练器与模型输入合同未改变。Python 核心支持 3.9+。真实模型依赖沿用训练 checkpoint 对应环境，另行验收。

## 任务与方法

| 赛道 | 输入 | 输出 | 本版实现 |
|---|---|---|---|
| Local | 5.0625 秒音乐 | 带时间的动作文本 | 复用现有原生 audio-only 模型，原始双窗文本保留 |
| Full song | 180–300 秒音乐与公共音乐结构 | 开放词汇动作总谱 | concat / global / hierarchical 三个 baseline |
| Editing | 源计划、文本指令、公开目标范围与保护条件 | 局部 patch 和新版计划 | 指令模型提出 patch，程序验证后事务式执行，可撤销 |

v1 编辑是 **给定目标范围的受约束编辑**。目标事件/允许修复区域由任务作者提供；不是已经实现从任意自由指令中自动解析“第二次副歌”等所有指代。v1 支持替换已有动作、轨道、时间、motif 引用及已有属性，尚不支持增删事件/新增属性，也不自动生成扩大的修复范围。这样可以先准确评估目标达成和保持性，再单列定位、修复范围预测等扩展任务。

- concat：逐窗原生生成，使用候选 index 0，按确定性所有权区间拼接；不声称进行了质量择优。
- global：外层指令模型一次读取公共结构和同一批局部文本候选，输出整曲计划。它是强全局提示对照。
- hierarchical：先生成全局段落/主题安排，再按段落读取相关候选及前段状态生成计划。

外层模型读取的是音乐工具结果和文本候选，不直接听音。Qwen 原生局部模型只接收波形，不接收全局计划、编辑文字或私有答案。分层 baseline 的收益需要通过真实实验测量，mock 仅复制/组织测试数据，不能证明方法有效。

完整协议：[PROTOCOL_zh.md](docs/PROTOCOL_zh.md)。资源：[RESOURCES_zh.md](docs/RESOURCES_zh.md)。相关工作：[RELATED_WORK.md](docs/RELATED_WORK.md)。精确数据合同：[CONTRACT.md](CONTRACT.md)。

## 1. CPU 上先验收软件

以下命令均在仓库根目录执行。核心 smoke 不需要 torch、CUDA、API 密钥或模型权重；输出目录必须是新目录。

```bash
python -m choreobench doctor
python -m choreobench demo --out-dir /tmp/choreobench-smoke --run
python -m choreobench validate-data --dataset /tmp/choreobench-smoke/dataset.json
python -m choreobench resources --dataset /tmp/choreobench-smoke/dataset.json --config /tmp/choreobench-smoke/hierarchical.json
```

Windows 将 `/tmp/...` 换成可写绝对路径。`demo --run` 生成合成 WAV，执行三个 mock baseline，并验证一个人工准备的局部 patch。所有结果都标记 synthetic，不进入科学结果。加 `--seconds 180` 可检查整曲长度的数据流。

```bash
python -m pip install -r choreobench/requirements-dev.txt
python -m pytest -c choreobench/pytest.ini -q
```

恢复使用原命令加 `--resume`。数据、代码、模型身份、参数、环境变化需要新输出目录；已经持久化的调用不会重发。结果未知的请求不会自动重试，避免隐藏额外采样或重复计费。原始响应与已发布计划不可覆盖。

## 2. 准备真实歌曲与数据划分

先完成原曲/艺人/近重复分组，再划 train/dev/test。缺少分组时可以做开发，但 release-check 会阻止将其标为正式可发布条件通过。不要把相邻切片或同曲不同版本拆到不同 split。

```bash
python -m pip install -r choreobench/requirements-audio.txt
python -m choreobench prepare-audio \
  --source /data/original/song01.flac \
  --out-wav /data/choreobench/audio/song01.wav \
  --out-song /data/choreobench/song01.json \
  --song-id song01 --split dev \
  --song-group-id composition01 --artist-group-id artist01 \
  --near-duplicate-group-id recording01 \
  --brief "Compose a solo dance with clear section contrast."
python -m choreobench assemble-data \
  --songs /data/choreobench/song01.json \
  --dataset-id pilot-v1 --out /data/choreobench/dataset.json
python -m choreobench validate-data --dataset /data/choreobench/dataset.json
```

prepare-audio 显式做声道均值、必要的重采样和 PCM16 量化，记录 receipt；不裁切歌曲、不自动归一化会削波的音频。解码失败时先用已有 ffmpeg 环境转成无损 WAV，并保留该转换来源。初始音乐结构为 unknown 全曲段，不会伪造拍点或段落。

原生专家要求完整 5.0625 秒波形。切窗时保持原始采样时钟；尾窗左移覆盖歌曲尾部，单独记录 observation 与 ownership，避免重叠输出重复。短于一窗时报错，不自动填零。

### 导入音乐结构

可在现有 BeatThis/All-In-One/SongFormer 环境先计算结构。本版提供 All-In-One JSON 适配，以及本项目整数 sample 格式导入；不会代替用户安装和运行外部工具。

```bash
python -m choreobench import-structure \
  --dataset /data/choreobench/dataset.json --song-id song01 \
  --structure /data/structure/song01.json --format allin1 \
  --audio-sha256 REPLACE_WITH_CANONICAL_WAV_SHA256 \
  --out /data/choreobench/dataset.with_structure.json
```

`--audio-sha256` 是分析所用规范波形的绑定声明，不证明工具准确。段落必须完整分割音频，时间以原波形为准；工具造成间隙、错位或尾部超长时明确失败，不自动补出虚假真值。相同功能标签不自动认定为相同 motif；repeat_group 需后续独立核对。

正式数据还需来源清单、分组审计与人工核验。不要只把 `grouping.status` 改成 reviewed 来冒充已经完成这些工作。

## 3. 在服务器运行真实 baseline

先复用已训练 Qwen 的运行环境，准备完整基座、processor/tokenizer、支持的 portable checkpoint、原 run_config 和冻结训练源码。当前 adapter 不接受任意未知 PEFT 目录，不支持隐式迁移输出协议。

修改 [native_concat.json](examples/native_concat.json) 与 [native_hierarchical.json](examples/native_hierarchical.json) 中的绝对路径、实际指令模型名和服务地址。concat 不调用外层模型，其 planner_provider 占位项不执行。若明确测试原始未微调基座，可删除 checkpoint/training_source_dir，并在实验记录中标注 base-only。

外层 API 使用兼容 `/v1/chat/completions` 的文本服务，不自动启动服务。密钥只放在 `CHOREOBENCH_API_KEY` 环境变量中。若服务不支持 JSON mode，改为 false；若其输出预算字段不同，配置 token_parameter。不要把密钥写进配置。远程服务权重身份只能声明/由服务端审计，本客户端不声称已核实。

```bash
python -m choreobench doctor
python -m choreobench resources \
  --dataset /data/choreobench/dataset.with_structure.json \
  --config choreobench/examples/native_hierarchical.json
python -m choreobench run \
  --dataset /data/choreobench/dataset.with_structure.json \
  --config choreobench/examples/native_concat.json \
  --out-dir /runs/pilot-concat
python -m choreobench run \
  --dataset /data/choreobench/dataset.with_structure.json \
  --config choreobench/examples/native_hierarchical.json \
  --out-dir /runs/pilot-hierarchical
```

global 对照使用同一配置，只把 mode 改为 global，使用新输出目录。三种方法应共享模型、seed、局部候选规则及相同数据。hierarchical 多出的调用必须计入成本；比较质量—成本曲线或另做预算匹配，不能把多调用收益全归给方法。

局部输出预算 `max_output_tokens` 与外层 `planner_max_output_tokens` 分开：完整总谱可能需要远多于4096个token，不能为了提高外层上限而超过原Qwen的上下文合同。外层服务还需要足够输入上下文容纳公共结构、全部候选和输出，服务器首轮应记录实际token用量，显式处理超长错误。

`max_calls` 是整次运行预算；先预估不足时不启动模型调用。`continue_on_model_error=true` 会把格式/截断等已知模型输出失败记为整首失败，继续其他歌曲；它不重采坏输出。网络、身份或未知结果属于基础设施错误，保留证据后停止。`complete=true` 只表示队列走完，必须另看 `failed_songs`、`all_songs_succeeded`。

输出：contract.json、calls/ 原始请求响应、派生 audio/、plans/、results.json。计划文件名为歌曲 ID 的 hash，通过 results.json 的 plans 列表定位，勿猜文件名。语义与音乐质量不会自动变成成功状态。

## 4. 文本编辑与撤销

先用 demo 的 source_plan.json、edit_task.private.json 了解结构。私有任务含 goals/semantic_goals；模型只收到白名单导出的公开指令、范围与保护条件。

```bash
python -m choreobench propose-edit \
  --plan /data/edit/source_plan.json --task /data/edit/task.private.json \
  --provider choreobench/examples/editor_provider.json \
  --out-dir /runs/edit-001
python -m choreobench evaluate-edit \
  --before /data/edit/source_plan.json --after /runs/edit-001/plan.json \
  --task /data/edit/task.private.json --out /runs/edit-001/evaluation.private.json
```

propose-edit 只进行一次有预算的模型调用，不用私有答案修补输出。允许区域和保护字段不得由模型扩大。空 operations 是弃答，不能因此获得编辑成功。

无需 API 的完整例子：

```bash
python -m choreobench apply-edit \
  --plan /tmp/choreobench-smoke/source_plan.json \
  --task /tmp/choreobench-smoke/edit_task.private.json \
  --patch /tmp/choreobench-smoke/example_patch.json \
  --out /tmp/choreobench-smoke/manual-edited.json
python -m choreobench undo-edit \
  --plan /tmp/choreobench-smoke/source_plan.json \
  --task /tmp/choreobench-smoke/edit_task.private.json \
  --patch /tmp/choreobench-smoke/example_patch.json \
  --after /tmp/choreobench-smoke/manual-edited.json \
  --out /tmp/choreobench-smoke/restored.json
```

撤销要求 after 正好对应这一次修改；后面又改过的版本不能被旧撤销覆盖。多轮任务需为每轮源计划重新绑定 task/source_plan_sha256。

## 5. 方法无关的提交与评测

外部方法可以直接输出本项目总谱，不必调用我们的 baseline。提交格式为：

```json
{
  "schema": "choreobench.submission.v1",
  "method_id": "your-method-v1",
  "dataset_sha256": "SHA256_OF_EXACT_DATASET_JSON_BYTES",
  "synthetic": false,
  "predictions": [
    {"song_id": "song01", "path": "plans/song01.json", "sha256": "SHA256_OF_PLAN_FILE_BYTES"}
  ]
}
```

预测路径相对 submission.json 解析。缺失/无效输出保留在所选 split 的歌曲分母中；不能仅挑成功歌曲。我们的结果可直接导出：

```bash
python -m choreobench export-submission \
  --run-dir /runs/pilot-concat --method-id concat \
  --out /runs/concat.submission.json
python -m choreobench evaluate-submission \
  --dataset /data/choreobench/dataset.with_structure.json \
  --submission /runs/concat.submission.json --split dev --track full_song \
  --out /runs/concat.evaluation.json
```

当前自动指标是结构、时间覆盖、字面目标和保持性诊断。音乐贴合、主题发展、动作语义和可执行性待独立评审；没有未经校准的美感总分或文本 BAS。模型生成的 motif ID 只作为声明，不证明舞蹈主题质量。

## 6. 盲评与统计

```bash
python -m choreobench review-pack \
  --dataset /data/choreobench/dataset.with_structure.json --split dev \
  --run concat=/runs/pilot-concat \
  --run hierarchical=/runs/pilot-hierarchical \
  --out-dir /runs/pilot-review --copy-audio
python -m choreobench review-import \
  --public /runs/pilot-review/public/pack.json \
  --private /runs/pilot-review/private/bindings.json \
  --reviews /runs/pilot-review/public/reviews.completed.json \
  --out /runs/pilot-review/report.private.json
```

只把 public/ 交给评审者，private/ 保留方法与原始身份。复制 reviews.template.json，填写真实 reviewer_id、分项评分及是否听取音频。未填写、漏项、模型评审、人工评审分别报告；audio_seen 是评审者声明，不是程序能够证明的事实。缺计划的歌曲先在 submission 评估中记失败，盲评阶段不能静默删除它们。

编辑盲评可使用 Python API `build_review_pack()`，item 增加 source_plan/edit_task；公开包自动重映射事件与任务身份，不暴露私有目标。暂未提供编辑任务批量 CLI。

歌曲级区间：准备 JSON 数组，每行包含 song_id 与数值 value，可重复多行评审；`python -m choreobench bootstrap --rows ratings.json --out interval.json` 先在歌内求均值再等权重采样。单歌不给置信区间。该工具不是“显著性保证”，成对方法比较应对每首歌的配对差值重采样。

## 7. 资源与实验顺序

1. CPU 软件检查：普通开发机，无真实音乐质量结论。
2. 服务器真实验收：一个合法 5.0625 秒样例，确认 checkpoint/处理器/格式/显存/耗时，再试一首完整歌曲。
3. Pilot：约 12 首完整歌曲用于协议和成本试验；完整任务答案、人工评审和组级隔离需补齐。
4. 固定协议后扩充正式数据，冻结测试集和评审规范，再比较多方法与消融。

现原生入口是单设备完整 BF16 Thinker 加载，不自动量化或多卡分片。可优先寻找单张80GB级GPU、128GB级主机RAM及足够模型/日志存储的服务器进行验收；这是容量规划起点，不是显存可运行保证。完整参数存储就约63.44GB，A3B不能解释为仅需加载3B。外层指令模型可用独立服务，避免与Qwen争用显存。真实内存与速度以pilot测量为准。

尚需用户准备：完整歌曲与分组、训练好的实际 checkpoint 与冻结环境、指令模型服务、编舞/音乐评审人员。当前未包含：正式数据发布、真实GPU结果、自动歌曲分组、完整音乐反事实资产生成、自由指代定位、事件增删、递归自改进训练或H3渲染。相关实验协议已记录，但不把它们写成现有代码能力。
