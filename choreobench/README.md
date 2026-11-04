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
- global：外层指令模型一次读取公共结构和同一批局部文本候选，输出整曲计划。它是未经整曲编舞专项训练的零样本全局提示对照，质量尚待真实评估。
- hierarchical：先生成全局段落/主题安排，再按段落读取相关候选及前段状态生成计划。每个身体部位按动作结束时间保留最近事件与声明的退出状态，另保留有界的主题历史，避免只看最后两条文本而遗漏仍重要的身体轨道。未知状态保持未知；时间空隙不意味着姿态持续不变。

外层模型读取的是音乐工具结果和文本候选，不直接听音。Qwen 原生局部模型只接收波形，不接收全局计划、编辑文字或私有答案。分层 baseline 的收益需要通过真实实验测量，mock 仅复制/组织测试数据，不能证明方法有效。

原hierarchical仍按导入的功能段落自由生成。新增run-harness把规划块、整包候选选择与来源编译独立接通；它不会将已有Qwen改成整曲模型。真正乐句检测、自动语义修复循环仍未实现。设计与剩余工作见 [HARNESS_DESIGN_zh.md](docs/HARNESS_DESIGN_zh.md)。

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

### 一次采样，复用同一批局部候选

先运行 concat 收集局部专家输出，再导出候选库：

```bash
python -m choreobench export-candidates \
  --run-dir /runs/pilot-concat --out /data/choreobench/local-candidates.json
```

在 global/hierarchical 的配置中增加 `"local_candidates_path": "/data/choreobench/local-candidates.json"`。保留与采集阶段一致的 local_provider 配置、seed、局部 token 上限、窗口、stride、候选数、歌曲划分和数据清单。然后用原来的 resources/run 命令启动新目录。相对候选库路径以 dataset.json 所在目录解析。

复用模式不创建局部模型 provider，不加载或审计当前本地 Qwen 权重；需要原音频和候选库，以及外层文本模型服务。候选库嵌入原始请求、响应、模型身份和成本，并验证当前派生波形、采样参数与原请求一致。更换机器时，可保留原 local_provider 配置中的历史路径声明，即使这些模型路径在新机器不存在；它们是采集身份，不用于重新加载。不能把不同模型的候选改名后复用。

`max_calls` 约束本轮新增调用；原局部采集成本单独保留，不能按零成本报告整个方法。候选库中的坏输出在后续方法中继续失败，不补采、不挑掉失败歌曲。unknown/pending 请求不能当成有效候选。合成数据与 mock 身份随库保留。哈希提供证据绑定，不证明作者或远程服务身份真实。

旧版本运行目录不会因新代码自动获得可恢复资格；版本变化后用新输出目录。可导出的既有局部调用能在新目录复用，但仍须通过严格来源与请求检查。

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

### 生成可审查的编辑目标

现在可用显式选择器编写任务。例如选择器 JSON `{"section_label":"chorus","occurrence":2,"body_scope":"upper"}` 表示第二个 chorus 段落的上半身；也支持 `{"section_id":"s001"}` 或整数采样区间 `{"start":160000,"end":320000,"track":"right_arm"}`。label 精确匹配工具给出的公开标签，occurrence 从1开始；track 精确匹配，body_scope 包含子部位。

```bash
python -m choreobench preview-target \
  --plan /data/edit/source_plan.json --dataset /data/choreobench/dataset.with_structure.json \
  --selector /data/edit/selector.json --out /data/edit/preview.json
python -m choreobench target-edit \
  --plan /data/edit/source_plan.json --dataset /data/choreobench/dataset.with_structure.json \
  --selector /data/edit/selector.json --task-id chorus2-upper \
  --instruction "Increase upper-body movement in the second chorus; preserve the steps and timing." \
  --constraints /data/edit/constraints.private.json --out-dir /data/edit/task-001
```

constraints 可省略，或包含 repair_event_ids/protected/goals/semantic_goals。生成 task.private.json、剔除私有目标的 task.public.json 和 resolution.json。目标定位不能自动理解指令的保护要求：若需保留步法/时间，应在 constraints 显式写保护字段，再人工核对任务。事件跨时间边界、整身事件覆盖了目标之外的身体部位时，preview 给出冲突，target-edit 拒绝扩大权限。当前仍不拆分或增删事件，也不把该工具描述成任意自由指令理解。

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

外部方法无需生成本项目 run 目录：`review-pack --dataset ... --submission /runs/external.submission.json --split dev --out-dir ... --copy-audio`。可重复 --submission，也可与 --run 混用；method_id 不得重复。来源哈希与 synthetic 声明只进入私有绑定，不泄露给评审者；无来源绑定的旧API材料标记 unknown，不自动视为正式结果。

编辑盲评可使用 Python API `build_review_pack()`，item 增加 source_plan/edit_task；公开包自动重映射事件与任务身份，不暴露私有目标。暂未提供编辑任务批量 CLI。

歌曲级区间：准备 JSON 数组，每行包含 song_id 与数值 value，可重复多行评审；`python -m choreobench bootstrap --rows ratings.json --out interval.json` 先在歌内求均值再等权重采样。单歌不给置信区间。该工具不是“显著性保证”，成对方法比较应对每首歌的配对差值重采样。

新增成对比较命令：

```bash
python -m choreobench compare --rows /runs/ratings.json \
  --method-a concat --method-b hierarchical --resamples 2000 --seed 7 \
  --out /runs/paired-comparison.json
```

每行包含 song_id/group_id/method_id/value，可带 reviewer_id 和 origin（human 或 model_judge）。先在歌曲内对每个方法求均值，再计算 B−A；主估计每歌等权，置信区间以完整来源组为单位重采样，另外报告等组权重诊断。单来源组不给置信区间；不混合人工与模型评分，不把重复 reviewer 或缺失评分自动计成有效样本。

默认任何歌曲缺一个方法即失败；显式 --allow-incomplete 才使用交集，并报告遗漏。完全不在 rows 中的歌曲无法由此工具发现，正式比较仍须与原数据清单核对。分数正差只代表B数值更大，是否更好取决于所选指标；不自动生成p值或领先结论。

## 7. 资源与实验顺序

1. CPU 软件检查：普通开发机，无真实音乐质量结论。
2. 服务器真实验收：一个合法 5.0625 秒样例，确认 checkpoint/处理器/格式/显存/耗时，再试一首完整歌曲。
3. Pilot：约 12 首完整歌曲用于协议和成本试验；完整任务答案、人工评审和组级隔离需补齐。
4. 固定协议后扩充正式数据，冻结测试集和评审规范，再比较多方法与消融。

现原生入口是单设备完整 BF16 Thinker 加载，不自动量化或多卡分片。可优先寻找单张80GB级GPU、128GB级主机RAM及足够模型/日志存储的服务器进行验收；这是容量规划起点，不是显存可运行保证。完整参数存储就约63.44GB，A3B不能解释为仅需加载3B。外层指令模型可用独立服务，避免与Qwen争用显存。真实内存与速度以pilot测量为准。

尚需用户准备：完整歌曲与分组、训练好的实际 checkpoint 与冻结环境、指令模型服务、编舞/音乐评审人员。当前未包含：正式数据发布、真实GPU结果、自动歌曲分组、完整音乐反事实资产生成、自由指代定位、事件增删、递归自改进训练或H3渲染。相关实验协议已记录，但不把它们写成现有代码能力。

## 8. 冻结模型的推理 harness

本轮实现不启动训练，不创建优化器，不更新权重。已有checkpoint是局部专家的训练成果；新增流程使用它的候选和现有指令模型做推理。运行无需新SFT数据，但真实歌曲、编辑任务、独立评审与开发校准数据仍是benchmark的必要输入。index0只是接口对照，LLM选择和既有偏好分数都不能直接证明编舞质量。

```bash
# 首先用现有run和export-candidates得到固定候选库。
python -m choreobench plan-units \
  --dataset /data/choreobench/dataset.with_structure.json --song-id song01 \
  --out /runs/song01.units.json
python -m choreobench run-harness \
  --dataset /data/choreobench/dataset.with_structure.json \
  --candidates /data/choreobench/local-candidates.json \
  --config choreobench/examples/harness_index0.json --out-dir /runs/harness-index0
python -m choreobench run-harness \
  --dataset /data/choreobench/dataset.with_structure.json \
  --candidates /data/choreobench/local-candidates.json \
  --config choreobench/examples/harness_llm.json --out-dir /runs/harness-llm
```

候选库、清单与选定歌曲必须严格匹配；新输出目录不可覆盖。run-harness自动沿用候选库的window/stride；独立plan-units使用非默认stride时须显式给 --stride-seconds。修改示例中的服务地址和真实模型名后再运行llm。

unit_config默认meter_status=unverified，按时间回退；审查计量结构后可显式声明reviewed并按重拍合并四小节，受max_unit_seconds约束。reviewed不是工具提供的正确概率，也不等于真正乐句检测。弱起、段落切入和尾部碎片保留。跨块窗口只按ownership中点分配一个选择责任，其他块读其交叠引用，不再做第二次冲突选择。

三种模式：index0固定选候选0，不调用模型；llm对每个有责任窗口的规划块调用一次文本模型，只能返回候选ID；scores读取绑定候选/清单/options的单窗与相邻分数，执行固定目标动态规划，不创建评分模型。合法分数下无完整路径按歌曲弃答，非法评分合同则停止。评分文件schema为choreobench.selection_scores.v1，必须有dataset_sha256、bank_sha256和songs；每歌含options_sha256/unary/transitions/score_provenance。选项对象在options/落盘，object_sha包括其record_sha256字段。分数本身需由经记录的外部来源提供，不能把全零占位当质量评审。

timing_policy默认strict：完整caption不能放入ownership时省略该片段、记录真实缺口，不填“休止”。proposed_crop可发出裁剪后的新排程提案，但明确标为pending语义检查。来源原文保持，入口/出口未知，模型并未预测到真实动作边界。complete仅指队列执行结束；all_songs_succeeded也只表示有结构合法计划，仍需查看compilation的coverage、needs_review及独立评价。

max_calls预检所需选择调用；max_output_tokens限制每次输出；max_request_characters限制实际消息字符数，超过则明确记失败，不截断动作。字符数不是tokenizer上限，服务真实输入上下文仍需另验。暂不自动重试或循环修复。已知输出格式/范围失败保留在歌曲分母，网络/身份/未知结果停止并保留成本。

产物包括contract.json、calls/、units/、options/、selections/、compilation/、plans/和results.json；同一配置加--resume恢复。可继续用export-submission和review-pack接统一benchmark。候选采集、当前文本推理和评分来源的synthetic状态合并传播，原局部采集成本单列。

## 9. 常驻局部专家服务

单次旧run已经复用一个NativeProvider，但不同CLI进程会重新加载。新增serve-local在一个前台服务进程中保持同一个原生adapter，首次有效请求懒加载，之后串行复用；不会重新训练模型。服务使用既有checkpoint校验，不自动把自定义checkpoint切换成任意通用推理引擎。

```bash
# 修改native_server.json中的绝对模型路径；保留原训练环境。
mkdir -p /shared/choreobench
export CHOREOBENCH_NATIVE_TOKEN='YOUR_SERVICE_TOKEN'
python -m choreobench serve-local \
  --provider choreobench/examples/native_server.json \
  --audio-root /shared/choreobench --host 127.0.0.1 --port 8765 \
  --api-key-env CHOREOBENCH_NATIVE_TOKEN
```

进程启动打印identity_sha256；将该值填写到native_service_concat.json的expected_identity_sha256，客户端使用同一token环境变量。健康接口是GET /health，生成接口是POST /v1/local-caption。默认只绑定本机；跨机器时显式设置服务地址，并让双方看到同样的绝对共享音频路径。当前不上传音频，根目录包含生成窗口所在run目录。模型配置相对路径以服务启动cwd解释，示例要求使用绝对路径。

```bash
# 另一个终端运行；--out-dir必须在服务允许的共享根目录内。
python -m choreobench run \
  --dataset /data/choreobench/dataset.with_structure.json \
  --config choreobench/examples/native_service_concat.json \
  --out-dir /shared/choreobench/collection
python -m choreobench export-candidates \
  --run-dir /shared/choreobench/collection \
  --out /data/choreobench/local-candidates.json
```

服务健康状态不加载权重，也不保证显存可容纳；第一次真实推理才完成加载验收。身份pin是服务端声明与版本绑定，不是客户端独立检查远端权重。生成失败不重发，加载验证失败后服务要求重启。服务串行处理原生调用，不自动批处理或多GPU分片。常驻显存和空闲租用成本仍在。

外层选择/编辑继续使用独立文本服务，可一直部署；音乐结构结果和候选库可离线缓存，采集后局部专家服务可停止。无需为每次编舞重新启动NLF、MotionScript或H3。真实GPU性能、质量与高并发可靠性未在本地CPU测试中验收。
