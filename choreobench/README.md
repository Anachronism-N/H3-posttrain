# ChoreoBench：音乐到文本编舞的 benchmark 框架与参考 baseline

实验安排统一见[音乐到文本编舞实验执行方案](../notes/44_cvpr_experiment_matrix_20261110_zh.md)：数据、对照、规模、评价、资源与执行顺序以该方案为准。

本目录将任务定义、方法实现和评估分开。核心产物是可编辑的自然语言编舞时间线；不要求 H3、3D 姿态或视频生成。名称为工作名称，当前交付是 **benchmark 软件、协议与参考 baseline**，尚不包含正式整曲数据集、正式大模型benchmark结果或人工评测结论。原始Omni加任务提示的开发期GPU验收与失败诊断另记于验收文档，不等于科学结果。

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

hierarchical 按导入的段落生成；run-harness 保留候选选择/来源编译，并新增全曲设计→逐块创作的 coordinated 模式。局部音频观察仍为5.0625秒；两种输出的来源语义不同。真正乐句检测、自动语义修复循环尚未实现。实现与边界见 [第11节](#11-全曲协调生成-harness) 和 [HARNESS_DESIGN_zh.md](docs/HARNESS_DESIGN_zh.md)。

当前状态：已完成CPU合同与真实三首开发推理验收，尚无人评音乐质量结论；精确版本/失败/资源/最后结果见 [VALIDATION_20261104.md](docs/VALIDATION_20261104.md)，不将合成fixture或开发调参结果当正式benchmark成绩。

完整协议：[PROTOCOL_zh.md](docs/PROTOCOL_zh.md)。资源：[RESOURCES_zh.md](docs/RESOURCES_zh.md)。相关工作：[RELATED_WORK.md](docs/RELATED_WORK.md)。精确数据合同：[CONTRACT.md](CONTRACT.md)。

整曲编舞与局部编辑的主指标、重复/变奏评价、指标校准集、标注规模与实现次序见 [评价指标与Benchmark方案](docs/BENCHMARK_METRICS_20261109_zh.md)。这是评价设计，不代表已完成标注、指标校准或正式排行榜。

最新进度核对、Windows页面测试修复及利用已有三种规划条件推进的安排见 [2026-11-09进度与执行次序](docs/PROGRESS_20261109_zh.md)。

指定提交9eb7d3b、main新增训练成果及其他分支的接入边界见 [分支审查与接入建议](docs/BRANCH_REVIEW_20261109_zh.md)。

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

原始Omni的零样本编舞对照可在`local_provider`中设置固定公开`task_prompt`，优先使用原始Qwen3-Omni Instruct；必须删除checkpoint，不能改变已训练专家的前缀。该提示仅在ChoreoBench的实例上应用，不改原训练adapter；仍使用完整5.0625秒波形与严格双窗解析，不传入歌曲私有答案、前段计划或编辑指令。提示原文/哈希和实际格式化前缀写入运行证据，身份标记`choreobench-prompted-omni-base-v1`，不称为已训练编舞专家。基座是否遵循指令须真实短窗验收，失败不补写或回退mock。

Linux服务器可先使用独立单窗留证入口（配置必须为base-only native、真实完整5.0625秒canonical WAV、新输出目录）：

```bash
python -m choreobench.acceptance \
  --provider /data/choreobench/base_provider.json \
  --audio /data/choreobench/real-window.wav \
  --out-dir /runs/native-base-window-acceptance --max-output-tokens 512
```

保存身份/配置、实际模板与token/EOS、加载前后/峰值显存、主机RSS与分阶段时延；失败写acceptance.json后非零退出，不补采或回退。`accepted=true`仅表示该真实短窗通过格式和EOS验收，不是音乐质量或整个服务器长期容量验收。

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

尚需用户准备：完整歌曲与分组、训练好的实际 checkpoint 与冻结环境、指令模型服务、编舞/音乐评审人员。当前未包含：正式数据发布、正式GPU benchmark结果、自动歌曲分组、完整音乐反事实资产生成、自由指代定位、事件增删、递归自改进训练或H3渲染。相关实验协议已记录，但不把它们写成现有代码能力。

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


## 多任务编辑库存（独立于评分包）

同歌多编辑任务/多seed使用独立库存与尝试文件，不能通过换song_id或method_id规避评分接口。新入口不调用模型：

~~~bash
python -m choreobench audit-edit-inventory \
  --inventory /runs/edit-study/inventory.json \
  --attempts /runs/edit-study/attempts.json \
  --out /runs/edit-study/audit.json
~~~

库存预先列出methods、seeds与任务的dataset/source_plan/task JSON文件及字节SHA；尝试按task_id/method_id/seed定位，returned绑定after/patch，failed保留错误，未提供的格仍missing。实际patch按原权限执行并与after精确比较，不将合法/空patch当目标语义通过。原任务、数据与源plan哈希和完整分母复核；task/seed结果保留在原歌内，无自动质量总分或CI。完整格式见[合同](CONTRACT.md#independent-multi-task-edit-inventory)。后续已接通`prepare-edit-semantics`/`analyze-edit-semantics`：显式匿名包到task/method/seed桥、逐目标pass/partial/fail/unknown、完整上下文审阅范围和硬失败优先。实际裁定未提交时仍pending；已审时间上的问题率统计尚未接通。格式与命令见[独立编辑语义合同](CONTRACT.md#independent-per-goal-edit-semantics)和[执行说明](docs/HISTORY_CONTROL_20261106_zh.md#独立编辑语义裁定)。

初步论文草稿已加入真实有界修订实验：三策略共享原稿/诊断，回听是固定音乐观察，不是已训练Captioner或专家真值。新collector限制8项诊断/400字符理由，保护动作文字不增添可识别精确时间声明。原v21/v22失败和内部接受决定分别保留；accepted不是质量提升，详见[最新修订记录](../notes/39_bounded_music_reinspection_and_revision_20261110_zh.md)。

## 10. 内网公开盲评页面

公开包生成后，可在launcher对应主机启动只读网页服务，不启动模型、不占GPU。服务仅提供index/pack/空评分模板/署名和pack绑定的音频白名单，不列目录，不提供private、任意文件路径或评分写入接口；音频支持HTTP Range拖动。评分在浏览器填写并导出JSON，按既有review-import导入，空评分始终pending。

```bash
# review-pack已生成匿名pack、空评分模板及音频资产后，独占生成新的index.html。
python -m choreobench review-page \
  --public /runs/review/public/pack.json --out /runs/review/public/index.html
python -m choreobench.review_server \
  --public-dir /runs/review/public --host 127.0.0.1 --port 8780 \
  --proxy-prefix /proxy/8780
```

`review-page`只读取匿名公开包，可选`--planning-music /data/anonymous-planning-music.json`提供原采样时钟下的匿名song_id→music完整映射，作为独立辅助视图；不改评分包或要求私有绑定。生成器总是嵌入新空评分模板，不读已填写评分/名册/音频，不调用模型或上传。输入字段、计划时钟和音频相对assets路径严格检查，JSON的`<`安全转义；已有HTML拒绝覆盖，使用新输出/目录，不能删除评分工件来重建。生成报告含包/页面/辅助输入SHA与项目数，不是质量或音频身份验收。

页面另有手动“结构化时间观察”与独立`review-issues.json`导出，原`reviews.v1`不增加字段。点击事件只填整数采样点/公开ID，不自动判错；合理主题回归及无法判断分别记录，空列表不表示无问题。组织者可运行`python -m choreobench check-review-issues --public /runs/review/public/pack.json --issues /runs/review/review-issues.json --out /runs/review/issues.checked.json`检查包/项目SHA、原样本半开范围、每个事件相交和声明来源；它不认证音乐判断、名册完成或产生分数。详见[执行说明](docs/HISTORY_CONTROL_20261106_zh.md#结构化时间观察与独立原件)。另有`analyze-review-ranges --public PACK --issues ISSUES --ranges RANGES --out REPORT`，从单独审阅原件统计已审范围内问题区间并集；原件绑定issues文件字节SHA并列明checked_types。未审/缺记录/未检查类型为NA，合理回归与未知单列，不把空列表或零比例判为质量合格。完整字段见[合同](CONTRACT.md#reviewed-time-interval-diagnostics)。

页面提供四维评分锚点、公开事件ID/样本范围评论、同歌统一音频及导出；编辑方案显示公开指令/原计划，生成方案两个edit评分禁用。若填音乐分却无本人听音声明，导出被阻止；勾选不证明实际听音。可选仓库浏览器回归需已安装Playwright/Chromium，不自动下载安装或填分。

端口需先确认空闲，不覆盖其他评审页面。通过环境VSCODE_PROXY_URI对应端口访问；页面使用相对音频路径，按匿名歌曲数支持`#case-1`至`#case-N`，非法锚点回到首case。公开文件或页面变更需重启以加载新快照；服务启动验收音频哈希，请求期间拒绝被替换的音频。不把服务健康或浏览器播放通过当人工音乐质量结论。


## 11. 全曲协调生成 harness

新增 `mode=coordinated`，与原 select-only 的 index0/llm/scores 分开。它不训练新模型、不改变 H3 或局部音频输入合同。

### 11.1 当前 pipeline 与时间语义

```text
完整 canonical 音频
  → 5.0625 秒观察窗 → 原始 Omni + 固定公开任务 prompt / 合法已训练专家
  → 每窗固定 K 候选，冻结原回答、时钟、seed、EOS 与身份
  → 独立公共音乐结构 sidecar（可选，绑定原 dataset 与音频 SHA）
  ├─ index0 / llm / scores：选择整包回答 → 来源逐字编译 → strict 缺口诊断
  └─ coordinated：全曲主题/段落意图 → 逐规划块创作 → 验证 → 提交 → 因果历史
  → 可编辑 JSON Plan → 受约束 patch → 独立盲评
```

输入不是2秒音频：每次听完整5.0625秒，输出被要求写 `[0–2s]` 和 `[2–5.0625s]`。它是兼容已有双窗解析器的协议选择，不是模型能力上限、拍点/小节结果或动作真值。原始基座是创造可能的编舞，不是恢复视频中的真实动作。

观察窗、所有权、公共音乐段、规划块和生成事件是不同时间尺度。coordinated 的事件可跨原2秒caption边界，但必须位于当前规划块内；所有时间为原canonical采样时钟下的整数下标。

### 11.2 公共结构与冻结候选

```bash
# 在同一 canonical WAV 上得到可追溯的真实 Beat This detections 后运行。
python -m choreobench.music_structure \
  --dataset /data/choreobench/dataset.json \
  --detections /data/choreobench/beatthis/detections.v1.jsonl \
  --split dev \
  --out /data/choreobench/public-structure.json
python -m choreobench run-harness \
  --dataset /data/choreobench/dataset.json \
  --candidates /data/choreobench/local-candidates.json \
  --config choreobench/examples/harness_coordinated.json \
  --out-dir /runs/coordinated
```

使用前修改示例的实际服务/model，并可加 `public_structure_path` 指向新sidecar。sidecar 严格绑定原dataset bytes、歌曲和音频SHA/时钟；**不改旧清单或候选身份，不允许私有审计/目标/参考答案进入结构**。index0/llm/scores 可消费相同sidecar，作为同信息条件对照。

`music_structure` 消费带版本/来源合同的拍点预测与canonical音频谱变化：在预测downbeat处比较前后3秒标准化log频带谱，选择间距至少20秒的高novelty峰。输入检测格式与旧结果迁移见第12节；不根据文件名推断Beat This身份。参数为未校准工程初值，全部功能标签 `unknown`、`repeat_group=null`；不称verse/chorus、真正乐句、人工真值或All-In-One复现。保存配置、原novelty与输入/code哈希。独立All-In-One等预测可按严格sidecar字段导入，工具本身仍需运行/验收。

计量 `unverified` 时默认10.125秒回退；reviewed声明后才按实际重拍合并四小节，不把它当真正乐句。时间短尾目标 `min_time_block_seconds=2`：同一段内合并短尾，超过容量则平衡最后两块；不跨公开段落、不改变音频/观察窗、不强行合并弱起和本来很短的section。它不是生成动作最短时长保证。

### 11.3 全曲设计与逐块创作

1. 默认启用的全曲大纲读取全部原局部候选和公开结构，没有提供拼接总谱供复制。生成至多8个动作主题，以及每公共section的编舞意图和许可主题。
2. 每块读取相关原候选、整曲section意图、当前段允许的主题定义、逐身体部位最近已提交动作、主题使用历史与最近8条事件。
3. 模型生成新的动作/排程，而不是只选候选ID。每块默认至多3事件，由 `max_events_per_unit`（1–8）限制，不用多调用自动制造更多动作。
4. 程序验证单位范围、integer时间、顺序、身体冲突、当前段主题与实际展示的局部call_id。分配稳定ID；attributes/entry/exit 保持未知空对象，禁止补编支撑或朝向。
5. 只有通过验证并提交的块进入后续记忆。失败保留原响应和前缀、停止该歌，不把前缀当完整总谱，不自动填rest或语义修复。

**来源语义不同：** index0/llm/scores 是整包选择和原文编译；coordinated 是新编舞提案。call_id只绑定创作输入，不证明新动作被音频支持、与原文语义相同、贴拍或可执行。

### 11.4 解码、预算与恢复

- 每首新增调用 = 启用时1次纲要 + 每规划块1次生成；纲要 `outline_max_output_tokens` 与块 `max_output_tokens` 独立，`max_calls` 启动前检查。
- `structured_unit_output=true` 为服务端公开动态JSONschema：限定root、unit_id、类型/整数范围、当前主题和实际输入call_id；控制器仍检查start<end、顺序、冲突及重复引用。后端拒绝时停止，不自动回退普通JSONmode或重发。
- `boundary_ids` 自由排程以完整对象分支约束实际样本最短时长，时间字段先于动作；每块索引归零。HTTP200不能证明服务端确实执行关系约束，控制器继续复检。
- 显式 `event_schedule_policy=balanced_full_unit` 先公开可行整块排程（默认1–3段，最多按事件预算至8段），模型选一整套区间并创作动作，schema固定时间/数组顺序，控制器逐项匹配。它限制自由并发，不事后拉长旧动作或补休止；覆盖1.0是设计约束，非编舞质量证据。推荐的grid示例显式开启，默认策略仍为free。
- 已验收后端不支持uniqueItems，故不发送该关键字，唯一性由控制器检查。示例开启结构化输出；其他服务必须验收或显式更换配置。
- 字符预检覆盖包含response_schema的完整内部request，不是tokenizer上下文上限或provider实际HTTP体字节数。真实输入+输出预算应另验；超长不截掉音频、候选或答案。
- `--resume` 只恢复同输入、代码、模型和配置；未决结果不重发。新结构/提示/预算/代码用新目录。
- 入口发现自身源码身份与导入时不同，调用模型前拒绝，避免等待进程混版本；这不是完整部署安全证明，仍应冻结代码。

产物包括contract/calls/units/options/outlines/blocks/diagnostics/plans/results，可经既有export-submission和review-pack提交/盲评。API费用、原局部采集、失败和额外试验都单列，不按缓存零成本报告整个方法。

### 11.5 与直接生成的区别及当前状态

局部提案本来就是音频模型 + task prompt。global是文本模型一次读局部候选/草稿生成；它不直接听音。本轮没有“音频模型直接听整曲生成总谱”的公平对照，no-audio不能替代它。

coordinated 新增确定的外部状态、公开音乐证据、全曲设计、逐块授权、验证后提交和可审计成本；尚无自动音乐critic、语义修复、长期主题搜索或训练。**工程机制已实现，不代表音乐/动作质量已改善。**

最后三首真实开发结果、复制退化与修订失败、短动作/缺口和未完成项统一记于 [VALIDATION_20261104.md](docs/VALIDATION_20261104.md)。这三首用于提示/参数开发，不是冻结隐藏测试；先完成独立盲评和结构/时长审查，再讨论pilot扩容和正式实验。

针对 `c1bba3c` 的历史审查及五项复现见 [CODE_REVIEW_20261105_zh.md](docs/CODE_REVIEW_20261105_zh.md)。本轮修复、checkpoint/网格时间验收与实验顺序见 [NEXT_STEPS_20261105_zh.md](docs/NEXT_STEPS_20261105_zh.md)。旧实验结果不自动代表修复后版本。

## 12. 来源修复与下一轮实验

本轮修复空目标编辑的虚假成功、检测合成来源丢失、固定非原文标记、模型校验失败后对象重用，以及Windows音频内容变化的请求检查。原c1bba3c审查保留为历史记录。模型仍冻结，未增加训练；真实已训练checkpoint、音乐效果及新参数需要另做服务器验收。

checkpoint验收支持显式类型，配置参考examples/native_server.json；checkpoint不能附task_prompt覆盖训练前缀。

```bash
python -m choreobench.acceptance \
  --provider /data/choreobench/native-checkpoint.json \
  --artifact-type portable-checkpoint \
  --audio /data/choreobench/acceptance-5.0625s.wav \
  --out-dir /runs/checkpoint-acceptance --max-output-tokens 512
```

base-only与portable-checkpoint分别绑定实际audit和加载验证。无Unix resource/procfs或nvidia-smi时不可用资源值记null并附原因；CLI帮助可在Windows导入。真实验收仍要求CUDA、权重及相容环境；CPU stub通过不代表实际checkpoint已验收。

结构检测JSONL的每行必须符合 `choreobench.beat_detections.v1`：

```text
schema,id,status="ok",audio_sha256,sample_rate,num_samples,time_unit="seconds",
synthetic:boolean,detector:{name,version,model_sha256,code_sha256,parameters},beats,downbeats
```

精确字段、完整歌曲集合、canonical音频SHA/时钟、升序时间和同检测器downbeats属于beats都必须通过。非合成检测需要模型SHA；代码SHA始终需要；synthetic fixture允许model_sha256=null，但合成标记会传播到sidecar和harness。身份声明与哈希不等于独立认证。可选source_binding保留旧文件/原记录/运行回执的hash。

混合train/dev/test清单可在分析时用重复的 `--split dev` 等选项，仅输出harness当前选中split的sidecar；输入检测仍须完整覆盖原dataset且逐条校验，原dataset SHA不变。默认不指定split则分析全量；sidecar歌曲集合必须与该次harness选中集合一致。迁移不接受split筛选。

旧Beat This结果可以封装为新合同，不重跑检测或改写旧文件。detector-identity.json需按真实日志填写上面的detector对象；真实/合成和时间单位必须显式声明，与旧记录已有身份相矛盾时拒绝。

```bash
python -m choreobench.music_structure --wrap-legacy \
  --dataset /data/choreobench/dataset.json \
  --detections /data/choreobench/beatthis/detections.old.jsonl \
  --out /data/choreobench/beatthis/detections.v1.jsonl \
  --detector-identity /data/choreobench/beatthis/detector-identity.json \
  --synthetic false --legacy-time-unit seconds
```

可加 `--run-receipt` 绑定原输出文件；迁移保留原文件及逐条字节hash。不能通过补填身份把fixture变成真实模型证据。

新 [harness_coordinated_grid.json](examples/harness_coordinated_grid.json) 显式采用0.25秒时间网格和0.5秒事件下限作为开发起点。模型用start_boundary/end_boundary选择公开边界表，控制器编译到精确原采样下标；过短动作拒绝，整块不可满足则在模型调用前记录该歌失败。时间网格不是节拍/动作真值，下限不是物理质量评分。旧global_samples协议保留，不自动修改旧journal。

该下限约束coordinated生成，不自动施加到旧计划或既有编辑patch；修改事件时间的编辑仍需独立时长/语义复核。

编排诊断改为实际 `literal_source_copies`，比较引用原回答和解析片段的逐字相等比例；不再固定声明“非原文”。无目标编辑记NA/not_evaluable，不能记为成功或把NA转0后汇总；已有明确且已满足的目标允许合法no-op。

评审音频请求先对实际有界内容核验SHA并关闭源句柄，再从同一bytes服务GET/Range/HEAD；每请求临时RAM与完整音频大小相关。评分仍只在浏览器填写/导出，由独立导入器处理。

完整服务器步骤、资源、消融和正式数据门槛见 [NEXT_STEPS_20261105_zh.md](docs/NEXT_STEPS_20261105_zh.md)。

最新v11生成结果、v12评审名册及后续质量/编辑推进决策见 [PROGRESS_20261106_zh.md](docs/PROGRESS_20261106_zh.md)。共享大纲接口与历史验收见 [HISTORY_CONTROL_20261106_zh.md](docs/HISTORY_CONTROL_20261106_zh.md)；早期b1eb0a1审查保留于 [PROGRESS_20261105_zh.md](docs/PROGRESS_20261105_zh.md)。

## 13. 无训练的大纲与历史消融

已实现coordinated配置 `use_outline/use_history`（默认均true）、完整request大小检查、真实排程饱和诊断，以及不调用模型的实验准备入口：

```bash
python -m choreobench prepare-ablation \
  --dataset /data/choreobench/dataset.json \
  --candidates /data/choreobench/local-candidates.json \
  --config /data/choreobench/coordinated-grid.json \
  --out-dir /runs/outline-history-v1
python -m choreobench.experiments --verify /runs/outline-history-v1
```

产物包括四组配置（main/no_outline/no_history/no_outline_no_history）、独立run路径、来源/源码/Python身份及调用预算清单、argv命令数组；不会自行运行模型或训练。关闭大纲不发起大纲调用且无生成主题，关闭历史不把前块动作/终态/计数/前缀hash交给模型。四组同块seed、候选和排程保持一致；上下文和实际费用会不同。

准备失败或预算不足时保留清单并exit2；ready_to_run不等于模型/服务/质量验收。按需执行某组命令，每组执行前先verify；修改输入、配置或代码须新目录。详见 [ABLATIONS_20261105_zh.md](docs/ABLATIONS_20261105_zh.md)。

## 14. 固定同一大纲的历史对照与人评汇总

真实v10显示：即使大纲请求与seed相同，两个独立run仍可能产生不同大纲。严格历史对照现在可先从预先指定的完成run导出公共大纲，再由两组共同读取：

```bash
python -m choreobench export-outlines --run-dir /runs/v10/main --out /data/shared-outlines.json
python -m choreobench prepare-history \
  --dataset /data/dataset.json --candidates /data/local-candidates.json \
  --config /data/coordinated-grid.json --outlines /data/shared-outlines.json \
  --out-dir /runs/history-fixed-v1
python -m choreobench verify-history --experiment /runs/history-fixed-v1
```

按commands.json分别执行两组后，运行 `audit-history --experiment /runs/history-fixed-v1 --out /runs/history-fixed-v1/audit.json` 检查实际大纲与请求，不只检查配置。共享来源变更禁止恢复，原大纲成本单列，新run只调用逐块文本模型。旧prepare-ablation保留四组独立生成大纲的语义，不能用它声称中间内容固定。

`analyze-reviews` 从公开包、私有绑定和原始评分文件验证完整歌曲×方法分母，保留缺计划、空评分、NA；按指定人工或单一模型裁判版本分析。可添加`--reviewer-roster /data/reviewer-roster.json`核验预登记评审者×方案×当前维度：整位未提交、空评分和未分配方案单独报告，未登记评分不会替代缺失项或混入均值。每格已有一份分数不代表全部评审者完成；有名册时`--compare`还要求`reviewer_roster.complete_for_roster=true`。可同时添加`--review-design /data/review-design.json`绑定完整参加者×全包项目×适用维度设计，防止名册少登记一项后“11/11完成”冒充“12/12完成”。设计缺项/多项时比较阻断；`review_design.complete_for_design`只说明任务分配匹配，不表示已有质量评分。名册及独立设计均绑定公开包SHA，字段与使用示例见[执行说明](docs/HISTORY_CONTROL_20261106_zh.md#预登记评审名册与评分完成率)。无名册时不猜人数，不认证真实听音或ICC。分组未审时只描述，满足完整配对与来源组条件才计算可选比较。所有命令、资源、真实进度和限制见 [共享大纲历史对照与人评汇总](docs/HISTORY_CONTROL_20261106_zh.md)。


## 15. 已有修订候选的独立审查（零模型调用）

`python -m choreobench.candidate_review` 从预登记的完整 revision experiment manifest 读取三臂/全部步骤，验证原稿、patch、candidate、task、decision、内部 verdict、最终提交与调用账本的哈希和身份；不构造 provider，不更新权重，不重采样。

输入库存为 `choreobench.candidate_review_inventory.v1`：

```json
{
  "schema": "choreobench.candidate_review_inventory.v1",
  "experiments": [
    {"experiment_dir": "/data/revision/song01", "manifest_sha256": "REPLACE_WITH_ORIGINAL_MANIFEST_FILE_SHA256"}
  ]
}
```

```bash
python -m choreobench.candidate_review prepare \
  --inventory /data/review/inventory.private.json --out-dir /data/review/candidates-v1 \
  --seed 7 --context-seconds 5
python -m choreobench.candidate_review analyze \
  --review-dir /data/review/candidates-v1 --out /data/review/preferences.pending.private.json
python -m choreobench.candidate_review analyze-targets \
  --review-dir /data/review/candidates-v1 --out /data/review/targets.pending.private.json
```

输出根的 `public/` 是所有非空候选与各自 before 的匿名 A/B 配对，包括内部 uncertain/rejected 以及当前时间 guard 拒绝的候选；不暴露哪一侧是 before、策略、诊断或内部接受标签。`targets/public/` 是全部三臂选定位置去重后的原稿必要性审查，不显示模型诊断。两页都保留完整歌曲与时间线、突出目标和共同上下文；局部模式不评分长期结构，整曲组织另到原评审包测。

只分发两处 public 目录。private 保存原件映射、当前 guard 离线结果、全体步骤/失败/no-op 分母和重复内容计数；它们不经 review_server 公开。公共页面和 blank forms 不是评分，audio_seen 与实际已审区间初始均空/false。重复 before 或相同 candidate 保留方法分母，不算新增独立歌曲。

用已有只读服务分别部署（端口只是示例，先确认空闲）：

```bash
python -m choreobench.review_server --public-dir /data/review/candidates-v1/public \
  --host 0.0.0.0 --port 8795 --proxy-prefix /proxy/8795
python -m choreobench.review_server --public-dir /data/review/candidates-v1/targets/public \
  --host 0.0.0.0 --port 8796 --proxy-prefix /proxy/8796
```

评审者分别导出既有分维 reviews/time issues、独立候选偏好 `candidate-preferences.json` 和原稿必要性 `target-judgments.json`。新增格式绑定 pack、focus、每个 item SHA 和真实匿名 reviewer ID；choice 为 a/b/tie/unknown/null，必要性为 yes/no/unknown/null，须另填理由与实际已审半开采样区间。导入示例：

```bash
python -m choreobench.candidate_review analyze --review-dir /data/review/candidates-v1 \
  --reviews /data/review/r1/candidate-preferences.json /data/review/r2/candidate-preferences.json \
  --out /data/review/preferences.analysis.private.json
python -m choreobench.candidate_review analyze-targets --review-dir /data/review/candidates-v1 \
  --reviews /data/review/r1/target-judgments.json /data/review/r2/target-judgments.json \
  --out /data/review/targets.analysis.private.json
```

缺少听音或共同范围覆盖时判断保持 pending；model_judge 与 human 分开，重复 reviewer×pair/target 不累计。逐评审映射为 candidate/before/tie/unknown，仅记录可能误接受/漏选好候选，不自动形成共识、显著性或质量赢家。尚未提供真实名单时 full_design_complete=null，不将一份评分冒充两名评审完成；原稿选点抽查不估计全曲诊断召回率。当前 guard 是字面检查，不是语义校验，历史 accepted 不回写。

后续审查补齐了派生配置/命令到实际run合同的对应，以及raw模型返回到patch、观察和内部判定的语义链；同一reviewer ID在一次导入中不能跨条目/文件改变origin或model身份。分析中的`song_id_count`只是歌曲ID数，`independent_song_count=null`，不由ID不同推断来源独立。合成/回放声明会继续传播；旧receipt缺少声明时保持unknown，声明为真实也不等于独立认证。

新prepare先随机排列非空pairs再编号，避免编号固定暴露策略遍历顺序。示例seed=7仅用于开发；正式评审私下冻结整数seed，评审期间不分发seed、private映射或guard结果。重复正文仍可能使角色被猜出，不能保证内容不可推断。新包的SHA和匿名映射会变化，使用新目录并保留旧包，不混导旧评分。实际人评安排和审阅边界见[最新审计与实验方案43](../notes/43_candidate_review_audit_and_next_experiments_20261110_zh.md)。

服务器v24已对原三首v22产物用新绑定规则重审通过（0模型调用/0GPU），新版先评选点 `/proxy/8799/`、后评候选 `/proxy/8798/`；这是原内网代理下的相对入口，不是完整访问地址。旧v23包与页面保留，旧导出不混入新pack；两位评审24条主判断尚待实际提交。详见方案43第8节。


## 16. 固定原前缀的历史内容重放（有界开发探针）

`python -m choreobench.memory_replay` 对最多三首、每首一个预先选定的非首块，比较full与metadata_only。只重放该块，不重生成全曲、不更新后续历史、不修改原Plan或候选，不调用音频、大纲、critic或训练。

metadata_only只从三处历史event移除action：recent_committed_events、boundary_context.body_parts.latest_event、motif_memory.recent_examples。保留时间/轨道/ID、已声明exit状态、gap、主题计数、大纲意图/主题description和原局部候选；因此不是“完全没有动作语义”。不额外添加反复制prompt，不截断正文；系统提示、JSON序列化规则、源seed、schema/排程与输出token cap共用。source_prefix_sha256仍绑定完整原前缀，不冒称投影后前缀SHA。

输入库存为如下格式（SHA是原文件字节哈希，路径为示例）：

```json
{
  "schema": "choreobench.memory_replay_inventory.v1",
  "cases": [
    {"run_dir": "/data/coordination/song01/main", "contract_sha256": "ORIGINAL_CONTRACT_FILE_SHA256",
     "results_sha256": "ORIGINAL_RESULT_FILE_SHA256", "song_id": "song01", "unit_id": "pu0018"}
  ]
}
```

```bash
python -m choreobench.memory_replay --inventory /data/replay/inventory.private.json \
  --provider /data/replay/text-provider.private.json --out-dir /data/replay/runs
# 只有输入/provider/代码全部不变时，才在同一命令上加 --resume
```

provider沿现有chat_completions/replay合同。加载前验证原contract/result、上一个已提交块、目标块、原始started/raw、原请求的历史/候选/主题/时钟，以及原响应确实编译为保存目标块；派发上限为case数×2。原模型服务名称可以不同，本次两臂使用同一实际服务/权重/环境；full重放不承诺与旧版本推理逐位一致。

输出保留started/raw、各臂原prefix、新events与prefix＋该块plan、完整失败和tokens/延迟；非法或截断不重抽，基础设施/未知结果阻止后续派发，原件不覆盖。相同条件恢复复用已落盘请求，不新增模型调用。逐字历史复制数、字符数和区间仅用于诊断；动作质量需另外读内容，整曲质量及误差累积尚未被本局部探针验证。它是独立开发入口，不是run-harness新增全曲metadata_only模式；boundary_compact/动作复杂度预算/oracle内容修订仍另行开发。

完成结果的恢复会先核对全部调用库存、请求/响应、导出原件和统计，缺件或变化时拒绝，不能通过补发调用修复历史。provider身份在合同中深拷贝，每次调用前后核验，漂移停止后续请求；源请求对照冻结的公共输入重建payload，拒绝未知字段/私有评价目标，原system正文保持原件。

用新代码只读审核旧v25结果时，使用独立入口，不使用`--resume`：

```bash
python -m choreobench.memory_replay --audit-only --out-dir /data/replay/runs
```

`--audit-only`不接受provider/inventory/resume，不创建provider或写入run目录。它按保存合同寻找原来源，核验完整结果、原请求、调用记录、四份派生JSON和统计；保留历史code哈希，不以当前code哈希冒充旧执行代码。只支持已完成结果，缺来源、缺调用或原件变化会报错；原绝对来源路径需仍可访问，不自动迁移或改写绑定。输出`new_model_calls=0`与`quality_verified=false`；一致性审核不是部署权重认证或编舞质量评价。

本轮发现、修复范围和下一项6/9调用的固定时段动作预算方案见[报告46](../notes/46_memory_replay_review_and_duration_experiment_20261110_zh.md)。
