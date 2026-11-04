# 音乐编舞 Harness 的执行设计与迁移方案

目标是把已有5秒音乐编舞专家扩展成可评测的整曲规划与局部编辑系统。第一阶段冻结局部专家，增加可追溯的候选选择和有限修复；global保留为零样本对照。本文以 `20f7cc2` 的实现为基线，规划块、受限选择器和自动修复循环属于待实现设计；本文新增文档不会自动启用这些能力。

## 当前划分与三个时间尺度

当前 `audio.make_windows` 只生成完整的5.0625秒观察窗；`baseline._outer_plan` 按导入的 `music.sections` 逐段调用外层模型。代码没有音乐乐句检测，也没有每四小节划分。`prepare-audio` 默认只有一个全曲unknown段，未导入真实结构时，hierarchical仍可能对整曲做一次大的section生成。

| 层 | 含义 | 当前或建议边界 | 是否可当真实动作边界 |
|---|---|---|---|
| 功能段落 | intro、verse、chorus等结构 | 外部工具预测，保留原始时间 | 否 |
| 规划块 | 控制候选组织、上下文和编辑范围的工程单元 | 建议先试四个完整小节；不完整部分单独记录 | 否 |
| 专家观察窗 | 现有模型实际读取的波形 | 固定5.0625秒，首版维持原采样配置 | 否 |
| 计划事件 | 输出中的动作及其排程 | 局部caption的名义时间与经过记录的ownership变换 | 属于提议的排程，非观测到的动作起止 |

“四小节规划块”不等于音乐学意义的乐句。真实乐句还依赖旋律、和声收束与重复结构；先用可复现的计量结构做工程分组，再用人工审查或额外模型研究语义乐句。没有必要把乐句检测作为首轮实验的前置依赖。

选择这一尺度的假设是：固定秒数不随速度改变，而按小节组织更便于安排段落对比、主题回归和局部修改。它是否提高编舞质量，需要比较固定时间块、四小节块、四小节加功能段落三种设置；共用候选、模型与调用预算。不得预先宣称最优。

## 整体执行流程

下图的“新增”节点均为设计。已实现的固定section生成路径仍保留作baseline。

```mermaid
flowchart TD
    A["完整歌曲"] --> B["规范WAV与整数采样时钟：已有"]
    B --> C["All-In-One或Beat This：外部工具"]
    C --> D["段落与重拍证据"]
    D --> E["构造规划块与交叠索引：新增"]
    B --> F["5.0625秒观察窗：已有"]
    F --> G["已训练Qwen，固定K个候选：已有接口"]
    G --> H["原始候选库与采集成本：已有"]
    E --> I["任务合同与组织目标：新增"]
    H --> J["候选整包选择与图搜索：新增"]
    I --> J
    M["已提交历史与主题记忆：已有基础"] --> J
    J --> K["按候选来源编译计划：新增"]
    K --> L["结构和编辑硬校验：已有<br/>候选引用和裁剪校验：新增<br/>语义审查：另接"]
    L -->|"通过已定义的检查"| N["版本化提交与状态更新：已有基础"]
    N --> M
    L -->|"候选或局部问题"| R["有限换候选或patch修复：新增控制器"]
    R --> J
    N --> O["完整计划"]
    O --> P["显式目标定位与受约束编辑：已有"]
    O --> Q["独立盲评与歌曲组统计：已有"]
    P --> Q
```

Harness持有任务状态、可调用工具、预算、证据、版本和转移规则。局部Qwen提供动作提案；文本LLM提供有限判断或编辑提案；确定性程序执行来源、时间和权限检查。最终隐藏评估位于运行控制器之外，不能被用来搜索本首测试歌的最优答案。

## 音乐分析工具的实际接法

最小部署先采用一种完整计量结果，避免混合不一致的beats与downbeats。All-In-One可同时提供功能段落；Beat This可作为节拍对照或替换计量来源。

| 工具 | 已核验官方接口 | 输出与用途 | 必须保留的边界 |
|---|---|---|---|
| [All-In-One](https://github.com/mir-aidj/all-in-one#usage-for-python) | `allin1.analyze(wav, include_activations=True)` | beats、downbeats、beat_positions、segments；功能段落用于粗层组织 | 不输出真正乐句；默认[小节解码](https://github.com/mir-aidj/all-in-one/blob/main/src/allin1/postprocessing/metrical.py)限定每小节3或4拍 |
| [Beat This](https://github.com/CPJKU/beat_this#python-class) | `File2Beats(checkpoint_path="final0", device="cuda", dbn=False)(wav)` | beats与downbeats；变速情况下仍使用实际时间序列 | 无DBN路径避免其DBN的固定3/4拍配置，但仍可能错拍、漏拍；不输出功能段落 |
| [librosa](https://github.com/librosa/librosa/blob/main/librosa/segment.py) | `segment.recurrence_matrix`；[`sequence.dtw`](https://github.com/librosa/librosa/blob/main/librosa/sequence.py) | 后续给重复段落、近似结构对应提供声学相似性证据 | 相似不等于同一乐句，也不等于应该复用同一舞蹈；需要适配特征与校准 |

Python调用示意可在独立音乐分析环境执行：

```python
import allin1
from beat_this.inference import File2Beats

# 首轮可只运行All-In-One；两种结果用于对照时分开保存。
analysis = allin1.analyze("/data/canonical/song.wav", include_activations=True)
beats, downbeats = File2Beats(
    checkpoint_path="final0", device="cuda", dbn=False
)("/data/canonical/song.wav")
```

分析工具与Qwen必须消费同一canonical WAV。不要让一个工具解码MP3、另一个工具读取不同解码器产生的波形；[All-In-One说明](https://github.com/mir-aidj/all-in-one#concerning-mp3-files)特别指出这种做法可能带来时间偏移。保存原始结果、模型/版本、波形SHA与转换回执；秒只在导入时转换一次为整数采样点。

当前 `import-structure --format allin1` 只保留beats、downbeats和segments，不保留beat_positions或activations；这些额外证据先放独立sidecar，后续再版本化扩展合同，不能悄悄加入严格Song schema。标准结果没有已校准的“边界正确概率”，不要把logit或激活直接当置信度。

All-In-One代码为MIT；Beat This官方说明代码和发布权重为MIT。音频、训练数据与其他依赖许可仍按各自来源处理。首轮不需要重新训练这两个模型。

## 规划块的明确构造算法

首版参数建议为 `bars_per_block=4`、`max_unit_seconds=20`、`fallback_seconds=10.125`。这些是待消融的工程初值，不是论文已验证超参数。

1. 验证功能段落形成全曲分割，实际downbeat时间严格递增且落在音频范围内。重复时间、越界和缺失字段明确失败，不静默去掉。
2. 连续两个downbeat之间构成一个预测小节。使用实际downbeat序列，不以全局BPM从零秒外推。BPM可能变化，首拍也可能不在零秒。
3. 在每个功能段落内部，从第一个可用完整小节开始，连续四小节组成一个 `metrical_block`。不跨verse/chorus等段落边界。
4. 首个downbeat前的音频、末尾未封闭小节、段落切入的小节、剩余不足四小节，分别保留 `partial_fragment` 与具体原因；覆盖时间不能丢失。短碎片可被相邻任务读作上下文，但不重复写事件。
5. 对错拍、缺重拍或复杂拍号不能可靠解释的片段，记录 `meter_unverified`，改用明确标注的10.125秒 `time_block`，或进入人工审查。工具不提供可信的判错概率，首版先结合输出一致性检查和抽样人工复核；规则与阈值在开发集冻结。
6. 得到规划块后，再建立它与专家观察窗、ownership及名义caption区间的交叠索引。规划块只增加上下文和定位能力，不拉伸、补零或重切现有模型波形。

为限制上下文，四小节超过20秒时，在可用重拍处减少合并小节数并记录实际bars；单个预测小节也过长时进入time_block或人工审查，不强行解释为正常乐句。计量、时长和外层输入token上限都应由控制器检查。

例：假设120 BPM、4/4，downbeats为 `0.72,2.72,4.72,6.72,8.72,...`，四小节块为 `[0.72,8.72)`；`[0,0.72)` 是前缀片段。如果功能段落在7.90秒结束，保留该原边界，将剩余部分标为不完整，不能擅自把段落改成8.72秒。四小节在不同速度下时长不同；这也说明它不能替换5.0625秒专家窗口。

规划sidecar的设计示例：

```json
{
  "schema": "choreobench.planning_units.proposed.v1",
  "audio_sha256": "CANONICAL_WAV_SHA256",
  "units": [
    {
      "unit_id": "chorus2-block1",
      "section_id": "s005",
      "kind": "metrical_block",
      "start": 512000,
      "end": 640000,
      "bars": 4,
      "boundary_source": "tool_downbeats",
      "semantic_phrase_verified": false,
      "window_refs": ["w006", "w007"]
    }
  ]
}
```

示例假定16kHz、32–40秒。该JSON是拟议接口，不是现有CLI可直接读取的schema；fragment和fallback条目应使用其真实kind及缺失bars值。

## 局部候选如何采集

保持当前输入合同：规范波形的完整5.0625秒，无大纲、前文、歌词、目标动作或编辑指令。模型每次返回原固定双窗caption。外层不能通过“更有力量”文字指令直接控制这个audio-only专家；需要从不同采样候选中选择，或另行训练条件输入。

首个pilot先试 `K=2`，再视候选差异与成本扩到4。固定seed派生、temperature/top-p、最大输出和checkpoint。每个候选保留原始回答、两个caption、实际观察窗、原始请求与停止原因。select_only必须整包选择一次响应的两段caption；混合A前半段与B后半段须另立新组合、记录父来源并重新检查，不能仍称为一次专家原始提案。

现有 `candidates_per_window` 与 `export-candidates` 已能完成采集和不可变保存。坏输出保留并进入失败分母；候选库不包含测试参考动作或人工终评答案。改变窗口/seed/K需新采集身份，不能在看到评测反馈后补采并假装同一预算。

## 规划块与观察窗不能硬对齐

例如一个规划块是32–40秒，现有两个相关观察窗可能为 `[30.375,35.4375)` 和 `[35.4375,40.5)`。它们保留原长度；选择器可同时读到它们与规划块的交叠情况。一个观察窗跨两个规划块时，全曲只产生一个候选选择，两块共享该选择，不能在边界两侧分别选两个冲突答案。

首版选择顺序仍沿专家window图推进；规划块提供组织目标与历史索引。提交和状态更新使用实际已提交的ownership时钟，不把“四小节末尾”当作模型已预测的动作结束。总谱显示可按规划块分组，跨块事件只引用同一稳定ID。

需要补验的现有语义问题是：`_concat` 将caption对应的名义区间裁到ownership，却原样保留整段动作文字。例如12秒音频的尾窗左移后，中间ownership可能只剩1.875秒，原两秒caption若写“转两圈”，裁完时间并不能证明动作仍完整。

新编译器应记录 `source_interval`、`scheduled_interval`、`partial_source`。完整包含的动作可保留；部分裁剪、次数动作和跨边界事件进入 `needs_review`，不能以格式合法当作语义保持。可选择保留跨块事件、做有依据的局部改写，或报告未解决区间；具体策略作为消融。不得把未知区间填成“休止”来提高覆盖率。首版也不把所有事件自动snap到拍点后再宣称同步改善。

## 外层模型只做明确的有限决策

先让程序生成每个规划块的合同：可见音乐结构、局部候选ID、前后可用上下文、用户要求、可变范围、受保护字段。主题回归或渐强是 `design_goal`；它们不是音频中已经观测到的事实。重复的chorus标签不能自动证明两段旋律相同；重复组需声学证据或人工核对，保留不确定状态。

第一条新增方法 `select_only` 限制外层输出候选ID、比较理由、证据引用和不确定项。动作文本与名义区间由程序从原候选读取。生成器不能在同一接口中擅自写新动作、改时间、改变身体轨道或抹除来源。

```json
{
  "schema": "choreobench.selection.proposed.v1",
  "decisions": [
    {"window_id": "w006", "candidate_index": 1},
    {"window_id": "w007", "candidate_index": 0}
  ],
  "uncertain": [],
  "reason": "Choose a different arm action while retaining the earlier step theme."
}
```

程序验证每个window恰好一个合法candidate；无法选择时允许明确弃答。需要新动作或修改时走单独patch接口，记录 `proposed_by_editor`；改后不再称为原候选逐字支持。

动作事实表可以作为第二阶段模块：从caption中提取部位、方向、动作和显式入口/出口，并附原文span；缺失为null。“抬右腿”不能自动推出“最后右脚支撑”。LLM提取结果仍是解释，不能直接变成硬性人体物理约束。当前concat通常只有body轨道与空entry/exit，生成可靠细粒度编辑前必须先解决事实拆分与审核。

## 候选选择如何计算

每个观察窗是一层节点，每个完整回答是一个节点。先使用相邻窗口图，计算本地偏好与相邻衔接；主题长期依赖确有收益后再扩为带状态beam search。

```text
dp[i,k] = unary[i,k] + max_j(dp[i-1,j] + transition[i-1,j,i,k])
```

`unary` 只来自明确的公开要求、已收集的独立偏好或经校准的评审；`transition` 表示文字动作的衔接判断。评分方向、缺评策略、平分时规则和权重在开发集冻结。这个搜索仅在给定分数上找路径，不会自动产生音乐理解能力。无可信音乐评审时，先报告“文字组织选择”，不能把文本LLM分数改名为音乐适配分。

transition比较的是ownership编译后真正相邻的片段，而不是重叠观察窗原始的“末句→首句”。部分裁剪尚未审核时，衔接保持unknown。若分数依赖历史主题或其他上下文，普通一阶DP不再保证完整目标最优；应扩展状态或使用预算内beam search。

硬约束由程序处理：候选存在、window唯一、时间身份、权限/保护、合法轨道与引用。动作语义、真实可执行性和音乐贴合由软审查或人工判断；未知状态既不自动通过，也不作为确定冲突剔除。

为控制成本，保存候选对评审缓存，缓存键包括源回答、时间编译策略、公开合同、所用上下文、rubric、评分器及提示版本；只有同一评价请求才复用。可以每个相邻window一次请求返回全部K×K比较，但必须对每项显式绑定候选ID且记录全部token。单次请求批量处理不等于少了评审工作量。改用局部beam时，保存被丢弃路径和原因，不仅保留最终赢家。

例：180秒、默认stride约36个window，K=2为72次局部采样；相邻候选对最多 `35×2×2=140` 项。若按相邻window批量返回则最多35次边界评审请求，另计组织、编辑、故障与独立终评成本。缓存复用避免再次跑Qwen，质量和实际延迟仍需pilot测量。

## Harness 的任务状态与有限修复

建议以现有journal为基础新增控制器，状态至少包括：

```text
prepared -> analyzed -> candidates_ready -> selection_proposed
         -> compiled -> checked -> committed
                          |
                          +-> repair_proposed -> compiled
                          +-> failed / abstained / needs_review
```

每次proposal记录父版本、输入哈希、公开合同、模型/提示版本、候选选择、检查结果和成本。只有检查完成后的版本才能更新已提交记忆。检查失败不覆盖前一版本，不偷偷增加随机重采。

首轮修复策略是先换已有候选，再考虑小范围patch；按问题所在window及明确批准的相邻修复区执行。`max_repair_rounds=2` 可作起始预算，另有整曲调用、输出token和墙钟上限。达到上限、无合法候选、连续版本重复或修改无可验证进展时，明确停止并保留失败/待审。超时且提供方结果未知时继续沿用现有禁止自动重发规则。

硬检查失败可以生成确定的修复问题；语义批评应附候选原文或音频证据，并允许“证据不足”。内部critic不接触私有测试答案。最终独立人评与内部critic分开，不能用同一模型的自评分作为方法成功证据。

这一步借鉴 [DeepSeek原型](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/main/docs/007-architecture.md)的持久任务、依赖与恢复模式。首版可用Python有限状态机与现有JSON journal实现，不必为跑整首歌引入完整多worker运行时。单曲搜索/修复属于推理过程；跨开发歌曲持续更新提示或策略是harness优化；只有机制满足递归改进定义时才讨论RSI。

## 精准编辑如何接入

已有显式选择器可解析section_id、section_label+occurrence、采样区间、track/body_scope，生成private/public task与resolution。用户说“第二次副歌上半身动作更大，脚步不变”时，下一步自由指代grounder只提出selector与候选保护条目，再由程序解析公开时间/事件索引；歧义不自动猜测。

范围解析后，编辑器只读取源计划、公开指令、目标IDs、修复IDs和保护字段。全曲版本、源计划哈希、所有未授权事件及保护字段被冻结。每次编辑输出patch、改后计划、检查报告和可撤销版本。当前schema只支持修改已有事件字段，不能把事件增删或整个body事件拆成手脚两轨说成已实现。

同一次用户请求的多轮尝试始终针对同一初始计划和冻结task提出累计patch，最终检查initial→final，避免每轮rebase掩盖累积漂移。下一个用户请求才绑定新接受版本。循环检测使用动作内容hash并排除provenance，防止每轮新日志使重复动作看起来不同。隐藏测试的private goals只在最终artifact固定后评价，不能把失败答案返给修复器。

若计划只有body文字且无法确认手脚独立事实，先标记任务粒度不支持，而不是声称脚步已保持。未来可用经人工复核的动作事实表建立分轨。编辑评价分别测定位、语义目标、保护保持和衔接，避免把给定target IDs的条件编辑与自由指代端到端任务混在一起。

## 可迁移工作与复用方式

直接复用优先调用既有接口；论文机制迁移则需要本任务的适配与实验。涉及权重、数据或代码拷贝时记录固定revision和各自许可，不能把第三方成绩当成我们的复现实验。

| 工作 | 我们复用的具体部分 | 可直接使用的程度 |
|---|---|---|
| All-In-One / Beat This | 音乐功能结构、节拍与重拍预测 | 可独立运行官方工具；现有All-In-One importer已接。真实乐句层须另做 |
| librosa recurrence / DTW | 重复结构候选、段落声学对应 | 可调用函数；我们需补特征、时间映射、阈值与审查 |
| [MAESTRO](https://midotronn.github.io/MAESTRO/) | [select_dance](https://github.com/midotronn/MAESTRO/blob/cfc77a3a1e607dfaebd0c381573d6a943616e11a/agentlodge/agent/selector.py#L117)及候选schedule选择 | 借候选→诊断→选择机制。公开selector使用运动指标与显式方法名，异常有固定fallback，不直接作为盲评器；项目页完整storyboard/编辑能力未在当前入口核实 |
| [AtomicDance](https://github.com/oceanflowlab/AtomicDance) | [infer_plan/infer_completion](https://github.com/oceanflowlab/AtomicDance/blob/7b85001babcab5379550e6f0f8cf3abc7779c8cf/infer_atomic.py)及动作原型检索 | 借计划/variation/transition分工；公开入口按window_size分块、类别为封闭集合，不等于全曲开放文本规划。代码MIT；README仍称权重将发布，本轮未确认下载可用 |
| [LODGE](https://github.com/li-ronghui/LODGE/blob/f38642ad67bc3402444f5f914010a911e5bd5c89/infer_lodge.py) | 全局粗计划与局部受约束生成 | 借组织原则；3D运动张量不直接接文本总谱。当前仓库未核到明确LICENSE，拷贝前需核实 |
| [MotionFix](https://github.com/atnikos/motionfix/blob/e6b437e6a17f26b2066f9c142600146f035da6bf/src/evaluator/evaluate_edits.py) | source/instruction/target三元组；分开edit_accuracy与motion_preservance | 迁移任务与标注设计，几何评估器不能直接评分文字。LICENSE为MIT但README仍有限制性表述；数据、AMASS、身体模型另行核清 |
| [DeepSeek prototype](https://github.com/deepseek-ai/dsh-superagent-prototype/blob/8c32a9dc5d72c033816c6a8110b37f40cf94fba4/src/app/store.ts#L138) | Store.atomic/afterCommit、Effects.enqueue、Prompts.save的父版本检查 | 借事务、恢复和版本约束，代码MIT；Effects默认重试不能原样用于计费调用的未知结果，继续保留started/raw阻断 |
| [ClawMark](https://github.com/evolvent-ai/ClawMark/blob/d1b641b3171e584e69a3763c269069f32a13b574/src/clawmark/orchestrator.py#L46) | stage执行、快照、阶段checker与最终checker | 借多轮编辑的累积检查；checker不能替代美感评审。许可CC BY-NC 4.0，直接拷贝需符合许可 |

项目内也有可复用基础：`music_planner_search.py` 的候选身份、独立音频评审合同与经审查数据导出；`motionscript_v2/selection.py` 的可逆摘要与源事件保留。前者需转换到ChoreoBench候选schema，后者依赖实测姿态事件，只适合离线标注，不可把音乐生成文字伪装成实测动作。两者均不是已有的整曲候选图选择器。

## 待实现模块与验收

下列文件名是建议，不是当前目录已有接口；当前可运行命令仍以README与CLI help为准。

| 模块 | 输入 → 输出 | 最低验收 |
|---|---|---|
| `music_tools.py` | canonical WAV → 原始结构结果与绑定回执 | 秒/样本一致、真实工具身份、不混不同解码波形 |
| `planning_units.py` | sections+downbeats → blocks/fragments/fallback与交叠索引 | 变速、弱起、残缺尾、段落非重拍边界、未知拍号均有测试；全曲无缝且不重复 |
| `selection.py` | 固定候选+公开合同+偏好 → 每window唯一选择 | 整包选择、跨block一致、坏候选/缺评/平分/全失败显式保留 |
| `compile_selected.py` | 选择+原请求 → plan+来源与部分裁剪回执 | 文本逐字来源、时钟、无默默retiming；次数动作裁剪被标待审 |
| `controller.py` | 任务状态+工具结果 → 有预算的运行轨迹 | 中断恢复、未知结果、版本重复、预算终止与失败分母 |
| `edit_grounder.py` | 文本指令+公开结构 → selector候选 | 区分自由定位与给定范围；歧义可弃答；保护范围不由编辑器扩大 |

建议实现顺序：先规划块与窗口映射，再select_only及编译器，再有限修复，最后自由指代编辑。global与原hierarchical保留作比较路径，不因新设计删除弱基线。

## 数据与实验的先后

先做真实短窗与一首整曲验收，再做小型整曲pilot。冻结候选库后让人工从同一批候选组织：人工也做不好时优先改善局部提案；人工明显更好时优先补选择监督。四小节块、候选数量、重复处理和修复预算的取舍在开发集完成。

首批新增监督可以是候选偏好、相邻衔接判断、错误定位与最小修复、编辑前后计划及保护条件。无需先写每首歌唯一的长总谱；保留多种可接受选择和不确定答案。教师生成的整曲文字只能作待审草稿。

正式评估同时报告缺失/失败、结构有效性、音乐贴合、主题发展、非预期重复、编辑目标与保持性及成本。按原曲/艺人/近重复组隔离数据；独立终评不得进入测试歌曲搜索或策略调参。文本排程的拍点分数不等于生成视频中的实际动作同步，后者需独立渲染和测量协议。

资源优先沿用 [RESOURCES_zh.md](RESOURCES_zh.md)：首次本地专家采集需要真实模型服务器；候选已固定后，选择/控制程序可在CPU运行，外层LLM由独立服务提供。人工编舞与音乐评审、完整音频来源、实际checkpoint和冻结环境是必要输入。本文的新增设计尚未执行真实模型质量实验。
