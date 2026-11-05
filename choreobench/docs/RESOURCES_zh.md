# 运行资源与成本预算

核验记录：2026-11-04初始容量规划，2026-11-05追加真实开发推理。本文区分CPU协议工具、局部模型、外层规划模型与人评；前8节保留容量/成本原则，第9–10节为已测边界与当前建议，不给普适运行保证。字段以 [CONTRACT.md](../CONTRACT.md) 为准，数据与评估要求见 [PROTOCOL_zh.md](PROTOCOL_zh.md)。实际启动命令见集成后的 [README](../README.md)。

## 1. 四类运行条件

| 工作 | 依赖与资源 | 需要实测/自备 |
|---|---|---|
| JSON数据校验、计划/patch检查、字面诊断、盲评交换 | 核心目标Python>=3.9；不需要CUDA、大模型或外部API。小pilot可从普通4核/8GB机器开始测试，此为工程估算 | 大规模manifest/审计记录的峰值RAM、磁盘吞吐；CPU通过不代表模型或benchmark质量通过。 |
| canonical音频、切窗与公共结构预测 | 音频处理依赖与所选结构工具另算；常用NumPy/SciPy/soundfile，压缩媒体解码可能需ffmpeg | 音频授权、原文件/采样率/时长，工具权重、模型版本及其CPU/GPU预算。固定BPM不能替代真实beat工具。 |
| 真实 native Qwen局部expert | 本地完整基座、原processor、兼容PyTorch/Transformers及实际checkpoint；portable LoRA路径还需PEFT | 见下节。模型文件存在、身份/源码合同和真实推理均需通过；没有mock自动降级。 |
| 外层规划/编辑/评审 | 远程文本API，或独立本地指令模型；音频质量评审还需实际听音能力或人工 | API endpoint/实际模型身份、配额/token、成本和数据使用条件；本地模型独立GPU/RAM、上下文与服务预算。 |

远程外层API不替代本地已训练Qwen expert。文本LLM只读公共音乐证据/候选时应标为text-planner；不能把其对文字的点评称为真实听音评价。本地外层模型若与Qwen共卡，峰值显存相加；先串行或分离服务，再考虑并发。核心实验不要求H3、骨架恢复或视频渲染；若做可选RGB验证，另列模型、GPU、渲染秒数、存储和误差归因。

## 2. Qwen3-Omni-30B-A3B 的实际加载边界

官方 [Captioner模型卡](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner) 将它定位为audio-to-text模型；它不等同于已训练的编舞checkpoint。官方 [Qwen3-Omni代码仓库](https://github.com/QwenLM/Qwen3-Omni) 提供通用推理说明，不能替代本项目原训练合同。

本仓库 [load_planner_adapter](../../scripts_omnidance/exp/ms/music_planner_adapters.py) 当前使用 `Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(..., dtype=bfloat16, local_files_only=True)`，加载Thinker完整子树及其MoE专家，不执行Talker/code2wav；随后 `.to(device)`。它没有在此入口配置4/8bit量化、`device_map`、CPU offload或tensor parallel。因此：

- `A3B`不是“只加载3B参数”，不能按6GB估算BF16常驻权重。
- 多卡总显存足够不代表当前单device入口自动分片；不能承诺2×40GB、2×48GB就能运行。
- 官方支持其他推理引擎，不等于本仓库的微调checkpoint、采样与身份校验已接通该引擎。更换引擎需另做等价性验收与新运行身份。

### 参数存储项与实测显存

2026-11-04只读取了官方小型metadata，没有下载权重：[固定revision](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner/tree/a2bd106cbf527db5676e79662674da22b0545ec0) 为 `a2bd106cbf527db5676e79662674da22b0545ec0`，HF safetensors metadata记录 **31,719,205,488个BF16参数**；[权重索引](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner/resolve/a2bd106cbf527db5676e79662674da22b0545ec0/model.safetensors.index.json) 的 `total_size` 为 **63,438,410,976 bytes，即约63.44GB /59.08GiB**。

这是该快照的权重tensor存储项，不是完整推理峰值。通用估算：

`M_weights = sum(numel(tensor) * bytes(dtype))`

`M_peak >= M_weights + M_adapter + M_KV + M_activations + M_workspace + M_allocator/driver`

实际加载还可能产生CPU或设备端临时副本；已训练adapter部分恢复为FP32，按实际tensor身份另计。主机RAM至少要容纳实际加载路径所需内容，不能只预留几GB。若在CPU先完整构造再搬GPU，可先按128GB级主机RAM、单张80GB级GPU作为优先验收候选；这只是容量规划起点，**不是可运行保证**。48GB级单卡通常连完整BF16参数存储项都容不下；要采用量化/offload/分片，需新增并验证相应路径。

官方 [推理说明](https://github.com/QwenLM/Qwen3-Omni#quickstart) 建议FlashAttention等方式，但其长视频显存表不是本项目5.0625秒audio-only测量。不同Transformers版本、attention backend、输入长度、输出长度、batch和并发都会影响峰值与数值。优先复用已训练checkpoint对应的锁定环境；若升级依赖，记录新版本并回归真实样例，不盲目以latest替换冻结训练源码。

服务器首轮验收应记录：模型/processor/checkpoint哈希与兼容合同、软件/CUDA/driver/GPU、加载前后显存、峰值allocated/reserved、主机RSS、5.0625秒真实音频样例、实际token/EOS/格式结果、单窗时间及异常。先batch1串行验证，再提高并发。显存不足、未知checkpoint或格式失败应显式停止并留证，不回退mock或悄悄缩短音频。

## 3. 所需模型与文件

- local expert基座本地目录；完整processor/tokenizer/config与权重文件。
- 若使用用户已训练模型：实际支持的完整portable checkpoint、原run_config、trainer_state/trainable tensors、adapter导出与所需冻结训练源码。不能只凭文件夹名、模型别名或一份任意PEFT权重声称复现。
- 有权限的完整歌曲、canonical音频与来源/分组审计。只有分散5秒片段不能验收fullsong赛道。
- 公共音乐结构结果及工具身份；私有参考、编辑目标、盲评文件分别保管。
- 外层LLM配置与实际模型身份；API密钥由环境或服务管理，不进JSON配置、日志、Git或可分享证据。
- 每次运行独立输出与journal目录，保留原始响应、请求身份、计划、patch、失败与评审。已有结果的恢复不等于可以改变模型/提示/数据后继续同一journal。

## 4. 调用数、GPU时间与API费用

设歌曲时长 `T`、局部观察窗 `L=5.0625s`、stride `s<=L`。`T>=L`时，均匀步进并把最后一窗左移到歌曲末尾，典型窗数为：

`W = ceil((T-L)/s) + 1`

最终以实现输出的去重窗口清单为准。短于L的输入不能无声补零。180–300秒、stride=L约需36–60窗；stride=4秒约45–75窗。模型观察区间可能重叠，计划所有权区间应独立记录。

设每窗候选数K，局部调用约为 `C_local = sum_songs(W*K)`；另计全局规划、逐段协调、候选选择、每轮编辑、语义评审及失败重试：

新增候选库复用路径：concat首次采集后用 export-candidates 保存局部证据，global/hierarchical共享同一批候选，无需再次加载本地Qwen。复用机器仅需音频、候选库和外层服务；必须保留首次采集身份和成本。resources区分计划局部工作量、新增局部调用、复用调用与原采集成本。max_calls仅限制当前运行的新调用，论文需另外报告首次采集成本及所采用的摊销口径。候选库包含原始响应，存储大小以真实pilot测量，不按纯caption文字大小估算。

`C_total = C_local + C_plan + C_edit + C_review + C_attempts_extra`

`GPU_hours = sum(actual_model_seconds * assigned_GPU_count)/3600`

`API_cost = sum(input_tokens*input_rate + output_tokens*output_rate + audio_seconds*audio_rate + other_billed_units)`

各价格单位需统一；不填过时固定价格。provider只返回部分usage时其余标unknown，不能记0。墙钟与GPU小时分别报，缓存命中和新调用分开；全局规划、judge和预处理也计入公平预算。串行耗时可由实测均值乘调用数作初估，但并行吞吐、排队、限流和加载时间必须另测。

journal可复用已完整落盘的请求结果；若服务端已计费而响应未落盘，请求结果未知。普通本地journal不能保证账单“恰好一次”；记录request ID与未知状态，需提供方幂等支持或账单核对，显式授权重试后再继续。

## 5. 音频、模型和日志存储

未压缩音频字节数：`S_audio = sum(T * sample_rate * channels * bytes_per_sample)`。例如200首、平均240秒、16kHz单声道：PCM16约1.536GB，float32约3.072GB，均不含容器头、原始高采样率文件与副本。压缩文件大小取决于实际码率；需同时记录canonical身份。

磁盘预算至少覆盖：基座快照＋checkpoint/adapter＋原曲/音频变体＋原始调用与token日志＋计划/patch版本＋私有评审＋备份。一个基座的共享只读快照可由多次运行复用，勿每次完整复制。身份哈希审计可能逐字节扫描大量权重，带来I/O启动成本；缓存必须有明确失效策略，不能以文件名代替内容身份。

`S_logs = sum(request_bytes + response_bytes + identity/diagnostic_bytes)`，按pilot实测P50/P95再乘调用数并留失败/修订容量。裁剪音频可按窗口身份缓存，但每曲重叠候选不必复制K份波形。备份、私有评估访问权限和保留周期由发布者明确。

## 6. 人工标注与评审预算

把工作拆开估算：音频来源/分组审计、音乐结构复核、编舞质量评审、编辑指令与目标/保护标注、复核和仲裁。不要把一次听歌时间当完整标注时间。

`H_annotation = sum_items(annotators * active_minutes_per_item)/60 + H_adjudication + H_training`

全曲听音时长只是下界，实际还需阅读总谱与比较多个版本。建议pilot先测每种任务的活跃时间、争议率、无效条目率与费用，再确定正式规模；6+6首的试验只估算流程与成本，不证明统计功效。正式比较分配盲化身份、随机顺序和独立评审，避免方法作者同时充当唯一标签来源和终评者。

编辑目标先人审，机器字段goals仅提供字面检查；自由动作语义、音乐贴合、合理变奏和边界可执行性需要独立判断。按歌曲组汇总再统计，不能把同曲几十个窗或多条编辑视为独立歌曲扩大样本量。

## 7. 服务器执行前的资源确认表

记录实际可用GPU及显存/占用、RAM/可用磁盘、模型与checkpoint路径和身份、依赖锁定版本、真实音频权限/分组、外层API或本地服务、每轮调用/token/GPU时间预算、私有评估存放位置与人工排期。缺少某项则明确阻塞对应真实实验；fixture仍可验收软件，但不能替代缺少的模型/数据/评审。

本文件没有启动下载、申请GPU、调用付费API或执行训练。实际命令只采用集成后的README和该版本CLI帮助，不将尚未实现的接口当作部署说明。

外层完整总谱的输出长度可能超过局部caption。当前实现将 `max_output_tokens`（局部）与 `planner_max_output_tokens`（外层）分开记录，后者缺省继承前者；估算token上限时应分别乘对应调用数。输入上下文还包括全曲公共结构及候选文本，实际API服务上下文上限需另验收。使用 `resources` 命令得到声明工作量与预算，不把输出上限当实际账单。

## 8. 冻结模型推理与常驻服务

run-harness不新增SFT/DPO/RL训练，不创建优化器或更新权重。模型checkpoint来自已有训练；运行输入是完整歌曲/公开结构/候选/brief，而不是新增训练样本。正式benchmark仍需来源分组、编辑任务、独立评审与开发校准数据。若局部专家的动作覆盖不足，流程跑通不能证明永远无需继续训练。

| 常驻或缓存对象 | 资源使用 | 何时可停止 |
|---|---|---|
| serve-local原生Qwen专家 | 单实例完整权重、串行推理；沿用前述80GB级候选验收与原环境，非自动量化/分片 | 候选采集完成后可停，后续直接复用候选库 |
| 文本选择/编辑服务 | 独立LLM服务或API；模型不更新；每规划块选择仍花输入/输出token | 无任务时可停或按所用平台保留常驻 |
| 音乐结构工具 | 可离线运行并缓存结果，未必须包装成常驻HTTP | 每歌分析完成后无需持续占用GPU |
| CPU harness/评估 | 规划块、候选绑定、图搜索、来源编译、日志与统计 | 单次任务完成即可退出 |

常驻减少重复加载，不免除显存、推理、空闲租用和标注成本。不要把完整Qwen与额外大文本模型默认共置同一张80GB卡；外层优先独立服务。HTTP客户端必须读取同一绝对共享音频路径，候选采集run目录放在serve-local允许的audio_root下。健康检查仅审计身份且不加载权重；首个真实请求仍必须测加载峰值、样例格式和延迟。

新run-harness先在已有候选库上运行index0 CPU验收，再接llm单曲；scores模式只消费外部已记录分数，未附带一个训练好的音乐critic。严格编译可能保留真实缺口，proposed_crop提升时间覆盖也不证明动作语义可执行。max_request_characters只是字符预检，服务端真实tokenizer/上下文上限、输出吞吐和并发须单独测量。常驻接口的软件测试使用假原生模型与真实本地HTTP，不构成真实GPU验收。


## 9. 真实开发验收：测量与容量规划分开

### 9.1 模型存储

只读safetensors文件头得到以下tensor存储项，不是GPU峰值：

| 本地快照 | BF16 tensor bytes | 用途 |
|---|---:|---|
| Qwen3-Omni Instruct完整快照 | 70,519,637,090 | 包含Thinker/Talker/code2wav |
| 仅Thinker | 63,438,410,976（59.08 GiB） | 本轮实际局部加载；不执行语音输出组件 |
| Qwen3-32B | 65,524,246,528 | 初始文本服务，原生40,960上下文不足以容纳旧全曲请求 |
| Qwen3-VL-32B-Instruct | 66,714,780,128 | 最后外层服务的纯文本模式 |

A3B不是仅加载3B。当前局部入口不量化、不自动device_map/offload/多卡分片；增加总卡数不解决单设备容量限制。

### 9.2 首窗实测

完整5.0625秒音频、batch1、原始Instruct + 固定公开任务、局部输出预算512：

- input212/output120 tokens，genuine EOS、严格双窗格式通过。
- 加载42.3576s、生成9.3238s，权重哈希审计另计。
- 峰值allocated63,578,429,952 bytes，reserved63,772,295,168 bytes（59.39 GiB）。
- 加载期主机峰值RSS120,137,420,800 bytes（111.89 GiB）；常驻RSS不能代替加载峰值。
- A800-SXM4-80GB、driver580.105.08、torch2.8.0+cu129、Transformers5.18.0；实际依赖版本/hash写入证据包。

128GiB主机RAM仍只是起点，需要额外余量用于临时副本/页缓存。节点准入取min(主机MemAvailable,cgroup余量)，计入共享内存与缓存，不只看free输出。不为加速默认复制权重到紧张的tmpfs。

### 9.3 外层上下文与显存

普通Qwen3-32B的40,960上限在真实旧global请求明确HTTP400失败；保留原始响应，不截断候选。改用已有VL32B纯文本（image/video限制0），原配置262,144，仅部署65,536，没有RoPE扩展。

旧global实际输入39,267 + 请求输出16,384 = 55,651 tokens；新coordinated首轮纲要输入约24,958/25,235/21,385。这些是具体开发请求，不可由歌曲秒数或字符数担保新输入一定可用。每模型/提示/歌曲都应做实际tokenizer预检。

同节点BF16 TP=2，gpu_memory_utilization=0.7，服务每卡nvidia-smi约57,283MiB，包含预分配KV；运行过程有额外workspace波动，不能称实际请求峰值57GiB。VL32B文本64层×8KV头×128head_dim，BF16 KV约256KiB/token（整模型）；服务常驻KV、并发、prefill和输出长度均影响资源。

### 9.4 实际调用与成本

三首252.97/230.03/217.21秒，K=2共100/92/86局部调用，全EOS、33,806 output tokens。平均单次生成7.18/7.25/8.41s，调用时延和约2,107.65s；不含加载、哈希、队列、I/O、音乐分析与文本生成，不当端到端墙钟或全部GPU小时。

音乐结构采用既有Beat This CPU输出和新谱变化CPU分析，未部署All-In-One/Demucs/NATTEN，也无自动音乐critic。

最后coordinated配置：纲要4096、块1024 tokens、每块3事件，27/26/23规划单位，新增外层28/27/24次，音频模型新增0。事件预算与token预算修订分开留证；这些不是质量分或完成时长承诺。

## 10. 当前部署与调度建议

| 阶段 | 最低建议资源 | 吞吐扩展 |
|---|---|---|
| 单真实短窗验收 | 1×80GB GPU + 128GiB级主机RAM起点 | 首个请求通过前不复制多实例 |
| 首次局部采集 + 完整文本流程 | 1音频卡 + 2文本TP卡，独立服务 | 经准入后独立单卡音频副本；相同候选规则，坏样本不补采 |
| 已有冻结候选的协调harness | 2文本TP卡 + CPU | 2组TP副本共4卡，可在一节点并行三case |
| 全流程并行pilot | 2音频实例 + 2组文本TP共6卡起点 | 按实测瓶颈与RAM准入扩容，不默认占用64卡 |
| 规划/编译/字面评估/网页/音乐sidecar | CPU | 不为每种baseline/角色单独加载模型 |

规划、选择、编辑可共用同一文本服务；局部采集结束可停止音频模型。index0无需模型，scores只消费外部绑定分数，没有内置质量评分器。

最终优化仅占一个节点4张A800，两组同权重/tokenizer的TP=2服务。启动前核实卡上只有holder和主机cgroup余量；未迁移在途请求、未终止其他节点训练/推理。最终全部文本服务停，节点8卡实测每卡唯一holder约975MiB；原网页继续CPU运行。

准入失败显式留证停止，不杀未知进程、缩短音频、回退mock或以holder利用率报告实验吞吐。未来调度要重新核实时状态；本节的空闲节点/显存是历史验收，不是当前可用资源承诺。

精确三首结果、短事件/缺口、失败修订与质量待审见 [VALIDATION_20261104.md](VALIDATION_20261104.md)。先独立盲评原三首并冻结协议，再谈正式数据、样本量和多方法质量—成本比较。

## 11. 修复版的资源与执行顺序

新增acceptance --artifact-type portable-checkpoint可验收实际已训练导出；基座与checkpoint类型、audit和实际加载校验必须一致，不能覆盖checkpoint任务前缀。前述基座峰值仍只是历史测量，checkpoint需重新测量。无resource/procfs/nvidia-smi等测量时报告null及原因，不记作零占用；CLI帮助不依赖真实GPU。当前修复回归使用CPU替身，没有重新占用集群或执行训练。

boundary_ids网格最多4097边界/块，控制器编译后复检最短时长。新下限不会启动额外采样、自动重试或拉长动作；不可行块使整曲失败并保留原分母。因此必须在开发集检验失败率和音乐/动作质量，不能只报通过样本的成本。

评审音频每请求校验完整有界内容后从同份bytes提供响应，临时RAM为O(完整文件字节数)，并发请求叠加；HEAD和Range也要校验完整内容。旧检测JSONL可显式迁移而不重跑Beat This或音频模型；保存原文件/行hash，不自动把未知来源认证为真实。

完整升级、候选缓存、结构输入迁移、单checkpoint首验和后续实验见 [NEXT_STEPS_20261105_zh.md](NEXT_STEPS_20261105_zh.md)。本轮仍不增加SFT/DPO/RL；下一阶段主要资源是冻结模型推理、完整歌曲准备与独立评审。
