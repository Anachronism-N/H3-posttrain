# MOSS「离散舞种＋时间戳动作」复核与优化

日期：2026-10-28。分支：`codex/planner-moss-captioner-64gpu`。复核起点：`416bab86e375abe0944c94622fd43b6dabbb6564`。本次优先完善 MOSS；没有启动远程训练或占用 A800。

## 本次修复了什么

1. **真实恢复漏洞**：旧恢复函数验证 marker 中记录的原绝对路径，却从当前 checkpoint 目录读取 scheduler／optimizer／RNG。复制 checkpoint 后，修改复制件 scheduler，原件仍完好，校验可能通过。现在从唯一 portable `trainable.pt` 引用推导原目录，再把成员映射到当前目录；验证实际读取的字节，支持复制及改名，拒绝清单漏项、重复引用、`..` 和符号链接逃逸。校验失败先广播给全部 rank，避免其余 rank 进入后续恢复 barrier。
2. **舞种决策与动作生成接线**：上一版 JEV 只在旁路分类，普通生成不读取分类结果。新增 v2 协议及 MOSS 推理入口，实际先选合法舞种，再把该舞种的 assistant header token 放进模型输入，续写动作。
3. **训练／推理 token 边界**：对完整字符串直接做 BPE 编码时，header 末尾与动作开头可能合并成一个 token。v2 明确使用 `encode(header) + encode(actions) + im_end`，保存每个合法类别的 header token IDs，推理预填完全相同的 IDs；不能把旧协议 checkpoint 改个名字当作 v2 使用。
4. **文本与标注一致性**：新解析器核验舞种白名单、时间戳语法、严格正序／不重叠、源时长、可选完整覆盖，逐项比较文本动作与 `target_windows` 的动作正文和边界。已知独立分类标签必须与文本舞种一致；v2 不把软分布默认为某个硬类别。

这是对输入、标签、梯度与输出接线的改进，不表示动作语义或音乐卡点已通过真实模型评测。

## v1 与 v2

历史 `music-planner-caption-contract-v1` 保持原协议，JEV 仍可作为并行辅助头。新的 `music-planner-caption-contract-v2` 在旧合同字段之外增加 `dance_caption_format`，显式启用「有限类别 header＋时间戳动作」协议。

正式协议按用户提供的拼接规则统一为英文。使用 [v2 合同](../../examples/planner_training/caption_contract_moss_v2.example.json) 和 [23值词表](../../examples/planner_training/dance_styles_23.json)：

```text
Dance style: contemporary
[0.000-1.960s] The dancer steps to the side.
[1.960-5.062s] The dancer raises the left arm.
```

以上是格式示意，不是已审核的动作样本。正式 `dance_caption_format` 使用 `music-planner-dance-caption-v2`：首行严格为 `Dance style: <合法值>`，后续每个物理行一个窗口；起止秒值必须恰好三位小数，ASCII 连字符、右端小写 `s`。动作正文为英文。程序检查结构，不声称能仅靠字符规则证明语言或动作语义正确。视觉排版软换行不改变数据；文件中若动作真的被拆成多行，生产检查会拒绝，不能静默合并后当作原始标注。

方括号保留给时间标记。正式 profile 拒绝常见帧号引用，如 `by frame 24`，以及明确的机器 metadata 字段；不会自动删帧号或改写动作来冒充RGB修订。草稿实验可显式把 `forbid_frame_references` 设为false，并使用独立caption版本／运行指纹。历史 `dance-caption-v1` 的宽松格式保留兼容，但不作为本次正式英文协议。

`time_tolerance_seconds` 仅用于输出小数舍入、覆盖和 metadata 边界比较，不容许负时间、重叠或空区间，也不会把动作吸附到节拍或改写秒值。例如 5.0625 秒源可以按0.001秒容差接受5.063秒末端，返回值仍保留5.063。

窗口由小节划分，数量由实际数据决定，不写死为两段。`[start-end]` 表示动作发生在该窗口内，不表示动作恰好在start开始、end结束，也不代表整窗保持同一姿态。训练指令明确了这一语义，没有把小节边界当成动作onset／offset监督。

`require_full_coverage=true` 要求从0开始、相邻窗连续并覆盖到源末端；false允许真实间隙，但仍禁止重叠／越界。`target_windows` 若存在，其每条 `caption` 必须与正文对应动作一致。程序不自动补动作来填空窗。

## 数据与监督

保留既有完整源音频 manifest 的 ID、来源组、音频路径／SHA、duration、split、caption版本和审核声明。`target_text` 放最终连续文本，`target_windows` 放同一文本的内部时间窗元数据。完整音乐不因文本分 chunk 而裁成多个输入样本。

正式词表已经按用户提供的顺序写入 `dance_styles_23.json`，并与现有 `vlm_label.py` 的22个 `GENRES` 做AST回归核对。最后追加 `unknown`；`freestyle / other` 是真实类别，不能代替无法判别的unknown。独立舞种 labels 沿用前一版的 ID／来源组／音频SHA／split／label_source／label_version 绑定：

- Caption 中必须有白名单内的一个舞种。
- 有独立标签时，标签必须与 Caption header 是同一离散类别，冲突直接拒绝。
- 缺独立标签时屏蔽分类 loss，不从动作正文猜标签，也不把缺失记录改写成视觉unknown；开启分支的训练集至少需要一个真实标签。
- v1 继续支持软分布；v2 已知标签必须对应单一类别。

正式合同额外声明 `genre_supervision`：**Caption/header 有23个合法值，JEV语义分类头只有22个真实舞种**。视觉 `unknown` 是标注弃权，不直接教成“音乐的第23类”。这类样本的分类 loss 被屏蔽，unknown header 的语言模型 loss 也被屏蔽，但header仍作动作上下文，可靠动作正文和EOS继续训练。原生音乐prefix长度在拼接header前显式记录，不能因header loss=-100而把教师header误当输入侧特征。

数据读取会分别记录缺少独立标签与被屏蔽的视觉unknown，并输出这批train／val manifest的23值数量和22类有效监督数量。若只有unknown，不能声称已训练语义head。该统计不代表尚未扫描的全量数据分布。

用户说明现有来源主要是 `runtime/historical_dance_style_reuse_v1.jsonl` 的历史235B VLM `genre`，尚未经人工审核。训练可以作为weak-label实验，但不能升级成reviewed。`style_metadata_not_in_target`、来源、置信度、RGB证据、审核状态均留在metadata；只有 `target_text` 的舞种行与动作行进入语言监督。新舞种标注应仅依据原始RGB中的动作，至少两帧证据；现有历史标签是否满足该要求，本次未作独立视觉验收。

两阶段修订的实际运行端尚未完成、没有修订后真实样例，本次没有假装执行该流程。严格文本校验也不能证明手部在两帧以上清楚可见；仍须真实RGB／手部ROI修订、保留可靠粗动作，无法确认的左右或保持时长用中性措辞。用户提供的contemporary原始草稿仅用于格式roundtrip测试，不表示认可其手部细节。

这轮没有给时间戳另加回归 loss。训练仍是Caption token CE加有标签源的舞种CE；正常类别的header、时间戳、动作和EOS参与语言监督，视觉unknown按上述规则屏蔽header。JEV取输入prefix最后一个token的最终decoder表示，因果attention阻止它读取后面的真值Caption。`--genre-weight` 默认仍为0，联合训练必须显式设置正权重。

一个optimizer update的目标是：全部有效Caption token的NLL总和／全局有效token数，加上 `genre_weight ×` 已标注真实舞种样本的CE总和／全局有效舞种样本数。分母跨rank和梯度累积窗口计算，缺标签、视觉unknown和尾部占位不计入舞种分母；不是把各卡不等长文本的平均loss再简单平均。示例舞种权重为0.1，只是初始实验值。这是有监督的联合训练，不是DPO或RL。

## 如何训练 MOSS

先用数据检查验证格式、类别和独立标签：

```bash
python -m scripts_omnidance.exp.ms.music_planner_train \
  --family moss --model /models/MOSS-Music-8B-Instruct \
  --train /shared/data/planner/train.jsonl --val /shared/data/planner/val.jsonl \
  --out /shared/runs/moss-style-timed-v2-smoke \
  --caption-contract /shared/data/planner/caption_contract_v2.json \
  --genre-classes /shared/data/planner/classes.json \
  --train-genre-labels /shared/data/planner/train-genres.jsonl \
  --val-genre-labels /shared/data/planner/val-genres.jsonl \
  --genre-weight 0.1 --check-data-only
```

将正式 `dance_styles_23.json` 作为上述 `classes.json` 的内容，三位小数英文合同作为 `caption_contract_v2.json`。标签和Caption版本应对应同一数据快照；初始化前会再次检查读入的数据合同／词表与实际adapter一致，模型加载期间若文件被改写则提前拒绝。

8节点、每节点8张A800可用 [MOSS 单模型64卡配置示例](../../examples/planner_training/moss_single64_v2.example.json)，替换真实 hostname 和路径。micro=1、accum=1、global batch64。也可以沿用32卡配置保持原global batch32；两者的update数量不同，比较应报告源曝光与epoch预算。

多机 plan／preflight／verify／commands 流程见[前一版训练指南](MOSS_QWEN_64GPU_JEV_20261027.md)。先做2步 smoke，再新进程 resume 到后续 update，核验全部 rank 的真实梯度、舞种头确实遇到标签、全验证诊断和 optimizer／scheduler／RNG 恢复。语义验收与正式训练另行执行。

代码、合同与 token 协议进入指纹。旧实验不能直接换新源码做严格 resume；保留旧快照继续旧实验，v2应建立独立新训练。当前没有通用旧 checkpoint warm start 工具，也没有给未按 v2 训练的模型自动赋予受舞种控制能力。

## 如何生成

新增入口 `music_planner_moss_generate`：

```bash
python -m scripts_omnidance.exp.ms.music_planner_moss_generate generate \
  --model /models/MOSS-Music-8B-Instruct \
  --checkpoint /shared/runs/moss-style-timed-v2/checkpoint-0000100 \
  --row /shared/inference/audio.json --output /shared/inference/result.json \
  --device cuda:0 --max-new-tokens 1024
```

`audio.json` 只需要 `id`、`source_group_id`、`audio_path`、`audio_sha256`、`audio_duration_seconds`、`audio_window_seconds`。接口只读取这些字段，即使传入含真值Caption的完整样本，也不消费目标或舞种标签。

默认执行：

```text
音乐 → 输入侧 JEV 前向 → 合法舞种 argmax
     → 原生音乐 prefix＋该舞种的assistant header
     → 原生 MOSS 续写时间戳动作 → 严格格式验收
```

默认路线要求checkpoint有真实 `jev_head_exercised` 记录，在22个真实类别中取argmax；概率向量明确伴随 `probability_classes`，不能错按23值解释。加 `--selected-genre` 可以指定23值中的一个，包括 `unknown`；结果明确记作人工指定，不算分类头自己预测正确。

可选 `--min-style-confidence` 在用户明确设置时启用弃权：22类最大概率低于阈值则使用unknown header，单独记录阈值、原因与未校准状态；默认不编造已校准阈值。音乐规划器弃权与训练数据的视觉证据不足是不同原因，均在metadata记录，正文仍只写合法header和动作。当前默认路线有两次独立prefill，没有共享KV缓存。

加载顺序为：验证完整 checkpoint／基础模型身份 → 原生portable adapter → 恢复合同和head → 原生MOSS position／cache／EOS兼容补丁 → 预测和生成。多机训练记录过内容库存SHA时，推理加载重新核对真实基础模型内容；历史未记录内容身份的情况只报告config／文件大小核验，不冒称权重内容已验证。

最终自然文本在结果的 `caption` 字段；另保留完整原始序列、生成 token、special token、强制 header IDs、终止原因、真实forward计数和校验失败信息。首个EOS后还有token、预算截断、空动作、类别重复、时间不合法等情况，`caption=null`，不会清洗或改写后伪装成功。`ConfiguredPlannerAdapter.generate()` 在v2下也路由到这条受约束路径，不会悄悄退回自由生成。

## 仍需真实实验确认

类别白名单、输入条件和格式合法不等于动作语义遵循。必须用同一音乐分别指定不同合法舞种，盲评动作是否随之改变；再评估时间覆盖、动作丰富度和音乐节奏关系。概率未校准，不把softmax当作可信质量分。

CPU测试验证解析、因果前缀、token协议、条件实际送入模型、恢复和原始输出保留。当前没有运行真实MOSS权重／PEFT／A800／NCCL训练，也没有证明卡点或编舞质量提升。本次不修改H3渲染器。

规划器训练与推理已复用同一英文格式解析器。渲染器数据准备应复用 `music_planner_caption_format.parse_caption`，原样使用同一 `caption` 字符串；不要再转成旧中文／两位小数示例。本次没有接管H3训练入口，不能宣称渲染器训练数据已完成这种验收。

## 本地验证记录

2026-10-28，Python 3.9／CPU PyTorch环境分两组执行同一工作树的7个相关测试文件：

```text
test_music_planner_train.py
test_music_planner_scaling.py
test_music_planner_cluster.py
test_music_planner_jev.py
  134 passed, 9 skipped

test_music_planner_schema.py
test_music_planner_caption_format.py
test_music_planner_moss_generate.py
  208 passed

合计：342 passed, 9 skipped
```

7项跳过源于未安装DeepSpeed，1项源于本机CPU PyTorch没有可用Gloo transport，1项源于Windows账号不能创建文件符号链接。测试包含真实小型head张量导出／恢复、23值header与22类head对应关系、unknown屏蔽后prefix不泄漏、token边界与原生生成调用的受控fixture，以及复制／改名checkpoint后修改实际成员文件被拒绝。原生MOSS生成、PEFT和ZeRO相关调用部分使用测试替身；这些计数仅涵盖列出的测试文件，不是全仓测试或64卡验收。

新增生成入口的CLI帮助和修改模块的Python编译检查也在本地通过。下一步服务器验收仍按先smoke、再新进程resume、最后语义评测的顺序进行。
