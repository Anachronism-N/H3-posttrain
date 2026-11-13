# main 分支训练结果审查与正式实验衔接

审查范围：`main@ff1c23c`（2026-11-13 19:05 +08:00），相对上次记录 `ce4fc96` 新增25个提交；对照 benchmark 分支 `099cbe5` 的 TX A800 归档。已 fetch 两分支，未将 main 的训练改动合并到正在运行的 benchmark。本文替代[上一轮执行计划](51_paper_materials_and_48h_experiment_sprint_20261113_zh.md)中“SFT仍需等待首批结果”的状态判断。

结论：main 已报告训练后的5秒 captioner、checkpoint 比较和解码实验，应立即准备加入正式384条评测。新结果支持“训练模型学到了 sol61 的表达方式，解码影响跨样本文字重复”，尚不能证明其音乐编舞质量胜过六个已跑通的通用模型。两分支的试验集、生成配置和指标实现不同，不能拼接分数计算提升。

本轮复核了提交的文档和代码，并执行了两个CPU指标反例；未获取模型权重或重新运行服务器实验。main 本次增量没有提交对应逐样本生成JSONL、实际解码脚本或adapter权重，以下训练数字均标为文档报告。用户说明这些原件之后会上传HF，目前尚无确定下载路径或revision。

## 1 新增结果改变了哪些判断

| 问题 | 新增证据 | 当前结论 |
|---|---|---|
| SFT有没有可用候选 | v1交付文档声明 `sol61_full_20261112_161243/checkpoint-0001000`，音频塔末4层和projector的LoRA，r32/alpha64 | 已有待核验候选，应优先评测；不是仍停留在训练设计 |
| 重复能否低成本改善 | 同一80条val解码表中，greedy与T=0.7差别明显 | 优先冻结开发集选出的解码，不为解决开头重复再开大训练 |
| 是否优于通用模型 | 20条pilot上词汇召回较高，但通用模型有接线问题，样本配对亦待核对 | 有专用表达能力的积极迹象；正式优势尚待同384条比较 |
| 是否必须多训一轮 | 2-epoch运行在第一轮后NLL仅下降0.0168；与文字多样性并不一致 | 两天内优先已有checkpoint评测，不追加低学习率长训练 |
| harness有没有新结果 | 本次main增量集中在训练、局部评测和文档 | 没有新增完整E3/E4质量表，不能用这批局部结果替代整曲结论 |

来源：[训练诊断](https://github.com/Anachronism-N/H3-posttrain/blob/ff1c23c/scripts_omnidance/exp/ms/planner/docs/SOL61_TRAINING_DIAGNOSIS.md)、[v1交付说明](https://github.com/Anachronism-N/H3-posttrain/blob/ff1c23c/scripts_omnidance/exp/ms/planner/docs/SOL61_PLANNER_V1_DELIVERY.md)。

## 2 值得保留的解码结果

下表来自诊断§5.6，为80条固定val的文档报告。词汇召回不是动作正确率，开头多样性不是实际编舞多样性。checkpoint绑定仍需按下一节核对。

| 解码 | 参考动作词汇召回 | 身体部位词汇召回 | 开头种类 | 开头熵/bit | 最常见开头占比 | 字符长度比 |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 0.803 | 0.904 | 16 | 2.81 | 0.413 | 0.84 |
| T=0.7 | 0.871 | 0.945 | 54 | 4.80 | 0.100 | 0.93 |
| T=0.9 | 0.841 | 0.940 | 64 | 5.36 | 0.075 | 0.95 |

采样配置另含top_p=0.9、repetition_penalty=1.05、no_repeat_ngram_size=4、num_beams=1。保守表述是“在这组开发样本上，采样明显改善了表面文字重复，并提高了参考词汇覆盖”。不要写成“仅靠解码已解决音乐理解/舞蹈多样性”，也不要因某个指标更高宣称全面优胜。单次采样尚不能估计随机种子方差。

## 3 必须先核对的结果与代码问题

### P1 checkpoint名称与分数绑定有冲突

[诊断§5.2](https://github.com/Anachronism-N/H3-posttrain/blob/ff1c23c/scripts_omnidance/exp/ms/planner/docs/SOL61_TRAINING_DIAGNOSIS.md#L143)把 `161243/@1000` 指为run-B，记录action/body/entropy=0.837/0.913/3.04；§5.6又将同一路径的greedy记为0.803/0.904/2.81，接近§5.2的run-A。§5.5中 `161243/@1527` 的0.822/0.894/2.88也对应前表run-A，而非run-B的0.836/0.917/2.90。

另外，“1000之后body recall从0.913降至0.853”取的是run-B@1000和run-A@1500，不能作为同一训练轨迹退化的证据。需以adapter SHA、base revision、run_config、eval ID清单及raw生成重建表格；当前不能断言是哪一行标错，也不能据此否定全部实测。

### P1 20条样本对齐声明不能由当前脚本复现

文档称基线使用 `sha256('many:'+id)` 排序后的前20条，和80条SFT的前缀一致。但[baseline runner](https://github.com/Anachronism-N/H3-posttrain/blob/ff1c23c/scripts_omnidance/exp/ms/planner/jobs/e2-baselines-job/e2_baselines.py#L124)按原始ID排序；[SFT evaluator](https://github.com/Anachronism-N/H3-posttrain/blob/ff1c23c/scripts_omnidance/exp/ms/planner/eval/eval_many.py#L72)按SHA排序。服务器可能使用了未提交的版本，因此结论是“待raw ID核验”，不是声称历史20条必然不同。

### P1 词汇召回不能叫动作正确性，文档指标与实现也不一致

[audit_sol61.py](https://github.com/Anachronism-N/H3-posttrain/blob/ff1c23c/scripts_omnidance/exp/ms/planner/eval/audit_sol61.py#L164)将动作、方向、身体词作为集合求召回，不绑定左右、动作对象和时间关系，也不惩罚额外动作。本轮直接运行该版本函数得到：

| 参考 | 候选 | action recall | body recall | hallucinated flag |
|---|---|---:|---:|---|
| The dancer raises the left arm and lowers the right arm. | The dancer lowers the left arm and raises the right arm. | 1.0 | 1.0 | false |
| The dancer raises both arms. | The dancer raises both arms and kicks both legs. | 1.0 | 1.0 | false |

第一个例子证明关系颠倒不可见；第二个说明额外动作不影响召回。开放编舞中额外腿动作未必错误，因此也不能反过来将“参考未提及”直接定义为幻觉。

| 字段 | 选择协议描述 | 当前实现 |
|---|---|---|
| density | 每秒独特动作标签 | 每100字符的分句数量 |
| distinct openings | 前60字符 | 首句前4个空格分词；entropy另用前3个词 |
| self repetition | 1−独特6gram/总数 | 属于重复8gram的所有出现次数占比，短文本另缩窗口 |
| hallucinated limbs | 参考未出现的部位 | face/eyes/tail等固定正则黑名单 |
| 长度淘汰 | 字符长度比越界 | 实际按density ratio淘汰 |

因此可将现有值保留为辅助诊断，并按实际实现命名；不能写“hallucination=0说明没有编造身体动作”。无需在两天内重造语义抽取器。协议/代码来源：[选择协议](https://github.com/Anachronism-N/H3-posttrain/blob/ff1c23c/scripts_omnidance/exp/ms/planner/docs/M2D_PLANNER_SELECTION_PROTOCOL_SOL61.md)、[实际实现](https://github.com/Anachronism-N/H3-posttrain/blob/ff1c23c/scripts_omnidance/exp/ms/planner/eval/audit_sol61.py)。

### P1 当前main evaluator不能直接复现交付方案

[eval_many.py](https://github.com/Anachronism-N/H3-posttrain/blob/ff1c23c/scripts_omnidance/exp/ms/planner/eval/eval_many.py#L11)硬编码另一工作树模块路径，且传递 `device_map`，但main的 [load_planner_adapter](https://github.com/Anachronism-N/H3-posttrain/blob/ff1c23c/scripts_omnidance/exp/ms/music_planner_adapters.py#L559)已删除该参数。若真正加载当前main实现，会出现不接受该参数的问题；若旧路径生效，则实际执行代码不是本次main。

同一evaluator固定 `do_sample=False`，无temperature CLI，无法直接复现T=0.7。应接入一个明确固定版本的SFT推理入口、保存实际导入文件路径与SHA，不把“路径能找到旧代码”视为版本一致。main基线异常还会写成空文本，被audit混入质量统计；需要把生成失败与可评文本分开。

### P2 训练因果结论需收窄

- `1e-4`的200步NLL约等于`1e-5`的1527步，是到达相近NLL所需更新数的观察；warmup同时20→100，不是单因素证明，也不是7倍真实编舞质量或实测GPU时长收益。
- 一轮/两轮作业使用cosine schedule，总步数不同会改变第一轮内的学习率轨迹：以提交中的函数计算，step1527的一轮/两轮倍率分别约0与0.50466；32/64 GPU与accum同时变化，不能只归因于accum=2。
- NLL未升不能证明没有任何生成退化；NLL与词汇/开头指标排序不同也不能推出NLL总是“反着排真实质量”。
- 1.3统一assistant token权重使相同logits的报告loss线性缩放；不能直接把2.0与未加权1.5解释为质量下降。应另报未加权NLL。
- 新增腿部token加权已从collate传入loss，但 `PLANNER_LEG_WEIGHT` 的值未写入run_config/fingerprint，validation也使用它；导出时应补实际值并另报未加权NLL。尚无完成的生成对照，不要把“多写knee/leg”自动解释为更好的下肢编舞。

## 4 为什么与benchmark分支看起来不同

| 维度 | main试跑 | benchmark正式E2归档 |
|---|---|---|
| 样本 | 20/80条val，曾用于选checkpoint/解码 | 冻结384条test |
| 数据池 | 过滤后217,373条：195,367/10,938/11,068 | 原sol61盘点220,667条：198,315/11,098/11,254 |
| 基线 | 4个模型有表内结果；MOSS-Music失败，MOSS-Audio未列 | 六模型共2,304注册输出 |
| 提示 | SFT原生audio-only；通用模型另用短任务提示 | 固定任务合同与已跑通的模型原生适配 |
| 解码 | 通用模型greedy，默认512上限；SFT报告2048和T=0.7 | 冻结本地基线采样与1024预算，具体以各行生成合同为准 |
| BLEU | 自定义平均sentence BLEU，0–1，正则分词 | corpus SacreBLEU，0–100，13a分词 |
| ROUGE-L | 自定义，无stemming | rouge_score，有stemming |
| 硬件 | Nebula L20Z | TX A800 |

过滤掉842条gate-false与2,452条无验证音频记录后，220,667−842−2,452=217,373；这两个数据规模可分别报告原始池与训练可用子集，并非数据凭空增减。main引用的数据构建器 `data/build_sol61.py` 不在本次main树中，构造代码及清单也需随HF原件归档。

不同BLEU实现不能仅乘100后横向比较。硬件差异要求时间/吞吐分层报告，但不是禁止内容质量比较的理由；只有确认同样本、音频、参考、模型身份和评测协议后，才有共同质量比较。无需为统一显卡而重跑已成功的六模型。

main的20条报告SFT词汇召回0.877，Qwen3 0.224；这不是与先前Qwen3参考指标较高或MOSS-Music音乐证据分较高发生矛盾：新模型、新样本、新维度、新适配混在了一起。MOSS接线失败更不构成其质量低的证据。

## 5 一两天内的更新执行顺序

### 先接SFT，保留现有基线与整曲生成

1. **HF上传后先核身份。** 固定HF repo/revision；绑定adapter SHA、base revision/hash、训练contract、processor/tokenizer、原始生成与对应ID清单、实际解码参数/seed/EOS/输出预算及执行代码SHA。run-A/B名称冲突通过原件重建，不凭表格猜。
2. **冻结一个SFT候选。** 以开发集选出的@1000、T=0.7作为待核验候选，保留audio-only输入方式，不随意套入通用模型prompt。检查12条开发样本的加载、EOS、完整输出和音频输入，随后不再根据正式测试挑参数。
3. **只新增384条SFT输出。** 直接读取既有 `test384` ID/音频清单，不从main val重新抽样；保留原参考与现有1024预算。原生采样及防复读设置明确作为方法配置记录，不能把不同解码结果包装成单因素SFT增益。
4. **使用同一评测实现。** 在新SFT行上计算既有BLEU/ROUGE/BERTScore/CLAP及文字、音乐评分；失败单列，配对统计按共同有效样本/来源计算。20条pilot只作开发分析，不混进正式表。
5. **不替换当前E3/E4专家。** 现有MOSS-Music与共享K2候选的四方法对比继续；临时将ours换成SFT会同时改变专家与harness，破坏比较。核心完成后才可单列少量同harness换专家的补充实验。

若要严格归因“训练带来的收益”，可选做同一基座、同输入和同解码下LoRA开启/关闭的小对照；它不是重新比较多个训练基座。当前主任务仍是专用模型对通用模型，时间不足时不把此消融加入必做清单。

### 只增加一个小的音乐依赖性验证

在12条预选开发来源上固定模型和解码，准备原音频seed42、同音频seed43、错配音频seed42、静音seed42，至多48次生成；已有完全同合同响应可复用。wrong-audio使用预先固定且音乐证据不同的配对。

同时观察文本变化、词汇/开头诊断和独立音乐适配判断。文本变了只代表敏感，不自动证明更懂音乐；变化须超过同音频采样差异且有合理音乐对应。原音频与静音仍同样高的参考召回，会提示指标更像在量表达风格。此检查针对SFT是否利用音频，区别于已完成的CLAP错配检索，不重复大规模CLAP计算。

### harness优化继续限额

沿用上一轮唯一v2：突出上一最终动作、减少短块堆积、用既有修订预算处理显式衔接冲突。仅3首开发歌，最多6–8小时；不增加训练或复杂工具，不抢正式比较队列。若SFT正式评测接入与该优化争资源，先交付SFT主表。

## 6 论文措辞与交付优先级

| 可写内容 | 还不能写 |
|---|---|
| sol61过滤子集、专用audio-LoRA模型与原生输入合同 | 原始池与训练子集是同一个规模 |
| 开发集上解码明显改变表面重复的描述性结果，原件核验后使用 | 解码已经解决音乐依赖或实际舞蹈多样性 |
| 单次训练/解码诊断及其与参考指标不同步 | 1000步是跨运行确定最优，或已证明7倍编舞质量效率 |
| 基于原384条补齐后的SFT对六模型正式表 | 用有接线问题的20条pilot证明全面领先 |
| 任务化的多维评价协议及其独立验证 | regex词汇召回是动作正确率，黑名单零命中是零幻觉 |

最先交付：**HF原件身份核验 → SFT同384条主表 → E3/E4质量表与小型指标验证**。训练延长、腿部加权新实验和额外模型列表后置。当前变化让模型贡献可以更快进入实证阶段，并没有消除正式对比与音乐依赖性验证的必要性。
