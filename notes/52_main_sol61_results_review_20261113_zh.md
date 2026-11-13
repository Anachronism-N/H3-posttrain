# main 分支训练结果审查与正式实验衔接

审查范围：`main@d210dc6`，相对上次记录 `ce4fc96` 新增26个提交；主体审查 `ff1c23c`（2026-11-13 19:05 +08:00）的25个提交后，又补查了MOSS修复增量。最初对照 benchmark 分支 `099cbe5` 的 TX A800 归档，同步期间该分支更新到 `3c8025e`，新增E3/E4终态及音乐不确定性审计，已补充到第7节。已 fetch 两分支，未将 main 的训练改动合并到 benchmark。本文替代[上一轮执行计划](51_paper_materials_and_48h_experiment_sprint_20261113_zh.md)中“SFT仍需等待首批结果”和“E3/E4仍需继续派发”的状态判断。

结论：main 已报告训练后的5秒 captioner、checkpoint 比较和解码实验，应立即准备加入正式384条评测。新结果支持“训练模型学到了 sol61 的表达方式，解码影响跨样本文字重复”，尚不能证明其音乐编舞质量胜过六个已跑通的通用模型。两分支的试验集、生成配置和指标实现不同，不能拼接分数计算提升。

本轮复核了提交的文档和代码，并执行了两个CPU指标反例；未获取模型权重或重新运行服务器实验。main 本次增量没有提交对应逐样本生成JSONL、实际解码脚本或adapter权重，以下训练数字均标为文档报告。用户说明这些原件之后会上传HF，目前尚无确定下载路径或revision。

## 1 新增结果改变了哪些判断

| 问题 | 新增证据 | 当前结论 |
|---|---|---|
| SFT有没有可用候选 | v1交付文档声明 `sol61_full_20261112_161243/checkpoint-0001000`，音频塔末4层和projector的LoRA，r32/alpha64 | 已有待核验候选，应优先评测；不是仍停留在训练设计 |
| 重复能否低成本改善 | 同一80条val解码表中，greedy与T=0.7差别明显 | 优先冻结开发集选出的解码，不为解决开头重复再开大训练 |
| 是否优于通用模型 | 20条pilot上词汇召回较高，但通用模型有接线问题，样本配对亦待核对 | 有专用表达能力的积极迹象；正式优势尚待同384条比较 |
| 是否必须多训一轮 | 2-epoch运行在第一轮后NLL仅下降0.0168；与文字多样性并不一致 | 两天内优先已有checkpoint评测，不追加低学习率长训练 |
| harness有没有新结果 | main增量集中在局部；benchmark同期已更新E3/E4全部终态 | 见第7节：harness完成20/24、编辑联合成功20/40，当前没有证明优势 |

来源：[训练诊断](https://github.com/Anachronism-N/H3-posttrain/blob/ff1c23c/scripts_omnidance/exp/ms/planner/docs/SOL61_TRAINING_DIAGNOSIS.md)、[v1交付说明](https://github.com/Anachronism-N/H3-posttrain/blob/ff1c23c/scripts_omnidance/exp/ms/planner/docs/SOL61_PLANNER_V1_DELIVERY.md)。

最新增量 `d210dc6` 已将MOSS-Music从生成失败更新为20条结果：动作词汇召回0.151、身体部位词汇召回0.426、长度比1.16、分句密度比1.54、self-repetition 0.207、固定黑名单命中率0.100。不能继续说main上的MOSS没有结果，也不能将该黑名单10%命中解读为“10%的编舞存在身体幻觉”。MOSS-Audio仍未列入这组pilot。

新增 [MOSS runner](https://github.com/Anachronism-N/H3-posttrain/blob/d210dc6/scripts_omnidance/exp/ms/planner/eval_harness/e2_baselines.py)修复了原生prompt、单GPU加载、Whisper层兼容与解码音频重复注入，且采用SHA排序，与SFT的选样规则一致。它重写了Whisper forward并按mask长度不匹配清除音频，因此“输出有限且句子通顺”只证明能运行，不能替代与已验证原生路径的数值/音频条件等价检查；本轮不因它可运行而替换已完成的384条MOSS结果。[新增结果与修复说明](https://github.com/Anachronism-N/H3-posttrain/blob/d210dc6/scripts_omnidance/exp/ms/planner/docs/SOL61_TRAINING_DIAGNOSIS.md#L337)。

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

`d210dc6` 新增的MOSS专用runner已按SHA排序，修复了该模型新执行路径的选样规则；旧jobs runner没有修改，其他四模型历史结果的ID绑定仍需核验，不能因新增一个正确runner就视为整表对齐已获证明。

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
| 基线 | d210dc6后5个模型有表内结果，包含MOSS-Music；MOSS-Audio未列 | 六模型共2,304注册输出 |
| 提示 | SFT原生audio-only；通用模型另用短任务提示 | 固定任务合同与已跑通的模型原生适配 |
| 解码 | 通用模型greedy，默认512上限；SFT报告2048和T=0.7 | 冻结本地基线采样与1024预算，具体以各行生成合同为准 |
| BLEU | 自定义平均sentence BLEU，0–1，正则分词 | corpus SacreBLEU，0–100，13a分词 |
| ROUGE-L | 自定义，无stemming | rouge_score，有stemming |
| 硬件 | Nebula L20Z | TX A800 |

过滤掉842条gate-false与2,452条无验证音频记录后，220,667−842−2,452=217,373；这两个数据规模可分别报告原始池与训练可用子集，并非数据凭空增减。main引用的数据构建器 `data/build_sol61.py` 不在本次main树中，构造代码及清单也需随HF原件归档。

不同BLEU实现不能仅乘100后横向比较。硬件差异要求时间/吞吐分层报告，但不是禁止内容质量比较的理由；只有确认同样本、音频、参考、模型身份和评测协议后，才有共同质量比较。无需为统一显卡而重跑已成功的六模型。

main的20条报告SFT词汇召回0.877，Qwen3 0.224，新补MOSS-Music为0.151；这不是与先前Qwen3参考指标较高或MOSS-Music音乐证据分较高发生矛盾：新模型、新样本、新维度、新适配混在了一起。MOSS现已跑通，其早期接线失败不构成质量低的证据，新词汇分也不能直接推翻原384条音乐适配结果。

## 5 一两天内的更新执行顺序

### 先接SFT，保留现有基线与整曲生成

1. **HF上传后先核身份。** 固定HF repo/revision；绑定adapter SHA、base revision/hash、训练contract、processor/tokenizer、原始生成与对应ID清单、实际解码参数/seed/EOS/输出预算及执行代码SHA。run-A/B名称冲突通过原件重建，不凭表格猜。
2. **冻结一个SFT候选。** 以开发集选出的@1000、T=0.7作为待核验候选，保留audio-only输入方式，不随意套入通用模型prompt。检查12条开发样本的加载、EOS、完整输出和音频输入，随后不再根据正式测试挑参数。
3. **只新增384条SFT输出。** 直接读取既有 `test384` ID/音频清单，不从main val重新抽样；保留原参考与现有1024预算。原生采样及防复读设置明确作为方法配置记录，不能把不同解码结果包装成单因素SFT增益。
4. **使用同一评测实现。** 在新SFT行上计算既有BLEU/ROUGE/BERTScore/CLAP及文字、音乐评分；失败单列，配对统计按共同有效样本/来源计算。20条pilot只作开发分析，不混进正式表。
5. **保留原E3/E4专家与完整成绩。** 最新归档已全部终态，原MOSS-Music与共享K2候选的四方法比较不再需要继续派发。将ours替换成SFT后覆盖旧行，会同时改变专家与harness，破坏比较；有余量时单列同harness换专家的补充实验。

若要严格归因“训练带来的收益”，可选做同一基座、同输入和同解码下LoRA开启/关闭的小对照；它不是重新比较多个训练基座。当前主任务仍是专用模型对通用模型，时间不足时不把此消融加入必做清单。

### 只增加一个小的音乐依赖性验证

在12条预选开发来源上固定模型和解码，准备原音频seed42、同音频seed43、错配音频seed42、静音seed42，至多48次生成；已有完全同合同响应可复用。wrong-audio使用预先固定且音乐证据不同的配对。

同时观察文本变化、词汇/开头诊断和独立音乐适配判断。文本变了只代表敏感，不自动证明更懂音乐；变化须超过同音频采样差异且有合理音乐对应。原音频与静音仍同样高的参考召回，会提示指标更像在量表达风格。此检查针对SFT是否利用音频，区别于已完成的CLAP错配检索，不重复大规模CLAP计算。

### harness优化继续限额

根据新增终态，v2首先解决进度与预算：为每个未完成块保留一次合法提交的调用预算；预算收紧时停止可选反思并采用已有合规候选；对无效修订限次，继续保留合法性检查。逐调用检查500次未改变草稿的步骤，其中合法读取不能统称浪费。上述策略保持原总预算，不靠放宽上限改善完成率。

上一轮“突出上一最终动作、减少短块堆积、用既有修订处理显式冲突”作为后续内容改进。唯一v2仍只在3首开发歌、6–8小时内验证；不增加训练或复杂工具。若与SFT接入争资源，先交付SFT主表。开发有效后新增独立v2行，不覆盖原24首结果。

## 6 论文措辞与交付优先级

| 可写内容 | 还不能写 |
|---|---|
| sol61过滤子集、专用audio-LoRA模型与原生输入合同 | 原始池与训练子集是同一个规模 |
| 开发集上解码明显改变表面重复的描述性结果，原件核验后使用 | 解码已经解决音乐依赖或实际舞蹈多样性 |
| 单次训练/解码诊断及其与参考指标不同步 | 1000步是跨运行确定最优，或已证明7倍编舞质量效率 |
| 基于原384条补齐后的SFT对六模型正式表 | 用有接线问题的20条pilot证明全面领先 |
| 任务化的多维评价协议及其独立验证 | regex词汇召回是动作正确率，黑名单零命中是零幻觉 |

最先交付：**HF原件身份核验 → SFT同384条主表 → E3/E4真实终态表与小型指标效度验证**。训练延长、腿部加权新实验和额外模型列表后置。当前变化让模型贡献可以更快进入实证阶段，并没有消除正式对比与音乐依赖性验证的必要性。

## 7 同期benchmark新增终态 对harness结论的实质更新

审查期间远端新增 `78a08ed`、`3c8025e`。E3全部96槽位已经终态，87合法完成、2停滞、7预算耗尽；E4全部144评价槽位已有状态记录。终态包括失败/unknown，不代表全部质量可评。本节直接核对了归档JSON与[更新后的结果说明](https://github.com/Anachronism-N/H3-posttrain/blob/3c8025e/notes/49_expanded_baselines_and_agent_experiments_20261111_zh.md)。

| 整曲方法 | 合法完成/24 | 停滞 | 预算失败 | 平均外层调用/歌 | 平均输入token/歌 |
|---|---:|---:|---:|---:|---:|
| workflow | 24 | 0 | 0 | 24.54 | 397,211 |
| Self-Refine | 24 | 0 | 0 | 71.63 | 1,215,833 |
| ReAct | 19 | 2 | 3 | 74.88 | 1,457,553 |
| ours | 20 | 0 | 4 | 76.08 | 1,526,376 |

ours的调用量与输入token量分别约为workflow的3.10倍、3.84倍，完成率更低；共享K2成本另计。这是可写入论文的可靠性/成本结果，不能包装为harness已经取得优势。来源：[mechanical summary](../experiments/tx_a800_20261113/results/e34/mechanical/summary.json)。

| 单轮编辑 | 严格联合成功/40需改题 | 比率 | 需改unknown |
|---|---:|---:|---:|
| No-op | 0 | 0% | 14 |
| ReAct | 21 | 52.5% | 18 |
| ours | 20 | 50.0% | 19 |

ReAct比ours多1题，不足以确立方法优劣；原按歌曲配对区间为差值[0,0.075]。全部8个控制题（实际库存为4已满足、4歧义）不可用：6个作者失败、2个原稿失败。不能宣称已验证“无需修改时保持/歧义时合理弃权”。69/144评价有review，75unknown；不能把可评事实61/61保持解释为全部40题保护率100%。No-op原稿未改却有3个保护事实被判矛盾，是语义judge异常的直接线索。来源：[edit summary](../experiments/tx_a800_20261113/results/e34/edit_metrics/summary.json)。

### 新的质量指标并没有建立harness领先

- 音乐证据均分workflow/Self-Refine/ReAct/ours为3.8047/3.8161/3.8405/3.8391，有效歌曲24/24/19/20。不能直接按四个不同集合的均值排序。原分数上共同20歌Self-Refine减ours约0.0547，未校正95%区间[0.0022,0.1046]；它仍是易错模型评分，不是人类质量结论。
- CLAP三组与ours的共同歌曲配对区间均跨0，没有建立方法差异；不能只看不同覆盖下的四均值。
- 全部87个可评完整总谱的长程评分都是5/5。指标完全饱和，无法支持方法有效或等效，也不能因为合成控制曾通过就继续当有效主指标。
- 复读只有44个有效评审，43个基础设施unknown、9个生成失败。低复读均值不能外推到全部24首。
- 2,164个边界中16个被判冲突、1,906个无冲突、242个unknown。失败产生unknown，不代表无冲突。

应先用开发材料检验长程协议能否区分有发展/无发展、正常主题复现/机械重复、可执行/显式冲突，再冻结新版本追加评价。原表保留，新版本不按ours是否获胜挑阈值。小型独立人评用于验证这些自动评价，不能用另一个未经校准的judge互相背书。来源：[最终可靠性审核](../experiments/tx_a800_20261113/results/e34/final_integrity_summary.json)。

### uncertain离线审计使覆盖变化 不代表生成改善

`3c8025e`使用已有评分做假设性分支优先级修正，没有新增生成/评分，也未替换原官方分数：

- E2有9条numeric变unknown，共同可评375来源；音乐证据均分排序仍为MOSS-Music、MOSS-Audio、Qwen3、Music-Flamingo、Qwen2.5、Audio-Flamingo3。
- E3有226个numeric块变unknown，2001降至1775/2260；过滤后的均分上升，是条件集合变化，不是系统优化收益。
- 全部历史raw及冻结代码采用情况尚待核实，不能直接将假设值替换为新正式分数。

来源：[敏感性汇总](../experiments/tx_a800_20261113/results/music_uncertainty_audit_v1/sensitivity_summary.json)。

现阶段论文可以呈现任务/数据/基准、专用模型开发证据，以及“更复杂的agent不自动带来更好的完成率或编辑成功”的实测发现。harness作为benchmark baseline可以报告真实表现；若要主张方法优越性，需要独立有效评价和新冻结版本的证据，而不能从当前结果推出。
