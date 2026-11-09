# 音乐→动作 Caption：实验复盘、六点分析与下一阶段计划

日期：2026-11-09。代码基线：`02f478a8b3a341f309e82a09ed55b8a823fd1391`。

## 0. 结论与边界

- 主目标保持为**学习音乐到动作 caption 的条件映射**，不是把闭集动作词改成开放词汇，也不是只提高英语流畅度。
- 动作词汇、方向、身体部位及句型复用正常；不以 n-gram 高、词汇少、caption 长或短本身判错。真实重复动作必须保留。
- 下一版舞种走 **JEV 分类头**，Captioner 输出动作正文；本轮实际没有启用 JEV。不能把生成 unknown 的失败等同于 JEV 分类失败。
- 本轮全量标准 SFT 与 EOS20 对照均执行完成，但音乐→动作映射质量未通过；所有原始标注、失败输出、断点及日志保留。
- 本文只做分析、CPU参数/原生特征测算与下一步规划，**没有启动新的训练、修改 caption、裁剪实际音频或替换共享依赖**。只同步文档，不混入其他工作区未提交代码。
- 固定生成评审仍为同三例，各方法使用相同设置的初始化对照；本文不把三例结果推成全验证集统计。

## 1. 本轮实际执行情况

基座为官方 **Qwen/Qwen3-Omni-30B-A3B-Captioner**，固定官方 revision `a2bd106cbf527db5676e79662674da22b0545ec0`；只执行 Thinker 的音频→文本路径，无 Talker、视频生成或3D渲染。数据来自指定 HF 标注目录；未下载视频或重新生成 caption。

训练193,278条；全局验证10,342条，测试10,114条；训练期间使用256条固定监控验证子集。220,631条候选完成媒体核验，213,735条通过；原标注中36条潜在非动作文本与1条训练版本不兼容单独隔离，不静默修改。原视频来源与精确PCM联合分组，样本ID、来源组、精确音频和路径跨split零交叉；歌曲ID及非精确音乐重复仍未完整验证。

| 实验 | 训练来源 | 预算 | 状态 |
| --- | ---: | ---: | --- |
| 首阶段标准SFT | 76,784 | 2,400更新、1 epoch | 完成，后续发现部分来源分组交叉风险；仅历史诊断 |
| 首阶段学习率×2 | 76,784 | 2,400更新、1 epoch | 完成，较低NLL不等于更好生成 |
| 全量标准SFT | 193,278 | 6,040更新、1 epoch | 官方基座重新初始化，完成 |
| 全量EOS权重20 | 同上 | 同上 | 官方基座重新初始化，完成 |

全量两组各8张A800-80GB，BF16基座/计算，FP32可训参数，ZeRO-3，microbatch1，accum4，global batch32，LoRA rank32/alpha64，语言LoRA学习率1e-5，投影5e-5，cosine，warmup20。音频最后4层LoRA、投影可训，融合专家LoRA关闭。各最终checkpoint45个成员SHA核查通过，优化器/调度器/各rank RNG/portable adapter完整；备份和自有holder恢复核验通过。单次最终训练本身未做resume；保存/新进程恢复已由先前独立冒烟验证，二者不混称。

### 最终同三例的技术检查（不是音乐/动作质量验收）

| 解码 | 标准SFT EOS / 完整时间覆盖 | EOS20 EOS / 完整时间覆盖 |
| --- | --- | --- |
| 原始greedy、显式EOS | 2/3、2/3 | 0/3、0/3 |
| repetition penalty1.05、禁重复4gram | 3/3、0/3 | 3/3、0/3 |
| 原生采样：T0.6、top-k20、top-p0.95、seed42 | 3/3、3/3 | 3/3、3/3 |

EOS20没有证明最终greedy收益。采样可以正常结束但具体动作/顺序与音乐对应仍未通过。禁重复4gram会误伤合法闭集重复，并带来措辞变化，不推荐作为修复主线。

## 2. 哪些内容重复：词汇正常复用与错误循环必须分开

### 2.1 正常、不得删除的闭集动作描述

下面的词和句型都不是错误：

- `wrist moves left/right/up/down relative to shoulder`；
- `elbow angle decreases/increases`；
- `knee angle decreases/increases`；
- `one side / the other side`；
- 相同动作在不同时间窗、重复舞步或周期中的多次出现。

语义闭集并不等于逐字重复必须减少。评价目标是**部位、方向、侧别、先后/并发、次数与音乐条件的对应**，而不是distinct-n或句式多样性。

实际训练清单中99.70%同时含one side和other side；这只是表达规范的描述，不是质量否决。unknown舞种57.03%，contemporary36.37%，此项用于分类标签覆盖诊断，不要求captioner通过文字生成解决。

### 2.2 确认存在的病理循环例子

固定case C的最终标准greedy输出，只有一个 `[0.000-2.720s]` 窗口，随后以下12子句块**完整重复19次**，再输出一个不完整尾块：

1. One side wrist moves left relative to shoulder.
2. The other side wrist moves right relative to shoulder.
3. One side wrist moves up relative to shoulder.
4. The other side wrist moves up relative to shoulder.
5. One side elbow angle decreases.
6. The other side elbow angle decreases.
7. One side wrist moves right relative to shoulder.
8. The other side wrist moves left relative to shoulder.
9. One side wrist moves down relative to shoulder.
10. The other side wrist moves down relative to shoulder.
11. One side elbow angle increases.
12. The other side elbow angle increases.

输出达到2,048-token上限，末尾断在句中；没有EOS，也没有进入后半观察范围。按相同标点切分口径，输出239段、参考53段；前10个子句各出现20次，后2个各19次。**问题是无依据地循环次数膨胀、无法推进或停止，而不是left/right等动作词重复。** 参考也有重复，但并没有这段19周期的输出。

EOS20三个case的greedy都落入相似的双侧腕左右/上下、肘弯伸循环，到token上限。两个不同音乐的采样结果也共享一段开头；仅此现象不证明映射无效，因为音乐可能对应相同动作，必须结合换音乐对照判断。

禁重复解码产生过 `elbow angle doubles`、`elbow angle halves` 等参考未支持的幅度词。零字面重复并不表示动作更准确。后续不采用“重复率越低越好”的验收标准。

## 3. 当前音乐→caption方法和loss是什么

### 3.1 结构与输入输出

原生音频经16kHz log-mel、音频tower、投影进入语言decoder；学习条件自回归概率：

`p_theta(y | a) = product_t p_theta(y_t | a, y_<t)`。

本轮audio-only前缀不含目标caption或目标舞种。Captioner分支没有加入合同task_instruction的文本：配置存在不代表实际送入模型；这是原生Captioner协议选择，不是已经确认的错误。见：

- `scripts_omnidance/exp/ms/music_planner_adapters.py:223`；
- `scripts_omnidance/exp/ms/music_planner_schema.py:302`。

assistant-only mask屏蔽输入音频前缀，目标拼接native im_end；caption所有动作、时间戳和舞种行均在当前文本CE中。

### 3.2 标准SFT：条件因果token CE

令有效assistant token集合为S，

`L_caption = -sum_{(i,t) in S} log p_theta(y_it | a_i, y_i,<t) / |S|`。

分母为一次global optimizer update内所有有效assistant token；不是每样本各自平均再平均，也不是音乐contrastive loss。global batch32经8 ranks/4次accum正确合并支持数。因果shift与assistant CE在 `music_planner_token_loss.py:34`；联合目标在同文件71行。EOS20是本次本地独立配方扩展，不是main原始代码已有开关。

EOS20实际优化：

`L_EOS20 = (sum CE_all + 19 * sum CE_EOS) / N_assistant`。

日志和验证NLL保持普通未加权口径。本轮JEV/窗口count权重不参与，没有独立动作token加权，没有显式音乐/文本匹配、错配负例或节拍对齐loss。**仅SFT本来就能学映射，但在多token文字骨架可预测时，没有额外机制要求模型区分相近音乐。**

### 3.3 为什么loss下降不一定等于映射变好

- Teacher forcing给模型此前正确动作词，下一词可能主要依赖闭集序列规律。
- 同音乐可以有多个编舞。闭集内不同事件/侧别/次数仍需学习，不应把这个理由当放弃映射。
- 每个token同权的CE让“relative to shoulder”等高频骨架也贡献loss；动作信息词变化可能被整体平均掩盖。
- 舞种行和数字边界都是额外任务；可以拆走，而不改动作事实。

标准SFT100到6040步三例token加权NLL：

| step | 真音乐 | 错配音乐 | 静音 |
| --- | ---: | ---: | ---: |
| 100 | 0.372157 | 0.378042 | 0.554583 |
| 6040 | 0.330727 | 0.333369 | 0.396873 |

最终2/3例错配音乐NLL略低。此诊断支持“对音频有响应，但具体音乐辨别尚无充分证据”，不证明模型完全忽略音频。三例、随机错配、teacher forcing不是总体匹配准确率，也不是踩拍真值。

### 3.4 怎么更直接学习与验证映射

**先确认测量，再改变目标。**

1. 用冻结的现有256验证来源做正确/随机错配/相近节奏风格音乐打分；视频/完整生成仍固定三例。不得把可能同歌曲、同PCM或本就有相同合理caption的音乐无脑当负例。
2. 比较动作token局部CE：部位、方向、侧别、动作变化、必要次数；不只比较骨架、舞种和数字。闭集parser只用于诊断，无法明确解析的字段记unknown，不能生成新事实。
3. 原音频、错配音频、静音固定同一y，报告 `Delta = L_wrong - L_true`、来源组bootstrap不确定性与候选排序；正确caption不唯一，同PCM多caption作为多正例而非冲突自动删。
4. 若下述去舞种/时间戳与扩大音频scope的小样本仍没有音乐收益，再试条件匹配loss，不直接全量新增loss。

可选的一正一负条件ranking（**仅计划，未实现/运行**）：

`s(a,y) = mean_t log p_theta(y_t | a,y_<t)`，

`L_match = softplus((s(a_negative,y) - s(a_positive,y)) / tau)`，

`L_total = L_caption + lambda_jev L_jev + lambda_match L_match`。

正负使用同y、同长度/同动作token支持，减少单纯文字流畅度的捷径。负例必须排除同歌曲/来源/精确PCM及多正例关系，困难负例要验证确实不匹配。错误负例会压制合法编舞，不能以“不同样本ID”代替负例依据。先比较lambda_match0与小权重；tau/权重由开发验证定，不使用测试集选择。batch32全配对会产生大量30B条件前向，不推荐；先每个样本至多一个负音乐、少量更新使用。有额外前向/可能额外反向的真实成本，不能声称免费。

备选为音频表示与固定文本/动作表示的多正例InfoNCE：成本更低，但文本embedding若只区分通用词不区分侧别/顺序，检索指标容易误导；只有诊断證明表征可用后再引入。检索与生成两个指标必须分别报告。

## 4. JEV分类头：从caption文本任务里拆出舞种

用户目标明确：**舞种是分类，不让Qwen Captioner直接生成舞种行。**

本轮实际以 `--no-enable-jev` 运行，genre_classes/labels没有接入，分类loss为0。下一版应复用已存在本地JEV头：

- `music_planner_jev.py:121`：LayerNorm(H) → Linear(H,256) → GELU → Linear(256,C)，FP32；
- `music_planner_jev.py:194`：取decoder**最后输入prefix token**状态，不用teacher-forced动作或真实舞种；
- H=2048时参数量为 `2H + H*256 + 256 + 256*C + C`；若C18则533,266参数，约0.53M，比LoRA小得多。C须核对实际分类表，不默认18个类全部可学。

分类有标注集合G：

`L_jev = -sum_{i in G} log softmax(JEV(h_prefix(a_i)))_{g_i} / |G|`。

每个样本计一次，不按caption长度倍增。主任务与分类使用独立global支持分母。先小预算比较lambda_jev0与0.1起点；0.1只是候选，不是已确认最优。

**建议协议**：

- 原始 `motion_caption`、舞种行原样留在raw；转换版本只把舞种行移至分类label，caption正文保持原事件与顺序。
- 外部输出是 `{dance_style: JEV预测, action_caption: Captioner动作正文}`；如需要显示舞种行，由应用组合，不把JEV输出计入decoder生成。
- 先使用可靠已存在舞种label，unknown按未知监督mask/拒识处理；不把unknown强行当多数舞种训练，不从音乐genre替代dance style。
- 报告known覆盖率、每类support、macro-F1/balanced accuracy、混淆矩阵、置信度校准/拒识；top1总体准确率不足以代表稀有类。
- 若希望JEV反过来帮助动作生成，可用预测概率条件或联合梯度另设对照；不能训练时注入真实舞种、推理时只给预测而不披露协议差异。先做分类输出分离，不默认额外注入。
- 它借用了本地JEV决策头设计，并非调用外部JEV服务或宣称复现私有算法。旧集群JEV48.1% top1为不同标签/数据实验，不能当本轮分类验收。

## 5. 训练速度和扩大参数：实测、算术和未知分别列出

### 5.1 实测速度

以训练管理器stage_start/exit文件时间差计量，包含原生模型加载、预处理、训练、评测和checkpoint保存；不冒称纯GPU kernel时间。

| 全量实验 | 实测wall耗时 | 更新平均wall | 样本/秒（整节点） | GPU-hours |
| --- | ---: | ---: | ---: | ---: |
| 标准SFT，8卡 | 51.6353小时 | 30.7760秒/update | 1.03976 | 413.0823 |
| EOS20，8卡 | 51.5318小时 | 30.7143秒/update | 1.04185 | 412.2545 |

虽然只训33.95M参数，仍需30B级MoE底座前向、跨decoder反向链路和ZeRO参数gather；**可训参数小，不等于底座计算被省掉**。还包含每100步保存和本地/共享音频读取。只从高GPU利用率不能判断瓶颈，应分开profile：数据等待、audio forward、decoder forward/backward、ZeRO通信、loss、保存。

提速优先级：小数据预处理/本地缓存复用（不重复hash不变文件）→原生tensor形状/动态长度优化→bucket与通信profile→经过合同验证的microbatch/accum对照。当前代码microbatch1有packing限制，不能直接改CLI micro2而声称支持；任意packing不能串音频或改变loss分母。不要先降精度、卸载到CPU或清除共享缓存作为快捷方案。

### 5.2 CPU元模型实际形状测算

以下是精确形状与rank计算，**不是新GPU训练或速度结果**。融合专家LoRA保持关闭，基座BF16，可训FP32。

| 范围 | 可训参数（含投影、不含JEV） | 参数+梯度+Adam一二阶总状态，按16 bytes/param | 说明 |
| --- | ---: | ---: | --- |
| 当前rank32、音频最后4层 | 33,950,976 | 0.506 GiB | 基线 |
| rank64、音频最后4层 | 63,638,784 | 0.948 GiB | 单纯扩大rank |
| rank128、音频最后4层 | 123,014,400 | 1.833 GiB | 优先级低于音频scope诊断 |
| rank32、音频最后16层 | 42,798,336 | 0.638 GiB | 仅扩大音频覆盖 |
| rank64、音频最后16层 | 81,333,504 | 1.212 GiB | 推荐容量候选，须隔离rank/scope影响 |
| rank64、全部32层音频LoRA | 104,926,464 | 1.564 GiB | 更广音频适配 |
| 全音频tower解冻 + rank32语言LoRA | 674,665,856 | 10.053 GiB | 另需实现scope/优化器/export及验证 |

当前分解：language LoRA26,738,688；audio LoRA2,949,120；projector4,263,168。音频tower原生647,927,168参数。主模型H2048，128 experts，每token选8；不能把A3B当作总权重只有3B。

表中的Adam状态是全实验总量，理想ZeRO3可除以world size，但**不是总显存预算**；另有冻结权重、激活、临时gather、通信buffer、FP32副本等。本轮训练快照每卡实际显存约27GiB（采样时点），不是覆盖所有最长样本的peak；原始解码单卡加载约61GiB。不要用27GiB平均值当所有新配置的硬安全上限。

### 5.3 时间和效果如何变化

- rank翻倍会使对应LoRA参数/状态近似翻倍，**不会使30B底座必然多算一倍**；不能用参数比线性外推训练耗时。
- 扩大音频层会把可适配范围前移，可能增加反向激活与计算；全音频解冻增加权重梯度，不同于增加低秩矩阵。
- 时间必须在相同样本/长度分布、global batch、dtype、save/eval规则下测量固定更新。建议预热后记录100个真实更新的samples/s、token/s、p50/p95 update时间、peak显存；和基线profile同口径。
- 新epoch耗时只能在取得实测吞吐后按 `T = N / measured_samples_per_second` 做预算，不在本文给未经实测的未来耗时承诺。已有51.6小时是过去实际消耗，不是预测。
- 更多容量可能改善方向/侧别/顺序辨别，也可能只更强地拟合文字规律、过拟合或破坏预训练音频能力。必须以映射对照判断；不承诺“参数越大效果越好”。
- 全语言MoE解冻需处理数百GiB优化器状态和通信/检查点，不能凭8卡总显存就承诺安全；本次不推荐。先rank/scope小预算，更符合映射诊断。

## 6. 不学时间戳、裁短音乐：建议先去输出时间戳，再按已有窗裁音频

### 6.1 同5.0625秒音乐、不生成数字时间戳

建议第一对照：舞种移交JEV，然后从动作正文移除行首数值时间标记，保持全部动作文字、窗口顺序、真实重复。raw及metadata保留绝对start/end，转换记录清楚。可保留换行表示顺序块；不要求模型生成精确秒数。如果下游不需要窗数，可进一步独立测试整段顺序正文，但不能在同一个对照里同时压缩事件。

同一确定性256样本、真实native tokenizer、每条包含EOS：

| 目标 | 平均token |
| --- | ---: |
| 当前完整舞种+时间戳+动作 | 381.629 |
| 移除舞种行（JEV负责） | 375.574 |
| 再移除数字时间戳、保留动作/块顺序 | 335.270 |

时间戳约占去舞种后目标10.73%；舞种行约6.05 token。去时间戳可移除数值边界学习与相关loss竞争、适度减少输出长度，**不等于训练速度提高10.73%**，因为audio/decoder底座和通信等开销仍在。更重要的是任务简化，不是纯token节省。

代价：不再有秒级动作定位；无法声称已完成踩拍、时长覆盖或时间窗复现。应换成动作内容/顺序、停止、音乐辨别评价，不再用原timestamp格式门槛否决新合同。数字序列不学不等于模型无时序理解，仍需学动作顺序。

### 6.2 裁短音乐

CPU实际native processor形状（零波形只用于形状测量，没有改真实音频）：

| 音频秒数 | audio feature帧 | prefix token |
| --- | ---: | ---: |
| 1.0 | 100 | 23 |
| 2.0 | 200 | 36 |
| 2.53125 | 253 | 43 |
| 5.0625 | 506 | 76 |

2秒相对5.0625秒减少40个prefix token；若完整动作仍335token，decoder总序列长度不会同比缩至40%。只有监督动作对应缩短才可能显著减负。**不能裁2秒音乐却仍监督完整5秒caption**。

推荐利用已有caption窗口边界形成真实配对：

1. 从原视频/旁路音频取完整窗口 `[start,end]`；本窗口caption作为目标，去数字时间戳，绝不按一个2秒固定切点截断无法定位的事件。
2. 优先1–3秒的已有完整窗口。当前训练547,544个窗口中351,800个在1–3秒范围，窗口时长p10=0.86、median1.68、p90=2.9825秒；这只是候选规模，尚未裁剪/准入。
3. 所有派生窗继承原视频/歌曲来源split；音频裁剪采样索引及实际帧数、历史pad交集、目标来源/绝对时间完整留元数据。不是重新打caption。
4. 很短窗口难判断风格/节奏相位；JEV可消费原完整观察范围，captioner消费局部范围，作为明确的双分支协议，不能偷偷把额外长音频算作短输入。
5. 固定2秒无重叠裁剪只有当事件有足够时间证据可以准确归属时才做。跨边界动作不能根据文字顺序猜精确归属。
6. 限制原来源贡献权重/记录曝光秒数和事件数，避免窗多样本被倍增采样。全量拆窗不是免费提速：行数从193k增至数十万，1 epoch更短不成立。等来源秒数/等事件预算和等更新预算要同时报告。

局部caption首句只写“the other side”或“then”时可能依赖前窗；应显式标记上下文依赖，在独立对照处理，不默默补动作/猜侧别。音频增加上下文而仅监督中心窗也是新协议，必须明确实际观察范围与目标范围。

## 7. 下一阶段实验矩阵与判定标准

不按未验证配方直接再做193k全量。也不因为音乐不能唯一决定编舞就停止学习映射。先在保事实新合同下小预算比较，模型生成固定三例。

### 7.1 必做的实现/记忆验证

- 在同一实际序列上比较cache/无cache next-token logits与ZeRO/portable前向，而不是自由生成分叉后比完整文字。已知旧三例128token对照2例不一致，尚不能只归结BF16误差。
- 验证新合同无caption/header泄漏、JEV只取音频prefix、时间戳移除可逆到原数据、裁剪没有串split。
- 单case → 8case有界真实过拟合：保全部动作，检查方向/侧别/顺序/次数和EOS，不将loss下降当完整记忆通过。已有40步小实验不是充分记忆验收。
- 保存/新进程恢复、实际梯度、原生EOS、完整adapter与JEV头一起portable恢复。不能因为旧合同smoke通过就跳过新合同。

### 7.2 第一轮：2×2保事实因子对照（所有方法JEV输出舞种）

共同：固定2k–5k原来源样本、同split/seed/global batch、同曝光预算、无EOS额外权重、无禁重复动作词；原始caption完整保留为raw。小预算上限与checkpoint检查点执行前冻结。

| 配置 | 时间戳目标 | 音频LoRA层 | rank | 改变的目的 |
| --- | --- | --- | --- | --- |
| P0 | 保留 | 最后4 | 32 | 新JEV协议基线 |
| P1 | 不生成 | 最后4 | 32 | 单独测去数字边界 |
| P2 | 保留 | 最后16 | 32 | 单独测音频适配范围 |
| P3 | 不生成 | 最后16 | 32 | 测两因素交互 |

第一轮scope对照保持rank不变，避免把rank和层数混成一个因素。若音频scope有收益，再以胜出协议比较rank32/64（81.33M为16层rank64）；必要时对比全audio LoRA，再决定全音频解冻。不是一次跑所有rank/层组合。

### 7.3 第二轮：已有完整短窗口对照

用胜出协议，比较全5秒正文与1–3秒完整原窗正文；保留绝对时间元数据，JEV观察范围独立明确。同源训练/验证不串组。按源秒数/动作事件曝光匹配，报告窗口数变化。若只缩短音频但无内容收益，不因快而判胜。

### 7.4 第三轮：仅在音乐辨别仍弱时做直接匹配目标

CE+JEV 与 CE+JEV+条件ranking两配置；保同数据、scope、rank、时间协议和负例策略。先一负例有界预算，记录开销和false-negative复核。若匹配提高但生成事件错误/实际音频贴合变差，不升级全量。

### 判定

- **映射**：相近节奏/风格困难错配中正确音乐有可复现优势，组级区间与每case证据，不只是比静音低loss。
- **内容**：按闭集事件匹配部位/方向/侧别、次数与顺序/并发；不追求同义词多样性，不强制删除真实重复。存在多编舞参考时做多正例评估。
- **停止**：EOS、预算截断、完整动作覆盖；无时间戳版不使用秒级时间覆盖门槛。
- **JEV**：known-label覆盖、macro指标及拒识，单独验收，不用动作生成unknown当分类错误。
- **技术**：真实前反向、保存恢复、数值有限、prefix和metadata绑定、无泄漏；代码/shape检查不等于质量通过。
- 开始前固定指标和预算，不因结果不好持续增加epoch。若小数据也不能完整记忆，先解决实现/优化；若只提高文本loss而音乐辨别不变，不扩大全量。

## 8. 建议资源（规划，不是已经占用）

本轮全部训练和最终推理自有holder恢复。使用更多节点已获用户授权，但下次启动仍检查真实非holder任务、租约、host/cgroup内存和磁盘；不停止其他训练，不清共享内存/依赖。

### 最低可推进布局

- 1个训练节点，优先8×A800复用验证过的ZeRO3配置；4卡是可测试的更紧凑候选，不冒称已经通过。P0–P3顺序完成。
- 1个推理节点，1×A800加载已有Thinker做同三例推理与256条非生成打分；缓存分类头训练可CPU或小GPU。
- CPU数据准备2–4worker，首次配对/形状/边界检查后复用清单与cache，不重复全文件hash。

### 建议吞吐布局

- 4个训练节点各8卡，并行P0–P3有界实验，避免在已启动任务中改world size。
- 1个推理节点按需2–3卡分别做同三例的初始化/候选及打分；高并发模型加载先看host/cgroup余量。
- 其余节点CPU做标签关联、完整窗清单、负例/同歌曲来源核对，GPU留给其他任务或按现有规则holder。不为占满8节点复制同一实验。

预算口径：每训练节点至少170GiB真实可用CPU内存作为先前成功布局的准入起点，建议预留约220GiB并复查cgroup；每新官方模型本地副本约63.5GB（十进制），样本缓存/日志/断点另计，按120GiB本地空闲起步。最终8卡checkpoint本轮约0.695GB/个，备份保存策略需按checkpoint数预算；新scope扩大后文件量随可训参数变化。

每推理任务优先1卡80GB，模型占约61GiB并需激活/KV余量；CPU可用约140GiB起点、含权重副本的本地磁盘约80–100GiB。这些是准入预算，不是峰值保证。新增rank64与16层scope先100真实更新profile；未通过最坏长度显存或吞吐检查不继续。

全音频解冻候选仍可先1训练节点8卡验证，状态算术约10.05GiB全局，不含权重/激活/通信；不能把静态算术当实际启动成功。全语言MoE解冻不在当前实验矩阵，另需多节点通信、FP32优化器与checkpoint预算和实现支持。

## 9. 可追溯性与未同步内容

本报告分析的训练代码基线为main@02f478a；按用户要求，文档同步到codex/music-choreo-benchmark分支的notes目录。文中的planner代码相对路径/行号指main@02f478a，不能据此声称本benchmark分支已部署这些训练实现。本次提交仅为文档，不合并两项实验合同。没有把原独立训练工作区的EOS、汇总工具等未提交代码混入本次文档提交，也没有修改冻结ChoreoBench。报告中的本轮结果来自本地真实运行日志、最终run_config、adapter元数据、完整checkpoint审计、同三例原输出与音频消融；本轮扩展不冒称main已经完整包含。

证据类别：

- 当前合同/损失与原生prefix：main中的 `music_planner_adapters.py`、`music_planner_schema.py`、`music_planner_token_loss.py`、`music_planner_train.py`；
- JEV头：main中的 `music_planner_jev.py`；
- 吞吐：两个完整训练stage_start/exit时间差；
- 参数：CPU init_empty_weights元模型，按实际Linear形状和rank计算，未使用GPU；
- token与短音频shape：固定256实际训练来源native tokenizer；1/2/2.53125/5.0625秒CPU原生processor；
- 质量：最终6040步同三例18条输出及相同配置官方初始化；
- 来源独立：全部新split来源组与精确音频交叉检查；歌级非精确相似仍未验证。

未上传原视频、音频、原标注大文件、机器地址、凭据或用户其他任务文件。本报告保留实际测量与建议的边界：参数算术不是性能实测，过去wall耗时不是未来承诺，技术通过不是映射/语义通过。
