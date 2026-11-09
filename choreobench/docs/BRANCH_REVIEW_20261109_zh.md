# 指定提交与各分支的审查和接入建议

2026-11-09，已fetch全部远端heads。GitHub对用户给出的`9eb7d3d`返回“No commit found”；可核验的相近结果提交为`9eb7d3b1348d9ab65fc7d124d40f89ebde8a9a06`。本次以该提交及下列固定分支快照进行只读审查，未合并其他训练分支、启动训练或读取服务器权重。

## 1 对9eb7d3b的review结论

该提交相对`a597879`只改三份文档：英文稿件、进度说明和验收记录，没有改训练器、生成器或评价算法。未发现这份diff新增的算法bug或结果算术矛盾。

- 三首独立受约束基线的27＋26＋23＝76次调用，另有3次编辑，合计79次。输入264,573和输出35,948 tokens与分项一致。
- 编辑合法patch、非目标字段保持、精确undo与目标语义待评分开，没有把字符串改变直接算作编辑成功。
- 独立块基线同时移除大纲、主题权限和历史；稿件已披露它不是历史单变量，以及旧main与新基线跨服务session。
- 新基线同样达到合法性和完整覆盖，是有用的控制结果，不能再用这两项证明全局规划的质量收益。

仍需补充可复核的运行证据。原v17目录未随仓库提供，本机没有其原始请求/响应、保护/撤销回执及浏览器验收工件。本轮核对的是公开记录和代码，不是重新计算服务器原件。建议在受控只读位置提供运行工件，并导出脱敏的配置/来源身份、每任务状态、成本与校验清单；不必公开音频、私有目标、人员身份或完整内部日志。

两个先前问题已由`973545a`处理，均不是9eb7d3b这份文档diff新增：Windows页面测试读文件未指定UTF-8，以及稿件将单次输出差异写成输入干预的因果效果。修复后本地非UTF-8模式页面测试29项通过；[修复提交的四组CI](https://github.com/Anachronism-N/H3-posttrain/actions/runs/42604836040)全部成功。Linux/Windows各覆盖Python 3.10与3.12；此结果不代表新训练分支也通过同一套模型验收。

## 2 分支地图

日期为各远端head的提交日期，按北京时间；不代表某个服务器作业今天仍在运行。

| 分支与固定head | 日期 | 实际用途与证据 | 对当前整曲任务的建议 |
|---|---|---|---|
| `codex/music-choreo-benchmark` `973545a` | 11-09 | 整曲推理、编辑、评价工具、真实开发pilot与论文材料 | 继续作为benchmark和harness主线 |
| `main` `02f478a` | 11-06 | 新的Qwen3-Omni Captioner训练、数据构建、评测和真实训练记录 | 最值得进一步接入已训练局部专家；先核checkpoint及协议，不直接覆盖当前adapter |
| `codex/planner-encoder-experiments` `120fe6d` | 10-29 | MOSS末4/8层LoRA范围、冻结MusicFM dense特征融合 | 文档明确尚无真实服务器训练验收；是后续训练研究，不是现有质量成果 |
| `codex/planner-moss-captioner-64gpu` `a9107c3` | 10-28 | MOSS/Captioner训练范围、style+timed协议、可选JEV和多节点配置 | CPU计划/测试不等于64GPU训练完成；暂不为接入benchmark而启动 |
| `experiments/music-planner-fair-20261025` `7ed6373` | 10-26 | 三族短窗训练、test500、固定三例H3视频与实际整体偏好 | MOSS checkpoint670是旧协议下可复用候选，不是整曲任务最终赢家 |
| `codex/planner-search-rsi` `a138930` | 10-25 | 有限候选采样、来源记录、恢复、盲评与偏好数据暂存 | 部分原生采样/身份核验已被当前provider复用；未实现自动权重更新或完成RSI |
| `codex/planner-cot-multitask` `4a6e643` | 10-25 | direct与短计划多段SFT、分段loss及非teacher-forced生成审计 | 新机制仍需训练和真实验收；不能只换mode让旧checkpoint支持新协议 |
| `codex/planner-genre-decision` `462a850` | 10-25 | 输入音频特征的舞种头、校准路由与Captioner联合header训练 | 新协议未有真实质量验收；风格控制是可选条件，不能替代全曲组织 |
| `codex/motionscript-optimization` `c7a976e` | 10-24 | 姿态处理、事件压缩、可见性支持、RGB32有限实验及分段描述 | 作为标注/参考证据工具；扩点与润色尚无已验证caption准确性优势 |

10月各分支大多是相互独立的实验路线，不能因为分支名含64gpu、RSI或encoder就推断对应实验已经完成。继承公平训练分支的真实训练记录，也不证明新增机制已有训练效果。main中的`music_planner_encoder_fusion.py`和`music_planner_musicfm.py`已与encoder分支对应文件字节一致，不应重复合入并称为新能力。

## 3 main中的真实训练成果与接入限制

完整入口为 [planner README](https://github.com/Anachronism-N/H3-posttrain/blob/02f478a8b3a341f309e82a09ed55b8a823fd1391/scripts_omnidance/exp/ms/planner/README.md)、[训练记录](https://github.com/Anachronism-N/H3-posttrain/blob/02f478a8b3a341f309e82a09ed55b8a823fd1391/scripts_omnidance/exp/ms/planner/docs/TRAINING_RECORD.md)及同目录`results_summary.json`。该路线使用Qwen3-Omni-30B-A3B-Captioner，原生输入5.0625秒、24kHz、121,500采样，正文采用可变时间窗；默认无舞种header，JEV变体另行声明。

记录包含full-v4的196,097 train / 10,990 val / 11,119 test及实际SFT过程。末尾交接状态推荐`sft198k_5515`（`sft_full_20261102_232838/checkpoint-0005515`）作为当前该路线的实用候选，`strat_full_20261106_022254/checkpoint-0003665`作为密度权衡对照；不按NLL最低直接宣布生成质量最好。部分实验在11月6日记录仍为in_flight，不能据此断言11月9日的服务器状态。

因此应纠正范围表述：**整个仓库已有训练成果；当前ChoreoBench已记录的整曲试验仍是base-only候选。** Git不含权重，接入还需要真实checkpoint、base、run_config/完成标记、导出身份及加载回执。

5.0625秒时长一致提供了复用基础，但采样预处理、训练prefix、可变窗解析和checkpoint结构都要核对。采样率不同本身不代表不可兼容，也不能通过强行改写为旧双窗文本来假装协议一致。当前NativeProvider身份核验支持的fair/direct协议不自动涵盖main新导出、short-plan或joint-genre。应新增明确provider/合同版本，保留原始回答和解析失败，再导出可被整曲规划器读取的候选；验收成功不等于质量已经提高。

旧fair分支的 [模型选择记录](https://github.com/Anachronism-N/H3-posttrain/blob/7ed6373a0de6cb343967bec194e9ac31454e237d/results/music_planner_final_model_recommendation_20261025_v1.json)确有MOSS checkpoint670、test500和三例真实用户整体视频偏好：两例偏好MOSS胜Flamingo，其中一例仅略优，另一例Flamingo更好。这是旧短窗/固定H3条件下的实用选择依据，不是当前整曲v11/v17的人评；它也不能与main新数据/目标协议的分数直接排名。

## 4 main中需优先处理的具体问题

### 全局CE缩放被解释为动作密度加权

[music_planner_token_loss.py:25](https://github.com/Anachronism-N/H3-posttrain/blob/02f478a8b3a341f309e82a09ed55b8a823fd1391/scripts_omnidance/exp/ms/music_planner_token_loss.py#L25)对所有非-100的监督token赋同一个权重w，随后仍以未加权的token数归一化。因此单独caption任务的损失为`w × CE`，没有区分动作内容、时间戳或其他受监督token。启用辅助头时，还会改变caption项相对辅助项的权重。

这不证明实验一定无效，但不足以将效果归因为“选择性增加动作密度监督”。全局缩放也不能简单等同调整学习率，Adam、梯度裁剪及多任务权重会影响训练动力学。先把现有实验准确标为caption-loss scaling；若将来研究内容密度，需明确哪些token/段落获得不同权重并做独立消融，当前不因此重训。

### 评测脚本依赖原部署目录且不保留完整原回答

[eval_many.py](https://github.com/Anachronism-N/H3-posttrain/blob/02f478a8b3a341f309e82a09ed55b8a823fd1391/scripts_omnidance/exp/ms/planner/eval/eval_many.py)在第10–12行将原机器绝对目录加入模块搜索路径，再导入顶层adapter。新机器可能找不到模块，或读入另一个部署快照，而不是正在review的源码。后续生成先清理输出，再保存清理后的text，缺少对应原始text/token记录，不能直接充当ChoreoBench的原始候选审计。

迁移时应显式指定依赖/模型路径并记录实际加载源码身份；分别保存raw与normalized文本、规范化版本、停止原因及失败。不把删除格式错误后的结果作为模型原始格式成功，也不默默丢掉被清理掉的内容。

### 快速开始命令缺少必需输出参数

main的README训练示例没有`--out`，而 [music_planner_train.py:1200](https://github.com/Anachronism-N/H3-posttrain/blob/02f478a8b3a341f309e82a09ed55b8a823fd1391/scripts_omnidance/exp/ms/music_planner_train.py#L1200)要求它必须提供。按示例原样运行会在参数解析时失败。实际trainer位于`scripts_omnidance/exp/ms/`，README的模块命令路径本身正确；布局表中的`planner/trainer/`等描述需与实际树同步。接入指南至少应给独立输出目录和部署依赖，避免读者靠猜测补环境。

此外，audio-swap导致caption相似度下降仅说明输入敏感性，不独立证明编舞与音乐正确对应。动作子句数、长度比及词表F1可用于诊断，尚不能代替开放文本编舞的独立评价。

## 5 推进顺序

1. 继续在benchmark分支收取已有v17规划/编辑原件的人评，不等待训练分支整合。原整曲baseline冻结保留。
2. 并行核验main推荐的已完成checkpoint与旧MOSS候选，优先解决输出协议和真实加载身份。只需小规模原生生成验收，不新增训练；随后在固定局部音乐上盲评候选质量。
3. 只有局部专家收益可辨时，才在相同全曲规划配置下替换候选源，比较“专家变化”而非同时改变大纲、历史和时间菜单。变动候选需新身份、新目录，不能污染原base-only结果。
4. 扩展每歌多编辑任务前，补独立任务清单和task/seed身份汇总；保留缺失/失败/待评分母。
5. genre、short-plan、MusicFM融合保留为后续研究。无需为了继续整曲benchmark而同时训练这些路线，也不整支覆盖当前已有审计和推理实现。

当前可直接复用的是已有checkpoint、原生生成/身份核验经验和评价约束。正式benchmark仍需来源组审计、校准标注及新歌评价；各训练分支的旧参考指标不自动转化为该任务的质量结论。
