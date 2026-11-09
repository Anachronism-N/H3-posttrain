# 缓存前向校准与 ZeRO / portable 等价性实测

本报告为原标准SFT@6040的服务器实测记录，数值来自本地原始快照；不是在GitHub CI中重跑30B。原训练实现引用 `main@02f478a` 的实际部署快照，诊断工具基于CaptionLab分支扩展。

## 结论

1. **同权重、同token/attention/音频feature、同FP32参数存储时，ZeRO与portable前向通过严格一致性核查。** 原标准SFT@6040，三例、30个预指定位置的完整词表logits逐位一致；全部1,715个监督目标token的log-prob逐位一致。BF16 autocast和FP32计算参考两种模式都观察到这一结果。
2. **现有单卡“冻结基座BF16存储＋可训adapter FP32”的portable不是原ZeRO的逐位等价计算策略。** 在冻结同输入排除CPU特征差异后，最大逐目标log-prob差为0.4697；不是adapter丢张量，因为恢复已逐张量验证。仅把归一化权重或归一化＋embedding改FP32的单卡实验未恢复严格一致，不部署这两种猜测修复。
3. **缓存与整段前向不应设置一个很宽的容差就宣称通过。** 同形状完整前向重复、同形状缓存前向重复均逐位相同；不同长度完整前向本身也有BF16差异。FP32下两个case的cache/full误差收敛至约6e-5以内logit量级，另一个case有near-tie MoE expert路由变化。固定参考路由的诊断干预使该残差收敛，说明剩余差异不能全部简单归为BF16。

本次所有任务是前向与状态恢复诊断，没有训练更新、没有自由生成、没有修改原checkpoint或标注；N2单卡和N8八卡自有holder恢复全部核验通过。此报告不是模型编舞质量通过证据，也不代表所有checkpoint/输入的普遍等价定理。

## 方法与控制

- 使用原训练代码快照及run_config，核对训练adapter和原runtime代码SHA。
- 原ZeRO checkpoint复制到独立诊断目录后，按原world8、optimizer组和精度配置创建fresh engine；恢复Adam/调度器/RNG与FP32可训张量，逐张量对比保存状态；前后优化器摘要一致。
- 三例原音频SHA通过；token IDs、labels、attention mask与feature mask严格绑定。
- 测量完整前向两次、同目标前缀的短序列完整前向、缓存两次、FP32输入与BF16输入。
- cache沿同一预定目标序列推进，保留native三维position与rope delta；不以自由生成分叉作误差指标。
- 对比原ZeRO BF16计算、ZeRO FP32计算、portable多卡FP32存储下BF16/FP32计算，以及当前BF16单卡portable。
- TF32关闭；没有通过修改共享transformers或DeepSpeed依赖来通过检查。

## ZeRO / portable 严格结果

| 比较 | 三例所选logits | 1,715目标token概率 | 说明 |
|---|---|---|---|
| ZeRO FP32存储/BF16计算 vs portable相同存储/计算 | 逐位相同 | 逐位相同 | 通过限定范围的严格比较 |
| ZeRO FP32存储/FP32计算 vs portable相同存储/计算 | 逐位相同 | 逐位相同 | 高精度参考也通过 |
| 原ZeRO vs 当前单卡BF16基座存储portable | 不相同 | 不相同 | 存储/算术策略不同，不能叫strict parity |

当前BF16单卡与原ZeRO，在同feature下三个case的最大逐目标log-prob差分别约0.4697、0.3447、0.3538；整段NLL差约+0.00157、−0.00125、+0.00083。所选30个位置1个top1变化。小平均差不能代表每个关键动作或EOS位置无影响。

参数值恢复正确不等于数值路径相同：原ZeRO所有参数FP32存储、BF16计算，portable冻结参数BF16存储会影响归一化、embedding/residual和MoE实现的dtype路径。没有证明仅某一个模块造成全部差异，所以已否决“只转norm即可修复”的未经验证方案。

## 输入特征独立因素

相同音频在两个节点的CPU预处理feature最大差约4.03e-5，BF16取整后也不完全相同；两个processor配置在CPU同节点比较一致。没有猜测为某个特定CPU/库bug。

因此本次严格ZeRO/portable比较重放**原节点已实际消费过的feature tensor**，其hash与ZeRO快照相同。保留原音频、不重新裁剪或补音。这消除了输入差异导致的混淆。

同一组feature下，将输入FP32留到模型入口还是先用原adapter mover降BF16，在本次BF16计算测量中输出一致；它不是本次严格差异的单独根因。跨节点feature内容变化仍需记录，不能仅凭token prefix hash相同宣称音频输入一致。

## 缓存校准与 MoE 路由实证

- 原单卡BF16策略：完整前向重复和cache重复误差均为0；full与cache最大目标log-prob差约0.185。full与同长度前缀的无cache前向也出现变化，说明不是随机dropout，也不能把整段未来token参与矩阵计算的长度差异忽略。
- 在FP32存储的BF16计算中，cache/full最大目标log-prob差可达0.345；这再次说明计算策略必须写入比较合同，而不是给所有路径统一一个误差值。
- 同FP32存储、FP32计算：两个case的full/cache最大logit差约4.96e-5、5.91e-5，目标log-prob差约8.58e-6、1.43e-5；不发生所选位置top1变化。
- 另一个case的FP32 full/short与full/cache仍有约0.0869目标log-prob差。router trace发现最早在decoder第12层、输入token95，expert top8中95与75互换；第8/9 expert概率间距只有约2.05e-8，gate logit变化约3.34e-6，随后24个层出现路由变化。
- 仅在诊断中固定完整前向的expert IDs、重新用当前logits计算路由权重，目标位置104的full/short log-prob差由0.08691降至约2.15e-5，logit差由0.5114降至约3.24e-5。这是实际干预结果，不是根据相关性猜测。
- 第三case也发现很晚层的一个near-tie路由切换，但对所测位置误差很小；不是所有路由切换都会造成明显输出变化。

**固定route干预不用于部署。** 它依赖参考完整序列，不是推理时可用条件；本次用于判断误差来源。

## 后续应采用的验证合同

1. 将checkpoint张量一致、输入tensor一致、计算策略一致和概率/greedy一致分别记录。
2. 严格训练/推理对照优先用本次已实测通过的多卡FP32参数存储路径或原ZeRO配置；保留单卡BF16推理为不同数值策略，不自动宣称错误或逐位等价。
3. cache/full比较按重复基线、长度基线、精度和路由边界分别报告，不扩大阈值把明显残差吞掉。未来输入超出当前经验包络需重新校准，不用三例包络宣称总体保证。
4. 关键动作/EOS在小margin位置容易被数值变化翻转。模型质量问题仍需真实输出审核；本次没有证明数值修正会解决病理复读或音乐适配。
5. 如需更紧凑的等价单卡/少卡实现，应另作算术路径对照并真实验证；本次两个单卡轻量修复候选均未通过，不替换生产loader。

## 证据位置（服务器私有目录，不随GitHub提交）

- 完整账本：`numeric_equivalence_summary.json`。
- 原始快照与对照：`evidence/cache_calibration`、`evidence/zero_original`、`evidence/zero_fp32_reference`、`evidence/matched_portable`。
- 精度匹配比较：`evidence/zero_portable_same_inputs_comparison.json`。
- 路由干预：`evidence/route_calibration/summary.json`。
- 输入控制：`evidence/portable_frozen_inputs`与`evidence/frozen_batches`。
- 未通过的轻量修复：`evidence/FP32_norm_fix_comparison.json`、`evidence/hybrid_norm_embedding_comparison.json`。

171项CPU测试、35项subtest通过；这些与真实30B前向结果是独立证据。

## GitHub同步范围与复现

- 本报告及脱敏数值汇总：[数值结果JSON](../captionlab/numeric_calibration_results.json)。case仅用A/B/C代称，不含音频、caption正文、机器地址或绝对路径。
- 工具：`captionlab/cache_calibrate.py`、`zero_parity.py`、`portable_storage_parity.py`、`portable_frozen_inputs.py`、`moe_route_calibrate.py`、`numeric_report.py`及`numeric_common.py`；CPU指标测试：`tests/test_captionlab_numeric.py`。
- 原checkpoint副本、实际batch张量与完整runtime工件由操作员在受控服务器提供，不能用脱敏JSON代替源数据审计。工具没有内置holder/资源调度器；真实执行先确认资源和任务归属。
- 本分支的CaptionLab不是原30B训练器。`--original-code`必须指向匹配训练的独立快照；原基座、权重版本、tokenizer/processor和全部portable状态需绑定。`cache_calibrate`使用原runtime factory，factory按已有README合同显式提供。
- 运行参数可用 `python -m captionlab.cache_calibrate --help` 和 `python -m captionlab.zero_parity --help` 查看。`zero_parity`要求原world8，多卡portable平衡加载也必须检查实际显存，不能以CPU测试通过代替GPU准入。
- 所有输出使用独立新目录；`numeric_report`汇总本轮规定目录结构的原始快照，若复现中缺某项，应保留缺失账本而非伪造文件。
- 未改现有CaptionLab评分内核、harness、训练器、已发布checkpoint和冻结ChoreoBench数据。
