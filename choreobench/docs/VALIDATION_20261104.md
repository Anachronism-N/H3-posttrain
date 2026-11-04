# 本地验收记录：2026-11-04

本记录只描述实际执行的软件验收。真实音乐数据、训练权重和GPU实验不包含在本次结果中。

## 环境

- Windows，Python 3.9.18。
- NumPy 1.26.4、SciPy 1.13.1、soundfile 0.13.1。
- 当前环境未安装 torch、transformers、peft、safetensors；没有加载真实规划器，没有调用付费API。
- HTTP provider 测试使用受控假响应；不能视为已验证所有兼容服务或真实模型。

## 已执行

1. `python -m pytest -c choreobench/pytest.ini -q`：**151 passed**。覆盖数据/时钟与分组、跨平台ID、音频身份与准备、三种baseline、预算与恢复、原始响应/请求绑定、模型输出失败、编辑保护/私有目标隔离、盲评绑定、歌曲级统计、方法无关提交及CLI联调。
2. `python -m choreobench demo --out-dir ... --run`：短合成音频端到端通过，包含concat/global/hierarchical和手工patch评估。
3. `python -m choreobench demo --out-dir ... --seconds 180 --run`：180秒合成音频跑通三种baseline与编辑流程。这里只验证长度、数据流与预算合同。
4. prepare-audio：180秒PCM输入完成规范化与receipt；另有24kHz双声道→16kHz单声道回归，检查样本数、SHA、来源记录及拒绝覆盖。
5. CLI集成测试验证盲评音频复制、原始身份隐藏、缺评保持pending、提交缺失预测保留分母和精确版本撤销。
6. `python -m compileall -q choreobench` 与Git whitespace检查通过。
7. 原有 `tests/test_music_planner_search.py` 回归：**60 passed, 8 skipped**。8项依赖torch，当前环境跳过。

## 未完成的服务器验收

曾尝试同时收集原有search及search_identity测试；后者在导入时因缺torch失败，未执行，不能记为通过。没有为掩盖此限制安装或模拟真实torch环境。后续必须在真实模型环境运行该identity测试及至少一个真实样例。

待服务器执行：完整checkpoint兼容性与加载、真实5.0625秒原生推理、EOS/输出格式、实际显存/延迟、一首完整歌曲、真实外层服务、编辑语义质量及人工音乐适配评审。CPU mock通过不证明音乐理解、整曲编舞质量、模型训练成功或SOTA。

## 结果解释

- 本地自动通过项只验证实现合同；音乐贴合、主题发展和实际可执行性仍待独立评审。
- 正式歌曲集、编辑目标与独立评审需要按PROTOCOL_zh.md补齐。synthetic数据或预测明确排除科学发布资格。
- GitHub CPU workflow已提供，用于Linux/Windows与不同Python版本检查；本记录不提前声称远端CI已经通过。
- 真实运行建议顺序：doctor → 一个真实短窗 → 一首整曲 → 小型pilot → 冻结协议后的正式实验。
