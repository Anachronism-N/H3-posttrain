# 本地验收记录：2026-11-04

本记录只描述实际执行的软件验收。真实音乐数据、训练权重和GPU实验不包含在本次结果中。

## 环境

- Windows，Python 3.9.18。
- NumPy 1.26.4、SciPy 1.13.1、soundfile 0.13.1。
- 当前环境未安装 torch、transformers、peft、safetensors；没有加载真实规划器，没有调用付费API。
- HTTP provider 测试使用受控假响应；不能视为已验证所有兼容服务或真实模型。

## 已执行

1. `python -m pytest -c choreobench/pytest.ini -q`：**154 passed**。覆盖数据/时钟与分组、跨平台ID、音频身份与准备、三种baseline、预算与恢复、原始响应/请求绑定、模型输出失败、编辑保护/私有目标隔离、盲评绑定、歌曲级统计、方法无关提交及CLI联调。
2. `python -m choreobench demo --out-dir ... --run`：短合成音频端到端通过，包含concat/global/hierarchical和手工patch评估。
3. `python -m choreobench demo --out-dir ... --seconds 180 --run`：180秒合成音频跑通三种baseline与编辑流程。这里只验证长度、数据流与预算合同。
4. prepare-audio：180秒PCM输入完成规范化与receipt；另有24kHz双声道→16kHz单声道回归，检查样本数、SHA、来源记录及拒绝覆盖。
5. CLI集成测试验证盲评音频复制、原始身份隐藏、缺评保持pending、提交缺失预测保留分母和精确版本撤销。
6. `python -m compileall -q choreobench` 与Git whitespace检查通过。
7. 原有 `tests/test_music_planner_search.py` 回归：**60 passed, 8 skipped**。8项依赖torch，当前环境跳过。

首次远端CI中Linux/Python 3.10与3.12通过，Windows/Python 3.12暴露了音频元数据误报。Windows路径stat与句柄fstat的ctime语义可能不同（[CPython源码](https://github.com/python/cpython/blob/v3.12.12/Modules/posixmodule.c#L2015)）；现改为同一API的前后句柄比较，并保持原句柄打开、重开路径检查替换。增加了路径元数据差异不误报，以及读取中修改/路径替换仍拒绝的三项回归。修复后的远端矩阵结果以GitHub Actions为准。

## 未完成的服务器验收

曾尝试同时收集原有search及search_identity测试；后者在导入时因缺torch失败，未执行，不能记为通过。没有为掩盖此限制安装或模拟真实torch环境。后续必须在真实模型环境运行该identity测试及至少一个真实样例。

待服务器执行：完整checkpoint兼容性与加载、真实5.0625秒原生推理、EOS/输出格式、实际显存/延迟、一首完整歌曲、真实外层服务、编辑语义质量及人工音乐适配评审。CPU mock通过不证明音乐理解、整曲编舞质量、模型训练成功或SOTA。

## 结果解释

- 本地自动通过项只验证实现合同；音乐贴合、主题发展和实际可执行性仍待独立评审。
- 正式歌曲集、编辑目标与独立评审需要按PROTOCOL_zh.md补齐。synthetic数据或预测明确排除科学发布资格。
- GitHub CPU workflow已提供，用于Linux/Windows与不同Python版本检查；本记录不提前声称远端CI已经通过。
- 真实运行建议顺序：doctor → 一个真实短窗 → 一首整曲 → 小型pilot → 冻结协议后的正式实验。

## 第二轮优化验收

- 本地完整ChoreoBench测试：**286 passed**。新增候选库导出/跨方法复用、保留坏样本、无局部provider加载、成本区分、编辑选择器/跨界拒绝、身体状态与主题历史、来源组配对统计、评分来源隔离、外部提交盲评与私有来源绑定等覆盖。
- 180秒合成音频：首次concat采集36次局部调用；同一候选库分别用于concat/global/hierarchical，三者都复用36个局部回答，新增调用分别为0/1/3（该夹具为2个段落，因此hierarchical为1次outline+2次section）。三条运行均完成且原参数resume返回相同结果，resources估算与新增调用一致。
- 检查了section请求实际包含逐身体部位的boundary_context；后段确实收到前段退出信息。该信息仍是计划声明，不能据此认定物理连续性或编舞质量提高。
- 首次180秒验收脚本误传了相对于工作目录的候选库路径，被按dataset目录解析后明确拒绝；改为绝对路径后完成。代码未为通过验收猜测或重写用户路径，README明确了相对路径基准。
- 候选库逐窗口只检查文件元数据，完整内容哈希在歌曲边界/最终检查，避免对整个库每窗重新扫描；仍会读取音频并生成/校验lossless窗口，不能声称零CPU或零磁盘成本。
- 此轮仍没有真实GPU推理、付费API调用或人工音乐质量结果。远端四组CPU CI以对应提交的Actions记录为准。
