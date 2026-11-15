# 两个结构性质疑的核查：MOSS 时间戳 与 用 VL 模型做外层规划

日期：2026-11-15。对应问题："MOSS-Music 为什么有显式时间戳？它是不是在描述音乐而不是动作？"以及"用 Qwen3-VL-30B 做外层规划是否合理？"

## 1. MOSS 候选里的时间戳来自我们的提示词，不是模型自己加的

`provider_moss.json` 的 `task_prompt` 原文：

> Listen to the music and describe a dance choreography in two time windows: [0-2s] and [2-5.062s]. Describe grounded, varied body movements for each window. Output only the two time-marked choreography descriptions.

所以 `[0.000s - 2.000s]` 这种前缀是**任务模板要求它分两个时间窗写**，模型照做了。对照 SFT v3 的候选（同一批窗口）：

| 来源 | 样例 |
|---|---|
| MOSS-Music | `[0.000s - 2.000s] In this opening segment, the choreography begins with a slow, deliberate stance, emphasizing grounded movements. Dancers perform a deep lunge...` |
| SFT v3 | `A leftward arm sweep begins with the dancer opening one knee, then curling that hand upward as both knees bend and straighten in succession...` |

**两者的差别是真的**：
- MOSS 的句子**同时描述音乐与动作**（"slow, deliberate stance"、"meditative groove"、"mirroring the..."），动作描述更笼统，长度约 812 字符；
- SFT v3 只描述**运动学**（哪只手、往哪、膝盖如何），约 610 字符，更接近可执行的动作指令。

因此用户此前的判断成立：扩展集应当用 SFT v3 重采，MOSS 的候选风格会污染下游计划。**注意**：两套候选池的文本唯一率都是 100%——计划里出现的"整段复用同一句"是外层规划器反复挑同一条候选造成的，不是专家只会说一句话。

## 2. 用 Qwen3-VL-30B 做外层规划的合理性

事实层面：

- 外层模型**只接收文本**（候选文本、声学摘要、大纲、历史），从不接收音频或视频；
- 因此"VL"部分完全未被使用，它实际上是把它当作一个 **Qwen3-30B-A3B 文本模型**在用；
- 这不构成方法上的错误（文本能力足够），但有两个真实代价：
  1. **无法直接听音乐**：所有音乐信息都经过"专家候选 + 10 秒块声学摘要"两层中介，模型没有独立核对音乐的能力；
  2. **命名与叙事容易误导**：论文里若写"多模态规划器"会被审稿人质疑，应如实写成"文本规划器（Qwen3-VL-30B 的文本接口）"。

更合理的替代（按代价排序）：
- 用同尺寸的**纯文本**模型（如 Qwen3-30B-A3B 文本版），可去掉误导；
- 用**能听音频**的模型（Qwen3-Omni-30B）做外层，让规划器直接接触音乐——但需要重新验证解码与稳定性；
- 保持现状但明确披露：音乐理解来自冻结专家与确定性声学摘要，规划器只做结构化写作。

当前实验保持第 3 种，并在所有文档中写明"文本模式、不直接听音"。
