# 结构化编排与定点修订：新组成方法开发记录

日期：2026-11-14。基线为 `codex/music-choreo-benchmark` 的 `53f65e7`。本轮只新增一个方法 `structured`，不训练、不换专家、不重采候选、不提高预算。**真实开发运行尚未执行**，原因见第 6 节；本文所有"已验证"均指 CPU 合同与状态机，不是编舞质量。

## 1. 为什么只改这一处

v2 已修复重复草稿、停滞与预算浪费，但总纲、motif 和历史只是"提供给模型"，没有变成必须满足的条件。结构合法也不等于音乐贴合、动作可执行。本轮因此只做两件事：

1. 让总纲对每个公开段落给出**可核验的编排意图**，并强制其证据来源真实可用；
2. 让逐块生成拿到**当前段要求 + 上一块最后动作**，并在草稿之后做一次**只针对具体问题**的扫描与修订。

没有新增"记忆""检索""多候选挑优"等概念；证据、权限、时钟和总预算与另外三种方法完全相同。

## 2. 结构化段落设计

新的大纲合同 `choreobench.structured_outline.v1`，每个公开段落必须给出：

| 字段 | 取值 | 作用 |
|---|---|---|
| `primary_motifs` | 已声明 motif 的子集 | 该段主要动作主题 |
| `development` | introduce / repeat / vary / contrast / resolve / unknown | 主题如何发展 |
| `quantity_change` | increase / decrease / steady / unknown | 计划中的动作量变化，**不是**测得的音频能量 |
| `transition_from_previous` | {kind: continue / return / shift / unknown, description} | 与上一段的承接 |
| `evidence` | [{source, note}] | 该判断依据 |
| `uncertainty` | 文本 | 证据不足处显式保留 |

程序强制执行的诚实性约束（`read_outline`）：

- 每个 `evidence.source` 必须属于白名单，且**在本段真实可用**：没有 beat 预测就不能引用 `public_beat_density`，没有候选窗口就不能引用 `local_caption_density`；
- 第一段的 `transition_from_previous.kind` 必须是 `unknown`（没有上一段）；
- `development != unknown` 时 `primary_motifs` 不能为空；
- 任一段落至少一条证据；
- 段落的 `motif_ids` 别名等于 `primary_motifs`，因此共享的块校验器只能引用本段主题。

`quantity_change` 明确标注为设计目标；`local_caption_density` 是"提供了多少条候选文本"，不是音频能量。提示中禁止声称检测到副歌/乐句/节拍同步或测得音频能量。

## 3. 逐块生成

在原有块请求（`_unit_request`，包含候选、公开结构、事件时钟、排期菜单、历史）之上追加：

- `section_requirement`：当前段的完整结构化设计；
- `block_requirement`：`role`（first/middle/last/only）、`required_motifs`、`development`、`quantity_change`、`transition` 与重复说明；
- `previous_block_last_action`：上一块最后一个事件的动作与时间（整块视角，不是最近 8 条全局事件的近似）；
- `revision_policy`：只有四类问题允许修订、每块至多一次、合理主题回归不是缺陷。

`required_motifs` 由确定性规则产生：段落要求 introduce/repeat/vary 时首块必须带主主题；要求 resolve 时末块必须带主主题。

## 4. 定点修订

草稿通过结构校验后，`detect_issues` 只报告四类**具体**问题：

| 代码 | 触发条件 |
|---|---|
| `missing_required_motif` | 本块被要求带主主题，但事件里没有 |
| `action_timing_contradiction` | 动作文本含 ≥3 个分句，时长不足 0.5 s/分句 |
| `missing_transition` | 段落要求 continue/return，但本块起点距上一块结束 ≥1 s |
| `mechanical_repetition` | 与上一块最后动作**逐字相同**，且该段没有要求 repeat |

控制规则：

- **无问题 → 立即提交**，不产生额外模型调用，也不做泛泛反思；
- 有问题 → 在 observation 中列出（最多 3 条，带事件 ID 与具体数值），只允许一次 `revise_block`；修订后不再二次扫描；
- `development == repeat` 时逐字重复**不算**问题（合理主题回归不惩罚）；
- 仍可 `commit_block` 作为逃生口，避免为了修而修；
- v2 的合法同草稿自动提交、完成自动结束、剩余块预留调用全部保留。

## 5. 预算与长曲风险

调用上限仍为 `1 + 4U`，未提高任何一种上限。`structured` 的典型用量是 `1 + U`（无问题直提），最坏 `1 + 2U`。结果中新增：

- `structured_statistics`：检出问题的块数、修订调用数、自动提交数、按当前速率外推的整曲输入 token；
- `token_projection_ratio`：外推总量 / `max_input_tokens`。

两项都是报告，不改变限额；超限仍在派发前拒绝。

## 6. 已验证 / 未验证

已验证（CPU，替代模型响应）：

- 新增 15 项测试：四类问题各自的触发与不触发、重复段落不判缺陷、大纲诚实性 5 个反例、干净草稿零额外调用（1+2 次）、缺主题触发恰好一次定向修订、`structured` 必须搭配 v2；
- 全量回归 **1668 passed, 4 skipped**（跳过项为环境缺可选浏览器）。

未验证：

- **真实三首开发歌的 12 个运行尚未执行。** TX 侧资产（`/opt/music_action_runs/...`、`/apdcephfs_gy4/share_302533218/cedricnie/choreobench_runs/...`）与 Qwen3-VL-32B 文本服务在本机不可达，且没有该机器的 SSH 通道；本地也没有 MOSS-Music K=2 候选目录。按要求不重采候选、不换专家，因此没有用替代资产伪造对照。
- 因此没有四方法逐歌结果、成本表、盲评结论，也没有"是否值得进入正式 24 首"的证据。

## 7. 资产到位后的执行

```bash
# 1) 12 个运行（workflow=v1，其余=v2；已完成且合同一致的格子自动复用）
python experiments/structured_dev_20261114/run_matrix.py \
  --dev-root /path/to/dev_catalog --song-id SONG1 --song-id SONG2 --song-id SONG3 \
  --provider /path/to/provider.json --out-root /path/to/runs/structured_dev_v1

# 2) 逐歌成本/失败表 + 匿名成对包（A/B 顺序按对交换，方法映射写入 private）
python experiments/structured_dev_20261114/report.py \
  --out-root /path/to/runs/structured_dev_v1 \
  --song-id SONG1 --song-id SONG2 --song-id SONG3 --out /path/to/report

# 3) 单独运行新方法
python -m choreobench run-composition --source VIEW.json --song-id SONG \
  --provider provider.json --out-dir runs/SONG/structured --method structured \
  --controller-version v2 --seed 42 --max-request-characters 300000 \
  --max-input-tokens 2000000 --max-total-output-tokens 150000
```

`report.py` 输出 `report.json`、`review-pairs.public.json`、`review-pairs.private.json`；公开包只含匿名计划与评分模板，方法名不外泄。评审要求逐条引用事件 ID 与真实动作文本，音乐判断只能依据随包的独立音乐证据。

## 8. 进入正式 24 首的条件

先看 12 格是否全部完整交付、调用与 token 是否不超过同歌 workflow 的合理倍数、`blocks_with_detected_issues` 的分布是否说明检测器在真实输出上有效（若为 0，说明约束没有生效）。质量上只采用匿名成对比较：音乐贴合、主题发展、动作可执行性、边界衔接四项，A/B 交换；不使用已饱和的长程 5/5 分作为主判据。没有可辨收益就如实保留，不扩大正式实验。

代码入口：[composition_structured.py](../choreobench/composition_structured.py)、[composition_workflows.py](../choreobench/composition_workflows.py)、[CLI](../choreobench/cli.py)、[矩阵脚本](../experiments/structured_dev_20261114/run_matrix.py)、[报告脚本](../experiments/structured_dev_20261114/report.py)。
