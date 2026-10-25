# 规划器候选搜索与 RSI 数据准入

本工具只处理“原始音乐 → 舞种／动作编舞文本”的规划器。它实现有限批次的随机候选生成、外部音乐评审请求、评审导入、训练数据暂存与复核；不执行模型训练、不自动修改权重、不创建在线 RL 循环。

对应模块：`scripts_omnidance.exp.ms.music_planner_search`。

## 已实现的边界

- 三族模型 `moss`、`music_flamingo`、`omni_captioner` 均复用现有 adapter 的原生 `prefix()`、音频 SHA 校验、完整 5.0625 秒波形检查、预处理和上下文长度限制。
- 实际采样路径应用上游 MOSS KV-cache 兼容补丁，以及缺失原生 EOS 时的 assistant EOS bridge。补丁修改当前模型实例的推理 hook／调用参数；不改训练权重、原生前缀、音频或公平基线源码。每个候选记录实际 requested/applied source ref、SHA 及无需补丁的原因。
- 独立采样函数明确使用 `do_sample=True`、temperature/top-p/top-k；每个源生成 2–32 个候选。没有调用原本的 greedy `generate()` 后仅更换 seed。原公平基线接口及解码设置保持原样。
- 每个候选保留完整解码文本、包含特殊 token 的解码文本、全部新增 token、EOS、停止原因、预算截断标志、采样参数、seed、原生前缀哈希、音频 SHA 与长度、模型内容身份。
- 不裁切输入音乐，沿用当前 `[0-2s]`、`[2-5.062s]` 两个文本窗口。采样不会自动改变模型的任务前缀、补训舞种能力或适配新 caption 契约。Captioner 继续使用其原生音频前缀。
- 格式检查复用 `music_planner_eval.parse_windows`。词数、动作类别数、重复率等仅是文本诊断，不作为音乐适配 reward，不凭这些诊断选“最佳编舞”。
- 没有独立评审时只有待审候选，不产生优选结果。独立评审不能证明实际人体执行或逐帧卡点。

采样允许产生重复方案，工具不伪造差异。完全相同的 chosen/rejected 文本不会导出为偏好对；最高分批准方案并列时保留并列状态，不按 hash 随意选一个。

候选池现使用 `music_planner_candidate_pool_v2`。旧 v1 缺少上述实际推理补丁证据，须重新生成并送审，不能修改 schema 字符串复用旧批准。若生成越过第一个 EOS 后又在预算尾部输出 EOS，原始 token 全保留，标记 `tokens_after_eos` 并阻断准入，不清理文本来伪造自然结束。

## 从当前 fair 数据开始

在仓库根目录执行。路径示例指向已有本地数据和权重；工具不下载模型、不启动调度器。

```bash
python -m scripts_omnidance.exp.ms.music_planner_search prepare \
  --data /path/to/music_planner_fixed2s_baseline \
  --split train --limit 8 --out-dir /path/to/search_input_v1
```

该目录必须包含现有 `train.jsonl`、`val.jsonl`、`test.jsonl`。命令产生：

- `selected_rows.json`：仅选定 train 音频及身份，不复制目标 caption、舞种、参考图或视频。`--split dev` 对应原始 `val`。
- `isolation.json`：三 split 全量成员清单，用原始 `source_group_id`、精确音频 SHA 检查交叉；歌曲和近重复状态仍是 `unknown`，缺失 group ID 保留 `null`。

`prepare` 不猜测歌曲归属，不把不同 row ID 当成不同歌曲。读取 test 身份只为检查隔离，不读取其动作作为搜索信息，也不生成 test 候选。

当前歌曲／近重复关系未知时，**可以生成候选和送审**。这种候选即使获批，仍不能进入训练导出；须完成独立歌曲／近重复审计后，更新清单、重新绑定候选与评审。不能把 `unknown` 改成 `verified` 来绕过审计。

```bash
python -m scripts_omnidance.exp.ms.music_planner_search generate \
  --rows /path/to/search_input_v1/selected_rows.json \
  --isolation /path/to/search_input_v1/isolation.json \
  --family moss --model /path/to/MOSS-Music-8B-Instruct \
  --checkpoint /path/to/completed_planner_checkpoint \
  --model-id OpenMOSS-Team/MOSS-Music-8B-Instruct \
  --base-model-id OpenMOSS-Team/MOSS-Music-8B-Instruct \
  --model-version planner-checkpoint-100 \
  --count 4 --seed 42 --temperature 0.9 --top-p 0.95 --top-k 50 \
  --max-new-tokens 2048 --max-length 8192 --device cuda:0 \
  --out /path/to/candidate_pool_v2.json
```

没有微调 checkpoint 时省略 `--checkpoint`。另外两族改变 family、模型路径和真实模型身份即可；不要把模型别名用于掩盖相同基座。调用会读取并哈希实际本地模型顶层文件及 checkpoint adapter 文件，可能耗时；`model_sha256` 是该文件内容身份清单的规范哈希，不是模型名称的哈希。

模型加载后还会检查真实波形的单声道、有限值、完整时长及字节身份。已合入上游严格时长检查，不再允许旧的 24 kHz 五帧过长例外。上下文不足则报错，不截断音乐或前缀。`--device` 只选择已有设备，不分配服务器资源。CPU 测试使用微型 mock，不代表实际大模型已经运行。

所有输出默认拒绝覆盖；新一轮使用新路径。

## 分组清单契约

`isolation.json` 使用 `music_planner_isolation_v1`：

```json
{
  "schema": "music_planner_isolation_v1",
  "version": "audit-version",
  "verification": {
    "source_group_id": {"status": "verified", "evidence_sha256": "真实审计证据的64位小写SHA256"},
    "song_group_id": {"status": "unknown", "evidence_sha256": null},
    "near_duplicate_group_id": {"status": "unknown", "evidence_sha256": null}
  },
  "members": [
    {
      "id": "source-id",
      "split": "train",
      "audio_sha256": "原始音频的64位小写SHA256",
      "source_group_id": "已有的源分组",
      "song_group_id": null,
      "near_duplicate_group_id": null
    }
  ]
}
```

上方是字段说明，不是可直接用于审批的伪证据。真实清单必须包含 train/dev/test 全量成员。已知 source、歌曲、近重复 group 或精确音频 SHA 不能跨 split；未知 group 可以为 null。所有分组状态 verified、所有成员 group 完整、证据 SHA 有效，才允许训练导出。

证据 SHA 是外部分组审计的声明与追溯标识，工具不能仅凭一个哈希证明歌曲分组的语义正确；应保存实际审计产物并独立核验。工具不会自动生成“歌曲已核验”的声明。若更新隔离清单，原 run/request/review 哈希均不再有效，需新一轮生成和送审。

## 固定 rubric 和独立评审

先创建 policy。下面的 Python 示例用于真正保存固定 rubric 的内容哈希：

```python
import hashlib, json
from pathlib import Path
text = (
    "听原始音乐，评估编舞方案的音乐适配、舞种兼容和动作衔接。"
    "每项0至5分；不得按篇幅、稀有动作数、格式或原视频动作一致性给高分。"
    "信息不足标记uncertain。只评音乐和动作文本，不能认证真实执行或逐帧卡点。"
)
policy = {
    "schema": "music_planner_review_policy_v1",
    "rubric": {"id": "planner-music-text", "version": "v1", "text": text,
               "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()},
    "judge": {"kind": "human", "id": "实际评审者ID", "version": "review-protocol-v1",
              "model_id": None, "base_model_id": None, "model_sha256": None},
    "thresholds": {"min_dimension_score": 3.0, "min_confidence": 0.8,
                   "preference_margin": 1.0}
}
Path("review_policy_v1.json").write_text(
    json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
```

阈值是明确的起步配置，应在开发数据和人工校准上确定，不是已验证的最优设置。一次搜索中固定 rubric、judge 和阈值；改变任何一项都需新请求和新评审。

```bash
python -m scripts_omnidance.exp.ms.music_planner_search review-request \
  --pool /path/to/candidate_pool_v2.json --policy review_policy_v1.json \
  --out /path/to/review_request_v1.json
```

请求只含原音频路径／SHA、候选原文、身份与格式是否合格，不含原视频或动作真值。评审可由人工或独立音频模型完成。音频模型的 judge 使用：

```json
{
  "kind": "audio_model",
  "id": "独立音频评审任务ID",
  "version": "固定推理与prompt版本",
  "model_id": "实际独立评审模型ID",
  "base_model_id": "实际基座ID",
  "model_sha256": "实际评审模型文件身份的64位小写SHA256"
}
```

相同 model ID、相同 base model ID 或相同模型内容身份均拒绝作为独立评审，改显示名称不能绕过。纯文本模型不能冒充音频 judge。该工具导入评审，不负责调用外部模型；应保存真实音频输入和评审日志。声明 `original_audio_reviewed=true` 不会自动证明模型确实听过音频，实际执行仍需审计。

评审结果文件的严格结构如下；所有字段都需保留，不接受未知字段：

```python
import json
from pathlib import Path
request = json.loads(Path("/path/to/review_request_v1.json").read_text(encoding="utf-8"))
response = {
    "schema": "music_planner_reviews_v1",
    "request_sha256": request["request_sha256"],
    "judge": request["policy"]["judge"],
    "rubric": request["policy"]["rubric"],
    "reviews": [{
        "candidate_sha256": c["candidate_sha256"],
        "source_id": c["source_id"], "audio_sha256": c["audio_sha256"],
        "original_audio_reviewed": False,
        "verdict": "uncertain",
        "scores": {"music_fit": 0.0, "genre_compatibility": 0.0, "transition_coherence": 0.0},
        "confidence": 0.0, "reason": "未评审；填写真正的音乐依据后再导入",
        "allow_preference_use": False
    } for c in request["candidates"]]
}
Path("reviews_to_complete.json").write_text(
    json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
```

这个未完成模板会被拒绝作为音乐评审。完成真实评审后才把 `original_audio_reviewed` 改为 true，并填写 approve/reject/uncertain、三项分数、置信度及依据。允许平局和不确定；不要求每组一定有胜者。`allow_preference_use` 须由评审者明确批准。

## 推荐使用盲评包

原 `review-request` 用于内部追溯，含候选身份和诊断。对外评审优先使用独立 harness：

```bash
python -m scripts_omnidance.exp.ms.music_planner_search_harness blind-request \
  --pool /path/to/candidate_pool_v2.json --policy review_policy_v1.json \
  --seed 42 --public-out /path/to/blind_public_v1.json \
  --private-out /path/to/blind_private_v1.json
```

只把 public 文件给评审者。public 使用随机密钥派生的 opaque ID 和打乱顺序，保留真实音频引用／SHA、完整波形范围、固定 rubric 和动作原文；没有模型／checkpoint 身份、candidate index、源 split、已有分数或格式是否合格。private 保存反查关系、密钥、原 run／policy／request 身份，不能随评审包发送。

真实 `audio_path` 仍可能含来源名称，故不声称完全遮蔽歌曲来源；工具主要去除模型与排名偏置信息。它不为隐藏文件名而改变音频、伪造 URL 或创建不存在的音频引用。需要更严的来源盲化时，可另做经 SHA 核对的中性音频副本并重新生成候选证据。

盲评响应的顶层为 `music_planner_blind_reviews_v1`，含 public 的 `request_sha256`、约定的 `judge`、同一 `rubric` 和 `reviews` 数组。每条使用以下字段：

```json
{
  "opaque_id": "public里的opaque_id",
  "audio_sha256": "public里audio_ref.sha256",
  "original_audio_reviewed": false,
  "verdict": "uncertain",
  "scores": {"music_fit": 0.0, "genre_compatibility": 0.0, "transition_coherence": 0.0},
  "confidence": 0.0,
  "reason": "未评审；完成真实音乐评审后填写",
  "allow_preference_use": false
}
```

保持未评审状态的模板不能作为完成评审导入。完成后执行：

```bash
python -m scripts_omnidance.exp.ms.music_planner_search_harness import-blind \
  --pool /path/to/candidate_pool_v2.json --policy review_policy_v1.json \
  --public /path/to/blind_public_v1.json --private /path/to/blind_private_v1.json \
  --response /path/to/completed_blind_reviews_v1.json \
  --out /path/to/restored_reviews_v1.json --receipt-out /path/to/blind_import_receipt_v1.json
```

回填逐项核对 public、private、原候选、原音频、rubric 和 judge 版本，再转成原严格 reviews schema；未知／重复 opaque ID、错误音频、文本或 mapping 篡改均拒绝。记录回填 receipt。两个输出路径须不同且不存在，避免只写出 public 后发现 private 已存在而丢失新 mapping。

`restored_reviews_v1.json` 可直接交给下述 export/check。回填不会改变 train/dev/test 准入条件。

## 导入、数据准入和复核

```bash
python -m scripts_omnidance.exp.ms.music_planner_search export \
  --pool /path/to/candidate_pool_v2.json --policy review_policy_v1.json \
  --reviews /path/to/completed_reviews_v1.json --out /path/to/reviewed_staging_v1.json

python -m scripts_omnidance.exp.ms.music_planner_search check \
  --pool /path/to/candidate_pool_v2.json --policy review_policy_v1.json \
  --reviews /path/to/completed_reviews_v1.json --staging /path/to/reviewed_staging_v1.json
```

`check` 会按原 pool、policy、reviews 重新计算整个确定性导出，核对每个文本、来源、split、approval、哈希和分差，而不只检查文件能否解析。只有 pool/policy 时也可执行 check，结果保持待独立评审。

准入条件：

1. 同一个源的全部候选都有有效、严格绑定的独立音乐评审。
2. train 源，且全量 source/song/near-duplicate 分组隔离已核验；dev 只允许 review，test 禁止候选搜索和 RSI。
3. 候选真实 EOS 停止、两个非空文本窗口合法；截断、未知停止、保留 token 不合格。
4. 明确 approve，三项分数及 confidence 全部达到固定门槛；最高分批准方案不能并列。
5. 偏好对还需双方明确允许 preference use、双方置信度合格、无 uncertain、两份文本不同、均通过格式检查、分差达到正 margin。格式失败不拿来充当容易区分的音乐负样本。

输出 `music_planner_reviewed_staging_v1` 包含：

- `pseudo_sft`：每个源最多一份批准方案，标记 `generated_plan`，明确不是原视频观察动作。
- `preference_pairs`：同一原音频的 chosen/rejected、独立评审分差和完整来源，可供后续 DPO 数据准备。
- `decisions`：待审、并列、无合格批准、dev review-only 或分组未核验阻断等具体状态。
- 固定模型、原音频、候选、run、rubric、judge、review 与隔离清单身份。

三项评审分数的算术平均仅用于**已明确获批方案间**的排序；它是外部音乐／文本评审分，不是本工具的格式规则分。原始视频观察标签和生成规划伪标签应分池保存，不能把伪标签写回原 MotionScript 真值。

导出是独立 staging schema，**不能静默送入现有 fair 弱标签训练器**。现有训练入口要求弱标签的 semantic/human approval 为 false；这里保留真实 review approval，不伪装成公平基线数据。本工具没有新增 SFT／DPO trainer，也没有自动更新权重。应先人工审查数据，再由专门的后续训练任务消费。

## 可复现的有限实验

先在固定 train-side 音乐上运行 4 候选，独立评审并人工抽查；dev 用来校准 rubric 和比较策略，不进入训练导出。test 只在策略冻结后走原最终评估，不进入此工具的搜索。

每次固定模型、音频清单、窗口、token budget 和评审，单独改变 temperature/top-p 或候选数；记录批准率、待审／阻断率、重复文本率、音乐适配人工偏好及推理成本。分数提高不等同于已经学会实际舞蹈执行。

生成可复核的 source 级预算与选择报告：

```bash
python -m scripts_omnidance.exp.ms.music_planner_search_harness report \
  --pool /path/to/candidate_pool_v2.json --policy review_policy_v1.json \
  --reviews /path/to/restored_reviews_v1.json --out /path/to/search_report_v1.json
```

省略 `--reviews` 时只记录候选数、生成 token／上限、重复文本数等预算诊断，保持待独立评审。报告分别统计 train 与 dev，保留每个 source 的配对数据；基线固定 candidate index 0，明确称为 **single stochastic sample baseline**，不是原公平实验的 greedy baseline。token 预算只覆盖生成调用，不含 judge、音频处理耗时或真实硬件费用；不同模型 tokenizer 的 token 数不能直接当统一成本单位。

若用同一份评分选择并报告优选，平均分差只称为描述性 selection 结果，存在挑选偏差，**不给泛化置信区间**。格式、词数、动作类别数仍不作为音乐质量分。

若需要独立复核，先固定上述选择，再用第二名独立评审者／独立模型、相同 rubric 对原基线与选定方案评审；按新 policy 再走一次盲评回填。然后：

```bash
python -m scripts_omnidance.exp.ms.music_planner_search_harness report \
  --pool /path/to/candidate_pool_v2.json --policy review_policy_v1.json \
  --reviews /path/to/restored_reviews_v1.json \
  --audit-policy /path/to/independent_audit_policy_v1.json \
  --audit-reviews /path/to/independent_audit_reviews_v1.json \
  --seed 42 --out /path/to/search_independent_audit_v1.json
```

同一个 reviewer ID、同模型／同基座／同模型内容身份的第二评分被拒绝作为独立 audit。独立 audit 在 source 层计算配对差值和 bootstrap，train/dev 分别统计，小样本明确标记；它只描述这些源的独立评审分，不能证明未见歌曲泛化。歌曲／近重复仍 unknown 时，报告显式保留未核验状态。审计可发现选择分升高、独立复核反而下降的情况。常规 reviews 本身不证明执行过盲评，应同时查对应回填 receipt。

训练集与开发集、测试集必须按歌曲及近重复分组隔离。当前 song/near-duplicate unknown 时，先做候选评审实验，得到空训练导出是正确状态；不要为通过门槛填造分组。

CPU 验证：

```bash
python -m pytest -q tests/test_music_planner_search.py
```

测试覆盖实际 torch 随机采样及 RNG 恢复、三族 adapter 前缀路径及完整音频检查（mock processor/model）、实际上游缓存／EOS hook 的调用和 provenance、越过 EOS、音频与评审身份绑定、盲包／回填及篡改拒绝、分 split 的 source 配对报告、独立审计反转选择结果、同模型自评拒绝、平局、不完整评审、截断、heldout 隔离、unknown grouping 阻断训练、确定性导出及 CLI 全流程。没有运行实际预训练大模型或人工音乐质量验收；官方原生模型的真实 HF 解码验证依赖本地模型源码／processor 与 transformers。
