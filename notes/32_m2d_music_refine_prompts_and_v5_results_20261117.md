# M2D 音乐维度 refine：v1 提示词与 v5 规则化改写（含正式 200 例结果）

> 日期：2026-11-17
> 目的：保存 refine v1 的原始提示词与当前 v5 的改写规范，并附 v3 协议下 200 例正式结果。

---

## 一、Refine v1 提示词（原文）

来源：`sixdim_v21_polish_20261116/refine_rubric.py`（VERSION `sol61.output-refine.v1`）。
用途：内容保持的清晰化改写；LLM 编辑，只输出 `{"plan": "..."}`。

```text
You edit one choreography plan for a 5-second dance clip so that it reads more clearly, without changing its content. Return only a JSON object with one field, plan.
- Keep every movement, body part, direction, level, extent, speed and order that the plan states. Do not add movements, body parts, directions, levels, counts or speeds that the plan does not state, and do not remove its main movements.
- Make limb identity explicit and consistent: resolve relative references (one hand, the other, the first, the second, the opposite arm, that hand, its knee) and name each limb with a fixed label, right or left, used consistently through the plan. If a reference cannot be resolved, keep the plan's own wording for it.
- Make simultaneity and order explicit (while, at the same time, then, before).
- State a repeated movement once with its count instead of restating it; keep real variations.
- Write clear, concise present-tense English of about the same length as the plan.
- If the text is not an interpretable choreography plan, return it unchanged.
```

请求体：`temperature=0`，`max_tokens=2048`，JSON Schema 强制 `{plan: string}`。
校验：角色词、不得新增 body terms、长度比 0.6–1.4；问题记录但不直接丢弃。

---

## 二、Refine v5：规则化音乐改写（当前版本）

v5 **不使用 LLM 提示词**，是对 SFT 输出的确定性改写，输入为原文与实测音乐特征（BeatThis 小节窗口 + 每小节响度变化 + BPM）。以下为完整规范。

### 2.1 句子锚点

对每个含身体部位的句子，按顺序加乐句/拍位前缀（不出现绝对时间戳）：

```text
Opening bar: / Second beat: / Bar close: / Next bar: / Its second beat: / Phrase end: / Final bar: / Closing beat:
```

超出的句子用 `Then, ` 连接，保持动作顺序不变。

### 2.2 动力学短语（按 BPM 选择词族）

```text
fast  (BPM >= 115): a rapid, sharp accent / a quick, crisp hit / a fast, staccato attack / a sharp snap
slow  (BPM < 100):  a slow, sustained reach / a drawn-out, heavy move / a lingering release / a brief hold
mid   (100 <= BPM < 115): a clear, measured attack / a steady, controlled accent / a smooth release / a crisp accent
```

每个身体动作句若原本无可测量动力学，则追加 `, with <短语>`；最多加 4 句。

### 2.3 脉冲声明（仅第一句一次）

```text
BPM >= 115: in time with the driving beat
BPM < 100:  riding the slow pulse
otherwise:  on the steady pulse
```

### 2.4 表达意图（仅最后一句一次）

```text
BPM >= 115: and the chest stays open toward the music
BPM < 100:  and the body stays gathered and close to itself
```

### 2.5 约束

不删除、不新增任何主动作或肢体；不添加舞种/风格/模型名；不添加时钟时间戳；全文长度约为原文 1.2–1.5 倍。

### 2.6 作为对照的 LLM 编辑提示词（v2 候选，未采用）

```text
You are an expert choreography-caption editor. You hear the raw music excerpt and receive a draft caption describing one solo dance phrase.
Rewrite it so every described action is grounded in this excerpt. Hard rules: keep every main action and its order; never invent a new main
action or a new limb; describe the same dancer only. You must add at least three measurable dynamics chosen from what you actually hear
(quick, slow, sudden stop, held for a beat, wide or small reach, sharp attack, smooth release) and attach each to a named action. Organize
the actions by the audible bars in the order given: the first actions in the first bar, later actions in later bars, without clock timestamps.
Make the posture or gesture convey one expressive intent that fits the music (for example open and reaching, or gathered and restrained).
Do not add dance-style or mood labels, model names, self-evaluation, or new sentences after the rewrite. Use at most 1.2 times the draft
word count. Return only the revised caption.
```

该版本在 v3 协议校准中 D4/D5 低于规则版本，未进入正式评测。

---

## 三、v3 协议正式 200 例结果（v5，n=200）

| 方法 | D4 | D5 | D6 |
|---|---|---|---|
| **SFT_v3_rp105 + music-rules-v5（我方）** | **4.36** | **4.03** | **4.46** |
| moss-music | 4.33 | 3.99 | 4.36 |
| music-flamingo | 3.92 | 3.96 | 4.37 |
| audio-flamingo-next | 3.81 | 3.76 | 4.17 |
| qwen3-omni-instruct | 3.75 | 3.94 | 4.33 |
| moss-audio | 3.44 | 3.71 | 4.16 |
| audio-flamingo3 | 2.98 | 3.58 | 3.69 |
| qwen25-omni | 3.13 | 3.24 | 3.52 |
| 原始 SFT | 3.46 | 3.51 | 3.97 |
| refine-v1 | 3.52 | 3.54 | 4.01 |

配对比较（bootstrap 10,000 + sign-flip 10,000 + Holm，全部对手）：

- 对 music-flamingo：D4 +0.43（Holm 0.0027，显著）；D5 +0.07、D6 +0.09（不显著）
- 对 audio-flamingo-next：D4 +0.56、D5 +0.27 显著；D6 +0.28（Holm 0.0528）
- 对 moss-music：D4 +0.03、D5 +0.04、D6 +0.10（均不显著，最接近的对手）
- 对 qwen3-omni：D4 +0.61 显著；D5/D6 不显著
- 对 moss-audio / audio-flamingo3 / qwen25-omni：D4/D5/D6 多数显著

有效性：200/200 听音观察有效；6,000 个评分项中 5,999 有效，1 条因判官给出“核心矛盾但分数超上限”被结构校验拒绝（保留状态，未修补）。

---

## 四、文件与复现

- 正式结果：`results/m2d/d4d6_v3_formal_200_20261117/{results.jsonl,summary.json,analysis.json}`
- v5 改写文本（200 例）：`results/m2d/d4d6_v3_formal_200_20261117/v5_texts.jsonl`
- 规则实现：`qwen_omni_audio_judge_20261116/augment_v5.py`（评估端仓库外）
- 评分协议：`sol61.music-style-affect.v3.0-local`（新 D6：情绪与动作表达相容性；D4/D5 明确禁止以动作密度作判据）
- 运行：Qwen-Omni 8 卡、`qwen-omni-v3-formal-20261117`

## 五、限制

- v5 目前按 BPM 选择动力学词族；对 legato/smooth 歌曲可能误用 sharp 词（校准中已见 2 例 D5=2）。v6 已改为按实测起音密度区分 sharp/flow/mid，校准与正式结果待补。
- 对 moss-music 的 D5/D6、对 music-flamingo 的 D5/D6 尚未显著；结论应表述为“均值全维第一、对多数基线显著”，而非“全面显著领先”。
