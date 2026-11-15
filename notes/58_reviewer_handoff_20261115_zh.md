# 人工评审交接说明（唯一剩余缺口）

## 用哪个包

`artifacts/sr-dev/human_pack_ext8/index.html` —— **优先用这个**。

- 8 首**全新留出**歌曲（另 1 首因基线失败不可配对）（FMA Electronic，艺人互不重复，从未参与任何开发）
- 两臂：基线 workflow vs 我们的方法（声学引导 + 保语汇复读重写）
- 主题编号已隐藏、动作文本完整不截断、位次按种子随机（O 在 A 位 3 首、B 位 3 首）
- 每人约 20 分钟；打开 HTML 即可，音频内嵌同目录 `audio/`

填完点“生成结果 JSON”，把内容存成 `answers_<名字>.json`。

## 怎么分析

```bash
python experiments/structured_dev_20261114/human_analyze_compact.py \
  --bindings /ssd/2/niezichao.nzc/artifacts/sr-dev/human_pack_ext8/bindings.private.json \
  --answers answers_a.json --answers answers_b.json \
  --out /ssd/2/niezichao.nzc/artifacts/sr-dev/human_analysis_ext6.json
```

输出：每位评审的胜负数、逐维度票数、两人一致性（Cohen's κ）与逐条分歧。弃权不会被插补。

## 需要几人

至少 2 人（一位舞蹈/MIR 背景更佳）。单人可以看方向，但无法区分"评审偏好"与"方法效果"。

## 已有结果（供对照，不要给评审看）

| 来源 | 结果 |
|---|---|
| Qwen3-VL-8B（隐藏标签、换序） | +6.00/20，CI [5.12, 6.88]，**8–0** |
| Qwen3-VL-30B | +4.44/20，CI [3.38, 5.56]，**8–0** |
| 作者侧独立阅读 | 6/6 偏好我们的方法 |

评审结果若与上述方向一致，即可支撑论文中的内容质量主张；若不一致，应以人工结果为准并重新表述结论。
