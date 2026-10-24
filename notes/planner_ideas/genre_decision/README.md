# 音乐规划器的舞种决策分支 v1

这是独立实验分支 `codex/planner-genre-decision`。继承公平实验代码，但不修改它的 `TASK_INSTRUCTION`、训练入口、数据或 fingerprint。新增协议为 `planner-selected-genre-prompt-v1`，其结果应与公平基线分别报告。

## 本轮实现与边界

实现链路：原 `PlannerAdapter.prefix` → 仅输入 prefill → 取音频占位 token 的最终 hidden states → masked mean pooling＋MLP → 舞种概率 → 独立 calibration 温度 → 选中舞种进入动作生成 prompt → 连续自然语言 caption。

这些特征是 **decoder 上下文化的音频 token 表示**，含模型原生输入上下文，不是裸音乐 encoder 输出。导出调用不接触答案、真值舞种、视频；不会对 teacher-forced caption 做池化。MOSS/Music Flamingo 有原生任务指令，Captioner 保持原生 audio-only 输入。

当前 CPU 合成数据测试验证代码、梯度、数据隔离与条件接线。**尚未执行真实预训练模型/GPU 导出、真实舞种头训练、条件 SFT 或规划器质量验收。** 双向全局音乐 encoder、Laya/RLCD loss、动态舞种语义匹配仍是后续对照；本轮不冒称复现 Jev 私有算法。

## 数据必须先准备

现有公平动作数据没有舞种字段。这里不会从动作 caption 猜造舞种，必须提供独立真实舞种标注及来源。音乐类型与舞种应分别定义。一段音乐可适配多种舞蹈时，可用软分布；若任务是同时出现多个舞种，应另做多标签 BCE，本版互斥 softmax 不适用。

创建 `classes.json`，列表顺序固定，名称会用于条件 prompt：

```json
["hip hop", "jazz", "ballet"]
```

建立实验自己的 source manifests，保留原始完整 5.0625 秒音乐，不修改公平数据原件。一条输入示例：

```json
{"id":"source-a","source_group_id":"song-a","split":"train","audio_path":"/data/a.wav","audio_sha256":"实际音频文件的64位小写SHA256","audio_window_seconds":[0.0,5.0625]}
```

独立标签 `train-labels.jsonl` 中每行使用同一 ID、来源组、音频 SHA、split：

```json
{"id":"source-a","source_group_id":"song-a","audio_sha256":"与source相同的实际SHA256","split":"train","genre":"hip hop","label_provenance":"实际数据集字段或人工标注记录路径","label_version":"dance-style-v1"}
```

软标签以 `"genre_distribution":{"hip hop":0.7,"jazz":0.3}` 替换 `genre`；缺失类别按零处理，总和须为一。标注来源字段是追溯记录，不构成质量认证。

需要互不重叠的 `train`、`val`、`cal`：训练头、挑选 checkpoint、最后拟合温度。按歌曲/原始来源组一次性划分。若从原训练集合划出 calibration，写入新实验 manifest，并记录样本减少；不要重写公平 manifest。代码拒绝三个集合之间的相同 ID、source_group_id 或 exact audio SHA。近重复歌曲/不同编码同曲还需上游按来源组聚类，SHA 检查不能识别全部近重复。`test` 仅最终使用，不能参与模型选择和温度拟合。

## 可执行 CLI

以下从仓库根目录运行；真实模型导出沿用现有 planner 运行依赖。缓存头训练只需 torch、numpy。路径为需替换的真实本地输入。`--model-id` 必须绑定确切基础模型 revision＋adapter checkpoint，而非 `latest`；它是声明式身份，需要操作者确保对应实际权重。导出文件还绑定本实验及原 `music_planner_adapters.py` 代码 SHA，混用 family/model/protocol 的缓存会失败。

```bash
python -m scripts_omnidance.exp.planner_ideas.genre_decision export \
  --family music_flamingo --model-path /models/music-flamingo \
  --checkpoint /runs/fair-mf/checkpoint-best --model-id mf-baseSHA-adapterSHA \
  --sources /data/genre/sources-train-val-cal.jsonl --out-dir /data/genre/features

python -m scripts_omnidance.exp.planner_ideas.genre_decision train \
  --features /data/genre/features/features.jsonl --classes /data/genre/classes.json \
  --train-labels /data/genre/train-labels.jsonl --val-labels /data/genre/val-labels.jsonl \
  --cal-labels /data/genre/cal-labels.jsonl --out-dir /runs/genre/mf-v1 \
  --epochs 20 --batch-size 32 --lr 0.001 --seed 42

python -m scripts_omnidance.exp.planner_ideas.genre_decision predict \
  --features /data/genre/features/features.jsonl \
  --head /runs/genre/mf-v1/genre_head.pt --out /runs/genre/mf-v1/probabilities.jsonl

python -m scripts_omnidance.exp.planner_ideas.genre_decision generate \
  --family music_flamingo --model-path /models/music-flamingo \
  --checkpoint /runs/fair-mf/checkpoint-best --model-id mf-baseSHA-adapterSHA \
  --head /runs/genre/mf-v1/genre_head.pt --sources /data/genre/sources-predict.jsonl \
  --protocol planner-selected-genre-prompt-v1 --max-new-tokens 1024 \
  --out /runs/genre/mf-v1/conditioned-captions.jsonl
```

`moss` 使用同样接口。`omni_captioner` 可做原生无 prompt 的 prefill 特征导出和舞种头训练；**条件生成显式报 unsupported**。其 audio-only 模板不会因为额外拼接文字就自动支持指令。将来若训练新增风格条件 token，必须另建训练/推理协议并验收，当前未实现。

最终测试先单独导出 test source manifest，再运行 `predict --final-test`；这个模式检查 test 与已使用的 train/val/cal 三集合隔离。普通 `predict` 可用于训练集诊断，不等于最终测试。

`train` 输出 `genre_head.pt` 和 `training_report.json`，包含验证 NLL、Brier、argmax accuracy、10-bin ECE、calibration 前后指标和数据来源 SHA。软标签 ECE 的期望正确率为标注分布在预测类别上的概率，argmax accuracy 只是诊断。训练只在 CPU 上更新小头；缓存中的 planner 参数保持不变。检查点选择看 val NLL，温度只在 cal 拟合，不使用 test。

## 已接通的条件 SFT API

第一轮 `generate` 是已有模型上的新条件 prompt，输出明确记录 `conditioning_sft_performed=false`。它不是已训练条件模型的质量结果。若其指令跟随不足，可在新协议下继续 SFT：

```python
from scripts_omnidance.exp.ms.music_planner_adapters import load_planner_adapter
from scripts_omnidance.exp.planner_ideas.genre_decision import (
    CONDITION_PROTOCOL, GenreConditionedPlanner, conditioned_collate,
)

base = load_planner_adapter("music_flamingo", "/models/music-flamingo",
                            device="cuda", lora_rank=32)
# selected_genre 必须来自已连接的真实舞种标注或明确的训练期预测混合策略。
batch = conditioned_collate(base, [row_with_action_target], "hip hop", CONDITION_PROTOCOL)
inputs = base._move(batch)
inputs["labels"] = batch["labels"].to(next(base.model.parameters()).device)
loss = base.model(**inputs).loss
loss.backward()
# 由实验自己的优化器/保存逻辑更新；这段展示接线，不是完整分布式训练入口。
prediction = GenreConditionedPlanner(base, "hip hop", CONDITION_PROTOCOL).generate(audio_only_row)
```

`conditioned_collate` 和 `GenreConditionedPlanner.generate` 使用同一个条件 prefix；目标仍为动作文本，程序最后统一前置 `Dance style: ...`。不能在训练只用真值风格、推理全用预测风格后假定二者完全一致。后续要测试预测风格混合、条件 dropout，以及分类出错时的动作鲁棒性。只训练独立舞种头而不向 caption 生成器传条件，不会自动提升动作 caption。

## 如何判断有效

先保持同一批来源与解码预算，比较：原公平 planner；舞种头＋条件 prompt；再到条件 SFT。记录舞种头准确性/概率质量、输出风格一致性、动作具体性、跨 chunk 连贯性和音乐节奏适配。按手势舞等稀有类别分层检查。不要以仅舞种正确或文本更长判定编舞改善。本次实验范围仅为规划器。

Jev 借鉴的是“快的概率决策与自由文本生成分工”。[TypeSafe 官方](https://docs.typesafe.ai/models)当前不接受原生音频、不开放客户微调。开放参考 [Laya](https://github.com/NandhaKishorM/laya) 的决策头/校准实现属于独立路线，其 RL 项并无普适优于 CE 的保证。本版先给 CE/soft CE 和[温度校准](https://proceedings.mlr.press/v70/guo17a.html)一个可比较基线。

CPU 验证命令：

```bash
python -m pytest tests/test_planner_genre_decision.py -q
```
