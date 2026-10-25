# 音乐规划器的舞种决策与不确定性路由

这是独立实验分支 `codex/planner-genre-decision`。已合并公平实验上游 `4e22b8ef8016ee3bdfa85670a9d71aba7d277686`，保留双方历史，不修改上游 `TASK_INSTRUCTION`、训练入口、数据或 fingerprint。条件协议为 `planner-selected-genre-prompt-v1`，新增路由协议为 `planner-genre-uncertainty-route-v1`，结果应与公平基线分别报告。三个方向的执行先后和证据边界参见[实验总指南](../EXPERIMENT_SEQUENCE_20261025.md)。

## 本轮实现与边界

实现链路：原 `PlannerAdapter.prefix` → 仅输入 prefill → 取音频占位 token 的最终 hidden states → masked mean pooling＋MLP → 舞种概率 → 独立 calibration 温度 → 置信/分差路由 → 条件生成或原生无指定舞种 fallback → 连续自然语言 caption。

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

本次上游严格音频处理和实验代码发生变化，旧缓存/旧 head 不能静默用于新在线生成；应重新导出特征并训练对应 head。相互匹配的旧 head＋旧 cache 可离线复核其概率，但不代表新版本生成协议已通过。本工具不会为了兼容而忽略代码 SHA。

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
  --head /runs/genre/mf-v1/genre_head.pt --out /runs/genre/mf-v1/probabilities.jsonl \
  --top-k 3 --min-confidence 0.6 --min-margin 0.15

python -m scripts_omnidance.exp.planner_ideas.genre_decision generate \
  --family music_flamingo --model-path /models/music-flamingo \
  --checkpoint /runs/fair-mf/checkpoint-best --model-id mf-baseSHA-adapterSHA \
  --head /runs/genre/mf-v1/genre_head.pt --sources /data/genre/sources-predict.jsonl \
  --protocol planner-selected-genre-prompt-v1 --max-new-tokens 1024 \
  --top-k 3 --min-confidence 0.6 --min-margin 0.15 \
  --out /runs/genre/mf-v1/conditioned-captions.jsonl
```

`moss` 使用同样接口。`omni_captioner` 可做原生无 prompt 的 prefill 特征导出和舞种头训练；**条件生成显式报 unsupported**。其 audio-only 模板不会因为额外拼接文字就自动支持指令。将来若训练新增风格条件 token，必须另建训练/推理协议并验收，当前未实现。

普通 `predict` 和 `generate` 只接受明确的 train/val/cal/predict split，遇到 test 会拒绝。最终测试先单独准备 test manifest，再显式运行 `predict --final-test` 或 `generate --final-test`；要求 head 中完整记录 train/val/cal 身份，并检查 test 的 ID、source_group_id、exact audio SHA 都不与三集合重叠。冻结路由阈值后才使用 test，不能看测试结果调参。近重复分组仍依赖外部审计。所有预测/生成输出拒绝覆盖，重新运行须使用新路径。

`train` 输出 `genre_head.pt` 和 `training_report.json`，包含验证 NLL、Brier、argmax accuracy、10-bin ECE、calibration 前后指标和数据来源 SHA。软标签 ECE 的期望正确率为标注分布在预测类别上的概率，argmax accuracy 只是诊断。训练只在 CPU 上更新小头；缓存中的 planner 参数保持不变。检查点选择看 val NLL，温度只在 cal 拟合，不使用 test。

## 不确定性路由与原生生成补丁

`predict` 和 `generate` 使用同一个 `decide`：`predicted_genre` 总保留分类最高项，`selected_genre` 只在最高概率达到 `--min-confidence` 且前两项概率差达到 `--min-margin` 时给出。最高项并列时始终 fallback。`--top-k` 只返回候选舞种列表，不会把多个风格同时强塞进动作 prompt；类别不足 k 时返回全部并记录实际数量。

默认 0.6/0.15 是实验起点，**不是已验证的质量门槛**。这里的 confidence 仅指温度缩放后的最高类别概率；拟合过温度不等于真实数据分布上已校准，更不等于动作正确。输出记录概率、top-k、top2 margin、阈值、路由原因，并始终不作语义验收声明。

低置信或分差不足时，`native_unconditioned_fallback` 使用原 `PlannerAdapter._prompt()`；不指定舞种，不给 caption 强加猜测出的 `Dance style:`。这保留原模型的原生音乐输入及动作任务。条件通过时才使用选中风格 prompt，并在结果前统一加入舞种。条件 SFT 的训练接口仍是显式指定舞种的 `conditioned_collate`；native fallback 对应原始前缀，不声称模型已学过新的“未知舞种”token。

`generate` 在实际参与生成的同一个 `RoutedGenrePlanner` 实例上安装上游补丁：MOSS 必须是已准备的原生 `moss_music` 实例，接通其单 token KV-cache decode 修复；所有支持族检查原生 generation EOS，只有缺失/无效时才接通原生 `<|im_end|>` 参数桥。有效 EOS 不改写配置。补丁按实例安装一次，多源条件/fallback 交替不会因重复包装共享模型而失败。输出 `generation_patches` 区分 requested/applied、源码 SHA 和适用性；actual generation 保留上游返回的应用凭据、原始文本、token 和停止原因，并绑定 head 和输入 manifest SHA。补丁不替代模型质量验证，不改变模型权重。

生成还检查第一枚原生 EOS 必须恰在输出末尾，且两段时间窗完整、文字非空。空输出、预算截断、EOS 后继续吐字、错窗/缺窗均标记 `caption_valid=false`，给出 `extraction_error`，`caption_text` 为 null；完整原始文本/token 仍保存在 `action_generation`，不偷偷修补。通过这个门槛只代表格式和完成性，不能证明动作或音乐适配正确。

## 已接通的条件 SFT API

第一轮 `generate` 是已有模型上的新条件 prompt，输出明确记录 `conditioning_sft_performed=false`。它不是已训练条件模型的质量结果。若其指令跟随不足，可在新协议下继续 SFT：

```python
from scripts_omnidance.exp.ms.music_planner_adapters import load_planner_adapter
from scripts_omnidance.exp.planner_ideas.genre_decision import (
    CONDITION_PROTOCOL, RoutedGenrePlanner, conditioned_collate, prepare_generation_patches,
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
generator = RoutedGenrePlanner(base, CONDITION_PROTOCOL)
generator.select_genre("hip hop")
prepare_generation_patches(generator)
prediction = generator.generate(audio_only_row)
```

`conditioned_collate` 和选中舞种时的 `RoutedGenrePlanner.generate` 使用同一个条件 prefix；目标仍为动作文本，CLI 最后统一前置 `Dance style: ...`。不能在训练只用真值风格、推理全用预测风格后假定二者完全一致。后续要测试预测风格混合、条件 dropout，以及分类出错时的动作鲁棒性。只训练独立舞种头而不向 caption 生成器传条件，不会自动提升动作 caption。

## 如何判断有效

先保持同一批来源与解码预算，比较：原公平 planner；舞种头＋条件 prompt；再到条件 SFT。记录舞种头准确性/概率质量、输出风格一致性、动作具体性、跨 chunk 连贯性和音乐节奏适配。按手势舞等稀有类别分层检查。不要以仅舞种正确或文本更长判定编舞改善。本次实验范围仅为规划器。

Jev 借鉴的是“快的概率决策与自由文本生成分工”。[TypeSafe 官方](https://docs.typesafe.ai/models)当前不接受原生音频、不开放客户微调。开放参考 [Laya](https://github.com/NandhaKishorM/laya) 的决策头/校准实现属于独立路线，其 RL 项并无普适优于 CE 的保证。本版先给 CE/soft CE 和[温度校准](https://proceedings.mlr.press/v70/guo17a.html)一个可比较基线。

CPU 验证命令：

```bash
python -m pytest tests/test_planner_genre_decision.py -q
```

除原有梯度/soft CE/校准/保存恢复测试，本轮覆盖置信度和 margin 分别触发 fallback、并列路由、非有限 checkpoint/logits、test 显式准入、拒绝覆盖、真实调用上游 EOS 包装器及 MOSS cache-position 回调、两源条件与 fallback 共用补丁实例。这里仍使用小型接口模型，不是官方大模型或服务器 GPU 实验。
