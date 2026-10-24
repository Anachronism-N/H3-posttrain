# 规划器短规划与分段 loss 实验

这是独立的规划器实验入口，不改变 `music_planner_train.py`、公平比较的标签、原生音频输入或冻结的基线协议。实验仅输入音乐，学习“简短音乐证据 → 简短编舞安排 → 最终动作 caption”。没有加入视频渲染、RL 或自动生成 CoT 真值，也没有为当前数据补造舞种。

当前交付是实现和 CPU 可执行验证，尚未训练真实 MOSS / Music Flamingo / Qwen3-Omni Captioner 权重，不能据此声称编舞质量提高。CPU 测试使用真实 PyTorch 梯度及 mock 音频 processor；不等同于三个真实原生 processor 的验收。

## 数据与人工审核

保留公平基线现有 `train.jsonl` / `val.jsonl` 每行的全部字段：`id`、`source_group_id`、`split`、`audio_path`、`audio_sha256`、`audio_window_seconds`、`target_text`、`target_windows`、`caption_version`、`semantic_approval: false`、`human_review_completed: false`。仍是一条完整 5.0625 秒音乐对应两段动作文本；不能把物理裁切的 2 秒音频放入此入口。

在每行额外添加下面两个字段。下面内容是 schema 示例，不能直接当作实际歌曲标注；SHA、证据、计划和审核人都必须换成真实记录。

```json
{
  "music_evidence": {
    "text": "前段打击乐较疏，后段打击乐密度和能量上升。",
    "provenance": {
      "source": "human_audio_review",
      "human_approved": true,
      "reviewer": "真实审核人或审核记录编号",
      "version": "music-evidence-v1",
      "input_modalities": ["audio"],
      "audio_sha256": "与该行audio_sha256相同的64位小写SHA256"
    }
  },
  "short_plan": {
    "text": "前段采用较收敛的运动，后段增大动作幅度，并保留可自然衔接的结束姿态。",
    "provenance": {
      "source": "human_choreography_plan",
      "human_approved": true,
      "reviewer": "真实审核人或审核记录编号",
      "version": "short-plan-v1",
      "audio_sha256": "与该行audio_sha256相同的64位小写SHA256",
      "target_caption_sha256": "SHA256(target_text.strip().encode('utf-8'))"
    }
  }
}
```

音乐证据的来源白名单是 `human_audio_review`、`human_reviewed_audio_teacher`、`human_reviewed_mir`，每种都要求明确的人工认可、审核人及版本。`input_modalities` 必须恰好是 `["audio"]`。不能给音乐证据生成器输入目标视频或 MotionScript，再称作音乐观察。

短计划的来源白名单是 `human_choreography_plan`、`human_reviewed_teacher_plan`。计划可以总结训练动作安排，但应承认它是一种编舞选择，不得声称音乐唯一决定左右手或实际视频动作。其审核同时绑定音乐 SHA 与最终 caption SHA；更换动作 target 后需要重新审核计划。

这里的 `human_approved` 只针对新增音乐证据/短计划，不把原有弱动作标签自动升级为完整的语义人工验收，所以原有 `semantic_approval` 和 `human_review_completed` 仍保持 false。程序核验的是来源声明与绑定，不会代替听音乐、检查目标动作的审核。

缺字段、缺审核、来源不在白名单、音乐证据声明视频输入、SHA 不匹配、NaN/Infinity、保留的对话/分段标签都会明确失败。入口不会自动填充这些内容。训练和调参只接受 train/val；检查 ID、source_group、精确音频 SHA 和音频路径跨 split 不重叠。歌曲近重复隔离仍需数据侧另做，现有检查不会把它宣称为已完成。

## 目标格式和 loss

assistant 的完整训练目标是：

```text
<music_evidence>
简短、可核验的音乐观察。
</music_evidence>
<short_plan>
简短的编舞选择与段落安排。
</short_plan>
<caption>
[0-2s] Steps left while lifting both arms. [2-5.062s] Lowers arms and turns right.
</caption>
```

最终对外提供的是 `<caption>` 中的自然语言文本。前两个部分是待实验证实是否有益的中间输出，不能把它们当作可信度保证。

目标全文只分词一次，再用 tokenizer 的字符 offset mapping 分配 token 所属段。不会把三段分别 encode 后拼接，也不会假设这种拼接与全文分词等价。跨边界 token 归其最后字符所在段；标签、段间空白也受监督。EOS 是唯一的原生 `<|im_end|>`，追加在完整目标末尾并归 caption 段。输入 prefix 的 label 为 -100，分段编号为 -1，均不计 loss。支持者必须提供准确 offset mapping，缺少该能力时明确拒绝。

每个源样本的优化目标是：

```text
L = (w_music × mean_NLL_music
   + w_plan × mean_NLL_plan
   + w_caption × mean_NLL_caption)
    / (w_music + w_plan + w_caption)
```

默认相对权重为 `0.2 0.3 1.0`，只是可运行的实验起点。每段先按自己的有效 token 数取均值，再按权重归一；这与把所有目标 token 一起取均值不同。`--accum` 内按源样本平均。验证同时记录每段 token NLL、有效 token 数和加权目标，`best_val_nll` 使用最终 caption 段 NLL；这仍只是参考文本似然，不是编舞质量分数，也不能跨 tokenizer 直接比较。

## 检查与训练

从仓库根目录执行，使用具备 torch/numpy/scipy/soundfile 的 Python。真实模型训练还需要与原生 adapter 对应的 transformers/PEFT 环境、完整本地模型权重及 processor。不会下载模型。

```bash
python -X utf8 -m scripts_omnidance.exp.ms.music_planner_cot train \
  --family moss --model /models/MOSS-Music-8B-Instruct \
  --train /data/short-plan/train.jsonl --val /data/short-plan/val.jsonl \
  --out /runs/moss-short-plan-v1 --check-data-only
```

检查不创建输出目录，不加载模型；它检查 schema/来源绑定，真正消费音频时才核验文件 SHA 与原生处理后的长度。

```bash
python -X utf8 -m scripts_omnidance.exp.ms.music_planner_cot train \
  --family moss --model /models/MOSS-Music-8B-Instruct \
  --train /data/short-plan/train.jsonl --val /data/short-plan/val.jsonl \
  --out /runs/moss-short-plan-v1 --device cuda:0 --dtype bfloat16 \
  --epochs 1 --accum 4 --rank 32 --alpha 64 \
  --segment-weights 0.2 0.3 1.0 --save-every 100 --max-updates 2 --verify-reload
```

这是单设备实验，microbatch 固定为 1。默认开启现有模型的 gradient checkpointing，LoRA 和 audio bridge 的可训练范围复用现有原生 loader，trainable 保持 FP32。它没有实现多卡/DDP/ZeRO3，`WORLD_SIZE > 1` 会失败；不要用 torchrun 启动，也不要把此入口的训练成本或数值当成冻结公平实验的直接续跑。

去掉 `--max-updates` 可执行设定 epoch 预算。正式训练前先做两个更新的 smoke 并检查生成。恢复时保留原配置，只提高更新上限或去掉上限：

```bash
python -X utf8 -m scripts_omnidance.exp.ms.music_planner_cot train \
  --family moss --model /models/MOSS-Music-8B-Instruct \
  --train /data/short-plan/train.jsonl --val /data/short-plan/val.jsonl \
  --out /runs/moss-short-plan-v1 --device cuda:0 --dtype bfloat16 \
  --epochs 1 --accum 4 --rank 32 --alpha 64 \
  --segment-weights 0.2 0.3 1.0 --save-every 100 \
  --resume /runs/moss-short-plan-v1/checkpoint-0000002
```

检查点复用现有 train 的 trainable/AdamW/scheduler/RNG/游标保存和严格恢复。新 run fingerprint 包含新增模块、原 train、原 adapters、原 eval 四份源码 SHA、数据 manifest SHA、模型身份、loss 权重和训练配置。数据/来源/权重/关键配置变化后拒绝续跑。模型权重身份默认沿用基线的 stat+config/processor SHA；这不是完整权重内容 SHA 的保证，可通过 `--weights-manifest` 绑定已有权重清单。

## 生成与提取最终 caption

推理 JSONL 只需要音频字段，完全不需要金标音乐证据、计划或 target：

```json
{"id":"val-source-001","split":"val","audio_path":"/data/audio/001.wav","audio_sha256":"真实音频文件的64位小写SHA256","audio_window_seconds":[0.0,5.0625]}
```

```bash
python -X utf8 -m scripts_omnidance.exp.ms.music_planner_cot generate \
  --family moss --model /models/MOSS-Music-8B-Instruct \
  --checkpoint /runs/moss-short-plan-v1/checkpoint-0000002 \
  --input /data/short-plan/val_audio_only.jsonl \
  --out /runs/moss-short-plan-v1/val_generations.jsonl \
  --device cuda:0 --dtype bfloat16 --max-new-tokens 2048
```

必须提供完整实验 checkpoint 目录及其父目录的 run_config，程序核验完成标记、portable trainable SHA、adapter 导出全部文件的 SHA/成员、当前四份实现源码 SHA、实验协议与基座 config/processor/权重成员身份，再恢复 `ShortPlanAdapter`，不会退回原始无 CoT 协议。源码变化后须使用该 checkpoint 对应的原实验代码，不能静默换预处理。`moss` 与 `music_flamingo` 的任务提示在训练/推理一致；`omni_captioner` 始终保留原生 audio-only user prefix，不新增 system 或文本指令，分段输出格式通过 assistant SFT 学习。

生成路径只挑选音频路径/SHA/时间范围作为输入，不读取 target、音乐证据、短计划或舞种。输出保留原始文本、token、EOS/停止原因、独立 `caption` 字段。仅在三段完整、最终两段时间窗正确且原生 EOS 结束时标记 `caption_valid: true`；否则保留 raw 输出、给出错误并使 caption 为 null，不偷偷补全文本或时间窗。输入仍只接受 train/val，test 留给独立终测。

## CPU 验收

```bash
python -X utf8 -m pytest tests/test_music_planner_cot.py -q
```

测试包含会在分段边界合并 token 的 tokenizer fixture、因果移位/分段 loss/EOS 的真实梯度、三种 family 的 mock prefix、生成不得读取金标、超长拒绝、错窗/缺窗拒绝、真实训练入口两次更新、checkpoint 读取及第三次恢复更新。它验证实现和状态，不证明底座能力、实际音乐理解或手部细粒度编舞表现。

建议先与同数据的直接 SFT 比较，再考虑增加 RL：使用相同音乐和相近生成预算，评估音乐依据正确性、计划/最终动作一致性、时间完整性、重复/矛盾率及人工偏好。同一音乐允许多份合理编舞；仅因不同于原视频动作不能判错。文字“踩拍”也不能当作实际动作对齐的验证。
