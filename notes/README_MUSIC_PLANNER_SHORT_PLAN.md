# 规划器短规划与直接输出消融实验

这是独立的规划器实验入口，不改变 `music_planner_train.py`、公平比较的标签、原生音频输入或冻结的基线协议。Captioner 的 direct-caption 可以直接使用已有弱动作标签；有经审核中间标注后，再比较“仅最终动作 caption”与“简短音乐证据 → 简短编舞安排 → 最终动作 caption”。没有加入视频渲染、RL 或自动生成 CoT 真值，也没有为当前数据补造舞种。

当前协议为 `music-planner-captioner-ablation-v3/<mode>/<annotation-policy>`。两个模式的训练、续跑、生成都通过 `--mode` 显式选择：

| mode | assistant 目标 | 有效 loss 段 | 推理结果 |
| --- | --- | --- | --- |
| `direct-caption` | 原始两段自然语言 caption＋原生 EOS | 仅 caption，其他段 token 数为 0 | 直接返回自然语言 caption |
| `short-plan`（默认） | 音乐证据＋短计划＋caption＋原生 EOS | 三段各自归一后加权 | 解析完整三段，提取最终 caption |

direct 默认 `weak-caption`，不要求 `music_evidence` / `short_plan`，不会把不用的中间标注塞进 prompt/target，也不会生成中间段。short-plan 必须 `reviewed-shared`；direct 指定 `--require-shared-annotations` 时也采用该策略。这种受审核数据约束的消融需要相同 train/val manifest，不能把更多弱标签的 direct 与较少审核数据的 short-plan 称为等数据实验。schema、审核策略、提示、权重和 checkpoint 身份独立，跨协议加载/续跑明确拒绝；旧 v1/v2 checkpoint 使用旧实验源码，不能在 v3 下静默解释。

当前交付是实现和 CPU 可执行验证，尚未训练真实 MOSS / Music Flamingo / Qwen3-Omni Captioner 权重，不能据此声称编舞质量提高。CPU 测试使用真实 PyTorch 梯度及 mock 音频 processor；不等同于三个真实原生 processor 的验收。

## 数据与人工审核

保留公平基线现有 `train.jsonl` / `val.jsonl` 每行的全部字段：`id`、`source_group_id`、`split`、`audio_path`、`audio_sha256`、`audio_window_seconds`、`target_text`、`target_windows`、`caption_version`、`semantic_approval: false`、`human_review_completed: false`。仍是一条完整 5.0625 秒音乐对应两段动作文本；不能把物理裁切的 2 秒音频放入此入口。

仅 short-plan 或启用 `--require-shared-annotations` 的 direct 需要在每行额外添加下面两个字段。普通 direct 可保留原弱标签 manifest。下面内容是 schema 示例，不能直接当作实际歌曲标注；SHA、证据、计划和审核人都必须换成真实记录。

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

需要中间标注的策略下，缺字段、缺审核、来源不在白名单、音乐证据声明视频输入、SHA 不匹配都会明确失败；所有策略都拒绝 NaN/Infinity 和保留对话/分段标签。入口不会自动填充标注。训练和调参只接受 train/val；检查 ID、source_group、精确音频 SHA 和音频路径跨 split 不重叠。歌曲近重复隔离仍需数据侧另做，现有检查不会把它宣称为已完成。

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

short-plan 目标全文只分词一次，再用 tokenizer 的字符 offset mapping 分配 token 所属段。不会把三段分别 encode 后拼接，也不会假设这种拼接与全文分词等价。跨边界 token 归其最后字符所在段；标签、段间空白也受监督。short-plan 缺少准确 offset mapping 会拒绝；direct 全部 token 均归 caption，直接完整 encode，不强求没有用途的 offset。EOS 是唯一原生 `<|im_end|>`，追加在目标末尾并归 caption 段。输入 prefix 的 label 为 -100，分段编号为 -1，均不计 loss。

每个源样本的优化目标是：

```text
L = (w_music × mean_NLL_music
   + w_plan × mean_NLL_plan
   + w_caption × mean_NLL_caption)
    / (w_music + w_plan + w_caption)
```

short-plan 默认相对权重为 `0.2 0.3 1.0`，只是可运行的实验起点。每段先按自己的有效 token 数取均值，再按权重归一；这与把所有目标 token 一起取均值不同。`--accum` 内按源样本平均。direct-caption 默认使用 `0 0 1`，若显式提供参数则只允许前两项为 0、最后一项为正，loss 退化为最终 caption＋EOS 的平均 NLL。未启用的段报告 token 数 0、NLL null，避免除零或把缺失值报成零损失。

验证的 `weighted_segment_nll` 与训练一致，对每个源样本的加权段均值再平均；`segment_nll` 另按各段全部有效 token 汇总，二者聚合方式在记录中明确区分。`best.json` 和 `best_val_nll` 使用最终 caption 段 NLL；这仍只是参考文本似然，不是编舞质量分数，也不能跨 tokenizer 直接比较。

即使使用同一 tokenizer，short-plan 的 caption NLL 以金标音乐证据和短计划作为 teacher-forced 前文，而 direct-caption 没有这些前文，条件不同，不能直接比较两个数来判断 CoT 是否更好。该指标只用于各模式内部选择 checkpoint；跨模式应比较不读金标前段的自由生成与人工偏好。

每个更新会从实际有效 labels 统计 `supervised_segment_tokens`（三个段）和 `supervised_tokens_total`，checkpoint 游标与 outputs.json 也保留累计计数，续跑不清零。相同源样本曝光不等于相同 token 或算力预算；消融必须同时报告这些 token 数、最大生成预算与耗时，不能仅把 epoch 对齐就称为等成本。

## 检查与训练

从仓库根目录执行，使用具备 torch/numpy/scipy/soundfile 的 Python。真实模型训练还需要与原生 adapter 对应的 transformers/PEFT 环境、完整本地模型权重及 processor。不会下载模型。

```bash
python -X utf8 -m scripts_omnidance.exp.ms.music_planner_cot train \
  --mode short-plan --family moss --model /models/MOSS-Music-8B-Instruct \
  --train /data/short-plan/train.jsonl --val /data/short-plan/val.jsonl \
  --out /runs/moss-short-plan-v3 --check-data-only
```

检查不创建输出目录，不加载模型；它检查 schema/来源绑定，真正消费音频时才核验文件 SHA 与原生处理后的长度。

```bash
python -X utf8 -m scripts_omnidance.exp.ms.music_planner_cot train \
  --mode short-plan --family moss --model /models/MOSS-Music-8B-Instruct \
  --train /data/short-plan/train.jsonl --val /data/short-plan/val.jsonl \
  --out /runs/moss-short-plan-v3 --device cuda:0 --dtype bfloat16 \
  --epochs 1 --accum 4 --rank 32 --alpha 64 \
  --segment-weights 0.2 0.3 1.0 --save-every 100 --max-updates 2 --verify-reload
```

这是单设备实验，microbatch 固定为 1。默认开启现有模型的 gradient checkpointing，LoRA 和 audio bridge 的可训练范围复用现有原生 loader，trainable 保持 FP32。它没有实现多卡/DDP/ZeRO3，`WORLD_SIZE > 1` 会失败；不要用 torchrun 启动，也不要把此入口的训练成本或数值当成冻结公平实验的直接续跑。

去掉 `--max-updates` 可执行设定 epoch 预算。正式训练前先做两个更新的 smoke 并检查生成。恢复时保留原配置，只提高更新上限或去掉上限：

```bash
python -X utf8 -m scripts_omnidance.exp.ms.music_planner_cot train \
  --mode short-plan --family moss --model /models/MOSS-Music-8B-Instruct \
  --train /data/short-plan/train.jsonl --val /data/short-plan/val.jsonl \
  --out /runs/moss-short-plan-v3 --device cuda:0 --dtype bfloat16 \
  --epochs 1 --accum 4 --rank 32 --alpha 64 \
  --segment-weights 0.2 0.3 1.0 --save-every 100 \
  --resume /runs/moss-short-plan-v3/checkpoint-0000002
```

检查点复用现有 train 的 trainable/AdamW/scheduler/RNG/游标保存和严格恢复。新 run fingerprint 包含本模块、分段 loss 模块、原 train、原 adapters、原 eval 五份源码 SHA、数据 manifest SHA、审核策略、模型身份、loss 权重/分块配置和训练配置。数据/来源/权重/关键配置变化后拒绝续跑。模型权重身份默认沿用基线的 stat+config/processor SHA；这不是完整权重内容 SHA 的保证，可通过 `--weights-manifest` 绑定已有权重清单。

以下是同审核数据 direct 消融，使用独立输出目录，训练/续跑保留模式与审核策略，不复用 short-plan checkpoint；省略 `--require-shared-annotations` 则是允许现有弱标签的普通 direct 训练：

```bash
python -X utf8 -m scripts_omnidance.exp.ms.music_planner_cot train \
  --mode direct-caption --require-shared-annotations --family moss --model /models/MOSS-Music-8B-Instruct \
  --train /data/short-plan/train.jsonl --val /data/short-plan/val.jsonl \
  --out /runs/moss-direct-v3 --device cuda:0 --dtype bfloat16 \
  --epochs 1 --accum 4 --rank 32 --alpha 64 \
  --segment-weights 0 0 1 --max-updates 2 --verify-reload

python -X utf8 -m scripts_omnidance.exp.ms.music_planner_cot train \
  --mode direct-caption --require-shared-annotations --family moss --model /models/MOSS-Music-8B-Instruct \
  --train /data/short-plan/train.jsonl --val /data/short-plan/val.jsonl \
  --out /runs/moss-direct-v3 --device cuda:0 --dtype bfloat16 \
  --epochs 1 --accum 4 --rank 32 --alpha 64 --segment-weights 0 0 1 \
  --resume /runs/moss-direct-v3/checkpoint-0000002
```

## 生成与提取最终 caption

推理 JSONL 只需要音频字段，完全不需要金标音乐证据、计划或 target：

```json
{"id":"val-source-001","split":"val","audio_path":"/data/audio/001.wav","audio_sha256":"真实音频文件的64位小写SHA256","audio_window_seconds":[0.0,5.0625]}
```

```bash
python -X utf8 -m scripts_omnidance.exp.ms.music_planner_cot generate \
  --mode short-plan --family moss --model /models/MOSS-Music-8B-Instruct \
  --checkpoint /runs/moss-short-plan-v3/checkpoint-0000002 \
  --input /data/short-plan/val_audio_only.jsonl \
  --out /runs/moss-short-plan-v3/val_generations.jsonl \
  --device cuda:0 --dtype bfloat16 --max-new-tokens 2048
```

必须提供完整实验 checkpoint 目录及其父目录的 run_config，程序核验完成标记、portable trainable SHA、adapter 导出全部文件的 SHA/成员、当前五份实现源码 SHA、实验/审核协议与基座 config/processor/权重成员身份，再恢复 `ShortPlanAdapter`。源码变化后使用对应原实验代码，不能静默换预处理。`moss` 与 `music_flamingo` 的任务提示训练/推理一致；`omni_captioner` 始终保留原生 audio-only user prefix，不新增 system 或文本指令，输出协议通过 assistant SFT 学习。

direct 模式生成把上面命令改为 `--mode direct-caption`、对应 direct checkpoint，并使用另一新输出文件；经审核的 direct checkpoint 还要保留 `--require-shared-annotations`。它直接验证自然语言时间窗，不要求或解析三段标签。

生成路径只挑选音频路径/SHA/时间范围作为输入，不读取 target、音乐证据、短计划或舞种。输出保留原始文本、token、EOS/停止原因、独立 `caption` 字段。short-plan 需三段完整，两个模式都需最终两段时间窗正确且原生 EOS 结束，才标记 `caption_valid: true`；否则保留 raw 输出、给出错误并使 caption 为 null，不偷偷补全文本或时间窗。输入仍只接受 train/val，test 留给独立终测。

完成判定同时检查原始 `new_token_ids`：不超过申请的生成 token 预算，首个原生 EOS 必须恰好是最后一个 token。即使停止原因字符串写着 eos，出现 EOS 后缀或预算超限也会判失败；不会先截断或清理 token 再声称成功。

## 上游原生生成兼容补丁与训练身份

本分支以双亲 merge 同步公平分支至 `236d4221039647bd2a229ba28a211d88eb5ab844`，不修改共享基线源码。真实生成加载默认接入上游 MOSS position buffer/cache 和原生 EOS 三种补丁：MOSS 先核验官方原生源码与 shard，恢复非持久的 FP32 `inv_timescales` buffer，再通过 KV cache 长度恢复单 token cached decode；仅当模型未声明有效 EOS 时，才向原生 generate 提供已训练的 assistant 终止 token。position 恢复有单独的 buffer receipt，不改持久模型参数、prefix、原始输出或解码惩罚。Captioner 不执行 MOSS 补丁，保留 EOS 兼容处理。

`generate_caption` 通过实例 `self.generate` 调用，确保这些 wrapper 实际执行。每条结果的 `generation_compatibility` 保存请求/实际应用的补丁源码路径与 SHA、未修改的训练 fingerprint、EOS 详情；已有有效 EOS 时该项补丁实际应用为 null，仍有请求源码记录。可用 `--no-generation-compatibility` 做明确的原生行为诊断，此时不会偷偷应用补丁。

补丁属于 eval-only 依赖，不加入不可变的训练 `code_sha256`，也不为应用补丁重算或伪造训练 fingerprint。五份训练/预处理核心源码仍须与 checkpoint 完全一致；补丁版本通过每次生成单独的 receipts 记录。旧核心源码 checkpoint 不会因为补丁可用而被当成兼容。CPU 测试调用上游 cache/EOS wrapper，并以独立 spy 核验官方 position 恢复函数已接入；真实 MOSS 权重验收另做。

## Captioner 现有弱标签起跑

不需要先造 CoT。下面使用原有 train/val 弱 caption manifest，输出只有最终自然语言动作文本。Captioner 无 system prompt，不通过提示词切换 short/direct；切换协议需要另行训练对应 checkpoint。

```bash
python -X utf8 -m scripts_omnidance.exp.ms.music_planner_cot train \
  --mode direct-caption --family omni_captioner \
  --model /models/Qwen3-Omni-30B-A3B-Captioner \
  --train /data/weak/train.jsonl --val /data/weak/val.jsonl \
  --out /runs/captioner-direct-v3 --device cuda:0 --dtype bfloat16 \
  --epochs 1 --accum 4 --loss-chunk-size 64 --max-updates 2 --verify-reload
```

仍是单设备实验，不能保证 30B 单卡显存足够，也没有实现或验收本实验的多卡 DS。先完成两个更新、生成与显存测量，再决定训练配置；不会自动启动昂贵实验。

## 可选精确分块 CE

所有路径只选择 assistant 有效位置，再做 FP32 CE，prefix 不参与全词表 FP32 CE。`--loss-chunk-size 64` 进一步把有效位置分块，在每个 chunk 的 checkpoint 重算区内部执行词表 FP32 转换和 CE，反向时逐块重算，不把所有 FP32 块的 CE 图长期保留。0 表示一次处理全部 assistant 有效位置。

数学目标、分段计数、EOS 与梯度保持一致，分块值进入训练指纹，续跑不可静默修改。CPU 有 dense CE 对照的 FP32/BF16 loss 与 logits 梯度测试，及 saved_tensors_hooks 测试确认分块路径不保留所有 FP32 词表块。它仍会接收原生模型完整 logits，不能据此声称消除了模型 logits/骨干/optimizer 的显存，尚无 30B 实测显存收益。

## 不读金标中间段的验证生成审计

每个协议使用自己已训练的 checkpoint，在固定 val 音频上真实自由生成。输入可用前述 audio-only JSONL，也可用完整 val manifest；审计只读取 id/split/音频字段，金标音乐证据、计划、caption 不进入 generation。核验它属于 checkpoint 绑定的 val 集，test 禁止进入。

```bash
python -X utf8 -m scripts_omnidance.exp.ms.music_planner_cot_audit audit \
  --mode direct-caption --family omni_captioner \
  --model /models/Qwen3-Omni-30B-A3B-Captioner \
  --checkpoint /runs/captioner-direct-v3/checkpoint-0000002 \
  --input /data/val_audio_only.jsonl --out /runs/captioner-direct-v3/audit.json \
  --max-new-tokens 2048 --max-length 8192
```

对 short-plan 的独立 checkpoint 运行同样命令，换 `--mode short-plan` 与输出路径。经审核的 direct 还需 `--require-shared-annotations`，与其训练协议一致。

报告保留每条 raw 文本、完整 token、首 EOS/预算判定、最终 caption、raw/final 的重复与时间窗统计、输入/输出 token、耗时、checkpoint 身份和 eval-only 补丁 receipt。没有 teacher-forced NLL，也不输出 CoT 优劣分或动作执行质量分。

```bash
python -X utf8 -m scripts_omnidance.exp.ms.music_planner_cot_audit compare \
  --left /runs/captioner-direct-v3/audit.json \
  --right /runs/captioner-short-v3/audit.json \
  --out /runs/teacher_free_comparison.json --require-shared-annotations
```

比较强制相同 family/基座身份、val 音频面板、上下文与生成预算。严格消融还要求相同审核 train/val manifest、seed/rank/alpha/学习率/accum/epochs/dtype/checkpointing 等训练合同、相同初始化方式和 checkpoint 实际 source_exposures；不匹配则拒绝。去掉严格标志可作描述性比较，但报告明确 `controlled_ablation: false`，不能称为受控 CoT 收益。即使严格合同匹配，short 的目标 token 更多、训练计算仍可能不同，需一起报告 token 与耗时。

## CPU 验收

```bash
python -X utf8 -m pytest tests/test_music_planner_cot.py -q
```

测试包含会在分段边界合并 token 的 tokenizer fixture、因果移位/分段 loss/EOS 的真实梯度、三种 family 的 mock prefix、生成不得读取金标、超长拒绝、错窗/缺窗拒绝、两个模式各自真实训练入口两次更新、checkpoint 读取及第三次恢复更新、CLI 生成、跨模式恢复/加载拒绝、direct 目标零中间段、上游原生 MOSS cached decode 与缺失 EOS wrapper 真实调用。它验证实现和状态，不证明底座能力、实际音乐理解或手部细粒度编舞表现。

建议先与同数据的直接 SFT 比较，再考虑增加 RL：使用相同音乐和相近生成预算，评估音乐依据正确性、计划/最终动作一致性、时间完整性、重复/矛盾率及人工偏好。同一音乐允许多份合理编舞；仅因不同于原视频动作不能判错。文字“踩拍”也不能当作实际动作对齐的验证。
