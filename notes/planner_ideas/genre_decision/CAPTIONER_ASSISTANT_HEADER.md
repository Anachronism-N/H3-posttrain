# Captioner：在 assistant 侧学习舞种与动作

新入口是 `scripts_omnidance.exp.planner_ideas.captioner_genre`。它保留 Captioner 原生 audio-only user prefix，不添加音乐指令或舞种 prompt。**先训练一个输出“舞种 header＋动作”的联合模型，再从该模型提取特征训练 Jev 思路的独立舞种概率头，最后由概率头决定是否预填 assistant header。** 这是一项新协议，不把原 Captioner 或原 fair checkpoint 当作已经支持风格条件。

总体执行顺序见 [Captioner 优先指南](../CAPTIONER_FIRST_20261025.md)和[三分支实验指南](../EXPERIMENT_SEQUENCE_20261025.md)。本文件只描述此入口实际实现的接口。

## 协议与训练目标

协议名为 `captioner-assistant-style-header-v1`。一条 assistant 训练目标的可读形式是：

```text
Dance style: gesture dance.
[0-2s] ... [2-5.062s] ...
```

真实 token 序列明确规定为 `encode(header) + encode(action) + native_im_end`。**header 和 action 分段编码**；并不假设它等于 `encode(header + action)`。训练的 header 末尾 newline 与条件推理预填的 header token 完全一致；原生音频 user prefix 的任何 token 均不改变。无新增 special token、无词表扩容，也不解冻视觉、talker 或 lm_head。

SFT 对 header 与动作分别计算平均因果 NLL，再按默认权重 `0.25:1.0` 组合；EOS 属于动作段，原生 prefix 不计 loss。两项权重必须为正。训练记录两个分段的累计监督 token、更新数和 source exposure，恢复时继续累计。

## 数据准备

首先划出 joint train、joint val、cal 和最终 test；**cal/test 必须在联合 SFT 之前留出**。动作行沿用现有完整 5.0625 秒音乐＋两个动作文本窗口的 schema。舞种必须来自独立真实标注，不能由动作 caption 猜出或由程序自动补造。每条动作行恰好关联一条同 ID、source_group_id、audio SHA、split 的硬舞种标签：

```json
{"id":"source-a","source_group_id":"song-a","audio_sha256":"实际64位小写SHA256","split":"train","genre":"gesture dance","label_provenance":"实际数据集字段或人工标注记录","label_version":"dance-style-v1"}
```

`classes.json` 是固定的舞种名称列表，例如 `["hip hop", "jazz", "gesture dance"]`。可以使用中文“手势舞”，但标注、训练与推理必须保持同一套名称和顺序；不要把中英文同义名称随意当成两个独立类别。程序没有删去或特殊排除手势舞。

联合训练接受的动作 rows 仍须保留公平弱标签的 `semantic_approval:false`、`human_review_completed:false`；独立舞种来源不会自动升级原动作标注质量。检查标签来源是字符串/版本/身份绑定，并不等于程序完成过语义审查。

程序按 ID、source_group_id、精确音频 SHA 核对 joint train/val 隔离。head `val` 不能来自 joint train，可以使用 joint val 或独立验证源；导出 `cal` 或 `test` 特征时，还核对它们不与 joint train/val 重叠。head 自身的 train/val/cal 隔离由原 `genre_decision train` 检查。歌曲近重复仍需独立审计，代码不会宣称已完成这一步。若初始化 fair 模型曾消费 cal/test，需在数据审计时说明；本入口不能仅凭旧 checkpoint 的 manifest SHA 恢复其全部歌曲归属。

## 可执行训练与恢复

从仓库根目录运行。下方路径应换成实际本地文件；不会下载模型或启动调度器。`check` 检查动作/舞种关联、split 和音频字节，不加载模型。

```bash
python -m scripts_omnidance.exp.planner_ideas.captioner_genre check \
  --train /data/joint/train.jsonl --val /data/joint/val.jsonl \
  --train-labels /data/joint/train-labels.jsonl --val-labels /data/joint/val-labels.jsonl \
  --classes /data/joint/classes.json

python -m scripts_omnidance.exp.planner_ideas.captioner_genre train \
  --model /models/Qwen3-Omni-30B-A3B-Captioner \
  --init-checkpoint /runs/fair-captioner/checkpoint-0000670 \
  --train /data/joint/train.jsonl --val /data/joint/val.jsonl \
  --train-labels /data/joint/train-labels.jsonl --val-labels /data/joint/val-labels.jsonl \
  --classes /data/joint/classes.json --out /runs/captioner-joint-v1 \
  --rank 32 --alpha 64 --epochs 1 --accum 4 --max-updates 2 \
  --device cuda:0 --dtype bfloat16
```

`--init-checkpoint` 是权重初始化：支持已完成的 Captioner fair 或明确的 direct-caption 协议（v2、v3 weak-caption/reviewed-shared），拒绝 short-plan/joint 或使用过 test 选模的来源。原 fair 可以是八卡 ZeRO3；这里只核验/消费其 portable trainable 与 adapter，**不恢复其分布式优化器**。它只重新启用已声明的 LoRA/projector 范围，核对名字、shape、FP32 dtype 和 rank/alpha，不能对全部参数 `requires_grad_(True)`。原模型权重与初始化内容身份进入新 run fingerprint。

省略 `--init-checkpoint` 时从本地原始 Captioner 配置新的 LoRA。新协议训练目前只支持单设备、microbatch=1，不支持 torchrun/DDP/ZeRO3；该 MoE 模型的实际显存和吞吐须在目标设备验证。

恢复使用原参数、同一输出目录，并去掉 init 参数：

```bash
python -m scripts_omnidance.exp.planner_ideas.captioner_genre train \
  --model /models/Qwen3-Omni-30B-A3B-Captioner \
  --train /data/joint/train.jsonl --val /data/joint/val.jsonl \
  --train-labels /data/joint/train-labels.jsonl --val-labels /data/joint/val-labels.jsonl \
  --classes /data/joint/classes.json --out /runs/captioner-joint-v1 \
  --rank 32 --alpha 64 --epochs 1 --accum 4 --max-updates 3 \
  --resume /runs/captioner-joint-v1/checkpoint-0000002 \
  --device cuda:0 --dtype bfloat16
```

`init` 和 `resume` 互斥。恢复保留最初初始化身份，恢复本 run 的 trainable、AdamW、scheduler、RNG 和游标。去掉 `--max-updates` 才跑完设定 epoch 预算。正数更新、两段非零监督 token 和完整 portable checkpoint 是“确实接受过该协议 SFT”的技术门槛，**不代表收敛或质量验收**。

程序绑定实际基础权重完整内容 SHA，以及 config/processor、字典、数据、实现源码和 checkpoint adapter 内容。完整基础权重哈希可能耗时；不是只相信一个模型别名。改变模型、字典或来源后不能静默续跑。

## 联合模型 → 概率头 → 条件生成

完成 joint checkpoint 后，才导出它的 input-only 特征。`head-sources.jsonl` 中每行包含 `id/source_group_id/split/audio_path/audio_sha256/audio_window_seconds`，不需要动作或舞种真值。可包含 head train/val/cal，cal 必须独立；特征导出始终没有 assistant header 或答案。

```bash
python -m scripts_omnidance.exp.planner_ideas.captioner_genre export-features \
  --model /models/Qwen3-Omni-30B-A3B-Captioner \
  --checkpoint /runs/captioner-joint-v1/checkpoint-0000003 \
  --input /data/joint/head-sources.jsonl --out-dir /data/joint/head-features-v1 \
  --device cuda:0 --dtype bfloat16

python -m scripts_omnidance.exp.planner_ideas.genre_decision train \
  --features /data/joint/head-features-v1/features.jsonl --classes /data/joint/classes.json \
  --train-labels /data/joint/head-train-labels.jsonl --val-labels /data/joint/head-val-labels.jsonl \
  --cal-labels /data/joint/head-cal-labels.jsonl --out-dir /runs/captioner-joint-head-v1

python -m scripts_omnidance.exp.planner_ideas.captioner_genre generate \
  --model /models/Qwen3-Omni-30B-A3B-Captioner \
  --checkpoint /runs/captioner-joint-v1/checkpoint-0000003 \
  --head /runs/captioner-joint-head-v1/genre_head.pt \
  --input /data/joint/predict-sources.jsonl --out /runs/captioner-joint-v1/predictions.jsonl \
  --top-k 3 --min-confidence 0.6 --min-margin 0.15 --max-new-tokens 1024 \
  --device cuda:0 --dtype bfloat16
```

head 自动绑定**确切 joint checkpoint 的 trainable、marker、adapter 成员内容 SHA、基础权重及字典**。旧 base/fair head 即便改 model-id 字符串也会被拒绝。继续更新 joint checkpoint 后，应从新 checkpoint 重新导出特征并训练/校准对应 head，不能跨 checkpoint 偷换特征分布。

最高概率及 top2 分差通过门槛时，程序在 **assistant 侧**预填已训练的风格 header tokens，再生成动作。低置信/并列时走 `joint-self-style`：不预填 header，由**同一个已训练联合模型**自己生成 header＋动作；这不是旧基线 fallback，也不声称联合模型能按未经训练的新指令工作。用户音频前缀两条路径都完全保持原生格式。

默认门槛只是实验配置。Jev 思路的概率头和拟合温度不是已验证的舞种质量保证，也没有加入 RL。输出单独记录 head 概率/route、生成舞种、`forced_header_token_ids`、`raw_new_token_ids`、`raw_tail_text`、`full_response_text` 和 `style_header_generated_by_joint_model`。强制 header 不能记作联合模型自己预测正确；`style_agreement` 明确标记是否由 head 强制，语义质量不作自动认可。

EOS 桥仅在原生 generation EOS 缺失时传入已训练的 `<|im_end|>`，不改模型权重或 user prefix。预算与输入 token 数包括 assistant header；高置信路线的生成 token 预算不包含已预填的 header，self-style 路线需要在生成预算中产生 header，比较时须分别记录。首 EOS 后继续吐字、截断、未知舞种 header、token 协议不符、错窗/缺窗均使 `caption_text=null`，完整 raw 文本/token/停止原因和补丁凭据保留。

普通生成不接受 test。最终生成显式加 `--final-test`，同时核验 head train/val/cal 与 joint train/val 的 ID/source/audio SHA 隔离。输出拒绝覆盖。

## 验证范围

```bash
python -m pytest tests/test_captioner_genre.py tests/test_planner_genre_decision.py -q
```

CPU 测试包含真实 autograd 的两个更新和第三步恢复、header/action 边界合并 tokenizer fixture、原生 audio-only prefix、未训练与跨协议拒绝、八卡来源 portable 初始化及精确冻结范围、同 joint 特征/head 绑定、高置信预填/低置信自生成、cal/test 泄漏拒绝和 EOS 包装器调用。使用小型模型与 processor 接口 fixture；尚未在官方预训练 Captioner 或服务器 GPU 上执行此新协议训练，不能据此声称舞种、音乐理解或编舞质量改善。
