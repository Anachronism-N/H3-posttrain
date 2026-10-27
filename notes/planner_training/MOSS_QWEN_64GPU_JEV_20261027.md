# MOSS / Qwen Captioner 多机训练与 JEV 舞种辅助分支

本文件记录 `416bab8` 时的 v1 协议。2026-10-28 已新增 MOSS 的「离散舞种＋时间戳动作」v2协议、受舞种控制的生成入口及恢复校验修复；当前使用请先阅读[后续复核与运行说明](MOSS_STYLE_TIMED_REVIEW_20261028.md)。下文关于JEV并行输出、尚未控制文本生成的描述适用于历史v1。

日期：2026-10-27。上游基线：`7ed6373a0de6cb343967bec194e9ac31454e237d`；开发分支：`codex/planner-moss-captioner-64gpu`。

## 目标与交付边界

最终规划器输入音乐，输出你们约定的自然语言编舞 Caption；两个候选模型是 MOSS-Music-8B-Instruct、Qwen3-Omni-30B-A3B-Captioner。训练资源为用户提供的 8 节点，每节点 8 张 80 GB A800。本实现准备训练入口、联合舞种分支、分布式启动计划和验收，不远程启动或占用这些服务器。

Caption 格式、舞种列表尚待提供。因此没有把假定的舞种词表、舞种 header、JSON 输出格式或固定两段时间窗写进新训练协议。旧 `music_planner_train` 不传新参数时仍接受原 fixed2s 数据；新协议由独立合同文件显式选择。历史公平实验应继续使用它自己的代码快照和指纹，不混用新训练结果。

JEV 在本文指借鉴“生成任务与概率决策任务分工”的本地可训练分类分支，**不是 TypeSafe 官方 Jev 的权重或 RLCD 复现**。官方目前接受文本输入、提供托管决策模型，并不支持客户 LoRA 微调；不能把它直接当原生音乐 encoder 使用。参考：[TypeSafe 官方模型说明](https://docs.typesafe.ai/models)。

## 两个模型如何分配 64 张卡

| 模式 | MOSS | Captioner | 每模型 global batch |
| --- | --- | --- | --- |
| `split32`，建议第一轮 | 4 节点 × 8 卡 | 另 4 节点 × 8 卡 | 32，micro=1、accum=1 |
| `single64` | 一次只运行其中一个模型，8 节点 × 8 卡 | 同左 | 64，micro=1、accum=1 |

原实验 global batch32；32 卡每模型可以保留这个 batch，有利于对照。64 卡单模型使用 batch64 时，每个 epoch 的 optimizer update 数约减半，不能直接照搬旧的 update 次数来声称同等数据曝光。比较时同时报告 epoch、source exposures、更新次数、学习率和有效 batch。当前原生音频拼接明确只支持 micro=1，不通过复制或丢弃样本伪造小于 world size 的 global batch。

使用 ZeRO-3 做模型参数／优化器分片，不是把两个模型放进同一个计算图，也没有实现额外的 tensor/expert parallel。Qwen 的“激活参数约 3B”不等于只需要存放 3B 权重。新入口保留历史 FP32 参数分片与 FP32 可训练参数／Adam 状态、BF16 计算协议；这也意味着不应按纯 BF16 权重估算存储和通信成本。真实吞吐、峰值显存、网络开销必须在集群 smoke 中测量，不能由 CPU 测试推定。

## 训练与 JEV 的计算路径

```text
完整源音乐 → 模型原生音频处理与音频 encoder
           → audio projector / MOSS DeepStack
           → 原生因果语言 decoder
              ├─ assistant token → Caption CE
              └─ 最后一个输入 prefix token → FP32 MLP → 舞种概率 CE
```

MOSS 可以使用合同里的任务指令。Captioner 保持原生单音频、无额外文本指令的 user prefix，目标格式通过 assistant 监督学会；不把任务指令强塞进它的输入。[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner)说明它是 audio-only、text-output、单轮模型，建议音频不超过 30 秒。合同中的最大时长只是数据约束，不表示任意长音频已做过模型兼容或显存验收。

新舞种头在 DeepSpeed 初始化前注册到模型中。一次原生 teacher-forced forward 同时产生 Caption logits 和舞种 logits。通过 decoder 最终 norm 的 hook，仅抽取原生输入 prefix 最后一个有效 token 的表示；原生因果 attention 使该位置看不到后面的真值答案。不会对完整 teacher-forced Caption 做 pooling，也不导出全部 decoder 层的 hidden states。特征协议固定为 `native-causal-prefill-last-valid-token-v1`，与旧离线“音频 token 平均池化”头不能互换。

训练范围仍以 LoRA 为主：语言模型 attention/MLP、可配置最后 N 层音频 encoder LoRA、完整音频 projector／MOSS merger，以及新增舞种头；其余基础参数冻结。`--audio-last-n` 默认4。这不是两个基础模型全参数微调，也没有加入新的 MusicFM 编码器。

联合 loss 为：

```text
L = 全局有效 assistant token 的 CE 总和 / 全局有效 assistant token 数
  + genre_weight × 全局有标签源的舞种 CE 总和 / 全局有标签源数
```

舞种 loss 按有标签源归一化，不按 Caption 长度归一化。尾批空 rank／空 accumulation 槽保持参与 collective 和带零梯度的计算图，但不计入 token、样本或舞种监督数量。若某个 update 没有已知舞种标签，其分类项为零。缺标签不代表“其他”或负类。舞种权重建议先从0.1开始做对照，它是起始超参，没有已有实验收益保证。

v1 舞种头支持单个硬标签或一个概率和为1的软类别分布。软分布可以表达一首音乐有多种合理编舞风格；它不等于多标签 BCE。若最终 list 是层级或“多个舞种同时存在”的定义，应先扩展任务定义和 loss，而不是把多热向量归一化后冒充概率。

## 数据合同与待用户补充的信息

先复制并编辑 [caption_contract.example.json](../../examples/planner_training/caption_contract.example.json)。`caption_version` 标识一套真正固定的格式；`task_instruction` 供 MOSS 使用；`required_markers` 是可选的字面标记检查；`require_target_windows` 决定是否要求内部时间窗元数据。JSON 只是训练配置和内部样本容器，模型的监督目标仍是原样自然语言文本。

新合同数据每行必需：

```json
{
  "id": "source-unique-id",
  "source_group_id": "original-song-or-dedup-group",
  "split": "train",
  "audio_path": "/shared/audio/source.wav",
  "audio_sha256": "实际音频文件的64位小写SHA256",
  "audio_duration_seconds": 5.0625,
  "audio_window_seconds": [0.0, 5.0625],
  "target_text": "这里放你们最终确认的完整自然语言 Caption，原样训练。",
  "caption_version": "与合同完全相同的版本",
  "semantic_approval": false,
  "human_review_completed": false
}
```

可提供 `song_id`、`near_duplicate_group_id` 增强跨 split 隔离检查；声明相同的组禁止同时进 train/val。未提供的身份不被视为已经通过歌曲／近重复审计。程序检查源身份和格式，不替代语义审核。声称人工或语义审核为真时必须给 `review_provenance`，程序仍只记录声明，不自行批准内容。

每条样本对应完整源音频，一组局部动作描述可以组成同一条完整 Caption；不因文字分 chunk 而物理裁成多个音乐训练样本。文件实际时长必须与声明相符，重采样到16k，不截断、不补齐到假定长度；上下文超限报错，不静默截掉后半段标签。`target_windows` 若提供，要求有序、不重叠、在源时长内；这些窗口验证不等于动作 onset 已被核实，也不会自动从文本解析出时间语义。

收到舞种 list 后创建固定顺序的 `classes.json`，内容是至少两个不同非空字符串的 JSON array。顺序属于模型语义，调整顺序或新增类不能直接严格 resume 旧头。不要现在用随意列举的舞种代替正式词表。

独立舞种标签文件 `train-genres.jsonl` / `val-genres.jsonl` 的一行格式为：

```json
{
  "id": "与Caption数据相同的id",
  "source_group_id": "与Caption数据相同的来源组",
  "audio_sha256": "与Caption数据相同的SHA256",
  "split": "train",
  "genre": "正式词表中的一个字符串",
  "label_source": "真实数据集舞种字段或人工标注记录位置",
  "label_version": "dance-style-label-v1"
}
```

软标签把 `genre` 换成 `genre_distribution`，值为“类别名→概率”的对象，两者只能有一个。可以只标一部分源，未出现行的样本按 unknown 处理；开启分支时训练集至少有一个真实标签。禁止未知类别、重复ID、跨 split 和源 SHA 不匹配。音乐风格与舞种应分开定义，不能把 music genre 自动当作 dance style 真值。

## 如何执行

数据先行检查，不加载基础模型：

```bash
python -m scripts_omnidance.exp.ms.music_planner_train \
  --family moss --model /models/MOSS-Music-8B-Instruct \
  --train /shared/data/train.jsonl --val /shared/data/val.jsonl \
  --out /shared/runs/moss-caption-v1 \
  --caption-contract /shared/data/caption_contract.json --check-data-only
```

开启舞种联合训练时，在训练入口或 cluster spec 中增加正式文件配置：

```bash
--genre-classes /shared/data/classes.json \
--train-genre-labels /shared/data/train-genres.jsonl \
--val-genre-labels /shared/data/val-genres.jsonl \
--genre-weight 0.1 --genre-lr 0.0001 --genre-head-dim 256
```

复制 [split32.example.json](../../examples/planner_training/split32.example.json)，替换八台真实 hostname、各节点一致的仓库／Python／模型／数据路径和输出目录。示例是两组各32卡 smoke 配置，不含伪造词表。路径、端口、job ID 和资源分配应分别检查。

```bash
python -m scripts_omnidance.exp.ms.music_planner_cluster plan \
  --spec /shared/config/split32.json --output /shared/config/planner-plan.json

# 在全部八个节点分别运行；每个节点只写自己的收据。
python -m scripts_omnidance.exp.ms.music_planner_cluster preflight \
  --plan /shared/config/planner-plan.json --receipts /shared/config/receipts

# 所有真实节点收据齐全且一致后核验，再打印每个host对应的命令。
python -m scripts_omnidance.exp.ms.music_planner_cluster verify \
  --plan /shared/config/planner-plan.json --receipts /shared/config/receipts
python -m scripts_omnidance.exp.ms.music_planner_cluster commands \
  --plan /shared/config/planner-plan.json --receipts /shared/config/receipts --format shell
```

`plan` 不需要 CUDA。`preflight --dry-run` 仍检查实际 hostname、真实本地文件和数据，只跳过 GPU 运行环境验收，不会被当作预检通过。工具只生成命令、不做 SSH 或自动占卡。torchrun 使用固定 node rank/world 的静态多机模式，关闭 elastic 自动重启；严格 checkpoint 不支持悄悄更换 world size。每个模型 job 的节点需要相同代码、依赖版本、完整模型文件、数据合同、标签内容和可共同访问的输出目录；预检记录实际文件内容哈希与运行环境，不仅比较文件名／大小。

不要直接复用仓库旧 `setup_venv.sh` 的视频训练环境：其 `wan_reqs.txt` 中的旧 Transformers 固定版本不涵盖当前 Captioner 接口。应将现有已经跑通两模型的 planner 环境复制并冻结，利用 `expected_versions` 约束实际版本，再跑 smoke。[Qwen 官方 README](https://github.com/QwenLM/Qwen3-Omni/blob/main/README.md) 当前建议 Transformers 5.2 或更新版本；这是上游接口参考，不代替本项目的兼容验收，也不是要求盲目升级现有环境。预检本身不联网下载权重或安装依赖。

先完成每模型2个真实 update 的 smoke（带全验证集诊断），确认所有 rank 有真实梯度、projector／语言 LoRA／舞种头确实更新、loss 有限、checkpoint 文件完整。随后以同配置、新进程 `--resume` 到至少下一个 update，核验模型、optimizer、scheduler、RNG、cursor 的恢复。smoke 的 `--max-updates` 是绝对总 update 上限，续到3必须设3，不能再次设2。正式 full run 使用独立输出目录／计划，`max_updates=0`；不要把2步 smoke 的结果当作正式训练或质量验收。

没有正式 Caption／舞种 list 时可以使用历史协议做基础兼容 smoke；这只证明旧任务链路，不能声称最终协议或新舞种监督已训练成功。新入口当前支持新训练与同配置严格 resume；没有实现任意旧实验 checkpoint 的 warm start，也不能把换词表／换格式后重启 optimizer 叫作严格恢复。

推理加载新的完整 checkpoint 时，先用原 `load_planner_adapter` 载入匹配的基础模型与 portable PEFT adapter，再调用 `music_planner_schema.restore_adapter_protocol(native, checkpoint)`，恢复训练时的 Caption 合同、时长策略及可选舞种头。该 helper 核验 checkpoint／导出物绑定，但不替代基础权重内容身份检查。MOSS 必须先应用仓库已有 `restore_moss_position_buffer(adapter, model_path)`，再进行包括 `predict_jev` 在内的音频 forward；文本生成还需沿用现有 cache／EOS 兼容补丁。`music_planner_jev.predict_jev(adapter, audio_row)` 返回独立概率结果，只消耗音乐字段。旧固定窗口评测器不会自动理解未来的新格式，应在格式确定后另接对应的生成格式验收，而不是沿用旧 `parse_windows` 判分。

## 优化重点与验收

1. Caption CE 先选有效 assistant 位置，再按 chunk 做 FP32 CE；可用 checkpoint 重算临时块，避免为整段音频 prefix 生成额外 FP32 loss 副本。基础模型仍产生原生词表 logits，这不是消除全部 logits 显存的 fused LM-head 实现。
2. Captioner MoE 采用显式 ZeRO-3 leaf 策略，实际识别模块并记录，不假设每个 rank 的动态路由相同。参考：[DeepSpeed ZeRO-3 leaf modules](https://deepspeed.readthedocs.io/en/latest/training.html#zero3-leaf-modules)。
   舞种头使用独立 Linear 类型，避免被默认 `lower_precision_safe_modules` 的精确类匹配纳入 BF16 通信；保持 head 的 FP32 计算和通信协议。DeepSpeed gather 会恢复参数存储 dtype，因此这里针对的是 BF16 通信往返精度，不声称已复现 dtype 崩溃。依据：[官方 autocast 源码](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/torch_autocast.py)、[参数 gather 源码](https://github.com/deepspeedai/DeepSpeed/blob/v0.18.0/deepspeed/runtime/zero/partition_parameters.py)。
3. 保留源级打乱、完整数据曝光、尾批和全局 token／有标签源归一化，避免扩大卡数后 loss 随 world size 改变。
   多机模式每节点只由一个进程重新核验完整模型文件内容，所有 rank 对齐核验结果；共同 fingerprint 使用已验证的内容身份，避免不同节点相同内容但 mtime 不同而误拒绝。未经该内容验证的普通单机入口保留历史身份检查。
4. 保存 trainable portable 权重与独立舞种头，同时保存可恢复的分布式 optimizer／RNG 等状态。只有权重导出成功不等于能严格恢复训练；舞种头的词表顺序、特征协议必须随 checkpoint 保存。支持 constant／cosine 学习率与按总更新预算的 warmup ratio；正式训练会按 `eval_every` 真正执行验证 NLL，并按验证集选择 checkpoint，而非仅写评估请求。
5. Caption 格式错误、舞种预测质量、动作质量分别评估。验证 NLL 只能在同模型 tokenizer 内比较；跨 MOSS／Captioner 仍需固定验证音乐、生成预算、盲评动作丰富度／可执行性／时间覆盖。舞种 head 的概率没有自动校准保证，应另留 calibration 数据拟合温度或选阈值，不能看到 softmax 就称“可信置信度”。

最少对照为同数据／曝光预算的 Caption-only 与 Caption+JEV。观察舞种 macro-F1、逐类召回、NLL／Brier／ECE，以及 Caption 质量是否下降。若舞种很好但动作更模板化，优先降低辅助权重、查类别不平衡和标签定义，而非直接扩大训练。舞种结果是并行输出；当前没有把 head 的 argmax 自动插入 Caption 或强制 header，最终如何与自然文本合并应由你们确认的格式决定。

本地验证使用 CPU 小模型／合成张量，检验因果边界、梯度、loss 等价性、源隔离、计划与恢复协议。真实 MOSS／30B Captioner 的32／64卡训练、NCCL带宽、A800显存余量和语义提升仍需上述集群验收，不由本次代码提交代替。

## 本次验证记录

在本地 Python3.9／CPU Torch2.8、Accelerate1.10.1 环境执行：

```bash
python -m pytest -q -rs \
  tests/test_music_planner_train.py \
  tests/test_music_planner_scaling.py \
  tests/test_music_planner_schema.py \
  tests/test_music_planner_jev.py \
  tests/test_music_planner_cluster.py
```

结果为 **154 passed，8 skipped**。7个跳过项缺少 DeepSpeed；另1个真实双 rank Gloo 测试因当前 Windows Torch 无可用 transport 跳过。两份 rank 模型各自 backward 后的平均梯度与单个 global objective 的数值对照通过，但它不替代真实 collective。实际 CPU Accelerate 训练入口的区间验证、保存、新进程身份恢复以及“前两步未知舞种、恢复第三步才出现标签”的闭环均通过。

另外，既有 adapter 的分片 dtype／属性保留、完整音频长度边界及 caption-only 参数组兼容性定向回归 **27 passed**。尝试扩展运行完整原生 adapter 测试时，环境缺少 Transformers／PEFT／DeepSpeed／官方模型 metadata；相应依赖案例没有通过本地验收，不将其并入成功计数。已修复其中发现的 caption-only 参数组兼容回归。实际模型环境仍应运行这些原生 adapter 检查及多机 smoke。未安装预训练模型或在本地模拟 GPU 成功回执。
