# MOSS encoder 实验：原生 LoRA 范围与冻结 MusicFM 融合

日期：2026-10-29。分支：`codex/planner-encoder-experiments`，基于 `a9107c3`。本指南给出基础对照、扩大原生 encoder 适配范围、第二音乐 encoder 融合三个实验的训练和生成步骤。没有启动服务器训练；代码检查不能代替真实模型和 A800 验收。

## 实验定义

八节点资源规划中的节点2、5、6分别承担以下任务；节点编号仅用于区分实验，配置中的 hostname 必须替换成服务器实际 `hostname`。

| 实验 | 配置 | 原生音频 encoder | 额外音乐表示 | 其他训练范围 |
|---|---|---|---|---|
| M0：MOSS 基础 SFT | [moss_encoder_baseline.example.json](../../examples/planner_training/moss_encoder_baseline.example.json) | 最后4层 LoRA | 无 | LM LoRA＋完整原生 audio adapter／DeepStack mergers |
| E5：扩大原生适配范围 | [moss_native_encoder8.example.json](../../examples/planner_training/moss_native_encoder8.example.json) | 最后8层 LoRA | 无 | 与 M0 相同 |
| E6：MusicFM 融合 | [moss_musicfm_fusion.example.json](../../examples/planner_training/moss_musicfm_fusion.example.json) | 最后4层 LoRA | 冻结 MusicFM dense 序列 | M0 范围＋独立可训练融合模块 |

E5 的“8层”指最后8层上的 LoRA，不是全参数训练8层。它使用实际原生模块名选择目标，语言侧目标保持一致；请求的层数超过原生 encoder 层数时拒绝启动。

E6 使用官方 MusicFM25Hz 的帧序列：先离线提取并冻结特征，训练时只读取缓存，不在每个 rank 加载 MusicFM。这不是端到端微调 MusicFM。当前实现把帧特征按绝对秒数插值到原生 MOSS 12.5Hz 音频 token 网格，经 FP32 LayerNorm／投影／GELU／投影和小幅可训练门控，残差加入最终 `audio_adapter` 输出；原生 DeepStack、占位 token 位置和完整音频输入保持原义。

三份配置都使用单节点8张80GB A800、microbatch 1、梯度累积4次、有效 global batch32；LoRA rank32／alpha64、LoRA lr=1e-5、projector lr=5e-5、相同数据、epoch预算和验证规则。E6 新模块 lr=1e-4、hidden_dim=256是起始实验值。先比较 M0→E5、M0→E6，不在首轮叠加 JEV、loss改进或RL。

## 正式数据与纯 SFT

三组都使用[23值词表](../../examples/planner_training/dance_styles_23.json)和[英文 v2 合同](../../examples/planner_training/caption_contract_moss_v2.example.json)。将它们分别部署为 `/shared/data/planner/classes.json` 和 `/shared/data/planner/caption_contract_v2.json`，保持文件内容一致；不要在加载大模型期间修改合同、词表或缓存索引。

正文仍是：

```text
Dance style: contemporary
[0.000-1.960s] The dancer steps to the side.
[1.960-5.062s] The dancer raises the left arm.
```

这只是格式示意。第一行是22个真实舞种加 `unknown` 中的一个值，每个后续物理行是一个三位小数时间窗和英文动作。窗口表示动作发生的包含区间，不能把小节边界当作动作精确 onset／offset。原始标注的语义质量和手部可靠性仍需画面审核；正式合同拒绝常见帧号引用，不会自动执行 RGB 修订。

三组均显式设置 `enable_jev=false`、`genre_weight=0`。`--genre-classes` 此时只提供 Caption 词表，不安装辅助头，也不需要 `train_genre_labels`／`val_genre_labels`。Caption token CE直接学习正常舞种 header、时间戳、动作及EOS。沿用正式合同的视觉弃权策略：`unknown` 首行不计 CE，动作正文及EOS继续计入。缺标签与视觉 `unknown` 不是同一回事，`freestyle / other` 仍是真实类别。

按原歌曲、原视频和近重复组冻结 train／val；本轮不得因为 E6 缺缓存就静默删样本。三个实验使用完全相同的音频和目标文本快照。完整音频输入对应一条训练记录，不因为目标有多个动作窗口而重复裁音频。验证 NLL只适合同一模型、同一目标协议的辅助比较；仍需生成盲评和格式、动作质量评测。

## E6：离线准备 MusicFM 缓存

服务器需预先准备以下本地资产：

- 官方 [MusicFM 源码](https://github.com/minzwon/musicfm)，例如 `/models/src/musicfm`；目录中含 `model/musicfm_25hz.py` 和 `modules/features.py`。
- 与官方 MusicFM25Hz 匹配的预训练 checkpoint 和 mel 统计文件，例如 `/models/MusicFM/msd_pretrained.pth`、`/models/MusicFM/msd_stats.json`。实际文件名以下载资产为准。
- 本地 `wav2vec2-conformer-rope-large-960h-ft` 的 `config.json`，例如 `/models/wav2vec2-conformer-rope-large-960h-ft/config.json`。它用于离线构造 Conformer；不从网络自动获取配置或下载另一份权重。
- 同一提取环境的 Torch／torchaudio、Transformers、NumPy、SciPy、soundfile、einops以及官方源码依赖。先在既有环境验收兼容性，不把本地 CPU 测试当作预训练模型提取已跑通。

在一个进程、一个 writer 中提取 train 和 val。以下路径需要替换成实际资产：

```bash
python -m scripts_omnidance.exp.ms.music_planner_musicfm prepare \
  --manifest /shared/data/planner/train.jsonl \
  --manifest /shared/data/planner/val.jsonl \
  --output-dir /shared/musicfm-cache \
  --musicfm-root /models/src/musicfm \
  --checkpoint /models/MusicFM/msd_pretrained.pth \
  --stats /models/MusicFM/msd_stats.json \
  --conformer-config /models/wav2vec2-conformer-rope-large-960h-ft/config.json \
  --device cuda:0 --layer-ix 7
```

该入口只消费音频身份、路径、完整时长和 `[0,duration]` 窗口，忽略 Caption 和标签；核验源音频 SHA，要求有限的原始单声道音频，按固定预处理重采样到24kHz，以 eval／冻结模式调用官方 dense latent，不做波形mask、不池化、不裁窗。它保存完整 FP32 `[T,1024]` safetensors及 `index.json`，绑定模型、源码、统计、Conformer配置、运行库和提取协议身份。

同一缓存目录不要由多个进程同时追加。准备全部条目后再生成训练计划；训练期间不要追加或重写 index。约22万条、每条约5秒时，单路 FP32 dense 特征约110GB量级，实际大小依样本时长和文件开销而变；提前检查共享存储空间。

可单独检查数据覆盖，默认只查索引及元数据：

```bash
python -m scripts_omnidance.exp.ms.music_planner_musicfm validate \
  --index /shared/musicfm-cache/index.json \
  --manifest /shared/data/planner/train.jsonl \
  --manifest /shared/data/planner/val.jsonl
```

追加 `--verify-bytes` 会读出全部对应特征，核对真实SHA、内嵌身份、shape、dtype及有限值，读盘成本接近整份缓存。训练时每次实际加载特征仍核验其真实文件。缓存与元数据匹配只证明身份一致，不证明特征对舞蹈规划有收益。

## 数据检查与单节点启动

先替换三份 example 中的路径、Python解释器、实际 hostname 和独立输出目录；可以复制到 `/shared/config/moss-encoder-baseline.smoke.json`、`moss-native-encoder8.smoke.json`、`moss-musicfm-fusion.smoke.json`。所有节点使用同一源码快照和固定环境。

E6 的数据检查命令如下。M0删掉 `--musicfm-cache`／`--fusion-*`，E5再把 `--audio-last-n` 改为8：

```bash
python -m scripts_omnidance.exp.ms.music_planner_train \
  --family moss --model /models/MOSS-Music-8B-Instruct \
  --train /shared/data/planner/train.jsonl --val /shared/data/planner/val.jsonl \
  --out /shared/runs/moss-musicfm-fusion-smoke \
  --caption-contract /shared/data/planner/caption_contract_v2.json \
  --genre-classes /shared/data/planner/classes.json \
  --no-enable-jev --genre-weight 0 --audio-last-n 4 \
  --musicfm-cache /shared/musicfm-cache/index.json \
  --fusion-hidden-dim 256 --fusion-lr 0.0001 \
  --batch 32 --micro 1 --accum 4 --check-data-only
```

`--check-data-only` 不加载或训练大模型；它验证合同、正文／窗口元数据、划分身份、类别和缓存覆盖，不能证明显存够用或完整特征字节已扫描。

对每个实验分别执行计划、实际节点预检、回执验证、命令生成。以E6为例：

```bash
python -m scripts_omnidance.exp.ms.music_planner_cluster plan \
  --spec /shared/config/moss-musicfm-fusion.smoke.json \
  --output /shared/config/moss-musicfm-fusion.smoke.plan.json

# 在该任务配置指定的实际 hostname 上执行；不手工伪造回执。
python -m scripts_omnidance.exp.ms.music_planner_cluster preflight \
  --plan /shared/config/moss-musicfm-fusion.smoke.plan.json \
  --receipts /shared/config/moss-musicfm-fusion.smoke.receipts

python -m scripts_omnidance.exp.ms.music_planner_cluster verify \
  --plan /shared/config/moss-musicfm-fusion.smoke.plan.json \
  --receipts /shared/config/moss-musicfm-fusion.smoke.receipts

python -m scripts_omnidance.exp.ms.music_planner_cluster commands \
  --plan /shared/config/moss-musicfm-fusion.smoke.plan.json \
  --receipts /shared/config/moss-musicfm-fusion.smoke.receipts --format shell
```

工具不执行 SSH、不自动调度训练。完成预检后，在输出命令指明的节点和工作目录运行对应行。真实 preflight 要求8张可见80GB A800、原生 API、模型和代码内容、数据及输出目录一致；`--dry-run`不能形成GPU准入证据。E6每个任务节点的 preflight 还会完整读取并核验对应特征，约110GB量级缓存意味着显著读盘成本；它不在每个 rank 重复全量扫描。后续 `verify`／`commands`复核索引身份，不再次逐文件全扫。训练单条加载仍逐项核验。

## Smoke、新进程恢复与正式训练

三组 example 都是 `phase=smoke`、`max_updates=2`。这表示总共2个 optimizer update，不是完整训练。检查所有 rank 的真实 backward、LoRA／projector及E6融合参数组的非零梯度、有限 loss、实际checkpoint调用、完整验证诊断和完整checkpoint；不要求纯SFT出现JEV梯度。

随后为每组新建一份 resume spec，保持原始 `out`、模型、数据、训练超参和 world size不变，只调整：

```json
{
  "job_id": "moss-musicfm-fusion-resume-smoke",
  "phase": "smoke",
  "max_updates": 3,
  "resume": "/shared/runs/moss-musicfm-fusion-smoke/checkpoint-0000002"
}
```

这是对原job的修改字段示意，不是可独立运行的完整spec。使用新的 plan 文件和 receipts 目录重新执行上述四步，再在新进程运行生成的命令。`max_updates=3`是绝对上限，表示在恢复后的update2基础上继续至少一次；不能重复设置2。核验 optimizer、scheduler、RNG、cursor和新update实际发生。相同global batch不能使不同world size的checkpoint变成可直接恢复。

正式训练从原始模型启动一个独立新任务：为原smoke spec设置新的 `job_id`／`out`，删除 `resume`，改 `phase=full`、`max_updates=0`，重新生成计划及回执。三组使用相同epoch／数据曝光预算；示例 `epochs=2`只是共同起点，实际正式预算应在启动前冻结。不要把两步smoke后更换数据、词表或层数叫作严格恢复。源码、模型内容、数据、词表、缓存、可训练参数与world size进入训练身份；更换实验范围应建立新run。

需要双节点提速时，在新任务spec中改为：

```json
{
  "hosts": ["ACTUAL-node-5", "ACTUAL-node-6"],
  "accum": 2
}
```

每节点8卡、micro=1，16×2仍为global batch32。两个节点分别 preflight，再运行各自 node rank 命令。新任务必须与其他并行任务使用不重叠的节点、独立 `out`／`job_id`／rendezvous端口；不能把单节点run直接改hosts后resume。`flexible`也可把最多8个并行job放入同一计划，总节点数不超过8；同一计划用于模型对比时要求train／val manifest完全一致。网络可访问不等于双节点线性提速，应实测吞吐、通信、保存和验证开销。

## 生成：模型自由输出舞种和动作

新入口 `music_planner_encoder_generate`面向上述MOSS encoder实验。它从完整checkpoint恢复基础模型、PEFT、正式合同、可选融合模块及原生position／cache／EOS兼容补丁；不要求存在JEV头。完整 `Dance style:` 首行及动作全部由模型生成，不预填真值舞种。

M0／E5：

```bash
python -m scripts_omnidance.exp.ms.music_planner_encoder_generate \
  --model /models/MOSS-Music-8B-Instruct \
  --checkpoint /shared/runs/moss-native-encoder8-full/checkpoint-0000100 \
  --row /shared/inference/audio.json \
  --output /shared/inference/native8-result.json \
  --device cuda:0 --max-new-tokens 1024
```

checkpoint路径中的update100只是示意，换成实际完整checkpoint。音频行需要 `id`、`source_group_id`、`audio_path`、`audio_sha256`、`audio_duration_seconds`、`audio_window_seconds=[0,duration]`，不需要Caption或舞种真值。接口只读取音频输入白名单字段。

E6生成前，新音乐也要用前面的 `prepare`命令提取特征：将 `--manifest` 换成包含待生成音频行的JSONL，将 `--output-dir` 换成独立 `/shared/musicfm-inference-cache`。保持完全相同的MusicFM权重、层号、统计、Conformer配置、源码和提取环境／协议。不能向正在训练的原缓存追加新条目。

```bash
python -m scripts_omnidance.exp.ms.music_planner_encoder_generate \
  --model /models/MOSS-Music-8B-Instruct \
  --checkpoint /shared/runs/moss-musicfm-fusion-full/checkpoint-0000100 \
  --musicfm-cache /shared/musicfm-inference-cache/index.json \
  --row /shared/inference/audio.json \
  --output /shared/inference/musicfm-result.json \
  --device cuda:0 --max-new-tokens 1024
```

训练恢复要求原训练索引身份保持一致；推理处理新音频允许不同index，只要求相同extractor身份并核验实际音频及特征。基础模型未训练融合时，不能在推理命令里临时添加MusicFM缓存。

成功结果的 `caption` 是原始模型文本，严格验收23值首行、三位小数时间窗、时间覆盖和首个原生assistant EOS等条件。失败保留原始token、文本、停止原因和错误，`caption=null`，CLI退出码2；不会自动修补成成功结果。格式正确仍不等于动作内容可靠。基础对照、E5、E6用同一组音乐、相同生成预算及自由生成路线比较，再分别统计格式失败、舞种、动作丰富度／可执行性／连贯性和音乐适配。

当前自由生成入口只支持MOSS，没有补齐Qwen Captioner同等入口。本轮不宣称两模型生成评测已经统一，也不实现training-free重排。

## 实现边界与服务器验收

本轮融合只支持原生单完整clip、microbatch1，不支持多个音频packed输入。MusicFM保持完整25Hz时间网格，原生网格为12.5Hz，不按padding长度缩放或把整段归一化到同一时间轴。静音／错配音频消融必须重新提取对应特征，不能把原音乐缓存搭配静音输入。

先在服务器验证实际约5.0625秒样本及手头最长音频，包括原生预处理、有效长度、融合时间对齐、PEFT保存恢复、ZeRO-3生命周期、生成prefill与cached decode。正式合同允许到30秒不代表所有长音频／依赖版本已完成实测；不支持的packing、网格或边界应明确失败。

本地验证使用CPU小模型、受控张量及原生接口替身；尚未运行真实预训练MOSS／MusicFM、真实PEFT／DeepSpeed／A800／NCCL训练或生成，也未证明单节点显存余量、双节点加速比例或质量提升。

## 本地验证记录

2026-10-29，Python3.9／CPU Torch环境执行以下10个相关测试文件，结果 **477 passed，9 skipped，68.67秒**：

```bash
python -m pytest -q -rs \
  tests/test_music_planner_train.py \
  tests/test_music_planner_scaling.py \
  tests/test_music_planner_cluster.py \
  tests/test_music_planner_jev.py \
  tests/test_music_planner_schema.py \
  tests/test_music_planner_caption_format.py \
  tests/test_music_planner_moss_generate.py \
  tests/test_music_planner_musicfm.py \
  tests/test_music_planner_encoder_fusion.py \
  tests/test_music_planner_encoder_experiments.py
```

7项因未安装DeepSpeed跳过，1项因本机CPU Torch没有可用Gloo transport跳过，1项因Windows账号无法创建文件符号链接跳过。测试包含真实WAV重采样与缓存核验、真实小型网络梯度、checkpoint重算／累积前向、融合与原生generation cache补丁共存、portable融合权重导出／恢复、相同extractor的新音频缓存、未知舞种头未训练时的自由生成及失败原文保留。接口替身不等于预训练模型已跑通。

修改模块编译、三份JSON配置以及新提取／生成CLI帮助也通过。本记录仅涵盖所列范围，不是全仓测试或GPU验收。
