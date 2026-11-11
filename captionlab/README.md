# CaptionLab：冻结模型的音乐→动作文本诊断

## sol61 自由生成 baseline 与质量评价

训练中的 captioner 尚不可用时，可先用 `sol61_captions` 冻结评测集、运行开源音频模型、导出听音盲评包并校准评价指标。完整命令、指标定义与集群资源见 [执行方案](../notes/48_sol61_baseline_evaluation_20261111_zh.md)；六模型比较、agent 对照、统一预算及空白结果表见 [扩展实验表](../notes/49_expanded_baselines_and_agent_experiments_20261111_zh.md)。后者包含待接入能力，不能直接用替换现有配置模型路径来运行全部基线。新增入口：

- `python -m captionlab.sol61`：固定 HF 版本下载、全划分审计、抽样、按原 SHA 绑定 WAV；GT 与生成输入分开。
- `python -m captionlab.baseline_eval`：音频＋固定任务 prompt 的自由生成；保留失败、截断、未知调用及原始响应。
- `python -m captionlab.review_bundle`：匿名音频/文字评审包及解除盲法；参考一致性另开阶段。
- `python -m captionlab.quality_judge`：文字 judge 与证据校验；不对未听过的音乐打分。
- `python -m captionlab.choreo_quality`、`python -m captionlab.metric_validation`：量表、质量分项与配对差异、人评一致性及自动指标相关性。

这些入口接受 sol61 无时间戳正文，**不经过下方旧 teacher-forcing 的双时间窗合同**。首版指标需要独立人评验证，CPU 测试不等于模型或指标质量结果。

## 原检查点 teacher-forcing 诊断

独立于 ChoreoBench 的推理/数据准备工具，对应 [8a833c9 训练复盘](../notes/34_music_caption_analysis_20261109_zh.md) 与 [下一轮执行方案](../notes/35_captionlab_experiments_20261109_zh.md)。没有训练入口，不修改冻结候选、整曲规划协议或 v11/v17 结果。

已实现：固定验证集的真音乐/随机候选/显式困难候选/同长度静音清单、来源排除、原始目标保留、逐 token 因果 NLL、来源组 bootstrap、失败分母，以及舞种/时间戳的可回溯目标转换。**这些是测量工具，不是新的模型质量结果。** 当前 `run` 不自动识别动作 token，`explicit_action` 为 NA；汇总器可读取另行核验的 `action_mask`，不能把整段动作正文都冒称为信息 token。

## 安装与 CPU 验证

从仓库根目录运行：

```bash
python -m pip install -r captionlab/requirements-cpu.txt
python -m pytest -c captionlab/pytest.ini -q
python -m captionlab --help
```

数据准备和已有分数汇总不需要 Torch。`runtime` 单测/真实评分需要 Torch；在**单独 CPU 测试环境**可安装 `torch --index-url https://download.pytorch.org/whl/cpu`。真实服务器沿用原检查点环境及锁定版本，不把 CPU Torch 覆盖进其 CUDA 环境。

## 1. 显式来源合同

`sources.json` 是 `{ "schema":"captionlab.sources.v1", "genre_classes":[...], "records":[...] }`。每条 `records` 恰好包含：

```json
{
  "schema": "captionlab.source.v1",
  "id": "original-sample-id",
  "split": "val",
  "source_group_id": "original-video-group",
  "song_id": null,
  "near_duplicate_group_id": null,
  "raw_caption": "Dance style: contemporary.\n[0-2s] One side wrist moves left.\n[2-5.062s] The other side knee bends.\n",
  "genre_header_template": "Dance style: {genre}.\n",
  "audio": {
    "path": "audio/original.wav",
    "sha256": "REPLACE_WITH_ACTUAL_FILE_SHA256",
    "pcm_sha256": "REPLACE_WITH_EXISTING_CANONICAL_PCM_SHA256",
    "sample_rate": 24000,
    "num_samples": 121500,
    "source_start_seconds": null,
    "pad_intervals": null
  },
  "windows": [
    {"start": 0, "end": 2, "context_dependency": "independent"},
    {"start": 2, "end": 5.0625, "context_dependency": "unknown"}
  ]
}
```

这是字段说明，SHA 占位符不能通过校验。请从原标注/媒体核验表导出真实字段，**不要为通过校验猜歌曲ID、PCM、历史补零或事件时间**。所有 PCM 身份必须使用原核验的同一规范；此工具不自行定义新 PCM hash，也不声称独立复核该声明。`song_id/near_duplicate_group_id=null` 明确表示未核实；`pad_intervals=null` 是未知，`[]` 才是已知无补零。区间为半开采样索引。

没有舞种行时 `genre_header_template=null`；有则必须明确原文模板（含结尾换行），不会用LLM猜或重写。动作窗必须是已有权威边界，正文行首时间标记目前只接受 `[数字-数字s]`，显示值与权威时间默认最多相差 0.001 秒。`raw` 模式同样检查这个源合同，**不接受任意无时间戳 prose**。保留 CRLF、Unicode、动作空白与真实重复。

尽量输入整个已核验 split 清单，再用固定 case IDs 选 256 项。若只输入 val，隔离检查只覆盖已提供的 val，无法发现未提供的 train/test 重复。已知 source/song/近重复/文件 SHA/PCM/规范化路径跨 split 会拒绝；同 split 的关联合并成组件，不能互作负例。相同 PCM 的多个 caption 保留，不删成一个标签；此版本保守排除其负例资格，尚未实现完整多正例检索评价。

## 2. 冻结诊断，不调用模型

`fixed_val_ids.json` 是原固定监控验证 ID 的数组，不重新抽样取代原 256 项。困难配对 `hard_pairs.json` 示例：

```json
[
  {"case_id":"sample-a", "donor_id":"sample-c",
   "mismatch_review":"unreviewed", "review_evidence":""}
]
```

`hard` 表示操作员预选的相近节奏/风格候选，代码不从舞种或 BPM 自动推导其语义不匹配。只有实际听音/动作审查后才能填 `reviewed_mismatch` 和证据。已知多正例关系应先并入来源/近重复关联，不能只因 ID 不同声明错误。随机候选始终为 `unreviewed`。

```bash
mkdir -p .captionlab-runs/probe-v1
python -m captionlab prepare --sources /data/sources.json --case-ids /data/fixed_val_ids.json --hard-pairs /data/hard_pairs.json --target-mode raw --max-cases 256 --seed 42 --out .captionlab-runs/probe-v1/manifest.json
```

未给 `--case-ids` 才按 seed+ID 的 SHA 顺序选最多256项 val。随机及困难候选必须与真音乐相同采样率、采样数和**已声明**补零布局；未知补零仍标记未知。同一 case 的 random 与 hard 不能复用同一 donor。没有合适 donor/没有提供 hard 时，保留 `unavailable_conditions`，不凑伪负例；真实分数缺失也不会补零。`forward_calls_per_checkpoint` 给实际可执行调用数，上限为 256×4=1024。

## 3. 运行原检查点的 teacher forcing

**本分支的旧 NativeProvider 不兼容报告中所有 main/未同步运行合同。** 因此 `run` 接受一个显式的本地 `load_runtime(**kwargs)` factory；必须沿实际训练快照构造原 adapter，不能随意换本分支的旧 loader。factory 返回：

```python
return {
    "adapter": original_adapter,
    "receipt": {
        "protocol_id": "the-exact-original-input-target-contract",
        "checkpoint_mode": "portable",
        "target_contract_verified": True,
        "portable_tensors_verified": True,
        # 附真实加载审计的来源/成员/张量核验记录，不以占位True代替核验。
    },
}
```

adapter 需提供 `collate([row])`、`_move(batch)`、`eos_token_ids()`、`model`。原运行环境的 factory 负责必要的 schema wrapper、依赖、基座/portable 加载与准确恢复回执；`target_contract_verified` 包括完整目标、原始分段token边界、无静默截断、音频prefix不含目标caption/真实舞种。当前通用内核无法独立证明这些模型特定事实。只有收到实际训练快照后才能完成这最后一层接线；**仓库没有把 6040 检查点宣称为已加载通过**。

runner提供 `audio_duration_seconds=num_samples/sample_rate`，以及含 `start/end/caption` 的 `target_windows`；只有metadata的caption按原解析器规则strip，`target_text`字节不变。main的v2 `ConfiguredPlannerAdapter`还要求其原舞种header、三位小数时间标记等格式；`actions_timed/actions_untimed`不符合原v2，必须是另行验证的新目标factory，不能认为补齐metadata便自动兼容。MusicFM缓存融合也不在本轮普通Captioner桥的支持范围，不能随音频更换继续使用原特征。

`runtime.json` 的字段：

```json
{
  "schema": "captionlab.runtime.v1",
  "protocol_id": "the-exact-original-input-target-contract",
  "target_mode": "raw",
  "checkpoint_mode": "portable",
  "factory": {"path":"/run/original_factory.py", "sha256":"ACTUAL_SHA256", "kwargs":{}},
  "artifacts": [
    {"role":"model", "path":"/model/ACTUAL_WEIGHT_FILE", "sha256":"ACTUAL_SHA256"},
    {"role":"tokenizer", "path":"/model/tokenizer.json", "sha256":"ACTUAL_SHA256"},
    {"role":"runtime", "path":"/run/original_adapter.py", "sha256":"ACTUAL_SHA256"},
    {"role":"checkpoint", "path":"/run/checkpoint/ACTUAL_ADAPTER_FILE", "sha256":"ACTUAL_SHA256"}
  ]
}
```

`artifacts` 可有同 role 多条，应列全实际消费的权重分片、配置/processor/tokenizer、adapter模块/权重、训练快照依赖；相对路径相对runtime.json。代码逐条核 SHA，**不能自动证明操作员没有漏列文件**。factory 是可信本地程序，不是从数据文本提取的脚本。基座对照使用 `base_only`，不列 checkpoint；换checkpoint/合同必须新config及输出目录。

```bash
python -m captionlab run --manifest .captionlab-runs/probe-v1/manifest.json --runtime-config /run/runtime.json --out-dir .captionlab-runs/sft6040-probe-v1
```

内核用原 collate 的有效 labels 构造连续 assistant 后缀，检查单EOS、label与输入一致、完整上下文、因果shift；三条件逐token一致才接受。`eval/no_grad/use_cache=False`，每次最多32个有效位置转FP32计算CE；仍可能保留模型原生完整logits，不承诺低显存到任意卡。没有 `generate`，没有反向、训练或自动重试。静音使用同几何全零波形，沿原处理器重新取特征；音频字节冻结到新的run目录，记录实际SHA，不按相同placeholder IDs复用音频/KV。

写入 `manifest.json/runtime_audit.json/load_receipt.json/scores.jsonl/summary.json`。失败保留已成功的逐条记录、`failure.json` 和含全分母的 `failure_summary.json`，不自动resume。已有输出路径拒绝覆盖。

## 4. 汇总与解释

```bash
python -m captionlab summarize --manifest .captionlab-runs/probe-v1/manifest.json --scores /run/scores.jsonl --out /run/summary-recomputed.json --resamples 2000 --seed 42
```

也可汇总外部原生评分器导出的记录。每行字段为 `case_id, condition_id, target_sha256, target_token_ids, token_nll, eos_token_id, model_identity, tokenizer_identity`；目标数组不含padding，包含末尾EOS，NLL同长度有限非负。可选 `action_mask` 是同长度bool列表，EOS=false；同case各条件完全一致。缺失/全false是NA，不能当动作分数0。原生目标分词无法确认时不得猜mask。

主要报告 `mean_NLL(candidate,y)-mean_NLL(true,y)`，正值表示正确音乐在**该目标**上的条件似然更高，不表示唯一正确编舞。`all_assistant` 与显式动作子集分开；不把原目标删除header/时间戳后的分数冒充只改变了loss mask。重采样以来源组件为单位，先组内平均再等权跨组；单组不给CI。该区间条件于冻结donor集合，共享donor及未发现同歌关系仍限制独立性；若要推广到候选总体需重复donor选择。

排名保留并列；未审错配/缺候选/缺记录逐项呈现。已审subset的统计只用于该subset。**低NLL、击败静音、EOS正常、候选排名、distinct-n都不单独构成舞蹈质量验收。** 对文本动作内容、次数、方向和真实音乐贴合仍需要独立审阅。

## 5. 后续目标/完整短窗准备

```bash
python -m captionlab targets --sources /data/sources.json --keep-timestamps --out /data/actions-timed.json
python -m captionlab targets --sources /data/sources.json --out /data/actions-untimed.json --crop-out /data/complete-short-window-plan.json
```

舞种按显式模板移入JEV标签，unknown或词表外不作为多数类监督；标签本身仍是来源声明。去时间戳仅移除行首marker，正文、顺序、空白和重复不改；原件与char/UTF-8 byte spans可回溯，`restore_raw`核验。

短窗只取原标注完整1–3秒窗，输出采样索引（nearest、半值向上）、原/绝对时间、历史补零交集、上下文依赖、继承split与原音频身份；**只生成清单，没有裁剪音频，没有接通新的短窗训练器/JEV头**。crop样本仍需实际裁剪/核验、上下文依赖处理、曝光预算冻结。新目标不是当前检查点原合同，不能直接将其teacher forcing解释为重新训练后效果。

## 6. Cache数值校准与原ZeRO/portable一致性

参见[实测报告38](../notes/38_captioner_numeric_parity_20261109_zh.md)及[脱敏数值结果](numeric_calibration_results.json)。本轮同输入、同FP32参数存储下，原三例30个所选位置logits和1715个监督目标token概率在ZeRO/portable间逐位一致；当前单卡BF16存储及cache/full的任意形状比较不能据此宣称普遍逐位等价。

新增工具独立于现有 `prepare/run/summarize`，不更改其评分合同，不修改原模型库或训练器：

- `cache_calibrate`：重复full、短前缀full、重复cache、输入精度控制。
- `zero_parity`：原8-rank完整ZeRO checkpoint副本恢复、前向及优化器状态不变检查；可增加FP32参考。
- `portable_storage_parity`：多卡同FP32存储、BF16/FP32计算与固定目标cache控制。
- `portable_frozen_inputs`：冻结原feature重放、单卡轻量dtype干预；这些候选在本轮未通过，不能当部署修复。
- `moe_route_calibrate`：观测near-tie路由切换与参考expert IDs诊断干预；干预不可用于实际生成。
- `numeric_report`：从本轮完整私有证据目录汇总限定范围结论；GitHub脱敏JSON不能替代那些tensor。

```bash
python -m captionlab.cache_calibrate --help
python -m captionlab.zero_parity --help
python -m pytest -c captionlab/pytest.ini tests/test_captionlab_numeric.py -q
```

上述GPU入口需要匹配原训练的CUDA环境、完整官方模型、原代码快照和已审核factory/工件清单；本仓库不下载模型、不提供未经鉴权的第三方runtime。不自动分配GPU、释放holder或修改原checkpoint。执行前由操作员提供独立输出/checkpoint副本并核查资源；不要在共享训练环境安装CPU Torch覆盖CUDA依赖。

数值工具现会在ID派生文件读写前拒绝路径分隔、Windows保留名称和大小写冲突，合法Unicode ID保持原样。`numeric_report`要求实际三个源快照及三案例×两精度的唯一完整记录，核验非空支持、布尔声明与有限差值；重复控制由指标导出。汇总器未直接核验的恢复、资源及操作行为保留为声明或null，不能当独立认证。已发布数值JSON保留原记录，不由新汇总器改写；实现说明见[最新审查与推进文档](../notes/39_bounded_music_reinspection_and_revision_20261110_zh.md#7-最新代码审阅发现的校准工具问题)。

## 7. 已完成无时间戳CE/JEV训练对照

[服务器训练报告41](../notes/41_captioner_untimed_ce_jev_pilot_20261110_zh.md)记录五组2k训练/256验证、各4 epoch/252步的完整结果，附[脱敏预算和指标JSON](untimed_ce_jev_pilot_results.json)。两组纯动作CE去掉JEV后，末4层三例EOS为1/3、末16层0/3；去JEV没有消除病理循环，JEV分类本轮等于多数类基线，模型质量未通过。

该结果不是本目录CLI自动启动了训练：实际运行复用了main@02f478a的已验证独立训练快照和私有运行桥。此次提交只更新结果/文档，不包含原始模型、音频、caption、tensor或未提交训练实现，不修改harness。动作闭集词复用和真实重复不判错；未知标签和未审随机候选的评价边界保留。

## 8. 标注 JSONL 的 CPU 审计

`label_audit` 对本地原交付文件做流式统计，帮助核对分母和来源字段。对应[最新 main 标注审查与实验方案45](../notes/45_latest_main_caption_audit_and_next_experiments_20261110_zh.md)。不加载模型，不修改标签或划分，不按词汇指标选择训练样本。

```bash
python -m captionlab.label_audit \
  --input /data/qwen35_captions/train_captions.jsonl \
  --input /data/qwen35_captions/valtest_captions.jsonl \
  --out /data/audits/qwen35_labels_v1.json
python -m pytest -c captionlab/pytest.ini tests/test_captionlab_label_audit.py -q
```

输入为每行一个JSON对象，预期字段 `id`、`split`、`source_group_id`、`caption_version`、`motion_caption`，可选 `windows[*].caption`。缺失和坏字段逐项统计，不推测别名；split作为原始非空字符串保留，不把`valtest`静默改成`val`或`test`。每条物理行均进入总行数，包括空行、坏JSON、非法UTF-8和非对象；正文与窗口特征各自给出可用分母。所有输入只读，输出必须是新文件。

报告包含输入SHA256/字节数/行数、唯一及重复ID、跨声明split的ID/来源组冲突、原caption版本计数、正文字符范围、全文和逐窗口精确句重复、跨窗口重复、8-gram与正确归一化的动作词命中密度。逐字节扫描前后检查文件身份，不一次加载整份多GB标注；内存随单条记录和ID/来源索引增长。句切分、大小写和分母定义写入报告；这些数值不与旧 `caption_tier.py` 的指标混称。

`audit_complete=true`只代表扫描完成，`quality_verified=false`、`training_approved=null`始终保留。字符范围不是语言鉴定，英文词正则不覆盖全部语言；缺乏英语词不等于差标签。重复动作可能合理。工具不核验视频事实、源事件保真、窗口时钟/覆盖、兜底来源、真实歌曲隔离或实际训练消费关系，也不输出A/B/C质量等级。source_group声明相同/不同均需结合外部音视频证据解释。

服务器回传摘要可用于继续分析；完整报告含输入本地路径，保持在实验目录。不要为跑工具下载或覆盖共享模型依赖。
