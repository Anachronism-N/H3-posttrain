# 小节编舞四方法实验的服务器入口

完整设计见 [实验规划](../../notes/60_bar_window_global_style_agent_experiments_20261115_zh.md)。本目录提供参数计划、部署模板和新代码入口。此次仅编写代码并做静态检查，**没有运行测试、训练、模型推理或实验**；需要在服务器完成真实音频消费检查。

## 已提供的代码

| 功能 | 入口 |
|---|---|
| Beat This 真实 downbeat 的 2/3/4 小节分窗 | `python -m choreobench.bar_windows` |
| 变长原生专家候选采集与缓存 | `python -m choreobench.untimed_catalog --windows-manifest ...` |
| 22 类输出、15 训练支持类的全局 JEV 导入 | `python -m choreobench.global_style` |
| 四方法矩阵及复用稿消融 | `python -m choreobench.bar_agent_experiment` |
| 全曲三维和局部六维音频评审 | `experiments/structured_dev_20261114/judge_rubric_v2.py` |

新候选采集只支持 `kind=native`、`audio_contract=variable_duration_v1`，一次进程加载一次专家。旧 `native_service` HTTP 合同仍固定时长，不能用于新小节主实验。外层文本规划仍可使用原有已部署服务。

## 1 准备真实检测和共享窗口

复用已有、与完整歌曲音频 hash 相同的 Beat This 检测。若是数据构造的原始格式，先转换；不要将 librosa 输出混入此入口。

```bash
python -m choreobench.music_structure \
  --dataset "$DATASET" --detections "$BEATTHIS_DETECTIONS" \
  --out "$BOUND_DETECTIONS" --wrap-legacy \
  --detector-identity "$DETECTOR_IDENTITY" \
  --synthetic false --legacy-time-unit seconds --run-receipt "$BEATTHIS_RECEIPT"

python -m choreobench.bar_windows \
  --dataset "$DATASET" --detections "$BOUND_DETECTIONS" \
  --out "$BAR_WINDOWS" --target-seconds 5.0625 --max-window-seconds 20
```

检测时钟与 dataset 必须相同。窗口包含真实完整小节和明确的首尾残片，超过容量时报错，不静默回退到固定秒数。

## 2 生成一次公共候选

复制已验证的 native provider 配置，增加 `"audio_contract": "variable_duration_v1"`。基础模型使用不含固定 `[0–2s]/[2–5s]` 的 untimed 动作提示；训练 checkpoint 使用其训练来源和原任务提示，不擅自覆盖。模型 checkpoint、加载器、推理精度和解码参数保持可追溯。

```bash
python -m choreobench.untimed_catalog \
  --dataset "$DATASET" --provider "$VARIABLE_NATIVE_PROVIDER" \
  --windows-manifest "$BAR_WINDOWS" --out-dir "$CATALOG" \
  --seed 42 --candidates-per-window 2 --max-output-tokens 1024
```

所有四方法的 source 都使用同一个 `$CATALOG/results.json`。先在开发材料验证 processor 实际使用完整波形；文件长度回执不等于模型内部未截断的证明。不能仅把 readiness 改为 true 就代替检查。

## 3 全曲舞种和声学摘要

用已部署 JEV 导出原生概率，整理为 `{"songs": [...]}`。每首包含 `song_id/audio_sha256/sample_rate/num_samples/windows`；每窗包含整数 `start/end`、音频 hash、22 维 probabilities，可附 `ownership_start/ownership_end`。模型身份字段见 `choreobench/global_style.py:validate_identity`，必须来自真实权重，不填猜测的 SHA。

```bash
python -m choreobench.global_style \
  --dataset "$DATASET" --predictions "$JEV_NATIVE_PREDICTIONS" \
  --model-identity "$JEV_IDENTITY" --out "$GLOBAL_STYLE_PRIOR"

python -m choreobench analyze-music \
  --audio "$CANONICAL_SONG_WAV" --out "$SONG_MUSIC_SUMMARY"
```

导入器不执行 JEV 推理，不更改概率。JEV 内部可保留已验证的原生短窗，全局按覆盖时长聚合，编舞边界仍来自小节。旧七类先验也可读取，但不能与新模型逐歌混选。`presentation.txt` 开头只有一次 `Dance style is xxx.`；事件正文不含固定标题。

## 4 冻结 inventory 并运行四方法

复制 [inventory.example.json](inventory.example.json) 到集群运行目录，填写 source、style_prior 和每首 music_summary。先用三个开发样本；正式新留出改为 `panel=new_holdout`、12 首及对应 `expected_songs=12`。清单中每个路径可相对 inventory 文件或使用绝对路径。

```bash
# 只检查输入和登记矩阵，不调用任何模型。
python -m choreobench.bar_agent_experiment \
  --inventory "$INVENTORY" --provider "$OUTER_PROVIDER" \
  --out-dir "$MATRIX_OUT" --with-ablations --dry-run

# 仅在服务器完成真实 preflight 后执行。
python -m choreobench.bar_agent_experiment \
  --inventory "$INVENTORY" --provider "$OUTER_PROVIDER" \
  --out-dir "$MATRIX_OUT" --with-ablations --execute --resume
```

注意：若 dry-run 后修改了 inventory/readiness，输入身份已经变化，必须用新的输出目录重新 dry-run，不能对旧目录 `--resume`。`protocol.json` 是规划，不是 inventory；空部署字段会明确失败。

`--with-ablations` 的四次物理运行是 W+R、SF、RA、O；逻辑主结果 W 使用 W+R 的改写前稿及其截止成本，W-plusR 使用后稿。O-noR 同样取 O 的原稿。因此不会为了消融再次生成相同 W。零重复也保存前稿。真正生成失败和未生成的消融结果全部登记。

导出位置：

```text
matrix_contract.json                 冻结输入与代码身份
registry.json                        每个注册格的终态、失败和逻辑结果
runs/{W,SF,RA,O}/{song_id}/           原始请求响应和计划
review/pairs.public.json             匿名 O 对三个 baseline 的配对
review/bindings.private.json         方法映射，仅分析端持有
review/whole3.cases.json              全曲音频三维评分输入
review/unavailable.json              无完整计划的注册项
```

**不要把 `runs/W/.../plan.json` 直接当 W 主结果**：开启消融时它是 W-plusR 后稿。使用 runner 导出的逻辑结果和 review 包。统计总物理耗费时，不把共享一次执行的 W 和 W-plusR 重复相加。

恢复只跳过已登记终态，失败不会自动换 seed 重跑。存在未登记的中断目录时保留并报错，先审查其请求记录；不能删掉重跑来隐藏已花成本。单个 matrix 目录有进程锁。

## 5 音频评分

全曲三维入口要求完整音频，服务必须先证明能接收完整歌曲：

```bash
python experiments/structured_dev_20261114/judge_rubric_v2.py \
  --cases "$MATRIX_OUT/review/whole3.cases.json" --rubric whole3 \
  --base-url "$AUDIO_JUDGE_URL" --model "$AUDIO_JUDGE_MODEL" \
  --out "$WHOLE3_SCORES" --seed 42 --max-tokens 4096
```

局部六维使用同一入口的 `--rubric local6`。先按规划为每首生成固定前/中/后三个 20 秒 WAV 和对应 cases；当前 runner 不自动裁这些 WAV。cases 必须保留全曲偏移，D6 适用性从不看候选的音频观察预先确定。

```json
{
  "cases": [{
    "case_id": "anonymous_case_1", "song_id": "song_1", "time_unit": "seconds",
    "events": [{"event_id": "e1", "start": 60, "end": 64,
                "track": "body", "action": "A concrete movement description."}],
    "audio_path": "/data/song_1_60_80.wav", "audio_sha256": "FILL_REAL_SHA256",
    "audio_global_start_seconds": 60, "audio_global_end_seconds": 80,
    "D6_applicable": false,
    "D6_applicability_reason": "Frozen audio-only observation: no clear development event."
  }]
}
```

每维返回 score/status、引用原句、event_ids、reason 和音乐证据。服务原始响应先保存到 `.calls.jsonl`，解析失败不丢弃。音频随请求实际发送，但该事实仍不证明服务内部完整编码。全曲模型不支持长音频时不能用局部片段冒充 whole3；规划中的分层音频评审尚需单独编排，先报告局部结果及覆盖。

旧四维评审脚本亦已修正全文、秒单位、匿名、分数验证及原始记录；它们仍是四维历史协议，不与新九维汇总混算。最终歌曲级 bootstrap/Holm 和全维论文汇总仍按规划执行，不把旧四维总分直接换名。

## 6 交付边界

代码已写，真实变长泛化、服务音频消费、部署内重试与真实物理请求成本、GPU兼容和实验效果均未在此机器运行验证。现有 controller 的账本按 dispatch 记录，provider 若内部重试需额外记物理请求；不把目标预算协议当成已验证服务行为。

建议服务器先运行新增测试：`test_choreobench_bar_windows_v2.py`、`test_bar_agent_experiment.py`、`test_harness_evaluation_contract.py` 及全局风格/共享证据回归测试，再做开发 preflight。正式输出目录和评测文件不得覆盖旧结果。
