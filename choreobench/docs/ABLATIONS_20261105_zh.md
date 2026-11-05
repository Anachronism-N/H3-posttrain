# 冻结模型的大纲与历史消融实验

本轮继续复用已有局部候选和冻结文本服务，不训练、不重新采集音频候选。新增代码把大纲/历史消融、完整请求大小检查和排程计数接入实际harness，供下一轮独立评审使用。没有在本机执行真实GPU或付费API实验。

**四组实验已具备实际运行入口**

`coordinated` 新增严格布尔配置use_outline和use_history，默认均true；非coordinated模式显式提供这些字段会报错。四组仅改变这两个开关，复用同一dataset、候选、公共结构、模型配置、时间策略、seed、输出预算和max_calls。

| 组名 | 全曲大纲 | 已提交历史 | 每首预期新文本调用 |
|---|---|---|---:|
| main | 有 | 有 | U+1 |
| no_outline | 无 | 有 | U |
| no_history | 有 | 无 | U+1 |
| no_outline_no_history | 无 | 无 | U |

U为实际规划块数；已知源失败或整块时长不可行的歌曲保留失败分母，不产生这些生成调用。四组block call_id与派生seed保持一致。相同seed和预算上限不等于相同token账单；大纲关闭少一次调用，历史关闭减少上下文，必须报告实际使用量。

no_outline不会发起大纲请求，也不把生成的段落意图或主题定义送入块请求。控制器保留明确标为deterministic_empty_placeholder的内部结构，motifs为空、motif_id仅允许null；计划provenance和outline_metadata明确是否由模型生成。因此这项消融同时移除了主题规划能力，不能用主题ID数量比较质量。

no_history不构造或发送boundary_context、recent_committed_events、前块动作文本、主题使用计数、声明终态或source_prefix_sha256。控制器仍保存并验证完整前缀，但不把它交给模型。当前局部候选、公开结构及大纲仍按其他开关提供；重叠观察窗本来就可能包含邻近时间的局部文本，关闭的是提交历史，不是删除这些共同证据。

**准备实验不调用模型**

在实际运行服务器、实际路径和冻结代码下准备；输出目录必须全新。若已有候选库，下面不需要先部署音频模型。

```bash
python -m choreobench prepare-ablation \
  --dataset /data/choreobench/dataset.json \
  --candidates /data/choreobench/local-candidates.json \
  --config /data/choreobench/coordinated-grid.json \
  --out-dir /runs/outline-history-v1
python -m choreobench.experiments --verify /runs/outline-history-v1
```

基础配置参考 [harness_coordinated_grid.json](../examples/harness_coordinated_grid.json)，提前填写实际文本服务，并按需要设置public_structure_path。准备函数不实例化provider，不访问服务health，不读取密钥值，不运行推理或训练。

输出包括：

```text
configs/main.json
configs/no_outline.json
configs/no_history.json
configs/no_outline_no_history.json
commands.json
manifest.json
```

runs下各独立输出路径只记录映射，不预先创建。manifest绑定dataset、音频、候选、配置和可选sidecar的字节SHA、Python执行路径/版本及当前choreobench源码闭包；commands使用argv数组，不拼接shell命令。verify在每组执行前重新检查来源、配置、源码、运行环境和命令清单。

prepared只表示准备产物已发布；executed、results_verified、model_or_service_verified均为false。ready_to_run仅要求已知输入准备有效、max_calls足够，不表示服务可用、模型身份正确、上下文可容纳或质量已验证。预算不足或已知歌曲准备失败时，CLI仍保留清单并返回exit2，不自动扩大预算或删除歌曲。

**按需执行一组，再决定是否执行其余组**

可以使用commands.json的argv列表，执行前先运行其中preflight_argv。以下示例只运行main，不会自动运行全部四组：

```python
import json
import subprocess
from pathlib import Path

root = Path('/runs/outline-history-v1')
commands = json.loads((root / 'commands.json').read_text(encoding='utf-8'))
subprocess.run(commands['preflight_argv'], cwd=commands['cwd'], check=True, shell=False)
selected = next(row for row in commands['runs'] if row['variant_id'] == 'main')
if not selected['ready_to_run']:
    raise RuntimeError('Prepared experiment is not ready')
subprocess.run(selected['argv'], cwd=commands['cwd'], check=True, shell=False)
```

这一步才会实际调用文本服务。供应方身份、费用、模型上下文和JSONschema支持需要真实验收，密钥仍通过配置指向的环境变量提供。四组共计约4U+2S次文本调用，其中S为可准备歌曲数；局部候选只采集一次，不能把原采样账单重复四次或记成免费。

代码、来源或配置变化时重新准备新目录。实验preflight只在执行前检查，不是模型服务认证或防止检查后文件变化的不可变执行环境；各run仍有自己的contract、来源复检和journal。同一组仅可在完全相同条件下--resume，不自动重发未知结果。

**输入大小检查覆盖结构化解码**

max_request_characters现在限制整个内部harness request的compact JSON Unicode字符数，包含messages、response_schema、call_id、seed和输出预算等字段。超限在该请求写入调用journal和发送模型前拒绝，不截断文本或缩减schema；此前完成的大纲/块调用和成本仍保留。

free模式密集网格可能展开大量anyOf分支。构造器先用完整最小分支建立保守大小下界，确定超限即在大规模复制前拒绝；最后再检查实际完整request。它不误把字符数当作token、UTF8字节、经过provider转换的HTTP体或后端语法编译资源上限。真实服务仍需tokenizer/上下文验收。

**排程诊断补齐实际策略与饱和程度**

每首diagnostics新增schedule_diagnostics；event_timing_policy也明确记录event_schedule_policy。报告逐块实际事件数量、允许菜单数量、轨道计数、最短/中位时长、配置上限饱和率，以及balanced菜单中可用最大数量的饱和率。

配置最大值与短块实际允许数量分开：某块只能容纳1事件，选择1事件不能算达到配置上限3。free模式不枚举可行数量，allowed_event_counts记null，不把配置范围当可行性证明。跨块事件或不完整单位分区会报错，不静默漏计。无事件的块相关时长为NA。

这些都是字面诊断，不能当作音乐性、动作自适应或物理质量评分。它们用于定位上一轮75/76块选满3事件这类现象，然后由独立评审判断是否合理。

**下一步验收**

先在现有冻结候选上选main和一个有针对性的消融，按相同排程约束输出盲评包。记录实际模型身份、全部失败、输入/输出token、延迟和原候选成本，评价音乐贴合、长程结构、复读和过渡。完整四组可按预算随后执行，不能凭软件fixture结果宣称大纲或历史有效。

同排程下有无局部音频候选、节拍引导的非均分菜单、自动语义critic和修复仍是后续工作，本次没有实现或声称验证。现有base-only候选也可用于这些已实现消融，不以新增训练或找到checkpoint作为前置条件。

## Linux 首轮真实执行：main 对 no_history

2026-11-05指定分支快进到658ffe4后，在独立工作区冻结代码和原双窗解析器，用已有三首完整录音的原始Omni候选准备实验。只执行main与no_history；no_outline及组合组仅生成配置，未在本轮发请求。

两组均开启大纲，采用同一VL32B BF16 TP=2纯文本服务、0.25秒网格、0.5秒下限、balanced_full_unit、每块至多3事件、同一seed/候选/公共结构/输入字符与输出预算。按每首实际单位数27/26/23，各组预登记28/27/24次请求，两组合计158次上限；原局部278次调用仅计一次，本轮新增音频采样0。

新版Linux完整软件回归为712 passed、148 subtests passed。prepare-ablation与每组preflight在冻结代码及相同Python执行路径下通过，manifest保留prepared/executed=false原状态；真实执行另存各run contract/results和联合回执，不把准备清单改写成成功报告。加载前一次核验文本权重和tokenizer，与旧模型字节一致；health只是本地服务声明，不当作远程tensor认证。

比较时同时核验两组大纲原内容是否相同、公共来源及块seed是否一致，no_history模型请求不含boundary_context/recent_committed_events。所有失败和未决结果保留，不能把前缀当整曲。

真实执行六条完整合法计划、158次文本请求，新增音频调用0。三首每组事件数均为81/78/69，每组76/76块达到配置上限3事件；正文不逐字相同。本文的饱和率仅是字面诊断，不证明动作密度合理或历史有效。

**实际不是严格的历史单变量内容对照。** 三组纲要的内部请求与HTTP请求逐项相同（包括temperature=0和seed），服务声明也相同，但三组生成纲要内容均不同：第一首motifs相同、section意图不同；另两首两部分均不同。配置只改变use_history，不保证生成的中间产物相同。没有筛掉不一致配对、重采大纲或把差异归因于某个确定的底层原因。干净的大纲内容控制配对为0/3，因此此轮是执行/诊断与配置对照，不足以独立证明历史造成音乐、复读或过渡差异。

后续若要隔离历史，应先冻结并验证同一份公共大纲，再将该份大纲实际提供给两组；固定seed或独立再调用一次不等价于固定内容。当前prepare工具仍为各组独立纲要生成，本轮没有宣称已经接通共享纲要输入。冻结共享大纲需新合同/目录和验收，不能通过修改旧日志或私有评分反选大纲实现。

六条匿名方案已部署独立3case评审页，HTTP/Range/Chromium/空评分及实际GPU恢复通过。人评仍pending，详见 [VALIDATION_20261104.md](VALIDATION_20261104.md) 第14节。没有基于自动指标选择赢家，no_outline和组合组本轮未执行；下一步先收集这些现有方案的独立听音问题，再准备严格内容控制对照。
