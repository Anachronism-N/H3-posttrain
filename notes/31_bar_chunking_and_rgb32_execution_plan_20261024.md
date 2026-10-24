# 按小节构造编舞标签与 RGB 32 关键点实跑方案


> **后续集群实证更新（2026-10-24）**：本篇初始“32点尚未实际运行”状态已被后续产物超越。cached-box直接32点及必要画内支持对照已完成；新fresh-detector入口也已真实完成三个case，每条81帧，由RGB新检测框并直接查询32点，不使用旧框或旧pose条件。原始数据、多个检测返回、最大面积选择及新增三栏叠加已独立验真，新坐标的whole／fixed动作caption与十二路必要支持控制也已完成。红衣画外足点说明finite valid不等于可见，尚未建立新增caption准确优势。按检测拍点C50已完成但不是小节首拍检测；真实BPM长度窗与BeatThis downbeat小节窗独立推进，不能用旧C重命名替代。

日期：2026 年 10 月 24 日。代码状态基线：`codex/motionscript-optimization` 的 `a34dc7023e543672898e6436dd1484ff98c52349`。

编舞规划器采用音乐输入、自然语言编舞 caption 输出。标注侧先识别音乐小节，再对相应视频窗口生成动作描述；规划器仍可读取完整音乐片段，渲染器读取参考图与编舞文本。

32 点实验补充的是掌面和足部的动作观测，音乐分段组织的是这些动作在时间上的描述。两项先独立比较，再考虑组合。用户在本篇初始计划时明确要求：**实际运行 RGB 32 点提取，不能继续只停留在代码和资产准备。** 后续真实终态见本文末的执行记录。 首轮按已有三个真实案例、单 GPU 顺序执行，再核对原视频叠加结果。

## 当前实现与待完成事项

| 工作 | 当前状态 | 下一步 |
|---|---|---|
| 原身体表示的整段与固定分段 MS | 已有实际执行与候选统计 | 核对正文质量和完整覆盖 |
| 按检测拍点分段 | 同 50 条对照已有报告；未识别小节首拍 | 与真正按小节的策略分开 |
| 按小节首拍分段 | 当前入口未实现完整流程 | 检测 downbeat，构建并共享窗口清单 |
| NLF 官方 canonical 资产和 32 点规格 | 已取得并记录兼容性核验 | 在实际服务器复核文件身份后使用 |
| 从真实 RGB 提取 32 点 | cached-box与新fresh-detector均已实跑3条 | 保来源分离，核验新框选择和新增点可靠性 |
| 32 点视频叠加及动作核对 | 新旧三栏叠加与fresh／cached十二路动作支持控制已完成 | 未建立准确优势；画内仍不等于可见，未批准掌足监督 |
| 新标签训练与 H3 跟随收益 | 尚未验收 | 用通过核对的小样本分开验证 |

MS 原身体分段处理、开放词汇 VLM 标注、NLF 资产准备和 RGB 32 点推理是不同任务。旧分段池处理完成、VLM 能生成文字或 32 点规格存在，都不能作为“服务器已经实际提取 32 点”的证据。

## 按小节分段的输入与时间映射

需要配对音乐、舞蹈视频、实际视频时间轴，以及音频与视频的时间对应关系。BPM 表示速度，beat 表示拍点，downbeat 表示小节首拍。不能把音量峰值、每四个拍点或视频第 0 秒直接认定为小节首拍。

优先在原始长音乐或较长上下文中检测小节，再映射回当前约 5.0625 秒视频。只有短音频时也可以预测，但必须允许小节位置不确定。

对于纯裁剪且未变速的配对媒体，可使用经核验的偏移映射：

```text
视频内时间 = 检测音频内时间 + 已核实的音画时间偏移
```

若在整首歌上检测，还要先减去当前配对音频在整首歌中的起点。变速、拼接或补帧可能需要更完整的时间映射，不能只套一个常数。当前已有记录中的音画偏移仍为 unknown，未知样本不能标成已经验证的小节对齐标签。

原始歌曲时间、配对音频时间和视频相对时间应分别保存；不要把 CSV 的声明时长当作实际媒体结束位置。

## 检测小节首拍

第一版可比较 Beat This 的拍点与首拍输出。其 `File2Beats` 接口返回 `beats, downbeats`。相邻两个可靠首拍构成候选小节：

```text
downbeat[k] → downbeat[k+1]
```

这不要求预先把所有歌曲都指定为 4/4。拍号及每小节拍数可用于核验、解释和细分；无法确定时允许保留 unknown。[Beat This 官方接口](https://github.com/CPJKU/beat_this)

Beat This 默认 minimal 后处理选择预测峰，并将首拍对齐到附近拍点，不能自动保证正确的拍号序列。其可选 DBN 后处理默认建模 3／4 拍长；复杂拍号、复合拍和变拍需另行核验，不能宣称一个默认配置覆盖所有音乐。[后处理实现](https://github.com/CPJKU/beat_this/blob/main/beat_this/model/postprocessor.py)、[madmom 首拍跟踪接口](https://madmom.readthedocs.io/en/v0.16.1/modules/features/downbeats.html)

至少保存实际拍点、首拍、模型与后处理版本、音频哈希、检测上下文、时间映射和回退原因。检测分数不是经过本项目校准的准确率。

## 小节层与标注窗口层

先构造不可混淆的小节时间轴，再决定真正送给描述器的窗口。原始小节边界始终保留，合并窗口不能反过来改写原始音乐结构。

假设视频长 5.0625 秒，首拍映射到视频中为 0.70、2.70、4.70 秒。原始划分为：

| 区间 | 类型 |
|---|---|
| `[0.00, 0.70)` | 首部残段 |
| `[0.70, 2.70)` | 一个完整小节 |
| `[2.70, 4.70)` | 一个完整小节 |
| `[4.70, 5.0625)` | 尾部残段 |

可以比较两种标注策略：

1. **严格小节策略**：保留上面四段，对短边缘段单独记录输出、无描述或弃权。只有中间两段是完整小节。
2. **合并短边缘策略**：把首尾短段分别并入相邻小节，形成 `[0.00, 2.70)` 和 `[2.70, 5.0625)`。窗口包含完整小节与边缘残段，不能再称为每段恰好一个完整小节。

固定约 5.06 秒、覆盖完整视频、每窗恰好一个完整小节，通常不能同时满足。首轮保留视频完整覆盖，并在元数据中明确窗口包含的原始小节和残段。

### 段长调整

默认一小节一段，约 1.5～4 秒作为需要验证的起始目标范围。四分音符计 BPM 的 4/4 音乐，在 60／120／180 BPM 时，一小节分别约 4／2／1.33 秒。

- 小节过短时，比较合并相邻两个完整小节。
- 小节过长时，在可靠拍点处细分；只有符合实际拍号时才称半小节。
- 首尾残段保留真实范围；需要合并时记录合并关系。
- 弱节奏、首拍不足、疑似漏拍或相位不确定时，回退固定时长，并使用独立策略标记。

不得仅依据全局 BPM 生成等距首拍，不得为了段长整齐伪造音乐边界。首拍不可靠与动作描述失败应分别统计。

## 用小节窗口生成 caption

完整姿态预处理一次，再按最终标注窗口切取坐标，逐窗运行 MotionScript。保留根位移、朝向、质量掩码及连续支持段；不在每个 chunk 中重置几何，也不删除缺失帧后拼接两侧动作。

用真实时间轴按 `[start, end)` 选样本，分别保存计划窗口与实际首末样本位置。视频末时刻不同于末帧采样时刻，不能互相替代。

原版 MotionScript 在本方案中仍不提供逐动作准确时间。窗口时间标记只说明描述来自哪里。描述器不额外读取窗口两侧动作后再把全部文字标回核心窗口；跨小节动作允许在相邻窗口描述各自部分，不强行补成两次完整动作。

最终 caption 使用连续自然语言，内部清单可采用结构化元数据。以下只是格式示例，动作必须由视频支持：

> [0.00–2.70秒] 舞者向右侧移动，同时抬起双臂，上身逐渐转向侧面。\
> [2.70–5.06秒] 舞者继续转动上身，随后向左回移，双臂逐渐放下。

按小节切分不证明动作踩拍。不要自动补写“每次动作落在鼓点上”，也不要把一个小节等同于一个完整舞步。

## 当前代码的接入位置

`ms_beat_chunk_compare.py` 当前使用 onset＋beat DP，选择 2／4／8 拍的窗口，并明确记录 `bar_or_downbeat_detector=false`、`bars_or_downbeats_verified=false`。它是按拍对照，不能改一个名称就当作按小节实现。

建议按以下顺序接入：

1. 音频分析新增首拍检测和可复用缓存，绑定音频身份、检测上下文和时间映射。
2. 保存原始首拍、小节与残段，再生成标注窗口及合并关系。
3. 扩展 `motionscript_v2/chunks.py` 的来源校验与窗口策略；现有策略只有 fixed／beat／explicit，不能假称已有 bar 策略。
4. 让原版 MS 入口和视觉观察入口消费同一份、与视频身份绑定的窗口清单。当前 MS 与 VLM 批处理分别使用自身的固定窗口逻辑，接入前不能假定它们已共享小节窗口。
5. 分别导出调用终态、正文覆盖和训练准入。部分窗口有文字不等于完整编舞 caption。

第一轮保留整段、固定时长、现有按拍对照，再增加严格小节和合并短边缘两组。固定姿态来源、规则与改写方式，比较动作事实、漏动作、跨界矛盾、完整覆盖和 H3 跟随，不用文字变短或重复指标下降代替准确度。

## RGB 32 点的目的与边界

| 点组 | 数量 | 用途 |
|---|---:|---|
| 身体锚点 | 22 | 腕、肘、肩、踝等共同参照 |
| 双手掌面锚点 | 4 | 每手两个掌面位置，与腕部形成掌面几何 |
| 双足表面锚点 | 6 | 每足脚跟与两个前足位置，描述足部局部朝向 |
| 合计 | 32 | 不以识别细手指动作或表情为目标 |

腕和肘的位置不变时，手掌仍可翻面；新增掌面点为这种变化提供显式几何依据。足部点用于足纵轴和近似平面，不直接证明承重或地面接触。

旧完整 `pose[72]` 已有腕、足旋转，模板可恢复部分朝向。应同时比较旧 body22、旧完整 pose／模板和直接 32 点，不能凭点数认定新信息更多或更准。

实际提取应把身体锚点与新增点一起从 RGB 查询，使用 `get_weights_for_canonical_points` 和 `estimate_poses_batched`。canonical 数组定义查询位置，逐帧结果由图像预测。不能用固定模板补十个点冒充新的图像观测，也不能在直接接口失败时静默改成 SMPL 拟合。

32 点的布局不同于旧原版 MS 的输入布局；提取成功后仍需独立的连续几何、细节算子与文本融合，不是把点数常量改成 32 就自动扩展原版动作词表。

## 已具备的服务器资产

官方 canonical 数组已取得，32 点规格已构建。兼容性核验涵盖实际模型内嵌的全部 55 个关节和 1024 个顶点，float32 对应值完全匹配；这解除此前规格准备的阻碍，不证明新增点预测准确。

| 资产 | 已记录位置或身份 |
|---|---|
| NLF 模型 | `models/nlf/nlf_l_multi_0.3.2.torchscript` |
| canonical 目录 | `models/nlf/canonical_assets_official_v020` |
| 32 点规格 | 上述目录中的 `body_palm_foot32.json` |
| 三个案例 | `wJPptuhclZmKAEWs`、`wJPrpFsThWAcBQIw`、`wJQTclrlOLEYYfTB` |

路径基于已提交服务器记录，实际执行前仍需核对文件存在、SHA、环境及输入配对。完整位置、数组／模型／规格 SHA 见 [官方资产兼容性记录](../results/nlf_official_canonical_match_20261024_v1.json)。该资产记录的 `real_rgb32_extraction_completed=false` 是资产核验当时的状态，不代表后续提取尚未运行；最新真实提取终态另行记录。

## 三案例真实运行安排

目标是实际调用 NLF，生成三份真实视频的 32 点缓存，并完成可回看的核验。先单 GPU、三个案例顺序执行，批大小 8、增广 3；不在第一轮扩展到全池。运行负责人指定服务器入口、可用 GPU 与现有 Python 环境，保留日志、退出码和资源身份记录。

执行前核对 PyTorch／Torchvision、NumPy／SciPy、FFmpeg／FFprobe；确认实际模型导出两个直接点方法。不要为通过接口检查更换成另一份未记录权重，也不要把 `--help`、CPU 载入或 mock 测试写成真实推理成功。

下面是现有 CLI 的执行模板。`RGB32_REPO`、`RGB32_PYTHON`、`RGB32_GPU` 需设置为实际值；其余服务器路径来自已提交记录，须在目标节点验证。模板没有在本次文档提交中执行。

```bash
set -euo pipefail
: "${RGB32_REPO:?设置实际仓库目录}"
: "${RGB32_PYTHON:?设置含所需依赖的Python绝对路径}"
: "${RGB32_GPU:?设置实际可用GPU编号}"

RGB32_ROOT=/apdcephfs_gy4/share_302533218/cedricnie
RGB32_MODEL="$RGB32_ROOT/models/nlf/nlf_l_multi_0.3.2.torchscript"
RGB32_SPEC="$RGB32_ROOT/models/nlf/canonical_assets_official_v020/body_palm_foot32.json"
RGB32_VIDEO="$RGB32_ROOT/OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/videos"
RGB32_BOXES="$RGB32_ROOT/diff/logs_omnidance/exp/MS/nlf_eval3"
RGB32_OUT="$RGB32_ROOT/diff/logs_omnidance/exp/MS/rgb32_fixed3_$(date -u +%Y%m%dT%H%M%SZ)"

cd "$RGB32_REPO"
mkdir "$RGB32_OUT"
"$RGB32_PYTHON" -c 'import numpy, scipy, torch, torchvision'

for RGB32_SID in wJPptuhclZmKAEWs wJPrpFsThWAcBQIw wJQTclrlOLEYYfTB; do
  CUDA_VISIBLE_DEVICES="$RGB32_GPU" "$RGB32_PYTHON" \
    scripts_omnidance/exp/ms/ms_landmarks.py extract \
    --video "$RGB32_VIDEO/$RGB32_SID.mp4" \
    --boxes-nlf "$RGB32_BOXES/$RGB32_SID.npz" \
    --spec "$RGB32_SPEC" --model "$RGB32_MODEL" \
    --device cuda --bs 8 --num-aug 3 \
    --ffprobe /opt/ms_tools/ffprobe --ffmpeg /usr/local/bin/ffmpeg \
    --out "$RGB32_OUT/raw/$RGB32_SID" \
    2>&1 | tee "$RGB32_OUT/$RGB32_SID.extract.log"

  "$RGB32_PYTHON" scripts_omnidance/exp/ms/ms_landmarks.py describe \
    --bundle "$RGB32_OUT/raw/$RGB32_SID" \
    --chunk-seconds 2 --min-tail-seconds 1.5 \
    --out "$RGB32_OUT/description/$RGB32_SID" \
    2>&1 | tee "$RGB32_OUT/$RGB32_SID.describe.log"
done
```

此处先复用固定窗口，隔离“新增点是否有用”的变量；按小节窗口完成并核验后，才使用现有 `describe --windows` 接口消费绑定视频 SHA 的窗口清单。独立输出目录已经存在时入口会拒绝复用，不得覆盖旧结果或把旧缓存改名当作新提取。每例单独记录结果；中断后先检查已完成产物，只对缺失或失败案例使用新的输出目录重试。

### 运行后的核验与视频叠加

每例至少检查：

- `raw.npz` 的三维点为 `[实际帧数,32,3]`，二维像素点为 `[实际帧数,32,2]`，逐点 uncertainty／valid 为 `[实际帧数,32]`，名称顺序与规格一致。
- 原视频、框缓存、规格、模型、原始坐标及 manifest receipt 的 SHA 对应；记录真实 PTS、首末范围、帧率、模型方法和运行配置。
- `smpl_fitting_used=false`、`template_fk_used=false`、`body_anchors_queried_together_with_extras=true`，来源为 `nlf_direct_rgb_landmarks`。
- 把直接输出的 `points2d_px` 叠加到实际原视频：身体、双掌和双足使用不同颜色，并显示名称；缺点、越界和异常跳点明确标记，不插值伪装为可靠观测。
- 回看翻掌、足部朝向、左右、遮挡和人物切换，与旧 body22／完整 pose 同窗比较。uncertainty 数值不是可见性或校准概率，不能机械套旧接口阈值。

现有提取／描述入口不等于已有专用叠加评审器；如没有对应工具，需要另行实现，不能记录一个不存在的 overlay 命令。先保留原始 2D 点和连续视频，叠加工具不得改变原始预测。

三个案例实际推理结束、产物完整、叠加可回看，是工程完成；掌足方向准确和动作有用，是后续语义判断。两者分别记录。若新增点不可靠，优先检查分辨率、遮挡和估计器，不能用更多模板点或更长文字掩盖问题。

## 后续训练的准入与实验顺序

先完成三案例 RGB32 真运行和核验，再扩大有限样本，比较新增细节的正确性与误报。按小节实验先保持旧坐标来源，与增点实验分开；随后才比较组合效果。

规划器仍可用完整音乐片段预测整段带时间标记的自然语言文本。若标签的小节相位依赖长音频、而学生只看短音频，应检查短输入是否足以推断该相位，必要时给规划器同样的音乐上下文。不要把不确定的小节定位当作短音频必然可以恢复的真值。

渲染器先使用真实视频支持的 caption 测试动作和粗时间顺序，再接规划器输出。原版 MS、直接 RGB32 几何和开放视觉描述保留独立来源；候选完成、全窗口正文覆盖、语义核对及训练准入分别保存。标签格式正确、退出码为零和资产就绪都不等于生成效果通过。

## 关联文档与实现

- [原始分段方案与执行补充](30_motionscript_chunk_annotation_plan_20261024.md)
- [32 点接口与表示说明](../scripts_omnidance/exp/ms/motionscript_v2/LANDMARKS.md)
- [32 点实际提取与描述入口](../scripts_omnidance/exp/ms/ms_landmarks.py)
- [现有按拍对照入口](../scripts_omnidance/exp/ms/ms_beat_chunk_compare.py)
- [原版 MS 固定分段入口](../scripts_omnidance/exp/ms/ms_chunk_caption.py)
- [窗口校验与坐标切片](../scripts_omnidance/exp/ms/motionscript_v2/chunks.py)
- [同 50 条整段与分段候选统计](../results/ms_beat_chunk_same50_verified_20261024_v1.json)

## 最新计划的实际推进与来源区分

- 远端 `3eabafb10745f55f5032b4623bac985dbe0f3e47` 已获取并安全合并。本地真实进度先提交保护，正常三方合并保留了notes30/LANDMARKS追加和用户其他未提交工作，未reset/stash或覆写在途Python源。新远端仅四份文档，没有新增detector32/BPM/bar运行代码。
- 旧模板实际direct32三case产物在 `MS/point32_direct_fixed3_v1`；官方canonical已bitwise兼容，提取/2D叠加/旧pose72及24父链对照/必要mask原文与源SHA保存。必要控制阻止了红衣画外足面事件，但画内合掌/鞋跟遮挡疑点仍未解除，不以工程成功批准细节监督。
- 新独立NLF32入口 `ms_nlf_rgb32.py` 已在N2 GPU2真实完成固定三case：actual `get_weights_for_canonical_points(32)`＋`detect_poses_batched(images, weights32)`同时返回新人物框与32点，不再以旧box/pose缓存作模型条件。crop features对查询点共享，但detector是另一个网络、aug3及内部batch仍有额外计算，不能称整个过程只有一次网络forward。先新fixed3与旧cached-box32对照，再扩大有限样本；不联SMPL-X1079拟合或把SMPL旧24与新SMPL-X布局混淆。
- BPM长度策略单独按实际估计 `60×拍数/BPM` 构造标注窗口并调用描述器，首0属于长度近似而非小节首拍；必须标tempo_grid、beat_phase_verified=false。此前C脚本只将该策略存在comparison字段，没有真正生成这些BPM窗的MS正文，新增真实运行不能用C50重命名替代。
- 真正小节策略将读取实际BeatThis等模型的beats/downbeats，保原始小节与首尾残段，分别执行strict_bar与merge_short_edges；MS和视觉源消费同一份videoSHA/实际PTS/音频SHA绑定窗口清单。未知音画偏移保为假设/未验证，不因为未知而伪造barphase，也不让缺downbeat模型阻碍独立BPM近似小试。
- 在途12k固定窗MS/32B视频候选作为现有基线继续完成，所有Python依赖SHA保持；新方法用独立源码与输出目录。新增32点、BPM长度、检测拍点、downbeat小节是不同轴的对照，不能同时换多项后只归因为“增点”或“按BPM”。

### Fresh-detector32真实执行与独立原始数据核验

新产物：`MS/nlf_rgb32_fresh_detector_fixed3_v1/finished.json`，schema为`nlf-fresh-detector32-terminal-v1`，`complete=true`、`expected=actual=3`。入口源码SHA256为`8875af505ead1e86313bcafe4aaf18f76c87eac0287d6dd1a2c0877e61c985d5`，固定三case输入SHA为`14a38a179bc0a1f0afc4fd9d6eef6e74797564db696cad20503e495b57d34361`。原模型、canonical规格与旧接口依赖未改变。

| Case | 实际帧数 | 有检测帧数 | 返回多个检测的帧数 | 原始numeric valid点帧 | 画内必要支持点帧 |
|---|---:|---:|---:|---:|---:|
| wJPptuhclZmKAEWs | 81 | 81 | 0 | 2592 | 2592 |
| wJPrpFsThWAcBQIw | 81 | 81 | 37 | 2592 | 1575 |
| wJQTclrlOLEYYfTB | 81 | 81 | 0 | 2592 | 2592 |

主会话独立读取三份raw及manifest/receipt，实际核验SHA、32点shape、ragged offsets/counts、全部检测几何、最大XYWH面积选择及原始二维点对应，未修改产物。37帧返回多个检测不等于37帧确有多个人；每帧选最大面积不保证同一身份追踪。当前三条没有零检帧，零检NaN/mask政策有测试覆盖，不能声称真实零检案例已经验证。

必要支持仅为`valid & finite(2D) & 0<=x<width & 0<=y<height`。红衣1017个点帧不满足画内必要支持；画内仍可能遮挡或定位错误。叠加、掌足事件与旧cached-box32的比较独立进行，数值差异、工程exit0与全finite均不构成准确度或训练批准。运行资源owner确认CLI自然exit0、GPU2项目holder已恢复，未抢占在途16张VLM卡。

新三栏叠加与CPU比较已随后完成，实际小报告为 [`nlf_rgb32_fresh_detector_fixed3_execution_compare_20261024_v1.json`](../results/nlf_rgb32_fresh_detector_fixed3_execution_compare_20261024_v1.json)，SHA256 `8dc98a5e470b45fac3e5ed02a4fec45e911ff7e6b493e6aef6d6752826407461`。三条叠加视频均实际解码81帧@16fps。新旧同源视频、规格、模型、点名与PTS一致；必要画内mask逐值一致，主会话独立核对比较NPZ及其分位数。画内点位差中位数分别约0.92／2.17／1.59px，仅表示新旧结果差异，不是准确率或body逐值不变的证据。红衣足面仍全画外，画内掌／鞋跟遮挡问题没有解决。后续新的fresh坐标事件caption及maxgap0必要支持控制也已真实完成，独立终态和实际结果见下一节，不使用旧cached-box描述或这份叠加报告冒充新caption。

### Fresh坐标动作描述与必要支持控制收尾

实际收尾报告：[`nlf_rgb32_fresh_fixed3_finite_closeout_20261024_v1.json`](../results/nlf_rgb32_fresh_fixed3_finite_closeout_20261024_v1.json)，SHA256 `026896f861081b4d02f9a1c7f775cf5774dea7c61611547abcbf0e0f6edcc41f`；它绑定新fresh提取／叠加实证和独立控制终态`7b9ecc068b530886571835b096870853aae82a7e42b4a0d9ad3bc9fac53f5d9d`。

三个源各执行fresh／cached × numeric／necessary四路，共12路真实CPU控制。全部使用同20fps插值、0.025s平滑、maxgap0、最短事件0.15s、角阈20°、整段及固定`[0,2)`／`[2,5.0625)`窗口；保原raw、所有报告与坐标、候选正文及支持信息，不跨拒绝点补gap。主独立读取12份报告与coords并核SHA、事件依赖点全valid、红衣脚部拒绝及旧cached基线逐事件一致。

| Case | fresh numeric事件／足事件 | fresh necessary事件／足事件 | cached numeric事件／足事件 | cached necessary事件／足事件 |
|---|---:|---:|---:|---:|
| 楼梯 | 31／0 | 31／0 | 35／0 | 35／0 |
| 红衣 | 36／2 | 27／0 | 36／1 | 28／0 |
| 商店 | 27／2 | 27／2 | 29／2 | 29／2 |

红衣新旧均由2592 numeric点帧降至1575必要支持，排除1017点帧；其画外足派生坐标保持NaN、足事件为0。商店画内的两个足候选仍有heel与遮挡疑点，不批准精确方向／接触监督。事件数减少或点差较小都不证明准确优势。

**有限结论**：NLF阶段fresh检测并一起查询32点已实际可执行，三源提取、叠加与whole／fixed动作支持控制均结束；裁切、遮挡和鞋跟定位限制仍在，未建立caption准确度优势，未批准GT、可见性、tracking或训练监督。未启动新的50条或全12k NLF提取；无需再重复同数学CLI来“证明”已有12路控制完成。

### 全量与新增音乐分窗的未完成边界

2026-10-24 04:35:46 UTC的全量失败快照为44个SID、45个窗口：38个两帧证据拒绝、6个正文超过140词、1个abstain却正文非空。所有45窗均正常stop/EOS且未达1200 token cap，不能归因为OOM或截断；此快照不是最终全12k分母。原raw和失败记录保留，旧两窗pair-enum只修格式且caption逐字未变，不可冒充全失败池的动作事实修复。

当前8分片仍继续生成，唯一原MS后验进程仍活且最终complete/receipt尚未产生；不重复启动、不修改运行源。BPM长度窗由新独立入口推进，BeatThis以新隔离环境复用兼容依赖、先CPU真实3条及同50音乐检测，不修改现有环境，不以部署或测试通过替代首拍/小节caption实际运行。整体12k可用正文、最终导出、最后Git远端HEAD与资源终态仍待真实验收。
