# 手部细节与跨段一致性的舞蹈标注优化执行方案

日期：2026 年 10 月 24 日。核对基线：`codex/motionscript-optimization` 的 `153fc66cec95098a039d6d78b9d135acb85d2834`。

保留可靠的全身动作表示，同时按舞蹈内容和画面质量启用掌面、完整手部及足部细节。**手部关键点不能因部分视频模糊而整体删除。手势舞和手部清晰、动作重要的片段，应当保留并优先验证手部观测通路。** 原32点是一个掌足几何试验，不是所有舞蹈都必须使用的固定点集，也不是手部表示的上限。

本方案分开推进观测质量、动作语义与跨 chunk 文本一致性。先复用现有结果完成有限样本对照，再新增手部 ROI 与跨段改写模块，不把多个改动混在一起归因。本次交付记录方案和接口，后续新增模块及实验不据此标为已经执行。

## 已完成的优化与实际边界

| 优化 | 已有实现或执行 | 解释边界 |
|---|---|---|
| 姿态与旋转处理 | SLERP插值、四元数平滑、连续yaw及统一坐标处理 | 降低处理伪影，仍需避免抹掉真实急停和快速动作 |
| 缺失支持处理 | 短缺口插值、长缺口保留、逐关节／逐点掩码 | 数值有限、画内和低uncertainty均不等于已证实可见 |
| 事件冗余处理 | v2去重及重复运动模式压缩，保留来源事件 | 不是原版MS全池已统一使用的新语义标签 |
| 分段对照 | 整段、固定时长、按拍、BPM近似及首拍小节方案已有有限对照 | 时间定位细化不保证动作识别准确率提高 |
| 直接RGB32 | 三个真实案例完成预测、叠加及支持条件对照 | 尚未建立新增点或caption准确性优势 |
| 独立视觉描述 | 真实VLM窗口观察、来源与生成终态核验 | 格式candidate不等于动作事实正确 |
| 审计与数据评审 | SHA／PTS／类型校验，已有标注对比页面 | 保证可回查，不能替代语义核对 |

红衣案例的六个足点在81帧均处于画外，仍有有限预测值；必要支持筛选后不应保留为实际足部事实。另两例仍有掌部遮挡和脚跟定位疑点。该结果支持加强可观测性处理，不足以证明所有掌足点都无用。[RGB32真实执行与对照](../results/nlf_rgb32_fresh_detector_fixed3_execution_compare_20261024_v1.json)、[有限实验收尾](../results/nlf_rgb32_fresh_fixed3_finite_closeout_20261024_v1.json)

固定分段MS全池已有12,000源处理与独立后验，其中只有10,327源具备完整B正文，1,673源存在正文缺口；不能由处理完成推断全部是完整训练目标。[全池MS记录](../results/ms12000_independent_canonical_verified_20261024_v1.json)

严格小节同50试验有211窗，61窗无描述；合并短边缘后136窗中21窗无描述。首拍不足、无采样和partial支持另行记录，不把不同窗口数量下的重复指标直接当准确率。[小节窗口对照](../results/music_windows_same50_finite_trials_20261024_v1.json)

## 手部需求与观测质量分别判断

“动作是否重要”和“是否看得清”是两个不同判断：

| 内容需求 | 画面质量 | 标注处理 |
|---|---|---|
| 手势舞或主要动作依赖手部 | 清晰、连续支持充分 | 优先保留手部轨迹、掌向及必要手形变化 |
| 手势舞或主要动作依赖手部 | 模糊、遮挡或过小 | 尝试原始高清素材／局部估计；仍不清楚则记录手部信息缺失，不能仅凭身体正文判为完整编舞标签 |
| 全身舞蹈，手部主要用于配合 | 手腕和手臂可靠，手指不清楚 | 保留摆臂、手腕路径和双手关系，不强制写手指细节 |
| 手部并非主要动作 | 不清晰 | 身体动作照常保留，手部细节弃权，不补写手形 |

原32点每手只有两个掌面位置，加上腕部可构成掌面几何；它不能完整表达握拳、伸指、捏合等手形。若这些信息属于预期caption，应为清晰手势舞增加完整手部候选表示。

建议分三个观测层级：

1. **身体层**：可靠身体关节、完整旧pose，以及腕部相对躯干的运动。始终作为主要动作基础。
2. **掌面层**：在支持充分时消费已有掌面点和朝向候选，描述翻掌、掌面变化；不由非退化平面直接认定朝向正确。
3. **手形层**：对有必要且能看清的片段，比较每手21点的手部估计。腕点与身体腕点需显式对应，不能重复计数或混用布局。

层级由样本需求和逐段支持选择，不按一个总分自动宣称可靠。重要但暂不可观测的手部，仍是待补信息，不应被统计成“没有手部动作”。

## 手部提取的具体路线

### 先复用已有Wholebody观测

仓库已有 `dwpose_full_extract.py`，输出134点布局：左手 `[92,113)`、右手 `[113,134)`，每手21点。包含像素坐标、分数、valid、人物连续性标记和帧索引。历史2k缓存见 [标注来源说明](27_dance_annotation_motionscript_vlm_and_dimensions_20261020.md)。

先核对选中SID是否已有同一视频、同一时间范围的有效缓存，复用时保留原始分数和坐标。`score>0.3` 是该脚本的既有门槛，不是已经校准的动作准确率。缓存缺少足够媒体绑定信息时，另建核对记录，不修改旧缓存伪造来源。

### 对关键样本增加局部ROI

拟实现的局部流程为：

```text
原视频与真实PTS
→ 身份稳定的人物轨迹
→ 左右手连续ROI及清晰度记录
→ 手部专用估计器
→ 映射回原图的21点与手形轨迹
→ 质量支持和动作语义核对
```

优先使用原始较高清素材；若只有已模糊的规范化视频，裁剪放大不会增加真实细节。保存裁剪范围、缩放／旋转／镜像变换及ROI到原图的映射，防止相同手部运动因裁剪漂移变成假动作。

人物身份先确定，手部再与该人物的腕、肘关系关联。双手交叉、转身和镜像时不能只按画面左右决定解剖左右；身份不确定时保留unknown。保留腕部对齐残差，不为“接上骨架”而隐藏原估计偏差。

首轮可将MediaPipe Hand Landmarker作为手部专用候选，与现有DWPose手点比较。其提供21点、手侧及视频模式跟踪；不将其默认阈值当本项目最优配置。[官方能力与接口](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)

先比较原图2D坐标和局部归一化手形。手部模型所谓world坐标可能以手的几何中心为原点，不能直接拼到NLF全身相机3D；需要3D融合时另行处理坐标、尺度和时间一致性。[坐标约定](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)

保持原生连续帧，快速手势不默认沿用4 FPS视觉抽帧。若原视频只有16 FPS，重采样不能作为新增观测。对分辨率不足或运动模糊造成的未知，模型先验可以给出候选，但不能转成确定事实。

### 手部事实的描述层级

- 腕、臂可靠时：描述抬落、横移、交叉／分离及相对身体的位置变化。
- 掌面有可靠支持时：增加朝向和翻转；单纯2D投影不无条件推出精确3D掌向。
- 手形可辨时：保留张开、握拳、伸指、捏合等实际变化。复杂手势名称需要片段证据，不凭少量点或动作类别猜测。
- 只看到双手靠近时写靠拢，不直接写拍手、接触或爱心手势。遮挡期间不补齐手指状态。

身体层不能吞掉手势舞的主要动作；同时不要求所有窗口强行描述每根手指。最终caption保留对编舞及渲染有意义、且有证据的动作。

## 足部及其他动作优化

穿鞋不等于没有步法信息。鞋尖、鞋跟的可见轨迹和整体朝向可以作为候选，但应区分外观锚点、解剖足点与地面接触。画外足部直接记缺失，不用模型的有限输出补成真实观测。

在增点前，优先使用已有完整pose和连续轨迹，补充以下动作关系：躯干与骨盆相对转动、双臂交替／同步、膝部变化与身体升降配合、移动方向及真实重复模式。骨盆移动不是完整重心或承重真值，相机运动也可能污染全身位移。

将已核实的微动作组织成主要动作组合，避免所有肘膝变化逐条扩成长段。几何事实、视觉事实与音乐节拍分别保留来源；“动作踩拍”需要实际音画对齐证据。

## 跨chunk一致性与LLM改写

三类处理应分开：坐标去噪、动作连续关系、语言衔接。坐标处理在完整连续支持段上完成，避免每段重新初始化，也不要把真实急停和重拍磨平。LLM负责整理已知事实，不能修复未观测到的动作轨迹。

### 边界核对

对每个相邻窗口检查末端与起始状态、速度方向、身体朝向及必要手形。首轮可用边界前后各约0.3～0.5秒作为核对起点，遇到长动作按证据调整；这是试验参数，不是已验证的最佳宽度。

额外片段只用于判断延续、重复、切换、停顿或不确定，核心窗口仍保持原时间范围。不能把扩大的观察窗所有动作贴回一个较小窗口。原版MS没有导出的逐动作时间，不从句序伪造精确事件边界。

纯文本LLM接收核对后的关系；VLM可查看边界原帧／短片段来提出候选意见。两个chunk都写抬手，可能是同一次延续，也可能是两次动作，仅靠同一句文字不能决定。

### 全文一次改写

向同一次改写提供整段各chunk原文、固定时间范围、实际支持状态、边界核对结果和手部需求。允许合并冗余叙述、表达同时关系、统一主语、保留真实重复模式；禁止新增未证实动作、改变左右／方向／次数、移动时间边界或把缺失窗口补成泛泛的舞蹈句。

推荐的指令模板如下。它是拟实现模块的prompt示例，不代表现有脚本已经执行该流程。

```text
根据以下分段事实，输出供舞蹈视频生成使用的连续自然语言caption。
在每段前保留给定时间标记，不输出JSON。
保留主要身体动作；对手势舞保留已核实的掌向与手形变化。
只有边界核对支持时才使用“继续”“保持”等连续关系。
同一动作的重复叙述可合并；真实多次动作必须保留或明确概括。
不新增动作、接触、舞步名称、人物外观或精确发生时刻。
左右、方向和次数有冲突时列入核对问题，不自行选择一个答案。
不可观测的细节与弃权窗口不能用常识补全。
输入：原始chunk文本、固定时间、支持状态及边界核对结果。
```

例如，证据若确认双手由胸前抬至脸侧、随后保持高位并改变手形，可整理为以下形式；具体事实为格式示例，不能直接用作数据：

> [0.00–2.00秒] 双手从胸前抬至脸侧，手掌张开。\
> [2.00–4.00秒] 双手保持在脸侧，由张开变为握拳，同时双肘向外展开。

不能为缩短文本将其改成“双手随音乐摆动”，从而删除手势舞的关键监督。若掌向、握拳或持续关系没有证据，则删除这些未证实细节或退回核对，不能用润色掩盖。

训练与H3接收自然语言正文；内部另存原文、改写文、来源句／帧、修改类型和问题清单。LLM自检及格式检查不等于事实正确。当前共享窗口入口主要逐窗观察后拼接，跨段核对及全文改写仍需单独实现。

## 分阶段执行与产物

以下样本量为建议，尚未因此执行新增实验。优先复用已有三案例、同50源与对比页面；旧骨架三案例和音乐窗口前三源是不同集合，不能混当同一对照。

| 阶段 | 操作 | 建议交付物 |
|---|---|---|
| 0 固定样本 | 建立约60条小试：清晰手势舞、全身且手部可辨、模糊／遮挡各约20；按来源组记录 | SID与媒体时间清单、组别、重要动作与可判断范围 |
| 1 复用与观测审计 | 核对现有body／pose／DWPose／RGB32／MS／VLM缓存，逐部位记录质量 | 来源及覆盖清单、手部ROI候选、缺失原因 |
| 2 手部小试 | 先复用DWPose；只对必要片段增加手部ROI估计与叠加 | 原图手点、变换、逐帧质量、左右关联、原视频叠加 |
| 3 动作信息对照 | 固定窗口与身体来源，比较身体、掌面、手形信息的标签贡献 | 原始及增量事实、正确／错误／未知与主要动作遗漏 |
| 4 跨段改写 | 同一标签比较原始拼接、文本改写、加入边界证据后改写 | 三版caption、句／帧来源、冲突与真实重复保留记录 |
| 5 渲染与规划验证 | 先用已核对caption测H3，再训练音乐规划器小试 | 固定条件生成对照、手部跟随与整体动作评分 |

同50池不一定包含足够手势舞，缺少时从真实数据补充，不把普通摆臂强行归为手势舞。保留模糊手势舞的需求与缺失记录，避免只保清晰样本后误报所有手势舞都可处理。

先在同一RGB32缓存内比较“只消费身体／增加掌面／增加足部”的作用，避免不同查询点集改变身体预测后仍把差异都归因于额外点。旧完整pose另作基线。手部21点采用独立估计器时，身体主干保持同源并明确这是新增观测通路的比较。

改写实验尽量保持模型、文字预算和时间范围一致。先比较事实保持与边界关系，再比较语言流畅度；单纯流畅不能抵消新动作幻觉或手部主要事实丢失。

### 可复用的现有入口

下面仅使用现有CLI。变量必须指向实际可用环境、规范化目录中的SID及独立输出；审计输入与输出目录分开。DWPose命令需要实际CUDA provider，不是局部ROI手部提取命令。

```bash
set -euo pipefail
: "${ANNOTATION_REPO:?设置实际仓库目录}"
: "${ANNOTATION_PYTHON:?设置实际Python绝对路径}"
: "${ANNOTATION_LABELS:?设置已有标签目录}"
: "${ANNOTATION_IDS:?设置已核验的小试SID清单}"
: "${ANNOTATION_GPU:?设置实际可用GPU编号}"
: "${ANNOTATION_OUT:?设置独立的新输出目录}"

mkdir "$ANNOTATION_OUT"
"$ANNOTATION_PYTHON" "$ANNOTATION_REPO/scripts_omnidance/exp/ms/ms_caption_audit.py" \
  --input "$ANNOTATION_LABELS" --out "$ANNOTATION_OUT/caption_audit.json" --top-k 20

CUDA_VISIBLE_DEVICES="$ANNOTATION_GPU" "$ANNOTATION_PYTHON" \
  "$ANNOTATION_REPO/scripts_omnidance/exp/ms/dwpose_full_extract.py" \
  --ids "$ANNOTATION_IDS" --out "$ANNOTATION_OUT/dwpose" \
  --shard 0 --nshard 1 --num_threads 1
```

已有相同SID缓存优先核验和复用，不为该模板重复生成。新手势舞不在此脚本的规范化媒体目录时，先实现并核验输入适配；该命令没有任意视频／ROI参数，不能据此宣称已经支持新局部流程。

### 待实现的独立接口

1. 样本需求与部位质量清单：分别保存手部重要性、画内情况、像素尺寸、模糊、遮挡、连续支持及未校准模型分数；阈值在小试上确定。
2. 手部ROI提取与估计适配：保留PTS、人物／手侧关联及原图变换，输出与旧缓存分开的新来源。
3. 边界关系核对：输入连续轨迹及可选原帧，输出延续／重复／切换／不确定和依据。
4. 全文改写：输出带原时间的自然语言正文与内部变更记录，保留原始标签。
5. 独立核对：检查主要动作、手形、左右、次数及连续关系，区分自动格式检查和实际事实判断。

这些接口当前不能作为已存在命令调用。已有标注对比页面可复用，新增手部ROI、连续关系和改写差异需扩展展示。[当前评审页面交付记录](../results/annotation_compare_review_delivery_20261024_v1.json)

## 评价与扩大条件

按清晰手势舞、一般全身舞、低观测质量三组分别记录：主要动作事实正确性、遗漏、手形可判断覆盖、左右错误、误报、边界关系正确性和真实重复保留。画内支持率、模型分数、文字重复和解析成功率分开报告，不当动作准确率。

手势舞应单独核对手形及其变化有没有被caption保留，H3有没有跟随；全身舞应确认增加细节没有损害主要身体动作描述。人工核对可先固定一批参考事实，再将候选版本匿名打乱比较，避免只选成功案例。

扩大到全池前，至少要看到新增手部信息在相关样本上有可核对收益，正文误报和边界矛盾没有被改写掩盖。无收益的细点只保留为研究候选，重要但不可观测的手部继续记录未知。允许生成有明确来源的弱标签候选，不把它们写成全量人工通过。

CPU可完成缓存审计、清单与改写请求准备；手部推理先用有限样本测量实际资源，已有GPU DWPose入口单worker单卡；VLM试验复用已具备的模型与有限资源，吞吐以实测为准。方案不要求重跑整个12k姿态池或扩大集群。

音乐并不唯一决定某一组精确手势。规划器学习的是与音乐相容的编舞方案；渲染器先用实际视频支持的caption验证手部与身体跟随，不能只凭规划器逐字匹配某一个原视频标签评价编舞能力。

## 关联资料

- [分段标注与训练任务](30_motionscript_chunk_annotation_plan_20261024.md)
- [按小节窗口与RGB32设计及执行补充](31_bar_chunking_and_rgb32_execution_plan_20261024.md)
- [32点的表示与真实执行边界](../scripts_omnidance/exp/ms/motionscript_v2/LANDMARKS.md)
- [事件重复处理与独立视觉观察](../scripts_omnidance/exp/ms/motionscript_v2/REPETITION.md)
- [DWPose原始Wholebody缓存入口](../scripts_omnidance/exp/ms/dwpose_full_extract.py)
- [现有逐窗视觉描述与文本拼接](../scripts_omnidance/exp/ms/ms_visual_music_windows.py)

## 四策略caption优化实际执行（2026-10-24）

- 用户授权实际GPU推进，只比较整段、固定约2秒、BPM长度、小节合并短边缘；不新增beat/strict组，不重训Omni/H3，不改在途12k producer。
- 同50独立输入计划 `MS/caption_chunk4_experiment_20261024_v1/plan_first50.jsonl`：50源×4策略，395窗＝346原MS candidate＋33 no_output＋16 partial_support。报告/收据/完整coords已核SHA。固定2秒保min_tail=1.5，所以约5.0625秒实际分[0,2)及[2,5.0625)。
- 新入口 `scripts_omnidance/exp/ms/ms_caption_rewrite.py` 保R0原MS；可观察的真正no_output另作video-only补充S，不冒MS。R1全文与R2边界增强使用同S、同prompt和采样；partial原文单列但完整正文仍空。边界前后各0.4秒原PTS帧只产生关系AI候选，不将扩窗事实贴核心窗；单窗R2引用R1。
- 真实CPUprepare TK005o自然exit0：3同源×4策略12组，7core补充manifest、8boundary RGB manifest。冻结v1源SHAafe7b081617f047216cba26c226322157148aed76c1b9a1543c57dafc5ad3731，原始响应未修补。
- v1实际四TP2均admission/live，whole/bpm/merged三源终态均exit1（多数JSON缺括号、非截断），fixed2s未完成全部源；已有少量R1/视觉补充候选并不代表效果通过。Lgl原RGB有限帧可核V→OK→心形，原MS整段空；视觉补充位置和双手同相位仍有误差。
- v1部分受管任务被标记killed，随后有stdout BrokenPipe/恢复读共享盘阻塞；保旧进程与结果，不硬杀未知进程或清近期租约。v2将由主会话直接托管日志，避开受阻卡位；新manager只隔离非关键进度print失败，实际收据落盘错误仍抛出。
- v2加入vLLM StructuredOutputsParams JSON结构约束及统一the dancer、禁止模板伪秒词的文本保真约束；不修旧raw，文字动作仍自由。R1/R2内部同facts/prompt/schema可比较，v1-v2不声称只变一个因素。当前v2入口及评审36项CPU测试通过，真实vLLM采样参数/XGrammar三schema建文法通过；不冒GPU语义验收。v2冻结SHA924b716bc13477726c287ee98321463f445a71d342e8c69a448aa13b0d527798。
- 评审使用新 `review_web/build_caption_chunk4_review.py`，已在原annotation页追加v1真实candidate/failed/pending及独立视觉补充，保旧视图、评分和模型manifest；版本固定validator source。先验收各3源再扩同50，当前未扩量。所有候选semantic/human/formal=false。

### v2失败复盘与v3精确窗口约束

- v2四策略各3源全部数据worker exit1：whole R1候选2/失败1（R2只复用）；fixed R1/R2各失败3；BPM/merged R1各候选2/失败1，独立R2均失败。多数normalSTOP raw已能JSON解析，但2窗返回3/4窗；qKg whole/BPM原1窗生成174个虚构空窗，最终4096token截断。不能把通用JSON语法约束当保窗/引用成功。v1/v2共83版本视图已真实发布，25评分/manifest未改。
- 诊断还发现恢复holder继承父SIGTERM/SIGINT忽略，导致后续精准SIGTERM无效。新8卡恢复已使用本地eval并在子exec前重置信号，实际fresh核8holder均不忽略TERM；旧N1故障2卡不强杀、不放其锁。
- v3逐request用prefixItems和minItems=maxItems锁原窗口数量/顺序/const ID，事实引用仅本窗enum并限制长度，边界固定6PARTS顺序与每侧真实帧enum；空事实窗口只能空正文。R1/R2同S时schema相同。原MS/视觉事实raw_text只在审计，进入模型的text仅机械中性化代词/随机秒词；原MS status仅audit，不阻止独立视觉事实写入该窗。新prompt不从事实列表顺序或重叠证据伪造then-then。
- v3当前40项CPU测试通过；主线程真实vLLM/XGrammar对12真实源schema accept+terminated、44错count/ID/ref/order拒绝；独立只读另验74反例全拒和28条实际规范化忠实。以上不是GPU效果结论。
- 当前冻结入口SHA6052fbb8f799c8a88833b8f564a935dad4562e16b8faf63f3f265143c0c43232；独立源码 `runtime/source_v3`、四新out `whole_v3/fixed2s_v3/bpm_v3/mergedbar_v3`，同原first50计划和固定3源。
- 主会话实际受管任务：N3 fixed2s GPU2/3为6c2Uh8，N8 BPM GPU4/5为gdGF7L，N6 merged GPU4/5为vDj7Z1，N6 whole GPU6/7为Zgp3ag（等待merged首真实generation后fresh预算）；日志文件固定，不由子代理持长GPU任务。先真实3源验收再放行50，不覆旧raw。

- v3后续真实GPU三源验收已通过：四策略各exit0，R1各3candidate，R2各3candidate视图（whole3条及BPM单窗1条为R1复用，不算独立R2提升）。20次独立rewrite、8次边界、7次视觉补空；未再增窗/截断。实际独立reader/source_v3 raw/收据/窗数/引用验收通过，原页新增42v3列，保41v1+42v2。小报告 `results/caption_chunk4_v3_pilot_20261024.json`。
- 依据用户授权和上述技术验收，四策略同50已写明确有限实验许可，全部full50.live实际出现（不是已经完成50）。R2虽有边界引用，正文多数与R1相近/轻微润色，暂未建立连续性改善。继续核覆盖、主动作遗漏和真实重复，不因格式成功宣称编舞质量通过。

### 网页review部署与当前结论（2026-10-24）

- 实际可用入口为 http://28.216.19.69:8765/#annotation；本机8765服务PID3341119从旧diff页面目录提供HTML，此前只改repo index不等于上线。此次只给该实际页面新增独立annotation按钮/脚本，保旧模型UI；未重启、不改评分或模型manifest。历史8775入口本次去代理直连拒绝，不冒其仍在线。
- 当前默认最新v3/R2四策略四列，可切R0/R1/视觉补充与历史版本，维持固定三case。R1/R2全文新增原分段时间跳转。实际page/data HTTP200、JSON与磁盘SHA一致、三视频三音乐Range206字节匹配；3个脚本语法及3case×3stage九组实际筛选测试通过。
- 17:51:20同50只读快照：whole已实际exit0，R1/R2各candidate50，但全窗非空正文只有46/50（原MS45＋视觉补空1），4partial仍空。fixed2s/bpm/mergedbar完整row分别25/23/18，仍在途，不拿不同分母判策略胜负。
- 固定三case边界审计链已跑通，但多个R1/R2正文完全相同，其他主要标点/连接词变化，未建立连续性收益。非空却语义贫乏的MS不触发现仅no_output视觉补充；Lgl小节后段仍可能漏主要OK→heart，不能以“有caption”当信息充分。
- mergedbar有限处理中有1次视觉补充失败：0.48秒原窗三event只引用单帧，违反至少两独立真实帧；正常STOP282/1200，非OOM/截断。该窗保留空，不虚构caption；改写JSONcandidate并不意味着所有窗正文完整。
- 评审优先核主要动作与手势遗漏、左右/先后及真实重复；当前不宣布最佳分段方式、音乐跟拍或训练增益。

### 代理入口与同50终态更新（2026-10-24）

- 用户要求用code-server代理而非IP：当前已由主会话在原空闲8775端口启动repo评审服务，保8765、既有focused manifest与评分；本机8156/proxy/8775页面与数据HTTP200，固定3case各视频/音乐6项Range206逐字节匹配。外部用户HTTPS域在执行环境直连超时，不冒外部浏览器已实测。入口为 https://launcher-ql3qjy14j2-codeserver.ide.taiji.woa.com/proxy/8775/#annotation 。补slash跳转保location.hash。
- 更新结论口径：边界增强本来应小幅修正，不能以R1/R2文本变化小判无收益，也不以边界引用数判成功；应评估必要衔接/重复错误是否修复且原动作事实保留。
- 最新四策略同50数据worker均已自然exit0（WRerXS终态已回收，另实读200份run.final核验）；R1/R2各50条格式合格记录。整段/固定2秒/BPM全窗非空正文各46/50，小节merged为44/50，不等同语义通过。
- 每组4个partial_support窗仍为空，这是原始姿态支持不足，不是队列卡住。merged另3个no_output窗的独立视觉补充已真实生成但被拒：AztczvBPGEjgAoaQ w0002[1.50,1.98)、KchlAfjUzORkuOtV w0001[0.54,1.06)及w0003[2.64,4.24)。原raw/generation正常STOP，事件引用仅单帧，未满足每event两不同真实帧；保原空，不修响应或猜动作。
- 小节组较慢是更多分窗、边界和空窗补充的串行模型调用及共享盘取帧/验源IO，实际源数持续增长后完成，不把单次GPU0/IO wait判死锁。网页仍按用户偏好只固定3case最新四策略，不发布全部50。

### 手部ROI真实验证与三模型固定窗基准导出

- 手部ROI已真实执行DWPose CUDA提取并保存双手21点原始数据和裁剪。首次严格禁CPU节点导致ONNX pose图形状/控制节点初始化失败；后续允许必要CPU形状节点但要求profile中真实CUDA节点，旧失败回执保留不改。
- 初次max32取前32帧漏掉末1.125秒，已修为全片均匀预算并保首末（24项CPU测试全部通过）。最终独立版本 `MS/bar_annotation_optimized_20261024_v1/hand_roi_gpu_fixed3_v3` 三源各32真实帧覆盖0–5秒；hwrl47、Lgl61、qKg50张ROI裁剪，worker_exit0、无资源manager failure、resources_safe及CUDA profile全部实际存在。裁剪支持不等于手形分类或关键点准确率通过。
- 有限实际ROI回看可辨Lgl首V、约2秒OK及末段两手心形；qKg约2秒局部ROI也有心形形态，不能用早期稀疏全图观察否定整个片段的手势。指尖细节、解剖左右和完整连续时序仍需按原帧核，不以单帧证明动作次数。
- 旧固定窗VLM12000 indexed final均已完整SHA/窗口/状态核验，11710源两窗完整。现在已CPU实际导出 `MS/music_planner_fixed2s_baseline_20261024_v1/{train,val,test}.jsonl`，数量10720/490/500；source与精确audio-SHA跨split交叉0，旧507验证来源未进入train。歌曲与近重复完整隔离未验证，不冒song-level split。
- 基准仅绑定原完整音轨0–5.0625秒与相同target_text，不绑定字面任务prompt；模型adapter按原生模板处理。Omni-Captioner保持audio-only原生chat/audio特殊token，不加system或舞蹈instruction；MOSS与MusicFlamingo使用合法原生音频/任务模板，不输入GT动作/genre答案。原音频24k mono可由各adapter完整重采样，不物理拆两窗。在上述数据导出时，三模型权重和训练适配仍待准备；后续启动状态见下方执行记录。
- 此前新bar全文拒收主要为纯帧说明误算独立动作、正文合理合并但漏事实引用。纯帧说明已归证据元数据，实际持续/调整动作仍作事实；不关闭校验、不无条件补满ID、旧raw和拒收回执不改。

### 2026-10-25 三模型真实启动与稳定ROI组合重验

- 三个官方模型已完成下载、官方摘要核验与节点本地stage。最终adapter SHA为 `ca7b63c18aa2452108b2c5b9ec4b36de8e0c430b401f81be26d4f5b1242ddd02`，评测/训练/adapter联合CPU测试在启动前66项通过；不是完整预训练GPU验收。
- 主会话实际启动N2 MusicFlamingo、N4 Omni-Captioner、N7 MOSS，各8卡。全相同10720/490数据、global32、2epoch。执行合同为真实2更新→新进程恢复3更新→同输出继续至670更新；前3次计入21440源曝光。当前尚无新三模型编舞效果排名。
- 初始v1进程在模型加载前因read_split对11210个音频逐条Path.resolve阻塞。只读/proc syscall与路径证实24rank重复newfstatat/FUSE metadata请求，非模型OOM、非导入失败。main精确pidfd终止其owned torchrun，旧输出/失败回执保留，三节点resources_safe确认后才重启。
- 新训练源码仅将预检音频路径改为词法abspath；id/source_group/declared audio SHA跨split校验及实际消费前stat/内容SHA保留。新增不得预检resolve音频的回归，67项联合测试通过；全实际train/val清单预检通过。未声称canonical/symlink/hardlink别名审计或song/near duplicate隔离通过。新train SHA为 `5df08aa4ed5b814527b4f150a2a67d3d4fd470a7aec193af8992d8c50dd6bb61`，冻结source_v2，主会话三个v2 manager已启动；须以实际更新、梯度、完整断点和新进程恢复证据决定烟测通过。
- MotionScript主线为真实音乐小节分窗（显式短窗合并/失败回退）→全窗口RGB事实→独立部位支持→同帧全身＋原生手ROI→一次全文整理＋必要边界小修→事实引用/支持检查与最多一次repair。原MS/关键点是辅助与对照，不冒完整语义真值；不清晰部位unknown，不连带丢可靠部位。
- stable_roi3_v1在模型加载前因冻结包遗漏ms_hand_roi.py及其dwpose依赖而失败，不作为ROI语义负评。新source_stable_roi_v2补齐15份py，85项CPU测试无跳过、真实worker import/AST闭包与3份ROI内容/PTS核验通过。main fresh捕获N3 GPU2/3后，已实际启动stable_roi3_v2完整ROI融合3case GPU验证；模板input_count12000是计划大小，targetcount3才是本次实际任务量。组合GPU尚未验收时，不称pipeline稳定，也不启动新12k全量标签。
- 后续三族v2已越过预检并真实加载完整预训练模型，但尚未完成optimizer更新：MOSS/Flamingo被检查出冻结BF16参数对应的ZeRO实际分片仍FP32；Captioner融合专家的空分片placeholder被ndim检查误当错误原形状。三族均由manager安全恢复，正在同步base/分片dtype和改用ds_shape；不移除精度/专家scope门控，不把tinyCPU通过当真实ZeRO训练通过。
- stable_roi3_v2真实终态为3源failed/0最终caption；六个RGB＋ROI观测窗五个正常STOP，一个1800token length。其四个event不同，失败是末尾连续空白循环，并非无限event数组或4096token不足。三个全文改写均由真实XGrammar拒绝uniqueItems后未调用模型；dispatch_error保留，generation缺失是派发拒绝的结果，不是全文动作被生成后抹掉。core结果校验本已拒绝重复引用；正在去除不支持语法项并限定grammar无界空白，须真实编译和再次3caseGPU验收。
- 语法与空白同批修复已完成：bar固定真实XGrammar backend及disable_any_whitespace=True，1800token与事件/支持/窗口/引用预算不变。46项真实vLLM解释器CPU回归无跳过，三实际rewrite与六实际visual schemas真实编译，旧3670字符空白反例被新grammar拒绝。新bar SHA 1c97d1960004f33b549a05439995e2256ac6ad5313ab1ea48047222a6975f414；source_stable_roi_v3补齐依赖，main已经fresh捕获并启动N3 GPU2/3三case完整重验，task j43MyJ，仍待真实新terminal与语义review，不开12k。
- 用户明确要求网页review后，main已发布stable_roi3_v2诊断版至现有8775 annotation：三case九视图，R0原MS/R1实际RGB+ROI观察/R2失败状态。18项builder测试通过，旧视图/评分未变，bundle SHA 71a12a35c0d3e33c23bbefce5249fc25963d63fe43126b777bc772d060619d89。index默认ROI三列、列明检查项、RGB/手ROI lazy缩略图及原尺寸链接、时间seek、actualdispatch_error具体原因；旧dispatch文件仅独立SHA绑定，明确不是旧terminal receipt覆盖。8156/proxy/8775的HTML/JSON/六图片200与三个视频Range206/SHA一致，外部HTTPS未亲测。
- 三音乐模型统一bb8b049b adapter（79项CPU回归）已越过真实全模型、PEFT和分片精度/专家形状检查；首次前向被DeepSpeed外层autocast未在DS config开启的断言拒绝，仍0更新。三节点均安全恢复，正在最小修复DS torch_autocast配置，保冻结BF16/可训FP32。MOSS v3另因本次导入生成三个pyc混入模型清单在准入前拒绝；main校验pyc源mtime/size/magic与官方源码SHA后将三个文件完整保留移到同节点独立隔离目录（未删权重/官方代码），v4新增PYTHONDONTWRITEBYTECODE=1才进入真实前向。正式2→3→670尚未获烟测批准。








### 原生AMP统一启动与ROI v3真实完成

- MotionScript stable_roi3_v3真实处理完固定3源：3份非空弱标签、2份所有窗口完整、0failed；Lgl首窗仍abstain，不能把它当静止。完整generation、事实引用和支持校验已过，不代表手势语义已人工接受。main严格用执行source_stable_roi_v3重建请求/事实后，追加三case九视图至现有评审网页，原v2失败版和其他历史视图/评分保留；bundle SHA ab0522c719daac851c0e755863219ea7c8538fa20a94bd7f80b39d0f892003fe。顶部入口改为“MotionScript／手部 ROI 图像”，ROI图片默认展开，以免与旧音乐direct-token模型预设混淆。
- 原生DS0.17.5明确要求torch_autocast下所有参数存储FP32；仅加config不能支持冻结BF16/可训FP32混合存储。采用三族相同的BF16载入后精确升FP32存储、冻结属性不变、原生BF16算子autocast；LoRA/projector/Adam仍FP32，NLL仍FP32。主/次实际ZeRO分片同步且检查，非DS原BF16存储不变。run_config记录load/compute=bf16、base/trainable=float32。通信字段官方JSON枚举fp32，解析为torch.float32。不修改DS库或跳过断言。
- 最终train SHA 4f046c5080958a1e6aa315951d6a0301f95d3a4b9c6ae72f3a3572e6472ae9e3，adapter仍bb8b049b；联合90CPUtests＋8subtests通过，含真实DS配置解析、混合存储拒绝到生产升存储通过、隔离CPU真实engine.forward。不能当8GPU通信或断点恢复验收。main冻结source_v4并真实fresh核三节点内存/8holder身份后，启动三统一fixed2s_v5 queues：N2 Flamingo task oYtos5，N4 Captioner RXuy7B，N7 MOSS TqYKxr。各8GPU、PYTHONDONTWRITEBYTECODE=1、全相同10720/490；仍先真实2更新→新进程3更新→继续670门控，尚未据启动声称任何更新已完成。
- v5真实前向已通过原生AMP初始化：Flamingo由processor顶层padding=False覆盖原生audio max_length，导致506有效帧卷积253与固定1500位置embedding不匹配；MOSS/Captioner在modules_to_save投影上出现ds_grads_remaining缺失。初步根因是PEFT包装的属性代理让DS复用子模块pre_bwd_fn/post_bwd_fn闭包，而计数落在另一实例，不是训练数值失败。三节点均resources_safe，仍0optimizer更新；正在限定adapter修复分区音频padding与实例本地DS钩子属性隔离，保原生模型/全部可训scope/冻结/AMP和完整断点预算，不填计数、不删钩子、不改DS或PEFT安装包。


### 完整精选联合标注与自动审核（2026-10-25）

- 已扩大为精选222,939条、448批；权威输入curated222939_inputs_v1/manifest.summary.json，正式正文仅Dance style行＋timed MotionScript，split和失败分母不改。
- 主out curated222939_production_v1继续；新增独立out curated222939_collaborative_v1，ordinal20000:21000，N5G0/G1各500条ROI/BeatThis＋N3G0/G1 TP2联合标注。独立out避免原无限分片同输出竞态；未来重复计算不能冒零重复，覆盖按source＋split去重。新params仅hand_roi_root变化，不冒完整参数摘要相同。
- I/O v2在真实Ceph publication errno22，原ordinal500–502三failed保留。v3使用独占mkdir＋hardlink noReplace，extract receipt最后提交，全资产SHA后才producer ready；目录整体不是原子。六failed公告是三unique来源。三失败的独立预处理→联合消费→每源8B审核CPU条件链已由主会话托管，等待现有任务resources_safe及fresh精确授权，不覆盖原终态。
- 新N6G2常驻独立8B审协作批、联合修订及RAQ额外修订；固定500成员随机25＋风险5，按工程阈值增抽，当前label文件/语义/原视频/ROI/模型/request/raw/generation/final绑定。独立checkpoint但同家族；抽样pass不是人工、全片语义或训练批准。新模型从N5已核本地15文件传N6独立目录，source fresh SHA＋stream SHA＋目标落盘SHA全核后上卡，不再反复读共享权重。
- 常驻修复N6G0/G1的v1实际已生成，但封装错误因ready producer terminal不含manifest_record；v2读checked plan、核row/plan/sourceRefs原摘要，保failed/null final，仅派生joint input安全处理。首条已有结果离线恢复且预算不重提；主机到N6同inode不同mount device的游标已按381byteprefix、hint/job/终态/预算有据迁移。
- 已确认每源反复读取215,078,356byte精选train.csv；v2仅显式original_csv按3split进程内首次stable SHA核，后续本节点dev/inode/size/mtime_ns/ctime_ns全相等才复用，改变立即拒绝，其它媒体/ROI/请求逐源SHA仍保留。真实N6全链CSV仅一次hash。最终watcher f46950a5ea6d0b6d406bd297b990f4c7ca9d8a4d45da3739873e1ae0e806b077，source_gpu_v3 0ff1f304db79170095c97cc44e3d6e7dff38a4fd658691ad502ba49ae5c60548。
- RAQ第二窗原GPU raw有确定双臂圆周运动，但no clear V/OK/heart/fist否定句触发整窗手形拒绝；离线只保双臂verbatim事实，未知细节metadata，其它两窗和预算不变、0新VLM。支持校验＋新独立8B抽样pass真实完成，不是模型rewrite或人审批准。现有代理#raq-scope有旧新全文、14body＋2ROI及两张原生同帧上下文PNG，18证据文件HTTP/SHA已核，旧评分保留。
- 其他5代表不是RAQ同类否定误判：raw真肯定V-sign/fist但缺原生支持，整窗raise连带丢其它事实；部分raw同时在uncertainties承认手形模糊。现有完整句筛选只确定能恢复ifc一条双臂粗动作，多数仍被原part支持不足拒绝；后续保守逐句离线验证，不补ref/漏部位、不细拆逗号强化claim、不将模糊拳句准入。
- 最小覆盖runtime/ms_curated_overlay.py已运行：共享仅curated222939_overlay_v2的progress.json/candidates.jsonl，本地可重建JSON缓存，未开始批次不扫222k目录，不用共享SQLite代真值。最近直接核对主312终态(151完整/158不完整/3失败)、协作26终态(13完整/13不完整)、修复29源(6完整修订/18不完整/5deferred)；修订与原源重叠不能相加。全222939联合标签和448批审核尚未完成，训练/人工/全片批准不成立。
- 旧12k任务仍占9GPU，全部源属于完整精选且同MS核心协议；已有542终态保留。没有现成源边界暂停接口，停止可能中断在途源；已向用户询问重分配授权，未得到答复前不停止旧任务，不碰三模型训练节点。


### 10-25 持久账本损坏与审核恢复边界

- 全精选主completed.jsonl在byte532233有940byte持久损坏，原SHA d74abe92289ada7d68da9f97c5444e177ae91b94ee76dd8be955a6556137e280。实字节是MVuipekTKQDQprcC真实terminal序列化前247byte＋ytykxzHnSDAecSEP真实完整terminal693byte；两个不可变terminal、joint、原500成员与源绑定皆实核。派生hint由真实terminal重建，不手补JSON，原坏行/错误/字节证据保留；主流水append本来有flock，不能误称缺锁，跨客户端FUSE锁/O_APPEND疑点仍未最终定因。
- 原collector因JSONDecodeError停止；新增curated_repair_collector_recovery_v1.py保原input字节cursor和originstateSHA，在独立collector.recovery.state.json恢复；原collector.state(offset532233/seen754/SHA89c4...)绝不迁移/覆写，统一预算/jobs不改。源码f85efb1441930cfc31bfe4cafacf25e0fdb18aeb9da6ad80fb540b58519237cd，每条实际读raw增量SHA与checkpoint当前prefix比对，13tests和真实930source完整前缀重放通过。常驻任务bZFFWu已托管，最新实checkpoint1106来源继续增长；旧单次任务被killed不是完成，不将局部计数冒finalreceipt。
- 离线coarsebody salvage真实5prototype全部actualPTS/原SHA通过：ifc新增原双臂e4_c0(56/80)与OAXB原9facts全文失败重组得到完整候选；Gh/uq/aBS仍空/partial，不补missingpart/ref、不拆逗号、不准入任何新的指节/手形细节。两eligible源用独立8B实审sample-pass，receipt所有artifact及模型来源已核，不是人工或全片强确认。持续离线流已生成更多候选，原raw/预算不变。
- 增量salvage审核在fCqUnQFyKAygeVVC出现producer只hands事实标eligible而独立coarsebody判不eligible，原整流抛ValueError停机。新strict_resume_v2保存单源structured_pending/不call/不算reviewed，正常trueproducer/独立true继续。实际N6freshbuilder已完整验证旧111事务/45审计、原287016byte前缀、源图/原生ROI与来源SHA，并生成授权cfg52047c...；管理器GPU续跑是否成功仍须实际新事务核验，不能以预检完成代替。当前保守producerfalse/独立true仍待处理，不宣称所有资格差异已解决。
- 原3publication失败独立重跑现全部producerready，联合1完整＋2不完整；3源独立审核实际2条且不完整不批准。Oert两空窗和Ub后一空窗raw均有动作，但手part_support机械等于ROIcrop名单，完整句混手型与身体动作被支持门拒，不能叫静止或离线剥词变确定。新的独立coarse6-v2协议保原14/14/13图/PTS、事件仅头/躯干/两臂/两腿、support仍10，ROI名单明确仅cropinventory；18CPU tests/3nativegrammar与真实preview字节已过。需要合法TP2两卡后仅3visualcall、0textmodelcall，primary统一eachsource一次预算仍未预约；当前没有实际GPU结果，不预报修好。
- 资源最新实际已运行的是独立collaborative_v2 [21000,31000)10000条、N5G0/G1＋N3G6/G7，不是准备中的30000段；后者三启动被activelease安全拒、无放卡无worker，历史配置保留不执行。覆盖default只读真实v2，20批仍需独立抽检；主审核唯一v5正在source-only真实原流/旧原生评分缓存/固定policy验收，不依赖optionalshadow迁移，不新调用模型伪修旧error。
- 旧主审核历史扫描实找回80评分/7AI错误，旧summary曾只有6，风险集合变化不能丢历史错误；membership.policy必须原样，历史保留信息仅顶层metadata，防旧不可变批次文件拒绝。新增错误网页#salvage-quality只展示确认候选及必要majoruncertain，no_newfacts仅计数不强推人工，原评分和训练批准不变。
- 全目标仍222939联合标签＋448批独立抽检和必要错误复核未完成；可见完整候选/partial/ready/局部pass分别统计，源与修订不简单相加，原badline保留即使已恢复两个来源也不把invalid原始证据抹掉。

### 10-25 主审核实际出分、旧标签复用与新证据边界（05:18 UTC核验）

- 主review v5已实际加载N5G7本地8B，并完成新的第二批阶段性出分：train-b000001保持原500成员，498已终态、95已审、12个AI错误候选，仍pending、不可训练；首批历史80已审/7错误保留。协作v2第二context也已实际出分：train-b000042原500成员、60已终态、16实审（13pass/2AI错误/1uncertain），主独立核16份各6文件收据与350个RGB/ROI SHA，真实checkpoint不同及input/label绑定通过；原批仍review_complete=false、不可训练。不是只通过CPU预检，也不是448批审核完成。
- salvage v3已有新真实aTDE pass，主独立复核其6文件receipt、16帧原视频RGB、当前视频/音频SHA及独立checkpoint；该源此次0张ROI，不声称手ROI强验收。v3随后因XnY的w00000:e2_c0新增head相对raw event.parts而退出；实际state已durably提交178源/53份实审，mtime04:59:13Z、cursor486344，progress164/51仅上一轮16条刷新时点，不应回退断点。worker_exit1且source_unchanged=true、resources_safe全真；manager仍存活仅管理恢复holder。
- head范围拒绝已由主实际CPU重放确认是协议不一致：原冻结ms_bar_annotation.filter_supported_observation的377–382行明确允许head/torso作为原事件引用，在原part_support与该event自身refs有两帧交集时补入；XnY原head支持与该event refs的交集实际[10,14]，原raw+prepared/window+forbidden重算e2_c0全部字段与派生event完全一致。故不能把该head引用定性伪造。新v4只有exact原协议重放及自身双帧例外可通过，其它真实范围越界保typed pending，不能泛化补部位/帧或吞shared SHA错误。该同一句另含closed grip，严格coarse修复范围需归手形细节，不能删词或拆逗号强化成确定粗动作；证据支持、细节准入与语义批准分开核。
- 已核旧12k三个完整来源的current媒体内容、NLF、actualPTS、全部旧window/boundary/rewrite receipt与真实逐源ROI，并由主落盘runtime/oldbar3_explicit_alternative_candidates_v1.jsonl；只前置Dance style行，原timed MS逐字不改，复用不新增32B生成。两源ROI真实在fixed3根、一源full12000根；旧舞种仍path_only_unverified/historical_video_sha256=null。三源后续已N6G3真实独立8B各调用一次，全3 sample-pass；主独立核3份各6文件收据、48帧RGB、18原生同帧ROI及checkpoint/input/labelSHA，worker_exit0/source_unchanged=true、05:37Z资源safe，仍不获训练/人工/原500批批准。主纠正运行环境为原vLLM解释器并独立7项资源门通过；base160GiB及锁/租约/精确holder门保持。全旧12000一次CPU导入已由主任务dsDwJp启动（runtime/oldbar_explicit_import_v1.py SHA aadc2f7578c1e9d986c7dcee86f7f84297dcb34190c81dee1233a375fe53d66b，独立输出oldbar_explicit_import_v1）；1027结构完整仍仅潜在上限，实际逐源导入数必须读新回执，后续分批独立审核不省略。
- 五节点32张卡已有预处理/标签生成/修复/审核，包括旧同协议12k产物待显式导入，不误称全部已切换全精选新生成。另N1G4–7和N5G2–5共8个holder现场SIGTERM/SIGINT均ignored或blocked；已向用户仅请求精确回收这些holder的强制信号授权，未执行、不影响其余工作继续；N2/N4/N7训练节点未触碰。
- 覆盖05:13时点：ready_unique5090、complete_candidate1003、partial_candidate964、ready_not_joint3123、missing217849，reviewed209、AI错误候选47、pendinghuman57。05:42:41Z更新为ready5412、complete1053、partial1029、ready_not_joint3330、missing217527、reviewed278、AI错误候选57、pendinghuman71；旧池随后实际42检查/22完整导入，尚未接入base overlay不能直接相加。这些是当前选择下source去重计数，不等全体质量批准，AI错误候选不是人工真值。完整222939标签及448批审核仍未完成。
- 三源已验GPU源码在资源safe后受到迟到worker覆写（334/975/7d0版本）。主从原Write/Edit本会话记录精确恢复实际部署的worker f5507fe6...、manager55e6f2b3...、builder0bf49e64...，各完整SHA匹配，原新版另存runtime/oldbar_review_source_recovery_v1/。主用恢复的原verify_review_receipt再次核原3pass和代码身份c433ecd580ad6009d583c2eb4640a81443e05b0c4d7e2a76da6c99a5c236798f，0新模型、标签/收据/预算不改，恢复回执保存。不将迟到新版混进旧评分，也不删除对照证据。
- 审核v5已完成9项N6真实CPU验证并由主托管ZxKcwN：XnY的head原filter重放合法，但closed grip完整句严格细节pending；Khfv的w00000:e1_c1是先前primary repair成功事实、w00001:e2_c0来自最初main，pure要求等原repair、不强求等原main。新fresh预检核178/53/4280文件、原byteprefix并通过N6G3资源门，cfgSHA49eb26d2a647ca85b481297722519491a2db303a5ba34a7ae85ebc2cd3802d44，worker935672已实际live；尚须核其新generation/事务，不能以启动代修复完成。旧v4误隔离方案已撤销且未GPU。
- 主日志又出现原byte1192186稳定单换行（1byte，SHA01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b），原collector bZFFWu failclosed停止且留游标。新curated_repair_collector_emptyline_v2.py（SHA5c6c01e2f9ad70fc6515ef43384e81af2cface3b52f716c67496a24e1565752a）只处理该精确span/bytes及三stable snapshots，原decodeerror/raw仍存，记录零encoded来源、不推断来源全覆盖；原940byte splice恢复不变。主3拒绝负例及真空行测试通过后bYgHPa托管常驻，actual游标1212374、seen1732，已经越过空行消费新真实终态，原预算/jobs/originstate不改。
- 空行不是唯一问题：原4批terminal_index vs journal+旧splice恢复核到4个长期缺日志但原真实consumer/jointcomplete存在的源（BBuc/dBlo原b1、pbUX/tYhk原b3），复读仍缺、mtime跨度00:09至06:00Z，不是快照短暂错位。独立Evidence.terminal已经原manifest/CSV/媒体/plan/producer/consumer/joint/visual全链SHA核4源通过；新独立terminal-reconciliation方案不把这4赋空行byteoffset，也不回写原损坏journal。主审核v7显式空行＋这4恢复入口正在实现，未知坏line继续failclosed，不把重新统计或4源完整候选冒448批批准。
- 06:24前后修复审核v5已真实续审：Khfv新一次独立8B sample-pass，主核6文件/16RGB/0ROI、原178baseline事务引用和53真实评分全部不变；累计226处理/57实审并继续增长，XnY保持closed-grip细节pending，不把原合法head引用孤立作伪造。主批次v5仍卡新空行、协作v2仍持续审核；新v7仅准备独立代码，未热改原运行源。
- 单源JPEG字节缓存已完成独立CPU验收：3真实源11个window/boundary请求canonical与原顺序序列化字节及磁盘requestSHA全相等，读取179→122（body107→86、crop72→36）；首次同bytes全SHA/四重stat、命中仍stat、竞态及退出恢复门全过。模块curated_request_image_cache_v1.py SHA8e40c5f784ebd548a2ab7d27450cb75a8d4e145576886ed690330925b51c79e3。尚未接运行消费者或测GPU吞吐，不宣传读取次数改进为实际wall-time收益；union39合法不能混同ROI32帧预算。
- 旧标签导入两条excluded（ebUV、Bfx）均是参考重写元数据序列化键序问题：原completed导出递归sort_keys，原transaction digest却按原JSON字段序，语义dict相等但reference_repair_sha不同。主用原不可变result_ref恢复原字段顺序并走同一全链验证，两个真实source和3拒绝负例通过；新oldbar_refrepair_exact_result_v2.py（SHA2b0c98d2c867ebec617f03350d208a4e3257d5ccf075941be287f42149566d5d）单次输出oldbar_refrepair_recovery_v2/receipts/92419becadd1659b46fa38bb8dcc7869787a05d19883f564d5acbe0cf63e0e61.json，2完整未批准候选、0新模型，旧aad运行源/raw/receipt/预算不改。不能把这两次工程排除当成语义质量错误或直接加进尚未接入它们的默认覆盖计数。
- 旧标签overlay原先要求在整次强核期间live base都不变，live每120s刷新造成饥饿。已批准先将经过前后stat/双SHA/各state-split-batch分母一致强核的base pair保存内容寻址不可变快照，再与逐源强核候选合并；明确base_snapshot_only=true/latest_live_base_claimed=false，发布仍核自己snapshot及来源SHA，旧score/坏日志保。RAQ旧版w1仍是用户否决的loose V-only文字，不能借新v3双臂caption的pass；旧original_visual_SHA及原句绑定风险，保alternative-only不默认选择。新快照版仍待实际正式派生及源去重验收，不以18tests过代替发布成功。
- 06:57Z主已对空闲N5G7旧v5 manager912575发一次精确pidfd SIGTERM（identity/start/cgroup/argv/UID皆核，worker915498在hrtimer_nanosleep；只该owned manager，不信号进程组或未知任务），任务Vbbj0q等原base安全归还后fresh builder启动stage1 v7。新v7原940＋精确newline/cross-mount身份桥接、N5原80 cached/7历史error/policy及0模型CPU已实核；stage1明确0reconciliation，不能冒四漏源已恢复到原人口。四源码final SHA422aabcf.../26ad8856.../5f86dba8.../926fc3df...已核，writer已shutdown。资源安全回执与真实新出分仍需核，不绕过base门。
- 终态对账24测试（23隔离＋1真实）已全部通过。主启动正式单次jkQTaO/out curated_terminal_reconciliation_v1，pin源码1ef0825d120775c25f628ed1c1374ac008de41b19f1397dfdf6f8b834038d2a8；原测试快照2000成员仅6source/media全链实核（四漏源＋原splice二源），1797索引对1792journal差5，四完整恢复＋PUL原不完整保unresolved。不能把路径存在性或四候选恢复当成全批质量通过。
- 新PUL漏日志源原terminal incomplete_candidate/visualfailed，不是可丢来源。主原Evidence及classifier实核为仅rewrite失败（正文有未被原部位事实支持的部位），独立curated_pending_terminal_register_v2.py SHA49ab929c687758882010585f0b473876ef6ac41d4bad9a0c8866874bc7de75f3已预检与wrongterminalSHA负例过，WJYNWu真实注册原jobs/ready（jobSHA b46bedafd5de463c3af5e20a9f2db464a5668bf5bfe34ea7367db4b4595759ca），原collector seen未添该source、original_byte_offset=null、未预约/重置budget，仍不完整；新request/receipt单独记录其原journal快照缺失证据。这样继续原GPU修复队列，不伪造原日志行或完整标签。
- 07:00Z正式对账jkQTaO已exit0：四完整漏源原不可变记录落curated_terminal_reconciliation_v1，snapshotSHA973b0e258a0ca3b83d5bd2eaa4f84ebc551c0d2008037d6d06f41116286abe1e；主四源hint/consumerSHA/null-offset复核后生成admission-main-review-v7-v1.json（SHA65ea5e194d408549be976799218b1309a8060f33cc6ff741480605e34bf22b60）。N5实际四来源全chain/同SID追加去重/跨mountprefix及12负例通过，0模型；PUL仍是原不完整来源不冒complete。
- 旧主审核manager停止后，其vLLM EngineCore917075成为PPID1并占GPU7；原base正确保locks/lease，不signal unknown。主精确核该engine UID/cgroup/exe与旧worker一致、start_ticks536079787和原日志04:54:30加载PID吻合、物理GPU7 solecompute且正常TERM，再pidfd仅TERM该owned引擎（不强杀/不信号组），保清理回执；随后原resources_safe/holder_restored真实出现。首v7构建器扫描在繁忙日志快照期间未完成empty证明而安全拒、无GPU；零写诊断实际scan又通过后新tag重预检，未绕过任何门。
- 新含四源v7审核已由主pO8omg托管：actual80原score/7历史错误CPU复用0模型与4reconciled强核通过，cfgSHA0774e91ee7ba57bd83090b359bf4e3716b345f4d889d925e330e45cb5e2b30bf；07:07:29Z admission、07:07:31Z live，worker955401精确start536877895，原snapshot/empty/四恢复来源绑定保持，尚须核新generation/批summary。不能据live或raw-index计数宣称全448审核完成。
- 主审核v7后续实际重新产出首批summary：原500成员、80已审/7历史错误保，原membership和scores不变，仍review_complete=false/training false；后续批次处于原source/label强核IO，不能把此历史cache保留当作所有后续VLM已完成。
- 为减少纯全文失败的额外32B改写，新增curated_rewrite_only_cpu_batch_v1.py（最终SHA0f8c684daa401c8a9a7deac3f1b6f86c166c45a8cd9a610a5ca8b1b10fe5290f）复用原offline事实逐字拼接，全原observations/部位自身双帧/实际媒体-PTS-RGB-ROI/成功window六receipt及失败全文实际artifact inventory强核，0model/0budget。主纠正实施中错误全raw/originaldescription细节否决门：未准入的handshape/uncertain事件只是证据metadata，不能连带丢已经原filter支持的粗动作句；只当前admitted raw_text自身命中才pending，不删词/不拆逗号强造声明。固定3真实source（wqGY/jqo/SDj）全部complete候选且原预算不变；9测试全过，包括原jqo被拒openhand raw事件不牵连6个粗动作事实的正例、wrongSHA/parts/frames/预算/幂等负例。有限500readyprefix一次批处理0aSNPh由主托管，后续独立8B审核仍需实际完成，不把CPU3候选当质量批准。
- 07:29Z有限500prefix CPU修复0aSNPh已exit0，正式manifest SHA60dcf3861b00665db5b79d2a8f47e2f67130c3de62323b47160c3b63cb810358，实际14完整候选/0模型/0预算，主逐14ref字节SHA与joint_complete核过；403需非rewrite工作、77实际admitted细手形、2不确定、2原final并非failed、2中性主体约束仍pending。不能把工程pending当语义错误，也不能以原被拒事件细节拒可靠coarsefacts。固定3候选独立审核已CPUprepare48RGB+6同帧ROI，GPU实际审核尚待安全准入。
- 07:36前后NMyy8W旧标签去重汇总首次真实发布成功：原base快照更新时间07:07:59Z、稳定读取07:09:08Z，原1232完整候选＋207强核导入，7来源重叠/200新增来源/203完整选择净增，合并选中1435complete、1211partial、3890ready_not_joint、216403missing。主核全source去重、448原batch总和222939、所有immutable快照SHA和候选导出SHA；RAQ旧原第二窗被否决标签仅alternative，无selection，不借v3审分，原AI错误保。此非最新livebase，new14CPU及两refrepair另root不混算，质量/训练批准false；不可变发布回执ee2e62307b41ae8c3507519681bf9b880b0fabc419d99a26f96d49c619335b05。
- 五节点即时盘点40卡：20标签生成含可复用旧12k、7ROI/BeatThis、2修复、3独立审核、8ignoredholder；32工作卡瞬时16卡GPUutil0参与CPU/Ceph等待，不是持续平均利用率或可任意占用的空卡。失败队列前500内部400需要reobserve、95纯rewrite、2style、3media，不是全数据错误率。下一步固定3未预约空窗coarse6/10support的新观察CPU预览已11/8/11原图通过，N5G0/1有限3源资源链正在准备，现producer任务和原产物不动，主须safe归还后才调度，不碰训练节点或强杀未授权holder。

### 10-25 三源粗动作真实观察、零模型误拒恢复与消费者缓存接入（08:38 UTC）

- N5G0/G1 原 COLLAB2 producer 经精确 owned-manager pidfd TERM、原 safe/restored/锁租约现场核验后，有限三源新配置 SHA `9f5e51a9e2461e864c824add7370ff89fcb1b84db1eae13c59a26b81d77f6806` 于08:21:05Z正式准入；08:22:12Z实际32B native TP2加载完成。真实共3 visual、0 text，每源 canonical attempt-budget 恰1且未重置，不再只停在CPU预览。原粗动作 v1 的 Auuit/AzPom 两源 partial、DxNC complete，不能把 manager exit0 或3次调用当作全部质量通过。
- 两 partial 的真实原始回应有完整粗动作句和原帧引用；原 frozen filter 删除非必需的 torso 标签后，v1 validate_coarse 用 parts 列表全等（并以zip对齐）误拒为“补造引用”。原 package 再把已派生 e1_c0 当 raw e1..e32重新验证也存在协议不一致。未热改已运行的 v1、未删除失败 final、未重跑模型；新增独立 `curated_coarse_empty_filter_recovery_v1.py`，只允许精确该 equality 错误，原 raw 格式/粗动作/自身双帧支持/画外/fine-detail 等门全部保持，派生 parts 必须是原 parts 子集、句与 frame refs 逐字不变。package 第二遍仅接受当前已真实重放的 exact projection，finally恢复私有回调，不一般放松 raw validator。
- 收尾 wrapper 另误用 repair q.digest 验 generation schema；实际原 native generation 使用 bar/core 的 digest，真实5d9b…与 generation一致而 q.digest 为0167…。原 generation/原 queue failure 保留，按原 core.validate_generation 核验，不迁改原收据或假造 worker_exit。原实际5/6 artifact inventory、原 request/preview/binding/raw/checkpoint/预算/media/PTS/ROI和其它窗/style/facts/usedIDs 全链强核。9项实际raw正负例通过；正式零模型恢复 `9eqla8` 已得到3完整联合候选，原新增模型仍3、此次新增模型0/预算0。恢复源码 SHA `f9085da43857bc3f05204dd1557674d0d7f645a740f7e260420748dca8acd957`，测试 `b24f318ff8bf198f58a6d67fda4e5719e3757350ada2fc48eebcb5c8250c3df2`。新根 `curated_coarse_empty_filter_recovery_v1`，两源净修复、第三源为同源替代；独立8B审核尚未做，不能与base直接相加或宣称批准。
- finite 原 resources_safe 全真且原真实 failure/last_result_proof/三个终态保存。新增严格恢复 builder/wrapper（299427dc…/a36bd6bd…）通过13 CPU tests及真实65 refs gate，只改变资源 namespace、不改变原每路5000/argv/env/model/out/selection/reuse。主 fresh 构建后正式恢复两 producer，任务 OTQeqS/M5fsbq；真实现存 worker975314/975313身份与父树已核。原未完成5000分片仍 pending，不把临时 TERM 或安全退出算数据完成。
- 08:25只读核查：base export 7081 unique（无重复），1397 complete/1347 partial/4337 ready_not_joint、215858 missing，更新08:20:29Z；oldbar合并 export7059 unique、1613 complete，但 frozen base updated07:45:57Z/read07:48:06Z，不能当最新live或与当前base相加。主reviewv7真实 reviewed290→291；salvage processed620→624而reviewed仍107，近期多为资格/无新事实处理，不是模型未加载。两8B在08:18–08:24已有79/80 resource samples全部util0，多个consumer/producer/repair/importer现场 request_wait_answer，支持 Ceph/FUSE 等待限速，不代表所有阶段已测 wall-time 占比。八 ignored holder未强杀；N2/N4/N7训练未触碰。
- 缓存只先接 N1 s2 GPU0/1，s3维持原任务作对照。主原native参数/模型/source全SHA CPU检查通过，精确原manager1159205 TERM、信号前已归属引擎1160178后为PPID1，只有精确该captured owned engine正常TERM，未信号未知/ignored任务。交接脚本在恢复holder到达但safe尚未写的瞬间保守拒 unknowncompute；后主重新只读核到原safe全真与两份exact restored holder，未重复任何TERM，独立保存真实transition SHA50abb60f…与signal capture fb2520a6…。
- 新 cache builder65512155…/manager31a7ee2e…保原argv仅entrypoint替换、原semantic params/model/out/selection不变；11 CPU tests和N1本机check-plan已实过，完整 old admission/live/failure/safe/restored/fresh locks/lease integer-ns/UUID/160GiB门通过。正式缓存消费者 UPptzE 于08:38:25Z准入、08:38:28Zlive，原run/terminal/事务保留。已有11真实请求字节等值及读取179→122仅证明IO次数改善；新GPU生产吞吐与cache事件仍须实际核验，不能预报加速倍数。
- 当前仍未完成222939联合标签及448批抽检/错误处理/必要人工。三源恢复 complete 与 CPU prepare 不是VLM pass；所有 human/semantic/training/batch approval 仍 false。
- 08:58Z缓存版已出现10条真实source cache事件（499实际图像读取、73命中），无worker共享错误。主独立重构前三条部署样本7个window request，与磁盘实际请求全字典相等，0新模型；request semantic输入未被cache改变。550 stat计数是126 miss×4＋23 hit×2的原缓存完整保护门，不能因数量多砍去检查。新的IO路线限定一来源前瞻CPU prepare，GPU当前源与下一源首媒体SHA/probe/uniondecode重叠，父仍原verify_prepared全部强核；LOCK_NB前必须join，prepared/receipt中断窗口需保真补尾，尚未部署或证明wall-time提速。
- 09:00Z N6本机有限3审核预检已287source refs/真实完整8Bcheckpoint SHA通过，0GPU/lease/signal。随后主精确owned-manager929053/引擎938194正常TERM并核原resources_safe，原state662来源及110既有评分保留。新finite builder在prior_safe误将transaction中的带byte_offset输入流切片ref扁平化后，拿历史切片SHA验当前持续追加全文件，安全拒绝；没有新8Breview、没有新review预算。正在只改未部署的新resource wrapper修复按byte span/source notice/prefix的原协议校验，原流/raw/score/state不改。原salvage恢复zrviq7已托管，当前只能据实际新live/generation确认恢复，不把builderrunning当已复核。
- 09:18:01Z直接base计数已1523 complete/1474 partial/4523 ready_not_joint/215419 missing、7520 ready unique，仍all_sources_joint_complete=false/all_batches_reviewed=false。缓存版s2已25来源、1243图像读取/145命中；不将miss四stat/hit两stat保护门计数解释为隐藏额外元数据开销。
- 新prepare-ahead实现经主接管修正：仅stage run启用helper，prepare/export原模式保留不spawn；CPU子环境移除CONDA_*/PYTHONPATH/HOME及明确conda PATH/LD段，CVD空、父env不变；helper启动-B -u并argv全SHA，unknown-current不推进ready_seen/物理cursor，原limit包含resumed/producer_failed/中间prepared，二次TERM/INT收尾不打断exact child回收。3隔离CPU tests含真实原native loop事件重叠/双LOCK_NB前join、部分prepared原save_once补尾/原verify完整拒篡改、exec helper与active subprocess回收，全过；另3真实生产来源make_job/validate_job均与原stored binding全等、0模型/0写。主冻结wrapper02d792a4…/testffa4bc54…，新资源适配4ce54ccd…保原N1s3 nativeargs/model/selection/out/分母。
- 主仅N1s3 G2/G3精确owned-manager1165253及信号前归属的EngineCore1166178正常TERM，原safe/restored/原产物与55734分片未完成边界保。fresh配置3677d106…已实构建；通用cache builder的打印启动argv仍指老cache manager，首次该错误入口被原语义门安全拒，无新queue/GPU。主使用配置本来SHA绑定的新prepare-ahead manager正确入口（不热改配置/源），09:23:35Z正式admitted、09:23:38Zlive，任务oHiOx3。新run v3-5bec0c92547e4b1ea7e577094f8b486c协议prepare_ahead_enabled=true；当前仍原旧终态SHA恢复扫描阶段，无timing/newhelper记录，不能宣已隐藏IO或提高吞吐。固定首3源独立实际timing/CPUhelper/终态验收R3mFyW已托管。
- 追加流误拒已最小修正：prepare38cbc9cd…/driverd2a5566a…/managera7ac544d…/test1056901c…，24CPU tests包括合法append、midprefixmutation/截断/换inode/source错配与热路径不扫旧评分。原v5 AppendCursor与resume.references分别保原physicaldev/inode/fullprefix、notice/input/byte_range SHA；动态stream不入wholefile sourcefreeze。主N6真实1808930bytes/9fad5d3d…前缀和physical门实过，另主节点只读662输入/notice全内容映射实核（不冒N6跨mount physical已过）。原score/fullsource强核仍在资源交接/退出，currentcall保持job/candidate/预览/RGB/ROI与parentFD/kernelFLOCK/leaseheartbeat/UUID，仅不重扫历史几千路径。
- 原salvage恢复zrviq7仍实际活在resume全旧证据核验，未出fresh配置，GPU3是原管理器已安全恢复holder，不冒运行8B。修复后有限3审核prGCoM与审核后恢复OX3QmF已主托管：只有原新watcher真实恢复、最新actual admission/currentmodel/ownedmanager身份核后才再临时调度，缺safe/未知compute不信号，原110+评分/断点全保。三源实际8B审核仍未发生；所有候选quality与原448批/训练批准false。
- 09:32Z预取首3真实source验收已完成：ZdNq已有prepared，sourcewall22.43秒/verify1.38秒；下一源fEHM CPUprepare10.98秒、main锁前wait0.0068秒/verify1.52秒/sourcewall77.62秒；第三Ylb CPUprepare8.44秒、wait0.0701秒/verify1.22秒/sourcewall71.76秒。actual3终态2complete/1partial；唯一CPUhelper3181463、直接父3157108、CVD空、argv带完整selfSHA实核。主独立重构3源共9个window请求，原batch.json_bytes与实际request.json原bytes全相等，0新模型。证明至少这两源准备已与前源重叠，但没有同源无预取walltime对照，不能算整体加速倍数或声称全部IO解决。
- 修复后的finite支持直接用原safe/state/110评分交接，不需“先恢复8B再立即停下”。主精确核未取卡的原resume CPU builder1036351及其主托管父（无配置、无GPU锁、原state字节SHA不变），仅pidfd正常TERM，记录coarse3_original_resume_deferred_main_v1.json SHAa6ca14fe…。旧prGCoM/OX3QmF两个没有子进程的等待脚本已精确取消，未信号GPU/未知任务；原safeholder当前再次实核。直接修复后的单次builder ouKvTB 正在构建coarse3_n6_launch_main_direct_v3.json，未完成、不重复启动；主将真实构建回执通过后才有限3review，原模型/评分/数据不改。原base管理器resources_safe后仍h.wait，故恢复必须等待safe而非manager exit；只等待中无子进程的错误supervisor已精确停止，不拿exit255当模型失败或全目标完成。
- 09:45Z三个新增旧reference-repair排除已真strict恢复：原ExactResultImporter复用result_ref原插入序再走原完整filter/rewrite/source验证，OG83Bn单次3/3 imported、0新模型/0预算。新根oldbar_refrepair_next3_recovery_v2，原候选MIXO 2e3ffd6e…/dGT10afa4a8…/QV46b82f37…，原batch162/38/272；原修复metadata语义全相等、receipt唯一差异原reference_repair_sha，原failed五产物与成功repair六产物均actualSHA绑定，原raw/receipt/文本不改。QV一个纯frame-range句的fact有exactissue省略合法，未造引用填满。主独立核3complete/原timedMS保持/待独立审核，receipt d0abc7c8f54c5e351f64d1c7420c205a3bd40952708ab6555cb6da7bfeac8652；不将额外根与base重叠来源直接相加。
- 同时新增curated_rewrite_only_cpu_next500_v1.py，只取原失败队列物理行500–999、仍500来源有限窗口，完整前1000行prefix及选段byteoffset/length/hash绑定，append合法/short窗口拒/first500不重处理CPU测试通过，复用原0f8c…strictderive/部位facts/native receipt/唯一预算。MWesZu正式单次运行，新OUT隔离；未出最终manifest前不报修复数量。
- 后续MWesZu真实exit0：500去重来源→15完整候选，405需非rewrite工作/75 admitted细手claim/3中性主体/2非failed仍pending；新manifest0cbf3ec4b7870a1ca226d6e44fd5899d0f91d826fb70601cf66d715175a4964f，主逐15candidateRef bytesSHA/fulljoint核验通过。三条词法阻塞已有原facts.text等native neutral，仅her→the dancer’s等主体替换；新curated_cpu_subject_normalization_pilot3_v1.py复用原全filter/media/PTS/ROI/window/failedfiveartifact/style验证，原facts.raw_text/source_clause/observations/原failedfinal不改，不删uncertain/细手/时间词，不新增动作声明。kmXTxu实际3complete/0模型/0预算；主逐3candidateSHA和facts/observations/final全等验收，原预算before/after一致，独立审核未做。
- 直接有限审核旧配置56b17204…真实构建结束，但原score保全＋cfg全ref检查逐file缓存命中仍Path.resolve深父链，N6多次FUSE revalidate新fstatat阻塞；原PID1048440完成并发布配置，后新manager尚未创建queue/拿lease即仍CPU核验。主核exactmanager argv、无queue/锁、GPU原safeholdersolecompute后仅pidfd正常TERM，保原cfg/原state/110评分；不取消或重签未知lease，未加载新审核模型。
- 新显式IO执行coarse3_n6_content_cache_execution_v1.py只换私有SHA缓存key为lexicalabspath＋完整7stat字段，实际stat/open始终原suppliedpath，MISS before/fd-before/fd-after/path-after＋fullSHA仍原门，cachehit仍只是current stat，不谎称每次hitfullhash；原Path.resolve路径域/原CRC/模型/当前source证据校验不改，旧CSV种子隔离不导外缓存。3独立tests证明ctime/inode变拒陈旧内容、link/../与lexical碰撞仍按实际文件、同inodecachehit不能绕原base realpath域以及读中变动不提交cache。下版源码作为SELF manager/driver独立执行ref绑定原38cbc/a7ac/d2a等完整源SHA，未热改旧frozen/source。Rn1B1i新build/hME8IN有限3review/SfVype旧watcher恢复均主托管；10:14实际仍strong核旧origin request/JPEG读，不能宣传已解决全部IO或已完成8B审核。
- 10:04直接base1629complete/1555partial/4736ready-not-joint/215019missing，mainreview367条/38AIerror/0closedbatch，旧CPU导入1113处理/616imported/492pending/5excluded（另3已新root恢复）。全量仍未完成。已独立只读核N8原s0/s1完整watch两个GPU对仍活跃，无safe；准备下一pair仅N8s0 GPU4/5，原s1和N1s2对照不动。新独立N8s0resource adapter正在实现/纯CPU测试，保exact oldconfige2ff…/同model/同原args/env/out/55735分母与完整safe/fresh/锁租约/N8本机stage门，不节点假别名、不自动TERM。其实际推广收益须后续3source request bytes/P/V/W实核，所有新candidate仍非quality/训练批准。
- N8独立最终资源适配入口2a2a7127d3f73e18f2836f6cdef68972b45d3a6616507d2592845cd0fc1d5d49/test836c7c147850ae4b3dfe8542537727a188c99c7f569b7cae5e7af914e188b541，3CPU组主独立全过，真实临时双leaseFD/native acquire＋refresh集成证明合法heartbeat不得误判历史ownedmtime变动，换inode/staleheartbeat拒。旧889e版虽CPU门通过但未GPU，迟到writer最终重freeze后主重做N8本机CPUsource/modelstage检查（kL2xHu），source/原model/N8stage门实过，无alias N1，writer停止后才实际交接。
- 主xjNuCI只精确N8s0旧manager1950517和信号前归属EngineCore1950716 normal pidfd TERM，原sharedrun/source/params/failure保，另一对s1及训练不动。原safe/restored/physicalUUID/fresh完整locks/lease integer-ns/160GiB/完整旧receipt时序主核，transition2c2f73c6…、newconfig5042c5049a865dbacda30dfd314fec058038786ba338461ed3ae7dd8c67c185e；10:24:54Z准入/10:24:56Zlive，主实时PID2361616与原args/env/model/out/selection逐字段保持已核。实际新helper/3source timing尚待LLpLSC独立验收，不能把部署成功当整体加速或质量通过。
- 10:20读取base更新10:16:51Z：1656complete/1583partial/4794ready-not-joint/214906missing，mainreview384条/39AIerror/0closedbatch；原N8s0与s1此前合计约620完整/659partial，均原完整分片watch未完成。现在仅s0启新预取，对照s1继续，无全体quality批准。
- N8s0实际首3source已LLpLSC验收完成（10:32:31Z）：Ghhy首prepare8.05秒/sourcewall93.34秒；Jklz前瞻prepare7.82秒、锁前wait0.0643秒/mainverify0.857秒/sourcewall53.24秒；KSD前瞻prepare6.18秒、wait0.0642秒/verify1.252秒/sourcewall56.17秒。3真实终态2complete/1partial，唯一CPUhelper2363579/父2361616/CVD空/argv全SHA实际核；主3源7个window请求按原batch.json_bytes重构，与磁盘原字节全等、0模型验证。不声称同源无预取对照或整体倍数。Task52仅此单对部署/3source实施验收完成，不等全量。
- 新显式元数据执行的build Rn1B1i已真实完成：config7592eae7…，但原manager在所有source/旧110分核完后才核历史capture20min TTL，因重核超时安全拒，未创建queue/取得lease/加载8B。这说明改变hashcachekey不解决所有证据重扫等待，不能把capture历史年龄当currentlease真值。主新revision保持原score/media/候选SHA不重签，移序为全source/model/history强核后再次实际fresh_holder（原FLOCK/lease/holder/UUID/160GiB gate），当前capture必须60s内；最终放卡前再核实际parent-FD/kernelFLOCK/currentleaseinode/heartbeat/currentUUID/normalTERM holder。旧历史capture仍全绑定、不作为当前放卡授权；未延长旧TTL或伪time。
- revision oldconfig7592→newconfig5404683f…/newSELF216a65ac…，source/code执行修订明确而原数据证据全保，TyLSoJ实际通过、gjRMpm已托管原native manager完整强核，再新鲜资源准入；X3zNcY等该新queue真实resources_safe后复用原v5恢复110+历史评分/断点。原nativePID/locks/leases/unknown策略不改、最多3review、任何partial审核不重试。4资源/内容单测通过；测试曾快速删除重建在Ceph复用同inode/size/ns造成缓存旧指纹命中，已明确作为原stat-cache已知边界而不宣其能检测完全复现指纹，真正inode替换测试保旧inodeallocated，MISS四snapshot/读中变化/同inode域逃逸/当前资源过期与lease变化拒绝实测。
- 11:22直接覆盖1809complete/1709partial/5077ready-not-joint/214344missing、8595readyunique，mainreview408/42AIerror/0closedbatch；推广后N8s0 run新增23complete/21partial＋641历史resumed，无sharedfailure。仍未全部222939联合标签或448批闭环；本轮零模型15＋subject3＋旧repair3额外候选各根待独立审核/同源去重接入，不能直接加base计数。

### 10-25 已生成三源审核离线恢复与新增日志故障（15:20 UTC）

- 原三源第一次current审核actualworker已启动但0dispatch，binding.input/frame内容相同而source_refs.stat.device跨挂载1048620/1048634不同；coarse3_review_mount_bridge_v1.py先原source_references逐文件fullSHA及路径/inode/size/mtime/ctime全核，仅device例外，在独立proof里保历史/当前身份对；原evidence/input/receipt bytes不改。N6真3source/48RGB/15ROI准备复核通过，零派发历史summary/ledger/stage均immutable snapshot保存后，newresume7e0d7af1…/testbd81c4b6…原11CPUtests主独立通过，config8d07eb22…actualsafe/FLOCK/lease/currentresource完整链允许续审，source预算不重置。
- 续审实际12:54:37Z准入、13:00:47Zlive；13:09:30Zworker_exit与13:09:35Zsafe/restored全真。真实review-dispatch=3、模型load=true，但原LocalReviewer在生成事务后用prepared.source_identity历史device做batch.verify_source，三条都抛sourceassetchanged；这不是3未调用或3质量错误，更不能清预算再调。原review各6产物已存在、normalstop/EOS/generation/request/raw/schema/receipt绑定完整，raw final均pass。旧summary/ledger维持pending/error作为原工程故障证据。
- 主在N6完整重hash三视频/音频/8原refs并核inode/size/ctime_ns/mtime_ns全等，唯一差异确为device1048620→1048634（15:17实机强核），actualinput/label/frame/ROI与原SHA无变化，不以stat代替内容证据。新增coarse3_completed_review_recovery_v1.py仅离线：原m.validate_candidate全部coarse/原facts/媒体PTS/ROI/CSV绑定再核，原old.verify_review_receipt逐6artifactSHA/request原semantic-only＋nativeRGBROI bytes/raw strict schema/final/json/gen模型sha验证；使用actual reviewer.model/code.identity原完整执行treeSHA与原teacher attestation reference，不把简化pathSHA当完整attestation。
- JC6QXm实际exit0恢复3sample-pass、original_model_calls3/recovery_model_calls0、不重派不改原媒体/evidence/claims/raw/评分/pendingsummary。独立新根coarse3_completed_review_recovery_v1：summarySHA147be057a30544fa424b9db0ace1361b49d6837fb894bbf46da3b9f610fea95e，各原receipt a427f460…/253d8a6f…/a39f7fc0…，主已经实际验证并保原失败结果/模型执行证据。3pass只代表稀疏sample审核，不批准原500批/448总体、人评或训练，手侧/手形与style仍按原不确定边界。
- 修复队列收集器bYgHPa在原completed byte2053650遇5bytes `te"}\n` 而安全停，前prefix2053650SHA7fb515…和原stateSHA8451c3fc…/seen2941绑定，旧cursor保。不是空行，suffix能匹配多个terminal，不能猜QIW或把它赋该byteoffset。仅b5/b6限定terminal_index与captured journalprefixunion对账发现4缺：afg2723/VHF3140完整，SQQ2775/QIW2883失败；原Evidence.terminal4全manifest/plan/JOINT/VISUAL/sourceidentity/mediaSHA/CSV/params强核实过，两个failed原classify需reobserve[＋rewrite]。新恢复正在实现：精确5bytes/3stable/prefix/indexunion/4原terminal强核授权下返回4exactrows；其来源是独立权威terminalreconciliation、sourcebyteoffset=null、不把5bytes叫含4SID、原raw/errors/旧state保，沿原collect.admit/classifier/masterbudgets恢复未处理来源。未generic skip、未回写原journal，任何新坏line仍failclosed。
- 原salvage自动恢复0x0Bn6再次因先historicalcapture后全662source/110score proof耗时导致原20minTTL stale门拒，无新watchermodel/数据丢失。新prepare_salvage_current_resource_restore_v1.py(a5341f7c…)/tests4357964c…主12CPUtests独立全过，先原完整proof662/110与旧plan同bytes/sourceSHA、本机model全核→actualfreshholder/PID/start/UUID/normalTERM/pidfd/160GiB/原GPUFLOCK+原output reviewer.lock只读检查，最后放卡前再核actualparent-FD/kernelFLOCK/entry.old原整数nslease/newownedleaseinode/heartbeat。历史capture不改时间、不延20min TTL，actualR/S动态state不当identity。aZLr51实际build与0xpgff真实构建成功后再launch已主托管；未出新live之前不报原审核恢复完成。
- 15:06最新base updated15:02:18Z：2344complete/2144partial/6071ready-not-joint/212380missing、10559readyunique，mainreview469/46AIerror/0closedbatch；旧CPUimport2328processed/1287imported/1030pending/11excluded不能与base相加。N1prefetch256newsourcewall/CPUprepare2770.52秒/交接等待17.54秒（newrun142complete114partial+549历史resume），N8prefetch226newsourcewall/1933.0秒/14.19秒（122complete104partial+641历史resume），持续证明prepare重叠但没有同源无预取walltime对照，不给整体倍数。全222939标签/448审核闭环仍未完成。
- 预取持续验收09:45时13sourcewall/13handoff，实际累计CPUprepare133.63秒、锁前handoff等待0.848秒、最大0.072秒、父13源prepareP0，证明多来源非一次偶然准备重叠；没有同源无预取对照，仍不推断整体倍数。实际run09:41已有7complete/3partial＋549旧终态复用，非全55734完，不用resumed数量当新生成。

### 10-25 收集器实际越过新片段与下一三源准备（15:57 UTC）

- canonical manifest/prod/out 参数重跑的 offset2053650 prepare 已实际成功：4原终态、925项原证据完整强核，2complete/2failed，0模型、0预算。最终共享恢复源码09c32ccb…/testsbb0dccc9…冻结不变；原5bytes、3stable、prefix、813index/809shared/4missing及独立record/receipt/sourceoffsetnull保留。主只启动一次新watch ZaS00j；15:55:15Z实际collector.recovery.state offset2095419（已越2053655）、seen3005，四原终态SHA全已登记，physical-cursor-authorization已存在。QIW原reobserve、SQQ原reobserve+rewrite两job/announced SHA实核通过，budget仍不存在；没有把片段分配给某SID、没有回写原journal或清重试预算。watch继续沿原admit/classify/checkpoint增量处理，未知新坏行仍failclosed。
- 主审核v8四新文件已落地且主独立验收：watch65972a15…/manager2273e1fa…/builder80183807…/test615de6ff…。ciSSrh实际完整20 CPU tests/2.006s/exit0，覆盖原469评分与batch0至少80/7/原policy保全、exactfragment编码源0和独立4reconciliation/sourceoffsetnull、未知片段/前缀/成员/SHA错误拒绝、原pidfd/lease资源门。xWl1WK在真实N5节点完整Preparation925artifact/217.326秒/4source/2complete/2failed通过，0GPU锁/信号/源state写。原早期4reconciliation不能丢；此次新4不是原rawencoded4。此时尚未v8实际资源交接、source_only配置或GPU启动，不能报主审核恢复完成。
- 下一组三空窗mrDP w0/Ecd w1/FSj w3的worker45f9b691…/testc104349c…主完整读取后实际MJxZRE 12tests/13.587s/exit0；另独立审计护栏12tests/5.277s通过、所有147write opens仅临时fixture/0禁止操作。原nativeRGB/ROI请求顺序字节、13/10/14图、原采样、10support/6bodyeventparts、raw整句/refs不变/partsubset、FSj已有neutralfacttext明确派生、shared-load与ambiguous-dispatch预算边界均覆盖。
- oAKWPr实际CPUprepare-only/39秒/exit0，独立OUT curated_coarse_reobserve_fixed3_v1已冻结protocol ddc491a7…/jobs.manifest a27ec388…/source.freeze98a6d925…/cpu.prepare.evidence70092696…；3源原budget before/after全部null、0load/0visual/0text/0reserve。原非target facts/observations/style/captions/usedIDs保持，FSj不伪旧failedfinal成功。新N6G0/1原primary TP2 managed callback/resource入口尚在实现，不能借旧N5入口或N6G3 reviewer身份、不能并发另模型或先reserve；GPU观察与独立审核仍未做。
- 15:49:57Z主覆盖实际2453complete/2244partial/6287ready-not-joint/211955missing，10984ready_unique；已审602/samplepass532/AI错误137/待人工161是当前选择汇总口径，不与各新根或旧import直接相加。全部222939联合标签与448批质量闭环仍未完成。N1/N8预取只证明准备重叠，近期N5完整证据核验仍耗217秒、原salvage构建仍live，整体IO瓶颈未解决。三模型训练N2/N4/N7未触碰；八blocked/ignoredholder没有强杀。

### 10-25 后续真实执行与八节点扩容讨论（16:28 UTC）

- 原salvage v1构建真实失败：safe_chain误对Python资源manager引用走JSON checked，fullproof662/110后JSONDecodeError；原错误日志保留、0新GPU/0模型。显式typed-source v2只替该单一源码引用为完整SHA+四次7字段stat稳定的bytes校验，JSON引用仍原严格parser。主独立17tests通过；新df177b4f…/test779b3fe5…已冻结，Dudz6d实际N6build/run受管执行，尚无newlive，不能报审核恢复完成。
- 新139物理失败队列来源（start1374/count139）已iaA032实际完成：5完整候选、108非rewrite/24细手claim/2原协议回放失败，0模型/0预算，next1513。manifest SHA bc54950a9219268e33b07bebb09841e9df8f51584d2cde55c6ac627dac41e7d8；主逐5result原bytes SHA/joint/allnonempty/无语义训练批准实核。新根仍独立审核待办，不直接加base。
- 主v8源准备后遇旧532233历史load.receipt被重启覆盖；v9实际先修2原splice源，再发现旧4reconciliation同类历史load差异。没有改原hint/record或借新load证明旧call，枚举真实immutable loads按原native JSON序列化恢复exact历史SHA：0000/1950582→2642c7dd…、0003/1165501→5f00958c…、0002/1159453→3599e28c…；保存content-addressed原字节，原模型25文件inventory/8code/实际oldload/hint/record全部绑定。v9私有Evidence factory作用于2原事件和4原对账来源，原v7/v8门、旧4/新4offsetnull、500policy/469及80/7历史保留。主补NormalizedReader原CPUhelper接口遗漏，独立最终15tests已过；DS7igU正在N5实际全scan+8reconciliation+native80/7+freshbuild/run，尚未新GPU分，不报全审核恢复成功。
- 固定mrDP/Ecd/FSj N6G0/1资源入口初版在patchedlease_snapshot读active.cfg前尚未初始化而启动前拒，原primarysafe已经完成，0新预算/模型。独立v2仅移动initializer并明确新exec/config桥接，原ad19/testd18/failedcfg68c/预检8037历史保留、原TTL不延。主39项CPU回归（含真实nestedvalidate→原fresh_pair→patchedsnapshot）通过，gdBjUr受管v2实际16:24:45Zadmission/16:24:50Zlive。16:28:13Z mrDP原预算已恰1且request存在、raw/receipt尚未生成；其它2源未预约，没有候选，不能以live冒3调用完成。受管链只有finite原resources_safe后才恢复原primaryargv/env/out/revision，未完成全目标。
- 已真实3samplepass的另一组Auuit/AzPom/DxNC已source去重派生：旧barbase时点11:51:39Z，3源0新增source、selectedcomplete2724→2727/partial1758→1755，仅该冻结快照；不冒当前覆盖。旧评分保alternative，新revision各自绑定receipt/input/labelSHA。review网页新增固定SHA路由coarse3-review，原server.py不改；主11CPUtests后精确normalTERM旧server505139，oftpD1原8775/同manifest受管恢复，原人工评分前缀保持（实际评分目录当时0文件）。原首页最小相对链接，原媒体权限/Range/评分接口保持。主真实HTTP /coarse3-review与/proxy/8775/coarse3-review均200且页面SHA431f7505…，63RGBROI图+3视频全部66HEAD200、Range206。外部code-server浏览器未亲测；samplepass非人审/全批/训练批准。
- 用户随后明确“当前8节点都可以使用”并要求讨论IO提速，取代该次原3训练节点保留范围。主实际8节点GPU/磁盘只读核，N2/N4/N7各8卡均本项目hold_gpu且normalTERM支持，无大模型训练进程；每节点容器可见磁盘余7.7–8.9TB，实际底层/配额仍需核。尚未取新节点GPU。主建议优先node-local JSON/JPEG输入证据闭包、local事务spool/唯一可靠提交器、历史审核local闭包+增量，原预算必须先共享durable、所有artifact写回后才能terminal；不能只搬MP4或只加worker放大Ceph小文件压力。计时必须含锁前/锁后/模型chat/validator/commit，原sourcewall不能直接减P/V当IO占比。跨sourcebatch另单独验证，不泛改原全局状态或共享loadreceipt。
- 两只读协作者最新网关empty-stream失败已向用户明确说明；非CPU/GPU模型错误、不把失败调用当成功交付，已主读取源码和独立实测的证据按实际保留。全222939标签/448质量闭环仍未完成。



