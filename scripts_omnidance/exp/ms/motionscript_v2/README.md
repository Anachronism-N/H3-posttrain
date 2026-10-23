# MotionScript 独立优化分支：第一版

本目录属于 `codex/motionscript-optimization`，基线为主分支提交 `abb8375078ba0c0506609292451b293a28a14d1b`。新入口是上一级的 `ms_optimize.py`。旧 `nlf_extract.py`、`ms_caption.py`、`ms_lite.py` 及训练入口保持独立，便于同一批数据比较。

本版复用已有 NLF NPZ，完成坐标质量控制、可审计动作事件和完整坐标缓存。无需重新运行 NLF、占用 GPU 或调用远程 LLM。这里的事件和描述是**待核验候选**，不代表真实舞蹈准确率、音乐同步效果或 H3 改善。

后续追加的封闭词表／复读处理见 [REPETITION.md](REPETITION.md)：包括可回查的事件压缩、标签分布审计，以及不输入旧 MS 先验的开放词汇视觉观察。默认只导出请求；只有显式运行 `ms_visual_observe.py observe` 才会将采样帧发送给指定端点。

编舞规划器所需的分段标注方案见 [MotionScript 分段标注与编舞规划器训练方案](../../../../notes/30_motionscript_chunk_annotation_plan_20261024.md)：采用先切窗口、再调用原版规则的思路，区分窗口时间与动作时间，并说明固定时长／按拍分段的小批量验证。该方案尚未接入当前命令或训练导出。

## 已实现的改动

- 对轴角先转为旋转，再通过 SLERP 插值；旋转平滑使用符号连续、归一化的四元数。连续 yaw 在各个有效段内展开，避免先折回 ±180° 再普通平滑造成假反转。
- 按秒设置平滑宽度与允许的缺失时长；短缺口可插值，长缺口和首尾缺失保持无效，不把缺口两边拼成连续动作。
- 使用真实输入时间戳（若有），否则使用 `frame_index / fps`；输出固定间隔时间轴不超过源视频最后一帧的时间。16→20 FPS 是插值，不增加实际观测。
- 区分根节点可用性、各关节可用性和观测／插值支持。可选使用已存 NLF uncertainty 做门控；其原始数值不是经过校准的置信概率。
- 保存 float32 几何数组、float64 时间轴、有效标记及来源协议，避免只有文字结果、没有可重用坐标。
- 提取左右肘膝弯伸、相对躯干的上臂抬落、躯干倾斜、肩髋相对扭转、根朝向变化及水平位移。检测采用变化量、持续时间和反转迟滞阈值，按有效区间处理。
- 所有事件保留身体部位、起止时间、量值、单位、所需关节和来源，按开始时间排序。首版不将同一固定时间 bin 的左右事件自动合并，也不强制填满某个部位的事件数量。
- 生成带出处的英文测量描述，或导出供 GPT／其他 LLM 做事实保持改写的 JSON 请求；**不自动调用任何 API**。
- 保留原始事件，并另外生成去除高度重叠副本、压缩已核实弯伸／抬落交替的 `compact_caption`。紧凑 LLM 请求允许按已核实事件组写一句，避免机械要求每个原语都生成一句；所有原始 ID 和时间仍须覆盖。
- 可选用 v2 坐标运行原版 MotionScript 规则，单独记录其结果；不将新规则描述冒充历史 `ms_orig`。

## 输入与环境

Python 3.10+，NumPy 1.26+、SciPy 1.13+。本地回归环境为 NumPy 1.26.4 / SciPy 1.13.1。

输入是原 `nlf_extract.py` 已保存的 NPZ，至少有：

| 字段 | 约定 |
|---|---|
| `pose` | `[T,72]`，SMPL 轴角，弧度 |
| `j3d` | `[T,J,3]`，J≥22，**米**；沿用原提取脚本除以 1000 后的结果 |
| `valid` | `[T]`，原检测是否有效 |
| `fps` | 正数源帧率；多帧且提供 timestamps 时可省略 |
| `timestamps` | 可选 `[T]`，非负、严格递增、秒；存在时优先用于时间轴 |
| `unc` | 可选 `[T,J]`，启用 uncertainty 阈值时必须存在 |

另需现有 `smplh_neutral_joints.npz`，含 `J_template[52,3]` 和按父节点先后排列的 `parents[52]`。模板和 NLF 私有缓存无需提交 GitHub。

输入文件名须为裸 SID，只包含字母、数字、下划线、短横线；保留 `protocol` / `manifest` 名称。可先使用 ID 清单做小批次。

## 运行

以下为需在持有原数据与模板的环境中运行的示例，路径须替换为实际值：

```bash
python scripts_omnidance/exp/ms/ms_optimize.py \
  --nlf /path/to/MS/nlf \
  --template /path/to/smplh_neutral_joints.npz \
  --ids /path/to/review_ids.txt \
  --out /path/to/MS/ms_optimization_v1 \
  --emit-llm-payload
```

`--out` 必须是独立的新目录，不能包含输入、与 NLF 输入目录互为祖先，或使用已有原版标注目录。脚本不修改源 NPZ，不自动覆盖历史标签。

可调参数包括 `--fps 20`、`--max-gap-seconds 0.2`、`--smooth-seconds 0.0625`、`--min-duration-seconds 0.15`、`--min-angle-change-degrees 20`、`--min-distance-change-m 0.12`。这些是待校准的工程起点，不是已经验证的最佳舞蹈阈值。

`max_gap_seconds` 的定义是两个可靠源观测之间的时间差，减去一个名义源帧间隔。例如 16 FPS 缺 3 帧，对应缺失时长 0.1875 秒。首尾不外推。长缺口内转过几圈属于未知，后续片段不会被假定为恢复了完整转圈数。

如果确需 uncertainty 门控，另传 `--max-joint-uncertainty VALUE`，先在人工核对集上确定阈值与可用覆盖率；默认不假定通用阈值。

### 可选：原版规则对照

```bash
python scripts_omnidance/exp/ms/ms_optimize.py \
  --nlf /path/to/MS/nlf \
  --template /path/to/smplh_neutral_joints.npz \
  --ids /path/to/review_ids.txt \
  --out /path/to/MS/ms_optimization_with_original_rules_v1 \
  --upstream-src /path/to/MotionScript/src \
  --upstream-shim /path/to/MotionScript/shim
```

额外需要原 MotionScript 源码、当前集群使用的 `text2pose` import shim 和 CPU PyTorch。此模式只支持 20 FPS，对每段完整有效的身体序列独立调用原版规则，保留段落时间位置。原算法自身的语言筛选、聚合及 Euler 通道限制依然存在。

本地环境缺少这些依赖；可选原版调用的分段与传参使用 stub 做了契约测试，**没有通过真实上游运行验证**。默认 NumPy/SciPy 路径不依赖 PyTorch、集群绝对路径或旧脚本的 CUDA 环境副作用。

## 输出与恢复

每个 SID 输出：

- `<sid>.coords.npz`：完整坐标、根旋转、时间轴和质量标记。
- `<sid>.json`：质量元数据、事件、带出处测量描述、可选原版结果／LLM payload；固定保留 `semantic_approval=false`、`human_review_completed=false`、`hand_fingers_observed=false`。
- `caption` 是完整事件的审计描述，`compact_caption` 是压缩后的描述视图；`event_summary` 保存原事件、去重映射和每次发生的时间。二者不是两份独立训练样本。
- `<sid>.receipt.json`：对应 JSON 的内容哈希。

数据集另输出 `protocol.json`、`manifest.json` 和 `manifest.receipt.json`。协议绑定输入选择、模板、实现源码、参数、依赖版本及可选上游源码哈希；清单绑定各输入、坐标和报告哈希。

加 `--resume` 只复用相同选择、源码、配置和输入内容的产物，同时核对报告及坐标完整性。更改规则、阈值或 ID 选择时应使用新目录。文件采用同目录临时文件加原子替换，先写坐标与 receipt，最后写报告／清单作为完成标记。崩溃后留下的孤立坐标或 receipt 可重建；报告／清单存在但 receipt 缺失或不匹配时明确失败，需要先人工检查，不能假装该样本完整。哈希与 NPZ 解析使用同一份读取到的字节快照，避免输入路径被并发替换时记录与实际数据不一致。

### 坐标约定

| 数组 | 含义 |
|---|---|
| `time`, `source_time` | 输出与源时间戳，float64，秒 |
| `body` | `[N,22,3]`，每帧移除根 yaw，y 加估计骨盆离地高度 |
| `global_joints` | `[N,22,3]`，固定 Y-up 相机轴系，保留根朝向与绝对骨盆平移 |
| `root_rotation` | 完整根旋转矩阵，是几何旋转的主要依据 |
| `translation` | 相对首个可用输出样本的骨盆位移，旋转到其初始朝向坐标系 |
| `yaw_deg` | 相对初始朝向、各有效段独立展开的连续角度 |
| `coords` | `[N,54,3]`：52 个模板关节＋根 Euler＋位移，供原版规则对照 |
| `valid`, `joint_valid` | 根／地面支持及单关节可用性；完整身体消费者还需检查全部关节 |
| `observed`, `interpolated` | 输出时间是否匹配可靠源观测，或由重采样／短缺口插值得到；平滑可改变观测时刻的数值 |
| `joint_observed`, `joint_interpolated` | 单关节及父链的时间观测／插值支持；事件按其所需关节统计，不以根节点覆盖代替 |
| `segment_id`, `joint_segment_id` | 根与各关节的源支持段编号，无效为 -1；即使低帧率隐藏缺口，也禁止跨段提取事件 |
| `source_detection_valid`, `source_joint_valid` | 原检测标记及经过有限性、可选 uncertainty 与父链检查的源关节标记 |
| `source_uncertainty`, `source_boxes` | 输入有对应字段时原样保留，便于回看观测质量 |

固定轴变换不等于恢复真实世界坐标。`global_joints` 仍是单目估计；若将来接 MG-MotionLLM 或 DEMO，需要继续做其关节、坐标、尺度等规范化。不能把已去 yaw 的 `body` 或带两个虚拟点的 `coords` 直接当作目标模型输入。

地面采用有效段内 1.5 秒滑动低分位足部高度，是启发式估计，可能受相机移动、长时间腾空和地形影响。首版因此不命名跳跃、落地、脚接触、承重、拍手、手指动作、走跑类别或舞种。根转向和 X/Z 位移保留明确坐标符号，不将其猜成解剖左右。

## 后续验证与扩展

先用同一批人工可核对的舞蹈短片比较：原版、v2 坐标＋原版规则、v2 事件描述。分别记录左右／方向错误、漏动作、误报、边界误差及弃权覆盖。描述变长和事件数量增多不等于准确度提高。

新 schema 尚未接入 `compose_caption.py`、`music_scale_data.py` 或 H3 teacher；不得标为 `motion_source=original` 直接替换训练数据。审核通过后再单独实现导出和固定参考图／种子的教师对照。

增加手足观测点、学习式 MG／DEMO 对照和实际模型效果对照属于下一阶段。已有 52 个模板点不表示有真实手指观测；从同一拟合增加网格点也不会自动恢复新动作。

完整回归测试需额外安装 pytest，并从仓库根目录运行（只运行 unittest discover 不会收集坐标模块的 pytest 测试）：

```bash
python -m pytest -q tests
```

测试使用有明确生成方式的合成输入，验证数学、掩码、事件与缓存契约；不把它们称作真实舞蹈准确率测试。

本次本地完整回归为 **119 项通过**。其中视觉入口实际执行了 FFmpeg 抽帧及双视频流一致性检查；没有调用远程模型、运行真实上游 MotionScript，或进行 Omni/H3 训练。
