# 从视频补充旧骨架未表达的动作信息：32 点实验

> **2026-10-24集群实测更新**：下文“canonical公开入口尚未核实、未运行真实NLF”等为代码提交时的开发环境边界，不再代表当前执行状态。已从官方v0.2.0 `nlf_data_files.zip` 用Range取得两小数组，与本地0.3.2模型的55关节＋1024顶点subset逐值匹配，随后固定三个真实视频完成32点直接提取及RGB叠加。此处更新是工程执行与验证结果，**不是点准确率或监督质量批准**。

## 最新真实验证结果

- 官方canonical数组55×3 joints和10475×3 vertices取得且CRC/SHA核验，实际模型内嵌1079点float32匹配maxabs0，未用普通T-pose或867点buffer补全；许可按NLF非商业研究边界，不在Git提交派生数组。来源与匹配小报告见 `results/nlf_official_canonical_match_20261024_v1.json`。
- 固定楼梯/红衣/商店三case均真实81帧×32点提取成功，原生2D像素点、3D米坐标、uncertainty/有限valid保留；没有SMPL拟合/FK补点或推理音频。单GPU结束后恢复项目holder。
- 关键负例：红衣源视频裁到大腿，6足点各81帧全在画外，raw却仍有有限坐标及valid=true。这说明数值有效不等于可见/正确，原foot proxy事件不能直接作为实际脚部事实；4掌点也有画外时刻。
- 楼梯/商店的新增点多数画内，但合掌、脚跟和模糊存在无法可靠判定的定位；掌足平面不退化也不等于方向预测正确。旧完整pose保留腕足旋转，不能把点数增加当准确新增信息。
- 原RGB/direct32原生2D/旧j3d22＋完整24旋转的三栏视频已输出到独立 `MS/point32_direct_fixed3_v1/review/<SID>`；30fps展示对原16fps按PTS零阶保持，未插值真实观测，最后4.17ms CFR舍入明确记录。本地图片和视频不上传GitHub。
- 后续候选派生mask只将画外点判为缺少必要画内支持，不能把画内或低uncertainty标成已验证可见性；原raw与source不改变。准确细节/动作语义人评仍需逐窗原视频核对，不要求为当前12k弱标注流水线伪造全量批准。

以下保留初版设计与当时的环境说明。


本轮目标是增加**由 RGB 支持的动作信息**，而不是增加模板点数或句式数。用户确认不需要细到手指关节／指尖，首版固定为32点。设计参考 [MotionScript分段标注与编舞规划器训练方案](../../../../notes/30_motionscript_chunk_annotation_plan_20261024.md)。独立入口为 `scripts_omnidance/exp/ms/ms_landmarks.py`；不改原提取器、历史缓存或 Omni/H3 训练。

## 点集决策

| 部分 | 点数 | 用途 |
|---|---:|---|
| 身体锚点 | 22 | 与新点在同一次查询中预测，提供腕、肘、肩、踝、膝等参照 |
| 双手掌面锚点 | 4 | 每手食指根侧、小指根侧各一点，与手腕构成掌面；不描述手指弯曲 |
| 双足表面锚点 | 6 | 每足脚跟、前脚掌内侧／外侧各一点，用于足纵轴与足底近似平面 |
| 合计 | 32 | 不含指尖、指节链或面部表情点 |

掌面点的技术名称为 `left/right_index_mcp`、`left/right_little_mcp`，它们是定位掌面的两个查询位置，不是本轮要识别的手指动作类别。足部使用 `left/right_heel`、`left/right_big_toe`、`left/right_little_toe`。

相较只有腕／踝等中心位置的 body22，这些点让掌面、足部朝向及局部形态有了显式几何依据。**不能把“旧事件输入未表达”写成“旧原始数据绝对没有”**：旧 `pose[72]` 本身含腕、足旋转；固定模板也可提供部分朝向近似。是否比完整旧 pose／模板增加准确信息，必须另做真实视频对照，不能仅凭点数认定。

## 实际预测路径

```text
原视频 RGB ＋ 已缓存的人物框
＋ NLF 自身 canonical 空间中的32个具名查询位置
→ get_weights_for_canonical_points
→ estimate_poses_batched（直接图像点预测）
→ 逐帧32点、逐点uncertainty、实际视频时间戳
→ 逐点质量处理与重采样
→ 完整连续坐标／相对角轨迹＋额外细节事件
```

canonical 坐标仅定义“要找人体的哪个位置”；它不是输出坐标。每帧输出由图像预测，不从SMPL pose执行FK生成，不把手部保持在模板姿态，也不在直接接口不可用时退回SMPL拟合。

该直接接口及其与参数拟合接口的区别来自 [NLF固定版本源码](https://github.com/isarandi/nlf/blob/29046738a8101c15dc366ab45945b45e40debf17/nlf/pt/multiperson/multiperson_model.py)。点的解剖名称及索引依据 [SMPL-X关节名称](https://github.com/vchoutas/smplx/blob/1265df7ba545e8b00f72e7c557c766e15c71632f/smplx/joint_names.py)、[足部顶点索引](https://github.com/vchoutas/smplx/blob/1265df7ba545e8b00f72e7c557c766e15c71632f/smplx/vertex_ids.py)。

所有身体锚点和新增点一起查询，避免新非参数点与旧拟合骨架不一致。更换查询点集会影响绝对坐标恢复和TTA尺度对齐，不保证旧身体点逐值不变；分析优先使用同次预测的局部相对几何。

## 运行

### 1. 构造查询规格

需使用与当前NLF匹配的真实canonical资产。两份数组不在公开Git仓库中，不能从一个不存在的raw地址下载；从已有NLF训练／发布资产目录定位：

- `canonical_joints/smplx.npy`：`[55,3]`，米。
- `canonical_verts/smplx.npy`：`[10475,3]`，米。

普通SMPL-X T-pose或旧52点模板不是这些数组的替代品。脚本绑定输入SHA、固定解剖索引和实际查询坐标，但哈希不证明资产确为官方版本。

尚未核实两份数组的小型公开官方下载入口，需要已有NLF原始canonical资产或作者提供。只有TorchScript文件不保证能完成点规格构建；模型源码中的 `cano_all` 只含选中的1024个顶点和关节点，不能当完整顶点表反查任意足部点。

```bash
python scripts_omnidance/exp/ms/ms_landmarks.py build-spec \
  --canonical-joints /models/nlf/assets/canonical_joints/smplx.npy \
  --canonical-vertices /models/nlf/assets/canonical_verts/smplx.npy \
  --out /work/specs/body_palm_foot32.json
```

### 2. 从真实RGB预测额外点

需要PyTorch、Torchvision、可导出直接点接口的NLF TorchScript模型，以及FFmpeg/FFprobe。当前项目原路径是 `nlf_l_multi_0.3.2.torchscript`；必须以实际artifact导出方法为准，脚本保存方法schema并在缺接口时失败。

```bash
python scripts_omnidance/exp/ms/ms_landmarks.py extract \
  --video /data/videos/clip.mp4 \
  --boxes-nlf /data/MS/nlf/clip.npz \
  --spec /work/specs/body_palm_foot32.json \
  --model /models/nlf/nlf_l_multi_0.3.2.torchscript \
  --device cuda --bs 8 --num-aug 3 \
  --out /work/point32_raw/clip
```

本版一次处理一个clip，输出目录必须是独立的新目录。原NPZ只读取 `box`、`valid` 和时间／FPS，不读取其pose作为新点几何。复用已有框不等于新增人物跟踪；旧框若切人，新点也会切人。帧数和时间一致不证明素材配对正确，操作前仍需确认它们对应同一个视频。

探测与解码均固定首视频流，不缩放、不自动旋转、不丢帧；带旋转元数据的源片需先规范化并重新提框。输出保留实际PTS相对首帧的时间，单目相机轴为X右、Y下、Z前；不声称恢复世界坐标。未提供标定时NLF使用55度默认FOV，实际尺度可能偏差。

NLF直接坐标由毫米转米；uncertainty按直接接口原尺度保存，**不能套旧 `joint_uncertainties` 的阈值**。不确定性不是可见性或校准概率。直接预测也可能来自遮挡推断，不能把有数值称作可靠观测。

输出 `raw.npz` 与带哈希的 `manifest.json`。除3D点外保留直接接口的 `points2d_px`（像素，不转米），用于逐帧叠加核验；描述缓存也保留 `raw_points2d_px`。保留视频、框缓存、模型、点规格SHA、完整规格、方法schema、增广次数和坐标约定；`smpl_fitting_used=false`、`template_fk_used=false`。

### 3. 保留连续信息并提取细节

```bash
python scripts_omnidance/exp/ms/ms_landmarks.py describe \
  --bundle /work/point32_raw/clip \
  --out /work/point32_description/clip \
  --fps 20 --smooth-seconds 0.025 \
  --min-angle-change-degrees 20 --min-duration-seconds 0.15
```

`landmarks.py` 不假设22点或父链，逐点插值、保留长缺口，保存原始预测与质量标记。`point_observed` 只表示输出时刻匹配一个可用的原预测，不表示传感器真值或像素可见性。门控参数 `--max-point-uncertainty` 只在经过当前直接接口尺度的人工校准后使用。

输出 `coords.npz` 保留完整连续32点和相对角轨迹，不强制先量化为少数动作标签。`report.json` 给出事件、测量量、逐点支持、来源和配置；描述保持 `semantic_approval=false`。之后交给LLM时可以使用这些连续信息与视频证据，不必只输入旧MS闭集句子。

首版测量包括：

- 掌面法向与前臂的相对夹角。
- 掌面绕前臂、相对于上臂／前臂平面的滚转。手臂近乎伸直或法向投影退化时弃权。
- 足底近似平面法向、足纵轴与胫骨的相对夹角。

这些量对共同刚体旋转和平移不变，避免只因人体／相机整体朝向改变就报局部细节变化。它们仍是几何代理，不能直接命名为孤立的腕关节旋转、真实承重、接触、踮脚或完整舞步；脚跟较高也不自动证明踮脚。局部点抖动、遮挡和近退化平面仍需真实数据核验。

## 与分段编舞方案对齐

新增点是分段动作描述的补充证据，训练正文仍应围绕主要身体动作及其组织方式。不会要求每段出现掌面／足部细节，也不把每个数值变化都写成一句。无可靠新增信息时，可以保留原主要动作描述；与原规则／视频冲突时需复核。

先对整段轨迹完成质量处理和平滑，再按真实时间切片，不在窗口内重置朝向、地面或位置。固定约2秒分段可直接运行：

```bash
python scripts_omnidance/exp/ms/ms_landmarks.py describe \
  --bundle /work/point32_raw/clip \
  --out /work/point32_chunks/clip \
  --chunk-seconds 2 --min-tail-seconds 1.5
```

不足1.5秒的末段并入前段；例如真实媒体长度5.0625秒时得到 `[0,2)`、`[2,5.0625)`。媒体结束来自解码帧duration或视频流时间，不能拿最后一个姿态样本时间代替。没有可用媒体结束信息时拒绝自动分段，不用名义FPS猜一个结束时刻。

按拍策略使用外部已确定的实际拍点窗口清单，本命令不检测beat，也不把全局BPM等距网格伪装成拍点。示例仅说明格式，不是真实标注：

```json
{
  "schema": "motionscript_windows_v1",
  "strategy": "beat",
  "video_sha256": "actual source video SHA256",
  "beats": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
  "beat_source": {
    "detector": "actual detector and version",
    "reliability": "operator review pending; not a calibrated score",
    "audio_sha256": "actual paired audio SHA256",
    "time_frame": "video-relative seconds after applying the documented audio/video offset"
  },
  "windows": [{"start": 0, "end": 2}, {"start": 2, "end": 5.0625}]
}
```

```bash
python scripts_omnidance/exp/ms/ms_landmarks.py describe \
  --bundle /work/point32_raw/clip \
  --out /work/point32_beat_chunks/clip \
  --windows /work/windows/clip.json
```

外部清单需绑定同一视频SHA，覆盖完整媒体、连续非重叠；按拍内边界须命中所提供的拍点。拍点来源／可靠性字段仅是声明，不是程序已经核验音乐与视频配对、音画偏移或beat准确性。拍点不可靠时应选固定分段。

每个窗口独立从其内部样本重新提取候选，不把整段事件或扩大上下文的结果贴进较小窗口。保存计划半开窗口、实际首末样本、原始全局时间和点支持段；跨长缺口不拼接。无事件不等于静止。窗口边界不是动作边界，也不意味着每个动作踩拍、停止或重新开始。

这些 `chunks` 仍标为 `caption_is_training_target=false`，供组装／核对标签使用；没有接原版MS的音乐分段调用，也没有实现规划器最终自然语言标签或训练导出。它们的来源是直接32点，不得标为原版MotionScript。

按参考方案首先做A整段、B固定分段、C按拍分段时，应保持同一坐标来源和原版规则。新增32点作为后续D组，在相同窗口下比较是否补充了准确且有用的细节。不要同时改分段、点集和改写器后把收益归到增点。规划器仍可输入完整音乐、输出整段分段文本，H3无需按chunk分别生成。

## 如何判断新增信息是否有用

先选一批能回看原视频、涵盖手掌翻转和足部变化的短片，区分三个问题：

1. **提取是否准确。** 把32点投影回视频核对左右、掌面／足面方向、遮挡与跳点，不只看骨架是否大致像人。当前缓存还未提供专用投影评审器，需用原视频和坐标可视化工具核验。
2. **是否超过旧表示。** 同时比较旧body22位置、旧完整pose／模板、直接32点。对能由旧pose恢复的朝向，不计作新增自由度；单独记录直接预测新增／改善的有效细节及新增误报。
3. **是否改善监督。** 在相同视频和文本预算下比较原标签与加入已核实细节的标签；检查描述覆盖、事实错误、冗余，再做Omni与H3对照。不要把所有新事件机械拼成更长的训练正文。

可查询更多点只说明架构允许。整人crop的分辨率、运动模糊、鞋和手部遮挡可能让10个新增点没有实用增益；此时应改ROI分辨率／估计器或移除不可靠点，而非继续增加模板点。

## 当前验证边界

本地完整回归252项通过，其中本轮直接点／坐标／细节算子／分段133项。运行了真实FFmpeg合成视频的时间探测／RGB解码，以及注入测试预测器的完整缓存链路。数学测试覆盖：旧身体锚点不变时新增掌面信息、共同SO(3)／平移不变性、退化弃权、逐点缺失／长缺口、单位转换、来源绑定、半开窗口／短尾及候选事件的全局采样索引。

测试预测器和canonical数组明确为合成fixture，不能当作真实NLF效果。本环境未安装PyTorch／Torchvision，未取得匹配NLF权重、canonical资产或实际舞蹈样本；尚未执行真实模型的32点提取，也未训练Omni/H3。当前交付是可在持有这些资产的环境运行的独立实验代码，不是新增细节准确率或生成效果的验收。

从仓库根目录执行完整回归：

```bash
python -m pytest -q tests
```
