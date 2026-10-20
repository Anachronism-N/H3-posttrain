# 舞蹈数据打标操作说明：MotionScript、VLM、姿态、音乐及多维度融合

更新：2026-10-20。本文按当前脚本、真实标注样例和目录核验整理，不把历史设想写成已实现流程。本次是文档与只读核查，未重新启动NLF、VLM或DWPose打标，也未占用GPU。

## 1. 先区分五种任务

| 任务 | 输入 | 方法 | 输出 | 当前状态 |
|---|---|---|---|---|
| 数据可用性质检 | 视频画面 | VLM语义过滤 | 字幕、转场、换装、跳帧、清晰度及pass/quarantine/reject | 有代码、已有精选产物；235B历史批次的精确启动配置未完全追溯 |
| 低层身体动作描述 | 视频→3D姿态 | NLF＋SMPL-H模板FK＋原版MotionScript规则 | 身体部位、相对位置、弯伸、位移及相对时序的英文动作句 | 主线已有产物 |
| 舞种及高层风格 | 视频 | 本地Qwen3-VL，vLLM | 舞种、质感、情绪、身体主导、道具及有序动作要点 | 235B批处理及32B三case对照已有产物 |
| 音乐统计 | 原配乐 | librosa起音＋自相关/DP、RMS/FFT；另有M2拍相位/CLAP | BPM、拍点、能量、起音密度、频谱；辅助表征 | 有代码和缓存；不是音乐决定具体舞步的标签 |
| 细粒度手脸足补充 | 视频→2D全身关键点 | DWPose Wholebody＋待验证事件规则 | 原始134点/分数/可见性；候选手腕、足踝、协调事件 | 2k关键点缓存完成；新增动作规则尚未作为正式标签落地 |

MotionScript原版不是VLM，NLF也不是MotionScript：前者根据3D动作规则生成语言，后者估计身体几何。VLM的舞种判断与MotionScript的具体动作应保留各自来源，不通过简单拼接宣称相互验证。

**当前音乐学生完全禁用text encoder的要求不改变离线打标用途。** 文字可用于离线教师监督、数据检查和评估；不意味着推理时要恢复caption、空白文本或TC参考prefix。标注文本、教师hidden和音乐学生token是三类不同产物。

## 2. 路径与目录约定

### 2.1 共用变量

下面命令均使用这些已核验的根目录；`${...}`只是展开绝对路径的简写，不是要求改变工作目录。

```sh
R=/apdcephfs_gy4/share_302533218/cedricnie
REPO="$R/H3-posttrain"
S="$REPO/scripts_omnidance/exp/ms"
EXP="$R/diff/logs_omnidance/exp"
MS="$EXP/MS"
DATA="$R/OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main"
FILTER="$R/OmniDance_processed/pose_preprocess_outputs/mainline/vlm_filter_235b"
```

源码版本以`H3-posttrain`中的git文件为准；历史执行树是`$R/diff/scripts_omnidance/exp`，结果在`$EXP`。两树可能不同，复现前记录所执行脚本SHA，不因文件同名假定版本一致。

### 2.2 输入、ID与现有标签

| 内容 | 路径（按上面变量展开） | 说明 |
|---|---|---|
| 规范化视频 | `$DATA/videos/<sid>.mp4` | NLF/VLM/audio打标脚本按sid固定查找 |
| 配对音乐 | `$DATA/audio/<sid>.wav` | 不能用生成视频mux音轨冒充源配乐 |
| 参考首帧 | `$DATA/first_frames/<sid>.png` | 数据/生成检查，不是MotionScript输入 |
| 主线候选ID表 | `$MS/ids_all.txt` | 每行一个裸sid；历史11992条候选 |
| 选定2k ID表 | `$MS/selected_2k_ids.txt` | 实际存在；不是`ids_2k.txt` |
| 固定三个case ID表 | `$MS/eval3_ids.txt` | 楼梯、红衣、商店 |
| NLF估计 | `$MS/nlf/<sid>.npz` | 原生帧率、身体参数/有效mask |
| 原版MotionScript | `$MS/ms_orig/<sid>.json` | 当前目录11992个JSON；不是全都非空 |
| 三case原版MotionScript | `$MS/ms_orig_eval3/<sid>.json` | 用于格式、教师及事实核对 |
| VLM结构化分片 | `$MS/vlm/labels_s0.jsonl`至`labels_s3.jsonl` | JSONL，含解析label与原始回答raw |
| 三case32B标签 | `$MS/vlm/eval3_32b.jsonl` | 不是235B全量文件 |
| 音频统计 | `$MS/audio_feats.jsonl` | 每行一个sid/特征或err |
| 合并2k标签 | `$MS/captions_orig_2k.json` | `dims/caption/caption_structured/motion_original` |
| 原富文本 | `$EXP/H3AR/full12k_rich_prompts.json` | 原11992候选的rich文本；生成来源需另追溯 |
| 独立VLM动作CSV | `$EXP/E10/val47_motion_captions.csv` | `sample_id,motion_caption`，用于明确的VLM备选 |
| DWPose2k缓存 | `$MS/dwpose_2k/<sid>.npz`、`<sid>.coverage.json` | 原始点/分数/人物跟踪；覆盖不等于正确 |
| DWPose2k汇总 | `$MS/dwpose_2k_summary.json` | 身体/足/脸/左右手覆盖 |
| 三caseDWPose | `$MS/dwpose_eval3/` | 足部缺失case必须弃权 |
| M2辅助音乐缓存 | `$R/diff/data/omnidance/audio_cache/m2_feats/beat/`、`clap/` | 40Hz六通道与512维全局CLAP |

历史全量验收记录：原版MS11992条中11903非空、67无句、22低观测弃权，见[notes/22 §27.7](22_h3_bidir_blur_attribution_20261013.md)。本次重新确认目录文件数与若干真实样例；全量逐JSON复算遇CephFS慢读取后停止，**没有把历史比例当本次重新实测的准确率**。目录与ID池规模都不代表精选集规模。

### 2.3 模型、规则与Python环境

| 内容 | 已核验路径 | 用途/资源 |
|---|---|---|
| NLF TorchScript | `$R/models/nlf/nlf_l_multi_0.3.2.torchscript` | 单目SMPL身体估计，单worker用1GPU |
| SMPL-H关节模板 | `$R/models/smplh/smplh_neutral_joints.npz` | CPU FK，`J_template/parents`；不发布受限模型文件 |
| MotionScript上游 | `$R/MotionScript/src/` | `captioning.py/captioning_data.py/posecodes.py` |
| 本地import shim | `$R/MotionScript/shim/text2pose/posescript` | 兼容`text2pose.posescript.captioning`导入；实际目录存在 |
| Qwen结构化主模型 | `$R/models/Qwen3-VL-235B-A22B-Instruct` | 本地vLLM，当前脚本默认TP8，即一节点8GPU |
| Qwen三case备选 | `$R/models/Qwen3-VL-32B-Instruct` | 可通过`--model`指定；不要因参数小就认定同等精度 |
| Qwen动作句备选 | `$R/models/Qwen3-VL-8B-Instruct` | E10 transformers单GPU；脚本默认另指`/dev/shm/od/vlm/Qwen3-VL-8B-Instruct` |
| DWPose工具树 | `$R/pose-preprocess/third_party/scail_pose/DWPoseProcess/` | 本地Wholebody ONNX调用 |
| DWPose权重 | `$R/pose-preprocess/third_party/scail_pose/pretrained_weights/DWPose/yolox_l.onnx`、`dw-ll_ucoco_384.onnx` | detector＋pose；必须实际CUDA provider |
| 音画观察模型 | `$R/models/Qwen2.5-Omni-7B` | 音视频联合意见，不是人工标签 |
| CLAP | `$R/models/clap-htsat-unfused` | M2全局音乐语义表征，不是动作事实标签 |
| NLF/MS/DWPose环境 | `/opt/venv_eval/bin/python` | 有系统site-packages；存在不等于所有节点依赖一致 |
| vLLM环境 | `/opt/venv_vllm/bin/python` | 235B/可选32B视频结构化标签 |
| H3/辅助音乐环境 | `/opt/venv_h3/bin/python` | M2与离线H3教师编码 |

环境和权重可能只在特定节点预缓存。启动前先告知节点/GPU需求，核验GPU占用、主机/cgroup内存、模型目录及依赖；不直接运行历史`run_vlm_node.sh/run_nlf_node.sh`占卡包装器，因为它们会操作holder和后台worker。本文不要求安装/重跑所有模型。

## 3. 选样、视频规范与质量过滤

### 3.1 ID必须绑定实际媒体

1. 选择一个固定ID清单或通过精选CSV的`video_stem`选样。
2. 检查同sid视频/配乐是否存在、配对是否正确、是否已裁到目标舞蹈段。
3. 用视频解码读取真实帧数/FPS/宽高；记录裁剪起点、时长、音频偏移。不能仅信文件名`704x1280`。
4. 原NLF头注释称81帧/16fps，但代码读取输入视频**全部帧**；它不自动裁5秒、转16fps或缩放704p。
5. DWPose取原生连续帧的前`81/16=5.0625s`；E10看前5.0625秒均匀16帧；结构化VLM脚本没有显式起止裁剪。不同任务要主动统一被观察片段，否则同sid标签可能不对应同一时间范围。

本地现有精选CSV还有需要注意的数据边界：header中`video_stem/error`等字段重复；实查`pass/subset.csv`首行`video=x`，不是有效路径。不要将CSV的`video`列无验证直接交模型，按`video_stem`与规范化媒体映射核对，保留并报告错误行，不默默当通过。历史222939条pass及其train/val/test划分不等于MotionScript11992候选或选定2k全部来自当前精选集。

### 3.2 VLM语义质检脚本与字段

源码在主仓库之外，必须保留这些绝对路径：

- `$R/pose-preprocess/python/vlm_semantic_filter.py`：异步HTTP客户端，当前完整prompt/schema在42–58行。
- `$R/pose-preprocess/python/vlm_inference_server.py`：一个GPU一个本地vLLM服务，端口按列表顺序递增。
- `$R/pose-preprocess/python/vlm_filter_direct.py`：历史小规模transformers直调8B，模型/输入/输出硬编码，**没有通用CLI**。
- `$R/pose-preprocess/run_vlm_filter.sh`：stage7包装器，注意其默认输出根与实际`mainline/vlm_filter_235b`不同。

质检四组维度：

| JSON组 | 字段 | 观察要求 |
|---|---|---|
| `subtitle_overlay` | `has_text/severity/text_type` | 检查烧录字幕/水印/logo遮挡；场景本来就有的商店招牌不当字幕覆盖 |
| `scene_continuity` | `has_scene_cut/has_costume_change/is_edited_compilation/person_consistency` | 场景变化、换装、拼接与人物一致性分开 |
| `frame_coherence` | `has_frame_jump/jump_count` | 检查采样中跳帧；5帧无法排除采样点之间的短异常 |
| `face_body_clarity` | `face/body` | clear、blurred、occluded等枚举，见源prompt |
| 总判定 | `overall_pass/reason` | 模型判断＋后续规则，不替代人工事实检查 |

当前客户端默认均匀取全片5帧、最长边不超过672、JPEG quality85；请求temperature0.1、max_tokens600、JSON response_format。没有输入配乐或旧caption。`VLMResult.category`规则优先级：请求错误→rejected；遮挡字幕minor/major、至少1次frame jump、指定模糊/遮挡或overall_pass false→rejected；否则转场/换装/拼接→edited_clips；其余pass。注意不是所有prompt枚举都被规则直接拒绝，例如too_small/person_inconsistent未单独列拒绝，需要观察overall_pass与人工检查。

现有类别及输出：

```text
$FILTER/vlm_filter_results.csv
$FILTER/per_stem/<sid>.json
$FILTER/pass/subset.csv
$FILTER/edited_clips/subset.csv
$FILTER/rejected/subset.csv
$FILTER/splits/train.csv
$FILTER/splits/val.csv
$FILTER/splits/test.csv
$FILTER/splits/split_summary.json
```

`split_summary.json`记录222939个pass：train200568、val11110、test11261，seed42、按source_width×source_height分层90/5/5，part049_val的37条纳入test。四组质检维度**不是四个数据split**。目录`235b`名称不能单独证明历史模型/抽帧/完整prompt版本；现有代码的8B默认也不能反向覆盖235B历史批次，需要运行日志/模型revision另行绑定。

复现客户端示例（仅命令模板，不在本次执行）：

```sh
# INPUT_CSV应先核验：video绝对路径或相对DATA_ROOT可解析，video_stem为有效ID。
# ENDPOINT使用已经部署且与客户端鉴权方式兼容的内部服务地址。
INPUT_CSV=/path/to/verified_input.csv
DATA_ROOT="$DATA"
ENDPOINT=http://localhost:8000/v1/chat/completions
/opt/venv_vllm/bin/python "$R/pose-preprocess/python/vlm_semantic_filter.py" \
  --input-csv "$INPUT_CSV" --dataset-root "$DATA_ROOT" \
  --output-root "$MS/vlm_filter_repro" \
  --endpoints "$ENDPOINT" --model "$R/models/Qwen3-VL-8B-Instruct" \
  --n-frames 5 --frame-size 672 --concurrency 8 --max-videos 3 --resume
```

此处`/path/to/verified_input.csv`是明确占位，不是已有文件。当前server包装器传`--api-key EMPTY`，客户端`session.post`没有Authorization header，两者直接配对可能返回401；执行前应核验服务鉴权，不能把该示例当已验证的一键server/client部署。不要把内部媒体传到外部API。

**恢复风险：** client的error JSON也可能被`--resume`读回；`result_from_dict`没有完整schema检查，错误JSON缺省字段有被默认当正常结果的风险。运行后必须独立检查error/字段完整性/真bool枚举，再分类；失败ID用单独新清单和新输出重跑，不盲删历史错误记录。本文记录问题，不擅自修改质检代码。

## 4. 原版MotionScript：详细步骤

### 4.1 第一步：NLF单目身体估计

实际入口：`$S/nlf_extract.py:10–67`。

```sh
# 单GPU示例；已有输出会跳过，不会因此重新验证其内容。
CUDA_VISIBLE_DEVICES=0 /opt/venv_eval/bin/python "$S/nlf_extract.py" \
  --ids "$MS/eval3_ids.txt" --out "$MS/nlf_repro" --bs 27 \
  --shard 0 --nshard 1
```

数据流：decord全部帧RGB→`[T,3,H,W]`→NLF `detect_smpl_batched`→逐帧取检测框面积最大的人。**这不是跨帧身份跟踪**，多人时可能切换主体；最好先选单人片段并检查人物一致性。NLF没有接受任何人时可缺boxes键，脚本记录NO_DETECTION，不填假姿态。

NPZ字段：

| 字段 | shape/dtype含义 |
|---|---|
| `pose` | `[T,72]` float32，24个SMPL旋转向量 |
| `betas` | `[T,10]` float32，估计体型参数；后续FK使用固定模板，不是由betas生成个体骨架 |
| `trans` | `[T,3]` float32，NLF原返回translation；后续主适配使用`j3d[:,0]`骨盆，不把此字段默认当米单位 |
| `j3d` | `[T,24,3]` float32，NLF输出除1000，单位米 |
| `unc` | `[T,24]` float32，原关节不确定性，不是校准后的动作置信度 |
| `box` | `[T,5]` float32，原检测框 |
| `valid` | `[T]` bool，是否有选中人 |
| `fps` | 视频实际平均FPS |

无检测帧的pose/j3d等保留NaN；输出先写`<sid>.npz.tmp.npz`再原子replace。多worker动态领取靠`O_EXCL <sid>.npz.lock`；异常可能留孤锁。不能盲删锁，先核验worker/PID和实际文件；已有NPZ被直接skip，不等于finite/ID/时长通过。

### 4.2 第二步：身体坐标适配

实际入口：`$S/ms_caption.py:75–118`，不是DWPose2D→MotionBERT的历史设想。

1. `valid.mean()<0.7`：返回弃权，不生成假动作。
2. 其余按有效帧插值pose与骨盆；长缺口也可能被插值跨过，因此有效率通过仍需看缺口长度。
3. 前22个身体关节旋转来自NLF `pose[:66]`；52关节SMPL-H模板的其余手指旋转固定为单位矩阵。
4. 相机坐标通过diag(1,-1,-1)转换，body用米、y朝上；骨盆相对首帧位移。
5. 身体每帧去yaw保持朝向+z，同时另存yaw变化用于转身统计；首帧位移朝向也对齐。
6. 足部几何＋深度回归及1.5秒滑动低分位估地面，减小单目竖直漂移；不是接触/承重真值。
7. 根欧拉角按源适配约定转为度，yaw相对首帧；body、translation与角度平滑并重采样20Hz。
8. 拼接`coords[T20,54,3]`：52关节＋一行根欧拉角＋一行translation。虚拟根信息不能当普通空间关节点解释。

**手指边界：** 模板产生52关节不代表观测了52关节。`hand_fingers_observed=False`必须保留；不能用当前MS声称真实握拳/张指、掌心朝前或精确手指屈曲。

### 4.3 第三步：调用原版动作规则

```sh
CUDA_VISIBLE_DEVICES='' /opt/venv_eval/bin/python "$S/ms_caption.py" \
  --nlf "$MS/nlf_repro" --ids "$MS/eval3_ids.txt" \
  --out "$MS/ms_orig_repro" --workers 4 --shard 0 --nshard 1
```

这里示范3case CPU流程，批量历史使用16workers/节点；workers不是GPU数。父进程先导入依赖，避免Pool initializer缺matplotlib等后无限补worker。上游原版与本地坐标适配应分别记录SHA。

实际调用：

```python
cap.main(coords, save_dir=..., babel_info=False,
         simplified_captions=False,
         apply_transrel_ripple_effect=False,
         apply_stat_ripple_effect=False,
         random_skip=False, motion_tracking=True, verbose=False)
```

每样本Python/NumPy/torch seed0。原版结合posecode静态关系与motioncode动态变化，再聚合部位、左右、同时/先后等语言；不是简单每秒写一句。常见原规则包括角度、距离、X/Y/Z相对位置、与垂直轴关系、地面关系、弯伸、靠近/远离、位移与朝向变化。源定义：

- `$R/MotionScript/src/captioning.py`：原算法入口。
- `$R/MotionScript/src/captioning_data.py:44`：posecode阈值/类别与模板；例如角度45/75/105/135/160度、距离0.20/0.50/0.80米、相对位置±0.15米、onGround0.10米。
- `$R/MotionScript/src/posecodes.py`：几何运算。

motioncode一些阈值是类别转移/计数，不是直接米或度，不能把全部阈值统一解释成距离。修改阈值或模板后是新规则版本，应单独消融而不是仍称等价原版。

输出JSON：

```json
{
  "sid": "wJPrpFsThWAcBQIw",
  "valid_ratio": 1.0,
  "ms": ["The left hand moves farther from his right elbow gradually.", ""],
  "ms_nonagg": ["..."],
  "kin": {"speed": 0.0, "travel": 0.0, "net_turn": 0.0},
  "source": "MotionScript original rules; NLF/SMPL-H body-only input adapter",
  "hand_fingers_observed": false
}
```

这是**schema节选**：首句摘自真实样例，省略正文并用0.0示意数值类型，不是该case实测kin值。真实完整样例在`$MS/ms_orig_eval3/wJPrpFsThWAcBQIw.json`。`ms`是聚合句列表，`ms_nonagg`是非聚合结果，可能含空字符串；低观测时只有`sid/valid_ratio/ms:null`。正文拼接应过滤空句，不能把空数组/空句等同成功标签。

## 5. 运动学维度：原始连续数值与分档要分开

`$S/ms_caption.py:121–155`输出kin：

| 维度/字段 | 实际计算 | 注意 |
|---|---|---|
| `speed/speed_p90` | 前22body关节20Hz差分速度及分位数 | 单目估计/平滑后指标，非动作质量 |
| `arm_speed/leg_speed/torso_speed` | 指定部位关节速度，torso相对骨盆 | 部位索引按SMPL，不复用DWPose索引 |
| `accel/sharp` | 二阶差分加速度；高于2×加速度中位数的比例 | 噪声可被当sharp，不能直接等同popping |
| `travel/disp` | 骨盆水平路径长度/首尾净位移 | camera运动和单目漂移可污染 |
| `turn/net_turn` | 累计yaw变化绝对量/首尾yaw差 | 累计转角与净转角不同，不能互换 |
| `pelvis_h_med/min/range` | 相对估计地面的骨盆高度 | 低位动作依赖地面估计与人体模板 |
| `jumps/airborne_frac` | 两脚最低点高于0.10米、连续至少2帧的段数/比例 | 估计离地，非真实腾空接触标注 |
| `hands_above_head_frac` | 双手最高点相对头高于0 | 模板手腕近似，不含手形 |
| `repetition` | 平均速度自相关，约0.4–2.5秒lag取峰 | 重复性指标，不是音乐同步或拍号 |
| `symmetry` | 左右臂速度相关系数 | 速度相关，不是镜像几何一致性 |
| `frames` | 重采样20Hz后帧数 | 不是原视频帧数 |

保留原连续数值，类别分档在compose阶段另做。不要只保存“高能量”而丢原速度、阈值与来源池。

## 6. VLM舞种、高层动作与风格标注

### 6.1 235B结构化主线

实际入口`$S/vlm_label.py`，只给视频＋分析指令，不喂旧caption，也不喂音乐。它描述**画面里的舞蹈**，不能据此直接标音乐本身的情绪或精确拍点。

```sh
# 235B默认TP8：需要提前安排一节点8GPU，不能在2卡上直接使用这个默认模型。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /opt/venv_vllm/bin/python "$S/vlm_label.py" \
  --ids "$MS/eval3_ids.txt" --out "$MS/vlm/labels_repro_235b.jsonl" \
  --model "$R/models/Qwen3-VL-235B-A22B-Instruct" \
  --tp 8 --frames 16 --bs 3 --shard 0 --nshard 1
```

模型存在不代表8GPU已空闲；本次没有执行此命令。历史全量通过`labels_s0..s3`分片，每条ID只属于一个分片；改nshard/ID顺序前先记录划分。

实际vLLM设置：`LLM.chat`，max_model_len16384、GPU fraction0.88、temperature0、max_tokens400、每prompt一个视频、视频抽帧数默认16、processor max_pixels=448×800。通过本地file媒体输入，不是外部服务API。

完整prompt及schema：`$S/vlm_label.py:7–28`。开头要求分析约5秒单人舞蹈，回复单个JSON，包含下表**全部15个键**。

| 键 | 枚举/格式 | 含义 |
|---|---|---|
| `genre` | 下列22类之一 | 主要舞种 |
| `genre_2nd` | 同22类或`none` | 第二候选，不是正式第二标签已监督 |
| `genre_conf` | high/medium/low | 模型自报置信度，非校准概率 |
| `sub_style` | 短自由文本 | 子风格；可能夹场景/人物信息，需审计 |
| `energy` | low/medium/high | 视觉运动能量 |
| `tempo_feel` | slow/moderate/fast | 视觉速度感，不是测得BPM |
| `quality` | sharp and percussive/smooth and flowing/bouncy/mixed | 动作质感，不是视频清晰度 |
| `mood` | 1–3词 | 可见表达/观感，不是真实心理状态 |
| `body_focus` | arms/legs and footwork/torso and hips/whole body | 主导部位 |
| `level` | upright/crouched or low/includes floor work | 高低层次 |
| `travel` | stays in place/moves around a little/travels across the floor | 位移程度 |
| `turns` | none/some turns/spins | 转身观感 |
| `jumps` | none/some jumps/many jumps | 跳跃观感 |
| `key_moves` | 3–5个短动作短语，按时间顺序 | 没有精确秒级边界；不得虚构时间戳 |
| `props` | none或道具名 | 可见扇子/袖子/椅子等，不等于服装完整描述 |

schema检查按上述15项逐一验证；字段数量与rich文本的五个章节是不同概念。

22类原词表：

```text
chinese classical; chinese folk; ballet; contemporary; lyrical;
modern jazz; jazz funk / heels; k-pop; hip-hop; popping; locking;
breaking; house; krump; waacking; street jazz; latin; belly dance;
indian / bollywood; other ethnic / folk; pop / casual; freestyle / other
```

输出一行：`{"sid":"...","label":{...},"raw":"模型原始JSON文本"}`。真实32B例在`$MS/vlm/eval3_32b.jsonl`，楼梯genre=k-pop、body_focus=arms、key_moves按观察顺序；商店sub_style实际包含`in a store setting`，说明自由字段有场景泄漏，不能直接拿去做跨人物/场景动作监督。

`parse`仅regex提取大括号＋json.loads，不验证字段类型/枚举/时序事实。`label:null`依然记为done，续跑不自动重试。应独立检查15键、类别枚举、key_moves长度/字符串和同sid重复，保留raw；失败用单独新输出重跑。历史解析失败0只表示JSON可解析，不是动作准确率。

### 6.2 独立8B视频动作句

入口`$REPO/scripts_omnidance/exp/e10/caption_motion.py`，不是把MS正文交LLM改写，而是**重新看视频描述动作**：前5.0625秒均匀16帧，高448且宽按28倍数，deterministic generate最多220 tokens。

```sh
CUDA_VISIBLE_DEVICES=0 /opt/venv_eval/bin/python \
  "$REPO/scripts_omnidance/exp/e10/caption_motion.py" \
  --csv "$EXP/E2/full_joint.csv" --out "$MS/vlm/motion_only_repro.csv" \
  --vlm "$R/models/Qwen3-VL-8B-Instruct" --max_dur 5.0625 --shard 0 --nshard 1
```

输入CSV必须有`sample_id,video`。这个脚本**没有`--limit/--rows`**；上面命令会处理该CSV所含全部待处理行，先生成并核验小量CSV再运行小试，不能因固定推理case是3个就假定此CSV仅3行。

指令要求5–7句、具体部位/方向/方式、开头/中间/末尾顺序，禁止服装、外观、背景、镜头、情绪、风格/质量及不可见事实。输出`sample_id,motion_caption`；短句数与动作事实没有代码强验，仍需人工review。其caption不能标成“原版MotionScript”等价结果。

### 6.3 原rich文本与事实保持改写

已有rich文本组织为技术动态、编舞词汇、表达意图、镜头、人物外观/环境五章；存储路径见2.2。当前读取/融合脚本没有证明该历史rich生成模型及精确抽帧方式，不能把它归为当前235B结构化脚本输出。

原版MS可能语法不自然、时间连接词绕、含性别代词。当前原则：先保留原文；同内容格式对照差异小则继续原版，不无谓全量LLM改写。

- **structured格式**：`compose_caption.py --format_probe`仅加标题/换行，正文动作不变。
- **neutral代词**：`music_text_condition.py:227–248`用规则替换he/she/his/her/him等为the dancer或the dancer's，保存原字符位置和替换表；不改左右、方向和动作。
- **LLM事实保持改写**：是可选后续流程，当前没有已核验的统一批处理CLI。输入原动作事实/顺序，要求输出同事实、去人物/场景泄漏，不新增步法；每条保存原文、改写文、模型/prompt、变更理由和人审。不能写不存在的`rewrite_ms.py`命令。

```sh
# 仅相同动作正文排版对照：不需要VLM或GPU。
/opt/venv_eval/bin/python "$S/compose_caption.py" \
  --motion_source original --ms "$MS/ms_orig_eval3" \
  --format_probe --out "$MS/format_body_repro.json"
```

## 7. 音乐维度与节拍辅助方法

### 7.1 主标注音频统计

入口`$S/audio_feats.py`，在独立进程运行，避免librosa/numba与torch/ONNX混合导入造成段错误。

```sh
CUDA_VISIBLE_DEVICES='' /opt/venv_eval/bin/python "$S/audio_feats.py" \
  --ids "$MS/eval3_ids.txt" --out "$MS/audio_feats_repro.jsonl" --workers 4
```

读取整个配对WAV，mono22050Hz；librosa起音强度→60–180BPM内的带120BPM先验自相关峰→纯NumPy Ellis DP拍点。**此主脚本不是直接调用librosa.beat.beat_track**；另一个M2入口会调用。

输出：`sid,dur,bpm,beats,rms,rms_seg[3],onset_rate,centroid`，beats秒、centroid Hz、onset_rate峰/秒。RMS由2048窗口/512步长统计，三段均分的是特征帧序列。`err`行是失败，不要默认缺音乐即0BPM。短/静音/弱节奏片段可能仍产生先验驱动拍点，需要人工或已校准探针验证。BPM不表示具体动作、拍号、强拍或卡点真值。

### 7.2 M2六通道拍相位与CLAP

入口`$REPO/scripts_omnidance/exp/h3ar/m2_feats.py`：输入不是裸ID表，必须每行`sid<TAB>wav绝对路径`。

```sh
# AUDIO_LIST是预先核验的两列TSV，不可直接用ids_all.txt。
AUDIO_LIST=/path/to/verified_id_audio.tsv
CUDA_VISIBLE_DEVICES='' /opt/venv_h3/bin/python \
  "$REPO/scripts_omnidance/exp/h3ar/m2_feats.py" \
  --phase beat --list "$AUDIO_LIST" --out "$MS/m2_repro" --shard 0 --nshard 1
CUDA_VISIBLE_DEVICES=0 /opt/venv_h3/bin/python \
  "$REPO/scripts_omnidance/exp/h3ar/m2_feats.py" \
  --phase clap --list "$AUDIO_LIST" --out "$MS/m2_repro" \
  --clap "$R/models/clap-htsat-unfused" --shard 0 --nshard 1
```

beat保存`[207,6]` float16、40Hz网格，列顺序：归一化起音、25ms sigma高斯拍点脉冲、sin相位、cos相位、归一化RMS、tempo/200。网格`t=i/40`最后时间5.15秒，207格名义覆盖5.175秒；不可将40Hz标签说成16fps视频原生帧。

少于2个拍点时M2会补周期拍点，必须记录这种先验风险；不能拿补出来的phase当真实听觉标注。CLAP输出L2归一化512维float16全局向量，是风格/情绪表征，不是逐帧舞步或动作顺序。缓存有文件不代表节拍检测已正确校准。

## 8. DWPose手、脸、足与协调：缓存与候选规则

### 8.1 已执行的134点原始缓存

```sh
CUDA_VISIBLE_DEVICES=0 /opt/venv_eval/bin/python "$S/dwpose_full_extract.py" \
  --ids "$MS/eval3_ids.txt" --out "$MS/dwpose_repro" \
  --shard 0 --nshard 1 --num_threads 1
```

源码明确禁止`DWPOSE_ALLOW_CPU=1`，必须detector和pose session实际CUDAExecutionProvider；先导入torch加载CUDA库再初始化ONNX。原视频前5.0625秒按原生连续帧提取，不抽16帧；首个人取最大框，后续IoU≥0.3跟踪，拒跳到不相交的人，缺帧保留NaN。

布局和字段来自`dwpose_full_extract.py:27–64,122–207`：

| 点组 | 半开范围 | 数量 |
|---|---|---:|
| 身体 | `[0,18)` | 18 |
| 足 | `[18,24)` | 6 |
| 脸 | `[24,92)` | 68 |
| 左手 | `[92,113)` | 21 |
| 右手 | `[113,134)` | 21 |

DWPose原COCO Wholebody133插入neck并重排为当前134点，**不要直接用SMPL22关节索引或裸COCO索引解读**。`joint_layout`保存名字、映射和分组。

NPZ：`xy[T,134,2]` pixel坐标、`score[T,134]`、`boxes[T,4]`、`valid[T,134]`、`tracked[T]`、`missing_reason[T]`（0接受/1无检测/2跟踪拒绝）、`frame_indices`及fps/h/w/schema/threshold等。`valid=tracked & score>0.3`；低分坐标不改，缺失NaN不补假点。每条另写coverage JSON；重新读缓存会验证完整shape/dtype/ID/layout，孤coverage无NPZ直接报错。

实查2k汇总身体97.04%、足88.03%、脸99.13%、左右手90.77%/90.40%，是点×帧分数覆盖，不是姿态/动作准确率。三case红衣足部覆盖仅约4.5%，不可因此生造足尖/抬跟/接触标签。

### 8.2 候选扩展维度（尚未正式打标）

| 部位/维度 | 候选事件 | 必须门控 | 不能声称 |
|---|---|---|---|
| 手形 | 可见张指/弯指/变化 | 手像素尺寸、21点连续分数、遮挡 | 完整真实3D屈指或掌向 |
| 手腕 | 掌轴与前臂二维夹角变化 | 腕/掌/肘可见、持续窗口 | 精确旋前旋后/掌心前后 |
| 肩 | 去躯干倾斜/全身升降后的肩残差 | 肩与躯干同时可靠 | 肩胛骨3D旋转 |
| 头颈 | 头肩相对侧倾/摆动 | 侧脸、遮挡或小脸弃权 | 精确3D点头/旋转 |
| 髋/躯干隔离 | 肩髋相对摆动 | 去平移/转身、躯干点连续 | 真实骨盆前后倾 |
| 足踝/足尖 | 抬跟、足轴变化 | 足点/鞋/地面可见 | 真实承重、接触力 |
| 脸 | 嘴/眉眼几何变化 | 面部尺度/连续可见 | 开心/愤怒等心理真值 |
| 跨部位协调 | 肩→肘→腕峰值、相位/先后 | 时间连续、长缺口弃权 | 原MS完全没时序或新规则已证明创新 |

完整历史候选与消融原则见[notes/22 §27.4](22_h3_bidir_blur_attribution_20261013.md)。**当前2k DWPose只是观测缓存，compose没有消费这些新增事件**。实施扩展时建议独立新版本，并按正例/负例/不可判断人工核对precision、覆盖率、时间边界误差；门控开/关、各部位单独消融，不以更多文字当更好标签。

## 9. MS-lite及其他3D方法的边界

`$S/ms_lite.py`是已实现的另一个确定性三段事件描述器，复用NLF坐标，输出`segs/kin`与`<sid>_c20.npz`，可标抬臂/弯肘/踢腿/低位/转身/换步等。参数有`--nlf --out --workers --limit --gq`，没有`--ids/--shard`。

```sh
CUDA_VISIBLE_DEVICES='' /opt/venv_eval/bin/python "$S/ms_lite.py" \
  --nlf "$MS/nlf_repro" --out "$MS/ms_lite_repro" --workers 4 --limit 3 --gq 0.25,0.55
```

这是**新输出示例路径**；当前未发现叫`$MS/ms_lite`的历史目录。原script头部“原版大多空句”的旧判断已被11992原版结果纠正，不作为现在选择lite的依据。lite不是原版MotionScript等价复现，不能把同坐标适配和不同事件阈值合并归因。

MotionBERT、VideoPose3D、GVHMR、WHAM在当前主线只查到历史规划，不是上述NLF流程的实际依赖，也没有已验收的统一打标CLI。不要把“DWPose→3D lifting”原设想写成已经执行。

## 10. 合并标签：实际维度来源与优先级

### 10.1 可复现合并命令

```sh
CUDA_VISIBLE_DEVICES='' /opt/venv_eval/bin/python "$S/compose_caption.py" \
  --vlm "$MS/vlm/labels_s*.jsonl" --ms "$MS/ms_orig" --motion_source original \
  --audio "$MS/audio_feats.jsonl" --rich "$EXP/H3AR/full12k_rich_prompts.json" \
  --count 2000 --seed 0 --out "$MS/captions_orig_2k_repro.json"
```

`--vlm`通配符要加引号，由Python glob展开；新输出不能覆盖正式标签。合并交集是有可解析VLM label且MS非空的sid；原版正文不自动混入VLM key_moves。`--count`从舞种×能量×层次×身体主导桶轮转选样，不是均匀随机，也不自动等于精选集/split。默认count0保留全部可用交集。

### 10.2 逐维度决策

| 合并`dims`字段 | 实际来源/规则 |
|---|---|
| `genre/coarse` | 合法VLM genre优先；非法/low置信且rich编舞章节有关键词时rich兜底；再fallback freestyle；COARSE合并为chinese/classical_western/contemporary/street/latin/ethnic/pop |
| `genre_conf/rich_genre/sub_style` | 保留VLM置信、rich候选及自由子风格，不代表一致真值 |
| `energy` / `energy_vlm` | NLF kin.speed在当前合并交集33.3/66.7分位分low/medium/high；保留VLM值 |
| `travel` / `travel_vlm` | disp>0.35米或travel>1.6米→跨场；否则travel高于当前池上分位→少量移动，否则原地；另保留VLM |
| `level` / `level_vlm` | pelvis_h_min<0.45米→low floor work；<0.7→low crouches；否则upright |
| `jumps` / `jumps_vlm` | kin.jumps≥2或=1形成句；VLM jumps=none时可否决，不因此声明VLM必正确 |
| `turns` / `turns_vlm` | abs(net_turn)>270度→spins；>120→turns；否则VLM spins可补spins，存在来源冲突风险 |
| `focus` / `focus_vlm` | 部位速度除各上分位后取最大；三部位均>0.95则whole body |
| `bpm/tempo` | 有BPM时<85慢/>125快/否则中；缺音乐统计才用VLM tempo_feel |
| `quality/mood/props/key_moves` | VLM直接字段；key_moves在original模式只保存在dims，不拼入原MS正文 |
| `motion_source/seg_events` | original或lite；original一般无lite三段事件列表，不硬填 |

阈值及仲裁是代码实现，不是统一人工真值。当前分位数在合并VLM/MS交集上计算，**不是严格train-only统计**；如果这些数值作为学习目标/归一化或评估分档，需先固定train池及阈值并保存，避免val/test参与数据驱动阈值选择。本文不把现合并逻辑冒称已完成无泄漏修复。

### 10.3 四种文本不要混用

```text
caption             = 舞种/音乐tempo/能量/位移/层次/质感/道具上下文 + motion
caption_structured  = 同样上下文和motion，仅标题/换行不同
motion_original     = 原版MS非空句拼接；lite模式则是自定义三段事件正文
实际dance_only目标  = "Dance style: {genre}. " + motion_original
```

真正用于舞蹈表示监督的正文入口是`text_embed.py:52–57`与`music_text_cache.py:30–36`：舞种＋原版动作，排除描述音乐BPM的段落。**保存所有dims不意味着全部dims已经进入训练**。`text_embed.py`是旧片段级mean-pooled5120向量；`music_text_cache.py`是原生逐token raw H teacher缓存，含正文token/有效长度和来源协议；它们不是同一目标。

neutral teacher只改变代词，记录原prompt与replacement；oracle aswap若缺原MS，会从`E10/val47_motion_captions.csv`用video-observed VLM fallback，并保存`source.kind/path/sha256`。不要把VLM fallback混称相同MS来源的单变量实验。旧oracle prefix/context仍是历史教师产物；最新音乐-only学生不能读取其prefix/context作为输入。

## 11. 时间、置信度与多来源标注的建议保存格式

现有MS的句列表/相对连接词和VLM key_moves**没有统一的精确事件时间戳schema**。不应凭句序生成每秒标签。若要扩展时间维度，建议独立版本保存：

```json
{
  "sample_id": "example_id",
  "source_window_s": [0.0, 5.0625],
  "sources": {"video_sha256": "...", "method": "DWPose_rule_vNext", "script_sha256": "..."},
  "events": [{"start_s": 1.0, "end_s": 1.5, "body_part": "left_wrist",
              "direction": "visible upward", "evidence": "2D reliable points",
              "confidence": "medium", "status": "observed"}],
  "unobservable_parts": ["foot_contact"],
  "review_status": "pending"
}
```

这是**建议schema与虚构示例，不是已存在产物**。事件边界需有实际原生帧索引/可靠连续区间，左右按舞者解剖侧，不按画面左/右；镜像、转背及换人须人工核验。不能用16帧VLM承诺毫秒级时间边界，不用插值跨长缺口造事件。

## 12. 音画联合模型意见：辅助验收，不当人工标签

`$S/music_av_judge.py`使用本地Qwen2.5-Omni-7B Thinker，真实视频和配乐联合输入：源`[1,5)`四秒，视频32帧/8fps、音频mono16kHz/64000samples；局部0秒是原视频1秒。prompt/schema在13–27行，字段`music_description/motion_description/correspondences/synchrony_score/confidence/limitations`。

它要求最多3个事件、score真整数1–5、时间范围0≤t<4、不精确到0.125秒以下；不提供beat标签或原/换音乐身份。输入aud/video token均实际存在并检查。相关命令是**评审意见实验**而非全量训练打标：

```sh
# 校准只选三GT及正确/替代配乐；输出目录必须是新的，运行前需要1GPU和依赖核验。
CUDA_VISIBLE_DEVICES=0 /opt/venv_eval/bin/python "$S/music_av_judge.py" \
  --exp_root "$EXP" --model "$R/models/Qwen2.5-Omni-7B" \
  --calibration --out "$MS/av_opinion_repro" --shard 0 --nshard 1
```

自动同步/节拍指标尚需GT与错配校准；不以模型5分、运动量更大或音画同时存在就认定舞蹈卡点。纯音乐CoT猜编舞不是对当前视频动作的观测标签。

## 13. 人工评审与数据验收

### 13.1 现有页面和记录

当前已核验HTTP200的数据页：在现有8766评审首页点击“MotionScript 数据评审”，路由`data-review`。服务源码：

`$REPO/scripts_omnidance/exp/review_web/server.py:495–567`。

数据页固定加载：

```text
$MS/captions_orig_2k.json
$MS/native_full_lr3e7_b8_e20/split.json
```

这页的train1800/val200是**历史原生实验划分**，不是自动切到当前所有音乐实验split，也不是整个222939精选集。页面显示原视频、原音乐、实际dance_only文本目标和全部dims，可筛舞种/train/val。

人工项：动作正确度1–5；flags=`wrong`动作事实不符、`side`左右/时间错误、`style`舞种待纠正；自由note记录漏掉的部位/具体动作。

追加记录目录：`$EXP/review_web/reviews/<reviewer>.jsonl`，key=`MS/data/<sid>`。当前只读核验目录为空，**不能声称已有网页数值评分**；用户在会话中反馈另行记录于实验文档，不伪造写入网页评分。

### 13.2 最小验收清单

- 媒体可完整解码，ID/audio/image/video对应，源时段与帧率真实记录。
- NLF有效帧比、最长缺口、主体切换、NaN只出现在未观测帧；不能只看文件存在。
- MS JSON合法、source为原版、ms至少一非空句；低观测/无句/err分别计数。
- VLM输出15键与类型/枚举可用，label:null和重复sid处理；原始raw保留。
- 人工按部位、左右、转身方向、动作先后、是否可见核对；标“不确定”，不强选yes/no。
- 跳跃/转圈/足接触需连续视频证据，不能由单帧或低覆盖足点推断。
- 音乐BPM/拍点只与同源配乐相连，起点与音视频偏移真实记录，静音/短片段弃权或单列。
- 每条融合标签保留方法/模型revision/脚本与输入SHA/阈值/原始与修订版本；修订不覆盖原文。
- split按源歌曲/近重复音频及视频源组审计；随机clip划分不等于没有泄漏。
- 同内容格式/代词清洗/新增事件规则分别单变量测试；标签数量、解析率、分数覆盖不是准确率。

## 14. 常见错误及处理边界

| 问题 | 原因/处置 |
|---|---|
| 原版MS多数空句的旧说法 | 两初样本过度外推，已撤回；使用后续全量非空统计，仍保留真正无句/低观测 |
| NLF多人忽左忽右 | 每帧最大框选择不跟踪；先过滤多人或实现独立跟踪版本并审计 |
| 模板手指被当真实手形 | FK手指单位旋转无观测；原MS身体标签与DWPose可见手形分开 |
| MS worker不断重启 | Pool initializer依赖失败；先父进程import，核验matplotlib等，不盲增worker |
| VLM JSON合法但错误 | parser只json.loads；补独立schema/事实review，不把解析率当准确率 |
| failed sid续跑被跳过 | VLM label:null/audio err/质检errorJSON已占done；保留错误记录，用失败清单与新输出重跑 |
| 音频段错误 | librosa/numba与torch/ONNX同进程兼容问题；CPU音频统计独立运行，主audio_feats用NumPy DP |
| 足覆盖不足仍出抬跟/接触 | 未按可见性门控；输出unknown/弃权，不插值造标签 |
| 转圈与原视频冲突 | yaw/地面/左右或文字表达有误差；看连续视频，保存原/修订及理由，不自动加舞步 |
| 结构化dims都被当训练目标 | 当前dance_only仅genre＋motion；记录实际编码正文，额外维度作为候选单列 |
| CSV同sid但时间不同 | NLF全片、DWPose前5.0625、M2名义5.175、AV[1,5)不同；主动绑定source window |
| 共享盘慢读/环境挂起 | 优先已核验的本地venv、同SHA缓存；不以超时/活PID或holder100%当完成 |
| 质检server/client返回401 | 现server wrapper启api-key但client无header；核验鉴权后运行，不混写历史质检成功 |

## 15. 推荐实际工作顺序与资源告知

1. 固定ID与视频/配乐时段，按可靠精选信息预筛；先核验CSV路径错误与重复字段。
2. 先3个不同case核对NLF→原版MS，身体和左右/顺序人工确认后再扩大。
3. VLM独立看同片段标舞种/高层字段；235B需要整节点8GPU，8B/32B小量对照另标模型来源。
4. CPU音频特征与MS语言规则可并行，NLF/DWPose逐GPU/逐ID分片，不改变标签定义来填满GPU。
5. 合并raw维度与正文，保存冲突；只把授权维度送入教师目标。
6. DWPose细粒度规则仍作为新版本候选，不能覆盖原MS正式标签。
7. 人工数据review确认事实，再独立测试生成遵循/音乐响应；当前noTC学生推理始终没有外部文字。

资源应分别告知**最低可运行**与**建议并行吞吐**：原版MS本身CPU即可；NLF/DWPose单worker1GPU可横向分片；本地235B标签默认TP8；独立生成/打标批次可按可用节点并行。明确不需要的节点提前让用户安排其他任务，扩占前再次协调。本文文档工作只需CPU/文件读取，不保留GPU。

关联：[原版/格式/DWPose历史核验](22_h3_bidir_blur_attribution_20261013.md)、[训练与生成实际目标](25_music_h_training_clip_dpo_and_generation_20261019.md)、[最新无TC学生与空间诊断](26_music_h_diagnosis_and_solution_plan_20261020.md)。
