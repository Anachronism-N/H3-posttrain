# Pose Align / Retarget Plan

本文记录如何基于已有的 3D SMPL pose pkl 和参考人物图片，生成与参考人物尺寸、位置更匹配的 aligned pose video。目标输出是类似 SCAIL `rendered_aligned.mp4` 的骨骼条件视频，用于后续 Wan Animate / pose-only 推理。

## 1. 已确认的输入格式

示例文件：

```text
eval/pkl/211.pkl
```

读取结果：

```text
dict keys:
  smpl_poses: (3261, 72) float32
  smpl_trans: (3261, 3) float32
  full_pose:  (3261, 24, 3) float32
```

含义判断：

- `smpl_poses` 是每帧 24 个 SMPL joints 的 axis-angle pose 参数，`24 * 3 = 72`。
- `smpl_trans` 是每帧的全局平移。
- `full_pose` 是每帧 24 个 SMPL joints 的 3D 坐标，最适合直接用于投影和渲染。
- `full_pose` 的骨长在全序列中稳定，说明它是完整 3D skeleton 序列。
- `full_pose[..., 2]` 更像人体竖直高度，而不是相机深度；不能直接套 `u = f * X / Z`，需要先转换成 camera coordinates。

## 2. 与 SCAIL Align 的关系

SCAIL 的 `--use_align` 思路是：

```text
driving video -> NLF 3D pose
reference image -> DWPose 2D keypoints
first driving 3D pose + reference 2D keypoints -> solve new camera intrinsics
new camera intrinsics -> render whole 3D pose sequence
```

对应代码位置：

```text
SCAIL/scail_pose/NLFPoseExtract/process_pose.py
SCAIL/scail_pose/NLFPoseExtract/align3d.py
SCAIL/scail_pose/NLFPoseExtract/nlf_render.py
```

我们的 pkl 已经有 `full_pose` 3D skeleton，所以可以复用同类思想：用参考图的 2D DWPose 点去约束 `full_pose` 的投影相机，而不是直接缩放 RGB pose video。

## 3. 总体方案

推荐流程：

```text
eval/pkl/211.pkl full_pose
  -> SMPL 24 joints to COCO/DWPose-like body joints
  -> convert world/body coordinates to camera coordinates
  -> reference image DWPose 2D keypoints
  -> optimize camera/depth/scale/shift parameters
  -> project all frames with one fixed parameter set
  -> render aligned skeleton video
```

输出建议：

```text
eval/pose/211_aligned.mp4
```

后续推理传入：

```bash
python examples/wanvideo/model_inference/Wan2.2-Animate-14B_pose_only.py \
  --pose_video eval/pose/211_aligned.mp4 \
  --input_image <reference_image>
```

## 4. SMPL 24 Joints 到 COCO/DWPose 点的映射

可参考 SCAIL 的 `process_data_to_COCO_format`。推荐映射：

```text
SMPL 15 -> 0  head / nose-like
SMPL 12 -> 1  neck
SMPL 17 -> 2  left shoulder
SMPL 19 -> 3  left elbow
SMPL 21 -> 4  left wrist / hand
SMPL 16 -> 5  right shoulder
SMPL 18 -> 6  right elbow
SMPL 20 -> 7  right wrist / hand
SMPL 2  -> 8  left hip
SMPL 5  -> 9  left knee
SMPL 8  -> 10 left ankle / foot
SMPL 1  -> 11 right hip
SMPL 4  -> 12 right knee
SMPL 7  -> 13 right ankle / foot
```

注意：

- SCAIL 返回的是 `(18, dim)`，但实际主干 align 只需要前 14 个 body points。
- 手腕不建议参与主 align，只用于后续可视化或手部跟随。

## 5. 坐标转换

`full_pose` 的坐标轴需要先转成相机坐标。根据当前文件的数值分布，建议初始转换：

```python
root = full_pose[align_frame, 0]

X = full_pose[:, :, 0] - root[0]
Y = -(full_pose[:, :, 2] - root[2])
Z = full_pose[:, :, 1] - root[1] + depth
```

解释：

- `X` 使用原始第 0 维，作为画面水平轴。
- `Y` 使用原始第 2 维取负，保证头部在画面上方、脚在画面下方。
- `Z` 使用原始第 1 维加一个正的 `depth` offset，作为相机深度。
- `root` 建议使用 align 帧的 pelvis/root，而不是每帧 root。如果每帧重新 root-center，会消掉原始全局位移。

初始 `depth` 范围：

```text
depth in [1.0, 4.0]
```

快速观察中，`depth=1.5~3.0` 的投影 bbox 比较合理。

## 6. 参考图 2D Keypoints

对参考人物图片跑 DWPose，得到 body keypoints：

```text
ref_body: shape roughly (18, 2 or 3)
```

统一到目标画布像素坐标：

```python
ref_xy[:, 0] *= width
ref_xy[:, 1] *= height
```

参考图和输出 pose video 必须使用同一目标分辨率，并且采用同样的 resize/crop 规则。建议与 SCAIL 一致：

```text
keep aspect ratio resize -> center crop
```

不要直接拉伸参考图或 pose video。

## 7. 选择 Align 点

优先不要使用手部点，因为手部动作和参考图差异通常很大，容易带偏整体尺度。

推荐两套策略：

```text
feet-align:
  neck + hips + knees + ankles

body-align:
  neck + shoulders + hips
```

选择逻辑：

```text
如果参考图下半身关键点完整，使用 feet-align。
如果脚踝/膝盖缺失，使用 body-align。
```

对照 COCO-like index：

```text
neck: 1
shoulders: 2, 5
hips: 8, 11
knees: 9, 12
ankles: 10, 13
```

## 8. 相机参数求解

优化参数：

```text
depth: 相机深度偏移
m:     全局尺度，相当于焦距缩放
s:     y 方向相对尺度
p:     x 方向主点偏移
q:     y 方向主点偏移
```

投影公式：

```python
f = max(height, width) / (2 * tan(fov / 2))

u = f * m * X / Z + width / 2 + p
v = f * m * s * Y / Z + height / 2 + q
```

优化目标：

```text
min sum_i weight_i * || projected_joint_i - ref_joint_i ||^2
```

建议 bounds：

```text
depth in [1.0, 4.0]
m     in [0.5, 2.0]
s     in [0.85, 1.15]
p     in [-width, width]
q     in [-height, height]
```

如果希望先做稳定版本，可以固定 `depth=2.0`，只优化：

```text
m, s, p, q
```

这更接近 SCAIL 的 `solve_new_camera_params_central/down`。

## 9. Align Frame 的选择

简单版本：

```text
align_frame = 0
```

更稳版本：

```text
从前 30 帧或前 1 秒中选择一个姿态完整、bbox 正常、身体尽量直立的帧。
```

避免使用：

- 人物蹲下、跳跃、倒地的帧。
- 人物侧身极端遮挡的帧。
- 关键点明显异常的帧。

如果参考图是站姿，而 align 帧是蹲姿，点对点优化会把尺度估计带偏。

## 10. 整段应用同一组参数

求解出的参数必须对整段视频固定使用：

```python
params = solve_once(full_pose[align_frame], ref_keypoints)

for frame in frames:
    camera_xyz = convert_to_camera(frame, root=full_pose[align_frame, 0], depth=params.depth)
    projected = project(camera_xyz, params)
    render(projected)
```

不要逐帧求解参数，否则 pose video 会产生尺度和位置抖动。

## 11. 渲染输出

输出方式有两种：

```text
方案 A: 只渲染 body skeleton。
方案 B: body 使用 3D 投影，face/hands 使用 DWPose 2D 并按头/腕关节跟随。
```

当前 `211.pkl` 只有 SMPL body skeleton，没有 face/hands。因此第一版建议直接渲染 3D body skeleton。

如果后续同时有 driving video 的 DWPose face/hands：

- face 按 head/neck 投影偏移做整体平移。
- hands 按 wrist 投影偏移做整体平移。
- hands 可围绕 wrist 按 `m` 或 `sqrt(m * m * s)` 做局部缩放。
- face 可参考 SCAIL `scale_faces`，用参考图脸部关键点平均半径与 driving 第一帧脸部平均半径求比例，并 clamp 到 `[0.8, 1.5]`。

## 12. 质量检查

生成 `211_aligned.mp4` 后检查：

- 人物头顶、脚底是否与参考人物大致同尺度。
- pelvis/neck 是否落在参考人物身体中心附近。
- 人物是否上下颠倒。如果颠倒，调整 `Y = +(full_pose[..., 2] - root[2])`。
- 人物是否过大/过小。如果过大/过小，收紧或调整 `m/depth` bounds。
- 是否保留了期望的全局平移。如果人物被固定在中心，检查是否误用了每帧 root-center。
- 投影是否有穿帮点飞出画面。如果有，检查 `Z` 是否接近 0，必要时增大 `depth` 下界。

## 13. 最小实现伪代码

```python
import pickle
import numpy as np
from scipy.optimize import minimize

with open("eval/pkl/211.pkl", "rb") as f:
    data = pickle.load(f)

full_pose = data["full_pose"]  # T, 24, 3

height, width = 1280, 704
fov = np.deg2rad(55)
f = max(height, width) / (2 * np.tan(fov / 2))

align_frame = 0
root = full_pose[align_frame, 0].copy()

def smpl24_to_coco18(joints):
    mapping = {
        15: 0,
        12: 1,
        17: 2,
        19: 3,
        21: 4,
        16: 5,
        18: 6,
        20: 7,
        2: 8,
        5: 9,
        8: 10,
        1: 11,
        4: 12,
        7: 13,
    }
    out = np.zeros((18, joints.shape[-1]), dtype=joints.dtype)
    for src, dst in mapping.items():
        out[dst] = joints[src]
    return out

def to_camera(joints, depth):
    x = joints[:, 0] - root[0]
    y = -(joints[:, 2] - root[2])
    z = joints[:, 1] - root[1] + depth
    return np.stack([x, y, z], axis=-1)

def project(cam, m, s, p, q):
    x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
    u = f * m * x / z + width / 2 + p
    v = f * m * s * y / z + height / 2 + q
    return np.stack([u, v], axis=-1)

# ref_xy 需要来自参考图 DWPose，并已转成像素坐标
# ref_valid 是有效关键点 mask
ref_xy = ...
ref_valid = ...

if ref_valid[[10, 13]].all():
    align_indices = [1, 8, 11, 9, 12, 10, 13]
else:
    align_indices = [1, 2, 5, 8, 11]

align_indices = [i for i in align_indices if ref_valid[i]]

src_coco = smpl24_to_coco18(full_pose[align_frame])
src_points = src_coco[align_indices]
target_points = ref_xy[align_indices]

def objective(params):
    depth, m, s, p, q = params
    cam = to_camera(src_points, depth)
    pred = project(cam, m, s, p, q)
    err = target_points - pred
    return np.sum(err ** 2)

result = minimize(
    objective,
    x0=[2.0, 1.0, 1.0, 0.0, 0.0],
    bounds=[
        (1.0, 4.0),
        (0.5, 2.0),
        (0.85, 1.15),
        (-width, width),
        (-height, height),
    ],
)

depth, m, s, p, q = result.x

projected_all = []
for frame in full_pose:
    coco = smpl24_to_coco18(frame)
    cam = to_camera(coco, depth)
    xy = project(cam, m, s, p, q)
    projected_all.append(xy)
projected_all = np.stack(projected_all)

# TODO: render projected_all to eval/pose/211_aligned.mp4
```

## 14. 推荐落地顺序

1. 先写一个 preview 脚本，把 `full_pose` 用固定 `depth=2.0, m=1, s=1, p=0, q=0` 投影并渲染，确认坐标轴方向正确。
2. 对参考图跑 DWPose，保存 reference body keypoints。
3. 实现相机参数优化，打印 `depth, m, s, p, q`。
4. 渲染 aligned video，并和参考图叠图检查头、肩、胯、脚位置。
5. 将 aligned video 送入 pose-only 推理。

