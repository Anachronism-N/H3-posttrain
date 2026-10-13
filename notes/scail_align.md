对齐模式（--use_align）详细流程
整个对齐流程解决的核心问题是：驱动视频里的人和参考图里的角色体型/身材比例不同，如何让提取的姿态"适配"参考角色的身材？

整体流程图

驱动视频 driving.mp4          参考图 ref_image.jpg
        │                              │
        ▼                              ▼
  DWPose 提取 2D 姿态           DWPose 提取 2D 姿态
  NLF 提取 3D 关键点            NLF 提取 3D 关键点
        │                              │
        └──────────────┬───────────────┘
                       ▼
              【对齐模式核心】
         判断是否能看到脚（下半身）
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   看到脚（≥4个下肢点）        看不到脚（上半身）
   solve_new_camera_params_down  solve_new_camera_params_central
          └────────────┬────────────┘
                       ▼
            求解新相机内参 K_new
            (m, s, p, q 四个参数)
                       ▼
          shift_dwpose_according_to_nlf
          （用新内参重新投影所有3D点到2D）
                       ▼
            scale_faces (脸部缩放)
                       ▼
              render_nlf_as_images
              输出: rendered_aligned.mp4
第一阶段：双路姿态提取

# 驱动视频：DWPose → 2D关键点
poses, scores, det_results = zip(*detector_return_list)

# 驱动视频：NLF → 3D关键点（米制单位，真实3D坐标）
nlf_results = process_video_nlf(model_nlf, vr_frames, det_results)

# 参考图：DWPose → 2D关键点
poses_ref, scores_ref, det_results_ref = ...

# 参考图：NLF → 3D关键点
nlf_results_ref = process_video_nlf(model_nlf, vr_frames_ref, ...)
此时得到：

poses_ref：参考图中人物在图像坐标系下的 2D 关节位置（归一化到 [0,1]）
nlf_results：驱动视频第一帧的 3D 关节坐标（世界坐标，单位：米）
第二阶段：判断对齐策略（上半身 vs 全身）

# 有效的下肢关键点（脚踝、膝盖等）
lower_body_indices = [9, 10, 12, 13]

# 统计参考图中可见的下肢关键点数量
for i in range(len(poses_2d_subset)):
    if poses_2d_subset[i] != -1.0 and ...:   # 点可见且3D点有效
        if i in lower_body_indices:
            valid_lower_indices.append(i)

# 策略选择
if len(valid_lower_indices) >= 4:
    # 全身可见：以脚为锚点对齐（更稳定）
    solve_new_camera_params_down(...)
else:
    # 上半身：以躯干中心对齐
    solve_new_camera_params_central(...)
为什么要区分？ 脚部在图像中的位置是"身高"信息的最强锚点，如果能看到脚，用脚对齐最稳定；否则只能用上半身的比例推断。

第三阶段：求解新相机内参（核心数学）
这是整个对齐的数学核心，位于 align3d.py。

问题建模：

驾驶视频里的 3D 关节点 (X, Y, Z) 通过相机内参 K 投影到图像：


u = f * X/Z + cx
v = f * Y/Z + cy
原始内参是根据图像尺寸估算的：


# 标准相机内参（假设 FOV 约 53°）
K_orig = [[f, 0, W/2],
           [0, f, H/2],
           [0, 0,   1]]
目标： 找到新相机内参 K_new，使驱动视频第一帧的 3D 点投影到 2D 后，与参考图中对应关节的 2D 位置尽量吻合。

新内参的参数化方式：


K_new = [[f * m,     0,    W/2 + p],
          [0,     f * m * s, H/2 + q],
          [0,         0,      1    ]]
参数	含义	约束范围
m	整体缩放（对应身材高矮）	[0.7, 1.4]
s	纵横比缩放（体型胖瘦）	[0.8, 1.15]
p	水平平移（左右位移）	[-W, W]
q	垂直平移（上下位移）	[-H, H]
优化目标（两种策略的差异在权重）：


# central（上半身模式）：躯干中心点权重 ×8，其余 ×1
error = error0 * 8 + error_rest * 1

# down（全身模式）：脚部点权重 ×4，躯干中心点 ×1
error = error0 * 1 + error_feet * 4
用 scipy.optimize.minimize 求解最优 (m, s, p, q)。

第四阶段：用新内参重新投影所有帧

shift_dwpose_according_to_nlf(
    collect_smpl_poses(nlf_results),  # 所有帧的3D关节点
    poses,                             # 所有帧的DWPose 2D关节点
    ori_camera_pose,                   # 原始内参
    new_camera_intrinsics,             # 求解出的新内参
    target_H, target_W,
    scale_x=scale_m,
    scale_y=scale_m * scale_s         # 注意：y方向乘以纵横比
)
这一步将每一帧的 3D 关节点用新相机内参重新投影，替换掉原来的 2D 关节坐标。效果上等同于：把驱动视频里演员的姿态"套"在参考图角色的体型上。

第五阶段：脸部缩放补偿

scale_face = scale_faces(list(poses), list(poses_ref))
3D 对齐后，身体比例变了但 DWPose 的脸部 68 点是 2D 检测结果，不受 3D 投影影响。scale_faces 单独计算：


scale_n = mean_dist(ref脸部点到鼻尖) / mean_dist(驱动脸部点到鼻尖)
scale_n = clip(scale_n, 0.8, 1.5)   # 防止极端值
然后对脸部 68 个关键点和身体关键点同时应用这个缩放，保证脸部大小和身体比例一致。


  用新相机内参重新投影 →  高个子的3D骨架投影出来的2D看起来和矮个子一样大
★ Insight ─────────────────────────────────────

1. 这是"虚拟相机调整"而非"关节坐标缩放"：传统做法是直接缩放 2D 关键点坐标，但这会破坏透视关系。SCAIL 的做法是保持 3D 点不变，通过调整"虚拟相机的焦距和主点位置"来改变投影结果——这在透视正确性上更优，不会产生肢体比例失真。

2. m 和 s 的物理意义：m 是"从驱动演员视角到参考角色视角需要拉近/推远多少"（改变感知身高）；s 是纵横比补偿（解决驱动演员和参考角色躯干宽高比不同的问题）。

3. 分策略加权的设计思想：全身模式以脚为高权重锚点，因为脚在图像底部，决定了感知身高；上半身模式以躯干中心为高权重锚点，保证核心姿态的位置准确。这种非均等加权比普通最小二乘更鲁棒。