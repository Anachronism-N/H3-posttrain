# Wan2.2-Animate 推理逻辑与形状变化

> 适用代码：`diffsynth/pipelines/wan_video.py`、`diffsynth/models/wan_video_animate_adapter.py`
> 重点对象：`Wan-AI/Wan2.2-Animate-14B`

---

## 1. 总览

Wan Animate 的推理输入主要有：

| 输入 | 作用 |
|------|------|
| `prompt` | 文本语义条件，由 UMT5 编码 |
| `input_image` | 目标人物/首帧身份条件，一路走 CLIP，一路走 VAE |
| `animate_pose_video` | 姿态/骨架条件视频，走 VAE 后加到 DiT patch feature |
| `animate_face_video` | 面部驱动视频，走 motion encoder + face encoder 后通过 adapter attention 注入 DiT |
| `animate_inpaint_video` | 换人/替换模式中的背景或 inpaint 参考视频，可选 |
| `animate_mask_video` | 换人/替换模式中的 mask，可选 |

核心链路：

```text
prompt
  -> UMT5 text_encoder
  -> context

input_image
  -> CLIP image encoder
  -> clip_feature
  -> DiT context 前缀

input_image
  -> VAE
  -> y 条件 latent
  -> 和噪声 latent 在 channel 维拼接

animate_pose_video
  -> VAE
  -> pose_latents
  -> Conv3d 对齐 hidden_dim
  -> 加到 DiT patch feature 的第 1..T latent 帧

animate_face_video
  -> motion_encoder
  -> 每帧 512 维 motion vector
  -> face_encoder
  -> 时间压缩后的 face motion tokens
  -> 每 5 个 transformer block 通过 attention residual 注入

DiT denoise loop
  -> scheduler step
  -> VAE decode
  -> output video
```

---

## 2. 推理入口

入口函数是：

```text
diffsynth/pipelines/wan_video.py
WanVideoPipeline.__call__()
```

Animate 相关参数：

```python
animate_pose_video: Optional[list[Image.Image]] = None
animate_face_video: Optional[list[Image.Image]] = None
animate_inpaint_video: Optional[list[Image.Image]] = None
animate_mask_video: Optional[list[Image.Image]] = None
```

示例：

```text
examples/wanvideo/model_inference/Wan2.2-Animate-14B.py
```

示例中 `num_frames=81`，但 `animate_pose_video` 和 `animate_face_video` 会取 `81 - 4 = 77` 帧：

```python
animate_pose_video = VideoData(...).raw_data()[:81-4]
animate_face_video = VideoData(...).raw_data()[:81-4]
```

原因是 Wan VAE 的时间压缩比例为 4。`81` 帧视频对应：

```text
latent_T = (81 - 1) // 4 + 1 = 21
```

而 `77` 帧条件视频对应：

```text
pose/face latent_T = (77 - 1) // 4 + 1 = 20
```

这 20 个条件 latent 时间步会对齐到主视频 latent 的第 `1..20` 步；第 0 步由 `input_image` / 首帧条件负责。

---

## 3. Pipeline Unit 顺序

`WanVideoPipeline` 初始化时会注册一组 unit。与 Animate 强相关的是：

```text
WanVideoUnit_ImageEmbedderCLIP
WanVideoUnit_ImageEmbedderVAE
WanVideoUnit_AnimateVideoSplit
WanVideoUnit_AnimatePoseLatents
WanVideoUnit_AnimateFacePixelValues
WanVideoUnit_AnimateInpaint
```

其中：

| Unit | 输出 | 作用 |
|------|------|------|
| `ImageEmbedderCLIP` | `clip_feature` | 把 `input_image` 编码成 CLIP 图像 embedding |
| `ImageEmbedderVAE` | `y` | 把 `input_image` 编码成 VAE 条件 latent，并带 mask |
| `AnimateVideoSplit` | 裁剪后的 animate 视频 | 如果有 `input_video`，按其长度裁剪 Animate 条件 |
| `AnimatePoseLatents` | `pose_latents` | 把姿态视频编码成 VAE latent |
| `AnimateFacePixelValues` | `face_pixel_values` | 把面部视频转成像素张量，正负 CFG 分支分别处理 |
| `AnimateInpaint` | `y` | 换人/替换模式下，用 inpaint video + mask 覆盖 VAE 条件 |

---

## 4. 基础形状约定

下面以常见设置为例：

```text
B = 1
num_frames = 81
height = 720
width = 1280
VAE z_dim = 16
VAE spatial downsample = 8
VAE temporal downsample = 4, 公式为 (T - 1) // 4 + 1
DiT hidden_dim = 5120
DiT patch_size = (1, 2, 2)
DiT num_heads = 40
head_dim = 5120 / 40 = 128
```

主视频 latent 形状：

```text
latents:
[B, 16, 21, 90, 160]
```

其中：

```text
21 = (81 - 1) // 4 + 1
90 = 720 // 8
160 = 1280 // 8
```

DiT patch 后空间再除以 2：

```text
patch_H = 90 // 2 = 45
patch_W = 160 // 2 = 80
tokens_per_latent_frame = 45 * 80 = 3600
total_video_tokens = 21 * 3600 = 75600
```

---

## 5. 文本条件：prompt -> context

代码位置：

```text
diffsynth/pipelines/wan_video.py
WanVideoUnit_PromptEmbedder
```

处理：

```text
prompt
  -> tokenizer: google/umt5-xxl/
  -> text_encoder: models_t5_umt5-xxl-enc-bf16.pth
  -> context
```

形状可近似理解为：

```text
context:
[B, text_len, text_dim]
通常 text_len = 512
```

进入 DiT 前会经过：

```python
context = dit.text_embedding(context)
```

变成：

```text
[B, 512, 5120]
```

注意：这里用的是 UMT5，不是 `xlm-roberta-large/`。`xlm-roberta-large/` 是 OpenCLIP 文本侧配套，在 DiffSynth 的 Wan Animate 正常推理里通常不走。

---

## 6. 参考图条件一：input_image -> CLIP -> context 前缀

代码位置：

```text
diffsynth/pipelines/wan_video.py
WanVideoUnit_ImageEmbedderCLIP
```

处理：

```python
image = pipe.preprocess_image(input_image.resize((width, height)))
clip_context = pipe.image_encoder.encode_image([image])
```

`image_encoder` 使用：

```text
models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
```

这里主要使用 CLIP 的图像侧，得到 `clip_feature`。进入主干时：

```python
clip_embdding = dit.img_emb(clip_feature)
context = torch.cat([clip_embdding, context], dim=1)
```

逻辑：

```text
input_image
  -> CLIP image encoder
  -> clip_feature
  -> dit.img_emb
  -> 拼到 text context 前面
```

作用：让 DiT 在 cross-attention 上同时看到文本 prompt 和参考图身份/外观条件。

---

## 7. 参考图条件二：input_image -> VAE -> y

代码位置：

```text
diffsynth/pipelines/wan_video.py
WanVideoUnit_ImageEmbedderVAE
```

处理：

```text
input_image resize 到 [height, width]
  -> 构造一个 81 帧视频：
     第 0 帧是 input_image
     后续帧为 0
  -> VAE encode
  -> 得到 y latent
  -> 拼接 mask
```

形状：

```text
vae_input:
[3, 81, 720, 1280]

VAE encode 后:
[16, 21, 90, 160]

mask:
[4, 21, 90, 160]

y = concat(mask, vae_latent):
[20, 21, 90, 160]

y unsqueeze:
[1, 20, 21, 90, 160]
```

进入主干时：

```python
x = latents
if y is not None and dit.require_vae_embedding:
    x = torch.cat([x, y], dim=1)
```

主 latent 原本：

```text
latents: [1, 16, 21, 90, 160]
```

拼接后：

```text
x: [1, 36, 21, 90, 160]
```

这正好对应 Animate DiT 的 `in_dim=36`：

```text
16 noisy video latent channels
+ 20 image/mask condition channels
= 36 channels
```

---

## 8. 姿态条件：animate_pose_video -> pose_latents -> patch feature 加法

代码位置：

```text
diffsynth/pipelines/wan_video.py
WanVideoUnit_AnimatePoseLatents
```

处理：

```python
animate_pose_video = pipe.preprocess_video(animate_pose_video)
pose_latents = pipe.vae.encode(animate_pose_video, ...)
```

如果 `animate_pose_video` 为 77 帧，且输出分辨率是 720x1280：

```text
animate_pose_video pixel:
[1, 3, 77, 720, 1280]

pose_latents:
[1, 16, 20, 90, 160]
```

进入 Animate Adapter：

```text
diffsynth/models/wan_video_animate_adapter.py
WanAnimateAdapter.after_patch_embedding
```

定义：

```python
self.pose_patch_embedding = torch.nn.Conv3d(
    16, 5120,
    kernel_size=(1, 2, 2),
    stride=(1, 2, 2)
)
```

因此：

```text
pose_latents:
[1, 16, 20, 90, 160]

pose_patch_embedding 后:
[1, 5120, 20, 45, 80]
```

主干中，`x` 先做 DiT patchify：

```python
x = dit.patchify(x, control_camera_latents_input)
```

可近似理解为：

```text
x:
[1, 36, 21, 90, 160]

patchify / patch_embedding:
[1, 5120, 21, 45, 80]
```

然后姿态加到第 `1..20` 个 latent 时间步：

```python
x[:, :, 1:] += pose_latents
```

形状对齐：

```text
x[:, :, 1:]:
[1, 5120, 20, 45, 80]

pose_latents after pose_patch_embedding:
[1, 5120, 20, 45, 80]
```

直觉：

```text
第 0 个 latent 时间步：主要由 input_image / 首帧条件负责
第 1..20 个 latent 时间步：叠加姿态视频提供的动作结构
```

---

## 9. 面部条件：animate_face_video -> motion_encoder -> face_encoder

代码位置：

```text
diffsynth/pipelines/wan_video.py
WanVideoUnit_AnimateFacePixelValues

diffsynth/models/wan_video_animate_adapter.py
WanAnimateAdapter.after_patch_embedding
```

### 9.1 face_pixel_values

`animate_face_video` 先转成像素张量：

```python
face_pixel_values = pipe.preprocess_video(animate_face_video)
```

形状：

```text
face_pixel_values:
[1, 3, 77, 512, 512]  # 推荐理解，面部 crop 通常是 512x512
```

CFG 里有一个细节：

```text
positive 分支：真实 face_pixel_values
negative 分支：torch.zeros_like(face_pixel_values) - 1
```

姿态 latent 是 shared input，正负分支共享；面部视频在正负 CFG 分支不同。

### 9.2 motion_encoder

在 adapter 中，先把时间维展开成逐帧图片：

```python
face_pixel_values = rearrange(face_pixel_values, "b c t h w -> (b t) c h w")
```

形状：

```text
[1, 3, 77, 512, 512]
  -> [77, 3, 512, 512]
```

然后：

```python
motion_vec = self.motion_encoder.get_motion(face_pixel_values)
```

`motion_encoder` 是：

```python
Generator(size=512, style_dim=512, motion_dim=20)
```

内部结构：

```text
face frame
  -> EncoderApp
  -> 512 维 appearance feature
  -> 多层 EqualLinear
  -> 20 维 motion coefficient
  -> Direction 正交方向基
  -> 512 维 motion vector
```

形状变化：

```text
[77, 3, 512, 512]
  -> EncoderApp
[77, 512]
  -> fc
[77, 20]
  -> Direction
[77, 512]
```

这里的 `20` 维可以理解为低维表情/头部运动系数；`Direction` 将它映射回模型使用的 `512` 维 motion space。

### 9.3 face_encoder

逐帧 motion vector 重新组成时间序列：

```python
motion_vec = rearrange(motion_vec, "(b t) c -> b t c", t=T)
```

形状：

```text
[77, 512]
  -> [1, 77, 512]
```

然后进入：

```python
self.face_encoder = FaceEncoder(in_dim=512, hidden_dim=5120, num_heads=4)
```

`FaceEncoder` 做三件事：

1. `CausalConv1d(512 -> 1024 * 4, stride=1)`，沿时间建模，并拆成 4 组 head/token。
2. 两个 `CausalConv1d(1024 -> 1024, stride=2)`，时间维压缩约 4 倍。
3. `Linear(1024 -> 5120)`，投影到 DiT hidden_dim。

形状近似：

```text
[1, 77, 512]
  -> conv1_local + reshape
[4, 77, 1024]
  -> conv2 stride=2
[4, 39, 1024]
  -> conv3 stride=2
[4, 20, 1024]
  -> out_proj
[4, 20, 5120]
  -> reshape
[1, 20, 4, 5120]
  -> 在 token 维加 padding token
[1, 20, 5, 5120]
```

随后 `after_patch_embedding` 又在时间维前面补一个空时间步：

```python
pad_face = torch.zeros(B, 1, H, C)
motion_vec = torch.cat([pad_face, motion_vec], dim=1)
```

最终：

```text
motion_vec:
[1, 21, 5, 5120]
```

这与主视频 latent 时间步 `21` 对齐。

---

## 10. 面部条件如何注入 DiT

代码位置：

```text
diffsynth/pipelines/wan_video.py
model_fn_wan_video
```

主干 patch feature 先 flatten 成 token：

```python
x = rearrange(x, 'b c f h w -> b (f h w) c')
```

形状：

```text
[1, 5120, 21, 45, 80]
  -> [1, 75600, 5120]
```

进入 transformer blocks：

```python
for block_id, block in enumerate(dit.blocks):
    x = block(x, context, t_mod, freqs)

    if pose_latents is not None and face_pixel_values is not None:
        x = animate_adapter.after_transformer_block(block_id, x, motion_vec)
```

`after_transformer_block` 每 5 层执行一次：

```python
if block_idx % 5 == 0:
    residual_out = self.face_adapter.fuser_blocks[block_idx // 5](x, motion_vec, ...)
    x = residual_out + x
```

对于 40 层 DiT：

```text
block 0, 5, 10, 15, 20, 25, 30, 35 注入
共 8 次
```

`FaceBlock` 是 cross-attention 风格：

```text
Query: 主视频 token x
Key/Value: 面部 motion_vec
输出: residual_out
最后: x = x + residual_out
```

具体形状：

```text
x:
[1, 75600, 5120]

motion_vec:
[1, 21, 5, 5120]
```

`FaceBlock` 会按时间维拆开：

```text
每个 latent 时间步：
  Query 数量 = 45 * 80 = 3600 个视频 token
  Key/Value 数量 = 5 个 face motion token
```

attention 中可理解为：

```text
q:
[21, 40, 3600, 128]

k/v:
[21, 40, 5, 128]

attn output:
[1, 75600, 5120]
```

因此面部信息的注入方式不是“拼 channel”，而是：

```text
面部 motion tokens 作为 K/V
视频 token 作为 Q
cross-attention 输出 residual
加回 DiT hidden state
```

---

## 11. Denoise Loop

每个 timestep 的主流程：

```text
1. model_fn_wan_video(...)
   -> 得到 noise_pred_posi

2. 如果 cfg_scale != 1:
   -> 再跑 negative 分支或拆 cfg_merge
   -> noise_pred = neg + cfg_scale * (pos - neg)

3. scheduler.step(...)
   -> 更新 latents
```

对应代码：

```text
diffsynth/pipelines/wan_video.py
WanVideoPipeline.__call__()
```

Animate 相关输入在 denoise loop 前已经准备好：

```text
latents
context
clip_feature
y
pose_latents
face_pixel_values
```

denoise loop 中每次都会重新通过 DiT 使用这些条件预测噪声。

---

## 12. Decode

denoise 完成后，如果传了 Animate 条件，会移除前面的参考 latent：

```python
if vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
    inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]
```

然后 VAE decode：

```python
video = self.vae.decode(inputs_shared["latents"], ...)
```

输出为视频帧。

---

## 13. 只有面部图片，没有面部视频怎么办

当前 DiffSynth 实现有一个重要条件：

```python
if pose_latents is not None and face_pixel_values is not None:
    x, motion_vec = animate_adapter.after_patch_embedding(...)
```

也就是说，Animate adapter 需要 `pose_latents` 和 `face_pixel_values` 同时存在才启用。只有 `animate_pose_video` 而没有 `animate_face_video` 时，当前代码不会进入这条 Animate adapter 注入路径。

如果只有单张面部图片，最直接的做法是把它重复成静态 face video：

```python
from PIL import Image

num_frames = 81
face = Image.open("face_crop.png").convert("RGB").resize((512, 512))
animate_face_video = [face.copy() for _ in range(num_frames - 4)]
```

效果含义：

```text
重复脸图
  -> 每帧 motion_encoder 输出接近相同的 motion vector
  -> face_encoder 得到静态/弱变化的 face motion tokens
  -> Animate adapter 可以启用
  -> 姿态条件也可以正常进入主干
```

局限：

```text
单张脸图没有时间运动信息
无法提供自然眨眼、张嘴、转头、口型节奏
模型只能从 prompt、姿态和扩散先验中猜测细节
```

更好的方案：

| 条件 | 建议 |
|------|------|
| 有驱动脸视频 | 用驱动脸视频预处理得到 `animate_face_video` |
| 只有脸图 | 重复脸图作为静态 `animate_face_video` |
| 需要自然表情/口型 | 先用其他模型生成脸部 motion/video，再作为 `animate_face_video` |

---

## 14. 一页形状速查

以 `num_frames=81, height=720, width=1280` 为例：

| 阶段 | 形状 |
|------|------|
| noise / latents | `[1, 16, 21, 90, 160]` |
| input image VAE 条件 `y` | `[1, 20, 21, 90, 160]` |
| `cat([latents, y], dim=1)` | `[1, 36, 21, 90, 160]` |
| DiT patchify | `[1, 5120, 21, 45, 80]` |
| pose video pixel | `[1, 3, 77, 720, 1280]` |
| pose VAE latent | `[1, 16, 20, 90, 160]` |
| pose patch embedding | `[1, 5120, 20, 45, 80]` |
| pose 加入位置 | `x[:, :, 1:] += pose_embedding` |
| face video pixel | `[1, 3, 77, 512, 512]` |
| face 展平成帧 | `[77, 3, 512, 512]` |
| motion_encoder 输出 | `[77, 512]` |
| 恢复时间序列 | `[1, 77, 512]` |
| face_encoder 输出 | `[1, 20, 5, 5120]` |
| 时间维 pad 后 motion_vec | `[1, 21, 5, 5120]` |
| flatten video tokens | `[1, 75600, 5120]` |
| 每个时间步视频 token | `45 * 80 = 3600` |
| face adapter K/V token | `5` 个 / latent 时间步 |
| DiT 输出 noise_pred | `[1, 16, 21, 90, 160]` |
| VAE decode | `81` 帧视频 |

---

## 15. 关键代码位置

| 逻辑 | 文件 |
|------|------|
| 推理入口 | `diffsynth/pipelines/wan_video.py::WanVideoPipeline.__call__` |
| 文本编码 | `diffsynth/pipelines/wan_video.py::WanVideoUnit_PromptEmbedder` |
| CLIP 图像编码 | `diffsynth/pipelines/wan_video.py::WanVideoUnit_ImageEmbedderCLIP` |
| input image VAE 条件 | `diffsynth/pipelines/wan_video.py::WanVideoUnit_ImageEmbedderVAE` |
| 姿态 VAE latent | `diffsynth/pipelines/wan_video.py::WanVideoUnit_AnimatePoseLatents` |
| 面部像素条件 | `diffsynth/pipelines/wan_video.py::WanVideoUnit_AnimateFacePixelValues` |
| inpaint/mask 条件 | `diffsynth/pipelines/wan_video.py::WanVideoUnit_AnimateInpaint` |
| DiT 主干函数 | `diffsynth/pipelines/wan_video.py::model_fn_wan_video` |
| Animate Adapter | `diffsynth/models/wan_video_animate_adapter.py::WanAnimateAdapter` |
| motion encoder | `diffsynth/models/wan_video_animate_adapter.py::Generator` |
| face encoder | `diffsynth/models/wan_video_animate_adapter.py::FaceEncoder` |
| face adapter attention | `diffsynth/models/wan_video_animate_adapter.py::FaceBlock` |

