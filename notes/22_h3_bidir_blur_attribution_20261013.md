# 22. H3 双向舞蹈微调"脸/手/肢体模糊"：方案总结、10-13 人评与归因对照（2026-10-13）

## 1. 现象
原始 H3 Ref2VA（E3 官方 pipeline + 富文本 VLM prompt）被人工认可可用；所有在我们舞蹈数据上做的双向 LoRA 微调，
都出现脸、手、部分肢体模糊，快动作拖影，且通常训练越久越糊。音频替换（语音/静音）时模型仍然跳舞且仍糊。

## 2. 已尝试方案（按大类，详见 notes/19、20、21）
| 大类 | 做法 | 结果（人评为准） |
|---|---|---|
| 推理设置 | shift 5→12/3、20→50 步、prompt 风格 | 采样器无问题；**清单 caption → 富文本 prompt** 是一大因素（原始 H3 + 富文本 ≈ E3） |
| CFG 蒸馏保护 | H-A 普通 FM → 真 CFG3 可救回；H-A2 DiffSynth 去蒸馏适配器；H-A3 inverse guidance g=4 | 有改善但未通过；两者都用清单 caption 训练，后被否决 |
| 富文本训练 | H-A4 = H-A2 配方 + 富文本侧车（full12k，750 步） | 身份/服饰保持、风格接近 E3，但手脸仍糊 |
| 数据 | H-A5hr 704×1280 原生分辨率；H-A6dup 不插帧（最近帧重复） | 768×1344 推理时手明显清楚；HR/dup 训练结果待人评（dup 1499 已训完，HR 798/1499） |
| AR | H-B2base/B3/B4m（TF/DF 混训、guidance loss、VAE 段对齐分块）、H-C 一致性蒸馏、ODE 轨迹 | TF 目标全部涂抹，停；双向未通过前不再做 AR/ODE |
| 音乐条件 | 节拍写入 prompt、MERT 注入、5 专家 MoE、音乐对比 DPO；富文本版 M1abR/M2bR | 旧清单版结论作废；富文本版 708/886 已出 3case，待人评 |
| 社区方案 | fal Realism LoRA（单用/叠加）、akatz 配方（ostris 助手 + guidance 3.5）、Hybrid BF16（b25–49 AdaLN 合并） | fal 叠加更差排除；后两者纳入 10-13 对照 |
| 同数据 CFG 对照 | 2091 精选、seed42、globalbatch8、300 更新：Hybrid-G3.5/FM、OST1-G3.5/FM、REF-G3.5/FM/inverse3.5、DS-FM/G3.5 | 见 §3 |

## 3. 10-13 人评（同 3 case，富文本，joint + aswap）
- HYB25base（未训练 Hybrid）手部已有些不清晰；HYB25G100/300 比训练前更差，人体整体模糊（最差）。
- OS1G100/300（ostris 助手 + direct3.5）脸/手/部分肢体模糊，但好于 HYB25G。
- REF35（原 Ref2VA + direct3.5，无助手）类似，且 300 比 100 更糊。
- OS1FM（ostris 助手 + 普通 FM）与 OS1G 差不多。

**结论**：助手有/无、FM / direct / inverse guidance、原底座 / Hybrid 全部同样退化且随步数加重 →
**模糊的共性原因不在 CFG 训练方式**。不再新增 CFG/助手/Hybrid 变体；HYB25FM、DSG35、REFINV35 在 216/176/174 步停止
（目录内 `STOPPED_user_cfg_not_key_20261013`，已有 100/200 权重保留）。DSFM8、REFFM 已正常训完 300 作为普通 FM 参照。

## 4. 剩余共性嫌疑与诊断
所有配方共享：训练目标视频（480p 裁剪 + 2/3 RIFE 插帧帧 + 视频 VAE 编码）、LoRA r64 / lr 5e-5、clean 音乐条件、富文本 prompt。
1. **训练目标本身已降级**（最可能解释"越训越像它"）。诊断：`h3ar/diag_decode_train_target.py` 把 3 条快动作训练 latent
   用 VAE 解码 → `logs_omnidance/exp/H3AR/diag_train_target_decode/*_target.mp4`，与同尺寸源片 `*_src.mp4` 对比（待人评）。
2. **微调过程本身**（lr/步数/LoRA 漂移）：与数据无关的退化。
3. 已发现的小不一致：评测音乐 5.06 s < 视频 5.17 s，采样时音频尾部 4 个 latent mask=1 但从不更新（训练时全 clean）。
   原始 H3 同样受影响，不能解释"微调后才糊"；为保持本批可比性暂不修。

## 5. 归因对照：H3 自生成数据（SG512）vs 真实数据（GT512）
- 目的：区分 §4 的 1 与 2。用**原始 H3 Ref2VA 自己生成的视频**作训练目标，其余完全相同。
  若 SG 训练后仍随步数变糊 → 微调过程本身的问题；若 SG 保持清晰而 GT 变糊 → 训练数据降级是根因。
- 数据：2091 精选中每隔 4 取 1 共 512 个 ID（`H3AR/SG512/sg512.txt`，五节点缓存逐字节一致）。
- 生成：`h3ar/sg_gen.py` 在**完全相同的缓存条件**（富文本侧车按 loader 同方式拼接、clean 音乐 mask0、参考锚点）上，
  原始 Ref2VA、无 LoRA、cfg1、官方 shift 12/3、50 步，从噪声采样视频 latent（每 ID 固定种子）。单卡约 11 min/条；
  N1/N3/N4/N7/N8 共 40 卡并行（`h3ar/sg_node.sh`，cephfs mkdir 认领），输出 `H3AR/SG512/lat/`。
  冒烟对照解码：`H3AR/SG512/check_0_1_{sg,gt}.mp4`。
- 训练：`h3ar/sg_build_cache.py` 生成节点本地 `/opt/h3cache/sg512`（input_latents 替换为 SG）与 `gt512`（原样），
  各自带 posi/nega 侧车；两组均 = REFFM 配方（原 Ref2VA、普通 FM、富文本、r64、lr5e-5、seed42、globalbatch8），
  `CFG_REPEAT=5`、300 更新、存 100/200/300。评测同 3 case；SG 组另需对比"SG 数据本身"与其训练结果。
- 注意：512×5 / 8 ≈ 4.7 epoch，比 2091 集 300 步（约 1.15 epoch）重复更多，所以 GT512 必须同时训练作对照，不拿 REFFM 直接比。

## 6. 进展（2026-10-13 晚）
- SG512 生成 512/512 完成（40 卡约 2.5 h，无报错）。SG512 训练 N4 17:40 起、GT512 训练 N6 18:05 起；两者 training_args 仅数据路径/输出不同
  （seed42、lr5e-5、r64、repeat5、max300、save100、cfg1）。SG512 step-100 已存、416 张量有限。
  训练 loss：SG 明显更低（前 100 步均值 0.07 vs GT 0.12）——SG 目标就是底座自己的输出，更易拟合，不代表画质。
- 空闲节点追加 4 组 GT512 单变量对照（同 512 ID、同 seed/步数，只改一项），19:23 启动：
  | 名称 | 节点 | 改动 | 检验 |
  |---|---|---|---|
  | GT512LR | N3 | lr 1e-5 | 漂移速度 |
  | GT512HI | N7 | 只训高噪声半段（时间步索引 [0,0.5)，sigma 1→0.69；`OD_H3_TB_MAX=0.5`） | 只学结构/运动、细节交给底座能否不糊 |
  | GT512LO | N8 | 只训低噪声半段（sigma 0.69→0；`OD_H3_TB_MIN=0.5`） | 细节变糊是否来自低噪声段拟合降级数据 |
  | GT512R16 | N1 | LoRA rank 16 | 容量 |
  `loss.py` 新增 `OD_H3_TB_MIN/MAX`（默认不变）；`q2_20261010.sh` 支持 `CFG_REPEAT/CFG_LR/CFG_RANK` 覆盖。
- 已训完：DSFM8 300、REFFM 300、H-A6dup 1499（正常退出、权重全有限）。H-A5hr 约 800/1499 继续。
- 网页置顶预设【10-13 归因】含 E3、原始 H3、GT512/SG512 及上述四组的 100/300；评测在 N2 队列中自动生成。
