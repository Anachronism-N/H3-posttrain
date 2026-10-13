# 21. H3 后训练画质下降：CFG 蒸馏相关调研（2026-10-10）

## 1. 问题根源
H3 发布的 checkpoint 是 **CFG 蒸馏**的（采样 cfg_scale=1 即带引导效果）。直接用标准 flow-matching 损失做 LoRA 微调，
会把模型拉回"无 CFG 的条件速度"，表现为细节糊、动作幅度小、对比度/饱和度下降。这与 FLUX.1-dev（guidance 蒸馏）、
FLUX.1-schnell（步数蒸馏）社区微调遇到的是同一类问题。

## 2. 现成方案
| 方案 | 来源 | 做法 | 代价 / 注意 |
|---|---|---|---|
| **A. DeCFG 训练适配器** | ModelScope `DiffSynth-Studio/MiniMax-H3-TrainingAdapter`（本地已有 `models/H3-TrainingAdapter/model_ref2va.safetensors`） | 训练时把一个 LoRA 融进 DiT，把蒸馏模型临时拉回"蒸馏前"行为；新 LoRA 在这个底座上训；**推理时丢掉适配器，只加载新 LoRA**。思路来自 DiffSynth 差分 LoRA 和 ostris ai-toolkit 的 training adapter（FLUX.1-schnell-training-adapter） | 零额外前向。参数 `--preset_lora_path ... --preset_lora_model dit`（`training_module.py:363` 用 `pipe.load_lora` 直接融进底座权重，在 DeepSpeed 初始化之前）。我们之前因与 ZeRO-3 冲突关掉——可改为**离线把适配器融进一份底座权重**再训，彻底绕开冲突 |
| **B. 逆 CFG 训练** `--training_cfg_scale s` | DiffSynth 上游 `loss.py` / 文档；FL2VA 示例建议 s=4 | 额外一路无梯度的无条件前向，把模型输出按 `v_raw=(v_cond+(s-1)v_uncond)/s` 还原成原始速度再算 FM 损失，等于让 LoRA 学"蒸馏后空间"里的目标 | 每步多 1 次前向（约 +25–30% 时间）；**缓存阶段必须用同一 s**（s=1 时缓存会删掉无条件 prompt_embeds/packed）→ 我们现有 20k 缓存需要补无条件输入（负 prompt 是常量 " "，可只编码一次后在线拼 packed） |
| C. 降学习率 / 小步数 | IAmIronMan42/MiniMax-H3-FineTuning 用 lr 1e-5 | 漂移更小 | 只是减缓，不解决根因 |
| D. 推理端补救 | — | 微调后的 LoRA 推理时开 cfg>1（如 2–4，负 prompt " "）或降 LoRA scale（0.5–0.8） | 零训练，可立刻在现有 H-A/H-B 上试，用来**验证是否是 CFG 丢失**；cfg>1 推理慢 2× |

其他：HF 上社区 H3 LoRA/变体（Singularity、QuantFunc 4bit、lightx2v Turbo）未公开讨论微调画质与 CFG 的处理。

## 3. 同时发现的推理设置偏差（我们自己的问题）
- 官方 DiffSynth 管线默认 **video flow_shift=12、audio 3、50 步**（`minimax_h3_audio_video.py:105`）；E3 原始 H3 结果用的是官方默认。
- 我们的 `sample_h3_ar.py` 用 **shift=5（音视频同）、20 步**，`h3_cd.py` 的老师网格也是 shift 5。
  → H3AR 下所有评测（含 zs_bi_motion、H-A、H-B）都偏离官方设置，可能本身就带来画质差距；同设置内部对比仍然公平，但与 E3 原始 H3 对比不公平。
- 训练端时间步采样用 DiffSynth 的 H3 模板默认（shift 2.22），与上游一致。
- 社区 FIXES.md 提到的"时间步 = 1−σ、速度为 x0−ε"是原生 H3 约定；DiffSynth 的 model_fn 已在内部转换（`t=1−timestep/1000`，返回 `-v`），我们的代码按 DiffSynth 约定写，符号无误。

## 4. 建议的验证顺序
1. **零训练诊断**（1 节点，约 1–2 h）：H-A step-676 双向，① 官方设置（shift 12/3、50 步）；② 再加 cfg=3；③ LoRA scale 0.5。与原始 H3 同设置对比，人工 review。
2. **A：融合适配器重训 H-A**（N7+N8，200 步，其余不变）vs 现有 H-A step-200。
3. **B：training_cfg_scale=4**（需补无条件输入的代码），作为 A 的对照。
4. 胜出者作为之后 H-B / H-C 的默认设置；采样和 CD 网格改用官方 shift 12。

## 5. 10-10 晚 诊断结果（N2/N6，前 6 条 val joint，网页预设"【10-10 诊断】"）
| 配置 | 结论（抽帧 + 拉普拉斯清晰度/帧差） |
|---|---|
| 原始 H3 双向：我们的采样器 shift5/20 步 vs 官方 shift12/3、50 步 vs 原 caption | 三者画质相当（lapvar 329–335），**双向采样器本身没问题**；shift/步数/prompt 不是主因 |
| H-A676 双向 cfg1（官方 shift） | 手/手臂运动拖影，动作幅度小（帧差 10.3 vs 原始 14–15） |
| **H-A676 双向 + 真 CFG=3** | 手部清晰、动作幅度恢复 → **证实 H-A 退化主因是微调冲掉了 CFG 蒸馏** |
| 原始 H3 未训练逐块 AR（shift5 / shift12） | 画面清晰（lapvar 308–356），问题只是漂移/场景变化 |
| H-B500 逐块 AR cfg1（shift5 / shift12） | 第 1 个块（第 10 帧）就严重涂抹/重影，shift 无影响（lapvar 233） |
| H-B500 逐块 AR + 真 CFG=3 | 明显改善但仍有快动作拖影、头发涂抹、衣服长出纹理/字 → **CFG 丢失是一部分原因，AR 老师还有额外退化** |

训练数据检查：精选片段 24fps 中 2/3 帧是 RIFE 插帧（插帧相位清晰度低约 12%），快动作无明显重影；旧 36k（H-A 用的 f0）里有转场/换装样本。

## 6. 社区仓库调研（用户提供的 4 个 HF 仓库）
| 仓库 | 训练方式 | 对我们的价值 |
|---|---|---|
| akatz-ai/MiniMax-H3-Character-Swap-LoRA | **ostris ai-toolkit**；`assistant_lora_path` = ostris `minimax_h3_ref2va_training_adapter_v1`（冻结，训练时生效、推理丢弃）+ **`do_guidance_loss: true, guidance_loss_target: 3.5`**；r16、lr 5e-5、1000 步、AdamW8bit；训练配置 json 公开 | **完整可抄的 CFG 保护配方**（适配器 + guidance loss 同时用） |
| fal/MiniMax-H3-Realism-People-LoRA | fal 自家 trainer（未开源）；r32、1500 步；结论：训练分辨率比 rank/步数更重要；数据严格 24.000fps、慢动作重定时 | 经验：高分辨率桶、真 24fps |
| alvdansen/h3-keyframe-animation | diffusers 模块树训练后转 fused qkv；r64、12k 步；推理 shift 12/3、25–50 步、cfg 无 | 推理设置参考；需要转 qkv 布局 |
| Veda-Sparse/...-Veda-8NFE | 稀疏注意力预测器（非 LoRA），配合 larryvrh/MiniMax-H3-Turbo-Lora 8 步；代码 veda-sparse/Miowtion | 推理加速；Turbo LoRA 可作为少步参考 |
- 这 4 个都是任务 LoRA（换人/写实/关键帧），**没有可直接用作舞蹈/AR 初始化的权重**；可复用的是配方与 ostris 适配器。
- ostris ai-toolkit 的 guidance loss：`target = v_uncond + g·(target − v_uncond)`（v_uncond 由当前模型无梯度前向得到），与 DiffSynth `--training_cfg_scale` 数学上等价。
- ostris 适配器（v1/v2/v3，ComfyUI 布局，qkv 的 lora_B 行序需从 [role][head][dim] 转为官方 [head][role][dim]）已下载到 `models/H3-TrainingAdapter-ostris/`；DiffSynth 官方 Ref2VA 适配器已离线融进 DiT：`models/H3_Ref2VA_decfgDS/`（`h3ar/fuse_adapter.py`，训练用，推理仍用原 DiT）。
