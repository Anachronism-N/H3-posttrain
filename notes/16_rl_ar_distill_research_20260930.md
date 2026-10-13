# 16. 调研汇总:RL 对齐 / 双向→AR 蒸馏 / H3+HyperFlow / 注意力加速 / Jev reward(2026-09-30)

> 背景:OmniDance 主线 = Wan2.2-TI2V-5B + 音乐分支(MERT → CausalAudioEncoder → 8 个注入点的逐帧 audio cross-attn + AdaIN 全局调制),
> full SFT 多阶段,stage4 停在 step 11398/16000(最新 ckpt step-11000)。已知核心问题:**音频捷径**(joint 条件下模型偏听音乐、忽视文本)、
> 只能生成 81 帧(~3s)、50 步推理慢。
> 本文把本轮所有论文/项目的阅读分析放在一处,每节末尾给"对我们的结论"。

目录
1. KVPO(ODE-native GRPO,KV 语义探索)
2. FlowAct-R2(流式多模态参考 DiT + 主动 Agent)—— 未开源,列为后续参考
3. 双向音频/音乐驱动视频模型 → AR 蒸馏:相关工作调研
4. 可行性:我们的数据 + MiniMax-H3 + HyperFlow 训练框架
5. DraftAttention2(arXiv 2609.32628,低分辨率引导的混合精度稀疏注意力)
6. Jev(jev-arena)作为 RL 额外 reward + 开源同类
7. 综合路线图与优先级

---

## 1. KVPO: ODE-Native GRPO for Autoregressive Video Alignment via KV Semantic Exploration

- arXiv 2605.14278(2026-05-14),清华 + HKUST + **Video Rebirth**(即 HyperFlow 同一家);代码 github.com/Richard-Zhang-AI/KVPO(已开源,含权重/prompt)。
- 对象:**已蒸馏的少步流式 AR 视频模型**(LongLive、MemFlow;Self-Forcing 系,4 步 ODE,KV = sink 3 帧 + local 9 槽)。

### 1.1 动机
现有视频 RL(Flow-GRPO / DanceGRPO 等)用 **SDE 代理 + 噪声探索**:与蒸馏 AR 模型的**确定性 ODE**不匹配;噪声扰动主要改低层外观,不改高层语义/情节(长视频最需要的)。

### 1.2 方法
1. **因果语义探索 CHR(Causal History Routing)**:所有分支**共享同一初始噪声 x_T**;在 pivot 块 b* 之后的扰动窗口(5 个块)里,
   sink 3 帧不动,local 9 槽中**最近 3 帧固定**,其余 **6 槽从更早的非 sink 历史中随机抽取**(Ω_L={4..L-3})。
   只在**前 2/4 个 ODE 步**做路由(布局/运动在早期步决定)。→ 分支语义不同但仍在数据流形上。
2. **TVE(Trajectory Velocity Energy)代理策略**:把分支缓存的中间态 z 放到**未扰动上下文 K_<b** 下重放:
   `E_θ(X^g) = Σ_{b∈B} Σ_s (1/d)·‖v_θ(z^g_{b,s}, t_s, K_<b) − û^g_{b,s}‖²_F`,
   `π_θ(g) ∝ exp(−E_θ(X^g)/τ)`(Gibbs,只依赖组内相对 TVE)。
3. **优化**:PPO(非对称 clip 0.1/0.2)+ 离散 KL:`L = L_PPO + β·Σ_g π_θ(g)[log π_θ(g) − log π_ref(g)]`。
   展开梯度 = **按优势加权的对比式 flow-matching**:高奖励分支 → 拉近其速度目标,低奖励 → 推远。
   附加保护:若本轮没有任何分支奖励超过未扰动锚点 r⁰,**梯度置零**。
4. Reward:HPSv3(VQ)+ VideoAlign(MQ、TA);长视频按段计算后平均。

### 1.3 结果与成本
| 设置 | 基线 VQ/MQ/TA | +Astrolabe | +KVPO |
|---|---|---|---|
| LongLive 短视频 | 8.86 / 1.80 / 0.02 | 9.98 / 1.87 / 0.03 | **10.21 / 1.89 / 0.06** |
| LongLive 长视频(多 prompt) | 6.34 / 1.41 / −0.19 | 7.26 / 1.44 / −0.18 | **8.14 / 1.50 / −0.14** |
| MemFlow 长视频 | 6.30 / 1.39 / −0.20 | 6.52 / 1.35 / −0.23 | **6.96 / 1.44 / −0.17** |

- 消融:TVE 换成隐空间 L2 → VQ 掉到 6.02(低于基线);β=0 崩溃(β=5 最好);6/9 槽最佳(9/9 失稳、3/9 多样性不足)。
- 成本:LoRA r256、lr 1e-5、G=8、**32×H200**、~960 s/iter、约 **1000 GPU·h**。
- 审慎:只与 Astrolabe 比;"TA +200%" 实为 0.02→0.06;π 是构造的代理分布而非真实似然。

### 1.4 对我们的结论
- **不能直接用**:我们是双向注意力、50 步、一次出 81 帧,**没有 KV cache/历史帧**。
- **可迁移的是目标函数**(TVE + Gibbs + PPO + KL + 锚点门控),它只要求"共享 x_T + 能在未扰动条件下重放"。探索源换到我们的**条件空间**:

| KVPO | OmniDance 对应 |
|---|---|
| 历史 KV 6/9 槽路由 | **音频上下文路由**:每个 latent 帧的 4 个 audio token 在邻近时间窗内随机取,最近帧固定(结构上对应"保留最近 3 帧") |
| 未扰动上下文重放 | 真实音乐 + 原 prompt 重放 |
| 只扰前 2/4 步 | 只扰 50 步中的前 20–30% |
| HPSv3 / MQ / TA | Pose-BAS / 2D-MM Align(节拍)+ 文本对齐(见 §6)+ 身份一致性 + HPSv3 |

- **最大价值**:RL 能在 joint 条件下同时奖励"卡拍"与"遵循文本动作",**直接惩罚音频捷径**分支(SFT 的 condition_quota 只能间接调配)。
- 风险:算力。704×1280×81 帧×50 步×G=8 在 H20 上非常慢 → 先 480p/49 帧/~20 步 做可行性;或先蒸馏少步学生(§3/§4)再做 KVPO(它本就为少步 ODE 设计)。
- 一旦有了 AR 学生(§3),KVPO 原版 CHR 可直接使用。

---

## 2. FlowAct-R2: Beyond Talking Avatar via Streaming Multimodal References and Proactive Agent Planning

- arXiv 2609.35728(2026-09-28),字节 Intelligent Creation。**技术报告,未开源**,底座闭源(Seedance 2.0 Mini R2V)。**列为后续参考工作**。
- 仅有定量:对 Vidu-S1 人评 GSB 画质 +54.76%、实时交互 +40.48%;对 Vidu-S2 定性比较参考切换。无消融。

### 2.1 方法要点
1. **Streaming Multimodal Reference DiT**:把 Diffusion Forcing(逐 token 独立噪声等级)用于分块 AR 流式推理,**块内保留双向时空建模**;
   条件也变成流——滚动动作 prompt、驱动音频、随时更新/切换的图/音/视频参考。
2. **video-driven RoPE**:把中途到达的参考块对齐到生成时间轴。
3. **(R+I)2V**:参考图与首帧分开、显式训练,锚定身份/外观。
4. **部分加噪的历史运动帧**作条件 → 对自生成的不完美历史鲁棒、减轻误差累积;coarse-to-fine 推理;专门的音频驱动训练(口型/表情/头姿)。
5. **Proactive Interaction Agent**:离线(人设、长时程议程、可复用多模态技能)+ 在线(按观众事件发起/叠加/延后/打断/恢复),以 idle/listening/speaking/skill 时段输出滚动 prompt 与参考。跳舞是一种技能(文本触发 / 参考视频驱动)。
6. 实时 720p、小时级。

### 2.2 对我们的结论(后续参考)
- **加噪历史帧的分块续写** → 解决"只有 3 秒"问题。TI2V-5B 本来就把首帧 latent 作干净条件,扩展成"前 k 帧历史 + 小噪声"续写改动小,可直接用 OmniDance 训练。
- **(R+I)2V**:续写时首帧来自上一段生成,需额外参考图防身份漂移。
- **滚动 prompt + 离线编舞规划**:LLM 读音乐结构(段落/节拍/情绪)→ 分段动作 prompt,与已有 `diffsynth/models/wan_dance_planner.py`、plan_schedule 思路一致,可作长视频推理上层。
- 实时交互 Agent、直播场景与我们研究问题无关,不跟。

---

## 3. 双向音频/音乐驱动视频模型 → AR 蒸馏:相关工作调研

(已逐条打开 arXiv/GitHub 核实;标 *待核实* 的为未能直接确认的细节。)

### 3.1 通用配方(CausVid → Self-Forcing → LongLive 一脉)
1. **因果初始化**:给双向老师加 block-causal mask,用老师多步 ODE 轨迹做回归(ODE init / causal ODE regression)。
2. **少步分布匹配 + 自回归自展开**:DMD(generator + real score=老师 + fake score)在训练时**用真实 KV cache 自己 roll out**(Self-Forcing),消除 teacher/diffusion forcing 的训练-推理分布差。
3. **长视频技巧**:sink 帧 + 滚动 local KV、短窗口注意力、prompt 切换时 KV 重缓存(LongLive)、记忆检索(MemFlow)。
- 经验(FastVideo CausalWan-MoE):I2V 时**输入图作为单帧**而非复制成一个 chunk,否则首帧过渡生硬。

### 3.2 音频/语音驱动的双向 → AR 蒸馏
| 工作 | arXiv / 代码 | 底座 | 配方要点 | 速度 | 开源程度 |
|---|---|---|---|---|---|
| **TalkingMachines**(Character.AI) | 2506.03099 | Wan2.1 I2V-14B → 18B 音频驱动模型 | 双向老师 → **稀疏因果** AR 学生的非对称蒸馏;DiT/VAE 解码分卡、CUDA stream 重叠 | 实时 FaceTime 式 | 训练码 *待核实*(似未开源) |
| **Live Avatar**(阿里 Quark,ECCV26 Spotlight) | 2512.04677 / Alibaba-Quark/LiveAvatar | **Wan2.2-S2V-14B** | DMD 蒸到 4 步 + 块自回归;**Timestep-forcing Pipeline Parallelism**(TPP,按去噪步分卡流水);滚动一致性机制;支持 10000 s+ | 45 FPS(5×H800);单 80GB 卡可离线跑 | 推理 + LoRA 权重开源,**训练代码未放** |
| **SoulX-FlashTalk**(Soul AI Lab) | 2512.23379 / Soul-AILab/SoulX-FlashTalk | 14B(音频驱动数字人) | **Self-correcting Bidirectional Distillation:chunk 内保留双向注意力**,chunk 间自回归;**多步回溯自纠错**防长时崩溃;混合序列并行、并行 VAE | 首帧 0.87 s、32 FPS | 推理开源,训练 *待核实* |
| **Mutual Forcing** | 2026-04(arXiv 号 *待核实*) | 音视频角色生成 | 不走蒸馏,直接训练**原生快速因果音视频模型**,双模式自进化保证长时音画同步 | — | *待核实* |

要点:以上全是**语音→口型/数字人**;**尚未找到"音乐→舞蹈"的 AR 蒸馏开源工作**(FlowDance 2608.15818、MACE-Dance 均为双向/离线)——这正是我们的空白点。

### 3.3 联合音视频模型 → AR(对 H3 最相关)
- **OmniForcing**(京东 Joy Future Academy,**ECCV 2026 Oral**,arXiv 2603.11647,github.com/OmniForcing/OmniForcing,**完整训练代码 + 数据处理开源**):
  - 底座 LTX-2(19B:14B 视频 + 5B 音频双流双向)→ 流式 AR,单卡 ~25 FPS,首块延迟 0.7 s(老师 197 s),VBench 与老师持平或略好。
  - **非对称 block-causal 对齐**:以 1 s 物理时间为 macro-block 同步两模态(视频 3 latent 帧×384 token vs 音频 25 token),**零截断 Global Prefix** 全局可见防同步漂移。
  - **Audio Sink Token + Identity RoPE**:解决音频 token 稀疏导致的 softmax 塌缩/梯度爆炸。
  - **Joint Self-Forcing DMD**:两模态互相适应对方的预测漂移;模态独立的滚动 KV cache。
  - v2 管线简化为 2 阶段(causal ODE 回归 → self-forcing DMD,直接以原始 LTX-2 作老师),块布局 4-3-3-3-3,SDPA。**最低 8×H200(141GB),推荐 32×H200**。
- 对 H3 的映射:H3 是**单流**打包序列 + 模态专属 AdaLN,做 block-causal mask 比双流更简单;而我们的任务里**音乐是输入**(不需生成),
  音频 token 可作为"干净条件流"(当前块 + 少量前瞻全可见),只需把**视频**变因果——比 OmniForcing 的联合生成更容易,本质是"H3 版 LiveAvatar"。

### 3.4 可直接起步的代码库(针对 Wan2.2-5B)
| 代码库 | 状态 | 说明 |
|---|---|---|
| **quanhaol/Wan2.2-TI2V-5B-Turbo** | 训练 + 推理开源 | 基于 Self-Forcing 框架对 **TI2V-5B** 做 DMD(4 步、去 CFG,1280×704×121 帧);**双向**少步(非因果);16×A100 · 4000 iter · 48 h |
| guandeh17/Self-Forcing、GoatWu/Self-Forcing-Plus | 开源 | Wan2.1-1.3B 因果 AR 的原始实现 |
| tianweiy/CausVid | 开源 | ODE init + 非对称 DMD 的原始实现 |
| NVlabs/LongLive | 开源 | 长视频 AR(KVPO 的底座之一) |
| hao-ai-lab/FastVideo(CausalWan2.2-I2V-A14B-Preview) | 预览权重 | Wan2.2 MoE 的 self-forcing,含 I2V 首帧处理经验 |
| AnyFlow-FAR-Wan2.1-1.3B | 权重开源 | flow-map 蒸馏的因果版,任意步数 |

### 3.5 结论与推荐路径
**(i) 我们的 Wan2.2-5B 音频模型(最快路径,预计数周级,8×H20 可做)**
1. **Stage A(双向少步)**:fork `Wan2.2-TI2V-5B-Turbo`,接入我们的音频分支(MERT → CausalAudioEncoder → injector),以 stage4 ckpt 作 real-score 老师、拷贝作 fake-score,做 DMD 4 步。
   这一步本身就有价值(推理 50 步 → 4 步 ≈ 10×+),也为 RL 提供便宜 rollout。5B×3 份模型在 H20 可放下。
2. **Stage B(因果初始化)**:block-causal mask(chunk = 3 latent 帧 ≈ 12 帧视频),首帧/参考图作**单帧 sink**;用老师 ODE 轨迹回归。
   **我们的音频注入天然适配因果**:injector 是**逐帧** cross-attn(帧 t 只看 audio_emb[:, t]),CausalAudioEncoder 是因果卷积 → 只需给每个 chunk 放开"当前 chunk + 1 个 latent 帧前瞻"的音频(舞蹈对节拍有预备动作)。
3. **Stage C(Self-Forcing DMD)**:KV cache 自展开;可选 SoulX 式"**chunk 内双向**"以保运动质量、降低改造难度。之后可接 LongLive 式长视频与 **KVPO 原版**(§1)。

**(ii) H3**
- 技术上可行:套 **OmniForcing** 配方(且音乐作干净条件流更简单),许可证允许对 H3 做衍生。
- 但成本高:33B × DMD 三份模型(LoRA 共享底座也要 ~70 GB 权重 + 32k token 激活),OmniForcing 19B 已需 ≥8×H200,H3 估计需 **16–32 张 H200 级**;8×H20 只适合做 LoRA 级小实验。
- 建议顺序:先 §4 的 H3 零样本(audio reuse + HyperFlow 8 步)判断上限,再决定是否投入 H3→AR。

---

## 4. 可行性:我们的数据 + MiniMax-H3 + HyperFlow 训练框架

### 4.1 已核实的事实(本地权重/仓库 + 之前 A/B 实测)
- **H3-Base**:33B dense 单流 Omni-Transformer(50 层、hidden 5376、56 头、ffn 14336;本地 `transformer/` 62 GB bf16;约 13B 参数在 AdaLN 分支),
  文本编码器 = Qwen3-VL-32B 第 50 层隐状态;VisualVAE f16t4d24(因果),patch 1×2×2(等效 32× 空间、4× 时间);AudioVAE 32 kHz 立体声 → 40 Hz latent;
  视频/音频**各自独立 sigma shift**(12 / 3),**已 CFG 蒸馏**;24 fps,4–15 s,短边默认 768。
- 两个 checkpoint:**FL2VA**(t2va / 首尾帧)与 **Ref2VA**(≤9 图、≤3 视频、≤3 音频参考)。
- **关键发现:Ref2VA 原生支持 `audio reuse`**(提示词指南:"Copying all or part of an audio signal … Referencing beat, rhythm")——
  即把给定音乐**直接复用**为目标视频音轨,并以 `<Picture 1>` 作参考/关键帧。**这意味着 H3 零训练即可做 MI2V/MTI2V(音乐+图+文→舞蹈)**。
- **HyperFlow**:只开源推理(`TwoTimeEmbedder`、LoRA 加载、8 步 sigma 网格 `schedule.py`),**训练代码未开源**;方法来自 AnyFlow(arXiv 2605.13724,NVlabs/Show Lab/MIT,on-policy flow-map 蒸馏;已放出 AnyFlow-FAR-Wan2.1-1.3B 等权重,训练代码是否开源需核实)。
- 我们的 8×H20 实测:HyperFlow 8 NFE ≈ **131–137 s/条**,基座 49 NFE ≈ **432–455 s/条**(4 卡 Ulysses),加速 3.2–3.5×;显存 80–86 GB/卡;质量两轮评测持平或略优(详见 `hyperflow_exp/HyperFlow技术详解.md`)。
- **许可证(MiniMax H3 Community License)**:允许修改与 Model Derivatives(可以微调/蒸馏 H3 本身);
  **禁止用 H3(及其衍生)的输出去改进其他 AI 模型**(第 IV.3 条)→ **不能用 H3 生成视频当老师数据去蒸馏/训练我们的 Wan2.2 模型**;年收 >$2000 万商用需授权;有地域限制(按我们此前整理:排除 EU/UK/韩/美)。

### 4.2 结论:可以,但分三级推进

| 级别 | 做什么 | 训练? | 成本 | 风险 |
|---|---|---|---|---|
| **L0 零样本** | Ref2VA + `[reference generation + audio reuse]`,OmniDance 首帧作 `<Picture 1>`、音乐作 `<Audio 1>`,把我们的编舞 caption 改写成 H3 结构化 prompt;叠加 HyperFlow 8 步 | 否 | 47 条 val × ~130 s / 4 卡 ≈ 1.7 h(两组并行) | Ref2VA 的参考图是"参考"不是严格首帧;音乐时间对齐程度未知 |
| **L1 领域 LoRA SFT** | 在 OmniDance 上对 H3 做 LoRA:音频 latent 作**干净条件行**(同 HyperFlow 的"条件行 r=t"思想:音频 timestep 固定在 clean,只对视频去噪计损失) | 是(需自写训练环) | 见 4.3 | 无官方训练代码;33B 显存/速度 |
| **L2 复现 HyperFlow/AnyFlow** | 在 L1 模型上做 flow-map 自蒸馏(双时间 embedder + 8 步网格),或先试"L1 LoRA + 官方 HyperFlow LoRA 叠加" | 是 | 老师 rollout 成本极高(见 4.3) | 训练码需按论文复现 |

### 4.3 资源估算(H20 ×8)
- 序列长度(768×1344、5 s):视频 ≈ 31 latent 帧 × (24×42) ≈ 31k token + 音频 200 + 文本/参考 → ~32–35k。
- 单样本前向 ≈ 2·31e9·32e3 + 注意力 50·4·N²·d ≈ 2e15 + 1.1e15 ≈ **3e15 FLOPs**;fwd+bwd(梯度检查点)≈ 1.2e16。
  H20 BF16 148 TFLOPS、MFU 按 40% → **~200 GPU·s/样本**,8 卡 ~25 s/样本 → 1 万样本/epoch ≈ 70 h。480p 约快 2.5×。
- 显存:权重 FSDP 8 路切分(62 GB → ~8 GB/卡)+ Ulysses SP + 检查点可行;**文本编码(Qwen3-VL-32B,~64 GB)与 VAE 编码必须离线预计算**。
- L2 的 data-free 自蒸馏:老师每条 49 NFE ≈ 432 s × 4 卡,1 万条 ≈ **4800 GPU·h** 仅用于生成 → 用我们的**真实 OmniDance 数据**替代老师 rollout(on-policy 部分再少量补)更现实。

### 4.4 数据适配清单
- 分辨率:OmniDance 704×1280(9:16)→ H3 768×1344 或 480p 档;帧率转 24 fps;时长切 4–15 s。
- 音频:重采样 32 kHz 立体声 → H3 AudioVAE latent(替代 MERT 特征)。
- 文本:编舞 caption → H3 `subject_definitions / summary / integrated_multimodal_description` 格式(LLM 批量改写,参考 `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`)。
- 评测:沿用 Pose-BAS / 2D-MM Align / AV-Align / 身份一致性,与 Wan2.2-audio stage4 在同一 val(`part049_val.csv`)上对比。

### 4.5 建议
**先做 L0**(不训练,一两个小时出结论):若 H3+audio reuse 的卡拍与身份已明显优于我们 stage4,则主线值得迁到 H3(L1→L2);否则 H3 只作上界参照,主线仍是 Wan2.2-5B(更适合做 AR 蒸馏与 RL,§3/§1)。

---

## 5. DraftAttention2: Fast Video Diffusion with Low-Resolution-Guided Mixed-Precision Attention(arXiv 2609.32628)

- 2026-09-26,南大 / HKUST(GZ) / Duke / NVIDIA / 浙大 / Adobe;代码 github.com/anemoi-project/anemoi(Anemoi 视频生成推理框架)。**training-free**。

### 5.1 方法
1. **Draft 注意力图**:每帧切成紧凑 2D 区域(ragged 2D 分区,最小周长、<1 块 padding),对 Q/K 做**区域平均池化 + 通道最大池化**:
   `A = (1−λ)·softmax(Q̄K̄ᵀ/√d) + λ·softmax(QᵐᵃˣKᵐᵃˣᵀ/√d)`(默认 λ=0.2)。
2. **稀疏 + 精度联合分配**:每个 head 内对全部 R² 区域对按 A 排序:前 T16 → 16bit,次 T8 → INT8(QKᵀ INT8、PV FP8),再 T4 → NVFP4,其余跳过。
3. **理论**:误差界 = 稀疏项 2V_max·δ + 注意力质量加权的量化项 Σ m⁽ᵖ⁾ε_p;证明"把中等重要的块用低比特算回来"优于直接丢弃(比 v1 界更紧),高精度应给注意力质量最大的块。
4. **内核**:Q/K 重排+池化+量化融合;单 kernel 按精度分相(≤2 次切换)、共享 online-softmax 统计与 FP32 累加器。
5. 保护:标准模型前 10 步 dense;少步模型第 1(4 步)/前 2(8 步)步 dense,且每步前 2 层 dense。

### 5.2 结果(RTX 5090)
- HunyuanVideo-13B(720p,129 帧)与 **MiniMax-H3**(1344×768,**362 帧**):纯稀疏 2.28×/2.23×;混合精度最高 ~2.5–2.6× DiT 加速,VBench 与 dense 接近,PSNR/LPIPS 优于 SVG2/SVG-EAR/SVOO/Sol-Attn;对 SageAttn3 同速更好。
- **少步 H3(FL2VA Turbo 4/8 步 LoRA)**:1.6–2.1× 加速,质量几乎不掉,优势比多步更明显(少步无法纠正近似误差)。

### 5.3 对我们的结论
- **硬件不兼容**:Anemoi 只提供 **SM89(4090)/ SM120(5090)** 内核,NVFP4 需 Blackwell。**H20 是 SM90(Hopper),不能直接用**;需移植(Hopper 有 FP8/INT8 tensor core,可参考 SageAttention2/SpargeAttn 的 SM90 实现)。
- **对 Wan2.2-5B 收益有限**:704×1280×81 帧 ≈ 21×(22×40) ≈ 18.5k token,远小于论文 109k token 场景,注意力占比与可稀疏度都更低;舞蹈依赖长程时间注意力,激进稀疏有损运动(论文表中 SVG2 在 HunyuanVideo 的 Dynamic Degree 从 72 掉到 9.7 即是警示)。
- **适用场景**:(a) 若走 H3 路线(§4),它正好在 H3 + turbo LoRA 上验证过,长序列收益大;(b) **RL rollout 加速**(rollout 是 GRPO 的主成本)。
  注意:rollout 用近似注意力、训练重放用 dense 会带来 rollout/训练策略不一致(类似 LLM RL 的 vLLM↔FSDP mismatch),KVPO 的 TVE 目标 û 来自 rollout,会把近似误差带进目标——需小规模验证或只在 rollout 的后期步使用。
- 优先级:**低**。等确定走 H3 或 RL rollout 成为瓶颈再做 SM90 移植。

---

## 6. Jev(jev-arena)作为 RL 额外 reward

### 6.1 Jev 是什么(据 github.com/NanmiCoder/jev-arena 及其"模型原理"页)
- **TypeSafe 的 "System One" 判断模型 `jev-1.13`**,经 OpenRouter `POST /api/alpha/decisions` 调用。**不生成文字,只返回带类型的判断与概率**:
  `noul`(是/否 → P(是))、`choice`(选项分布 + confidence)、`score`(有序档位,返回可落在两档之间的连续下标 + 分布)。
- 特性:70–500 ms/次;$0.042/百万输入 token,输出不计费;同一 state 上多个问题并行、只发一次文本;限 1200 请求/分钟、state+最长问题 ≤32k token。
- 边界(官方明确):**只吃文本**(不支持图/音/视频);不能微调;英文为主;校准是批级别的,单条不保证。
- jev-arena 本身只是一个"Jev vs DeepSeek 评论打标对决场"(Node.js):1 万条评论 Jev 203 s / $0.84 vs DeepSeek 823 s / $1.50,
  三项全对率 62.69% vs 67.26%(GPT-6 Astra 复核,非人工金标)——**便宜 4× 快、略不准**。它与视频无关,可借鉴的是 Jev 的"类型化 + 校准概率"判断接口。

### 6.2 作为 reward 的设计
Jev 看不到视频,必须经过"视频 → 文本"的中间层:

```
rollout 视频 ──(本地 Qwen3-VL 结构化描述: 动作/肢体/节奏感/镜头/身份/瑕疵)──► caption
prompt ──(离线 LLM 拆成原子子句, DSG 式)──► {c_1..c_k}
Jev: state = caption, questions = { c_i: noul("描述中的舞蹈包含: c_i"),
                                     fidelity: score("动作与文本描述的吻合程度", 5 档),
                                     artifact: noul("描述中出现肢体畸形/穿模/闪烁") }
R_text = mean_i P(c_i) (+ 0.x·score) ;  R_artifact = 1 − P(artifact)
```

- **优点**:输出是**连续校准概率**(不像 VLM yes/no 二值)→ GRPO 组内优势不易并列、方差小;一次请求问十几个问题,成本可忽略
  (每步 G=8×batch 32=256 条 ≈ 256 次请求,远低于 1200 次/分钟);擅长"文本 A 是否支撑文本 B"(官方用例:校验引用/抽取)。
- **直接对准音频捷径**:在 joint / text_only rollout 上奖励"文本动作子句被满足",只听音乐不看文本的分支 R_text 低。
- **风险**:
  1. **信息瓶颈 + 被 hack 的是 captioner**:生成器可能学会"让 captioner 写出关键词"的画面;需 captioner 与 Jev 双冻结、定期人工抽检、并用视觉直读 reward 交叉校验。
  2. 不感知音乐:节拍仍需 Pose-BAS / 2D-MM Align(`scripts/eval_pose_bas.py`、`notes/metric_report_2d_mm_align_av_align_wav2clip.md`)。
  3. 外部 API:集群需走代理(star-proxy),有网络/限流/版本漂移风险(需锁定 `jev-1.13-20260917` 版本号);校准在其领域,不在舞蹈描述上 → 先用 ~200 条人工标注核对。
  4. 不可微调,英文为主 → caption 与问题用英文。

### 6.3 开源同类(可本地部署、多数能直接看视频)
| 工作 | 形式 | 与 Jev 的关系 |
|---|---|---|
| **VQAScore**(t2v_metrics) | VLM 回答 "Does this show {text}?" 取 **P("Yes")** | 最接近"校准概率判断",且直接吃图/视频;可换 Qwen3-VL 作骨干 |
| **VisionReward**(THUDM,AAAI 2026) | 把偏好拆成一系列**是/否判断题**,线性加权求和,支持视频 | 等价于"Jev 式清单 + 学到的权重",可解释 |
| **VideoAlign / VideoReward** | VQ / MQ / TA 三维回归 | KVPO 用的就是它,可直接复用 |
| **UnifiedReward(-Think)**(CodeGoat24) | 统一多模态奖励模型,含 CoT 版本,开源训练脚本 | 可按我们数据微调(Jev 不能) |
| DSG(Davidsonian Scene Graph) | prompt → 原子问题 + VQA 回答 | 提供"prompt 拆子句"的做法 |

### 6.4 建议
- **本地主 reward 用"Jev 式"接口但自建**:Qwen3-VL(8B/32B)对每个原子子句取 P("Yes")(VQAScore 式,直接看视频帧),在 ~200 条人工标注上做温度标定——
  得到与 Jev 同样"类型化 + 概率"的输出,但能看视频、可控、可微调。
- **Jev 作为廉价第二意见**:caption → Jev 的文本蕴含概率与本地 VLM 分数做集成(取均值或 min),降低单一裁判被 hack 的风险。
- 总 reward(按条件分别启用):
  `R = w_beat·BeatAlign + w_text·TextAlign(VLM ⊕ Jev) + w_id·Identity(与参考图) + w_vq·HPSv3/VQ + w_mq·MQ`,身份低于阈值时整体置 0(硬约束);
  audio_only rollout 关闭 text 项,text_only rollout 关闭 beat 项。

---

## 7. 综合路线图与优先级

| 优先级 | 事项 | 依赖 | 产出 |
|---|---|---|---|
| P0 | H3 Ref2VA + audio reuse + HyperFlow 零样本跑 OmniDance val(§4 L0) | 空闲 8 卡(当前被 wm_dataset 占) | H3 能否作为新主线/上界 |
| P0 | reward 管线原型:Pose-BAS + Qwen3-VL VQAScore 式文本对齐 + Jev 第二意见,在 stage4 已有生成样本上离线打分,核对与人工判断一致性 | 无需训练 | 可信的 reward |
| P1 | 条件空间探索 + TVE 的 GRPO 小规模验证(stage4 step-11000 + LoRA,480p/49 帧/~20 步,G=4) | P0 reward | 缓解音频捷径 |
| P1 | 双向→AR:按 §3 推荐路径把 Wan2.2-5B-audio 转成分块因果少步学生(ODE 初始化 + DMD/Self-Forcing) | §3 | 流式/长舞蹈;之后可用原版 KVPO |
| P2 | FlowAct-R2 式加噪历史续写 + (R+I)2V + 编舞规划滚动 prompt | P1-AR 或独立 | 长视频 |
| P2 | H3 领域 LoRA / HyperFlow 复现(§4 L1/L2) | P0 结论为正 | 高质量少步 H3 舞蹈 |
| P3 | DraftAttention2 SM90 移植(用于 H3 推理或 RL rollout) | 走 H3 或 rollout 成瓶颈 | 1.5–2× DiT 加速 |

待决事项:① stage4 是否续跑完 16k 步;② GPU 从哪来(本机被 wm_dataset round1 占满);③ H3 许可证(IV.3)限制了"H3 输出 → 训练 Wan2.2"这条路,需在路线选择时考虑。
