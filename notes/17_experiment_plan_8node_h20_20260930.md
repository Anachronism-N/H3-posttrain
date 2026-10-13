# 17. 8 节点 H20(64 卡)后续实验安排(2026-09-30)

> 依据:`notes/16_rl_ar_distill_research_20260930.md`(调研结论)、stage4 训练日志、valtest 推理日志、HyperFlow A/B 实测。
> 目标(按优先级):① 把 stage4 收尾,拿到最终底座;② **少步学生**(一切下游的成本前提);③ 用 RL 缓解**音频捷径**;
> ④ 双向 → **AR 流式**,做长舞蹈;⑤ 判断 **H3** 是否值得转主线。

---

## 0. 关键成本事实(决定了整个排期)

| 项 | 实测 / 估算 | 来源 |
|---|---|---|
| stage4 训练 | 18.1 s/step(8 卡,global batch 32);剩 4600 步 ≈ 23 h | `OSS_fullSFT_stage4_*/train.log` |
| **stage4 推理** | **≈ 25 min/条/卡**(704×1280×81 帧,50 步,text_first CFG) | `logs_omnidance/valtest/*.log` |
| H3 + HyperFlow 8 步 | ≈ 131–137 s/条(4 卡 Ulysses);49 步基线 ≈ 432 s/条 | `hyperflow_exp/HyperFlow技术详解.md` |
| 4 步无 CFG 学生(预估) | 4 次前向 vs 50×3 次 → **≈ 40 s/条/卡**(约 35× 快) | 按前向次数折算 |
| 480p×49 帧 | token ≈ 0.27×,单次前向约 0.2× | 按序列长度折算 |

**推论**:在 25 min/条的老师上直接做 GRPO(每个 prompt 8 条)不可行;用老师采 ODE 轨迹也不可行(1 万条 ≈ 4200 GPU·h)。
→ **DMD 4 步学生是关键路径**:RL rollout、AR 的 ODE 初始化、大规模评测都要靠它。

---

## 1. 节点分工总览

记为 N1–N8(每个节点 8×H20 96GB)。原则:**默认每个实验只用单节点**(稳健,不依赖跨节点 IB);只有 DMD 主实验在 G0 验证 IB 可用后才跨两个节点。

| 节点 | 第 0–2 天 | 第 2–7 天 | 第 7–14 天 |
|---|---|---|---|
| **N1** | E1 stage4 续训到 16k | E2′ 评测最终 stage4 → E9 长视频续写 SFT | E9 续 |
| **N2** | E3 H3 零样本 A/B(4+4 卡) | 过 G1:H3 LoRA(E8);未过:并入 RL | 同左 |
| **N3** | E4 reward/评测服务搭建与校准 | reward 服务常驻(给 RL 供分)+ 评测队列 | 同左 |
| **N4** | E5a DMD 4 步 @480p/49 帧 | E6 AR @480p(ODE 对 → 因果回归 → Self-Forcing) | E6 AR @704p |
| **N5** | E5b DMD 4 步 @704×1280/81 帧(E5a 冒烟通过后启动) | E5b 续(IB 可用则与 N6 组成 16 卡) | E6 AR @704p |
| **N6** | E5c DMD 消融(见 §5) | E5b 扩容 或 E5c 续 | E7 RL @704p |
| **N7** | E7-0 RL 框架调试(用公开 Turbo 模型做替身) | E7-1 RL @480p,本文方法组 | E7 RL @704p |
| **N8** | E2 基线评测 + 音频捷径诊断 | E7-1 RL @480p,噪声/SDE 基线组 | E7/E8 灵活 |

空闲的卡一律挂 `gpu_holder`(沿用现有占卡规范,避免被集群回收)。

---

## 2. 第 0 天:基础设施(所有节点,约半天)

1. **环境**:`wan` conda 环境放在 gy2 cephfs 上,各节点直接用;H3 环境跑 `hyperflow_exp/setup_new_node.sh`,解压到本地 `/opt/conda/envs/hyperflow`。
2. **模型拷到本地盘**(cephfs 并发读很慢,§15 记录过 8 进程加载约 1 h):
   - Wan-AI(22 GB)+ MERT + stage4 ckpt 拷到各节点本地盘或 `/dev/shm`(494 GB);
   - H3 只拷 Ref2VA 需要的部分(transformer + Qwen3-VL-32B 文本编码器 + VAE,约 140 GB)到 N2/N8 本地。
3. **G0 跨节点检查**:两节点间跑 NCCL all-reduce 带宽测试(IB 用 `mlx5_bond_*`,GID index 3)。通过则 E5b 允许跨节点;不通过则所有实验单节点。
4. **统一评测集与协议**(§3)固化成一个 csv 和一个 `eval_all.sh`,所有实验共用。
5. **目录约定**:输出放 `models/train/E<id>_<desc>/`,日志放 `logs_omnidance/exp/E<id>/`。启动时必须带 `DIFFSYNTH_MODEL_BASE_PATH`、`DIFFSYNTH_SKIP_DOWNLOAD=true`、`WANDB_MODE=offline`;每个实验一个幂等启动脚本(参照 `scripts_omnidance/resume_stage3.sh`)。

---

## 3. 统一评测协议

- **评测集**:
  - `OD-Val-47`:即 `OmniDance/part049_val.csv`,主表用;
  - `OD-Val-200`:从未进入 `metadata_train.csv` 的 main/pass 样本里再抽 200 条,只给学生模型用(老师太慢)。
- **条件**:joint / audio_only / text_only 三种。
- **音频捷径诊断**(本项目的核心指标):
  - **文本替换**:图和音乐不变,换成另一条样本的编舞 prompt,计算 TextAlign(新 prompt) 与 TextAlign(原 prompt) 的差 → **文本可控性 Δtext**;
  - **音频替换**:prompt 不变,换音乐,计算 BeatAlign 的差 → **音频可控性 Δaudio**;
  - 两个 Δ 都要高才说明模型同时在用两种条件。
- **指标**:
  - 节拍:Pose-BAS(`scripts/eval_pose_bas.py`)、2D-MM Align、AV-Align;
  - 文本:VQAScore 式 TextAlign(§4);
  - 身份与首帧:ffPSNR、参考图与生成帧的 ArcFace/DINO 相似度;
  - 画质:VBench 经典 7 维(`hyperflow_exp/eval/run_classic.py`)、HPSv3;
  - 运动:运动幅度、坏帧检查(`logs_omnidance/valtest/analyze_valtest.py` 里的 flat/cast);
  - 效率:秒/条。
- **人评**:每个关键节点抽 30 对做 GSB 盲评(画质、卡拍、跟文本、身份四项)。

---

## 4. 各实验详细设计

### E1 stage4 收尾(N1,约 25 h)
- 从 `step-11000` 做 warm-start(只加载权重),输出到新目录 `OSS_fullSFT_stage4_resume_from11k`(**不能写回原目录**,步数计数会从 0 重来,会覆盖旧 ckpt)。
- 配置:`max_steps=5000`,warmup 200 步,lr 1e-5 恒定,3000→5000 步线性衰减到 0.2×(对齐原来 14k→16k 的衰减窗口),condition_quota 不变。
- 产出:`S4-final`,作为 DMD 第二轮的老师和 RL 的初始化。在它出来之前,下游一律先用 `S4-11000`。

### E2 基线评测 + 音频捷径诊断(N8 第 0–1 天;N1 在 E1 结束后评 S4-final)
- 在 OD-Val-47 上评 S4-11000,3 种条件加 2 种替换,共约 235 条 × 25 min ≈ 98 GPU·h,一个节点约 12 h。
- 同时从中取约 400 条带标签的视频(条件、是否替换都已知)交给 E4 做 reward 校准。
- 产出:基线表,以及**量化后的音频捷径**(Δtext 很低就是证据)。

### E3 H3 零样本 A/B(N2,约 1 天)
- 用 Ref2VA 的 `[reference generation + audio reuse]` 模式:参考图作 `<Picture 1>`,音乐作 `<Audio 1>`;编舞 caption 用 Qwen3-VL-8B 或共享的 235B vLLM 服务,按 `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` 改写成 H3 的结构化 prompt。
- GPU 0–3 跑 HyperFlow 8 步,GPU 4–7 跑 49 步基线;OD-Val-47 × {joint, 文本替换},约 8 h。
- 额外检查:输出音轨和输入音乐的波形互相关,确认 audio reuse 是不是真的原样复用;首帧和参考图的一致性。
- **G1(第 1–2 天)**:H3 在 Pose-BAS、身份一致性、TextAlign 上**都不低于** S4 → 开 E8,分 2 个节点;否则 H3 只作上界参照。

### E4 reward 与评测服务(N3)
- 服务组件,以 HTTP 常驻,RL 和评测共用:
  - **TextAlign**:Qwen3-VL-8B(本地 `models/Qwen3-VL-8B-Instruct`),把 prompt 预先拆成 5–10 个原子子句,每个子句问一次,取 P("Yes") 的平均(VQAScore 式);
  - **BeatAlign**:DWPose 抽骨架,算 Pose-BAS 和 2D-MM Align;
  - **Identity**:ArcFace/DINO 对参考图的相似度;
  - **VQ/MQ**:下载 HPSv3 和 VideoAlign(KVPO 用的同一套);
  - **Jev(可选)**:Qwen3-VL 写描述 → Jev 判断每个子句是否被满足,作为第二意见;需要 OpenRouter key,走 star-proxy 代理,固定 `jev-1.13-20260917` 版本。
- **G2 校准(第 2 天)**:用 E2 的 400 条视频:
  - TextAlign 能否区分原 prompt 和替换后的 prompt:AUC ≥ 0.8;
  - BeatAlign 能否区分原音乐和替换后的音乐:AUC ≥ 0.7;
  - 与 100 对人评的 Spearman 相关 ≥ 0.5;
  - 吞吐:单条视频的全部 reward ≤ 20 s(8 卡并发)。

  不达标就先修 reward,**不开 RL**。

### E5 DMD 4 步学生(关键路径)
- **代码底子**:fork `quanhaol/Wan2.2-TI2V-5B-Turbo`(Self-Forcing 框架里 TI2V-5B 的 DMD 实现),接入我们的 WanAudioModel(MERT → CausalAudioEncoder → injector)。三份模型:
  - generator:从 S4 初始化;
  - real score:冻结的 S4 老师,带我们的 text_first CFG(文本 2.0、音频 2.0);
  - fake score:从 S4 拷贝。
- **多条件蒸馏**:每个 batch 按 joint 60 / audio_only 20 / text_only 20 采样,老师各自用对应条件的 CFG,保证学生三种条件都能用。
- **E5a**(N4):480p×49 帧,batch 8,约 16 s/iter,4000 iter ≈ 18 h。先跑 200 iter 冒烟,确认 loss、梯度和样例正常,再放行 E5b。
- **E5b**(N5,G0 通过则加 N6 组成 16 卡):704×1280×81 帧,约 80 s/iter(8 卡),4000 iter:单节点约 3.7 天,两节点约 2 天。先用 S4-11000 作老师,S4-final 出来后续训一小段。
- **G3**:学生对比老师(OD-Val-47):
  - Pose-BAS ≥ 95%,TextAlign ≥ 95%;
  - ffPSNR ≥ 35;
  - VBench 美学/成像分差距在 2% 以内;
  - 速度 ≥ 20×。

### E5c DMD 消融(N6,第 1–4 天,480p)
二选一或依次做:
- (a) 老师 CFG 里有没有音频引导;
- (b) 只蒸 joint,还是按三种条件配比蒸。

看哪种设定学生的 Δtext 和 Δaudio 更好;这也直接回答"蒸馏会不会放大音频捷径"。

### E6 双向 → AR 流式(N4,之后 N4+N5)
1. **ODE 对**:用 E5a 学生(4 步、无 CFG,480p 约 4 s/条)在训练集 prompt 上生成 2 万组轨迹,约 25 GPU·h。老师太慢,不用老师生成(OmniForcing v1 的做法也是用蒸馏后的学生生成 ODE 对)。
2. **因果初始化**:
   - block-causal mask:首帧/参考图作单帧 sink,之后每 4 个 latent 帧一个 chunk,81 帧 = 1 + 5×4;
   - 音频可见范围:当前 chunk 加 1 个 latent 帧的前瞻(舞蹈有预备动作);
   - 然后做 causal ODE 回归。
3. **Self-Forcing DMD**:用真实 KV cache 自展开;另开一组"chunk 内双向"的变体(SoulX 式)对比。
4. **G5**:480p 连续生成 30 s 不崩;身份漂移(逐段 ArcFace)< 阈值;每段 Pose-BAS 不低于 E5a。通过后在 N4+N5 上做 704p。
5. 有了 AR 学生之后,可以直接跑 KVPO 原版的历史 KV 路由(第 3 周)。

### E7 RL:条件空间探索 + TVE(本文方法)
- **E7-0**(N7,第 0–3 天,先把框架跑通):用公开的 Wan2.2-TI2V-5B-Turbo(无音频)作替身策略,只用 TextAlign 作 reward。
  - 实现 KVPO 那套目标:TVE → Gibbs → PPO(clip 0.1/0.2)+ 离散 KL(β 扫 {1, 5, 10})+ 锚点门控;
  - 先在"共享 x_T + prompt 改写/子句 dropout"这种探索方式上把代码和稳定性调通。
- **E7-1**(N7 本文方法组 / N8 基线组,第 3–7 天,策略用 E5a 480p 学生):
  - **本文方法组**:共享 x_T;前 1/4 步里做**音频上下文路由**(每个 latent 帧的 4 个音频 token 在 ±k 帧窗口内随机取,最近帧固定)加子句 dropout;用真实条件回放计算 TVE。
  - **基线组**:Flow-GRPO/DanceGRPO 式,转成 SDE 后加噪声探索。
  - **Reward**(按条件切换):
    `R = 0.35·BeatAlign + 0.35·TextAlign + 0.15·HPSv3 + 0.15·MQ`,Identity 低于阈值则 R = 0;
    audio_only 条件去掉 TextAlign 项,text_only 条件去掉 BeatAlign 项。
  - **规模**:G=8,每轮 16 个 prompt,LoRA r128,lr 1e-5,每轮约 10 min,300 轮约 2 天。
- **G4**:reward 上升,同时 Identity 和 VQ 不下降;**Δtext 明显提升**(即音频捷径缓解);KL 稳定。通过后第 2 周在 N6–N8 上用 E5b 704p 学生放大。

### E8 H3 领域 LoRA(条件实验:G1 通过才开;N2 + N8)
1. 预计算:在 1 万条样本上离线算好 Qwen3-VL-32B 的文本隐状态和 VisualVAE/AudioVAE latent,约 1 天,单节点。
2. LoRA SFT @480p:音频 latent 作为干净条件行(timestep 固定为 clean,只对视频算损失),FSDP + Ulysses,每个 epoch 约 28 h。
3. 叠上官方 HyperFlow LoRA 看 8 步效果;不行再考虑按 AnyFlow 复现 flow-map 蒸馏。
- 注意许可证 IV.3:**H3 的输出不能用来训练 Wan2.2**,两条线的数据流必须分开。

### E9 长视频续写 SFT,FlowAct-R2 式(N1,E1 结束后)
- 在 S4-final 上把首帧条件扩展成最近 k 个历史 latent 帧(k=1–2),给历史帧加 σ~U(0, 0.3) 的噪声扰动,防止误差累积;
- 参考图作为额外一帧放在最前面,单独给 RoPE 偏移,即 (R+I)2V 雏形;
- 滚动 prompt 暂时手工分段,后续接 `wan_dance_planner`;
- 评测:3 段拼接(约 10 s)的身份漂移、段落衔接处的抖动、每段的 Pose-BAS。
- 它和 E6 AR 是**并行的两条长视频路线**:E9 不需要蒸馏,E6 可以实时。第 2 周末对比两者。

---

## 5. 时间线与决策点

```
第 0 天   G0 基础设施 / IB 检查          E1、E3、E4、E5a 冒烟、E7-0、E2 启动
第 1 天   E1 完成(约 25 h)             G1:H3 零样本结论
第 2 天   G2:reward 校准通过?          E5a 完成 → G3(480p)
第 3–7 天 E6 AR@480p;E7-1 两组 RL@480p;E5b 704p;E9 启动;(E8)
第 7 天   G4:RL 是否缓解音频捷径       G3(704p)
第 7–14 天 AR@704p;RL@704p;长视频两条路线对比(G5);KVPO 原版 CHR 用在 AR 学生上
```

算力合计(前两周):约 64 卡 × 14 天 ≈ 21,500 GPU·h,按上表不会有节点长时间空闲。

## 6. 风险与应对

| 风险 | 应对 |
|---|---|
| 8 个节点不是同一个多机作业,跨节点 IB 不通 | 全部单节点设计;E5b 在单节点跑约 3.7 天 |
| cephfs 并发读很慢 | 第 0 天把模型拷到本地盘;错峰启动 |
| DMD 学生模式坍缩、音频捷径被放大 | E5c 消融;多条件配比;G3 里加 Δtext 检查 |
| reward 被 hack(VLM 打分偏好某些画面) | 多个 reward 组合;Identity 作硬约束;定期人评;Jev 作第二意见 |
| H3 许可证 | 两条线数据隔离;不对外发布 H3 衍生权重,除非确认合规 |
| 集群回收空闲卡 | 空闲时挂 `gpu_holder`,禁止宽匹配 `pkill` |

## 7. 开工前需要的信息
1. 8 个节点的 IP 和登录方式(之前候选节点用的是 36000 端口 + root);是不是同一个 Taiji 多机作业(`/etc/taiji/hostfile` 里是否有 8 行)。
2. 本机上 wm_dataset round1 的任务是否继续占用;新节点里是否包含本机。
3. Jev 需要 OpenRouter API key(可选)。
4. 优先级确认:如果资源要收缩,建议保留顺序为 E1 > E5 > E4 > E7 > E3 > E6 > E9 > E8。
