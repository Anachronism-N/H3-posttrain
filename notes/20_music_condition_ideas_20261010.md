# 音乐条件增强 idea 记录（2026-10-10，待 H3 AR 主线稳定后尝试）

背景：H3（33B）只通过自带音频 VAE（DAC 风格卷积编码器，32 kHz，hop 800，40 帧/秒、32 通道的声学 latent）接收音乐，
没有专门表示节拍、曲风的语义特征，音乐能否控制舞蹈全靠 DiT 自己从声学 latent 里学。下面两条思路都是想让音乐信号更"可读"。

---

## M1：给 H3 额外接入 MERT 或节拍特征作为音乐条件

### M1a 作为 condition 注入
- 特征：MERT-v1-330M 25 层隐状态（24 kHz，75 帧/秒，1024 维），用可学习层权重加权求和（同 Wan2.2 S4 的做法）。
- 时间对齐：75 帧/秒插值到 H3 音频 latent 的 40 帧/秒，逐音频 token 对应；或插值到视频 latent 帧率，按时间加到视频 token 上。
- 注入方式（由简到繁）：
  1. 残差加法：MLP 投到 H3 hidden dim，**零初始化门控**后加到音频 token（或视频 token）上，训练开始时等价于原始 H3，不破坏画质；
  2. 额外的音乐 token 流：作为新的 token 组拼进序列，复用 H3 的联合注意力（注意 AR 时要遵守块因果 mask 和 lookahead）；
  3. cross-attention adapter：每 N 层加一层交叉注意力，输出零初始化。
- 节拍特征（可与 MERT 拼接或单独用）：beat / downbeat / onset 强度、节拍相位 sin/cos、tempo，逐帧嵌入后按时间加到视频 token 上。
  beats 已有缓存（`logs_omnidance/exp/beats.json`、`E2/beat_v2.jsonl`）；注意 librosa 与 torch 同进程会崩，需离线预计算。

### M1b 写入 prompt
- 把音乐信息写成文本：BPM、节拍/重拍时间点、曲风、情绪、分段能量（如 "0–2s 舒缓，2–5s 强拍密集"）。
- 来源：librosa 节拍/能量 + 音频理解模型（Qwen-Audio 类）或 MERT 标签头给曲风。
- 优点：零训练成本，可以先在原始 H3 上零样本测，再进训练数据。
- 风险：文本时间戳精度有限，H3 不一定能把文本里的时间点对齐到帧；可能只影响整体风格而不是逐拍同步。

### 评测
aswap（换音乐后动作是否跟着变）、BAS / BAS_alt 差值、G2 音乐-动作对比判别器、人工 review（音乐同步分）。
画质必须同时人工确认不下降（自动指标看不出画质差距，见 10-10 review 结论）。

---

## M2：多 encoder 融合的 MoE 架构编码音乐

- 思路：不同编码器擅长不同层面——
  - H3 音频 VAE：声学细节、与 H3 原生空间一致；
  - MERT：音乐语义、节奏、和声；
  - 节拍/onset 特征：精确的节奏时间点；
  - CLAP / wav2clip 类：全局曲风和情绪；
  - （可选）BEATs、Whisper encoder 等通用音频表示。
- 结构：各 encoder 冻结、特征离线缓存 → 各自线性投到公共维度并对齐到同一时间网格 → 逐时间步的 router（softmax 或 top-k 门控）
  加权融合各"专家" → 零初始化后注入 H3（注入方式同 M1a）。
- 训练细节：负载均衡损失防止 router 塌缩到单一 encoder；可加 encoder dropout，避免依赖某一路。
- 消融：单 encoder vs 直接拼接 vs MoE；top-1 vs soft；router 是否逐时间步。
- 可解释性（可作论文卖点）：可视化 router 在不同音乐段落（前奏/副歌/鼓点密集段）偏向哪个 encoder。
- 风险：计算和显存（33B 主干上增量不大，主要是特征缓存的存储）；多路条件可能放大"音频捷径"问题，需要 aswap 检验。

---

## 优先级与前置条件
1. 先等 H3 AR 主线（H-A → H-B，以及 H-B0 的 TF / 混训 / DF 对比）出结果。
2. M1b（写入 prompt）最便宜，可最先在原始双向 H3 上零样本试。
3. M1a（MERT/节拍注入）其次，在双向 H3 上做 LoRA + 零初始化 adapter；有效再迁移到 AR 版本。
4. M2（MoE 融合）在 M1a 证明"额外音乐特征有用"之后再做。

---
## 2026-10-10 22:10 M1a 启动（N4）
- 代码：`h3ar/h3_music.py`（MusicInjector：25 层 MERT 的 softmax 层权重 → LayerNorm → MLP 1024→1024→5376，末层零初始化；
  16 fps MERT 线性插值到 40 Hz 音频 latent 网格，加到**目标音频 token**（双声道同加）的 embedding 上，挂在 DiT._embed 之后）。
  上游补丁（备份 `.bak_20261010music`）：unified_dataset 在 OD_H3_MUSIC=1 时给缓存样本附 `od_cache_path`；train.py 在 DeepSpeed 初始化前 `h3_music.setup`；
  loss.py 调 `set_from_inputs`。缓存→MERT 映射 `h3ar/build_music_map.py`（N4 f0_cur 2091/2091 有 MERT）。注入器权重随 LoRA 一起存（`od_music.*`），采样器自动识别。
- 训练：H-A2 同配方（DiffSynth DeCFG 适配器融合底座、lr 5e-5、LoRA r64），+ 6.56M 注入器参数；N4 精选 f0_cur ×3 → 785 步，约 59 s/步。
  输出 `models/train/H3M1a_mert_decfgDS_lr5e-5_cur`；N2 自动评测 step 100/300/500/784（双向，官方 shift，joint+aswap 前 6 条），网页预设"【M1a 音乐注入】"。
- 对照：H-A2 主线同步数（注意数据量不同：H-A2 用 12k，M1a 用 2.1k）。判据：aswap 动作是否随新音乐变化（姿态距离、G2 判别器差值为正）+ 人工音乐同步分；画质不得下降。

## 2026-10-11 00:10 M2 多 encoder MoE 启动准备（N2）
- 用户：AR 只需验证 N1（H-B2base）可行即可 → N2 的 H-B2pre 已停；N2 改跑 M2。
- 5 个专家（`h3ar/h3_music.py: MusicMoE`）：MERT 低层(0–12，可学层权重) / MERT 高层(13–24) / 节拍(onset、节拍脉冲、节拍相位 sin/cos、RMS、tempo，40 Hz) /
  H3 自带音频 VAE latent（音乐自身，40 Hz） / CLAP 全局 embedding（laion/clap-htsat-unfused，曲风情绪，广播到每个时间步）。
  各自投到 1024 维 → LayerNorm → 逐时间步 softmax router（零初始化 = 起始均匀）→ 加权和 → 零初始化 MLP → 加到目标音频 token。
  训练时 expert dropout 0.1；负载均衡项 0.01·5·Σ ḡ²（均匀时=1）；日志每 20 步打印各专家平均门控（[OD H3-MOE] gate ...），可直接看"哪个 encoder 被用"。
- 特征：`h3ar/m2_feats.py --phase beat|clap` → `diff/data/omnidance/audio_cache/m2_feats/{beat,clap}/<id>.npy`（N2 训练 7,086 条 + val 音频）。
  坑：librosa beat_track（numba）在 venv_h3 里段错误 → 节拍用 /opt/venv_od 跑；CLAP 在 venv_h3（transformers 5 的 get_audio_features 返回 ModelOutput，取 pooler_output）。
- 训练：H-A2 配方 + MoE，N2 f1_cur 7.1k，~886 步，`models/train/H3M2_moe5_decfgDS_lr5e-5_f1cur`；N5/N3 评测，网页"【音乐控舞 idea】"。
