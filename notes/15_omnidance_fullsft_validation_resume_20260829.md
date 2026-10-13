# 15. OmniDance fullSFT Checkpoint 验证与断点续训报告(2026-08-29)

> 背景:2026-08-28 18:37 stage3 训练中断于 step 5203/12000(会话被杀,无 error)。
> 此前人工抽测 stage3 早期 checkpoint(step-2000/3000/4500)出现"纯色/色块画面"(蓝闪/橙/绿方块),疑似训练崩溃。
> 本报告记录全量 checkpoint 自动化验证结论、问题定位、以及断点续训方案。

## 一、结论速览

1. **全部 12 个 checkpoint(S1-6000、S2-6000、S3-500~5000)在标准推理配置下生成视频全部正常** —— 无纯色、无色偏、有真实运动。
2. NaN/Inf 扫描全部干净(990 keys/ckpt,max_abs ~7.7 稳定)。
3. **"纯色画面"未能在本地复现,问题大概率出在此前人工测试的推理环境/配置,而非训练本身**:
   - 最可能原因:从 HF 下载的 checkpoint 旁缺少 `args.json`。推理脚本依赖 ckpt 同目录的
     `args.json` 自动恢复 `enable_global_audio_modulation=true` 与 `audio_sink_bias_mode=pass_mass`,
     缺省会 fallback 到 `legacy`,行为不匹配。
   - 次要可能:cfg_mode / 分辨率 / 帧数等推理参数与训练分布不一致。
4. Stage3 已于 2026-08-29 从 step-5000 权重断点续训(新输出目录),预计 ~60h 完成。

## 二、验证方法

### 2.1 测试集

- `OmniDance/part049_val.csv`(47 条),取前 3 条:样本 `wJPptuhclZmKAEWs` / `wJPrpFsThWAcBQIw` / `wJQTclrlOLEYYfTB`
- 路径重映射:`/data/raw/nzc/OmniDance/tar/...` → 本机 `OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/`

### 2.2 推理配置(与 inference_bundle probe 工具链一致)

```
脚本: examples/wanvideo/model_inference/Wan2.2-TI2V-5B-Audio.py
包装: inference_bundle/toolkit/run_batch.py
ckpt_type=onlyCKPT, cfg_mode=text_first, cfg_scale=2.0, audio_scale=2.0
audio_mode=real(另测 none 作隔离对照), text_mode=original
num_frames=81, num_inference_steps=50, seed=42, 704x1280, save_with_audio
```

复现脚本:`logs_omnidance/valtest/run_valtest_queue.sh`(幂等,支持队列;8 卡并行)。

### 2.3 自动化指标(`logs_omnidance/valtest/analyze_valtest.py`)

| 指标 | 含义 | 纯色/坏视频表现 | 正常表现 |
|---|---|---|---|
| flat_ratio | 空间 std<8 的帧占比 | →1.0 | 0 |
| color_cast | max/min 通道均值比 | >3(色偏) | ~1.0 |
| motion | 帧间像素差 | ~0(冻结) | 3~18 |
| ffPSNR | 首帧 vs 输入图 | <12 | ~37 |
| gtPSNR | 与 GT 视频逐帧 PSNR | 低 | ~19 |

## 三、验证结果

### 3.1 指标汇总(每配置 3 样本)

| tag | OK | flat | std | cast | motion | ffPSNR | gtPSNR |
|---|---|---|---|---|---|---|---|
| S1-6000_none | 3/3 | 0.00 | 51.7 | 1.0 | 3.7 | 37.7 | 19.6 |
| S2-6000_real | 3/3 | 0.00 | 51.4 | 1.0 | 3.8 | 37.7 | 19.5 |
| S3-500_real | 3/3 | 0.00 | 51.3 | 1.0 | 3.8 | 37.7 | 19.6 |
| S3-1000_real | 3/3 | 0.00 | 51.5 | 1.0 | 3.9 | 37.7 | 19.5 |
| S3-1500_real | 3/3 | 0.00 | 51.1 | 1.0 | 3.8 | 37.7 | 19.6 |
| S3-2000_real | 3/3 | 0.00 | 51.0 | 1.0 | 5.1 | 37.7 | 19.5 |
| S3-2500_real | 3/3 | 0.00 | 51.3 | 1.0 | 3.8 | 37.7 | 19.5 |
| S3-3000_real | 3/3 | 0.00 | 51.3 | 1.0 | 5.7 | 37.7 | 19.6 |
| S3-3500_real | 3/3 | 0.00 | 51.5 | 1.0 | 4.1 | 37.7 | 19.5 |
| S3-4000_real | 3/3 | 0.00 | 51.2 | 1.0 | 4.2 | 37.7 | 19.5 |
| S3-4500_real | 3/3 | 0.00 | 51.5 | 1.0 | 3.8 | 37.7 | 19.8 |
| S3-4500_none | 3/3 | 0.00 | 51.5 | 1.0 | 3.5 | 37.7 | 19.8 |
| S3-5000_real | 3/3 | 0.00 | 51.1 | 1.0 | 4.0 | 37.7 | 19.6 |

### 3.2 交叉证据

- **不同 checkpoint 同一样本视频 md5 全部不同** → checkpoint 确实加载并各自生效(排除"没加载/加载同一个"假说)。
- checkpoint 加载日志:`[load] audio-like dense subset: loaded 125 keys (missing=865, unexpected=0)` + `remaining dense subset: loaded 865 keys` → 990 keys 全量命中,0 unexpected。
- 目视抽帧(S3-4500 第 0/40/80 帧):画面为清晰完整的舞者,帧间姿势有变化,运动量(diff 12.8~16.7)与 GT(13.5~17.8)同量级。

### 3.3 训练侧佐证

- 三阶段 loss 平稳:S1 0.27→0.21,S2 ~0.22,S3 ~0.21-0.24;grad_norm 稳定 0.05~0.09。
- 中断前最后一步(5203)loss 0.49、grad_norm 0.076,无发散迹象。
- 注:loss 曲线整体平坦、下降缓慢,是本配置(full SFT, lr=1e-5, 混合 condition)的固有表现,非崩溃信号。

## 四、断点续训方案

### 4.1 关键坑(重要)

- `train.py` 的步数计数器从 0 开始、**无 resume 步数偏移**;`save_steps=500` 会从 step-500 重新保存。
- 因此**绝不能把续训输出写回原 stage3 目录**,否则会覆盖现有 step-500~5000 checkpoint 并截断 metrics.jsonl。

### 4.2 实施配置

- 新输出目录:`models/train/OSS_fullSFT_stage3_resume_from5k`
- warm-start:`--train_checkpoint_path <S3>/step-5000.safetensors`(仅权重,无 optimizer state)
- 步数映射:本地 1~6800 ≈ 原始 5204~12000(override 配置 `fullSFT_stage3_resume.yaml`,后加载覆盖前)
- LR:warmup 300 步(保护新 optimizer),decay 4800→6800 对齐原 10000→12000,min_factor 0.2
- 启动脚本:`scripts_omnidance/resume_stage3.sh`(幂等;setsid 脱离会话)
- 日志:`logs_omnidance/s3_resume.log` + 输出目录 `train.log`

### 4.3 启动后状态(2026-08-29 16:20 左右启动)

- step≈361,loss 0.05~0.33(与中断前同水平 → warm-start 生效),grad_norm ~0.06
- ~32s/step,8×H20 满载,预计 **~60h** 完成 6800 步
- HF 上传守护已重启(30min/轮,自动上传新 checkpoint 到 `NZC415/m2d`)

## 五、复现清单

```bash
# 1. 全 checkpoint 验证(8 卡并行,约 6h)
bash logs_omnidance/valtest/run_valtest_queue.sh <gpu> "<tag>:<ckpt_path>:<audio_mode>" ...

# 2. 指标分析
python3 logs_omnidance/valtest/analyze_valtest.py   # → analysis_summary.csv

# 3. 续训(幂等)
bash scripts_omnidance/resume_stage3.sh
```

## 六、遗留事项

1. **HF 下载侧缺 args.json 问题**:需向 HF repo 补传各 stage 目录的 `args.json` / `training_config.yaml`(本地 `scripts_omnidance/upload_ckpts_hf.py` 已支持辅助文件 md5 增量上传)。
2. 人工测试"纯色画面"的确切原因未最终定位(本地无法复现);建议用 §2.2 配置重测,重点核对 args.json 是否随 ckpt 一起下载。
3. kill 训练进程时避免使用 `pkill -f "train.py"` 这类宽匹配(会误杀 `dummy_train.py` 等无关进程)。
4. cephfs 并发大文件读极慢(8 进程并发加载模型 ~1h);多卡推理建议错峰启动。
