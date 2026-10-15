#!/usr/bin/env python3
"""OmniDance 人工评审 demo 服务（仅标准库）。

用法:  python3 server.py --port 8765 [--rebuild]
访问:  http://<本机IP>:8765/   或 code-server 代理 http://<本机IP>:8156/proxy/8765/

- 自动扫描 logs_omnidance/exp 下所有 {joint,aswap,tswap,audio_only,text_only} 推理结果目录 + E9 长视频
- 视频/音频经 /media 提供（支持 Range，可拖动），只允许访问 EXP 与 OmniDance_processed
- 评审结果写入 logs_omnidance/exp/review_web/reviews/<评审人>.jsonl（追加，取最新）
"""
import argparse, csv, gzip, json, os, re, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/apdcephfs_gy4/share_302533218/cedricnie"
EXP = f"{ROOT}/diff/logs_omnidance/exp"
DATA = f"{ROOT}/OmniDance_processed"
OUT = f"{EXP}/review_web"
REVIEWS = f"{OUT}/reviews"
MANIFEST = f"{OUT}/manifest.json"
ALLOWED = [os.path.realpath(EXP), os.path.realpath(DATA)]
CONDS = ["joint", "aswap", "tswap", "audio_only", "text_only"]
SKIP = {"review", "review_web", "logs", "kps", "shards", "r0", "r0b", "r1", "r1v2", "__pycache__"}
ID_RE = re.compile(r"wJ[A-Za-z]{14}")
METRIC_KEYS = ["bas", "bas_alt", "ta", "ta_alt", "id_dino", "motion", "g2", "g2_alt"]
CTYPE = {".mp4": "video/mp4", ".wav": "audio/wav", ".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json"}

# 模型说明（label 匹配前缀即可）
DESC = {
    "E3_h3/H3-Ref2VA": "原始 H3（33B 双向，零样本 Ref2VA，原长 prompt）",
    "E3_h3/out_retake": "原始 H3 retake（音乐作干净对齐音轨）",
    "E3_h3/out_seed43": "原始 H3 seed43",
    "H3AR/val_zs_bi_motion": "原始 H3 零样本双向，动作 prompt（H3-AR 基线）",
    "H3AR/val_zs_ar_motion": "原始 H3 零样本 + 块因果 AR 采样（未训练，AR 下限）",
    "H3AR/val_HA200_bi": "H-A 双向 H3 LoRA step-200（干净音乐条件）",
    "H3AR/val_HA676_bi": "H-A 双向 H3 LoRA 最终 step-676",
    "H3AR/val_HB0mix500_ar": "H-B0 混训 AR 老师 step-500（原始 H3 起训，精选数据，逐块采样）",
    "H3AR/val_HBmix100_ar": "H-B 混训 AR 老师 step-100（H-A 起训，精选数据，逐块采样）",
    "H3AR/val_HB0mix700_ar": "H-B0 混训 AR 老师 step-700（原始 H3 起训，精选数据，逐块采样）",
    "H3AR/val_HBmix300_ar": "H-B 混训 AR 老师 step-300（H-A 起训，精选数据，逐块采样）",
    "H3AR/val_HCsmoke100_cm4": "H-C 冒烟 step-100：4 步/块一致性采样（老师 H-B0 step-600）",
    "H3AR/val_HBmix500_ar": "H-B 最终 step-500（逐块 AR，20 步/块）",
    "H3AR/val_HB0mix1046_ar": "H-B0 最终 step-1046（逐块 AR，20 步/块）",
    "H3AR/val_zs_bi_musicprompt": "原始 H3 双向 + prompt 写入 BPM/节拍点（idea M1b）",
    "H3AR/diag_zs_bi_s12n50_motion": "诊断：原始 H3 双向，官方 shift12/3、50 步，动作 prompt",
    "H3AR/diag_zs_bi_s12n50_orig": "诊断：原始 H3 双向，官方 shift12/3、50 步，原 caption",
    "H3AR/diag_zs_bi_s12n20_motion": "诊断：原始 H3 双向，shift12、20 步",
    "H3AR/diag_HA676_bi_s12n50_motion": "诊断：H-A676 双向，官方 shift、50 步，cfg1",
    "H3AR/diag_HA676_bi_s12n50_cfg3": "诊断：H-A676 双向，官方 shift、50 步，真 CFG=3（验证 CFG 蒸馏丢失）",
    "H3AR/diag_HB500_ar_s12n20_motion": "诊断：H-B500 逐块 AR，shift12、20 步/块，cfg1",
    "H3AR/diag_HB500_ar_s12n20_cfg3": "诊断：H-B500 逐块 AR，shift12、20 步/块，真 CFG=3",
    "H3AR/diag_zs_ar_s12n20_motion": "诊断：原始 H3 未训练逐块 AR，shift12",
    "H3AR/val_zsbase_bi_s12n50": "原始 H3 双向零样本（官方 shift12/3、50 步）——H-A2 的对照",
    "H3AR/val_HA2ds": "H-A2 双向（修正 CFG：训练时融合 DiffSynth DeCFG 适配器，推理丢弃），lr5e-5，精选 12k",
    "H3AR/val_HA2dslr1e-4_": "H-A2 双向 lr1e-4 消融（DiffSynth 适配器，N1 精选 5.6k）",
    "H3AR/val_HB2pre": "H-B2 预跑：修正 CFG 的 AR 老师（TF+DF 混训，从 H-A2 step-200 起，N1 精选 5.6k，逐块 AR shift12 20 步/块）",
    "H3AR/val_M1a": "M1a：H-A2 配方 + MERT 音乐特征零初始化注入（目标音频 token），N4 精选 2.1k",
    "H3AR/val_W1b": "W1b：音乐对比 DPO（正=原音乐，负=别的音乐/错位音乐）+ SFT，修正 CFG 底座，N6 精选 3.6k",
    "H3AR/val_M2b": "M2b：M2 + 音乐 MoE 单独 lr 1e-3（M2 的 router 在 lr 5e-5 下不学）",
    "H3AR/val_M2": "M2：多 encoder MoE 音乐融合（MERT 低/高层、节拍、H3 音频 VAE、CLAP）+ H-A2 配方，N2 精选 7.1k",
    "H3AR/val_HB2base": "H-B2base：修正 CFG 的 AR 老师，直接从原始 H3 起（TF+DF 混训，N1 精选 5.6k），逐块 AR 20 步/块",
    "H3AR/val_HC2": "H-C2：CF++ 阶段 2 的 4 步 AR 学生（因果一致性蒸馏，老师 H-B2base step-200），每块 4 步",
    "H3AR/val_HB2base400_arW": "AR 诊断：H-B2base400 推理时保留去蒸馏适配器（与训练底座一致）",
    "H3AR/val_HB2base400_arcfg3": "AR 诊断：H-B2base400 + 真 CFG=3",
    "H3AR/val_HA2ds400_ar": "AR 诊断：双向 H-A2 step-400 直接逐块 AR（没有 AR 训练）",
    "H3AR/val_falReal1_bifal": "社区 fal Realism-People LoRA（原始 H3 + LoRA，触发词 r34l1sm，双向）",
    "H3AR/val_falHA2_400_bifal": "fal Realism LoRA 叠加 H-A2 step-400（双向，触发词 r34l1sm）",
    "H3AR/val_HB2_": "H-B2 正式 AR 老师：从最终 H-A2 step-750 起，TF+DF 混训，精选 full12k（N7+N8），逐块 AR",
    "H3AR/val_HB2base708_arcfg3": "AR：H-B2base 最终版 + 真 CFG=3",
    "H3AR/val_HB3cfg": "H-B3：AR 修复版（原始蒸馏底座 + inverse-CFG guidance loss g=4，其余同 H-B2base）",
    "H3AR/val_HA3cfg": "H-A3 双向：原始蒸馏底座 + inverse-CFG guidance loss g=4（DiffSynth 官方方案），full12k，lr5e-5",
    "H3AR/val_zsbase_bihr": "原始 H3 双向，官方分辨率 768×1344（手部诊断）",
    "H3AR/val_HA2ds750_bihr": "H-A2 750 双向，推理用 768×1344（训练是 480×832；手部诊断）",
    "H3AR/val_HA2ds750_biorig": "H-A2 750（清单 caption 训练）+ 富文本 prompt 采样（E3 同款 prompt）",
    "H3AR/val_HA4rich": "H-A4：富文本 prompt 训练（E3 同款），DiffSynth 适配器修正 CFG，full12k；富文本 prompt 采样",
    "H3AR/val_HA3cfg750_biorig": "H-A3 750 + 富文本 prompt 采样",
    "H3AR/val_zsbase_biorig": "原始 H3 + 我们的采样器 + 富文本 prompt（与 E3 同 prompt，验证采样器无问题）",
    "H3AR/val_zsbase_biospeech": "音频探针：原始 H3，音乐替换为英文朗读语音（LibriSpeech），富文本 prompt",
    "H3AR/val_zsbase_biosilence": "音频探针：原始 H3，音乐替换为静音",
    "H3AR/val_zsbase_bionoise": "音频探针：原始 H3，音乐替换为白噪声",
    "H3AR/val_HA4rich300_biospeech": "音频探针：H-A4 300，语音替换音乐",
    "H3AR/val_HA4rich300_biosilence": "音频探针：H-A4 300，静音替换音乐",
    "H3AR/val_HA4rich300_biorigcfg3": "H-A4 300 + 真 CFG=3（检验 CFG 蒸馏残余丢失）",
    "H3AR/val_HA2os": "H-A2 双向（修正 CFG：训练时融合 ostris v3 适配器，推理丢弃），lr5e-5，精选 2k",
    "H3AR/val_HYB25base": "开源b25–49规则重建的Hybrid BF16，未做舞蹈SFT；训练/推理都保留此底座",
    "H3AR/val_HYB25G": "Hybrid BF16 + direct guidance3.5 双向舞蹈LoRA；同精选集、全局batch8、首阶段300步",
    "H3AR/val_OS1G": "Ref2VA + ostris-v1训练助手 + direct guidance3.5；推理不带助手，同精选集、全局batch8、首阶段300步",
    "H3AR/val_REF35": "纯原始Ref2VA + direct guidance3.5（无训练助手）；同精选集/globalbatch8/300步",
    "H3AR/val_OS1FM": "Ref2VA + ostris-v1训练助手 + 普通FM（guidance1）；同精选集/globalbatch8/300步",
    "H3AR/val_DSFM": "Ref2VA + DiffSynth训练助手 + 普通FM；同精选集/4GPU×累积2/globalbatch8/300步，采样批次顺序与8GPU组可能不同",
    "H3AR/val_DSFM8": "Ref2VA + DiffSynth训练助手 + 普通FM；同2091精选集/8GPU/globalbatch8/seed42/300更新",
    "H3AR/val_DSG35": "Ref2VA + DiffSynth训练助手 + direct guidance3.5；与DSFM8单变量对照，同2091精选集/300更新",
    "H3AR/val_HYB25FM": "Hybrid BF16 + 普通FM；与Hybrid guidance3.5单变量对照，同2091精选集/300更新",
    "H3AR/val_REFFM": "原Ref2VA + 普通FM，无训练助手；同2091精选集/300更新",
    "H3AR/val_HRFM512": "704×1280，原始Ref2VA底座，普通FM；512样本(303精选+209)/repeat5/seed42/lr5e-5——与HRANC512只差训练目标",
    "H3AR/val_HRANC512": "704×1280，原始Ref2VA底座，锚定σ≥0.5；与HRFM512完全相同数据/底座/seed",
    "H3AR/val_zsbase_bi768orig": "原始H3 + 富文本，768×1344（H3官方默认分辨率）推理",
    "H3AR/val_CANHI05300_bi768orig": "锚定σ≥0.5 300（480p训练）在 768×1344 推理",
    "H3AR/val_MCFG1_base": "原始H3 + 推理音乐引导 w=1：v + w·(v(音乐) − v(同一音乐错位50%))；推理多一次前向",
    "H3AR/val_MCFG2_base": "原始H3 + 推理音乐引导 w=2",
    "H3AR/val_MCFG1_CANHI05300": "锚定σ≥0.5 300（2091）+ 推理音乐引导 w=1",
    "H3AR/val_ANCMCFG1": "训练蒸馏音乐引导：σ≥0.5 目标=原始H3的音乐引导速度(w=1)，σ<0.5 普通FM；lr1e-4/2091/300步；推理不再需要引导",
    "H3AR/val_DPOANC1K": "音乐对比DPO(β=1000)+SFT项锚定原始H3(σ≥0.5)；2091精选/富文本/300步",
    "H3AR/val_DPOANC5K": "音乐对比DPO(β=5000)+锚定(σ≥0.5)；2091精选/富文本/300步",
    "H3AR/val_CANHI05LR": "锚定σ≥0.5 + lr 1e-4（锚定下能否用更大lr加快适配）；2091精选/300步",
    "H3AR/val_DUPANC": "480×832 无插帧(dup) + 锚定σ≥0.5；full12k dup数据/400步（与HRANC同预算）",
    "H3AR/val_HRANC": "704×1280 原生分辨率 + 锚定σ≥0.5；full12k hr数据/400步",
    "H3AR/val_zsbase_bihrorig": "原始H3 + 富文本，704×1280 推理（高分辨率基线）",
    "H3AR/val_CANHI05": "锚定σ>=0.5，2091精选全集，900步（300/600/900）",
    "H3AR/val_CANADD5": "FM+5×锚定，2091精选全集，900步（300/600/900）",
    "H3AR/val_ANCHI07": "锚定：sigma>=0.7 的训练步目标=冻结原始H3预测（同x_t/t/条件），sigma<0.7 普通FM；GT512/300步",
    "H3AR/val_ANCHI05": "锚定：sigma>=0.5 锚定原始H3，sigma<0.5 普通FM；GT512/300步",
    "H3AR/val_ANCADD1": "锚定：全时间步 FM + 1×||v-v_base||²；GT512/300步",
    "H3AR/val_ANCADD5": "锚定：全时间步 FM + 5×||v-v_base||²；GT512/300步",
    "H3AR/val_RC_SG512300_L0-42": "根因：SG512 300 LoRA 只在前42步（sigma>0.7，高噪声）生效，后8步用原始H3",
    "H3AR/val_RC_SG512300_L42-50": "根因：SG512 300 LoRA 只在最后8步（sigma<0.7，低噪声）生效，前42步用原始H3",
    "H3AR/val_RC_GT512300_L42-50": "根因：GT512 300 LoRA 只在最后8步（低噪声）生效",
    "H3AR/val_RC_SG512300_n100": "根因：SG512 300，100步采样（检验轨迹弯曲/离散误差）",
    "H3AR/val_RC_GT512300_n100": "根因：GT512 300，100步采样",
    "H3AR/val_DIAG_DSbase_cfg3": "诊断：DiffSynth去蒸馏底座，无LoRA，推理真CFG3（负prompt空格）——去蒸馏底座+CFG本身是否清晰",
    "H3AR/val_DIAG_DSFM8300_keepDS_cfg3": "诊断：DSFM8 300 LoRA + 推理保留去蒸馏底座 + 真CFG3（训练/推理底座一致的去蒸馏方案）",
    "H3AR/val_DIAG_REFFM300_cfg3": "诊断：REFFM 300（原Ref2VA+普通FM）+ 推理真CFG3",
    "H3AR/val_DIAG_SG512300_cfg3": "诊断：SG512 300（自生成数据训练）+ 推理真CFG3",
    "H3AR/val_GT512C": "GT512 从300续训到600（tag里的300=累计600步）——看更久是否继续变糊",
    "H3AR/val_SG512C": "SG512 从300续训到600（累计600步）——自生成数据训练更久是否仍保持",
    "H3AR/val_GT512LR": "GT512 单变量：lr 1e-5（其余同GT512）——检验漂移速度",
    "H3AR/val_GT512HI": "GT512 单变量：只在高噪声半段训练（时间步索引[0,0.5)，sigma 1→0.69）——只学结构/运动，细节留给底座",
    "H3AR/val_GT512LO": "GT512 单变量：只在低噪声半段训练（sigma 0.69→0）——检验细节变糊是否来自低噪声段拟合",
    "H3AR/val_GT512R16": "GT512 单变量：LoRA rank 16（其余同GT512）——检验容量",
    "H3AR/val_GT512": "归因对照：原Ref2VA+普通FM，512个精选ID的真实视频latent（=REFFM配方，repeat5/300更新）",
    "H3AR/val_SG512": "归因对照：同512 ID，但训练目标换成原始H3自己生成的视频（cfg1/shift12/50步）；其余与GT512完全相同",
    "H3AR/val_REFINV35": "原Ref2VA + inverse guidance3.5；同direct3.5目标但损失/梯度缩放不同，同2091精选集/300更新",
    "E2_S4final/S4-final": "Wan2.2-5B 最终版 S4-final（50 步老师）",
    "E2/S4-11000": "Wan2.2 S4-11000 704p 老师",
    "E2_teacher480": "Wan2.2 S4-11000 480p 老师",
    "E2_S4r3000": "Wan2.2 S4r-3000",
    "E6/E6_3b": "Wan2.2 AR 学生（Self-Forcing DMD，from 2c3000）",
    "E6/E6_3a": "Wan2.2 AR 学生（Self-Forcing DMD，from 2a1500）",
    "E6/E6_3c": "Wan2.2 AR 学生（Self-Forcing DMD lr5e-6）",
    "E6/E6_2": "Wan2.2 AR 因果 ODE 初始化",
    "E5/E5b5": "Wan2.2 DMD 双向学生 704p（最佳 b5）",
    "E5/": "Wan2.2 DMD 双向学生",
    "E7": "E7 自蒸馏（已停）",
    "E9": "E9 长视频（33s）",
    "E10": "E10 stage5 / CFG / 音频强度消融",
    "E11": "E11 参考帧/增强消融",
}

PRESETS = [
    {"name": "【10-15 文本×音乐】原始H3 704p：空prompt/H3模板+通用句/富文本 × 有音乐/静音（看文本与音乐各自驱动什么）", "cond": "joint", "view": "models",
     "models": ["H3AR/val_zsbase_bihrempty", "H3AR/val_zsbase_bihrempty_silence", "H3AR/val_zsbase_bihrgeneric", "H3AR/val_zsbase_bihrgeneric_silence", "H3AR/val_zsbase_bihrorig", "H3AR/val_zsbase_bihrorig_silence"]},
    {"name": "【10-15 文本控幅度/音乐→文本】原始H3 704p 不训练：平静文本 / 夸张文本 / 通用句 / 富文本 / 按音乐逐秒强度+节拍生成的文本（看文本能否控制动作幅度、音乐写进文本能否跟拍）", "cond": "joint", "view": "models",
     "models": ["H3AR/val_zsbase_bihrcalm", "H3AR/val_zsbase_bihrenergetic", "H3AR/val_zsbase_bihrgeneric", "H3AR/val_zsbase_bihrorig", "H3AR/val_zsbase_bihrtimed"]},
    {"name": "【10-15 配对噪声】同SG512目标/同配方：普通FM(随机噪声) vs 配对噪声(生成该目标的那个噪声)，480p", "cond": "joint", "view": "models",
     "models": ["H3AR/val_zsbase_biorig", "H3AR/val_SG512100_biorig", "H3AR/val_SG512300_biorig", "H3AR/val_SG512P100_biorig", "H3AR/val_SG512P300_biorig"]},
    {"name": "【10-15 704p 噪声段切换】HA5hr300（704p普通FM）LoRA 全程 / 仅高噪声σ≥0.5(0:47步) / 仅低噪声(47:50) / 细分 0:10(σ1→.98) 10:30(σ.98→.89) 30:47(σ.89→.5) vs 原始H3：糊来自哪段", "cond": "joint", "view": "models",
     "models": ["H3AR/val_zsbase_bihrorig", "H3AR/val_HA5hr300_bihrorig", "H3AR/val_HA5hr300hi_bihrorig", "H3AR/val_HA5hr300lo_bihrorig", "H3AR/val_HA5hr300s0-10_bihrorig", "H3AR/val_HA5hr300s10-30_bihrorig", "H3AR/val_HA5hr300s30-47_bihrorig"]},
    {"name": "【10-15 704p 预览】480p训练的同512 FM300 / 锚定300 用704推理（训练推理分辨率不一致，仅参考）", "cond": "joint", "view": "models",
     "models": ["H3AR/val_zsbase_bihrorig", "H3AR/val_GT512300_bihrorig", "H3AR/val_ANCHI05300_bihrorig"]},
    {"name": "【10-15 704p 单变量】同512样本/同原始底座/同seed：普通FM vs 锚定（只差训练目标）", "cond": "joint", "view": "models",
     "models": ["H3AR/val_zsbase_bihrorig", "H3AR/val_HRFM512100_bihrorig", "H3AR/val_HRFM512300_bihrorig",
                "H3AR/val_HRANC512100_bihrorig", "H3AR/val_HRANC512300_bihrorig", "H3AR/val_HA5hr300_bihrorig", "H3AR/val_HRANC100_bihrorig", "H3AR/val_HRANC300_bihrorig"]},
    {"name": "【10-14 分辨率】同模型 480×832 / 704×1280 / 768×1344：原始H3 与 锚定300（看手脸是否随分辨率变清楚）", "cond": "joint", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "H3AR/val_zsbase_biorig", "H3AR/val_zsbase_bihrorig", "H3AR/val_zsbase_bi768orig",
                "H3AR/val_CANHI05300_biorig", "H3AR/val_CANHI05300_bi768orig", "H3AR/val_HA5hr300_bihrorig",
                "H3AR/val_HRANC100_bihrorig", "H3AR/val_HRANC200_bihrorig", "H3AR/val_HRANC400_bihrorig"]},
    {"name": "【10-14 音乐引导】推理音乐引导 / 训练中蒸馏音乐引导（锚定）——切 aswap 看换音乐后的动作", "cond": "aswap", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "H3AR/val_zsbase_biorig", "H3AR/val_MCFG1_base_biorig", "H3AR/val_MCFG2_base_biorig",
                "H3AR/val_CANHI05300_biorig", "H3AR/val_MCFG1_CANHI05300_biorig",
                "H3AR/val_ANCMCFG1100_biorig", "H3AR/val_ANCMCFG1200_biorig", "H3AR/val_ANCMCFG1300_biorig",
                "H3AR/val_DPOANC1K300_biorig", "H3AR/val_DPOANC5K300_biorig"]},
    {"name": "【10-14 锚定+音乐】锚定+音乐对比DPO / 高lr / 704×1280 锚定（同3case，joint 与 aswap 都要看）", "cond": "aswap", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "H3AR/val_zsbase_biorig", "H3AR/val_W1b446_bi", "H3AR/val_ANCHI05300_biorig",
                "H3AR/val_DPOANC1K100_biorig", "H3AR/val_DPOANC1K300_biorig", "H3AR/val_DPOANC5K100_biorig", "H3AR/val_DPOANC5K300_biorig",
                "H3AR/val_CANHI05LR100_biorig", "H3AR/val_CANHI05LR300_biorig",
                "H3AR/val_zsbase_bihrorig", "H3AR/val_HRANC100_bihrorig", "H3AR/val_HRANC200_bihrorig", "H3AR/val_HRANC400_bihrorig",
                "H3AR/val_DUPANC100_biorig", "H3AR/val_DUPANC200_biorig", "H3AR/val_DUPANC400_biorig"]},
    {"name": "【10-14 锚定训练】高噪声段锚定原始H3 vs 普通FM（GT512同数据，同3case）", "cond": "joint", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "H3AR/val_zsbase_biorig", "H3AR/val_GT512100_biorig", "H3AR/val_GT512300_biorig",
                "H3AR/val_ANCHI07100_biorig", "H3AR/val_ANCHI07300_biorig", "H3AR/val_ANCHI05100_biorig", "H3AR/val_ANCHI05300_biorig",
                "H3AR/val_ANCADD1100_biorig", "H3AR/val_ANCADD1300_biorig", "H3AR/val_ANCADD5100_biorig", "H3AR/val_ANCADD5300_biorig",
                "H3AR/val_CANHI05300_biorig", "H3AR/val_CANHI05600_biorig", "H3AR/val_CANHI05900_biorig",
                "H3AR/val_CANADD5300_biorig", "H3AR/val_CANADD5600_biorig", "H3AR/val_CANADD5900_biorig"]},
    {"name": "【10-14 根因】LoRA只在高噪声步/只在低噪声步生效、100步采样（同3case joint）", "cond": "joint", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "H3AR/val_zsbase_biorig", "H3AR/val_SG512300_biorig", "H3AR/val_RC_SG512300_L0-42_biorig",
                "H3AR/val_RC_SG512300_L42-50_biorig", "H3AR/val_GT512300_biorig", "H3AR/val_RC_GT512300_L42-50_biorig",
                "H3AR/val_RC_SG512300_n100_biorig", "H3AR/val_RC_GT512300_n100_biorig"]},
    {"name": "【10-14 CFG推理诊断】微调后推理开真CFG3 / 保留去蒸馏底座（同3case）", "cond": "joint", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "H3AR/val_zsbase_biorig", "H3AR/val_DIAG_DSbase_cfg3_biorig",
                "H3AR/val_DSFM8300_biorig", "H3AR/val_DIAG_DSFM8300_keepDS_cfg3_biorig",
                "H3AR/val_REFFM300_biorig", "H3AR/val_DIAG_REFFM300_cfg3_biorig",
                "H3AR/val_SG512300_biorig", "H3AR/val_DIAG_SG512300_cfg3_biorig",
                "H3AR/val_HA4rich300_biorig", "H3AR/val_HA4rich300_biorigcfg3"]},
    {"name": "【10-13 归因】H3自生成数据(SG512) vs 真实数据(GT512)，同512 ID同配方（原Ref2VA+普通FM）", "cond": "joint", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "H3AR/val_zsbase_biorig", "H3AR/val_GT512100_biorig", "H3AR/val_GT512300_biorig",
                "H3AR/val_SG512100_biorig", "H3AR/val_SG512300_biorig", "H3AR/val_GT512C300_biorig", "H3AR/val_SG512C300_biorig",
                "H3AR/val_GT512LR100_biorig", "H3AR/val_GT512LR300_biorig", "H3AR/val_GT512HI100_biorig", "H3AR/val_GT512HI300_biorig",
                "H3AR/val_GT512LO100_biorig", "H3AR/val_GT512LO300_biorig", "H3AR/val_GT512R16100_biorig", "H3AR/val_GT512R16300_biorig",
                "H3AR/val_REFFM100_biorig", "H3AR/val_REFFM300_biorig",
                "H3AR/val_DSFM8100_biorig", "H3AR/val_DSFM8300_biorig", "H3AR/val_HA6dup500_biorig", "H3AR/val_HA6dup1499_biorig",
                "H3AR/val_HA5hr500_bihrorig", "H3AR/val_M1abR708_biorig", "H3AR/val_M2bR886_biorig"]},
    {"name": "【Hybrid / 社区CFG】双向舞蹈：同精选集3case（H-A4数据预算不同，仅参考）", "cond": "joint", "view": "models",
     "models": ["H3AR/val_zsbase_biorig", "H3AR/val_HA4rich300_biorig", "H3AR/val_HYB25base_biorighyb",
                "H3AR/val_HYB25G100_biorighyb", "H3AR/val_HYB25G300_biorighyb", "H3AR/val_OS1G100_biorig", "H3AR/val_OS1G300_biorig",
                "H3AR/val_REF35100_biorig", "H3AR/val_REF35300_biorig", "H3AR/val_OS1FM100_biorig", "H3AR/val_OS1FM300_biorig",
                "H3AR/val_DSFM8100_biorig", "H3AR/val_DSFM8300_biorig", "H3AR/val_DSG35100_biorig", "H3AR/val_DSG35300_biorig",
                "H3AR/val_HYB25FM100_biorighyb", "H3AR/val_HYB25FM300_biorighyb", "H3AR/val_REFFM100_biorig", "H3AR/val_REFFM300_biorig",
                "H3AR/val_REFINV35100_biorig", "H3AR/val_REFINV35300_biorig"]},
    {"name": "【H-A5hr / H-A6dup】高分辨率 / 无插帧（100 / 300 / 500；后续固定 3 case）", "cond": "joint", "view": "models",
     "models": ["H3AR/val_zsbase_biorig", "H3AR/val_HA4rich100_biorig", "H3AR/val_HA4rich300_biorig", "H3AR/val_HA4rich500_biorig",
                "H3AR/val_HA5hr100_bihrorig", "H3AR/val_HA5hr300_bihrorig", "H3AR/val_HA5hr500_bihrorig",
                "H3AR/val_HA6dup100_biorig", "H3AR/val_HA6dup300_biorig", "H3AR/val_HA6dup500_biorig"]},
    {"name": "【10-12 模糊诊断】音乐 vs 语音 vs 静音 vs 白噪声；H-A4 300 真 CFG3（原配 joint，前 6 条）", "cond": "joint", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "H3AR/val_zsbase_biorig", "H3AR/val_zsbase_biospeech", "H3AR/val_zsbase_biosilence", "H3AR/val_zsbase_bionoise",
                "H3AR/val_HA4rich300_biorig", "H3AR/val_HA4rich300_biospeech", "H3AR/val_HA4rich300_biosilence", "H3AR/val_HA4rich300_biorigcfg3"]},
    {"name": "【★ H-A4 富文本训练】E3 基准 / 原始 H3 富文本 / H-A4 step-100 / 300 / 500 / 750（原配 joint）", "cond": "joint", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "H3AR/val_zsbase_biorig", "H3AR/val_HA4rich100_biorig", "H3AR/val_HA4rich300_biorig",
                "H3AR/val_HA4rich500_biorig", "H3AR/val_HA4rich750_biorig"]},
    {"name": "【10-12 富文本 prompt】E3（可用基准）/ H-A2 750 清单 vs 富文本 / H-A4 富文本训练（原配 joint）", "cond": "joint", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "H3AR/val_zsbase_biorig", "H3AR/val_HA2ds750_bi", "H3AR/val_HA2ds750_biorig", "H3AR/val_HA3cfg750_biorig",
                "H3AR/val_HA4rich100_biorig", "H3AR/val_HA4rich300_biorig", "H3AR/val_HA4rich500_biorig", "H3AR/val_HA4rich750_biorig"]},
    {"name": "【10-12 双向检查】原始 H3 / H-A2 750 / H-A3（同 prompt 同 seed，原配 joint）", "cond": "joint", "view": "models",
     "models": ["H3AR/val_zsbase_bi_s12n50", "H3AR/val_HA2ds750_bi", "H3AR/val_HA3cfg300_bi", "H3AR/val_HA3cfg500_bi", "H3AR/val_HA3cfg750_bi",
                "H3AR/val_zsbase_bihr", "H3AR/val_HA2ds750_bihr"]},
    {"name": "【10-11 手部诊断】480p vs 768×1344，原始 H3 / H-A2 / +fal（原配 joint，前 6 条）", "cond": "joint", "view": "models",
     "models": ["H3AR/val_zsbase_bi_s12n50", "H3AR/val_zsbase_bihr", "H3AR/val_HA2ds300_bi", "H3AR/val_HA2ds750_bi", "H3AR/val_HA2ds750_bihr",
                "H3AR/val_falReal1_bifal", "H3AR/val_falHA2_400_bifal"]},
    {"name": "【10-11 AR 拖影诊断】H-B2base 适配器/CFG/双向 LoRA 直接 AR（原配 joint，前 3 条）", "cond": "joint", "view": "models",
     "models": ["H3AR/val_zsbase_bi_s12n50", "H3AR/diag_zs_ar_s12n20_motion", "H3AR/val_HB2base200_ar", "H3AR/val_HB2base400_ar",
                "H3AR/val_HB2base400_arW", "H3AR/val_HB2base400_arcfg3", "H3AR/val_HA2ds400_ar",
                "H3AR/val_HB2base708_ar", "H3AR/val_HB2base708_arcfg3", "H3AR/val_HB2_100_ar", "H3AR/val_HB2_300_ar", "H3AR/val_HB2_500_ar", "H3AR/val_HB2_700_ar",
                "H3AR/val_HB3cfg200_ar", "H3AR/val_HB3cfg400_ar", "H3AR/val_HB3cfg708_ar"]},
    {"name": "【社区 LoRA】fal Realism-People：原始 H3 / +fal / H-A2 / H-A2+fal（原配 joint）", "cond": "joint", "view": "models",
     "models": ["H3AR/val_zsbase_bi_s12n50", "H3AR/val_falReal1_bifal", "H3AR/val_HA2ds300_bi", "H3AR/val_HA2ds500_bi", "H3AR/val_falHA2_400_bifal"]},
    {"name": "【M1a 音乐注入】H-A2 vs H-A2+MERT（换音乐 aswap）", "cond": "aswap", "view": "models",
     "models": ["H3AR/val_zsbase_bi_s12n50", "H3AR/val_HA2ds100_bi", "H3AR/val_HA2ds300_bi", "H3AR/val_HA2ds500_bi",
                "H3AR/val_M1a100_bi", "H3AR/val_M1a300_bi", "H3AR/val_M1a500_bi", "H3AR/val_M1a785_bi", "H3AR/val_M1a785_bi"]},
    {"name": "【双向主线】原始 H3 / 旧 H-A676 / H-A2（适配器）/ H-A3（guidance loss）（原配 joint）", "cond": "joint", "view": "models",
     "models": ["H3AR/val_zsbase_bi_s12n50", "H3AR/val_HA676_bi", "H3AR/val_HA2ds100_bi", "H3AR/val_HA2ds300_bi", "H3AR/val_HA2ds750_bi",
                "H3AR/val_HA3cfg100_bi", "H3AR/val_HA3cfg300_bi", "H3AR/val_HA3cfg500_bi", "H3AR/val_HA3cfg750_bi"]},
    {"name": "【AR 主线】原始 H3 / AR 老师 H-B2base / 4 步学生 H-C2（原配 joint）", "cond": "joint", "view": "models",
     "models": ["H3AR/val_zsbase_bi_s12n50", "H3AR/val_HBmix500_ar", "H3AR/val_HB2base200_ar", "H3AR/val_HB2base400_ar", "H3AR/val_HB2base600_ar",
                "H3AR/val_HB2base700_ar", "H3AR/val_HC2100_cm4", "H3AR/val_HC2200_cm4", "H3AR/val_HC2300_cm4", "H3AR/val_HC2500_cm4", "H3AR/val_HC2700_cm4"]},
    {"name": "【音乐控舞 idea】H-A2 / M1a MERT / M2(b) MoE / W1b DPO（换音乐 aswap）", "cond": "aswap", "view": "models",
     "models": ["H3AR/val_zsbase_bi_s12n50", "H3AR/val_HA2ds300_bi", "H3AR/val_HA2ds750_bi", "H3AR/val_M1a300_bi", "H3AR/val_M1a500_bi", "H3AR/val_M1a785_bi",
                "H3AR/val_M2300_bi", "H3AR/val_M2b100_bi", "H3AR/val_M2b300_bi", "H3AR/val_M2b500_bi", "H3AR/val_M2b886_bi",
                "H3AR/val_W1b100_bi", "H3AR/val_W1b300_bi", "H3AR/val_W1b446_bi"]},
    {"name": "【10-10 诊断】画质问题：采样设置 / 真 CFG（原配 joint，前 6 条）", "cond": "joint", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "H3AR/val_zs_bi_motion", "H3AR/diag_zs_bi_s12n50_motion", "H3AR/diag_zs_bi_s12n50_orig",
                "H3AR/val_HA676_bi", "H3AR/diag_HA676_bi_s12n50_motion", "H3AR/diag_HA676_bi_s12n50_cfg3",
                "H3AR/val_HBmix500_ar", "H3AR/diag_HB500_ar_s12n20_motion", "H3AR/diag_HB500_ar_s12n20_cfg3", "H3AR/diag_zs_ar_s12n20_motion"]},
    {"name": "画质诊断：原始 H3 vs H-A 双向微调（原配 joint）", "cond": "joint", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "H3AR/val_zs_bi_motion", "H3AR/val_HA200_bi", "H3AR/val_HA676_bi"]},
    {"name": "H3 AR 进度：老师最终版 vs 4 步学生（换音乐）", "cond": "aswap", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "H3AR/val_zs_bi_motion", "H3AR/val_HBmix500_ar", "H3AR/val_HB0mix1046_ar",
                "H3AR/val_HB0mix700_ar", "H3AR/val_HCsmoke100_cm4"]},
    {"name": "M1b：节拍写入 prompt vs 普通动作 prompt（原始 H3）", "cond": "aswap", "view": "models",
     "models": ["H3AR/val_zs_bi_motion", "H3AR/val_zs_bi_musicprompt"]},
    {"name": "H3 主线：零样本双向 / H-A / 零样本 AR / H-B0 / H-B（换音乐）", "cond": "aswap", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "H3AR/val_zs_bi_motion", "H3AR/val_HA200_bi", "H3AR/val_HA676_bi",
                "H3AR/val_zs_ar_motion", "H3AR/val_HB0mix500_ar", "H3AR/val_HB0mix700_ar", "H3AR/val_HBmix100_ar", "H3AR/val_HBmix300_ar"]},
    {"name": "H3 主线（原配 joint）", "cond": "joint", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "H3AR/val_zs_bi_motion", "H3AR/val_HA200_bi", "H3AR/val_HA676_bi",
                "H3AR/val_zs_ar_motion", "H3AR/val_HB0mix500_ar", "H3AR/val_HB0mix700_ar", "H3AR/val_HBmix100_ar", "H3AR/val_HBmix300_ar"]},
    {"name": "音乐控舞主对比：H3 / Wan2.2 最终版 / Wan2.2 AR（换音乐）", "cond": "aswap", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "E3_h3/out_retake", "H3AR/val_zs_bi_motion", "E2_S4final/out/S4-final",
                "E5/eval/E5b5_dmd704_lr4e-6_accum2_ema_it3000", "E6/eval/E6_3b_sfdmd_from2c3000_lr2e-6"]},
    {"name": "音乐控舞主对比（原配 joint）", "cond": "joint", "view": "models",
     "models": ["E3_h3/out/H3-Ref2VA", "E3_h3/out_retake", "H3AR/val_zs_bi_motion", "E2_S4final/out/S4-final",
                "E5/eval/E5b5_dmd704_lr4e-6_accum2_ema_it3000", "E6/eval/E6_3b_sfdmd_from2c3000_lr2e-6"]},
    {"name": "Wan2.2 最终版 单模型多条件（看可控性）", "cond": "joint", "view": "conds", "models": ["E2_S4final/out/S4-final"]},
    {"name": "原始 H3 单模型多条件", "cond": "joint", "view": "conds", "models": ["E3_h3/out/H3-Ref2VA"]},
    {"name": "DMD 学生 vs 老师", "cond": "joint", "view": "models",
     "models": ["E2_teacher480/out/S4-11000_480p", "E2/out/S4-11000", "E5/eval/E5a2_dmd480_lr2e-6_ema_it3000",
                "E5/eval/E5b5_dmd704_lr4e-6_accum2_ema_it3000"]},
    {"name": "Wan2.2 AR 系列（E6）", "cond": "joint", "view": "models",
     "models": ["E6/eval/E6_2a_causalODE_b4_lr1e-4", "E6/eval/E6_2c_causalODE_b4_lr3e-5", "E6/eval/E6_3a_sfdmd_from2a1500_lr2e-6",
                "E6/eval/E6_3b_sfdmd_from2c3000_lr2e-6", "E6/eval/E6_3c_sfdmd_from2a3000_lr5e-6"]},
    {"name": "E10 stage5 / CFG / 音频强度", "cond": "joint", "view": "models",
     "models": ["E10/val/S4final_intended/out/S4final_intended", "E10/val/S5_step2000/out/S5_step2000",
                "E10/val/S4f_cfg5/out/S4f_cfg5", "E10/val/S4f_as5/out/S4f_as5", "E10/val/S4f_as8/out/S4f_as8"]},
    {"name": "E11 参考帧/增强消融", "cond": "joint", "view": "models",
     "models": ["E11/val/E11a_ref40-80_aug1", "E11/val/E11b_ref20-80_aug2", "E11/val/E11c_ref40-80_aug1_intended",
                "E11/val/E11d_ctrl_ref0_noaug", "E11/val/E11e_ref20-80_aug2_intended"]},
    {"name": "E9 长视频 33s（AR / 参考帧锚定）", "cond": "long", "view": "models",
     "models": ["E9/longAR_E6_3b_8seg", "E9/longAR_E6_3b_8seg_refanchor", "E9/long_E9a_it3000_8seg"]},
]


def slim(r):
    return {k: round(float(r[k]), 4) for k in METRIC_KEYS if isinstance(r.get(k), (int, float))}


def load_metrics(rel, dp, gt):
    cands = [os.path.join(dp, "metrics")]
    parts = rel.split("/")
    if len(parts) == 3 and parts[1] == "out":
        cands.append(os.path.join(EXP, parts[0], "metrics_TA"))
    m = {}
    for md in cands:
        if not os.path.isdir(md):
            continue
        for f in sorted(os.listdir(md)):
            if not f.endswith(".jsonl"):
                continue
            for line in open(os.path.join(md, f)):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                c, sid = r.get("cond"), r.get("key", "").split("/")[-1]
                if c == "gt":
                    gt.setdefault(sid, slim(r))
                elif c:
                    m.setdefault(c, {})[sid] = slim(r)
    return m


def make_model(rel, dp, conds, gt):
    entries = {}
    for c, files in conds.items():
        d = {}
        for f in files:
            ids = [t for t in re.split(r"[_.]", f) if ID_RE.fullmatch(t)]
            if not ids:
                continue
            sid = ids[0]
            src = ids[1] if len(ids) > 1 and ids[1] != sid else None
            d[sid] = [f, src] if src else [f]
        if d:
            entries[c] = d
    lab = []
    for p in rel.split("/"):
        if p not in ("out", "eval", "val") and (not lab or lab[-1] != p):
            lab.append(p)
    label = "/".join(lab)
    desc = ""
    for k in sorted(DESC, key=len, reverse=True):
        if label.startswith(k) or rel.startswith(k):
            desc = DESC[k]
            break
    return {"id": rel, "label": label, "group": rel.split("/")[0], "desc": desc, "conds": entries,
            "metrics": load_metrics(rel, dp, gt)}


def read_csv(path):
    csv.field_size_limit(10 ** 9)
    with open(path) as f:
        return {r["sample_id"]: r for r in csv.DictReader(f)}


def build_manifest():
    t0 = time.time()
    gt, models = {}, []
    load_metrics("E2/out/_", f"{EXP}/E2/out/_", gt)  # GT 指标以 E2/metrics_TA（原长 prompt）为准
    for dp, dns, fns in os.walk(EXP):
        rel = os.path.relpath(dp, EXP)
        dns[:] = sorted(d for d in dns if d not in SKIP and not d.startswith("metrics") and ("smoke" not in d or d.startswith("val_HCsmoke")))
        conds = {}
        for c in CONDS:
            if c in dns:
                files = sorted(f for f in os.listdir(os.path.join(dp, c)) if f.endswith(".mp4"))
                if files:
                    conds[c] = files
        if conds:
            dns[:] = [d for d in dns if d not in CONDS]
            models.append(make_model(rel, dp, conds, gt))
        elif rel.startswith("E9/") and any(f.endswith(".mp4") for f in fns):
            models.append(make_model(rel, dp, {"long": sorted(f for f in fns if f.endswith(".mp4"))}, gt))
    models = [m for m in models if m["conds"]]

    samples = {}
    joint = read_csv(f"{EXP}/E2/full_joint.csv")
    aswap = read_csv(f"{EXP}/E2/full_aswap.csv")
    tswap = read_csv(f"{EXP}/E2/full_tswap.csv")
    mcap_p = f"{EXP}/E10/val47_motion_captions.csv"
    mcap = {k: v.get("motion_caption", "") for k, v in read_csv(mcap_p).items()} if os.path.exists(mcap_p) else {}
    for sid, r in joint.items():
        samples[sid] = {"prompt": r["prompt"], "audio": r["audio_path"], "video": r["video"], "image": r["image_path"],
                        "aswap_src": aswap.get(sid, {}).get("swap_src", ""), "tswap_src": tswap.get(sid, {}).get("swap_src", ""),
                        "motion_caption": mcap.get(sid, ""), "gt_metrics": gt.get(sid, {})}
    man = {"exp": EXP, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "samples": samples, "models": models,
           "presets": PRESETS, "conds": CONDS + ["long"]}
    os.makedirs(OUT, exist_ok=True)
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, ensure_ascii=False)
    os.replace(tmp, MANIFEST)
    print(f"[manifest] {len(models)} models, {len(samples)} samples, {time.time() - t0:.1f}s", flush=True)
    return man


_lock = threading.Lock()
_cache = {"gz": None}


def manifest_gz(rebuild=False):
    with _lock:
        if rebuild or _cache["gz"] is None:
            if rebuild or not os.path.exists(MANIFEST):
                build_manifest()
            with open(MANIFEST, "rb") as f:
                _cache["gz"] = gzip.compress(f.read(), 5)
        return _cache["gz"]


def safe_name(s):
    s = re.sub(r"[^\w\-]", "_", s or "")[:40]
    return s or "anon"


def load_reviews():
    out = {}
    if not os.path.isdir(REVIEWS):
        return []
    for f in sorted(os.listdir(REVIEWS)):
        if not f.endswith(".jsonl"):
            continue
        for line in open(os.path.join(REVIEWS, f)):
            try:
                r = json.loads(line)
                out[(r["reviewer"], r["key"])] = r
            except Exception:
                pass
    return list(out.values())


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode())

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path.rstrip("/").split("/")[-1] if u.path not in ("/", "") else ""
        try:
            if path in ("", "index.html"):
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8", {"Cache-Control": "no-cache"})
            elif path == "manifest":
                body = manifest_gz(rebuild=q.get("rebuild", ["0"])[0] == "1")
                self._send(200, body, extra={"Content-Encoding": "gzip", "Cache-Control": "no-cache"})
            elif path == "reviews":
                self._json(load_reviews())
            elif path == "media":
                self._media(q.get("p", [""])[0])
            elif path == "oos":  # OOS 15 s T2VA test page (built by OOS_H3/make_page.py)
                with open(os.path.join(EXP, "OOS_H3", "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8", {"Cache-Control": "no-cache"})
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        u = urlparse(self.path)
        if not u.path.endswith("/review"):
            return self._json({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length", 0))
        if n > 1 << 20:
            return self._json({"error": "too large"}, 413)
        try:
            r = json.loads(self.rfile.read(n))
            assert isinstance(r, dict) and r.get("key")
        except Exception:
            return self._json({"error": "bad json"}, 400)
        r["reviewer"] = safe_name(r.get("reviewer"))
        r["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(REVIEWS, exist_ok=True)
        with _lock, open(os.path.join(REVIEWS, r["reviewer"] + ".jsonl"), "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._json({"ok": True})

    def _media(self, p):
        real = os.path.realpath(p)
        if not any(real.startswith(a + os.sep) for a in ALLOWED) or not os.path.isfile(real):
            return self._json({"error": "forbidden or missing"}, 404)
        size = os.path.getsize(real)
        start, end, code = 0, size - 1, 200
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            s, e = m.groups()
            if s == "":
                start = max(0, size - int(e))
            else:
                start = int(s)
                end = min(int(e), size - 1) if e else size - 1
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", CTYPE.get(os.path.splitext(real)[1].lower(), "application/octet-stream"))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "max-age=3600")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(real, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    manifest_gz(rebuild=a.rebuild)
    srv = ThreadingHTTPServer((a.host, a.port), H)
    srv.daemon_threads = True
    print(f"[serve] http://{a.host}:{a.port}/", flush=True)
    srv.serve_forever()
