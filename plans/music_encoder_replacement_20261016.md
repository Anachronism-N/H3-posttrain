# 专用音乐 encoder 替换 H3：候选比较＋两条训练路线

## 用户确认的范围
- 当前改用专用音乐 encoder，视频生成也使用同一个微调 encoder；不是只算CLIP分数，也不是额外MERT残差注入。
- 同时尝试 **严格CLIP** 和 **原生latent桥接＋CLIP**，允许桥接监督。冻结H3文本encoder、DiT及audio decoder；不用caption检索，不增加DiT/video loss。
- 微调及桥接仅用舞蹈视频配乐，沿用原版MotionScript＋舞种caption。已有2k（1800 train/200 val）为首轮公平对照；3个固定生成case独立留出，joint/aswap各3条。

## 1. 候选及选择标准
截至2026-10-16官方资料核查：
- **MuQ-large-msd-iter**：约300M，24kHz，25Hz时序hidden、1024维；公开MSD权重，官方建议FP32避免NaN，CC-BY-NC4.0。作为首要候选，不能称普适最强；论文MARBLE 77.0是私有160k小时迭代版，不能套给公开版。
- **MusicFM-MSD**：24kHz、25Hz、1024维；官方有节拍/结构分析动机与微调方法，强时序候选。HF仓库元数据标注`license: mit`，但README为空；这是仓库级许可元数据证据，不能等同于已核验具体`pretrained_msd.pt`权重的完整授权条款，也不能仅依据代码MIT扩大权重使用或再分发范围。须记录所用权重revision/hash及该证据边界，不发布权重。
- **MERT-v1-330M**：24kHz、75Hz、1024维，本地已有；保持作为复现基线，不继续用先前16Hz压缩缓存伪称其原始时序网格。
- **MuQ-MuLan**：音乐文本语义对照。官方配置音频底座MuQ，audio_transformer depth=0；官方`create_MuLan_from_config`明确传入`frozen_pretrained=False`，不能只依据底层类默认冻结开关认为官方加载流程冻结底座。尚未实测加载及权重对比，不能声称其底座与独立MuQ权重相同或未经过语义训练；核对加载后的参数/结构/hash，若实测底座相同，不重复算一个独立encoder。512维全局输出只作语义诊断，不重复铺成生成帧。
- **CLaMP3-SAAS**：强全局风格检索参考，但默认先把MERT的5s窗口平均，约5s的本项目片段会失去帧级结构，不列入首批密集时间接口替换。
- **Music2Latent**：约10Hz/64维音频codec，目标偏重建，不冒充最强音乐语义模型；保留调研结论，首批不扩大到codec替换。
- 更大的音频LLM或音乐生成模型不因体量大就算更好的时序音乐encoder。

先比较前三个的原始时序hidden：相同train/val、相同时间窗口、仅浅层线性音频head（无MLP）、相同文本监督。分别报告a2t/t2a R@k、舞种/粗舞种命中、同舞种动作区分、随机/多数类基线、真实frame rate及有限值。补做冻结音频特征上的节拍/起音局部可读性诊断，不把文字检索当生成评分。
- 验证集用于选择配置，报告这一选择偏差；固定3case生成人评独立于2k，必要时从精选val另选仅表征测试集（不新增视频生成case）。
- 相同/近重复配乐分组检查，避免同曲泄漏或硬负样本冲突；不未经核实给音频赋唯一编舞。
- 按本项目结果选择前两个dense候选进入替换，优先MuQ与MusicFM；MERT保持基线。若MuQ不占优或不兼容，不强行坚持它。

## 2. 共享训练／生成结构
```
舞蹈配乐（双声道）
  → 对每声道重采样24kHz（不先平均声道）
  → 预训练音乐encoder，保留逐帧hidden
  → 小型时序adapter（约256隐藏通道，局部卷积＋线性32维输出）
  → 按真实时间戳对齐到H3的40Hz／双声道32维normalized latent z
       ├─ 有效区间pool(mean/std) → 单层线性音频head → 5120维 → CLIP
       └─ 原H3 audio patch projection → 冻结DiT → 原视频decoder
```
- adapter不是训练专用projector：encoder＋adapter在训练和生成都必须保留。线性CLIP head只训练使用，不加入推理。
- 两个输入音轨条件（reference audio与target/retake audio）均由新的encoder＋adapter生成，彻底取代原生编码侧。原生encoder只作离线桥接教师或对照，不能偷偷留作推理信息旁路。
- 接口保持`[2,32,T]`、40Hz、原生声道排列、原生normalized坐标；不得把1024/5120向量直接塞入32维接口，也不对生成latent做L2单位化。
- 首轮5.0625s的原生有效latent为203帧；124/24秒视频目标207帧，沿已验证repeat_clean政策填尾部，mask0、有效AdaLN time1；原生基线与新encoder所有配置用同一政策，padding不进入CLIP池化。
- 保留音乐时序，而非把全局embedding复制207次；25→40Hz只提高接口采样密度，不能声称产生新的节拍信息。
- 浅层head的“弱”只指表达层数（LayerNorm＋单层线性、无MLP），不意味着参数量比上一轮2.7M MLP更小：冻结时序特征双声道mean/std池化的4096→5120 probe有20,979,712参数（20,971,520线性权重＋8,192 LayerNorm参数）；共享生成latent池化的128→5120 CLIP head有655,616参数（655,360线性权重＋256 LayerNorm参数）。这两种head不是同一个接口；用各自同head的原始/训练encoder消融报告骨干真实增量。全局CLIP仍不能保证逐拍控制，明确该边界。

## 3. 两条路线及必要对照
**A．严格CLIP**
- 从公开音乐encoder与同规格随机adapter开始；全encoder＋adapter＋弱音频head，仅对称InfoNCE，冻结文本和DiT。
- 不加原生latent MSE/音频重建/video loss。先做接口有限值/尺度检查，再按固定3case生成验收；未知坐标兼容性是结果，不能预先承诺可用。

**B．桥接＋CLIP**
1. 使用同样舞蹈配乐，冻结原始H3 audio-VAE编码侧提取逐帧normalized教师latent。
2. 冻结音乐骨干，只训adapter对齐教师latent，按有效帧/时间/声道计算归一化逐帧MSE；不额外引入音频重建训练loss。
3. 再解冻音乐encoder＋adapter，做`L = L_CLIP + λ L_bridge`，音频侧浅head对齐冻结H3文本；λ明确记录，首轮只小范围对照，不遍历大规模无效组合。
4. 保存 **bridge-only** 检查点，和B比较，避免将接口恢复算作CLIP音乐增益；另以同bridge初始化`λ=0`作最小消融，区分初始化与持续保真约束。
- 桥接可能把新音乐表示拉回原生声学空间、削弱语义优势；音乐encoder可能已丢失无法恢复的声学相位。若bridge-only在留出集都无法形成可用接口，先报告该失败，不靠大loss或重复全局向量掩盖，不擅自改DiT。
- A/B是不同完整配方，不能把其差异直接归因于单一loss；同初始化λ对照才隔离兼容约束。

## 4. 实现范围与文件
优先修改已有训练/评测入口，必要新文件仅共享音乐后端与H3代理：
- `diff/scripts_omnidance/exp/ms/native_audio_full_clip.py`：复用数据/分组划分、DDP、InfoNCE、有限值、checkpoint/hash；新增独立backend选择，保留可复现原生配置。
- `diff/scripts_omnidance/exp/ms/text_embed.py`：复用冻结H3目标；不改变已生成缓存含义。
- 必要新模块`diff/scripts_omnidance/exp/ms/music_encoder_adapter.py`：MuQ/MusicFM/MERT可微前向、原生时间接口、共享adapter与序列长度mask；明确绕开库的no_grad/frozen开关，禁用破坏逐帧对齐的随机mask（或同步验证策略）。
- `diff/scripts_omnidance/exp/h3ar/sample_h3_ar.py`：新checkpoint入口，代理audio_vae.encode_audio；保留sample_rate/hop_length/decoder，支持CPU offload调用与有限值断言；不调用G4文本hook、h3_music残差。
- `diff/scripts_omnidance/exp/ms/native_audio_validate.py`：扩展共享backend，音频重建及native接口比较只作为诊断，不当新训练loss。
- `diff/scripts_omnidance/exp/ms/eval_gen_motion.py`、`review_web/server.py`：固定三case结果与caption评审；区分encoder、adapter、loss路线、桥接-only。
- `H3-posttrain/notes/22_h3_bidir_blur_attribution_20261013.md`及必要实验计划/结果文件：记录官方证据、许可、配置、分割、负结果和生成验收；同步必要代码到代码仓再提交推送，不发布账号/权重/原视频/受限模型。

## 5. 验收与资源安排
1. 先审查官方源码、固定模型revision/hash与独立本地venv，禁止信任未经检查remote code。下载并预读本地盘，避免重蹈cephfs冷读拖住GPU。MuQ默认FP32，先测再决定AMP，不无依据BF16。
2. 合成测试＋真实3case小前向：shape/time/channel、归一化、可微梯度、参数真实更新、缓存不能代替全encoder训练；确认text/DiT无更新。
3. 预训练encoder浅probe并行分节点；adapter桥接阶段多卡并行；每条路线起点/数据/日志独立，既有running任务不覆盖不修改。只用受管理后台任务，不nohup/setsid逃逸。
4. checkpoint有限值/hash；验证train/val loss、检索随机基线、固定head骨干增量、有效latent漂移、原decoder重建；shape正常不能算声学兼容成功。
5. 固定3case×joint/aswap，各配置同图/prompt/seed/704p/50步/音轨，完整视频及音轨解码；1s后运动学＋连续视频人评看音乐风格/节奏差异和手脸清晰度，不能将动作变化量当音乐遵循。
6. 首轮不自动全量生成或扩case；不足才在已选数据上增训练数据（按曲目组留出且复用相同200验证基准）。每个自然里程碑发布中文结论；失败修复后重新核验，不以任务启动、指标升高宣称目标完成。

## 2026-10-16 实施修正与恢复约束
- 共享后端已实现为`music_encoder_adapter.py`，训练入口为`music_clip_bridge.py`，冻结特征probe入口为`music_linear_probe.py`；复用原生CLIP的数据/损失/有限值工具，不改运行中的原生训练入口。
- 2个骨干×4配方共8组训练完成，真实梯度覆盖316,343,296骨干参数；3个声明trainable但未参与forward的位置卷积张量计8,389,760参数，不能按声明数宣传全更新。详细指标、hash、声学诊断见`results/ms_20261016/music_*`及notes §28。
- 已发现缓存warmup的原wav→24k与在线原wav→32k→24k链路不完全一致，三条验证音频的latent差MuQ约3.8%–4.6%、MusicFM约5.6%–9.9%，FP16缓存自身误差远小于此。保留当前配置/产物；统一预处理须用新缓存与单独warmup对照，不修改活跃生成的代码来掩盖差异。
- 第二批生成在仅检查GPU空闲后并发冷加载，8/24分片遭主机/cgroup OOM。失败不归因于训练NaN；保留首轮失败日志，正常活跃分片不重启，仅精确清理已失去父任务的编码worker。恢复仍只3case joint/aswap，输出文件名不改变，用独立`shard_N_retry_memory1.log`保留证据。
- **恢复准入**：启动前同时读取`/sys/fs/cgroup/memory/memory.usage_in_bytes`、`memory.limit_in_bytes`和`/proc/meminfo`的MemAvailable，主机与cgroup两者都至少有180GiB可用；确认目标GPU只有已核对的占卡进程或完全空闲，非holder进程不得停止。当前保守限制为每个恢复节点至多4个既有H3 sampler，再添加1个冷加载。每节点两个失败分片串行运行，前一个退出后重新检查预算才启动后一个；预算不足直接停止队列，不sleep轮询/重试、不批量kill。
- 监控用受工具管理的后台任务及完成事件；不以GPU占卡100%当真实利用率，不以kernel failcnt当前为0排除先前OOM。现有正常任务全部完成后，再考虑修改共享加载/缓存代码降低峰值内存。

## 官方证据
- https://github.com/tencent-ailab/MuQ （24kHz、FP32、公开MSD权重与论文差异、权重CC-BY-NC）
- https://huggingface.co/OpenMuQ/MuQ-large-msd-iter/raw/main/config.json （1024dim、label_rate25）
- https://arxiv.org/html/2501.01108v2#Sx4 （77.0私有MuQ_iter，75.4 MusicFM，74.4 MERT，非公开MuQ排行榜）
- https://github.com/minzwon/musicfm （25Hz、时序任务/微调支持、MSD与FMA区别）
- https://huggingface.co/m-a-p/MERT-v1-330M （75Hz、24层1024、24kHz、各层须实测）
- https://raw.githubusercontent.com/tencent-ailab/MuQ/main/src/muq/muq_mulan/models/audio.py （冻结开关、no_grad、return_mean=False序列输出）
- https://huggingface.co/OpenMuQ/MuQ-MuLan-large/raw/main/config.json （音频MuQ、tf_depth0；不误称新独立骨干）
- https://raw.githubusercontent.com/sanderwood/clamp3/main/preprocessing/audio/extract_mert.py （默认5秒无重叠池化，首批不作为密集时间替换）
- https://github.com/SonyCSLParis/music2latent （重建codec≈10Hz/64维、非音乐语义冠军）
