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

## 2026-10-17 补齐诊断的实施口径
- 视频完整性独立入口`music_generation_validate.py`按两份CSV前三行固定映射八配置48条。先核实成功分片的两次新音频编码、203帧latent和repeat_clean尾部4帧/time1，再以decord完整读124帧及ffmpeg完整视频/音轨解码、记录文件与日志hash。没有成功日志的运行中文件不计通过；partial报告保留`complete=false`。既有环境无ffprobe，使用已有decord/ffmpeg，不把已选双流存在与解码冒称恰好两个声明流。
- 时序可读性入口`music_temporal_probe.py`只读三个原始网格缓存，在同1800/200划分上拟合1024→3带截距ridge（固定alpha10，不验证调参）。共同25Hz网格0.04–4.96秒、训练统计标准化、按真实source范围和音频完整窗mask，分别预测起音谱通量、log-RMS及5秒DP节拍代理。DP有120BPM先验且未经人工核验，不是节拍真值。报告R2/MAE/Pearson、逐片相关及常量无定义计数，同head的片段均值广播/1秒循环错位、同网格64维logmel声学基线；不把代理可读性当作舞蹈生成验收。
- 近重复入口`music_duplicate_audit.py`只检查前5.0625秒真实音频，不把padding作重复。精确SHA256与训练统计标准化后的logmel/chroma/RMS签名top-k筛候选，±0.48秒移位且至少3秒真实重叠后比对时序谱/色度/包络。默认疑似阈值0.98/0.98/0.95，附0.95/0.98/0.995敏感性；候选门控可能漏检，疑似连通簇不等于曲目ID，也不能排除同歌不同段。仅报告跨split风险，未经人校验不更改既有划分、不重新训练。
- 新诊断先源码/合成测试再工具后台运行；与`JbHM6s`最后四条串行恢复分离，既有H3 sampler与源码不修改，不启动第四视频case。所有模型权重/派生音频PT仍不发布，发布诊断代码与汇总JSON。

## 2026-10-17 用户评审后的后续事项（暂不启动）
- 用户要求其他音乐维度之后测试：速度/拍点相位、重拍/小节、节奏型与切分/律动、音色/乐器、乐句/段落、情绪张力与风格。同数据/划分/读出容量/时间窗口，复杂节奏与结构采用独立可信标签，不能只追加未经核验的自动标签；5秒不足的结构维度可延长表征测试窗口，但不增加视频生成case。
- 多encoder融合仍是候选方向，先补测互补性与融合浅probe，和参数量相近的单encoder、简单拼接对比，再决定是否动态门控；不把交替更优直接称创新已验证。
- 用户提出比较音乐表征的文本输入token替换、残差叠加及输出hidden追加等接口。需要独立训练音乐→对应文本空间的token适配器：已有归一化池化CLIP向量不是输入embedding或逐tokenhidden，维度相同不意味着空间等价。保留人物/首帧/音轨绑定模板、冻结text/DiT、固定3case。
- 建议先固定当前可用音频条件，比较局部音乐槽的等长替换与残差叠加，提供同槽零增量和错配音乐对照；输出hidden追加另作一组，不与输入槽混为同空间。文本路有增量后才决定是否移除原音频路。当前仅方案讨论，未修改互斥入口或启动新训练/推理。

## 2026-10-17 自主推进：反事实评测与数据扩量
- 用户授权自动规划/实施/修复/持续跟进与推送；先完成正确音乐对照指标及MusicFM嵌套数据规模实验，不同时改文本接口或启动后置其他维度。
- 已核查八节点目前无本轮训练/推理，显存约975MiB的100%占卡程序不算实验任务。只停止核对过的holder并用工具管理后台，训练/cache/评测并行；高分辨率H3冷加载仍受主机/cgroup准入守卫。
- 数据源是精选train与已有11992原版MotionScript/VLM舞种交集，无需重跑NLF；核验空句/音频质量。保留既有验证200和固定人物/替换音轨6ID，扩量全候选进行音频近重复门控/精确hash审计。任何疑似组连到验证或生成音轨都保守排出训练；训练每候选组只留代表。该组是声学候选，不宣称已完整识别歌曲或严格song-isolated。
- 构造同验证200、嵌套1800/6000训练及固定1800训练统计归一化。新缓存统一原wav→32k→24k用于MusicFM，教师原wav→32k→原始冻结H3encoder；全部新路径保留旧链路/results。
- 显式split/归一化ID输入、max_steps/定期step验证与实际吞吐计时支持数据规模对照：1800与6000相同更新次数，再额外6000更多更新次数；lambda0/lambda1分开记录。桥接warmup与CLIP步预算明确，不把多训练步当纯数据收益。先真实小步DDP及梯度/缓存接口验收，再长任务。
- 已有48+原生6+GT3视频只提一次DWPose，统一[1,5)实际时间/置信度/缺失处理，分body/arms/legs零lag起音与运动相关、固定事件拍点评分。每视频分别配实际条件音乐和另一音乐，报告margin及原生增量；32个保拍点密度相位shift提供零分布。低覆盖/静音/缺事件null不填0，不挑最佳lag。指标先以GT实际原音乐是否优于替换音乐作校准，若失败明确限制，不凭高BAS宣称音乐遵循。
- 输出质量ROI锐度/检测覆盖仅代理，结合连续帧视觉审查；仍只三个生成case。计划/代码/实际wall时间/finite/hash、负结果和验收均推送，权重/缓存PT/原配乐视频不发布。

## 2026-10-17 mentor建议的故事线（参考讨论，不立即实施新系统）
- 贡献定位考虑从单纯music-driven avatar/video生成转向音乐条件编舞规划：舞种/舞步组合、可编辑方案、候选选择与动作分解辅助学习；H3是保持视觉质量的rendering backbone。核心需是更准确、自由、可验证的控制，不是换名称或网页包装。
- 最少对照应含直接H3自然语言prompt、相同舞种/动作信息的手工计划、音乐条件LLM规划，以及局部编辑与组合的控制成功率/不受编辑部分的保留率。必须区分LLM引入更多文字信息与音乐encoder改进，不把已有原生文本能力说成新贡献。
- 当前其它维度与LLM/APP训练仍后置。正在进行的反事实诊断及一次小规模数据量对照用于决定是否继续音乐encoder路线；若数据扩量仍无可感知收益，不无限遍历学习率/轮数，以已可用H3的编舞控制验证为下一决策。

## 2026-10-17 正式规模实验执行与生成验收
- 1800/6000各100更新bridge完成；随后1800 lambda0/1各560、6000 lambda0/1各560及1875共六配置全部完成正式更新。固定global batch16、文本归一化clean1800、验证200顺序；具体指标和实测耗时见notes §28.14。只在最后更新验证，因此best不代表多轮择优。
- N4已存在字节相同缓存时cp no-clobber返回非零，第一版lambda0/560未开始；恢复先核验hash并确认无同输出活跃进程。另一重复尝试因已有正常训练占端口而在rendezvous前失败，保留日志、不杀正常任务、不再重复启动，等待现有560任务完成。
- 综合权重审计在有完整本地缓存的N4运行，实际校验best/last文件SHA256、CPU读取全张量finite、split/norm/最后验证/finished事件和输入文件hash。N6缺其他主机记录的本地cache路径曾导致审计退出，不能解释为权重坏；无修改训练源码或输入来掩盖。
- 新生成队列`music_scale_generate.py`仅接受已审计CLIP配置，每配置固定3人物joint/aswap各3、704×1280、124帧、24fps、bi/50步、shift12/audio_shift3、seed42、generic prompt、repeat_clean。每节点只新增一个冷加载，主机/cgroup各180GiB准入、目标GPU非holder不得停止；输出/日志/最终JSON新路径拒覆盖，成功后完整decord/ffmpeg解码。
- 1800两组12条先生成，6000四组24条在权重审计完成后安排到其余节点并行。仅三人物，不因扩量增加case。后续DWPose复用原生/GT缓存，F1/BAS继续只作校准不足的诊断；原decoder声学检查及连续帧检查不等于音乐收益，最终需明确相对原生/bridge-only的可感知增量或负结果。

## 2026-10-17 执行收尾
- 六训练配置、16份best/last实际审计、36视频完整124帧音画解码以及45条反事实/姿态缓存全部正确结束。完整JSON与元数据汇总发布至`results/ms_20261017/music_scale_all6_*`和`music_scale_final_acceptance_v4.json`；最终解释见notes §28.15，不以小数据12条结果替代全量。
- 扩量/更多更新对检索有配方依赖的收益，但原生声学恢复和生成音乐遵循没有可靠一致增量；GT事件拍点校准失败仍在，三人物及多配置视频不独立。明确是未证明收益，不是证明无法控制。停止自动扩大数据或扫更多配方，当前固定case连续听看人评后再决策文本接口/可编辑编舞路线，不擅自训练DiT或开发LLM/APP。
- 原评审开发机SSH缺可信key后，用户允许替代方案：当前节点8766独立服务，固定八模型（六新+原生+历史bridge-only）/三人物/三预设清单，保留旧8765及旧manifest。`server.py --manifest`禁止全树rescan；`index.html`显式每条件音轨防止focused清单aswap回落原音轨。页面/API和48video+6audio Range均通过，计算节点访问code-server代理超时，后续用户明确确认https代理8766可访问；内网直连报readLoopPeekFailLocked，不影响已确认代理入口。连续音画人评仍单独记录，不把入口可访问当效果验收。

## 2026-10-17 连续音画验收补充
- 新增已有Omni7B的联合音画校准入口，实际同裁[1,5)音频/video并核验联合token时间，三个GT分别正确/错配配乐六次前向，不用旧纯音乐CoT冒充视频评审。
- 六次运行完成，但GT正确/错配分数5/3、3/3、3/3，泛化描述和两对平局不足以校准评分器；保留raw及JSON围栏续写解析修复后的完整意见，不扩大无效自动ranking、不称人评。连续新配置效果验收仍需真实听看反馈；用户已确认HTTPS8766代理可访问。

## 2026-10-17 首轮真实效果验收完成
- 用户已连续听看lambda0与lambda1扩量预设并提供自然语言反馈：joint各配置差异小，aswap动作改变但保留母题，与原H3总体相近，lambda1也肉眼难区分，画质均未退化。原话及边界见notes §28.17和final acceptance的human_review字段。
- 以两个预设整体定性反馈完成本轮首轮效果验收，不要求必须提交网页数值评分；不冒称36条逐视频穷尽人评、不捏造逐拍偏好或bridge-only独立评分。结论为画质保留、扩量和更多更新未建立明确可感知音乐控制增量；检索改善不作视频收益。过去“仍待人评”是阶段状态，以本节为最新事实。
- 本轮受控规模实验到此收尾，不重复训练/加数据以满足goal。下一方向优先另行设计音乐文本接口/可编辑编舞控制的可归因验证，其他维度/融合/LLM/APP与DiT训练不在本轮自动扩大范围。

## 用户新增输入约束：仅音乐＋参考图快速推理
- 用户要求后续验证只给音乐和参考图，不给文本prompt；本轮先比较原生H3与MusicFM6000/1875的lambda0和lambda1，不扩大六配置全量训练或新增人物。每配置固定三人物、joint/aswap各3，共18条；50步/704×1280/124帧24fps/seed42/shift12和audio_shift3/repeat_clean保持与前轮一致，只改语义文本。
- 复用`sample_h3_ar.py --prompt_mode empty`：实际prompt为单个空白字符`" "`，无通用舞蹈句、无H3自然语言人物/音乐绑定模板、无MotionScript/CoT/软token注入；不移除模型所需文本编码器或Ref2VA的结构标识`<Picture 1>: / <Audio 1>:`和参考条件打包。上游拒绝长度0字符串，空白占位没有语义指令，不冒称完全删除文本模态。
- sampler增加实际input_conditions日志（prompt/图/音轨/seed/encoder），empty禁止caption/soft-token/prefix组合；验收支持显式native_tags和require_empty_prompt，不把原生模型缺replacement日志当失败。合成测试空文本分支不调用模板，非空语义与假replacement声明拒绝。无训练loss/权重或DiT更改。
- 八节点启动前无活跃采样/训练，主机/cgroup空闲均超过180GiB；每节点只新增一个冷加载，18分片按队列串行，每次重新准入。完整18结束后再完整音画解码并提供新评审入口。当前启动不等于视频验收结束，准确命令在`music_empty_inference_plan_v1.json`，旧结果不覆盖。

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
