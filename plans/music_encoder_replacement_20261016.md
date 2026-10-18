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
- 八节点启动前无活跃采样/训练，主机/cgroup空闲均超过180GiB；每节点只新增一个冷加载，18分片按队列串行，每次重新准入。18分片全部exit0，严格完整音画与实际空白输入核验18passed/0missing/errors已通过。准确命令、完成任务和逐条验收分别为`music_empty_inference_plan_v1.json`、`music_empty_inference_jobs_v1.json`及`music_empty_inference_integrity_v1.json`，旧结果不覆盖。
- 原8766入口第一预设已加入【仅音乐+参考图·无文本】原生/λ0/λ1三列，合并11模型/3人物/4预设并保留原有文本对照。页面/API200、18视频及6实际音轨Range206已核验。推理请求已完成，空文本条件下的音乐效果需看当前新结果，不能用旧有文本的人评宣称这批有效。

## 2026-10-18 无文本诊断与目标对齐边界
- 用户明确无文本动作不同、动作小且不像舞蹈；新增18empty pose缓存一次，重读同权重generic/原生/GT共39视频，对照实际音乐来源caption。body速度比原生0.5773/λ0 0.5032/λ1 0.4641，原生本身也退化；不把运动量当caption/音乐分数。六视觉图与缺失动作证据见notes28.19、`music_empty_caption_visual_review_v1.json`与完整motion audit。
- 参考图当前是reference-only而非硬首帧条件：sampler的keyframes/retake_video为None。实际楼梯浅黄衣/灰裤在empty原生及两音乐encoder都变深色衣裤，不能把输入正确/解码成功当参考保真正确；不单独归因训练骨干。维持无外部语义文本要求，先验收一致的参考条件，不偷偷恢复通用舞蹈句。
- 共用encoder/adapter与CLIP的池化投影对齐不等于生成行为一致。若需要训练与部署学生完全同输入，应在相同图/音频/噪声/timestep/CFG/tail下训练空文本学生通过冻结H3的生成路径；教师可用内部caption或GT作为监督而学生推理无caption。该路线需新增生成监督并保留冻结DiT到encoder/adapter的梯度，超出此前CLIP/桥接-only限制，需独立授权/方案，当前只分析不实施。没有承诺音乐唯一恢复caption或两条件输出逐像素一致。

## 2026-10-18 已确认的新主线：音乐预测原生文本 hidden

用户确认先走文本侧输出hidden，推理先完全不提供人物/ID/动作文字；不足后才加入仅身份信息。A输入embedding及首帧固定后置，C旧全局接口复原和D离散动作文字不纳入本轮。

```text
训练教师：参考图结构prefix + 舞种/原版MotionScript → 冻结H3文本encoder → raw逐token hidden
训练学生：参考图结构上下文 + 音乐 → MusicFM → 共享text adapter → raw逐token hidden
部署学生：执行完整同一MusicFM/adapter/最后5120维投影 → 原DiT文本条件入口
```

- 原32D音频adapter生成确实使用，但旧CLIP head的归一化全局向量不是原生text token目标。本轮最后投影属于部署adapter，不能再只用于训练loss。
- CLIP不强制池化；首轮直接raw token MSE/cosine和长度监督，后续可对实际输出序列做无额外语义head的对比学习。检索相似不保证音乐对应动作，caption也不是音乐唯一编舞答案。
- 保留图像结构prefix、视觉tags；空白占位的正文hidden丢弃后接新tail。旧text_embedding实际prepend，不等于本轮tail替换；prompt=None整体绕过图像编码也不采用。
- 真实音乐只进入新text adapter；原生reference/retake两路均固定相同长度静音，并保持repeat_clean尾部和有效time1。原生静音latent不是零latent，换音乐必须核实原生条件hash不变。真实音乐只在生成之后用于mux。
- 先同次teacher拆拼身份oracle，再独立结构prefix回填，核验raw条件、tags、packed和首步DiT预测；固定三个源人物joint/aswap各三。Teacher oracle只说明接口和条件可达性，绝不是训练好的音乐模型。
- 原始caption逐token序列不池化/不复制均值/不插值压成K槽；训练集预检2000条图均存在，train1800原文平均392.688 tokens、最长1309，Qmax仅由训练集确定。部署由音乐+安全图上下文预测长度并实际截取，不读GT长度，不把补零行当自动mask。
- 首轮公开MusicFM骨干冻结eval，训练共享adapter；如未解冻不称encoder已微调。TC/DiT/VAE冻结，无新增生成/FM/DiT/video训练loss。共享所有读取/裁剪/重采样/BN政策，checkpoint保存部署全部映射。
- 新代码为`ms/music_text_adapter.py`（原始token输出/长度/严格重载）、`ms/music_text_condition.py`（明确tail回填）、`ms/music_text_oracle_generate.py`（工具管理三节点串行队列）、`ms/music_text_cache.py`（TC-only逐token目标/CPU预检），既有sampler加入显式oracle和fixed_native_silence参数。原实验默认路径不改。
- 独立CPU复核已通过：全部部署层有限梯度、短/长query前缀误差1.55e-6、padding不变、checkpoint重载完全一致且拒覆盖。真实首个case的独立prefix和首步DiT预测均逐值一致，packed一致；完整六条视频及效果尚待完成，不提前宣称训练/音乐控制成功。
- 正式训练门禁：完成oracle技术验收并查看视频，再提取1800/val200新目标、small-fit、正式adapter训练与正确/错配/零音乐三case验收。失败保留证据，区分接口、拟合、泛化、生成效果，不无限扩数据扫参数。

### 首批joint oracle实际观察（阶段状态）
- 三条joint sampler exit0，完整124帧/24fps/704×1280及音轨解码通过，三人物×参考图+六时间采样帧对照图已实际查看。可以看到转身、步移、手臂变化；不能把采样帧观察等同连续听看或音乐正确度。
- 楼梯参考人物为女性浅黄上衣/灰裤，oracle输出男性外观、深色衣裤，人物保真失败。原版MS中的His/He可能提供错误性别语义，但无去代词反事实前不认定是唯一原因；过去原生empty也有参考衣着漂移。不要把接口精确回填成功包装成全部生成质量验收通过。
- N2两条TC-only教师缓存真实exit0，未加载video/audio VAE；实际tail462/479、prefix7437。只验证提取程序，不冒称全1800/200缓存完成。
- 原管理队列六条均exit0，全六124帧/音轨完整解码完成，各音乐条件的原生静音输入/ref/retake/mask hash完全一致。aswap六时间采样图已Read：楼梯浅黄上衣/灰裤及女性外观保留，其他两case也有抬臂/腿部/躯干动作。与joint的caption来源和长度都不同，不能直接把差别归因代词，更不能当训练音乐增量。
- 独立SFoT83三个joint已exit0且完整音画解码通过，同图/seed/native静音不变，仅teacher将He/She/His/Her等转换为the dancer，原动作/舞种/顺序保留并记录replacement。采样图实际Read：楼梯女性外观、浅黄上衣和灰裤恢复，仍有转身/步移/手臂动作；红衣和商店人物也基本保留。支持教师性别措辞是影响因素，不称唯一根因或普适保证。学生未加ID或其他语义文字，首帧未固定。
- 选择中性代词版为首轮教师目标，已启动N2/N3/N6/N7四分片TC-only提取完整1800train/200val。旧两条原代词smoke缓存不混入新训练。
- N2/N3两音乐分片均exit0，独立重读2000条并集、shape/finite和文件SHA通过；公开骨干冻结eval。缓存/在线部署统一FP16 roundtrip、32k读取/24k重采样。实测batch8缓存与batch1在线并非逐值相同：特征max abs .00390625/RMS .000198649，随机adapter max abs3.2544e-5，属待训练权重复核的数值偏差，不伪称完全一致。
- 真实MusicFM+随机adapter完整checkpoint经H3→隔离music进程部署，原始token输出逐值与保存前前向一致；未输入GTcaption/GTlength，最后投影实际执行。这是部署全链自测，不是训练或音乐控制效果。35步合成小集loss5.2639→1.4235，仅说明训练梯度可用。

### 接续训练配置与验收门禁
- 教师缓存管理`xoMaSk`四分片全部exit0，finished报告/唯一并集2000覆盖1800train/200val；中性原文train均431.154tokens、最长1423，val无超过train上限。这是报告和覆盖检查，完整tensor/hash仍由后续审计验收，不以marker冒充。
- `Vrzqj3`已从inotify等待进入全2000教师文件SHA重读；PID192857存活，CephFS `generic_file_buffered_read`，fd指向的教师PT持续推进。该IO等待不是GPU训练，不重复启动或删锁/kill正常队列。
- 随后N5 GPU7串行执行8条small-fit 300更新、再clean1800正式1875更新/val200，每批8、adapter lr1e-4；raw MSE用train教师能量作标量损失尺度，加cosine和log长度loss，部署输出仍原始hidden不单位化。小集回归不改善、梯度scope/finite错误、checkpoint完整重载或缓存/在线训练权重检查失败时停止，不越门禁正式训练。
- 全部candidate/last权重保存完整冻结MusicFM及adapter，不只保存训练head；训练生成前向同一最终投影。checkpoint输出strict重载必须逐值一致；cached/online训练后实测相对误差≤.01且预测长度相同，报告具体误差而非以形状宣称完全一致。
- 训练前审查补齐门禁：small-fit除rawMSE改善，还要求log长度loss改善及训练长度MAE≤max(2,平均长度5%)；验证选candidate同时计入长度loss。best和last均检查保存时输出、完整strict重载与online预测长度/误差，不用只验last代替所选best。验证超过train Qmax的文本只计可表示前缀并明确报告数量，不声称完整token重建。
- `OYdbCB`以真实训练pipeline PID192857/inotify等待正式验收报告，通过后自动核实所选checkpoint SHA→三节点joint/aswap六条部署→同三人物zero music joint三条→默认六条完整解码。zero只置零新text adapter音乐输入，原生音频保持固定静音，真实音乐仍用于mux，避免干预与播放混淆。生成后仍需零音乐完整解码/连续视觉和音乐效果/发布，不把后台注册当完成。
- 音乐隔离进程与H3共用GPU，新学生unit在完成TC前向并取安全图context后先offloadTC，再执行独立音乐进程，避免额外骨干冷加载OOM。原始图prefix输出保留，仍不通过caption查表。
- 该阶段配置/管理队列已经建立，不等于真实训练结束。生成效果由后续三人物joint/aswap正确/错配/零音乐验证，不能拿teacher oracle/检索/token loss替代。

### 用户要求复用旧完整链：免训练文本注入诊断
- 不必等新adapter训练才验证旧权重。补做已训练MusicFM6000/lambda0/1875的完整encoder→32Dadapter→时间mean/std→CLIPhead→L2，最后全局输出作为一个5120文本token，图结构prefix保留，两路原生音频固定静音，学生不输入任何语义文字。
- 旧checkpoint strict加载476个state项，包含adapter与head；N8真实全链自测通过，tail[1,5120]有限。训练1800文本mean/std估计pooled向量期望norm=272.4553528，用该单一全局尺度放大单位CLIP方向；没有inverse L2、没有逐token恢复、没有新训练恢复器。输出仍标准化CLIP方向而非原生rawhidden坐标，必须作为边界明确的诊断。
- 原训练前向的池化/末端投影均真正执行；不复制该token成K条，既不检索caption也不读生成GT长度。不能预先断言有效或无效，也不能以单token失败否定原生逐token主线。
- 工具管理AT358S在N2/N3/N6各GPU7串行三源人物joint/aswap六条；独立输出`val_musicreplace_text_legacy_fullclip_v1`。原Vrzqj3训练、OYdbCB后续生成继续，不取消已批准任务、不重复启动正常任务。

### 真实small-fit失败与恢复（原日志保留）
- 原Vrzqj3全2000教师文件SHA审计通过后，smallfit300实际运行但门禁exit1：训练长度MAE31.375>平均长度5%(20.875)，student/teacher能量比.0033186，rawMSE12.626仍接近zero目标能量13.855。正式训练未启动；OYdbCB因上游complete=false退出，未生成新模型视频。不是缓存损坏、也不是成功验收。
- 保留v1全部metrics/log/checkpoints；RcB9Vp独立小集恢复对照：相同8train/8val，rawtokenMSE+length，cosine权重0，lr3e-4/1000更新，长度末端bias由train平均log长度初始化。多个优化因素一起调整，仅用作修复拟合，不单独宣称cosine是唯一根因。
- 新增能量比≥.1门禁，原长度/改善/完整checkpoint/online门禁不降低。部署仍5120raw输出，teacher/student未加入身份文字；TC/DiT/骨干冻结不变。训练未通过不启动formal，不把更低cos loss当幅值对齐。

### 小集修复实际通过并启动正式训练
- RcB9Vp raw小集1000步exit0：train rawMSE5.50198、长度MAE0、能量比.405596；独立8val rawMSE5.68923、长度MAE102，不以小集验证代表正式泛化。参数变化包含cosine0/lr3e-4/更多更新/长度bias初始化，不能单归因cosine。
- best/last完整checkpoint、所有adapter梯度、cached/online相对RMS1.5386e-6和预测长度385一致验收通过。正式NFcYQG已N5 GPU7启动1800train/200val、1875步、batch8、lr3e-4/cos0/train均log长度bias，公开骨干冻结eval；新目录`music_text_train1800_raw_neutral_v2`，未重启失败v1。加载/审计未结束时不把它称有效GPU更新完成。
- 新生成followup读取明确formal_summary/新v2报告，旧失败OYdbCB不继续称存活，零音乐与三case输出树只在正式门禁通过后创建。

### 旧完整链六视频实际验收与受限结论
- AT358S六sampler均exit0，三人物joint/aswap完整124帧/音轨解码全通过，原生两路静音hash不变、旧CLIPhead实际部署执行、没有外部语义prompt。
- 六真实音乐旧全链embedding确实区分：同case原/换音乐cosine为-.14028/.14883/.07284；但两组六时间采样图实际Read，三人物整体动作顺序接近，只有细节和后段变化；楼梯仍深蓝衣裤/肩包，参考浅黄上衣/灰裤不保真。
- joint/aswap全124帧RGB MAE(.01866/.01744/.01026，像素0–1)不是音乐准确度，所有配对无完全相同帧。采样图不是连续听看、caption不是音乐唯一真值；目前未建立该单token尺度配方的可用音乐控制增量，不能把差异小归因旧head完全没运行，也不能否定所有原生逐token方案。
- 8766原入口已重载工具管理yonqEA，保留旧11模型/4预设，新增三列完整旧链/原raw教师/中性教师，合计14模型/3人物/6预设；页面/manifest200、15新增视频Range206核验。第一新预设【10-18 文本侧·旧完整CLIP链】，第二【10-18 接口教师oracle·不是音乐预测】。teacher不能冒充音乐预测成绩。
- 新正式NFcYQG仍正常读取teacher文件；已完成小集门禁不等于正式数据泛化或视频效果，新4GfX7J等正式报告接续生成。此处只结束旧完整链诊断，不缩小全实验目标。

### 正式rawtoken训练完成与生成恢复
- NFcYQG正式1800/200、1875更新exit0，实际15000曝光。最后val rawMSE5.381603、cosine loss.202404、长度MAE137.71、student/teacher能量比.454974；长度泛化不足，不能宣传完整caption恢复或生成收益。原始token幅值和语义回归确有学习，但trainprobe也长度MAE148.59。
- 约496.7万adapter参数，507tensor完整MusicFM+adapter权重；best=last1875。独立重读两权重SHA/finite和冻结范围通过，SHA cbb79c6338c2673d0c2f650a14da815234894055ffc14f0a56f1ec1afd1750fc。保存时/strict重载输出一致；cached/online相对RMS2.3863e-7、预测长度259一致。正式训练区间272.44s含过程val；finished361.10s含checkpoint/部署审计；49m26s launcher还含CephFS目标读取，不能把IO算GPU训练。
- 4GfX7J初次生成启动器在写plan前导入music_text_condition触发缺diffsynth(PYTHONPATH)异常，未进入采样；保留失败日志/空输出树。修复为通用原子JSON写入不导入模型模块，fp88pX使用独立v2输出树/新recovery_v3报告恢复，绝不覆盖原失败目录或重训。
- 恢复后只用音乐/参考图、两路固定静音、同完整adapter和预测长度，default joint/aswap六条+zerojoint三条。实际视频验收/效果/网页后续独立结束，不以训练exit0冒充完整实验成功。

### 正式回归审计：token误差下降未建立音乐依赖
- 新`music_text_regression_audit.py`固定同一图结构context，val200分别正确音乐/一位置循环错配/真实静音MusicFM编码；train1800原始target构造逐位置均值和全局均值基线。严格原文件SHA/teacher协议沿用，目标mask相同，不增加生成case或训练。
- tokenweighted rawMSE：正确5.38160354、错配5.38160507、静音5.38213766、逐位置均值4.75674750。宏平均正确5.53759473、错配5.53759485，差1.19e-7；模型还不如逐位置平均target。模型可改变length(验证MAE137.71，比固定train平均长度431的MAE155.57略低)，但rawtoken内容对换音乐不敏感。
- 正确/错配输出RMS宏平均.002778、正确/静音.012215；这不是DiT动作/音乐准确度。不能把cosine≈.795和相对zero MSE下降归因音乐语义学习；更像主要拟合文本位置/共享分布。多种正文写法/原始词位置不等价、直接回归平均化是后续训练目标需要修正的候选原因，尚未实证唯一根因。
- 保持fp88pX已运行六条正确/aswap+三zero生成，完整效果核验后决定下一训练对照；不跳过负结果继续扩量，不擅自新增video/DiT预测loss。

### 正式九视频完成与本轮结论（2026-10-18）
- fp88pX所有9生成exit0；default joint/aswap6＋同三参考图zerojoint3均独立124帧/24fps/704×1280完整音画解码及log/input审计通过。全部原生静音waveform/ref/retake/mask hash相同，无外部语义文本，无teacher/GTlength输入，训练最后映射确实部署。原失败启动v1目录/log保留。
- 实际Read三套参考图＋六时间采样图：原音乐/换音乐/零音乐都保持相近小手势、后勾腿、脸/头侧抬臂、单臂重心等母题；楼梯变白蓝运动衣裤/肩包，仍非参考浅黄上衣/灰裤。采样视觉不是连续听看或新用户认可，画质/动作变化不能包装成音乐准确度。
- 原/换音乐全帧RGB MAE .02822/.01068/.02124，原/零 .05569/.01546/.03152；数值不是动作/音画对齐准确度。换音乐还改变token长度449→410、271→265、305→312，zero297/208/202；packed随长度变化，本轮未排除长度改变引发视频差异，不能据此证明token内容学习音乐。
- 新8766第一预设【10-18 正式文本token·仅音乐+参考图】，原生空文本/旧完整CLIP链/新rawtoken/新rawtoken零音乐，固定三case；joint四列，aswap零音乐列明确缺失(只做三joint，不自动扩case)。旧结果保留，16模型/3人物/7预设，页面/manifest200及9新增视频Range206通过。
- **本轮技术闭环完成，效果为负结果**：旧完整链能注入但单token未建立增益；原生rawtoken训练/部署同链已实现，直接逐词位置回归主要拟合共享文本分布/均值，正确音乐未优于错配且不如train位置mean。生成也未建立可用音乐舞蹈增量，参考保真不足。不能把所有失败归因丢adapter，也不能否定文本侧全路线。
- 本轮训练/推理/完整解码/表示依赖审计和采样视觉分析均实际完成；没有宣称连续音画人评、穷尽偏好打分或三个case统计显著。不继续盲目扩数据/扫参数，后续应针对序列监督/音乐条件依赖设计对照；如采用生成预测loss需独立授权，ID/首帧/A输入端仍后置。

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
