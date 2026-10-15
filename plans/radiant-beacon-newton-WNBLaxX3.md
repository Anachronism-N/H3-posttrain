# Hybrid BF16 双向音乐舞蹈与社区 CFG 对照

## 目标与已确认选择
- 用户选择 **Hybrid BF16**：复现开源合并规则，在此底座上继续训练双向音乐控制舞蹈。
- M1abR/M2bR目前只能证明输出不同；先不判定音乐encoder改造有优势，也不新增encoder实验。保留已运行训练，不擅自终止。
- 双向未经用户人评认可，不启动ODE生成、AR训练或蒸馏。推理固定同3个case，每条件3条；joint+aswap共6视频，不做全量。

## 当前策略与候选依据
当前 H-A4/5/6 = Ref2VA + 训练时DiffSynth DeCFG助手 + 富文本 + 视频LoRA rank64/lr5e-5；音乐latent作为clean条件(mask0)，只算视频FM损失，未启用guidance loss。H-A5/6只改分辨率/插帧。旧H-A3用清单caption，不足以排除富文本guidance路线。

采用代码 https://github.com/scottmudge/ComfyUI_MinimaxH3HybridLoader ，对应公开权重 https://huggingface.co/smhfacct/Minimax-H3-fl2va-ref2va-hybrid-models 。其推荐b25–49为FL2VA全底座，只用Ref2VA blocks25..49的 `adaln_proj.linear.{weight,bias}`，其余含final AdaLN和输出头仍用FL2VA。这是作者经验合并，不是后训练，也不预先声称舞蹈更好。公开INT8不能转BF16恢复原精度；从本地官方BF16原权重按同规则重建，并记录来源与hash。代码遵其许可证、权重遵H3许可证。

## 实施步骤
### 1. 最小代码与权重准备
- 扩展现有 `diff/scripts_omnidance/exp/h3ar/fuse_adapter.py`，提供明确的hybrid merge模式：流式按分片替换严格限定的50个AdaLN张量。检查两源config/key/shape/dtype相容，非选中tensor不变；禁止覆盖原始模型或已有实验产物。底座输出用新目录，必要新权重文件属于任务所需。
- 保留现有LoRA融合行为；对ostris v1复核alpha/r、role-major到head-major QKV转换，融合助手只在训练使用。
- 在 `DiffSynth-Studio-upstream-20260930/diffsynth/diffusion/loss.py` 增加仅新任务显式启用的direct guidance实现：`target = stopgrad(u) + g*(target-stopgrad(u))`，g=3.5，FP32计算；旧inverse实现默认不变。u为当前模型含LoRA的无梯度文本negative分支，不称冻结teacher；参考图及clean音乐仍保留，因此这是文本CFG，不是音乐CFG。
- 通过H3训练入口 `examples/minimax_h3/model_training/train.py` 显式记录loss模式/seed/max_train_steps；`diffsynth/diffusion/runner.py`对新任务提供同步optimizer-step硬上限，旧任务无上限行为保持不变。不要编辑正在执行的bash副本；新启动用冻结副本。
- 核验所选训练ID正/负侧车全覆盖、Lp边界、锚点与packed shape/位置。复用已有480x832侧车；本轮不做HR guidance，避免复用硬编码尺寸的negative缓存。

### 2. 两条新双向训练（核实空闲节点后）
优先使用此前候选N3/N4，但不假定现时空闲；仅8卡单机，N7+N8现有任务不动。
- **HYB-G3.5**：Hybrid BF16 + direct guidance3.5；不叠加未经验证的Ref2VA DeCFG助手。训练和推理均用Hybrid底座，不丢弃Hybrid。
- **OS1-G3.5**：原Ref2VA + 冻结ostris-v1助手 + 同direct guidance3.5，复现akatz/ai-toolkit的CFG组合思路；推理回原Ref2VA，不带助手。
- 共用完全相同精选训练IDs、rich prompt、480x832/124帧、clean音乐、rank64/lr5e-5、随机seed、8卡global batch8；关闭TF/MERT/MoE/DPO。首先300个optimizer updates，保存100/300与完成状态，按实际训练样本数记录，不把单机300与双机300当等预算。
- 这两组是**整套方案比较，不是单变量CFG证明**。若需解释CFG增益，用同ID同底座同助手的FM(g=1)顺序补对照；不拿H-A4 full12k与小集结果宣称某loss胜出。
- Hybrid训练前只检查3case接口/参考身份/clean音乐路径能运行，不要求未训练舞蹈效果先获通过才允许用户已要求的SFT。

### 3. 推理与网页
- `q2_20261010.sh`增加明确新双向job；`train_ha2.sh`透传入口参数并记录正常/异常退出；确认缓存和侧车后才启动。
- `eval_sched.sh`增加显式Hybrid底座模式，`eval_jobs.txt`加100/300的rows3；OS1推理原底座。沿用bi50步、shift12/3、cfg1。
- `sample_h3_ar.py`已有--w可复用，不增加无关抽象。若所有评测卡用于训练，不挤占/终止训练，等空闲后执行3case。
- `review_web/server.py`增加清楚预设：原底座/H-A4/Hybrid训练前后/OS1，标明预算差异；只报实际视频，不把未生成checkpoint显示为已有。
- 人评看手脸细节、人体结构、动作幅度、参考身份与换音乐的真实响应，不用拉普拉斯指标直接替代质量判断。先通过双向，再讨论encoder增强。

## SSH与资源校验
- 已核实失败来自探索代理额外 `StrictHostKeyChecking=yes`，不是已证实集群配置坏；系统当前为Port36000、UserKnownHostsFile=/dev/null、StrictHostKeyChecking=no。不能把弱化校验当修复，也不能等价重试已失败命令绕过检查。
- 不盲目修改全局SSH配置、ssh-keyscan后直接信任，或在规划中继续远程尝试。实施时只能用可信来源主机公钥建立特定known_hosts；本地宿主公钥可从 `/etc/ssh/ssh_host_rsa_key.pub`验证，其他节点须用户/平台可信来源。若无可信材料，先做本地准备并报告远端启动阻碍。
- 修复可信访问后再核实GPU compute PID，区分holder与训练/采样；确认节点内存/磁盘/cache/weight和主机是否空闲。新任务启动仅释放目标节点holder，不终止现有任务；后台进程走受工具管理的启动方式，不自行nohup/setsid脱管。
- 实施核实：本地平台注入公钥严格匹配八节点，已使用 `/tmp/od_known_hosts_20261012`。N3/N4现有缓存不同realpath，通过 `cfg_rich_cur/<rank>/<i>.pth`统一2091精选IDs；正负侧车remap后全量校验通过，两节点每条cache SHA256完全相同。repeat=2保证足够300次globalbatch8更新（DataLoader原总523，但代码严格max_train_steps300）。
- OSTv1 208模块融合完成；N4原采样正常收尾后已开始direct3.5双向训练。Hybrid BF16从ceph读取很慢，N1合并进行中，不使用不完整底座启动；N3等待。H-A4已750正常结束，空闲N7恢复3卡评测。
- Hybrid专用队列 `eval_jobs_hybrid.txt` 配合 `JOBS=... eval_sched.sh`，无Hybrid manifest的worker不认领，避免原底座错误回退。新推理脚本要从更新后的冻结副本启动。
- 用户要求所有空闲节点开展训练：N7增加纯Ref2VA+direct3.5（无助手），N8增加ostris-v1助手+FM(g1)，同globalbatch8/300更新/seed42。N3 Hybrid完整传输并校验13分片SHA256及节点本地辅助symlink后已开训。
- N7/N8的旧f0编号与N1不同，不能套N1侧车：初次尝试因subshell在`||`列表中使set-e失效，已停止仅这两项不合格新任务；修成显式exit守卫，避免在OR-list包裹subshell。原模型输出H3REF_rich_g3p5_cur/H3OS1_rich_fm_cur作废不得评测。N1权威2091缓存和正负侧车复制至N7/N8新目录 `/opt/h3cache/cfg_raw_f0_v2`，身份映射 `/opt/h3cache/cfg_meta_n1_f0.json`；全量哈希与N4一致后用 `CFG_CACHE=/opt/h3cache/cfg_raw_f0_v2 CFG_READY=/opt/h3cache/.cfg_rich_cur_v2_ready CFG_OUT=H3REF_rich_g3p5_cur_v2` 或 `H3OS1_rich_fm_cur_v2` 重启。不覆盖旧f0、侧车或实验输出。评测只取v2输出。
- 用户要求高分辨率先人评：H-A5hr step100现有6joint/6aswap（旧产物），只需先review前3case；300尚未生成，不将高分辨率预判成改善。当前H3AR已clean music视频only训练和采样，不生成/更新/解码audio，音频out head仍计算但忽略，保留音频条件tokens。
- v2修复完成：N7/N8新实体cache及sidecar全量2091验证，SID/cache SHA256与N4完全一致；两项v2训练已重启。所有64卡投入训练；现阶段没有评测卡，不用抢卡，等最先完成的节点做3case。监控job更新为`1cfc4295`，替代已删除的`311462d2`；Hybrid任务i2JZzJ、OST1任务SDL2kK、REF35_v2任务Hc3yZn、OS1FM_v2任务o9o6ux。

## 2026-10-13 完成首轮后的资源安排
- 首轮Hybrid-G3.5、OST1-G3.5、REF-G3.5-v2、OST1-FM-v2均停在300更新且正常退出，最终张量有限性和SHA256全通过；M1abR708与M2bR886最终权重也已验证。M2bR模型已完整保存886，但共用shell在执行中被修改导致尾部EOF错误，不能把shell日志的done当模型退出码；不重训已完整模型。以后运行中的shell及其子脚本均使用冻结副本。
- 新五组同2091数据/globalbatch8/seed42/300更新：N1 **DS-FM8**（DiffSynth助手+普通FM，取代未真正启动的4卡方案）；N3 **Hybrid-FM**；N4 **DS-G3.5**；N7 **REF-FM**；N8 **REF-inverse3.5**。已实际启动并产生有限loss更新，勿重复启动。对应日志`/tmp/q2_{ds_rich_fm,hyb_rich_fm,ds_rich_g35,ref_rich_fm,ref_rich_inverse35}.log`。
- N5高分辨率、N6无插帧原训练仍在跑，不中断；不追加新的encoder实验，不用尚未人评可用的双向做ODE/AR。
- N2已释放，Hybrid13分片传输SHA256及本地Ref2VA辅助链接已验证；正在运行`bash /tmp/eval_bidir_review_n2_20261013.sh`，日志`/tmp/eval_bidir_review_n2.log`，六卡0..5每3分钟错峰，固定3case每条件；首任务OST1-G300。不要重复启动第二个worker；新增启动尝试因检测到现有采样而已退出。
- 已生成OS1-G100、REF-G100、OST1-FM100各3joint+3aswap且全格式可解码，需刷新网页供人评；首轮四种300结果及Hybrid100仍待评测。专用队列`eval_jobs_bidir_review.txt`优先首轮最终300，之后新对照100/300，不排AR或全量。
- 目前有效利用目标：七节点56卡训练，N2六卡集中评测（需错峰完成，启动期间不能说六卡都已计算）；N2另2卡不适合独立大模型训练，按主机内存预算决定是否加采样，不盲目占满制造OOM。
- 09:08已核五组新训练持续更新且全部loss有限，无Traceback/OOM：DSFM8第6步、HybridFM第7步、DSG3.5第5步、REFFM第7步、REFinverse3.5第5步。所有启动脚本使用冻结副本，seed42/globalbatch8/accum1/max300，4卡方案未启动不再安排。
- 网页09:06已确认OS1G100/REF35100/OST1FM100各3joint+3aswap可见且新对照预设生效。监控旧`1cfc4295`已不存在，当前为`f5c5a28b`（每15分钟，会话内、最长3天），覆盖新5组训练和N2最终评测；全部实验结束和结果展示验收前不取消。

## 验证与完成标准
- CPU单位测试：g1退化为普通FM；固定u时direct MSE = inverse MSE*g²、正支梯度同scale且负支无梯度；不可据此称优化全等价。
- Hybrid全部选中tensor逐值来自Ref2VA，非选中来自FL2VA；源权重不变；ostris融合布局/倍率验证。
- Python编译、bash语法、正负sidecar全覆盖断言、同ID顺序/global batch/max optimizer step验证。
- GPU冒烟：finite loss、rank同步、LoRA可训练/助手冻结、clean音频不加噪，100/300检查点完整可读，异常退出不标done。
- 新实验正常结束并生成3case网页结果后汇报；未通过人评不启动下一阶段。
