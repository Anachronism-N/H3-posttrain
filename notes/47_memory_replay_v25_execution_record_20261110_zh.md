# 固定前缀记忆重放 v25 执行记录

原实验记录从实验总方案移至此处保存，以下执行记录正文保持不变。

正文中的“本方案”及章节编号指[迁移前的归档版本](https://github.com/Anachronism-N/H3-posttrain/blob/51c1b1c2a228306b2d570e2bd4292d498d9c371a/notes/44_cvpr_experiment_matrix_20261110_zh.md)；后续实验按[统一实验执行方案](44_cvpr_experiment_matrix_20261110_zh.md)安排。

## 14. 服务器开发实验：三处固定前缀的记忆重放（v25）

执行基线 `2e8bc71`。按本方案第12节先做E5小探针：三首已有开发歌各预选一个已知复制/过密块，比较原full历史与metadata_only，共上限**6次文字调用**。这是源请求重放，不是重新生成整曲，不做新音频采集、大纲、诊断、critic或训练，也不把开发旧歌当新测试。

### 14.1 冻结控制与实际干预

| case | 原规划块 | 范围 | 预选依据 |
|---|---|---|---|
| fma-1881 | pu0018 | 173.46–177.26秒 | 长动作正文在1.25秒槽中复制，含多次状态转换 |
| fma-3855 | pu0018 | 168.855–171.14秒 | 0.75秒内pivot/hop/lunge等顺序动作过多 |
| fma-4151 | pu0009 | 82.265–92.39秒 | M8 freeze短语在邻近多个块中原样复用 |

两臂读取**同一个历史原前缀**，不使用full本轮新输出为metadata_only的前缀；候选库、当前主题定义/全曲section意图、原音乐结构、时间菜单/响应schema、源seed、1024-token上限和系统prompt一致。实际服务为同一Qwen3-VL-32B-Instruct、BF16 TP=2；本轮两臂同环境，full不保证与旧v11服务逐位复现。

metadata_only仅删除以下三处event.action，不另加反复制指令，不截断或摘要正文：recent_committed_events、各body_parts.latest_event、各motif_memory.recent_examples。其余时间、轨道、ID、exit、gap、计数和原前缀SHA保留，outline/motif.description及原候选照常可见，因此不是完全无动作语义。JSON序列化格式与原请求保持一致；源前缀SHA绑定完整原前缀，不伪称投影后哈希。

新入口 `choreobench/memory_replay.py` 在派发前验证原contract/result、已提交前块/目标块、started/raw绑定、请求中的历史/主题/候选/时钟，以及原响应确实编译为保存块。随后使用同一机械验证器检查新输出，保留非法/截断/空输出及真实成本，不自动修复、换seed或补采；基础设施/未知结果阻断后续请求。相同输入/代码/provider才允许恢复。

命令与输入合同见 [ChoreoBench README第16节](../choreobench/README.md#16-固定原前缀的历史内容重放有界开发探针)。该入口是独立有界开发工具，不是已经实现run-harness全曲metadata_only模式；boundary_compact、时长相关动作预算和oracle内容编辑仍未执行。

### 14.2 验证与原件保存

- 合成新专项8项通过，关联coordination/history回归共65项通过；覆盖三条历史action路径移除、同prompt/seed/token cap、固定前缀、源文件错配、非首块、重复case、非法输出/基础设施错误和零新调用恢复。
- 本轮全量CPU回归：**1322 passed、4 optional browser skipped、148 subtests passed**，显式禁用CUDA；不把软件测试数量当编舞质量指标。
- 实际文本基座25个权重/config/处理器文件与之前已核验工件逐个SHA一致，加载回执、vLLM日志、GPU身份另存。普通API身份字段不是远端权重认证。
- 证据在仓库父目录 `choreobench_runs/20261110_memory_replay_v25/`：预选输入、固定请求和原件哈希、源码快照、started/raw、原prefix/原目标events/新events、新prefix＋块plan、成本、恢复及资源回执。仅代码、文档和可公开汇总推送，不上传音频、私有路径/日志或凭据。

逐字历史复制、字符数或较短正文只作诊断；另直接读取完整新动作和真实前段内容，判断是否减少冗余串、是否保住合理衔接、短槽是否仍过载。助手初审不是直接听音或人工盲评，本局部探针不能证明全曲长期质量和误差累积改善。具体六调用结果及结束资源状态在本节后续执行表中记录，不用历史accepted或测试通过替代。


### 14.3 已完成的六调用结果

六次均正常停止并通过原机械约束，两臂分别输出九事件，实际每个case都选择同一三事件菜单。没有非法、截断、未知结果或未知usage；同条件resume结果完全一致、**0新增调用**。原v11/v22结果与候选不覆盖。

| case/模式 | 原样复制此前action的事件数 | action总字符 | 新事件数 | 实际输入/输出tokens |
|---|---:|---:|---:|---:|
| fma-1881 full | 3/3 | 1097 | 3 | 6176 / 445 |
| fma-1881 metadata_only | 0/3 | 473 | 3 | 4647 / 383 |
| fma-3855 full | 1/3 | 1219 | 3 | 5906 / 479 |
| fma-3855 metadata_only | 0/3 | 464 | 3 | 4167 / 373 |
| fma-4151 full | 0/3 | 1150 | 3 | 6948 / 475 |
| fma-4151 metadata_only | 0/3 | 651 | 3 | 5646 / 370 |

逐字复制定义为新event.action与**完整固定原前缀中任一event.action的Unicode字符串相等**，不是归一化词汇相似度、语义质量或病理重复率。full合计4/9、metadata_only为0/9；字符由3466降到1588。但原声full虽然0个逐字匹配，仍明显重用原freeze结构并追加措辞，说明该指标会漏掉近似复制；不能以0复制当成功。

成本合计 **33490输入tokens、2525输出tokens、6请求**；full为19030/1399，metadata_only为14460/1126。API返回延迟合计136.74秒，不含加载/哈希/控制器耗时；不换算成完整端到端GPU小时。输入量不同是历史移除干预的自然结果，不称等实际token成本比较。原音频候选/大纲成本另外保留，本轮没有重采。

### 14.4 实质动作初审，而非以“短/不同”判胜

- **电子case：局部冗余减少，但短槽仍紧。** full完整复用长串，174.71–175.96秒包含body roll→肩背isolation→低lunge→pivot→手臂sweep→再次chest isolation；metadata_only缩为“body roll后肩背isolation”，去掉多个先后转换。第一/第三槽仍包含起身、chop/twist/步法或下沉/方向改变/肩部wave/arm thrust，1.25–1.3秒是否足够仍待更明确预算。不是去重即证明可执行。
- **摇滚case：仍未解决关键时间负担。** 169.605–170.355秒的metadata_only仍写“dynamic spin landing on one leg, followed by staccato arm pumps and floor touches ... head snaps”；仅0.75秒、含转身落地再多次动作。它减少了full的pivot/hop/lunge串，却仍过密。相邻两个槽又复用arm pumps/floor touches/head snaps短语；零“完整旧action匹配”并不意味着语义重复消失。
- **原声case：有局部结尾内容变化，连续性未被证明。** full末槽继续“freeze→foot tap→arm thrust→pause”，metadata_only换成一腿后伸、前倾、手指模仿strumming、弯膝并保持stillness；不再机械复用M8长句，结尾操作更少。但前一原事件状态为空，不能断言新首槽step-hop与此前freeze之间真实姿态衔接已成立。乐器模仿/末strum也是候选文字启发，未由本轮直接听音确认。

两臂全是body轨道，JSON时间/主题/引用合法且原前缀相同；这些只证明合同与局部比较有效。**初步支持减少完整历史正文能减轻本批块的长串照抄，不能直接宣布metadata_only全曲最优或音乐贴合提升。** 目前是三个专选问题块、单seed的开发探针，不覆盖未见歌曲/整曲轨迹/记忆与预算交互。两臂共同仍选三个槽，当前提示与菜单的事件数量偏好未随去历史解决。

### 14.5 资源已释放与下一步

实际占用本节点GPU0/1启动文本BF16 TP=2，2–7不动；六调用和resume核验后按完整argv、CUDA与稳定starttime确认本次服务再SIGTERM。已确认无本次模型进程驻留，两张使用卡恢复**各一个holder**，八卡约975MiB占卡；旧CPU评审服务保留，没有新建近乎相同人评页。

下一项优先做**时长相关动作复杂度**与等长度控制/oracle内容编辑的小探针，重点保留并发动作、减少不可能在短槽内顺序完成的转换，允许选择少phrase；不是再提高最短时长阈值。boundary_compact可用于研究保留直接边界动作与复制的权衡，但不未经实验就把全部历史禁掉。暂不扩整曲、音频重听或训练，不因这批字符变少就写质量胜利；助手实质初审先行，真正争议的音乐贴合和正式推广仍另做独立评价。
