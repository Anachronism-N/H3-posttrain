# 下一步实验计划与部署验收

当前工作针对完整音乐到文本编舞。先复用已有5秒编舞模型和冻结文本模型，验证整曲组织与编辑，不启动新的SFT、DPO或RL。H3渲染、人体恢复、真正乐句检测与RSI不是本轮已经完成的功能。

本轮承接 [c1bba3c审查](CODE_REVIEW_20261105_zh.md)，修复来源/评估与服务状态问题，并增加实际checkpoint验收、可编译动作时间与最短时长检查。旧三首的真实实验数字仍属于旧版本开发记录，不因代码修复自动成为新版本结果。

升级时冻结并保留旧运行目录，重启采用新源码的模型/评审服务，并重新核对health身份和客户端pin。新代码、模型、检测合同或配置使用新输出目录，不把旧journal迁移成新版本成功记录。

**本轮交付与明确的边界**

| 工作 | 本轮行为 | 仍需服务器或人工完成 |
|---|---|---|
| 空目标编辑 | 无字面目标不生成虚假联合成功；全部目标为空且约束保持时not_evaluable | 语义目标的独立评审、合理弃权与已满足目标的区别 |
| 音乐结构来源 | 版本化检测输入，显式来源/时钟/合成标记，保留旧输出的迁移证据 | 原始检测器是否真实运行及模型/输入身份核对 |
| 原文复制诊断 | 实际比较动作与引用输入的raw/解析片段，不再固定声称非原文 | 改写不等于动作创新，主题回归与复读仍需语义判断 |
| 模型加载 | 失败实例不能跳过校验继续使用；常驻服务仍要求重启 | 真实checkpoint的加载峰值、EOS和输出格式 |
| 人评音频 | 请求核验实际内容，并服务同一份已验证字节，含Range/HEAD | 真实听音、独立评分和评审一致性 |
| 动作时间 | 可输出块内boundary ID，由控制器编译为原采样时钟；可设置最短时长 | 时间网格不是拍点或动作边界；阈值需开发校准 |

**动作时间的新接口**

新配置 [harness_coordinated_grid.json](../examples/harness_coordinated_grid.json) 使用：

```json
{"event_time_format":"boundary_ids","event_grid_seconds":0.25,"min_event_seconds":0.5}
```

块请求中的event_clock公开精确的boundary_samples。模型返回 `start_boundary/end_boundary` 的整数索引，控制器转换为最终plan的全局start/end采样下标。它不改变原音频、观察窗、公共段落或候选合同。时间网格按最近采样点取整，恰好半采样时向上；真实unit.end始终是最后一个边界，尾部不补零或补到下一格。每块最多4097个边界，避免极密网格失控。

最短时长向上取整到采样点，并在解码后复检。示例0.5秒是待校准的开发规则，不是人体物理阈值：不满足则该歌失败并保留原响应/成本，不自动拉长、补rest、补采或重试。如果某个真实规划块整体短于下限，预处理即记录整曲失败，模型新增调用0。短促accent可能需要不同任务配置，不能在看到测试结果后临时豁免。

原 `event_time_format=global_samples` 保留，`min_event_seconds=0` 保留原来的至少1采样点检查。这些默认值支持原协议；新例子显式开启网格和下限。改变时间格式、下限、模型或来源后必须用新目录，不能恢复旧journal。

下限用于coordinated生成，不自动重判旧计划或约束既有编辑patch。编辑如果改变时间，仍应独立检查时长和动作语义，不能把生成时通过的规则当成编辑后仍成立。

诊断 `literal_source_copies` 给出逐字匹配数、比例和事件ID；只比较所引用输入的原回答或已解析片段。非引用输入相同、语义相近或简单改写不在此字面指标定义中。它不能充当新颖性或音乐性评分。

**服务器执行顺序**

以下为Linux服务器示例，`/models`、`/data/choreobench`、`/runs` 都是需按实际环境替换的路径。仓库不提供私有权重或音频，不自动下载模型。

1. 准备合法完整歌曲及canonical音频、dataset.json、实际基座/portable checkpoint/processor和匹配源码。先做普通校验；正式发布前再通过分组、近重复、许可等release检查。

   ```bash
   python -m choreobench doctor
   python -m choreobench validate-data --dataset /data/choreobench/dataset.json
   # 仅当来源和分组已完成真实审计时检查正式发布条件。
   python -m choreobench validate-data --dataset /data/choreobench/dataset.json --release-check
   ```

2. 从 [native_server.json](../examples/native_server.json) 准备实际配置，路径和max_length必须与checkpoint匹配。已训练checkpoint不能同时使用task_prompt覆盖训练协议。验收输入必须是真实完整5.0625秒窗，不在验收入口内部裁切或填零。

   ```bash
   python -m choreobench.acceptance \
     --provider /data/choreobench/native-checkpoint.json \
     --artifact-type portable-checkpoint \
     --audio /data/choreobench/acceptance-5.0625s.wav \
     --out-dir /runs/checkpoint-acceptance --max-output-tokens 512
   ```

   accepted只覆盖声明权重身份、真实加载校验、EOS和严格双窗格式，不代表编舞质量。先batch1；记录峰值显存、加载期RAM、token及延迟。原基座提示对照应独立使用base-only类型及新输出目录，不能沿用checkpoint验收结论。

3. 常驻部署同一原生专家。调用端与服务端必须看到同一绝对音频路径；候选采集输出放在允许的audio-root中。读取 `/health`，把identity_sha256填写到客户端配置，token通过环境变量提供。

   ```bash
   python -m choreobench serve-local \
     --provider /data/choreobench/native-checkpoint.json \
     --audio-root /runs --host 127.0.0.1 --port 8765
   python -m choreobench run \
     --dataset /data/choreobench/dataset.json \
     --config /data/choreobench/native-service-concat.json \
     --out-dir /runs/checkpoint-collection
   python -m choreobench export-candidates \
     --run-dir /runs/checkpoint-collection \
     --out /data/choreobench/checkpoint-candidates.json
   ```

   客户端配置参考 [native_service_concat.json](../examples/native_service_concat.json)。先核算窗口数乘K是否在max_calls内，不以出错后补采筛掉坏输出。候选采集完成可停止音频模型；之后的编排复用候选，无需重新加载Qwen。

4. 在同一canonical音频上准备可追溯的检测记录，再构建独立public sidecar。v1检测记录与旧JSONL的转换方式见主README新增说明；必须显式提供检测器身份和synthetic声明，不根据文件名推断真实来源。原文件不覆盖，转换不重跑音频模型。

   ```bash
   python -m choreobench.music_structure \
     --dataset /data/choreobench/dataset.json \
     --detections /data/choreobench/detections.v1.jsonl \
     --split dev \
     --out /data/choreobench/public-structure.json
   ```

   混合清单下，--split与harness所选splits一致；检测文件仍完整覆盖原dataset，输出sidecar只包含所选歌曲并保留原dataset SHA。旧记录封装示例：

   ```bash
   python -m choreobench.music_structure --wrap-legacy \
     --dataset /data/choreobench/dataset.json \
     --detections /data/choreobench/detections.old.jsonl \
     --out /data/choreobench/detections.v1.jsonl \
     --detector-identity /data/choreobench/detector-identity.json \
     --synthetic false --legacy-time-unit seconds
   ```

   --synthetic false必须有实际检测来源依据，不是迁移默认值；fixture应声明true。源文件、原行和可选运行回执的hash会保留，矛盾声明拒绝。

5. 准备两份本地配置：来源于harness_index0.json的固定候选对照，及harness_coordinated_grid.json的逐块合成。二者填写完全相同的public_structure_path；后者填写真实文本服务、token上限和总调用预算。

   ```bash
   python -m choreobench run-harness \
     --dataset /data/choreobench/dataset.json \
     --candidates /data/choreobench/checkpoint-candidates.json \
     --config /data/choreobench/index0.json --out-dir /runs/checkpoint-index0
   python -m choreobench run-harness \
     --dataset /data/choreobench/dataset.json \
     --candidates /data/choreobench/checkpoint-candidates.json \
     --config /data/choreobench/coordinated-grid.json --out-dir /runs/checkpoint-coordinated
   ```

   每首外层预算为1次大纲加规划块数；示例max_calls=100不是任意规模数据的预算保证。上下文需用真实服务的tokenizer检查，max_request_characters不是token限制。原候选采样成本与本轮新增文本调用分别报告。

6. 输出盲评包，保留失败歌曲分母；使用可识别的独立评分者听完整歌曲，评分结束前不查看方法绑定。

   ```bash
   python -m choreobench review-pack \
     --dataset /data/choreobench/dataset.json --split dev \
     --run index0=/runs/checkpoint-index0 \
     --run coordinated=/runs/checkpoint-coordinated \
     --out-dir /runs/checkpoint-review --copy-audio
   ```

   这条命令导出评审包，不会自动生成质量分或完整网页；页面生成与评分者部署使用实际公开资产，私有方法绑定不发布。评审需分开音乐贴合、动作明确性、段落对比、主题回归/变奏、非预期重复、过渡与编辑语义。

**实验规划与升级条件**

| 阶段 | 工作与产物 | 通过条件 |
|---|---|---|
| 软件回归 | 五项缺陷负测、checkpoint stub验收、网格端到端和恢复 | 正常路径及反例均通过；无真实模型质量宣称 |
| 已训练模型验收 | 单真实短窗的identity/EOS/格式/资源报告，再运行原三首 | 真实checkpoint校验通过；所有失败保留，不用基座代替 |
| 开发诊断 | 三首独立盲评、短动作/缺口/编辑错误清单 | 找出可复现的质量问题并冻结下一版规则；不作显著性/领先结论 |
| 新歌pilot | 例如另外6首开发、6首预先冻结留出；按歌曲来源组隔离 | 检查音乐覆盖、评分一致性和成本；小样本不证明正式统计功效 |
| 正式benchmark | 依据pilot方差、效应及标注预算确定规模，冻结协议/模型/私有答案 | 合法授权、分组/近重复审核、独立测试、可审计评价和歌曲组统计 |

第一轮方法比较使用2×2设计：原始基座与已训练专家，各自比较index0和coordinated。同一专家内部共享同一候选库；两个专家必须分别采样，不能伪称跨模型共享完全相同候选。随后再比较有无大纲、有无历史、无结构/原始预测结构/打乱结构；一次只研究明确变量，并对齐总预算。

编辑先做明确段落、时间、身体范围的任务，每条保留私有字面/语义目标、保护范围和预先许可的修复区。无字面目标时联合字面成功为NA；仅有语义目标时等待独立评审；两类目标都空时not_evaluable。不能把NA记0或1混入数值bootstrap，应明确可评任务分母。已有目标原本满足时，合法no-op可以通过字面目标检查；“是否应弃权”另作任务定义。

**资源与停止条件**

实际checkpoint首验沿用单80GB级GPU和128GiB级RAM作为容量起点，不构成峰值保证。已有文档中的基座测量不能替代checkpoint测量。文本服务可用独立服务/API；若沿用已验收的VL32B BF16 TP=2路径，需两张相应GPU，不能假定与音频专家同卡共存。CPU负责音频预处理、结构后处理、编排、日志和盲评交换。

公共评审服务每个音频请求读取并核验完整有界文件，再从相同bytes返回Range/HEAD；临时内存与文件大小及并发数相关，不再仅以元数据担保内容身份。它不会持有全库音频常驻内存或向服务器提交评分。

缺少合法checkpoint、真实音频或服务配额时，只能完成相应软件验证。模型身份不符、输出超预算、时间规则不可满足或请求结果未知时停止对应路径并留证，不回退mock，不隐式重试，也不启动补训。

RSI安排在质量评审和冻结留出集之后：明确改进的是提示、工具选择还是规划策略；搜索只使用开发集，独立验证通过再准入新版本，并保留全部尝试成本和回滚点。当前交付不把单次重写或多次手动调参称为RSI。
