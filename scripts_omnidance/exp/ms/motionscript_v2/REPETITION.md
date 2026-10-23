# 封闭几何词表、重复监督与开放动作描述

MotionScript 的测量算子、动作原语和模板有限，而组合出的描述仍可很多；点数增加和同义词增加不会自动增加观测到的动作种类。原 MS 高频原语与模板可能使 Omni 监督过于集中，但现有实验尚未证明它是生成复读的唯一原因。应区分：同一证据被多次描述、真实动作多次发生、训练语料分布集中，以及模型解码时自行循环复读。

本分支实现数据端的处理与验证接口，不修改现有 Omni/H3 训练，也不宣称已经消除真实模型的生成重复。历史已结束的 N7 实验使用原 MS＋genre，不能用它评价尚未完成的融合标签效果。

## 1. 压缩重复叙述，保留真实重复动作

`selection.py` 位于事件提取之后，保留全部 `source_events`，再生成可逆回查的描述视图：

- 同来源、同部位、同测量／方向、近乎相同证据且时间高度重叠的副本可以合并，`duplicate_map` 保留原 ID。
- 不同时间发生的相同动作不是重复数据，不删除。
- 连续支持段内至少四次、量值衔接且严格交替的肘膝弯伸／上臂抬落，可用一句“repeatedly bends and extends”概括。`occurrences` 仍保留每次 ID、时间和量值，不臆造周期数、节拍或完整舞步。
- 不跨根／关节支持段、不合并左右、不强制 top-K、不用同义词制造新动作类。

`ms_optimize.py --emit-llm-payload` 现在导出紧凑事件组请求，要求所有原始 ID 含别名恰好覆盖一次。旧的一事件一句话请求仅保留作审计用途。请求不是模型已经正确改写的证据；当前不自动调用 LLM，也不直接把候选放入训练。

这个步骤解决冗余叙述，**不会突破规则词表**。新增动作语义走独立的视觉分支。

## 2. 独立视觉观察：允许规则词表以外的描述

新入口为 `scripts_omnidance/exp/ms/ms_visual_observe.py`。它不把原 MS、rich text、genre、固定动作枚举作为观察先验。输入实际视频抽出的有序帧和真实索引；模型返回自由文本动作、证据帧及不确定性。

```bash
# 只抽帧并生成 request.json，不调用任何模型。
python scripts_omnidance/exp/ms/ms_visual_observe.py prepare \
  --video /data/videos/clip.mp4 --model YOUR_VLM_MODEL \
  --out /work/visual_clip_v1 --sample-fps 8 --max-frames 48

# 可选：显式将该请求发送到自己的多图 VLM 服务。
python scripts_omnidance/exp/ms/ms_visual_observe.py observe \
  --bundle /work/visual_clip_v1 \
  --endpoint http://127.0.0.1:8000/v1/chat/completions

# 或导入在其它环境得到的 JSON，二选一。
python scripts_omnidance/exp/ms/ms_visual_observe.py import-response \
  --bundle /work/visual_clip_v1 --response /work/model_response.json
```

抽帧需要 FFmpeg/FFprobe；本地回归实际生成并解码了合成视频。准备请求使用真实 decoded-frame 时间戳，不假定视频是16 FPS，也不假定窗口一定是5.0625秒。可指定 `--start/--end`。8 FPS／48帧是默认预算，不保证足以看清快速舞步；这些是有序抽帧输入，不是完整连续视频观察。

时间戳探测与图片提取固定使用同一个首视频流，避免多视频流文件的索引／画面错配。JSON 中的重复键明确拒绝；导入若在写入原响应后中断，只能用完全相同的响应继续生成候选。

只有显式 `observe` 会发出请求；可从 `MS_VLM_API_KEY` 环境变量读取认证信息，密钥不会保存到数据。模型名称是请求或服务的声明，脚本没有核验服务器上的官方权重身份。本次开发未调用 VLM，也未做真实舞蹈标注。

响应格式：

```json
{
  "motion_caption": "A concise chronological description using supported movement facts.",
  "events": [
    {"id": "v1", "description": "An observed action, expressed freely.", "frame_indices": [0, 8]}
  ],
  "uncertainties": [],
  "abstain": false
}
```

`description` 没有动作类别枚举。新增名称必须有视觉支持；给出两个帧索引只通过格式检查，不证明动作或顺序正确。模型不确定时允许弃权，不强制最少字数；真实重复动作应在简短正文中概括其模式，不复写同一句。正文上限140词，防止规则条目被机械扩成长段；该上限也不是质量指标。

模型提供的证据帧范围仅是观察范围，不能冒称精确动作边界。原响应、帧、视频和请求均保留哈希。候选固定为 `semantic_approval=false`，模型自报 confidence、通过 JSON 检查或自评得分都不能自动批准训练。

## 3. 审核后导出，来源与旧 MS 分开

审核者回看实际视频、采样帧及候选，核对具体动作、左右、顺序、漏动作和重复模式。审核文件需要这些字段；SHA须填写实际文件值：

```json
{
  "approved": true,
  "video_observed": true,
  "reviewer": "actual reviewer identifier",
  "reviewer_type": "human",
  "candidate_sha256": "actual candidate.json SHA256",
  "video_sha256": "actual source video SHA256",
  "request_sha256": "actual request.json SHA256"
}
```

明确委托的助手审核使用 `reviewer_type=assistant_delegated`；导出会保留该身份，不能冒称完成人工审核。这里是外部审核记录的绑定，不是程序能自动证明审核者确实观察过视频。

```bash
python scripts_omnidance/exp/ms/ms_visual_observe.py export \
  --bundle /work/visual_clip_v1 --review /work/review.json \
  --out /work/reviewed/clip.json
```

导出重验原响应与候选、视频、请求和帧。结果标为 `independent_video_open_vocabulary_reviewed`；不能伪装成 `motion_source=original`，也不会修改训练入口。如果候选有错误，需要重新观察／审核，而不是把旧 SHA 的批准沿用到新文本。

建议后续以审核后的视觉动作正文为语义监督候选，将 MS 几何事件作为可查证的补充材料，避免机械拼入全部微动作句。几何事实与视觉事实冲突时记录不确定性并复核，不默认任一路是真值。

## 4. 审计分布，再决定训练采样

```bash
python scripts_omnidance/exp/ms/ms_caption_audit.py \
  --input /path/to/labels --out /work/caption_audit.json --top-k 20
```

审计只读原标签，按来源分开统计精确句重复、连续复读、高频句、部位覆盖和结构化谓词的样本级出现率。左右、数字和时段保留。不能因为两段视频有相同句子就认定它们重复；只用明确 SID、来源组或内容哈希关联，并说明身份尚未核实的情况。

也支持本分支的开放视觉 `motion_caption` 与自由事件描述。同一 v2 记录的完整描述与紧凑描述按配对方式比较，不重复计算样本；带不同事件 ID 或时间的同模板句子不会被精确重复指标合并，因此该指标不等同于完整的措辞单调度。

原始样本出现率与来源组去重后的出现率同时输出；同一视频多个窗口不是独立歌曲／人物。英语关键词统计只是有限词表的文字提及，不能当作开放动作空间覆盖或真实动作频率。语料中的多个来源不得混为一次统一的动作分布。

不自动删除真实高频动作，不从未审标签生成逆频率采样权重。审核后再考虑按来源、舞种／音乐、身体部位和动作模式分层采样，并限制单一来源曝光。后续 Omni 对照应控制数据量、有效 token、训练步数、解码设置及正文长度，按独立来源划分测试集：

1. 原 MS 监督。
2. 相同证据的紧凑 MS 监督，分离文字冗余效应。
3. 审核后的开放视觉监督，检验是否真正补充动作覆盖。

同时测目标文本复读、模型原始解码复读和视频动作重复；不要靠 `repetition_penalty` 或随机同义词掩盖数据／模型问题。模型是否改善，需要实际训练和生成对照才能确定。
