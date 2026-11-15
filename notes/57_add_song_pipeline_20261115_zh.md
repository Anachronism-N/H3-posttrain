# 新增歌曲到正式语料的完整流程（待批准执行）

本地语料已用尽：`choreobench-data/dataset` 与 `fma-mp3` 都只有同样的 11 首。要扩大留出集，必须**新增音频并重新采集候选**。下面是可直接执行的最小流程，预计每首歌 20–30 分钟（单卡）。

## 步骤

### 1. 选定新歌并下载音频

需要 Free Music Archive 的完整曲目（30 秒片段不够，整曲编排至少 2 分钟）。选歌规则建议：与现有 11 首**不同艺人/来源组**，避免分组泄漏。

```bash
# 以 <FMA_TRACK_ID> 为例；下载后用 prepare-audio 归一化
python -m choreobench prepare-audio \
  --source /path/to/<track>.mp3 \
  --out-wav /ssd/2/niezichao.nzc/artifacts/choreobench-data/dataset/audio/fma-<id>.wav \
  --out-song /ssd/2/niezichao.nzc/artifacts/choreobench-data/dataset/songs/fma-<id>.json \
  --song-id fma-<id> --split test --brief "free-form solo dance, no props"
```

### 2. 重建数据集清单

```bash
python -m choreobench assemble-data \
  --song <每首 song JSON> --out /path/to/dataset-12.json
python -m choreobench validate-data --dataset /path/to/dataset-12.json
```

### 3. 采集候选（需要 GPU + MOSS-Music 8B 本地权重）

```bash
python -m choreobench run \
  --dataset /path/to/dataset-12.json \
  --config /path/to/catalog_config.json \
  --out-dir /ssd/2/niezichao.nzc/artifacts/sr-dev/catalog_song12
```

`catalog_config.json` 与现有目录一致：`window_seconds=5.0625`、`stride_seconds=5.0625`、`candidates_per_window=2`、`seed=7`、`max_output_tokens=1800`，provider 用 MOSS-Music 的 native 配置。

### 4. 导出候选并生成视图

```bash
python -m choreobench export-candidates \
  --run /ssd/2/niezichao.nzc/artifacts/sr-dev/catalog_song12 \
  --out /ssd/2/niezichao.nzc/artifacts/sr-dev/catalog_song12/candidates.json
```

视图（`views_*/*.json`）由 `choreobench.untimed_catalog.run_catalog` 产出的合同派生，需要一条小驱动把新歌写入 `complete_catalog_song.v1` 结构；现有 11 首的视图即由此生成，可直接复用同一脚本。

### 5. 用现有流程跑正式臂

```bash
echo "fma-<id>\tviews_new\tmusic_new" >> /ssd/2/niezichao.nzc/artifacts/sr-dev/formal_song_list.tsv
/ssd/2/niezichao.nzc/artifacts/sr-dev/run_formal.sh \
  /ssd/2/niezichao.nzc/artifacts/sr-dev/formal_song_list.tsv \
  /ssd/2/niezichao.nzc/artifacts/sr-dev/provider_text.json
```

## 需要什么

| 资源 | 说明 |
|---|---|
| 新歌音频 | 需要下载 FMA 完整曲目（网络 + 存储） |
| GPU | 每首约 24–30 个窗口 × 2 候选，MOSS-Music 8B，单卡 20–30 分钟 |
| 批准 | 这会改变正式语料，必须先冻结新语料再跑，不能在结果出来后再增删歌曲 |

## 为什么必须这么做

现有 11 首中基线只完成了 6 首，导致**可配对的 test 只有 2 首**，任何分集结论都没有统计意义。加 2–4 首能让 test 配对达到 4–6 首，才有机会判断内容质量是否真的更好。
