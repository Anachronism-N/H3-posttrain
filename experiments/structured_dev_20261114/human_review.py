#!/usr/bin/env python3
"""Build and analyse a human review pack for the blind pairs.

build   -> pack/pairs.json, pack/template.json, pack/RUBRIC.md, pack/audio/*
analyze -> per-criterion paired means, per-song winners, inter-rater sign agreement

The public pack never contains method names. The private bindings file is only
read by `analyze`, after the ratings are collected.
"""
import argparse
import json
import shutil
from pathlib import Path
from statistics import mean

CRITERIA = ("musical_fit", "motif_development", "action_executability", "boundary_connection")
RUBRIC = """# 评审说明（匿名）

你会看到同一首歌的两个编舞文本计划（A / B，顺序已随机）。它们来自同一批音频候选、
同一外层模型、同一预算，只有编排策略不同。方法名不会显示。

## 你可以听到 / 看到
- 原曲音频（`audio/<song_id>.wav`，16 kHz 单声道整曲）
- 公共声学摘要（每 10 秒块：相对能量、起音密度、能量趋势；样本相似度给出的
  “可能重复/对比”，**不是**检测到的副歌或乐句）
- 两个计划的事件表：`event_id`、起止秒、轨道、动作文本、主题 id

## 四个维度，各自 1–5 分
| 维度 | 1 分 | 3 分 | 5 分 |
|---|---|---|---|
| musical_fit 音乐贴合 | 多处与听到的节奏/能量变化明显不合 | 局部贴合，整体较通用 | 主要变化处有合理动作发展，时间安排清楚 |
| motif_development 主题发展 | 缺少发展，重复难以解释 | 有主题与对比，发展较弱 | 主题回归、变奏、对比和收尾组织清楚 |
| action_executability 可执行性 | 文字含糊、自相矛盾或明显超出所给时间 | 大体可理解，部分动作需澄清 | 动作与时间关系明确，舞者可据此试排 |
| boundary_connection 边界衔接 | 相邻动作多处断裂或冲突 | 大部分可衔接，部分依赖未说明条件 | 可判断的边界上衔接明确、合理 |

## 规则
- 必须引用具体 `event_id` 与原文动作来说明理由。
- **正常持续、留白和合理的主题回归不是缺陷**；机械照搬、无依据的重复才是。
- 未知的身体状态（支撑脚、朝向）不要当作缺陷。
- 听不清或无法判断时留空并注明，不要强行给分。
- 不要因为 A/B 位置推断方法。

## 交付
把填好的 `template.json`（每对一个文件或一个合并文件均可）交回。字段：
`pair_id, reviewer_id, heard_audio(bool), A:{四个维度}, B:{...}, preferred("A"/"B"/"tie"), notes`
"""


def build(pairs_path, audio_dir, music_dir, out):
    bundle = json.loads(Path(pairs_path).read_text())
    out = Path(out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    pairs = []
    for pair in bundle["pairs"]:
        song = pair["song_id"]
        source = Path(audio_dir) / (song + ".wav")
        target = out / "audio" / (song + ".wav")
        if source.is_file() and not target.exists():
            shutil.copyfile(source, target)
        summary = json.loads((Path(music_dir) / (song + ".json")).read_text())
        evidence = dict(
            blocks=[{k: b[k] for k in ("block_id", "start_seconds", "end_seconds", "relative_energy_db",
                                       "onset_rate_per_second", "energy_trend")} for b in summary["blocks"]],
            relations=summary["relations"][:12],
            note="acoustic measurement only; not chorus/verse/phrase/beat detection")
        pairs.append(dict(pair_id=pair["pair_id"], song_id=song,
                          audio="audio/%s.wav" % song, acoustic_evidence=evidence,
                          A=pair["A"], B=pair["B"], criteria=list(CRITERIA)))
    (out / "pairs.json").write_text(json.dumps(dict(schema="choreobench.human_review_pack.v1", criteria=list(CRITERIA),
                                                    pairs=pairs), ensure_ascii=False, indent=1) + "\n")
    template = dict(schema="choreobench.human_review_ratings.v1", reviewer_id="",
                    ratings=[dict(pair_id=p["pair_id"], song_id=p["song_id"], heard_audio=False,
                                  A={c: None for c in CRITERIA}, B={c: None for c in CRITERIA},
                                  preferred="", notes="") for p in pairs])
    (out / "template.json").write_text(json.dumps(template, ensure_ascii=False, indent=1) + "\n")
    (out / "RUBRIC.md").write_text(RUBRIC)
    return dict(pairs=len(pairs), out=str(out))


def _check(ratings, pack_ids):
    seen = set()
    for row in ratings["ratings"]:
        pid = row["pair_id"]
        if pid not in pack_ids or pid in seen:
            raise ValueError("unknown or duplicated pair_id: " + str(pid))
        seen.add(pid)
        for side in ("A", "B"):
            for criterion in CRITERIA:
                value = row[side][criterion]
                if value is not None and (type(value) is not int or not 1 <= value <= 5):
                    raise ValueError("scores must be 1-5 integers or null")
        if not ratings.get("reviewer_id"):
            raise ValueError("reviewer_id is required")
    return ratings


def analyze(pack_dir, bindings_path, ratings_paths, out):
    pack = json.loads((Path(pack_dir) / "pairs.json").read_text())
    pack_ids = {p["pair_id"] for p in pack["pairs"]}
    bindings = {b["pair_id"]: b for b in json.loads(Path(bindings_path).read_text())["pairs"]}
    reviewers = {}
    for path in ratings_paths:
        value = json.loads(Path(path).read_text())
        reviewers[value["reviewer_id"]] = _check(value, pack_ids)
    # per reviewer: map positional scores to method names
    per_reviewer = {}
    for rid, value in reviewers.items():
        rows = {}
        for row in value["ratings"]:
            b = bindings[row["pair_id"]]
            score = {b["A"]: row["A"], b["B"]: row["B"]}
            if None in score["B"].values() or None in score["S4"].values():
                continue
            rows[row["song_id"]] = {
                "delta": {c: score["S4"][c] - score["B"][c] for c in CRITERIA},
                "total_B": sum(score["B"].values()), "total_S4": sum(score["S4"].values())}
        per_reviewer[rid] = rows
    # aggregate
    agg = {c: [] for c in CRITERIA}
    winners = {"S4": 0, "B": 0, "tie": 0}
    for rid, rows in per_reviewer.items():
        for song, row in rows.items():
            for c in CRITERIA:
                agg[c].append(row["delta"][c])
            d = row["total_S4"] - row["total_B"]
            winners["S4" if d > 0 else "B" if d < 0 else "tie"] += 1
    agreement = None
    if len(per_reviewer) >= 2:
        rids = list(per_reviewer)
        hits = total = 0
        for song in set(per_reviewer[rids[0]]) & set(per_reviewer[rids[1]]):
            a = per_reviewer[rids[0]][song]["total_S4"] - per_reviewer[rids[0]][song]["total_B"]
            b = per_reviewer[rids[1]][song]["total_S4"] - per_reviewer[rids[1]][song]["total_B"]
            total += 1
            hits += int((a > 0) == (b > 0) and a != 0 and b != 0)
        agreement = dict(songs=total, same_direction=hits)
    result = dict(schema="choreobench.human_review_analysis.v1", reviewers=list(per_reviewer),
                  per_criterion_mean_delta={c: round(mean(v), 2) if v else None for c, v in agg.items()},
                  total_wins=winners, inter_reviewer=agreement, per_reviewer=per_reviewer,
                  note="S4 minus B on a 1-5 scale per criterion; null scores excluded, never imputed.")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pairs", required=True)
    b.add_argument("--audio-dir", required=True)
    b.add_argument("--music-dir", required=True)
    b.add_argument("--out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--pack", required=True)
    a.add_argument("--bindings", required=True)
    a.add_argument("--ratings", action="append", required=True)
    a.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cmd == "build":
        print(json.dumps(build(args.pairs, args.audio_dir, args.music_dir, args.out)))
    else:
        print(json.dumps(analyze(args.pack, args.bindings, args.ratings, args.out)))


if __name__ == "__main__":
    main()
