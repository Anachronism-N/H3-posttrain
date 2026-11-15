#!/usr/bin/env python3
"""Build a single-page human review pack (one reviewer, ~20 minutes).

Everything is in one HTML file: audio players, blinded plans, and a form that
exports JSON. Answers are per criterion (A/B/cannot tell) plus one overall
choice, so a review can be completed without reading raw JSON.

Two independent reviewers are the target; this only lowers the effort per reviewer.
"""
import argparse
import json
import random
from pathlib import Path

RUBRIC = {
    "musical_fit": "动作量随时间的变化是否跟得上你听到的能量与起音变化",
    "motif_development": "主题是否被有意引入、变奏、回归，而不是机械照搬",
    "action_executability": "每个动作在给定时间内是否可读、可做",
    "boundary_connection": "相邻动作之间是否衔接得上",
}


def compact(events, limit=600, hide_motif=False):
    return [dict(id=e["event_id"], start=round(e["start"], 2), end=round(e["end"], 2),
                 track=e.get("track"), action=e["action"][:limit],
                 motif=None if hide_motif else e.get("motif_id"))
            for e in events]


def blocks(summary):
    return [dict(start=round(b["start_seconds"], 1), end=round(b["end_seconds"], 1),
                 energy=round(b["relative_energy_db"], 1), onsets=round(b["onset_rate_per_second"], 2),
                 trend=b["energy_trend"]) for b in summary["blocks"]]


HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>编舞计划盲评</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:1100px;margin:0 auto;padding:18px;line-height:1.5;color:#1c1c1e}
h1{font-size:20px;margin:0 0 6px} .muted{color:#666;font-size:13px}
.song{border:1px solid #d9d9de;border-radius:10px;padding:14px;margin:18px 0}
.song h2{font-size:16px;margin:0 0 8px}
audio{width:100%;margin:6px 0 10px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:800px){.cols{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{border-bottom:1px solid #eee;padding:4px 6px;text-align:left;vertical-align:top}
th{background:#f6f6f8;position:sticky;top:0}
details summary{cursor:pointer;font-weight:600;font-size:13px}
.q{margin:8px 0;font-size:13.5px}
label{margin-right:12px;white-space:nowrap}
textarea{width:100%;min-height:52px;font-size:13px}
.bar{position:sticky;bottom:0;background:#ffffffe6;backdrop-filter:blur(4px);padding:10px 0;border-top:1px solid #ddd;margin-top:16px}
button{font-size:14px;padding:8px 14px;border-radius:8px;border:1px solid #888;background:#fff;cursor:pointer}
button.primary{background:#0a84ff;border-color:#0a84ff;color:#fff}
#out{width:100%;height:150px;font-family:ui-monospace,monospace;font-size:12px}
</style></head><body>
<h1>编舞计划盲评</h1>
<p class="muted">同一首歌的两份编舞文本计划（A / B，顺序已随机化）。两者使用同一批音频候选、同一外层模型、同一预算，只有编排策略不同。方法名不会显示。预计 20 分钟。
评分口径：每首歌先听音频（可拖到任意位置），再看两份计划；每个维度选“A 更好 / B 更好 / 无法判断”，最后选整体更愿意用于排练的一份。
正常持续、留白和有意回归的主题都不是缺陷；无法判断时请选“无法判断”，不要猜。</p>
<div id="app"></div>
<div class="bar"><button class="primary" onclick="exportAnswers()">生成结果 JSON</button>
<button onclick="copyAnswers()">复制</button> <span id="progress" class="muted"></span></div>
<textarea id="out" placeholder="点击“生成结果 JSON”后，把这里的内容保存为 answers_<你的名字>.json"></textarea>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const CRIT = Object.keys(DATA.rubric);
const answers = {};
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function table(events){return '<table><thead><tr><th>#</th><th>起止(s)</th><th>动作</th><th>主题</th></tr></thead><tbody>'+
 events.map((e,i)=>`<tr><td>${i+1}</td><td>${e.start}–${e.end}</td><td>${esc(e.action)}</td><td>${e.motif||''}</td></tr>`).join('')+'</tbody></table>';}
function render(){
 const app=document.getElementById('app');
 app.innerHTML = DATA.songs.map((s,si)=>`
  <section class="song" data-song="${s.song_id}">
   <h2>歌曲 ${si+1} / ${DATA.songs.length} · ${s.song_id} <span class="muted">(${s.seconds}s)</span></h2>
   <audio controls preload="none" src="${s.audio}"></audio>
   <details open><summary>声学摘要（10 秒块）</summary>
    <table><thead><tr><th>起止(s)</th><th>相对能量 dB</th><th>起音/秒</th><th>趋势</th></tr></thead><tbody>
    ${s.blocks.map(b=>`<tr><td>${b.start}–${b.end}</td><td>${b.energy}</td><td>${b.onsets}</td><td>${b.trend}</td></tr>`).join('')}
    </tbody></table></details>
   <div class="cols">
    <div><details open><summary>计划 A · ${s.A.length} 个事件</summary>${table(s.A)}</details></div>
    <div><details open><summary>计划 B · ${s.B.length} 个事件</summary>${table(s.B)}</details></div>
   </div>
   <div class="q"><b>整体</b>：你更愿意用哪份计划排练？
     ${['A','B','无法判断'].map(v=>`<label><input type="radio" name="overall-${s.pair_id}" value="${v}" onchange="set('${s.pair_id}','overall','${v}')"> ${v}</label>`).join('')}
   </div>
   ${CRIT.map(c=>`<div class="q"><b>${c}</b> ${esc(DATA.rubric[c])}
     ${['A','B','无法判断'].map(v=>`<label><input type="radio" name="${c}-${s.pair_id}" value="${v}" onchange="set('${s.pair_id}','${c}','${v}')"> ${v}</label>`).join('')}
   </div>`).join('')}
   <div class="q"><label><input type="checkbox" onchange="set('${s.pair_id}','heard_audio',this.checked)"> 我完整听过这首歌</label></div>
   <div class="q"><textarea placeholder="理由（请引用具体事件编号和动作原文，可留空）" oninput="set('${s.pair_id}','reason',this.value)"></textarea></div>
  </section>`).join('');
}
function set(pairId,key,value){answers[pairId]=answers[pairId]||{};answers[pairId][key]=value;progress();}
function progress(){const done=Object.values(answers).filter(a=>a.overall).length;document.getElementById('progress').textContent=`已完成 ${done}/${DATA.songs.length}`;}
function exportAnswers(){
 const payload={schema:'choreobench.human_answers.v1',reviewer_id:prompt('你的名字或代号（用于登记，不会与方法绑定）','reviewer-1')||'anonymous',
  pairs:DATA.songs.map(s=>Object.assign({pair_id:s.pair_id,song_id:s.song_id},answers[s.pair_id]||{}))};
 document.getElementById('out').value=JSON.stringify(payload,null,1);
}
function copyAnswers(){const t=document.getElementById('out');if(!t.value)exportAnswers();t.select();document.execCommand('copy');}
render();
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="review-pairs.public.json")
    parser.add_argument("--music-root", required=True, help="directory with <song_id>.json summaries")
    parser.add_argument("--audio-root", required=True, help="directory with <song_id>.wav")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--hide-motif", action="store_true",
                        help="hide theme ids so the review judges content, not labelling")
    parser.add_argument("--seed", type=int, default=13, help="position randomisation seed")
    parser.add_argument("--bindings", help="private bindings to read; written to <out>/bindings.private.json")
    args = parser.parse_args()
    bundle = json.loads(Path(args.pairs).read_text())
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    songs, bindings = [], []
    for pair in bundle["pairs"]:
        summary = json.loads((Path(args.music_root) / (pair["song_id"] + ".json")).read_text())
        source = Path(args.audio_root) / (pair["song_id"] + ".wav")
        target = out / "audio" / source.name
        if not target.exists():
            target.symlink_to(source.resolve())
        duration = round(max(e["end"] for e in pair["A"] + pair["B"]), 1)
        flip = rng.random() < 0.5
        first, second = (pair["B"], pair["A"]) if flip else (pair["A"], pair["B"])
        songs.append(dict(pair_id=pair["pair_id"], song_id=pair["song_id"], seconds=duration,
                          audio="audio/" + source.name, blocks=blocks(summary),
                          A=compact(first, hide_motif=args.hide_motif),
                          B=compact(second, hide_motif=args.hide_motif)))
        if args.bindings:
            private = {x["pair_id"]: x for x in json.loads(Path(args.bindings).read_text())["pairs"]}
            source_binding = private[pair["pair_id"]]
            bindings.append(dict(pair_id=pair["pair_id"], song_id=pair["song_id"],
                                 comparison=pair["comparison"],
                                 A=source_binding["B"] if flip else source_binding["A"],
                                 B=source_binding["A"] if flip else source_binding["B"],
                                 flipped=flip))
    payload = dict(schema="choreobench.human_pack.v1", criteria=list(RUBRIC), rubric=RUBRIC,
                   instructions="one reviewer, ~20 minutes, forced choice, blind positions",
                   songs=songs)
    text = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    (out / "index.html").write_text(text)
    if bindings:
        (out / "bindings.private.json").write_text(json.dumps(dict(
            schema="choreobench.review_bindings.v1", seed=args.seed, pairs=bindings),
            ensure_ascii=False, indent=1) + "\n")
    (out / "template.json").write_text(json.dumps(
        dict(schema="choreobench.human_answers.v1", reviewer_id="", pairs=[
            dict(pair_id=s["pair_id"], song_id=s["song_id"], overall=None, heard_audio=None,
                 reason="", **{c: None for c in RUBRIC}) for s in songs]), ensure_ascii=False, indent=1) + "\n")
    (out / "README.md").write_text(
        "# 单人盲评（约 20 分钟）\n\n"
        "1. 打开 `index.html`（直接双击，或用浏览器打开；音频在同目录 `audio/`）。\n"
        "2. 每首歌：先听音频，再展开 A/B 计划对比。\n"
        "3. 每个维度选 A 更好 / B 更好 / 无法判断，勾选“完整听过”，最后选整体偏好并写一句理由（可留空）。\n"
        "4. 点“生成结果 JSON”，把文本框内容保存为 `answers_<你的名字>.json` 交回。\n\n"
        "也可以不用网页，直接填 `template.json`。\n\n"
        "规则：方法名不可见；正常持续、留白、有意回归的主题不是缺陷；无法判断就选“无法判断”。\n\n"
        "注意：本包隐藏了所有主题编号（motif id），以免根据标签而不是内容打分；"
        "「主题发展」请依据动作文本本身的重复与变化判断。\n\n"
        "位次已按种子随机化，两份计划不是固定顺序。\n")
    print(json.dumps(dict(songs=len(songs), out=str(out / "index.html"),
                          bytes=(out / "index.html").stat().st_size), ensure_ascii=False))


if __name__ == "__main__":
    main()
