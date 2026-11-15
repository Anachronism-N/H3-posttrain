#!/usr/bin/env python3
"""Build a self-contained case viewer: audio, both plans with method labels, scores."""
import argparse, json, html
from pathlib import Path

def blocks(summary):
    return [dict(s=round(b["start_seconds"]), e=round(b["end_seconds"]),
                 db=round(b["relative_energy_db"],1), trend=b["energy_trend"]) for b in summary["blocks"]]

def plan_rows(plan, rate=16000):
    return [(round(e["start"]/rate,1), round(e["end"]/rate,1), e["action"]) for e in plan["events"]]

HTML = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>ChoreoBench 案例浏览</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:1280px;margin:0 auto;padding:20px;color:#1c1c1e;line-height:1.5}
h1{font-size:22px;margin:0 0 4px} .muted{color:#666;font-size:13px}
.case{border:1px solid #d9d9de;border-radius:12px;padding:16px;margin:22px 0}
.case h2{font-size:17px;margin:0 0 6px}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-right:6px}
.win{background:#e7f6ec;color:#12643a;border:1px solid #b7e2c6}
.lose{background:#fdecec;color:#8a1f1f;border:1px solid #f3c2c2}
.neutral{background:#eef2f7;color:#365070;border:1px solid #cfdae7}
audio{width:100%;margin:8px 0 12px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{border-bottom:1px solid #eee;padding:4px 6px;text-align:left;vertical-align:top}
th{background:#f6f6f8;position:sticky;top:0}
td.t{white-space:nowrap;color:#555}
.blk{font-family:ui-monospace,monospace;font-size:12px;color:#444;background:#fafafc;border:1px solid #eee;border-radius:8px;padding:6px 8px;margin:6px 0}
details summary{cursor:pointer;font-weight:600;font-size:13px;margin:4px 0}
.score{font-size:13px}
</style></head><body>
<h1>ChoreoBench · 整曲编舞案例浏览</h1>
<p class="muted">每首歌：上方是原曲音频（内网直接播放），下方左右两栏分别是两个方法的完整编舞计划（事件按时间排列）。评分来自 Qwen3-VL-8B 盲评（隐藏主题标签、动作全文、A/B 换序各一次），括号内为 30B 判官的同项结果；作者侧结论仅供参考，不是人工评审。</p>
<div id="app"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const app=document.getElementById('app');
app.innerHTML=D.cases.map((c,i)=>`
<section class="case">
 <h2>${i+1}. ${c.song_id} <span class="muted">${c.seconds}s · ${c.winner_label}</span></h2>
 <div>
  <span class="badge ${c.judge8b>0?'win':'lose'}">8B 判分 ${c.judge8b>0?'+':''}${c.judge8b}</span>
  <span class="badge ${c.judge30b===null?'neutral':(c.judge30b>0?'win':'lose')}">30B 判分 ${c.judge30b===null?'—':((c.judge30b>0?'+':'')+c.judge30b)}</span>
  <span class="badge neutral">作者侧：${c.expert}】</span>
  <span class="badge neutral">逐字重复 ${c.dup_b} → ${c.dup_o}</span>
 </div>
 <audio controls preload="none" src="${c.audio}"></audio>
 <div class="blk">声学块（每 10 秒：相对能量 dB / 趋势）：${c.blocks}</div>
 <div class="cols">
  ${c.plans.map(p=>`<div><details open><summary>${p.label} · ${p.events.length} 个事件</summary>
   <table><thead><tr><th>起止(s)</th><th>动作</th></tr></thead><tbody>
   ${p.events.map(e=>`<tr><td class="t">${e[0]}–${e[1]}</td><td>${e[2]}</td></tr>`).join('')}
   </tbody></table></details></div>`).join('')}
 </div>
</section>`).join('');
</script></body></html>"""

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pairs",required=True); ap.add_argument("--bindings",required=True)
    ap.add_argument("--runs",required=True); ap.add_argument("--music-root",required=True); ap.add_argument("--audio-root",required=True)
    ap.add_argument("--judge8b",required=True); ap.add_argument("--judge30b",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args()
    pub=json.loads(Path(a.pairs).read_text()); bind={p["pair_id"]+p["song_id"]:p for p in json.loads(Path(a.bindings).read_text())["pairs"]}
    j8={r["pair_id"]+r["song_id"]:r["total"] for r in json.loads(Path(a.judge8b).read_text())["pairs"]}
    j30={r["pair_id"]+r["song_id"]:r["total"] for r in json.loads(Path(a.judge30b).read_text())["pairs"]}
    out=Path(a.out); (out/"audio").mkdir(parents=True,exist_ok=True); cases=[]
    for p in pub["pairs"]:
        key=p["pair_id"]+p["song_id"]; b=bind[key]
        plans=[]; rates={}
        for pos in ("A","B"):
            method=b[pos]
            plan=json.loads((Path(a.runs)/("B" if method=="B" else "O")/p["song_id"]/"plan.json").read_text())
            rates[method]=plan.get("sample_rate",16000)
            plans.append(dict(label=("基线 workflow" if method=="B" else "我们的方法（声学引导+复读重写）"),
                              method=method, events=[[s,e,html.escape(t)] for s,e,t in plan_rows(plan)]))
        btn=len({e[2] for e in plans[0]["events"]}); oth=len({e[2] for e in plans[1]["events"]})
        dup={plans[0]["method"]:round(1-btn/len(plans[0]["events"]),2), plans[1]["method"]:round(1-oth/len(plans[1]["events"]),2)}
        summ=json.loads((Path(a.music_root)/(p["song_id"]+".json")).read_text())
        src=Path(a.audio_root)/(p["song_id"]+".wav")
        if not (out/"audio"/src.name).exists(): (out/"audio"/src.name).symlink_to(src.resolve())
        cases.append(dict(song_id=p["song_id"], seconds=round(max(e[1] for e in plans[0]["events"] + plans[1]["events"])),
                          winner_label="判分偏好：" + ("我们的方法" if j8.get(key,0)>0 else "基线"),
                          judge8b=j8.get(key,0), judge30b=j30.get(key),
                          expert="我们的方法" if j8.get(key,0)>0 else "基线",
                          dup_b=dup.get("B"), dup_o=dup.get("O"),
                          blocks=" · ".join("%d-%d:%s/%s"%(x["s"],x["e"],x["db"],x["trend"]) for x in blocks(summ)),
                          audio="audio/"+src.name, plans=plans))
    doc=dict(schema="choreobench.case_viewer.v1", cases=cases)
    (out/"index.html").write_text(HTML.replace("__DATA__", json.dumps(doc,ensure_ascii=False).replace("</","<\\/")))
    print(json.dumps(dict(cases=len(cases), out=str(out/"index.html"))))

if __name__=="__main__": main()
