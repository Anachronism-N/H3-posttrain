"""Cost/latency/repetition/coverage metrics for panel20 plans."""
import json, re, statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path('/ssd/2/niezichao.nzc/artifacts/bar-style-agent-run-20261115/panel20_results')
DUR=json.loads(Path('/ssd/2/niezichao.nzc/artifacts/bar-style-panel20/panel20.json').read_text())
DUR={s['song_id']:s['audio']['num_samples']/s['audio']['sample_rate'] for s in DUR['songs']}
SENT=re.compile(r'(?<=[.!?])\s+')
rows=[]
for p in sorted(ROOT.glob('shard*/runs/*/*/plan.json')):
    arm=p.parent.name; sid=p.parent.parent.name
    plan=json.loads(p.read_text()); rate=plan['sample_rate']; ev=plan['events']
    secs=[(e['end']-e['start'])/rate for e in ev]
    text=' '.join(e['action'] for e in ev)
    sents=[s.strip() for s in SENT.split(text.strip()) if s.strip()]
    dup=len(sents)-len(set(sents))
    calls=list(p.parent.glob('calls/*.raw.json'))
    latency=0.0; itok=otok=0; n=0
    for f in calls:
        d=json.loads(f.read_text()); r=d.get('result') or {}; c=r.get('costs') or {}
        if isinstance(c.get('latency_seconds'),(int,float)): latency+=c['latency_seconds']
        if isinstance(c.get('input_tokens'),int): itok+=c['input_tokens']; ott=otok
        if isinstance(c.get('output_tokens'),int): otok+=c['output_tokens']
    rows.append(dict(song=sid,arm=arm,events=len(ev),coverage=sum(secs)/DUR.get(sid,1),
                     mean_event=sum(secs)/len(ev) if ev else 0,dup_rate=dup/len(sents) if sents else 0,
                     calls=len(calls),latency=latency,input_tokens=itok,output_tokens=otok))
agg=defaultdict(list)
for r in rows: agg[r['arm']].append(r)
out={}
for arm,rs in agg.items():
    out[arm]=dict(n=len(rs),
        mean_calls=round(statistics.mean(r['calls'] for r in rs),1),
        mean_latency_s=round(statistics.mean(r['latency'] for r in rs),1),
        mean_input_tokens=round(statistics.mean(r['input_tokens'] for r in rs)),
        mean_output_tokens=round(statistics.mean(r['output_tokens'] for r in rs)),
        mean_coverage=round(statistics.mean(r['coverage'] for r in rs),3),
        mean_events=round(statistics.mean(r['events'] for r in rs),1),
        mean_duplicate_rate=round(statistics.mean(r['dup_rate'] for r in rs),3))
(ROOT/'panel20_extra_metrics.json').write_text(json.dumps(dict(schema='panel20.extra_metrics.v1',arms=out),ensure_ascii=False,indent=1)+'\n')
for arm in ('W','SF','RA','O'): print(arm,out.get(arm))
