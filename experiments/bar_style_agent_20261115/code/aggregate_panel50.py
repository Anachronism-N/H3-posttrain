import json, subprocess, statistics
from collections import Counter
from pathlib import Path
B=Path('/ssd/2/niezichao.nzc/artifacts/bar-style-agent-run-20261115/panel20_results')
OSS='oss://futurelab-dpro-hz/niezichao.nzc/m2d_train_v1/runs'; CFG='/ssd/2/niezichao.nzc/.secrets/ossutilconfig_beijing'
def fetch(name):
    t=B/(name+'-registry.json')
    if not t.exists():
        subprocess.run(['ossutil','--config-file',CFG,'cp',f'{OSS}/{name}/runs/registry.json',str(t)],
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    return json.loads(t.read_text()) if t.exists() else None
rows=[]
for i in range(4):
    for name in (f'bar-style-panel20-shard{i}-v3', f'bar-style-panel43-matrix-shard{i}'):
        d=fetch(name)
        if d: rows.extend(d.get('runs',[]))
for i in range(2):
    d=fetch(f'bar-style-aioz-matrix-shard{i}')
    if d: rows.extend(d.get('runs',[]))
out={}
for arm in ('W','SF','RA','O'):
    cells=[r for r in rows if r.get('arm')==arm]
    comp=[c for c in cells if (c.get('result') or {}).get('full_song_succeeded')]
    calls=[(c.get('result') or {}).get('calls_used') for c in comp if (c.get('result') or {}).get('calls_used')]
    fails=Counter((c.get('result') or {}).get('stop_reason') or c.get('error') or 'missing'
                  for c in cells if not (c.get('result') or {}).get('full_song_succeeded'))
    out[arm]=dict(registered=len(cells),complete=len(comp),rate=round(len(comp)/len(cells),3) if cells else None,
                  mean_calls=round(statistics.mean(calls),1) if calls else None,failures=dict(fails))
(B/'panel50_metrics.json').write_text(json.dumps(dict(schema='panel50.metrics.v1',runs=len(rows),arms=out),ensure_ascii=False,indent=1)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=1))
