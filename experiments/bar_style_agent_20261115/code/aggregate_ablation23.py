"""23-song ablation table: 3 valid variants from abl43 plus 3 corrected reruns."""
import json, subprocess
from collections import Counter
from pathlib import Path

OSS = 'oss://futurelab-dpro-hz/niezichao.nzc/m2d_train_v1/runs'
B = Path('/ssd/2/niezichao.nzc/artifacts/bar-style-agent-run-20261115/panel20_results')
CFG = '/ssd/2/niezichao.nzc/.secrets/ossutilconfig_beijing'


def fetch(name):
    t = B / (name + '-registry.json')
    if not t.exists():
        subprocess.run(['ossutil', '--config-file', CFG, 'cp',
                        OSS + '/' + name + '/runs/registry.json', str(t)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(t.read_text()) if t.exists() else None


def summarize(cells):
    comp = [c for c in cells if (c.get('result') or {}).get('full_song_succeeded')]
    calls = [(c.get('result') or {}).get('calls_used') for c in comp
             if (c.get('result') or {}).get('calls_used')]
    fails = Counter((c.get('result') or {}).get('stop_reason') or c.get('error') or 'missing'
                    for c in cells if not (c.get('result') or {}).get('full_song_succeeded'))
    return dict(complete=len(comp), registered=len(cells),
                rate=round(len(comp) / len(cells), 3) if cells else None,
                mean_calls=round(sum(calls) / len(calls), 1) if calls else None,
                failures=dict(fails))


valid = ['no_style', 'no_acoustic', 'no_style_no_acoustic']
rows = {v: [] for v in valid}
for i in range(4):
    reg = fetch('bar-style-panel43-ablation-shard%d' % i)
    if not reg:
        continue
    cfg = json.loads(Path('/ssd/2/niezichao.nzc/artifacts/bar-style-abl43-shard-%d/config.json' % i).read_text())
    order = [v for v, _ in cfg['variants']]
    bysong = {}
    for r in reg['runs']:
        bysong.setdefault(r['song_id'], []).append(r)
    for sid, group in bysong.items():
        for v, r in zip(order, group):
            if v in rows:
                rows[v].append(r)

corrected = fetch('bar-style-ablfix23b')
order = ['no_motif', 'no_repair', 'motif_variation']
if corrected:
    bysong = {}
    for r in corrected['runs']:
        bysong.setdefault(r['song_id'], []).append(r)
    for sid, group in bysong.items():
        for v, r in zip(order, group):
            rows.setdefault(v, []).append(r)

out = {}
for v in valid + order:
    if not rows.get(v):
        print('missing', v)
        continue
    out[v] = summarize(rows[v])

baseline = json.loads((B / 'panel43_metrics.json').read_text())['arms']['O']
(B / 'panel23_ablation.json').write_text(
    json.dumps(dict(schema='choreobench.panel23_ablation.v1',
                    baseline_o=dict(complete=baseline['complete'] - 12,
                                    registered=baseline['registered'] - 20),
                    variants=out), ensure_ascii=False, indent=1) + '\n')
print(json.dumps(out, ensure_ascii=False, indent=1))
