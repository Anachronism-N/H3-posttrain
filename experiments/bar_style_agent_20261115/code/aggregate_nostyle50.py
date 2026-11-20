"""Matched 50-song no-style comparison: O vs RA with style prior removed."""
import json, collections
from math import factorial
from pathlib import Path

B = Path('/ssd/2/niezichao.nzc/artifacts/bar-style-agent-run-20261115/panel20_results')
o, ra = {}, {}

# 20 original songs: O no-style from the 20-song ablation (index 0 per song),
# RA no-style from bar-style-ra-nostyle20.
uniform = B / 'bar-style-o-nostyle20-registry.json'
if uniform.exists():
    for r in json.loads(uniform.read_text())['runs']:
        o[r['song_id']] = (r.get('result') or {}).get('full_song_succeeded') is True
else:
    for i in range(4):
        reg = json.loads((B / ('bar-style-panel20-ablation-shard%d-registry.json' % i)).read_text())
        by = collections.defaultdict(list)
        for r in reg['runs']:
            by[r['song_id']].append(r)
        for sid, group in by.items():
            o[sid] = (group[0].get('result') or {}).get('full_song_succeeded') is True
reg = json.loads((B / 'bar-style-ra-nostyle20-registry.json').read_text())
for r in reg['runs']:
    ra[r['song_id']] = (r.get('result') or {}).get('full_song_succeeded') is True

# 23 replication songs.
reg = json.loads((B / 'bar-style-ra-nostyle-registry.json').read_text())
for r in reg['runs']:
    tgt = o if r['arm'] == 'O' else ra
    tgt[r['song_id']] = (r.get('result') or {}).get('full_song_succeeded') is True

# 7 AIOZ short tracks.
for f, tgt in (('bar-style-aioz-nostyle-registry.json', o), ('bar-style-aioz-ranostyleb-registry.json', ra)):
    reg = json.loads((B / f).read_text())
    for r in reg['runs']:
        tgt[r['song_id']] = (r.get('result') or {}).get('full_song_succeeded') is True

songs = sorted(set(o) & set(ra))
both = sum(1 for s in songs if o[s] and ra[s])
oo = sum(1 for s in songs if o[s] and not ra[s])
ro = sum(1 for s in songs if ra[s] and not o[s])
ne = sum(1 for s in songs if not o[s] and not ra[s])
n = oo + ro
k = min(oo, ro)
p = 2 * sum(factorial(n) // (factorial(i) * factorial(n - i)) for i in range(k + 1)) / 2 ** n if n else 1
out = dict(schema='choreobench.nostyle50.v1', songs=len(songs),
           o_complete=sum(1 for s in songs if o[s]), ra_complete=sum(1 for s in songs if ra[s]),
           both=both, o_only=oo, ra_only=ro, neither=ne, mcnemar_p=round(min(1, p), 4))
(B / 'nostyle50.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
