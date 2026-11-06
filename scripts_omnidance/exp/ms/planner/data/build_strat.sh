#!/bin/bash
set -e
python - <<'PY'
import json,re,os,hashlib,shutil
from pathlib import Path
MARK=re.compile(r'\[\s*\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*s\s*\]')
base=Path('/ssd/2/niezichao.nzc/artifacts/captioner-jev-data-v1/full-v4')
out=Path('/ssd/2/niezichao.nzc/artifacts/captioner-jev-data-v1/strat-v1')
if out.exists(): shutil.rmtree(out)
(out/'audio').mkdir(parents=True)
def density(t):
    parts=[p for p in MARK.split(t) if p.strip()]
    cl=[s for p in parts for s in re.split(r'[;。]',p) if s.strip()]
    return len(cl)/max(len(parts),1)
def bucket(n):
    return str(min(n,5))  # 1,2,3,4,5+
def load(split):
    rows=[json.loads(l) for l in (base/(split+'.jsonl')).open() if l.strip()]
    out_rows=[]; stats={}
    for n in ('1','2','3','4','5'):
        cand=[r for r in rows if bucket(len(MARK.findall(r['target_text'])))==n and 5<=density(r['target_text'])<=40]
        cand.sort(key=lambda r: hashlib.sha256(('strat:'+r['id']).encode()).hexdigest())
        stats[n]=len(cand); out_rows.extend(cand)
    return out_rows, stats
train,ts=load('train'); val,vs=load('val')
# cap val to 800 balanced across buckets
from collections import defaultdict
byb=defaultdict(list)
for r in val: byb[bucket(len(MARK.findall(r['target_text'])))].append(r)
valsel=[]
for k in sorted(byb): valsel.extend(byb[k][:160])
for split,rows in (('train',train),('val',valsel)):
    with (out/(split+'.jsonl')).open('w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    for r in rows:
        src=base/'audio'/(r['id']+'.wav'); dst=out/'audio'/(r['id']+'.wav')
        if not dst.exists(): os.link(src,dst)
for n in ('caption_contract.json','caption_contract_nostyle.json'):
    shutil.copyfile(base/n,out/n)
import statistics as st
print('train rows',len(train),'bucket sizes',ts)
print('val rows',len(valsel),'bucket sizes',vs)
print('mean chars',st.mean(len(r['target_text']) for r in train),'mean density',st.mean(density(r['target_text']) for r in train))
PY
cd /ssd/2/niezichao.nzc/artifacts/captioner-jev-data-v1 && tar -cf strat-v1.tar -C strat-v1 .
ossutil -c /ssd/2/niezichao.nzc/.secrets/ossutilconfig -e oss-cn-hangzhou.aliyuncs.com cp strat-v1.tar oss://futurelab-dpro-hz/niezichao.nzc/m2d_train_v1/data/strat-v1.tar --force
echo DONE
