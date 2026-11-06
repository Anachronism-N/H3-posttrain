import argparse, hashlib, json, os, wave
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def sha_file(path):
    d=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): d.update(b)
    return d.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--merged',type=Path,required=True); ap.add_argument('--planner',type=Path,required=True); ap.add_argument('--audio',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
    if a.out.exists(): raise SystemExit('exists')
    meta={}
    for line in a.planner.read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line); sid=r.get('id')
        if isinstance(sid,str) and isinstance(r.get('audio_sha256'),str): meta[sid]=(r['audio_sha256'],r.get('source_group_id'))
    rows=[]; rejected=Counter()
    for line in a.merged.read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line); sid=r.get('id'); text=r.get('motion_caption')
        if sid not in meta or not isinstance(text,str) or not text.strip(): rejected['schema']+=1; continue
        if len(text)>10000: rejected['over_10k']+=1; continue
        if '<|' in text or '<audio>' in text: rejected['reserved']+=1; continue
        if text.startswith('Dance style:'):
            parts=text.split('\n',1)
            if len(parts)!=2 or not parts[1].strip(): rejected['empty_body']+=1; continue
            text=parts[1]
        rows.append(dict(id=sid,source_group_id=meta[sid][1] or r.get('source_group_id'),audio_sha256=meta[sid][0],target_text=text,split=r.get('split','train')))
    # exact audio validation
    def check(row):
        p=a.audio/(row['id']+'.wav')
        try:
            with wave.open(str(p),'rb') as w:
                if (w.getframerate(),w.getnchannels(),w.getnframes())!=(24000,1,121500): return None,'geometry'
            if sha_file(p)!=row['audio_sha256']: return None,'sha'
        except Exception: return None,'unreadable'
        return row,None
    with ThreadPoolExecutor(max_workers=32) as ex: results=list(ex.map(check,rows))
    kept=[]
    for row,err in results:
        if err: rejected[err]+=1
        else: kept.append(row)
    # union-find groups by source_group + audio
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(x,y):
        x,y=find(x),find(y)
        if x!=y: parent[y]=x
    for r in kept:
        union(('id',r['id']),('g',r['source_group_id'] or '')); union(('id',r['id']),('a',r['audio_sha256']))
    clusters=sorted({find(('id',r['id'])) for r in kept},key=lambda x:hashlib.sha256((str(a.seed)+':'+str(x)).encode()).hexdigest())
    n=len(clusters); nt=int(n*.9); nv=int(n*.05)
    assign={c:'train' if i<nt else 'val' if i<nt+nv else 'test' for i,c in enumerate(clusters)}
    for r in kept: r['split']=assign[find(('id',r['id']))]
    a.out.mkdir(parents=True); (a.out/'audio').mkdir()
    for r in kept: os.link(a.audio/(r['id']+'.wav'),a.out/'audio'/(r['id']+'.wav'))
    for split in ('train','val','test'):
        vals=[r for r in kept if r['split']==split]
        data=[dict(id=r['id'],source_group_id=r['source_group_id'],split=split,audio_path='audio/'+r['id']+'.wav',audio_sha256=r['audio_sha256'],audio_duration_seconds=5.0625,audio_window_seconds=[0.0,5.0625],target_text=r['target_text'],caption_version='m2d-full-v3',semantic_approval=False,human_review_completed=False) for r in vals]
        (a.out/(split+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data),encoding='utf-8')
        print(split,len(data))
    contract=dict(schema='music-planner-caption-contract-v1',caption_version='m2d-full-v3',task_instruction='Listen to the music and describe the dance.',audio_min_seconds=1.0,audio_max_seconds=10.0,required_markers=[],require_target_windows=False)
    (a.out/'caption_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n'); (a.out/'caption_contract_nostyle.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
    summary=dict(schema='m2d-full-v3',input=198315,kept=len(kept),rejected=dict(rejected),split_counts={s:sum(1 for r in kept if r['split']==s) for s in ('train','val','test')},audio_geometry='24kHz mono 121500',audio_sha_verified=True,nostyle=True)
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
