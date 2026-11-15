#!/usr/bin/env python3
"""Absolute (single-plan) scoring with a text-only judge, to avoid pair-position bias."""
import argparse, json, urllib.request
from pathlib import Path
CRITERIA=("musical_fit","motif_development","action_executability","boundary_connection")
SYSTEM=("Rate one anonymous timed choreography score for a song. You cannot hear the audio; the supplied "
 "acoustic summary is fixed-parameter measurement. Score 1-5 on musical_fit (movement amount follows the "
 "block evidence), motif_development (themes varied rather than pasted unchanged), action_executability "
 "(each action fits its stated duration), boundary_connection (adjacent events join coherently). Times are "
 "seconds. Sustained movement and intentional theme return are not defects. Cite at least one event id. "
 'Return only {"scores":{"musical_fit":n,"motif_development":n,"action_executability":n,'
 '"boundary_connection":n},"reason":"..."}.')
def call(url,model,system,payload,max_tokens=600,seed=5):
    body=json.dumps(dict(model=model,temperature=0,max_tokens=max_tokens,seed=seed,
        response_format={"type":"json_object"},
        messages=[dict(role="system",content=[dict(type="text",text=system)]),
                  dict(role="user",content=[dict(type="text",text=json.dumps(payload,ensure_ascii=False))])])).encode()
    r=urllib.request.Request(url.rstrip("/")+"/chat/completions",data=body,headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(r,timeout=900) as resp: return json.loads(resp.read())
def first_object(t):
    s=t.find("{"); d=0; ins=False; esc=False
    for i in range(s,len(t)):
        c=t[i]
        if ins:
            if esc: esc=False
            elif c=="\\": esc=True
            elif c=='"': ins=False
        elif c=='"': ins=True
        elif c=='{': d+=1
        elif c=='}':
            d-=1
            if d==0: return t[s:i+1]
    raise ValueError("unterminated")
def main():
    p=argparse.ArgumentParser(); p.add_argument("--pairs",required=True); p.add_argument("--music-root",required=True)
    p.add_argument("--out",required=True); p.add_argument("--base-url",default="http://127.0.0.1:8017/v1")
    p.add_argument("--model",default="MOSS-Audio-8B-Instruct"); p.add_argument("--action-chars",type=int,default=400)
    a=p.parse_args(); bundle=json.loads(Path(a.pairs).read_text()); rows=[]
    for pair in bundle["pairs"]:
        summ=json.loads((Path(a.music_root)/(pair["song_id"]+".json")).read_text())
        blocks=[{k:b[k] for k in ("block_id","start_seconds","end_seconds","relative_energy_db","energy_trend")} for b in summ["blocks"]]
        row=dict(pair_id=pair["pair_id"],song_id=pair["song_id"],versions=[])
        for pos in ("A","B"):
            payload=dict(song_id=pair["song_id"],acoustic_blocks=blocks,
                plan=[dict(event_id=e["event_id"],start=e["start"],end=e["end"],action=e["action"][:a.action_chars]) for e in pair[pos]])
            try:
                resp=call(a.base_url,a.model,SYSTEM,payload); val=json.loads(first_object(resp["choices"][0]["message"]["content"]))
                ok=all(isinstance(val["scores"].get(c),int) for c in CRITERIA)
                if not ok: raise ValueError("missing criteria")
                row["versions"].append(dict(position=pos,scores=val["scores"],reason=val.get("reason"),error=None))
            except Exception as e: row["versions"].append(dict(position=pos,scores=None,error=repr(e)))
        rows.append(row); print(row["pair_id"],"ok" if all(v["error"] is None for v in row["versions"]) else "FAIL",flush=True)
    Path(a.out).write_text(json.dumps(dict(schema="choreobench.absolute_text.v1",judge_model=a.model,pairs=rows),indent=1))
if __name__=="__main__": main()
