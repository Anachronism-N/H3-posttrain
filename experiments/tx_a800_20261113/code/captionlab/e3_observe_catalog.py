"""Observe exact catalog PCM windows without reading any candidate response."""
import json,time,shutil,subprocess
from pathlib import Path

root=Path('/opt/music_action_runs/20261113_e3_observe_catalog_v1')
catalog=Path('/opt/music_action_runs/20261113_e3_moss_catalog_v1')
start=time.monotonic()
while not all((catalog/('shard'+str(i))/'complete.json').exists() for i in range(4)):
 for i in range(4):
  p=catalog/('manager'+str(i))/'catalog_exit.json'
  if p.exists() and json.loads(p.read_text())['exit_code']!=0:raise RuntimeError('Original catalog failed; no synthetic replacement audio')
 if time.monotonic()-start>86400:raise TimeoutError('Original catalog incomplete')
 time.sleep(30)
items=[];assets=[];bindings=[]
for i in range(4):
 cat=catalog/('shard'+str(i))/'catalog';contract=json.loads((cat/'contract.json').read_text())
 songs={s['song_id']:s for s in contract['dataset_snapshot']['songs']}
 for source in contract['song_windows']:
  sid=source['song_id']
  for w in source['windows']:
   path=cat/'audio'/sid/('%04d.wav'%w['index']);import hashlib
   digest=hashlib.sha256(path.read_bytes()).hexdigest();rate=songs[sid]['audio']['sample_rate'];count=w['observation_end']-w['observation_start']
   assets.append(dict(path=str(path),sha256=digest,sample_rate=rate,num_samples=count));items.append(dict(item_sha256=digest,audio_sha256=digest,case_id=sid+'_'+str(w['index']),method_id='observer',source_group_id=sid,generation_status='ok',text=''));bindings.append(dict(song_id=sid,window=w,audio_sha256=digest))
(root/'items.json').write_text(json.dumps(dict(items=items),indent=2));(root/'audio_assets.json').write_text(json.dumps(dict(assets=assets),indent=2));(root/'bindings.json').write_text(json.dumps(dict(bindings=bindings),indent=2));(root/'freeze.json').write_text(json.dumps(dict(registered=len(items),source_songs=24,candidate_text_read=False,reference_read=False,source='same original full-song PCM windows already materialized; no rewriting',fallible_audio_model=True),indent=2))
raise SystemExit(subprocess.call(['/opt/venv_music_action_20261106/bin/python','-u',str(root/'manager.py'),'--config',str(root/'config.json')]))
