"""Reuse exact already-existing merged-window plans before their original sources are claimed."""
import hashlib,io,json,os,sqlite3,tarfile,time
from pathlib import Path
import motion_revision_codec_v3 as codec
from stage2_local_publication_v1 import write_once
LOCAL=Path('/root/stage2_motion_revision_exact_plan_reuse_20261028_v1')
PLANS=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/ms_truebar_first50_merge_short_edges_v1')


def run():
 LOCAL.mkdir(exist_ok=True);source=sqlite3.connect('/root/stage2_motion_revision_inputs_20261028_v2/catalog.db',isolation_level=None,timeout=60);model=sqlite3.connect('/root/stage2_motion_revision_claim_service_20261028_v3/model_pool.db',isolation_level=None,timeout=60);source.execute('CREATE TABLE IF NOT EXISTS exact_plan_reuse(source_key TEXT PRIMARY KEY,plan_ref TEXT,status TEXT,evidence TEXT)');plans=[]
 for p in PLANS.glob('*.windows.json'):
  raw=p.read_bytes();plan=json.loads(raw);assert plan['params']['policy']=='merge_short_edges';plans.append((plan,dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest())))
 while not (LOCAL/'STOP').exists():
  applied=0;pending=0
  for original,ref in plans:
   key=original['id'].casefold()
   if source.execute('SELECT 1 FROM exact_plan_reuse WHERE source_key=?',(key,)).fetchone():continue
   s=source.execute('SELECT payload FROM sources WHERE source_key=?',(key,)).fetchone()
   if not s:pending+=1;continue
   row=json.loads(s[0]);ar=row['revision_input_archive'];member=row['revision_input_member'];raw=Path(ar['path']).read_bytes();assert len(raw)==ar['bytes'] and hashlib.sha256(raw).hexdigest()==ar['sha256']
   with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tar:context=json.load(tar.extractfile(member))
   assert context['manifest']['video_sha256']==original['binding']['video_sha256'] and context['manifest']['duration_seconds']==original['binding']['media_end'],'Original exact plan refers to different video'
   plan=dict(context['plan']);plan.update(windows=[dict(window_id=i,**w) for i,w in enumerate(original['windows'])],raw_windows=original['raw_windows'],raw_downbeats_audio_seconds=original.get('downbeats_seconds',[]),original_existing_merged_plan_ref=ref,existing_plan_boundary_reused_exactly=True);context.update(plan=plan,window_plan_sha256=hashlib.sha256(codec.encoded(plan)).hexdigest(),window_frames=codec.select_window_frames(context['manifest'],plan),original_existing_window_plan_reused=True)
   payload=codec.encoded(context);path=LOCAL/(original['id']+'.tar')
   if not path.exists():
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:
     with tarfile.open(fileobj=f,mode='w',format=tarfile.PAX_FORMAT) as tar:m=tarfile.TarInfo(member);m.size=len(payload);tar.addfile(m,io.BytesIO(payload))
     f.flush();os.fsync(f.fileno())
   data=path.read_bytes();updated=dict(row,revision_input_archive=dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)),revision_input_sha256=hashlib.sha256(payload).hexdigest());model.execute('BEGIN IMMEDIATE')
   try:
    state=model.execute('SELECT state,owner FROM sources WHERE source_key=?',(key,)).fetchone()
    if state is not None and state[0]!='unused':
     model.execute('COMMIT');source.execute('INSERT INTO exact_plan_reuse VALUES(?,?,?,?)',(key,json.dumps(ref),'already_claimed_no_budget_changed',json.dumps(state)));print(json.dumps(dict(event='existing_plan_already_claimed_hold',id=original['id'],models_retried=0)),flush=True);continue
    value=json.dumps(updated,separators=(',',':'));source.execute('UPDATE sources SET payload=? WHERE source_key=?',(value,key))
    if state:model.execute('UPDATE sources SET payload=? WHERE source_key=? AND state="unused"',(value,key))
    model.execute('COMMIT');source.execute('INSERT INTO exact_plan_reuse VALUES(?,?,?,?)',(key,json.dumps(ref),'exact_original_plan_reused','No source claims or model budgets changed'));applied+=1
   except BaseException:model.execute('ROLLBACK');raise
  if applied:print(json.dumps(dict(event='existing_merged_window_plans_reused',sources=applied,pending_unprepared=pending,new_model_calls=0,original_boundaries_unchanged=True)),flush=True)
  if not pending:break
  time.sleep(60)
 source.close();model.close()
if __name__=='__main__':run()
