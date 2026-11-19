"""Incrementally join01-02 final MS commits into the same corrected revision input catalog."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,io,json,os,sqlite3,tarfile,time
from pathlib import Path
import motion_revision_prepare_catalog_v2 as original
import motion_revision_codec_v3 as codec
from stage2_local_publication_v1 import write_once
BASE=original.ROOT/'external_ms_transfer_user_20261028_1793159583747185218'
LOCAL=Path('/root/stage2_motion_revision_inputs_20261028_v2')
SERVICE=Path('/root/stage2_motion_revision_final_MS_intake_20261028_v1')


def run():
 SERVICE.mkdir(exist_ok=True);db=sqlite3.connect(LOCAL/'catalog.db',isolation_level=None,timeout=60);db.execute('CREATE TABLE IF NOT EXISTS incremental_commits(path TEXT PRIMARY KEY,sha256 TEXT,source_count INTEGER,status TEXT,error TEXT)');assignments={};expected=0
 for worker in ('a100-extra-01','a100-extra-02'):
  ticket=json.loads((BASE/'assignments'/(worker+'.ticket.json')).read_bytes());path=Path(ticket['assignment_path']);raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==ticket['assignment_sha256'];rows=[json.loads(x) for x in raw.splitlines() if x];expected+=len(rows);assignments[worker]=(ticket,rows)
 write_once(SERVICE/'input_group_bound.json',dict(original_new_MS_epoch=str(BASE),original_assignment_sources=expected,into_same_input_catalog=str(LOCAL/'catalog.db'),existing_MS_not_restarted=True,revision=codec.REVISION))
 original.codec=codec
 def work(item):
  path,worker=item
  try:
   raw=path.read_bytes();c=json.loads(raw);ticket,assigned=assignments[worker];assert c['schema']=='ms_unified_novlm_batch_commit_v1' and c['edition']==BASE.name and c['worker']==worker and c['assignment_sha256']==ticket['assignment_sha256'] and c['prefetch_drained'] is True;files={x['path']:x for x in c['files']};selected=[]
   for r in c['receipts']:
    if r['status']!='extracted_rule_draft_unapproved':continue
    row=r['source_row'];i=row['assignment_index'];assert {k:v for k,v in row.items() if k!='assignment_index'}==assigned[i] and r['vlm_called'] is False and r['quality_approved'] is False and r['training_eligible'] is False;directory=path.parent/'cases'/row['split']/row['id'];ref=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest());selected.append(dict(id=row['id'],split=row['split'],video_path=row['video_path'],rule_draft_ref=dict(path=str(directory/'draft.json'),sha256=files[str(directory/'draft.json')]['sha256']),fullframe_manifest_ref=dict(path=str(directory/'fullframe_manifest.json'),sha256=files[str(directory/'fullframe_manifest.json')]['sha256']),_commit=ref))
   if not selected:return (path,hashlib.sha256(raw).hexdigest(),[],None)
   values=original.packed(selected);contexts=[original.prepare(row,values) for row in selected];return (path,hashlib.sha256(raw).hexdigest(),contexts,None)
  except Exception as e:return (path,None,[],repr(e))
 def store(path,digest,contexts,error):
  if error:
   db.execute('INSERT OR IGNORE INTO incremental_commits VALUES(?,?,?,?,?)',(str(path),digest,0,'blocked',error));print(json.dumps(dict(event='latest_MS_input_blocked',commit=str(path),error=error,models_called=0)),flush=True);return
  pending=[]
  for c in contexts:
   if not db.execute('SELECT 1 FROM sources WHERE source_key=?',(c['row']['id'].casefold(),)).fetchone():pending.append(c)
  name='latest_MS_'+hashlib.sha256(str(path).encode()).hexdigest();p=SERVICE/(name+'.tar');entries=[]
  if pending:
   fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
   with os.fdopen(fd,'wb') as f:
    with tarfile.open(fileobj=f,mode='w',format=tarfile.PAX_FORMAT) as tar:
     for c in pending:raw=codec.encoded(c);member=c['row']['split']+'/'+c['row']['id']+'.json';m=tarfile.TarInfo(member);m.size=len(raw);tar.addfile(m,io.BytesIO(raw));entries.append((c,member,hashlib.sha256(raw).hexdigest()))
    f.flush();os.fsync(f.fileno())
   ar=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
  db.execute('BEGIN IMMEDIATE')
  try:
   for c,member,sha in entries:
    row=dict(id=c['row']['id'],split=c['row']['split'],video_path=c['row']['video_path'],revision_input_archive=ar,revision_input_member=member,revision_input_sha256=sha,revision=codec.REVISION);db.execute('INSERT INTO sources(source_key,id,split,payload) VALUES(?,?,?,?)',(row['id'].casefold(),row['id'],row['split'],json.dumps(row,separators=(',',':'))))
   if entries:db.execute('INSERT INTO metadata_packs VALUES(?,?)',(name,json.dumps(ar)))
   db.execute('INSERT INTO incremental_commits VALUES(?,?,?,?,?)',(str(path),digest,len(pending),'ready',None));db.execute('COMMIT')
  except BaseException:db.execute('ROLLBACK');raise
 with ThreadPoolExecutor(max_workers=8) as pool:
  while not (SERVICE/'STOP').exists():
   known={x[0] for x in db.execute('SELECT path FROM incremental_commits')};pending=[]
   for worker in assignments:
    for b in (BASE/'results'/worker/'batches').iterdir():
     p=b/'commit.json'
     if str(p) not in known and p.exists():pending.append((p,worker))
   for offset in range(0,len(pending),32):
    for item in pool.map(work,pending[offset:offset+32]):store(*item)
    count=db.execute('SELECT COALESCE(sum(source_count),0) FROM incremental_commits WHERE status="ready"').fetchone()[0];blocked=db.execute('SELECT count(*) FROM incremental_commits WHERE status="blocked"').fetchone()[0];print(json.dumps(dict(event='latest01_02_MS_joined_to_revision',added_unique_sources=count,blocked_commits=blocked,original_assigned_sources=expected,new_VLM_calls=0)),flush=True)
   time.sleep(60)
 db.close()
if __name__=='__main__':run()
