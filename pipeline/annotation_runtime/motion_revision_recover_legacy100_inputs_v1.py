"""Recover existing early pilot metadata and actual beat-cache names, without model rerun."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,io,json,os,sqlite3,tarfile,time
from pathlib import Path
import motion_revision_codec_v3 as codec
import motion_revision_prepare_catalog_v2 as inputs
from stage2_local_publication_v1 import write_once
ROOT=Path('/root/stage2_motion_revision_inputs_20261028_v2')

def run():
 db=sqlite3.connect(ROOT/'catalog.db',isolation_level=None,timeout=60);errors=dict(db.execute('SELECT source_key,error FROM errors'));rows,_=inputs.gather();rows=[r for r in rows if r['id'].casefold() in errors and 'external_a100_fullframe_pilot_20261026_v1/results/a100-extra-01' in r['fullframe_manifest_ref']['path']];ready=[];bad=[]
 def one(row):
  try:
   dr=Path(row['rule_draft_ref']['path']).read_bytes();mr=Path(row['fullframe_manifest_ref']['path']).read_bytes();assert hashlib.sha256(dr).hexdigest()==row['rule_draft_ref']['sha256'] and hashlib.sha256(mr).hexdigest()==row['fullframe_manifest_ref']['sha256'];d=json.loads(dr);m=json.loads(mr);p=Path(row['fullframe_manifest_ref']['path']).parent/'actual_audio_beats.json';br=p.read_bytes();original_b=json.loads(br);assert 'schema' not in original_b and original_b['status']=='success' and original_b['dbn'] is False and original_b['float16'] is False and original_b['dtype']=='torch.float32';binding=m['metadata_provenance']['media_binding'];assert binding['video_path']['sha256']==m['video_sha256'] and Path(binding['audio_path']['original_path']).stem==row['id'] and m['metadata_provenance']['models']['beatthis']==original_b['checkpoint'];b=dict(schema='ms_beatthis_audio_clockonly_v1',id=row['id'],split=row['split'],clock='audio_only',beats=original_b['beats'],downbeats=original_b['downbeats']);plan=codec.merged_plan(m,b,dict(path=str(p),sha256=hashlib.sha256(br).hexdigest(),original_schema=None,adapter='recorded_legacy_actual_audio_beats_without_schema_v1',source_audio_sha256=binding['audio_path']['sha256'],original_raw_arrays_preserved=True));rich=inputs.RICH/(row['id']+'.txt');v=dict(schema='motion_revision_prepared_metadata_v1',revision=codec.REVISION,row=row,draft=d,manifest=m,plan=plan,style_hint=codec.style_hint(rich.read_text() if rich.exists() else ''),window_frames=codec.select_window_frames(m,plan),rich_body_forwarded_to_VLM=False,original_draft_sha256=hashlib.sha256(dr).hexdigest(),original_manifest_sha256=hashlib.sha256(mr).hexdigest(),window_plan_sha256=hashlib.sha256(codec.encoded(plan)).hexdigest(),quality_approved=False,training_eligible=False,original_beat_inference_not_repeated=True);return v,None
  except Exception as e:return None,dict(id=row['id'],error=repr(e))
 with ThreadPoolExecutor(max_workers=8) as pool:
  for v,e in pool.map(one,rows):
   if v:ready.append(v)
   else:bad.append(e)
 out=ROOT/'recovered_legacy100_metadata';out.mkdir(exist_ok=False);db.execute('CREATE TABLE IF NOT EXISTS resolved_input_errors(source_key TEXT,error TEXT,reason TEXT,resolved_unix REAL)')
 for offset in range(0,len(ready),50):
  part=ready[offset:offset+50];name='recovered_legacy100%06d'%(offset//50);path=out/(name+'.tar');fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);entries=[]
  with os.fdopen(fd,'wb') as f:
   with tarfile.open(fileobj=f,mode='w',format=tarfile.PAX_FORMAT) as tar:
    for v in part:raw=codec.encoded(v);member=v['row']['split']+'/'+v['row']['id']+'.json';i=tarfile.TarInfo(member);i.size=len(raw);tar.addfile(i,io.BytesIO(raw));entries.append((v,member,hashlib.sha256(raw).hexdigest()))
   f.flush();os.fsync(f.fileno())
  ar=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size);db.execute('BEGIN IMMEDIATE')
  try:
   for v,member,digest in entries:
    row=dict(id=v['row']['id'],split=v['row']['split'],video_path=v['row']['video_path'],revision_input_archive=ar,revision_input_member=member,revision_input_sha256=digest,revision=codec.REVISION);key=row['id'].casefold();assert not db.execute('SELECT1 FROM sources WHERE source_key=?'.replace('SELECT1','SELECT 1'),(key,)).fetchone();db.execute('INSERT INTO sources(source_key,id,split,payload) VALUES(?,?,?,?)',(key,row['id'],row['split'],json.dumps(row)));db.execute('INSERT INTO resolved_input_errors VALUES(?,?,?,?)',(key,errors[key],'Original actual_audio_beats cache and separately exported draft reused',time.time()));db.execute('DELETE FROM errors WHERE source_key=?',(key,))
   db.execute('INSERT INTO metadata_packs VALUES(?,?)',(name,json.dumps(ar)));db.execute('COMMIT')
  except BaseException:db.execute('ROLLBACK');raise
 write_once(inputs.OUT/'legacy100_inputs_recovered.json',dict(recovered_sources=len(ready),not_recovered=bad,old_raw_bytes_preserved=True,beat_model_calls=0,VLM_model_calls=0,observed_unix=time.time()));db.close();print(json.dumps(dict(early_pilot_input_references_fixed=len(ready),blocked=len(bad),old_beats_reused=True,model_calls=0)),flush=True)
if __name__=='__main__':run()
