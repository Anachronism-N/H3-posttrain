"""Publish original MS metadata packs and a portable manifest once, without re-extraction."""
import hashlib,json,os,sqlite3,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
RT=Path(__file__).parent
ROOT=RT.parent/'original_MS_221887_shared_v1'

def put(path,value):
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'w') as f:json.dump(value,f);f.flush();os.fsync(f.fileno())

def run():
 ROOT.mkdir(exist_ok=False);packs=ROOT/'metadata_packs';packs.mkdir();db=sqlite3.connect('file:/root/stage2_motion_revision_inputs_20261028_v2/catalog.db?mode=ro',uri=True);rows=[json.loads(x[0]) for x in db.execute('SELECT payload FROM sources ORDER BY ordinal')];blocked=db.execute('SELECT count(*) FROM errors').fetchone()[0];db.close();unique={r['revision_input_archive']['sha256']:r['revision_input_archive'] for r in rows}
 fd=os.open(ROOT/'source_manifest.jsonl',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'w') as f:
  for r in rows:
   ref=r['revision_input_archive'];value=dict(id=r['id'],split=r['split'],video_path=r['video_path'],original_metadata_archive=str(packs/(ref['sha256']+'.tar')),original_metadata_archive_sha256=ref['sha256'],original_metadata_archive_bytes=ref['bytes'],original_metadata_member=r['revision_input_member'],original_metadata_sha256=r['revision_input_sha256'],contains='Original MotionScript draft/events, original fullframe manifest, immutable merged-edge window plan and rich-only dance-style candidates; not VLM rewritten caption')
   f.write(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n')
  f.flush();os.fsync(f.fileno())
 put(ROOT/'export_started.json',dict(source_count=len(rows),metadata_packs=len(unique),expected_bytes=sum(x['bytes'] for x in unique.values()),blocked_inputs_not_included=blocked,source_manifest_ready=True,metadata_copy_in_progress=True,observed_unix=time.time()))
 print(json.dumps(dict(event='original_MS_manifest_ready',sources=len(rows),manifest=str(ROOT/'source_manifest.jsonl'),metadata_packs=len(unique))),flush=True)
 def copy(ref):
  destination=packs/(ref['sha256']+'.tar');temporary=destination.with_suffix('.copying');digest=hashlib.sha256();total=0;fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with Path(ref['path']).open('rb') as source,os.fdopen(fd,'wb') as out:
   for block in iter(lambda:source.read(8*2**20),b''):digest.update(block);total+=len(block);out.write(block)
   out.flush();os.fsync(out.fileno())
  assert total==ref['bytes'] and digest.hexdigest()==ref['sha256'],'Original metadata pack changed';os.link(temporary,destination);temporary.unlink();return total
 copied=0
 with ThreadPoolExecutor(max_workers=4) as executor:
  for i,size in enumerate(executor.map(copy,unique.values()),1):
   copied+=size
   if i%200==0:print(json.dumps(dict(event='original_MS_packs_copied',packs=i,of=len(unique),copied_bytes=copied)),flush=True)
 fd=os.open(packs,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);put(ROOT/'EXPORT_COMPLETE.json',dict(source_count=len(rows),metadata_packs=len(unique),copied_bytes=copied,source_manifest=str(ROOT/'source_manifest.jsonl'),original_MS_not_rewritten=True,VLM_captions_included=False,quality_approved=False,training_eligible=False,completed_unix=time.time()));print(json.dumps(dict(event='original_MS_export_complete',sources=len(rows),shared_root=str(ROOT),copied_bytes=copied)),flush=True)
if __name__=='__main__':run()
