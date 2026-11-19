from pathlib import Path
import json,hashlib,datetime,tarfile,io
import numpy as np
P=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_unified_clusters80k_20261027_v1')
ROOT=Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher')
import sys
sys.path.insert(0,str(ROOT/'unified_code_snapshot_v1'))
import ms_fullframe_pose2d_draft_v2 as rules
reports=[]
for gpu in range(4):
 commits=sorted((P/'results/a100-extra-03/batches'/('gpu'+str(gpu))).glob('*/commit.json'))
 if not commits:raise ValueError('new epoch batch not reliably committed')
 path=commits[0];commit=json.loads(path.read_bytes());assert commit['worker']=='a100-extra-03' and commit['gpu']==gpu and commit['edition']==P.name
 assert commit['unique_sources_attempted']==20 and commit['unique_sources_with_action_events']==20
 archive=path.parent/'batch.tar';raw=archive.read_bytes();assert hashlib.sha256(raw).hexdigest()==commit['batch_sha256'];expected={str(Path(x['path']).relative_to(path.parent)):x['sha256'] for x in commit['publications']};payloads={}
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as t:
  for m in t:
   assert m.isfile() and not m.issym() and not m.islnk() and '..' not in Path(m.name).parts and not Path(m.name).is_absolute()
   b=t.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==expected[m.name];payloads[m.name]=b
 events=0;frames=0
 for r in commit['receipts']:
  row=r['source_row'];i=r['assignment_index'];assert i%4==gpu;relative='cases/'+row['split']+'/'+row['id'];manifest=json.loads(payloads[relative+'/fullframe_manifest.json']);draft=json.loads(payloads[relative+'/draft.json']);beats=json.loads(payloads[relative+'/beats.json']);bound={'id':row['id'],'split':row['split'],'npz_path':str(path.parent/relative/'pose.npz'),'fullframe_manifest_path':str(path.parent/relative/'fullframe_manifest.json')}
  with np.load(io.BytesIO(payloads[relative+'/pose.npz']),allow_pickle=False) as z:data={k:z[k] for k in z.files}
  mask=rules.validate_observations(bound,manifest,data);assert rules.fullframe_binding(manifest,data)==draft['source']['fullframe_binding'];assert data['schema'].item()=='ms_fullvideo_pose2d_cache_v1' and data['bbox'].shape==(len(data['times']),4)
  assert all(draft[k]==v and manifest[k]==v and r[k]==v for k,v in rules.FLAGS.items()) and draft['events']
  assert manifest['duration_seconds']==float(manifest['metadata_provenance']['actual_probe']['streams'][0]['duration'])
  for e in draft['events']:assert e['text'].strip() and mask[np.ix_(e['support']['frame_indices'],e['support']['joint_indices'])].all()
  assert beats['float16'] is False and beats['dbn'] is False and beats['device']=='cuda:0' and beats['audio_video_alignment_verified'] is False
  for name in ['beats','downbeats']:
   a=np.array(beats[name]);assert np.isfinite(a).all() and np.all(a>=0) and np.all(np.diff(a)>0) and np.all(a<=beats['actual_audio_metadata']['duration_seconds']+.02)
  frames+=len(data['times']);events+=len(draft['events'])
 reports.append({'GPU':gpu,'sources_verified':20,'events':events,'frames':frames,'commit_path':str(path),'archive_sha256':commit['batch_sha256'],'source_indices_mod4_correct':True,'actual_payloads_SHA_and_rule_support_verified':True})
report={'schema':'new_epoch_first_four_batches_acceptance_v1','created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker_id':'a100-extra-03','sources':80,'events':sum(r['events'] for r in reports),'frames':sum(r['frames'] for r in reports),'batches':reports,'GPU_models_reexecuted':False,'old_claims_or_outputs_modified':False,'semantic_approval':False,'quality_approved':False,'training_eligible':False,'needs_visual_revision':True}
for path in [ROOT/'logs/new_epoch03_first80_accepted.json',P/'results/a100-extra-03/first80_independent_acceptance.json']:
 with path.open('x') as f:json.dump(report,f,indent=2);f.write(chr(10));f.flush();__import__('os').fsync(f.fileno())
print(json.dumps(report))
