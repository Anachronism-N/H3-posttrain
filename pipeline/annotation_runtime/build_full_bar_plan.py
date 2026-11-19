import argparse,copy,hashlib,json,sys
from pathlib import Path
ROOT=Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS=ROOT/'diff/logs_omnidance/exp/MS';OUT=MS/'bar_annotation_optimized_20261024_v1'
REPO=ROOT/'H3-posttrain';sys.path.insert(0,str(REPO/'scripts_omnidance/exp/ms'))
import ms_visual_batch as batch
import ms_music_windows as music
from motionscript_v2.chunks import fixed_windows
FULL=MS/'chunk_caption12k_input_v1/manifest.jsonl'
LINEAGE=MS/'ms_chunk12k_final_export_v1/lineage.jsonl'
CHECKPOINT='8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_bytes()!=b:raise ValueError('Different existing plan '+str(p))
  return
 with p.open('xb') as f:f.write(b)
def dumps(rows):return ''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n' for r in rows).encode()
def merged_supported(windows,minimum=.75):
 result=copy.deepcopy(windows);changes=[];i=0
 while len(result)>1 and i<len(result):
  w=result[i]
  if w['end']-w['start']>=minimum:i+=1;continue
  before=copy.deepcopy(w)
  if i<len(result)-1:
   nxt=result[i+1];nxt['start']=w['start'];nxt['raw_window_indices']=w['raw_window_indices']+nxt['raw_window_indices'];result.pop(i)
  else:
   prev=result[i-1];prev['end']=w['end'];prev['raw_window_indices']+=w['raw_window_indices'];result.pop(i);i=max(0,i-1)
  changes.append({'short_window':before,'reason':'insufficient_temporal_context_for_visual_motion','minimum_seconds':minimum})
 return result,changes
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=12000);ap.add_argument('--output',required=True);args=ap.parse_args()
 fullraw=FULL.read_bytes();fullsha=hashlib.sha256(fullraw).hexdigest()
 if fullsha!='4b2efc8233fed8dd7cdb46acf8da318db76cb575faa1177e9e2bc113d453d283':raise ValueError('Source input changed')
 rows=[json.loads(l) for l in fullraw.decode().splitlines() if l];rows=rows[:args.count]
 wanted={r['id'] for r in rows};lineage={}
 with LINEAGE.open() as f:
  for line in f:
   r=json.loads(line)
   if r['id'] in wanted:lineage[r['id']]=r
 if set(lineage)!=wanted:raise ValueError('Prior verified lineage missing')
 oldpath=MS/'beatthis_first50_v1/detections.jsonl';oldreceipt=MS/'beatthis_first50_v1/run_receipt.json'
 detectors={}
 for sourcepath,receiptpath in [(oldpath,oldreceipt)]+[(OUT/f'beat_shard{i}_v1/detections.jsonl',OUT/f'beat_shard{i}_v1/run_receipt.json') for i in range(4)]:
  if not sourcepath.exists():continue
  receipt=json.loads(receiptpath.read_text())
  if receipt['model']['checkpoint_sha256']!=CHECKPOINT:raise ValueError('Wrong BeatThis model')
  with sourcepath.open() as f:
   for line in f:
    try:d=json.loads(line)
    except json.JSONDecodeError:break
    if d['id'] not in wanted:continue
    if d['id'] in detectors:raise ValueError('Duplicate detection source')
    detectors[d['id']]=(d,receipt,sourcepath,receiptpath)
 if set(detectors)!=wanted:raise ValueError('Still missing detections: '+str(len(wanted-set(detectors))))
 plans=[];stats={'sources':len(rows),'window_count':0,'boundary_count':0,'detected_bar_sources':0,'fixed_fallback_sources':0,'short_context_merged_sources':0}
 for row in rows:
  sid=row['id'];prior=lineage[sid];d,receipt,detectpath,receiptpath=detectors[sid];actual=prior['actual_media']
  if prior['status']!='processed' or not actual['media_end_verified_from_video']:raise ValueError('Prior actual media missing')
  if d['audio_sha256']!=prior['audio_sha256']:raise ValueError('Audio source SHA mismatch '+sid)
  if d['source_row']['audio_path']!=row['audio_path']:raise ValueError('Wrong detection row')
  n=actual['decoded_video_frames'];fps=actual['video_fps'];expectedtimes=[i/fps for i in range(n)]
  binding=dict(id=sid,input_sha256=prior['input_sha256'],video_sha256=prior['video_sha256'],audio_sha256=prior['audio_sha256'],
   actual_video_pts_sha256=batch.digest(expectedtimes),media_start=actual['media_start'],media_end=actual['media_end'],
   video_pts_offset=actual['video_pts_offset'],video_fps=fps,decoded_video_frames=n)
  original_db=d['downbeats'] if d.get('status')=='ok' else [];db=[v for v in original_db if 0<=v<=binding['media_end']]
  source={'method':'beat_this.inference.File2Beats','checkpoint_name':'final0','dbn':False,'device':'cpu','synthetic':False,
   'downbeats_origin':'model_downbeat_head_postprocessed','checkpoint_sha256':CHECKPOINT,'audio_sha256':binding['audio_sha256'],
   'library_source_sha256':receipt['runtime']['beatthis_inference_sha256'],'library_version':d['library_versions']['beat-this'],
   'normalized_audio_video_offset':'unknown','original_downbeats_seconds':original_db,'model_timestamps_outside_video_omitted':[v for v in original_db if v not in db]}
  barplan=music.bar_windows(binding,db,source=source,policy='merge_short_edges',min_edge_seconds=1.5)
  if len(db)<2:
   stats['fixed_fallback_sources']+=1
   barplan['windows']=fixed_windows(0,binding['media_end'],2.,1.5)
   for w in barplan['windows']:w['raw_window_indices']=[i for i,r in enumerate(barplan['raw_windows']) if r['start']<w['end'] and r['end']>w['start']]
   barplan['status']='ready_fixed_fallback';barplan['fallback_reason']='insufficient_detected_downbeats' if d.get('status')=='ok' else 'beat_detection_failed: '+d.get('error_message','unknown');barplan['fallback_is_verified_bar']=False
  else:stats['detected_bar_sources']+=1
  original_windows=copy.deepcopy(barplan['windows']);final,changes=merged_supported(original_windows)
  if changes:stats['short_context_merged_sources']+=1
  barplan['windows']=final;barplan['pre_visual_support_windows']=original_windows;barplan['visual_support_merges']=changes
  barplan['annotation_policy']='detected_bar_merge_edges_visual_min075_v1';barplan['one_window_is_exactly_one_bar']=False
  barplan['params']['visual_min_seconds']=.75;barplan['params']['short_interior_bars_merged']=bool(changes)
  barplan['expected_time_grid_provenance']='prior_verified_constant_fps_metadata; new_consumer_must_probe_actual_video_pts'
  refs={'manifest':{'path':str(FULL),'sha256':fullsha},'lineage':{'path':prior['report_path'],'sha256':prior['report_sha256']},
   'beat_receipt':{'path':str(receiptpath),'sha256':sha(receiptpath)},
   'beat_source':{'path':str(REPO/'models/BeatThis/source_manifest.json'),'sha256':receipt['model']['source_manifest_sha256']}}
  # Full detector JSONL keeps every raw model output; the row below carries the exact selected detection plus lineage references.
  plan={'schema':'bar_visual_annotation_plan_v1','id':sid,'manifest_record':row,'binding':binding,'window_manifest':barplan,'windows':copy.deepcopy(final),
   'plan_status':barplan['status'],'data_source':'full12k_detected_bar_visual_support_v1','source_refs':refs,
   'beat_detection':d,'beat_output_path':str(detectpath),'time_grid_provenance':barplan['expected_time_grid_provenance'],
   'prior_media_probe_metadata':actual,'cache_video_identity_verified':False,'semantic_approval':False,'human_review_completed':False,'formal_training':False}
  plans.append(plan);stats['window_count']+=len(final);stats['boundary_count']+=max(0,len(final)-1)
 output=Path(args.output);data=dumps(plans);save(output,data)
 report=dict(stats,plan_path=str(output),plan_sha256=hashlib.sha256(data).hexdigest(),source_input_sha256=fullsha,
  media_reprobed_this_stage=False,actual_pts_verification_required_at_worker=True,original_labels_modified=False,semantic_approval=False,formal_training=False)
 save(output.with_suffix('.summary.json'),(json.dumps(report,indent=2)+'\n').encode());print(json.dumps(report))
if __name__=='__main__':main()
