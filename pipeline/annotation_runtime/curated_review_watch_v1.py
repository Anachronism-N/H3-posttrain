"""Resident independent reviewer: consumes each fixed curated batch as it completes."""
import argparse,hashlib,json,os,sys,tempfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--results-root',required=True);p.add_argument('--out',required=True);p.add_argument('--model',required=True);p.add_argument('--ms-source',required=True);p.add_argument('--annotator-model-reference',required=True);p.add_argument('--poll-seconds',type=int,default=60);a=p.parse_args()
sys.path.insert(0,a.ms_source);import ms_batch_review as review
manifest=Path(a.manifest);summary=json.loads(manifest.read_text());assert summary['total']==222939 and len(summary['batches'])==448
root=Path(a.results_root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
reviewer=review.LocalReviewer(a.model,out,annotator_model_reference=a.annotator_model_reference,hand_roi_root=str(root/'roi'),ffprobe='/opt/ms_tools/ffprobe',ffmpeg='/usr/local/bin/ffmpeg')
def atomic(path,value):
 with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:json.dump(value,f,ensure_ascii=False);f.flush();os.fsync(f.fileno());tmp=Path(f.name)
 tmp.replace(path)
processed={};waiting_for_human=set()
while True:
 newly_reviewed=0
 for b in summary['batches']:
  bid=b['batch_id'];ready=root/'batches'/bid/'batch_ready.json'
  if not ready.exists():continue
  readysha=hashlib.sha256(ready.read_bytes()).hexdigest()
  if processed.get(bid)==readysha:continue
  plans=review.read_batch_list(ready);assert len(plans)==1 and plans[0]['source_ids']==json.loads(ready.read_text())['source_ids']
  assert plans[0]['source_count']==b['count'] and plans[0]['split']==b['split']
  result=review.review_batch(plans[0],out,reviewer)
  assert result['batch_id']==bid
  processed[bid]=readysha;newly_reviewed+=1
  if not result['acceptance']['eligible_for_training']:waiting_for_human.add(bid)
  else:waiting_for_human.discard(bid)
  print(json.dumps({'batch':bid,'reviewed_sources':result['sampled_reviewed_denominator'],'AI_error_candidates':result['confirmed_error_count'],'eligible_for_training':result['acceptance']['eligible_for_training']}),flush=True)
 statuses=[]
 for b in summary['batches']:
  q=out/b['batch_id']/'summary.json'
  if q.exists():statuses.append(json.loads(q.read_text()))
 count=sum(x['acceptance']['review_complete'] for x in statuses)
 value={'schema':'curated_independent_review_watch_v1','total_source_count':222939,'expected_batch_count':448,'batch_review_count':len(statuses),'review_complete_batch_count':count,'sampled_sources_reviewed':sum(x['sampled_reviewed_denominator'] for x in statuses),'AI_error_candidate_count':sum(x['confirmed_error_count'] for x in statuses),'awaiting_human_or_label_completion_batches':sorted(waiting_for_human),'one_resident_reviewer':True,'all_batches_reviewed':count==448,'all_source_labels_completed':len(statuses)==448 and all(x['acceptance']['all_source_labels_completed'] for x in statuses),'human_approval_automatically_granted':False}
 atomic(out/'progress.json',value)
 if count==448 and value['all_source_labels_completed']:
  atomic(out/'run.summary.json',value);break
 time.sleep(a.poll_seconds)
