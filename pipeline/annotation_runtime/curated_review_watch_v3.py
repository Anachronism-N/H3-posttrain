"""Incremental independent batch review with fixed full-batch membership."""
import argparse,collections,hashlib,json,os,sys,tempfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--results-root',required=True);p.add_argument('--out',required=True);p.add_argument('--model',required=True);p.add_argument('--ms-source',required=True);p.add_argument('--annotator-model-reference',required=True);p.add_argument('--poll-seconds',type=int,default=60);a=p.parse_args()
sys.path.insert(0,a.ms_source);import ms_batch_review as review
summary=json.loads(Path(a.manifest).read_text());assert summary['total']==222939 and len(summary['batches'])==448
root=Path(a.results_root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);stream_path=root/'completed.jsonl'
reviewer=review.LocalReviewer(a.model,out,annotator_model_reference=a.annotator_model_reference,hand_roi_root=str(root/'roi'),ffprobe='/opt/ms_tools/ffprobe',ffmpeg='/usr/local/bin/ffmpeg')
descriptors={b['batch_id']:b for b in summary['batches']};members={};completed=collections.defaultdict(dict);processed={};offset=0;input_inode=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,value):
 with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:json.dump(value,f,ensure_ascii=False);f.flush();os.fsync(f.fileno());tmp=Path(f.name)
 tmp.replace(path)
def batch_rows(bid):
 if bid not in members:
  b=descriptors[bid];assert sha(b['path'])==b['sha256'];rows=[json.loads(l) for l in Path(b['path']).read_text().splitlines() if l];assert len(rows)==b['count'] and all(r['batch_id']==bid and r['split']==b['split'] for r in rows);members[bid]={r['id']:r for r in rows}
 return members[bid]

while True:
 dirty=set()
 if stream_path.exists():
  st=stream_path.stat();identity=(st.st_dev,st.st_ino)
  if input_inode is None:input_inode=identity
  assert identity==input_inode and st.st_size>=offset,'consumer stream replaced or truncated'
  with stream_path.open('rb') as f:
   f.seek(offset)
   while True:
    pos=f.tell();line=f.readline()
    if not line or not line.endswith(bytes([10])):offset=pos;break
    v=json.loads(line);bid=v['batch_id'];original=batch_rows(bid);assert v['id'] in original and v['split']==descriptors[bid]['split']
    if v.get('joint_candidate_ref'):
     ref=v['joint_candidate_ref'];assert sha(ref['path'])==ref['sha256'];r=json.loads(Path(ref['path']).read_text());assert r['id']==v['id'] and r['batch_id']==bid;r['label_path']=ref['path'];r['label_sha256']=ref['sha256'];r['label_status']='completed' if r['joint_complete'] else 'partial'
    else:
     r={**original[v['id']],'label_status':'failed','label':{'dance_style':'unknown','timed_motion_script':''},'consumer_status':v['status']}
    if v['id'] in completed[bid]:assert completed[bid][v['id']]==r,'consumer source revision changed without explicit repair'
    else:completed[bid][v['id']]=r;dirty.add(bid)
    offset=f.tell()
 for bid in sorted(dirty):
  source_members=batch_rows(bid);rows=[]
  for sid,original in source_members.items():rows.append(completed[bid].get(sid,{**original,'label_status':'pending'}))
  # Stable membership/random order and pending source records are retained.
  result=review.review_batch_ready(rows,out,reviewer,batch_id=bid,expected_source_ids=list(source_members),sample_count=25,risk_count=5)
  print(json.dumps({'batch':bid,'labels_terminal':len(completed[bid]),'source_count':len(source_members),'sources_reviewed':result['sampled_reviewed_denominator'],'AI_error_candidates':result['confirmed_error_count'],'state':result['state'],'eligible_for_training':result['acceptance']['eligible_for_training']}),flush=True)
 statuses=[]
 for bid in completed:
  path=out/bid/'summary.json'
  if path.exists():statuses.append(json.loads(path.read_text()))
 count=sum(x['acceptance']['review_complete'] for x in statuses)
 value={'schema':'curated_independent_review_watch_v1','incremental_labels_reviewed_without_waiting_full_batch':True,'total_source_count':222939,'expected_batch_count':448,'batch_review_count':len(statuses),'review_complete_batch_count':count,'sampled_sources_reviewed':sum(x['sampled_reviewed_denominator'] for x in statuses),'AI_error_candidate_count':sum(x['confirmed_error_count'] for x in statuses),'awaiting_human_or_label_completion_batches':[x['batch_id'] for x in statuses if not x['acceptance']['eligible_for_training']],'one_resident_reviewer':True,'all_batches_reviewed':count==448,'all_source_labels_completed':len(statuses)==448 and all(x['acceptance']['all_source_labels_completed'] for x in statuses),'human_approval_automatically_granted':False,'consumer_stream_offset':offset}
 atomic(out/'progress.json',value)
 if count==448 and value['all_source_labels_completed']:atomic(out/'run.summary.json',value);break
 time.sleep(a.poll_seconds)
