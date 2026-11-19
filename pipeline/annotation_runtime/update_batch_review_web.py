import argparse,json,hashlib,os,tempfile,time
from pathlib import Path
ROOT=Path('/apdcephfs_gy4/share_302533218/cedricnie');EXP=ROOT/'diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1';WEB=ROOT/'diff/logs_omnidance/exp/review_web/batch_quality_v1';PRODUCTION=EXP/'curated222939_production_v1';MANIFEST=EXP/'curated222939_inputs_v1/manifest.summary.json'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def refresh():
 summary=load(MANIFEST);batches=[]
 pilot=EXP/'independent_review_pilot25_v1'
 for batch in summary['batches']:
  bid=batch['batch_id'];ready=PRODUCTION/'batches'/bid/'batch_ready.json';review=EXP/'curated_independent_reviews_v1'/bid/'summary.json';queue=review.with_name('review_web.json')
  if ready.exists() or review.exists():
   b={'batch_id':bid,'split':batch['split'],'count':batch['count'],'label_status':'ready' if ready.exists() else 'pending','review_status':'pending'}
   if ready.exists():v=load(ready);b['joint_complete_count']=v['joint_complete_count'];b['label_ref']={'path':str(ready),'sha256':sha(ready)}
   if review.exists():v=load(review);b.update(review_status=v.get('state','reviewed'),reviewed=v['sampled_reviewed_denominator'],error_count=v['confirmed_error_count'],acceptance=v['acceptance']);b['review_summary_ref']={'path':str(review),'sha256':sha(review)}
   if queue.exists():b['review_queue_ref']={'path':str(queue),'sha256':sha(queue)}
   batches.append(b)
 for p in pilot.glob('*/summary.json'):
  v=load(p);q=p.with_name('review_web.json');b={'batch_id':v['batch_id'],'split':'train','count':v['source_count'],'pilot_only':True,'label_status':'ready','review_status':'reviewed','reviewed':v['sampled_reviewed_denominator'],'error_count':v['confirmed_error_count'],'acceptance':v['acceptance'],'review_summary_ref':{'path':str(p),'sha256':sha(p)}}
  if q.exists():b['review_queue_ref']={'path':str(q),'sha256':sha(q)}
  batches.insert(0,b)
 progress=[]
 for i in range(4):
  q=EXP/'queues'/('curated_producer_p%d_v2'%i);r={'task':'producer','shard':i,'admitted':(q/'admission.json').exists(),'failed':(q/'failure.json').exists()};progress.append(r)
 for i in range(4):
  q=EXP/'queues'/('curated_consumer_s%d_v1'%i);progress.append({'task':'joint_visual','shard':i,'admitted':(q/'admission.json').exists(),'failed':(q/'failure.json').exists()})
 value={'schema':'curated_batch_quality_web_v1','total_sources':summary['total'],'expected_batches':len(summary['batches']),'counts':summary['counts'],'updated_utc':time.strftime('%FT%TZ',time.gmtime()),'batches':batches,'GPU_tasks':progress,'completed_all_labels':False,'all_batches_reviewed':False,'AI_errors_require_human_review':True,'scope':'complete_curated_not_only_12k','caption_format':'Dance style: xxx + timed MotionScript'}
 WEB.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',dir=WEB,delete=False) as f:json.dump(value,f,ensure_ascii=False);f.flush();os.fsync(f.fileno());p=Path(f.name)
 p.replace(WEB/'index.json');print(json.dumps({'web_index':str(WEB/'index.json'),'ready_batches':len(batches),'all_labeling_completed':False}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 while True:
  refresh()
  if not a.watch:break
  time.sleep(60)
