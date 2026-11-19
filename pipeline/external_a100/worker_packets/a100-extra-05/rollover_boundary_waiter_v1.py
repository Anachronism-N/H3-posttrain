"""One-shot safe-boundary continuation. Never stops or retries existing GPU jobs."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys,time
P=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_unified_clusters80k_20261027_v1')
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',required=True);p.add_argument('--gpu',type=int,choices=range(4),required=True);a=p.parse_args()
if a.worker not in ['a100-extra-04','a100-extra-05','a100-extra-06'] or a.root!=Path('/tmp/ms_pose2d_draft_20261026T1427Z')/('worker-'+str(int(a.worker[-2:])-4)):raise ValueError('wrong own node scope')
previous=a.root/'logs'/('unified_gpu'+str(a.gpu)+'_progress.json');newprogress=a.root/'logs'/('rollover_gpu'+str(a.gpu)+'_progress.json');started=time.monotonic();waitrecord=a.root/'logs'/('rollover_waiter_gpu'+str(a.gpu)+'.json')
def report(status,**fields):
 v={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'waiter_pid':os.getpid(),'worker':a.worker,'gpu':a.gpu,'status':status,**fields};tmp=waitrecord.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(v,f);f.write(chr(10));f.flush();os.fsync(f.fileno())
 tmp.replace(waitrecord);print(json.dumps(v),flush=True)
report('waiting_existing_lane_safe_completion',next_source_scope='20000 per node',old_workers_unchanged=True)
while time.monotonic()-started<3*86400:
 raw=previous.read_bytes()
 if bytes([0]) in raw or not raw.endswith(bytes([10])):
  report('blocked_partial_or_NUL_progress',cursor_retained=True);sys.exit(1)
 v=json.loads(raw);pid=v['pid'];proc=Path('/proc/'+str(pid));alive=False
 if proc.exists():
  try:alive=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])==v['start_ticks']
  except OSError:alive=False
 if newprogress.exists():
  report('existing_next_ticket_consumption_not_restarted',next_progress=str(newprogress));sys.exit(0)
 if v['status'] in ['completed','failed_sealed_no_retry'] and not alive and v.get('ended_unix'):
  report('safe_old_terminal_boundary',old_pid=pid,old_status=v['status'],old_published=v['published'],old_source_failed=v['failed']);break
 if not alive and v['status'] not in ['completed','failed_sealed_no_retry']:
  report('blocked_old_exit_without_terminal',old_pid=pid,old_status=v['status'],no_retry=True);sys.exit(1)
 report('waiting_existing_lane_safe_completion',old_pid=pid,old_processed=v['unique_processed'],old_published=v['published'],old_status=v['status'])
 time.sleep(60)
else:
 report('waiter_safety_cap_expired',old_workers_unchanged=True);sys.exit(1)
args=[str(a.root/'venv/bin/python'),'-u',str(a.root/'unified_lane_rollover_v1.py'),'--root',str(a.root),'--worker',a.worker,'--gpu',str(a.gpu),'--ticket',str(P/'assignments'/(a.worker+'.ticket.json')),'--manifest',str(P/'assignments'/(a.worker+'.jsonl')),'--queue',str(P/(a.worker+'.queue.json')),'--previous-progress',str(previous),'--code-root',str(a.root/'unified_code_snapshot_v1'),'--scail-root',str(a.root/'unified_scail_local_v1'),'--beat-checkpoint',str(a.root/'model_cache/BeatThis/final0.verified.ckpt'),'--relay-host','11.239.36.34']
log=a.root/'logs'/('rollover_gpu'+str(a.gpu)+'_v1.log');report('starting_exact_next_unconsumed_ticket',old_pid=pid,no_old_source_retried=True)
with log.open('x') as f:
 result=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(a.gpu),'CUDA_DEVICE_ORDER':'PCI_BUS_ID','PYTHONWARNINGS':'ignore','PYTHONDONTWRITEBYTECODE':'1'})
report('next_lane_completed' if result.returncode==0 else 'next_lane_failed_sealed_no_retry',returncode=result.returncode,log=str(log),automatic_retry=False)
sys.exit(result.returncode)
