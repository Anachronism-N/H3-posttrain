from pathlib import Path
import argparse,hashlib,json,os,sys,time,importlib.util
ROOT=Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher');P=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_unified_run_20261027_v1')
sys.path.insert(0,str(ROOT))
import unified_access_transport_v3 as transport
spec=importlib.util.spec_from_file_location('lane_v4',ROOT/'unified_lane_worker_v4.py');lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)
args=argparse.Namespace(root=ROOT,worker='a100-extra-03',gpu=0,ticket=str(P/'assignments/a100-extra-03.ticket.json'),manifest=str(P/'assignments/a100-extra-03.jsonl'),code_root=ROOT/'unified_code_snapshot_v1',scail_root=ROOT/'unified_scail_local_v1',beat_checkpoint=ROOT/'model_cache/BeatThis/final0.verified.ckpt',relay_host=None,resume_first3=True)
a=transport.Access(args.worker);ticket,rows,sha,sigs,helper=lane.verify_assignment(args,a)
v=json.loads((ROOT/'logs/unified_gpu0_progress.json').read_bytes());target=P/'results/a100-extra-03/batches/gpu0'/(v['run_id']+'-batch000')/'commit.json'
raw=target.read_bytes();commit=json.loads(raw);lease=json.loads((ROOT/'gpu_locks/physical_gpu0.lease.json').read_bytes())
marker={'edition':lane.EDITION,'worker':args.worker,'assignment_sha256':sha,'code_signatures':sigs,'indices':[0,4,8],'commit_path':str(target),'commit_sha256':hashlib.sha256(raw).hexdigest(),'owner_pid':v['pid'],'verified_by_pid':os.getpid(),'verified_unix':time.time(),'gpu_uuid':lease['gpu_uuid'],'recovery_scope':'only readonly verified published first3;no model or source retry'}
instance=lane.Lane(args,a,ticket,rows,sha,sigs,helper);instance.verify_first3(marker)
sys.path.insert(0,str(args.code_root));import ms_unified_novlm_v1 as core
core.write_once(instance.marker,core.json_bytes(marker));a.publish_json(marker,str(P/'results/a100-extra-03/first3_verified.json'))
print(json.dumps({'published_first3_verified':True,'models_executed':0,'claims_modified':False,'indices_excluded':[0,4,8],'marker_path':str(instance.marker),'remaining_gpu0':247}))
