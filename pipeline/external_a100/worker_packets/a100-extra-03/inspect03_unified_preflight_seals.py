from pathlib import Path
import json,hashlib,datetime
ROOT=Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher');P=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_unified_run_20261027_v1')
t=json.loads((P/'assignments/a100-extra-03.ticket.json').read_bytes());rows=[json.loads(l) for l in Path(t['assignment_path']).read_bytes().splitlines()];claims=[]
for r in rows:
 p=Path(t['claims_root'])/r['split']/(r['id']+'.json')
 if p.exists():claims.append(json.loads(p.read_bytes()))
oldpids={0:[],1:[],2:[],3:[]}
for p in (ROOT/'gpu_locks').glob('physical_gpu*.lease.json'):
 v=json.loads(p.read_bytes());gpu=v.get('physical_gpu_index')
 if gpu in oldpids:oldpids[gpu].append({'file':str(p),'pid':v['pid'],'alive':Path('/proc/'+str(v['pid'])).exists(),'claims_for_pid':sum(c.get('pid')==v['pid'] for c in claims),'status':v.get('status')})
print(json.dumps({'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'own_claims':len(claims),'claimed_assignment_indices':[c.get('assignment_index') for c in claims],'leases':oldpids,'source_GPU_retries':0}))
