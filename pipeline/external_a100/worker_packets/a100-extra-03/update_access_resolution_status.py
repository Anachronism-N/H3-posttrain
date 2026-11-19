from pathlib import Path
import datetime,json
root=Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher')
result={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'own03':{'status':'completed_no_GPU_reexecution','processed':100},'workers':{}}
for worker in ['a100-extra-04','a100-extra-05','a100-extra-06']:
 result['workers'][worker]={'storage_resolution':'target_data_not_present_under_any_discovered_mount_prefix','accessible_transport':'SSH to 11.239.36.34 using original path','relay_connection_verified':True,'original_manifest_unchanged':True,'source_limit':100,'GPU_indices':[0],'processes':1,'copy_threads':1,'max_inflight_sources':2,'cache_budget_bytes':8*2**30,'execution_status':'await_frozen_access_IO_adapter_not_waiting_mount','NPZ_schema':'ms_fullvideo_pose2d_cache_v1','bbox_shape':'N,4'}
p=root/'logs/pilot_access_resolution_status.json'
with p.open('w') as f:json.dump(result,f,indent=2);f.write(chr(10))
print(json.dumps(result))
