"""Read current draft progress only; never inspect media, evidence trees or weights."""
import argparse
import json
from pathlib import Path
import time
BAR=Path(__file__).resolve().parent.parent

def poll(names):
    lanes=[];pools={}
    for name in names:
        q=BAR/'queues'/name;config=BAR/'runtime'/(name+'_launch.json')
        if not config.exists():lanes.append(dict(lane=name,status='no_launch'));continue
        cfg=json.loads(config.read_bytes());path=Path(cfg['draft_out'])/'progress.json'
        value=json.loads(path.read_bytes()) if path.exists() else dict(processed=0,selected=cfg['targetcount'])
        failure=json.loads((q/'failure.json').read_bytes()).get('error') if (q/'failure.json').exists() else None
        value.pop('completed_source_keys',None)
        value.update(lane=name,node=cfg['node'],gpus=cfg['gpus'],failure=failure,
            native_admitted=(q/'admission.json').exists(),child_started=(q/'live.json').exists(),
            resources_safe=(q/'resources_safe.json').exists(),draft_only=True,training_eligible=False)
        if cfg.get('pool_root'):
            value.update(scope='worker',pool_worker_id=cfg['pool_worker_id'],selected_scope='pool_eligibility_not_worker_completed')
            pool_root=cfg['pool_root']
            if pool_root not in pools:
                pool_path=Path(pool_root)/'progress.json'
                pools[pool_root]=json.loads(pool_path.read_bytes()) if pool_path.exists() else dict(scope_pool=True,N=cfg['targetcount'],status='no_pool_progress')
                pools[pool_root]['pool_root']=pool_root
        lanes.append(value)
    return dict(UTC=time.strftime('%FT%TZ',time.gmtime()),lanes=lanes,pools=list(pools.values()),
        generated_draft_sources=sum(x.get('processed',0) for x in lanes),quality_approved_inferred=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--lane',action='append',required=True);p.add_argument('--interval',type=float,default=60);p.add_argument('--watch',action='store_true')
    args=p.parse_args();previous=None
    while True:
        value=poll(args.lane)
        if previous:
            elapsed=time.monotonic()-previous[0]
            value['new_drafts_since_previous']=value['generated_draft_sources']-previous[1]
            value['sample_drafts_per_hour']=value['new_drafts_since_previous']*3600/elapsed
        print(json.dumps(value,ensure_ascii=False),flush=True)
        previous=(time.monotonic(),value['generated_draft_sources'])
        if not args.watch:break
        time.sleep(args.interval)
if __name__=='__main__':main()
