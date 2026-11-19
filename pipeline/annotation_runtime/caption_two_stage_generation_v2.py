"""Three basic check classes in the stage-one draft lane; revision is deferred."""
import copy
from pathlib import Path
import time
import caption_batch8_production_consumer_v2 as ownership
import caption_two_stage_light_v1 as light


def run_consumer(args,runtime,resource_admission=None):
    light.need(callable(resource_admission) and resource_admission(args,runtime) is True,'Actual caller-owned resource safety required')
    light.need(args.stage=='run' and args.batch_size==8 and args.max_num_seqs>=8 and not args.retry_failed,'Batch8 without retries required')
    # Exclusivity and GPU authorization are startup checks, not repeated per window.
    handoff=ownership.verify_handoff(args,ownership.Inputs())
    selected=handoff['admission']['selected']
    light.need(selected,'Nonempty exclusive source assignment required')
    out=Path(args.out).resolve();expected=copy.deepcopy(vars(args));inputs=light.Inputs()
    light.write_once(out/'protocol.json',dict(recipe=light.RECIPE,handoff_ref=args.handoff_ref,
        selected=[x['meta'] for x in selected],configuration=expected,**light.FLAGS))
    pending=[];reports=[];started=time.perf_counter()
    def group(entries):
        light.need(vars(args)==expected,'Source assignment/configuration changed')
        r=light.generate_batch(entries,args,runtime,out,resource_admission)
        reports.append(r)
    for item in selected:
        light.need(vars(args)==expected,'Source assignment/configuration changed')
        pending.append(light.load_entry(args.source_root,item['meta'],out,inputs))
        if len(pending)==3:
            group(pending);pending=[];inputs=light.Inputs()
    if pending:group(pending)
    report=dict(recipe=light.RECIPE,source_count=len(selected),new_model_calls=sum(r['new_model_calls'] for r in reports),
        chat_batch_sizes=[n for r in reports for n in r['chat_batch_sizes']],
        media_sha_reads=0,media_probes=0,history_review_reads=0,
        synchronous_semantic_checks=0,synchronous_rewrite_calls=0,
        wall_seconds=time.perf_counter()-started,**light.FLAGS)
    light.write_once(out/'summaries'/(light.key(report)+'.json'),report)
    return report
