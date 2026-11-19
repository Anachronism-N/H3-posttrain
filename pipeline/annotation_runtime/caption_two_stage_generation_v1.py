"""Finite stage-one drafts over an exclusive original producer assignment."""
import copy
from contextlib import contextmanager
import inspect
from pathlib import Path
import time

import caption_batch8_production_consumer_v2 as ownership
import caption_two_stage_pipeline_v1 as drafts

RECIPE=drafts.RECIPE
FLAGS=drafts.FLAGS


def run_consumer(args,runtime,resource_admission=None):
    drafts.need(callable(resource_admission) and resource_admission(args,runtime) is True,'Actual caller-owned resources required')
    drafts.need(args.stage=='run' and args.batch_size==8 and args.max_num_seqs>=8 and not args.retry_failed,'Native stage-one batch8 required')
    out=Path(args.out).resolve()
    drafts.need(out.parent==drafts.BAR and out.name.startswith('caption_two_stage_'),'Independent draft output required')
    bar,pipeline=drafts.native.native()
    inputs=ownership.Inputs()
    handoff=ownership.verify_handoff(args,inputs)
    selected=handoff['admission']['selected']
    drafts.need(selected,'Nonempty exclusive remaining draft assignment required')
    started=time.perf_counter()
    protocol=dict(recipe=RECIPE,stage='draft_generation',handoff_ref=args.handoff_ref,
        selected=selected,configuration=vars(args),**FLAGS)
    ref_cache=dict(production_manifest_summary=inputs.evidence.checked(handoff['admission']['manifest_summary_ref']),
        production_batch_rows={},production_source_plan=inputs.evidence.load(args.source_plan))
    source=inspect.getsource(drafts.generate_batch)
    source=source.replace("need(not (out/'batch.claim.json').exists()", "need(not (out/'batches'/BATCH_KEY/'claim.json').exists()")
    source=source.replace("pipeline.save(out/'batch.claim.json'", "pipeline.save(out/'batches'/BATCH_KEY/'claim.json'")
    results=[];pending=[];styles={}
    with bar.batch.bundle_lock(out):
        if (out/'protocol.json').exists():
            drafts.need(pipeline.load(out/'protocol.json')==protocol,'Cannot resume another draft assignment/recipe')
        pipeline.save(out/'protocol.json',protocol)
        queue=out/'revision_queue.jsonl'
        completed=set()
        if queue.exists():
            for line in queue.read_bytes().splitlines():
                item=bar.core.strict_json(line)
                drafts.need(item['schema']==RECIPE and item['training_eligible'] is False,'Unexpected revision queue recipe')
                completed.add(item['id'])
        configuration=bar.params(args,bar.code_identity())
        configuration.update(caption_recipe=RECIPE,support_and_semantic_filter_deferred=True)
        def group(rows):
            batch_key=bar.batch.digest([dict(id=e['row']['id'],binding=e['binding']) for e in rows])
            namespace=dict(drafts.__dict__,BATCH_KEY=batch_key)
            # The lane already owns its outer lock; opening another lock fd would deadlock.
            private=source.replace('with bar.batch.bundle_lock(out), ExitStack() as locks:', 'with ExitStack() as locks:')
            exec(compile(private,str(Path(__file__).resolve()),'exec'),namespace)
            result=namespace['generate_batch'](rows,args,runtime,out,resource_admission,styles,ref_cache)
            results.append(result)
            print(pipeline.raw_json(dict(event='draft_group_completed',sources=len(rows),window_calls=result['window_call_count'],
                revision_queue=str(queue),**FLAGS)).decode(),flush=True)
        for item in selected:
            sid=item['meta']['id']
            if sid in completed:continue
            original=ownership.load_source(item,args,configuration,ref_cache)
            drafts.need(original['entry'] is not None,'Failed producer deferred; no invented draft')
            entry=original['entry']
            entry['root']=out/'sources'/sid
            styles[sid]=original['style']
            pending.append(entry)
            if len(pending)==3:
                group(pending);pending=[]
        if pending:group(pending)
        inputs.verify()
        report=dict(recipe=RECIPE,source_count=len(selected),new_source_count=sum(x['source_count'] for x in results),
            new_window_calls=sum(x['window_call_count'] for x in results),groups=results,
            synchronous_rewrite_calls=0,synchronous_review_calls=0,synchronous_style_calls=0,
            stage2_revision_queue=str(queue),wall_seconds=time.perf_counter()-started,**FLAGS)
        pipeline.save(out/'summaries'/(pipeline.digest(report)+'.json'),report)
        if not (out/'generation_report.json').exists():
            pipeline.save(out/'generation_report.json',report)
        return report
