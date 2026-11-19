"""Two disjoint TP1 reviewer lanes; exact-version receipts, CPU idle/prepare safe."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import curated_collab3_review_adapter_v1 as aggregation

RT = Path(__file__).resolve().parent
FLAGS = dict(semantic_approval=False, human_review_completed=False, formal_approval=False,
             training_eligible=False, source_quality_pass=False, human_approval_automatically_granted=False)


def native_review(source):
    sys.path.insert(0, str(Path(source).resolve()))
    spec = importlib.util.spec_from_file_location('collab3_8b_native_review', Path(source)/'ms_batch_review.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_reference_cache(native):
    cache={}
    def references(row):
        refs=[]
        def visit(value):
            if isinstance(value,dict):
                if 'path' in value and 'sha256' in value:
                    key=(value['path'],value['sha256'])
                    if key not in cache:
                        identity=native.batch.file_identity(Path(value['path']))
                        aggregation.need(identity['sha256']==value['sha256'],'native immutable reference SHA mismatch')
                        cache[key]=identity
                    identity=cache[key]
                    aggregation.need(native.batch.snapshot(Path(value['path']))==identity['stat'],
                                     'native immutable reference changed after first SHA verification')
                    refs.append(copy.deepcopy(identity))
                for child in value.values():visit(child)
            elif isinstance(value,list):
                for child in value:visit(child)
        visit(row.get('sourceRefs',row.get('source_refs',{})))
        for key in ('raw_rgb_manifest','hand_roi','visual_result_ref','style_result_ref','source_identity'):
            visit(row.get(key))
        return refs
    native.source_references=references
    return cache


def assigned(bid, gpu):
    aggregation.need(gpu in (6, 7) and bid.startswith('train-b') and bid[7:].isdigit(), 'exact N7 GPU/batch required')
    index = int(bid[7:])
    aggregation.need(64 <= index < 124, 'batch outside original collab3 reservation')
    return True


class Watcher:
    def __init__(self, adapter, input_root, out, gpu, reviewer=None, native=None, state=None, initial_source_review_limit=0, continuation_authority=None):
        aggregation.need(gpu in (6, 7), 'only physical N7 GPU6/7 reviewer lanes')
        self.adapter, self.review = adapter, adapter.review
        self.input_root, self.out = Path(input_root).resolve(), Path(out).resolve()
        self.gpu, self.reviewer, self.native = gpu, reviewer, native
        aggregation.need(type(initial_source_review_limit) is int and initial_source_review_limit>=0,'nonnegative reviewer-only initial source budget')
        self.initial_source_review_limit=initial_source_review_limit
        self.continuation_authority=Path(continuation_authority) if continuation_authority else None
        self.continuation_ref=None
        self.state = state if state is not None else {}
        aggregation.need(self.state.get('gpu', gpu) == gpu and self.state.get('plan_ref', adapter.plan_ref) == adapter.plan_ref,
                         'review lane/state plan mismatch')
        self.state.update(gpu=gpu, plan_ref=adapter.plan_ref)
        for key in ('batches', 'receipts', 'pending_sources', 'initial_review_sources'):
            self.state.setdefault(key, {})
        self.input_stat = None
        self.session_model_ref = None

    def continuation(self):
        path=self.continuation_authority
        if path is None or not path.exists():return False
        value=aggregation.strict(path.read_bytes())
        aggregation.need(value.get('schema')=='curated_collab3_review_continuation_v1' and
                         value.get('node')=='N7' and value.get('gpu')==self.gpu and
                         value.get('plan_ref')==self.adapter.plan_ref and value.get('authorized') is True and
                         value.get('fixed_full500_random25_risk5') is True,
                         'explicit same-plan/lane full-policy continuation authorization required')
        self.continuation_ref=aggregation.reference(path)
        return True

    def read_inputs(self):
        path = self.input_root/'progress.json'
        if not path.exists():
            return []
        stat = path.stat()
        token = (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        retry=self.continuation() and any(v.get('pending_review_budget') for v in self.state['batches'].values())
        if token == self.input_stat and not retry:
            return []
        progress = aggregation.strict(path.read_bytes())
        aggregation.need(progress['plan_ref'] == self.adapter.plan_ref and progress['full_source_denominator'] == 222939
                         and progress['full_batch_denominator'] == 448, 'current aggregate input/denominator mismatch')
        changed = []
        for bid, batch in progress['batches'].items():
            if not assigned(bid, self.gpu):
                continue
            ref = batch['input_ref']
            expected = self.input_root/'batches'/bid/'review-required.json'
            aggregation.need(Path(ref['path']).resolve() == expected, 'explicit physical review input path mismatch')
            old=self.state['batches'].get(bid,{})
            if old.get('input_ref') != ref or (retry and old.get('pending_review_budget')):
                changed.append((bid, ref))
        self.input_stat = token
        return changed

    def cached_result(self, version, row):
        entry = self.state['receipts'].get(version)
        if not entry:
            return None
        record = self.adapter.checked(entry['result_ref'])
        aggregation.need(record['exact_version_review_key'] == version and record['source_key'] == row['source_key']
                         and record['label_file_ref'] == {'path':row['label_path'],'sha256':row['label_sha256']},
                         'cached receipt source/current revision mismatch')
        # Reuse immutable closures; no repeated source-media decode/hash/model call.
        native_receipt = record['result'].get('review_receipt')
        if native_receipt:
            path = Path(native_receipt['path'])
            aggregation.need(aggregation.sha(path) == native_receipt['sha256'], 'native review receipt changed')
        return record['result']

    def recover_native_receipt(self, source, root, exact_binding):
        native_root=root/'native';transaction=native_root/'review'
        if not (transaction/'receipt.json').exists():return None
        native=self.native
        aggregation.need(native is not None,'existing native receipt cannot be redispatched')
        evidence=native.read_json(native_root/'evidence.json')
        evidence_receipt=native.read_json(native_root/'evidence.receipt.json')
        aggregation.need(evidence_receipt['sha256']==aggregation.sha(native_root/'evidence.json') and
                         evidence['binding']['source_id']==source['id'] and
                         evidence['binding']['label_file_ref']['path']==source['label_path'] and
                         evidence['binding']['label_file_ref']['sha256']==source['label_sha256'] and
                         evidence['binding']['label_sha256']==exact_binding['label_payload_sha256'] and
                         evidence['binding']['input_row_sha256']==native.batch.digest(source),
                         'orphan native receipt evidence/source/label binding mismatch')
        for frame in evidence['frames']:
            aggregation.need(aggregation.sha(native_root/frame['file'])==frame['sha256'],'orphan native RGB SHA changed')
        for crop in evidence.get('hand_rois',[]):
            aggregation.need(aggregation.sha(crop['path'])==crop['sha256'],'orphan native ROI SHA changed')
        receipt=native.read_json(transaction/'receipt.json');binding=receipt['binding']
        aggregation.need(binding['model_sha256']==self.reviewer.identity()['sha256'] and
                         binding['code_sha256']==self.reviewer.code['sha256'] and
                         binding['label_file_ref']['sha256']==source['label_sha256'] and
                         binding['label_sha256']==exact_binding['label_payload_sha256'] and
                         binding['evidence_sha256']==aggregation.sha(native_root/'evidence.json'),
                         'orphan native receipt model/code/exact evidence mismatch')
        request=native.review_request(source,evidence,native_root,self.reviewer.model)
        # Existing receipt transaction validates generation/request/raw/final SHA; llm=None forbids a new dispatch.
        result=native.core.transaction(transaction,request,None,None,lambda v:native.validate_review(v,evidence),binding)
        if result['status']=='candidate':
            return {**result['json'],'transaction_status':'candidate','source_id':source['id'],
                    'review_receipt':aggregation.reference(transaction/'receipt.json'),
                    'label_sha256':exact_binding['label_payload_sha256'],'evidence':evidence,'evidence_root':str(native_root)}
        return dict(source_id=source['id'],status=result['status'],error=result.get('error'),
                    label_sha256=exact_binding['label_payload_sha256'],evidence=evidence,evidence_root=str(native_root))

    def audit(self, input_ref, row, binding, run):
        aggregation.need(row['source_ordinal']%2==self.gpu-6,'cross-lane source ownership claim rejected')
        version = binding['exact_version_review_key']
        result = self.cached_result(version,row)
        if result is not None:
            return result
        root = self.out/'versions'/version
        resultpath = root/'result.json'
        if resultpath.exists():
            # Crash after atomic result but before lane state: adopt, never redispatch.
            ref = aggregation.reference(resultpath)
            record = self.adapter.checked(ref)
            aggregation.need(record['exact_version_review_key'] == version and record['source_key'] == row['source_key']
                             and record['label_file_ref'] == binding['label_file_ref'], 'orphan exact-version result mismatch')
            self.state['receipts'][version] = {'result_ref':ref}
            return self.cached_result(version,row)
        if not run:
            return dict(source_id=row['id'],status='pending_gpu_review',model_calls=0)
        native_receipt=root/'native/review/receipt.json'
        if self.initial_source_review_limit and len(self.state['initial_review_sources'])>=self.initial_source_review_limit and not native_receipt.exists() and not self.continuation():
            return dict(source_id=row['id'],status='pending_initial_three_case_review_budget',model_calls=0)
        aggregation.need(self.reviewer is not None and self.native is not None, 'explicit independent native reviewer required')
        checked = self.adapter.submission_binding(input_ref,row['id'])
        aggregation.need(checked['binding']['exact_version_review_key'] == version, 'current label changed before submission')
        source = copy.deepcopy(checked['row'])
        aggregation.need(source.get('annotator_model_sha256'), 'annotator checkpoint identity missing')
        # Explicit model-reference identity is validated by manager and native runtime.
        result = self.recover_native_receipt(source,root,binding)
        if result is None:
            self.state['initial_review_sources'][version]=dict(source_key=source['source_key'],label_file_ref=binding['label_file_ref'])
            aggregation.atomic(self.out/'state.json',self.state)
            result = self.reviewer(source, root/'native', prepare_only=False)
        aggregation.need(result.get('source_id') == row['id'], 'review returned different source')
        aggregation.need(result.get('label_sha256') == binding['label_payload_sha256'], 'review returned different semantic label SHA')
        # A label revision arriving during the call invalidates applicability, not its preserved AI evidence.
        current = True
        try:
            self.adapter.submission_binding(input_ref,row['id'])
        except (ValueError,FileNotFoundError):
            current = False
        result['applicable_to_current_label']=current
        record = dict(schema='curated_collab3_exact_8b_review_v1',source_key=row['source_key'],source_id=row['id'],
                      original_batch_id=row['batch_id'],exact_version_review_key=version,label_file_ref=binding['label_file_ref'],
                      label_revision=binding['label_revision'],label_payload_sha256=binding['label_payload_sha256'],
                      explicit_review_input_ref=input_ref,source_binding=binding,current_version_at_commit=current,
                      result=result, AI_errors_are_candidates_not_human_truth=True, **FLAGS)
        aggregation.atomic(resultpath,record)
        self.state['receipts'][version]={'result_ref':aggregation.reference(resultpath)}
        return result

    def publish(self, bid, manifest, results):
        rows = {r['id']:r for r in manifest['records']}
        full_sampled = manifest['sampling']['selected_source_ids']
        sampled = [sid for sid in full_sampled if rows[sid]['source_ordinal'] % 2 == self.gpu-6]
        reviewed = [sid for sid in sampled if results.get(sid,{}).get('status') in ('pass','confirmed_error','uncertain')
                    and results[sid].get('applicable_to_current_label',True)]
        confirmed = [sid for sid in reviewed if results[sid]['status']=='confirmed_error']
        if self.native is not None:
            plan = dict(batch_id=bid,records=manifest['records'])
            web = self.native.review_queue(plan,results,representative_limit=3)
        else:
            web = dict(batch_id=bid,items=[],optional_uncertain_items=[],all_error_source_ids=confirmed)
        for item in web.get('items',[])+web.get('optional_uncertain_items',[]):
            sid=item['source_id'];binding=manifest['exact_label_bindings'][sid]
            item.update(source_key=rows[sid]['source_key'],label_file_ref=binding['label_file_ref'],
                        label_revision=binding['label_revision'],exact_version_review_key=binding['exact_version_review_key'],
                        original_member_sha256=binding['original_member_sha256'],physical_result_root=binding['physical_result_root'],
                        original_ROI_ref=binding['roi_ref'], **FLAGS)
        web.update(AI_errors_are_human_review_candidates=True,all_candidate_receipts_preserved=True,**FLAGS)
        root=self.out/'batches'/bid
        aggregation.atomic(root/'review_web.json',web)
        summary=dict(schema='curated_collab3_8b_batch_review_v1',batch_id=bid,source_count=500,sampling_denominator=500,
                     full_source_denominator=222939,full_batch_denominator=448,sampling=manifest['sampling'],
                     sampled_reviewed_denominator=len(reviewed),AI_error_candidate_count=len(confirmed),
                     selected_source_ids=full_sampled,lane_selected_source_ids=sampled,individually_reviewed_source_ids=reviewed,
                     unsampled_source_status='not_individually_reviewed',
                     pending_selected_source_ids=[sid for sid in sampled if sid not in reviewed],
                     all_source_labels_completed=all(self.review.label_complete(r) for r in manifest['records']),
                     no_full500_generation_wait=True, fixed_random25_risk5=True,
                     lane_review_complete=len(reviewed)==len(sampled), full_batch_review_complete=False,
                     both_lane_receipts_required_for_full_sample_review=True, **FLAGS)
        aggregation.atomic(root/'summary.json',summary)
        return summary

    def poll(self, run=False):
        self.adapter.documents.clear()  # Each input snapshot is validated once; receipts are separate immutable files.
        dirty=self.read_inputs()
        for bid,ref in dirty:
            manifest=self.adapter.checked(ref,self.input_root/'batches'/bid/'review-required.json')
            aggregation.need(manifest['source_count']==500 and manifest['sampling_denominator']==500 and
                             manifest['sampling']==self.review.sampling_plan(manifest['records'],bid,sample_count=25,risk_count=5),
                             'exact original500 random25+risk5 required')
            rows={r['id']:r for r in manifest['records']}
            results={}
            for sid in manifest['sampling']['selected_source_ids']:
                row=rows[sid]
                if row['source_ordinal'] % 2 != self.gpu-6:
                    continue
                if not self.review.label_reviewable(row):
                    results[sid]=dict(source_id=sid,status='label_incomplete')
                    continue
                binding=manifest['exact_label_bindings'][sid]
                results[sid]=self.audit(ref,row,binding,run)
            summary=self.publish(bid,manifest,results)
            self.state['batches'][bid]=dict(input_ref=ref,summary=summary,
                pending_review_budget=any(v['status']=='pending_initial_three_case_review_budget' for v in results.values()))
        progress=dict(schema='curated_collab3_8b_lane_progress_v1',gpu=self.gpu,expected_physical_node='N7',
                      physical_node='N7' if run else None,
                      assigned_batch_denominator=60,assigned_source_denominator=15000,full_source_denominator=222939,
                      full_batch_denominator=448,current_batch_count=len(self.state['batches']),
                      individually_reviewed=sum(v['summary']['sampled_reviewed_denominator'] for v in self.state['batches'].values()),
                      AI_error_candidate_count=sum(v['summary']['AI_error_candidate_count'] for v in self.state['batches'].values()),
                      exact_version_receipt_count=len(self.state['receipts']),idle_waiting_for_selected_source_arrival=not dirty,
                      independent_8B_resident_runtime=bool(run and getattr(self.reviewer,'_loaded',None) is not None),
                      GPU_review_requested=run, model_calls=0 if not run else None,
                      native_generation_receipts_are_model_call_truth=True, signals_sent=0, resource_locks_acquired=0,
                      initial_source_review_limit=self.initial_source_review_limit,initial_review_source_count=len(self.state['initial_review_sources']),
                      continuation_authority_ref=self.continuation_ref,
                      no_producer_attempt_budget_consumed=True, **FLAGS)
        aggregation.atomic(self.out/'state.json',self.state)
        aggregation.atomic(self.out/'progress.json',progress)
        return progress


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,choices=(6,7),required=True)
    parser.add_argument('--plan',default=str(RT/'curated_collaborative3_source_plan_v1.json'))
    parser.add_argument('--plan-sha256',default=aggregation.PLAN_SHA)
    parser.add_argument('--input-root',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--model-reference',required=True)
    parser.add_argument('--annotator-model-reference',required=True)
    parser.add_argument('--ms-source',default=str(RT/'source_curated_review_v2'))
    parser.add_argument('--poll-seconds',type=float,default=30)
    parser.add_argument('--initial-source-review-limit',type=int,default=0)
    parser.add_argument('--continuation-authority')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args(argv)
    aggregation.need(args.poll_seconds>0,'positive polling interval')
    out=Path(args.out).resolve();inputroot=Path(args.input_root).resolve()
    aggregation.need(out != inputroot and not out.is_relative_to(inputroot) and not inputroot.is_relative_to(out),
                     'review output must not overlap CPU input tree')
    review=aggregation.load_sampling(args.ms_source)
    adapter=aggregation.Adapter(args.plan,args.plan_sha256,review)
    native=None;reviewer=None
    if args.run:
        aggregation.need(os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.gpu),'exact single physical GPU CVD required')
        import curated_collab3_8b_review_manager_v1 as manager
        manager.node()
        native=native_review(args.ms_source)
        install_reference_cache(native)
        reviewer=native.LocalReviewer(args.model,out,model_reference=args.model_reference,
            annotator_model_reference=args.annotator_model_reference,hand_roi_root=str(adapter.root/'roi'),
            ffprobe='/opt/ms_tools/ffprobe',ffmpeg='/usr/local/bin/ffmpeg')
        reviewer.identity()  # Different actual weight content, not a renamed 32B deployment.
    statepath=out/'state.json'
    state=aggregation.strict(statepath.read_bytes()) if statepath.exists() else None
    if args.run:
        session=dict(model_reference=aggregation.reference(args.model_reference),
                     annotator_model_reference=aggregation.reference(args.annotator_model_reference),
                     checkpoint_identity=reviewer.identity()['sha256'])
        state=state if state is not None else {}
        aggregation.need(state.get('session_model_ref',session)==session,'existing receipt tree belongs to different reviewer/teacher checkpoint')
        state['session_model_ref']=session
    watcher=Watcher(adapter,inputroot,out,args.gpu,reviewer,native,state,
                    initial_source_review_limit=args.initial_source_review_limit,continuation_authority=args.continuation_authority)
    while True:
        value=watcher.poll(run=args.run)
        print(json.dumps(value,sort_keys=True),flush=True)
        if not args.watch:
            return value
        time.sleep(args.poll_seconds)


if __name__=='__main__':
    main()
