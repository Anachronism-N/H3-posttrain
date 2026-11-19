"""Independent fixed-three CPU rewrite audit; prepare never loads a model."""
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
BAR = RT.parent
CANDIDATES = BAR / 'curated_rewrite_only_cpu_candidates_v1'
OUT = BAR / 'curated_cpu_rewrite_reviews_pilot3_v1'
IDS = ('wqGYgdMiTDqyyVuS', 'jqoZXbNoNchzsYOd', 'SDjBFHHLfNrWZyQw')
SCHEMA = 'curated_cpu_rewrite_independent_review_pilot3_v1'
BATCH = 'cpu-rewrite-independent-pilot3-v1'
TEACHER = BAR.parent / '32b_replica_stage_receipts/source_reference_v1.json'
OLD_SHA = 'f5507fe6b42f9d5bda22346d369bebf76c1455d14dea1234433ef16869cd23ef'
CPU_SHA = '0f8c684daa401c8a9a7deac3f1b6f86c166c45a8cd9a610a5ca8b1b10fe5290f'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


cpu = module('independent_cpu_producer', RT / 'curated_rewrite_only_cpu_batch_v1.py')
q, off, v5 = cpu.q, cpu.off, cpu.v5
original_bar = cpu.bar
need = v5.need
need(cpu._BASE_SHA(RT / 'curated_rewrite_only_cpu_batch_v1.py') == CPU_SHA, 'Frozen CPU producer changed')
need(cpu._BASE_SHA(RT / 'oldbar_explicit_review_v1.py') == OLD_SHA, 'Frozen f550 independent reviewer changed')


def library_modules():
    return {name: value for name, value in sys.modules.items()
            if name.startswith(('ms_', 'motionscript_v2'))}


# Original frozen replay and independent review have different ms_* versions.
# Keep both real implementations intact instead of deleting validators or AST.
_original_modules = library_modules()
_original_path = list(sys.path)
for _name in _original_modules:
    sys.modules.pop(_name, None)
old = module('independent_f550_review', RT / 'oldbar_explicit_review_v1.py')
_review_modules = library_modules()
_review_path = list(sys.path)
for _name in _review_modules:
    sys.modules.pop(_name, None)
sys.modules.update(_original_modules)
sys.path[:] = _original_path
audit, batch, core = old.audit, old.batch, old.core
NO_APPROVAL = {**old.NO_APPROVAL, 'eligible_for_training': False, 'original_curated_batch_completed': False,
               'original_curated_batches_completed': False, 'original_population_quality_approved': False,
               'full_batch_semantic_pass_claimed': False}


@contextmanager
def review_library():
    previous = library_modules()
    paths = list(sys.path)
    for name in previous:
        sys.modules.pop(name, None)
    sys.modules.update(_review_modules)
    sys.path[:] = _review_path
    try:
        yield
    finally:
        _review_modules.update(library_modules())
        for name in library_modules():
            sys.modules.pop(name, None)
        sys.modules.update(previous)
        sys.path[:] = paths


def check_output(out):
    out = Path(out).resolve()
    allowed = out == OUT or out.is_relative_to(Path('/tmp'))
    protected = (CANDIDATES, cpu.PRODUCTION, cpu.QUEUE, RT)
    need(allowed and all(out != p and not out.is_relative_to(p) and not p.is_relative_to(out)
                        for p in protected), 'Output must be isolated pilot3 default or /tmp, never production/input/runtime')
    return out


def candidate_path(sid):
    need(sid in IDS, 'Only the fixed three source IDs are allowed')
    return CANDIDATES / 'revisions' / cpu.REVISION / 'train' / sid / 'result.json'


def load_population():
    rows = [q.checked(q.identity(candidate_path(sid))) for sid in IDS]
    refs = [q.identity(candidate_path(sid)) for sid in IDS]
    population = dict(schema=SCHEMA, independent_batch_id=BATCH, population_kind='fixed-three-pilot',
                      population_source_count=3, sample_count=3, source_ids=list(IDS), candidate_refs=refs,
                      candidate_rows_sha256=[batch.digest(r) for r in rows],
                      original_curated_batch_mapping={r['id']: r['batch_id'] for r in rows},
                      original_population_not_adopted=True, **NO_APPROVAL)
    validate_population(rows, population)
    return rows, population


def validate_population(rows, population):
    need(population.get('schema') == SCHEMA and population.get('independent_batch_id') == BATCH
         and population.get('population_kind') == 'fixed-three-pilot'
         and population.get('population_source_count') == 3 and population.get('sample_count') == 3,
         'Three-source pilot cannot impersonate the original 500/448 population')
    need(len(rows) == 3 and [r['id'] for r in rows] == list(IDS)
         and all(r['split'] == 'train' and r['source_key'] == 'train/' + r['id'] for r in rows),
         'Exact fixed-three original source/split membership required')
    need(population['source_ids'] == list(IDS)
         and population['candidate_rows_sha256'] == [batch.digest(r) for r in rows]
         and population['original_curated_batch_mapping'] == {r['id']:r['batch_id'] for r in rows}
         and all(population.get(k) is False for k in NO_APPROVAL), 'Independent population binding/approval changed')
    for row, ref in zip(rows, population['candidate_refs']):
        need(Path(ref['path']).resolve() == candidate_path(row['id']) and q.checked(ref) == row,
             'Wrong candidate path/SHA/current contents')


def validate_claim(candidate, visual, original, windows, style):
    need(candidate['schema'] == cpu.SCHEMA and candidate['global_revision'] == cpu.REVISION
         and candidate['status'] == 'rewrite_only_cpu_joint_candidate'
         and candidate['source_dedup_key'] == candidate['source_key']
         and candidate['original_batch_id'] == candidate['batch_id'], 'Exact CPU candidate/revision required')
    for document in (candidate, visual):
        need(all(document.get(k) is False for k in q.FLAGS)
             and document.get('model_calls') == 0 and type(document['model_calls']) is int
             and document.get('extra_source_attempts') == 0 and type(document['extra_source_attempts']) is int
             and document.get('attempt_budget_consumed') is False
             and document.get('model_rewrite_performed') is False
             and document.get('VLM_dispatch_performed') is False
             and document.get('style_label_changed') is False
             and document.get('primary_worker_reservation_modified') is False
             and document.get('original_raw_or_terminal_modified') is False
             and document.get('original_population_quality_approved') is False,
             'Zero model/extra attempt/budget and no approval/production mutation required')
    need(candidate['joint_complete'] is True and candidate['all_windows_nonempty'] is True,
         'Nonempty complete CPU alternative required')
    need(visual['facts'] == original['facts'] and visual['window_observations'] == original['window_observations']
         and visual['original_windows'] == original['original_windows']
         and visual['boundary_observations'] == original['boundary_observations']
         and visual['original_final'] == original['final'] == candidate['original_final'],
         'Facts/raw_text/parts/refs/observations/windows and failed original must remain unchanged')
    # The frozen scope tests admitted raw_text/raw_source_clause only, never the
    # other rejected events in original_description. jqo is a genuine example.
    body = off.concatenate_facts(windows, original['facts'])
    need(visual['final']['json'] == body and visual['final']['status'] == 'candidate'
         and visual['final']['model_rewrite_performed'] is False
         and visual['final']['derivation'] == 'verbatim_admitted_fact_concatenation'
         and all(w['caption'].strip() for w in body['windows']), 'Each caption must concatenate exact supported raw_text/IDs')
    need(visual['final_timed_caption'] == original_bar.core.timed_text({'windows':windows}, body['windows'])
         == candidate['label']['timed_motion_script'], 'Verbatim timed motion claim changed')
    need(candidate['style_result'] == style and candidate['label']['dance_style'] == style['dance_style']
         and candidate['dance_style_label'] == style['dance_style']
         and candidate['target_text'] == 'Dance style: ' + style['dance_style'] + '\n' + visual['final_timed_caption'],
         'Original unchanged style/MS semantic claim required')


def validate_source(candidate, candidate_ref, *, ffprobe='/opt/ms_tools/ffprobe'):
    cpu.frozen()
    need(q.checked(candidate_ref) == candidate and Path(candidate_ref['path']).resolve() == candidate_path(candidate['id']),
         'Current candidate bytes/source differ')
    job_ref = candidate['original_job_ref']
    job = q.checked(job_ref)
    key = q.safe_key(job)
    need(key == 'train/' + candidate['id'] == candidate['source_key']
         and job['source_key'] == key and job['split'] == candidate['split']
         and job['batch_id'] == candidate['batch_id']
         and Path(job_ref['path']).resolve() == cpu.QUEUE / 'jobs' / (key + '.json')
         and Path(job['production']).resolve() == cpu.PRODUCTION
         and Path(job['queue_root']).resolve() == cpu.QUEUE
         and job['global_revision'] == 'repair-20261025-v1', 'Wrong original job/root/revision/source/split')
    original = q.checked(job['refs']['visual_result'])
    windows = cpu.scope(job, original)
    for name in ('plan_ref', 'style_result'):
        if job['refs'].get(name):
            cpu.register_shared(job['refs'][name])
    plan = q.checked(job['refs']['plan_ref'])
    cpu.register_shared(plan['source_refs']['manifest'])
    checked_job, source, style, derived, provenance, original_joint = off.candidate(job_ref, cpu.REVISION)
    need(checked_job == job, 'Original job changed during verification')
    root = Path(job['visual_root']).resolve()
    need(root.parent.name == 'sources' and root.name == job['id']
         and root.parent.parent.parent == cpu.PRODUCTION / 'visual'
         and cpu.re.fullmatch(r'shard-[0-9]{4}', root.parent.parent.name)
         and Path(job['refs']['plan_ref']['path']).resolve() == cpu.PRODUCTION / 'sources' / key / 'bar_plan.json'
         and Path(job['refs']['prepared']['path']).resolve() == root / 'prepared.json'
         and Path(job['refs']['prepared_receipt']['path']).resolve() == root / 'prepared.receipt.json'
         and job['refs']['raw_rgb_manifest'] == original['raw_rgb_manifest'], 'Exact original visual/plan/prepared lineage required')
    prepared = q.verify_prepared_content(root, plan, original_bar, ffprobe)
    need(prepared == q.checked(job['refs']['prepared']) and prepared['binding'] == original['binding'],
         'Original actual SHA/PTS/RGB/ROI prepared mismatch')
    q.verify_references_by_content(prepared)
    style_refs = cpu.check_style(job, style, plan, root, prepared)
    context = {**original, 'source_key':key, 'original_job_ref':job_ref, 'original_visual_result_ref':job['refs']['visual_result']}
    evidence = [cpu.replay_window(job, original, o, context, root, prepared) for o in original['window_observations']]
    failed = Path(original['final']['attempt_path']).resolve()
    need(failed.parent == root / 'rewrite' and cpu.re.fullmatch(r'attempt-[0-9]+', failed.name), 'Wrong failed provenance attempt')
    _, failed_refs = cpu.receipt_inventory(failed, 'failed')
    need(q.load(failed / 'final.json') == {k:v for k,v in original['final'].items() if k != 'attempt_path'},
         'Actual failed final provenance changed')
    destination = candidate_path(job['id']).parent
    need(candidate['visual_result_ref'] == q.identity(destination / 'visual_result.json')
         and candidate['original_evidence_manifest_ref'] == q.identity(destination / 'evidence.manifest.json')
         and candidate['original_visual_result_ref'] == job['refs']['visual_result']
         and candidate['original_result_ref'] == job['original_result_ref'], 'Wrong exact original/current refs')
    visual = q.checked(candidate['visual_result_ref'])
    validate_claim(candidate, visual, original, windows, style)
    refs = [copy.deepcopy(job_ref)] + list(copy.deepcopy(job['refs']).values())
    refs += [q.identity(root / name) for name in ('plan.json','plan.line.raw.json','binding.json')]
    refs += list(v5.immutable_references(prepared)) + style_refs + failed_refs
    union = q.checked(job['refs']['raw_rgb_manifest'])
    refs += [dict(path=str((root / 'rgb' / f['file']).resolve()),sha256=f['sha256']) for f in union['frames']]
    if prepared['hand_roi'].get('available'):
        roi_path = Path(prepared['hand_roi']['path']).resolve()
        refs += [q.identity(roi_path.with_name(name)) for name in ('source_manifest.json','extract_receipt.json')]
        refs += [dict(path=c['path'],sha256=c['sha256']) for c in q.load(roi_path)['crops']]
    refs += [dict(path=str(RT / name),sha256=sha) for name,sha in {**cpu.HELPERS, **v5.REPLAY_CODE_SHA}.items()]
    expected_audit = dict(schema=cpu.SCHEMA,source_key=key,id=job['id'],split=job['split'],batch_id=job['batch_id'],
                         original_job_ref=job_ref,original_source_refs=refs,original_window_evidence=evidence,
                         facts_raw_text_unchanged=True,facts_and_parts_full_original_replay=True,
                         actual_media_SHA_PTS_RGB_ROI_checked=True,admitted_claim_fine_or_uncertain_exclusion=True,
                         unadmitted_context_detail_does_not_suppress_admitted_coarse_facts=True,
                         model_calls=0,extra_source_attempts=0,attempt_budget_consumed=False,
                         primary_worker_reservation_modified=False,independent_review_required=True,**q.FLAGS)
    evidence_ref = candidate['original_evidence_manifest_ref']
    need(q.checked(evidence_ref) == expected_audit, 'Full original audit manifest/sourceRefs/window six-receipt replay changed')
    provenance.update(schema=cpu.SCHEMA,model_calls=0,alternative_to_primary_repair=True,source_dedup_key=key,
                      original_batch_id=job['batch_id'],original_population_quality_approved=False,
                      primary_worker_reservation_modified=False,original_evidence_manifest_ref=evidence_ref)
    derived.update(provenance)
    need(visual == derived, 'Full CPU visual candidate differs from original fact-only derivation')
    derived['result_ref'] = candidate['visual_result_ref']
    joint = off.pipeline.joint_candidate(source,derived,style,original_joint.get('annotator_model_sha256'))
    joint.update(provenance)
    for name in ('source_identity','actual_video_probe','style_result_ref'):
        if name in original_joint:
            joint[name] = copy.deepcopy(original_joint[name])
    joint['status'] = 'rewrite_only_cpu_joint_candidate'
    need(candidate == joint, 'Full current JOINT candidate/source identity/input media/label/provenance differs')
    for name, identity in candidate['source_identity']['files'].items():
        previous = prepared['source_identity']['files'][name]
        need(identity is None and previous is None or identity is not None and previous is not None
             and all(identity[k] == previous[k] for k in ('path', 'sha256')),
             'Source identity content/path must equal original visual bundle; cross-mount device stats are not content')
    for name in ('video_path','audio_path'):
        identity = candidate['source_identity']['files'][name]
        need(candidate[name] == identity['path'] == source[name], 'Current media/source path changed')
        q.verify_bytes(identity)
    need(candidate['actual_video_probe'] == prepared['probe'], 'Current candidate PTS evidence differs')
    need(Path(candidate['hand_roi']['path']).resolve() == cpu.PRODUCTION / 'roi' / job['id'] / 'roi_manifest.json',
         'Original native ROI source path differs')
    q.verify_job(job)
    for ref in refs:
        q.verify_bytes(ref)
    return dict(probe=prepared['probe'],source_hint=str(root),origin_root=str(root.parent.parent),
                original_fact_count=len(original['facts']),original_windows=len(windows),
                original_receipt_windows=len(evidence))


def review_row(candidate, candidate_ref, population_ref, verified):
    source_refs = dict(candidate=candidate_ref,original_job=candidate['original_job_ref'],
                       original_result=candidate['original_result_ref'],original_visual=candidate['original_visual_result_ref'],
                       original_evidence_manifest=candidate['original_evidence_manifest_ref'],population=population_ref)
    return dict(id=candidate['id'],source_id=candidate['id'],split=candidate['split'],
                video_path=candidate['video_path'],audio_path=candidate['audio_path'],
                label=copy.deepcopy(candidate['label']),label_status='completed',
                source_hint=verified['source_hint'],origin_root=verified['origin_root'],
                raw_rgb_manifest=candidate['raw_rgb_manifest'],hand_roi=candidate['hand_roi'],
                visual_result_ref=candidate['visual_result_ref'],sourceRefs=source_refs,
                annotator_model_sha256=candidate['annotator_model_sha256'],
                explicit_review_binding=dict(schema=SCHEMA,independent_batch_id=BATCH,population=population_ref,
                    candidate_sha256=batch.digest(candidate),label_sha256=batch.digest(candidate['label']),
                    target_text_sha256=old.text_sha(candidate['target_text']),
                    source_binding_sha256=batch.digest(candidate['source_binding']),
                    provenance_sha256=batch.digest(candidate['source_refs']),
                    original_curated_batch_id=candidate['batch_id'],original_failed_rewrite_is_provenance_only=True))


def verify_native_evidence(evidence, row, root, probe):
    old.verify_evidence(evidence, row, root, probe)
    need([f['index'] for f in evidence['frames']] == evidence['binding']['frame_indices'],
         'Evidence RGB indices differ from binding')
    need(len(evidence['frames']) == min(16, len(probe['times'])), 'All sixteen actual sampled RGB frames required')
    roi_manifest = q.load(row['hand_roi']['path'])
    need(all(crop in roi_manifest['crops'] for crop in evidence['hand_rois']),
         'Native ROI geometry/frame/source must equal original crop inventory')
    need(evidence['image_budget'] == dict(source_frames=len(evidence['frames']),
         native_hand_rois=len(evidence['hand_rois']), max_source_frames=16, max_ROIs=6), 'Exact native image budget required')
    need(evidence['frame_indices_are_original_global'] is True and evidence['source_RGB_saved_native_size'] is True,
         'Original global-PTS/native RGB evidence required')


class IndependentReviewer(old.ManifestReviewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        paths = [Path(v['path']) for v in self.code['files'].values()] + [Path(__file__).resolve()]
        self.code = batch.identity_tree(RT, paths)

    def identity(self):
        identity = super().identity()
        need(Path(self.annotator_model_reference).resolve() == TEACHER, 'Exact original teacher reference required')
        config = json.loads((Path(self.model) / 'config.json').read_bytes())
        need('8b' in config.get('_name_or_path', '').lower() or '8b' in Path(self.model).name.lower(),
             'Independent pilot reviewer must be the scheduled 8B checkpoint')
        return identity


def review_transaction(root, row, evidence, reviewer):
    transaction = Path(root) / 'review'
    dispatched = 0
    if (transaction / 'receipt.json').exists():
        need(audit.read_json(transaction / 'receipt.json').get('status') == 'candidate',
             'Failed/incomplete original transaction remains pending; never retry')
    else:
        need(not transaction.exists() or not any(transaction.iterdir()),
             'Partial request/raw/etc transaction remains pending; never retry')
        dispatched = 1
        reviewer._pilot_dispatches = getattr(reviewer, '_pilot_dispatches', 0) + 1
        result = reviewer(row, root, prepare_only=False)
        need(result.get('status') in ('pass','confirmed_error','uncertain'), 'Independent model transaction is not complete')
    checked, receipt = old.verify_review_receipt(root,row,evidence,reviewer)
    transaction_receipt = audit.read_json(transaction / 'receipt.json')
    need(transaction_receipt.get('semantic_approval') is False
         and transaction_receipt.get('human_review_completed') is False, 'Review receipt cannot claim semantic/human approval')
    return checked, receipt, dispatched


def execute(out=OUT, *, prepare_only=True, reviewer=None, ffprobe='/opt/ms_tools/ffprobe',ffmpeg='/usr/local/bin/ffmpeg'):
    out = check_output(out)
    need(prepare_only or reviewer is not None, 'Run requires externally admitted independent 8B reviewer')
    rows,population = load_population()
    out.mkdir(parents=True,exist_ok=True)
    lock = (out / 'independent.pilot3.lock').open('a+b')
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q.save_once(out / 'frozen.population.json',population)
        population_ref = q.identity(out / 'frozen.population.json')
        results = []
        for candidate,candidate_ref in zip(rows,population['candidate_refs']):
            base = dict(source_id=candidate['id'],split=candidate['split'],independent_batch_id=BATCH,
                        original_curated_batch_id=candidate['batch_id'],candidate_ref=candidate_ref,
                        candidate_sha256=batch.digest(candidate),label_sha256=batch.digest(candidate['label']),
                        AI_candidate_not_ground_truth=True,old_scores_reused=False,actual_VLM_calls=0,review_dispatches=0,**NO_APPROVAL)
            calls_before = getattr(reviewer, '_pilot_dispatches', 0)
            try:
                verified = validate_source(candidate,candidate_ref,ffprobe=ffprobe)
                row = review_row(candidate,candidate_ref,population_ref,verified)
                root = out / 'sources' / ('train-' + candidate['id']) / batch.digest(row)[:24]
                with review_library():
                    evidence = audit.prepare_evidence(row,root,ffprobe=ffprobe,ffmpeg=ffmpeg,hand_roi_root=row['hand_roi']['path'])
                    verify_native_evidence(evidence,row,root,verified['probe'])
                    base.update(evidence_ref=q.identity(root / 'evidence.json'),input_row_sha256=batch.digest(row),
                                real_source_hint=verified['source_hint'],real_hand_roi_ref=row['hand_roi'],
                                evidence_frames=len(evidence['frames']),native_hand_rois=len(evidence['hand_rois']),**verified)
                    if prepare_only:
                        result = dict(base,status='prepared',GPU_model_load_performed=False)
                        q.save_once(root / 'independent.prepare.receipt.json',result)
                    else:
                        need(Path(reviewer.annotator_model_reference).resolve() == TEACHER, 'Wrong original teacher reference')
                        reviewer.identity()
                        checked,receipt,calls = review_transaction(root,row,evidence,reviewer)
                        result = {**base, **checked, 'review_receipt':receipt, 'actual_VLM_calls':calls, 'review_dispatches':calls, **NO_APPROVAL}
                        # Resume dispatch counts are execution metadata, not immutable semantic receipt content.
                        terminal = {k:v for k,v in result.items() if k not in ('actual_VLM_calls', 'review_dispatches')}
                        q.save_once(root / 'independent.review.receipt.json',terminal)
                validate_source(candidate,candidate_ref,ffprobe=ffprobe)
                q.verify_bytes(candidate_ref)
                q.verify_bytes(candidate['original_evidence_manifest_ref'])
                results.append(result)
            except Exception as error:
                dispatches = getattr(reviewer, '_pilot_dispatches', 0) - calls_before
                results.append(dict(base,status='pending',error=type(error).__name__ + ': ' + str(error),
                                    review_dispatches=dispatches,actual_VLM_calls=None if dispatches else 0))
        reviewed = [r for r in results if r['status'] in ('pass','confirmed_error','uncertain')]
        summary = dict(schema=SCHEMA,population=population_ref,independent_batch_id=BATCH,
                       population_source_count=3,sample_count=3,selected_source_ids=list(IDS),
                       original_curated_batch_mapping=population['original_curated_batch_mapping'],
                       sampled_reviewed_denominator=len(reviewed),confirmed_error_count=sum(r['status']=='confirmed_error' for r in reviewed),
                       statuses=dict(Counter(r['status'] for r in results)),pending_source_ids=[r['source_id'] for r in results if r['status']=='pending'],
                       native_source_frames=sum(r.get('evidence_frames',0) for r in results),
                       native_hand_rois=sum(r.get('native_hand_rois',0) for r in results),
                       actual_VLM_calls=None if any(r['actual_VLM_calls'] is None for r in results)
                                       else sum(r['actual_VLM_calls'] for r in results),
                       review_dispatches=sum(r['review_dispatches'] for r in results),
                       independent_sample_review_completed=not prepare_only and len(reviewed)==3,
                       state='prepared_pending_independent_review' if prepare_only and all(r['status']=='prepared' for r in results)
                             else 'sample_review_completed_pending_human' if len(reviewed)==3 else 'pending',
                       GPU_model_load_performed=bool(not prepare_only and getattr(reviewer,'_loaded',None) is not None),
                       model_execution_scheduled_externally=not prepare_only,original_attempt_budget_consumed=False,
                       original_worker_reservation_modified=False,statistical_population_error_estimate=False,
                       confirmed_errors_are_AI_candidates_not_ground_truth=True,**NO_APPROVAL)
        batch.atomic_write(out / 'independent.ledger.jsonl',b''.join(q.encoded(r) for r in results))
        batch.atomic_write(out / 'independent.summary.json',batch.json_bytes(summary))
        return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    stage = p.add_mutually_exclusive_group(required=True)
    stage.add_argument('--prepare-only',action='store_true')
    stage.add_argument('--run',action='store_true')
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--model')
    p.add_argument('--model-reference')
    p.add_argument('--annotator-model-reference',type=Path,default=TEACHER)
    p.add_argument('--ffprobe',default='/opt/ms_tools/ffprobe')
    p.add_argument('--ffmpeg',default='/usr/local/bin/ffmpeg')
    args = p.parse_args(argv)
    reviewer = None
    if args.run:
        if not (args.model and args.model_reference):
            p.error('--run requires --model and --model-reference')
        need(args.annotator_model_reference.resolve() == TEACHER, 'Exact original teacher reference required')
        with review_library():
            reviewer = IndependentReviewer(args.model,args.out,model_reference=args.model_reference,
                       annotator_model_reference=args.annotator_model_reference,ffprobe=args.ffprobe,ffmpeg=args.ffmpeg)
    report = execute(args.out,prepare_only=args.prepare_only,reviewer=reviewer,ffprobe=args.ffprobe,ffmpeg=args.ffmpeg)
    print(json.dumps(report,ensure_ascii=False))
    return 0 if not report['pending_source_ids'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
