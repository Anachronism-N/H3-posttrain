"""Explicitly assigned ready sources, caller-owned runtime; no scheduling/resources."""
import copy
from contextlib import contextmanager, ExitStack
from pathlib import Path
import subprocess
import threading

import caption_cross_source_stage_v1 as cross
import caption_direct_facts_v1 as direct
import curated_fast_source_v1 as fast
import curated_transaction_cache_v1 as transactions

RECIPE = 'caption_batch8_direct_consumer_v1'
FLAGS = dict(direct.FLAGS, production_counts=0, production_claims=0)
ERRORS = (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError)


def need(value, message):
    if not value:
        raise ValueError(message)


class Runtime:
    """Caller must admit resources before scope; one thread/recipe for all batches."""
    def __init__(self, native_runtime, emit=lambda event: None):
        self.native_runtime, self.emit, self.active = native_runtime, emit, False

    def __call__(self):
        self.check(self.args)
        return self.native_runtime()

    def check(self, args):
        need(self.active and self.args is args and self.thread == threading.get_ident(), 'Caller runtime.scope required in one thread')
        need(vars(args) == self.inputs, 'Lane inputs/configuration changed')
        self.evidence.verify_stable()

    @contextmanager
    def scope(self, args):
        bar, _ = direct.native()
        need(not self.active and not getattr(bar, '_direct_lane_active', False), 'Concurrent/nested direct lane forbidden')
        need(args.stage == 'run' and args.boundary_review == 'off' and not args.retry_failed and args.batch_size == 8,
             'Explicit run batch8 boundary off/no retry required')
        evidence = direct.Evidence()
        assignment = evidence.checked(args.assignment_ref)
        need(assignment['schema'] == 'caption_direct_lane_assignment_v1' and assignment['exclusive_source_owner'] is True and
             assignment['out'] == str(Path(args.out).resolve()) and assignment['producer_root'] == str(Path(args.producer_root).resolve()) and
             assignment['selected_sources'] == args.selected_sources and bool(args.selected_sources), 'Explicit exclusive selected assignment required')
        need(assignment['mode'] in ('new_unassigned', 'handoff') and
             (assignment['mode'] != 'handoff' or assignment.get('old_consumer_stopped') is True) and
             (assignment['mode'] != 'new_unassigned' or not assignment.get('old_terminals')), 'Unassigned scope or stopped-consumer transition required')
        out, producer = Path(args.out).resolve(), Path(args.producer_root).resolve()
        need(out != producer and not out.is_relative_to(producer) and not producer.is_relative_to(out), 'New output must not overlap original ready root')
        self.args, self.inputs, self.evidence, self.assignment = args, copy.deepcopy(vars(args)), evidence, assignment
        self.source_evidence = {}
        self.thread, self.active, bar._direct_lane_active = threading.get_ident(), True, True
        self.execution = direct.reference(__file__)
        try:
            with transactions.transaction_cache(bar, self.emit), direct.direct_recipe(bar, args):
                self.configuration = {**bar.params(args, bar.code_identity()), 'consumer_recipe': RECIPE, 'consumer_execution_ref': self.execution}
                Path(args.out).mkdir(parents=True, exist_ok=True)
                yield self
        finally:
            self.active, bar._direct_lane_active = False, False


def iter_ready(args, runtime, terminal_seen):
    """No queue scan or source claim: explicit SID->ordinal assignment only."""
    runtime.check(args)
    bar, pipeline = direct.native()
    for sid, ordinal in args.selected_sources.items():
        need(isinstance(sid, str) and '/' not in sid and sid not in ('', '.', '..') and type(ordinal) is int, 'Invalid selected identity')
        terminal = Path(args.out) / 'sources/train' / sid / 'consumer.terminal.json'
        if sid in terminal_seen:
            previous = runtime.evidence.load(terminal)
            need(previous['consumer_recipe'] == RECIPE and previous['id'] == sid, 'Seen source must have this lane terminal')
            continue
        old = runtime.assignment.get('old_terminals', {}).get(sid)
        if old:
            previous = runtime.evidence.checked(old)
            need(runtime.assignment['mode'] == 'handoff' and previous['id'] == sid, 'Old terminal requires exact stopped-lane transition')
            if previous.get('joint_complete') is True:
                terminal_seen.add(sid)
                continue
        source = pipeline.source_dir(args.producer_root, dict(split='train', id=sid))
        evidence = runtime.source_evidence.setdefault(sid, direct.Evidence())
        producer = evidence.load(source / 'producer.terminal.json')
        need(producer['status'] == 'ready' and producer['id'] == sid, 'Selected producer is not ready')
        raw = evidence.raw(producer['plan_ref']['path'], producer['plan_ref']['sha256'])
        row = bar.validate_row(evidence.load(producer['plan_ref']['path']))
        record = row['manifest_record']
        need(record['id'] == sid and record['source_ordinal'] == ordinal and record['split'] == 'train' and
             producer['source_binding'] == row['binding'], 'Original selected source membership differs')
        evidence.raw(producer['roi_ref']['path'], producer['roi_ref']['sha256'])
        actual = evidence.load(source / 'actual_media.json')
        binding = dict(source_binding=row['binding'], plan_sha256=producer['plan_ref']['sha256'],
            full_source_row_sha256=bar.batch.digest(row), raw_line_sha256=bar.sha(raw),
            manifest_record_sha256=bar.batch.digest(record), params_sha256=bar.batch.digest(runtime.configuration))
        yield dict(row=row, raw_line=raw, binding=binding, root=Path(args.out) / 'visual/sources' / sid,
                   actual=actual, source_root=source)


def prepare_ready(entry, args, runtime, refcache):
    """Optional caller prefetch: CPU only in this same thread, no model requests."""
    runtime.check(args)
    bar, _ = direct.native()
    sid, root = entry['row']['id'], Path(entry['root'])
    need(args.selected_sources.get(sid) == entry['row']['manifest_record']['source_ordinal'] and
         root.resolve() == (Path(args.out) / 'visual/sources' / sid).resolve() and
         entry['binding']['params_sha256'] == bar.batch.digest(runtime.configuration), 'Prefetch outside assigned new recipe')
    with bar.batch.bundle_lock(root), fast.fast_source(bar, root):
        return (bar.verify_prepared(root, entry['row'], entry['raw_line'], entry['binding']) if (root / 'prepared.json').exists()
                else bar.prepare_source(entry['row'], entry['raw_line'], root, args, entry['binding'], refcache))


def boundary(entry, prepared, assets, staged, args, runtime):
    runtime.check(args)
    cross.check_inputs(staged, args)
    runtime.source_evidence[entry['row']['id']].verify_stable()
    assets.verify_boundary()
    bar, _ = direct.native()
    bar.batch.verify_source(prepared['source_identity'])
    bar.core.verify_references(prepared['source_refs'])


def run_ready_batch(entries, args, runtime, refcache, style_lookup):
    """At most three complete sources/eight windows; caller groups/prefetches next batch."""
    runtime.check(args)
    entries = tuple(entries)
    bar, pipeline = direct.native()
    need(0 < len(entries) <= 3 and sum(len(e['row']['windows']) for e in entries) <= 8, 'Up to three sources/eight whole original windows required')
    roots = [Path(e['root']).resolve() for e in entries]
    ids = [e['row']['id'] for e in entries]
    need(len(set(roots)) == len(roots) and len(set(ids)) == len(ids), 'Duplicate source/root collision')
    for entry, root in zip(entries, roots):
        sid, record = entry['row']['id'], entry['row']['manifest_record']
        need(args.selected_sources.get(sid) == record['source_ordinal'] and root == (Path(args.out) / 'visual/sources' / sid).resolve(), 'Source outside explicit lane selection/output')
        need(entry['binding']['params_sha256'] == bar.batch.digest(runtime.configuration), 'Cannot hot-resume another generation recipe')
        need(not (Path(args.out) / 'sources/train' / sid / 'consumer.terminal.json').exists(), 'Source terminal already committed; never redispatch')
    staged, items, prepared_rows, assets_rows = cross.StagedSources(entries, copy.deepcopy(vars(args))), [], {}, {}
    with ExitStack() as locks:
        for root in roots:
            locks.enter_context(bar.batch.bundle_lock(root))
        for i, entry in enumerate(staged.inputs):
            local = []
            try:
                with fast.fast_source(bar, entry['root']) as assets:
                    root = Path(entry['root'])
                    prepared = (bar.verify_prepared(root, entry['row'], entry['raw_line'], entry['binding']) if (root / 'prepared.json').exists()
                                else bar.prepare_source(entry['row'], entry['raw_line'], root, args, entry['binding'], refcache))
                    prepared_rows[i], assets_rows[i] = prepared, assets
                    forbidden = {p: set(v) for p, v in prepared['forbidden'].items()}
                    for manifest in prepared['windows']:
                        wroot = root / 'windows' / manifest['window']['window_id']
                        bar.core.save_json(wroot / 'manifest.json', {**manifest, 'binding': entry['binding'], 'union_manifest_sha256': prepared['union_manifest_sha256'], 'raw2d_mask': prepared['raw2d_mask'], **bar.FLAGS})
                        if len(manifest['frames']) >= 2:
                            local.append(dict(root=bar.attempt_root(wroot, False), request=bar.visual_request(args, root / 'rgb', manifest, forbidden),
                                validator=lambda value, m=manifest, f=forbidden: bar.filter_supported_observation(value, m, f),
                                binding={**entry['binding'], 'window': manifest['window'], 'frames_sha256': bar.batch.digest(manifest['frames'])}))
                items.extend(local)
            except ERRORS as error:
                staged.errors[i] = error
        runtime.check(args)
        cross.check_inputs(staged, args)
        staged.receipts = bar.transactions(items, runtime, args)
        results = []
        for i, entry in enumerate(entries):
            sid, root = entry['row']['id'], Path(entry['root'])
            try:
                if staged.errors[i] is not None:
                    raise staged.errors[i]
                with fast.fast_source(bar, root) as assets:
                    prepared = prepared_rows[i]
                    old_save = bar.core.save_json
                    def guarded_save(path, value):
                        if Path(path).parent == root / 'results':
                            boundary(entry, prepared, assets, staged, args, runtime)
                        return old_save(path, value)
                    process = bar.process_source
                    bar.process_source = transactions.clone(process, load=bar.load, verify_prepared=bar.verify_prepared,
                        prepare_source=bar.prepare_source, transactions=bar.transactions)
                    bar.core.save_json = guarded_save
                    try:
                        visual = cross.resume_source(staged, i, args, runtime, refcache)
                    finally:
                        bar.core.save_json, bar.process_source = old_save, process
                    style = copy.deepcopy(style_lookup(entry)) if style_lookup else None
                    if style is not None and style.get('status') in ('resolved', 'candidate'):
                        need(style.get('source') in ('historical_RGB_label', 'rich_chapter2', 'independent_RGB') and
                             style.get('metadata', {}).get('video_sha256') == entry['row']['binding']['video_sha256'], 'Reused style has no original same-video evidence')
                    if style is None:
                        style = pipeline.observe_dance_style(entry['row']['manifest_record'], entry['row'], visual, root, args, runtime, {})
                    need(style.get('id') == sid and style.get('source') not in ('music_genre', 'audio_genre'), 'Style is not same-source dance evidence')
                    windows = [m['window'] for m in prepared['windows']]
                    need(visual['final'] == direct.direct_final(windows, visual['facts'], []) and not visual['boundary_observations'] and visual['rewrite_fulltext_calls_per_attempt'] == 0,
                         'Native direct zero-boundary/rewrite contract differs')
                    model_ref = visual.get('model_identity_ref')
                    model_sha = bar.load(model_ref['path'])['sha256'] if model_ref else None
                    joint = pipeline.joint_candidate(entry['row']['manifest_record'], visual, style, model_sha)
                    joint.update(consumer_recipe=RECIPE, consumer_execution_ref=runtime.execution, source_identity=prepared['source_identity'],
                        actual_video_probe=prepared['probe'], batch_review_status='pending_independent_review', **FLAGS)
                    boundary(entry, prepared, assets, staged, args, runtime)
                    directory = pipeline.source_dir(args.out, entry['row']['manifest_record'])
                    style_path = pipeline.save(directory / 'style-results' / (pipeline.digest(style) + '.json'), style)
                    joint['style_result_ref'] = pipeline.ref(style_path)
                    joint_path = pipeline.save(directory / 'joint-results' / (pipeline.digest(joint) + '.json'), joint)
                    value = dict(id=sid, split='train', batch_id=entry['row']['manifest_record']['batch_id'],
                        status='joint_candidate' if joint['joint_complete'] else 'partial_candidate', joint_complete=joint['joint_complete'],
                        joint_candidate_ref=pipeline.ref(joint_path), consumer_recipe=RECIPE, params_sha256=entry['binding']['params_sha256'], **FLAGS)
                    boundary(entry, prepared, assets, staged, args, runtime)
                    pipeline.save(directory / 'consumer.terminal.json', value)
                    pipeline.append(Path(args.out) / 'completed.jsonl', value)
                    results.append(value)
            except ERRORS as error:
                value = dict(id=sid, status='source_error', error=type(error).__name__ + ': ' + str(error), consumer_recipe=RECIPE, **FLAGS)
                pipeline.save(root / 'failures' / (pipeline.digest(value) + '.json'), value)
                results.append(value)
        return results
