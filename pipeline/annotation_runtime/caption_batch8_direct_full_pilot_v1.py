"""Fixed three-source native batch8 pilot; runtime/admission are caller-owned."""
import copy
from contextlib import ExitStack, contextmanager
import functools
from pathlib import Path
import time

import caption_cross_source_stage_v1 as cross
import caption_direct_facts_v1 as direct

RT = Path(__file__).resolve().parent
BAR = RT.parent
DEFAULT_OUT = BAR / 'caption_batch8_direct_full_pilot_v1'
ORIGINAL_ROOT = BAR / 'curated222939_collaborative_v3'
SOURCES = ('VGfurWgIdrVQNyhI', 'HVsstNGHcxHNsWkM', 'VrydLuJOvAnzhiQt')
SCHEMA = 'caption_batch8_direct_full_pilot_v1'
FLAGS = dict(direct.FLAGS, production_counts=0, production_claims=0)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pilot_args(model, model_reference=None):
    bar, _ = direct.native()
    args = bar.parser().parse_args(['--plan', str(ORIGINAL_ROOT / 'sources/train' / SOURCES[0] / 'bar_plan.json'),
        '--model', str(model), '--out', str(DEFAULT_OUT), '--stage', 'run',
        '--batch-size', '8', '--max-num-seqs', '8', '--boundary-review', 'off',
        '--hand-roi-root', str(ORIGINAL_ROOT / 'roi'), '--ffprobe', '/opt/ms_tools/ffprobe',
        '--ffmpeg', '/usr/local/bin/ffmpeg'])
    args.model_reference = str(model_reference) if model_reference else None
    return args


class Inputs:
    def __init__(self):
        self.evidence = direct.Evidence()
        self.stats = {}

    def guard(self, path):
        path = Path(path).resolve(strict=True)
        s = path.stat()
        self.stats.setdefault(path, (s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns))

    def checked(self, ref):
        return self.evidence.checked(ref)

    def load(self, path):
        return self.evidence.load(path)

    def refs(self, value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value and isinstance(value['path'], str):
                self.evidence.raw(value['path'], value['sha256'])
            else:
                for child in value.values():
                    self.refs(child)
        elif isinstance(value, list):
            for child in value:
                self.refs(child)

    def verify(self):
        self.evidence.verify_stable()
        for path, before in self.stats.items():
            s = path.stat()
            need((s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) == before,
                 'Original source input changed before commit: ' + str(path))


def fixed_sources(inputs, args, configuration):
    """Same producer terminal/bar_plan/record/raw bytes as original consumer."""
    bar, pipeline = direct.native()
    entries, provenance = [], []
    need(Path(args.hand_roi_root).resolve() == (ORIGINAL_ROOT / 'roi').resolve(),
         'Original three-source ROI root required')
    for shard, sid in enumerate(SOURCES):
        source = pipeline.source_dir(ORIGINAL_ROOT, dict(split='train', id=sid))
        producer_path = source / 'producer.terminal.json'
        producer = inputs.load(producer_path)
        terminal_path = ORIGINAL_ROOT / 'consumers' / ('shard-%04d' % shard) / 'sources/train' / sid / 'consumer.terminal.json'
        terminal = inputs.load(terminal_path)
        joint = inputs.checked(terminal['joint_candidate_ref'])
        plan_ref = producer['plan_ref']
        inputs.evidence.raw(plan_ref['path'], plan_ref['sha256'])
        raw_line = inputs.evidence.raw(plan_ref['path'])
        plan = bar.validate_row(inputs.load(plan_ref['path']))
        row = plan['manifest_record']
        need(producer['status'] == 'ready' and producer['source_binding'] == plan['binding'] and
             row['id'] == plan['id'] == joint['id'] == terminal['id'] == sid and
             row['split'] == joint['split'] == terminal['split'] == 'train' and
             row['batch_id'] == joint['batch_id'] == producer['batch_id'], 'Original source identity differs')
        inputs.checked(producer['roi_ref'])
        inputs.guard(Path(producer['roi_ref']['path']).with_name('source_manifest.json'))
        inputs.guard(Path(producer['roi_ref']['path']).with_name('extract_receipt.json'))
        actual = inputs.load(source / 'actual_media.json')
        # Historical st_dev is mount-local; retain its record, bind reuse by SHA.
        for ref in actual['source_identity']['files'].values():
            if ref:
                inputs.evidence.raw(ref['path'], ref['sha256'])
        need(actual['source_identity']['files']['video_path']['path'] == row['video_path'] and
             actual['source_identity']['files']['video_path']['sha256'] == plan['binding']['video_sha256'] and
             actual['source_identity']['files']['audio_path']['sha256'] == plan['binding']['audio_sha256'],
             'Original actual media differs')
        for ref in actual['source_identity']['files'].values():
            if ref:
                inputs.guard(ref['path'])
        roi = inputs.checked(producer['roi_ref'])
        for crop in roi.get('crops', []):
            inputs.guard(crop['path'])
        style = copy.deepcopy(joint['style_result'])
        need(style.get('id') == sid and style.get('status') in ('resolved', 'candidate') and
             style.get('dance_style') not in (None, '', 'unknown') and
             style.get('source') in ('historical_RGB_label', 'rich_chapter2', 'independent_RGB') and
             style.get('metadata', {}).get('video_sha256') == plan['binding']['video_sha256'],
             'Only original same-video resolved dance style may be reused')
        if joint.get('style_result_ref'):
            need(inputs.checked(joint['style_result_ref']) == style, 'Original style artifact differs')
        inputs.refs(style.get('metadata', {}).get('source_refs', {}))
        for key in ('generation.json', 'receipt.json'):
            if key in style.get('metadata', {}):
                inputs.checked(style['metadata'][key])
        binding = dict(source_binding=plan['binding'], plan_sha256=plan_ref['sha256'],
            full_source_row_sha256=bar.batch.digest(plan), raw_line_sha256=bar.sha(raw_line),
            manifest_record_sha256=bar.batch.digest(row), params_sha256=bar.batch.digest(configuration))
        entries.append(dict(row=plan, raw_line=raw_line, root=Path(args.out) / 'sources' / sid, binding=binding))
        provenance.append(dict(style=style, actual=actual, original_actual_media_ref=inputs.evidence.ref(source / 'actual_media.json'), original_joint_ref=terminal['joint_candidate_ref'],
            original_producer_ref=inputs.evidence.ref(producer_path), original_plan_ref=plan_ref))
    need(sum(len(e['row']['windows']) for e in entries) == 8, 'Fixed pilot requires all original eight windows')
    return entries, provenance


@contextmanager
def native_times(bar, times):
    hooks = ((bar, 'prepare_source', 'cpu_prepare_seconds'),
             (bar, 'verify_prepared', 'source_verify_seconds'),
             (bar, 'persist_completion', 'native_persist_seconds'),
             (bar.core, 'transaction', 'native_receipt_check_seconds'))
    originals = []
    try:
        for module, name, metric in hooks:
            original = getattr(module, name)
            originals.append((module, name, original))
            @functools.wraps(original)
            def measured(*a, _fn=original, _metric=metric, **kw):
                begin = time.perf_counter()
                try:
                    return _fn(*a, **kw)
                finally:
                    times[_metric] += time.perf_counter() - begin
            setattr(module, name, measured)
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


class ChatClock:
    def __init__(self, llm, times, calls):
        self.llm, self.times, self.calls = llm, times, calls

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def chat(self, messages, *a, **kw):
        self.calls.append(len(messages))
        begin = time.perf_counter()
        try:
            return self.llm.chat(messages, *a, **kw)
        finally:
            self.times['chatwall_seconds'] += time.perf_counter() - begin


def run_pilot(args, runtime, resource_admission=None):
    """Admission(args, runtime) must inspect caller's actual loaded runtime/resources.

    No runtime is constructed here. Admission must not populate args.out. Native
    runtime_loader may write its own fresh identity there after recipe entry.
    Timing phases are nested wall measurements, not additive CPU/GPU estimates.
    """
    begin = time.perf_counter()
    need(callable(resource_admission), 'Caller resource/runtime admission required; default refusal')
    out = Path(args.out).resolve()
    need(out == DEFAULT_OUT.resolve() and not Path(args.out).is_symlink(), 'Fixed independent pilot output required')
    need(not out.exists(), 'Pilot output already exists; no resume or overwrite')
    need(callable(runtime) and args.stage == 'run' and args.batch_size == 8 and
         args.max_num_seqs >= 8 and args.boundary_review == 'off' and not args.retry_failed,
         'Run batch8 with boundary off and no retry required')
    need(resource_admission(args, runtime) is True, 'Caller resource/runtime admission rejected')
    bar, pipeline = direct.native()
    times = {k: 0. for k in ('cpu_prepare_seconds', 'source_verify_seconds', 'native_persist_seconds',
        'native_receipt_check_seconds', 'chatwall_seconds')}
    calls, results, inputs, ref_cache = [], [], Inputs(), {}
    with direct.direct_recipe(bar, args):
        configuration = bar.params(args, bar.code_identity())
        start = time.perf_counter()
        entries, provenance = fixed_sources(inputs, args, configuration)
        inputs.verify()
        times['input_prepare_check_seconds'] = time.perf_counter() - start
        out.mkdir(parents=True, exist_ok=False)
        def measured_runtime():
            llm, sampling = runtime()
            return ChatClock(llm, times, calls), sampling
        with ExitStack() as locks, native_times(bar, times):
            for entry in entries:
                locks.enter_context(bar.batch.bundle_lock(entry['root']))
            start = time.perf_counter()
            staged = cross.stage_sources(entries, args, measured_runtime, ref_cache)
            times['stage_wall_seconds'] = time.perf_counter() - start
            start = time.perf_counter()
            for i, entry in enumerate(entries):
                visual = cross.resume_source(staged, i, args, measured_runtime, ref_cache)
                prepared = bar.load(entry['root'] / 'prepared.json')
                windows = [m['window'] for m in prepared['windows']]
                need(visual['facts'] == bar.facts_from_observations(windows, visual['window_observations']) and
                     visual['final'] == direct.direct_final(windows, visual['facts'], []) and
                     not visual['boundary_observations'] and visual['rewrite_fulltext_calls_per_attempt'] == 0,
                     'Native admitted facts/direct windows or zero-call contract differs')
                identity_ref = visual.get('model_identity_ref')
                model_sha = bar.load(identity_ref['path'])['sha256'] if identity_ref else None
                joint = pipeline.joint_candidate(entry['row']['manifest_record'], visual, provenance[i]['style'], model_sha)
                joint.update(status='joint_candidate' if joint['joint_complete'] else 'partial_candidate',
                    source_identity=prepared['source_identity'], actual_video_probe=prepared['probe'],
                    original_actual_media_ref=provenance[i]['original_actual_media_ref'],
                    historical_source_identity=provenance[i]['actual']['source_identity'],
                    original_style_reused_without_model_call=True, original_joint_ref=provenance[i]['original_joint_ref'],
                    caption_recipe=direct.RECIPE, **FLAGS)
                results.append(dict(id=entry['row']['id'], visual=visual, joint_candidate=joint))
            times['resume_wall_seconds'] = time.perf_counter() - start
            cross.check_inputs(staged, args)
        start = time.perf_counter()
        inputs.verify()
        need(bar.params(args, bar.code_identity()) == configuration, 'Native configuration changed before commit')
        times['end_check_seconds'] = time.perf_counter() - start
        start = time.perf_counter()
        for result in results:
            pipeline.save(out / 'candidates' / (result['id'] + '.json'), result)
        times['candidate_persist_seconds'] = time.perf_counter() - start
        inputs.verify()
        report = dict(schema=SCHEMA, source_ids=list(SOURCES), source_count=3,
            window_count=sum(len(r['visual']['window_observations']) for r in results),
            joint_complete_candidates=sum(r['joint_candidate']['joint_complete'] for r in results),
            partial_candidates=sum(not r['joint_candidate']['joint_complete'] for r in results),
            visual_chat_batch_sizes=calls, boundary_model_calls=0, rewrite_model_calls=0,
            new_style_model_calls=0, results=results, timing_seconds=times,
            timing_phases_nested_not_additive=True, speedup_claim=None,
            independent_review_required=True, original_records_and_assets_unmodified=True,
            configuration=configuration, execution_ref=direct.reference(__file__),
            final_timing_file='timing.json', **FLAGS)
        times['end_to_end_before_report_commit_seconds'] = time.perf_counter() - begin
        start = time.perf_counter()
        pipeline.save(out / 'report.json', report)
        times['report_commit_seconds'] = time.perf_counter() - start
        times['end_to_end_through_report_commit_seconds'] = time.perf_counter() - begin
        pipeline.save(out / 'timing.json', dict(timing_seconds=times, excludes_final_timing_sidecar_write=True, **FLAGS))
    return report
