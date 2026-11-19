"""Finite, caller-owned native batch8 consumer; only its owned lane may resume."""
import argparse
import copy
from contextlib import ExitStack, contextmanager
import inspect
from pathlib import Path
import subprocess

import caption_cross_source_stage_v1 as cross
import caption_direct_facts_v1 as direct

RT = Path(__file__).resolve().parent
BAR = RT.parent
SCHEMA = 'caption_batch8_production_consumer_v2'
SOURCE_PLAN = RT / 'curated_collaborative3_source_plan_v1.json'
FLAGS = dict(direct.FLAGS)
LOCAL_ERRORS = (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError)


def need(condition, message):
    if not condition:
        raise ValueError(message)


class Inputs:
    def __init__(self):
        self.evidence = direct.Evidence()
        self.stats = {}

    def guard(self, path):
        path = Path(path).resolve(strict=True)
        self.stats.setdefault(path, self.fingerprint(path))

    @staticmethod
    def fingerprint(path):
        s = path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    def refs(self, value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and 'sha256' in value:
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
            need(self.fingerprint(path) == before, 'Source input changed before commit: ' + str(path))


def ready_lines(path, bar):
    """One finite read; only invalid/incomplete offsets get two immediate rereads."""
    if not path.exists():
        return
    with path.open('rb') as stream:
        end = path.stat().st_size
        while stream.tell() < end:
            offset = stream.tell()
            for attempt in range(3):
                line = stream.readline()
                try:
                    need(bool(line) and line.endswith(b'\n'), 'Incomplete ready line')
                    value = bar.core.strict_json(line)
                except ValueError:
                    if attempt == 2:
                        raise ValueError('Invalid ready line at %s@%d' % (path, offset))
                    stream.seek(offset)
                    continue
                break
            yield value


def assignment(args, inputs):
    bar, pipeline = direct.native()
    need(isinstance(getattr(args, 'handoff_ref', None), dict), 'Mandatory native handoff_ref required')
    handoff = inputs.evidence.checked(args.handoff_ref)
    need(handoff.get('schema') == 'caption_batch8_native_handoff_v1' and
         handoff.get('exclusive_source_owner') is True and handoff.get('old_resume_requires_exclusion') is True,
         'Exclusive native handoff/exclusion contract required')
    plan = inputs.evidence.load(args.source_plan)
    plan_ref = inputs.evidence.ref(args.source_plan)
    need(handoff['source_plan_ref'] == plan_ref and handoff['manifest_summary_ref'] == plan['manifest_summary'] and
         handoff['member'] == args.member and Path(handoff['new_out']).resolve() == Path(args.out).resolve() and
         Path(handoff['source_root']).resolve() == Path(args.source_root).resolve() == Path(plan['out']).resolve(),
         'Handoff original plan/member/newout/source root differs')
    need(Path(args.manifest).resolve() == Path(plan['manifest_summary']['path']).resolve(), 'Manifest summary differs from plan')
    summary = inputs.evidence.checked(plan['manifest_summary'])
    need(args.member in plan['members'], 'Explicit original consumer member required')
    member = plan['members'][args.member]
    need(member['mode'] == 'consumer' and (args.shard, args.nshard) == (member['shard'], member['nshard']),
         'Exact original consumer member/shard differs')
    original_selected = inputs.evidence.checked(plan['selected_sources'][args.member])
    selected = inputs.evidence.checked(handoff['exclusive_selected_sources_ref'])
    pool = {r['id']: r for r in original_selected}
    start, end = plan['source_range']['start'], plan['source_range']['end']
    need(start <= args.start_ordinal < args.end_ordinal <= end and 0 < len(selected) <= args.limit <= 500,
         'Finite exclusive selection/range budget differs')
    need(len({r['id'] for r in selected}) == len(selected) and len({r['source_ordinal'] for r in selected}) == len(selected) and
         all(bar.portable(r['id']) and r['split'] in ('train', 'val', 'test') and type(r['source_ordinal']) is int and
             args.start_ordinal <= r['source_ordinal'] < args.end_ordinal and r == pool.get(r['id']) and
             r['source_ordinal'] % args.nshard == args.shard for r in selected),
         'Exclusive selected source is outside exact original member/split/ordinal pool')
    old_cfg = inputs.evidence.checked(handoff['old_launch_ref'])
    old_out = Path(args.source_root).resolve() / 'consumers' / ('shard-%04d' % args.shard)
    queue = Path(old_cfg['queue']).resolve()
    need(old_cfg['mode'] == 'consumer' and old_cfg['collab3_member'] == args.member and
         old_cfg['collab3_source_plan'] == plan_ref and old_cfg['selected_sources'] == plan['selected_sources'][args.member] and
         old_cfg['node'] == member['node'] and old_cfg['gpus'] == member['gpus'] and Path(old_cfg['out']).resolve() == old_out and
         old_cfg['source_selection'] == dict(shard=args.shard, nshard=args.nshard, limit=0),
         'Native old launch exact lane/assignment differs')
    for key, name in (('old_admission_ref', 'admission.json'), ('old_resources_safe_ref', 'resources_safe.json')):
        need(Path(handoff[key]['path']).resolve() == queue / name, 'Handoff native queue artifact path differs')
    admitted = inputs.evidence.checked(handoff['old_admission_ref'])
    safe = inputs.evidence.checked(handoff['old_resources_safe_ref'])
    need(admitted['phase'] == old_cfg['phase'] == safe['phase'] and admitted['node'] == old_cfg['node'] and
         admitted['gpus'] == old_cfg['gpus'] and Path(admitted['out']).resolve() == old_out and
         admitted['config_sha256'] == handoff['old_launch_ref']['sha256'] and
         all(safe.get(k) is True for k in ('all_holders_safe', 'locks_released', 'leases_restored',
                                          'managed_parent_wait', 'unknown_jobs_not_signaled')),
         'Native old admission/resources_safe is not exact safely cleaned lane')
    need([(h['gpu'], h['uuid']) for h in admitted['holders']] ==
         [(h['gpu'], h['uuid']) for h in old_cfg['holders']], 'Native old admission GPU UUID differs')
    cleanup = inputs.evidence.checked(handoff['old_cleanup_ref'])
    live = inputs.evidence.load(queue / 'live.json')
    identity_keys = ('pid', 'uid', 'argv', 'start_ticks', 'cgroup', 'CVD', 'exe')
    worker = cleanup.get('old_worker_identity', {})
    need(live['phase'] == old_cfg['phase'] and isinstance(live.get('child_identity'), dict) and
         all(k in worker and worker[k] == live['child_identity'].get(k) for k in identity_keys),
         'Cleanup old worker identity differs from native live receipt')
    manager = cleanup.get('old_manager_identity', {})
    manager_safe = cleanup.get('old_manager_absent') is True or (cleanup.get('old_manager_quiescent') is True and
        all(k in manager and manager[k] == admitted['manager_identity'].get(k) == live['manager_identity'].get(k) for k in identity_keys))
    need(cleanup.get('schema') == 'caption_native_handoff_cleanup_v1' and cleanup.get('cleanup_done') is True and
         cleanup.get('old_worker_tree_absent') is True and manager_safe and
         cleanup.get('phase') == old_cfg['phase'] and cleanup.get('node') == old_cfg['node'] and
         cleanup.get('gpus') == old_cfg['gpus'] and cleanup.get('admission_ref') == handoff['old_admission_ref'] and
         cleanup.get('resources_safe_ref') == handoff['old_resources_safe_ref'] and
         type(cleanup.get('checked_unix_ns')) is int and cleanup['checked_unix_ns'] > 0,
         'Exact actual old-worker cleanup proof required')
    parameters = inputs.evidence.checked(old_cfg['consumer_params_evidence'])
    need(parameters['new_params_sha256'] == old_cfg['consumer_params_sha256'] == handoff['old_native_params_sha256'] and
         pipeline.digest(parameters['new_configuration']) == old_cfg['consumer_params_sha256'],
         'Native old parameter evidence differs')
    prefix = inputs.evidence.raw(handoff['old_completed_prefix_ref']['path'], handoff['old_completed_prefix_ref']['sha256'])
    need(Path(handoff['old_completed_path']).resolve() == old_out / 'completed.jsonl' and
         type(handoff['old_completed_bytes']) is int and handoff['old_completed_bytes'] == len(prefix) and
         (not prefix or prefix.endswith(b'\n')), 'Exact old completed prefix binding differs')
    actual_prefix = inputs.evidence.raw(handoff['old_completed_path'])
    need(actual_prefix == prefix, 'Stopped old completed tree differs from frozen prefix')
    need(Path(handoff['old_exclusion_ref']['path']).is_absolute(), 'Explicit old-resume exclusion reference required')
    exclusion = inputs.evidence.checked(handoff['old_exclusion_ref'])
    need(exclusion.get('schema') == 'caption_native_handoff_exclusion_v1' and exclusion.get('member') == args.member and
         exclusion.get('source_plan_ref') == plan_ref and exclusion.get('new_out') == handoff['new_out'] and
         exclusion.get('exclusive_selected_sources_ref') == handoff['exclusive_selected_sources_ref'] and
         exclusion.get('original_config_resume_forbidden') is True, 'Old resume requires explicit owned-source exclusion')
    producers = {m['shard']: name for name, m in plan['members'].items() if m['mode'] == 'producer'}
    need(set(producers) == set(range(4)), 'Original four producer stream assignment required')
    pools = {shard: {r['id']: r for r in inputs.evidence.checked(plan['selected_sources'][name])}
             for shard, name in producers.items()}
    return plan, summary, selected, pools


def freeze_admission(args, assignment_inputs):
    bar, pipeline = direct.native()
    plan, summary, selected, pools = assignment(args, assignment_inputs)
    wanted = {r['id']: r for r in selected}
    ready = {}
    source_root = Path(args.source_root)
    for shard in range(4):
        for hint in ready_lines(source_root / 'ready' / ('producer-%04d.jsonl' % shard), bar):
            sid = hint.get('id')
            if sid not in wanted:
                continue
            meta = wanted[sid]
            need(meta == pools[shard].get(sid) and
                 all(hint.get(k) == meta[k] for k in ('id', 'split', 'batch_id', 'source_ordinal')),
                 'Ready hint exact original member/split/ordinal differs')
            item = dict(meta=meta, hint=hint, ready_stream=shard)
            if sid in ready:
                need(ready[sid] == item, 'Conflicting duplicate ready source')
            ready[sid] = item
    candidates, skipped, skipped_refs = [], [], []
    old_out = source_root / 'consumers' / ('shard-%04d' % args.shard)
    for meta in selected:
        sid = meta['id']
        old_terminal = pipeline.source_dir(old_out, meta) / 'consumer.terminal.json'
        if old_terminal.exists():
            old = assignment_inputs.evidence.load(old_terminal)
            need(all(old.get(k) == meta[k] for k in ('id', 'split', 'batch_id')), 'Old terminal source differs')
            skipped.append(sid)
            skipped_refs.append(assignment_inputs.evidence.ref(old_terminal))
            continue
        need(sid in ready, 'Explicit exclusive selected source is not ready')
        path = pipeline.source_dir(source_root, meta) / 'producer.terminal.json'
        terminal = assignment_inputs.evidence.load(path)
        need(terminal == ready[sid]['hint'] and terminal['status'] in ('ready', 'failed'), 'Ready hint differs from producer terminal')
        candidates.append(dict(meta=meta, producer_terminal_ref=assignment_inputs.evidence.ref(path), ready_stream=ready[sid]['ready_stream']))
    return dict(schema=SCHEMA, selected=candidates, remaining_ids=[s['meta']['id'] for s in candidates],
        skipped_old_terminal_ids=skipped, skipped_old_terminal_refs=skipped_refs, handoff_ref=args.handoff_ref,
        original_assignment_ref=assignment_inputs.evidence.ref(args.source_plan), manifest_summary_ref=plan['manifest_summary'],
        single_frozen_ready_snapshot=True, selection=dict(source_root=str(source_root.resolve()), member=args.member,
            shard=args.shard, nshard=args.nshard, start_ordinal=args.start_ordinal, end_ordinal=args.end_ordinal,
            limit=args.limit, handoff_ref=args.handoff_ref), **FLAGS)


def verify_handoff(args, inputs):
    admission = freeze_admission(args, inputs)
    inputs.verify()
    return dict(selected_meta=[entry['meta'] for entry in admission['selected']],
                skipped_old_terminal_ids=admission['skipped_old_terminal_ids'], admission=admission)


def execution_identity(args, inputs):
    bar, _ = direct.native()
    style = Path(bar.__file__).with_name('ms_dance_style.py')
    refs = {name: inputs.evidence.ref(path) for name, path in
            (('consumer', __file__), ('stage', cross.__file__), ('direct', direct.__file__), ('style', style),
             ('source_cache', RT / 'curated_fast_source_v1.py'), ('request_images', RT / 'curated_request_image_cache_v1.py'),
             ('transaction_cache', RT / 'curated_transaction_cache_v1.py'))}
    style_reuse = list(args.style_reuse or [])
    refs['style_reuse'] = [inputs.evidence.ref(p) for p in style_reuse]
    if args.model_reference:
        refs['model_reference'] = inputs.evidence.ref(args.model_reference)
    code = bar.code_identity()
    for path in code:
        inputs.guard(path)
    return dict(schema=SCHEMA, caption_recipe=direct.RECIPE, execution_refs=refs,
                native_configuration=bar.params(args, code),
                style_max_tokens=args.style_max_tokens, out=str(Path(args.out).resolve()),
                source_plan=inputs.evidence.ref(args.source_plan), manifest=inputs.evidence.ref(args.manifest),
                selection=dict(source_root=str(Path(args.source_root).resolve()), member=args.member,
                    shard=args.shard, nshard=args.nshard, start_ordinal=args.start_ordinal,
                    end_ordinal=args.end_ordinal, limit=args.limit, handoff_ref=args.handoff_ref), **FLAGS)


@contextmanager
def owned_recipe(bar, args, protocol, fresh=False):
    """Retain the direct hook verbatim except its single fresh gate, after owned admission."""
    _, pipeline = direct.native()
    def verify_owned(out, current_args):
        if fresh:
            need({p.name for p in out.iterdir()} <= {'.lock'}, 'Fresh lane has unowned artifacts')
        else:
            need(pipeline.load(out / 'protocol.json') == protocol and protocol['schema'] == SCHEMA and
                 protocol['caption_recipe'] == direct.RECIPE, 'Owned direct recipe protocol differs')
    source = inspect.getsource(direct.direct_recipe)
    old = "        raise ValueError('Direct recipe requires a fresh isolated output; never resume an old generation recipe')"
    need(source.count(old) == 1, 'Direct fresh-gate anchor changed')
    source = source.replace(old, '        VERIFY_OWNED(out, args)')
    namespace = dict(direct.__dict__, VERIFY_OWNED=verify_owned)
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), namespace)
    with namespace['direct_recipe'](bar, args):
        yield


@contextmanager
def group_input_cache(bar, entries):
    """One immutable SourceAssets cache per locked source across stage and all resumes."""
    from curated_fast_source_v1 import SourceAssets, images
    originals = {name: getattr(bar, name) for name in ('load', 'prepare_source', 'verify_prepared', 'rgb_content')}
    old_sha = bar.batch.sha256
    assets = {Path(e['root']).resolve(): SourceAssets(e['root']) for e in entries}
    for cache in assets.values():
        cache.active = True
        cache.original_load = originals['load']
        cache.parse_bytes = bar.strict_json
    image_caches = {}
    for root, cache in assets.items():
        image = images.SourceImageCache(bar, root / 'rgb')
        image._data_url = cache.data_url
        image_caches[root / 'rgb'] = image
    def rgb_content(root, frames, header, hand_evidence=None):
        return image_caches[Path(root).resolve()]._rgb_content(root, frames, header, hand_evidence)
    def owner(path):
        key = Path(path).absolute()
        return next((a for a in assets.values() if key in a.allowed), None)
    def load(path):
        a = owner(path)
        return a.load(path, originals['load']) if a else originals['load'](path)
    def sha(path):
        a = owner(path)
        return a.capture(path)[1] if a else old_sha(path)
    def prepare(*a, **kw):
        value = originals['prepare_source'](*a, **kw)
        assets[Path(a[2]).resolve()].bind_prepared(value)
        return value
    def verify(*a, **kw):
        cache = assets[Path(a[0]).resolve()]
        cache.bind_prepared(load(Path(a[0]) / 'prepared.json'))
        return originals['verify_prepared'](*a, **kw)
    replacements = dict(load=load, prepare_source=prepare, verify_prepared=verify, rgb_content=rgb_content)
    # direct_recipe compiles process_source into its own frozen globals.
    namespace = bar.process_source.__globals__
    saved_namespace = {k: namespace[k] for k in replacements}
    for k, v in replacements.items():
        setattr(bar, k, v)
        namespace[k] = v
    bar.batch.sha256 = sha
    try:
        yield assets
    finally:
        for k, v in originals.items():
            setattr(bar, k, v)
        namespace.update(saved_namespace)
        bar.batch.sha256 = old_sha
        for cache in assets.values():
            cache.active = False


def load_source(item, args, configuration, ref_cache):
    bar, pipeline = direct.native()
    inputs = Inputs()
    producer = inputs.evidence.checked(item['producer_terminal_ref'])
    meta = item['meta']
    need(all(producer.get(k) == meta[k] for k in ('id', 'split', 'batch_id', 'source_ordinal')), 'Producer admission identity differs')
    if producer['status'] == 'failed':
        need(all(producer['manifest_record'].get(k) == meta[k] for k in meta), 'Failed producer manifest identity differs')
        return dict(item=item, inputs=inputs, producer=producer, entry=None)
    source = pipeline.source_dir(Path(args.source_root), meta)
    plan_ref = producer['plan_ref']
    need(Path(plan_ref['path']).resolve() == (source / 'bar_plan.json').resolve(), 'Producer plan path differs')
    raw_line = inputs.evidence.raw(plan_ref['path'], plan_ref['sha256'])
    plan = bar.validate_row(inputs.evidence.checked(plan_ref))
    row = plan['manifest_record']
    need(plan['id'] == meta['id'] and all(row.get(k) == meta[k] for k in meta) and
         producer['source_binding'] == plan['binding'], 'Original plan/split/ordinal binding differs')
    manifest_ref = plan['source_refs']['manifest']
    summary = ref_cache['production_manifest_summary']
    batch = next(b for b in summary['batches'] if b['batch_id'] == row['batch_id'])
    need(manifest_ref == {k: batch[k] for k in ('path', 'sha256')} and batch['split'] == row['split'], 'Plan not bound to original manifest batch')
    original = ref_cache['production_batch_rows'].get(batch['batch_id'])
    if original is None:
        raw = direct.Evidence().raw(batch['path'], batch['sha256'])
        original = {r['id']: r for r in (bar.core.strict_json(line) for line in raw.splitlines() if line.strip())}
        ref_cache['production_batch_rows'][batch['batch_id']] = original
    need(original.get(row['id']) == row, 'Exact original manifest record differs')
    inputs.guard(batch['path'])
    roi = inputs.evidence.checked(producer['roi_ref'])
    need(Path(producer['roi_ref']['path']).resolve() == (Path(args.hand_roi_root) / row['id'] / 'roi_manifest.json').resolve(),
         'Producer ROI path differs')
    for name in ('source_manifest.json', 'extract_receipt.json'):
        inputs.guard(Path(producer['roi_ref']['path']).with_name(name))
    for crop in roi.get('crops', []):
        inputs.guard(crop['path'])
    actual = inputs.evidence.load(source / 'actual_media.json')
    for ref in actual['source_identity']['files'].values():
        if ref:
            inputs.guard(ref['path'])
    need(actual['source_identity']['files']['video_path']['sha256'] == plan['binding']['video_sha256'] and
         actual['source_identity']['files']['video_path']['path'] == row['video_path'], 'Actual original video binding differs')
    if row.get('rich_path') and Path(row['rich_path']).is_file():
        inputs.guard(row['rich_path'])
    style = None
    # Target only the original assigned shard, never consume its completion as our receipt.
    old_members = ref_cache['production_source_plan']['members']
    member = next(m for m in old_members.values() if m['mode'] == 'consumer' and row['source_ordinal'] % m['nshard'] == m['shard'])
    old_terminal = Path(args.source_root) / 'consumers' / ('shard-%04d' % member['shard']) / 'sources' / row['split'] / row['id'] / 'consumer.terminal.json'
    if old_terminal.exists():
        old = inputs.evidence.load(old_terminal)
        if old.get('joint_candidate_ref'):
            joint = inputs.evidence.checked(old['joint_candidate_ref'])
            candidate = joint.get('style_result', {})
            if (candidate.get('id') == row['id'] and candidate.get('status') in ('resolved', 'candidate') and
                candidate.get('dance_style') not in (None, '', 'unknown') and candidate.get('source') in
                ('historical_RGB_label', 'rich_chapter2', 'independent_RGB') and
                candidate.get('metadata', {}).get('video_sha256') == plan['binding']['video_sha256']):
                if joint.get('style_result_ref'):
                    need(inputs.evidence.checked(joint['style_result_ref']) == candidate, 'Original style result differs')
                inputs.refs(candidate.get('metadata', {}).get('source_refs', {}))
                for key in ('generation.json', 'receipt.json'):
                    if key in candidate.get('metadata', {}):
                        inputs.refs(candidate['metadata'][key])
                style = copy.deepcopy(candidate)
    binding = dict(source_binding=plan['binding'], plan_sha256=plan_ref['sha256'],
        full_source_row_sha256=bar.batch.digest(plan), raw_line_sha256=bar.sha(raw_line),
        manifest_record_sha256=bar.batch.digest(row), params_sha256=bar.batch.digest(configuration))
    entry = dict(row=plan, raw_line=raw_line, root=Path(args.out) / 'visual/sources' / row['id'], binding=binding)
    return dict(item=item, inputs=inputs, producer=producer, entry=entry, style=style, actual=actual,
                original_actual_media_ref=inputs.evidence.ref(source / 'actual_media.json'))


def check_visual(visual, root, bar):
    prepared = bar.load(root / 'prepared.json')
    windows = [m['window'] for m in prepared['windows']]
    need([o['window'] for o in visual['window_observations']] == windows and
         [w['window_id'] for w in visual['final']['json']['windows']] == [w['window_id'] for w in windows] and
         not visual['boundary_observations'] and visual['rewrite_fulltext_calls_per_attempt'] == 0 and
         visual.get('caption_recipe') == direct.RECIPE, 'Native window/zero-call contract differs')
    return prepared


@contextmanager
def verified_transactions(bar):
    """Same-run native receipts reuse verified memory; real restart has an empty cache."""
    from curated_transaction_cache_v1 import ARTIFACTS
    original, verified, source_stats = bar.transactions, {}, {}
    def transactions(items, runtime, args):
        results, missing = {}, []
        for item in items:
            key = str(item['root'])
            if key not in verified:
                missing.append(item)
                continue
            previous = verified[key]
            need(item['request'] == previous['request'] and item['binding'] == previous['binding'],
                 'Same-run verified native request/binding changed')
            results[key] = copy.deepcopy(previous['result'])
        if missing:
            values = original(missing, runtime, args)
            for item in missing:
                key, root = str(item['root']), Path(item['root'])
                boundary = Inputs()
                for name in ARTIFACTS | {'receipt.json'}:
                    path = root / name
                    if path.is_file():
                        boundary.guard(path)
                need((root / 'receipt.json').is_file(), 'Native result lacks committed receipt')
                source = next(parent for parent in root.parents if parent.parent.name == 'sources')
                source_stats.setdefault(source.resolve(), {}).update(boundary.stats)
                verified[key] = dict(request=copy.deepcopy(item['request']), binding=copy.deepcopy(item['binding']),
                                     result=copy.deepcopy(values[key]))
            results.update(values)
        return results
    bar.transactions = transactions
    try:
        yield source_stats
    finally:
        bar.transactions = original


def completed_index(out, admission, configuration):
    bar, pipeline = direct.native()
    selected = {s['meta']['id']: s for s in admission['selected']}
    index = {}
    for value in ready_lines(out / 'completed.jsonl', bar):
        need(value['id'] in selected and value['id'] not in index, 'Unowned or duplicate completed source')
        need(value['params_sha256'] == bar.batch.digest(configuration) and value.get('caption_recipe') == direct.RECIPE and
             all(value.get(k) == selected[value['id']]['meta'][k] for k in ('id', 'split', 'batch_id', 'source_ordinal')) and
             value.get('producer_terminal_ref') == selected[value['id']]['producer_terminal_ref'],
             'Completed recipe/configuration/source differs')
        path = pipeline.source_dir(out, selected[value['id']]['meta']) / 'consumer.terminal.json'
        need(pipeline.load(path) == value, 'Completed differs from immutable lane terminal')
        if value.get('joint_candidate_ref'):
            direct.Evidence().checked(value['joint_candidate_ref'])
        index[value['id']] = value
    return index


def run_consumer(args, runtime, resource_admission=None):
    """One finite frozen-ready lane; caller manages resources and optional native runtime_loader.

    No GPU/runtime is constructed, no worker/capture/signal is managed here.
    Exactly one lane writer holds its bundle lock through the run. An interrupted
    native attempt resumes only in this lane and under identical owned protocol.
    """
    need(callable(runtime) and callable(resource_admission), 'Caller runtime/resource admission required')
    if not getattr(args, 'handoff_ref', None) and getattr(args, 'handoff', None) and getattr(args, 'handoff_sha', None):
        args.handoff_ref = dict(path=str(Path(args.handoff).resolve()), sha256=args.handoff_sha)
    need(isinstance(getattr(args, 'handoff_ref', None), dict), 'Mandatory native handoff_ref required')
    need(args.stage == 'run' and args.tp == 2 and args.batch_size == 8 and args.max_num_seqs >= 8 and
         args.boundary_review == 'off' and not args.retry_failed and args.limit > 0 and args.style_max_tokens > 0,
         'Finite run TP2/batch8/boundary-off/no-retry required')
    out = Path(args.out).resolve()
    source_root = Path(args.source_root).resolve()
    need(not Path(args.out).is_symlink() and not out.is_relative_to(source_root) and not source_root.is_relative_to(out),
         'Independent source/output roots required')
    need(Path(args.hand_roi_root).resolve() == source_root / 'roi', 'Original producer ROI root required')
    need(resource_admission(args, runtime) is True, 'Caller resource/runtime admission rejected')
    bar, pipeline = direct.native()
    identity_inputs = Inputs()
    protocol = execution_identity(args, identity_inputs)
    exists = out.exists() and any(p.name != '.lock' for p in out.iterdir())
    if exists:
        need((out / 'protocol.json').exists(), 'Output is not same-recipe owned resume; refuse')
        previous_protocol = pipeline.load(out / 'protocol.json')
        need({k: v for k, v in previous_protocol.items() if k != 'admission_sha256'} == protocol and
             'admission_sha256' in previous_protocol, 'Output is not same-recipe owned resume; refuse')
        protocol = previous_protocol
    with bar.batch.bundle_lock(out):
        assignment_inputs = Inputs()
        if exists:
            admission = assignment_inputs.evidence.load(out / 'admission.json')
            assignment(args, assignment_inputs)
            need(admission['schema'] == SCHEMA and admission['selection'] == protocol['selection'] and
                 pipeline.digest(admission) == protocol['admission_sha256'], 'Owned admission selection/digest differs')
        else:
            admission = freeze_admission(args, assignment_inputs)
            protocol['admission_sha256'] = pipeline.digest(admission)
        # Enter the single recipe once; fresh paths are still empty at this point.
        from curated_transaction_cache_v1 import transaction_cache
        transaction_events = []
        with transaction_cache(bar, transaction_events.append), verified_transactions(bar) as receipt_stats, owned_recipe(bar, args, protocol, fresh=not exists):
            code = protocol['native_configuration']['code_sha256']
            configuration = bar.params(args, code)
            configuration.update(production_execution_refs=protocol['execution_refs'], style_max_tokens=args.style_max_tokens)
            # Native digest is indent-based and preserves insertion order. Bind a stable
            # ordered configuration before either native receipts or sorted pipeline persistence.
            configuration = bar.core.strict_json(pipeline.raw_json(configuration))
            expected_args, expected_params = copy.deepcopy(vars(args)), bar.params
            params_sha = bar.batch.digest(configuration)
            pipeline.save(out / 'protocol.json', protocol)
            pipeline.save(out / 'admission.json', admission)
            index = completed_index(out, admission, configuration)
            ref_cache = dict(production_manifest_summary=assignment_inputs.evidence.checked(admission['manifest_summary_ref']),
                             production_batch_rows={}, production_source_plan=assignment_inputs.evidence.load(args.source_plan))
            reuse = pipeline.load_style_reuse(args.style_reuse)
            assignment_inputs.verify()
            identity_inputs.verify()
            calls = {'visual': [], 'style': []}
            loaded = []
            class ChatCounter:
                def __init__(self, llm):
                    self.llm = llm
                def __getattr__(self, name):
                    return getattr(self.llm, name)
                def chat(self, messages, *a, **kw):
                    key = 'visual' if messages[0][0]['content'] == bar.SYSTEM else 'style'
                    calls[key].append(len(messages))
                    return self.llm.chat(messages, *a, **kw)
            def measured_runtime():
                if not loaded:
                    llm, sampling = runtime()
                    loaded.append((ChatCounter(llm), sampling))
                return loaded[0]
            pending = []
            recovered = 0
            def commit(source, value):
                meta = source['item']['meta']
                need(vars(args) == expected_args and bar.params is expected_params, 'Native configuration changed before commit')
                value.update(params_sha256=params_sha, caption_recipe=direct.RECIPE,
                             source_ordinal=meta['source_ordinal'], producer_terminal_ref=source['item']['producer_terminal_ref'], **FLAGS)
                path = pipeline.source_dir(out, meta) / 'consumer.terminal.json'
                pipeline.save(path, value)
                if meta['id'] not in index:
                    pipeline.append(out / 'completed.jsonl', value)
                    index[meta['id']] = value
            def source_boundary(source, cache=None, staged=None):
                if cache is not None:
                    root = source['entry']['root']
                    source['inputs'].stats.update(receipt_stats.get(root.resolve(), {}))
                    cache.verify_boundary()
                    need(staged.entries == staged.inputs, 'Staged source inputs changed')
                need(vars(args) == expected_args and bar.params is expected_params, 'Native configuration changed before commit')
                source['inputs'].verify()
            def group(sources):
                entries = [s['entry'] for s in sources]
                # Shared initialization errors never become source failures.
                measured_runtime()
                with ExitStack() as locks:
                    for source in sources:
                        locks.enter_context(bar.batch.bundle_lock(pipeline.source_dir(out, source['item']['meta'])))
                        locks.enter_context(bar.batch.bundle_lock(source['entry']['root']))
                    with group_input_cache(bar, entries) as caches:
                        staged = cross.stage_sources(entries, args, measured_runtime, ref_cache)
                        for i, source in enumerate(sources):
                            meta, root = source['item']['meta'], source['entry']['root']
                            try:
                                visual = cross.resume_source(staged, i, args, measured_runtime, ref_cache)
                                prepared = check_visual(visual, root, bar)
                                style = source['style']
                                if style is None:
                                    style = pipeline.observe_dance_style(source['entry']['row']['manifest_record'],
                                        source['entry']['row'], visual, root, args, measured_runtime, reuse)
                                need(style.get('id') == meta['id'], 'Style source identity differs')
                                if style.get('dance_style') not in (None, '', 'unknown'):
                                    need(style.get('source') in ('historical_RGB_label', 'rich_chapter2', 'independent_RGB') and
                                         style.get('metadata', {}).get('video_sha256') == source['entry']['row']['binding']['video_sha256'],
                                         'Known style lacks same-video RGB/rich binding')
                                source['inputs'].refs(style.get('metadata', {}).get('source_refs', {}))
                                identity_ref = visual.get('model_identity_ref')
                                model_sha = bar.load(identity_ref['path'])['sha256'] if identity_ref else None
                                joint = pipeline.joint_candidate(source['entry']['row']['manifest_record'], visual, style, model_sha)
                                joint.update(source_identity=prepared['source_identity'], actual_video_probe=prepared['probe'],
                                    original_actual_media_ref=source['original_actual_media_ref'],
                                    historical_source_identity=source['actual']['source_identity'], caption_recipe=direct.RECIPE, **FLAGS)
                                directory = pipeline.source_dir(out, meta)
                            except RuntimeError as error:
                                raise pipeline.SharedFailure('Shared visual/style runtime failure: ' + str(error)) from error
                            except LOCAL_ERRORS as error:
                                value = dict(id=meta['id'], split=meta['split'], batch_id=meta['batch_id'], status='failed',
                                    error=type(error).__name__ + ': ' + str(error), manifest_record=source['entry']['row']['manifest_record'])
                            else:
                                value = None
                            source_boundary(source, caches[root.resolve()], staged)
                            if value is None:
                                style_path = pipeline.save(directory / 'style-results' / (pipeline.digest(style) + '.json'), style)
                                joint['style_result_ref'] = pipeline.ref(style_path)
                                snapshot = pipeline.save(directory / 'joint-results' / (pipeline.digest(joint) + '.json'), joint)
                                value = dict(id=meta['id'], split=meta['split'], batch_id=meta['batch_id'],
                                    status='joint_candidate' if joint['joint_complete'] else 'incomplete_candidate',
                                    joint_complete=joint['joint_complete'], joint_candidate_ref=pipeline.ref(snapshot),
                                    style_status=style['status'], visual_status=visual['status'])
                            commit(source, value)
            for item in admission['selected']:
                sid = item['meta']['id']
                if sid in index:
                    continue
                terminal = pipeline.source_dir(out, item['meta']) / 'consumer.terminal.json'
                if terminal.exists():
                    previous = pipeline.load(terminal)
                    need(previous['params_sha256'] == bar.batch.digest(configuration) and
                         previous.get('producer_terminal_ref') == item['producer_terminal_ref'] and
                         previous.get('caption_recipe') == direct.RECIPE and
                         all(previous.get(k) == item['meta'][k] for k in ('id', 'split', 'batch_id', 'source_ordinal')),
                         'Orphan terminal owned configuration/source differs')
                    if previous.get('joint_candidate_ref'):
                        direct.Evidence().checked(previous['joint_candidate_ref'])
                    pipeline.append(out / 'completed.jsonl', previous)
                    index[sid] = previous
                    recovered += 1
                    continue
                try:
                    source = load_source(item, args, configuration, ref_cache)
                except LOCAL_ERRORS as error:
                    # Preserve missing/corrupt admission inputs pending; don't mint a source receipt without binding.
                    raise ValueError('Admitted source input unavailable: ' + sid + ': ' + str(error)) from error
                if source['entry'] is None:
                    row = source['producer']['manifest_record']
                    source_boundary(source)
                    commit(source, dict(id=sid, split=row['split'], batch_id=row['batch_id'],
                           status='producer_failed', manifest_record=row))
                    continue
                pending.append(source)
                if len(pending) == 3:
                    group(pending)
                    pending = []
            if pending:
                group(pending)
            assignment_inputs.verify()
            identity_inputs.verify()
            need(vars(args) == expected_args and bar.params is expected_params, 'Native configuration changed before run summary')
            report = dict(schema=SCHEMA, caption_recipe=direct.RECIPE, selected_source_count=len(admission['selected']),
                terminal_count=len(index), recovered_completed_notices=recovered,
                counts={status: sum(v['status'] == status for v in index.values()) for status in sorted({v['status'] for v in index.values()})},
                admission_ref=pipeline.ref(out / 'admission.json'), protocol_ref=pipeline.ref(out / 'protocol.json'),
                finite_ready_snapshot=True, single_completed_writer=True, boundary_model_calls=0, rewrite_model_calls=0,
                visual_chat_batch_sizes=calls['visual'], style_chat_batch_sizes=calls['style'],
                new_style_model_calls=sum(calls['style']), groups_use_at_most_three_sources=True,
                final_short_group_not_padded=True, transaction_cache_events=transaction_events,
                independent_review_required=True, **FLAGS)
            pipeline.save(out / 'summaries' / (pipeline.digest(report) + '.json'), report)
            return report


def parser():
    bar, _ = direct.native()
    p = bar.parser()
    p.add_argument('--source-plan', default=str(SOURCE_PLAN))
    p.add_argument('--source-root', default=str(BAR / 'curated222939_collaborative_v3'))
    p.add_argument('--manifest', default=str(BAR / 'curated222939_inputs_v1/manifest.summary.json'))
    p.add_argument('--member', required=True)
    p.add_argument('--handoff', required=True)
    p.add_argument('--handoff-sha', required=True)
    p.add_argument('--start-ordinal', type=int, default=32000)
    p.add_argument('--end-ordinal', type=int, default=62000)
    p.add_argument('--style-reuse', action='append', default=[])
    p.add_argument('--style-max-tokens', type=int, default=1024)
    p.set_defaults(limit=500, shard=0, nshard=1)
    return p
