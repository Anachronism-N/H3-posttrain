"""Standalone CPU A/B: explicit verified local bytes, never a worker/dispatch entry."""
from __future__ import annotations
import argparse
import base64
import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from types import FunctionType, MappingProxyType, SimpleNamespace

RT = Path(__file__).resolve().parent
BAR = RT.parent
STAT_KEYS = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
FROZEN = {
    'curated_request_image_cache_v1.py': '8e40c5f784ebd548a2ab7d27450cb75a8d4e145576886ed690330925b51c79e3',
    'curated_coarse_empty_filter_recovery_v1.py': 'f9085da43857bc3f05204dd1557674d0d7f645a740f7e260420748dca8acd957',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot(path=None, fd=None):
    s = os.stat(path) if fd is None else os.fstat(fd)
    need(stat.S_ISREG(s.st_mode), 'not a regular file')
    return tuple(getattr(s, key) for key in STAT_KEYS)


def stable_bytes(path):
    path = Path(path)
    before = snapshot(path)
    with path.open('rb') as stream:
        opened = snapshot(fd=stream.fileno())
        raw = stream.read()
        finished = snapshot(fd=stream.fileno())
    need(before == opened == finished == snapshot(path) and len(raw) == before[2],
         'original changed during read')
    return raw, before


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def clone(fn, replacements):
    # The original bytecode and every validator predicate stay intact.
    result = FunctionType(fn.__code__, {**fn.__globals__, **replacements}, fn.__name__,
                          fn.__defaults__, fn.__closure__)
    result.__kwdefaults__ = fn.__kwdefaults__
    return result


@dataclass(frozen=True)
class Entry:
    source_key: str
    logical_path: str
    expected_sha256: str
    original_stat: tuple
    original_resolved_path: str
    local_path: str
    local_stat: tuple
    local_content_sha256: str


class Reader:
    def __init__(self, source_key, entries, local_enabled):
        self.source_key = source_key
        self.entries = MappingProxyType(dict(entries))
        self.local_enabled = local_enabled
        self.stats = {'read_count': 0, 'read_bytes': 0, 'sha_bytes': 0,
                      'ceph_read_count': 0, 'local_read_count': 0,
                      'ceph_read_bytes': 0, 'local_read_bytes': 0,
                      'read_seconds': 0.0, 'sha_seconds': 0.0,
                      'explicit_stat_calls': 0, 'explicit_stat_seconds': 0.0,
                      'explicit_resolve_calls': 0, 'explicit_resolve_seconds': 0.0}

    def _stat(self, path=None, fd=None):
        start = time.perf_counter()
        try:
            return snapshot(path, fd)
        finally:
            self.stats['explicit_stat_calls'] += 1
            self.stats['explicit_stat_seconds'] += time.perf_counter() - start

    def _resolve(self, path):
        start = time.perf_counter()
        try:
            return str(Path(path).resolve(strict=True))
        finally:
            self.stats['explicit_resolve_calls'] += 1
            self.stats['explicit_resolve_seconds'] += time.perf_counter() - start

    def read(self, path, expected_sha=None):
        logical = os.fspath(path)
        need(Path(logical).is_absolute(), 'logical path must be absolute')
        need(logical in self.entries, 'path outside explicit source closure')
        e = self.entries[logical]
        need(e.source_key == self.source_key and e.logical_path == logical,
             'source mapping or logical path mismatch')
        need(expected_sha is None or expected_sha == e.expected_sha256, 'expected SHA mismatch')
        need(self._resolve(logical) == e.original_resolved_path, 'original path identity changed')
        need(self._stat(logical) == e.original_stat, 'original identity changed')
        physical = e.local_path if self.local_enabled else logical
        expected_stat = e.local_stat if self.local_enabled else e.original_stat
        before = self._stat(physical)
        need(before == expected_stat, 'local/original file identity changed')
        with Path(physical).open('rb') as stream:
            opened = self._stat(fd=stream.fileno())
            need(opened == before, 'file changed between stat and open')
            start = time.perf_counter()
            raw = stream.read()
            self.stats['read_seconds'] += time.perf_counter() - start
            self.stats['read_count'] += 1
            self.stats['read_bytes'] += len(raw)
            domain = 'local' if self.local_enabled else 'ceph'
            self.stats[domain + '_read_count'] += 1
            self.stats[domain + '_read_bytes'] += len(raw)
            finished = self._stat(fd=stream.fileno())
        need(before == opened == finished == self._stat(physical) and len(raw) == before[2],
             'content changed during read')
        need(self._stat(logical) == e.original_stat and
             self._resolve(logical) == e.original_resolved_path, 'original changed during local read')
        sha = self.hash(raw)
        need(sha == e.expected_sha256 == e.local_content_sha256, 'content SHA changed')
        return raw

    def load(self, path):
        return json.loads(self.read(path))  # Identical semantics to frozen q.load.

    def checked(self, ref):
        return json.loads(self.read(ref['path'], ref['sha256']))

    def hash(self, raw):
        start = time.perf_counter()
        value = digest(raw)
        self.stats['sha_seconds'] += time.perf_counter() - start
        self.stats['sha_bytes'] += len(raw)
        return value

    def sha(self, path):
        return self.hash(self.read(path))

    def verify_bytes(self, ref):
        self.read(ref['path'], ref['sha256'])


class ReadRoot:
    """Only a validate_generation input; never installed as pathlib.Path."""
    def __init__(self, path, reader):
        self.path, self.reader = Path(path), reader

    def __truediv__(self, name):
        return ReadRoot(self.path / name, self.reader)

    def __fspath__(self):
        return str(self.path)

    def read_bytes(self):
        return self.reader.read(self.path)


class MetadataProfile:
    """Current-thread profile only; does not alter any filesystem method."""
    def __enter__(self):
        self.old = sys.getprofile()
        need(self.old is None, 'existing profiler must not be displaced')
        self.stats = {'os_stat_calls': 0, 'os_stat_seconds': 0.0,
                      'os_lstat_calls': 0, 'os_lstat_seconds': 0.0,
                      'os_fstat_calls': 0, 'os_fstat_seconds': 0.0,
                      'path_resolve_calls': 0, 'path_resolve_seconds': 0.0}
        self.pending = {}
        sys.setprofile(self._profile)
        return self

    def _profile(self, frame, event, arg):
        if event in ('c_call', 'c_return', 'c_exception') and arg in (os.stat, os.lstat, os.fstat):
            kind = 'os_' + arg.__name__
            key = (id(frame), kind)
            if event == 'c_call':
                self.pending[key] = time.perf_counter()
                self.stats[kind + '_calls'] += 1
            elif key in self.pending:
                self.stats[kind + '_seconds'] += time.perf_counter() - self.pending.pop(key)
        elif frame.f_code is Path.resolve.__code__:
            key = (id(frame), 'path_resolve')
            if event == 'call':
                self.pending[key] = time.perf_counter()
                self.stats['path_resolve_calls'] += 1
            elif event == 'return' and key in self.pending:
                self.stats['path_resolve_seconds'] += time.perf_counter() - self.pending.pop(key)

    def __exit__(self, *exc):
        sys.setprofile(self.old)


@contextmanager
def rgb_reader(private_bar, reader, rgb_root):
    original = private_bar.rgb_content

    def replacement(root, frames, header, hand_evidence=None):
        need(str(Path(root)) == str(rgb_root), 'source RGB root mismatch')
        content = [{'type': 'text', 'text': json.dumps(header, ensure_ascii=False, allow_nan=False)}]
        for frame in frames:
            raw = reader.read(Path(root) / frame['file'], frame['sha256'])
            content.extend([{'type': 'text', 'text': f"Source frame {frame['index']}, time {frame['time']:.9f}s, original PTS {frame['source_timestamp']:.9f}s."},
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii')}}])
            if hand_evidence:
                for part in ('left_hand', 'right_hand'):
                    for crop in hand_evidence[part]['crops']:
                        if crop['frame_index'] != frame['index']:
                            continue
                        raw = reader.read(crop['path'], crop['sha256'])
                        private_bar.need(reader.hash(raw) == crop['sha256'], 'hand ROI JPEG changed before request')
                        metadata = {k: crop[k] for k in ('part', 'frame_index', 'time', 'source_timestamp',
                            'crop_xyxy', 'source_frame_size', 'crop_pixel_size', 'ROI_to_source_affine',
                            'source_to_ROI_affine', 'supportpointindices', 'hand_support_xyxy', 'rawframe_sha256')}
                        metadata.update({'same_source_frame_as_body_context': True, 'not_another_time_observation': True,
                            'crop_adds_native_detail': False, 'anatomical_side_is_unreviewed': True})
                        content.extend([{'type': 'text', 'text': json.dumps(metadata, allow_nan=False)},
                            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii')}}])
        private_bar.need(sum(c['type'] == 'image_url' for c in content) <= 16, 'body plus ROI exceeds 16 image budget')
        return content

    private_bar.rgb_content = replacement
    try:
        yield
    finally:
        private_bar.rgb_content = original


def discover(worker, hint):
    start = time.perf_counter()
    refs, loaded = {}, {}

    def add(path, expected=None):
        path = str(path)
        raw, s = stable_bytes(path)
        sha = digest(raw)
        need(expected is None or expected == sha, 'discovery expected SHA mismatch: ' + path)
        if path in refs:
            need(refs[path][0] == sha, 'conflicting discovery SHA')
        refs[path] = (sha, s, str(Path(path).resolve(strict=True)))
        loaded[path] = raw
        return raw

    def checked(ref):
        return json.loads(add(ref['path'], ref['sha256']))

    job = checked(hint['job_ref'])
    checked(worker.consumer_ref)
    need((job['source_key'], job['target_window_id']) in worker.c_targets, 'not fixed three-source scope')
    prepared = checked(job['refs']['prepared'])
    original = checked(job['refs']['visual_result'])
    checked(job['refs']['plan_ref'])
    root = Path(job['visual_root'])
    manifest = checked(job['target_manifest_ref'])
    for name in ('binding.json', 'plan.json', 'plan.line.raw.json', 'prepared.receipt.json'):
        add(root / name)
    add(root / 'rgb/manifest.json', prepared['union_manifest_sha256'])
    if prepared['raw2d_mask'].get('available'):
        checked(prepared['raw2d_mask'])
    roi = prepared['hand_roi']
    if roi.get('available'):
        checked(roi)
        roi_root = Path(roi['path']).parent
        add(roi_root / 'source_manifest.json', roi['source_manifest_sha256'])
        add(roi_root / 'extract_receipt.json', roi['extract_receipt_sha256'])
    for frame in manifest['frames']:
        add(root / 'rgb' / frame['file'], frame['sha256'])
    for part in ('left_hand', 'right_hand'):
        for crop in manifest.get('hand_roi_evidence', {}).get(part, {}).get('crops', []):
            checked_crop = add(crop['path'], crop['sha256'])
    attempt = worker.OUT / 'revisions' / worker.REVISION / job['source_key'] / 'windows' / job['target_window_id'] / 'attempt-000001'
    receipt = json.loads(add(attempt / 'receipt.json'))
    for name, expected in receipt['files_sha256'].items():
        add(attempt / name, expected)
    preview = worker.OUT / 'request-previews' / job['source_key'] / job['target_window_id'] / 'request.json'
    add(preview)
    old_attempt = Path(next(o for o in original['window_observations'] if o['window']['window_id'] == job['target_window_id'])['attempt_path'])
    add(old_attempt / 'request.json', worker.TARGETS[job['id']][5])
    budget = worker.budget_path(job)
    add(budget)
    add(attempt.parent.parent.parent / 'canonical_budget_claim.json')
    return {'hint': hint, 'job': job, 'prepared': prepared, 'root': root, 'attempt': attempt,
            'preview': preview, 'budget': budget, 'refs': refs, 'loaded': loaded,
            'discovery_seconds': time.perf_counter() - start}


def build_bundle(source, local_root):
    start = time.perf_counter()
    root = local_root / source['job']['id']
    root.mkdir(mode=0o700)
    entries = {}
    total = 0
    for i, (path, (sha, original_stat, resolved)) in enumerate(sorted(source['refs'].items())):
        raw, now = stable_bytes(path)
        need(now == original_stat and digest(raw) == sha, 'original changed after discovery')
        local = root / ('%04d.blob' % i)
        with local.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        local.chmod(0o400)
        check, local_stat = stable_bytes(local)
        need(digest(check) == sha, 'local staging SHA mismatch')
        entries[path] = Entry(source['job']['source_key'], path, sha, original_stat, resolved,
                              str(local), local_stat, digest(check))
        total += len(raw)
    return entries, {'wall_seconds': time.perf_counter() - start,
                     'files': len(entries), 'bytes': total, 'reused_existing_bundle': False,
                     'ceph_reads': len(entries), 'ceph_read_bytes': total,
                     'local_verification_reads': len(entries), 'local_verification_bytes': total}


def replay(worker, source, reader):
    start = time.perf_counter()
    job, prepared, attempt = source['job'], reader.checked(source['job']['refs']['prepared']), source['attempt']
    q = SimpleNamespace(**{**worker.q.__dict__, 'load': reader.load, 'checked': reader.checked,
                            'sha': reader.sha, 'verify_bytes': reader.verify_bytes})
    private_bar = SimpleNamespace(**worker.bar.__dict__)
    make_globals = {'q': q, 'bar': private_bar}
    verify_window = clone(worker.verify_window, make_globals)
    private_bar.visual_request = clone(worker.bar.visual_request, private_bar.__dict__)
    private_args = clone(worker.base.worker_args, {'q': q})
    private_verify_args = clone(worker.verify_args, {'worker_args': private_args})
    make_request = clone(worker.make_request, {**make_globals, 'verify_window': verify_window,
                                              'verify_args': private_verify_args})
    with rgb_reader(private_bar, reader, source['root'] / 'rgb'):
        # The private function globals point to the private instance dictionary.
        private_bar.visual_request.__globals__['rgb_content'] = private_bar.rgb_content
        request = make_request(job, source['hint']['job_ref'], prepared, private_args())
    construction = time.perf_counter() - start
    start = time.perf_counter()
    manifest = verify_window(job, prepared)
    binding = reader.load(source['root'] / 'binding.json')
    row = reader.load(source['root'] / 'plan.json')
    receipt_prepared = reader.load(source['root'] / 'prepared.receipt.json')
    need(binding == prepared['binding'] == receipt_prepared['binding'], 'prepared binding differs')
    need(receipt_prepared['prepared_sha256'] == reader.sha(source['root'] / 'prepared.json'), 'prepared receipt SHA differs')
    need(row == json.loads(reader.read(source['root'] / 'plan.line.raw.json'))
         == reader.checked(job['refs']['plan_ref']), 'plan/raw/original source row differs')
    need(worker.bar.batch.digest(row) == binding['full_source_row_sha256'], 'original full source row digest differs')
    need(reader.load(source['root'] / 'rgb/manifest.json')['binding'] == binding, 'union binding differs')
    if prepared['raw2d_mask'].get('available'):
        reader.verify_bytes(prepared['raw2d_mask'])
    roi = prepared['hand_roi']
    if roi.get('available'):
        reader.verify_bytes(roi)
        for name, key in (('source_manifest.json', 'source_manifest_sha256'), ('extract_receipt.json', 'extract_receipt_sha256')):
            reader.read(Path(roi['path']).with_name(name), roi[key])
    receipt = reader.load(attempt / 'receipt.json')
    expected_binding = {**binding, 'repair_revision': worker.REVISION, 'current_job_ref': source['hint']['job_ref'],
        'original_canonical_job_ref': job['refs']['original_canonical_job'], 'ledger_bindings': job['ledger_bindings'],
        'window': manifest['window'], 'frames_sha256': worker.bar.batch.digest(manifest['frames']),
        'request_preview_ref': {'path': str(source['preview']), 'sha256': reader.entries[str(source['preview'])].expected_sha256}}
    need(reader.load(attempt / 'request.json') == reader.load(source['preview']) == request, 'exact original request differs')
    need(reader.load(attempt / 'binding.json') == receipt['binding'] == expected_binding, 'transaction binding differs')
    files = receipt['files_sha256']
    six = {'request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'response.json', 'final.json'}
    need(set(files) <= six and set(files) >= six - {'response.json'}
         and set(files) == {name for name in six if (attempt / name).exists()}, 'original transaction inventory differs')
    for name, sha in files.items():
        reader.read(attempt / name, sha)
    batch = SimpleNamespace(**{**worker.bar.batch.__dict__, 'sha256': reader.sha})
    original_generation = worker.bar.core.validate_generation
    generation_fn = clone(original_generation, {'batch': batch})
    generation = generation_fn(ReadRoot(attempt, reader), request)
    need(generation['model_identity_sha256'] == worker.expected_model_sha256, 'original annotator checkpoint differs')
    raw = worker.bar.core.strict_json(reader.read(attempt / 'response.raw.txt'))
    failure = None
    try:
        filtered = worker.validate_coarse(raw, manifest, prepared['forbidden'])
    except ValueError as error:
        failure = str(error)
        need(failure == 'No synthesized/repaired part references permitted', 'unrecoverable original validation error')
        filtered = worker.bar.filter_supported_observation(raw, manifest, prepared['forbidden'])
    # Same bounded exact raw recovery guards, with no package validator substitution.
    originals = {e['id']: e for e in raw['events']}
    for event in filtered['events']:
        clause, old = event['source_clause'], originals[event['source_clause']['raw_event_id']]
        need(clause['raw_clause_index'] == 0 and event['description'] == old['description']
             == clause['raw_source_clause'] == clause['original_description'], 'coarse sentence changed')
        need(event['frame_indices'] == old['frame_indices'] and set(event['parts']) <= set(old['parts']), 'coarse references added')
    final = reader.load(attempt / 'final.json')
    need(all(receipt.get(k) is False and final.get(k) is False for k in worker.bar.core.FLAGS), 'approval flags changed')
    if receipt['status'] == 'failed':
        need(final['status'] == 'failed' and final['error'] == receipt['error'] == 'ValueError: ' + failure,
             'failed original terminal differs')
    else:
        need(receipt['status'] == final['status'] == 'candidate' and final['json'] == reader.load(attempt / 'response.json') == filtered,
             'successful final differs from raw replay')
    need(reader.load(source['budget']) == reader.load(attempt.parent.parent.parent / 'canonical_budget_claim.json'), 'budget claim differs')
    validation = time.perf_counter() - start
    canonical = worker.bar.batch.json_bytes(request)
    return {'request': request, 'canonical_request_sha256': digest(canonical), 'canonical_request_bytes': len(canonical),
            'filtered': filtered, 'final': final, 'generation': generation, 'receipt': receipt,
            'construction_seconds': construction, 'validation_seconds': validation,
            'original_validate_coarse_error': failure,
            'native_generation_bytecode_unchanged': generation_fn.__code__ is original_generation.__code__,
            'native_window_bytecode_unchanged': verify_window.__code__ is worker.verify_window.__code__,
            'native_make_request_bytecode_unchanged': make_request.__code__ is worker.make_request.__code__}


def run(report_path, local_parent):
    sys.dont_write_bytecode = True
    before_globals = (Path.open, Path.read_bytes, Path.resolve)
    c = module('pilot3_fixed_scope', RT / 'prepare_curated_coarse_empty_n5_v1.py')
    worker = c.worker_module()
    worker.c_targets = c.TARGETS
    worker.consumer_ref = {'path': str(c.CONSUMER), 'sha256': c.CONSUMER_SHA}
    launch_raw = stable_bytes(c.LAUNCH)[0]
    need(digest(launch_raw) == '9f5e51a9e2461e864c824add7370ff89fcb1b84db1eae13c59a26b81d77f6806', 'original frozen launch changed')
    launch = json.loads(launch_raw)
    worker.expected_model_sha256 = launch['modelref']['model_identity_sha256']
    manifest_raw = stable_bytes(worker.OUT / 'jobs.manifest.json')[0]
    need(launch['jobs_manifest_ref'] == {'path': str(worker.OUT / 'jobs.manifest.json'), 'sha256': digest(manifest_raw)},
         'original launch manifest binding changed')
    manifest = json.loads(manifest_raw)
    need([(h['source_key'], json.loads(stable_bytes(h['job_ref']['path'])[0])['target_window_id']) for h in manifest['jobs']] == c.TARGETS,
         'manifest fixed scope differs')
    code_refs = {}
    for path in (Path(__file__), RT / 'test_curated_local_input_bundle_pilot3_v1.py', RT / 'curated_request_image_cache_v1.py',
                 RT / 'curated_coarse_empty_filter_recovery_v1.py', RT / 'prepare_curated_coarse_empty_n5_v1.py',
                 Path(worker.__file__), Path(worker.bar.__file__), Path(worker.bar.core.__file__)):
        code_refs[str(path)] = digest(stable_bytes(path)[0])
    for name, prefix in FROZEN.items():
        need(code_refs[str(RT / name)] == prefix, 'requested frozen reference SHA differs')
    local_root = Path(tempfile.mkdtemp(prefix='curated-local-input-pilot3-', dir=local_parent))
    rows = []
    budget_before = {}
    sources = [discover(worker, hint) for hint in manifest['jobs']]
    for source in sources:
        budget_before[str(source['budget'])] = digest(stable_bytes(source['budget'])[0])
    original_rgb = worker.bar.rgb_content
    for source in sources:
        # Native original Ceph reconstruction is an untimed equivalence oracle.
        native = worker.make_request(source['job'], source['hint']['job_ref'], source['prepared'], worker.worker_args())
        entries, build = build_bundle(source, local_root)
        modes = []
        results = []
        for local in (False, True):
            reader = Reader(source['job']['source_key'], entries, local)
            start = time.perf_counter()
            with MetadataProfile() as profile:
                value = replay(worker, source, reader)
            wall = time.perf_counter() - start
            need(worker.bar.batch.json_bytes(native) == worker.bar.batch.json_bytes(value['request']), 'native/A/B canonical bytes differ')
            results.append(value)
            modes.append({'mode': 'B_local_enabled' if local else 'A_ceph_explicit_reader', 'feature_enabled': local,
                'reader': reader.stats, 'metadata_profile': profile.stats, 'replay_wall_seconds': wall,
                **{k: v for k, v in value.items() if k not in ('request', 'filtered', 'final', 'generation', 'receipt')}})
        a, b = results
        need(all(a[k] == b[k] for k in ('request', 'filtered', 'final', 'generation', 'receipt')), 'A/B validation or final mismatch')
        rows.append({'source_key': source['job']['source_key'], 'window': source['job']['target_window_id'],
            'discovery_seconds': source['discovery_seconds'], 'bundle_build': build,
            'bundle_entries': [asdict(e) for e in entries.values()], 'A': modes[0], 'B': modes[1],
            'canonical_bytes_equal_native_A_B': True, 'raw_filtered_final_receipt_equal': True,
            'filtered_sha256': digest(worker.bar.batch.json_bytes(a['filtered'])),
            'final_sha256': digest(worker.bar.batch.json_bytes(a['final']))})
        need(worker.bar.rgb_content is original_rgb, 'original rgb_content changed')
        print(json.dumps({'source_key': rows[-1]['source_key'], 'bundle_build': build, 'A': modes[0], 'B': modes[1]}), flush=True)
    budget_after = {path: digest(stable_bytes(path)[0]) for path in budget_before}
    need(budget_before == budget_after, 'original three budgets changed')
    need(before_globals == (Path.open, Path.read_bytes, Path.resolve), 'global Path methods changed')
    test = module('pilot3_negative_tests', RT / 'test_curated_local_input_bundle_pilot3_v1.py')
    negatives = test.run_negative_tests(sys.modules.get(__name__) or SimpleNamespace(Reader=Reader, build_bundle=build_bundle,
                                                stable_bytes=stable_bytes, digest=digest, Entry=Entry), local_root)
    for path, sha in code_refs.items():
        need(digest(stable_bytes(path)[0]) == sha, 'code changed during pilot')
    report = {'schema': 'curated_local_input_bundle_pilot3_v1', 'source_count': 3, 'sources': rows,
        'local_root': str(local_root), 'node': os.uname().nodename, 'code_sha256': code_refs,
        'budget_sha256_before': budget_before, 'budget_sha256_after': budget_after,
        'original_budget_unchanged': True, 'negative_tests': negatives,
        'new_model_calls': 0, 'new_budget_claims': 0, 'GPU_used': False,
        'whole_worker_covered': False, 'full_verify_prepared_covered': False,
        'covered': ['native make_request bytecode', 'native verify_window bytecode and all source/PTS/ROI predicates',
                    'prepared receipt/binding/plan/union/mask/ROI content bindings', 'native validate_generation bytecode',
                    'native validate_coarse plus exact bounded filter recovery', 'original transaction receipt/files/final checks'],
        'limitations': ['No producer, queue, full source CSV, source media probe, full union frame loop, load_hand_roi or GPU dispatch replay.',
            'A uses the same verified explicit reader as B, so this is instrumented Ceph/local content comparison, not native wholeworker baseline.',
            'All timed replay content reads/counts/bytes and content SHA bytes are explicit and exact, including worker_args config. Import/discovery/native oracle are outside timed replay.',
            'Metadata profile counts current-thread os.stat/lstat/fstat and Path.resolve, not kernel syscalls inside open, C extensions or external subprocesses.',
            'Metadata times are nested/inclusive; do not sum resolve and lstat times. Profiling itself adds wall overhead.',
            'Discovery and native oracle warm Ceph caches. First new bundle build is separately reported, but not asserted cold-cache.',
            'No cross-request LRU or cached data URL: B reads and hashes verified local bytes each time. Original Ceph metadata validation remains.',
            'All original files are read-only; fixtures only mutate fresh node-local temp files.']}
    need(Path(report_path).parent == RT and not Path(report_path).exists(), 'report must be a new dedicated runtime JSON')
    with Path(report_path).open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({'report_path': str(report_path), 'report_sha256': digest(stable_bytes(report_path)[0]),
                      'negative_tests': negatives, 'original_budget_unchanged': True}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--local-parent', type=Path, default=Path('/tmp'))
    args = parser.parse_args()
    need(str(args.local_parent.resolve()) in ('/tmp', '/opt'), 'node-local dedicated temp parent required')
    run(args.report, args.local_parent)
