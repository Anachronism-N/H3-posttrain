"""Test-only, bounded, serial shared-filesystem publication A/B; never runs a lane."""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import tarfile
import tempfile
import time
import uuid

ROOT = Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher')
P = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1/external_ms_unified_clusters80k_20261027_v1')
SAMPLE = P / 'results/a100-extra-03/batches/gpu0/g0-1793043643552659367-0bb56122-batch007'
SHARED = P / 'results/a100-extra-03/publication_experiments'
FILES = {'pose.npz', 'fullframe_manifest.json', 'beats.json', 'draft.json', 'receipt.json'}
INPUT_LIMIT = 64 * 1024**2
LOCAL_BUNDLE_LIMIT = 8 * 1024**2
SECONDS_LIMIT = 120
started = time.perf_counter()
run = 'bench_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
local = ROOT / 'publication_experiments' / run
shared = SHARED / run
report = {
    'schema': 'publication_bundle_benchmark_report_v1',
    'test_only': True, 'no_annotation': True, 'new_source_count': 0,
    'production_counts_modified': False, 'production_pipeline_switched': False,
    'normal_running_unchanged_by_experiment': True,
    'observer': {'gpu_or_lane_modifications': 0, 'signals_to_other_processes': 0,
                 'normal_still_live': 'not_observed_by_this_filesystem_experiment'},
    'measurement_scope': 'launcher_to_shared_filesystem_microbenchmark_not_full_RPC_protocol',
    'worker_node': os.uname().nodename,
    'limits': {'input_bytes': INPUT_LIMIT, 'elapsed_seconds': SECONDS_LIMIT,
               'local_bundle_bytes_exclusive': LOCAL_BUNDLE_LIMIT, 'copy_threads': 1},
    'paths': {'script': str(ROOT / 'publication_bundle_benchmark.py'),
              'local': str(local), 'shared': str(shared), 'published_sample': str(SAMPLE)},
    'trials': [],
    'authorized_retry': {'attempt': 2, 'further_automatic_retries': False,
                         'first_failure_run_retained': 'bench_20261026T210902Z_f5cd6ddb'},
    'destination_readback_metadata_diagnostics': [],
    'readback_safety': {'expected_content_sha256': True, 'byte_exact_comparison': True,
                        'opened_fd_before_after_size_and_complete_read_length': True,
                        'destination_timestamps_diagnostic_only': True,
                        'published_source_timestamp_checks_unchanged': True},
    'central_observer_evidence': {'provided_by': 'team-lead', 'observation_time': '21:12',
                                  'active_gpu_lanes': 16, 'source_failed': 0,
                                  'frozen_code_unchanged': True,
                                  'new_published': {'a100-extra-03': 960,
                                                    'a100-extra-04': 80,
                                                    'a100-extra-05': 80,
                                                    'a100-extra-06': 80}},
    'compatibility': {'bundle_only_support_by_current_consumer': 'not_established',
                      'zero_expanded_files_in_experiment_bundle_only': True,
                      'zero_expanded_files_in_production': False,
                      'deployment': 'future_frozen_version_only_after_central_consumer_acceptance'},
}

def bounded():
    if time.perf_counter() - started >= SECONDS_LIMIT:
        raise TimeoutError('experiment 120s budget reached; production untouched')

def alarm_handler(signum, frame):
    raise TimeoutError('experiment 120s alarm; no signals sent to GPU/lane processes')

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()

def strict(raw):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError('duplicate JSON key')
            obj[key] = value
        return obj
    def bad(value):
        raise ValueError('nonfinite JSON')
    return json.loads(raw.decode('utf-8', 'strict'), object_pairs_hook=pairs, parse_constant=bad)

def no_redirect(path):
    current = path
    while not current.exists() and not current.is_symlink():
        current = current.parent
    if current.resolve(strict=True) != current:
        raise ValueError('redirected filesystem ancestor')

def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def make_dir(path):
    if not (path == local or local in path.parents or path == shared or shared in path.parents):
        raise ValueError('write outside isolated experiment roots')
    no_redirect(path)
    path.mkdir(parents=True, exist_ok=True)
    no_redirect(path)
    sync_dir(path.parent)

def read_once(path, cap, destination_readback=False):
    no_redirect(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
            raise ValueError('read type or size limit: ' + str(path))
        raw = stream.read(cap + 1)
        after = os.fstat(stream.fileno())
    timestamps_stable = (before.st_mtime_ns, before.st_ctime_ns) == (after.st_mtime_ns, after.st_ctime_ns)
    if destination_readback:
        report['destination_readback_metadata_diagnostics'].append({
            'path': str(path), 'opened_fd_bytes_before': before.st_size,
            'opened_fd_bytes_after': after.st_size, 'read_bytes': len(raw),
            'mtime_ns_before': before.st_mtime_ns, 'mtime_ns_after': after.st_mtime_ns,
            'ctime_ns_before': before.st_ctime_ns, 'ctime_ns_after': after.st_ctime_ns,
            'timestamp_change_observed': not timestamps_stable,
            'not_content_integrity_verdict': True,
        })
    if len(raw) != before.st_size or len(raw) > cap or before.st_size != after.st_size:
        raise ValueError('opened FD size/readlen/cap mismatch: ' + str(path) +
                         ' before=' + str(before.st_size) + ' after=' + str(after.st_size) +
                         ' read=' + str(len(raw)) + ' cap=' + str(cap))
    if not destination_readback and not timestamps_stable:
        raise ValueError('published source timestamps changed: ' + str(path))
    return raw

def write_once(path, raw):
    bounded()
    make_dir(path.parent)
    fd, temporary = tempfile.mkstemp(prefix='.bench-unpublished-', dir=path.parent)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fdatasync(stream.fileno())
    os.link(temporary, path, follow_symlinks=False)
    sync_dir(path.parent)
    os.unlink(temporary)
    sync_dir(path.parent)
    return {'path': str(path), 'bytes': len(raw), 'sha256': sha(raw)}

def verify_bytes(path, expected):
    bounded()
    raw = read_once(path, INPUT_LIMIT, destination_readback=True)
    if len(raw) != len(expected) or sha(raw) != sha(expected) or raw != expected:
        raise ValueError('destination payload byte/SHA mismatch: ' + str(path))
    return raw

def run_arm(kind, trial, payloads, index_value):
    destination = shared / ('trial%02d' % trial) / kind
    begin = time.perf_counter()
    payload_receipts = []
    if kind == 'baseline':
        for name, raw in payloads.items():
            payload_receipts.append(write_once(destination / name, raw))
    else:
        payload_receipts.append(write_once(destination / 'payload_bundle.tar', bundle_raw))
        payload_receipts.append(write_once(destination / 'member_index.json', index_raw))
    copy_seconds = time.perf_counter() - begin
    verification_start = time.perf_counter()
    if kind == 'baseline':
        for name, raw in payloads.items():
            verify_bytes(destination / name, raw)
    else:
        destination_archive = verify_bytes(destination / 'payload_bundle.tar', bundle_raw)
        destination_index = strict(verify_bytes(destination / 'member_index.json', index_raw))
        if destination_index != index_value or destination_index['archive_sha256'] != sha(destination_archive):
            raise ValueError('index/archive binding mismatch')
        with tarfile.open(fileobj=io.BytesIO(destination_archive), mode='r:') as pack:
            members = pack.getmembers()
            if len(members) != len(payloads) or {m.name for m in members} != set(payloads):
                raise ValueError('destination archive member set')
            for member in members:
                raw = pack.extractfile(member).read()
                if not member.isfile() or raw != payloads[member.name] or sha(raw) != index_value['members'][member.name]['sha256']:
                    raise ValueError('destination member byte/SHA mismatch')
    verify_seconds = time.perf_counter() - verification_start
    marker = {
        'schema': 'publication_benchmark_verified_marker_v1', 'test_only': True,
        'no_annotation': True, 'new_source_count': 0,
        'original_commit_sha256': report['input_verification']['commit_sha256'],
        'kind': kind, 'payload_publications': payload_receipts,
        'verified_members': len(payloads), 'payload_byte_exact': True,
        'payload_files': len(payload_receipts), 'marker_written_last': True,
        'production_consumer_visibility': False,
    }
    marker_start = time.perf_counter()
    marker_raw = encoded(marker)
    write_once(destination / 'benchmark_marker.json', marker_raw)
    verify_bytes(destination / 'benchmark_marker.json', marker_raw)
    marker_seconds = time.perf_counter() - marker_start
    result = {
        'trial': trial, 'kind': kind, 'destination': str(destination),
        'payload_file_count': len(payload_receipts), 'marker_file_count': 1,
        'all_final_file_count': len(payload_receipts) + 1,
        'payload_transferred_bytes': sum(p['bytes'] for p in payload_receipts),
        'unbundled_member_bytes': sum(map(len, payloads.values())),
        'processed_member_files': len(payloads), 'payload_sha_and_byte_exact': True,
        'copy_wall_seconds': copy_seconds, 'verify_wall_seconds': verify_seconds,
        'copy_plus_verify_wall_seconds': copy_seconds + verify_seconds,
        'marker_write_and_verify_wall_seconds': marker_seconds,
        'arm_total_wall_seconds': time.perf_counter() - begin,
        'average_copy_seconds_per_payload_file': copy_seconds / len(payload_receipts),
        'full_assignment_binding_rpc_calls': 0, 'ssh_rpc_calls': 0,
    }
    report['trials'].append(result)
    return result

signal.signal(signal.SIGALRM, alarm_handler)
signal.alarm(SECONDS_LIMIT)
try:
    no_redirect(local)
    no_redirect(shared)
    local.mkdir(parents=True, exist_ok=False)
    shared.mkdir(parents=True, exist_ok=False)
    sync_dir(local.parent)
    sync_dir(shared.parent)
    input_start = time.perf_counter()
    commit_raw = read_once(SAMPLE / 'commit.json', 4 * 1024**2)
    commit = strict(commit_raw)
    archive_start = time.perf_counter()
    archive_raw = read_once(SAMPLE / 'batch.tar', INPUT_LIMIT - len(commit_raw))
    archive_read_seconds = time.perf_counter() - archive_start
    archive_sha = sha(archive_raw)
    if commit.get('schema') != 'ms_unified_novlm_batch_commit_v1' or commit.get('worker') != 'a100-extra-03' or commit.get('gpu') != 0 or commit.get('edition') != P.name:
        raise ValueError('sample identity binding')
    expected = {}
    archive_publication = None
    for item in commit['publications']:
        path = Path(item['path'])
        if not path.is_relative_to(SAMPLE) or str(path) != item['path'] or '..' in path.parts:
            raise ValueError('publication path binding')
        name = path.relative_to(SAMPLE).as_posix()
        receipt = item['receipt']
        if receipt.get('path') != item['path'] or receipt.get('sha256') != item['sha256'] or not re.fullmatch('[0-9a-f]{64}', item['sha256']) or type(receipt.get('bytes')) is not int:
            raise ValueError('publication receipt/SHA fields')
        if name == 'batch.tar':
            if archive_publication is not None:
                raise ValueError('duplicate archive publication')
            archive_publication = item
        else:
            if name in expected:
                raise ValueError('duplicate publication member')
            expected[name] = {'bytes': receipt['bytes'], 'sha256': item['sha256']}
    if archive_publication is None or archive_sha != commit['batch_sha256'] or archive_sha != archive_publication['sha256'] or len(archive_raw) != archive_publication['receipt']['bytes']:
        raise ValueError('published archive SHA/byte mismatch')
    receipts = commit['receipts']
    if len(receipts) != 20 or len(expected) != 100 or commit['unique_sources_attempted'] != 20:
        raise ValueError('sample must be exactly 20 published sources/100 members')
    source_cases = {}
    for receipt in receipts:
        row = receipt['source_row']
        case = 'cases/' + row['split'] + '/' + row['id']
        if case in source_cases or receipt.get('status') != 'extracted_rule_draft_unapproved':
            raise ValueError('sample duplicate or unsuccessful source')
        source_cases[case] = {'split': row['split'], 'id': row['id'],
                             'source_ordinal': row['source_ordinal'],
                             'assignment_index': receipt['assignment_index']}
        if {case + '/' + name for name in FILES} - set(expected):
            raise ValueError('five outputs not bound to published source')
    if [r['assignment_index'] for r in receipts] != commit['assignment_indices']:
        raise ValueError('source receipt/assignment-index binding')
    selected_positions = [0, 6, 12, 19]
    selected_cases = [list(source_cases)[i] for i in selected_positions]
    payloads = {}
    validation_start = time.perf_counter()
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode='r:') as pack:
        members = pack.getmembers()
        if len(members) != 100 or len({m.name for m in members}) != 100 or {m.name for m in members} != set(expected):
            raise ValueError('published tar member set differs from commit')
        for member in members:
            bounded()
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or member.name != PurePosixPath(member.name).as_posix() or len(parts) != 4 or parts[0] != 'cases' or parts[-1] not in FILES or '..' in parts or member.size != expected[member.name]['bytes']:
                raise ValueError('unsafe/member type or bytes mismatch')
            raw = pack.extractfile(member).read()
            if sha(raw) != expected[member.name]['sha256']:
                raise ValueError('published member SHA mismatch')
            if '/'.join(parts[:-1]) in selected_cases:
                payloads[member.name] = raw
    member_verify_seconds = time.perf_counter() - validation_start
    if len(payloads) != 20:
        raise ValueError('selected sample must have 4 sources and 20 files')
    report['input_verification'] = {
        'commit_sha256': sha(commit_raw), 'archive_sha256': archive_sha,
        'archive_bytes': len(archive_raw), 'commit_bytes': len(commit_raw),
        'archive_source_read_calls': 1, 'commit_source_read_calls': 1,
        'archive_read_wall_seconds': archive_read_seconds,
        'archive_and_member_sha_verified': True, 'verified_archive_members': 100,
        'verified_sources': 20, 'original_member_hashes': expected,
        'member_verify_wall_seconds': member_verify_seconds,
        'source_read_and_full_validation_wall_seconds': time.perf_counter() - input_start,
        'media_read_calls': 0, 'annotation_generated': False,
    }
    pack_start = time.perf_counter()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:', format=tarfile.USTAR_FORMAT) as pack:
        for name, raw in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mode = 0o600
            pack.addfile(info, io.BytesIO(raw))
    bundle_raw = buffer.getvalue()
    index_value = {
        'schema': 'publication_benchmark_member_index_v1',
        'test_only': True, 'no_annotation': True, 'new_source_count': 0,
        'original_commit_path': str(SAMPLE / 'commit.json'),
        'original_commit_sha256': sha(commit_raw), 'original_archive_sha256': archive_sha,
        'original_assignment_sha256': commit['assignment_sha256'],
        'archive_name': 'payload_bundle.tar', 'archive_bytes': len(bundle_raw),
        'archive_sha256': sha(bundle_raw), 'source_count': 4, 'member_count': 20,
        'source_bindings': [source_cases[case] for case in selected_cases],
        'members': {name: dict(expected[name], original_case_path=str(SAMPLE / name)) for name in payloads},
    }
    index_raw = encoded(index_value)
    if len(bundle_raw) + len(index_raw) >= LOCAL_BUNDLE_LIMIT:
        raise ValueError('local selected bundle reaches exclusive 8MiB bound')
    report['sample'] = {'source_count': 4, 'selected_original_receipt_positions': selected_positions,
                        'selected_sources': index_value['source_bindings'], 'member_file_count': 20,
                        'member_bytes': sum(map(len, payloads.values())),
                        'bundle_bytes': len(bundle_raw), 'index_bytes': len(index_raw),
                        'local_bundle_build_wall_seconds': time.perf_counter() - pack_start}
    write_once(local / 'payload_bundle.tar', bundle_raw)
    write_once(local / 'member_index.json', index_raw)
    for trial, order in enumerate((('baseline', 'bundle'), ('bundle', 'baseline')), 1):
        bounded()
        for kind in order:
            run_arm(kind, trial, payloads, index_value)
    means = {}
    for kind in ('baseline', 'bundle'):
        arms = [t for t in report['trials'] if t['kind'] == kind]
        means[kind] = {key: sum(t[key] for t in arms) / len(arms)
                       for key in ('copy_wall_seconds', 'verify_wall_seconds',
                                   'copy_plus_verify_wall_seconds', 'marker_write_and_verify_wall_seconds',
                                   'arm_total_wall_seconds', 'payload_transferred_bytes')}
        means[kind]['trials'] = len(arms)
        means[kind]['payload_file_count_per_trial'] = arms[0]['payload_file_count']
    report['means'] = means
    report['microbenchmark_ratio'] = {
        'baseline_over_bundle_copy_plus_verify': means['baseline']['copy_plus_verify_wall_seconds'] / means['bundle']['copy_plus_verify_wall_seconds'],
        'baseline_over_bundle_including_test_marker': means['baseline']['arm_total_wall_seconds'] / means['bundle']['arm_total_wall_seconds'],
        'not_end_to_end_pipeline_speedup': True,
        'caveats': ['two_trials_4_sources_only', 'shared_FS_cache_and_metadata_noise',
                    'does_not_measure_SSH_or_20k_assignment_validation',
                    'test_marker_not_production_commit_or_consumer_acceptance'],
    }
    report['status'] = 'verified_test_only'
except Exception as error:
    report['status'] = 'stopped_test_only_normal_running_unchanged'
    report['error'] = type(error).__name__ + ': ' + str(error)
    report['retained_partial_experiment_outputs'] = True
finally:
    signal.alarm(0)
    report['experiment_wall_seconds_before_report'] = time.perf_counter() - started
    report_raw = encoded(report)
    # Failure reports use only the newly owned local directory after a timeout.
    def retain_report(path):
        no_redirect(path)
        with path.open('xb') as stream:
            stream.write(report_raw)
            stream.flush()
            os.fdatasync(stream.fileno())
        sync_dir(path.parent)
    if local.is_dir():
        retain_report(local / 'benchmark_report.json')
    if shared.is_dir() and report['experiment_wall_seconds_before_report'] < SECONDS_LIMIT:
        write_once(shared / 'benchmark_report.json', report_raw)
    print(json.dumps({k: report.get(k) for k in ('status', 'error', 'paths', 'sample', 'means', 'microbenchmark_ratio', 'experiment_wall_seconds_before_report')}, sort_keys=True))
