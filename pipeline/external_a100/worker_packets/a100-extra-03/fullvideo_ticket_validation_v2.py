"""Write-once ticket-bound full-video diagnostic; never a production launcher."""
from __future__ import annotations

import argparse
import copy
import datetime
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time

BASE = Path('/tmp/ms_pose2d_draft_20261026T1427Z')
KEYS = ('video_path', 'audio_path', 'npz_path', 'roi_manifest_path',
        'bar_plan_path', 'source_manifest_path')
ROLES = {'a100-extra-03': ('launcher', 3), 'a100-extra-04': ('worker-0', 1),
         'a100-extra-05': ('worker-1', 1), 'a100-extra-06': ('worker-2', 1)}
FLAGS = {'diagnostic_not_production': True, 'semantic_approval': False,
         'quality_approved': False, 'training_eligible': False,
         'needs_visual_revision': True}
SCHEMA = 'ms_fullvideo_pose2d_diagnostic_v1'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_asset(asset):
    path = Path(asset['local_path'])
    before = path.stat()
    if (not path.is_file() or before.st_size != asset['bytes']
            or sha(path) != asset['sha256']):
        raise ValueError('input bytes/SHA256 mismatch: ' + asset.get('key', path.name))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
        raise ValueError('input changed during verification')
    return {'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns,
            'ctime_ns': before.st_ctime_ns, 'inode': before.st_ino}


def verify_inputs(source):
    return {asset['key']: verify_asset(asset) for asset in source['assets']}


def validate_plan(plan_path):
    plan = read_json(plan_path)
    worker = plan.get('worker_id')
    if worker not in ROLES:
        raise ValueError('worker is outside this diagnostic ticket scope')
    role, count = ROLES[worker]
    root = BASE / role
    if (plan.get('role') != role or plan.get('root') != str(root)
            or plan.get('diagnostic_not_production') is not True
            or plan.get('source_count') != count or len(plan.get('sources', [])) != count):
        raise ValueError('plan worker/role/count/root scope mismatch')
    if not root.is_dir() or root.resolve() != root:
        raise ValueError('owned local diagnostic root is unavailable or redirected')
    ticket_path = Path(plan['ticket_path']).resolve(strict=True)
    code_root = Path(plan['code_root']).resolve(strict=True)
    if root not in ticket_path.parents or root not in code_root.parents:
        raise ValueError('ticket/code must be local copied packet assets')
    ticket = read_json(ticket_path)
    if (ticket.get('worker_id') != worker or ticket.get('source_count') != count
            or len(ticket.get('sources', [])) != count
            or ticket.get('role') != 'pose_beat_cpu_rule_validation_only'
            or ticket.get('allowed_gpu_indices') != [0]
            or ticket.get('diagnostic_not_production') is not True
            or ticket.get('formal_production_sources') != 0
            or ticket.get('no_automatic_retry_or_budget_reset') is not True
            or any(ticket.get(key) != value for key, value in FLAGS.items())):
        raise ValueError('ticket worker/GPU0/diagnostic scope mismatch')
    execution = ticket.get('validation_execution')
    if worker == 'a100-extra-03' and execution != 'sequential_three_sources_one_process':
        raise ValueError('03 requires sequential three-source single-process scope')
    if count == 1 and execution not in (None, 'sequential_one_source_one_process',
                                         'single_source_one_process'):
        raise ValueError('worker requires single-source single-process scope')
    if plan.get('output_namespace') != ticket.get('output_namespace'):
        raise ValueError('output namespace differs from ticket')
    namespace = PurePosixPath(plan['output_namespace'])
    if (namespace.is_absolute() or not namespace.parts or '..' in namespace.parts
            or namespace.parts[0] != worker
            or any(re.fullmatch(r'[A-Za-z0-9_-]+', part) is None for part in namespace.parts)):
        raise ValueError('unsafe ticket output namespace')
    if plan.get('ticket_sha256') and sha(ticket_path) != plan['ticket_sha256']:
        raise ValueError('copied ticket SHA256 mismatch')
    seen = set()
    for source, original in zip(plan['sources'], ticket['sources']):
        sid = source.get('id')
        if (not isinstance(sid, str) or re.fullmatch(r'[A-Za-z0-9_-]+', sid) is None
                or sid.casefold() in seen or sid != original.get('id')
                or source.get('split') != original.get('split') or source['split'] != 'train'):
            raise ValueError('source IDs/splits/order are not the unique ticket-bound list')
        seen.add(sid.casefold())
        if set(source.get('original_paths', {})) != set(KEYS):
            raise ValueError('original path binding requires all six ticket inputs')
        assets = source.get('assets', [])
        if len(assets) != len(KEYS) or {a.get('key') for a in assets} != set(KEYS):
            raise ValueError('source assets must bind exactly six unique inputs')
        for asset in assets:
            key = asset['key']
            if (source['original_paths'][key] != original[key]
                    or asset['original_path'] != original[key]
                    or asset['local_path'] != source[key]
                    or not Path(source[key]).is_absolute()
                    or type(asset.get('bytes')) is not int or asset['bytes'] <= 0
                    or re.fullmatch(r'[0-9a-f]{64}', asset.get('sha256', '')) is None):
                raise ValueError('asset is not bound to the copied ticket input')
            if role == 'launcher':
                if source[key] != original[key]:
                    raise ValueError('03 must read its original shared ticket inputs')
            elif root not in Path(source[key]).resolve().parents:
                raise ValueError('relay worker input must be inside its owned local root')
        if Path(source['video_path']).stem != sid or Path(source['audio_path']).stem != sid:
            raise ValueError('media filename does not match source ID')
    required_code = {'ms_hand_roi.py', 'dwpose_full_extract.py', 'ms_pose2d_draft.py',
                     'dwpose/wholebody.py', 'dwpose/onnxdet.py', 'dwpose/onnxpose.py',
                     'dwpose/__init__.py'}
    for relative in required_code:
        path = (code_root / relative).resolve(strict=True)
        if code_root not in path.parents or not path.is_file():
            raise ValueError('local official code snapshot is missing or redirected')
    if 'code_assets' in plan:
        if {str(Path(a['local_path']).relative_to(code_root)) for a in plan['code_assets']} != required_code:
            raise ValueError('copied code receipt set mismatch')
        for asset in plan['code_assets']:
            verify_asset(asset)
    return plan, ticket, root, code_root, namespace


def memory_contract(root):
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024
    readings = []
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if controllers == '':
            mount, limit_name, usage_name = Path('/sys/fs/cgroup'), 'memory.max', 'memory.current'
        elif 'memory' in controllers.split(','):
            mount = Path('/sys/fs/cgroup/memory')
            limit_name, usage_name = 'memory.limit_in_bytes', 'memory.usage_in_bytes'
        else:
            continue
        candidate = mount / relative.lstrip('/')
        candidates = [candidate, *candidate.parents]
        candidates = [p for p in candidates if p == mount or mount in p.parents]
        for directory in candidates:
            limit_path, usage_path = directory / limit_name, directory / usage_name
            if not limit_path.is_file() or not usage_path.is_file():
                continue
            limit_text = limit_path.read_text().strip()
            usage = int(usage_path.read_text())
            limit = None if limit_text == 'max' else int(limit_text)
            readings.append({'path': str(directory), 'limit_bytes': limit, 'usage_bytes': usage,
                             'headroom_bytes': None if limit is None else limit - usage})
    if not readings or available is None:
        raise RuntimeError('cannot independently verify cgroup and available memory')
    headrooms = [available] + [r['headroom_bytes'] for r in readings if r['headroom_bytes'] is not None]
    headroom = min(headrooms)
    disk = shutil.disk_usage(root).free
    result = {'scope': 'independent_small_memory_diagnostic_not_main_production_lifecycle',
              'cgroup_readings': readings, 'host_available_bytes': available,
              'effective_headroom_bytes': headroom, 'required_headroom_bytes': 16 * 2**30,
              'local_disk_free_bytes': disk, 'required_local_disk_bytes': 2 * 2**30}
    if headroom < 16 * 2**30 or disk < 2 * 2**30:
        raise RuntimeError('independent 16GiB memory / 2GiB local disk contract not met')
    return result


def gpu_gate(fresh):
    command = ['nvidia-smi', '-i', '0', '--query-gpu=uuid,memory.used,memory.free',
               '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if result.returncode or len(result.stdout.strip().splitlines()) != 1:
        raise RuntimeError('GPU0 resource query failed')
    uuid, used, free = [part.strip() for part in result.stdout.strip().split(',')]
    result = subprocess.run(['nvidia-smi', '-i', '0', '--query-compute-apps=gpu_uuid,pid',
                             '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError('GPU0 compute PID query failed')
    pids = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(',')]
        if len(parts) != 2 or parts[0] != uuid or not parts[1].isdigit():
            raise RuntimeError('GPU0 compute PID response is not verifiable')
        pids.append(int(parts[1]))
    foreign = [pid for pid in pids if pid != os.getpid()]
    if foreign or (fresh and (pids or int(used) != 0)) or int(free) < 4096:
        raise RuntimeError('GPU0 is occupied or not idle; no other process is modified')
    return {'physical_gpu_index': 0, 'cuda_visible_devices': '0', 'uuid': uuid,
            'memory_used_mib': int(used), 'memory_free_mib': int(free),
            'compute_pids': pids, 'foreign_compute_pids': [], 'fresh_idle_required': fresh}


def process_identity():
    rest = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    ticks = int(rest[19])
    boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                if line.startswith('btime '))
    start_unix = boot + ticks / os.sysconf('SC_CLK_TCK')
    return {'pid': os.getpid(), 'start_ticks': ticks, 'start_unix': start_unix,
            'start_time': datetime.datetime.fromtimestamp(start_unix, datetime.timezone.utc).isoformat(),
            'single_python_process': True, 'sources_execution': 'sequential',
            'child_process_scope': 'only own ffprobe/ffmpeg/nvidia-smi, no model subprocesses'}


class Progress:
    def __init__(self, root, process, worker):
        self.path = root / 'logs/fullvideo_validation_progress.json'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.value = {**FLAGS, **process, 'worker_id': worker, 'started': now(),
                      'status': 'starting', 'current_source': None, 'frames_done': 0,
                      'total_frames': 0, 'physical_gpu_index': 0}

    def update(self, **fields):
        self.value.update(fields, time=now())
        with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent,
                                         prefix='.fullvideo-progress-', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(self.value, stream, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
        print(json.dumps(self.value, allow_nan=False), flush=True)


def import_official(code_root):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code_root))
    modules = {}
    for name in ('dwpose_full_extract', 'ms_hand_roi', 'ms_pose2d_draft'):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != code_root / (name + '.py'):
            raise RuntimeError('official API import escaped copied code snapshot')
        modules[name] = module
    return modules['ms_hand_roi'], modules['ms_pose2d_draft']


def local_model_identity(api, root, code_root):
    code = code_root / 'dwpose'
    files = {'detector': api.file_identity(root / 'model_cache/DWPose/yolox_l.onnx'),
             'pose': api.file_identity(root / 'model_cache/DWPose/dw-ll_ucoco_384.onnx'),
             'wholebody': api.file_identity(code / 'wholebody.py'),
             'onnxdet': api.file_identity(code / 'onnxdet.py'),
             'onnxpose': api.file_identity(code / 'onnxpose.py'),
             'dwpose_init': api.file_identity(code / '__init__.py'),
             'layout_source': api.file_identity(code_root / 'dwpose_full_extract.py'),
             'roi_source': api.file_identity(code_root / 'ms_hand_roi.py')}
    for value in files.values():
        if root not in Path(value['path']).parents:
            raise ValueError('model/code identity is not local')
    return {'scail_root': str(code_root), 'files': files, 'sha256': api.digest(files),
            'joint_layout': api.joint_layout(), 'full_hash_once_per_process': True,
            'upstream_package_init_executed': False}


def probe_native(path, np):
    result = subprocess.run(['ffprobe', '-v', 'error', '-threads', '2', '-select_streams', 'v:0',
                             '-show_frames', '-show_streams', '-show_entries',
                             'frame=best_effort_timestamp_time,pts_time,width,height:'
                             'stream=width,height,duration,avg_frame_rate,time_base',
                             '-of', 'json', str(path)], capture_output=True, timeout=300)
    if result.returncode:
        raise RuntimeError('ffprobe full native-frame PTS query failed')
    value = json.loads(result.stdout)
    streams, frames = value.get('streams', []), value.get('frames', [])
    if len(streams) != 1 or not frames:
        raise ValueError('require one native video stream with actual decoded frame PTS')
    stream = streams[0]
    width, height = int(stream['width']), int(stream['height'])
    if width <= 0 or height <= 0:
        raise ValueError('invalid native video dimensions')
    times, origins = [], set()
    for frame in frames:
        key = 'best_effort_timestamp_time' if 'best_effort_timestamp_time' in frame else 'pts_time'
        if key not in frame:
            raise ValueError('native frame has no real PTS; FPS-derived timestamps forbidden')
        times.append(float(frame[key]))
        origins.add(key)
        if (int(frame.get('width', width)), int(frame.get('height', height))) != (width, height):
            raise ValueError('native video dimensions changed midstream')
    pts = np.asarray(times, dtype=np.float64)
    if not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError('native PTS must be finite and strictly increasing')
    return {'width': width, 'height': height, 'frame_count': len(pts), 'pts': pts,
            'time_origin': float(pts[0]), 'pts_fields': sorted(origins),
            'duration': stream.get('duration'), 'time_base': stream.get('time_base'),
            'avg_frame_rate_metadata_only': stream.get('avg_frame_rate')}


def end_actual_profiles(model, directory):
    result = {}
    for name, session in (('detector', model.session_det), ('pose', model.session_pose)):
        path = Path(session.end_profiling())
        if directory.resolve() not in path.resolve().parents:
            raise RuntimeError('profile output escaped owned case directory')
        events = read_json(path)
        nodes = [event for event in events if event.get('cat') == 'Node'
                 and event.get('args', {}).get('provider')]
        providers = {}
        for event in nodes:
            provider = event['args']['provider']
            providers[provider] = providers.get(provider, 0) + 1
        result[name] = {'providers': session.get_providers(),
                        'cuda_node_events': providers.get('CUDAExecutionProvider', 0),
                        'node_events_by_provider': providers,
                        'profile_path': str(path), 'profile_sha256': sha(path),
                        'evidence_scope': 'first up to four actual video frames only; no synthetic forward',
                        'shape_control_cpu_nodes_allowed': True}
    return result


def coverage(data, np):
    times, valid, tracked = data['times'], data['valid'], data['tracked']
    dt = np.diff(times)
    median = float(np.median(dt)) if len(dt) else None
    gap_indices = np.flatnonzero(dt > median * 1.5) if median is not None else np.array([], dtype=int)
    return {'frame_count': len(times), 'first_relative_time': float(times[0]),
            'last_relative_time': float(times[-1]), 'first_source_timestamp': float(data['source_timestamps'][0]),
            'last_source_timestamp': float(data['source_timestamps'][-1]),
            'tracked_frames': int(tracked.sum()), 'missing_subject_frames': int((~tracked).sum()),
            'valid_point_count': int(valid.sum()), 'total_point_slots': int(valid.size),
            'valid_point_fraction_score_threshold_not_probability': float(valid.mean()),
            'effective_valid_nonfinite_xy_count': int((~np.isfinite(data['xy'][valid])).sum()),
            'effective_valid_nonfinite_score_count': int((~np.isfinite(data['score'][valid])).sum()),
            'raw_nonfinite_xy_values': int((~np.isfinite(data['xy'])).sum()),
            'raw_nonfinite_score_values': int((~np.isfinite(data['score'])).sum()),
            'valid_scores_above_one_count': int((data['score'][valid] > 1).sum()),
            'time_median_step_s': median, 'time_min_step_s': float(dt.min()) if len(dt) else None,
            'time_max_step_s': float(dt.max()) if len(dt) else None,
            'native_pts_gap_policy': 'delta greater than 1.5 times median; diagnostic only, no fill',
            'native_pts_gaps': [{'after_frame_position': int(i), 'start': float(times[i]),
                                 'end': float(times[i + 1]), 'delta_s': float(dt[i])} for i in gap_indices],
            'score_threshold': 0.3, 'score_is_probability': False,
            'interpolation': False, 'three_dimensional_padding': False}


def extract_fullvideo(source, case_dir, api, model, probe, receipt, progress, np, torch):
    n, width, height = probe['frame_count'], probe['width'], probe['height']
    data = {'xy': np.full((n, 134, 2), np.nan, np.float32),
            'score': np.full((n, 134), np.nan, np.float32),
            'valid': np.zeros((n, 134), np.bool_), 'tracked': np.zeros(n, np.bool_),
            'bbox': np.full((n, 4), np.nan, np.float32),
            'frame_indices': np.arange(n, dtype=np.int64),
            'times': probe['pts'] - probe['time_origin'], 'source_timestamps': probe['pts']}
    reasons = []
    previous, decoder, profile_ended, done = None, None, False, 0
    infer_seconds = 0.0
    started = time.monotonic()
    progress.update(status='running', stage='fullvideo_decode_pose', frames_done=0, total_frames=n)
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2', '-noautorotate', '-i', source['video_path'],
               '-map', '0:v:0', '-an', '-sn', '-dn', '-vsync', '0', '-threads', '2',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
    receipt['fullvideo'] = {'schema': SCHEMA, **FLAGS, 'native_width': width, 'native_height': height,
                           'probed_frame_count': n, 'pts_source': 'ffprobe -show_frames',
                           'pts_fields': probe['pts_fields'], 'pts_origin': probe['time_origin'],
                           'original_roi_binding_not_impersonated': True,
                           'new_npz_never_passed_to_rules': True, 'frames_done': 0,
                           'decoder_command': command, 'decoded_frame_count': 0}
    try:
        with (case_dir / 'ffmpeg_decoder.stderr.log').open('xb') as errors:
            decoder = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=errors, bufsize=0)
            receipt['fullvideo']['decoder_pid'] = decoder.pid
            while True:
                buffer = bytearray(width * height * 3)
                view = memoryview(buffer)
                filled = 0
                while filled < len(buffer):
                    amount = decoder.stdout.readinto(view[filled:])
                    if not amount:
                        break
                    filled += amount
                if not filled:
                    del view, buffer
                    break
                if filled != len(buffer):
                    raise ValueError('partial native frame from ffmpeg')
                if done >= n:
                    raise ValueError('ffmpeg decoded more frames than the native PTS probe')
                frame = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
                before = time.monotonic()
                with torch.inference_mode():
                    points, scores, box, reason = api.select_raw_subject(model(frame), previous, .3)
                torch.cuda.synchronize()
                infer_seconds += time.monotonic() - before
                reasons.append(reason)
                if box is not None:
                    data['xy'][done], data['score'][done], data['bbox'][done] = points, scores, box
                    data['tracked'][done] = True
                    data['valid'][done] = api.point_valid(points, scores, width, height, .3, tracked=True)
                    previous = box
                done += 1
                receipt['fullvideo']['frames_done'] = done
                del frame, view, buffer, points, scores, box
                if done == min(4, n):
                    receipt['actual_video_cuda_profiles'] = end_actual_profiles(model, case_dir)
                    profile_ended = True
                    if any(value['cuda_node_events'] == 0 for value in receipt['actual_video_cuda_profiles'].values()):
                        raise RuntimeError('both real sessions require actual-video CUDA profile node evidence')
                if done % 30 == 0 or done == n:
                    progress.update(frames_done=done, total_frames=n,
                                    source_timestamp=float(probe['pts'][done - 1]),
                                    elapsed_source_seconds=time.monotonic() - started)
            returncode = decoder.wait(timeout=30)
            receipt['fullvideo']['decoder_returncode'] = returncode
            receipt['fullvideo']['decoded_frame_count'] = done
            if returncode != 0 or done != n:
                raise ValueError('ffmpeg exit/frame count differs from real native PTS; no frame padding')
    finally:
        if decoder is not None:
            if decoder.stdout is not None:
                decoder.stdout.close()
            if decoder.poll() is None:
                decoder.terminate()
                try:
                    decoder.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    decoder.kill()
                    decoder.wait(timeout=10)
            receipt['fullvideo']['decoder_returncode'] = decoder.returncode
            receipt['fullvideo']['decoded_frame_count'] = done
        if not profile_ended:
            try:
                receipt['actual_video_cuda_profiles'] = end_actual_profiles(model, case_dir)
            except Exception as error:
                receipt['profile_cleanup_error'] = type(error).__name__
        receipt['stage_seconds']['fullvideo_decode_plus_pose'] = time.monotonic() - started
        receipt['stage_seconds']['pose_forward_cuda_synchronized'] = infer_seconds
        receipt['fullvideo']['missing_decoded_frames'] = max(0, n - done)
        receipt['fullvideo']['real_native_pts_full_coverage'] = done == n
        if done:
            receipt['fullvideo']['coverage'] = coverage({key: value[:done] for key, value in data.items()}, np)
            receipt['fullvideo']['coverage']['coverage_scope'] = 'successfully processed native frame prefix'
        else:
            receipt['fullvideo']['coverage'] = {'frame_count': 0, 'missing_decoded_frames': n,
                                                'real_native_pts_full_coverage': False}
    stats = coverage(data, np)
    stats.update({'decoded_equals_probed_framecount': done == n, 'native_pts_strictly_increasing': True,
                  'missing_decoded_frames': n - done, 'frame_indices_contiguous': True,
                  'real_native_pts_full_coverage': True})
    if stats['effective_valid_nonfinite_xy_count'] or stats['effective_valid_nonfinite_score_count']:
        raise ValueError('effective valid wholebody observations are nonfinite')
    path = case_dir / 'new_fullvideo_wholebody134.npz'
    before = time.monotonic()
    with path.open('xb') as stream:
        np.savez_compressed(stream, **data, schema=np.array(SCHEMA), id=np.array(source['id']),
                            split=np.array(source['split']), subject_reason=np.asarray(reasons),
                            joint_layout=np.array(json.dumps(api.joint_layout(), sort_keys=True)),
                            video_sha256=np.array(next(a['sha256'] for a in source['assets'] if a['key'] == 'video_path')),
                            **{key: np.array(value) for key, value in FLAGS.items()})
        stream.flush()
        os.fsync(stream.fileno())
    npz_sha = sha(path)
    receipt['stage_seconds']['fullvideo_npz_write_and_hash'] = time.monotonic() - before
    receipt['fullvideo'].update({'status': 'pass', 'npz_path': str(path), 'npz_sha256': npz_sha,
                                'npz_bytes': path.stat().st_size, 'coverage': stats,
                                'subject_reason_counts': {reason: reasons.count(reason) for reason in set(reasons)}})
    del data


def audio_probe(source, root, case_dir, receipt, progress, np, torch):
    progress.update(stage='beatthis_audio', frames_done=receipt['fullvideo']['frames_done'])
    import soundfile as sf
    from beat_this.inference import File2Beats
    before = time.monotonic()
    binding = read_json(root / 'model_cache/BeatThis/local_source_manifest.json')
    checkpoint = Path(binding['checkpoint_path']).resolve(strict=True)
    if (root / 'model_cache/BeatThis').resolve() not in checkpoint.parents:
        raise ValueError('BeatThis checkpoint must be its existing local formal weight')
    if (not binding.get('official_sha1_match') or not binding.get('model_bytes_unchanged')
            or checkpoint.stat().st_size != binding['checkpoint_bytes']
            or sha(checkpoint) != binding['checkpoint_sha256']):
        raise ValueError('official local BeatThis checkpoint bytes/identity mismatch')
    info = sf.info(source['audio_path'])
    duration = float(info.duration)
    if not math.isfinite(duration) or duration <= 0 or info.frames <= 0:
        raise ValueError('soundfile actual audio duration is not positive finite')
    receipt['stage_seconds']['audio_info_and_binding'] = time.monotonic() - before
    receipt['audio'] = {**FLAGS, 'source_audio_path': source['original_paths']['audio_path'],
                        'actual_input_path': source['audio_path'], 'duration_seconds': duration,
                        'audio_sha256': next(a['sha256'] for a in source['assets'] if a['key'] == 'audio_path'),
                        'sample_rate': info.samplerate, 'channels': info.channels, 'sample_frames': info.frames,
                        'checkpoint_sha256': binding['checkpoint_sha256'], 'checkpoint_path': str(checkpoint),
                        'library': 'official beat_this.inference.File2Beats', 'device': 'cuda:0',
                        'float16': False, 'dbn': False, 'new_beats_not_substituted_for_bound_bar_plan': True,
                        'beat_accuracy_verified': False, 'audio_video_alignment_verified': False}
    beat = None
    try:
        before = time.monotonic()
        with checkpoint.open('rb') as checkpoint_stream:
            beat = File2Beats(checkpoint_path=checkpoint_stream, device='cuda:0', float16=False, dbn=False)
        receipt['audio']['checkpoint_loading'] = 'open local formal checkpoint file; no shortname/URL fallback'
        torch.cuda.synchronize()
        receipt['stage_seconds']['beatthis_model_load'] = time.monotonic() - before
        parameter_devices = sorted({str(p.device) for p in beat.model.parameters()})
        parameter_dtypes = sorted({str(p.dtype) for p in beat.model.parameters()})
        if str(beat.device) != 'cuda:0' or parameter_devices != ['cuda:0'] or beat.float16:
            raise RuntimeError('official BeatThis model parameters must actually be on cuda:0 without float16')
        receipt['audio']['actual_parameter_devices'] = parameter_devices
        receipt['audio']['actual_parameter_dtypes'] = parameter_dtypes
        before = time.monotonic()
        with torch.inference_mode():
            beats, downbeats = beat(source['audio_path'])
        torch.cuda.synchronize()
        receipt['stage_seconds']['beatthis_actual_audio_inference'] = time.monotonic() - before
        arrays = {'beats': np.asarray(beats, dtype=np.float64),
                  'downbeats': np.asarray(downbeats, dtype=np.float64)}
        for name, values in arrays.items():
            if (values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0)
                    or np.any(np.diff(values) <= 0) or np.any(values > duration + .02)):
                raise ValueError(name + ' must be finite, nonnegative, strictly increasing, within actual audio duration')
        receipt['audio'].update({'status': 'pass', **{name: values.tolist() for name, values in arrays.items()},
                                'empty_beats': len(arrays['beats']) == 0,
                                'empty_downbeats': len(arrays['downbeats']) == 0,
                                'validation_tolerance_s': .02})
        write_json(case_dir / 'actual_audio_beats_diagnostic.json', receipt['audio'])
    finally:
        del beat
        gc.collect()
        torch.cuda.empty_cache()


def original_rule_draft(source, root, namespace, case_dir, rules, receipt, progress, np):
    progress.update(stage='original_cache_cpu_rules')
    before = time.monotonic()
    roi = read_json(source['roi_manifest_path'])
    original_row = {**source, **source['original_paths'], 'diagnostic_not_production': True}
    prepared = read_json(source['source_manifest_path'])
    video_sha = next(a['sha256'] for a in source['assets'] if a['key'] == 'video_path')
    npz_sha = next(a['sha256'] for a in source['assets'] if a['key'] == 'npz_path')
    if (roi.get('id') != source['id'] or prepared.get('id') != source['id']
            or roi.get('video_sha256') != video_sha or prepared.get('video_sha256') != video_sha
            or prepared.get('binding_sha256') != roi.get('binding_sha256')
            or roi.get('source', {}).get('path') != source['original_paths']['video_path']
            or roi.get('raw', {}).get('path') != source['original_paths']['npz_path']
            or roi.get('raw', {}).get('sha256') != npz_sha):
        raise ValueError('original ROI/source/NPZ/video receipt binding mismatch')
    required = {'xy', 'score', 'valid', 'tracked', 'times', 'source_timestamps',
                'frame_indices', 'joint_layout', 'binding_sha256', 'video_sha256'}
    with np.load(source['npz_path'], allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise ValueError('original cache missing native observation fields')
        data = {key: archive[key] for key in required | ({'id', 'sid'} & set(archive.files))}
    view = copy.deepcopy(roi)
    view['raw']['path'] = source['npz_path']
    mask = rules.validate_observations(source, view, data)
    bars = rules.load_bar_windows(source, view)
    receipt['original_cache_coverage'] = coverage({**data, 'valid': mask}, np)
    receipt['original_cache_coverage'].update({'frame_indices': data['frame_indices'].tolist(),
                                             'times': data['times'].tolist(),
                                             'source_timestamps': data['source_timestamps'].tolist(),
                                             'effective_valid_count': int(mask.sum()),
                                             'observation_range_never_extended_by_new_npz': True})
    receipt['stage_seconds']['original_cache_validate_and_bars'] = time.monotonic() - before
    provenance = {'assets': source['assets'], 'original_paths': source['original_paths'],
                  'local_paths': {key: source[key] for key in KEYS},
                  'original_roi_file_unchanged': True, 'original_binding_preserved': True,
                  'roi_manifest_sha256': next(a['sha256'] for a in source['assets'] if a['key'] == 'roi_manifest_path'),
                  'in_memory_roi_path_view': {'raw.path': {'original': roi['raw']['path'],
                                                           'local': view['raw']['path']}},
                  'source_facts_and_observed_times_preserved': True,
                  'new_fullvideo_npz_not_used': True, **FLAGS}
    existing = root / 'results_unreviewed' / namespace / source['split'] / source['id'] / 'draft.json'
    before = time.monotonic()
    if source.get('id') and root.name == 'launcher' and existing.exists():
        raw = existing.read_bytes()
        draft = json.loads(raw)
        mode = 'copied_own_prior_draft_without_rule_generation_or_budget_reset'
        receipt['prior_draft'] = {'path': str(existing), 'sha256': hashlib.sha256(raw).hexdigest(),
                                  'bytes': len(raw), 'scope': 'same worker/namespace/split/id existing diagnostic',
                                  'rule_generate_called': False, 'automatic_retry': False}
    else:
        draft = rules.generate_draft(original_row, roi, data, mask, bars)
        draft['diagnostic_not_production'] = True
        draft['relay_provenance'] = provenance
        raw, mode = None, 'official_cpu_rules_original_observations_only'
    if (draft.get('id') != source['id'] or draft.get('split') != source['split']
            or any(draft.get(key) != value for key, value in FLAGS.items())
            or draft.get('schema') != rules.SCHEMA or not draft.get('rules')
            or not draft.get('ms_text') or not draft.get('events')
            or 'unknown_intervals' not in draft.get('observation', {})):
        raise ValueError('draft must retain official schema/unreviewed flags/actions/unknown/rule text')
    index_times = dict(zip(data['frame_indices'].tolist(), data['times'].tolist()))
    for event in draft['events']:
        support = event.get('support', {})
        indices, times = support.get('frame_indices', []), support.get('times', [])
        if (not event.get('text') or not support.get('signal') or not indices
                or len(indices) != len(times)
                or any(i not in index_times or not math.isclose(float(t), index_times[i], abs_tol=1e-6)
                       for i, t in zip(indices, times))):
            raise ValueError('action text/support frames must be real original cached observations')
    expected_source = {'npz_path': source['original_paths']['npz_path'],
                       'roi_manifest_path': source['original_paths']['roi_manifest_path'],
                       'bar_plan_path': source['original_paths']['bar_plan_path']}
    if any(draft['source'].get(key) != value for key, value in expected_source.items()):
        raise ValueError('draft source facts must remain original, not relay or new NPZ paths')
    if (draft['observation']['frame_count'] != len(data['times'])
            or draft['observation']['start'] != float(data['times'][0])
            or draft['observation']['end'] != float(data['times'][-1])):
        raise ValueError('draft observation extent differs from the original cache')
    path = case_dir / 'draft.json'
    if raw is not None:
        with path.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        rules.write_new(path, draft)
    write_json(case_dir / 'relay_provenance.json', provenance)
    receipt['stage_seconds']['cpu_rule_generate_or_own_copy_and_write'] = time.monotonic() - before
    receipt['draft'] = {'path': str(path), 'sha256': sha(path), 'mode': mode,
                        'status': draft['status'], 'action_event_count': len(draft['events']),
                        'window_count': len(draft['windows']), 'unknown_intervals': draft['observation']['unknown_intervals'],
                        'original_flags': {key: draft[key] for key in rules.FLAGS},
                        'relay_provenance_path': str(case_dir / 'relay_provenance.json'),
                        'rules_entry_path': rules.__file__, 'rules_entry_sha256': sha(rules.__file__),
                        'source_cache_bytes_preserved': True, 'new_npz_passed_to_rules': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan, ticket, root, code_root, namespace = validate_plan(args.plan)
    out = root / 'results_unreviewed' / namespace / 'fullvideo_gpu_v2'
    parent = out.parent
    parent.mkdir(parents=True, exist_ok=True)
    if root not in parent.resolve().parents:
        raise ValueError('output parent escaped owned local diagnostic root')
    out.mkdir(exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    sys.dont_write_bytecode = True
    process = process_identity()
    progress = Progress(root, process, plan['worker_id'])
    start = time.monotonic()
    summary = {**FLAGS, **process, 'schema': 'fullvideo_ticket_validation_summary_v2',
               'worker_id': plan['worker_id'], 'ticket_path': plan['ticket_path'],
               'ticket_sha256': sha(plan['ticket_path']), 'plan_sha256': sha(args.plan),
               'output_root': str(out), 'source_ids': [s['id'] for s in plan['sources']],
               'unique_source_ids': sorted({s['id'] for s in plan['sources']}),
               'formal_production_sources': 0, 'production_rate_reported': False,
               'production_speedup_claimed': False, 'automatic_retry': False,
               'shared_writeback_performed': False, 'status': 'running', 'cases': []}
    cases, preflight_errors = {}, {}
    try:
        progress.update(status='running', stage='ticket_input_preflight')
        for source in plan['sources']:
            case_dir = out / source['split'] / source['id']
            case_dir.mkdir(parents=True, exist_ok=False)
            receipt = {**FLAGS, **process, 'schema': 'fullvideo_ticket_case_receipt_v2',
                       'id': source['id'], 'split': source['split'], 'worker_id': plan['worker_id'],
                       'formal_production_sources': 0, 'started': now(), 'status': 'running',
                       'stage_seconds': {}, 'input_assets': source['assets'],
                       'automatic_retry': False, 'new_npz_schema': SCHEMA,
                       'new_npz_not_original_roi_cache': True}
            cases[source['id']] = (case_dir, receipt)
            before = time.monotonic()
            try:
                receipt['verified_input_stats'] = verify_inputs(source)
            except Exception as error:
                preflight_errors[source['id']] = type(error).__name__ + ': ' + str(error)
            receipt['stage_seconds']['input_preflight_sha_size'] = time.monotonic() - before
        if Path(sys.prefix).resolve() != (root / 'venv').resolve():
            raise RuntimeError('ticket requires the owned node-local venv')
        summary['memory_contract'] = memory_contract(root)
        summary['gpu0_initial_gate'] = gpu_gate(fresh=True)
        api, rules = import_official(code_root)
        import numpy as np
        import torch
        torch.set_num_threads(4)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('exactly one visible CUDA device (physical GPU0) is required')
        identity_start = time.monotonic()
        identity = local_model_identity(api, root, code_root)
        summary['model_identity_seconds'] = time.monotonic() - identity_start
        summary['local_pose_model_identity'] = identity
        summary['official_code_snapshot'] = {'code_root': str(code_root),
                                              'rule_entry_sha256': sha(code_root / 'ms_pose2d_draft.py'),
                                              'upstream_init_executed': False}
        write_json(out / 'run_identity.json', {**summary, 'cases': []})
        for source in plan['sources']:
            case_dir, receipt = cases[source['id']]
            case_start, model = time.monotonic(), None
            progress.update(status='running', current_source=source['id'], stage='input_verification',
                            frames_done=0, total_frames=0, elapsed_source_seconds=0)
            try:
                if source['id'] in preflight_errors:
                    raise ValueError(preflight_errors[source['id']])
                receipt['memory_contract'] = memory_contract(root)
                receipt['gpu0_gate'] = gpu_gate(fresh=False)
                before = time.monotonic()
                verify_inputs(source)
                receipt['stage_seconds']['input_reverify_before_models'] = time.monotonic() - before
                receipt['current_stage'] = 'probe_real_native_pts'
                before = time.monotonic()
                probe = probe_native(source['video_path'], np)
                receipt['stage_seconds']['ffprobe_full_native_pts'] = time.monotonic() - before
                receipt['current_stage'] = 'pose_model_load'
                progress.update(stage='pose_model_load', total_frames=probe['frame_count'])
                before = time.monotonic()
                model = api.load_cuda_model(identity, {'num_threads': 4, 'device': 0}, case_dir)
                torch.cuda.synchronize()
                receipt['stage_seconds']['pose_model_load'] = time.monotonic() - before
                receipt['actual_session_providers'] = {'detector': model.session_det.get_providers(),
                                                       'pose': model.session_pose.get_providers()}
                receipt['current_stage'] = 'fullvideo_pose'
                extract_fullvideo(source, case_dir, api, model, probe, receipt, progress, np, torch)
                del model
                model = None
                gc.collect()
                torch.cuda.empty_cache()
                receipt['current_stage'] = 'actual_audio_beatthis'
                audio_probe(source, root, case_dir, receipt, progress, np, torch)
                receipt['current_stage'] = 'original_cache_cpu_rules'
                original_rule_draft(source, root, namespace, case_dir, rules, receipt, progress, np)
                receipt['current_stage'] = 'final_input_byte_verification'
                before = time.monotonic()
                verify_inputs(source)
                receipt['stage_seconds']['final_input_sha_size'] = time.monotonic() - before
                receipt['all_input_bytes_unchanged'] = True
                receipt['status'] = ('partial' if receipt['draft']['status'] != 'draft_generated'
                                     or receipt['fullvideo']['coverage']['missing_subject_frames']
                                     or receipt['audio']['empty_beats'] else 'success')
            except Exception as error:
                receipt['status'] = 'failed'
                receipt['error'] = type(error).__name__ + ': ' + str(error)
                receipt['failed_stage'] = receipt.get('current_stage', 'input_verification')
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                receipt['finished'] = now()
                receipt['case_total_seconds'] = time.monotonic() - case_start
                write_json(case_dir / 'validation_receipt.json', receipt)
                summary['cases'].append({'id': source['id'], 'split': source['split'],
                                         'status': receipt['status'], 'receipt_path': str(case_dir / 'validation_receipt.json'),
                                         'error': receipt.get('error'), **FLAGS})
                progress.update(stage='case_finished', case_status=receipt['status'],
                                elapsed_source_seconds=receipt['case_total_seconds'])
    except Exception as error:
        summary['error'] = type(error).__name__ + ': ' + str(error)
        for source in plan['sources']:
            if source['id'] not in cases:
                continue
            case_dir, receipt = cases[source['id']]
            if (case_dir / 'validation_receipt.json').exists():
                continue
            receipt.update(status='failed', error=preflight_errors.get(source['id'], summary['error']),
                           failed_stage='run_preflight_or_initialization', finished=now())
            write_json(case_dir / 'validation_receipt.json', receipt)
            summary['cases'].append({'id': source['id'], 'split': source['split'], 'status': 'failed',
                                     'receipt_path': str(case_dir / 'validation_receipt.json'),
                                     'error': receipt['error'], **FLAGS})
    finally:
        summary['status_counts'] = {status: sum(case['status'] == status for case in summary['cases'])
                                    for status in ('success', 'partial', 'failed')}
        summary['cases_attempted'] = len(summary['cases'])
        summary['status'] = ('failed' if summary.get('error') or summary['status_counts']['failed']
                             else 'partial' if summary['status_counts']['partial'] else 'success')
        summary['finished'] = now()
        summary['wall_seconds'] = time.monotonic() - start
        write_json(out / 'summary.json', summary)
        progress.update(status=summary['status'], stage='finished', finished=summary['finished'],
                        summary_path=str(out / 'summary.json'), status_counts=summary['status_counts'],
                        elapsed_run_seconds=summary['wall_seconds'])
    return 1 if summary['status'] == 'failed' else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({**FLAGS, 'status': 'rejected_before_run_or_write_once_collision',
                          'error_type': type(error).__name__, 'automatic_retry': False,
                          'formal_production_sources': 0}), file=sys.stderr, flush=True)
        raise SystemExit(1)
