"""Validate 48 fixed music-replacement artifacts, not musical adherence or dance quality."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import math


TAGS = (
    'muq_bridge_only', 'muq_pureclip', 'musicfm_bridge_only', 'musicfm_pureclip',
    'muq_bridgeclip', 'muq_bridgeinit_lambda0', 'musicfm_bridgeclip',
    'musicfm_bridgeinit_lambda0',
)
DEFAULT_EXP_ROOT = Path(__file__).resolve().parent.parent.parent.parent / 'logs_omnidance' / 'exp'


def signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fixed_cases(exp_root):
    cases = []
    for condition, offset in (('joint', 0), ('aswap', 3)):
        csv_path = exp_root / 'E2' / f'full_{condition}.csv'
        with csv_path.open(newline='', encoding='utf-8') as stream:
            rows = list(itertools.islice(csv.DictReader(stream), 3))
        if len(rows) != 3:
            raise ValueError(f'{csv_path}: expected at least three rows')
        seen = set()
        for index, row in enumerate(rows):
            sample_id, audio_path = row['sample_id'], row['audio_path']
            if not sample_id or Path(sample_id).name != sample_id or sample_id in ('.', '..'):
                raise ValueError(f'{csv_path}: invalid sample_id {sample_id!r}')
            audio_stem = Path(audio_path).stem
            if not audio_path or not audio_stem or audio_stem in ('.', '..'):
                raise ValueError(f'{csv_path}: invalid audio_path {audio_path!r}')
            filename = f'{sample_id}_{audio_stem}.mp4'
            if filename in seen:
                raise ValueError(f'{csv_path}: duplicate fixed output {filename}')
            seen.add(filename)
            cases.append(dict(condition=condition, shard=index + offset,
                              sample_id=sample_id, audio_path=audio_path, filename=filename))
    return cases


def successful_log(log_dir, case, require_replacement=True, require_empty_prompt=False):
    """Never inspect a video until its shard has a complete, matching success log."""
    invalid = []
    for suffix in ('_retry_memory1', ''):
        path = log_dir / f"shard_{case['shard']}{suffix}.log"
        if not path.exists():
            continue
        before = signature(path)
        content = path.read_bytes()
        text = content.decode('utf-8', errors='replace')
        if not re.search(r'^\[h3ar\] done\s*$', text, flags=re.MULTILINE):
            continue
        sample_line = re.search(
            rf"^\[h3ar\] {re.escape(case['condition'])} {re.escape(case['sample_id'])} "
            r'\d+s(?:\s.*)?$', text, flags=re.MULTILINE,
        )
        calls = [re.search(
            rf'^\[music replacement\] encode call={call}\b[^\r\n]*'
            r'shape=\[2,\s*32,\s*203\]', text, flags=re.MULTILINE,
        ) for call in (1, 2)]
        tail = re.search(
            r'^\[h3ar\] repeat_clean audio tail: 4 rows fixed at effective time=1\s*$',
            text, flags=re.MULTILINE,
        )
        condition_record = None
        if require_empty_prompt:
            records = re.findall(r'^\[h3ar\] input_conditions (.+)$', text, flags=re.MULTILINE)
            if len(records) != 1:
                invalid.append(f'{path}: require exactly one empty-prompt input audit')
                continue
            condition_record = json.loads(records[0])
            if (condition_record.get('prompt_mode') != 'empty' or condition_record.get('semantic_text') != ' '
                    or condition_record.get('audio_path') != case['audio_path']
                    or condition_record.get('seed') != 42
                    or bool(condition_record.get('music_encoder_ckpt')) != require_replacement):
                invalid.append(f'{path}: empty-prompt input audit disagrees with fixed condition')
                continue
        if not sample_line or (require_replacement and not all(calls)) or not tail:
            invalid.append(f'{path}: done log lacks matching sample, both [2,32,203] calls, '
                           'or repeat_clean tail=4 time=1')
            continue
        done = re.search(r'^\[h3ar\] done\s*$', text, flags=re.MULTILINE)
        ordered = tail.start() < sample_line.start() < done.start()
        if require_replacement:
            ordered = ordered and calls[0].start() < calls[1].start() < tail.start()
        if not ordered:
            invalid.append(f'{path}: replacement/tail/sample/done evidence is out of order')
            continue
        if signature(path) != before:
            raise ValueError(f'{path}: log changed while being read')
        evidence = dict(encode_calls=[1, 2], encode_shapes=[[2, 32, 203], [2, 32, 203]],
                        repeat_clean_tail_rows=4, repeat_clean_effective_time=1,
                        matching_sample_done=True)
        if not require_replacement:
            evidence.update(encode_calls=None, encode_shapes=None, native_encoder=True)
        if condition_record is not None:
            evidence['input_conditions'] = condition_record
        return path, hashlib.sha256(content).hexdigest(), before, evidence
    if invalid:
        raise ValueError('; '.join(invalid))
    return None


def run_tool(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or result.stderr.strip():
        detail = result.stderr.strip()[-4000:] or f'exit code {result.returncode}'
        raise ValueError(f'{command[0]} failed: {detail}')
    return result.stdout


def probe_video(path):
    """Decode every video frame with decord and both selected streams with ffmpeg."""
    import decord

    try:
        reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
        frames, fps = len(reader), float(reader.get_avg_fps())
        if frames != 124 or not math.isfinite(fps) or abs(fps - 24) > 1e-6:
            raise ValueError(f'{path}: expected 124 frames at 24fps, got {frames} at {fps}')
        for index in range(frames):
            if tuple(reader[index].shape) != (1280, 704, 3):
                raise ValueError(f'{path}: frame {index} is not 704x1280 RGB')
        del reader
    except Exception as error:
        raise ValueError(f'{path}: full decord video decode failed: {error}') from error
    progress = run_tool([
        'ffmpeg', '-v', 'error', '-xerror', '-i', str(path),
        '-map', '0:v:0', '-map', '0:a:0', '-progress', 'pipe:1', '-f', 'null', '-',
    ])
    decoded_frames = re.findall(r'^frame=(\d+)\s*$', progress, flags=re.MULTILINE)
    if not decoded_frames or int(decoded_frames[-1]) != 124 or 'progress=end' not in progress:
        raise ValueError(f'{path}: ffmpeg did not finish decoding exactly 124 video frames')
    metadata = subprocess.run(['ffmpeg', '-hide_banner', '-i', str(path)],
                              capture_output=True, text=True, timeout=120)
    return dict(
        streams=[dict(codec_type='video', width=704, height=1280, fps=fps,
                      decoded_frames=frames, verified_by='decord full-frame decode'),
                 dict(codec_type='audio', verified_by='ffmpeg full selected-audio decode')],
        declared_stream_count_checked=False,
        stream_count_note='ffprobe unavailable: selected video/audio presence and decoding verified; '
                          'exact declared stream counts not independently checked.',
        ffmpeg_metadata_stderr=metadata.stderr,
        full_video_audio_decode=True, decode_method='decord + ffmpeg -xerror',
        ffmpeg_decoded_video_frames=int(decoded_frames[-1]),
    )


def validate(exp_root, tags=TAGS, native_tags=(), require_empty_prompt=False):
    expected = fixed_cases(exp_root)
    results = []
    for tag in tags:
        group_dir = exp_root / 'H3AR' / f'val_musicreplace_{tag}'
        for case in expected:
            video_path = group_dir / case['condition'] / case['filename']
            entry = dict(tag=tag, **case, video_path=str(video_path))
            try:
                log = successful_log(group_dir / 'logs', case,
                                     require_replacement=tag not in native_tags,
                                     require_empty_prompt=require_empty_prompt)
                if log is None:
                    entry.update(status='missing', reason='no_matching_success_log')
                else:
                    log_path, log_sha256, log_signature, log_evidence = log
                    entry.update(log_path=str(log_path), log_sha256=log_sha256,
                                 log_evidence=log_evidence)
                    if not video_path.is_file():
                        raise ValueError(f'{video_path}: completed shard has no expected video')
                    before = signature(video_path)
                    entry.update(probe_video(video_path))
                    entry['video_sha256'] = sha256(video_path)
                    if signature(video_path) != before or signature(log_path) != log_signature:
                        raise ValueError('video or successful log changed during validation')
                    entry.update(status='passed', video_size_bytes=before[2])
            except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError,
                    subprocess.SubprocessError) as error:
                entry.update(status='error', reason=str(error))
            results.append(entry)
    counts = {status: sum(row['status'] == status for row in results)
              for status in ('passed', 'missing', 'error')}
    total = len(tags) * len(expected)
    return dict(
        expected=total, expected48=48, passed=counts['passed'], missing=counts['missing'],
        errors=counts['error'], complete=counts['passed'] == total,
        full48_complete=counts['passed'] == 48 and total == 48,
        exp_root=str(exp_root), tags=list(tags), cases=results,
        scope='Generation log and audio/video artifact completeness only; '
              'NOT musical adherence, audio-condition sensitivity, dance quality, or human acceptance.',
    )


def atomic_write_new(out, result):
    # A hard-link publishes the finished JSON atomically without ever replacing an existing output.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=out.parent,
                                         prefix=f'.{out.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, out)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp_root', type=Path, default=DEFAULT_EXP_ROOT)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tags', nargs='+', default=list(TAGS),
                        help='Validate only these fixed groups; each group still uses the same six cases.')
    parser.add_argument('--native_tags', nargs='*', default=[], help='Explicit original-native groups: no music-replacement encode log required')
    parser.add_argument('--require_empty_prompt', action='store_true', help='Require actual empty semantic text and fixed music/seed evidence')
    parser.add_argument('--allow_partial', action='store_true',
                        help='Allow missing success logs, but never validation errors; complete remains false.')
    args = parser.parse_args(argv)
    exp_root, out = args.exp_root.absolute(), args.out.absolute()
    if not exp_root.is_dir():
        parser.error(f'experiment root does not exist: {exp_root}')
    if not out.parent.is_dir():
        parser.error(f'output parent directory does not exist: {out.parent}')
    if os.path.lexists(out):
        parser.error(f'refusing existing output: {out}')
    if len(set(args.tags)) != len(args.tags) or any(re.fullmatch(r'[A-Za-z0-9_-]+', tag) is None for tag in args.tags):
        parser.error('--tags must contain distinct bare configuration names')
    if shutil.which('ffmpeg') is None:
        parser.error('ffmpeg is not on PATH')
    try:
        import decord  # noqa: F401
    except ImportError:
        parser.error('decord is required; use /opt/venv_eval/bin/python')
    try:
        if len(set(args.native_tags)) != len(args.native_tags) or not set(args.native_tags) <= set(args.tags):
            parser.error('--native_tags must be unique members of --tags')
        result = validate(exp_root, args.tags, args.native_tags, args.require_empty_prompt)
        result['native_tags'] = args.native_tags
        result['required_empty_prompt'] = args.require_empty_prompt
        result['allow_partial'] = args.allow_partial
        atomic_write_new(out, result)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    summary = {key: result[key] for key in ('expected', 'expected48', 'passed', 'missing', 'errors', 'complete')}
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 0 if result['errors'] == 0 and (result['complete'] or args.allow_partial) else 1


if __name__ == '__main__':
    raise SystemExit(main())
