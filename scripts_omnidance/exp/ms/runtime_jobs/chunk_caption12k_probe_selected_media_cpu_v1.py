import hashlib
import json
import shutil
import socket
import wave
from pathlib import Path

import av

OUT = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/chunk_caption12k_input_v1')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    rows = [json.loads(line) for line in (OUT / 'manifest.jsonl').read_text().splitlines()]
    fixed = [json.loads(line) for line in (OUT / 'fixed3_manifest.jsonl').read_text().splitlines()]
    chosen = rows[11992:] + rows[:3] + fixed
    results = []
    for row in chosen:
        with av.open(row['video_path']) as container:
            stream = container.streams.video[0]
            frames = 0
            timestamps = []
            shapes = set()
            for frame in container.decode(stream):
                frames += 1
                shapes.add((frame.width, frame.height))
                if frame.pts is not None:
                    timestamps.append(float(frame.pts * frame.time_base))
            fps = float(stream.average_rate)
            interval = 1.0 / fps
            last_frame_interval = timestamps[-1] - timestamps[-2] if len(timestamps) > 1 else interval
            observed_end = timestamps[-1] + last_frame_interval
            video = dict(decoded_frames=frames, average_fps=fps, decoded_shapes=[list(shape) for shape in sorted(shapes)], first_frame_seconds=timestamps[0], last_frame_seconds=timestamps[-1], observed_end_seconds=observed_end, stream_duration_seconds=float(stream.duration * stream.time_base) if stream.duration is not None else 'unknown', time_base=str(stream.time_base), frame_timestamps_count=len(timestamps), constant_16fps_timestamps=all(abs((b - a) - 1.0 / 16) < 1e-7 for a, b in zip(timestamps, timestamps[1:])))
        with wave.open(row['audio_path'], 'rb') as audio:
            info = dict(sample_rate=audio.getframerate(), frames=audio.getnframes(), channels=audio.getnchannels(), sample_width_bytes=audio.getsampwidth(), duration_seconds=audio.getnframes() / audio.getframerate())
        checks = dict(decoded_81_frames=frames == 81, actual_16fps=abs(fps - 16) < 1e-7 and video['constant_16fps_timestamps'], actual_704x1280=shapes == {(704, 1280)}, actual_zero_start=abs(timestamps[0]) < 1e-7, actual_5_0625_end=abs(observed_end - 5.0625) < 1e-7, natural_audio_covers_window=info['duration_seconds'] >= observed_end - 1e-7)
        results.append(dict(id=row['id'], dataset_role=row['dataset_role'], video_path=row['video_path'], audio_path=row['audio_path'], source_group=row['source_group'], video=video, audio=info, checks=checks, passed=all(checks.values()), video_sha256=digest(row['video_path']), audio_sha256=digest(row['audio_path']), normalized_audio_offset_seconds='unknown', original_source_window_mapping='unknown'))
    report = dict(schema='chunk-caption12k-selected-media-probe-cpu-v1', host=socket.gethostname(), ffprobe_available=shutil.which('ffprobe') is not None, decoder='PyAV full CPU decode; ffprobe unavailable on N1', av_version=av.__version__, gpu_used=False, full_12000_media_scan=False, selected_clips=len(results), extra8_count=8, authority_smoke_count=3, fixed3_count=3, passed=all(row['passed'] for row in results), records=results)
    with (OUT / 'selected_media_probe_cpu.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(passed=report['passed'], selected_clips=report['selected_clips'], decoder=report['decoder'], records=[dict(id=row['id'], passed=row['passed'], checks=row['checks'], decoded_frames=row['video']['decoded_frames'], observed_end_seconds=row['video']['observed_end_seconds'], audio_duration_seconds=row['audio']['duration_seconds']) for row in results]), ensure_ascii=False))
    assert report['passed'], 'Selected-media protocol mismatch; do not admit failed clips'


if __name__ == '__main__':
    main()
