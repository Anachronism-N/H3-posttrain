"""Lossless fixed-window views of canonical mono PCM16 songs.

Observation and schedule ownership are different intervals. Neither is an
estimate of a real movement's onset or offset.
"""
import hashlib
import io
import math
import os
from pathlib import Path
import tempfile
import wave


WINDOW_SECONDS = 5.0625


def inspect_audio(path):
    """Read a complete canonical WAV; never resample, pad or downmix it."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        with wave.open(io.BytesIO(raw), "rb") as handle:
            if (handle.getnchannels() != 1 or handle.getsampwidth() != 2
                    or handle.getcomptype() != "NONE"):
                raise ValueError("Require uncompressed mono PCM16 WAV; use prepare-audio explicitly")
            rate, count = handle.getframerate(), handle.getnframes()
            pcm = handle.readframes(count)
    except (wave.Error, EOFError) as error:
        raise ValueError("Require readable canonical PCM16 WAV") from error
    if rate <= 0 or count <= 0 or len(pcm) != count * 2:
        raise ValueError("Truncated or empty PCM audio")
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                sample_rate=rate, num_samples=count, pcm=pcm)


def make_windows(num_samples, sample_rate, window_seconds=WINDOW_SECONDS, stride_seconds=None):
    """Return full observation windows and an exact nonoverlapping partition.

    The final full window shifts left. Ownership ends at the next observation
    start, so overlapping observations do not duplicate the output schedule.
    A nonintegral nominal sample count is rounded once, with its error recorded.
    """
    if type(num_samples) is not int or num_samples <= 0 or type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("Positive integer sample count/rate required")
    if type(window_seconds) not in (int, float) or not math.isfinite(window_seconds) or window_seconds != WINDOW_SECONDS:
        raise ValueError("Local native contract requires window_seconds=5.0625")
    stride_seconds = window_seconds if stride_seconds is None else stride_seconds
    if (type(stride_seconds) not in (int, float) or not math.isfinite(stride_seconds)
            or not 0 < stride_seconds <= window_seconds):
        raise ValueError("stride_seconds must be finite, positive and <= window_seconds")
    length = int(math.floor(window_seconds * sample_rate + .5))
    stride = int(math.floor(stride_seconds * sample_rate + .5))
    if length < 2 or stride < 1:
        raise ValueError("Sample clock cannot represent the requested window/stride")
    if num_samples < length:
        raise ValueError("Song shorter than one full 5.0625s window; no padding or short-window fallback")
    starts = list(range(0, num_samples - length + 1, stride))
    tail = num_samples - length
    if starts[-1] != tail:
        starts.append(tail)
    return [dict(index=i, observation_start=start, observation_end=start + length,
                 ownership_start=start,
                 ownership_end=starts[i + 1] if i + 1 < len(starts) else num_samples,
                 nominal_window_seconds=window_seconds,
                 sample_rounding_error=length - window_seconds * sample_rate,
                 boundaries_are_action_boundaries=False)
            for i, start in enumerate(starts)]


def write_window(audio, window, destination):
    """Publish a byte-identical PCM slice; verify rather than overwrite on resume."""
    start, end = window["observation_start"], window["observation_end"]
    if (type(start) is not int or type(end) is not int
            or not 0 <= start < end <= audio["num_samples"]):
        raise ValueError("Invalid observation interval")
    memory = io.BytesIO()
    with wave.open(memory, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(audio["sample_rate"])
        handle.writeframes(audio["pcm"][2 * start:2 * end])
    raw = memory.getvalue()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != raw:
            raise ValueError("Existing derived audio bytes changed: " + str(destination))
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(str(temporary), str(destination))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return dict(path=str(destination.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                sample_rate=audio["sample_rate"], num_samples=end - start,
                parent_audio_sha256=audio["sha256"], parent_start=start, parent_end=end,
                conversion="lossless PCM16 sample slice; no padding, resampling or mixing")
