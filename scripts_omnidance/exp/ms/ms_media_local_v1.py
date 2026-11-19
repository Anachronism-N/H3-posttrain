"""Native media subprocesses use a local cwd and no empty library search items."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def _media_env():
    env = os.environ.copy()
    if "LD_LIBRARY_PATH" in env:
        items = [item for item in env["LD_LIBRARY_PATH"].split(":") if item]
        if items:
            env["LD_LIBRARY_PATH"] = ":".join(items)
        else:
            del env["LD_LIBRARY_PATH"]
    return env


def probe_video(path, executable="ffprobe"):
    parent = Path(path).parent
    if not parent.is_absolute():
        raise ValueError("Absolute local media path required")
    command = [str(executable), "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames", "-show_entries", "stream=width,height,duration,nb_frames,time_base:frame=best_effort_timestamp,best_effort_timestamp_time,pkt_duration_time", "-of", "json", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=parent, env=_media_env())
    return json.loads(result.stdout)


def decode_bgr(path, width, height, executable="ffmpeg"):
    parent = Path(path).parent
    if not parent.is_absolute():
        raise ValueError("Absolute local media path required")
    command = [str(executable), "-v", "error", "-threads", "1", "-noautorotate", "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn", "-vsync", "0", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, cwd=parent, env=_media_env())
        try:
            size = width * height * 3
            while True:
                raw = process.stdout.read(size)
                if not raw:
                    break
                if len(raw) != size:
                    raise ValueError("Truncated native raw BGR frame")
                yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError(errors.read(4096).decode(errors="replace"))
        finally:
            process.stdout.close()
            process.wait()  # Closing pipe ends unfinished decoder; no process signals.

