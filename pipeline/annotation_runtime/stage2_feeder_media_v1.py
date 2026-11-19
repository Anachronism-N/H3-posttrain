"""Original single-copy/native12-frame media preparation, used only by CPU feeder."""
import hashlib,math,os,stat,subprocess
from pathlib import Path
def need(condition, message):
    if not condition:
        raise ValueError(message)
def select_frames(manifest):
    n = manifest["frame_count"]
    indices = sorted({round(i * (n - 1) / (min(12, n) - 1)) for i in range(min(12, n))})
    return [dict(frame_index=i, time=manifest["times"][i], source_pts=manifest["source_timestamps"][i]) for i in indices]
def stage_images(row, manifest, local, ffmpeg_path):
    src = Path(row["video_path"])
    before = src.stat()
    need(stat.S_ISREG(before.st_mode) and before.st_size <= 2**30, "Source video exceeds 1 GiB single-copy bound")
    dst = local / (row["id"] + src.suffix)
    digest = hashlib.sha256()
    total = 0
    with src.open("rb") as incoming, dst.open("xb") as output:
        for block in iter(lambda: incoming.read(8 * 2**20), b""):
            total += len(block)
            need(total <= 2**30, "Source grew beyond copy bound")
            output.write(block)
            digest.update(block)
    after = src.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns) and
         total == before.st_size and digest.hexdigest() == manifest["video_sha256"], "Source changed or single-copy video SHA differs")
    frames = select_frames(manifest)
    expression = "+".join("eq(n\,%d)" % x["frame_index"] for x in frames)
    scale = min(1., math.sqrt(200704 / (manifest["width"] * manifest["height"])))
    width, height = max(1, int(manifest["width"] * scale)), max(1, int(manifest["height"] * scale))
    env = os.environ.copy()
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = ":".join(x for x in env["LD_LIBRARY_PATH"].split(":") if x)
    subprocess.run([ffmpeg_path, "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1", "-noautorotate",
        "-i", str(dst), "-an", "-sn", "-filter_threads", "1", "-vf", "select=" + expression + ",scale=%d:%d" % (width, height),
        "-vsync", "0", "-threads", "1", "-q:v", "3", str(local / "frame-%03d.jpg")],
        check=True, cwd=local, env=env, capture_output=True)
    images = sorted(local.glob("frame-*.jpg"))
    need(len(images) == len(frames) <= 16, "Actual JPEG count differs from selected native frames")
    for p in images:
        raw = p.read_bytes()
        need(raw.startswith(b"\xff\xd8") and raw.endswith(b"\xff\xd9"), "Incomplete JPEG")
    return frames, images
