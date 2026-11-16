"""Retrieve one hash-verified public source video without downloading its tar."""
import hashlib
import json
import re
import sys
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TARGET = sys.argv[1] if len(sys.argv) > 1 else "qaeUjkBWdUaHwyAi"
ASSET = Path(__file__).resolve().parent
AUDIT = json.loads((ASSET / (TARGET + ".audit.json")).read_text(encoding="utf-8"))
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(total=1, backoff_factor=.2)))
BASE = "https://huggingface.co/datasets/avagelion/dance-data/resolve/main"
CALLS, TOTALS = [], {}
DOWNLOADED = 0


def get_range(group, part, start, count):
    global DOWNLOADED
    assert 0 <= start and 0 < count <= 4_200_000
    if DOWNLOADED + count > 32_000_000:
        raise ValueError("Bounded range-download budget reached")
    url = f"{BASE}/{group}/part_{part:04d}.tar?download=true&range_start={start}"
    response = SESSION.get(url, headers={"Range": f"bytes={start}-{start+count-1}"},
                           stream=True, timeout=(10, 25))
    if response.status_code != 206:
        response.close()
        raise ValueError(f"HTTP {response.status_code}; range required")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
    if not match or int(match[1]) != start:
        response.close()
        raise ValueError("Content-Range did not match requested start")
    raw = response.raw.read(count + 1)
    response.close()
    assert len(raw) <= count
    DOWNLOADED += len(raw)
    TOTALS[group, part] = int(match[3])
    CALLS.append(dict(group=group, part=part, start=start, returned_bytes=len(raw)))
    return raw


def parse(header, pos):
    if len(header) != 512 or header[257:262] != b"ustar":
        return None
    try:
        stored = int(header[148:156].strip(b"\0 ") or b"0", 8)
        if stored != sum(header[:148]) + 8 * 32 + sum(header[156:]):
            return None
        size = int(header[124:136].strip(b"\0 ") or b"0", 8)
        name = header[:100].split(b"\0", 1)[0].decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    return dict(name=name, size=size, type=header[156:157].decode(),
                header_offset=pos, data_offset=pos+512)


def main():
    lo, hi = 0, 53
    while len(sys.argv) < 3 and lo < hi:
        mid = (lo + hi) // 2
        raw = get_range("audio_tars", mid, 0, 2048)
        members = [parse(raw[i:i+512], i) for i in range(0, len(raw)-511, 512)]
        member = next(m for m in members if m and m["name"].endswith(".wav"))
        print("shard_probe", mid, member["name"], flush=True)
        if member["name"] <= TARGET + ".wav":
            lo = mid + 1
        else:
            hi = mid
    part = int(sys.argv[2]) if len(sys.argv) > 2 else lo - 1
    assert 0 <= part < 53
    print("audio_shard", part, flush=True)
    lo, hi = 0, 5000
    audio_member = None
    if len(sys.argv) > 3:
        hint = int(sys.argv[3])
        offset = hint * 244736 + 1024
        candidate = parse(get_range("audio_tars", part, offset, 512), offset)
        if candidate and candidate["name"] == TARGET + ".wav":
            audio_member, ordinal = candidate, hint
            lo = hi
    while lo < hi:
        mid = (lo + hi) // 2
        offset = mid * 244736 + 1024
        member = parse(get_range("audio_tars", part, offset, 512), offset)
        if not member or not member["name"].endswith(".wav"):
            raise ValueError("Audio constant-stride assumption failed")
        if member["name"] == TARGET + ".wav":
            audio_member, ordinal = member, mid
            break
        if member["name"] < TARGET + ".wav":
            lo = mid + 1
        else:
            hi = mid
    if audio_member is None:
        raise ValueError("Audio ID not found in sorted shard")
    print("audio_ordinal", ordinal, flush=True)
    audio_file = ASSET / (TARGET + ".wav")
    if audio_file.exists():
        assert hashlib.sha256(audio_file.read_bytes()).hexdigest() == AUDIT["audio"]["sha256"]
    else:
        audio_bytes = get_range("audio_tars", part, audio_member["data_offset"], audio_member["size"])
        assert hashlib.sha256(audio_bytes).hexdigest() == AUDIT["audio"]["sha256"]
        audio_file.write_bytes(audio_bytes)
    get_range("videos_tars", part, 0, 2048)
    tar_size = TOTALS["videos_tars", part]
    estimate = int((tar_size - 10240) / 5000 * ordinal)
    start = max(0, ((estimate - 2_000_000) // 512) * 512)
    raw = get_range("videos_tars", part, start, min(4_200_000, tar_size-start))
    headers = []
    for off in range(0, len(raw)-511, 512):
        member = parse(raw[off:off+512], start+off)
        if member and member["name"].endswith(".mp4"):
            headers.append(member)
    print("video_near_estimate", [(m["name"], m["header_offset"]) for m in headers], flush=True)
    found = next((m for m in headers if m["name"] == TARGET + ".mp4"), None)
    lower, upper = 0, tar_size
    for member in headers:
        if member["name"] < TARGET + ".mp4":
            lower = max(lower, member["data_offset"] + ((member["size"]+511)//512)*512)
        elif member["name"] > TARGET + ".mp4":
            upper = min(upper, member["header_offset"])
    while found is None and lower < upper:
        probe = ((lower+upper)//2//512)*512
        block = get_range("videos_tars", part, probe, min(1_800_000, tar_size-probe))
        nearby = []
        for off in range(0, len(block)-511, 512):
            member = parse(block[off:off+512], probe+off)
            if member and member["name"].endswith(".mp4"):
                nearby.append(member)
        if not nearby:
            raise ValueError("Video entry exceeds bounded header-search span")
        for member in nearby:
            if member["name"] == TARGET + ".mp4":
                found = member
                break
            if member["name"] < TARGET + ".mp4":
                lower = max(lower, member["data_offset"] + ((member["size"]+511)//512)*512)
            else:
                upper = min(upper, member["header_offset"])
        print("video_binary", [(m["name"], m["header_offset"]) for m in nearby], flush=True)
    if not found:
        raise ValueError("Target video not found")
    video = get_range("videos_tars", part, found["data_offset"], found["size"])
    digest = hashlib.sha256(video).hexdigest()
    assert digest == AUDIT["video"]["sha256"], "Video differs from frozen original"
    outfile = ASSET / (TARGET + ".mp4")
    outfile.write_bytes(video)
    provenance = dict(schema="herewechoreography.tar_member_range_extract.v1", case_id=TARGET,
        source_dataset="avagelion/dance-data", known_revision=AUDIT["hf_metadata_revision"],
        range_request_revision="main", video_sha256_matches_frozen_source=True,
        video_sha256=digest, video_file=outfile.name,
        audio_file=audio_file.name, audio_sha256=hashlib.sha256(audio_file.read_bytes()).hexdigest(),
        video_tar=f"videos_tars/part_{part:04d}.tar", audio_tar=f"audio_tars/part_{part:04d}.tar",
        ordinal_zero_based=ordinal, member=found, source_tar_size_bytes=tar_size,
        downloaded_bytes_total=DOWNLOADED, range_requests=CALLS,
        whole_tar_downloaded=False, video_frames_or_quality_judged=False)
    (ASSET / (TARGET + ".video_range_audit.json")).write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(saved_video=str(outfile), sha256=digest, member=found,
                         shard=part, ordinal=ordinal, downloaded_bytes_total=DOWNLOADED,
                         request_count=len(CALLS)), indent=2), flush=True)


if __name__ == "__main__":
    main()
