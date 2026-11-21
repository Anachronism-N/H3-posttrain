"""Resume the official, fixed-version NLF asset with verified byte ranges."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parent
URL = "https://github.com/isarandi/nlf/releases/download/v0.2.0/nlf_l_multi.torchscript"
SIZE = 495696900
ETAG = '"0x8DD1AD5CA003611"'
PARTS = 128
BOUNDS = [(i * SIZE // PARTS, (i + 1) * SIZE // PARTS - 1) for i in range(PARTS)]


def segment_path(start, stop):
    return ROOT / f"nlf-v020-{start}-{stop}.part"


def status():
    complete = [(a, b) for a, b in BOUNDS if segment_path(a, b).exists()
                and segment_path(a, b).stat().st_size == b - a + 1]
    report = {"schema": "nlf-download-progress-v1", "source": URL, "size_bytes": SIZE,
              "complete_segments": len(complete), "expected_segments": PARTS,
              "complete": len(complete) == PARTS, "verified_remote_etag": ETAG,
              "cached_bytes": sum(segment_path(a, b).stat().st_size for a, b in BOUNDS if segment_path(a, b).exists())}
    (ROOT / "nlf_download_status.json").write_text(json.dumps(report, indent=2) + "\n")
    return complete, report


def main():
    response = requests.head(URL, allow_redirects=True, timeout=(30, 30))
    response.raise_for_status()
    if int(response.headers.get("Content-Length", -1)) != SIZE or response.headers.get("ETag") != ETAG:
        raise ValueError("Official asset identity changed; cached ranges cannot be mixed")
    asset_url = response.url
    complete, _ = status()
    pending = [bound for bound in BOUNDS if bound not in complete]

    def download(bound):
        a, b = bound
        path = segment_path(a, b)
        offset = path.stat().st_size if path.exists() else 0
        if offset > b - a + 1:
            raise ValueError("Oversized cached range")
        started = time.monotonic()
        try:
            with requests.get(asset_url, headers={"Range": f"bytes={a + offset}-{b}", "If-Range": ETAG},
                              stream=True, timeout=(30, 30)) as stream:
                stream.raise_for_status()
                if (stream.status_code != 206 or stream.headers.get("Content-Range") != f"bytes {a + offset}-{b}/{SIZE}"
                        or stream.headers.get("ETag") != ETAG):
                    raise ValueError("Range identity mismatch")
                with path.open("ab") as output:
                    for block in stream.iter_content(65536):
                        output.write(block)
                        if time.monotonic() - started > 300:
                            raise TimeoutError("Bounded range transfer deadline reached")
            if path.stat().st_size != b - a + 1:
                raise ValueError("Incomplete range")
            return {"start": a, "complete": True}
        except (requests.RequestException, ValueError, TimeoutError) as error:
            return {"start": a, "complete": False, "error_type": type(error).__name__}

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        for future in concurrent.futures.as_completed([pool.submit(download, bound) for bound in pending]):
            print(json.dumps(future.result()), flush=True)
            status()
    complete, report = status()
    print(json.dumps(report), flush=True)
    if len(complete) != PARTS:
        return
    temporary = ROOT / "nlf_l_multi.v0.2.0.download"
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for a, b in BOUNDS:
            with segment_path(a, b).open("rb") as source:
                for block in iter(lambda: source.read(1048576), b""):
                    output.write(block)
                    digest.update(block)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("TorchScript archive CRC failure")
    destination = ROOT / "nlf_l_multi.v0.2.0.torchscript"
    temporary.replace(destination)
    metadata = {"schema": "nlf-official-asset-v1", "source": URL,
                "official_demo_alias": "https://bit.ly/nlf_l_pt", "release": "v0.2.0",
                "source_commit": "29046738a8101c15dc366ab45945b45e40debf17",
                "size_bytes": SIZE, "sha256": digest.hexdigest(), "remote_etag": ETAG,
                "zip_crc_verified": True, "publisher_sha256_available": False,
                "license_notice": "Official README: noncommercial research use",
                "torchscript_loaded": False, "model_api_verified": False}
    (ROOT / "nlf_asset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Complete official model archive verified", flush=True)


if __name__ == "__main__":
    main()
