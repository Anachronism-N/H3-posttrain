"""Read-only public review server with allowlisted routes and audio ranges."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import stat
from urllib.parse import unquote, urlsplit

from .io import file_sha, object_sha, read_json
from .reviews import validate_review_pack


def _range(header, size):
    if header is None:
        return 0, size - 1, 200
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if match is None or not any(match.groups()):
        raise ValueError("Invalid range")
    first, last = match.groups()
    if not first:
        count = int(last)
        if count < 1:
            raise ValueError("Empty suffix")
        return max(0, size - count), size - 1, 206
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or start > end:
        raise ValueError("Unsatisfiable range")
    return start, end, 206


def _signature(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def create_review_server(public_dir, host="127.0.0.1", port=0, proxy_prefix=""):
    root = Path(public_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("An existing public review directory is required")
    if proxy_prefix and not re.fullmatch(r"/proxy/[0-9]+", proxy_prefix):
        raise ValueError("Explicit proxy prefix must be /proxy/PORT")
    pack = validate_review_pack(read_json(root / "pack.json"))
    pages, assets = {}, {}
    for name in ("index.html", "pack.json", "reviews.template.json", "audio_attribution.json"):
        path = root / name
        if not path.is_file() and name == "audio_attribution.json":
            continue
        if path.resolve(strict=True) != path or not path.is_file():
            raise ValueError("Public route must be a regular file inside the public root")
        pages["/" + name] = (path.read_bytes(), mimetypes.guess_type(name)[0] or "application/octet-stream")
    for item in pack["items"]:
        audio = item["song"]["audio"]
        relative = PurePosixPath(audio["path"])
        if (relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != "assets"
                or any(part in (".", "..") for part in relative.parts) or "\\" in audio["path"]):
            raise ValueError("Public audio must be an explicit assets/FILENAME route")
        path = root.joinpath(*relative.parts)
        if path.resolve(strict=True) != path or not path.is_file():
            raise ValueError("Public assets must not escape through symlinks")
        before = path.stat()
        if file_sha(path) != audio["sha256"] or _signature(before) != _signature(path.stat()):
            raise ValueError("Public audio identity differs from the review pack")
        assets["/" + relative.as_posix()] = (path, _signature(before),
            mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    identity = dict(schema="choreobench.review_server.health.v1", public_pack_sha256=object_sha(pack),
                    review_items=len(pack["items"]), songs=len({i["song"]["song_id"] for i in pack["items"]}),
                    read_only=True, private_routes=False)
    pages["/health"] = (json.dumps(identity).encode("utf-8"), "application/json")
    pages["/"] = pages["/index.html"]

    class Handler(BaseHTTPRequestHandler):
        server_version = "ChoreoBenchReview/1"
        sys_version = ""

        def _headers(self, status, size, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "same-origin")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self'; object-src 'none'; base-uri 'self'")

        def _error(self, status):
            payload = b"Request not available"
            self._headers(status, len(payload), "text/plain; charset=utf-8")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def _serve(self):
            self.connection.settimeout(30)
            route = unquote(urlsplit(self.path).path)
            if proxy_prefix and (route == proxy_prefix or route.startswith(proxy_prefix + "/")):
                route = route[len(proxy_prefix):] or "/"
            if route in pages:
                payload, content_type = pages[route]
                self._headers(200, len(payload), content_type)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(payload)
                return
            asset = assets.get(route)
            if asset is None:
                self._error(404)
                return
            path, signature, content_type = asset
            try:
                if path.resolve(strict=True) != path:
                    self._error(404)
                    return
                with path.open("rb") as stream:
                    observed = os.fstat(stream.fileno())
                    if not stat.S_ISREG(observed.st_mode) or _signature(observed) != signature:
                        self._error(409)
                        return
                    try:
                        start, end, status = _range(self.headers.get("Range"), observed.st_size)
                    except ValueError:
                        self._headers(416, 0, content_type)
                        self.send_header("Content-Range", "bytes */" + str(observed.st_size))
                        self.end_headers()
                        return
                    self._headers(status, end - start + 1, content_type)
                    self.send_header("Accept-Ranges", "bytes")
                    if status == 206:
                        self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, observed.st_size))
                    self.end_headers()
                    if self.command == "HEAD":
                        return
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = stream.read(min(remaining, 256 * 1024))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            except OSError:
                self._error(404)

        def do_GET(self):
            self._serve()

        def do_HEAD(self):
            self._serve()

        def do_POST(self):
            self._error(405)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.review_identity = identity
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--proxy-prefix", default="")
    args = parser.parse_args()
    server = create_review_server(args.public_dir, args.host, args.port, args.proxy_prefix)
    print(json.dumps(dict(listening=server.server_address, **server.review_identity)), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
