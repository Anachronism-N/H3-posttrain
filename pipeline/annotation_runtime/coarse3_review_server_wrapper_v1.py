#!/usr/bin/env python3
"""Preserve the frozen review server; add one SHA-pinned, read-only HTML route."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT.parents[4] / 'H3-posttrain/scripts_omnidance/exp/review_web/server.py'
SERVER_SHA256 = '7326ed7bc63210ba32de9bc316811dbef60562c40c2ce67e10ab017f560c5475'
PAGE = ROOT / 'coarse3_delivery_derived_v1/snapshots/a32c05e6d0c036dbacc7419c20d1e6dfaf0751e7b90881f9f08830006061d40b/review.html'
PAGE_SHA256 = '431f750508baf19dd47d8de47fce06c4d1cb3cd5718e5a00d0b5b7ff574438f5'


def verified_bytes(path, expected):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('Frozen content SHA mismatch: ' + str(path))
    return content


def load_server():
    content = verified_bytes(SERVER, SERVER_SHA256)
    spec = importlib.util.spec_from_file_location('coarse3_frozen_review_server', SERVER)
    module = importlib.util.module_from_spec(spec)
    exec(compile(content, str(SERVER), 'exec'), module.__dict__)
    return module


legacy = load_server()


class H(legacy.H):
    def do_GET(self):
        path = urlparse(self.path).path.rstrip('/').split('/')[-1]
        if path != 'coarse3-review':
            return super().do_GET()
        try:
            body = verified_bytes(PAGE, PAGE_SHA256)
        except (OSError, ValueError):
            return self._json({'error': 'Pinned coarse3 review page unavailable or changed'}, 503)
        try:
            self._send(200, body, 'text/html; charset=utf-8', {
                'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff',
                'Content-Security-Policy': "default-src 'none'; img-src 'self'; media-src 'self'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'"})
        except (BrokenPipeError, ConnectionResetError):
            pass


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--manifest', help='Original existing focused manifest; legacy scoring/media are unchanged')
    return ap


def configure(args, ap):
    verified_bytes(PAGE, PAGE_SHA256)
    if args.manifest:
        if args.rebuild or not Path(args.manifest).is_file():
            ap.error('--manifest requires an existing file and cannot combine with --rebuild')
        legacy.MANIFEST = str(Path(args.manifest).absolute())
        legacy.FOCUSED_MANIFEST = True


def main():
    ap = parser()
    args = ap.parse_args()
    configure(args, ap)
    legacy.manifest_gz(rebuild=args.rebuild)
    srv = legacy.ThreadingHTTPServer((args.host, args.port), H)
    srv.daemon_threads = True
    print(f'[serve] http://{args.host}:{args.port}/ (frozen server + fixed coarse3-review)', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
