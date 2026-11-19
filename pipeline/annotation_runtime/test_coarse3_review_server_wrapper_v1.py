#!/usr/bin/env python3
"""Pure CPU handler tests; no server socket, signal, GPU or real scoring write."""
import argparse
from email.message import Message
import hashlib
from html.parser import HTMLParser
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import mock_open, patch
from urllib.parse import urlencode

PATH = Path(__file__).with_name('coarse3_review_server_wrapper_v1.py')
spec = importlib.util.spec_from_file_location('coarse3_wrapper_tested', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Capture(m.H):
    def __init__(self, path, method='GET', body=b'', headers=None):
        self.path, self.command = path, method
        self.rfile, self.wfile = io.BytesIO(body), io.BytesIO()
        self.headers = Message()
        for name, value in (headers or {}).items():
            self.headers[name] = value
        self.sent_headers = {}
        self.code = None

    def send_response(self, code):
        self.code = code

    def send_header(self, name, value):
        self.sent_headers[name] = str(value)

    def end_headers(self):
        pass


class Tests(unittest.TestCase):
    def test_frozen_server_identity_and_inherited_interfaces(self):
        self.assertEqual(hashlib.sha256(m.SERVER.read_bytes()).hexdigest(), m.SERVER_SHA256)
        for name in ('do_HEAD', 'do_POST', '_media', '_json', '_send'):
            self.assertIs(getattr(m.H, name), getattr(m.legacy.H, name))
        self.assertEqual(m.legacy.ALLOWED, [m.legacy.os.path.realpath(m.legacy.EXP), m.legacy.os.path.realpath(m.legacy.DATA)])
        self.assertEqual(m.legacy.REVIEWS, m.legacy.EXP + '/review_web/reviews')

    def test_exact_page_native_and_proxy_routes(self):
        body = m.verified_bytes(m.PAGE, m.PAGE_SHA256)
        for path in ('/coarse3-review', '/coarse3-review/', '/proxy/8775/coarse3-review', '/proxy/8775/coarse3-review?path=/etc/passwd'):
            handler = Capture(path)
            handler.do_GET()
            self.assertEqual(handler.code, 200)
            self.assertEqual(handler.wfile.getvalue(), body)
            self.assertEqual(handler.sent_headers['Content-Type'], 'text/html; charset=utf-8')
            self.assertEqual(handler.sent_headers['X-Content-Type-Options'], 'nosniff')
            self.assertIn("default-src 'none'", handler.sent_headers['Content-Security-Policy'])

    def test_head_fixed_route_no_body(self):
        handler = Capture('/proxy/8775/coarse3-review', 'HEAD')
        handler.do_HEAD()
        self.assertEqual(handler.code, 200)
        self.assertEqual(handler.wfile.getvalue(), b'')
        self.assertEqual(int(handler.sent_headers['Content-Length']), len(m.PAGE.read_bytes()))

    def test_page_sha_failure_closed(self):
        for failure in (ValueError('SHA mismatch'), OSError('unavailable')):
            handler = Capture('/coarse3-review')
            with patch.object(m, 'verified_bytes', side_effect=failure):
                handler.do_GET()
            self.assertEqual(handler.code, 503)
            self.assertNotIn(b'<html', handler.wfile.getvalue())

    def test_other_get_routes_forwarded_unchanged(self):
        for path in ('/', '/index.html', '/api/manifest', '/proxy/8775/api/manifest', '/fusion-review?sid=x',
                     '/data-review', '/api/reviews', '/oos', '/media?p=x', '/review.html', '/coarse3-review/other',
                     '/coarse3-review.html', '/coarse3-review%2f..%2findex.html'):
            handler = Capture(path)
            with patch.object(m.legacy.H, 'do_GET', autospec=True) as original:
                handler.do_GET()
                original.assert_called_once_with(handler)
            self.assertEqual(handler.path, path)

    def test_original_homepage_preserved(self):
        handler = Capture('/')
        handler.do_GET()
        self.assertEqual(handler.code, 200)
        self.assertEqual(handler.wfile.getvalue(), (Path(m.legacy.HERE) / 'index.html').read_bytes())

    def test_focused_manifest_configuration_and_rebuild_rejection(self):
        manifest = Path(m.legacy.EXP) / 'review_web/manifest_h3_ms_style_gt_eval3_v1.json'
        ap = m.parser()
        with patch.object(m.legacy, 'MANIFEST', m.legacy.MANIFEST), patch.object(m.legacy, 'FOCUSED_MANIFEST', False):
            args = ap.parse_args(['--host', '0.0.0.0', '--port', '8775', '--manifest', str(manifest)])
            m.configure(args, ap)
            self.assertEqual(args.port, 8775)
            self.assertEqual(m.legacy.MANIFEST, str(manifest.absolute()))
            self.assertTrue(m.legacy.FOCUSED_MANIFEST)
            handler = Capture('/api/manifest?rebuild=1')
            handler.do_GET()
            self.assertEqual(handler.code, 400)
            with patch.object(m.legacy, 'manifest_gz', return_value=b'original-gzip') as gzip:
                handler = Capture('/api/manifest')
                handler.do_GET()
                gzip.assert_called_once_with(rebuild=False)
                self.assertEqual(handler.code, 200)
                self.assertEqual(handler.sent_headers['Content-Encoding'], 'gzip')
                self.assertEqual(handler.wfile.getvalue(), b'original-gzip')
            with self.assertRaises(SystemExit), patch('sys.stderr', io.StringIO()):
                m.configure(ap.parse_args(['--manifest', str(manifest), '--rebuild']), ap)

    def test_original_media_range_head_and_allowlist(self):
        records = json.loads(m.PAGE.with_name('review.records.json').read_bytes())
        evidence = json.loads(Path(records[0]['evidence_ref']['path']).read_bytes())
        image = Path(records[0]['review_receipt']['path']).parent.parent / evidence['frames'][0]['file']
        url = '/proxy/8775/media?' + urlencode({'p': str(image)})
        handler = Capture(url, headers={'Range': 'bytes=0-15'})
        handler.do_GET()
        self.assertEqual(handler.code, 206)
        self.assertEqual(handler.wfile.getvalue(), image.read_bytes()[:16])
        self.assertTrue(handler.sent_headers['Content-Range'].startswith('bytes 0-15/'))
        handler = Capture(url, 'HEAD')
        handler.do_HEAD()
        self.assertEqual(handler.code, 200)
        self.assertEqual(handler.wfile.getvalue(), b'')
        self.assertEqual(handler.sent_headers['Content-Length'], str(image.stat().st_size))
        handler = Capture('/media?' + urlencode({'p': '/etc/passwd'}))
        handler.do_GET()
        self.assertEqual(handler.code, 404)
        handler = Capture(url, headers={'Range': 'bytes=' + str(image.stat().st_size + 1) + '-'})
        handler.do_GET()
        self.assertEqual(handler.code, 416)
        self.assertEqual(handler.wfile.getvalue(), b'')

    def test_scoring_original_storage_no_real_write(self):
        record = {'key': 'original-score-key', 'reviewer': 'reviewer/invalid', 'sync': 4}
        content = json.dumps(record).encode()
        handler = Capture('/api/review', 'POST', content, {'Content-Length': str(len(content))})
        output = mock_open()
        with patch('builtins.open', output), patch.object(m.legacy.os, 'makedirs') as mkdir:
            handler.do_POST()
        self.assertEqual(handler.code, 200)
        mkdir.assert_called_once_with(m.legacy.REVIEWS, exist_ok=True)
        args = output.call_args.args
        self.assertEqual(args[0], m.legacy.REVIEWS + '/' + m.legacy.safe_name(record['reviewer']) + '.jsonl')
        self.assertEqual(args[1], 'a')
        written = json.loads(output().write.call_args.args[0])
        self.assertEqual(written['key'], record['key'])
        self.assertEqual(written['sync'], 4)
        self.assertIn('ts', written)

    def test_html_relative_media_safe_dom_and_real_content(self):
        class Safe(HTMLParser):
            def __init__(self):
                super().__init__()
                self.images, self.videos = 0, 0
            def handle_starttag(self, tag, attrs):
                values = dict(attrs)
                if tag in ('script', 'iframe') or any(k.startswith('on') for k in values):
                    raise AssertionError('unsafe HTML')
                if tag in ('img', 'video'):
                    if not values['src'].startswith('media?p='):
                        raise AssertionError('nonrelative media')
                    if tag == 'img':
                        self.images += 1
                    else:
                        self.videos += 1
        body = m.PAGE.read_text()
        parser = Safe()
        parser.feed(body)
        self.assertEqual(parser.images, 63)
        self.assertEqual(parser.videos, 3)
        self.assertIn('非全批次通过、非人工批准、非训练准入', body)
        self.assertIn('href="./"', body)

    def test_no_new_port_socket_model_or_signal_in_import(self):
        source = PATH.read_text()
        self.assertNotIn('torch', source)
        self.assertNotIn('subprocess', source)
        self.assertNotIn('kill', source)
        with patch.object(m.legacy, 'ThreadingHTTPServer', side_effect=AssertionError('server must not start')):
            m.configure(m.parser().parse_args([]), m.parser())


if __name__ == '__main__':
    unittest.main()
