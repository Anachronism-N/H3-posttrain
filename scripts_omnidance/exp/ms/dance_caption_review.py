"""Independent rich/MS caption review; GET never creates or approves a review.

Only explicit, CSRF-protected human POSTs persist decisions. --self-test uses
synthetic TemporaryDirectory fixtures, never real captions or media.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import stat
import json
import os
from pathlib import Path
import re
import secrets
import socket
import tempfile
import threading
import time
import unittest
from urllib.parse import urlsplit
import wave

FUSION_PATH = Path(__file__).with_name('dance_caption_fusion.py')
_spec = importlib.util.spec_from_file_location('caption_review_fusion', FUSION_PATH)
fusion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fusion)
PROTOCOL = fusion.PROTOCOL
WINDOW = list(fusion.AUDIO_WINDOW_SECONDS)
CHECKS = ('facts_approved', 'narrative_approved', 'window_verified', 'conflicts_resolved')
MAX_JSON = 32 * 1024
MAX_INPUT_JSON = 32 * 1024 * 1024


class ReviewError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def digest(data):
    return hashlib.sha256(data).hexdigest()


def decode_json(data):
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(data, object_pairs_hook=fusion.unique_object,
                      parse_constant=invalid_constant)


def read_blob(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ReviewError(422, 'JSON/source input must be a regular nonsymlink file')
        data = stream.read(MAX_INPUT_JSON + 1)
        after = os.fstat(stream.fileno())
    if len(data) > MAX_INPUT_JSON:
        raise ReviewError(422, 'JSON input exceeds size limit')
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ReviewError(409, 'Input changed during read')
    return data


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()


def exact_window(value):
    return (isinstance(value, list) and len(value) == 2
            and all(type(item) in (int, float) for item in value) and value == WINDOW)


def valid_notes(row):
    return all(isinstance(row.get(key), str) and row[key].strip()
               for key in ('reviewer', 'window_evidence', 'notes')) and exact_window(row.get('reviewed_audio_window_seconds'))


class ReviewState:
    def __init__(self, manifest, rewrites, out, expected_sha=None, fixture_only=False):
        self.manifest_path = Path(manifest).resolve(strict=True)
        self.manifest_data = read_blob(self.manifest_path)
        self.manifest_sha = digest(self.manifest_data)
        if expected_sha is not None and self.manifest_sha != expected_sha:
            raise ReviewError(422, 'Manifest SHA does not match --expect-manifest-sha')
        self.manifest = fusion.load_manifest(self.manifest_path)
        if digest(read_blob(self.manifest_path)) != self.manifest_sha:
            raise ReviewError(409, 'Manifest changed while loading')
        self.records = {row['id']: row for row in self.manifest['records']}
        if len(self.records) != 32:
            raise ReviewError(422, 'This page requires exactly 32 preview records')
        self.rewrites_arg = Path(rewrites).absolute()
        self.rewrites = self.rewrites_arg.resolve()
        if self.rewrites.exists() and not self.rewrites.is_dir():
            raise ReviewError(422, '--rewrites must be a directory or pending directory')
        out_arg = Path(out).absolute()
        if not out_arg.parent.is_dir() or out_arg.suffix != '.json' or out_arg.is_symlink():
            raise ReviewError(422, '--out must be a nonsymlink .json in an existing directory')
        self.out = out_arg.parent.resolve() / out_arg.name
        if self.out == self.manifest_path or self.out == self.rewrites or self.rewrites in self.out.parents:
            raise ReviewError(422, 'Review output must be independent of manifest and rewrite directory')
        for record in self.records.values():
            sources = list(record.get('provenance', {}).values())
            sources.append(record.get('source_rich', {}).get('raw_source', {}))
            for source in sources:
                if source.get('path') and Path(source['path']).resolve() == self.out:
                    raise ReviewError(422, 'Review output cannot overwrite a manifest source file')
        self.fixture_only = fixture_only
        self.lock = threading.RLock()
        self.sessions = {}
        self.media = {}
        for sid, row in self.records.items():
            if not exact_window(row.get('audio_window_seconds')):
                raise ReviewError(422, 'Wrong manifest audio window: ' + sid)
            for kind, key in (('image', 'image_path'), ('audio', 'audio_path'), ('paired_video', 'paired_video_path')):
                source = row['provenance'][kind]
                path = Path(row[key])
                if not path.is_absolute() or str(path) != source['path'] or not re.fullmatch(r'[0-9a-f]{64}', source['sha256']):
                    raise ReviewError(422, 'Media provenance mismatch: ' + sid)
                pinned = path.resolve()
                if pinned == self.out:
                    raise ReviewError(422, 'Review output cannot overwrite source media')
                self.media[(sid, kind)] = (path, pinned, source['sha256'])
            original = row.get('source_video')
            evidence = row.get('provenance', {}).get('source_video')
            if isinstance(original, str) and isinstance(evidence, dict):
                path = Path(original)
                if (path.is_absolute() and str(path) == evidence.get('path')
                        and isinstance(evidence.get('sha256'), str)
                        and re.fullmatch(r'[0-9a-f]{64}', evidence['sha256'])
                        and path.is_file() and path.resolve() != self.out):
                    self.media[(sid, 'source_video')] = (path, path.resolve(), evidence['sha256'])
        self.source = dict(path=str(Path(__file__).resolve()), sha256=digest(read_blob(Path(__file__))))
        self.fusion_source = dict(path=str(FUSION_PATH.resolve()), sha256=digest(read_blob(FUSION_PATH)))
        # Validate existing human data immediately, without writing or replacing it.
        self.current()

    def assert_sources(self):
        if self.out.parent.resolve() != self.out.parent or not self.out.parent.is_dir():
            raise ReviewError(409, 'Review output parent realpath changed')
        if digest(read_blob(self.manifest_path)) != self.manifest_sha:
            raise ReviewError(409, 'Manifest changed; stop and inspect the source version')
        if self.rewrites_arg.resolve() != self.rewrites:
            raise ReviewError(409, 'Rewrite directory realpath changed')
        if digest(read_blob(FUSION_PATH)) != self.fusion_source['sha256']:
            raise ReviewError(409, 'Fusion protocol code changed; restart only after inspection')
        if digest(read_blob(Path(__file__))) != self.source['sha256']:
            raise ReviewError(409, 'Review server code changed; restart only after inspection')

    def candidates(self):
        result = {}
        for sid, record in self.records.items():
            path = self.rewrites / (sid + '.json')
            if path.is_symlink():
                raise ReviewError(409, 'Rewrite symlink rejected: ' + sid)
            if not path.exists():
                result[sid] = dict(status='pending', sha256=None, target=None, validation=None)
                continue
            if not path.is_file() or path.resolve().parent != self.rewrites:
                raise ReviewError(409, 'Rewrite path is not the expected source: ' + sid)
            data = read_blob(path)
            sha = digest(data)
            try:
                value = decode_json(data)
                if (value.get('protocol') != PROTOCOL or value.get('id') != sid
                        or value.get('split') != record['split']
                        or value.get('manifest_sha256') != self.manifest_sha
                        or value.get('rewrite_input_sha256') != record['rewrite_input_sha256']):
                    raise ReviewError(409, 'Rewrite is not bound to this manifest: ' + sid)
                validation = fusion.validate_target(value.get('target'), record)
                result[sid] = dict(status='ready' if validation['format_valid'] else 'invalid',
                                   sha256=sha, target=value.get('target'), validation=validation,
                                   raw_generation=value.get('raw_generation', ''),
                                   raw_source_location=value.get('raw_source_location'))
            except (ValueError, AttributeError, TypeError, KeyError) as error:
                result[sid] = dict(status='invalid', sha256=sha, target=None,
                                   validation=dict(format_valid=False, errors=[str(error)]),
                                   raw_generation=data.decode('utf-8', errors='replace'))
        return result

    def default_row(self, sid, candidate):
        return dict(id=sid, rewrite_sha256=candidate['sha256'],
                    **{key: False for key in CHECKS}, reviewed_audio_window_seconds=None,
                    reviewer='', window_evidence='', notes='')

    def build_document(self, rows, candidates):
        approved = sum(all(row[key] is True for key in CHECKS)
                       and valid_notes(row) and candidates[row['id']]['status'] == 'ready'
                       and row['rewrite_sha256'] == candidates[row['id']]['sha256'] for row in rows)
        return dict(protocol=PROTOCOL, manifest_sha256=self.manifest_sha,
                    preview_manifest=dict(path=str(self.manifest_path), sha256=self.manifest_sha),
                    rewrites_dir=str(self.rewrites),
                    stage='preview32_approved' if approved == 32 else 'partial',
                    records=rows, counts=dict(total=32, approved=approved,
                                             pending=sum(item['status'] == 'pending' for item in candidates.values())),
                    fixture_only=self.fixture_only, review_server_source=self.source,
                    fusion_protocol_source=self.fusion_source)

    def current(self):
        self.assert_sources()
        candidates = self.candidates()
        if self.out.is_symlink():
            raise ReviewError(409, 'Review output became a symlink')
        if not self.out.exists():
            rows = [self.default_row(sid, candidates[sid]) for sid in self.records]
            return self.build_document(rows, candidates), 'absent', candidates
        data = read_blob(self.out)
        saved = decode_json(data)
        if (not isinstance(saved, dict) or saved.get('protocol') != PROTOCOL
                or saved.get('manifest_sha256') != self.manifest_sha
                or saved.get('preview_manifest') != dict(path=str(self.manifest_path), sha256=self.manifest_sha)
                or saved.get('rewrites_dir') != str(self.rewrites)
                or saved.get('fixture_only', False) is not self.fixture_only):
            raise ReviewError(409, 'Existing review is bound to a different source; never overwrite it')
        mapped = fusion.review_record_map(saved)
        if set(mapped) != set(self.records):
            raise ReviewError(409, 'Existing review does not have the exact 32 IDs')
        rows = []
        for sid in self.records:
            row = dict(mapped[sid])
            if any(type(row.get(key)) is not bool for key in CHECKS):
                raise ReviewError(409, 'Existing review has invalid approval booleans: ' + sid)
            sha = row.get('rewrite_sha256')
            if sha is not None and sha != candidates[sid]['sha256']:
                raise ReviewError(409, 'Rewrite changed after binding; preserve and inspect human review: ' + sid)
            if sha is None:
                if any(row[key] for key in CHECKS):
                    raise ReviewError(409, 'Unbound existing approval rejected: ' + sid)
                row['rewrite_sha256'] = candidates[sid]['sha256']
            if any(row[key] for key in CHECKS) and (not valid_notes(row) or candidates[sid]['status'] != 'ready'):
                raise ReviewError(409, 'Existing approval lacks real evidence or valid target: ' + sid)
            rows.append(row)
        document = {**saved, **self.build_document(rows, candidates)}
        return document, digest(data), candidates

    def new_session(self):
        with self.lock:
            now = time.time()
            self.sessions = {key: value for key, value in self.sessions.items() if now - value['created'] < 43200}
            if len(self.sessions) >= 256:
                raise ReviewError(503, 'Too many review sessions')
            sid = secrets.token_urlsafe(32)
            self.sessions[sid] = dict(created=now, csrf=secrets.token_urlsafe(32), views={})
            return sid

    def session(self, sid):
        value = self.sessions.get(sid)
        if value is None or time.time() - value['created'] >= 43200:
            raise ReviewError(403, 'Open the review page again to create a session')
        return value

    def snapshot(self, sid):
        with self.lock:
            session = self.session(sid)
            document, revision, candidates = self.current()
            view = secrets.token_urlsafe(32)
            session['views'][view] = dict(revision=revision,
                                         rewrites={key: value['sha256'] for key, value in candidates.items()})
            while len(session['views']) > 8:
                del session['views'][next(iter(session['views']))]
            records = []
            reviews = {row['id']: row for row in document['records']}
            for key, row in self.records.items():
                records.append(dict(id=key, split=row['split'], genre=row['source_selected_caption']['dims']['genre'],
                                    rich_sections=row['source_rich']['motion_sections'],
                                    rich_raw_original=row['source_rich'].get('raw_original', ''),
                                    source_MS=row['source_MS'],
                                    original_MS=[dict(id='MS:' + str(index), text=text)
                                                 for index, text in enumerate(row['source_MS']['ms']) if text.strip()],
                                    paired_video_path=row['paired_video_path'], source_video_path=row.get('source_video'),
                                    source_video_review=dict(status='available_not_human_verified' if (key, 'source_video') in self.media else 'unverified_not_whitelisted',
                                        reason='仅完整原视频对比来源；是否映射rich/MS仍需用户真实证据，不构成自动审核' if (key, 'source_video') in self.media else
                                               '原视频未获manifest明确SHA来源白名单或实际文件缺失；不读取、不猜测映射，当前未核'),
                                    reviewed_audio_window_seconds=list(row['audio_window_seconds']),
                                    facts=row['facts'], excluded_rich_sentences=row.get('excluded_rich_sentences', []),
                                    time_scope=row['time_scope'], candidate=candidates[key], review=reviews[key],
                                    media={kind: 'media/' + key + '/' + kind for kind in ('image', 'paired_video', 'audio', 'source_video')
                                           if (key, kind) in self.media}))
            return dict(protocol=PROTOCOL, manifest_sha256=self.manifest_sha,
                        preview_manifest=document['preview_manifest'], rewrites_dir=str(self.rewrites),
                        stage=document['stage'], counts=document['counts'], csrf=session['csrf'],
                        view_token=view, revision=revision, fixture_only=self.fixture_only, records=records)

    def save(self, sid, request):
        fields = {'id', 'csrf', 'view_token', 'revision', *CHECKS,
                  'reviewer', 'window_evidence', 'notes', 'reviewed_audio_window_seconds'}
        if not isinstance(request, dict) or set(request) != fields:
            raise ReviewError(422, 'Unexpected POST schema; source hashes and paths cannot be supplied')
        with self.lock:
            session = self.session(sid)
            if not isinstance(request['csrf'], str) or not secrets.compare_digest(request['csrf'], session['csrf']):
                raise ReviewError(403, 'CSRF nonce mismatch')
            if not isinstance(request['view_token'], str):
                raise ReviewError(422, 'Invalid snapshot token')
            view = session['views'].get(request['view_token'])
            if view is None or request['revision'] != view['revision']:
                raise ReviewError(409, 'Unknown or stale snapshot; refresh before reviewing')
            if not isinstance(request['revision'], str) or not re.fullmatch(r'(?:absent|[0-9a-f]{64})', request['revision']):
                raise ReviewError(422, 'Invalid output revision')
            clip = request['id']
            if not isinstance(clip, str) or clip not in self.records:
                raise ReviewError(422, 'Unknown clip ID')
            if any(type(request[key]) is not bool for key in CHECKS):
                raise ReviewError(422, 'Approval fields must be actual JSON booleans')
            for key, maximum in (('reviewer', 256), ('window_evidence', 8192), ('notes', 8192)):
                if not isinstance(request[key], str) or not request[key].strip() or len(request[key]) > maximum or '\x00' in request[key]:
                    raise ReviewError(422, 'Actual nonempty ' + key + ' is required')
            if (not exact_window(request['reviewed_audio_window_seconds'])
                    or request['reviewed_audio_window_seconds'] != self.records[clip]['audio_window_seconds']):
                raise ReviewError(422, 'Reviewed audio window must equal actual manifest window [0.0, 5.0625]')
            # A directory lock avoids creating a separate lock file and serializes
            # cooperating server processes, including the first output creation.
            directory_fd = os.open(self.out.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                fcntl.flock(directory_fd, fcntl.LOCK_EX)
                document, revision, candidates = self.current()
                if revision != view['revision']:
                    raise ReviewError(409, 'Another reviewer saved; no data overwritten. Refresh first')
                candidate = candidates[clip]
                if candidate['sha256'] != view['rewrites'][clip]:
                    raise ReviewError(409, 'Candidate changed since viewing; review the new candidate first')
                if any(request[key] for key in CHECKS) and candidate['status'] != 'ready':
                    raise ReviewError(422, 'Pending or invalid candidate cannot receive any approval')
                row = next(row for row in document['records'] if row['id'] == clip)
                row.update({key: request[key] for key in (*CHECKS, 'reviewer', 'window_evidence', 'notes', 'reviewed_audio_window_seconds')})
                row['rewrite_sha256'] = candidate['sha256']
                row['reviewed_at'] = datetime.now(timezone.utc).isoformat()
                if all(row[key] for key in CHECKS):
                    fusion.approved_review(row, self.rewrites / (clip + '.json'))
                document = {**document, **self.build_document(document['records'], candidates)}
                document['last_saved_at'] = row['reviewed_at']
                self.atomic_save(document, revision, {key: value['sha256'] for key, value in candidates.items()})
                del session['views'][request['view_token']]
            finally:
                fcntl.flock(directory_fd, fcntl.LOCK_UN)
                os.close(directory_fd)
            return self.snapshot(sid)

    def atomic_save(self, document, expected_revision, expected_rewrites):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.out.parent, prefix='.' + self.out.name + '.',
                                             suffix='.tmp', delete=False) as stream:
                temporary = stream.name
                os.chmod(temporary, 0o600)
                stream.write(json_bytes(document))
                stream.flush()
                os.fsync(stream.fileno())
            self.assert_sources()
            if {key: value['sha256'] for key, value in self.candidates().items()} != expected_rewrites:
                raise ReviewError(409, 'Rewrite changed during save; no review overwritten')
            if self.out.is_symlink():
                raise ReviewError(409, 'Review output became a symlink')
            revision = digest(read_blob(self.out)) if self.out.exists() else 'absent'
            if revision != expected_revision:
                raise ReviewError(409, 'Output changed during save; no review overwritten')
            if revision == 'absent':
                os.link(temporary, self.out)
            else:
                os.replace(temporary, self.out)
                temporary = None
            fd = os.open(self.out.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            if temporary is not None:
                os.unlink(temporary)

    def open_media(self, clip, kind):
        source = self.media.get((clip, kind))
        if source is None:
            raise ReviewError(404, 'Media is not in the manifest whitelist')
        original, pinned, expected = source
        if original.resolve() != pinned:
            raise ReviewError(409, 'Media realpath changed')
        fd = os.open(pinned, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        stream = os.fdopen(fd, 'rb')
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise ReviewError(422, 'Media is not a regular file')
            sha = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                sha.update(block)
            after = os.fstat(fd)
            if sha.hexdigest() != expected or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ReviewError(409, 'Media SHA differs from actual manifest source')
            stream.seek(0)
            if kind == 'audio':
                with wave.open(stream, 'rb') as wav:
                    rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
                    frames = int(rate * WINDOW[1])
                    if (not 1 <= rate <= 192000 or channels not in (1, 2) or width not in (1, 2, 3, 4)
                            or wav.getnframes() < frames or frames * channels * width > 8 * 1024 * 1024):
                        raise ReviewError(422, 'Paired audio cannot supply the full requested PCM window')
                    pcm = wav.readframes(frames)
                    if len(pcm) != frames * channels * width:
                        raise ReviewError(422, 'Truncated paired WAV')
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wav:
                    wav.setnchannels(channels)
                    wav.setsampwidth(width)
                    wav.setframerate(rate)
                    wav.writeframes(pcm)
                stream.close()
                data = buffer.getvalue()
                return io.BytesIO(data), len(data), 'audio/wav'
            media_type = 'video/mp4' if kind in ('paired_video', 'source_video') else {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}.get(pinned.suffix.lower())
            if media_type is None:
                raise ReviewError(422, 'Unsupported source image type')
            return stream, after.st_size, media_type
        except Exception:
            stream.close()
            raise


PAGE = r'''<!doctype html>
<html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>32条 rich＋MS caption 人工审核</title>
<style>
body{background:#111827;color:#e5e7eb;font:16px/1.6 system-ui;margin:24px;max-width:1500px}h1{font-size:24px}button,select,input,textarea{font:inherit;padding:8px;background:#1f2937;color:#fff;border:1px solid #64748b;border-radius:4px}button{cursor:pointer}button:disabled{opacity:.5;cursor:default}.warn{background:#4b2d12;padding:14px;border:1px solid #f59e0b}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#1f2937;padding:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.media{display:flex;gap:12px;flex-wrap:wrap}img,video{width:min(320px,45vw);max-height:500px;object-fit:contain;background:#000}label{display:block;margin:10px 0}textarea{display:block;width:95%;min-height:90px}input[type=text]{width:90%}#status{white-space:pre-wrap;color:#fde68a}.checks label{padding:8px;border:1px solid #64748b}@media(max-width:900px){.grid{grid-template-columns:1fr}}small{overflow-wrap:anywhere}
</style>
<h1>32条 rich＋原始MS → 融合caption：独立人工审核</h1>
<div class="warn">格式通过 ≠ 事实或人工批准。rich/MS 源窗均为 unknown；MS句序及“seconds earlier/later”不是可信时间线。必须实际观看并写下窗证据，不可把“unknown”自动认作已核验。H3旧loader的107帧与本次81帧/16fps＝5.0625秒窗不等价。仅仅音频裁到5.0625秒不证明rich/MS映射到此窗；原视频可用也不代表人工已核。这里不运行H3、不生成视频、不扩大数据。缓存完成、rewrite run_complete或生成视频画质都不是caption批准。未生成或格式无效的candidate不能批准。</div>
<p id="overview"></p><small id="source"></small><p><button id="prev" type="button">上一条</button> <select id="picker"></select> <button id="next" type="button">下一条</button> <button id="refresh" type="button">刷新状态（将丢弃未保存表单）</button></p>
<h2 id="clip"></h2><div class="media"><div>参考图<br><img id="image" alt="实际参考图"></div><div>paired normalized video（静音；播放限制0–5.0625秒）<br><video id="video" controls muted playsinline preload="metadata"></video></div></div>
<p>外置配对音频：服务端真实PCM裁剪，仅[0, 5.0625]秒，不补齐、不播放整首。</p><audio id="audio" controls preload="metadata"></audio>
<details><summary>完整原视频与归一化短片来源对比（未自动核验源窗）</summary><pre id="sourcePaths"></pre><p id="sourceVideoStatus"></p><video id="sourceVideo" controls muted playsinline preload="none" hidden></video></details>
<p id="mediaStatus"></p><div class="grid"><section><h3>原始rich前三章（事实ID）</h3><pre id="rich"></pre><details><summary>source_rich.raw_original 原始完整全文（审查上下文，不进入动作目标）</summary><pre id="richRaw"></pre></details><h3>原始MS事实（保留原始ID与原文；句序不是时间线）</h3><pre id="ms"></pre><details><summary>source_MS 完整原始对象（含ms_nonagg/kin/出处；不是可信时间标注）</summary><pre id="msRaw"></pre></details><details><summary>实际teacher facts与被排除rich句子</summary><pre id="facts"></pre></details></section>
<section><h3>candidate motion_caption</h3><pre id="caption"></pre><h3>used_facts / conflicts / time_scope</h3><pre id="target"></pre><h3>服务器重新计算的格式检查（非语义审批）</h3><pre id="validation"></pre><details><summary>paired source窗元信息</summary><pre id="scope"></pre></details><small id="binding"></small></section></div>
<form id="review"><h3>四项独立人工判断（无人审时均为false）</h3><div class="checks">
<label><input type="checkbox" id="facts_approved"> 事实：每个动作有实际证据，无凭空添加或遗漏重要事实</label>
<label><input type="checkbox" id="narrative_approved"> 叙事：连续英文舞蹈叙事，不是关节动作清单，未泄漏外观/音频/精确时间</label>
<label><input type="checkbox" id="window_verified"> 窗：实际核验全部候选动作适用于本次[0,5.0625]秒配对窗，并记录观察证据</label>
<label><input type="checkbox" id="conflicts_resolved"> 冲突：已逐一核查conflicts（含unknown窗），在notes中说明如何解决</label></div>
<label>实际审查人 reviewer（必填）<input id="reviewer" type="text" maxlength="256" required autocomplete="off"></label>
<label>实际窗证据 window_evidence（必填；写明来源/窗/观察到的动作与对应证据，不接受空白）<textarea id="window_evidence" maxlength="8192" required></textarea></label>
<label>notes（必填；问题或批准依据/冲突处理）<textarea id="notes" maxlength="8192" required></textarea></label>
<label>已实际审查音频窗 reviewed_audio_window_seconds（必须明确选择） <select id="window" required><option value="">尚未确认</option value="confirmed">[0.0, 5.0625]秒（我已实际审查此窗）</option></select></label>
<button id="save" type="submit">显式保存当前条人工判断（不会自动全通过）</button></form><p id="status" role="status"></p>
<script>
'use strict';
const checks=['facts_approved','narrative_approved','window_verified','conflicts_resolved'];
const el=id=>document.getElementById(id);let state=null,index=0,dirty=false;
const show=(id,value)=>{el(id).textContent=typeof value==='string'?value:JSON.stringify(value,null,2);};
function render(){
 const row=state.records[index],candidate=row.candidate,review=row.review;
 show('overview',`${state.stage} · 四项全部批准 ${state.counts.approved}/32 · pending ${state.counts.pending}${state.fixture_only?' · SYNTHETIC FIXTURE ONLY（非真实人评）':''}`);
 show('source',`manifest=${state.preview_manifest.path}\nSHA256=${state.manifest_sha256}\nrewrites=${state.rewrites_dir}`);
 el('picker').replaceChildren();state.records.forEach((r,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=`${i+1}/32 ${r.id} [${r.candidate.status}]`;el('picker').append(option);});el('picker').value=String(index);
 show('clip',`${index+1}/32 ${row.id} · ${row.split} · ${row.genre} · ${candidate.status}`);
 el('image').src=row.media.image;el('video').src=row.media.paired_video+'#t=0,5.0625';el('video').muted=true;el('video').volume=0;el('audio').src=row.media.audio;
 show('sourcePaths',{paired_video_path:row.paired_video_path,source_video:row.source_video_path,source_window_mapping:row.time_scope});show('sourceVideoStatus',row.source_video_review);
 const original=el('sourceVideo');original.pause();original.muted=true;original.volume=0;original.hidden=!row.media.source_video;if(row.media.source_video){original.src=row.media.source_video;}else{original.removeAttribute('src');original.load();}
 show('mediaStatus','');show('rich',row.rich_sections.map(s=>`RICH:${s.number} [${s.title}]\n${s.text}`).join('\n\n'));show('richRaw',row.rich_raw_original||'Manifest未提供raw_original，不代补原文。');
 show('ms',row.original_MS.map(f=>f.id+'\n'+f.text).join('\n\n'));show('msRaw',row.source_MS);show('facts',{facts:row.facts,excluded_rich_sentences:row.excluded_rich_sentences});
 const target=candidate.target;show('caption',target&&typeof target.motion_caption==='string'?target.motion_caption:(candidate.status==='pending'?'PENDING：真实rewrite尚未生成；不可批准':candidate.raw_generation||'无有效candidate'));
 show('target',target?{used_facts:target.used_facts,conflicts:target.conflicts,time_scope:target.time_scope}:null);show('validation',candidate.validation);show('scope',row.time_scope);
 show('binding',`实际rewrite SHA256: ${candidate.sha256||'pending/unbound'} · 服务器绑定manifestSHA与rewriteSHA，表单不能修改`);
 checks.forEach(key=>{el(key).checked=review[key]===true;el(key).disabled=candidate.status!=='ready';});
 ['reviewer','window_evidence','notes'].forEach(key=>{el(key).value=review[key]||'';});el('window').value=Array.isArray(review.reviewed_audio_window_seconds)?'confirmed':'';
 el('save').disabled=false;el('prev').disabled=index===0;el('next').disabled=index===31;dirty=false;
}
async function reload(){const response=await fetch('api',{credentials:'same-origin',cache:'no-store'});const data=await response.json();if(!response.ok)throw new Error(data.error||String(response.status));state=data;render();}
function move(to){if(dirty&&!confirm('当前条有未保存人工判断，确定丢弃？')){el('picker').value=String(index);return;}index=Math.max(0,Math.min(31,to));render();show('status','');}
el('prev').onclick=()=>move(index-1);el('next').onclick=()=>move(index+1);el('picker').onchange=()=>move(Number(el('picker').value));
el('refresh').onclick=()=>{if(!dirty||confirm('刷新将丢弃未保存表单，确定？'))reload().then(()=>show('status','已刷新；未自动写入评分')).catch(e=>show('status',e.message));};
el('review').addEventListener('input',()=>{dirty=true;});
el('review').onsubmit=async event=>{event.preventDefault();const row=state.records[index];const body={id:row.id,csrf:state.csrf,view_token:state.view_token,revision:state.revision,reviewer:el('reviewer').value,window_evidence:el('window_evidence').value,notes:el('notes').value,reviewed_audio_window_seconds:el('window').value==='confirmed'?row.reviewed_audio_window_seconds:null};checks.forEach(key=>{body[key]=el(key).checked;});el('save').disabled=true;
 try{const response=await fetch('review',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw new Error(data.error||String(response.status));state=data;render();show('status','已保存这一条显式人工判断；stage='+state.stage);}catch(error){show('status',error.message+'。未自动重试、未覆盖；请保留笔记并人工刷新。');el('save').disabled=false;}};
const video=el('video');video.addEventListener('volumechange',()=>{if(!video.muted)video.muted=true;if(video.volume!==0)video.volume=0;});
video.addEventListener('timeupdate',()=>{if(video.currentTime>=5.0625){video.pause();video.currentTime=5.0624;}});video.addEventListener('seeking',()=>{if(video.currentTime>5.0625)video.currentTime=5.0624;});video.addEventListener('play',()=>{if(video.currentTime>=5.0624)video.currentTime=0;});
el('sourceVideo').addEventListener('volumechange',()=>{const original=el('sourceVideo');if(!original.muted)original.muted=true;if(original.volume!==0)original.volume=0;});
['image','video','audio','sourceVideo'].forEach(id=>el(id).addEventListener('error',()=>show('mediaStatus','实际媒体加载失败：'+id+'；不要在缺少媒体证据时批准。')));
reload().catch(error=>show('status',error.message));
let pendingPoll=false;setInterval(async()=>{if(pendingPoll||dirty||!state||state.records[index].candidate.status!=='pending')return;pendingPoll=true;try{const response=await fetch('api',{credentials:'same-origin',cache:'no-store'});const data=await response.json();if(!response.ok)throw new Error(data.error||String(response.status));if(dirty)return;const previous=state.records[index].candidate.sha256;state=data;if(state.records[index].candidate.sha256!==previous){render();show('status','实际candidate状态已刷新；未自动评分或批准。');}}catch(error){show('status',error.message);}finally{pendingPoll=false;}},15000);
</script></html>'''


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, state, allowed_hosts=(), public_origin=None):
        self.state = state
        self.public_origin = public_origin
        if public_origin is not None:
            parsed = urlsplit(public_origin)
            if (parsed.scheme not in ('http', 'https') or not parsed.hostname or
                    parsed.username is not None or parsed.password is not None or
                    parsed.path or parsed.query or parsed.fragment or
                    parsed.port is not None and not 1 <= parsed.port <= 65535):
                raise ValueError('--public-origin requires an exact http(s) origin without a path')
        super().__init__(address, ReviewHandler)
        hosts = {'127.0.0.1', 'localhost', '::1', socket.gethostname().lower()}
        try:
            hosts.update(socket.gethostbyname_ex(socket.gethostname())[2])
        except OSError:
            pass
        if address[0] not in ('0.0.0.0', '::'):
            hosts.add(address[0].lower())
        hosts.update(host.lower() for host in allowed_hosts)
        self.allowed_hosts = hosts


class ReviewHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, fmt, *args):
        if not self.server.state.fixture_only:
            super().log_message(fmt, *args)

    def check_host(self):
        hosts = self.headers.get_all('Host', [])
        if len(hosts) != 1 or len(hosts[0]) > 256:
            raise ReviewError(400, 'One valid Host header is required')
        host = hosts[0]
        parsed = urlsplit('http://' + host)
        public = urlsplit(self.server.public_origin) if self.server.public_origin else None
        try:
            port = parsed.port or (443 if public and parsed.hostname == public.hostname and public.scheme == 'https' else 80)
        except ValueError:
            raise ReviewError(400, 'Invalid Host port')
        public_host = bool(public and parsed.hostname == public.hostname and
                           port == (public.port or (443 if public.scheme == 'https' else 80)))
        local_host = parsed.hostname in self.server.allowed_hosts and port == self.server.server_port
        if (parsed.username is not None or parsed.password is not None or parsed.path or parsed.query or parsed.fragment
                or not (local_host or public_host)):
            raise ReviewError(403, 'Host not allowed; configure the exact review hostname or public origin')
        return self.server.public_origin if public else 'http://' + host

    def session_id(self):
        jar = cookies.SimpleCookie()
        try:
            jar.load(self.headers.get('Cookie', ''))
        except cookies.CookieError:
            raise ReviewError(403, 'Invalid session cookie')
        sid = jar.get('dance_review_session')
        if sid is None:
            raise ReviewError(403, 'Open / before accessing the review API')
        self.server.state.session(sid.value)
        return sid.value

    def send_headers(self, status, kind, size, extra=()):
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(size))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()

    def send_json(self, status, value):
        data = json_bytes(value)
        self.send_headers(status, 'application/json; charset=utf-8', len(data))
        if self.command != 'HEAD':
            self.wfile.write(data)

    def route(self):
        if len(self.path) > 1024 or '%' in self.path or '\\' in self.path:
            raise ReviewError(400, 'Encoded or unsafe request path rejected')
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or '..' in parsed.path:
            raise ReviewError(400, 'Invalid request path')
        return parsed.path

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            origin = self.check_host()
            if self.headers.get('Origin') not in (None, origin):
                raise ReviewError(403, 'Cross-origin request rejected')
            route = self.route()
            if route == '/':
                sid = self.server.state.new_session()
                data = PAGE.encode()
                script_sha = base64.b64encode(hashlib.sha256(PAGE.split('<script>')[1].split('</script>')[0].encode()).digest()).decode()
                csp = "default-src 'none'; img-src 'self'; media-src 'self'; connect-src 'self'; style-src 'unsafe-inline'; script-src 'sha256-" + script_sha + "'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
                self.send_headers(200, 'text/html; charset=utf-8', len(data),
                                  [('Set-Cookie', 'dance_review_session=' + sid + '; HttpOnly; SameSite=Strict; Path=/'),
                                   ('Content-Security-Policy', csp)])
                if self.command != 'HEAD':
                    self.wfile.write(data)
            elif route == '/api':
                self.send_json(200, self.server.state.snapshot(self.session_id()))
            else:
                match = re.fullmatch(r'/media/([A-Za-z0-9_-]+)/(image|paired_video|audio|source_video)', route)
                if match is None:
                    raise ReviewError(404, 'No such review resource')
                self.session_id()
                self.serve_media(*match.groups())
        except (ReviewError, ValueError, OSError, TypeError, KeyError, AttributeError) as error:
            self.send_json(error.status if isinstance(error, ReviewError) else 422, dict(error=str(error)))

    def serve_media(self, clip, kind):
        self.server.state.assert_sources()
        stream, size, media_type = self.server.state.open_media(clip, kind)
        with stream:
            start, end, status = 0, size - 1, 200
            ranges = self.headers.get_all('Range', [])
            if ranges:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', ranges[0]) if len(ranges) == 1 and len(ranges[0]) < 100 else None
                if match is None or not any(match.groups()):
                    self.send_headers(416, media_type, 0, [('Content-Range', 'bytes */' + str(size))])
                    return
                left, right = match.groups()
                if left:
                    start = int(left)
                    end = min(int(right), size - 1) if right else size - 1
                else:
                    suffix = int(right)
                    start, end = max(0, size - suffix), size - 1
                    if suffix == 0:
                        start = size
                if start >= size or end < start:
                    self.send_headers(416, media_type, 0, [('Content-Range', 'bytes */' + str(size))])
                    return
                status = 206
            extra = [('Accept-Ranges', 'bytes')]
            if status == 206:
                extra.append(('Content-Range', f'bytes {start}-{end}/{size}'))
            if kind == 'audio':
                extra.append(('X-Reviewed-Audio-Window-Seconds', '0,5.0625'))
            self.send_headers(status, media_type, end - start + 1, extra)
            if self.command != 'HEAD':
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = stream.read(min(256 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

    def do_POST(self):
        try:
            origin = self.check_host()
            if self.headers.get_all('Origin', []) != [origin] or self.headers.get('Sec-Fetch-Site') not in (None, 'same-origin'):
                raise ReviewError(403, 'Explicit same-origin POST required')
            if self.route() != '/review':
                raise ReviewError(404, 'No such POST route')
            if self.headers.get('Transfer-Encoding') is not None:
                raise ReviewError(400, 'Transfer-Encoding rejected')
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1 or not re.fullmatch(r'[0-9]{1,6}', lengths[0]):
                raise ReviewError(411, 'One bounded Content-Length is required')
            length = int(lengths[0])
            if not 1 <= length <= MAX_JSON:
                raise ReviewError(413, 'POST body exceeds allowed size')
            if self.headers.get('Content-Type', '').lower() not in ('application/json', 'application/json; charset=utf-8'):
                raise ReviewError(415, 'Expected application/json')
            sid = self.session_id()
            body = self.rfile.read(length)
            if len(body) != length:
                raise ReviewError(400, 'Truncated POST body')
            request = decode_json(body)
            self.send_json(200, self.server.state.save(sid, request))
        except (ReviewError, ValueError, OSError, TypeError, KeyError, AttributeError) as error:
            # Close after rejected requests, including unread oversized bodies.
            self.close_connection = True
            self.send_json(error.status if isinstance(error, ReviewError) else 422, dict(error=str(error)))


class CpuChecks(unittest.TestCase):
    """All POSTs and positive approvals below are explicit SYNTHETIC FIXTURE ONLY."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='caption-review-fixture-only-')
        self.root = Path(self.temporary.name)
        self.rewrites = self.root / 'rewrites'
        self.rewrites.mkdir()
        audio = self.root / 'fixture-only.wav'
        with wave.open(str(audio), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\x00\x00' * 96000)
        image = self.root / 'fixture-only.png'
        image.write_bytes(b'fixture-only-not-real-image')
        video = self.root / 'fixture-only.mp4'
        video.write_bytes(b'fixture-only-not-real-video')
        original = self.root / 'fixture-only-original.mp4'
        original.write_bytes(b'fixture-only-original-video')
        self.rows = []
        for number in range(32):
            sid = 'fixture_' + str(number)
            inp = {'fixture_only': sid}
            self.rows.append(dict(id=sid, split='train' if number < 24 else 'val', source_group=sid,
                                  rewrite_input=inp, rewrite_input_sha256=fusion.object_sha(inp),
                                  facts=[dict(id='RICH:1', text='Fluid arm extensions.', source='rich'),
                                         dict(id='MS:5', text='The elbows bend and extend.', source='MS')],
                                  audio_path=str(audio), image_path=str(image), paired_video_path=str(video), source_video=str(original),
                                  audio_window_seconds=WINDOW, time_scope=dict(status='unknown', rich_source_window=None, ms_source_window=None),
                                  provenance={key: fusion.source_record(path) for key, path in [('image', image), ('audio', audio), ('paired_video', video)]},
                                  source_rich=dict(raw_original='<script>not executable</script> FULL RAW FIXTURE ONLY', motion_sections=[dict(number=i, title='Fixture only', text='<script>not executable</script>') for i in (1, 2, 3)]),
                                  source_MS=dict(ms=['', '', '', '', '', 'The elbows bend and extend.']),
                                  source_selected_caption=dict(dims=dict(genre='fixture only'))))
        self.rows[0]['provenance']['source_video'] = fusion.source_record(original)
        self.manifest = self.root / 'fixture-only-manifest.json'
        self.manifest.write_bytes(json_bytes(dict(protocol=PROTOCOL, stage='preview32', records=self.rows, blocked_ids=[])))
        self.manifest_sha = digest(self.manifest.read_bytes())
        self.out = self.root / 'fixture-only-review.json'
        self.write_candidate(0)
        self.state = ReviewState(self.manifest, self.rewrites, self.out, fixture_only=True)
        self.server = ReviewServer(('127.0.0.1', 0), self.state)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.origin = 'http://127.0.0.1:' + str(self.server.server_port)
        _, headers, body = self.http('GET', '/')
        self.cookie = headers['Set-Cookie'].split(';')[0]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def write_candidate(self, number, invalid=False):
        row = self.rows[number]
        target = dict(motion_caption='The dancer performs fluid arm movements, bending the elbows and extending the arms with controlled transitions.',
                      used_facts=['RICH:1', 'MS:5'], conflicts=['Source window mapping is unknown.'],
                      time_scope=dict(status='unknown', reason='Synthetic fixture has no source window mapping.'))
        if invalid:
            target['motion_caption'] = 'The dancer follows the music.'
        (self.rewrites / (row['id'] + '.json')).write_bytes(json_bytes(dict(protocol=PROTOCOL,
                id=row['id'], split=row['split'], manifest_sha256=self.manifest_sha,
                rewrite_input_sha256=row['rewrite_input_sha256'], target=target, raw_generation='fixture-only')))

    def http(self, method, path, value=None, headers=None, raw=None):
        import http.client
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=10)
        actual_headers = {'Cookie': getattr(self, 'cookie', '')}
        if method == 'POST':
            actual_headers.update({'Origin': self.origin, 'Content-Type': 'application/json'})
        actual_headers.update(headers or {})
        body = raw if raw is not None else json_bytes(value) if value is not None else None
        connection.request(method, path, body=body, headers=actual_headers)
        response = connection.getresponse()
        result = (response.status, dict(response.getheaders()), response.read())
        connection.close()
        return result

    def view(self):
        status, _, data = self.http('GET', '/api')
        self.assertEqual(status, 200, data)
        return decode_json(data)

    def post(self, view, number=0, approve=False, **changes):
        value = dict(id=self.rows[number]['id'], csrf=view['csrf'], view_token=view['view_token'], revision=view['revision'],
                     **{key: approve for key in CHECKS}, reviewer='SYNTHETIC FIXTURE ONLY',
                     window_evidence='Synthetic fixture only, not actual source/human evidence.',
                     notes='CPU automated fixture only; never a real human approval.', reviewed_audio_window_seconds=WINDOW)
        value.update(changes)
        return self.http('POST', '/review', value)

    def test_get_default_false_and_xss_boundary(self):
        view = self.view()
        self.assertFalse(self.out.exists())
        self.assertEqual(view['stage'], 'partial')
        self.assertEqual(view['counts']['pending'], 31)
        self.assertTrue(view['fixture_only'])
        for row in view['records']:
            for key in CHECKS:
                self.assertIs(row['review'][key], False)
        _, headers, body = self.http('GET', '/')
        self.assertNotIn(b'<script>not executable</script>', body)
        self.assertNotIn(b'innerHTML', body)
        self.assertIn('script-src \'sha256-', headers['Content-Security-Policy'])
        self.assertIn(b'textContent', body)

    def test_raw_sources_and_original_video_strict_whitelist(self):
        view = self.view()
        first, second = view['records'][:2]
        self.assertIn('FULL RAW FIXTURE ONLY', first['rich_raw_original'])
        self.assertEqual(first['source_MS'], self.rows[0]['source_MS'])
        self.assertEqual(first['paired_video_path'], self.rows[0]['paired_video_path'])
        self.assertEqual(first['source_video_path'], self.rows[0]['source_video'])
        self.assertEqual(first['source_video_review']['status'], 'available_not_human_verified')
        self.assertEqual(second['source_video_review']['status'], 'unverified_not_whitelisted')
        self.assertIn('source_video', first['media'])
        self.assertNotIn('source_video', second['media'])
        status, headers, body = self.http('GET', '/media/fixture_0/source_video', headers={'Range': 'bytes=0-6'})
        self.assertEqual(status, 206)
        self.assertEqual(headers['Content-Type'], 'video/mp4')
        self.assertEqual(body, b'fixture')
        self.assertEqual(self.http('GET', '/media/fixture_1/source_video')[0], 404)
        self.assertFalse(self.out.exists())
        self.assertIs(first['review']['window_verified'], False)

    def test_range_and_audio_real_crop(self):
        status, headers, body = self.http('GET', '/media/fixture_0/paired_video', headers={'Range': 'bytes=2-8'})
        self.assertEqual(status, 206)
        self.assertEqual(body, b'xture-o')
        self.assertTrue(headers['Content-Range'].startswith('bytes 2-8/'))
        status, _, body = self.http('GET', '/media/fixture_0/audio')
        self.assertEqual(status, 200)
        with wave.open(io.BytesIO(body), 'rb') as wav:
            self.assertEqual(wav.getnframes(), 81000)
            self.assertEqual(wav.getnframes() / wav.getframerate(), 5.0625)
        self.assertEqual(self.http('GET', '/media/fixture_0/audio', headers={'Range': 'bytes=0-9'})[0], 206)
        self.assertEqual(self.http('GET', '/media/fixture_0/audio', headers={'Range': 'bytes=9999999-'})[0], 416)
        self.assertEqual(self.http('GET', '/media/fixture_0/audio', headers={'Range': 'bytes=0-2,4-5'})[0], 416)
        self.assertEqual(self.http('HEAD', '/media/fixture_0/image')[0], 200)

    def test_missing_window_evidence_and_window_rejected(self):
        view = self.view()
        for changes in ({'window_evidence': ''}, {'notes': ' '}, {'reviewer': ''},
                        {'reviewed_audio_window_seconds': None}, {'reviewed_audio_window_seconds': [0, 6]},
                        {'reviewed_audio_window_seconds': [False, 5.0625]}, {'facts_approved': 'true'}):
            self.assertEqual(self.post(view, approve=True, **changes)[0], 422)
        self.assertFalse(self.out.exists())

    def test_human_gate_pending_invalid_and_fusion_compatibility(self):
        view = self.view()
        self.assertEqual(self.post(view, number=1, approve=True)[0], 422)
        self.write_candidate(1, invalid=True)
        view = self.view()
        self.assertEqual(self.post(view, number=1, approve=True)[0], 422)
        status, _, data = self.post(view, approve=True)
        self.assertEqual(status, 200, data)
        saved = decode_json(self.out.read_bytes())
        self.assertEqual(saved['stage'], 'partial')
        self.assertEqual(saved['counts']['approved'], 1)
        self.assertTrue(saved['fixture_only'])
        fusion.approved_review(saved['records'][0], self.rewrites / 'fixture_0.json')
        self.assertTrue(all(saved['records'][0][key] for key in CHECKS))
        self.assertTrue(all(not row[key] for row in saved['records'][1:] for key in CHECKS))
        with self.assertRaises(ValueError):
            fusion.expansion_approval(self.out)

    def test_no_lost_updates_and_existing_review_binding(self):
        first, second = self.view(), self.view()
        self.assertEqual(self.post(first, approve=True)[0], 200)
        before = self.out.read_bytes()
        self.assertEqual(self.post(second, approve=False)[0], 409)
        self.assertEqual(self.out.read_bytes(), before)
        self.assertEqual(self.post(first, approve=False)[0], 409)
        restarted = ReviewState(self.manifest, self.rewrites, self.out, fixture_only=True)
        self.assertTrue(restarted.current()[0]['records'][0]['facts_approved'])
        saved = decode_json(before)
        saved['manifest_sha256'] = '0' * 64
        self.out.write_bytes(json_bytes(saved))
        with self.assertRaises(ReviewError):
            ReviewState(self.manifest, self.rewrites, self.out, fixture_only=True)
        self.assertEqual(self.out.read_bytes(), json_bytes(saved))

    def test_changed_rewrite_cannot_reuse_approval_or_snapshot(self):
        view = self.view()
        self.write_candidate(0, invalid=True)
        self.assertEqual(self.post(view, approve=False)[0], 409)
        self.assertFalse(self.out.exists())
        self.write_candidate(0)
        view = self.view()
        self.assertEqual(self.post(view, approve=True)[0], 200)
        before = self.out.read_bytes()
        self.write_candidate(0, invalid=True)
        self.assertEqual(self.http('GET', '/api')[0], 409)
        self.assertEqual(self.out.read_bytes(), before)

    def test_explicit_proxy_origin_and_relative_routes(self):
        proxy_origin = 'https://caption-review-proxy.invalid'
        with self.assertRaises(ValueError):
            ReviewServer(('127.0.0.1', 0), self.state, public_origin=proxy_origin + '/proxy/8773/')
        self.server.public_origin = proxy_origin
        view = self.view()
        self.assertTrue(all(not row['review'][key] for row in view['records'] for key in CHECKS))
        self.assertTrue(all(not route.startswith('/') for row in view['records'] for route in row['media'].values()))
        status, _, page = self.http('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn(b"fetch('api'", page)
        self.assertIn(b"fetch('review'", page)
        status, headers, _ = self.http('GET', '/')
        self.cookie = headers['Set-Cookie'].split(';')[0]
        self.origin = proxy_origin
        view = self.view()
        status, _, body = self.post(view, approve=True)
        self.assertEqual(status, 200, body)
        self.assertTrue(decode_json(body)['fixture_only'])
        self.assertEqual(self.http('GET', '/api', headers={'Host': 'caption-review-proxy.invalid:443'})[0], 200)
        self.assertEqual(self.http('GET', '/api', headers={'Host': 'caption-review-proxy.invalid:8773'})[0], 403)
        self.assertEqual(self.http('POST', '/review', {}, headers={'Origin': 'https://untrusted.invalid'})[0], 403)

    def test_csrf_origin_schema_paths_and_size(self):
        view = self.view()
        self.assertEqual(self.post(view, csrf='forged')[0], 403)
        self.assertEqual(self.post(view, rewrite_sha256='forged')[0], 422)
        value = dict(id='fixture_0')
        self.assertEqual(self.http('POST', '/review', value, headers={'Origin': 'http://external.invalid'})[0], 403)
        self.assertEqual(self.http('POST', '/review', value, headers={'Host': 'external.invalid'})[0], 403)
        self.assertEqual(self.http('POST', '/review', raw=b'x' * (MAX_JSON + 1))[0], 413)
        self.assertEqual(self.http('POST', '/review', raw=b'{"csrf":"a","csrf":"b"}')[0], 422)
        for path in ('/media/%2e%2e/audio', '/media/../audio', '/media/fixture_0/audio?path=/etc/passwd', '/media/unknown/image'):
            self.assertIn(self.http('GET', path)[0], (400, 404))
        self.assertFalse(self.out.exists())

    def test_media_whitelist_sha_and_realpath(self):
        source = self.root / 'fixture-only.png'
        source.write_bytes(b'changed')
        self.assertEqual(self.http('GET', '/media/fixture_0/image')[0], 409)
        other = self.root / 'not-whitelisted.png'
        other.write_bytes(b'fixture-only-not-real-image')
        source.unlink()
        source.symlink_to(other)
        self.assertEqual(self.http('GET', '/media/fixture_0/image')[0], 409)

    def test_all32_expansion_gate_fixture_only(self):
        for number in range(1, 32):
            self.write_candidate(number)
        view = self.view()
        for number in range(32):
            status, _, body = self.post(view, number=number, approve=True)
            self.assertEqual(status, 200, body)
            view = decode_json(body)
            self.assertEqual(view['stage'], 'preview32_approved' if number == 31 else 'partial')
        saved = decode_json(self.out.read_bytes())
        self.assertTrue(saved['fixture_only'])
        self.assertEqual(saved['counts']['approved'], 32)
        self.assertEqual(fusion.expansion_approval(self.out)['sha256'], digest(self.out.read_bytes()))

    def test_pending_directory_is_read_only_until_explicit_post(self):
        pending = self.root / 'not-yet-generated'
        out = self.root / 'pending-fixture-only-review.json'
        state = ReviewState(self.manifest, pending, out, fixture_only=True)
        sid = state.new_session()
        view = state.snapshot(sid)
        self.assertEqual(view['counts']['pending'], 32)
        self.assertFalse(pending.exists())
        self.assertFalse(out.exists())
        pending.mkdir()
        (pending / 'run_started.json').write_bytes(json_bytes(dict(complete=True, fixture_only=True,
                    provenance=dict(source_video=dict(path=str(self.root / 'fixture-only-original.mp4'))))))
        (pending / 'run_complete.json').write_bytes(json_bytes(dict(complete=True, fixture_only=True)))
        view = state.snapshot(sid)
        self.assertEqual(view['counts']['pending'], 32)
        self.assertEqual(view['counts']['approved'], 0)
        self.assertNotIn('source_video', view['records'][1]['media'])
        self.assertFalse(out.exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--rewrites', type=Path)
    parser.add_argument('--out', type=Path, help='Independent review JSON; existing human data must match the exact sources')
    parser.add_argument('--port', type=int, default=8773)
    parser.add_argument('--host', default='127.0.0.1', help='Loopback by default; exposure to other hosts is explicit')
    parser.add_argument('--allowed-host', action='append', default=[], help='Additional real HTTP hostname (no scheme or port)')
    parser.add_argument('--public-origin', help='Exact browser origin for a trusted reverse proxy; never inferred from request headers')
    parser.add_argument('--expect-manifest-sha', help='Refuse a different manifest byte SHA256')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CpuChecks))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    for name in ('manifest', 'rewrites', 'out'):
        if getattr(args, name) is None:
            parser.error('--' + name + ' is required')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be in [1,65535]')
    if args.expect_manifest_sha is not None and not re.fullmatch(r'[0-9a-f]{64}', args.expect_manifest_sha):
        parser.error('--expect-manifest-sha must be lowercase SHA256')
    for host in [args.host, *args.allowed_host]:
        if not re.fullmatch(r'[A-Za-z0-9_.:-]+', host):
            parser.error('Unsafe hostname')
    try:
        state = ReviewState(args.manifest, args.rewrites, args.out, args.expect_manifest_sha)
        server = ReviewServer((args.host, args.port), state, args.allowed_host, args.public_origin)
    except (ReviewError, ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        parser.error(str(error))
    print(json.dumps(dict(host=args.host, port=server.server_port, manifest_sha256=state.manifest_sha,
                          review_source=state.source, fusion_source=state.fusion_source,
                          out=str(state.out), stage=state.current()[0]['stage'],
                          writes='Explicit actual human POST only; GET does not save',
                          warning='rich/MS window unknown; old H3 loader107 frames not equivalent; no automatic approval'), ensure_ascii=False))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
