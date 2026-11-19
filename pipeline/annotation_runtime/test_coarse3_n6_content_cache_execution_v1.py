import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import time
import coarse3_n6_content_cache_execution_v1 as c


class ContentCacheTests(unittest.TestCase):
    def test_hit_avoids_parent_resolution_and_detects_content_stat_change(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            p = Path(d) / 'content'
            p.write_bytes(b'first')
            with patch.object(Path, 'resolve', side_effect=AssertionError('cache key resolved parents')):
                self.assertEqual(c.content_sha(p), hashlib.sha256(b'first').hexdigest())
                self.assertEqual(c.content_sha(p), hashlib.sha256(b'first').hexdigest())
                before = p.stat()
                p.write_bytes(b'other')
                os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000))
                self.assertEqual(c.content_sha(p), hashlib.sha256(b'other').hexdigest())

    def test_supplied_alias_tracks_new_target_not_stale_canonical_path(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root = Path(d)
            a, b, alias = root / 'a', root / 'b', root / 'alias'
            a.write_bytes(b'alpha')
            b.write_bytes(b'bravo')
            alias.symlink_to(a)
            first = c.content_sha(alias)
            alias.unlink()
            alias.symlink_to(b)
            self.assertNotEqual(c.content_sha(alias), first)
            self.assertEqual(c.content_sha(alias), hashlib.sha256(b'bravo').hexdigest())
            # Keep the former inode allocated: this case specifically tests
            # inode replacement, not filesystem timestamp/inode reuse.
            b.rename(root / 'old-b-retained')
            b.write_bytes(b'fresh')
            self.assertEqual(c.content_sha(alias), hashlib.sha256(b'fresh').hexdigest())
            a_dir, b_dir = root / 'one', root / 'two'
            a_dir.mkdir()
            b_dir.mkdir()
            (b_dir / 'sub').mkdir()
            (a_dir / 'link').symlink_to(b_dir / 'sub')
            (a_dir / 'file').write_bytes(b'lexical')
            (b_dir / 'file').write_bytes(b'actual')
            alias_dotdot = a_dir / 'link' / '..' / 'file'
            self.assertEqual(os.path.abspath(alias_dotdot), str(a_dir / 'file'))
            self.assertEqual(c.content_sha(a_dir / 'file'), hashlib.sha256(b'lexical').hexdigest())
            self.assertEqual(c.content_sha(alias_dotdot), hashlib.sha256(b'actual').hexdigest())
            (a_dir / 'link').unlink()
            with self.assertRaises(FileNotFoundError):
                c.content_sha(alias_dotdot)

    def test_mutation_during_full_read_rejected_without_cache_commit(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            p = Path(d) / 'changed'
            p.write_bytes(b'original')
            original_open = Path.open
            class ChangedRead:
                def __init__(self, stream):
                    self.stream, self.changed = stream, False
                def __enter__(self):
                    self.stream.__enter__()
                    return self
                def __exit__(self, *args):
                    return self.stream.__exit__(*args)
                def fileno(self):
                    return self.stream.fileno()
                def read(self, count):
                    value = self.stream.read(count)
                    if value and not self.changed:
                        self.changed = True
                        with original_open(p, 'wb') as target:
                            target.write(b'changed-longer-content')
                    return value
            def opened(path, *args, **kwargs):
                stream = original_open(path, *args, **kwargs)
                return ChangedRead(stream) if path == p and args == ('rb',) else stream
            cache_before = dict(c._cache)
            with patch.object(Path, 'open', opened):
                with self.assertRaisesRegex(RuntimeError, 'changed during full content SHA'):
                    c.content_sha(p)
            self.assertEqual(c._cache, cache_before)
        # A same-inode content-cache hit must never replace the original
        # queue/output realpath confinement guard.
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            import importlib.util
            import json
            import sys
            root = Path(d)
            allowed, outside = root / 'allowed', root / 'outside'
            allowed.mkdir()
            outside.mkdir()
            (allowed / 'source').write_bytes(b'same inode')
            os.link(allowed / 'source', outside / 'source')
            alias = allowed / 'alias'
            alias.symlink_to(allowed / 'source')
            digest = c.content_sha(alias)
            cache_size = len(c._cache)
            alias.unlink()
            alias.symlink_to(outside / 'source')
            self.assertEqual(c.content_sha(alias), digest)
            self.assertEqual(len(c._cache), cache_size)
            (outside / 'queue').mkdir()
            (outside / 'out').mkdir()
            (allowed / 'escape').symlink_to(outside, target_is_directory=True)
            cfg = root / 'launch.json'
            cfg.write_text(json.dumps({'gpus': [3], 'queue': str(allowed / 'escape/queue'),
                                      'out': str(allowed / 'escape/out')}))
            source = c.RT / 'resource_manager.py'
            spec = importlib.util.spec_from_file_location('test_original_domain_manager', source)
            manager = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(manager)
            argv = [str(source), '--config', str(cfg), '--config-sha', manager.sha(cfg),
                    '--manager-sha', manager.sha(source)]
            with patch.object(manager, 'EXPERIMENT', allowed), \
                 patch.object(manager, 'validate_pair', lambda value: None), \
                 patch.object(sys, 'argv', argv):
                with self.assertRaisesRegex(RuntimeError, 'paths outside current experiment'):
                    manager.main()

    def test_current_resources_are_verified_after_all_source_checks(self):
        calls = []
        cfg = {'holder_capture': {'path': '/historical', 'sha256': 'x'},
               'holders': [{'pid': 7, 'uuid': 'physical3'}], 'lease_snapshot_ns': {'/lease': None},
               'transition_ref': {'path': '/transition', 'sha256': 'y'}, 'launch_path': '/new-launch'}
        old = {'signals_sent': False, 'holders': cfg['holders'],
               'lease_snapshot_ns': cfg['lease_snapshot_ns'], 'budget': {'headroom': 200 * 2**30},
               'captured_unix_ns': time.time_ns() - 7200 * 10**9}
        fresh = {**old, 'captured_unix_ns': time.time_ns()}
        def need(ok, message):
            if not ok:
                raise RuntimeError(message)
        fake = SimpleNamespace(source_plan=lambda: calls.append('sources') or {},
            verify_config=lambda *a: calls.append('config'), check_node=lambda: calls.append('host'),
            verify_local_model=lambda *a: calls.append('model'),
            prior_safe=lambda *a: calls.append('history') or {}, checked=lambda *a: old,
            require_unreviewed=lambda *a: calls.append('once'),
            fresh_holder=lambda *a: calls.append('fresh-current-resources') or fresh, need=need,
            ref=lambda p: {'path': str(p), 'sha256': 'checked'},
            save_once=lambda *a: calls.append('durable-current-proof'))
        base = SimpleNamespace(check_holder=lambda *a: calls.append('current-exact-holder'), apps=lambda: [])
        manager = SimpleNamespace(c=fake, base=base)
        c.fresh_current_validate(manager, cfg)
        self.assertLess(calls.index('history'), calls.index('fresh-current-resources'))
        self.assertLess(calls.index('model'), calls.index('fresh-current-resources'))
        self.assertLess(calls.index('current-exact-holder'), calls.index('durable-current-proof'))
        fresh['captured_unix_ns'] = time.time_ns() - 61 * 10**9
        with self.assertRaisesRegex(RuntimeError, 'Current resource capture already stale'):
            c.fresh_current_validate(manager, cfg)
        fresh['captured_unix_ns'] = time.time_ns()
        fresh['lease_snapshot_ns'] = {'/lease': {'different': True}}
        with self.assertRaisesRegex(RuntimeError, 'Original exact lease changed'):
            c.fresh_current_validate(manager, cfg)


if __name__ == '__main__':
    unittest.main()
