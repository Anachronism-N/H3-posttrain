"""CPU fake-byte/mocked tests. Never bootstrap, capture, manage, SSH, or GPU."""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


m = module('render_test_subject', Path(__file__).with_name('music_planner_h3_render_manager.py'))
audit = module('render_test_audit', m.EVAL)
h3 = module('render_test_infer', m.RENDER / 'h3_video_infer.py')
SAFE = dict(all_holders_safe=True, locks_released=True, leases_restored=True,
            managed_parent_wait=True, unknown_jobs_not_signaled=True, admitted_gpu_count=1)


class CpuTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return h3.source(path)

    def test_wrong_frozen_source_sha(self):
        upstream = self.root / 'upstream'; upstream.mkdir()
        files = []
        for index in range(288):
            p = upstream / (str(index) + '.py'); p.write_bytes(b'fake bytes')
            files.append(h3.source(p))
        render = self.root / 'renderer'; render.mkdir()
        manifest = render / 'upstream_python_manifest.json'
        manifest.write_text(json.dumps(dict(files=files)))
        source = self.root / 'source.py'; source.write_bytes(b'fake code')
        config = self.write('config.json', {})
        own = audit.sha(m.__file__)
        frozen = {str(manifest): audit.sha(manifest), str(source): audit.sha(source)}
        bindings = dict(frozen, **{str(Path(m.__file__).resolve()): own})
        cfg = dict(source_sha256=bindings, environment_refs=[dict(path=str(source), sha256=frozen[str(source)])])
        with patch.object(m, 'RENDER', render), patch.object(m, 'PINNED', frozen), patch.object(audit, 'PINNED', {}):
            m.check_sources(cfg, config['path'], config['sha256'], own, audit)
            source.write_bytes(b'drifted bytes')
            with self.assertRaisesRegex(RuntimeError, 'SHA changed'):
                m.check_sources(cfg, config['path'], config['sha256'], own, audit)
            self.assertNotEqual(audit.sha(source), frozen[str(source)])

    def fixture(self):
        cases, captions, raw = [], [], []
        for index, sid in enumerate(m.IDS):
            image = self.root / (sid + '.png'); image.write_bytes(b'fake image')
            audio = self.root / (sid + '.wav'); audio.write_bytes(b'fake audio')
            cases.append(dict(id=sid, image_path=str(image), image_ref=h3.source(image), audio_ref=h3.source(audio)))
            captions.append(dict(id=sid, image_path=str(image), motion_caption='  Native exact dance prose.  ', caption_version='native-v1'))
            raw.append(dict(id=sid, raw_text=captions[-1]['motion_caption'], image_ref=cases[-1]['image_ref'], audio_ref=cases[-1]['audio_ref']))
        records_ref = self.write('h3_records.json', dict(records=captions))
        raw_path = self.root / 'raw.jsonl'
        raw_path.write_text(''.join(json.dumps(r) + '\n' for r in raw))
        raw_ref = h3.source(raw_path)
        values = {str(m.HANDOFF / 'protocol.json'): dict(fixed3=cases, data={})}
        cfg = dict(schema='music-planner-h3-render-manager-v1', node='N2', host='28.216.17.70',
                   gpus=[0], release_authorized=True, protocol=dict(path=str(m.HANDOFF / 'protocol.json'),
                   sha256=m.PINNED[str(m.HANDOFF / 'protocol.json')]), families={},
                   final_selection=dict(path=str(m.HANDOFF / 'final_selection.json'), sha256='mock'))
        completes = {}
        for node, (host, family) in audit.SCOPE.items():
            row = dict(fingerprint='fake', training_launch=dict(path=str(m.RT / (family + '_launch_clean_v11.json'))),
                full_complete=dict(path=str(m.EXP / 'queues' / (family + '_fixed2s_clean_v11') / 'full.complete.json')),
                resources_safe=dict(path=str(m.EXP / 'queues' / (family + '_fixed2s_clean_v11') / 'resources_safe.json')),
                checkpoint_selection=dict(path=str(m.EXP / 'runs' / (family + '_fixed2s_clean_v11') / 'train/best.json')),
                checkpoint_marker=dict(path='fake marker'), validation_candidates=[dict(path='fake%d' % i, sha256='mock') for i in range(4)], h3_records=records_ref, raw=raw_ref,
                manifest=dict(path=str(self.root / 'manifest.json'), sha256='mock'),
                generation_complete=dict(path=str(m.HANDOFF / (family + '.queue') / 'eval.complete.json'), sha256='mock'),
                generation_resources_safe=dict(path=str(m.HANDOFF / (family + '.queue') / 'resources_safe.json'), sha256='mock'),
                generation_exit=dict(path=str(m.HANDOFF / (family + '.queue') / 'fixed3.exit.json'), sha256='mock'))
            complete = dict(manifest=row['manifest'], raw=raw_ref, technical_collection_complete=True,
                            complete_native_GPU_outputs_verified=True, human_quality_accepted=False)
            completes[family] = complete
            values[row['generation_complete']['path']] = complete
            values[row['generation_resources_safe']['path']] = dict(SAFE)
            values[row['generation_exit']['path']] = dict(exit_code=0)
            values[row['manifest']['path']] = dict(complete=True, h3_handoff_approved=True,
                     pretrained_GPU_generation_executed=True, h3_records=records_ref)
            values[records_ref['path']] = dict(records=captions)
            cfg['families'][family] = row
        values[cfg['final_selection']['path']] = dict(schema='music-planner-final-selection-v1',
            test_used=False, selection_frozen_before_test=True, protocol=cfg['protocol'], data={}, decoding=audit.DECODE,
            families={f: {k: r[k] for k in ('training_launch', 'full_complete', 'resources_safe',
                     'checkpoint_selection', 'checkpoint_marker', 'fingerprint')} for f, r in cfg['families'].items()})
        fake = SimpleNamespace(stage=SimpleNamespace(validate_scope=Mock()), release_holder=Mock(), run_child=Mock())
        return cfg, values, completes, fake

    def check_fixture(self, cfg, values, completes, manager):
        with patch.object(audit, 'ref_json', side_effect=lambda r: values[r['path']]), \
             patch.object(audit, 'check_completed', return_value=({}, {})), \
             patch.object(audit, 'selected_checkpoint', return_value=(self.root, {}, dict(validation_candidates=[dict(path='fake%d' % i, sha256='mock') for i in range(4)]))), \
             patch.object(audit, 'check_worker_outputs', side_effect=lambda scope, *args: completes[scope['family']]), \
             patch.object(m, 'image_geometry'):
            return m.prerequisites(cfg, audit, manager, Mock(), Mock(), h3)

    def test_three_native_families_keep_raw_caption_verbatim(self):
        cfg, values, complete, manager = self.fixture()
        got = self.check_fixture(cfg, values, complete, manager)
        self.assertEqual(len(got), 3)
        self.assertEqual(next(iter(got.values()))[0]['motion_caption'], '  Native exact dance prose.  ')
        manager.release_holder.assert_not_called(); manager.run_child.assert_not_called()

    def test_any_fixed3_not_complete_refuses_gpu(self):
        for family in audit.SCOPE.values():
            cfg, values, complete, manager = self.fixture()
            complete[family[1]]['technical_collection_complete'] = False
            with self.assertRaisesRegex(RuntimeError, 'not complete'):
                self.check_fixture(cfg, values, complete, manager)
            manager.release_holder.assert_not_called(); manager.run_child.assert_not_called()

    def test_final_selection_or_nonzero_exit_refuses_gpu(self):
        for failure in ('final', 'exit'):
            cfg, values, complete, manager = self.fixture()
            if failure == 'final':
                values[cfg['final_selection']['path']]['families']['moss']['fingerprint'] = 'changed'
            else:
                values[cfg['families']['moss']['generation_exit']['path']]['exit_code'] = 1
            with self.assertRaises(RuntimeError):
                self.check_fixture(cfg, values, complete, manager)
            manager.release_holder.assert_not_called(); manager.run_child.assert_not_called()

    def test_no_three_or_caption_sha_changed(self):
        cfg, values, complete, manager = self.fixture()
        ref = next(iter(cfg['families'].values()))['h3_records']
        values[ref['path']]['records'].pop()
        with self.assertRaisesRegex(RuntimeError, 'exactly three'):
            self.check_fixture(cfg, values, complete, manager)
        cfg, values, complete, manager = self.fixture()
        ref = next(iter(cfg['families'].values()))['h3_records']
        value = json.loads(Path(ref['path']).read_text())
        value['records'][0]['motion_caption'] = 'Cleaned or truncated caption'
        Path(ref['path']).write_text(json.dumps(value))
        self.assertNotEqual(audit.sha(ref['path']), ref['sha256'])
        with self.assertRaisesRegex(RuntimeError, 'receipt SHA changed'):
            audit.ref_json(ref)
        with self.assertRaisesRegex(RuntimeError, 'caption/image changed'):
            self.check_fixture(cfg, values, complete, manager)
        manager.release_holder.assert_not_called()

    def test_busy_existing_flock_and_unsafe_symlink(self):
        lock = self.root / 'gpu0.lock'; lock.touch()
        with patch.object(audit, 'GPU0_LOCK', lock):
            fd = audit.open_gpu0_lock()
            try:
                with self.assertRaises(BlockingIOError):
                    audit.open_gpu0_lock()
                audit.check_gpu0_lock(fd)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                import os
                os.close(fd)
        link = self.root / 'bad.lock'; link.symlink_to(lock)
        with patch.object(audit, 'GPU0_LOCK', link), self.assertRaises(RuntimeError):
            audit.open_gpu0_lock()

    def test_actual_renderer_API_cumulative_counts_and_bad_metadata(self):
        captions = []
        ref = self.write('caption.json', {})
        out = self.root / 'video'; out.mkdir()
        cache = self.root / 'cache'; cache.mkdir()
        media = dict(verified=True, verification_method='CPU mock decode', frames=124, height=1280,
                     width=704, fps=24, video_streams=1, audio_streams=0, complete_decode=True)
        models = dict(files=[], weights_root=str(m.WEIGHTS), audio_vae_loaded=False)
        for index, sid in enumerate(m.IDS, 1):
            image = self.root / (sid + '.png'); image.write_bytes(b'fake image')
            caption = dict(id=sid, image_path=str(image), motion_caption='Exact caption.', caption_version='v1')
            captions.append(caption)
            (cache / (sid + '.condition.pt')).write_bytes(b'fake condition')
            (out / (sid + '.mp4')).write_bytes(b'fake video')
            case = dict(protocol=h3.PROTOCOL, id=sid, mode='video_generation', complete=True,
                motion_caption=caption['motion_caption'], caption_version='v1', prompt='[Shot 1] ' + caption['motion_caption'],
                sources=dict(caption_json=ref, image=h3.source(image)), condition_cache=h3.source(cache / (sid + '.condition.pt')),
                packed=dict(audio_pos_numel=0, tag2_count=0),
                audio_guard=dict(audio_vae_is_none=True, scheduler_audio_is_none=True, calls=dict(audio_head=0),
                      observed_video_calls=dict(text_encoder=0, dit=index * 50, video_vae_decode=index, video_vae_encode=0)),
                sampling=dict(video_scheduler_updates=50, target_audio_generated=False),
                video=h3.source(out / (sid + '.mp4')), media=media)
            (out / (sid + '.audit.json')).write_text(json.dumps(case))
        report = dict(protocol=h3.PROTOCOL, mode='video_generation', complete=True, completed_ids=m.IDS,
                      geometry=m.GEOMETRY, steps=50, seed=42, cfg=1., flow_shift=12., lora_source=None,
                      old_y_reused=False, model_sources=models, caption_source=ref, audio_guard=case['audio_guard'],
                      source_code=h3.source(m.RENDER / 'h3_video_infer.py'), wrapper_source=h3.source(m.RENDER / 'h3_video_only.py'))
        (out / 'run_complete.json').write_text(json.dumps(report))
        with patch.object(h3, 'verify_media', return_value=media):
            m.check_stage(out, 'video_generation', captions, ref, models, audit, h3, cache)
            for key, bad in (('steps', 49), ('complete', False), ('completed_ids', m.IDS[:2]), ('lora_source', {})):
                wrong = dict(report, **{key: bad})
                (out / 'run_complete.json').write_text(json.dumps(wrong))
                with self.assertRaises(RuntimeError):
                    m.check_stage(out, 'video_generation', captions, ref, models, audit, h3, cache)
            (out / 'run_complete.json').write_text(json.dumps(report))
            wrong = out / (m.IDS[0] + '.audit.json')
            case = json.loads(wrong.read_text()); case['sampling']['video_scheduler_updates'] = 0
            wrong.write_text(json.dumps(case))
            with self.assertRaisesRegex(RuntimeError, 'sampling differs'):
                m.check_stage(out, 'video_generation', captions, ref, models, audit, h3, cache)


if __name__ == '__main__':
    unittest.main(verbosity=2)
