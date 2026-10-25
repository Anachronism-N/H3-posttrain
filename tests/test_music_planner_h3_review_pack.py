"""CPU-only fixtures/mocks and one synthetic ffmpeg mux; never actual H3 gate/GPU."""
import copy
import importlib.util
import json
import math
import struct
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts_omnidance/exp/ms/music_planner_h3_review_pack.py'
spec = importlib.util.spec_from_file_location('review_pack_test_subject', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ReviewPackTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return m.source(path)

    def wav(self, path):
        with wave.open(str(path), 'wb') as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(24000)
            stream.writeframes(b''.join(struct.pack('<h', int(12000 * math.sin(2 * math.pi * 440 * n / 24000)))
                                        for n in range(121500)))
        return dict(channels=1, sample_rate=24000, frames=121500, duration_seconds=5.0625)

    def fixture(self):
        handoff = self.root / 'handoff'
        queue, out = handoff / 'h3_compare_v1/queue', handoff / 'h3_compare_v1/out'
        queue.mkdir(parents=True)
        out.mkdir()
        cfg = dict(queue=str(queue), out=str(out), families={}, weight_pre_audit={})
        cases, records, videos, receipts = [], {}, {}, []
        for sid in m.IDS:
            audio = self.root / (sid + '.wav')
            header = self.wav(audio)
            cases.append(dict(id=sid, audio_ref=m.source(audio), audio_header=header))
        cfg['protocol'] = self.write(handoff / 'protocol.json', dict(fixed3=cases))
        model = dict(files=[], weights_root=str(self.root), sha256='already full-SHA audited')
        for stage in ('cache', 'video'):
            cfg['weight_pre_audit'][stage] = self.write(self.root / (stage + '.models.json'), model)
        for family in sorted(m.FAMILIES):
            records[family] = [dict(id=sid) for sid in m.IDS]
            cfg['families'][family] = dict(h3_records=self.write(self.root / (family + '.records.json'), records[family]))
            for stage in ('cache', 'video'):
                destination = out / family / stage
                run_ref = self.write(destination / 'run_complete.json', dict(complete=True, completed_ids=m.IDS))
                self.write(queue / (family + '.' + stage + '.complete.json'), dict(report=run_ref, completed_ids=m.IDS))
                self.write(queue / (family + '.' + stage + '.exit.json'), dict(exit_code=0))
            for sid in m.IDS:
                original = out / family / 'video' / (sid + '.mp4')
                original.write_bytes((family + sid + 'private').encode())
                audit = self.write(original.with_suffix('.audit.json'), dict(prompt='PRIVATE PROSE', complete=True))
                videos[family, sid] = dict(video=m.source(original), case_audit=audit)
        config = handoff / 'h3_compare_v1/run_config_v3.json'
        config_ref = self.write(config, cfg)
        safe = dict(all_holders_safe=True, locks_released=True, leases_restored=True,
                    managed_parent_wait=True, unknown_jobs_not_signaled=True, admitted_gpu_count=1)
        safe_ref = self.write(queue / 'resources_safe.json', safe)
        complete = dict(technical_video_complete=True, families=sorted(m.FAMILIES), completed_ids=m.IDS,
                        human_quality_accepted=False, human_quality=False, winner=None,
                        config_sha256=config_ref['sha256'], manager_sha256=m.MANAGER_SHA,
                        other_seven_GPUs_untouched=True, model_sources=dict(cache=model, video=model))
        complete_ref = self.write(queue / 'h3.compare.complete.json', complete)
        h3 = SimpleNamespace(source=m.source, ffmpeg_binary=Mock(return_value='fake-ffmpeg'),
                             weight_sources=Mock(side_effect=AssertionError('must not rehash weights')))
        def ref_json(ref):
            self.assertEqual(m.source(ref['path'])['sha256'], ref['sha256'])
            return m.read_json(ref['path'])
        audit = SimpleNamespace(ref_json=ref_json)
        def safe_resources(ref, count, audit):
            m.require(audit.ref_json(ref), dict(safe, admitted_gpu_count=count), 'resources unsafe')
        fake = SimpleNamespace(bootstrap=Mock(return_value=(audit, Mock(), Mock(), Mock(), h3)),
            safe_resources=Mock(side_effect=safe_resources), check_sources=Mock(),
            prerequisites=Mock(return_value=records), check_stage=Mock(), WEIGHTS=self.root)
        patches = [patch.object(m, 'HANDOFF', handoff), patch.object(m, 'CONFIG', config),
                   patch.object(m, 'CONFIG_SHA', config_ref['sha256']), patch.object(m, 'frozen_manager', return_value=fake)]
        for item in patches:
            item.start(); self.addCleanup(item.stop)
        verified = dict(cases=cases, videos=videos, receipts=[complete_ref, safe_ref])
        return cfg, queue, out, fake, h3, verified

    def test_gate_requires_real_nine_video_stages_and_no_weight_rehash(self):
        cfg, queue, out, fake, h3, _ = self.fixture()
        verified, returned_h3 = m.verify_complete()
        self.assertEqual(len(verified['videos']), 9)
        self.assertIs(returned_h3, h3)
        self.assertEqual(fake.check_stage.call_count, 6)
        self.assertEqual([c.args[1] for c in fake.check_stage.call_args_list].count('video_generation'), 3)
        h3.weight_sources.assert_not_called()

    def test_gate_missing_completion_missing_video_exit_and_bad_sha_refuse(self):
        cfg, queue, out, fake, h3, _ = self.fixture()
        family = sorted(m.FAMILIES)[0]
        paths = [queue / 'h3.compare.complete.json', queue / (family + '.video.exit.json'),
                 out / family / 'video' / (m.IDS[0] + '.mp4')]
        for path in paths:
            original = path.read_bytes(); path.unlink()
            with self.assertRaises((OSError, RuntimeError)):
                m.verify_complete()
            path.write_bytes(original)
        path = out / family / 'video/run_complete.json'
        original = path.read_bytes(); path.write_bytes(original + b' ')
        with self.assertRaisesRegex(RuntimeError, 'run_complete SHA'):
            m.verify_complete()
        h3.weight_sources.assert_not_called()

    def test_each_unsafe_field_refuses_and_cache_complete_is_not_video_complete(self):
        cfg, queue, out, fake, _, _ = self.fixture()
        safe_path = queue / 'resources_safe.json'
        safe = m.read_json(safe_path)
        for key in safe:
            bad = dict(safe); bad[key] = 0 if key == 'admitted_gpu_count' else False
            safe_path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(RuntimeError, 'unsafe'):
                m.verify_complete()
        safe_path.write_text(json.dumps(safe))
        fake.check_stage.reset_mock()
        family = sorted(m.FAMILIES)[0]
        self.write(queue / (family + '.video.exit.json'), dict(exit_code=1))
        with self.assertRaisesRegex(RuntimeError, 'exit is not zero'):
            m.verify_complete()
        self.assertEqual(fake.check_stage.call_count, 1)  # Only cache checked; no video acceptance.

    def test_gate_rejects_false_technical_acceptance_or_model_drift(self):
        _, queue, _, _, _, _ = self.fixture()
        path = queue / 'h3.compare.complete.json'
        baseline = m.read_json(path)
        for key, value in [('technical_video_complete', False), ('human_quality_accepted', True),
                           ('winner', 'A'), ('families', ['moss']), ('completed_ids', m.IDS[:2]),
                           ('model_sources', {})]:
            bad = dict(baseline); bad[key] = value; path.write_text(json.dumps(bad))
            with self.assertRaises(RuntimeError):
                m.verify_complete()

    def test_no_output_before_gate_and_existing_paths_never_overwritten(self):
        public, private = self.root / 'public', self.root / 'private.json'
        with patch.object(m, 'verify_complete', side_effect=RuntimeError('incomplete')):
            with self.assertRaises(RuntimeError):
                m.build_pack(public, private)
        self.assertFalse(public.exists()); self.assertFalse(private.exists())
        for existing in (public, private):
            existing.write_text('KEEP')
            with patch.object(m, 'verify_complete') as gate:
                with self.assertRaises(RuntimeError):
                    m.build_pack(public, private)
                gate.assert_not_called()
            self.assertEqual(existing.read_text(), 'KEEP'); existing.unlink()
        with self.assertRaises(RuntimeError):
            m.output_paths(public, public / 'private.json')
        private.symlink_to(self.root / 'missing')
        with self.assertRaises(RuntimeError):
            m.output_paths(public, private)

    def test_anonymous_public_pack_private_exclusive_key_and_unfilled_scores(self):
        _, _, _, _, h3, verified = self.fixture()
        public, private = self.root / 'public', self.root / 'private.json'
        def fake_mux(command, **kwargs):
            Path(command[-1]).write_bytes(b'FAKE COPIED VIDEO: CPU MOCK ONLY')
            return SimpleNamespace(returncode=0)
        media = dict(width=704, height=1280, frames=124, fps=24, video_duration_seconds=124 / 24)
        with patch.object(m, 'verify_complete', return_value=(verified, h3)), \
             patch.object(m.subprocess, 'run', side_effect=fake_mux) as mux, \
             patch.object(m, 'verify_copy', return_value=media):
            m.build_pack(public, private)
        self.assertEqual(mux.call_count, 9)
        self.assertEqual(private.stat().st_mode & 0o777, 0o600)
        self.assertEqual(public.stat().st_mode & 0o777, 0o700)
        manifest = m.read_json(public / 'manifest.json')
        self.assertEqual(len(manifest['cases']), 3)
        self.assertEqual(manifest['axes'], m.AXES)
        text = (public / 'index.html').read_text() + (public / 'manifest.json').read_text()
        for token in [*m.FAMILIES, *m.IDS, str(self.root), 'PRIVATE PROSE', 'winner', 'accepted']:
            self.assertNotIn(token, text)
        for row in manifest['cases']:
            self.assertEqual([v['label'] for v in row['videos']], list('ABC'))
            for video in row['videos']:
                self.assertTrue((public / video['file']).is_file())
                self.assertEqual(video['frames'], 124)
        key = m.read_json(private)
        self.assertEqual(len(key['cases']), 3)
        self.assertRegex(manifest['bundle_id'], r'^[a-f0-9]{32}$')
        self.assertEqual(key['bundle_id'], manifest['bundle_id'])
        self.assertIn("const bundleId='" + manifest['bundle_id'] + "';", text)
        self.assertIn('bundle_id:bundleId', text)
        self.assertIn('有损编码', text)
        for public_case, private_case in zip(manifest['cases'], key['cases']):
            audio = public / public_case['original_audio']['file']
            self.assertEqual(m.source(audio)['sha256'], private_case['audio_ref']['sha256'])
            self.assertEqual(audio.read_bytes(), Path(private_case['audio_ref']['path']).read_bytes())
            self.assertEqual(public_case['original_audio']['sha256'], private_case['audio_ref']['sha256'])
        second_public, second_private = self.root / 'public2', self.root / 'private2.json'
        with patch.object(m, 'verify_complete', return_value=(verified, h3)), \
             patch.object(m.subprocess, 'run', side_effect=fake_mux), \
             patch.object(m, 'verify_copy', return_value=media):
            m.build_pack(second_public, second_private)
        self.assertNotEqual(m.read_json(second_public / 'manifest.json')['bundle_id'], manifest['bundle_id'])
        self.assertEqual({v['family'] for v in key['cases'][0]['videos']}, m.FAMILIES)
        self.assertFalse(private.is_relative_to(public))
        before = private.read_bytes()
        with self.assertRaises(RuntimeError):
            m.build_pack(public, private)
        self.assertEqual(private.read_bytes(), before)
        for call in mux.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index('-c:v') + 1], 'copy')
            self.assertEqual(command[command.index('-c:a') + 1], 'aac')
            self.assertNotIn('-shortest', command); self.assertNotIn('-t', command)
            self.assertNotIn('-ss', command); self.assertIn('-n', command)
            self.assertEqual(command[command.index('-map_metadata') + 1], '-1')
        self.assertIn(".value||null", text)
        self.assertIn("['all_unusable','都不可用']", text)
        self.assertIn("['unknown','无法判断']", text)
        self.assertNotIn('fetch(', text); self.assertNotIn('XMLHttpRequest', text)
        self.assertIn("connect-src 'none'", text)

    def test_changed_original_audio_refuses_before_media_and_no_index(self):
        _, _, _, _, h3, verified = self.fixture()
        audio = Path(verified['cases'][0]['audio_ref']['path'])
        with audio.open('r+b') as stream:
            stream.seek(50)
            stream.write(b'\x7f\x01')
        public, private = self.root / 'public', self.root / 'private.json'
        with patch.object(m, 'verify_complete', return_value=(verified, h3)), \
             patch.object(m.secrets.SystemRandom, 'shuffle', return_value=None), \
             patch.object(m.subprocess, 'run') as mux:
            with self.assertRaisesRegex(RuntimeError, 'original audio changed'):
                m.build_pack(public, private)
            mux.assert_not_called()
        self.assertFalse((public / 'index.html').exists())
        self.assertFalse((public / 'manifest.json').exists())

    def test_metadata_leak_and_packet_changes_are_rejected(self):
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            self.skipTest('ffmpeg unavailable')
        info = 'Input #0\n  Stream #0:0: Video: h264\n  Stream #0:1: Audio: aac, 24000 Hz, mono\n PRIVATE PROSE'
        h3 = SimpleNamespace(verify_media=Mock())
        header = dict(channels=1, sample_rate=24000, frames=121500)
        with patch.object(m.subprocess, 'run', return_value=SimpleNamespace(stderr=info)):
            with self.assertRaisesRegex(RuntimeError, 'identity leaked'):
                m.verify_copy('original', 'copy', header, ffmpeg, h3, ['PRIVATE PROSE'])
        info = info.replace(' PRIVATE PROSE', '')
        with patch.object(m.subprocess, 'run', return_value=SimpleNamespace(stderr=info)), \
             patch.object(m, 'packet_frames', side_effect=[['before'], ['after']]):
            with self.assertRaisesRegex(RuntimeError, 'packets/PTS/duration'):
                m.verify_copy('original', 'copy', header, ffmpeg, h3, [])

    def test_reused_frozen_check_stage_refuses_protocol_and_audio_guard_drift(self):
        frozen = m.frozen_manager()  # Import only; never bootstrap or manage.
        directory, cache = self.root / 'video', self.root / 'cache'
        directory.mkdir(); cache.mkdir()
        caption_ref = self.write(self.root / 'captions.json', {})
        models, records = dict(files=[]), []
        media = dict(verified=True, frames=124, height=1280, width=704, fps=24,
                     video_streams=1, audio_streams=0, complete_decode=True)
        h3 = SimpleNamespace(read_json=m.read_json, source=m.source, PROTOCOL='h3_caption_image_infer_v1',
                             verify_media=Mock(return_value=media))
        for index, sid in enumerate(m.IDS, 1):
            image = self.root / (sid + '.png'); image.write_bytes(b'CPU fake image')
            record = dict(id=sid, image_path=str(image), motion_caption='Private dance prose.', caption_version='v1')
            records.append(record)
            video = directory / (sid + '.mp4'); video.write_bytes(b'CPU fake video')
            condition = cache / (sid + '.condition.pt'); condition.write_bytes(b'CPU fake cache')
            guard = dict(calls=dict(audio_patch=0, audio_head=0, audio_reference_encode=0),
                         audio_vae_is_none=True, scheduler_audio_is_none=True,
                         observed_video_calls=dict(text_encoder=0, dit=50 * index,
                                                   video_vae_decode=index, video_vae_encode=0))
            self.write(directory / (sid + '.audit.json'), dict(protocol=h3.PROTOCOL, id=sid,
                mode='video_generation', complete=True, motion_caption=record['motion_caption'], caption_version='v1',
                sources=dict(caption_json=caption_ref, image=m.source(image)), condition_cache=m.source(condition),
                packed=dict(audio_pos_numel=0, tag2_count=0), audio_guard=guard,
                prompt='[Shot 1] ' + record['motion_caption'], sampling=dict(video_scheduler_updates=50,
                target_audio_generated=False), video=m.source(video), media=media))
        report = dict(protocol=h3.PROTOCOL, mode='video_generation', complete=True, completed_ids=m.IDS,
                      geometry=m.GEOMETRY, steps=50, seed=42, cfg=1., flow_shift=12., lora_source=None,
                      old_y_reused=False, model_sources=models, caption_source=caption_ref, audio_guard=guard,
                      source_code=m.source(frozen.RENDER / 'h3_video_infer.py'),
                      wrapper_source=m.source(frozen.RENDER / 'h3_video_only.py'))
        run = directory / 'run_complete.json'
        self.write(run, report)
        frozen.check_stage(directory, 'video_generation', records, caption_ref, models, Mock(), h3, cache)
        for key, value in [('geometry', dict(m.GEOMETRY, num_frames=123)), ('steps', 49), ('seed', 43),
                           ('cfg', 3.), ('flow_shift', 11.), ('lora_source', {}), ('old_y_reused', True),
                           ('model_sources', dict(files=['drift']))]:
            self.write(run, dict(report, **{key: value}))
            with self.assertRaises(RuntimeError):
                frozen.check_stage(directory, 'video_generation', records, caption_ref, models, Mock(), h3, cache)
        self.write(run, report)
        case_path = directory / (m.IDS[0] + '.audit.json')
        case = m.read_json(case_path)
        case['audio_guard']['calls']['audio_patch'] = 1
        self.write(case_path, case)
        with self.assertRaises(RuntimeError):
            frozen.check_stage(directory, 'video_generation', records, caption_ref, models, Mock(), h3, cache)

    def test_html_javascript_syntax_without_browser_network(self):
        node = shutil.which('node')
        if not node:
            self.skipTest('node unavailable')
        script = re.search(r'<script>(.*?)</script>', m.html_page('0123456789abcdef0123456789abcdef'), re.S).group(1)
        result = subprocess.run([node, '--check', '-'], input=script, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_cpu_ffmpeg_synthetic_copy_all_frames_audio_no_reencode(self):
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            self.skipTest('ffmpeg unavailable')
        original, audio, destination = self.root / 'source.mp4', self.root / 'source.wav', self.root / 'copy.mp4'
        header = self.wav(audio)
        subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
                        '-f', 'lavfi', '-i', 'color=c=black:s=704x1280:r=24', '-frames:v', '124',
                        '-an', '-c:v', 'libx264', '-threads', '1', '-preset', 'ultrafast',
                        '-pix_fmt', 'yuv420p', '-metadata', 'comment=PRIVATE PROSE', str(original)],
                       capture_output=True, check=True)
        original_ref, audio_ref = m.source(original), m.source(audio)
        subprocess.run(m.mux_command(ffmpeg, original, audio, destination), capture_output=True, check=True)
        infer_path = m.EXP / 'runtime/h3_renderer_v10/h3_video_infer.py'
        infer_spec = importlib.util.spec_from_file_location('review_cpu_media_verifier', infer_path)
        h3 = importlib.util.module_from_spec(infer_spec); infer_spec.loader.exec_module(h3)
        with patch.object(h3, 'ffmpeg_binary', return_value=ffmpeg):
            media = m.verify_copy(original, destination, header, ffmpeg, h3, ['PRIVATE PROSE'])
        self.assertEqual(media['frames'], 124)
        self.assertEqual((media['width'], media['height']), (704, 1280))
        self.assertEqual(media['video_duration_seconds'], 124 / 24)
        self.assertGreaterEqual(media['audio_decoded_duration_seconds'], 5.0625)
        self.assertEqual(m.source(original), original_ref); self.assertEqual(m.source(audio), audio_ref)


if __name__ == '__main__':
    unittest.main()
