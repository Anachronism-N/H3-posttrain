"""Full native three-source transaction fixture: fake external decode/chat, CPU only."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import caption_batch8_direct_full_pilot_v1 as pilot
import caption_direct_facts_v1 as direct
from test_caption_cross_source_stage_v1 import Fixture, FakeLLM
bar, pipeline = direct.native()


class FullFixture(Fixture):
    def __init__(self, base, empty_window=False, masked=False):
        super().__init__(base, empty_window=empty_window, masked=masked)
        self.original = Path(base) / 'original'
        self.out = Path(base) / 'new-pilot'
        self.args.out = str(self.out)
        self.args.hand_roi_root = str(self.original / 'roi')
        Path(self.args.hand_roi_root).mkdir(parents=True)
        self.sources = tuple(e['row']['id'] for e in self.entries)
        self.styles = []
        for shard, entry in enumerate(self.entries):
            plan = entry['row']
            if empty_window and shard == 0:
                plan['windows'][1]['start'] = plan['windows'][0]['end']
                plan['window_manifest']['windows'] = copy.deepcopy(plan['windows'])
            plan['manifest_record']['batch_id'] = 'train-b000000'
            source = pipeline.source_dir(self.original, plan['manifest_record'])
            plan_path = pipeline.save(source / 'bar_plan.json', plan)
            roi_path = pipeline.save(self.original / 'roi' / plan['id'] / 'roi_manifest.json',
                                     dict(id=plan['id'], crops=[]))
            pipeline.save(roi_path.with_name('source_manifest.json'), {'cpu_fixture': True})
            pipeline.save(roi_path.with_name('extract_receipt.json'), {'cpu_fixture': True})
            producer = dict(id=plan['id'], status='ready', batch_id='train-b000000',
                source_binding=plan['binding'], plan_ref=pipeline.ref(plan_path), roi_ref=pipeline.ref(roi_path))
            pipeline.save(source / 'producer.terminal.json', producer)
            actual = dict(source_identity=bar.batch.source_identity(plan['manifest_record']),
                          video_probe=self.probes[plan['manifest_record']['video_path']], audio_probe={})
            pipeline.save(source / 'actual_media.json', actual)
            style = dict(id=plan['id'], status='candidate' if shard == 1 else 'resolved',
                dance_style=('contemporary', 'k-pop', 'contemporary')[shard],
                source=('rich_chapter2', 'independent_RGB', 'rich_chapter2')[shard],
                metadata=dict(video_sha256=plan['binding']['video_sha256'], source_refs={}), **direct.FLAGS)
            self.styles.append(style)
            consumer = self.original / 'consumers' / ('shard-%04d' % shard) / 'sources/train' / plan['id']
            style_path = pipeline.save(consumer / 'original_style.json', style)
            joint = dict(id=plan['id'], split='train', batch_id='train-b000000', style_result=style,
                         style_result_ref=pipeline.ref(style_path))
            joint_path = pipeline.save(consumer / 'original_joint.json', joint)
            pipeline.save(consumer / 'consumer.terminal.json', dict(id=plan['id'], split='train',
                joint_candidate_ref=pipeline.ref(joint_path), batch_id='train-b000000'))
        self.original_bytes = {p: p.read_bytes() for p in self.original.rglob('*') if p.is_file()}
        self.native_load_roi = bar.load_hand_roi
        self.roi_calls = []

    def context(self):
        stack = ExitStack()
        stack.enter_context(patch.object(pilot, 'ORIGINAL_ROOT', self.original))
        stack.enter_context(patch.object(pilot, 'DEFAULT_OUT', self.out))
        stack.enter_context(patch.object(pilot, 'SOURCES', self.sources))
        stack.enter_context(patch.object(bar, 'probe_actual', side_effect=lambda path, _: copy.deepcopy(self.probes[path])))
        stack.enter_context(patch.object(bar.subprocess, 'run', side_effect=self.decode))
        stack.enter_context(patch.object(bar.core, 'structured_sampling', side_effect=self.sampling))
        def roi(row, probe, root):
            self.roi_calls.append((row['id'], root))
            # No fake native ROI claims: fixture ROI is intentionally unavailable.
            return self.native_load_roi(row, probe, None)
        stack.enter_context(patch.object(bar, 'load_hand_roi', side_effect=roi))
        return stack

    def run(self, llm=None, admission=None):
        llm = llm or FakeLLM()
        with self.context():
            report = pilot.run_pilot(self.args, self.runtime(llm),
                                    admission or (lambda args, runtime: True))
        return report, llm


class FullPipelineContracts(unittest.TestCase):
    def test_native_three_sources_eight_windows_batch8_direct_joint_and_immutable_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = FullFixture(tmp, masked=True)
            with patch.object(bar, 'prepare_source', wraps=bar.prepare_source) as prepare,                     patch.object(bar, 'verify_prepared', wraps=bar.verify_prepared) as verify:
                report, llm = f.run()
            self.assertEqual((prepare.call_count, verify.call_count), (3, 3))
            self.assertEqual([len(call) for call in llm.calls], [8])
            self.assertEqual(report['visual_chat_batch_sizes'], [8])
            self.assertEqual((report['source_count'], report['window_count']), (3, 8))
            self.assertEqual(len(f.decoder_calls), 3)
            self.assertEqual([sid for sid, _ in f.roi_calls], list(f.sources))
            self.assertEqual(report['joint_complete_candidates'], 3)
            self.assertEqual((report['boundary_model_calls'], report['rewrite_model_calls'], report['new_style_model_calls']), (0, 0, 0))
            for result, style in zip(report['results'], f.styles):
                visual, joint = result['visual'], result['joint_candidate']
                self.assertEqual(joint['style_result'], style)
                self.assertTrue(joint['target_text'].startswith('Dance style: ' + style['dance_style'] + '\n'))
                self.assertEqual(joint['label']['timed_motion_script'], visual['final_timed_caption'])
                self.assertEqual([w['window_id'] for w in visual['final']['json']['windows']],
                                 [o['window']['window_id'] for o in visual['window_observations']])
                for window in visual['final']['json']['windows']:
                    facts = [fact for fact in visual['facts'] if fact['window_id'] == window['window_id']]
                    self.assertEqual(window['caption'], ' '.join(fact['text'] for fact in facts))
                    self.assertEqual(window['used_fact_ids'], [fact['fact_id'] for fact in facts])
                    self.assertEqual(window['used_boundary_ids'], [])
                    for fact in facts:
                        self.assertGreaterEqual(len(fact['frame_indices']), 2)
                        self.assertNotIn('right foot', fact['text'])
                self.assertGreater(visual['coverage']['part_unknown_counts']['right_foot'], 0)
                self.assertEqual(visual['coverage']['omitted_fact_count'], 0)
                for key in pilot.FLAGS:
                    self.assertFalse(joint[key])
                    self.assertFalse(report[key])
            self.assertTrue(all(t >= 0 for t in report['timing_seconds'].values()))
            self.assertIsNone(report['speedup_claim'])
            persisted = pipeline.load(f.out / 'report.json')
            self.assertEqual({k: v for k, v in persisted.items() if k != 'timing_seconds'},
                             {k: v for k, v in report.items() if k != 'timing_seconds'})
            self.assertEqual(pipeline.load(f.out / 'timing.json')['timing_seconds'], report['timing_seconds'])
            self.assertEqual({p: p.read_bytes() for p in f.original_bytes}, f.original_bytes)
            self.assertFalse(list(f.original.rglob('prepared.json')))
            self.assertFalse(list(f.out.rglob('rewrite')))
            self.assertEqual(len(list(f.out.glob('sources/*/windows/*/attempt-0/receipt.json'))), 8)

    def test_uncertain_fact_and_frame_note_kept_as_admitted_native_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = FullFixture(tmp)
            class Qualified(FakeLLM):
                def chat(self, messages, sampling, use_tqdm=False):
                    outputs = super().chat(messages, sampling, use_tqdm)
                    for output in outputs:
                        value = json.loads(output.outputs[0].text)
                        value['events'][0]['description'] = 'The head appears to turn; the exact direction is uncertain.'
                        value['uncertainties'] = ['Frame note: hand detail is uncertain.']
                        output.outputs[0].text = json.dumps(value)
                    return outputs
            report, _ = f.run(Qualified())
            for result in report['results']:
                visual = result['visual']
                self.assertIn('the exact direction is uncertain', visual['final_timed_caption'])
                for obs in visual['window_observations']:
                    self.assertIn('Frame note: hand detail is uncertain.', obs['json']['uncertainties'])
                for window in visual['final']['json']['windows']:
                    local = [fact for fact in visual['facts'] if fact['window_id'] == window['window_id']]
                    self.assertEqual(window['caption'], ' '.join(fact['text'] for fact in local))

    def test_failed_one_window_retains_all_windows_and_partial_source_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = FullFixture(tmp)
            class OneBadWindow(FakeLLM):
                def chat(self, messages, sampling, use_tqdm=False):
                    outputs = super().chat(messages, sampling, use_tqdm)
                    outputs[1].outputs[0].text = 'not-json'
                    return outputs
            report, llm = f.run(OneBadWindow())
            self.assertEqual([len(c) for c in llm.calls], [8])
            self.assertEqual(report['partial_candidates'], 1)
            result = report['results'][0]
            self.assertEqual(result['visual']['status'], 'failed')
            self.assertEqual(result['joint_candidate']['status'], 'partial_candidate')
            self.assertFalse(result['joint_candidate']['joint_complete'])
            self.assertEqual(result['joint_candidate']['target_text'], '')
            self.assertEqual(len(result['visual']['final']['json']['windows']), 3)
            self.assertEqual(result['visual']['final']['json']['windows'][1]['caption'], '')
            self.assertEqual(result['visual']['window_observations'][1]['status'], 'failed')
            self.assertFalse(list(f.out.rglob('attempt-1')))

    def test_empty_actual_window_is_not_filled_with_static_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = FullFixture(tmp, empty_window=True)
            report, llm = f.run()
            self.assertEqual([len(c) for c in llm.calls], [7])
            result = report['results'][0]
            self.assertEqual(result['visual']['final']['json']['windows'][0]['caption'], '')
            self.assertEqual(result['joint_candidate']['status'], 'partial_candidate')
            self.assertEqual(report['window_count'], 8)
            self.assertFalse((f.out / 'sources/source0/windows/w00000/attempt-0').exists())

    def test_default_admission_refuses_before_any_input_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = FullFixture(tmp)
            with f.context(), patch.object(pilot, 'fixed_sources', side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError, 'admission required'):
                    pilot.run_pilot(f.args, lambda: self.fail('must not load'))
            self.assertFalse(f.out.exists())

    def test_changed_original_asset_or_style_before_commit_refuses_report(self):
        for kind in ('asset', 'style'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                f = FullFixture(tmp)
                original_resume = pilot.cross.resume_source
                def mutate(*a, **kw):
                    value = original_resume(*a, **kw)
                    if a[1] == 2:
                        path = (Path(f.entries[0]['row']['manifest_record']['video_path']) if kind == 'asset'
                                else next(f.original.rglob('original_style.json')))
                        path.write_bytes(b'changed input')
                    return value
                with patch.object(pilot.cross, 'resume_source', side_effect=mutate):
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        f.run()
                self.assertFalse((f.out / 'report.json').exists())
                self.assertFalse((f.out / 'candidates').exists())

    def test_crossmount_bridge_keeps_historical_device_and_uses_current_execution_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = FullFixture(tmp)
            for path in f.original.glob('sources/train/*/actual_media.json'):
                actual = pipeline.load(path)
                for ref in actual['source_identity']['files'].values():
                    if ref:
                        ref['stat']['device'] += 100
                path.write_bytes(pipeline.raw_json(actual))
            report, _ = f.run()
            for result in report['results']:
                joint = result['joint_candidate']
                prepared = bar.load(f.out / 'sources' / result['id'] / 'prepared.json')
                self.assertEqual(joint['source_identity'], prepared['source_identity'])
                historical = pipeline.load(joint['original_actual_media_ref']['path'])['source_identity']
                self.assertEqual(joint['historical_source_identity'], historical)
                self.assertNotEqual(joint['source_identity']['files']['video_path']['stat']['device'],
                                    historical['files']['video_path']['stat']['device'])

    def test_stage_and_resumes_share_cache_and_parse_manifest_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = FullFixture(tmp)
            stages, resumes, identities, parses = [], [], [], []
            old_stage, old_resume = pilot.cross.stage_sources, pilot.cross.resume_source
            old_identity, old_read = bar.batch.file_identity, Path.read_text
            manifest = Path(f.entries[0]['row']['source_refs']['manifest']['path'])
            def stage(*a, **kw):
                stages.append(id(a[3]))
                return old_stage(*a, **kw)
            def resume(*a, **kw):
                resumes.append(id(a[4]))
                return old_resume(*a, **kw)
            def identity(path):
                if Path(path) == manifest:
                    identities.append(path)
                return old_identity(path)
            def read(path, *a, **kw):
                if path == manifest:
                    parses.append(path)
                return old_read(path, *a, **kw)
            with patch.object(pilot.cross, 'stage_sources', side_effect=stage), \
                    patch.object(pilot.cross, 'resume_source', side_effect=resume), \
                    patch.object(bar.batch, 'file_identity', side_effect=identity), \
                    patch.object(Path, 'read_text', read):
                f.run()
            self.assertEqual(resumes, stages * 3)
            self.assertEqual((len(identities), len(parses)), (1, 1))

    def test_configuration_changed_after_last_resume_rejects_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = FullFixture(tmp)
            original_resume = pilot.cross.resume_source
            def changed(*a, **kw):
                value = original_resume(*a, **kw)
                if a[1] == 2:
                    f.args.visual_max_tokens += 1
                return value
            with patch.object(pilot.cross, 'resume_source', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'configuration changed'):
                    f.run()
            self.assertFalse((f.out / 'report.json').exists())
            self.assertFalse((f.out / 'candidates').exists())

    def test_repeated_output_refuses_without_reloading_or_redispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = FullFixture(tmp)
            _, llm = f.run()
            before = {p: p.read_bytes() for p in f.out.rglob('*') if p.is_file()}
            with self.assertRaisesRegex(ValueError, 'already exists'):
                f.run(llm)
            self.assertEqual([len(c) for c in llm.calls], [8])
            self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_unbound_or_music_genre_style_rejected_before_dispatch(self):
        for field, value in (('source', 'music_genre'), ('id', 'other-source')):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                f = FullFixture(tmp)
                path = next(f.original.rglob('original_joint.json'))
                joint = pipeline.load(path)
                joint['style_result'][field] = value
                path.write_bytes(pipeline.raw_json(joint))
                terminal_path = path.with_name('consumer.terminal.json')
                terminal = pipeline.load(terminal_path)
                terminal['joint_candidate_ref'] = pipeline.ref(path)
                terminal_path.write_bytes(pipeline.raw_json(terminal))
                llm = FakeLLM()
                with self.assertRaisesRegex(ValueError, 'resolved dance style'):
                    f.run(llm)
                self.assertEqual(llm.calls, [])
                self.assertFalse(f.out.exists())


if __name__ == '__main__':
    unittest.main()
