"""Small synthetic runner tests; no model imports, media decode or GPU allocation."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

RT = Path(__file__).resolve().parent

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

w = module('watch_test', RT / 'curated_repair_watch_v1.py')
q = module('repair_test', RT / 'source_repair_gpu_v2/curated_repair_queue_v1.py')
pipeline = module('pipeline_test', RT / 'source_curated_consumer_v1/curated_pipeline.py')


class WatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'repairs'
        self.out.mkdir()
        self.calls = []
        self.loads = []
        loaded = []
        def runtime():
            if not loaded:
                self.loads.append(1)
                loaded.append(object())
            return loaded[0]
        self.runtime = runtime
        self.bar = SimpleNamespace(core=SimpleNamespace(timed_text=lambda plan, captions:
            '\n'.join(f"[{v['start']:.3f}-{v['end']:.3f}] {c['caption']}" for v, c in zip(plan['windows'], captions))))
        self.worker = SimpleNamespace(stage='run', retry_failed=False, rewrite_reference_repair='off')
        self.repair_fn = q.repair_visual
        q.repair_visual = self.fake_repair
        self.addCleanup(setattr, q, 'repair_visual', self.repair_fn)

    def job(self, sid, unknown=False, semantic=False):
        key = 'train/' + sid
        directory = self.root / 'production' / key
        row = {'id': sid, 'split': 'train', 'batch_id': 'batch1',
               'video_path': '/video', 'audio_path': '/audio'}
        window = {'window_id': 'w1', 'start': 0., 'end': 2.}
        visual = {'id': sid, 'binding': {'source': sid}, 'original_windows': [window],
                  'coverage': {'full_window_caption': False}, 'window_observations': [{'window': window, 'status': 'candidate'}],
                  'final': {'status': 'failed'}}
        style = {'status': 'unknown' if unknown else 'candidate', 'dance_style': 'unknown' if unknown else 'hiphop'}
        producer = {'manifest_record': row}
        q.save_once(directory / 'visual.json', visual)
        q.save_once(directory / 'style.json', style)
        q.save_once(directory / 'producer.json', producer)
        joint = {'id': sid, 'style_result': style, 'style_result_ref': q.identity(directory / 'style.json'),
                 'annotator_model_sha256': 'test-model'}
        q.save_once(directory / 'joint.json', joint)
        refs = {n: q.identity(directory / f) for n, f in
                [('visual_result', 'visual.json'), ('style_result', 'style.json'),
                 ('joint_result', 'joint.json'), ('producer_terminal', 'producer.json')]}
        diagnostic = {'window_id': 'w1', 'body_status': 'present' if semantic else 'malformed',
                      'status': 'abstain' if semantic else 'failed',
                      'reason': 'no_admitted_motion_events' if semantic else 'malformed_response_body'}
        job = {'id': sid, 'split': 'train', 'source_key': key, 'batch_id': 'batch1',
               'queue_root': str(self.out), 'production': str(self.root / 'production'),
               'stages': ['reobserve'] + (['styleunknown'] if unknown else []), 'stage': 'reobserve',
               'failures': [{'stage': 'reobserve', 'window_id': 'w1'}],
               'window_diagnostics': [diagnostic], 'refs': refs, 'source_binding': {'source': sid},
               'original_result_ref': refs['joint_result'], 'original_result_sha256': refs['joint_result']['sha256']}
        path = self.out / 'jobs' / (key + '.json')
        q.save_once(path, job)
        hint = {'source_key': key, 'job_ref': q.identity(path)}
        return hint, job

    def fake_repair(self, path, out, revision, bar, runtime, worker):
        job = q.load(path)
        runtime()
        q.reserve_attempt(job, out, revision)
        self.calls.append(job['id'])
        result = {'id': job['id'], 'original_result_ref': job['original_result_ref'],
                  'window_observations': q.checked(job['refs']['visual_result'])['window_observations'],
                  'facts': [{'fact_id': 'w1:e1'}], 'boundary_observations': [],
                  'final': {'status': 'candidate', 'json': {'windows': [{'window_id': 'w1',
                            'caption': 'The dancer raises both arms.', 'used_fact_ids': ['w1:e1']}] }},
                  'all_windows_nonempty': True}
        q.save_once(Path(out) / 'revisions' / revision / job['source_key'] / 'result.json', result)
        return result

    def consume(self):
        return w.consume_once(self.out, 'r1', q, pipeline, self.bar, self.runtime, self.worker)

    def test_shared_runtime_joint_and_duplicate_ready(self):
        a, ja = self.job('A')
        b, jb = self.job('B', unknown=True)
        original = {r['path']: q.sha(r['path']) for j in [ja, jb] for r in j['refs'].values()}
        for hint in [a, b, a]:
            q.append(self.out / 'ready.jsonl', hint)
        state, count = self.consume()
        self.assertEqual((count, len(self.calls), len(self.loads)), (2, 2, 1))
        self.assertEqual(len(state['seen']), 2)
        result_a = q.load(self.out / 'joint/revisions/r1/train/A/result.json')
        result_b = q.load(self.out / 'joint/revisions/r1/train/B/result.json')
        self.assertTrue(result_a['joint_complete'])
        self.assertIn('Dance style: hiphop', result_a['target_text'])
        self.assertIn('[0.000-2.000]', result_a['label']['timed_motion_script'])
        self.assertFalse(result_b['joint_complete'])
        self.assertEqual(result_b['target_text'], '')
        self.assertEqual(result_b['unresolved_stages'], ['styleunknown'])
        self.assertFalse(result_a['training_eligible'])
        self.assertEqual(len((self.out / 'joint/review-required.jsonl').read_text().splitlines()), 2)
        self.consume()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(original, {p: q.sha(p) for p in original})

    def test_semantic_and_raq_deferred_without_budget(self):
        for sid in ['semantic', w.RAQ_ID]:
            hint, _ = self.job(sid, semantic=sid == 'semantic')
            q.append(self.out / 'ready.jsonl', hint)
        state, count = self.consume()
        self.assertEqual(count, 2)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.out / 'attempt-budget').exists())
        self.assertEqual(state['counts'], {'deferred': 2})

    def test_partial_line_and_truncation(self):
        hint, _ = self.job('partial')
        data = q.encoded(hint)
        path = self.out / 'ready.jsonl'
        path.write_bytes(data[:-1])
        self.assertEqual(self.consume()[1], 0)
        with path.open('ab') as stream:
            stream.write(b'\n')
        self.assertEqual(self.consume()[1], 1)
        path.write_bytes(b'')
        with self.assertRaises(q.SourceChanged):
            self.consume()

    def test_interrupted_budget_never_resubmits(self):
        hint, job = self.job('interrupted')
        q.reserve_attempt(job, self.out, 'r1')
        q.append(self.out / 'ready.jsonl', hint)
        self.consume()
        self.assertEqual(self.calls, [])
        self.assertEqual(q.load(self.out / 'worker-terminals/train/interrupted.json')['status'], 'interrupted_attempt')

    def test_recover_completed_visual_without_resubmit(self):
        hint, job = self.job('recover')
        selected = w.dispatch_job(job, hint['job_ref'])
        path = self.out / 'dispatch-jobs/train/recover.json'
        q.save_once(path, selected)
        self.fake_repair(path, self.out, 'r1', self.bar, self.runtime, self.worker)
        q.append(self.out / 'ready.jsonl', hint)
        self.consume()
        self.assertEqual(self.calls, ['recover'])
        self.assertTrue(q.load(self.out / 'joint/revisions/r1/train/recover/result.json')['joint_complete'])

    def test_dispatch_error_stops_before_next_source(self):
        for sid in ['dispatchfail', 'next']:
            hint, _ = self.job(sid)
            q.append(self.out / 'ready.jsonl', hint)
        fake = self.fake_repair
        def fail(path, out, revision, bar, runtime, worker):
            result = fake(path, out, revision, bar, runtime, worker)
            key = q.load(path)['source_key']
            q.save_once(Path(out) / 'revisions' / revision / key / 'rewrite/dispatch_error.json', {'error': 'engine unavailable'})
            return result
        q.repair_visual = fail
        with self.assertRaises(RuntimeError):
            self.consume()
        self.assertEqual(self.calls, ['dispatchfail'])
        self.assertFalse((self.out / 'attempt-budget/train/next.json').exists())
        self.assertTrue(q.load(self.out / 'worker-terminals/train/dispatchfail.json')['shared_dispatch_failure'])

    def test_different_revision_budget_never_resubmits(self):
        hint, job = self.job('oldbudget')
        q.reserve_attempt(job, self.out, 'r0')
        q.append(self.out / 'ready.jsonl', hint)
        self.consume()
        self.assertEqual(self.calls, [])
        self.assertEqual(q.load(self.out / 'worker-terminals/train/oldbudget.json')['status'], 'budget_already_reserved')
        job['stages'] = ['media', 'reobserve']
        self.assertIsNone(w.dispatch_job(job, hint['job_ref']))
        job['stages'] = ['styleunknown']
        job['failures'] = [{'stage': 'styleunknown'}]
        self.assertIsNone(w.dispatch_job(job, hint['job_ref']))

    def test_changed_job_sha_rejected(self):
        hint, _ = self.job('changed')
        Path(hint['job_ref']['path']).write_text('{}')
        q.append(self.out / 'ready.jsonl', hint)
        with self.assertRaises(q.SourceChanged):
            self.consume()


if __name__ == '__main__':
    unittest.main()
