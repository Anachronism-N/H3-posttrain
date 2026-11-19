"""Native finite production tests; external media and chat only are synthetic."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import caption_batch8_production_consumer_v1 as consumer
import caption_direct_facts_v1 as direct
from test_caption_batch8_direct_full_pilot_v1 import FullFixture, FakeLLM
bar, pipeline = direct.native()
import ms_dance_style as style_module


class ProductionFixture:
    def __init__(self, base, groups=2, empty=False, masked=False):
        self.base = Path(base)
        self.original = self.base / 'producer-root'
        self.out = self.base / 'new-production'
        self.probes, self.fixtures, self.entries, self.styles = {}, [], [], []
        self.args = None
        for group in range(groups):
            directory = self.base / ('fixture%d' % group)
            directory.mkdir()
            f = FullFixture(directory, empty_window=empty and group == 0, masked=masked)
            self.fixtures.append(f)
            self.probes.update(f.probes)
            if self.args is None:
                self.args = f.args
            for i, entry in enumerate(f.entries):
                row = copy.deepcopy(entry['row'])
                sid = 'source%d' % (group * 3 + i)
                row['id'] = sid
                row['binding']['id'] = sid
                row['window_manifest']['id'] = sid
                row['window_manifest']['binding']['id'] = sid
                record = row['manifest_record']
                record.update(id=sid, source_ordinal=3 + 9 * (group * 3 + i), split_ordinal=group * 3 + i)
                rich = self.original / 'rich' / (sid + '.txt')
                pipeline.save(rich.with_suffix('.unused.json'), {'fixture': True})
                rich.write_text('[2. Dance] Contemporary dance choreography. [3. Motion]')
                record['rich_path'] = str(rich)
                if masked:
                    mask = pipeline.load(row['raw2d_mask']['path'])
                    mask['id'] = sid
                    mask_path = self.original / 'masks' / (sid + '.json')
                    pipeline.save(mask_path, mask)
                    row['raw2d_mask'] = pipeline.ref(mask_path)
                self.entries.append(row)
                dance_style = copy.deepcopy(f.styles[i])
                dance_style['id'] = sid
                self.styles.append(dance_style)
        records = [row['manifest_record'] for row in self.entries]
        manifest = self.original / 'batch.jsonl'
        manifest.write_bytes(b''.join(pipeline.raw_json(r) for r in records))
        batch_ref = dict(**pipeline.ref(manifest), batch_id='train-b000000', split='train', count=len(records), first_source_ordinal=0)
        summary_path = self.original / 'manifest.summary.json'
        pipeline.save(summary_path, dict(batches=[batch_ref], total=len(records), counts={'train': len(records)}, batch_size=500))
        selected = []
        for row, dance_style in zip(self.entries, self.styles):
            row['source_refs']['manifest'] = {k: batch_ref[k] for k in ('path', 'sha256')}
            sid, record = row['id'], row['manifest_record']
            meta = {k: record[k] for k in ('id', 'split', 'batch_id', 'source_ordinal')}
            selected.append(meta)
            source = pipeline.source_dir(self.original, record)
            plan_path = pipeline.save(source / 'bar_plan.json', row)
            roi = pipeline.save(self.original / 'roi' / sid / 'roi_manifest.json', dict(id=sid, crops=[]))
            pipeline.save(roi.with_name('source_manifest.json'), {'cpu_fixture': True})
            pipeline.save(roi.with_name('extract_receipt.json'), {'cpu_fixture': True})
            producer = dict(**meta, status='ready', plan_ref=pipeline.ref(plan_path), roi_ref=pipeline.ref(roi), source_binding=row['binding'])
            pipeline.save(source / 'producer.terminal.json', producer)
            pipeline.append(self.original / 'ready' / ('producer-%04d.jsonl' % (record['source_ordinal'] % 4)), producer)
            actual = dict(source_identity=bar.batch.source_identity(record), video_probe=self.probes[record['video_path']], audio_probe={})
            pipeline.save(source / 'actual_media.json', actual)
            old = self.original / 'consumers' / ('shard-%04d' % record['source_ordinal']) / 'sources/train' / sid
            style_path = pipeline.save(old / 'original_style.json', dance_style)
            joint_path = pipeline.save(old / 'original_joint.json', dict(**meta, joint_complete=True, style_result=dance_style, style_result_ref=pipeline.ref(style_path)))
            terminal = dict(**meta, status='joint_candidate', joint_complete=True, joint_candidate_ref=pipeline.ref(joint_path))
            # Existing style artifacts alone are not old consumer terminal work.
            self.old_styles = getattr(self, 'old_styles', []) + [(old, terminal)]
        members, selected_refs = {}, {}
        for mode, count in (('producer', 4), ('consumer', 9)):
            for shard in range(count):
                name = mode + str(shard)
                members[name] = dict(mode=mode, shard=shard, nshard=count, node='CPU', gpus=[0, 1])
                path = pipeline.save(self.original / 'selected' / (name + '.json'),
                                     [r for r in selected if r['source_ordinal'] % count == shard])
                selected_refs[name] = pipeline.ref(path)
        self.source_plan = self.original / 'source-plan.json'
        pipeline.save(self.source_plan, dict(out=str(self.original), manifest_summary=pipeline.ref(summary_path),
            source_range=dict(start=3, end=3 + 9 * len(records)), members=members, selected_sources=selected_refs))
        vars(self.args).update(source_plan=str(self.source_plan), source_root=str(self.original), manifest=str(summary_path),
            member='consumer3', shard=3, nshard=9, start_ordinal=3, end_ordinal=3 + 9 * len(records), limit=len(records),
            style_reuse=[], style_max_tokens=1024, out=str(self.out), hand_roi_root=str(self.original / 'roi'))
        self.old_out = self.original / 'consumers/shard-0003'
        self.old_out.mkdir(parents=True, exist_ok=True)
        (self.old_out / 'completed.jsonl').write_bytes(b'')
        self.make_handoff(selected)
        self.original_bytes = {p: p.read_bytes() for p in self.original.rglob('*') if p.is_file()}

    def make_handoff(self, selected=None):
        import time
        proof = self.base / 'handoff-fixture'
        proof.mkdir(exist_ok=True)
        plan = pipeline.load(self.source_plan)
        if selected is None:
            selected = pipeline.load(plan['selected_sources']['consumer3']['path'])
        selected_ref = pipeline.ref(pipeline.save(proof / 'exclusive.json', selected))
        config = dict(native_fixture_only=True)
        params_sha = pipeline.digest(config)
        params_ref = pipeline.ref(pipeline.save(proof / 'params.json', dict(new_params_sha256=params_sha, new_configuration=config)))
        queue = proof / 'old-native-queue'
        manager_identity = dict(pid=111, uid=0, argv=['CPU-native-manager'], start_ticks=123, cgroup='CPU-fixture', CVD=[], exe='CPU-python')
        worker_identity = dict(pid=222, uid=0, argv=['CPU-native-worker'], start_ticks=456, cgroup='CPU-fixture', CVD=['CUDA_VISIBLE_DEVICES=0,1'], exe='CPU-python')
        holders = [dict(gpu=i, uuid='CPU-GPU%d' % i) for i in (0, 1)]
        cfg = dict(mode='consumer', holders=holders, collab3_member='consumer3', collab3_source_plan=pipeline.ref(self.source_plan),
            selected_sources=plan['selected_sources']['consumer3'], node='CPU', gpus=[0, 1],
            out=str(self.old_out), queue=str(queue), phase='native_cpu_fixture',
            source_selection=dict(shard=3, nshard=9, limit=0), consumer_params_evidence=params_ref,
            consumer_params_sha256=params_sha)
        cfg_ref = pipeline.ref(pipeline.save(proof / 'old-launch.json', cfg))
        admission_ref = pipeline.ref(pipeline.save(queue / 'admission.json', dict(phase=cfg['phase'], node='CPU',
            gpus=[0, 1], out=str(self.old_out), config_sha256=cfg_ref['sha256'], holders=holders, manager_identity=manager_identity)))
        pipeline.save(queue / 'live.json', dict(phase=cfg['phase'], child_identity=worker_identity, manager_identity=manager_identity))
        safe_ref = pipeline.ref(pipeline.save(queue / 'resources_safe.json', dict(phase=cfg['phase'],
            all_holders_safe=True, locks_released=True, leases_restored=True, managed_parent_wait=True, unknown_jobs_not_signaled=True)))
        cleanup_ref = pipeline.ref(pipeline.save(proof / 'cleanup.json', dict(schema='caption_native_handoff_cleanup_v1',
            phase=cfg['phase'], node='CPU', gpus=[0, 1], admission_ref=admission_ref, resources_safe_ref=safe_ref,
            cleanup_done=True, old_worker_tree_absent=True, old_worker_identity=worker_identity, old_manager_absent=True, checked_unix_ns=time.time_ns())))
        prefix_path = proof / 'old-completed-prefix.jsonl'
        prefix_path.write_bytes((self.old_out / 'completed.jsonl').read_bytes())
        exclusion_ref = pipeline.ref(pipeline.save(proof / 'exclusion.json', dict(schema='caption_native_handoff_exclusion_v1',
            member='consumer3', source_plan_ref=pipeline.ref(self.source_plan), new_out=str(self.out),
            exclusive_selected_sources_ref=selected_ref, original_config_resume_forbidden=True)))
        handoff = dict(schema='caption_batch8_native_handoff_v1', member='consumer3', source_plan_ref=pipeline.ref(self.source_plan),
            manifest_summary_ref=plan['manifest_summary'], new_out=str(self.out), source_root=str(self.original),
            old_launch_ref=cfg_ref, old_admission_ref=admission_ref, old_resources_safe_ref=safe_ref,
            old_cleanup_ref=cleanup_ref, old_completed_prefix_ref=pipeline.ref(prefix_path),
            old_completed_path=str(self.old_out / 'completed.jsonl'), old_completed_bytes=prefix_path.stat().st_size,
            exclusive_selected_sources_ref=selected_ref, exclusive_source_owner=True, old_native_params_sha256=params_sha,
            old_resume_requires_exclusion=True, old_exclusion_ref=exclusion_ref)
        handoff_path = proof / 'handoff.json'
        handoff_path.write_bytes(pipeline.raw_json(handoff))
        self.args.handoff_ref = pipeline.ref(handoff_path)

    def context(self):
        stack = ExitStack()
        stack.enter_context(patch.object(bar, 'probe_actual', side_effect=lambda path, _: copy.deepcopy(self.probes[path])))
        stack.enter_context(patch.object(bar.subprocess, 'run', side_effect=self.fixtures[0].decode))
        stack.enter_context(patch.object(bar.core, 'structured_sampling', side_effect=self.fixtures[0].sampling))
        native_roi = bar.load_hand_roi
        stack.enter_context(patch.object(bar, 'load_hand_roi', side_effect=lambda row, probe, root: native_roi(row, probe, None)))
        return stack

    def runtime(self, llm):
        return self.fixtures[0].runtime(llm)

    def run(self, llm=None):
        llm = llm or FakeLLM()
        with self.context():
            report = consumer.run_consumer(self.args, self.runtime(llm), lambda args, runtime: True)
        return report, llm

    def terminals(self):
        return [pipeline.load(p) for p in sorted(self.out.glob('sources/train/*/consumer.terminal.json'))]

    def joint(self, value):
        return pipeline.load(value['joint_candidate_ref']['path'])


class ProductionContracts(unittest.TestCase):
    def test_two_groups_batch8_exact_facts_new_terminal_unique_completed_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp, masked=True)
            caches = []
            old = consumer.cross.stage_sources
            def stage(*args, **kwargs):
                caches.append(id(args[3]))
                return old(*args, **kwargs)
            with patch.object(consumer.cross, 'stage_sources', side_effect=stage):
                report, llm = f.run()
            self.assertEqual([len(c) for c in llm.calls], [8, 8])
            self.assertEqual(caches[0], caches[1])
            self.assertEqual(report['terminal_count'], 6)
            notices = [json.loads(line) for line in (f.out / 'completed.jsonl').read_text().splitlines()]
            self.assertEqual(len(notices), len({n['id'] for n in notices}))
            self.assertEqual(len(f.terminals()), 6)
            for terminal in f.terminals():
                joint = f.joint(terminal)
                self.assertTrue(joint['target_text'].startswith('Dance style: '))
                visual = pipeline.load(joint['visual_result_ref']['path'])
                for w in visual['final']['json']['windows']:
                    facts = [fact for fact in visual['facts'] if fact['window_id'] == w['window_id']]
                    self.assertEqual(w['caption'], ' '.join(fact['text'] for fact in facts))
                    self.assertEqual(w['used_fact_ids'], [fact['fact_id'] for fact in facts])
                    self.assertEqual(w['used_boundary_ids'], [])
                    self.assertTrue(all(len(fact['frame_indices']) >= 2 for fact in facts))
                self.assertEqual(visual['rewrite_fulltext_calls_per_attempt'], 0)
                for key in direct.FLAGS:
                    self.assertFalse(joint[key])
            self.assertEqual({p: p.read_bytes() for p in f.original_bytes}, f.original_bytes)
            self.assertFalse(list(f.out.rglob('attempt-1')))

    def test_restart_same_owned_recipe_no_redispatch_no_duplicate_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp)
            report, llm = f.run()
            before = (f.out / 'completed.jsonl').read_bytes()
            second, _ = f.run(llm)
            self.assertEqual(second['terminal_count'], 6)
            self.assertEqual([len(c) for c in llm.calls], [8, 8])
            self.assertEqual(before, (f.out / 'completed.jsonl').read_bytes())
            f.args.visual_max_tokens += 1
            with self.assertRaisesRegex(ValueError, 'owned resume'):
                f.run(llm)
            self.assertEqual([len(c) for c in llm.calls], [8, 8])

    def test_failed_source_does_not_rerun_reliable_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp)
            class BadOne(FakeLLM):
                def chat(self, messages, sampling, use_tqdm=False):
                    outputs = super().chat(messages, sampling, use_tqdm)
                    if len(self.calls) == 1:
                        outputs[1].outputs[0].text = 'not-json'
                    return outputs
            report, llm = f.run(BadOne())
            self.assertEqual(report['counts'], {'incomplete_candidate': 1, 'joint_candidate': 5})
            terminal = f.terminals()[0]
            joint = f.joint(terminal)
            self.assertEqual(joint['target_text'], '')
            visual = pipeline.load(joint['visual_result_ref']['path'])
            self.assertEqual(visual['final']['json']['windows'][1]['caption'], '')
            f.run(llm)
            self.assertEqual([len(c) for c in llm.calls], [8, 8])
            self.assertFalse(list(f.out.rglob('attempt-1')))

    def test_empty_window_never_fills_static_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp, groups=1, empty=True)
            _, llm = f.run()
            self.assertEqual([len(c) for c in llm.calls], [7])
            joint = f.joint(f.terminals()[0])
            visual = pipeline.load(joint['visual_result_ref']['path'])
            self.assertEqual(visual['final']['json']['windows'][0]['caption'], '')
            self.assertFalse(joint['joint_complete'])
            self.assertEqual(joint['target_text'], '')

    def test_unknown_style_one_native_rgb_call_remains_unknown_not_genre_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp, groups=1)
            Path(f.entries[0]['manifest_record']['rich_path']).write_text('[2. Music] k-pop music. [3. Motion]')
            class UnknownStyle(FakeLLM):
                def chat(self, messages, sampling, use_tqdm=False):
                    if messages[0][0]['content'] == style_module.SYSTEM:
                        self.calls.append(copy.deepcopy(messages))
                        value = dict(dance_style='unknown', frame_indices=[], reason='Cannot distinguish the dance style.', abstain=True)
                        return [SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps(value), token_ids=[7], finish_reason='stop', stop_reason=None)], prompt_token_ids=[1])]
                    return super().chat(messages, sampling, use_tqdm)
            _, llm = f.run(UnknownStyle())
            self.assertEqual([len(c) for c in llm.calls], [8, 1])
            joint = f.joint(f.terminals()[0])
            self.assertEqual(joint['style_result']['dance_style'], 'unknown')
            self.assertEqual(joint['style_result']['status'], 'unknown')
            self.assertEqual(joint['target_text'], '')
            f.run(llm)
            self.assertEqual([len(c) for c in llm.calls], [8, 1])

    def test_source_mutation_rejects_commit_without_poisoning_other_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp, groups=1)
            original = consumer.cross.resume_source
            def mutate(*args, **kwargs):
                value = original(*args, **kwargs)
                if args[1] == 2:
                    Path(f.entries[2]['manifest_record']['video_path']).write_bytes(b'changed source')
                return value
            with patch.object(consumer.cross, 'resume_source', side_effect=mutate):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    f.run()
            self.assertEqual(len(f.terminals()), 2)
            self.assertFalse((f.out / 'sources/train/source2/consumer.terminal.json').exists())

    def test_midrun_restart_native_receipts_zero_resubmit_and_recovers_orphan_notice(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp, groups=1)
            llm = FakeLLM()
            original = pipeline.append
            stopped = False
            def append(path, value):
                nonlocal stopped
                if Path(path) == f.out / 'completed.jsonl' and not stopped:
                    stopped = True
                    raise OSError('CPU simulated interruption after immutable terminal')
                return original(path, value)
            with patch.object(pipeline, 'append', side_effect=append):
                with self.assertRaisesRegex(OSError, 'interruption'):
                    f.run(llm)
            self.assertEqual([len(c) for c in llm.calls], [8])
            report, _ = f.run(llm)
            self.assertEqual([len(c) for c in llm.calls], [8])
            self.assertEqual(report['recovered_completed_notices'], 1)
            self.assertEqual(report['terminal_count'], 3)
            self.assertEqual(len((f.out / 'completed.jsonl').read_text().splitlines()), 3)

    def test_unowned_nonempty_output_and_wrong_plan_member_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp, groups=1)
            f.out.mkdir()
            (f.out / 'old.json').write_text('{}')
            llm = FakeLLM()
            with self.assertRaisesRegex(ValueError, 'owned resume'):
                f.run(llm)
            self.assertEqual(llm.calls, [])
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp, groups=1)
            f.args.shard = 1
            with self.assertRaisesRegex(ValueError, 'member/shard'):
                f.run()

    def test_all_old_terminal_statuses_skip_without_new_attempt_or_reselection(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp, groups=1)
            for i, status in enumerate(('joint_candidate', 'incomplete_candidate', 'failed')):
                row = f.entries[i]['manifest_record']
                terminal = {k: row[k] for k in ('id', 'split', 'batch_id')}
                terminal.update(status=status)
                pipeline.save(pipeline.source_dir(f.old_out, row) / 'consumer.terminal.json', terminal)
            report, llm = f.run()
            self.assertEqual(report['terminal_count'], 0)
            self.assertEqual(llm.calls, [])
            admission = pipeline.load(f.out / 'admission.json')
            self.assertEqual(set(admission['skipped_old_terminal_ids']), {'source0', 'source1', 'source2'})
            self.assertEqual(admission['remaining_ids'], [])
            self.assertFalse(list(f.out.glob('visual/sources/*/prepared.json')))
            with patch.object(consumer, 'freeze_admission', side_effect=AssertionError('must not reselect')):
                f.run(llm)

    def test_stage_resume_share_immutable_inputs_and_exact_native_image_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp, groups=1)
            from curated_fast_source_v1 import SourceAssets
            captured, requests = [], []
            original_capture, original_rgb = SourceAssets.capture, bar.rgb_content
            def capture(cache, path, expected=None):
                key = cache.key(path)
                if key not in cache.values:
                    captured.append(key)
                return original_capture(cache, path, expected)
            original_chat = FakeLLM.chat
            def chat(llm, messages, *a, **kw):
                for msgs in messages:
                    content = msgs[1]['content']
                    header = json.loads(content[0]['text'])
                    wid = header['window']['window_id'] if 'window' in header else None
                    requests.append((wid, content))
                return original_chat(llm, messages, *a, **kw)
            with patch.object(SourceAssets, 'capture', capture), patch.object(FakeLLM, 'chat', chat):
                report, _ = f.run()
            self.assertEqual(len(captured), len(set(captured)))
            self.assertEqual(report['visual_chat_batch_sizes'], [8])
            self.assertEqual(report['new_style_model_calls'], 0)
            resumed = [e for e in report['transaction_cache_events'] if e.get('receipt_resume_full_validation') is True]
            self.assertGreaterEqual(len(resumed), 8)
            # Persisted request still has original per-window native messages, receipt and budgets.
            for path in f.out.glob('visual/sources/*/windows/*/attempt-0/request.json'):
                request = pipeline.load(path)
                root = path.parents[3]
                prepared = pipeline.load(root / 'prepared.json')
                window_id = path.parents[1].name
                manifest = next(m for m in prepared['windows'] if m['window']['window_id'] == window_id)
                forbidden = {p: set(v) for p, v in prepared['forbidden'].items()}
                with f.context():
                    expected = bar.visual_request(f.args, root / 'rgb', manifest, forbidden)
                self.assertEqual(request, expected)
                self.assertTrue(path.with_name('receipt.json').exists())
                self.assertTrue(path.with_name('generation.json').exists())

    def test_admission_content_change_rejected_before_redispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp, groups=1)
            _, llm = f.run()
            path = f.out / 'admission.json'
            admission = pipeline.load(path)
            admission['remaining_ids'] = []
            path.write_bytes(pipeline.raw_json(admission))
            with self.assertRaisesRegex(ValueError, 'admission selection/digest'):
                f.run(llm)
            self.assertEqual([len(c) for c in llm.calls], [8])

    def test_finite_limit_only_loads_selected_producer_terminals(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp)
            f.args.limit = 3
            handoff = pipeline.load(f.args.handoff_ref['path'])
            selected_path = Path(handoff['exclusive_selected_sources_ref']['path'])
            selected_path.write_bytes(pipeline.raw_json(pipeline.load(selected_path)[:3]))
            handoff['exclusive_selected_sources_ref'] = pipeline.ref(selected_path)
            exclusion_path = Path(handoff['old_exclusion_ref']['path'])
            exclusion = pipeline.load(exclusion_path)
            exclusion['exclusive_selected_sources_ref'] = handoff['exclusive_selected_sources_ref']
            exclusion_path.write_bytes(pipeline.raw_json(exclusion))
            handoff['old_exclusion_ref'] = pipeline.ref(exclusion_path)
            Path(f.args.handoff_ref['path']).write_bytes(pipeline.raw_json(handoff))
            f.args.handoff_ref = pipeline.ref(f.args.handoff_ref['path'])
            loads = []
            original = direct.Evidence.load
            def load(evidence, path):
                if Path(path).name == 'producer.terminal.json':
                    loads.append(Path(path))
                return original(evidence, path)
            with patch.object(direct.Evidence, 'load', load):
                report, llm = f.run()
            self.assertEqual(report['terminal_count'], 3)
            self.assertEqual([len(c) for c in llm.calls], [8])
            self.assertEqual({p.parent.name for p in loads}, {'source0', 'source1', 'source2'})

    def test_prepare_failure_terminal_is_local_other_sources_commit_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp, groups=1)
            original = bar.prepare_source
            def prepare(row, *a, **kw):
                if row['id'] == 'source0':
                    raise ValueError('CPU synthetic per-source prepare failure')
                return original(row, *a, **kw)
            with patch.object(bar, 'prepare_source', side_effect=prepare):
                report, llm = f.run()
            self.assertEqual(report['counts'], {'failed': 1, 'joint_candidate': 2})
            self.assertEqual([len(c) for c in llm.calls], [5])
            f.run(llm)
            self.assertEqual([len(c) for c in llm.calls], [5])

    def test_missing_or_unsafe_native_handoff_refuses_before_dispatch(self):
        for mutation in ('missing', 'unsafe', 'cleanup'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                f = ProductionFixture(tmp, groups=1)
                if mutation == 'missing':
                    del f.args.handoff_ref
                else:
                    handoff = pipeline.load(f.args.handoff_ref['path'])
                    key = 'old_resources_safe_ref' if mutation == 'unsafe' else 'old_cleanup_ref'
                    path = Path(handoff[key]['path'])
                    value = pipeline.load(path)
                    value['locks_released' if mutation == 'unsafe' else 'old_worker_tree_absent'] = False
                    path.write_bytes(pipeline.raw_json(value))
                    handoff[key] = pipeline.ref(path)
                    Path(f.args.handoff_ref['path']).write_bytes(pipeline.raw_json(handoff))
                    f.args.handoff_ref = pipeline.ref(f.args.handoff_ref['path'])
                llm = FakeLLM()
                with self.assertRaises(ValueError):
                    f.run(llm)
                self.assertEqual(llm.calls, [])
                self.assertFalse(list(f.out.glob('visual/sources/*/prepared.json')))

    def test_handoff_exact_member_pool_and_completed_prefix_are_mandatory(self):
        for mutation in ('ordinal', 'newout', 'prefix'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                f = ProductionFixture(tmp, groups=1)
                handoff = pipeline.load(f.args.handoff_ref['path'])
                if mutation == 'ordinal':
                    path = Path(handoff['exclusive_selected_sources_ref']['path'])
                    rows = pipeline.load(path)
                    rows[0]['source_ordinal'] += 1
                    path.write_bytes(pipeline.raw_json(rows))
                    handoff['exclusive_selected_sources_ref'] = pipeline.ref(path)
                elif mutation == 'newout':
                    handoff['new_out'] = str(f.base / 'another-out')
                else:
                    Path(handoff['old_completed_path']).write_bytes(b'{}\n')
                Path(f.args.handoff_ref['path']).write_bytes(pipeline.raw_json(handoff))
                f.args.handoff_ref = pipeline.ref(f.args.handoff_ref['path'])
                llm = FakeLLM()
                with self.assertRaises(ValueError):
                    f.run(llm)
                self.assertEqual(llm.calls, [])

    def test_one_source_over_eight_windows_keeps_all_original_windows_batch8_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = ProductionFixture(tmp, groups=1)
            row = f.entries[0]
            times = [i / 8. for i in range(9 * 4)]
            probe = dict(times=times, source_timestamps=times, actual_video_pts_sha256=bar.batch.digest(times),
                         media_start=0., media_end=4.5, video_fps=8., video_pts_offset=0.)
            f.probes[row['manifest_record']['video_path']] = probe
            row['binding'].update(actual_video_pts_sha256=probe['actual_video_pts_sha256'], decoded_video_frames=len(times), media_end=4.5)
            windows = [dict(start=i / 2., end=(i + 1) / 2., raw_window_indices=[i]) for i in range(9)]
            row['windows'] = windows
            row['window_manifest'].update(binding=copy.deepcopy(row['binding']), windows=copy.deepcopy(windows), raw_windows=copy.deepcopy(windows))
            source = pipeline.source_dir(f.original, row['manifest_record'])
            plan_path = source / 'bar_plan.json'
            plan_path.write_bytes(pipeline.raw_json(row))
            terminal_path = source / 'producer.terminal.json'
            producer = pipeline.load(terminal_path)
            producer.update(plan_ref=pipeline.ref(plan_path), source_binding=row['binding'])
            terminal_path.write_bytes(pipeline.raw_json(producer))
            stream = f.original / 'ready' / ('producer-%04d.jsonl' % (row['manifest_record']['source_ordinal'] % 4))
            stream.write_bytes(pipeline.raw_json(producer))
            actual_path = source / 'actual_media.json'
            actual = pipeline.load(actual_path)
            actual['video_probe'] = probe
            actual_path.write_bytes(pipeline.raw_json(actual))
            report, llm = f.run()
            self.assertEqual([len(c) for c in llm.calls], [8, 6])
            self.assertEqual(report['terminal_count'], 3)
            joint = f.joint(f.terminals()[0])
            visual = pipeline.load(joint['visual_result_ref']['path'])
            self.assertEqual(len(visual['final']['json']['windows']), 9)
            self.assertEqual(len(visual['window_observations']), 9)
            f.run(llm)
            self.assertEqual([len(c) for c in llm.calls], [8, 6])

    def test_native_parent_wait_alive_is_legal_only_with_stable_quiescent_identity(self):
        for mutation in ('valid_wait', 'nonquiescent', 'changed_manager', 'live_worker', 'wrong_uuid'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                f = ProductionFixture(tmp, groups=1)
                handoff = pipeline.load(f.args.handoff_ref['path'])
                cleanup_path = Path(handoff['old_cleanup_ref']['path'])
                cleanup = pipeline.load(cleanup_path)
                admitted = pipeline.load(handoff['old_admission_ref']['path'])
                cleanup.update(old_manager_absent=False, old_manager_quiescent=True,
                               old_manager_identity=copy.deepcopy(admitted['manager_identity']))
                if mutation == 'nonquiescent':
                    cleanup['old_manager_quiescent'] = False
                elif mutation == 'changed_manager':
                    cleanup['old_manager_identity']['start_ticks'] += 1
                elif mutation == 'live_worker':
                    cleanup['old_worker_tree_absent'] = False
                elif mutation == 'wrong_uuid':
                    admission_path = Path(handoff['old_admission_ref']['path'])
                    admitted['holders'][0]['uuid'] = 'OTHER-CPU-GPU'
                    admission_path.write_bytes(pipeline.raw_json(admitted))
                    handoff['old_admission_ref'] = pipeline.ref(admission_path)
                    cleanup['admission_ref'] = handoff['old_admission_ref']
                cleanup_path.write_bytes(pipeline.raw_json(cleanup))
                handoff['old_cleanup_ref'] = pipeline.ref(cleanup_path)
                Path(f.args.handoff_ref['path']).write_bytes(pipeline.raw_json(handoff))
                f.args.handoff_ref = pipeline.ref(f.args.handoff_ref['path'])
                llm = FakeLLM()
                if mutation == 'valid_wait':
                    preview = consumer.verify_handoff(f.args, consumer.Inputs())
                    self.assertEqual(len(preview['selected_meta']), 3)
                    _, llm = f.run(llm)
                    self.assertEqual([len(c) for c in llm.calls], [8])
                else:
                    with self.assertRaises(ValueError):
                        f.run(llm)
                    self.assertEqual(llm.calls, [])

    def test_parser_finite_cpu_caller_surface(self):
        args = consumer.parser().parse_args(['--plan', 'caller-plan', '--model', 'caller-model', '--out', 'caller-out',
            '--member', 'n4c3', '--shard', '3', '--nshard', '9', '--handoff', 'caller-handoff', '--handoff-sha', 'a' * 64])
        self.assertEqual((args.limit, args.start_ordinal, args.end_ordinal), (500, 32000, 62000))
        self.assertEqual((args.shard, args.nshard), (3, 9))


if __name__ == '__main__':
    unittest.main()
