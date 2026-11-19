"""CPU fake receipts and mocked SSH/Popen only; no real GPU-pass claims."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shlex
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts_omnidance/exp/ms/music_planner_completion_coordinator.py'
spec = importlib.util.spec_from_file_location('completion_under_test', SCRIPT)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
fixture_spec = importlib.util.spec_from_file_location('eval_cpu_fixtures', Path(__file__).with_name('test_music_planner_eval_manager.py'))
f = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(f)


class CoordinatorCPU(unittest.TestCase):
    def setUp(self):
        self.base = f.CPUHandoff('test_correct_scope_and_real_helper_completion')
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.work = self.base.handoff / 'coordinator_v1'
        self.work.mkdir()
        self.base.stack.enter_context(patch.object(f.m, '__file__', str(c.MANAGER)))
        self.coord = c.Coordinator(f.m, (f.manager, f.fixed, f.ev), self.base.protocol_ref,
                                   self.base.protocol, {}, self.work)

    def freeze(self, node='N2'):
        if node != 'N2':
            self.base.family(node)
        return self.coord.freeze_family(node)

    def freeze_all(self):
        for node in f.m.SCOPE:
            self.freeze(node)

    def capture(self):
        holder = dict(pid=42, uid=os.getuid(), start_ticks=1, argv=['CPU fake holder'], gpu=0, uuid='CPU fake GPU0')
        return subprocess.CompletedProcess([], 0, json.dumps(dict(holder=holder, signals_sent=False,
            other_seven_GPUs_untouched=True)), '')

    def dispatch(self):
        self.freeze()
        child = Mock()
        child.poll.return_value = None
        with patch.object(c.subprocess, 'run', return_value=self.capture()) as run,                 patch.object(c.subprocess, 'Popen', return_value=child) as popen:
            self.coord.dispatch('N2', 'val_panel')
        return child, run, popen

    def test_manager_SHA_gate_precedes_import(self):
        with patch.object(c.importlib.util, 'spec_from_file_location') as execute:
            with self.assertRaisesRegex(RuntimeError, 'before import'):
                c.load_manager(c.MANAGER, '0' * 64)
            execute.assert_not_called()
        module = c.load_manager()
        self.assertEqual(Path(module.__file__), c.MANAGER)

    def test_remote_SHA_guard_cannot_be_disabled_by_python_optimization(self):
        cfg = dict(host='CPU-not-an-SSH-host')
        command = self.coord.ssh('capture', cfg, dict(path='/CPU/no-config', sha256='0' * 64))
        argv = shlex.split(command[-1])
        argv[5] = '0' * 64
        # Execute only the rejecting guard in a local CPU interpreter; no manager import/capture/GPU.
        result = subprocess.run([sys.executable, '-B', '-O', *argv[1:]], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('manager SHA changed', result.stderr)

    def test_real_remote_completion_and_within_family_selection_frozen(self):
        frozen = self.freeze()
        self.assertEqual(frozen['identity']['cursor']['update'], 670)
        self.assertEqual(frozen['identity']['cursor']['source_exposures'], 21426)
        self.assertEqual(frozen['selection']['fingerprint'], self.base.cfg['fingerprint'])
        self.assertIsNone(frozen['audit']['content'])
        self.assertEqual(len(frozen['audit']['validation_candidates']), 4)
        self.assertEqual(frozen['protocol'], self.base.protocol_ref)

    def test_incomplete_training_never_freezes(self):
        Path(self.base.cfg['full_complete']['path']).unlink()
        with patch.object(c.subprocess, 'run') as ssh, patch.object(c.subprocess, 'Popen') as popen:
            self.assertIsNone(self.coord.freeze_family('N2'))
            ssh.assert_not_called()
            popen.assert_not_called()
        self.assertEqual(list(self.work.iterdir()), [])

    def test_wrong_budget_unsafe_resource_and_failure_rejected(self):
        self.base.update_complete(dict(optimizer_updates=669))
        with self.assertRaises(RuntimeError):
            self.freeze()
        self.base.update_complete(dict(optimizer_updates=670))
        safe = Path(self.base.cfg['resources_safe']['path'])
        value = json.loads(safe.read_text())
        value['all_holders_safe'] = False
        f.write_json(safe, value)
        with self.assertRaises(RuntimeError):
            self.freeze()
        value['all_holders_safe'] = True
        f.write_json(safe, value)
        f.write_json(safe.parent / 'failure.json', dict(error='CPU simulated training failure'))
        with self.assertRaisesRegex(RuntimeError, 'failure retained'):
            self.freeze()

    def test_no_three_family_final_selection(self):
        self.freeze()
        with self.assertRaisesRegex(RuntimeError, 'all three'):
            self.coord.build_final_selection()
        with self.assertRaisesRegex(RuntimeError, 'no three'):
            self.coord.config('N2', 'test')
        self.assertFalse((self.base.handoff / 'final_selection.json').exists())

    def test_final_build_uses_prior_frozen_rows_not_new_selection_calculation(self):
        self.freeze_all()
        expected = {family: copy.deepcopy(row['selection']) for family, row in self.coord.frozen.items()}
        # Real manager check_final_selection still performs its own strict current checks.
        with patch.object(self.coord, 'freeze_family', side_effect=AssertionError('must not refreeze')):
            final = self.coord.build_final_selection()
        self.assertEqual(final['families'], expected)
        self.assertEqual(final['data'], self.base.protocol['data'])
        self.assertFalse(final['test_used'])
        f.m.check_final_selection(self.coord.config('N2', 'test'), f.manager, f.fixed, f.ev, self.base.protocol)

    def test_frozen_selection_changed_or_wrong_checkpoint_rejected(self):
        self.freeze_all()
        Path(self.coord.frozen['moss']['selection']['checkpoint_marker']['path']).write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'SHA changed'):
            self.coord.build_final_selection()
        self.assertFalse((self.base.handoff / 'final_selection.json').exists())

    def test_busy_coordinator_lock_and_content_nofollow(self):
        lock = self.base.root / 'coordinator.lock'
        fd = c.coordinator_lock(lock)
        try:
            with self.assertRaises(BlockingIOError):
                c.coordinator_lock(lock)
        finally:
            os.close(fd)
        lock.write_text('foreign')
        with self.assertRaisesRegex(RuntimeError, 'content'):
            c.coordinator_lock(lock)
        link = self.base.root / 'link.lock'
        link.symlink_to(lock)
        with self.assertRaises((OSError, RuntimeError)):
            c.coordinator_lock(link)

    def test_stale_claim_never_recaptures_or_duplicates(self):
        self.freeze()
        claim = self.work / 'music_flamingo_val_panel.dispatch.json'
        f.write_json(claim, dict(pid=99999, old_task_handle='not_process_evidence'))
        original = claim.read_bytes()
        with patch.object(c.subprocess, 'run') as ssh, patch.object(c.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError, 'never duplicate'):
                self.coord.dispatch('N2', 'val_panel')
            ssh.assert_not_called()
            popen.assert_not_called()
        self.assertEqual(claim.read_bytes(), original)

    def test_actual_manager_configs_capture_before_holder_and_controlled_Popen(self):
        child, run, popen = self.dispatch()
        capture = json.loads((self.work / 'music_flamingo_val_panel.capture_config.json').read_text())
        run_cfg = json.loads((self.work / 'music_flamingo_val_panel.run_config.json').read_text())
        self.assertNotIn('holder', capture)
        self.assertEqual(run_cfg['holder']['gpu'], 0)
        self.assertEqual(run_cfg['schema'], 'music-planner-eval-manager-v1')
        self.assertEqual(run_cfg['gpus'], [0])
        self.assertTrue(run_cfg['release_authorized'])
        self.assertEqual(run_cfg['source_sha256'][str(c.MANAGER)], c.MANAGER_SHA)
        self.assertTrue(all(run_cfg['source_sha256'][p] == s for p, s in f.m.PINNED.items()))
        self.assertEqual(Path(run_cfg['queue']).parent, self.work)
        self.assertEqual(Path(run_cfg['out']).parent, self.work)
        self.assertNotEqual(c.file_ref(self.work / 'music_flamingo_val_panel.capture_config.json')['sha256'],
                            c.file_ref(self.work / 'music_flamingo_val_panel.run_config.json')['sha256'])
        claim = json.loads((self.work / 'music_flamingo_val_panel.dispatch.json').read_text())
        c.verify_file(claim['capture'])
        command = popen.call_args.args[0]
        self.assertEqual(command[:5], ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10'])
        self.assertNotIn('-p', command)
        self.assertIn("runpy.run_path", command[-1])
        self.assertNotIn('nohup', command[-1])
        self.assertNotIn('setsid', command[-1])
        self.assertFalse(command[-1].endswith('&'))
        self.assertEqual(run.call_count, 1)
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(popen.call_args.kwargs['stdin'], subprocess.DEVNULL)
        self.assertIs(self.coord.handles[0], child)
        with patch.object(c.subprocess, 'Popen') as again:
            with self.assertRaises(RuntimeError):
                self.coord.dispatch('N2', 'val_panel')
            again.assert_not_called()

    def test_capture_signals_or_wrong_gpu_blocks_Popen(self):
        self.freeze()
        value = json.loads(self.capture().stdout)
        value['signals_sent'] = True
        with patch.object(c.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, json.dumps(value), '')),                 patch.object(c.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError, 'capture'):
                self.coord.dispatch('N2', 'val_panel')
            popen.assert_not_called()
        self.assertFalse((self.work / 'music_flamingo_val_panel.dispatch.json').exists())

    def test_simulated_Popen_spawn_failure_retains_claim(self):
        self.freeze()
        with patch.object(c.subprocess, 'run', return_value=self.capture()),                 patch.object(c.subprocess, 'Popen', side_effect=OSError('CPU simulated SSH spawn failure')):
            with self.assertRaises(OSError):
                self.coord.dispatch('N2', 'val_panel')
        self.assertTrue((self.work / 'music_flamingo_val_panel.dispatch.json').is_file())
        with patch.object(c.subprocess, 'run') as ssh:
            with self.assertRaisesRegex(RuntimeError, 'never duplicate'):
                self.coord.dispatch('N2', 'val_panel')
            ssh.assert_not_called()

    def test_simulated_Popen_exit_without_receipts_blocks_progress(self):
        child, _, _ = self.dispatch()
        child.poll.return_value = 255
        with self.assertRaisesRegex(RuntimeError, 'without complete receipts'):
            self.coord.stage_receipt(self.coord.active['music_flamingo'])
        with self.assertRaisesRegex(RuntimeError, 'SSH failed'):
            self.coord.tick()
        self.assertEqual(self.coord.done, {})
        child.terminate.assert_not_called()
        child.kill.assert_not_called()

    def test_own_family_serial_and_other_family_need_not_wait(self):
        child, _, _ = self.dispatch()
        with self.assertRaisesRegex(RuntimeError, 'active'):
            self.coord.dispatch('N2', 'fixed3')
        self.freeze('N4')
        with patch.object(c.subprocess, 'run', return_value=self.capture()),                 patch.object(c.subprocess, 'Popen', return_value=Mock()) as popen:
            self.coord.dispatch('N4', 'val_panel')
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(set(self.coord.active), {'music_flamingo', 'omni_captioner'})
        self.assertIsNone(self.coord.final_ref)

    def test_incomplete_actual_queue_or_failed_worker_not_done(self):
        child, _, _ = self.dispatch()
        item = self.coord.active['music_flamingo']
        self.assertIsNone(self.coord.stage_receipt(item))
        queue = Path(item['cfg']['queue'])
        f.write_json(queue / 'failure.json', dict(error='CPU simulated worker failure'))
        with self.assertRaisesRegex(RuntimeError, 'failure retained'):
            self.coord.stage_receipt(item)
        child.terminate.assert_not_called()

    def prepare_cpu_outputs(self):
        rows = []
        for id_ in self.base.protocol['val32']['ids']:
            audio = self.base.root / (id_ + '.wav')
            audio.write_bytes(b'CPU fake audio ' + id_.encode())
            rows.append(dict(id=id_, source_group_id=id_ + '-group', audio_path=str(audio),
                audio_sha256=c.file_ref(audio)['sha256'], audio_window_seconds=[0.0, 5.0625]))
        rows = f.ev.select_rows(rows, 'val', limit=32, seed=42)
        self.base.protocol['val32']['ids'] = [r['id'] for r in rows]
        self.coord.splits = dict(val=rows, test=rows)
        self.base.protocol['data']['summary'] = f.ev.file_ref(Path(f.write_json(self.base.data / 'summary.json',
            dict(CPU_fake_not_real_GPU=True))['path']))
        selection = self.base.handoff / 'val32_selection.json'
        f.write_json(selection, self.base.protocol['val32']['ids'])
        self.base.protocol['val32']['selection_ref'] = f.ev.file_ref(selection)
        for row in self.base.protocol['fixed3']:
            audio, image = self.base.root / (row['id'] + '.wav'), self.base.root / (row['id'] + '.png')
            audio.write_bytes(b'CPU fake fixed3 audio ' + row['id'].encode())
            image.write_bytes(b'CPU fake image ' + row['id'].encode())
            row.update(audio_path=str(audio), image_path=str(image), audio_ref=f.ev.file_ref(audio),
                image_ref=f.ev.file_ref(image), audio_sha256=c.file_ref(audio)['sha256'], audio_window_seconds=[0.0, 5.0625])
        self.coord.protocol_ref = f.write_json(self.base.handoff / 'protocol.json', self.base.protocol)
        self.coord.protocol = self.base.protocol

    def make_worker_outputs(self, stage='val_panel', diagnostic=False):
        cfg = self.coord.config('N2', stage)
        out = Path(cfg['out'])
        out.mkdir()
        frozen = self.coord.frozen['music_flamingo']
        rows = self.coord.protocol['fixed3'] if stage == 'fixed3' else self.coord.splits['test' if stage == 'test' else 'val']
        expected = [(r, 'original', r) for r in rows] if stage == 'fixed3' else f.ev.make_conditions(rows, stage == 'val_panel')
        records = []
        for source, condition, donor in expected:
            tokens, text = ([7] * 2048, '') if diagnostic else ([7, 2], 'CPU fake raw dance caption')
            native = dict(raw_text=text, new_token_ids=tokens, eos_token_ids=[2])
            record = dict(id=source['id'], audio_ref=f.ev.file_ref(donor['audio_path']), failure=None,
                native_output=native, raw_text=text, new_token_ids=tokens, eos_token_ids=[2],
                generation_budget=2048, generated_tokens=len(tokens), stop_reason=f.ev.classify_stop(tokens, [2], 2048))
            if stage == 'fixed3':
                record['image_ref'] = source['image_ref']
            else:
                record.update(condition=condition, source_group_id=source['source_group_id'], audio_source_id=donor['id'],
                    audio_source_group_id=donor['source_group_id'], audio_sha256=donor['audio_sha256'],
                    audio_window_seconds=donor['audio_window_seconds'],
                    audio_transform='zero_waveform_preserve_length' if condition == 'silence' else 'none')
            records.append(record)
        (out / 'raw.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records))
        manifest = dict(schema='music_planner_fixed3_native_eval_v1' if stage == 'fixed3' else 'music_planner_ar_eval_v1',
            family='music_flamingo', complete=True, checkpoint_identity=frozen['identity'], source_ids=[r['id'] for r in rows],
            code=f.ev.file_ref(f.m.FROZEN / ('music_planner_fixed3_eval.py' if stage == 'fixed3' else 'music_planner_eval.py')),
            decoding=f.m.DECODE, human_quality_accepted=False, device='cuda:0' if stage == 'fixed3' else 'cuda')
        if stage == 'fixed3':
            version = 'music-planner-fixed3-native-v1:' + frozen['identity']['fingerprint'] + ':' + frozen['identity']['trainer_state']['sha256']
            handoff = dict(records=[dict(id=row['id'], image_path=row['image_path'], motion_caption=record['raw_text'], caption_version=version)
                                   for row, record in zip(rows, records)])
            manifest.update(h3_handoff_approved=True, pretrained_GPU_generation_executed=True,
                protocol=f.ev.file_ref(self.coord.protocol_ref['path']),
                checkpoint_selection=dict(receipt=f.ev.file_ref(cfg['checkpoint_selection']['path'])),
                h3_records=f.ev.file_ref(f.write_json(out / 'h3_records.json', handoff)['path']))
        else:
            manifest.update(checkpoint=frozen['checkpoint'], mode='test' if stage == 'test' else 'val', test_used_for_checkpoint_selection=False,
                data_summary=self.coord.protocol['data']['summary'], split=self.coord.protocol['data']['splits']['test' if stage == 'test' else 'val'],
                adapter_code=f.ev.file_ref(f.m.FROZEN / 'music_planner_adapters.py'),
                selection_ref=self.coord.protocol['val32']['selection_ref'] if stage == 'val_panel' else None)
        f.write_json(out / 'manifest.json', manifest)
        return cfg, manifest, records

    def test_CPU_positive_actual_receipt_resource_safe_gate_and_diagnostics(self):
        self.prepare_cpu_outputs()
        child, _, _ = self.dispatch()
        item = self.coord.active['music_flamingo']
        cfg, manifest, records = self.make_worker_outputs(diagnostic=True)
        self.coord.check_outputs(cfg, manifest, records)
        queue = Path(cfg['queue'])
        complete = dict(technical_collection_complete=True, complete_native_GPU_outputs_verified=True,
            quality_accepted=False, human_quality_accepted=False,
            manifest=f.ev.file_ref(Path(cfg['out']) / 'manifest.json'), raw=f.ev.file_ref(Path(cfg['out']) / 'raw.jsonl'))
        f.write_json(queue / 'eval.complete.json', complete)
        f.write_json(queue / 'resources_safe.json', dict(**{k: True for k in c.SAFE_KEYS}, admitted_gpu_count=1,
            training_complete=False))
        f.write_json(queue / 'val_panel.exit.json', dict(exit_code=0))
        f.write_json(queue / 'integrity_receipt.json', dict(config_sha256=item['config']['sha256'], manager_sha256=c.MANAGER_SHA,
            checkpoint_identity=self.coord.frozen['music_flamingo']['identity'], protocol=self.coord.protocol_ref,
            **{k: cfg[k] for k in ('training_launch', 'full_complete', 'resources_safe')}))
        f.write_json(queue / 'admission.json', dict(gpus=[0], other_seven_GPUs_untouched=True,
            config_sha256=item['config']['sha256'], source_sha256=cfg['source_sha256'],
            holder={k: v for k, v in item['cfg']['holder'].items() if k not in ('gpu', 'uuid')}))
        receipt = self.coord.stage_receipt(item)
        self.assertTrue(receipt['technical_collection_complete'])
        self.assertTrue(receipt['resources_safe'])
        self.assertFalse(receipt['h3_rendered'])
        self.assertFalse(receipt['human_quality_accepted'])
        child.terminate.assert_not_called()
        child.kill.assert_not_called()
        f.write_json(queue / 'val_panel.exit.json', dict(exit_code=1))
        with self.assertRaisesRegex(RuntimeError, 'worker exit'):
            self.coord.stage_receipt(item)

    def test_fixed3_strict_threecase_handoff_stop_schema(self):
        self.prepare_cpu_outputs()
        self.freeze_all()
        self.coord.build_final_selection()
        cfg, manifest, records = self.make_worker_outputs('fixed3')
        self.coord.check_outputs(cfg, manifest, records)
        manifest['h3_handoff_approved'] = False
        with self.assertRaisesRegex(RuntimeError, 'strict effective'):
            self.coord.check_outputs(cfg, manifest, records)
        manifest['h3_handoff_approved'] = True
        with self.assertRaisesRegex(RuntimeError, 'incomplete actual raw|passthrough mismatch'):
            self.coord.check_outputs(cfg, manifest, records[:2])
        records[0]['raw_text'] = ''
        records[0]['native_output']['raw_text'] = ''
        with self.assertRaisesRegex(RuntimeError, 'passthrough|strict stop'):
            self.coord.check_outputs(cfg, manifest, records)

    def test_final_test500_membership_and_protocol_gate(self):
        self.prepare_cpu_outputs()
        rows = [dict(self.coord.splits['val'][index % 32], id='CPU-test-%04d' % index) for index in range(500)]
        self.coord.splits['test'] = f.ev.select_rows(rows, 'test', limit=500, seed=42)
        self.freeze_all()
        self.coord.build_final_selection()
        cfg, manifest, records = self.make_worker_outputs('test')
        self.coord.check_outputs(cfg, manifest, records)
        self.assertEqual(len(records), 500)
        self.assertTrue(all(r['condition'] == 'original' for r in records))
        with self.assertRaisesRegex(RuntimeError, 'incomplete actual raw'):
            self.coord.check_outputs(cfg, manifest, records[:-1])
        manifest['test_used_for_checkpoint_selection'] = True
        with self.assertRaisesRegex(RuntimeError, 'budget/source/decoding'):
            self.coord.check_outputs(cfg, manifest, records)

    def test_output_paths_canonical_and_nonoverlapping(self):
        with self.assertRaises(RuntimeError):
            c.validate_paths(dict(queue=str(self.work / 'q'), out=str(self.work / 'q/sub')), self.base.handoff)
        with self.assertRaises(RuntimeError):
            c.validate_paths(dict(queue='/tmp/not_handoff_q', out=str(self.work / 'o')), self.base.handoff)


class StaticCPU(unittest.TestCase):
    def test_dry_run_is_read_only_no_SSH_Popen_or_GPU_dispatch(self):
        m = c.load_manager()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(c, 'load_manager', return_value=m), \
                patch.object(m, 'EXP', Path(directory)), \
                patch.object(c.subprocess, 'run', side_effect=AssertionError('no SSH')), \
                patch.object(c.subprocess, 'Popen', side_effect=AssertionError('no GPU')), \
                patch.object(c, 'coordinator_lock', side_effect=AssertionError('no lock write')):
            self.assertEqual(c.main(['--dry-run']), 0)
            queue = Path(directory) / 'queues/music_flamingo_fixed2s_clean_v11'
            queue.mkdir(parents=True)
            (queue / 'failure.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'failure retained'):
                c.main(['--dry-run'])

    def test_incomplete_holdout_is_rejected_before_dispatch(self):
        m = c.load_manager()
        helpers = m.load_helpers()
        with patch.object(helpers[1], 'holdout_rows', return_value=[]),                 patch.object(c.subprocess, 'run') as ssh, patch.object(c.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError, 'incomplete/changed fixed3'):
                c.static_inputs(m, helpers[1], helpers[2])
            ssh.assert_not_called()
            popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
