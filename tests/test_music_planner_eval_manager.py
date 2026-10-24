"""CPU fake-byte receipts exercise real frozen helpers; never proof of real GPU safety."""
import copy
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts_omnidance/exp/ms/music_planner_eval_manager.py'
spec = importlib.util.spec_from_file_location('safe_eval_under_test', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
manager, fixed, ev = m.load_helpers()
ORIGINAL_RT = m.RT


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return dict(path=str(path), sha256=m.sha(path))


class CPUHandoff(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.exp = self.root / 'experiment'
        self.rt = self.exp / 'runtime'
        self.data = self.root / 'data'
        self.handoff = self.exp / 'eval_handoff_clean_v10'
        self.handoff.mkdir(parents=True)
        self.data.mkdir()
        for split, count in (('train', 10713), ('val', 490), ('test', 500)):
            (self.data / (split + '.jsonl')).write_text('{}\n' * count)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for key, value in (('EXP', self.exp), ('RT', self.rt), ('DATA', self.data), ('HANDOFF', self.handoff)):
            self.stack.enter_context(patch.object(m, key, value))
        self.stack.enter_context(patch.object(manager, 'EXPERIMENT', self.exp))
        # Only host-address discovery is faked. No GPU query, worker, signal, or SSH is executed.
        self.stack.enter_context(patch.object(manager.stage.socket, 'getaddrinfo', return_value=[
            (None, None, None, None, ('28.216.17.70', 0)),
            (None, None, None, None, ('28.216.19.143', 0)),
            (None, None, None, None, ('28.216.18.144', 0))]))
        self.protocol = dict(code=[dict(path=p, sha256=s) for p, s in m.PINNED.items()],
            data=dict(path=str(self.data), splits={split: ev.file_ref(self.data / (split + '.jsonl'))
                                                for split in ('train', 'val', 'test')}), decoding=m.DECODE,
            val32=dict(ids=['id%02d' % i for i in range(32)], selection_ref=dict(path='/fake/val32.json')),
            fixed3=[dict(id=id_) for id_ in fixed.IDS])
        self.protocol_ref = write_json(self.handoff / 'protocol.json', self.protocol)
        self.cfg, self.result, self.path = self.family('N2')

    def family(self, node):
        host, family = m.SCOPE[node]
        launch = json.loads((ORIGINAL_RT / (family + '_launch_clean_v10.json')).read_text())
        old_out = Path(launch['out'])
        out = self.exp / 'runs' / (family + '_fixed2s_clean_v10')
        train = out / 'train'
        train.mkdir(parents=True)
        launch['out'] = str(out)
        launch['queue'] = str(self.exp / 'queues' / (family + '_fixed2s_clean_v10'))
        for phase in ('smoke', 'resume_smoke', 'full'):
            launch[phase + '_result'] = str(train / Path(launch[phase + '_result']).name)
            argv = [arg.replace(str(old_out), str(out)) for arg in launch[phase + '_argv']]
            for split in ('train', 'val'):
                index = argv.index('--' + split) + 1
                previous = argv[index]
                argv[index] = str(self.data / (split + '.jsonl'))
                launch['source_sha256'].pop(previous, None)
                launch['source_sha256'][argv[index]] = m.sha(argv[index])
                launch['dataset'][split] = dict(path=argv[index], sha256=m.sha(argv[index]),
                                               count=10713 if split == 'train' else 490)
            launch[phase + '_argv'] = argv
        launch_ref = write_json(self.rt / (family + '_launch_clean_v10.json'), launch)
        config = dict(family=family, data=dict(test_used=False,
            train=dict(sources=10713, sha256=m.sha(self.data / 'train.jsonl')),
            val=dict(sources=490, sha256=m.sha(self.data / 'val.jsonl'))), max_length=8192,
            trainer_sha256=m.PINNED[str(m.FROZEN / 'music_planner_train.py')],
            adapter_source_sha256=m.PINNED[str(m.FROZEN / 'music_planner_adapters.py')], model={})
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        config['fingerprint'] = fingerprint
        run_config = write_json(train / 'run_config.json', config)
        marker_refs = {}
        for update, nll in ((335, 2.0), (670, 1.0)):
            path = train / ('checkpoint-%07d' % update)
            names = ['trainable.pt', 'scheduler.pt', 'adapter/adapter_config.json',
                     'adapter/adapter_model.safetensors', 'adapter/planner_adapter.json']
            names += ['%s-rank%d.%s' % (kind, rank, suffix) for rank in range(8)
                      for kind, suffix in (('rng', 'pt'), ('writer', 'json'))]
            names += ['distributed/zero3/zero_pp_rank_%d_mp_rank_00_%s_states.pt' % (rank, kind)
                      for rank in range(8) for kind in ('optim', 'model')]
            refs = []
            for name in names:
                item = path / name
                item.parent.mkdir(parents=True, exist_ok=True)
                item.write_bytes(b'CPU fake tensor bytes: ' + name.encode())
                refs.append(ev.file_ref(item))
            marker = dict(schema='music-planner-train-v1', complete=True, world_size=8, zero3=True,
                fingerprint=fingerprint, trainable_sha256=m.sha(path / 'trainable.pt'), files=refs,
                cursor=dict(epoch=update // 335, position=0, update=update,
                            source_exposures=10713 * (update // 335), best_val_nll=nll))
            marker_refs[update] = write_json(path / 'trainer_state.json', marker)
            validation = dict(nll=nll, assistant_tokens=999, source_count=490,
                              comparison_scope='within-family-tokenizer-only', test_used=False)
            write_json(path / 'validation.json', validation)
        path = train / 'checkpoint-0000670'
        validation = json.loads((path / 'validation.json').read_text())
        selection = write_json(train / 'best.json', dict(checkpoint=str(path), **validation))
        marker = json.loads((path / 'trainer_state.json').read_text())
        result = dict(schema='music-planner-outputs-v1', family=family, mode='full', world_size=8,
            test_selection_performed=False, optimizer_updates=670, epoch_budget=2, epochs_completed=2,
            source_exposure_budget=21426, source_exposures=21426, full_budget_completed=True,
            checkpoint_complete=True, train_count=10713, val_count=490, global_batch=32,
            starting_update=3, invocation_optimizer_updates=667,
            initialization='official_base_then_matching_checkpoint', resume_verified=True,
            verified_new_process=True, resumed_next_update_performed=True, fingerprint=fingerprint,
            cursor=marker['cursor'], checkpoint=ev.file_ref(path / 'trainer_state.json'),
            checkpoint_files=marker['files'], run_config=ev.file_ref(train / 'run_config.json'),
            source_sha256={s: m.sha(self.data / (s + '.jsonl')) for s in ('train', 'val')})
        result_ref = write_json(Path(launch['full_result']), result)
        complete = write_json(Path(launch['queue']) / 'full.complete.json',
            dict(result=result, result_path=result_ref['path'], result_sha256=result_ref['sha256']))
        safe = write_json(Path(launch['queue']) / 'resources_safe.json', dict(all_holders_safe=True,
            locks_released=True, leases_restored=True, managed_parent_wait=True,
            unknown_jobs_not_signaled=True, admitted_gpu_count=8))
        cfg = dict(schema='music-planner-eval-manager-v1', node=node, family=family, host=host,
            gpus=[0], release_authorized=True, job='val_panel', queue=str(self.handoff / (family + '_queue')),
            out=str(self.handoff / (family + '_out')), training_launch=launch_ref, full_complete=complete,
            resources_safe=safe, protocol=self.protocol_ref, checkpoint_selection=selection,
            checkpoint_marker=marker_refs[670], fingerprint=fingerprint,
            source_sha256={**m.PINNED, str(SCRIPT): m.sha(SCRIPT)})
        return cfg, result, path

    def select(self, cfg=None, result=None, full=True):
        return m.selected_checkpoint(cfg or self.cfg, result or self.result, self.protocol, fixed, ev, full_hash=full)

    def update_complete(self, change):
        full = json.loads(Path(self.cfg['full_complete']['path']).read_text())
        full['result'].update(change)
        result_ref = write_json(Path(full['result_path']), full['result'])
        full['result_sha256'] = result_ref['sha256']
        self.cfg['full_complete'] = write_json(Path(self.cfg['full_complete']['path']), full)

    def test_correct_scope_and_real_helper_completion(self):
        m.validate_scope(self.cfg)
        _, result = m.check_completed(self.cfg, manager)
        self.assertEqual(result['optimizer_updates'], 670)
        self.assertIn('best_val_nll', result['cursor'])

    def test_wrong_node_family_host_and_gpu_rejected(self):
        for change in (dict(node='N4'), dict(family='moss'), dict(host='wrong'),
                       dict(gpus=[1]), dict(gpus=[False]), dict(gpus=[0, 1]), dict(release_authorized=False)):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                m.validate_scope(dict(self.cfg, **change))

    def test_missing_completion_rejected(self):
        Path(self.cfg['full_complete']['path']).unlink()
        with self.assertRaises(FileNotFoundError):
            m.check_completed(self.cfg, manager)

    def test_incomplete_real_budget_rejected(self):
        for change in (dict(optimizer_updates=669), dict(epochs_completed=1), dict(source_exposures=21425),
                       dict(full_budget_completed=False), dict(test_selection_performed=True)):
            self.update_complete(change)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                m.check_completed(self.cfg, manager)
            self.update_complete({key: self.result[key] for key in change})

    def test_unsafe_training_resources_rejected(self):
        self.cfg['resources_safe'] = write_json(Path(self.cfg['resources_safe']['path']),
                                               dict(all_holders_safe=False))
        with self.assertRaises(RuntimeError):
            m.check_completed(self.cfg, manager)

    def test_frozen_receipt_SHA_change_rejected(self):
        Path(self.cfg['checkpoint_selection']['path']).write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'SHA changed'):
            self.select()

    def test_stale_fingerprint_rejected(self):
        with self.assertRaises(RuntimeError):
            self.select(dict(self.cfg, fingerprint='old-fingerprint'))

    def test_wrong_family_checkpoint_real_helper_rejected(self):
        with self.assertRaises(ValueError):
            ev.checkpoint_identity(self.path, 'moss', self.data)

    def test_full_val_selection_and_exact_eight_rank_content(self):
        _, _, audit = self.select()
        self.assertTrue(audit['content']['whole_manifest_inventory_and_content_SHA_verified'])
        self.assertEqual(len([r for r in audit['content']['files'] if 'optim_states.pt' in r['path']]), 8)
        self.assertEqual(len(audit['validation_candidates']), 4)

    def test_val32_cannot_be_used_as_full_val_best(self):
        path = self.path / 'validation.json'
        validation = json.loads(path.read_text())
        validation['source_count'] = 32
        write_json(path, validation)
        self.cfg['checkpoint_selection'] = write_json(Path(self.cfg['checkpoint_selection']['path']),
                                                      dict(checkpoint=str(self.path), **validation))
        with self.assertRaises(ValueError):
            self.select()

    def test_nonminimal_or_cross_tokenizer_best_rejected(self):
        path = self.path.parent / 'checkpoint-0000335'
        validation = json.loads((path / 'validation.json').read_text())
        self.cfg['checkpoint_selection'] = write_json(Path(self.cfg['checkpoint_selection']['path']),
                                                      dict(checkpoint=str(path), **validation))
        self.cfg['checkpoint_marker'] = dict(path=str(path / 'trainer_state.json'),
                                             sha256=m.sha(path / 'trainer_state.json'))
        with self.assertRaisesRegex(RuntimeError, 'minimum'):
            self.select()
        validation['comparison_scope'] = 'across-tokenizers'
        write_json(path / 'validation.json', validation)
        self.cfg['checkpoint_selection'] = write_json(Path(self.cfg['checkpoint_selection']['path']),
                                                      dict(checkpoint=str(path), **validation))
        with self.assertRaises(ValueError):
            self.select()

    def test_checkpoint_legal_late_sidecars_separate_SHA(self):
        write_json(self.path / 'val_generation_request.json', dict(not_GPU_executed=True))
        _, _, audit = self.select()
        self.assertEqual({Path(r['path']).name for r in audit['content']['late_written_sidecars']},
                         {'validation.json', 'val_generation_request.json'})

    def test_checkpoint_missing_rank_extra_escape_and_symlink_rejected(self):
        extra = self.path / 'extra.pt'
        extra.write_bytes(b'extra')
        with self.assertRaises(ValueError):
            self.select()
        extra.unlink()
        extra.symlink_to(self.path / 'trainable.pt')
        with self.assertRaises(RuntimeError):
            self.select()
        extra.unlink()
        (self.path / 'distributed/zero3/zero_pp_rank_7_mp_rank_00_optim_states.pt').unlink()
        with self.assertRaises(ValueError):
            self.select()

    def test_checkpoint_same_size_edit_changes_fresh_stat(self):
        _, _, audit = self.select()
        item = self.path / 'adapter/adapter_model.safetensors'
        before = item.stat()
        item.write_bytes(b'x' * before.st_size)
        os.utime(item, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertNotEqual(m.checkpoint_snapshot(self.path), audit['stat_snapshot'])
        with self.assertRaises(ValueError):
            self.select()

    def test_sources_checked_with_real_helpers_and_config_SHA(self):
        cfgref = write_json(self.root / 'config.json', self.cfg)
        m.check_sources(self.cfg, cfgref['path'], cfgref['sha256'], m.sha(SCRIPT))
        with self.assertRaises(RuntimeError):
            m.check_sources(self.cfg, cfgref['path'], '0' * 64, m.sha(SCRIPT))
        bad = copy.deepcopy(self.cfg)
        bad['source_sha256'][str(m.MANAGER)] = '0' * 64
        with self.assertRaises(RuntimeError):
            m.check_sources(bad, cfgref['path'], cfgref['sha256'], m.sha(SCRIPT))

    def test_existing_lock_nonblocking_busy_and_nofollow(self):
        lock = self.root / 'gpu0.lock'
        lock.touch()
        a = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
        b = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
        try:
            fcntl.flock(a, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                fcntl.flock(b, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(m, 'GPU0_LOCK', lock), self.assertRaises(BlockingIOError):
                m.open_gpu0_lock()
        finally:
            os.close(a)
            os.close(b)
        link = self.root / 'link.lock'
        link.symlink_to(lock)
        with self.assertRaises(OSError):
            os.open(link, os.O_RDWR | os.O_NOFOLLOW)
        with patch.object(m, 'GPU0_LOCK', link), self.assertRaises(RuntimeError):
            m.open_gpu0_lock()
        with patch.object(m, 'GPU0_LOCK', lock):
            owned = m.open_gpu0_lock()
            try:
                lock.rename(self.root / 'old.lock')
                lock.touch()
                with self.assertRaises(RuntimeError):
                    m.check_gpu0_lock(owned)
            finally:
                os.close(owned)

    def test_real_lease_helpers_old_snapshot_atomic_fd_heartbeat_and_restore(self):
        path = self.root / 'lease'
        path.touch()
        os.utime(path, (1, 1))
        old = manager.lease_snapshot(path)
        e = dict(reserve=path, old=old, leasefd=None)
        manager.acquire_lease(e)
        self.assertEqual(e['lease_inode'], (old.st_dev, old.st_ino))
        self.assertGreater(path.stat().st_mtime, old.st_mtime)
        manager.refresh_leases([e])
        os.utime(e['leasefd'], ns=(old.st_atime_ns, old.st_mtime_ns))
        os.close(e['leasefd'])
        self.assertEqual(path.stat().st_mtime_ns, old.st_mtime_ns)
        os.utime(path, None)
        with self.assertRaisesRegex(RuntimeError, 'active lease'):
            manager.lease_snapshot(path)

    def test_lease_replacement_and_unsafe_content_rejected(self):
        path = self.root / 'lease'
        path.touch()
        os.utime(path, (1, 1))
        old = manager.lease_snapshot(path)
        path.unlink()
        path.write_text('foreign')
        with self.assertRaises(RuntimeError):
            manager.acquire_lease(dict(reserve=path, old=old, leasefd=None))
        with self.assertRaises(RuntimeError):
            manager.lease_snapshot(path)

    def test_holder_exact_identity_and_ignored_blocked_SIGTERM(self):
        holder = dict(pid=42, uid=os.getuid(), argv=manager.RESTORE_ARGV, start_ticks=9,
            cgroup='CPU-test-same-cgroup', CVD=['CUDA_VISIBLE_DEVICES=0'], OCCUPY=['OCCUPY_GPU=0'],
            exe='/usr/bin/python', sig_ign=0, sig_blk=0, uuid='CPU-test-GPU0-UUID', gpu=0)
        running = [dict(uuid=holder['uuid'], pid=42)]
        with patch.object(manager, 'identity', return_value=holder):
            manager.check_holder(holder, holder, 0, holder['uuid'], running)
            for change in (dict(pid=43), dict(start_ticks=10), dict(argv=['foreign']), dict(uid=os.getuid()+1),
                           dict(cgroup='foreign'), dict(OCCUPY=['OCCUPY_GPU=1']), dict(CVD=['CUDA_VISIBLE_DEVICES=1']),
                           dict(sig_ign=1 << (signal.SIGTERM-1)), dict(sig_blk=1 << (signal.SIGTERM-1))):
                with self.subTest(change=change), self.assertRaises(RuntimeError):
                    manager.check_holder(holder, dict(holder, **change), 0, holder['uuid'], running)
            with self.assertRaises(RuntimeError):
                manager.check_holder(holder, holder, 0, 'foreign-uuid', running)
            with self.assertRaises(RuntimeError):
                manager.check_holder(holder, holder, 0, holder['uuid'], running + [dict(uuid=holder['uuid'], pid=99)])
        with patch.object(manager.signal, 'pidfd_send_signal') as send:
            e = dict(fd=123, released=False)
            manager.release_holder(e)
            send.assert_called_once_with(123, signal.SIGTERM)
            self.assertTrue(e['released'])

    def test_worker_argv_explicit_CUDA_native_protocol(self):
        for job, expected in (('val_panel', 'cuda'), ('fixed3', 'cuda:0'), ('test', 'cuda')):
            cfg = dict(self.cfg, job=job)
            argv = m.worker_argv(cfg, self.path, self.protocol)
            self.assertEqual(manager.argv_option(argv, '--device'), expected)
            self.assertNotIn('--teacher-forced', argv)
            if job == 'fixed3':
                self.assertEqual(manager.argv_option(argv, '--checkpoint-selection-sha'), cfg['checkpoint_selection']['sha256'])
            else:
                self.assertEqual(manager.argv_option(argv, '--max-new-tokens'), '2048')
                self.assertEqual(manager.argv_option(argv, '--seed'), '42')
        self.assertIn('--counterfactual', m.worker_argv(self.cfg, self.path, self.protocol))

    def final_config(self):
        families = {self.cfg['family']: self.cfg}
        for node in ('N4', 'N7'):
            cfg, _, _ = self.family(node)
            families[cfg['family']] = cfg
        keys = ('training_launch', 'full_complete', 'resources_safe', 'checkpoint_selection',
                'checkpoint_marker', 'fingerprint')
        final = dict(schema='music-planner-final-selection-v1', test_used=False,
            selection_frozen_before_test=True, decoding=m.DECODE, data=self.protocol['data'],
            protocol=self.protocol_ref, families={family: {k: cfg[k] for k in keys} for family, cfg in families.items()})
        self.cfg['job'] = 'test'
        self.cfg['final_selection'] = write_json(self.handoff / 'final_selection.json', final)
        return final

    def test_shared_allcomplete_final_gate_and_once_only_claim(self):
        self.final_config()
        m.check_final_selection(self.cfg, manager, fixed, ev, self.protocol)
        path = m.claim_test(self.cfg)
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            m.claim_test(dict(self.cfg, out=str(self.handoff / 'second-run')))
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(json.loads(original)['automatic_retry'])

    def test_shared_final_cannot_omit_family_or_change_decoding(self):
        final = self.final_config()
        for change in (dict(families={self.cfg['family']: final['families'][self.cfg['family']]}),
                       dict(decoding=dict(m.DECODE, max_new_tokens=1024)), dict(test_used=True)):
            self.cfg['final_selection'] = write_json(self.handoff / 'final_selection.json', dict(final, **change))
            with self.assertRaises(RuntimeError):
                m.check_final_selection(self.cfg, manager, fixed, ev, self.protocol)

    def test_fixed3_cap_and_unknownstop_not_effective_handoff(self):
        _, identity, _ = self.select()
        cfg = dict(self.cfg, job='fixed3')
        out = Path(cfg['out'])
        out.mkdir()
        manifest = dict(complete=True, family=cfg['family'], checkpoint_identity=identity,
            pretrained_GPU_generation_executed=True, h3_handoff_approved=True,
            device='cuda:0', decoding=m.DECODE, human_quality_accepted=False,
            code=ev.file_ref(m.FROZEN / 'music_planner_fixed3_eval.py'),
            protocol=ev.file_ref(cfg['protocol']['path']),
            checkpoint_selection=dict(receipt=ev.file_ref(cfg['checkpoint_selection']['path'])))
        write_json(out / 'manifest.json', manifest)
        records = [dict(id=id_, raw_text='CPU fake text', new_token_ids=[1, 2], eos_token_ids=[2],
            generation_budget=2048, generated_tokens=2, stop_reason='eos',
            native_output=dict(native_stop=False, raw_text='CPU fake text', new_token_ids=[1, 2], eos_token_ids=[2]),
            failure=None, budget_truncation=False) for id_ in fixed.IDS]
        raw = out / 'raw.jsonl'
        raw.write_text('\n'.join(json.dumps(r) for r in records) + '\n')
        evidence = m.check_worker_outputs(cfg, self.path, identity, self.protocol, ev)
        self.assertFalse(evidence['human_quality_accepted'])
        write_json(out / 'manifest.json', dict(manifest, code=dict(path='wrong', sha256='old')))
        with self.assertRaises(RuntimeError):
            m.check_worker_outputs(cfg, self.path, identity, self.protocol, ev)
        write_json(out / 'manifest.json', manifest)
        records[0]['new_token_ids'] = [1] * 2048
        records[0]['native_output']['new_token_ids'] = records[0]['new_token_ids']
        records[0]['generated_tokens'] = 2048
        records[0]['stop_reason'] = 'budget_truncation'
        records[0]['budget_truncation'] = True
        raw.write_text('\n'.join(json.dumps(r) for r in records) + '\n')
        original = raw.read_bytes()
        with self.assertRaises(RuntimeError):
            m.check_worker_outputs(cfg, self.path, identity, self.protocol, ev)
        self.assertEqual(raw.read_bytes(), original)
        records[0]['new_token_ids'] = [1]
        records[0]['native_output']['new_token_ids'] = [1]
        records[0]['generated_tokens'] = 1
        records[0]['stop_reason'] = 'unknown_stop'
        records[0]['budget_truncation'] = False
        raw.write_text('\n'.join(json.dumps(r) for r in records) + '\n')
        with self.assertRaises(RuntimeError):
            m.check_worker_outputs(cfg, self.path, identity, self.protocol, ev)

    def panel_fixture(self, job):
        _, identity, _ = self.select()
        cfg = dict(self.cfg, job=job, out=str(self.handoff / (job + '_raw')))
        out = Path(cfg['out'])
        out.mkdir()
        protocol = copy.deepcopy(self.protocol)
        protocol['data']['summary'] = write_json(self.data / 'summary.json', dict(CPU_fixture=True))
        mode = 'test' if job == 'test' else 'val'
        ids = ['id%03d' % i for i in range(500)] if job == 'test' else protocol['val32']['ids']
        manifest = dict(complete=True, family=cfg['family'], checkpoint_identity=identity,
            checkpoint=str(self.path), device='cuda', decoding=m.DECODE, mode=mode,
            code=ev.file_ref(m.FROZEN / 'music_planner_eval.py'),
            adapter_code=ev.file_ref(m.FROZEN / 'music_planner_adapters.py'),
            data_summary=protocol['data']['summary'], split=protocol['data']['splits'][mode],
            test_used_for_checkpoint_selection=False)
        write_json(out / 'manifest.json', manifest)
        records = []
        for id_ in ids:
            for condition in (('original', 'mismatched', 'silence') if job == 'val_panel' else ('original',)):
                records.append(dict(id=id_, condition=condition, raw_text='CPU fake native text',
                    new_token_ids=[1, 2], eos_token_ids=[2], generation_budget=2048, generated_tokens=2,
                    stop_reason='eos', failure=None, native_output=dict(raw_text='CPU fake native text',
                        new_token_ids=[1, 2], eos_token_ids=[2], native_stop=False)))
        if job == 'test':
            self.stack.enter_context(patch.object(ev, 'validate_data', return_value=(dict(test=ids), {})))
            self.stack.enter_context(patch.object(ev, 'select_rows', return_value=[dict(id=i) for i in ids]))
        return cfg, identity, protocol, records, out / 'raw.jsonl', manifest

    def test_val_and_test_cap_unknownstop_emptyraw_are_quality_diagnostics_not_technical_failure(self):
        for job in ('val_panel', 'test'):
            with self.subTest(job=job):
                cfg, identity, protocol, records, raw, _ = self.panel_fixture(job)
                records[0].update(new_token_ids=[1] * 2048, generated_tokens=2048,
                                  stop_reason='budget_truncation')
                records[0]['native_output']['new_token_ids'] = records[0]['new_token_ids']
                records[1].update(new_token_ids=[1], generated_tokens=1, stop_reason='unknown_stop', raw_text='')
                records[1]['native_output'].update(new_token_ids=[1], raw_text='')
                raw.write_text('\n'.join(json.dumps(r) for r in records) + '\n')
                before = raw.read_bytes()
                result = m.check_worker_outputs(cfg, self.path, identity, protocol, ev)
                self.assertTrue(result['technical_collection_complete'])
                self.assertEqual(result['diagnostic_count'], 2)
                self.assertEqual(result['diagnostic_counts'], dict(budget_truncation=1, unknown_stop=1, empty_raw=1))
                self.assertEqual(result['stop_reason_counts'], dict(eos=len(records)-2, native_stop=0,
                                                                  budget_truncation=1, unknown_stop=1))
                self.assertEqual(result['budget_truncation_count'], 1)
                self.assertEqual(result['empty_raw_count'], 1)
                self.assertEqual(len(records), 500 if job == 'test' else 96)
                self.assertFalse(result['quality_accepted'])
                self.assertFalse(result['human_quality_accepted'])
                self.assertTrue(result['diagnostics_do_not_trigger_retry'])
                self.assertEqual(raw.read_bytes(), before)

    def test_val_and_test_invalid_tokens_stop_mismatch_or_no_native_interface_rejected(self):
        for job in ('val_panel', 'test'):
            cfg, identity, protocol, records, raw, _ = self.panel_fixture(job)
            for kind in ('stop_mismatch', 'over_budget', 'invalid_token', 'zero_tokens', 'no_native',
                         'native_ids_mismatch', 'native_eos_mismatch', 'failure'):
                with self.subTest(job=job, kind=kind):
                    changed = copy.deepcopy(records)
                    if kind == 'stop_mismatch':
                        changed[0]['stop_reason'] = 'budget_truncation'
                    elif kind == 'no_native':
                        changed[0].pop('native_output')
                    elif kind == 'native_ids_mismatch':
                        changed[0]['native_output']['new_token_ids'] = [9]
                    elif kind == 'native_eos_mismatch':
                        changed[0]['native_output']['eos_token_ids'] = [9]
                    elif kind == 'failure':
                        changed[0]['failure'] = dict(type='InterfaceError', message='CPU fake failure')
                    else:
                        tokens = [1] * 2049 if kind == 'over_budget' else [-1] if kind == 'invalid_token' else []
                        changed[0]['new_token_ids'] = tokens
                        changed[0]['native_output']['new_token_ids'] = tokens
                        changed[0]['generated_tokens'] = len(tokens)
                    raw.write_text('\n'.join(json.dumps(r) for r in changed) + '\n')
                    with self.assertRaises(RuntimeError):
                        m.check_worker_outputs(cfg, self.path, identity, protocol, ev)

    def test_fixed3_GPU_not_executed_cannot_approve(self):
        _, identity, _ = self.select()
        cfg = dict(self.cfg, job='fixed3', out=str(self.handoff / 'no_GPU'))
        out = Path(cfg['out'])
        out.mkdir()
        write_json(out / 'manifest.json', dict(complete=True, family=cfg['family'], checkpoint_identity=identity,
            code=ev.file_ref(m.FROZEN / 'music_planner_fixed3_eval.py'),
            protocol=ev.file_ref(cfg['protocol']['path']),
            checkpoint_selection=dict(receipt=ev.file_ref(cfg['checkpoint_selection']['path'])),
            pretrained_GPU_generation_executed=False, h3_handoff_approved=True,
            device='cuda:0', decoding=m.DECODE))
        (out / 'raw.jsonl').write_text('\n'.join(json.dumps(dict(id=i)) for i in fixed.IDS))
        with self.assertRaisesRegex(RuntimeError, 'completed GPU native handoff'):
            m.check_worker_outputs(cfg, self.path, identity, self.protocol, ev)

    def test_real_finish_resources_restores_lease_and_unlocks_only_GPU0(self):
        queue = self.root / 'cpu-resource-queue'
        queue.mkdir()
        lock = self.root / 'gpu0.lock'
        lock.touch()
        lockfd = os.open(lock, os.O_RDWR)
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease = self.root / 'lease0'
        lease.touch()
        os.utime(lease, (1, 1))
        old = manager.lease_snapshot(lease)
        readfd, writefd = os.pipe()
        self.addCleanup(os.close, writefd)
        holder = dict(pid=42, uid=os.getuid(), argv=manager.RESTORE_ARGV, start_ticks=9,
            cgroup='CPU-test-cgroup', CVD=['CUDA_VISIBLE_DEVICES=0'], OCCUPY=['OCCUPY_GPU=0'],
            exe='/usr/bin/python', sig_ign=0, sig_blk=0, uuid='CPU-only-UUID', gpu=0)
        e = dict(gpu=0, uuid=holder['uuid'], expected=holder, identity=holder,
            fd=readfd, reserve=lease, old=old, leasefd=None, released=False)
        manager.acquire_lease(e)
        def save(name, value):
            write_json(queue / name, value)
        with patch.object(manager, 'identity', return_value=holder), \
             patch.object(manager, 'uuid_for', return_value=holder['uuid']) as gpuquery, \
             patch.object(manager, 'apps', return_value=[dict(uuid=holder['uuid'], pid=42)]), \
             patch.object(manager.signal, 'signal') as sig:
            manager.finish_resources([e], [lockfd], queue, save, [], dict(child=None, started=False))
        sig.assert_any_call(signal.SIGTERM, signal.SIG_IGN)
        sig.assert_any_call(signal.SIGINT, signal.SIG_IGN)
        self.assertEqual(lease.stat().st_mtime_ns, old.st_mtime_ns)
        self.assertIsNone(e['leasefd'])
        self.assertTrue(all(call.args == (0,) for call in gpuquery.call_args_list))
        safe = json.loads((queue / 'resources_safe.json').read_text())
        self.assertEqual(safe['admitted_gpu_count'], 1)
        self.assertTrue(safe['leases_restored'])
        checkfd = os.open(lock, os.O_RDWR)
        try:
            fcntl.flock(checkfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(checkfd)

    def test_restore_wait_and_foreign_GPU_job_are_not_signaled_or_respawned(self):
        e = dict(gpu=0, uuid='CPU-only-UUID', fd=123, released=True,
                 queue=self.root, identity=dict(CVD=['CUDA_VISIBLE_DEVICES=0']))
        saved = {}
        def save(name, value):
            saved[name] = value
        with patch.object(manager.select, 'select', return_value=([], [], [])), \
             patch.object(manager.subprocess, 'Popen') as spawn, \
             patch.object(manager.signal, 'pidfd_send_signal') as sends:
            self.assertFalse(manager.restore_step(e, save, []))
            spawn.assert_not_called()
            sends.assert_not_called()
        self.assertTrue(saved['holder_exit_wait_gpu0.json']['lock_and_lease_retained'])
        with patch.object(manager.select, 'select', return_value=([123], [], [])), \
             patch.object(manager, 'apps', return_value=[dict(uuid=e['uuid'], pid=999)]), \
             patch.object(manager.subprocess, 'Popen') as spawn, \
             patch.object(manager.signal, 'pidfd_send_signal') as sends:
            self.assertFalse(manager.restore_step(e, save, []))
            spawn.assert_not_called()
            sends.assert_not_called()
        self.assertTrue(saved['restore_blocked_gpu0.json']['unknown_jobs_not_signaled'])

    def test_current_base_audit_real_SHA_LFS_gitblob_and_same_stat_rewrite(self):
        model = self.root / 'local-model'
        model.mkdir()
        (model / 'config.json').write_bytes(b'{}')
        (model / 'model.safetensors').write_bytes(b'CPU fake local weights')
        files = {}
        copied = {}
        for name in ('config.json', 'model.safetensors'):
            path = model / name
            data = path.read_bytes()
            git = hashlib.sha1(('blob %d\0' % len(data)).encode() + data).hexdigest()
            files[name] = dict(size=len(data), sha256=m.sha(path) if name.endswith('.safetensors') else None,
                               git_blob=git if name == 'config.json' else None)
            copied[name] = dict(size=len(data), sha256=m.sha(path), local_stat=manager.stage.file_stat(path))
        marker = dict(complete=True, family=self.cfg['family'], revision=manager.stage.MODELS[self.cfg['family']][1],
            destination=str(model), download_receipt_sha256='CPU-download-sha', files=copied,
            download_receipt=str(self.root / 'download.json'))
        stage_ref = write_json(self.root / 'stage-receipt.json', marker)
        bridge = dict(official_stage=dict(receipt=stage_ref))
        with patch.object(fixed, 'base_stage_identity', return_value=bridge), \
             patch.object(manager.stage, 'read_download', return_value=(None, 'CPU-download-sha', None, None, files)):
            audit = m.audit_base(model, self.cfg['family'], {}, fixed, ev, manager)
            self.assertTrue(audit['current_full_content_SHA_and_official_digest_verified'])
            self.assertEqual(audit['local_weight_hash_passes_this_audit'], 1)
            weight = model / 'model.safetensors'
            old = weight.stat()
            weight.write_bytes(b'x' * old.st_size)
            os.utime(weight, ns=(old.st_atime_ns, old.st_mtime_ns))
            # Simulate forged stat-only receipt while retaining the original independent content digest.
            marker['files']['model.safetensors']['local_stat'] = manager.stage.file_stat(weight)
            write_json(Path(stage_ref['path']), marker)
            with self.assertRaisesRegex(RuntimeError, 'local base bytes differ'):
                m.audit_base(model, self.cfg['family'], {}, fixed, ev, manager)


if __name__ == '__main__':
    unittest.main()
