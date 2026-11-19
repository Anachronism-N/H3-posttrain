"""Self-contained CPU tests; no capture, processes, signals, CUDA or vLLM imports."""
import copy
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

RT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('primary_driver_test', RT / 'caption_8b_primary_driver_v1.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
p = d.p


def config():
    mm = NS(mm_processor_kwargs={'max_pixels': 200704}, get_limit_per_prompt=lambda kind: {'image': 16, 'video': 0}[kind])
    return NS(model_config=NS(model=p.MODEL, tokenizer=p.MODEL, dtype='torch.bfloat16', quantization=None,
        max_model_len=16384, seed=0, trust_remote_code=False, mm_processor_kwargs=None, multimodal_config=mm),
        scheduler_config=NS(max_num_seqs=8), parallel_config=NS(tensor_parallel_size=1),
        structured_outputs_config=NS(backend='xgrammar', disable_any_whitespace=True),
        cache_config=NS(enable_prefix_caching=True, gpu_memory_utilization=0.88))


class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs, self.calls = kwargs, []
        self.vllm_config = config()

    def chat(self, messages, samples, **kwargs):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(samples)))
        core = p.original_filter(RT / 'source_curated_consumer_v1')
        raw = dict(events=[dict(id='e1', description='The dancer shifts the torso.', parts=['torso'], frame_indices=[0, 1])],
            part_support=[dict(part=part, frame_indices=[0, 1] if part == 'torso' else [],
                reason='visible' if part == 'torso' else 'unknown') for part in core.PARTS],
            uncertainties=['CPU fake only; not quality evidence'], abstain=False)
        return [NS(prompt_token_ids=[1, 2], outputs=[NS(text=json.dumps(raw), token_ids=[3, 4, 5], finish_reason='stop')])
                for _ in messages]


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='8b-driver-cpu-')
        self.root = Path(self.tmp.name)
        self.bar = self.root / 'bar'
        self.bar.mkdir()
        self.queue = self.bar / 'queues/prior'
        self.queue.mkdir(parents=True)
        self.bpatch = patch.object(d, 'BAR', self.bar)
        self.bpatch.start()
        self.addCleanup(self.bpatch.stop)
        self.mpatch = patch.object(d, 'PREPARED_MANIFEST', self.bar / 'manifestCPU_v2/manifest.json')
        self.mpatch.start()
        self.addCleanup(self.mpatch.stop)
        self.addCleanup(self.tmp.cleanup)

    def save(self, path, value):
        return p.write_once(path, value)

    def predecessor_fixture(self):
        original = dict(pid=11, uid=os.getuid(), argv=d.base.RESTORE_ARGV, start_ticks=1, cgroup='/cpu',
            CVD=['CUDA_VISIBLE_DEVICES=3'], exe='/opt/venv_eval/bin/python')
        current = dict(original, pid=12, start_ticks=2, gpu=3, uuid='CPU-fake-uuid', sig_ign=0, sig_blk=0)
        producer, plan = dict(path='/cpu/producer', sha256='a'), dict(path='/cpu/plan', sha256='b')
        admission = dict(node='N6', gpus=[2, 3], manager_sha256=d.BASE_SHA, model_identity_sha256=p.SOURCE_MODEL_SHA,
            phase='cpu_32b_bench', holders=[dict(gpu=2, uuid='CPU-fake-uuid-2', identity=dict(original, pid=21)),
                dict(gpu=3, uuid='CPU-fake-uuid', identity=original)], producer=producer, plan=plan)
        a = self.save(self.queue / 'admission.json', admission)
        self.save(self.queue / 'worker_exit.json', dict(exit_code=0, phase=admission['phase'], source_unchanged=True,
            summaries={'cpu-summary': {'count': 16}}, producer=producer, plan=plan))
        self.save(self.queue / 'holder_restored_gpu3.json', dict(gpu=3, uuid='CPU-fake-uuid', original_identity=original,
            holder_identity=current, managed_parent_wait=True, restore_argv=d.base.RESTORE_ARGV))
        safe = dict(phase=admission['phase'], all_holders_safe=True, locks_released=True, leases_restored=True,
                    managed_parent_wait=True, unknown_jobs_not_signaled=True)
        s = self.save(self.queue / 'resources_safe.json', safe)
        cap = dict(schema='caption_8b_primary_native_capture_v1', node='N6', gpus=[3], mode='consumer',
            captured_unix=time.time(), physical_addresses=[d.HOST], holders=[current], signals_sent=False,
            headroom={'headroom': d.base.SAFETY + d.base.LOAD}, lock_inventory={'3': {'exists': False, 'held': []}},
            lease_inventory={'3': {'exists': False}})
        return a, s, cap

    def prepared_fixture(self):
        root = self.bar / 'manifestCPU_v2'
        root.mkdir()
        core = p.original_filter(RT / 'source_curated_consumer_v1')
        records = []
        for sid, count in zip(p.CASES, [2, 3, 3]):
            for i in range(count):
                window = dict(window_id='w%d' % i, start=i * 2, end=i * 2 + 2)
                request = dict(model=p.MODEL, messages=[{'role': 'user', 'content': 'CPU fake body RGB'}],
                    sampling={'temperature': 0.0, 'seed': 0, 'max_tokens': 1800}, json_schema=core.OUTPUT_SCHEMA)
                records.append(dict(id=sid, window=window, request=request,
                    source_request_without_model_sha256=p.digest({k: v for k, v in request.items() if k != 'model'}),
                    manifest=dict(window=window, frames=[{'index': 0, 'time': i * 2}, {'index': 1, 'time': i * 2 + 1}]), forbidden={}))
        manifest = dict(case_ids=list(p.CASES), requests=records, settings=p.SETTINGS, original_filter_code_refs=core.code_refs,
            model_identity={'sha256': p.MODEL_SHA}, request_delta=['model'], budgets=dict(new_8B_window_calls=8,
            independent_32B_reviews=0, source_style_reruns=0, production_attempt_calls=0, production_repair_budget=0))
        mref = self.save(root / 'manifest.json', manifest)
        return NS(manifest=manifest, manifest_ref=mref, core=core)

    def worker_fixture(self):
        prepared = self.prepared_fixture()
        out = Path(prepared.manifest_ref['path']).parent / 'native-run'
        out.mkdir()
        cfg = dict(pilot_manifest=prepared.manifest_ref, out=str(out), modelref={'sha256': 'cpu-stage'})
        cfgpath = self.root / 'worker-config.json'
        cfg['config_path'] = str(cfgpath)
        self.save(cfgpath, cfg)
        stack = ExitStack()
        stack.enter_context(patch.object(d, 'validate', side_effect=lambda c: d.no_claims(c['pilot_manifest']['path'])))
        stack.enter_context(patch.object(d, 'prepared_scope', return_value=prepared))
        stack.enter_context(patch.object(d, 'parent_gate', return_value=dict(admitted=True, node='N6', gpus=[3], tp=1, test_only=True)))
        stack.enter_context(patch.object(d, 'check_model', return_value={}))
        self.addCleanup(stack.close)
        return cfgpath, prepared, out

    def test_CPU_build_has_explicit_proofs_and_frozen_native_single_card_argv(self):
        a, s, cap = self.predecessor_fixture()
        prepared = self.prepared_fixture()
        capture = self.save(self.root / 'capture.json', cap)
        args = NS(pilot_manifest=Path(prepared.manifest_ref['path']), manifest_sha=prepared.manifest_ref['sha256'],
            capture=Path(capture['path']), capture_sha=capture['sha256'], prior_admission=Path(a['path']),
            prior_admission_sha=a['sha256'], prior_resources_safe=Path(s['path']), prior_resources_safe_sha=s['sha256'],
            modelref=RT / 'reviewer8b_stage_N6_extension_v2.json', modelref_sha=d.STAGE_SHA,
            config_out=self.root / 'primary.config.json', out=Path(prepared.manifest_ref['path']).parent / 'new-native',
            queue=self.bar / 'queues/new', phase='cpu_8b_native')
        with patch.object(d, 'MANIFEST_SHA', prepared.manifest_ref['sha256']), patch.object(p, 'load_prepared', return_value=prepared), \
             patch.object(d.base, 'apps') as apps:
            result = d.build_config(args)
            cfg = d.checked(result)
            apps.assert_not_called()
            self.assertEqual(cfg['gpus'], [3])
            self.assertEqual(cfg['argv'], [d.PYTHON, '-B', str(d.SELF), 'worker', '--config-file', str(args.config_out)])
            self.assertEqual(cfg['source_sha256'][str(RT / 'resource_manager.py')], d.BASE_SHA)
            self.assertEqual(cfg['prior_admission'], a)
            self.assertEqual(cfg['prior_resources_safe'], s)
            self.assertIn(str(self.queue / 'holder_restored_gpu3.json'), cfg['source_sha256'])
            self.assertFalse(args.out.exists())
            self.assertFalse(args.queue.exists())
            cfg['argv'].append('--not-native')
            with self.assertRaisesRegex(ValueError, 'direct worker argv'):
                d.validate(cfg)

    def test_pinned_CPU_only_contract_is_preserved_not_a_GPU_reservation(self):
        root = d.PREPARED_MANIFEST.parent
        directory = root/'cpu_fake_batch8'
        directory.mkdir(parents=True)
        claim = dict(admission=dict(test_only=True), manifest_ref=dict(path=str(d.PREPARED_MANIFEST),sha256=d.MANIFEST_SHA))
        claim_path=directory/'run.claim.json'
        receipt_path=directory/'benchmark.receipt.json'
        self.save(claim_path,claim)
        self.save(receipt_path,dict(measured_real_model=False))
        before=(claim_path.read_bytes(),receipt_path.read_bytes())
        with patch.object(d,'CPU_FAKE_CLAIM_SHA',d.base.sha(claim_path)), patch.object(d,'CPU_FAKE_RECEIPTS',{'cpu_fake_batch8':d.base.sha(receipt_path)}):
            d.no_claims(d.PREPARED_MANIFEST)
            self.assertEqual(before,(claim_path.read_bytes(),receipt_path.read_bytes()))
            receipt_path.write_bytes(p.encoded(dict(measured_real_model=True)))
            with patch.object(d,'CPU_FAKE_RECEIPTS',{'cpu_fake_batch8':d.base.sha(receipt_path)}):
                with self.assertRaisesRegex(ValueError,'CPU Fake'):
                    d.no_claims(d.PREPARED_MANIFEST)
        (root/'native_primary.claim.json').touch()
        with self.assertRaisesRegex(ValueError,'reserved/claimed'):
            d.no_claims(d.PREPARED_MANIFEST)

    def test_original_base_is_unrebound_and_pinned(self):
        d.original_base()
        self.assertEqual(Path(d.base.__file__).resolve(), RT / 'resource_manager.py')
        with patch.object(d.base, '__file__', str(d.SELF)), self.assertRaisesRegex(ValueError, 'never be rebound'):
            d.original_base()

    def test_marker_only_runtime_denied(self):
        marker = NS(_ms_bar_model_sha256=p.MODEL_SHA, _ms_8b_pilot_settings=p.SETTINGS)
        with self.assertRaises(AttributeError):
            d.actual_state(marker)

    def test_actual_runtime_checks_every_required_field(self):
        llm = FakeLLM()
        self.assertTrue(d.actual_state(llm)['verified_actual_vllm_config'])
        cases = [('model_config', 'model', '/wrong'), ('model_config', 'tokenizer', '/wrong'),
            ('model_config', 'dtype', 'torch.float16'), ('model_config', 'quantization', 'awq'),
            ('model_config', 'seed', 1), ('model_config', 'max_model_len', 8192),
            ('model_config', 'trust_remote_code', True),
            ('parallel_config', 'tensor_parallel_size', 2), ('scheduler_config', 'max_num_seqs', 4),
            ('structured_outputs_config', 'backend', 'auto'), ('structured_outputs_config', 'disable_any_whitespace', False),
            ('cache_config', 'enable_prefix_caching', False), ('cache_config', 'gpu_memory_utilization', .95)]
        for group, key, value in cases:
            with self.subTest(group=group, key=key):
                llm.vllm_config = config()
                setattr(getattr(llm.vllm_config, group), key, value)
                with self.assertRaisesRegex(ValueError, 'actual vllm_config'):
                    d.actual_state(llm)
        llm.vllm_config = config()
        llm.vllm_config.model_config.multimodal_config.get_limit_per_prompt = lambda kind: 8
        with self.assertRaises(ValueError):
            d.actual_state(llm)

    def test_actual_multimodal_processor_config_not_deprecated_model_field(self):
        llm = FakeLLM()
        llm.vllm_config.model_config.mm_processor_kwargs = {'max_pixels': 100}
        self.assertTrue(d.actual_state(llm)['verified_actual_vllm_config'])
        llm.vllm_config.model_config.multimodal_config.mm_processor_kwargs = {'max_pixels': 100}
        with self.assertRaisesRegex(ValueError, 'actual vllm_config'):
            d.actual_state(llm)

    def test_engine_runtime_config_supported_without_markers(self):
        llm = NS(llm_engine=NS(vllm_config=config()))
        self.assertEqual(d.actual_state(llm)['tensor_parallel_size'], 1)

    def test_actual_sampling_keeps_schema_and_token_budget(self):
        request = dict(sampling={'temperature': 0.0, 'seed': 0, 'max_tokens': 1800}, json_schema={'type': 'object'})
        result = d.sampling(request, NS, NS)
        self.assertEqual(result.structured_outputs.json, request['json_schema'])
        self.assertEqual(result.max_tokens, 1800)
        result.structured_outputs.json['type'] = 'array'
        self.assertEqual(request['json_schema']['type'], 'object')
        with self.assertRaisesRegex(ValueError, 'actual sampling'):
            d.sampling(request, lambda **kw: NS(**dict(kw, max_tokens=1200)), NS)

    def test_predecessor_and_capture_bind_actual_restored_native_holder(self):
        a, s, cap = self.predecessor_fixture()
        self.assertEqual(d.validate_capture(cap, a, s)['pid'], 12)
        cap['holders'][0]['pid'] = 13
        with self.assertRaisesRegex(ValueError, 'actual native restored'):
            d.validate_capture(cap, a, s)

    def test_proof_booleans_alone_cannot_pass_native_predecessor(self):
        a, s, cap = self.predecessor_fixture()
        (self.queue / 'holder_restored_gpu3.json').unlink()
        with self.assertRaises(FileNotFoundError):
            d.predecessor(a, s)

    def test_predecessor_wrong_node_failed_stage_missing_safe_denied(self):
        a, s, cap = self.predecessor_fixture()
        with patch.object(d, 'checked', side_effect=lambda r: dict(d.p.strict(Path(r['path']).read_bytes()), node='N7') if r == a else d.p.strict(Path(r['path']).read_bytes())):
            with self.assertRaisesRegex(ValueError, 'N6 GPU2/3'):
                d.predecessor(a, s)
        self.save(self.queue / 'failure.json', {'error': 'CPU fake'})
        with self.assertRaisesRegex(ValueError, 'failure'):
            d.predecessor(a, s)

    def test_capture_old_prelaunch_wrong_gpu_and_occupied_lock_denied(self):
        a, s, good = self.predecessor_fixture()
        for key, value in [('captured_unix', time.time() - 1), ('gpus', [7]), ('signals_sent', True), ('node', 'N7')]:
            cap = copy.deepcopy(good)
            cap[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                d.validate_capture(cap, a, s)
        cap = copy.deepcopy(good)
        cap['lock_inventory']['3']['held'] = ['CPU fake FLOCK held']
        with self.assertRaisesRegex(ValueError, 'lock/lease'):
            d.validate_capture(cap, a, s)

    def test_cross_queue_safe_denied(self):
        a, s, cap = self.predecessor_fixture()
        outside = self.save(self.root / 'resources_safe.json', {'all_holders_safe': True})
        with self.assertRaisesRegex(ValueError, 'same native prior queue'):
            d.predecessor(a, outside)

    def test_prepared_exact_SHA_and_request_budget(self):
        prepared = self.prepared_fixture()
        with self.assertRaisesRegex(ValueError, 'exact original'):
            d.prepared_scope(prepared.manifest_ref)
        with patch.object(d, 'MANIFEST_SHA', prepared.manifest_ref['sha256']), patch.object(p, 'load_prepared', return_value=prepared):
            self.assertIs(d.prepared_scope(prepared.manifest_ref), prepared)
            prepared.manifest['requests'][0]['request']['sampling']['max_tokens'] = 1200
            with self.assertRaisesRegex(ValueError, 'token budget'):
                d.prepared_scope(prepared.manifest_ref)

    def test_same_SHA_in_other_root_cannot_bypass_unique_prepared_root(self):
        other = self.bar / 'other-root/manifest.json'
        reference = dict(path=str(other), sha256=d.MANIFEST_SHA)
        with patch.object(p, 'load_prepared') as load:
            with self.assertRaisesRegex(ValueError, 'unique original prepared'):
                d.prepared_scope(reference)
            load.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'unique original prepared'):
            d.no_claims(other)
        self.assertFalse(other.exists())

    def test_existing_reserved_or_claimed_root_never_redispatched(self):
        prepared = self.prepared_fixture()
        root = Path(prepared.manifest_ref['path']).parent
        d.no_claims(prepared.manifest_ref['path'])
        self.save(root / 'old-failed/run.claim.json', {'reserved_unique_window_calls': 8})
        with self.assertRaisesRegex(ValueError, 'no redispatch'):
            d.no_claims(prepared.manifest_ref['path'])

    def test_runtime_rechecks_actual_state_and_no_more_than_eight_calls(self):
        llm, gate = FakeLLM(), NS(calls=0)
        def gated():
            gate.calls += 1
        runtime = d.Runtime(llm, gated)
        runtime.chat(['CPU'] * 8, [NS()] * 8)
        self.assertEqual(runtime.attempted_window_calls, 8)
        with self.assertRaisesRegex(ValueError, 'exhausted'):
            runtime.chat(['CPU'], [NS()])
        self.assertEqual(len(llm.calls), 1)
        llm.vllm_config.parallel_config.tensor_parallel_size = 2
        with self.assertRaisesRegex(ValueError, 'actual vllm_config'):
            runtime.chat([], [])

    def test_native_worker_generates_exact_eight_and_preserves_source_fact_text(self):
        cfgpath, prepared, out = self.worker_fixture()
        receipt = d.worker(cfgpath, factory=(FakeLLM, NS, NS))
        self.assertEqual(receipt['new_window_calls'], 8)
        self.assertFalse(receipt['measured_real_model'])
        self.assertFalse(receipt['quality_approved'])
        self.assertEqual([len(c['captions']) for c in receipt['candidates']], [2, 3, 3])
        self.assertTrue(all(c['full_window_coverage'] for c in receipt['candidates']))
        for c in receipt['candidates']:
            self.assertTrue(all(row['caption'] == 'The dancer shifts the torso.' for row in c['captions']))
        load = p.strict((out / 'native_model_load.receipt.json').read_bytes())
        self.assertTrue(load['actual_runtime']['verified_actual_vllm_config'])
        self.assertEqual(len(list((out / 'annotation').rglob('generation.json'))), 8)
        self.assertTrue((out / 'original_budget_guard.exit.json').exists())
        with self.assertRaisesRegex(ValueError, 'no redispatch'):
            d.worker(cfgpath, factory=(lambda **kw: self.fail('redispatch'), NS, NS))

    def test_zero_generation_model_load_failure_is_native_terminal_preserved(self):
        cfgpath, prepared, out = self.worker_fixture()
        def failed(**kwargs):
            raise RuntimeError('CPU fake load OOM')
        with self.assertRaisesRegex(RuntimeError, 'load OOM'):
            d.worker(cfgpath, factory=(failed, NS, NS))
        failure = p.strict((out / 'native_driver.failure.json').read_bytes())
        self.assertEqual(failure['attempted_window_calls'], 0)
        self.assertEqual(failure['reserved_unique_window_calls'], 8)
        self.assertTrue(failure['native_parent_restores_resources'])
        self.assertTrue((out / 'annotation/run.claim.json').exists())
        self.assertFalse(list(out.rglob('generation.json')))
        self.assertTrue((out / 'original_budget_guard.exit.json').exists())
        with self.assertRaisesRegex(ValueError, 'no redispatch'):
            d.worker(cfgpath, factory=(FakeLLM, NS, NS))

    def test_kernel_flock_inode_device_matching(self):
        st = (self.root / 'bar').stat()
        line = '1: FLOCK ADVISORY WRITE 101 %x:%x:%d 0 EOF' % (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)
        self.assertTrue(d.kernel_lock_matches(line, st))
        self.assertFalse(d.kernel_lock_matches(line.replace('FLOCK', 'POSIX'), st))

    def test_manager_keeps_original_main_and_original_manager_SHA(self):
        originals = {name: getattr(d.base, name) for name in ('validate_pair', 'selected_sources', 'check_model', 'stage_summaries', 'check_sources')}
        original_file = d.base.__file__
        try:
            with patch.object(d.base, 'main') as main:
                d.manager()
                main.assert_called_once_with()
            self.assertEqual(d.base.__file__, original_file)
            self.assertIs(d.base.check_model, d.check_model)
            self.assertIs(d.base.stage_summaries, d.summaries)
            with self.assertRaisesRegex(ValueError, 'original base'):
                d.base.check_sources({}, '/cpu/no-config', 'cpu', d.base.sha(d.SELF))
        finally:
            for name, value in originals.items():
                setattr(d.base, name, value)

    def test_current_capture_recheck_is_default_deny_wrong_physical_node(self):
        a, s, cap = self.predecessor_fixture()
        capref = self.save(self.root / 'capture.json', cap)
        cfg = dict(holder_capture=capref, prior_admission=a, prior_resources_safe=s)
        with patch.object(d.subprocess, 'check_output', return_value='127.0.0.1'), patch.object(d.base, 'apps') as apps:
            with self.assertRaisesRegex(ValueError, 'not N6'):
                d.verify_capture_current(cfg)
            apps.assert_not_called()

    def parent_fixture(self):
        queue = self.bar / 'queues/current'
        queue.mkdir()
        cfg = dict(queue=str(queue), out=str(self.bar / 'native-run'), config_path=str(self.root / 'config.json'),
            _config_sha='CPU-fake-config-sha', argv=[d.PYTHON, '-B', str(d.SELF), 'worker', '--config-file', str(self.root / 'config.json')],
            source_sha256={str(d.SELF): d.base.sha(d.SELF)}, producer=d.ref(d.SELF))
        parent = dict(pid=101, uid=os.getuid(), argv=[d.PYTHON, '-B', str(d.SELF), '--config', cfg['config_path'],
            '--config-sha', cfg['_config_sha'], '--manager-sha', d.BASE_SHA], start_ticks=1, cgroup='/cpu',
            CVD=['CUDA_VISIBLE_DEVICES=3'], exe=d.PYTHON)
        child = dict(parent, pid=202, argv=cfg['argv'], start_ticks=2)
        admission = dict(manager_identity=parent, node='N6', gpus=[3], out=cfg['out'], manager_sha256=d.BASE_SHA,
            config_sha256=cfg['_config_sha'], model_identity_sha256=p.MODEL_SHA, source_sha256=cfg['source_sha256'],
            producer=cfg['producer'], holders=[dict(gpu=3, uuid='CPU-fake-uuid')])
        self.save(queue / 'admission.json', admission)
        self.save(queue / 'live.json', {'child_identity': child})
        lock, lease = self.root / 'fake-lock', self.root / 'fake-lease'
        lock.touch()
        lease.touch()
        target = {'/tmp/ms12k_gpu_3.lock': lock, '/tmp/od_reserve_3': lease}
        st = lock.stat()
        line = '1: FLOCK ADVISORY WRITE 101 %x:%x:%d 0 EOF' % (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)
        original_lstat, original_iterdir, original_readtext = Path.lstat, Path.iterdir, Path.read_text
        stack = ExitStack()
        stack.enter_context(patch.object(d.os, 'getpid', return_value=202))
        stack.enter_context(patch.object(d.os, 'getppid', return_value=101))
        stack.enter_context(patch.object(d.base, 'identity', side_effect=lambda pid: parent if pid == 101 else child))
        stack.enter_context(patch.object(d.base, 'uuid_for', return_value='CPU-fake-uuid'))
        apps = stack.enter_context(patch.object(d.base, 'apps', return_value=[]))
        def fake_iterdir(path):
            return iter([lock, lease]) if str(path) == '/proc/101/fd' else original_iterdir(path)
        stack.enter_context(patch.object(Path, 'iterdir', fake_iterdir))
        stack.enter_context(patch.object(Path, 'lstat', lambda path: original_lstat(target.get(str(path), path))))
        stack.enter_context(patch.object(Path, 'read_text', lambda path, *a, **kw: line if str(path) == '/proc/locks' else original_readtext(path, *a, **kw)))
        stack.enter_context(patch.object(d.os, 'readlink', side_effect=lambda path: next(name for name, actual in target.items() if actual == path)))
        self.addCleanup(stack.close)
        return cfg, parent, child, lock, lease, apps

    def test_parent_gate_requires_original_argv_identity_fds_flock_and_lease(self):
        cfg, parent, child, lock, lease, apps = self.parent_fixture()
        self.assertTrue(d.parent_gate(cfg, before_load=True)['admitted'])
        parent['argv'][-1] = d.base.sha(d.SELF)
        with self.assertRaisesRegex(ValueError, 'parent identity differs'):
            d.parent_gate(cfg)

    def test_parent_gate_missing_parent_fds_or_stale_lease_denied(self):
        cfg, parent, child, lock, lease, apps = self.parent_fixture()
        original_iterdir = Path.iterdir
        with patch.object(Path, 'iterdir', lambda path: iter([]) if str(path) == '/proc/101/fd' else original_iterdir(path)):
            with self.assertRaisesRegex(ValueError, 'FD inode missing'):
                d.parent_gate(cfg)
        os.utime(lease, (time.time() - 60, time.time() - 60))
        with self.assertRaisesRegex(ValueError, 'heartbeat stale'):
            d.parent_gate(cfg)

    def test_parent_gate_unknown_gpu_tree_and_preload_app_denied(self):
        cfg, parent, child, lock, lease, apps = self.parent_fixture()
        apps.return_value = [{'uuid': 'CPU-fake-uuid', 'pid': 999}]
        with self.assertRaisesRegex(ValueError, 'before native admitted load'):
            d.parent_gate(cfg, before_load=True)
        stranger = dict(child, pid=999, ppid=1, cgroup='/foreign')
        with patch.object(d.base, 'identity', side_effect=lambda pid: parent if pid == 101 else child if pid == 202 else stranger):
            with self.assertRaisesRegex(ValueError, 'unknown GPU app'):
                d.parent_gate(cfg)
        apps.return_value = [{'uuid': 'CPU-fake-uuid', 'pid': 202}]
        self.assertTrue(d.parent_gate(cfg)['admitted'])

    def test_full_staged_model_check_hashes_only_nonweight_inputs(self):
        root = self.root / 'raw-eight-b'
        root.mkdir()
        files = {}
        for name in ['model-%d.safetensors' % i for i in range(4)] + ['config-%d.json' % i for i in range(11)]:
            path = root / name
            path.write_bytes(('CPU fake ' + name).encode())
            st = path.stat()
            files[name] = dict(path=str(path), sha256=d.base.sha(path), stat=dict(size=st.st_size,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns, device=st.st_dev, inode=st.st_ino))
        identity = dict(root=str(root), files=files)
        identity['sha256'] = p.identity_digest(identity)
        origin = dict(complete=True, full_copy_SHA_verified=True, model='Qwen3-VL-8B-Instruct', files=files)
        stage = dict(node='N6', complete=True, full_copy_SHA_verified=True, destination_disk_SHA_rehashed=True,
            source_and_destination_SHA_verified=True, model_identity=identity, origin_ref={'path': '/cpu/origin', 'sha256': 'cpu'})
        cfg = dict(modelref=dict(path=str(RT / 'reviewer8b_stage_N6_extension_v2.json'), sha256=d.STAGE_SHA,
            model_identity_sha256=identity['sha256']), pilot_manifest={'path': '/cpu/prepared', 'sha256': 'cpu'})
        original_sha = d.base.sha
        with patch.object(p, 'MODEL', str(root)), patch.object(p, 'MODEL_SHA', identity['sha256']), \
             patch.object(d, 'checked', side_effect=lambda r: origin if r == stage['origin_ref'] else stage), \
             patch.object(d, 'prepared_scope', return_value=NS(manifest={'declared_checkpoint': p.checkpoint(identity)})), \
             patch.object(d.base, 'sha', wraps=original_sha) as sha:
            d.check_model(cfg)
            self.assertEqual(sha.call_count, 11)
            self.assertTrue(all(not str(c.args[0]).endswith('.safetensors') for c in sha.call_args_list))
            changed = root / 'model-0.safetensors'
            changed.write_bytes(b'changed weight stat must reject without rehash')
            with self.assertRaisesRegex(ValueError, 'stat changed'):
                d.check_model(cfg)

    def test_stage_pins_real_modelref_not_untrusted_marker(self):
        cfg = dict(modelref={'path': str(RT / 'reviewer8b_stage_N6_extension_v2.json'), 'sha256': d.STAGE_SHA,
                            'model_identity_sha256': p.MODEL_SHA})
        stage = d.checked(cfg['modelref'])
        self.assertEqual(len(stage['model_identity']['files']), 15)
        self.assertEqual(sum(n.endswith('.safetensors') for n in stage['model_identity']['files']), 4)
        self.assertEqual(p.identity_digest(stage['model_identity']), p.MODEL_SHA)
        origin = d.checked(stage['origin_ref'])
        self.assertEqual({n: r['sha256'] for n, r in stage['model_identity']['files'].items()},
                         {n: r['sha256'] for n, r in origin['files'].items()})
        altered = copy.deepcopy(stage)
        altered['node'] = 'N7'
        with patch.object(d, 'checked', return_value=altered), self.assertRaisesRegex(ValueError, 'N6 extension'):
            d.check_model(cfg)


if __name__ == '__main__':
    unittest.main()
