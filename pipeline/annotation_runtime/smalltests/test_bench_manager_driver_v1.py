"""CPU-only integration: native lifecycle reuse, fresh config, effective state, no free GPU driver."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location('bench_manager', RT / 'bench_manager_driver_v1.py')
d = importlib.util.module_from_spec(s)
s.loader.exec_module(d)


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original = json.loads(d.SOURCE_CFG.read_bytes())
        self.source_path = self.root / 'original.json'
        self.source_path.write_bytes(d.bench.encoded(self.original))
        self.source_sha = d.base.sha(self.source_path)
        self.capture_path = self.root / 'capture.json'
        import time
        proofs=[]
        for name, phase in d.SAFE_PHASES.items():
            safe=self.root/'queues'/name/'resources_safe.json'
            safe.parent.mkdir(parents=True)
            safe.write_bytes(d.bench.encoded(dict(phase=phase,all_holders_safe=True,locks_released=True,leases_restored=True)))
            proofs.append(d.ref(safe))
        self.capture = dict(schema='visual_bench_native_capture_v1',node='N6', gpus=[2,3], mode='consumer', signals_sent=False,
            captured_unix=time.time(),physical_addresses=[d.HOST],headroom=dict(headroom=d.base.SAFETY+d.base.LOAD+1),
            lock_inventory={str(g):dict(exists=False,held=[]) for g in [2,3]},
            lease_inventory={str(g):dict(exists=False) for g in [2,3]},resource_safe_refs=proofs,
            holders=[dict(gpu=2,pid=111,uuid='cpu-fake-2'),dict(gpu=3,pid=112,uuid='cpu-fake-3')])
        self.capture_path.write_bytes(d.bench.encoded(self.capture))
        self.capture_ref = d.ref(self.capture_path)
        self.output = self.root / 'visual_batch_fixed3_benchmark_CPU'
        self.output.mkdir()
        self.manifest_path = self.output / 'manifest.json'
        self.manifest = d.bench.strict((d.bench.DEFAULT_OUT / 'manifest.json').read_bytes())
        self.manifest['output'] = str(self.output)
        self.manifest_path.write_bytes(d.bench.encoded(self.manifest))
        self.patches = [patch.object(d,'SOURCE_CFG',self.source_path),patch.object(d,'SOURCE_CFG_SHA',self.source_sha),
            patch.object(d,'BAR',self.root),patch.object(d.bench,'BAR',self.root),patch.object(d.c,'validate',lambda cfg: None)]
        for p in self.patches:p.start()
        self.cfgpath = self.root / 'benchmarkcfg.json'
        self.queue = self.root / 'queues/benchmark'

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def build(self):
        return d.build_config(self.source_path,self.source_sha,self.capture_ref,self.manifest_path,self.cfgpath,self.queue,'bench_CPU')

    def test_builder_preserves_original_worker_and_fixed_budget(self):
        before = self.source_path.read_bytes()
        cfg = self.build()
        self.assertEqual(cfg['targetcount'],3)
        self.assertEqual(cfg['input_count'],222939)
        self.assertEqual(cfg['gpus'],[2,3])
        self.assertEqual(cfg['node'],'N6')
        self.assertEqual(cfg['env'],{**self.original['env'],'CUDA_VISIBLE_DEVICES':'2,3'})
        self.assertEqual(cfg['original_consumer_argv'],self.original['argv'])
        self.assertEqual(cfg['original_consumer_out'],self.original['out'])
        self.assertEqual(cfg['modelref'],self.original['modelref'])
        self.assertEqual(cfg['plan'],self.original['plan'])
        self.assertEqual(cfg['manifest_summary'],self.original['manifest_summary'])
        self.assertIn('--bench-driver',cfg['argv'])
        self.assertEqual(d.fixed_selected(cfg),(list(d.bench.SOURCES),222939))
        self.assertEqual(self.source_path.read_bytes(),before)
        self.assertFalse(self.queue.exists())
        self.assertTrue(all(not Path(p).exists() for p in cfg['original_retry_budget_paths']))
        with self.assertRaises(FileExistsError):self.build()

    def test_exact_safe_proofs_reject_wrong_queue_duplicate_and_phase(self):
        refs = self.capture['resource_safe_refs']
        d.safe_proofs(refs)
        wrong = self.root/'queues/curated_consumer_s0_v1/resources_safe.json'
        wrong.parent.mkdir(parents=True)
        wrong.write_bytes(d.bench.encoded(dict(phase='curated_review_extension_v1',
            all_holders_safe=True, locks_released=True, leases_restored=True)))
        with self.assertRaisesRegex(ValueError, 'exact N6'):
            d.safe_proofs([d.ref(wrong), refs[1]])
        with self.assertRaisesRegex(ValueError, 'exact N6'):
            d.safe_proofs([refs[0], refs[0]])
        path = Path(refs[0]['path'])
        value = d.bench.strict(path.read_bytes())
        value['phase'] = 'wrong-phase'
        path.write_bytes(d.bench.encoded(value))
        with self.assertRaisesRegex(ValueError, 'not safely restored'):
            d.safe_proofs([d.ref(path), refs[1]])

    def test_removed_N8_resume_branch_and_original_base_identity(self):
        self.assertFalse(hasattr(d, 'build_resume'))
        self.assertEqual(Path(d.base.__file__).resolve(), RT/'resource_manager.py')
        self.assertEqual(d.base.sha(d.base.__file__), d.BASE_SHA)

    def test_existing_started_budget_refuses_new_config(self):
        (self.output/'started.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'already started'):self.build()
        self.assertFalse(self.cfgpath.exists())

    def test_changed_fixed_source_windows_denied(self):
        self.manifest['cases'][0]['window_count']=3
        self.manifest_path.write_bytes(d.bench.encoded(self.manifest))
        with self.assertRaisesRegex(ValueError,'scope differs'):self.build()

    def test_driver_denied_without_real_parent_before_any_loader(self):
        cfg=self.build()
        with patch.object(d,'parent_gate',side_effect=ValueError('native parent missing')):
            with self.assertRaisesRegex(ValueError,'parent missing'):d.driver(self.cfgpath,self.manifest_path)
        self.assertFalse((self.output/'driver.started.json').exists())
        self.assertFalse((self.output/'native_model_load.receipt.json').exists())
        self.assertNotIn('vllm',sys.modules)

    def test_capture_ttl_physical_host_headroom_and_flock_failclosed(self):
        import copy
        import time
        for change, message in [({'captured_unix':time.time()-1201},'TTL'),
                                ({'physical_addresses':['not-N6']},'physical'),
                                ({'headroom':dict(headroom=0)},'headroom')]:
            capture=copy.deepcopy(self.capture)
            capture.update(change)
            with self.assertRaisesRegex(ValueError,message):d.validate_capture(capture)
        capture=copy.deepcopy(self.capture)
        capture['lock_inventory']['2']['held']=['occupied']
        with self.assertRaisesRegex(ValueError,'FLOCK'):d.validate_capture(capture)
        with patch.object(d.subprocess,'check_output',return_value=d.HOST),patch.object(d.base,'apps',return_value=[]),\
             patch.object(d.base,'identity',return_value={}),patch.object(d.base,'uuid_for',return_value='fake'),\
             patch.object(d.base,'check_holder') as check_holder,patch.object(d,'lock_snapshot',return_value=dict(exists=True,held=[])),\
             patch.object(d,'lease_snapshot',return_value=dict(exists=False)):
            with self.assertRaisesRegex(ValueError,'exact FLOCK/lease'):d.verify_capture_current(self.capture)
            check_holder.assert_called_once()

    def test_model_load_OOM_is_one_shot_no_request_resubmit(self):
        from types import ModuleType
        cfg=self.build()
        fake=ModuleType('vllm')
        calls=[]
        def load(**kwargs):
            calls.append(kwargs)
            raise MemoryError('CPU simulated model-load OOM')
        fake.LLM=load
        fake.SamplingParams=object
        with patch.dict(sys.modules,{'vllm':fake}),patch.object(d,'parent_gate',return_value=dict(admitted=True,caller_owned=True)):
            with self.assertRaisesRegex(MemoryError,'OOM'):d.driver(self.cfgpath,self.manifest_path)
            with self.assertRaisesRegex(ValueError,'already started'):d.driver(self.cfgpath,self.manifest_path)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['max_num_seqs'],8)
        self.assertTrue((self.output/'driver.started.json').exists())
        self.assertTrue((self.output/'original_budget_guard.exit.json').exists())
        self.assertFalse((self.output/'started.json').exists())
        self.assertFalse((self.output/'calls.jsonl').exists())
        self.assertTrue(all(not Path(p).exists() for p in cfg['original_retry_budget_paths']))

    def test_actual_effective_vllm_state_not_static_flag(self):
        cfg=self.build()
        loadpath=self.output/'load.json'
        loadpath.write_bytes(d.bench.encoded(dict(model_load_completed=True,model_identity_sha256=d.bench.MODEL_SHA,
            source_reference_sha256=d.bench.MODEL_REF_SHA)))
        model=SimpleNamespace(model='/opt/models/Qwen3-VL-32B-Instruct',dtype='torch.bfloat16',quantization=None,
            max_model_len=16384,seed=0,mm_processor_kwargs={'max_pixels':200704},
            multimodal_config=SimpleNamespace(get_limit_per_prompt=lambda key:16))
        vc=SimpleNamespace(model_config=model,scheduler_config=SimpleNamespace(max_num_seqs=8),
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            cache_config=SimpleNamespace(gpu_memory_utilization=0.88,enable_prefix_caching=True),
            structured_outputs_config=SimpleNamespace(backend='xgrammar',disable_any_whitespace=True))
        llm=SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=vc),_ms_bar_model_sha256=d.bench.MODEL_SHA)
        state=d.actual_state(llm,cfg,d.ref(loadpath))
        self.assertEqual(state,d.bench.EXPECTED_STATE)
        vc.scheduler_config.max_num_seqs=2
        self.assertEqual(d.actual_state(llm,cfg,d.ref(loadpath))['max_num_seqs'],2)
        model.mm_processor_kwargs['max_pixels']=123
        self.assertEqual(d.actual_state(llm,cfg,d.ref(loadpath))['max_pixels'],123)
        vc.cache_config.enable_prefix_caching=False
        with self.assertRaisesRegex(ValueError,'cache policy'):d.actual_state(llm,cfg,d.ref(loadpath))

    def test_native_manager_main_reused_without_signal_or_gpu_in_CPU_test(self):
        with patch.object(d.base,'main') as main:
            d.manager()
            main.assert_called_once_with()
            self.assertEqual(Path(d.base.__file__).resolve(),RT/'resource_manager.py')
            self.assertEqual(d.base.sha(d.base.__file__),d.BASE_SHA)
            self.assertEqual(d.base.validate_pair.__name__,'manager_validate')
            self.assertIs(d.base.selected_sources,d.fixed_selected)
            self.assertIs(d.base.stage_summaries,d.summaries)

    def test_parent_gate_requires_live_exact_flock_lease_and_gpu_tree(self):
        import os
        cfg = self.build()
        cfg.update(_config_path=str(self.cfgpath),_config_sha=d.base.sha(self.cfgpath))
        manager_pid,worker_pid=11001,11002
        parent=dict(pid=manager_pid,uid=os.getuid(),argv=['/opt/venv_vllm/bin/python','-B',str(d.SELF),'--config',str(self.cfgpath),
            '--config-sha',cfg['_config_sha'],'--manager-sha',d.BASE_SHA],start_ticks=10,cgroup='CPU-fake',CVD=[],exe='/cpu/python',ppid=1)
        child=dict(pid=worker_pid,uid=os.getuid(),argv=cfg['argv'],start_ticks=11,cgroup='CPU-fake',CVD=['CUDA_VISIBLE_DEVICES=2,3'],exe='/cpu/python',ppid=manager_pid)
        self.queue.mkdir(parents=True)
        (self.queue/'admission.json').write_bytes(d.bench.encoded(dict(manager_identity=parent,config_sha256=cfg['_config_sha'],
            out=cfg['out'],gpus=[2,3],node='N6',manager_sha256=d.BASE_SHA,source_sha256=cfg['source_sha256'],
            holders=[dict(uuid='cpu_uuid_2'),dict(uuid='cpu_uuid_3')])))
        (self.queue/'live.json').write_bytes(d.bench.encoded(dict(child_identity=child)))
        proc=self.root/'proc'
        fds=proc/str(manager_pid)/'fd'
        fds.mkdir(parents=True)
        tmp=self.root/'tmp'
        tmp.mkdir()
        lock_lines=[]
        for gpu in [2,3]:
            lock=tmp/('ms12k_gpu_%d.lock'%gpu)
            lease=tmp/('od_reserve_%d'%gpu)
            lock.touch()
            lease.touch()
            (fds/str(gpu*2)).symlink_to(lock)
            (fds/str(gpu*2+1)).symlink_to(lease)
            lock_lines.append('1: FLOCK ADVISORY WRITE %d %x:%x:%d 0 EOF'%(manager_pid,os.major(lock.stat().st_dev),os.minor(lock.stat().st_dev),lock.stat().st_ino))
        (proc/'locks').write_text('\n'.join(lock_lines))
        original_readlink=os.readlink
        def mapped_path(*args):
            path=Path(*args)
            text=str(path)
            if text.startswith('/proc'):
                return self.root/text.lstrip('/')
            if text.startswith('/tmp/ms12k_gpu_') or text.startswith('/tmp/od_reserve_'):
                return self.root/text.lstrip('/')
            return path
        def identity(pid):
            if pid==manager_pid:return parent
            if pid==worker_pid:return child
            return dict(uid=999,cgroup='unknown',ppid=1)
        with patch.object(d,'Path',mapped_path),patch.object(d.os,'getpid',return_value=worker_pid),patch.object(d.os,'getppid',return_value=manager_pid),\
             patch.object(d.os,'readlink',side_effect=lambda p:original_readlink(p).replace(str(tmp),'/tmp')),\
             patch.object(d.base,'identity',side_effect=identity),patch.object(d.base,'uuid_for',side_effect=lambda gpu:'cpu_uuid_%d'%gpu),\
             patch.object(d.base,'apps',return_value=[]):
            self.assertTrue(d.parent_gate(cfg,before_load=True)['caller_owned'])
            (proc/'locks').write_text('')
            with self.assertRaisesRegex(ValueError,'FLOCK'):d.parent_gate(cfg)
            (proc/'locks').write_text('\n'.join(line.replace(str(manager_pid),str(manager_pid+9)) for line in lock_lines))
            with self.assertRaisesRegex(ValueError,'FLOCK'):d.parent_gate(cfg)
            (proc/'locks').write_text('\n'.join(lock_lines))
            parent['argv'][-1]=d.base.sha(d.SELF)
            with self.assertRaisesRegex(ValueError,'identity|argv'):d.parent_gate(cfg)
            parent['argv'][-1]=d.BASE_SHA
            with patch.object(d.base,'apps',return_value=[dict(uuid='cpu_uuid_2',pid=999)]):
                with self.assertRaisesRegex(ValueError,'unknown GPU'):d.parent_gate(cfg)
            (tmp/'od_reserve_2').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'lease'):d.parent_gate(cfg)

    def test_stage_summary_requires_actual16_generation_batches_not_fakequality(self):
        cfg=self.build()
        result=dict(status='completed',actual_model_request_count=16,original_boundary_unchanged=True,
            production_counts=0,production_claims=0,variants={})
        for name,sizes in [('A',[2,2,2,2]),('B',[8])]:
            records = []
            bar = d.bench.pure_bar()
            for item in self.manifest['requests']:
                root = self.output / 'generations' / name / item['source_id'] / item['window_id']
                root.mkdir(parents=True)
                value = dict(events=[], part_support=[dict(part=p,frame_indices=[],reason='unknown') for p in bar.PARTS],
                             uncertainties=[],abstain=True)
                raw = json.dumps(value)
                filtered = bar.filter_supported_observation(value,item['manifest'],{p:set(v) for p,v in item['forbidden'].items()})
                rawpath,genpath,filterpath = root/'response.raw.txt',root/'generation.json',root/'filtered.json'
                rawpath.write_text(raw)
                filterpath.write_bytes(d.bench.encoded(filtered))
                genpath.write_bytes(d.bench.encoded(dict(model_call_performed=True,model_identity_sha256=d.bench.MODEL_SHA,
                    key=item['key'], request_sha256=item['refs']['request.json']['sha256'],
                    binding_sha256=item['refs']['binding.json']['sha256'],completion_tokens=20,finish_reason='stop',
                    raw_response_sha256=d.base.sha(rawpath))))
                records.append(dict(key=item['key'],finish_reason='stop',completion_tokens=20,raw_response=raw,
                    filtered_observation=filtered,raw_response_ref=d.ref(rawpath),generation_ref=d.ref(genpath),filtered_ref=d.ref(filterpath)))
            result['variants'][name]=dict(batches=[dict(size=n) for n in sizes],records=records)
        (self.output/'result.json').write_bytes(d.bench.encoded(result))
        (self.output/'original_budget_guard.exit.json').write_bytes(d.bench.encoded(dict(unchanged=True)))
        summaries=d.summaries(cfg,self.output)
        self.assertEqual(next(iter(summaries.values()))['model_request_count'],16)
        result['actual_model_request_count']=15
        (self.output/'result.json').write_bytes(d.bench.encoded(result))
        with self.assertRaisesRegex(ValueError,'count'):d.summaries(cfg,self.output)


if __name__=='__main__':unittest.main()
