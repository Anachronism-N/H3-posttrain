"""CPU-only real native helpers and fake-byte resource regressions; never GPU proof."""
import copy
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / 'scripts_omnidance/exp/ms'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

p = load('base_producer_cpu_test', SCRIPTS / 'music_planner_base_fixed3.py')
m = load('base_manager_cpu_test', SCRIPTS / 'music_planner_base_fixed3_manager.py')
a = load('base_native_adapter_cpu_test', p.ADAPTER)
legacy = load('base_legacy_cpu_fixture', REPO / 'tests/test_music_planner_eval_manager.py')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return p.file_ref(path)


def native_result(text='', tokens=None):
    return dict(raw_text=text, new_token_ids=[9] if tokens is None else tokens, eos_token_ids=[9],
                seed=42, decoding=p.NATIVE_DECODE, audio_input='full-waveform', silence_applied=False,
                processor_sample_rate=16000, n_samples=81000, input_tokens=3, native_stop=False)


class NativeModel(torch.nn.Module):
    def __init__(self, tokens, nonfinite=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.tokens = tokens
        self.nonfinite = nonfinite
        self.argmax_called = False
        self.generation_config = SimpleNamespace(eos_token_id=9)

    def forward(self, input_ids):
        values = torch.ones(1, 1, 10)
        if self.nonfinite is not None:
            values[0, 0, 0] = self.nonfinite
        return SimpleNamespace(logits=values)

    def generate(self, input_ids, **kwargs):
        output = self(input_ids)
        self.argmax_called = True
        output.logits.argmax(-1)
        assert kwargs['do_sample'] is False and kwargs['num_beams'] == 1
        assert kwargs['max_new_tokens'] == 2048 and kwargs['use_cache'] is True
        return torch.cat((input_ids, torch.tensor([self.tokens], dtype=torch.long)), dim=1)


class BaseProducerCPU(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = []
        for i, id_ in enumerate(p.IDS):
            audio = self.root / (id_ + '.wav')
            with wave.open(str(audio), 'wb') as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
                stream.writeframes(bytes([i, 0]) * 81000)
            image = self.root / (id_ + '.png')
            image.write_bytes(b'CPU image binding')
            ref = p.file_ref(audio)
            self.rows.append(dict(id=id_, audio_path=str(audio), audio_sha256=ref['sha256'],
                audio_window_seconds=[0., 5.0625], audio_ref=ref, image_ref=p.file_ref(image)))
        self.old = [dict(row, native_output=native_result()) for row in self.rows]

    def adapter(self, text, tokens, nonfinite=None):
        model = NativeModel(tokens, nonfinite)
        tokenizer = SimpleNamespace(decode=lambda *args, **kw: text, eos_token_id=9)
        adapter = a.PlannerAdapter('music_flamingo', model, SimpleNamespace(tokenizer=tokenizer))
        def prefix(row, silence=False):
            adapter._last_audio_info = dict(processor_sample_rate=16000, n_samples=81000)
            return dict(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
        adapter.prefix = prefix
        return adapter

    def test_real_native_generate_empty_and_budget_are_diagnostics(self):
        for text, tokens, diagnostic in [('', [9], 'empty_raw'), ('description', [1] * 2048, 'budget_truncation'),
                                          ('no windows', [9], 'no_exact_dual_window_format')]:
            with self.subTest(diagnostic=diagnostic):
                adapter = self.adapter(text, tokens)
                record = p.native_record(adapter, self.rows[0], torch)
                self.assertIsNone(record['failure'])
                self.assertEqual(record['raw_text'], text)
                self.assertEqual(record['new_token_ids'], tokens)
                self.assertIn(diagnostic, record['quality_diagnostics'])
                self.assertEqual(record['actual_forward_count'], 1)
                self.assertEqual(record['finite_native_logits_forward_count'], 1)
                self.assertTrue(record['native_logits_all_finite'])
                self.assertFalse(adapter.model._forward_hooks)

    def test_real_native_nan_inf_fail_before_argmax(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            adapter = self.adapter('ignored', [9], value)
            record = p.native_record(adapter, self.rows[0], torch)
            self.assertIn('NaN/Inf', record['failure']['message'])
            self.assertFalse(adapter.model.argmax_called)
            self.assertEqual(record['actual_forward_count'], 1)
            self.assertEqual(record['finite_native_logits_forward_count'], 0)
            self.assertFalse(record['native_logits_all_finite'])
            self.assertFalse(adapter.model._forward_hooks)

    def test_frozen_window_parser_5_0625_and_5_062_and_invalid(self):
        for text in ['[0-2s] dance [2-5.0625s] turn', '[0.000-2.000s] dance [2.000-5.062s] turn']:
            record = p.native_record(self.adapter(text, [9]), self.rows[0], torch)
            self.assertIsNone(record['failure'])
            self.assertTrue(record['matches_fixed_windows'])
            self.assertTrue(record['nonempty_window_text'])
            self.assertTrue(record['dual_window_format'])
        for text in ['[0-1s] dance [2-5.0625s] turn', '[0-2s] dance', '[0-2s] [2-5.0625s] turn']:
            record = p.native_record(self.adapter(text, [9]), self.rows[0], torch)
            self.assertIsNone(record['failure'])
            self.assertFalse(record['dual_window_format'])

    def test_effective_eos_and_input_tokens_must_match_trained(self):
        for change in [dict(eos_token_ids=[10]), dict(input_tokens=4)]:
            old = dict(native_result(), **change)
            record = p.native_record(self.adapter('raw', [9]), self.rows[0], torch, old)
            self.assertIn('effective EOS or native input length', record['failure']['message'])

    def test_no_native_forward_and_interface_error_are_technical_failures(self):
        adapter = self.adapter('raw', [9])
        adapter.generate = lambda *args, **kwargs: native_result('raw')
        self.assertIn('no real finite', p.native_record(adapter, self.rows[0], torch)['failure']['message'])
        adapter.generate = lambda *args, **kwargs: dict(raw_text=12)
        self.assertIsNotNone(p.native_record(adapter, self.rows[0], torch)['failure'])
        for result in [native_result(tokens=[True]), native_result(tokens=[-1]), native_result(tokens=[1] * 2049),
                       dict(native_result(), audio_input='cropped-waveform')]:
            with self.assertRaises(RuntimeError):
                p.validate_result(result)

    def test_exact_three_order_and_audio_binding_rejected(self):
        p.check_rows(self.rows, self.old)
        for rows, trained in [(self.rows[::-1], self.old), (self.rows[:2], self.old[:2]),
                              (self.rows, self.old[::-1])]:
            with self.assertRaises(RuntimeError):
                p.check_rows(rows, trained)
        changed = copy.deepcopy(self.old)
        changed[0]['audio_ref'] = self.rows[1]['audio_ref']
        with self.assertRaisesRegex(RuntimeError, 'binding'):
            p.check_rows(self.rows, changed)
        changed = copy.deepcopy(self.rows)
        changed[0]['audio_window_seconds'] = [0, 2]
        with self.assertRaises(RuntimeError):
            p.check_rows(changed, self.old)

    def test_tokenizer_template_and_full_prefix_exact_match(self):
        def make(template='same', audio=1.):
            tok = SimpleNamespace(get_vocab=lambda: {'a': 1}, get_added_vocab=lambda: {}, model_max_length=8192,
                                  all_special_ids=[9], special_tokens_map={'eos_token': '<end>'},
                                  chat_template=template, encode=lambda *args, **kw: [9])
            obj = SimpleNamespace(tokenizer=tok, processor=SimpleNamespace(chat_template=template),
                                  _prompt=lambda: 'native audio only', max_length=8192)
            obj.prefix = lambda row: dict(input_ids=torch.tensor([[1, 2, 3]]), features=torch.tensor([audio]))
            return obj
        evidence = p.compare_prefixes(make(), make(), self.rows, self.old, torch)
        self.assertEqual(len(evidence['records']), 3)
        self.assertFalse(evidence['base_processor_replaced'])
        self.assertFalse(evidence['records'][0]['trained_historical_prefix_ids_available'])
        serialized_copy = make()
        serialized_copy.tokenizer.chat_template = None
        same_effective = p.compare_prefixes(make(), serialized_copy, self.rows, self.old, torch)
        self.assertFalse(same_effective['tokenizer_chat_template_storage_equal'])
        self.assertTrue(same_effective['effective_processor_prompt_and_all_tensors_exact_equal'])
        ordered = make()
        reordered = make()
        ordered.tokenizer.all_special_ids = [8, 9, 10]
        reordered.tokenizer.all_special_ids = [10, 8, 9]
        p.compare_prefixes(ordered, reordered, self.rows, self.old, torch)
        reordered.tokenizer.all_special_ids = [10, 8, 11]
        with self.assertRaises(RuntimeError):
            p.compare_prefixes(ordered, reordered, self.rows, self.old, torch)
        changed_prompt = make()
        changed_prompt._prompt = lambda: 'different rendered user instruction'
        with self.assertRaises(RuntimeError):
            p.compare_prefixes(make(), changed_prompt, self.rows, self.old, torch)
        for obj in (make(template='changed'), make(audio=2.)):
            with self.assertRaises(RuntimeError):
                p.compare_prefixes(make(), obj, self.rows, self.old, torch)
        other = copy.deepcopy(self.old)
        other[0]['native_output']['input_tokens'] = 4
        with self.assertRaisesRegex(RuntimeError, 'length'):
            p.compare_prefixes(make(), make(), self.rows, other, torch)

    def test_real_frozen_loader_no_checkpoint_or_fresh_lora(self):
        parameter = SimpleNamespace(device=torch.device('cuda:0'), dtype=torch.bfloat16, is_floating_point=lambda: True)
        model = SimpleNamespace(parameters=lambda: iter([parameter]), named_parameters=lambda: [('base.weight', parameter)])
        model.to = Mock(return_value=model)
        model.eval = Mock(return_value=model)
        model.requires_grad_ = Mock(return_value=model)
        cls = SimpleNamespace(from_pretrained=Mock(return_value=model))
        transformers = SimpleNamespace(MusicFlamingoForConditionalGeneration=cls)
        with patch.dict(sys.modules, {'transformers': transformers}),              patch.object(a, 'load_native_processor', return_value=SimpleNamespace()),              patch.object(a, '_cast_parameter_dtype') as cast,              patch.object(a, 'configure_trainable') as lora,              patch.object(a, 'load_planner_adapter', wraps=a.load_planner_adapter) as loader:
            adapter = p.pure_base(a, 'music_flamingo', self.root, 8192, torch)
        loader.assert_called_once_with('music_flamingo', self.root, checkpoint=None, lora_rank=None,
                                      device='cuda:0', dtype=torch.bfloat16, max_length=8192)
        lora.assert_not_called()
        self.assertEqual(adapter.trainable_info, {})
        cast.assert_called_once_with(parameter, torch.bfloat16)
        model.to.assert_called_once_with('cuda:0')
        model.eval.assert_called_once_with()
        model.requires_grad_.assert_called_once_with(False)
        self.assertTrue(adapter.loading_evidence['loading_info_not_recorded'])
        self.assertTrue(adapter.loading_evidence['exact_per_parameter_restore_not_verified'])

    def test_processor_jinja_inventory_and_unlisted_resource_rejected(self):
        export = self.root/'adapter'; export.mkdir()
        template = export/'chat_template.jinja'; template.write_text('native template')
        ref = p.file_ref(template)
        self.assertIn('.jinja', p.PROCESSOR_SUFFIXES)
        p.check_processor_inventory(export, [ref])
        (export/'extra_tokenizer.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'inventory'):
            p.check_processor_inventory(export, [ref])

    def test_native_loading_warning_blocks_and_logs_are_not_restore_proof(self):
        import logging
        def warning_loader(*args, **kwargs):
            logging.getLogger('transformers').warning('Some weights were not initialized and are newly initialized: learned.weight')
            return SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError, 'missing/uninitialized loading warning'):
            p.pure_base(SimpleNamespace(load_planner_adapter=warning_loader), 'moss', self.root, 8192, torch)

    def test_no_cpu_fallback_or_peft_allowed(self):
        for device, names, peft in [('cpu', ['base.weight'], False), ('cuda:0', ['lora_A.weight'], False),
                                    ('cuda:0', ['modules_to_save.weight'], False), ('cuda:0', ['base.weight'], True)]:
            param = SimpleNamespace(device=torch.device(device), dtype=torch.bfloat16, is_floating_point=lambda: True)
            model = SimpleNamespace(parameters=lambda: iter([param]), named_parameters=lambda: [(n, param) for n in names])
            if peft: model.peft_config = {}
            adapters = SimpleNamespace(load_planner_adapter=lambda *args, **kw: SimpleNamespace(
                family='music_flamingo', model=model, trainable_info={}))
            with self.assertRaises(RuntimeError):
                p.pure_base(adapters, 'music_flamingo', self.root, 8192, torch)

    def test_generate_and_manager_outputs_keep_empty_raw_without_retry(self):
        cfg = dict(schema='music-planner-base-fixed3-manager-v1', node='N2', host=p.SCOPE['N2'][0],
            family='music_flamingo', gpus=[0], release_authorized=True,
            out=str(self.root/'music_flamingo_out'), queue=str(self.root/'music_flamingo_queue'),
            protocol={'path': '/fixed/protocol', 'sha256': 'fixed'})
        cfg['protocol'] = save(self.root/'protocol.json', {'fixed3': self.rows, 'decoding': p.DECODE})
        args = SimpleNamespace(config=self.root/'config.json', config_sha='config', producer_sha='producer',
                               audit={'path': '/manager/audit', 'sha256': 'frozen'})
        data = dict(protocol={'fixed3': self.rows}, trained_refs={}, run_config={'path': '/run', 'sha256': 'frozen'},
                    max_length=8192, export=self.root/'export', trained_raw=self.old,
                    processor_refs=[], expected_patch=None)
        adapter = self.adapter('', [9])
        adapters = SimpleNamespace(load_native_processor=Mock(return_value=SimpleNamespace()),
                                   PlannerAdapter=Mock(return_value=SimpleNamespace()))
        with patch.object(p, 'OUTPUT', self.root), patch.object(p, 'check_sources'),              patch.object(p, 'bindings', return_value=data), patch.object(p, 'check_audit'),              patch.object(p, 'pure_base', return_value=adapter), \
             patch.object(p, 'load_module', side_effect=lambda name, path: load(name, path) if path == p.EVALUATOR else adapters),              patch.object(p, 'compare_prefixes', return_value={}),              patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}),              patch.object(torch.cuda, 'is_available', return_value=True),              patch.object(torch.cuda, 'device_count', return_value=1), patch.object(torch.cuda, 'manual_seed_all'):
            p.generate(args, cfg)
        raw = [json.loads(line) for line in (Path(cfg['out'])/'raw.jsonl').read_text().splitlines()]
        self.assertEqual([r['id'] for r in raw], p.IDS)
        self.assertTrue(all(r['raw_text'] == '' and r['failure'] is None for r in raw))
        with patch.object(p, 'verify_ref', return_value=cfg['protocol']):
            save(Path(cfg['queue'])/'base_fixed3.exit.json', {'exit_code': 0})
            result = m.check_outputs(cfg, data, p)
        self.assertTrue(result['technical_collection_complete'])
        self.assertEqual(result['diagnostic_count'], 3)
        self.assertFalse(result['automatic_retry'])
        self.assertFalse((Path(cfg['out'])/'h3_records.json').exists())
        exit_path = Path(cfg['queue'])/'base_fixed3.exit.json'
        save(exit_path, {'exit_code': 7})
        with patch.object(p, 'verify_ref', return_value=cfg['protocol']), self.assertRaisesRegex(RuntimeError, 'exit0'):
            m.check_outputs(cfg, data, p)
        save(exit_path, {'exit_code': 0})
        data['trained_raw'][0]['native_output']['eos_token_ids'] = [10]
        with patch.object(p, 'verify_ref', return_value=cfg['protocol']), self.assertRaisesRegex(RuntimeError, 'effective EOS'):
            m.check_outputs(cfg, data, p)
        data['trained_raw'][0]['native_output']['eos_token_ids'] = [9]
        raw[0]['actual_forward_count'] = 0
        raw_path = Path(cfg['out'])/'raw.jsonl'
        raw_path.write_text(''.join(json.dumps(r)+'\n' for r in raw))
        manifest_path = Path(cfg['out'])/'manifest.json'
        manifest = json.loads(manifest_path.read_text()); manifest['raw'] = p.file_ref(raw_path)
        manifest_path.write_text(json.dumps(manifest))
        with patch.object(p, 'verify_ref', return_value=cfg['protocol']), self.assertRaisesRegex(RuntimeError, 'pre-argmax'):
            m.check_outputs(cfg, data, p)

    def test_snapshot_not_substitute_for_current_full_sha(self):
        model = self.root / 'model'
        model.mkdir()
        weight = model / 'weights'
        weight.write_bytes(b'CPU weights')
        cfg = dict(family='moss', queue=str(self.root), protocol={'path': '/fixed/protocol', 'sha256': 'fixed'},
                   source_sha256={str(m.PRODUCER.with_name('music_planner_base_fixed3_manager.py')): 'manager'})
        value = dict(schema='music-planner-base-fixed3-audit-v1', manager_sha256='manager',
            config_sha256='config', family='moss', model=str(model), protocol=cfg['protocol'],
            base=dict(files=[dict(path=str(weight), stat=p.file_stat(weight))], local_weight_hash_passes_this_audit=1,
                      current_full_content_SHA_and_official_digest_verified=False))
        ref = save(self.root / 'integrity_receipt.json', value)
        with self.assertRaisesRegex(RuntimeError, 'snapshot is not SHA'):
            p.check_audit(ref, cfg, 'config', model)
        value['base']['current_full_content_SHA_and_official_digest_verified'] = True
        ref = save(self.root / 'integrity_receipt.json', value)
        p.check_audit(ref, cfg, 'config', model)
        weight.write_bytes(b'NEW weights')
        with self.assertRaisesRegex(RuntimeError, 'changed after'):
            p.check_audit(ref, cfg, 'config', model)

    def test_exclusive_outputs_and_source_sha_drift(self):
        path = self.root / 'exclusive.json'
        p.new_json(path, {'first': True})
        with self.assertRaises(FileExistsError): p.new_json(path, {'first': False})
        self.assertEqual(json.loads(path.read_text()), {'first': True})
        source = self.root / 'source.py'; source.write_text('pass')
        cfg = dict(source_sha256={str(source): p.file_ref(source)['sha256'], str(Path(p.__file__).resolve()): p.file_ref(p.__file__)['sha256']})
        ref = save(self.root / 'config.json', cfg)
        with patch.object(p, 'PINNED', {str(source): cfg['source_sha256'][str(source)]}):
            p.check_sources(cfg, ref['path'], ref['sha256'], p.file_ref(p.__file__)['sha256'])
            source.write_text('FAIL')
            with self.assertRaisesRegex(RuntimeError, 'SHA drift'):
                p.check_sources(cfg, ref['path'], ref['sha256'], p.file_ref(p.__file__)['sha256'])


class BaseManagerCPU(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        for name, value in [('EXP', self.root / 'exp'), ('RT', self.root / 'exp/runtime'),
                            ('HANDOFF', self.root / 'handoff'), ('OUTPUT', self.root / 'handoff/base_fixed3_compare_v1')]:
            self.stack.enter_context(patch.object(m, name, value))
        m.HANDOFF.mkdir()
        self.cfg = dict(schema='music-planner-base-fixed3-manager-v1', node='N7', host=m.SCOPE['N7'][0],
            family='moss', gpus=[0], release_authorized=True, human_quality_accepted=False,
            out=str(m.OUTPUT / 'moss_out'), queue=str(m.OUTPUT / 'moss_queue'))
        for key, path in dict(training_launch=m.RT/'moss_launch_clean_v11.json',
            full_complete=m.EXP/'queues/moss_fixed2s_clean_v11/full.complete.json',
            resources_safe=m.EXP/'queues/moss_fixed2s_clean_v11/resources_safe.json',
            checkpoint_selection=m.EXP/'runs/moss_fixed2s_clean_v11/train/best.json').items():
            self.cfg[key] = dict(path=str(path), sha256='frozen')
        self.cfg['final_selection'] = dict(path='/fixed/final', sha256='frozen')
        self.cfg['protocol'] = dict(path='/fixed/protocol', sha256='frozen')
        self.cfg['trained_compare'] = dict(path='/fixed/compare', sha256='frozen')

    def predecessor(self, node='N7'):
        host, family = m.SCOPE[node]
        cfg = dict(self.cfg, node=node, host=host, family=family)
        complete, safe, exits, count = m.predecessor_paths(cfg)
        payload = dict(technical_video_complete=True, families=list(p.FAMILIES), completed_ids=p.IDS,
            human_quality_accepted=False) if node == 'N2' else dict(technical_collection_complete=True,
            complete_native_GPU_outputs_verified=True, source_count=500, worker_exit_code=0,
            final_selection=cfg['final_selection'])
        resources = dict(all_holders_safe=True, locks_released=True, leases_restored=True,
                         managed_parent_wait=True, unknown_jobs_not_signaled=True, admitted_gpu_count=count)
        cfg['predecessor'] = dict(complete=save(complete, payload), safe=save(safe, resources),
                                 exits=[save(path, {'exit_code': 0}) for path in exits])
        return cfg, payload, resources

    def test_scope_exact_family_gpu_paths_and_no_overwrite(self):
        m.validate_scope(self.cfg)
        for changes in [dict(node='N2'), dict(host='other'), dict(gpus=[1]), dict(gpus=[False]),
                        dict(gpus=list(range(8))), dict(release_authorized=False),
                        dict(out=str(m.HANDOFF/'elsewhere')), dict(human_quality_accepted=True)]:
            with self.assertRaises(RuntimeError): m.validate_scope(dict(self.cfg, **changes))
        Path(self.cfg['out']).mkdir(parents=True)
        with self.assertRaises(RuntimeError): m.validate_scope(self.cfg)

    def test_all_three_related_predecessors_complete_safe_exit_sha(self):
        for node in m.SCOPE:
            cfg, _, _ = self.predecessor(node)
            m.check_predecessor(cfg, p)
            changed = copy.deepcopy(cfg); changed['predecessor']['exits'] = []
            with self.assertRaises(RuntimeError): m.check_predecessor(changed, p)
            ref = cfg['predecessor']['exits'][0]
            Path(ref['path']).write_text('{"exit_code":1}')
            with self.assertRaisesRegex(RuntimeError, 'changed'): m.check_predecessor(cfg, p)
            cfg['predecessor']['exits'][0] = p.file_ref(ref['path'])
            with self.assertRaisesRegex(RuntimeError, 'exit0'): m.check_predecessor(cfg, p)

    def test_unsafe_or_incomplete_predecessor_rejected(self):
        cfg, payload, resources = self.predecessor()
        for key in ['all_holders_safe', 'locks_released', 'leases_restored', 'managed_parent_wait', 'unknown_jobs_not_signaled']:
            bad = dict(resources, **{key: False})
            cfg['predecessor']['safe'] = save(Path(cfg['predecessor']['safe']['path']), bad)
            with self.assertRaisesRegex(RuntimeError, 'not proven safe'): m.check_predecessor(cfg, p)
        cfg['predecessor']['safe'] = save(Path(cfg['predecessor']['safe']['path']), resources)
        cfg['predecessor']['complete'] = save(Path(cfg['predecessor']['complete']['path']), dict(payload, source_count=499))
        with self.assertRaisesRegex(RuntimeError, 'must finish'): m.check_predecessor(cfg, p)
        with self.assertRaises(RuntimeError): m.check_predecessor(self.cfg, p)

    def test_family_once_exclusive_permanent_and_test_ledger_untouched(self):
        once = m.claim_once(self.cfg, p)
        before = once.read_bytes()
        with self.assertRaises(FileExistsError): m.claim_once(self.cfg, p)
        self.assertEqual(once.read_bytes(), before)
        self.assertEqual(list(m.HANDOFF.glob('test_once*')), [])
        self.assertTrue(json.loads(before)['retained_on_failure'])

    def test_owned_heartbeat_stop_join_before_finish(self):
        native = SimpleNamespace(refresh_leases=Mock())
        watch = m.start_heartbeat([], native)
        self.assertTrue(watch[2].is_alive())
        m.stop_heartbeat(watch)
        self.assertTrue(watch[0].is_set())
        self.assertFalse(watch[2].is_alive())

    def test_capture_checks_predecessor_before_current_holder(self):
        cfg_path = Path(save(self.root/'config.json', self.cfg)['path'])
        native = SimpleNamespace()
        producer = SimpleNamespace(bindings=Mock(), IDS=p.IDS)
        audit = SimpleNamespace(check_completed=Mock(), check_protocol=Mock(), capture_gpu0=Mock())
        with patch.object(m, 'sha', return_value='frozen'), patch.object(m, 'check_sources'),              patch.object(m, 'load_helpers', return_value=(audit, native, None, None, producer)),              patch.object(m, 'check_predecessor', side_effect=RuntimeError('predecessor not finished')):
            with self.assertRaisesRegex(RuntimeError, 'predecessor not finished'):
                m.main(['capture', '--config', str(cfg_path), '--config-sha', 'frozen', '--manager-sha', 'frozen'])
        audit.capture_gpu0.assert_not_called()

    def test_actual_full_sha_binds_prior_base_content_not_prior_stat(self):
        expected = dict(current_full_content_SHA_and_official_digest_verified=True,
            identity=dict(official_stage=dict(revision='official')),
            files=[dict(path='/model/weight', sha256='content', size_bytes=3, stat={'old': True})])
        actual = copy.deepcopy(expected); actual['files'][0]['stat'] = {'current': True}
        m.check_base_expected(actual, expected)
        actual['files'][0]['sha256'] = 'drift'
        with self.assertRaisesRegex(RuntimeError, 'current full base SHA differs'):
            m.check_base_expected(actual, expected)

    def test_pinned_helper_and_producer_sha_required_before_import(self):
        cfg = dict(source_sha256={str(m.PRODUCER): 'bad'})
        with patch.object(m, 'sha', side_effect=[m.AUDIT_SHA, 'other']), patch.object(m, 'load_module') as execute:
            with self.assertRaisesRegex(RuntimeError, 'producer'): m.load_helpers(cfg)
            execute.assert_not_called()
        with patch.object(m, 'sha', return_value='other'), patch.object(m, 'load_module') as execute:
            with self.assertRaisesRegex(RuntimeError, 'helper SHA'): m.load_helpers(cfg)
            execute.assert_not_called()

    def test_manage_precise_release_gate_worker_and_cleanup_order(self):
        native = legacy.manager
        for failure in (None, 'headroom', 'worker', 'nonzero'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                for name, value in [('EXP', root/'exp'), ('RT', root/'exp/runtime'),
                                    ('HANDOFF', root/'handoff'), ('OUTPUT', root/'handoff/base_fixed3_compare_v1')]:
                    stack.enter_context(patch.object(m, name, value))
                m.HANDOFF.mkdir()
                cfg, _, _ = self.predecessor()
                cfg.update(out=str(m.OUTPUT/'moss_out'), queue=str(m.OUTPUT/'moss_queue'),
                    source_sha256={str(m.PRODUCER): 'producer-pin'})
                for key, relative in [('training_launch', 'runtime/moss_launch_clean_v11.json'),
                     ('full_complete', 'queues/moss_fixed2s_clean_v11/full.complete.json'),
                     ('resources_safe', 'queues/moss_fixed2s_clean_v11/resources_safe.json'),
                     ('checkpoint_selection', 'runs/moss_fixed2s_clean_v11/train/best.json')]:
                    cfg[key] = dict(path=str(m.EXP/relative), sha256='fixture')
                cfg['checkpoint_marker'] = dict(path=str(root/'checkpoint/trainer_state.json'), sha256='fixture')
                holder = dict(pid=os.getpid(), uid=os.getuid(), argv=native.RESTORE_ARGV, start_ticks=9,
                    cgroup='CPU-cgroup', CVD=['CUDA_VISIBLE_DEVICES=0'], OCCUPY=['OCCUPY_GPU=0'],
                    exe='/usr/bin/python', sig_ign=0, sig_blk=0, uuid='CPU-UUID', gpu=0)
                cfg['holder'] = holder
                model = root/'models/moss'; model.mkdir(parents=True)
                weight = model/'weight'; weight.write_bytes(b'CPU weight')
                stage_ref = save(root/'stage.json', {'files': {'weight': {}}})
                run_ref = save(root/'run.json', {'max_length': 8192})
                trained_ref = save(root/'trained.json', {})
                identity = dict(run_config=run_ref)
                base = dict(files=[dict(path=str(weight), stat=p.file_stat(weight), sha256=p.file_ref(weight)['sha256'],
                                       size_bytes=weight.stat().st_size)],
                    identity=dict(official_stage=dict(receipt=stage_ref, revision='official-revision')),
                    current_full_content_SHA_and_official_digest_verified=True, local_weight_hash_passes_this_audit=1)
                cfg['expected_base_audit'] = save(root/'previous-base-audit.json', {'base': base})
                export = root/'adapter'; export.mkdir()
                data = dict(protocol={'fixed3': []}, processor_refs=[], export=export, expected_base=base,
                    trained_refs={k: trained_ref for k in ('manifest', 'raw', 'generation_complete', 'generation_resources_safe', 'generation_exit')})
                trace, watches = [], []
                lock = root/'lock'; lock.touch()
                stack.enter_context(patch.object(legacy.m, 'GPU0_LOCK', lock))
                audit = SimpleNamespace(check_completed=Mock(return_value=({'env': {}, 'modelref': {}}, {})),
                    check_protocol=Mock(return_value={}),
                    selected_checkpoint=Mock(return_value=(root/'checkpoint', identity, {'validation_candidates': []})),
                    audit_base=Mock(side_effect=lambda *args: trace.append('current-full-sha') or base),
                    capture_gpu0=Mock(return_value={'holder': holder}), open_gpu0_lock=legacy.m.open_gpu0_lock,
                    check_gpu0_lock=legacy.m.check_gpu0_lock)
                fixed = SimpleNamespace(checkpoint_selection_identity=Mock())
                stack.enter_context(patch.object(p, 'bindings', return_value=data))
                stack.enter_context(patch.object(m, 'check_sources'))
                stack.enter_context(patch.object(m, 'check_outputs', return_value={'technical_collection_complete': True}))
                stack.enter_context(patch.object(native.stage, 'MODEL_BASE', root/'models'))
                stack.enter_context(patch.object(native.stage, 'validate_scope'))
                stack.enter_context(patch.object(native.stage, 'read_download', return_value=(None, 'digest', None, None, {})))
                stack.enter_context(patch.object(native.stage, 'check_staged'))
                stack.enter_context(patch.object(native, 'identity', return_value=holder))
                stack.enter_context(patch.object(native, 'uuid_for', return_value='CPU-UUID'))
                stack.enter_context(patch.object(native, 'apps', return_value=[{'uuid': 'CPU-UUID', 'pid': os.getpid()}]))
                stack.enter_context(patch.object(native, 'headroom', return_value={'headroom': 0 if failure == 'headroom' else 10**15}))
                stack.enter_context(patch.object(native.signal, 'signal'))
                sender = stack.enter_context(patch.object(native.signal, 'pidfd_send_signal', side_effect=lambda *args: trace.append(('pidfd-release', args))))
                stack.enter_context(patch.object(m.select, 'select', side_effect=lambda reads, writes, errors, timeout: (reads if timeout == 30 else [], [], [])))
                real_path = Path
                stack.enter_context(patch.object(m, 'Path', side_effect=lambda value: root/'lease' if str(value) == '/tmp/od_reserve_0' else real_path(value)))
                stack.enter_context(patch.object(m.os, 'dup2'))
                stack.enter_context(patch.object(native, 'wait_gpu_cleanup', side_effect=lambda *args: trace.append('gpu-cleanup')))
                real_start, real_finish = m.start_heartbeat, native.finish_resources
                def start(entries, manager):
                    watch = real_start(entries, manager); watches.append(watch); return watch
                stack.enter_context(patch.object(m, 'start_heartbeat', side_effect=start))
                def child(worker, argv, phase, entries, queue, save_, owned):
                    trace.append('worker')
                    self.assertEqual(worker['gpus'], [0])
                    self.assertEqual(worker['env']['CUDA_VISIBLE_DEVICES'], '0')
                    self.assertEqual(phase, 'base_fixed3')
                    self.assertNotIn('--checkpoint', argv)
                    self.assertNotIn('--limit', argv)
                    owned['started'] = True
                    if failure == 'worker': raise RuntimeError('CPU fake worker failed')
                    return 1 if failure == 'nonzero' else 0
                stack.enter_context(patch.object(native, 'run_child', side_effect=child))
                def finish(entries, locks, queue, save_, holders, owned):
                    trace.append('finish')
                    self.assertTrue(all(not watch[2].is_alive() for watch in watches))
                    # No actual holder was killed: let real native cleanup verify the original fake identity.
                    for entry in entries: entry['released'] = False
                    return real_finish(entries, locks, queue, save_, holders, owned)
                stack.enter_context(patch.object(native, 'finish_resources', side_effect=finish))
                m.manage(cfg, root/'config.json', 'config-pin', 'manager-pin', (audit, native, fixed, None, p))
                self.assertTrue(m.claim_path(cfg).exists())
                self.assertEqual(audit.audit_base.call_count, 1)
                self.assertFalse(audit.selected_checkpoint.call_args.kwargs['full_hash'])
                self.assertEqual(trace[0], 'current-full-sha')
                self.assertEqual(trace[-1], 'finish')
                self.assertFalse((root/'lease').exists())
                safe = json.loads((Path(cfg['queue'])/'resources_safe.json').read_text())
                self.assertEqual(safe['admitted_gpu_count'], 1)
                if failure == 'headroom':
                    sender.assert_not_called()
                    self.assertNotIn('worker', trace)
                else:
                    sender.assert_called_once()
                    self.assertLess(trace.index('gpu-cleanup'), trace.index('worker'))
                self.assertEqual((Path(cfg['queue'])/'failure.json').exists(), failure is not None)
                self.assertEqual((Path(cfg['queue'])/'base_fixed3.complete.json').exists(), failure is None)
                self.assertFalse(list(m.HANDOFF.glob('test_once*')))

    def test_help_no_heavy_imports(self):
        for name in ('music_planner_base_fixed3.py', 'music_planner_base_fixed3_manager.py'):
            code = "import runpy,sys; sys.argv=['entry','--help']; " +                    "runpy.run_path(" + repr(str(SCRIPTS/name)) + ",run_name='__main__')"
            result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)
            self.assertNotIn('torch', result.stderr)


class NativeResourceRegressions(unittest.TestCase):
    def run_fixture(self, name):
        case = legacy.CPUHandoff(name)
        try:
            case.setUp()
            getattr(case, name)()
        finally:
            case.doCleanups()

    def test_real_pidfd_exact_identity_and_no_unknown_signals(self):
        self.run_fixture('test_holder_exact_identity_and_ignored_blocked_SIGTERM')
        self.run_fixture('test_restore_wait_and_foreign_GPU_job_are_not_signaled_or_respawned')

    def test_real_flock_lease_cleanup_restore_gpu0_only(self):
        self.run_fixture('test_existing_lock_nonblocking_busy_and_nofollow')
        self.run_fixture('test_real_lease_helpers_old_snapshot_atomic_fd_heartbeat_and_restore')
        self.run_fixture('test_lease_replacement_and_unsafe_content_rejected')
        self.run_fixture('test_real_finish_resources_restores_lease_and_unlocks_only_GPU0')

    def test_real_current_full_sha_not_stat_only(self):
        self.run_fixture('test_current_base_audit_real_SHA_LFS_gitblob_and_same_stat_rewrite')


if __name__ == '__main__':
    unittest.main()
