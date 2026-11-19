"""CPU-only typed-source regression plus the frozen v1 resource contracts."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


v2 = module("salvage_current_v2_contract", RT / "prepare_salvage_current_resource_restore_v2.py")
m = v2.restore
frozen = module("salvage_current_v1_frozen_tests", v2.FROZEN_TEST)
frozen.m = m
ProofOrderTests = frozen.ProofOrderTests
CurrentResourceTests = frozen.CurrentResourceTests


class NativeReleaseTests(frozen.NativeReleaseTests):
    def test_native_signals_and_false_completion_logic_remain_original(self):
        self.assertEqual(m.SELF, RT / "prepare_salvage_current_resource_restore_v2.py")
        self.assertEqual(m.TEST, Path(__file__).resolve())
        with patch.object(m, "SELF", v2.FROZEN_ENTRY):
            super().test_native_signals_and_false_completion_logic_remain_original()


class TypedReferenceTests(unittest.TestCase):
    def test_true_nonJSON_python_source_full_SHA_and_no_JSON_parse(self):
        path = RT / "coarse3_zero_dispatch_resume_resource_v1.py"
        reference = {"path": str(path), "sha256": "7e0d7af1a380cb7fbabf073f834e731764aba2da225582f3005259409840aad8"}
        with patch.object(json, "loads", side_effect=AssertionError("source must not be JSON parsed")), \
                patch.object(m, "content_sha", side_effect=AssertionError("source bytes must get full SHA, not cache")):
            raw = v2.checked_source(reference)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), reference["sha256"])
        with self.assertRaises(json.JSONDecodeError):
            m.checked(reference)

    def test_source_content_SHA_absolute_path_and_each_stable_stat_gate(self):
        with tempfile.TemporaryDirectory(prefix="salvage_v2_bytes_") as tmp:
            path = Path(tmp) / "source.py"
            raw = b'"""Real Python, not JSON."""\nvalue = 1\n'
            path.write_bytes(raw)
            reference = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
            self.assertEqual(v2.checked_source(reference), raw)
            with self.assertRaisesRegex(RuntimeError, "Immutable reference"):
                v2.checked_source(dict(reference, path="relative.py"))
            with self.assertRaisesRegex(RuntimeError, "Immutable reference"):
                v2.checked_source(dict(reference, sha256="0" * 64))
            path.write_bytes(raw.replace(b"1", b"2"))
            with self.assertRaisesRegex(RuntimeError, "Immutable reference"):
                v2.checked_source(reference)
            path.write_bytes(raw)
            stable = path.stat()
            for position in range(4):
                for key in v2.STAT_KEYS:
                    with self.subTest(position=position, key=key):
                        samples = [stable] * 4
                        changed = {name: getattr(stable, name) for name in v2.STAT_KEYS}
                        changed[key] += 1
                        samples[position] = NS(**changed)
                        with patch.object(Path, "stat", side_effect=[samples[0], samples[3]]), \
                                patch.object(v2.os, "fstat", side_effect=[samples[1], samples[2]]):
                            with self.assertRaisesRegex(RuntimeError, "changed during full SHA"):
                                v2.checked_source(reference)

    def test_JSON_valid_and_malformed_refs_keep_original_parser(self):
        with tempfile.TemporaryDirectory(prefix="salvage_v2_json_") as tmp:
            path = Path(tmp) / "plan.py"
            for raw, expected in [(b'{"scope": 17}', {"scope": 17}), (b'{"scope":', None), (b'{} {}', None)]:
                path.write_bytes(raw)
                reference = m.ref(path)
                if expected is None:
                    with self.assertRaises(json.JSONDecodeError):
                        m.checked(reference)
                else:
                    self.assertEqual(m.checked(reference), expected)
                    with self.assertRaisesRegex(RuntimeError, "Immutable reference"):
                        m.checked(dict(reference, sha256="0" * 64))
            with self.assertRaisesRegex(RuntimeError, "Immutable reference"):
                m.checked({"path": "relative.json", "sha256": "0" * 64})

    def test_only_explicit_resource_call_changes_and_frozen_bytes_stay_bound(self):
        tree = ast.parse(v2.checked_source({"path": str(v2.FROZEN_ENTRY), "sha256": v2.FROZEN_ENTRY_SHA}))
        original_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "safe_chain")
        expected_function = copy.deepcopy(original_function)
        expected_call = ast.dump(ast.parse("checked(cfg['resource_manager'])", mode="eval").body)
        calls = [node for node in ast.walk(expected_function) if isinstance(node, ast.Call) and ast.dump(node) == expected_call]
        self.assertEqual(len(calls), 1)
        calls[0].func.id = "checked_source"
        namespace = {}
        exec(compile(ast.Module(body=[expected_function], type_ignores=[]), str(v2.FROZEN_ENTRY), "exec"), namespace)
        expected = namespace["safe_chain"].__code__
        actual = m.safe_chain.__code__
        for field in ("co_code", "co_consts", "co_names", "co_varnames", "co_freevars", "co_cellvars", "co_exceptiontable"):
            self.assertEqual(getattr(actual, field), getattr(expected, field))
        namespace = {}
        exec(compile(tree, str(v2.FROZEN_ENTRY), "exec"), dict(m.__dict__), namespace)
        for name in ("checked", "full_proof", "evidence", "fresh_current", "build", "admission", "before_release", "main"):
            self.assertEqual(getattr(m, name).__code__.co_code, namespace[name].__code__.co_code)
            self.assertEqual(getattr(m, name).__code__.co_consts, namespace[name].__code__.co_consts)
        self.assertEqual(m.PINNED[v2.FROZEN_ENTRY.name], v2.FROZEN_ENTRY_SHA)
        self.assertEqual(m.PINNED[v2.FROZEN_TEST.name], v2.FROZEN_TEST_SHA)
        self.assertEqual(m.launch_command(v2.FROZEN_ENTRY)[3], str(v2.SELF))
        self.assertEqual(m.launch_command(v2.FROZEN_ENTRY)[-1], m.content_sha(v2.TEST))
        with patch.object(v2, "FROZEN_ENTRY_SHA", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "Immutable reference"):
                v2.private_restore()
        with patch.object(v2, "FROZEN_TEST_SHA", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "Immutable reference"):
                v2.private_restore()

    def test_safe_chain_true_python_resource_preserves_receipt_plan_source_gates(self):
        native = frozen.native
        with tempfile.TemporaryDirectory(prefix="salvage_v2_chain_") as tmp:
            root = Path(tmp)
            rt = root / "runtime"
            rt.mkdir()
            queue = root / "queues" / "coarse3_n6_independent_zero_dispatch_resume_1792931193890113953"
            queue.mkdir(parents=True)
            plan_path = rt / "plan.json"
            plan_path.write_bytes(b'{"frozen_plan": 1}')
            source = RT / "coarse3_zero_dispatch_resume_resource_v1.py"
            resource = {"path": str(source), "sha256": "7e0d7af1a380cb7fbabf073f834e731764aba2da225582f3005259409840aad8"}
            config_path = rt / "safe.json"
            cfg = {"queue": str(queue), "node": "N6", "gpus": [3], "release_authorized": True,
                   "resource_manager": resource, "plan": m.ref(plan_path), "phase": "frozen_phase", "targetcount": 3,
                   "producer": resource, "source_sha256": {str(source): resource["sha256"]}, "out": "/fixture/out",
                   "modelref": {"model_identity_sha256": "fixture_model_sha"}, "argv": ["fixture_worker_argv"]}
            def write(path, value):
                path.write_text(json.dumps(value))
            write(config_path, cfg)
            config_ref = m.ref(config_path)
            identity = {key: "fixture" for key in native.IDENTITY_KEYS}
            identity.update(uid=os.getuid(), argv=['/opt/venv_eval/bin/python', '-B', '-u', str(source),
                            '--config', str(config_path), '--config-sha', config_ref['sha256'], '--manager-sha', m.BASE_SHA])
            holder = {"fixture_holder": 1}
            restored = {"gpu": 3, "uuid": "fixture_uuid", "original_identity": holder,
                        "restore_argv": native.RESTORE_ARGV, "holder_identity": {"argv": native.RESTORE_ARGV},
                        "managed_parent_wait": True}
            admission = {key: cfg[key] for key in ('node', 'gpus', 'phase', 'targetcount', 'producer', 'plan', 'source_sha256', 'out')}
            admission.update(config_sha256=config_ref['sha256'], manager_sha256=m.BASE_SHA, manager_identity=identity,
                             model_identity_sha256='fixture_model_sha', holders=[{'uuid': 'fixture_uuid', 'identity': holder}])
            live = {"manager_identity": identity, "argv": cfg['argv']}
            safe = dict(phase=cfg['phase'], all_holders_safe=True, locks_released=True, leases_restored=True,
                        managed_parent_wait=True, unknown_jobs_not_signaled=True)
            for offset, (name, value) in enumerate([('admission.json', admission), ('live.json', live),
                                                  ('holder_restored_gpu3.json', restored), ('resources_safe.json', safe)]):
                path = queue / name
                write(path, value)
                os.utime(path, ns=(100 + offset, 100 + offset))
            prepare = NS(base=native)
            with patch.object(m, "RT", rt), patch.object(m, "SAFE_PATH", config_path), patch.object(m, "SAFE_SHA", config_ref['sha256']):
                result = m.safe_chain(prepare, config_ref)
                self.assertEqual(result['restored'], restored)
                self.assertEqual(set(result['receipt_refs']), {'admission.json', 'live.json', 'holder_restored_gpu3.json', 'resources_safe.json'})
                with patch.object(m, "checked_source", side_effect=RuntimeError('source_SHA_gate')):
                    with self.assertRaisesRegex(RuntimeError, 'source_SHA_gate'):
                        m.safe_chain(prepare, config_ref)
                with patch.object(m, "verify_bindings", side_effect=RuntimeError('full_source_gate')):
                    with self.assertRaisesRegex(RuntimeError, 'full_source_gate'):
                        m.safe_chain(prepare, config_ref)
                plan_path.write_bytes(b'{malformed')
                with self.assertRaisesRegex(RuntimeError, 'Immutable reference'):
                    m.safe_chain(prepare, config_ref)
                # Bind the malformed fixture honestly; only JSON parsing should fail.
                cfg['plan'] = m.ref(plan_path)
                write(config_path, cfg)
                malformed_ref = m.ref(config_path)
                admission['plan'] = cfg['plan']
                admission['config_sha256'] = malformed_ref['sha256']
                identity['argv'][identity['argv'].index('--config-sha') + 1] = malformed_ref['sha256']
                write(queue / 'admission.json', admission)
                write(queue / 'live.json', live)
                os.utime(queue / 'admission.json', ns=(100, 100))
                with patch.object(m, "SAFE_SHA", malformed_ref['sha256']), \
                        patch.object(m, "checked_source", side_effect=AssertionError('plan must fail first')):
                    with self.assertRaises(json.JSONDecodeError):
                        m.safe_chain(prepare, malformed_ref)
                plan_path.write_bytes(b'{"frozen_plan": 1}')
                cfg['plan'] = m.ref(plan_path)
                write(config_path, cfg)
                admission['plan'] = cfg['plan']
                admission['config_sha256'] = config_ref['sha256']
                identity['argv'][identity['argv'].index('--config-sha') + 1] = config_ref['sha256']
                write(queue / 'admission.json', admission)
                write(queue / 'live.json', live)
                os.utime(queue / 'admission.json', ns=(100, 100))
                safe['unknown_jobs_not_signaled'] = False
                write(queue / 'resources_safe.json', safe)
                with self.assertRaisesRegex(RuntimeError, 'not actually safe'):
                    m.safe_chain(prepare, config_ref)


if __name__ == "__main__":
    unittest.main(verbosity=2)
