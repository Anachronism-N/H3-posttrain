"""CPU-only real Qwen3 thinker EOS tests; bridge fixtures test provenance, not quality."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
PATCH_PATH = REPO / "scripts_omnidance/exp/ms/music_planner_generation_eos_patch.py"
MODEL_PATH = Path(os.environ.get("CAPTIONER_MODEL_PATH", "/opt/music_planner_models/omni_captioner"))
FROZEN = Path(os.environ.get("CAPTIONER_FROZEN_SOURCE_DIR", str(
    REPO.parent / "diff/logs_omnidance/exp/MS/music_planner_fixed2s_training_20261024_v1/runtime/source_v11")))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CaptionerNativeCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODEL_PATH / "config.json").is_file():
            raise unittest.SkipTest("Official local Captioner config/processor required; no fake semantic fallback")
        import numpy as np
        import torch
        from transformers import AutoConfig, GenerationConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration
        cls.torch, cls.GenerationConfig = torch, GenerationConfig
        cls.native_class = Qwen3OmniMoeThinkerForConditionalGeneration
        torch.set_num_threads(1)
        cls.module = load("_test_captioner_eos_patch", PATCH_PATH)
        cls.adapters = load("_test_frozen_captioner_adapters", FROZEN / "music_planner_adapters.py")
        cls.processor = cls.adapters.load_native_processor("omni_captioner", MODEL_PATH)
        cls.official_config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
        cls.official_generation = GenerationConfig.from_pretrained(MODEL_PATH, local_files_only=True)
        config = copy.deepcopy(cls.official_config.thinker_config)
        text = config.text_config
        text.hidden_size, text.intermediate_size, text.moe_intermediate_size = 32, 64, 32
        text.num_hidden_layers, text.num_attention_heads, text.num_key_value_heads = 1, 4, 2
        text.head_dim, text.num_experts, text.num_experts_per_tok = 8, 2, 1
        text.rope_parameters["mrope_section"] = [1, 1, 2]
        audio = config.audio_config
        audio.d_model, audio.output_dim, audio.encoder_ffn_dim = 32, 32, 64
        audio.encoder_layers, audio.encoder_attention_heads, audio.downsample_hidden_size = 1, 4, 4
        vision = config.vision_config
        vision.depth, vision.hidden_size, vision.intermediate_size = 1, 32, 64
        vision.num_heads, vision.out_hidden_size, vision.deepstack_visual_indexes = 4, 32, []
        cls.config = config
        waveform = np.sin(np.arange(81000, dtype=np.float32) * .01)
        prefix_adapter = cls.adapters.PlannerAdapter("omni_captioner", None, cls.processor)
        cls.prompt = prefix_adapter._prompt()
        cls.batch = dict(cls.processor(text=[cls.prompt], audio=[waveform], sampling_rate=16000,
                                       return_tensors="pt", padding=False))
        cls.prefix_length = cls.batch["input_ids"].shape[1]
        cls.text_ids = cls.processor.tokenizer.encode("The dancer steps sideways.", add_special_tokens=False)
        cls.eos = cls.processor.tokenizer.eos_token_id
        cls.loop = [151644, cls.eos, 198]

    def adapter(self, peft=False):
        self.torch.manual_seed(42)
        model = self.native_class(copy.deepcopy(self.config)).cpu().eval()
        self.assertIsNone(model.generation_config.eos_token_id)
        model.generation_config = copy.deepcopy(self.official_generation)
        if peft:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"],
                                                    task_type="CAUSAL_LM"))
        return self.adapters.PlannerAdapter("omni_captioner", model, self.processor)

    def controlled_call(self, adapter, schedule, budget, native_calls):
        from transformers import LogitsProcessor, LogitsProcessorList
        prefix_length = self.prefix_length
        class ControlledLogits(LogitsProcessor):
            def __call__(self, input_ids, scores):
                index = input_ids.shape[1] - prefix_length
                token = schedule[index] if index < len(schedule) else schedule[-1]
                scores.fill_(-float("inf"))
                scores[:, token] = 0
                return scores
        native_model = adapter.model.get_base_model() if hasattr(adapter.model, "get_base_model") else adapter.model
        attribute = "_planner_generation_eos_native_generate" if "_planner_generation_eos_native_generate" in vars(native_model) else "generate"
        real_generate = getattr(native_model, attribute)
        def controlled(*args, **kwargs):
            native_calls.append(dict(kwargs))
            result = real_generate(*args, **kwargs, logits_processor=LogitsProcessorList([ControlledLogits()]),
                                   return_dict_in_generate=True, output_logits=True)
            native_calls[-1]["output"] = result
            return result
        with patch.object(native_model, attribute, side_effect=controlled),                 patch.object(adapter, "prefix", return_value=self.batch):
            adapter._last_audio_info = dict(processor_sample_rate=16000, n_samples=81000)
            return adapter.generate({}, max_new_tokens=budget)

    def test_official_config_native_api_and_training_eos(self):
        from transformers.generation import GenerationMixin
        self.assertIsNone(self.official_generation.eos_token_id)
        self.assertEqual(self.official_generation.max_new_tokens, 32768)
        self.assertIsNone(self.GenerationConfig.from_model_config(self.official_config.thinker_config).eos_token_id)
        adapter = self.adapter()
        self.assertIsNone(adapter.model.generation_config.eos_token_id)
        self.assertIs(self.native_class.generate, GenerationMixin.generate)
        self.assertEqual(adapter.eos_token_ids(), [151645])
        self.assertEqual(adapter._assistant_ids("A step.")[0, -1].item(), 151645)
        self.assertNotIn(self.adapters.TASK_INSTRUCTION, self.prompt)
        self.assertEqual(self.processor.tokenizer.encode("<|im_end|>", add_special_tokens=False), [151645])

    def test_real_native_first_eos_stop_and_pre_eos_logits_identical(self):
        adapter = self.adapter()
        model = adapter.model
        params = {name: value.detach().clone() for name, value in model.named_parameters()}
        config_before, generation_before = model.config.to_dict(), model.generation_config.to_dict()
        forward_before, prepare_before = model.forward, model.prepare_inputs_for_generation
        batch_before = {key: value.clone() for key, value in self.batch.items()}
        schedule = self.text_ids + [self.eos] + self.loop * 6 + [self.eos]
        old_calls = []
        old = self.controlled_call(adapter, schedule, len(schedule), old_calls)
        self.assertNotIn("eos_token_id", old_calls[0])
        self.assertEqual(old["new_token_ids"], schedule)
        # Frozen code mislabels a capped sequence ending in im_end as natural EOS.
        self.assertEqual(old["stop_reason"], "eos")
        self.assertEqual(len(old["new_token_ids"]), len(schedule))
        self.assertEqual(old["new_token_ids"].index(self.eos), len(self.text_ids))
        self.assertGreater(old["new_token_ids"].count(self.eos), 1)
        ref = self.module.apply_generation_eos_patch(adapter)
        new_calls = []
        new = self.controlled_call(adapter, schedule, len(schedule), new_calls)
        stopped = self.text_ids + [self.eos]
        self.assertEqual(new["new_token_ids"], stopped)
        self.assertEqual(new["stop_reason"], "eos")
        self.assertEqual(new["applied_generation_eos_patch_ref"], ref)
        self.assertEqual(old["raw_text"], self.processor.tokenizer.decode(
            schedule, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        self.assertEqual(new["raw_text"], self.processor.tokenizer.decode(
            stopped, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        self.assertEqual(old["raw_text"], new["raw_text"] + "\n" * 6)
        self.assertEqual(old["decoding"], new["decoding"])
        self.assertEqual(new["input_tokens"], old["input_tokens"])
        self.assertEqual(new["n_samples"], 81000)
        self.assertEqual(new_calls[0]["eos_token_id"], [self.eos])
        for key in old_calls[0]:
            if key == "output":
                continue
            if isinstance(old_calls[0][key], self.torch.Tensor):
                self.assertTrue(self.torch.equal(old_calls[0][key], new_calls[0][key]), key)
            else:
                self.assertEqual(old_calls[0][key], new_calls[0][key], key)
        for before, after in zip(old_calls[0]["output"].logits, new_calls[0]["output"].logits):
            self.assertTrue(self.torch.equal(before, after))
        self.assertTrue(self.torch.equal(old_calls[0]["output"].sequences[:, :self.prefix_length],
                                        new_calls[0]["output"].sequences[:, :self.prefix_length]))
        self.assertEqual(model.config.to_dict(), config_before)
        self.assertEqual(model.generation_config.to_dict(), generation_before)
        self.assertEqual(model.forward, forward_before)
        self.assertEqual(model.prepare_inputs_for_generation, prepare_before)
        for name, value in model.named_parameters():
            self.assertTrue(self.torch.equal(value, params[name]), name)
        for key, value in self.batch.items():
            self.assertTrue(self.torch.equal(value, batch_before[key]), key)
        self.assertEqual(self.module.apply_generation_eos_patch(adapter), ref)

    def test_first_generated_token_eos_and_real_peft_base_wrapper(self):
        adapter = self.adapter(peft=True)
        base = adapter.model.get_base_model()
        self.assertIsNone(base.generation_config.eos_token_id)
        ref = self.module.apply_generation_eos_patch(adapter)
        calls = []
        result = self.controlled_call(adapter, [self.eos] + self.loop * 3, 10, calls)
        self.assertEqual(result["new_token_ids"], [self.eos])
        self.assertEqual(result["stop_reason"], "eos")
        self.assertEqual(result["applied_generation_eos_patch_ref"], ref)
        self.assertIsNone(base.generation_config.eos_token_id)
        self.assertEqual(base._planner_generation_eos_patch_ref, ref)

    def test_native_valid_eos_collection_is_not_modified(self):
        for family in self.module.FAMILIES:
            with self.subTest(family=family):
                adapter = self.adapter()
                adapter.family = family
                adapter.model.generation_config.eos_token_id = [self.eos, 151643]
                before = dict(vars(adapter.model))
                generate_before = adapter.generate
                self.assertIsNone(self.module.apply_generation_eos_patch(adapter))
                self.assertEqual(adapter.model.generation_config.eos_token_id, [self.eos, 151643])
                self.assertEqual(dict(vars(adapter.model)), before)
                self.assertEqual(adapter.generate, generate_before)
                self.assertNotIn("applied_generation_eos_patch_ref", vars(adapter))


class EOSBridgeBoundaryCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load("_test_eos_bridge_boundary", PATCH_PATH)

    def fixture_adapter(self, eos=None, fallback=(151645,)):
        # These fixtures test wrapper arguments/provenance only, not model behavior.
        native_generate = Mock(return_value="native-opaque-output")
        model = types.SimpleNamespace(generation_config=types.SimpleNamespace(eos_token_id=eos),
                                      generate=native_generate)
        adapter = types.SimpleNamespace(family="omni_captioner", model=model,
            tokenizer=types.SimpleNamespace(encode=lambda *a, **k: [151645]),
            eos_token_ids=lambda: list(fallback), generate=lambda **k: dict(raw_text="unchanged"))
        return adapter, native_generate

    def test_invalid_native_eos_only_and_valid_call_eos_preserved(self):
        for invalid in (None, [], -1, True, [None]):
            adapter, native = self.fixture_adapter(eos=invalid)
            self.assertIsNotNone(self.module.apply_generation_eos_patch(adapter))
            self.assertEqual(adapter.model.generate(input_ids="prefix", use_cache=True), "native-opaque-output")
            self.assertEqual(native.call_args.kwargs, dict(input_ids="prefix", use_cache=True, eos_token_id=[151645]))
            adapter.model.generate(input_ids="prefix", eos_token_id=[151645, 151643])
            self.assertEqual(native.call_args.kwargs["eos_token_id"], [151645, 151643])
            self.assertEqual(adapter.model.generation_config.eos_token_id, invalid)
        for valid in (151645, [151645, 151643], (151645,)):
            adapter, native = self.fixture_adapter(eos=valid)
            self.assertIsNone(self.module.apply_generation_eos_patch(adapter))
            self.assertIs(adapter.model.generate, native)

    def test_wrong_training_eos_extra_specials_and_patch_ref_rejected(self):
        for fallback in ([151644], [151645, 151644], []):
            adapter, native = self.fixture_adapter(fallback=fallback)
            with self.assertRaisesRegex(ValueError, "trained assistant"):
                self.module.apply_generation_eos_patch(adapter)
            self.assertIs(adapter.model.generate, native)
        adapter, native = self.fixture_adapter()
        ref = self.module.apply_generation_eos_patch(adapter)
        self.assertEqual(self.module.apply_generation_eos_patch(adapter), ref)
        other, _ = self.fixture_adapter()
        other.model = adapter.model
        with self.assertRaisesRegex(ValueError, "different EOS patch identity"):
            self.module.apply_generation_eos_patch(other)
        adapter.model._planner_generation_eos_patch_ref = dict(ref, sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "different EOS patch identity"):
            self.module.apply_generation_eos_patch(adapter)
        adapter.model._planner_generation_eos_patch_ref = ref
        adapter.applied_generation_eos_patch_ref = dict(ref, sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "different EOS patch identity"):
            self.module.apply_generation_eos_patch(adapter)
        self.assertEqual(ref, self.module._file_ref(PATCH_PATH))

    def bridge_fixture(self, directory, entry="eval", trained_sha=None, eos=None, fail=False):
        root = Path(directory)
        refs = {}
        for key, name in (("adapter", "music_planner_adapters.py"), ("eval", "music_planner_eval.py"),
                          ("fixed3", "music_planner_fixed3_eval.py")):
            path = root / name
            path.write_text("fixture = " + repr(key))
            refs[key] = self.module._file_ref(path)
        refs["patch"] = self.module._file_ref(PATCH_PATH)
        out = root / "fresh"
        adapter, native = self.fixture_adapter(eos=eos)
        loader = Mock(return_value=adapter)
        adapter_module = types.SimpleNamespace(load_planner_adapter=loader)
        identity = dict(training_config=dict(adapter_source_sha256=trained_sha or refs["adapter"]["sha256"]),
                        fingerprint="original-fingerprint", trainer_state=dict(sha256="original-marker"))
        initial = dict(checkpoint_identity=identity, code=refs[entry], adapter_code=refs["adapter"], complete=False)
        def run():
            out.mkdir()
            (out / "manifest.json").write_text(json.dumps(initial))
            result = adapter_module.load_planner_adapter()
            if fail:
                raise RuntimeError("controlled bridge failure")
            initial["complete"] = True
            (out / "manifest.json").write_text(json.dumps(initial))
            (out / "raw.jsonl").write_text(json.dumps(dict(native_output=result.generate())) + "\n")
        evaluator = types.SimpleNamespace(main=run)
        fixed3 = types.SimpleNamespace(main=run, FROZEN=root, frozen_eval=lambda: None)
        modules = {"music_planner_adapters": adapter_module,
                   "_eos_generation_frozen_eval": evaluator, "_eos_generation_frozen_fixed3": fixed3}
        def fake_load(name, path):
            sys.modules[name] = modules[name]
            return modules[name]
        args = ["--frozen-source", str(root), "--entry", entry]
        for key in ("adapter", "eval", "patch") + (("fixed3",) if entry == "fixed3" else ()):
            args += ["--expected-" + key + "-sha256", refs[key]["sha256"]]
        args += ["--"] + (["generate"] if entry == "fixed3" else []) + ["--family", "omni_captioner", "--out", str(out)]
        return args, out, fake_load, adapter_module, loader, initial, refs

    def test_source_pins_fail_before_any_model_load(self):
        for key in ("adapter", "eval", "patch", "fixed3"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                args, out, fake_load, mod, loader, initial, refs = self.bridge_fixture(temporary, entry="fixed3")
                args[args.index("--expected-" + key + "-sha256") + 1] = "0" * 64
                with patch.object(self.module, "_load_module", side_effect=fake_load) as imports:
                    with self.assertRaises(SystemExit):
                        self.module.main(args)
                imports.assert_not_called()
                loader.assert_not_called()
                self.assertFalse(out.exists())

    def test_eval_fixed3_receipts_preserve_original_identity_and_restore(self):
        for entry in ("eval", "fixed3"):
            for eos in (None, [151645, 151643]):
                with self.subTest(entry=entry, eos=eos), tempfile.TemporaryDirectory() as temporary:
                    args, out, fake_load, mod, loader, initial, refs = self.bridge_fixture(temporary, entry=entry, eos=eos)
                    original_argv, original_path = sys.argv, sys.path[:]
                    with patch.object(self.module, "_load_module", side_effect=fake_load):
                        self.module.main(args)
                    self.assertIs(sys.argv, original_argv)
                    self.assertEqual(sys.path, original_path)
                    self.assertIs(mod.load_planner_adapter, loader)
                    for name in ("music_planner_adapters", "_eos_generation_frozen_eval", "_eos_generation_frozen_fixed3"):
                        self.assertNotIn(name, sys.modules)
                    manifest = json.loads((out / "manifest.json").read_text())
                    receipt = json.loads((out / "generation_eos_patch.json").read_text())
                    for key in ("checkpoint_identity", "code", "adapter_code"):
                        self.assertEqual(manifest[key], initial[key])
                    expected = refs["patch"] if eos is None else None
                    self.assertEqual(manifest["applied_generation_eos_patch_ref"], expected)
                    self.assertEqual(receipt["applied_generation_eos_patch_ref"], expected)
                    self.assertEqual(receipt["applied"], eos is None)
                    self.assertEqual(manifest["generation_eos_patch_receipt_ref"], self.module._file_ref(out / "generation_eos_patch.json"))
                    raw = json.loads((out / "raw.jsonl").read_text())["native_output"]
                    self.assertEqual(raw["raw_text"], "unchanged")
                    if eos is None:
                        self.assertEqual(raw["applied_generation_eos_patch_ref"], expected)
                    else:
                        self.assertNotIn("applied_generation_eos_patch_ref", raw)

    def test_training_source_mismatch_and_exception_restore_fail_closed(self):
        for mismatch, fail in ((True, False), (False, True)):
            with tempfile.TemporaryDirectory() as temporary:
                args, out, fake_load, mod, loader, initial, refs = self.bridge_fixture(
                    temporary, trained_sha="0" * 64 if mismatch else None, fail=fail)
                original_argv, original_path = sys.argv, sys.path[:]
                with patch.object(self.module, "_load_module", side_effect=fake_load):
                    with self.assertRaises(ValueError if mismatch else RuntimeError):
                        self.module.main(args)
                self.assertIs(sys.argv, original_argv)
                self.assertEqual(sys.path, original_path)
                self.assertIs(mod.load_planner_adapter, loader)
                manifest = json.loads((out / "manifest.json").read_text())
                self.assertFalse(manifest["complete"])
                if mismatch:
                    loader.assert_not_called()
                    self.assertIsNone(manifest["applied_generation_eos_patch_ref"])
                    self.assertFalse((out / "generation_eos_patch.json").exists())


if __name__ == "__main__":
    unittest.main()
