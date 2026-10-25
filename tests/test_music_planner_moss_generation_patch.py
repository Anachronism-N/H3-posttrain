"""CPU integration tests using the shipped MOSS classes and real HF generation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import types
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
PATCH_PATH = REPO / "scripts_omnidance/exp/ms/music_planner_moss_generation_patch.py"
MODEL_PATH = Path(os.environ.get("MOSS_MODEL_PATH", "/opt/music_planner_models/moss"))
FROZEN = Path(os.environ.get(
    "MOSS_FROZEN_SOURCE_DIR", str(REPO.parent / "diff/logs_omnidance/exp/MS/"
                                "music_planner_fixed2s_training_20261024_v1/runtime/source_v11")))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MossNativeCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODEL_PATH / "modeling_moss_music.py").is_file():
            raise unittest.SkipTest("Official local MOSS source/processor required; no fake generation fallback")
        import numpy as np
        import torch
        from transformers import Qwen3Config, WhisperConfig
        cls.torch = torch
        torch.set_num_threads(1)
        cls.patch_module = load("_test_moss_generation_patch", PATCH_PATH)
        cls.adapters = load("_test_frozen_moss_adapters", FROZEN / "music_planner_adapters.py")
        cls.processor = cls.adapters.load_native_processor("moss", MODEL_PATH)
        cls.native_class = cls.adapters._local_moss_class(
            MODEL_PATH, "modeling_moss_music", "MossMusicModel")
        cls.audio_config = WhisperConfig(
            d_model=16, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=32,
            downsample_hidden_size=2, output_dim=16, deepstack_encoder_layer_indexes=[0, 1],
            layer_norm_eps=1e-5, num_mel_bins=128, dropout=0., attention_dropout=0.,
            activation_dropout=0.)
        cls.text_config = Qwen3Config(
            vocab_size=len(cls.processor.tokenizer), hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            bos_token_id=cls.processor.tokenizer.bos_token_id,
            eos_token_id=cls.processor.tokenizer.eos_token_id)
        waveform = np.sin(np.arange(81000, dtype=np.float32) * .01)
        cls.batch = dict(cls.processor(
            text=cls.processor._build_default_prompt(cls.adapters.TASK_INSTRUCTION, True),
            audios=[waveform], return_tensors="pt"))
        cls.batch["audio_input_mask"] = cls.batch["input_ids"].eq(cls.processor.audio_token_id)
        cls.batch["audio_data"] = cls.batch["audio_data"].float()
        cls.prefix_length = cls.batch["input_ids"].shape[1]
        assert cls.prefix_length == 141 and cls.batch["audio_input_mask"].sum().item() == 64

    def adapter(self):
        self.torch.manual_seed(42)
        config = self.native_class.config_class(
            audio_config=self.audio_config, language_config=self.text_config,
            adapter_hidden_size=32, deepstack_num_inject_layers=2)
        model = self.native_class(config).cpu().eval()
        return self.adapters.PlannerAdapter("moss", model, self.processor)

    def generate(self, model, steps=3):
        with self.torch.no_grad():
            return model.generate(**self.batch, max_new_tokens=steps, min_new_tokens=steps,
                                  use_cache=True, do_sample=False, return_dict_in_generate=True,
                                  output_scores=True, output_logits=True)

    def test_unpatched_official_generation_reproduces_actual_stack(self):
        adapter = self.adapter()
        with self.assertRaises(RuntimeError) as caught:
            self.generate(adapter.model)
        self.assertIn("expanded size of the tensor (142)", str(caught.exception))
        self.assertIn("existing size (141)", str(caught.exception))
        # assertRaises clears exception traceback, so capture the real generation stack separately.
        try:
            self.generate(adapter.model, steps=2)
        except RuntimeError:
            stack = traceback.format_exc()
        self.assertIn("modeling_moss_music.py", stack)
        self.assertIn("mask_expanded = audio_input_mask.unsqueeze(-1).expand_as(inputs_embeds)", stack)
        self.assertIn("music_planner_adapters.py", stack)
        self.assertIn("generation/utils.py", stack)

    def test_prefill_logits_and_trained_state_are_exactly_unchanged(self):
        from transformers import DynamicCache
        adapter = self.adapter()
        model = adapter.model
        parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        config_before = model.config.to_dict()
        forward_before = model.forward
        checkpoint_before = model.gradient_checkpointing_enable
        hooks_before = model._register_llm_deepstack_hooks
        native = model.prepare_inputs_for_generation
        cache_old = DynamicCache(config=model.config.get_text_config(decoder=True))
        cache_new = DynamicCache(config=model.config.get_text_config(decoder=True))
        old = native(**self.batch, past_key_values=cache_old, use_cache=True)
        ref = self.patch_module.apply_moss_generation_patch(adapter)
        new = model.prepare_inputs_for_generation(**self.batch, past_key_values=cache_new, use_cache=True)
        self.assertEqual(set(old), set(new))
        for key in old:
            if key != "past_key_values":
                if isinstance(old[key], self.torch.Tensor):
                    self.assertTrue(self.torch.equal(old[key], new[key]), key)
                else:
                    self.assertEqual(old[key], new[key], key)
        with self.torch.no_grad():
            out_old = model(**old)
            out_new = model(**new)
        self.assertTrue(self.torch.equal(out_old.logits, out_new.logits))
        self.assertEqual(cache_old.get_seq_length(), 141)
        self.assertEqual(cache_new.get_seq_length(), 141)
        self.assertEqual(model.config.to_dict(), config_before)
        self.assertEqual(model.forward, forward_before)
        self.assertEqual(model.gradient_checkpointing_enable, checkpoint_before)
        self.assertEqual(model._register_llm_deepstack_hooks, hooks_before)
        for name, parameter in model.named_parameters():
            self.assertTrue(self.torch.equal(parameter, parameters[name]), name)
        self.assertEqual(self.patch_module.apply_moss_generation_patch(adapter), ref)
        self.assertEqual(set(ref), {"path", "sha256", "size_bytes"})
        self.assertEqual(ref["sha256"], hashlib.sha256(PATCH_PATH.read_bytes()).hexdigest())
        self.assertEqual(ref["size_bytes"], PATCH_PATH.stat().st_size)

    def test_real_three_token_dynamic_cache_matches_native_reference(self):
        from transformers import DynamicCache
        adapter = self.adapter()
        model = adapter.model
        self.patch_module.apply_moss_generation_patch(adapter)
        seen, injections = [], []
        native_apply = model._apply_deepstack_to_hidden_states
        def record_injection(hidden, mask, embeds):
            result = native_apply(hidden, mask, embeds)
            self.assertTrue(self.torch.equal(result[~mask], hidden[~mask]))
            self.assertTrue(self.torch.equal(result[mask], hidden[mask] + embeds.reshape(-1, 32)))
            injections.append(mask.clone())
            return result
        def record_forward(module, positional, options):
            cache = options["past_key_values"]
            seen.append(dict(ids=options["input_ids"].clone(), cache=cache,
                             cached=int(cache.get_seq_length()),
                             attention=(options["attention_mask"].clone()
                                        if options.get("attention_mask") is not None else None),
                             audio=options.get("audio_data"), mask=options.get("audio_input_mask"),
                             seqlens=options.get("audio_data_seqlens")))
        handle = model.register_forward_pre_hook(record_forward, with_kwargs=True)
        try:
            with patch.object(model, "_apply_deepstack_to_hidden_states", side_effect=record_injection), \
                    patch.object(model, "get_audio_features", wraps=model.get_audio_features) as encoder:
                actual = self.generate(model)
                self.assertEqual(encoder.call_count, 1)
        finally:
            handle.remove()
        self.assertEqual(actual.sequences.shape, (1, 144))
        self.assertEqual([r["ids"].shape[1] for r in seen], [141, 1, 1])
        self.assertEqual([r["cached"] for r in seen], [0, 141, 142])
        for row, length in zip(seen, (141, 142, 143)):
            if row["attention"] is not None:
                self.assertEqual(row["attention"].shape[1], length)
        self.assertTrue(all(r["cache"] is seen[0]["cache"] for r in seen))
        self.assertEqual(seen[-1]["cache"].get_seq_length(), 143)
        self.assertEqual(len(injections), 2)
        for mask in injections:
            self.assertTrue(self.torch.equal(mask, self.batch["audio_input_mask"]))
        for row in seen[1:]:
            self.assertIsNone(row["audio"])
            self.assertIsNone(row["mask"])
            self.assertIsNone(row["seqlens"])
        # Reference is the unmodified official prepare with an explicit cache_position,
        # not a fake generation fixture or a cache-disabled full-prefix workaround.
        reference = self.adapter().model
        cache = DynamicCache(config=reference.config.get_text_config(decoder=True))
        ids = self.batch["input_ids"].clone()
        attention = self.batch["attention_mask"].clone()
        for step in range(3):
            inputs = reference.prepare_inputs_for_generation(
                **dict(self.batch, input_ids=ids, attention_mask=attention),
                past_key_values=cache, cache_position=ids.new_tensor([cache.get_seq_length()]),
                use_cache=True)
            with self.torch.no_grad():
                out = reference(**inputs)
            logits = out.logits[:, -1].float()
            self.assertTrue(self.torch.equal(logits, actual.logits[step]))
            logits[:, self.processor.tokenizer.eos_token_id] = -float("inf")
            token = logits.argmax(-1, keepdim=True)
            ids = self.torch.cat((ids, token), dim=1)
            attention = self.torch.cat((attention, self.torch.ones_like(token)), dim=1)
        self.assertTrue(self.torch.equal(ids, actual.sequences))

    def test_real_two_token_decode_and_false_padded_audio_mask(self):
        from transformers import DynamicCache
        adapter = self.adapter()
        model = adapter.model
        self.patch_module.apply_moss_generation_patch(adapter)
        self.assertEqual(self.generate(model, steps=2).sequences.shape, (1, 143))
        cache = DynamicCache(config=model.config.get_text_config(decoder=True))
        first = model.prepare_inputs_for_generation(**self.batch, past_key_values=cache, use_cache=True)
        with self.torch.no_grad():
            out = model(**first)
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        grown = dict(self.batch,
                     input_ids=self.torch.cat((self.batch["input_ids"], token), dim=1),
                     attention_mask=self.torch.cat((self.batch["attention_mask"], self.torch.ones_like(token)), dim=1),
                     audio_input_mask=self.torch.cat((self.batch["audio_input_mask"],
                                                      self.torch.zeros_like(token, dtype=self.torch.bool)), dim=1))
        self.assertFalse(grown["audio_input_mask"][0, -1])
        current = model.prepare_inputs_for_generation(**grown, past_key_values=cache,
                                                       next_sequence_length=1, use_cache=True)
        self.assertEqual(current["input_ids"].shape, (1, 1))
        self.assertIsNone(current["audio_data"])
        self.assertIsNone(current["audio_input_mask"])
        self.assertIs(current["past_key_values"], cache)
        generated_audio_id = grown["input_ids"].clone()
        generated_audio_id[0, -1] = self.processor.audio_token_id
        generated = model.prepare_inputs_for_generation(
            **dict(grown, input_ids=generated_audio_id), past_key_values=cache, use_cache=True)
        self.assertEqual(generated["input_ids"].item(), self.processor.audio_token_id)
        self.assertIsNone(generated["audio_input_mask"])
        self.assertIsNone(generated["audio_data"])
        with self.assertRaisesRegex(ValueError, "one-token"):
            model.prepare_inputs_for_generation(
                **dict(grown, input_ids=self.torch.cat((grown["input_ids"], token), dim=1)),
                past_key_values=cache, use_cache=True)
        self.assertTrue(self.torch.equal(self.batch["audio_input_mask"],
                                        self.batch["input_ids"].eq(self.processor.audio_token_id)))
        with self.assertRaisesRegex(ValueError, "KV cache length"):
            model.prepare_inputs_for_generation(**grown, past_key_values=cache,
                                                  cache_position=token.new_tensor([0]), use_cache=True)

    def test_wrong_original_prefix_audio_masks_fail_closed(self):
        adapter = self.adapter()
        self.patch_module.apply_moss_generation_patch(adapter)
        mask = self.batch["audio_input_mask"]
        misplaced = mask.clone()
        audio_index = int(mask[0].nonzero()[0])
        text_index = int((~mask[0]).nonzero()[0])
        misplaced[0, audio_index] = False
        misplaced[0, text_index] = True
        for wrong in (mask[:, :-1], misplaced, mask.long(), None):
            with self.subTest(shape=None if wrong is None else wrong.shape):
                with self.assertRaisesRegex(ValueError, "original prefix"):
                    adapter.model.prepare_inputs_for_generation(
                        **dict(self.batch, audio_input_mask=wrong), use_cache=True)

    def test_peft_instance_cache_and_checkpoint_hooks_preserved(self):
        adapter = self.adapter()
        self.adapters.configure_trainable(adapter, rank=2, alpha=4, audio_last_n=2)
        adapter.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        base = adapter.model.get_base_model()
        checkpoint_functions = [layer._gradient_checkpointing_func for layer in base.language_model.layers]
        target = self.torch.tensor([[100, 101, 102]], dtype=self.torch.long)
        training = dict(self.batch,
                        input_ids=self.torch.cat((self.batch["input_ids"], target), dim=1),
                        attention_mask=self.torch.cat((self.batch["attention_mask"], self.torch.ones_like(target)), dim=1),
                        audio_input_mask=self.torch.cat((self.batch["audio_input_mask"],
                                                         self.torch.zeros_like(target, dtype=self.torch.bool)), dim=1),
                        labels=self.torch.cat((self.torch.full_like(self.batch["input_ids"], -100), target), dim=1),
                        use_cache=False)
        adapter.model.train()
        before_loss = adapter.model(**training).loss
        before_loss.backward()
        before_gradients = {name: parameter.grad.clone() for name, parameter in adapter.model.named_parameters()
                            if parameter.grad is not None}
        adapter.model.zero_grad(set_to_none=True)
        ref = self.patch_module.apply_moss_generation_patch(adapter)
        after_loss = adapter.model(**training).loss
        after_loss.backward()
        self.assertTrue(self.torch.equal(before_loss, after_loss))
        for name, parameter in adapter.model.named_parameters():
            if name in before_gradients:
                self.assertTrue(self.torch.equal(before_gradients[name], parameter.grad), name)
        adapter.model.zero_grad(set_to_none=True)
        adapter.model.eval()
        actual = self.generate(adapter.model)
        self.assertEqual(actual.sequences.shape, (1, 144))
        self.assertEqual([layer._gradient_checkpointing_func for layer in base.language_model.layers],
                         checkpoint_functions)
        # Invoke the real frozen PlannerAdapter.generate and verify its eval-only provenance.
        with patch.object(adapter, "prefix", return_value=self.batch):
            adapter._last_audio_info = dict(processor_sample_rate=16000, n_samples=81000)
            result = adapter.generate({}, max_new_tokens=3)
        self.assertEqual(len(result["new_token_ids"]), 3)
        self.assertEqual(result["applied_generation_patch_ref"], ref)

    def _fake_bridge_success(self, entry):
        # Fake producer/adapter here tests only argv/provenance plumbing, NOT native
        # generation or GPU success; the separate seven tests exercise real MOSS.
        module = self.patch_module
        refs = {key: module._file_ref(path) for key, path in (
            ("adapter", FROZEN / "music_planner_adapters.py"),
            ("eval", FROZEN / "music_planner_eval.py"),
            ("fixed3", FROZEN / "music_planner_fixed3_eval.py"),
            ("patch", PATCH_PATH))}
        args = ["--frozen-source", str(FROZEN),
                "--expected-adapter-sha256", refs["adapter"]["sha256"],
                "--expected-eval-sha256", refs["eval"]["sha256"],
                "--expected-patch-sha256", refs["patch"]["sha256"]]
        if entry == "fixed3":
            args += ["--entry", "fixed3", "--expected-fixed3-sha256", refs["fixed3"]["sha256"]]
        with tempfile.TemporaryDirectory(prefix="moss_bridge_cpu_") as temporary:
            out = Path(temporary) / "new_eval"
            worker_args = (["generate"] if entry == "fixed3" else []) + [
                "--family", "moss", "--model", "/unused-model", "--checkpoint", "/unused-checkpoint",
                "--out", str(out), "--device", "cuda"]
            original_manifest = dict(
                schema="fake_bridge_cpu_plumbing_not_GPU", code=refs[entry],
                adapter_code=refs["adapter"], complete=True,
                checkpoint_identity=dict(fingerprint="unchanged-original-training-fingerprint",
                                         training_config=dict(adapter_source_sha256=refs["adapter"]["sha256"])))
            fake_adapter = types.SimpleNamespace(generate=Mock(return_value=dict(
                raw_text="unmodified fake producer output", new_token_ids=[11, 12], eos_token_ids=[13])))
            fake_loader = Mock(return_value=fake_adapter)
            adapters = types.SimpleNamespace(__file__=refs["adapter"]["path"],
                                             load_planner_adapter=fake_loader)
            def apply_fake(adapter):
                native_generate = adapter.generate
                def stamped(*positional, **options):
                    return dict(native_generate(*positional, **options),
                                applied_generation_patch_ref=refs["patch"])
                adapter.generate = stamped
                return refs["patch"]
            seen_argv = []
            def fake_main():
                seen_argv.extend(sys.argv)
                out.mkdir()
                (out / "manifest.json").write_text(json.dumps(original_manifest))
                if entry == "fixed3":
                    adapter = adapters.load_planner_adapter("moss", Path("/unused-model"),
                                                           checkpoint=Path("/unused-checkpoint"), device="cuda")
                else:
                    adapter = adapters.load_planner_adapter(family="moss", model_path=Path("/unused-model"),
                                                           checkpoint=Path("/unused-checkpoint"), device="cuda")
                result = adapter.generate({}, max_new_tokens=2048)
                (out / "raw.jsonl").write_text(json.dumps(dict(native_output=result)) + "\n")
            evaluator = types.SimpleNamespace(main=fake_main)
            old_argv = sys.argv
            with patch.dict(sys.modules, clear=False):
                sys.modules.pop("music_planner_adapters", None)
                with patch.object(module, "_load_module", side_effect=[adapters, evaluator]) as imports, \
                        patch.object(module, "apply_moss_generation_patch", side_effect=apply_fake) as applied:
                    module.main(args + ["--", *worker_args])
            self.assertIs(sys.argv, old_argv)
            self.assertIs(adapters.load_planner_adapter, fake_loader)
            self.assertEqual(seen_argv, [refs[entry]["path"], *worker_args])
            self.assertEqual(imports.call_args_list[1].args[1], refs[entry]["path"])
            fake_loader.assert_called_once()
            applied.assert_called_once_with(fake_adapter)
            manifest = json.loads((out / "manifest.json").read_text())
            for key, value in original_manifest.items():
                self.assertEqual(manifest[key], value, key)
            self.assertEqual(manifest["applied_generation_patch_ref"], refs["patch"])
            receipt_path = out / "generation_patch.json"
            self.assertEqual(manifest["generation_patch_receipt_ref"], module._file_ref(receipt_path))
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["applied_generation_patch_ref"], refs["patch"])
            self.assertEqual(receipt["frozen_source_refs"][entry], refs[entry])
            self.assertTrue(receipt["original_training_identity_preserved"])
            self.assertFalse(receipt["cache_disabled"])
            record = json.loads((out / "raw.jsonl").read_text())
            self.assertEqual(record["native_output"]["applied_generation_patch_ref"], refs["patch"])
            self.assertEqual(record["native_output"]["new_token_ids"], [11, 12])
            self.assertEqual(record["native_output"]["raw_text"], "unmodified fake producer output")
            self.assertEqual({path.name for path in out.iterdir()}, {"manifest.json", "raw.jsonl", "generation_patch.json"})

    def test_fake_eval_bridge_success_argv_and_provenance_only(self):
        self._fake_bridge_success("eval")

    def test_fake_fixed3_bridge_success_argv_and_provenance_only(self):
        self._fake_bridge_success("fixed3")

    def test_eval_bridge_refuses_unpinned_source_before_loading(self):
        with self.assertRaises(SystemExit) as caught:
            self.patch_module.main([
                "--frozen-source", str(FROZEN), "--expected-adapter-sha256", "0" * 64,
                "--expected-eval-sha256", "0" * 64, "--expected-patch-sha256", "0" * 64,
                "--", "--family", "moss", "--out", "/unused"] )
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
