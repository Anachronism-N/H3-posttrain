"""CPU tests; set MOSS_OFFICIAL_MODEL_PATH for actual HF/DeepSpeed regressions."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "scripts_omnidance/exp/ms/music_planner_moss_position_buffer_patch.py"
spec = importlib.util.spec_from_file_location("_moss_position_patch_test", PATCH_PATH)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

NATIVE_SOURCE = '''import math
import torch
from torch import nn
class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, num_positions, embedding_dim):
        super().__init__()
        increment = math.log(10000.0) / (embedding_dim // 2 - 1)
        self.register_buffer("inv_timescales", torch.exp(-increment * torch.arange(embedding_dim // 2).float()), persistent=False)
    def forward(self, seq_len, device):
        scaled = torch.arange(seq_len, device=device, dtype=self.inv_timescales.dtype).unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1).unsqueeze(0)
'''


def load_native(root, namespace):
    package = types.ModuleType(namespace)
    package.__path__ = [str(root)]
    sys.modules[namespace] = package
    for name in ("configuration_moss_music", "modeling_moss_music"):
        path = root / (name + ".py")
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(namespace + "." + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


class FakeMoss(torch.nn.Module):
    def __init__(self, position_cls):
        super().__init__()
        self.config = types.SimpleNamespace(model_type="moss_music", audio_config=types.SimpleNamespace(d_model=1280))
        self.audio_encoder = torch.nn.Module()
        self.audio_encoder.embed_positions = position_cls(1500, 1280)
        self.weight = torch.nn.Parameter(torch.randn(3, 5))
        self.lora_A = torch.nn.Parameter(torch.randn(2, 5))
        self.register_buffer("other_buffer", torch.arange(3))
        self.prepare_inputs_for_generation = object()
        self._planner_native_forward = object()
        self.next_logits = torch.ones(1, 2, 17)

    def forward(self):
        self.last_output = types.SimpleNamespace(logits=self.next_logits)
        return self.last_output


class PositionBufferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="moss_position_unit_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "modeling_moss_music.py").write_text(NATIVE_SOURCE)
        self.native = load_native(self.root, "_position_fixture_" + str(id(self)))
        self.reference = self.native.SinusoidsPositionEmbedding(1500, 1280).inv_timescales.clone()
        (self.root / "config.json").write_text(json.dumps(dict(audio_config=dict(d_model=1280, max_source_positions=1500))))
        (self.root / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map={patch.BUFFER_KEY: "model.safetensors"})))
        self.save_source(self.reference.bfloat16())
        self.model = FakeMoss(self.native.SinusoidsPositionEmbedding)
        self.adapter = types.SimpleNamespace(family="moss", model=self.model)

    def save_source(self, tensor):
        save_file({patch.BUFFER_KEY: tensor}, str(self.root / "model.safetensors"))

    def test_exact_restore_keeps_parameters_other_buffers_config_hooks(self):
        self.model.audio_encoder.embed_positions.inv_timescales.fill_(float("nan"))
        self.model.weight.grad = torch.randn_like(self.model.weight)
        self.model.lora_A.grad = torch.randn_like(self.model.lora_A)
        parameters = {k: (v, v.detach().clone(), v.grad, v.grad.clone()) for k, v in self.model.named_parameters()}
        other = self.model.other_buffer
        config = self.model.config
        config_contents = copy.deepcopy(vars(config))
        native_forward = self.model._planner_native_forward
        prepare = self.model.prepare_inputs_for_generation
        hook = self.model.register_forward_hook(lambda m, i, o: None)
        self.addCleanup(hook.remove)
        hooks = self.model._forward_hooks.copy()
        receipt = patch.restore_moss_position_buffer(self.adapter, self.root)
        actual = self.model.audio_encoder.embed_positions.inv_timescales
        self.assertTrue(torch.equal(actual, self.reference))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertNotIn(patch.BUFFER_KEY, self.model.state_dict())
        self.assertEqual(receipt["before_stats"]["nan_count"], 640)
        self.assertEqual(receipt["source_tensor_ref"]["dtype"], "torch.bfloat16")
        self.assertEqual(receipt["canonical_tensor_ref"], patch._tensor_ref(self.reference))
        json.dumps(receipt, allow_nan=False)
        self.assertIs(self.model.other_buffer, other)
        self.assertIs(self.model.config, config)
        self.assertEqual(vars(config), config_contents)
        self.assertIs(self.model._planner_native_forward, native_forward)
        self.assertIs(self.model.prepare_inputs_for_generation, prepare)
        self.assertEqual(self.model._forward_hooks, hooks)
        for key, param in self.model.named_parameters():
            old, values, grad, grad_values = parameters[key]
            self.assertIs(param, old)
            self.assertTrue(torch.equal(param, values))
            self.assertIs(param.grad, grad)
            self.assertTrue(torch.equal(param.grad, grad_values))
        expected = self.native.SinusoidsPositionEmbedding(1500, 1280)(11, torch.device("cpu"))
        self.assertTrue(torch.equal(self.model.audio_encoder.embed_positions(11, torch.device("cpu")), expected))

    def test_native_and_peft_repeated_restore(self):
        self.model.audio_encoder.embed_positions.inv_timescales.fill_(-1e21)
        self.adapter.model = types.SimpleNamespace(get_base_model=lambda: self.model)
        first = patch.restore_moss_position_buffer(self.adapter, self.root)
        second = patch.restore_moss_position_buffer(self.adapter, self.root)
        self.assertEqual(first["canonical_tensor_ref"], second["canonical_tensor_ref"])
        self.assertEqual(second["before_stats"]["finite_count"], 640)

    def test_official_fp32_supported(self):
        self.save_source(self.reference)
        receipt = patch.restore_moss_position_buffer(self.adapter, self.root)
        self.assertEqual(receipt["source_tensor_ref"]["dtype"], "torch.float32")

    def test_wrong_official_values_shape_dtype_nan_and_negative_refused(self):
        wrong = self.reference.clone(); wrong[10] *= 2
        nan = self.reference.clone(); nan[10] = float("nan")
        negative = self.reference.clone(); negative[10] = -1
        for tensor in [wrong, self.reference[:639], self.reference.half(), nan, negative]:
            with self.subTest(dtype=tensor.dtype, shape=tensor.shape):
                self.save_source(tensor)
                old = self.model.audio_encoder.embed_positions.inv_timescales
                with self.assertRaises(ValueError):
                    patch.restore_moss_position_buffer(self.adapter, self.root)
                self.assertIs(self.model.audio_encoder.embed_positions.inv_timescales, old)

    def test_persistent_duplicate_wrong_shape_and_missing_key_refused(self):
        owner = self.model.audio_encoder.embed_positions
        owner._non_persistent_buffers_set.clear()
        with self.assertRaises(ValueError): patch.restore_moss_position_buffer(self.adapter, self.root)
        owner._non_persistent_buffers_set.add("inv_timescales")
        self.model.register_buffer("inv_timescales", torch.ones(640), persistent=False)
        with self.assertRaises(ValueError): patch.restore_moss_position_buffer(self.adapter, self.root)
        del self.model.inv_timescales
        owner._buffers["inv_timescales"] = torch.ones(639)
        with self.assertRaises(ValueError): patch.restore_moss_position_buffer(self.adapter, self.root)
        owner._buffers["inv_timescales"] = self.reference.clone()
        (self.root / "model.safetensors.index.json").write_text('{"weight_map":{}}')
        with self.assertRaises(ValueError): patch.restore_moss_position_buffer(self.adapter, self.root)

    def test_source_identity_and_shard_escape_refused(self):
        with tempfile.TemporaryDirectory(prefix="moss_wrong_official_") as temp:
            official = Path(temp)
            for source in self.root.iterdir():
                if source.is_file():
                    (official / source.name).write_bytes(source.read_bytes())
            (official / "modeling_moss_music.py").write_text(NATIVE_SOURCE + "# changed source\n")
            with self.assertRaises(ValueError): patch.restore_moss_position_buffer(self.adapter, official)
        (self.root / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map={patch.BUFFER_KEY: "../model.safetensors"})))
        with self.assertRaises(ValueError): patch.restore_moss_position_buffer(self.adapter, self.root)

    def test_generation_validator_preserves_finite_output_refuses_nan_inf(self):
        logits = torch.randn(2, 3, 17)
        values = logits.clone()
        self.assertIs(patch.validate_moss_generation_logits(logits), logits)
        self.assertTrue(torch.equal(logits, values))
        for value in [float("nan"), float("inf"), -float("inf")]:
            broken = logits.clone(); broken[1, -1, 5] = value
            with self.assertRaises(FloatingPointError): patch.validate_moss_generation_logits(broken)
        with self.assertRaises(ValueError): patch.validate_moss_generation_logits(torch.ones(1, 17))

    def test_generation_hook_observes_native_outputs_without_replacing(self):
        hook = patch.install_moss_generation_logits_validator(self.adapter)
        output = self.model()
        self.assertIs(output, self.model.last_output)
        self.assertIs(output.logits, self.model.next_logits)
        self.model.next_logits.fill_(float("nan"))
        with self.assertRaises(FloatingPointError): self.model()
        hook.remove()
        self.assertIs(self.model().logits, self.model.next_logits)


@unittest.skipUnless(os.environ.get("MOSS_OFFICIAL_MODEL_PATH"), "Set MOSS_OFFICIAL_MODEL_PATH for real official native/HF/ZeRO3 tests")
class RealNativeHFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path(os.environ["MOSS_OFFICIAL_MODEL_PATH"])
        cls.native = load_native(cls.root, "_position_real_official")
        from transformers import Qwen3Config, WhisperConfig
        from transformers.integrations.deepspeed import unset_hf_deepspeed_config
        unset_hf_deepspeed_config()
        audio = WhisperConfig(d_model=1280, encoder_layers=0, encoder_attention_heads=20, encoder_ffn_dim=32,
                              max_source_positions=16, num_mel_bins=128)
        audio.downsample_hidden_size = 2
        audio.downsample_rate = 8
        audio.output_dim = 16
        audio.deepstack_encoder_layer_indexes = []
        audio.layer_norm_eps = 1e-5
        language = Qwen3Config(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                               num_key_value_heads=1, head_dim=8, vocab_size=32, bos_token_id=1, eos_token_id=2)
        config_cls = sys.modules["_position_real_official.configuration_moss_music"].MossMusicConfig
        cls.config = config_cls(audio_config=audio, language_config=language, adapter_hidden_size=32)
        torch.manual_seed(7)
        cls.constructor = cls.native.MossMusicModel(cls.config)
        cls.reference = cls.constructor.audio_encoder.embed_positions.inv_timescales.clone()
        cls.temp = tempfile.TemporaryDirectory(prefix="moss_real_tiny_checkpoint_")
        cls.constructor.save_pretrained(cls.temp.name)
        # Released MOSS includes this non-persistent key; retain the exact layout on tiny weights.
        path = str(Path(cls.temp.name) / "model.safetensors")
        weights = load_file(path)
        weights[patch.BUFFER_KEY] = cls.reference.bfloat16()
        save_file(weights, path, metadata={"format": "pt"})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_real_hf_default_meta_loses_native_buffer(self):
        from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
        self.assertFalse(is_deepspeed_zero3_enabled())
        for dtype in [torch.float32, torch.bfloat16]:
            loaded, info = self.native.MossMusicModel.from_pretrained(
                self.temp.name, config=self.config, dtype=dtype, output_loading_info=True)
            actual = loaded.audio_encoder.embed_positions.inv_timescales
            self.assertIn(patch.BUFFER_KEY, info["unexpected_keys"])
            self.assertFalse(actual.is_meta)
            self.assertFalse(torch.equal(actual, self.reference))
            self.assertFalse(torch.equal(loaded.audio_encoder.embed_positions(5, torch.device("cpu")),
                                         self.constructor.audio_encoder.embed_positions(5, torch.device("cpu"))))
            print("REAL_HF_META_BUFFER", str(dtype), patch._buffer_stats(actual), flush=True)
            parameters = {name: (param, param.detach().clone()) for name, param in loaded.named_parameters()}
            other_buffers = {name: (value, value.clone()) for name, value in loaded.named_buffers()
                             if name != patch.BUFFER_KEY}
            receipt = patch.restore_moss_position_buffer(types.SimpleNamespace(family="moss", model=loaded), self.root)
            self.assertTrue(torch.equal(loaded.audio_encoder.embed_positions.inv_timescales, self.reference))
            for name, param in loaded.named_parameters():
                old, values = parameters[name]
                self.assertIs(param, old)
                self.assertTrue(torch.equal(param, values))
            for name, value in loaded.named_buffers():
                if name != patch.BUFFER_KEY:
                    old, values = other_buffers[name]
                    self.assertIs(value, old)
                    self.assertTrue(torch.equal(value, values))
            self.assertEqual(receipt["canonical_tensor_ref"]["sha256"], "941431bea09bd9b86db14c34ece2741fb1575a8fbccc29301dcaac38723b942e")

    def test_02_real_cpu_hf_zero3_preserves_constructor_buffer(self):
        import deepspeed
        from transformers.integrations.deepspeed import HfDeepSpeedConfig, is_deepspeed_zero3_enabled, unset_hf_deepspeed_config
        self.assertFalse(torch.cuda.is_available(), "This regression must run with CUDA_VISIBLE_DEVICES empty")
        self.assertFalse(torch.distributed.is_initialized())
        os.environ.setdefault("LOCAL_RANK", "0")
        with tempfile.TemporaryDirectory(prefix="moss_real_gloo_") as rendezvous:
            deepspeed.init_distributed(dist_backend="gloo", init_method="file://" + rendezvous + "/init",
                                      rank=0, world_size=1, auto_mpi_discovery=False)
            try:
                for dtype in [torch.float32, torch.bfloat16]:
                    ds_config = HfDeepSpeedConfig(dict(train_batch_size=1, train_micro_batch_size_per_gpu=1,
                                                      gradient_accumulation_steps=1,
                                                      zero_optimization=dict(stage=3, offload_param=dict(device="cpu")),
                                                      fp16=dict(enabled=False), bf16=dict(enabled=dtype == torch.bfloat16)))
                    self.assertTrue(is_deepspeed_zero3_enabled())
                    loaded, info = self.native.MossMusicModel.from_pretrained(
                        self.temp.name, config=self.config, dtype=dtype, output_loading_info=True)
                    self.assertTrue(hasattr(next(loaded.parameters()), "ds_id"))
                    actual = loaded.audio_encoder.embed_positions.inv_timescales
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertTrue(torch.equal(actual, self.reference))
                    self.assertTrue(torch.equal(loaded.audio_encoder.embed_positions(5, torch.device("cpu")),
                                                self.constructor.audio_encoder.embed_positions(5, torch.device("cpu"))))
                    print("REAL_HF_ZERO3_BUFFER", str(dtype), patch._tensor_ref(actual), flush=True)
                    del loaded, ds_config
                    unset_hf_deepspeed_config()
            finally:
                unset_hf_deepspeed_config()
                torch.distributed.destroy_process_group()

    def test_03_actual_official_640_native_restore_and_audio_positions(self):
        position_cls = self.native.SinusoidsPositionEmbedding
        model = FakeMoss(position_cls)
        adapter = types.SimpleNamespace(family="moss", model=model)
        reference = position_cls(1500, 1280).inv_timescales.clone()
        model.audio_encoder.embed_positions.inv_timescales.fill_(float("nan"))
        receipt = patch.restore_moss_position_buffer(adapter, self.root)
        self.assertEqual(receipt["source_tensor_ref"]["sha256"], "915bebc276a4b2cacd0ce5060e5d14c87fbd0cce7493b2eabf82b275a7a8b7b5")
        self.assertEqual(receipt["canonical_tensor_ref"]["sha256"], "941431bea09bd9b86db14c34ece2741fb1575a8fbccc29301dcaac38723b942e")
        self.assertTrue(torch.equal(model.audio_encoder.embed_positions.inv_timescales, reference))
        self.assertTrue(torch.equal(model.audio_encoder.embed_positions(150, torch.device("cpu")),
                                    position_cls(1500, 1280)(150, torch.device("cpu"))))
        print("REAL_OFFICIAL_RESTORE", json.dumps(receipt, allow_nan=False), flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
