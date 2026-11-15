"""Frozen audio-only SFT v3 provider with measured generation and artifact identity."""
import copy
import hashlib
import importlib.metadata
import json
import re
import sys
import time
import wave
from pathlib import Path

from transformers import GenerationConfig

PROMPT = "<audio-only: no text instruction>"
MAX_NEW_TOKENS = 1024
DECODING = dict(do_sample=True, temperature=0.7, top_p=0.9, top_k=20, num_beams=1,
                repetition_penalty=1.05, no_repeat_ngram_size=4,
                eos_token_id=[151645, 151643], pad_token_id=151643, use_cache=True)
ADAPTER_SOURCE = "/ssd/2/niezichao.nzc/artifacts/h3-jev-worktree-120fe6d/scripts_omnidance/exp/ms/music_planner_adapters.py"
EXPECTED_ADAPTER_SHA256 = "c1621d64f22275bcce51074ac708a4a8ec24e75cb9e365bbd45c63c79a03f9a7"
EXPECTED_LOADER_SHA256 = "01dcd73b7a4ba19dca26f9dc4e30142f539b27ecfc7979ae2600be8ca8c6fae5"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def _signature(path):
    stat = Path(path).stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _case_id(request):
    explicit = request.get("case_id")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip() or len(explicit) > 512:
            raise ValueError("A bounded nonempty canonical case_id is required")
        return explicit, "explicit_request_case_id"
    call_id = request["call_id"]
    if not isinstance(call_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*/untimed/[0-9]{4}/[0-9]{2}", call_id) is None:
        raise ValueError("Supply case_id outside the predeclared untimed catalog")
    return call_id, "predeclared_song_window_candidate_id"


class CheckpointProvider:
    def __init__(self, spec):
        self.spec = copy.deepcopy(spec)
        if spec.get("family", "omni_captioner") != "omni_captioner" or spec.get("task_prompt"):
            raise ValueError("SFT v3 requires the audio-only Captioner without task_prompt")
        self.model = Path(spec["model"]).resolve()
        self.checkpoint = Path(spec["checkpoint"]).resolve()
        self.max_length = spec.get("max_length", 8192)
        if type(self.max_length) is not int or self.max_length <= MAX_NEW_TOKENS:
            raise ValueError("The context must leave room for the audio prefix and frozen output")
        self.adapter_dir = (self.checkpoint / "adapter" if (self.checkpoint / "adapter").is_dir()
                            else self.checkpoint)
        self.adapter_source = Path(spec.get("adapter_source", ADAPTER_SOURCE)).resolve()
        if _sha(self.adapter_dir / "adapter_model.safetensors") != EXPECTED_ADAPTER_SHA256:
            raise ValueError("Adapter is not the selected SFT v3 checkpoint-0001000 export")
        if _sha(self.adapter_source) != EXPECTED_LOADER_SHA256:
            raise ValueError("Inference loader differs from the reviewed source")
        exports = {p.name: _sha(p) for p in sorted(self.adapter_dir.iterdir()) if p.is_file()}
        if "processor_config.json" not in exports:
            raise ValueError("Portable processor configuration is required; no silent fallback")
        index_path = self.model / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        names = sorted(set(index["weight_map"].values()))
        for name in names:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Model shard path escapes the base model")
        self.generation_config = GenerationConfig(**DECODING, max_new_tokens=MAX_NEW_TOKENS)
        self._generation_snapshot = copy.deepcopy(self.generation_config.to_dict())
        self._base_generation_path = self.model / "generation_config.json"
        self._base_generation_present = self._base_generation_path.is_file()
        assets = [self.model / "config.json", index_path, self.adapter_source, Path(__file__).resolve()]
        assets += [self.model / name for name in names]
        assets += [self.adapter_dir / name for name in exports]
        if self._base_generation_present:
            assets.append(self._base_generation_path)
        self._asset_signatures = {p: _signature(p) for p in assets}
        marker_path = self.checkpoint / "trainer_state.json"
        marker = json.loads(marker_path.read_text()) if marker_path.is_file() else {}
        self._identity = dict(
            kind="native", family="omni_captioner", synthetic=False,
            protocol="choreobench.sft_v3_frozen_decode.v2",
            model=str(self.model), checkpoint=str(self.checkpoint),
            base_config_sha256=_sha(self.model / "config.json"),
            base_index_sha256=_sha(index_path),
            base_weight_shards=[dict(name=name, size=(self.model / name).stat().st_size,
                                     sha256=_sha(self.model / name)) for name in names],
            adapter_exports=exports, adapter_source_sha256=EXPECTED_LOADER_SHA256,
            provider_source_sha256=_sha(__file__),
            checkpoint_fingerprint=marker.get("fingerprint"),
            checkpoint_marker_available=bool(marker),
            prompt=PROMPT, prompt_sha256=_object_sha(PROMPT), max_length=self.max_length,
            sampling=dict(copy.deepcopy(DECODING), max_new_tokens=MAX_NEW_TOKENS),
            resolved_generation_config=copy.deepcopy(self._generation_snapshot),
            base_generation_config_sha256=(_sha(self._base_generation_path)
                                           if self._base_generation_present else None),
            audio_clock=dict(sample_rate=16000, num_samples=81000),
            device_map=copy.deepcopy(spec.get("device_map", spec.get("device", "auto"))),
            seed_policy="SHA256('sol61-select:' + canonical case_id) first 8 hex digits",
            catalog_case_id_policy="predeclared song/untimed/window/candidate, independent of downstream method",
            runtime_versions={name: importlib.metadata.version(name)
                              for name in ("torch", "transformers", "peft")},
            training_source_identity_verified=False,
            processor_policy="load selected portable processor export explicitly; no base fallback",
            benchmark_first_line_protocol_equivalent=False)
        self._assert_unchanged()
        self.adapter = None

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def _assert_unchanged(self):
        if (self._base_generation_path.is_file() != self._base_generation_present
                or any(_signature(path) != signature for path, signature in self._asset_signatures.items())):
            raise ValueError("Frozen SFT provider artifact changed; start a new run")
        if self.generation_config.to_dict() != self._generation_snapshot:
            raise ValueError("Resolved SFT generation contract changed")

    def _load(self):
        self._assert_unchanged()
        if self.adapter is not None:
            return
        import importlib.util
        name = "_sol61_adapters_" + EXPECTED_LOADER_SHA256[:12]
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, self.adapter_source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                if sys.modules.get(name) is module:
                    del sys.modules[name]
                raise
        module = sys.modules[name]
        source_file = getattr(module, "__file__", None)
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        code = getattr(getattr(module, "load_planner_adapter", None), "__code__", None)
        if (source_file is None or origin is None or code is None
                or Path(source_file).resolve() != self.adapter_source
                or Path(origin).resolve() != self.adapter_source
                or Path(code.co_filename).resolve() != self.adapter_source):
            raise ValueError("Cached inference loader provenance differs from the verified source")
        exported_processor = module.load_native_processor("omni_captioner", self.adapter_dir)
        import torch
        device_map = self.spec.get("device_map", self.spec.get("device", "auto"))
        self.adapter = module.load_planner_adapter(
            "omni_captioner", self.model, checkpoint=self.checkpoint,
            max_length=self.max_length, device_map=device_map, dtype=torch.bfloat16)
        # The shared loader permits fallback; always use the verified export here.
        self.adapter.processor = exported_processor
        self._processor_export_verified = True
        self._assert_unchanged()

    def generate(self, request):
        budget = request["max_output_tokens"]
        if type(budget) is not int or budget != MAX_NEW_TOKENS:
            raise ValueError("The selected SFT v3 decode budget is exactly 1024 tokens")
        if request["response_kind"] != "local_caption":
            raise ValueError("The checkpoint provider accepts local_caption only")
        case_id, case_id_source = _case_id(request)
        audio = request["audio"]
        rate, samples = audio["sample_rate"], audio["num_samples"]
        if type(rate) is not int or type(samples) is not int or rate != 16000 or samples != 81000:
            raise ValueError("SFT v3 collection requires 16000 Hz mono PCM16: 81000 samples (5.0625 seconds)")
        path = Path(audio["path"])
        if _sha(path) != audio["sha256"]:
            raise ValueError("Audio SHA256 differs from the journaled observation")
        with wave.open(str(path), "rb") as handle:
            if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()) != (1, 2, rate, samples):
                raise ValueError("Observation must match its mono PCM16 waveform declaration")
        self._load()
        import torch
        adapter = self.adapter
        row = dict(id=case_id, audio_path=str(path), audio_sha256=audio["sha256"],
                   audio_window_seconds=[0.0, 5.0625])
        seed = int(hashlib.sha256(("sol61-select:" + case_id).encode()).hexdigest()[:8], 16)
        started = time.monotonic()
        batch = adapter.prefix(row)
        prefix = batch["input_ids"]
        if prefix.shape[1] + budget > adapter.max_length:
            raise ValueError("Generation budget exceeds the expert context")
        adapter.model.eval()
        parameter = next(adapter.model.parameters())
        torch.manual_seed(seed)
        if parameter.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode(), torch.autocast(parameter.device.type, dtype=torch.bfloat16,
                                                   enabled=parameter.dtype == torch.bfloat16):
            output = adapter.model.generate(**adapter._move(batch), generation_config=self.generation_config)
        sequence = output.sequences if hasattr(output, "sequences") else output
        if not torch.equal(sequence[:, :prefix.shape[1]].cpu(), prefix.cpu()):
            raise ValueError("Native generation changed its input prefix")
        tokens = sequence[0, prefix.shape[1]:].cpu().tolist()
        if len(tokens) > budget:
            raise ValueError("Model returned more tokens than the frozen budget")
        decoded = adapter.tokenizer.decode(tokens, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
        if tokens and tokens[-1] in DECODING["eos_token_id"]:
            stop = "eos"
        elif len(tokens) == budget:
            stop = "budget_truncation"
        else:
            stop = "incomplete_output"
        self._assert_unchanged()
        return dict(text=decoded, raw=dict(raw_text=decoded, new_token_ids=tokens,
                    stop_reason=stop, seed=seed, canonical_case_id=case_id,
                    case_id_source=case_id_source, decoding=dict(copy.deepcopy(DECODING), max_new_tokens=budget),
                    resolved_generation_config=copy.deepcopy(self._generation_snapshot),
                    input_tokens=int(prefix.shape[1]), model_loaded=True,
                    processor_export_verified=getattr(self, "_processor_export_verified", False),
                    audio_input="full-waveform"),
                    identity=self.identity, synthetic=False, stop_reason=stop,
                    costs=dict(input_tokens=int(prefix.shape[1]), output_tokens=len(tokens),
                               latency_seconds=time.monotonic() - started))


def create(spec, role="local", base_dir=None):
    if role != "local":
        raise ValueError("The checkpoint provider is a local audio model only")
    return CheckpointProvider(spec)
