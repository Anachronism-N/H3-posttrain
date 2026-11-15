"""Local expert provider backed by a trained Captioner checkpoint (sol61 v3).

The catalog run needs a provider whose identity and outputs are journaled by the
same machinery as the base-only path. This provider loads the official Captioner
base plus a portable adapter through the training-eval entry points and records
exactly what it verified:

- adapter export and trainable.pt content hashes;
- the checkpoint marker's fingerprint and trainable signature key set;
- the base model config hash and shard sizes;
- the inference adapter source hash.

It deliberately does NOT claim the training-source identity was verified: the
exact trainer file recorded by the run is not available here, so
``training_source_identity_verified`` is false and the report says so.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

# The trained Captioner takes audio only; this string is provenance metadata, not an input.
PROMPT = "<audio-only: no text instruction>"
DECODING = dict(do_sample=True, temperature=0.7, top_p=0.9, top_k=20, num_beams=1,
                repetition_penalty=1.05, no_repeat_ngram_size=4,
                eos_token_id=[151645, 151643], pad_token_id=151643, use_cache=True)
ADAPTER_SOURCE = "/ssd/2/niezichao.nzc/artifacts/h3-jev-worktree-120fe6d/scripts_omnidance/exp/ms/music_planner_adapters.py"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class CheckpointProvider:
    def __init__(self, spec):
        self.spec = json.loads(json.dumps(spec))
        self.model = Path(spec["model"]).resolve()
        self.checkpoint = Path(spec["checkpoint"]).resolve()
        self.device = spec.get("device", "cpu")
        self.max_length = int(spec.get("max_length", 8192))
        self.sampling = dict(DECODING)
        marker_path = self.checkpoint / "trainer_state.json"
        marker = json.loads(marker_path.read_text()) if marker_path.is_file() else {}
        adapter = self.checkpoint / "adapter" if (self.checkpoint / "adapter").is_dir() else self.checkpoint
        index = json.loads((self.model / "model.safetensors.index.json").read_text())
        names = sorted(set(index["weight_map"].values()))
        self.identity = dict(
            kind="native", family="omni_captioner", synthetic=False,
            model=str(self.model), checkpoint=str(self.checkpoint),
            base_config_sha256=_sha(self.model / "config.json"),
            base_weight_shards=[dict(name=n, size=(self.model / n).stat().st_size) for n in names],
            checkpoint_fingerprint=marker.get("fingerprint"),
            checkpoint_complete=marker.get("complete") is True,
            checkpoint_trainable_sha256=(_sha(self.checkpoint / "trainable.pt")
                                          if (self.checkpoint / "trainable.pt").is_file() else None),
            marker_trainable_sha256=marker.get("trainable_sha256"),
            checkpoint_marker_available=bool(marker),
            adapter_exports={p.name: _sha(p) for p in sorted(adapter.iterdir()) if p.is_file()},
            adapter_source_sha256=_sha(Path(__file__).resolve().parents[2] / "scripts_omnidance/exp/ms/music_planner_adapters.py"),
            prompt=PROMPT, prompt_sha256=_object_sha(PROMPT),
            max_length=self.max_length, sampling=dict(self.sampling),
            training_source_identity_verified=False,
            training_source_note="trainer file recorded by the run is unavailable; not verified",
            loaded=False)
        if (self.identity["checkpoint_trainable_sha256"] is not None
                and self.identity["checkpoint_trainable_sha256"] != self.identity["marker_trainable_sha256"]):
            raise ValueError("trainable.pt differs from the checkpoint marker")
        self.adapter = None

    def _load(self):
        if self.adapter is not None:
            return
        import importlib.util
        source = Path(self.spec.get("adapter_source", ADAPTER_SOURCE))
        name = "_sol61_adapters_" + _sha(source)[:12]
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        load_planner_adapter = sys.modules[name].load_planner_adapter
        device_map = self.spec.get("device_map", "auto")
        kwargs = dict(checkpoint=self.checkpoint, max_length=self.max_length)
        if device_map == "auto":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device"] = self.device
        self.adapter = load_planner_adapter("omni_captioner", self.model, **kwargs)
        self.loaded_learned_param_count = sum(
            1 for n, p in self.adapter.model.named_parameters() if p.requires_grad)

    def generate(self, request):
        self._load()
        import torch
        adapter = self.adapter
        audio = request["audio"]
        row = dict(id=request["call_id"], audio_path=audio["path"], audio_sha256=audio["sha256"],
                   audio_window_seconds=[0.0, 5.0625])
        seed = int(hashlib.sha256(("sol61-select:" + request["call_id"]).encode()).hexdigest()[:8], 16)
        batch = adapter.prefix(row)
        prefix = batch["input_ids"]
        budget = int(request["max_output_tokens"])
        if prefix.shape[1] + budget > adapter.max_length:
            raise ValueError("Generation budget exceeds the expert context")
        adapter.model.eval()
        parameter = next(adapter.model.parameters())
        started = time.monotonic()
        torch.manual_seed(seed)
        if parameter.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad(), torch.autocast(parameter.device.type, dtype=torch.bfloat16,
                                             enabled=parameter.dtype == torch.bfloat16):
            output = adapter.model.generate(**adapter._move(batch), max_new_tokens=budget, **DECODING)
        sequence = output.sequences if hasattr(output, "sequences") else output
        if not torch.equal(sequence[:, :prefix.shape[1]].cpu(), prefix.cpu()):
            raise ValueError("Native generation changed its input prefix")
        tokens = sequence[0, prefix.shape[1]:].cpu().tolist()
        text = adapter.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        eos = DECODING["eos_token_id"]
        stop = "eos" if tokens and tokens[-1] in eos else "budget_truncation"
        return dict(text=text,
                    raw=dict(raw_text=text, new_token_ids=tokens, stop_reason=stop, seed=seed,
                             loaded_learned_param_count=getattr(self, "loaded_learned_param_count", None),
                             decoding=dict(DECODING), input_tokens=int(prefix.shape[1]),
                             audio_input="full-waveform"),
                    identity=self.identity, synthetic=False, stop_reason=stop,
                    costs=dict(input_tokens=int(prefix.shape[1]), output_tokens=len(tokens),
                               latency_seconds=time.monotonic() - started))


def create(spec, role="local", base_dir=None):
    if role != "local":
        raise ValueError("The checkpoint provider is a local audio model only")
    return CheckpointProvider(spec)
