"""Inference-only LoRA activation chunking; weights and sampler remain unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, "/apdcephfs_gy4/share_302533218/cedricnie/DiffSynth-Studio-upstream-20260930")
import torch
import h3_video_infer as core
from diffsynth.core.vram.layers import LoRAHotLoadMixin

CORE_SHA = '77bafd9f6165f80328d4e272c901b21fbf45da3898161bcc039a572e1c86a2f8'
CHUNK_TOKENS = 1024
COUNTERS = {'layer_calls': 0, 'chunk_calls': 0, 'max_chunk_tokens': 0}

def chunked_lora(self, x, out):
    if torch.is_grad_enabled() or self.lora_merger is not None:
        raise RuntimeError('Chunked LoRA is supported only for ordinary inference; no training or merger')
    if x.ndim < 2 or out.shape[:-1] != x.shape[:-1]:
        raise ValueError('LoRA input/output token layouts differ')
    if len(self.lora_A_weights) != len(self.lora_B_weights):
        raise ValueError('Incomplete LoRA pairs')
    COUNTERS['layer_calls'] += 1
    for a, b in zip(self.lora_A_weights, self.lora_B_weights):
        a = a.to(device=x.device, dtype=x.dtype)
        b = b.to(device=x.device, dtype=x.dtype)
        for start in range(0, x.shape[-2], CHUNK_TOKENS):
            end = min(start + CHUNK_TOKENS, x.shape[-2])
            # Each row uses exactly the same two low-rank matmuls and one addition.
            # Only the temporary activation is chunked, never the condition sequence.
            delta = (x[..., start:end, :] @ a.T) @ b.T
            out[..., start:end, :].add_(delta)
            COUNTERS['chunk_calls'] += 1
            COUNTERS['max_chunk_tokens'] = max(COUNTERS['max_chunk_tokens'], end-start)
    return out

def self_test():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    tests = []
    for dtype in (torch.float32, torch.bfloat16):
        for shape in ((17, 13), (2, 2051, 13)):
            x = torch.randn(shape).to(dtype)
            base = torch.randn((*shape[:-1], 23)).to(dtype)
            module = SimpleNamespace(lora_A_weights=[torch.randn(4, 13).to(dtype) * .1, torch.randn(3, 13).to(dtype) * .1],
                                     lora_B_weights=[torch.randn(23, 4).to(dtype) * .1, torch.randn(23, 3).to(dtype) * .1], lora_merger=None)
            original = base.clone()
            with torch.inference_mode():
                for a,b in zip(module.lora_A_weights,module.lora_B_weights): original = original + (x @ a.T) @ b.T
                actual = chunked_lora(module, x, base.clone())
            tolerance = 2e-6 if dtype == torch.float32 else .008
            torch.testing.assert_close(actual, original, rtol=tolerance, atol=tolerance)
            tests.append({'dtype': str(dtype), 'shape': list(shape), 'maximum_error': float((actual.float()-original.float()).abs().max())})
    print(json.dumps({'chunked_lora_CPU_regression_passed': True, 'tests': tests,
                      'weights_and_sampling_unchanged': True, 'full_token_sequence_retained': True}), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption-json', type=Path)
    parser.add_argument('--weights-root', type=Path, default=Path('/dev/shm/h3/Ref2VA'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--condition-cache', type=Path)
    parser.add_argument('--lora', type=Path)
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cfg', type=float, default=1.)
    parser.add_argument('--flow-shift', type=float, default=12.)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test();return
    if hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest() != CORE_SHA:
        raise RuntimeError('Original inference source changed')
    if not args.lora or not args.condition_cache or not args.out:
        raise ValueError('Use a complete condition cache and the unchanged LoRA file')
    args.cache_only = False
    implementation = {'source': str(Path(__file__).resolve()), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'original_inference_sha256': CORE_SHA, 'method': 'inference_only_token_chunked_lora_add',
                      'chunk_tokens': CHUNK_TOKENS, 'full_sequence_retained': True, 'weights_unchanged': True}
    previous = LoRAHotLoadMixin.lora_forward
    atomic = core.atomic_json
    def audited_json(path, value):
        if isinstance(value, dict): value = {**value, 'runtime_lora_implementation': implementation}
        return atomic(path, value)
    LoRAHotLoadMixin.lora_forward = chunked_lora
    core.atomic_json = audited_json
    try:
        core.run(args)
        atomic(args.out / 'lora_chunk_runtime_verified.json', {**implementation, 'complete': True, 'actual_call_counts': COUNTERS})
    finally:
        LoRAHotLoadMixin.lora_forward = previous
        core.atomic_json = atomic

if __name__ == '__main__': main()
