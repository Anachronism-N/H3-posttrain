"""Per-GPU worker for the 9 fixed H3 jobs: verbatim prompt, image reference, pure video output."""
import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

import diffsynth
from diffsynth.pipelines.minimax_h3_audio_video import ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio
from h3_video_only import build_video_only_pipeline


def log(message):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), message), flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_oss(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(dst) + ".partial")
    shutil.copyfile(src, partial)
    os.replace(partial, dst)


def build_pipeline(weights, margin_gb):
    vram_config = {"offload_dtype": torch.bfloat16, "offload_device": "cpu",
                   "onload_dtype": torch.bfloat16, "onload_device": "cpu",
                   "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
                   "computation_dtype": torch.bfloat16, "computation_device": "cuda"}
    transformer = sorted(str(p) for p in (weights / "transformer").glob("model*.safetensors"))
    text_encoder = sorted(str(p) for p in (weights / "text_encoder").glob("model*.safetensors"))
    vae = [str(weights / "video_vae" / "source" / "model.safetensors")]
    for name, paths in (("transformer", transformer), ("text_encoder", text_encoder), ("video_vae", vae)):
        if not paths or not all(os.path.isfile(p) for p in paths):
            raise FileNotFoundError("missing %s weights under %s" % (name, weights))
    total_gb = torch.cuda.mem_get_info("cuda")[1] / 1024 ** 3
    t0 = time.time()
    pipe = build_video_only_pipeline(
        weights, device="cuda", torch_dtype=torch.bfloat16, load_text_encoder=True, load_mode="all",
        model_configs=[ModelConfig(path=transformer, **vram_config),
                       ModelConfig(path=vae, **vram_config),
                       ModelConfig(path=text_encoder, **vram_config)],
        vram_management=True, vram_limit=total_gb - margin_gb)
    if pipe.audio_vae is not None or pipe.scheduler_audio is not None:
        raise RuntimeError("video-only pipeline loaded audio components")
    info = {"load_s": round(time.time() - t0, 1), "vram_limit_gib": round(total_gb - margin_gb, 1),
            "vram_management": True,
            "weights": {"transformer": transformer, "text_encoder_count": len(text_encoder), "video_vae": vae},
            "gpu": torch.cuda.get_device_name(0), "diffsynth": diffsynth.__file__}
    return pipe, info


def load_lora(pipe, path, alpha):
    import re
    from safetensors import safe_open
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("missing lora %s" % path)
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        keys = list(stream.keys())
    pattern = re.compile(r"^(?:token_refiner\.)?blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))"
                         r"\.lora_[AB]\.default\.weight$")
    bad = [key for key in keys if not pattern.fullmatch(key)]
    if not keys or bad:
        raise ValueError("unexpected native H3 LoRA keys: %r" % bad[:5])
    for key in keys:
        pair = key.replace(".lora_A.", ".lora_B.") if ".lora_A." in key else key.replace(".lora_B.", ".lora_A.")
        if pair not in keys:
            raise ValueError("incomplete LoRA pair for %s" % key)
    pipe.load_lora(pipe.dit, lora_config=str(path), alpha=alpha)
    return {"file": str(path), "keys": len(keys), "alpha": alpha,
            "sha256": sha256_file(path), "bytes": path.stat().st_size}


def instrument(pipe, timings, stats):
    original_dit = pipe.dit.forward

    def dit_forward(*args, **kwargs):
        torch.cuda.synchronize()
        started = time.time()
        out = original_dit(*args, **kwargs)
        torch.cuda.synchronize()
        timings["dit_s"].append(time.time() - started)
        return out
    pipe.dit.forward = dit_forward

    for name in ("encode_video", "decode_video"):
        original = getattr(pipe.video_vae, name)

        def wrapped(*args, __original=original, __name=name, **kwargs):
            torch.cuda.synchronize()
            started = time.time()
            out = __original(*args, **kwargs)
            torch.cuda.synchronize()
            timings["vae_%s_s" % __name].append(time.time() - started)
            return out
        setattr(pipe.video_vae, name, wrapped)

    if isinstance(pipe.text_encoder, torch.nn.Module):
        def pre_hook(module, inputs):
            torch.cuda.synchronize()
            module._t0 = time.time()

        def post_hook(module, inputs, output):
            torch.cuda.synchronize()
            timings["text_encoder_s"].append(time.time() - module._t0)
        pipe.text_encoder.register_forward_pre_hook(pre_hook)
        pipe.text_encoder.register_forward_hook(post_hook)

    original_model_fn = pipe.model_fn

    def model_fn(*args, **kwargs):
        packed = kwargs.get("packed") or {}
        if isinstance(packed, dict):
            stats["packed_seq_len"] = int(packed.get("seq_len", 0))
        return original_model_fn(*args, **kwargs)
    pipe.model_fn = model_fn


def frame_stats(frames):
    sample = [np.asarray(frame.convert("L").reduce(4), dtype=np.float32) for frame in frames]
    picks = [sample[0], sample[len(sample) // 2], sample[-1]]
    return {"mean": [round(float(x.mean()), 2) for x in picks],
            "std": [round(float(x.std()), 2) for x in picks],
            "near_constant": any(float(x.std()) < 2.0 for x in picks)}


def run_job(pipe, timings, stats, info, job, out_root, oss_outputs, spec):
    job_id = job["job_id"]
    out_dir = out_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_bytes = Path(job["prompt"]).read_bytes()
    prompt_digest = hashlib.sha256(prompt_bytes).hexdigest()
    if prompt_digest != job["prompt_sha256"]:
        raise RuntimeError("prompt sha256 mismatch: %s != %s" % (prompt_digest, job["prompt_sha256"]))
    prompt = prompt_bytes.decode("utf-8")
    reference_digest = sha256_file(job["reference"])
    if reference_digest != job["reference_sha256"]:
        raise RuntimeError("reference sha256 mismatch")
    generation = job["request"]["generation"]
    if (generation["seed"], generation["width"], generation["height"], generation["num_frames"], generation["steps"],
            float(generation["cfg"])) != (spec["seed"], spec["width"], spec["height"], spec["num_frames"],
                                          spec["steps"], float(spec["cfg"])):
        raise RuntimeError("request.json generation params deviate from the task spec: %r" % generation)
    reference = Image.open(job["reference"]).convert("RGB")
    for values in timings.values():
        values.clear()
    stats.clear()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    def progress(timesteps):
        total = len(timesteps)
        for index, timestep in enumerate(timesteps):
            yield timestep
            log("  %s step %d/%d %.2fs" % (job_id, index + 1, total,
                                           timings["dit_s"][-1] if timings["dit_s"] else -1))
    frames = pipe(prompt=prompt, height=spec["height"], width=spec["width"], num_frames=spec["num_frames"],
                  num_inference_steps=spec["steps"], seed=spec["seed"], cfg_scale=spec["cfg"],
                  flow_shift=spec["flow_shift"], references=[{"type": "image", "image": reference}],
                  progress_bar_cmd=progress)
    generate_s = round(time.time() - started, 1)
    if len(frames) != spec["num_frames"]:
        raise RuntimeError("pipeline returned %d of %d frames" % (len(frames), spec["num_frames"]))
    stats_frame = frame_stats(frames)
    encode_started = time.time()
    output = out_dir / ("%s.mp4" % job_id)
    write_video_audio(video=frames, audio=None, output_path=str(output), fps=spec["fps"], audio_sample_rate=32000)
    encode_s = round(time.time() - encode_started, 1)
    with av.open(str(output)) as container:
        video_streams, audio_streams = len(container.streams.video), len(container.streams.audio)
        rate = str(container.streams.video[0].average_rate)
        decoded, dims = 0, None
        for frame in container.decode(video=0):
            dims = (frame.width, frame.height)
            decoded += 1
    check = {"decoded_frames": decoded, "width": dims[0], "height": dims[1], "fps": rate,
             "video_streams": video_streams, "audio_streams": audio_streams}
    ok = (decoded == spec["num_frames"] and dims == (spec["width"], spec["height"])
          and rate in ("24/1", "24") and video_streams == 1 and audio_streams == 0)
    if not ok:
        raise RuntimeError("output check failed: %r" % check)
    digest = sha256_file(output)
    report = {"job_id": job_id, "case_id": job["case_id"], "variant": job["variant"], "status": "ok",
              "prompt": prompt, "prompt_sha256": prompt_digest, "reference_sha256": reference_digest,
              "params": {"seed": spec["seed"], "width": spec["width"], "height": spec["height"],
                         "num_frames": spec["num_frames"], "num_inference_steps": spec["steps"],
                         "cfg_scale": spec["cfg"], "flow_shift": spec["flow_shift"], "fps": spec["fps"]},
              "weights": info["weights"], "weights_commit": info.get("weights_commit"),
              "lora": info.get("lora"),
              "diffsynth_commit": "974cfa37f27ac55eba3b6d10efa21f876900572d", "gpu": info["gpu"],
              "timings": {"load_s": info["load_s"], "generate_s": generate_s, "encode_s": encode_s,
                          "dit_s": [round(v, 3) for v in timings["dit_s"]],
                          "text_encoder_s": [round(v, 3) for v in timings["text_encoder_s"]],
                          "vae_encode_s": [round(v, 3) for v in timings["vae_encode_video_s"]],
                          "vae_decode_s": [round(v, 3) for v in timings["vae_decode_video_s"]]},
              "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
              "stats": dict(stats, frame_stats=stats_frame),
              "verification": dict(check, near_constant=stats_frame["near_constant"]),
              "output": {"file": output.name, "bytes": output.stat().st_size, "sha256": digest},
              "total_s": round(time.time() - started, 1)}
    (out_dir / ("%s.json" % job_id)).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    copy_oss(output, oss_outputs / output.name)
    copy_oss(out_dir / ("%s.json" % job_id), oss_outputs / ("%s.json" % job_id))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs_json", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--oss_outputs", required=True)
    parser.add_argument("--weights_root", required=True)
    parser.add_argument("--code_root", required=True)
    parser.add_argument("--status_file", required=True)
    parser.add_argument("--gpu_index", type=int, required=True)
    parser.add_argument("--vram_margin_gb", type=float, default=5)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    args = parser.parse_args()
    jobs = json.loads(Path(args.jobs_json).read_text())
    spec = jobs["spec"]
    status_file = Path(args.status_file)

    def write_status(**extra):
        body = {"gpu_index": args.gpu_index, "pid": os.getpid(), "updated_ts": time.time(),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}
        body.update(extra)
        temporary = Path(str(status_file) + ".tmp")
        temporary.write_text(json.dumps(body, indent=1, ensure_ascii=False))
        os.replace(temporary, status_file)
    write_status(state="loading")
    code = Path(args.code_root).resolve()
    if Path(diffsynth.__file__).resolve() != code / "DiffSynth-Studio/diffsynth/__init__.py":
        raise SystemExit("unexpected DiffSynth source %s" % diffsynth.__file__)
    pipe, info = build_pipeline(Path(args.weights_root), args.vram_margin_gb)
    if args.lora:
        info["lora"] = load_lora(pipe, args.lora, args.lora_alpha)
        log("gpu %d lora loaded: %s" % (args.gpu_index, json.dumps({k: info["lora"][k] for k in ("file", "keys", "alpha")})))
    timings = {"dit_s": [], "vae_encode_video_s": [], "vae_decode_video_s": [], "text_encoder_s": []}
    stats = {}
    instrument(pipe, timings, stats)
    log("gpu %d ready: %s" % (args.gpu_index, json.dumps({k: info[k] for k in ("load_s", "gpu", "vram_limit_gib")})))
    failures = {}
    for index, job in enumerate(jobs["jobs"]):
        write_status(state="running", job=job["job_id"], index=index, failures=failures)
        log("gpu %d job %s (%d/%d)" % (args.gpu_index, job["job_id"], index + 1, len(jobs["jobs"])))
        try:
            report = run_job(pipe, timings, stats, info, job, Path(args.out_root), Path(args.oss_outputs), spec)
            log("gpu %d %s ok %.1fs (gen %.1fs, peak %.1f GiB)" % (
                args.gpu_index, job["job_id"], report["total_s"], report["timings"]["generate_s"],
                report["peak_gpu_memory_gib"]))
        except Exception as error:
            failures[job["job_id"]] = "%s: %s" % (type(error).__name__, str(error)[:500])
            log("gpu %d %s FAILED: %s\n%s" % (args.gpu_index, job["job_id"], failures[job["job_id"]],
                                              traceback.format_exc()[-1500:]))
            prompt_head = Path(job["prompt"]).read_text(encoding="utf-8")[:200]
            failure = {"job_id": job["job_id"], "status": "fail", "error": failures[job["job_id"]],
                       "traceback": traceback.format_exc()[-4000:], "prompt_head": prompt_head,
                       "params": spec, "gpu": info.get("gpu")}
            target = Path(args.oss_outputs)
            target.mkdir(parents=True, exist_ok=True)
            (target / ("%s.json" % job["job_id"])).write_text(json.dumps(failure, indent=1, ensure_ascii=False))
        write_status(state="running", job=None, index=index + 1, failures=failures)
    write_status(state="finished", failures=failures, info=info)
    log("gpu %d finished: %d jobs, %d failed" % (args.gpu_index, len(jobs["jobs"]), len(failures)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
