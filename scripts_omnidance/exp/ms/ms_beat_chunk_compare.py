"""Independent detected-beat C vs matched original-MS A/B, CPU only.

Music beats are an uncalibrated onset/DP estimate, not bar ground truth or
observed action boundaries. Librosa runs in an isolated subprocess; original
MS uses whole-clip v2 coordinates, never per-window coordinate resets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

SCHEMA = "motionscript_detected_beat_C_compare_v1"
ESTIMATOR = {"method": "audio_feats.beat_dp_on_librosa_onset_strength", "tempo_range_bpm": [60, 180],
             "tempo_prior_bpm": 120, "tightness": 100., "sample_rate": 22050, "hop_samples": 512,
             "calibrated": False, "bar_or_downbeat_detector": False}
CONTEXT = None


def audio_estimate(path):
    # Do not import torch here: the existing audio_feats contract isolates it.
    import librosa
    import soundfile as sf
    from scipy.signal import resample_poly, find_peaks
    from audio_feats import beat_dp
    content = Path(path).read_bytes()
    audio, rate = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)
    if len(audio) == 0 or rate <= 0 or not np.isfinite(audio).all():
        raise ValueError("WAV must contain finite nonempty decoded samples")
    mono = audio.mean(axis=1)
    divisor = math.gcd(int(rate), 22050)
    y = resample_poly(mono, 22050 // divisor, int(rate) // divisor)
    onset = librosa.onset.onset_strength(y=y, sr=22050, hop_length=512)
    peaks, _ = find_peaks(onset, height=float(onset.mean() + .5 * onset.std()), distance=3)
    bpm, beats = beat_dp(onset, 22050, hop=512)
    stamps = np.arange(len(onset)) * 512 / 22050
    strengths = np.interp(beats, stamps, onset)
    peak_times = peaks * 512 / 22050
    alignment = float(np.mean([bool(len(peaks)) and np.min(np.abs(peak_times - beat)) <= .10 for beat in beats])) if beats else 0.
    return {"audio_sha256": hashlib.sha256(content).hexdigest(), "decoded_sample_count": len(audio),
            "native_sample_rate": int(rate), "native_channels": audio.shape[1], "duration_seconds": len(audio) / rate,
            "estimator": ESTIMATOR, "bpm_estimate": float(bpm), "detected_beats_seconds": beats,
            "rms": float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))),
            "onset_std": float(onset.std()), "onset_max": float(onset.max()),
            "onset_peak_times_seconds": peak_times.tolist(), "detected_beat_onset_strength": strengths.tolist(),
            "beat_peak_alignment_fraction": alignment,
            "librosa_version": librosa.__version__, "soundfile_version": sf.__version__,
            "reliability_is_calibrated_confidence": False}


def choose_windows(estimate, media_end, *, beats_per_chunk=4, min_seconds=1.5, max_seconds=4.):
    from motionscript_v2.chunks import fixed_windows
    if beats_per_chunk not in (2, 4, 8) or not (0 < min_seconds <= max_seconds):
        raise ValueError("beat count must be 2/4/8 and duration limits positive/ordered")
    beats = np.asarray(estimate["detected_beats_seconds"], dtype=float)
    if not np.isfinite(beats).all() or (np.diff(beats) <= 0).any():
        raise ValueError("detected beat timestamps must be finite/strictly increasing")
    original_indices = np.flatnonzero((beats > 0) & (beats < media_end))
    beats = beats[original_indices]
    reasons = []
    if estimate["rms"] < 1e-5: reasons.append("near_silent_audio")
    if estimate["onset_std"] < 1e-3 or estimate["onset_max"] <= 0: reasons.append("weak_onset_envelope")
    if len(beats) < 4: reasons.append("insufficient_detected_beats")
    if len(beats) and (beats[0] > 1.5 or media_end - beats[-1] > 1.5): reasons.append("insufficient_estimated_beat_coverage")
    intervals = np.diff(beats)
    cv = float(intervals.std() / intervals.mean()) if len(intervals) else None
    if cv is not None and cv > .25: reasons.append("irregular_estimated_beat_intervals")
    if estimate["beat_peak_alignment_fraction"] < .4: reasons.append("weak_detected_beat_to_onset_peak_alignment")
    reliability = {"heuristic": "energy/onset/coverage/interval/peak-alignment screen; not calibrated beat accuracy",
                   "interval_cv": cv, "minimum_peak_alignment": .4, "maximum_interval_cv": .25,
                   "minimum_beats": 4, "maximum_uncovered_edge_seconds": 1.5, "reasons": reasons}
    windows = []
    if not reasons:
        previous, anchor = 0., -1
        choices = [beats_per_chunk, *[number for number in (4, 2, 8) if number != beats_per_chunk]]
        while previous < media_end:
            options = []
            for count in choices:
                index = anchor + count
                if 0 <= index < len(beats):
                    boundary = float(beats[index])
                    if min_seconds <= boundary - previous <= max_seconds and media_end - boundary >= min_seconds:
                        options.append((choices.index(count), index, boundary, count))
            if options:
                _, index, boundary, count = min(options)
                windows.append({"start": previous, "end": boundary, "detected_boundary_beat_index": int(original_indices[index]),
                                "beat_steps_from_previous_anchor": count, "first_edge_is_media_start": previous == 0})
                previous, anchor = boundary, index
            else:
                if media_end - previous > max_seconds:
                    # All internal boundaries must remain actual detected beats;
                    # never invent a partial beat or global-BPM time grid.
                    reasons.append("cannot_meet_duration_limits_on_detected_boundaries")
                    break
                windows.append({"start": previous, "end": media_end, "detected_boundary_beat_index": None,
                                "last_edge_is_actual_media_end": True, "short_clip_or_tail_merged": media_end - previous < min_seconds})
                break
    if reasons:
        windows = fixed_windows(0, media_end, 2., 1.5)
    return {"strategy": "fixed_fallback" if reasons else "detected_beats", "windows": windows,
            "fallback_reason": reasons or None, "beats_per_chunk_requested": beats_per_chunk,
            "optional_beat_counts": [2, 4, 8], "duration_limits_seconds": [min_seconds, max_seconds],
            "detected_internal_boundaries_seconds": [window["end"] for window in windows[:-1]] if not reasons else [],
            "reliability": reliability, "window_boundaries_are_action_boundaries": False,
            "bars_or_downbeats_verified": False}


def word_stats(texts):
    words_total = fourgrams = extra = 0
    counts = []
    for text in texts:
        words = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", text.casefold())
        counts.append(len(words));words_total += len(words)
        grams = Counter(tuple(words[i:i+4]) for i in range(max(0, len(words)-3)))
        fourgrams += sum(grams.values());extra += sum(value-1 for value in grams.values())
    return {"caption_denominator": len(counts), "nonempty_count": sum(count > 0 for count in counts),
            "total_words": words_total, "mean_words_all": words_total / len(counts) if counts else 0.,
            "fourgram_occurrences": fourgrams, "extra_repeated_fourgrams_within_caption": extra,
            "repeat_fourgram_micro_fraction": extra / fourgrams if fourgrams else 0.,
            "definition": "token regex words; no timing headers; repeated count-1 within each individual caption/window, no cross-window concatenation"}


def initialize(context):
    global CONTEXT
    CONTEXT = context
    from ms_chunk_caption import init_worker
    init_worker(context)


def run_one(row):
    from ms_optimize import atomic_write, json_bytes, sha256, save_npz
    from ms_chunk_caption import media_bounds, describe_windows
    from motionscript_v2.coordinates import preprocess
    from motionscript_v2.chunks import fixed_windows
    sid, ctx = row["id"], CONTEXT
    out = Path(ctx["out"])
    try:
        for key in ("nlf_path", "video_path", "audio_path"):
            if not Path(row[key]).is_file(): raise ValueError("missing consumed " + key)
        source = Path(row["nlf_path"]).read_bytes()
        input_sha = hashlib.sha256(source).hexdigest()
        video_sha = sha256(Path(row["video_path"]))
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--estimate-audio", row["audio_path"]],
                                capture_output=True, text=True, check=True, timeout=180,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        audio = json.loads(result.stdout)
        with np.load(io.BytesIO(source), allow_pickle=False) as archive:
            nlf = {key: archive[key] for key in archive.files}
        end, media = media_bounds(row, nlf, ffprobe=ctx["ffprobe"])
        if abs(audio["duration_seconds"] - end) > 1 / audio["native_sample_rate"]:
            raise ValueError("decoded WAV length differs from actual normalized video duration")
        baseline_path = Path(ctx["baseline"]) / f"{sid}.json"
        baseline_bytes = baseline_path.read_bytes()
        receipt = json.loads(baseline_path.with_suffix(".receipt.json").read_bytes())
        if receipt["report_sha256"] != hashlib.sha256(baseline_bytes).hexdigest(): raise ValueError("altered A/B baseline report")
        baseline = json.loads(baseline_bytes)
        if any(baseline[key] != value for key, value in (("input_sha256",input_sha),("video_sha256",video_sha),("audio_sha256",audio["audio_sha256"]))):
            raise ValueError("C source bytes differ from matched A/B")
        if baseline["source_group"] != row["source_group"] or baseline["split"] != row["split"]: raise ValueError("A/B source_group or split differs")
        arrays, quality = preprocess(nlf, ctx["template"], fps=20., max_gap_seconds=.2, smooth_seconds=.0625)
        plan = choose_windows(audio, end, beats_per_chunk=ctx["beats_per_chunk"], min_seconds=ctx["min_seconds"], max_seconds=ctx["max_seconds"])
        captions = describe_windows(arrays, plan["windows"], Path(ctx["src"]), Path(ctx["shim"]) if ctx["shim"] else None)
        coords = out / f"{sid}.coords.npz"
        save_npz(coords, arrays)
        # Comparison-only tempo grid: not detected beats and never source C.
        target = 60 * ctx["beats_per_chunk"] / audio["bpm_estimate"]
        report = {"schema": SCHEMA, "id": sid, "fingerprint": ctx["fingerprint"], "source_group": row["source_group"], "split": row["split"],
                  "manifest_record": row, "source": "original_MS_on_whole_v2_coordinates_with_audio_estimated_beat_windows",
                  "input_sha256": input_sha, "video_sha256": video_sha, "audio_sha256": audio["audio_sha256"],
                  "audio": audio, "media": media, "plan": plan, "C_windows": captions,
                  "coords_file": coords.name, "coords_sha256": sha256(coords), "quality": quality,
                  "baseline_report_path": str(baseline_path), "baseline_report_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
                  "A_whole_clip": baseline["whole_clip_original"], "B_fixed_windows": baseline["chunks"],
                  "tempo_grid_comparison_only": {"strategy": "tempo_grid_not_detected_beats", "target_seconds": target,
                                                 "windows": fixed_windows(0, end, target, 1.5), "used_as_C": False},
                  "audio_video_time_assumption": "normalized WAV t=0 aligns normalized video t=0; original-source/audio offsets unknown, not independently verified",
                  "audio_video_alignment_verified": False, "motions_on_beats_verified": False,
                  "accepted": False, "formal_approval": False, "semantic_approval": False, "human_review_completed": False}
        payload = json_bytes(report)
        atomic_write(out / f"{sid}.receipt.json", json_bytes({"report_sha256": hashlib.sha256(payload).hexdigest()}))
        atomic_write(out / f"{sid}.json", payload)
        return report, None
    except Exception as error:
        failed = {"id": sid, "error_type": type(error).__name__, "error": str(error), "formal_approval": False}
        atomic_write(out / f"{sid}.error.json", json_bytes(failed))
        return None, failed


def run(args):
    from ms_chunk_caption import load_manifest
    from ms_optimize import atomic_write, json_bytes, overlaps, sha256, source_hashes, upstream_hashes
    if args.limit < 0 or args.workers < 1: raise ValueError("invalid limit/workers")
    choose_windows({"detected_beats_seconds":[], "rms":0., "onset_std":0., "onset_max":0., "beat_peak_alignment_fraction":0.}, 5.,
                   beats_per_chunk=args.beats_per_chunk, min_seconds=args.min_seconds, max_seconds=args.max_seconds)
    manifest = Path(args.manifest).resolve()
    rows, manifest_sha = load_manifest(manifest)
    rows = rows[:args.limit] if args.limit else rows
    out, baseline = Path(args.out).resolve(), Path(args.baseline).resolve()
    if out.exists(): raise ValueError("C output must be a new independent directory")
    for path in (baseline, manifest.parent, Path(args.template).resolve().parent, Path(args.upstream_src).resolve()):
        if overlaps(out, path): raise ValueError("C output must be separate from source/baseline directories")
    for row in rows:
        for key in ("nlf_path","video_path","audio_path"):
            if overlaps(out, Path(row[key]).parent): raise ValueError("C output must be separate from media/NLF")
    template_bytes = Path(args.template).read_bytes()
    with np.load(io.BytesIO(template_bytes), allow_pickle=False) as archive:
        template = {key: archive[key] for key in archive.files}
    base_manifest = json.loads((baseline / "manifest.json").read_bytes())
    base_ids = [record["id"] for record in base_manifest["records"]]
    if [row["id"] for row in rows] != base_ids[:len(rows)]: raise ValueError("C selection is not the same ordered A/B prefix")
    config = {key: getattr(args,key) for key in ("workers","limit","beats_per_chunk","min_seconds","max_seconds","ffprobe")}
    here = Path(__file__).resolve().parent
    protocol = {"schema":SCHEMA,"manifest_sha256":manifest_sha,"selected_ids":[row["id"] for row in rows],"config":config,"estimator":ESTIMATOR,
                "code_sha256":{**source_hashes(),Path(__file__).name:sha256(Path(__file__)),"audio_feats.py":sha256(here/"audio_feats.py"),
                               "ms_chunk_caption.py":sha256(here/"ms_chunk_caption.py"),"ms_landmarks.py":sha256(here/"ms_landmarks.py")},
                "upstream_sha256":upstream_hashes(Path(args.upstream_src),Path(args.upstream_shim) if args.upstream_shim else None),
                "template_sha256":hashlib.sha256(template_bytes).hexdigest(),"baseline_manifest_sha256":sha256(baseline/"manifest.json"),
                "formal_approval":False}
    fingerprint = hashlib.sha256(json_bytes(protocol)).hexdigest()
    ctx = {"out":str(out),"baseline":str(baseline),"src":str(Path(args.upstream_src).resolve()),"shim":str(Path(args.upstream_shim).resolve()) if args.upstream_shim else None,
           "template":template,"fingerprint":fingerprint,**config}
    initialize(ctx)
    out.mkdir()
    atomic_write(out/"protocol.json",json_bytes({"fingerprint":fingerprint,"protocol":protocol}))
    reports,errors=[],[]
    if args.workers==1:
        iterator=map(run_one,rows);pool=None
    else:
        pool=mp.get_context("spawn").Pool(args.workers,initializer=initialize,initargs=(ctx,));iterator=pool.imap_unordered(run_one,rows)
    try:
        for report,error in iterator:
            if error: errors.append(error);print(json.dumps(error),flush=True)
            else: reports.append(report);print(json.dumps({"id":report["id"],"strategy":report["plan"]["strategy"],"C_windows":len(report["C_windows"])}),flush=True)
    finally:
        if pool:pool.close();pool.join()
    order={row["id"]:index for index,row in enumerate(rows)}
    reports.sort(key=lambda row:order[row["id"]]);errors.sort(key=lambda row:order[row["id"]])
    stats={}
    for mode,key in (("A","A_whole_clip"),("B","B_fixed_windows"),("C","C_windows")):
        views=[report[key] for report in reports] if mode=="A" else [view for report in reports for view in report[key]]
        raw=lambda view:" ".join(text for segment in view["original_captions"] for text in segment["ms"] if text.strip())
        stats[mode]={"source_clip_denominator":len(reports),"view_denominator":len(views),"status_counts":dict(Counter(view["status"] for view in views)),
                     "abstain_reasons":dict(Counter(view["abstain_reason"] for view in views if view["status"]=="abstain")),
                     "raw_text":word_stats([raw(view) for view in views]),
                     "all_windows_candidate_clip_count":sum(all(view["status"]=="candidate" for view in ([report[key]] if mode=="A" else report[key])) for report in reports)}
    cviews=[view for report in reports for view in report["C_windows"]]
    summary={"schema":SCHEMA,"fingerprint":fingerprint,"selected_count":len(rows),"processed_count":len(reports),"failed_count":len(errors),"errors":errors,
             "complete":not errors,"formal_approval":False,"accepted":False,"semantic_approval":False,"human_review_completed":False,
             "strategy_counts":dict(Counter(report["plan"]["strategy"] for report in reports)),
             "fallback_reasons":dict(Counter(reason for report in reports for reason in (report["plan"]["fallback_reason"] or []))),
             "bpm_mean":float(np.mean([report["audio"]["bpm_estimate"] for report in reports])) if reports else None,
             "C_duration_mean":float(np.mean([view["window_end"]-view["window_start"] for view in cviews])) if cviews else None,
             "A_B_C_same_processed_source_stats":stats,"tempo_grid_is_not_C":True,"beat_boundaries_are_verified_action_boundaries":False,
             "scope":"finite matched input only; onset/DP beats not calibrated/bar annotated; no semantic acceptance or training approval",
             "records":[{"id":report["id"],"report_sha256":sha256(out/(report["id"]+".json")),"coords_sha256":report["coords_sha256"]} for report in reports]}
    chunks=b"".join((json.dumps({"id":report["id"],"strategy":report["plan"]["strategy"],"audio_sha256":report["audio_sha256"],**view},ensure_ascii=False,allow_nan=False)+"\n").encode("utf-8") for report in reports for view in report["C_windows"])
    atomic_write(out/"chunks.jsonl",chunks)
    payload=json_bytes(summary)
    atomic_write(out/"summary.receipt.json",json_bytes({"summary_sha256":hashlib.sha256(payload).hexdigest(),"chunks_sha256":hashlib.sha256(chunks).hexdigest()}))
    atomic_write(out/"summary.json",payload)
    return summary


def parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",required=True);ap.add_argument("--baseline",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--template",required=True);ap.add_argument("--upstream-src",required=True);ap.add_argument("--upstream-shim")
    ap.add_argument("--ffprobe",default="/opt/ms_tools/ffprobe");ap.add_argument("--workers",type=int,default=1);ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--beats-per-chunk",type=int,choices=(2,4,8),default=4);ap.add_argument("--min-seconds",type=float,default=1.5);ap.add_argument("--max-seconds",type=float,default=4.)
    return ap


if __name__=="__main__":
    try:
        if len(sys.argv)==3 and sys.argv[1]=="--estimate-audio":
            with contextlib.redirect_stdout(io.StringIO()):estimate=audio_estimate(sys.argv[2])
            print(json.dumps(estimate,allow_nan=False))
        else:
            summary=run(parser().parse_args());print(json.dumps(summary,ensure_ascii=False),flush=True);sys.exit(0 if summary["complete"] else 1)
    except Exception as error:
        print(type(error).__name__+": "+str(error),file=sys.stderr);sys.exit(1)
