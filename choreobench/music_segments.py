"""Predicted-bar-aware whole-window observations for an untimed captioner.

The native model still reads 5.0625 seconds. Downbeats anchor observation starts,
not real motion boundaries. Ownership partitions cover the song without overlap.
No global BPM extrapolation, audio stretching, padding, or fake timestamps.
"""
import argparse
import bisect
import copy
import json
import re
from fractions import Fraction
from pathlib import Path

from .audio import make_windows
from .baseline import _Journal, _journal_accounting, _sealed
from .audio import inspect_audio, write_window
from .dataset import resolve_audio
from .io import file_sha, file_lock, object_sha, read_json, read_json_snapshot, write_json
from .providers import create_provider
from .schema import keys, validate_dataset
from .schema import validate_song


UNTIMED_PROMPT = (
    "Listen to this music excerpt and propose one coherent possible solo choreography phrase. "
    "Describe only human movements, their sequence and simultaneous coordination, relative energy "
    "and a natural transition. Match the audible music without claiming unique correct choreography. "
    "Use concise connected movement text, not a list of isolated poses. Do not mention clothing, "
    "appearance, scenery, dance genre labels, numeric times or timestamps. Do not treat the excerpt "
    "cutoff as evidence that the full song ends. Return a single English paragraph without headers."
)


def _length(rate):
    return (81 * rate + 8) // 16


def make_music_observations(song, mode="predicted_downbeat", max_shift_seconds=1.5):
    validate_song(song)
    if mode not in ("fixed", "predicted_downbeat"):
        raise ValueError("Unknown observation segmentation mode")
    shift = Fraction(str(max_shift_seconds)) * song["audio"]["sample_rate"]
    if not 0 <= shift <= 3 * song["audio"]["sample_rate"]:
        raise ValueError("max_shift_seconds must be in [0,3]")
    shift = shift.numerator // shift.denominator
    rate, count = song["audio"]["sample_rate"], song["audio"]["num_samples"]
    length = _length(rate)
    original = make_windows(count, rate)
    tail = count - length
    beats = song["music"]["downbeats"]
    starts, reasons = [0], ["canonical_origin"]
    if mode == "fixed":
        starts = [w["observation_start"] for w in original]
        reasons = ["fixed_seconds" for _ in starts]
    else:
        while starts[-1] < tail:
            ideal = min(starts[-1] + length, tail)
            if ideal == tail:
                next_start, reason = tail, "real_full_window_tail"
            else:
                # Only backward anchoring keeps neighboring complete observations
                # connected. Variable tempo uses actual predicted downbeats.
                candidates = beats[bisect.bisect_left(beats, max(starts[-1]+1, ideal-shift)):
                                   bisect.bisect_right(beats, ideal)]
                if candidates:
                    next_start, reason = candidates[-1], "predicted_downbeat_backward_anchor"
                else:
                    next_start, reason = ideal, "no_nearby_downbeat_time_fallback"
            starts.append(next_start); reasons.append(reason)
    windows = [dict(index=i, observation_start=start, observation_end=start+length,
        ownership_start=start, ownership_end=starts[i+1] if i+1<len(starts) else count,
        nominal_window_seconds=5.0625, sample_rounding_error=length-5.0625*rate,
        boundaries_are_action_boundaries=False, boundary_source=reasons[i],
        predicted_downbeat_at_start=start in beats,
        ownership_is_proposed_context_not_action_duration=True) for i,start in enumerate(starts)]
    return dict(schema="choreobench.untimed_observations.v1", song_id=song["song_id"],
        audio_sha256=song["audio"]["sha256"], sample_rate=rate, num_samples=count, music_sha256=object_sha(song["music"]),
        mode=mode, max_shift_seconds=max_shift_seconds, windows=windows,
        predicted_meter_verified=False, whole_window_caption_has_no_measured_action_times=True,
        native_audio_length_unchanged=True, trained_untimed_captioner_loaded=False)


def paired_observation(song, target_sample, max_shift_seconds=1.5):
    """A same-seed local test of start anchoring, not of whole-song coverage."""
    validate_song(song)
    rate,count=song["audio"]["sample_rate"],song["audio"]["num_samples"]
    if type(target_sample) is not int or not 0<=target_sample<count:
        raise ValueError("Probe target outside song")
    original=make_windows(count,rate)
    fixed=min(original,key=lambda w:(abs(w["observation_start"]-target_sample),w["observation_start"]))
    length=_length(rate);start=fixed["observation_start"];tail=count-length
    shift=Fraction(str(max_shift_seconds))*rate
    if not 0<=shift<=3*rate:raise ValueError("Shift exceeds bound")
    margin=shift.numerator//shift.denominator
    candidates=[b for b in song["music"]["downbeats"] if max(0,start-margin)<=b<=min(tail,start+margin)]
    anchored=min(candidates,key=lambda b:(abs(b-start),b)) if candidates else start
    return dict(song_id=song["song_id"],target_sample=target_sample,audio_sha256=song["audio"]["sha256"],
        observation_seconds=5.0625, fixed=dict(observation_start=start,observation_end=start+length),
        predicted_downbeat=dict(observation_start=anchored,observation_end=anchored+length),
        start_shift_samples=anchored-start,changed=anchored!=start,
        boundary_source="predicted_downbeat" if candidates else "time_fallback",
        meter_verified=False,caption_output_times_inferred=False)


def run_untimed_probe(inventory_path, provider_spec, output_dir):
    """Six native wave-only calls at most; no inferred caption timestamps."""
    inventory_path, output = Path(inventory_path).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(str(output))
    inventory, inv_sha = read_json_snapshot(inventory_path)
    keys(inventory, ("schema", "cases", "seed"), where="untimed probe inventory")
    if (inventory["schema"] != "choreobench.untimed_segment_probe.v1" or not isinstance(inventory["cases"], list)
            or not 1 <= len(inventory["cases"]) <= 3 or type(inventory["seed"]) is not int
            or not 0 <= inventory["seed"] < 2**63):
        raise ValueError("Untimed pilot requires one to three cases and a fixed valid seed")
    sources, prepared = {str(inventory_path): inv_sha}, []
    seen = set()
    for entry in inventory["cases"]:
        keys(entry, ("dataset_path", "dataset_sha256", "song_id", "target_sample"), where="untimed probe case")
        path = Path(entry["dataset_path"]).resolve()
        dataset, sha = read_json_snapshot(path)
        validate_dataset(dataset)
        if sha != entry["dataset_sha256"]:
            raise ValueError("Untimed probe dataset identity changed")
        sources[str(path)] = sha
        song = next(s for s in dataset["songs"] if s["song_id"] == entry["song_id"])
        if song["song_id"] in seen:
            raise ValueError("One paired probe per song")
        seen.add(song["song_id"])
        audio = inspect_audio(resolve_audio(song, path))
        if any(audio[k] != song["audio"][k] for k in ("sha256", "sample_rate", "num_samples")):
            raise ValueError("Untimed probe canonical waveform differs")
        sources[audio["path"]] = audio["sha256"]
        pair = paired_observation(song, entry["target_sample"])
        prepared.append((song, audio, pair, dataset["synthetic"]))
    provider = create_provider(provider_spec, role="local", base_dir=inventory_path.parent)
    if provider.identity.get("kind") != "native" or provider.identity.get("task_prompt") != UNTIMED_PROMPT:
        raise ValueError("Untimed placeholder pilot requires explicit base-only native task prompt")
    def unchanged():
        for path, sha in sources.items():
            if file_sha(path) != sha:
                raise ValueError("Untimed source bytes changed")
    unchanged()
    contract = dict(schema="choreobench.untimed_segment_contract.v1", sources=sources,
        provider_identity=copy.deepcopy(provider.identity), prompt=UNTIMED_PROMPT, seed=inventory["seed"],
        code_sha256={p.name: file_sha(p) for p in Path(__file__).parent.glob("*.py")},
        maximum_new_calls=2*len(prepared), trained_untimed_captioner_loaded=False, placeholder_base_omni=True,
        whole_window_output_not_timestamped=True, predictions_not_motion_truth=True,
        synthetic=any(synthetic for _, _, _, synthetic in prepared) or provider.identity.get("synthetic", True))
    output.mkdir(parents=True, exist_ok=False)
    with file_lock(output / ".run.lock"):
        write_json(output / "contract.json", contract)
        journal = _Journal(output, object_sha(contract), contract["maximum_new_calls"])
        rows = []
        fatal = False
        for song, audio, pair, _ in prepared:
            sid = song["song_id"]
            write_json(output / sid / "paired_windows.json", pair)
            for mode in ("fixed", "predicted_downbeat"):
                interval = pair[mode]
                derived = write_window(audio, interval, output / sid / mode / "audio.wav")
                request = dict(call_id=sid+"/untimed-segment/"+mode, seed=inventory["seed"],
                    max_output_tokens=512, response_kind="local_caption", audio=derived)
                row = dict(song_id=sid, mode=mode, observation_start=interval["observation_start"],
                    observation_end=interval["observation_end"], changed_anchor=pair["changed"], quality_verified=False)
                try:
                    unchanged()
                    response = journal.call(provider, request)
                    if provider.identity != contract["provider_identity"]:
                        raise ValueError("Native provider identity drift")
                    value = response["text"]
                    timed = bool(re.search(r'\[\s*\d+(?:\.\d+)?\s*[-–]|\b\d+(?:\.\d+)?\s*(?:seconds?|samples?|ms)\b', value, re.I))
                    row.update(status="complete", text=value, words=len(value.split()), timestamp_format_violation=timed,
                               untimed_format_verified=not timed, audio_sha256=derived["sha256"])
                except (ValueError, TypeError, KeyError) as error:
                    row.update(status="failed", error=str(error), retained_in_denominator=True)
                    raw=journal.records.get(request["call_id"])
                    if provider.identity != contract["provider_identity"] or not raw or not raw.get("success") or raw.get("result", {}).get("stop_reason") in (
                            None,"http_error","infrastructure_error","malformed_api_response"):
                        fatal=True
                write_json(output / sid / mode / "result.json", row)
                rows.append(row)
                if fatal:break
            if fatal:break
        unchanged()
        result=_sealed(dict(schema="choreobench.untimed_segment_result.v1",contract_sha256=object_sha(contract),rows=rows,
            complete=not fatal and len(rows)==contract["maximum_new_calls"],calls_used=len(journal.started),
            call_records={k:r["record_sha256"] for k,r in journal.records.items()},**_journal_accounting(journal),
            trained_untimed_captioner_loaded=False,new_text_planner_calls=0,quality_verified=False,
            native_waveform_seconds=5.0625,predicted_downbeat_verified=False))
        write_json(output / "results.json",result)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--song-id")
    parser.add_argument("--out")
    parser.add_argument("--probe-inventory")
    parser.add_argument("--provider")
    parser.add_argument("--out-dir")
    parser.add_argument("--mode",choices=["fixed","predicted_downbeat"],default="predicted_downbeat")
    args=parser.parse_args(argv)
    if args.probe_inventory is not None:
        if args.provider is None or args.out_dir is None or args.dataset is not None:
            parser.error("Probe requires --provider and --out-dir, without --dataset")
        result=run_untimed_probe(args.probe_inventory,read_json(args.provider),args.out_dir)
        print(json.dumps(dict(complete=result["complete"],calls_used=result["calls_used"],quality_verified=False)))
        return
    if args.dataset is None or args.song_id is None or args.out is None:
        parser.error("Preparation requires --dataset, --song-id and --out")
    dataset=read_json(args.dataset)
    song=next(s for s in dataset["songs"] if s["song_id"]==args.song_id)
    result=make_music_observations(song,args.mode)
    write_json(args.out,result)
    print("Prepared %d whole-window untimed observations; no model calls"%len(result["windows"]))


if __name__=="__main__":main()
