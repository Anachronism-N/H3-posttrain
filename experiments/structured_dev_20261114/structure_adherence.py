#!/usr/bin/env python3
"""Does Module S actually use the acoustic summary?

Generates the whole-song outline twice for the same song: once with the real
   summary and once with a shuffled one (block evidence cyclically shifted, energies/onsets
swapped, relations dropped). Then compares

- the returned per-block guidance (identical output would mean the summary is
  not driving anything), and
- agreement between each unit's declared quantity_change and the acoustic
  direction of the blocks that overlap it.

This is an implementation probe, not quality evidence.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from choreobench import composition_music as cm                      # noqa: E402
from choreobench.composition_workflows import _source                 # noqa: E402
from choreobench.io import read_json                                  # noqa: E402
from choreobench.music_structure_summary import load as load_summary   # noqa: E402
from choreobench.providers import create_provider                      # noqa: E402

RISE_DB = 1.5


def cfg():
    return dict(seed=42, max_output_tokens=2048, outline_max_output_tokens=8192,
                max_request_characters=300000, max_events_per_unit=3, structured_unit_output=True,
                use_outline=True, use_history=True, draft_revision_limit=1, event_time_format="boundary_ids",
                event_grid_seconds=.25, min_event_seconds=.5, event_schedule_policy="balanced_full_unit",
                unit_config=dict(bars_per_block=4, fallback_seconds=10.125, max_unit_seconds=20,
                                 meter_status="unverified", min_time_block_seconds=2),
                outline_proposal_char_limit=160, outline_candidates_per_window=1, max_calls=1)


def shuffled(summary):
    value = copy.deepcopy(summary)
    blocks = value["blocks"]
    permutation = [(index + 1) % len(blocks) for index in range(len(blocks))]
    for index, block in enumerate(blocks):
        mirror = summary["blocks"][permutation[index]]
        block["relative_energy_db"], block["onset_rate_per_second"] = (
            mirror["relative_energy_db"], mirror["onset_rate_per_second"])
        block["energy_trend"] = mirror["energy_trend"]
    value["relations"] = []
    value["_control_shuffled"] = True
    value["_control_permutation"] = permutation
    value["_control_identifiable"] = len(blocks) > 1
    return value


def acoustic_direction(summary, unit):
    """Deterministic reference direction from the blocks overlapping one unit."""
    blocks = [b for b in summary["blocks"] if b["start"] < unit["end"] and b["end"] > unit["start"]]
    if not blocks:
        return "unknown"
    energies = [b["relative_energy_db"] for b in blocks]
    mean = sum(energies) / len(energies)
    if mean >= RISE_DB:
        return "increase"
    if mean <= -RISE_DB:
        return "decrease"
    return "steady"


def probe(provider, source, summary, label):
    request = cm.outline_request(source["song"], source["options"], source["units"], summary, cfg())
    response = provider.generate(request)
    if response.get("stop_reason") not in ("stop", "eos"):
        return dict(label=label, valid=False, error="stop_reason=" + str(response.get("stop_reason")))
    try:
        outline = cm.read_outline(response["text"], source["song"], summary, source["units"])
    except Exception as error:
        return dict(label=label, valid=False, error=str(error))
    guidance = {g["unit_id"]: g for g in outline["block_guidance"]}
    units = source["units"]["units"]
    matches = []
    for unit in units:
        want = acoustic_direction(summary, unit)
        got = guidance[unit["unit_id"]]["quantity_change"]
        if want == "unknown":
            continue
        matches.append((unit["unit_id"], got, want, got == want))
    consistency = (sum(1 for m in matches if m[3]) / len(matches)) if matches else None
    return dict(label=label, valid=True, consistency=consistency, compared=len(matches),
                guidance={k: dict(quantity_change=v["quantity_change"], motif_id=v["motif_id"])
                          for k, v in guidance.items()},
                mismatches=[dict(unit_id=m[0], declared=m[1], acoustic=m[2]) for m in matches if not m[3]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--music-summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = _source(args.source, args.song_id, cfg(), {})
    summary = load_summary(args.music_summary)
    provider = create_provider(read_json(args.provider), "planner", Path(args.out))
    normal = probe(provider, source, summary, "normal")
    control = probe(provider, source, shuffled(summary), "shuffled")
    identical = normal.get("guidance") == control.get("guidance")
    report = dict(schema="choreobench.s_music_adherence.v1", song_id=args.song_id,
                  units=len(source["units"]["units"]), rise_db=RISE_DB,
                  normal=normal, shuffled=control,
                  guidance_identical=identical,
                  note=("If guidance is identical under a shuffled summary, Module S is not using the "
                        "acoustic evidence even though it is in the prompt. Consistency is agreement "
                        "with a deterministic block-energy direction, not a quality score."))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "adherence.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(song=args.song_id,
                          normal=normal.get("consistency"), shuffled=control.get("consistency"),
                          identical=identical,
                          normal_valid=normal.get("valid"), shuffled_valid=control.get("valid"))))


if __name__ == "__main__":
    main()
