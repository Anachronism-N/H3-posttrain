"""Deterministic synthetic fixtures, never a released music benchmark."""
import math
import struct
import wave
from pathlib import Path

from .dataset import inspect_wav
from .io import object_sha, write_json
from .schema import validate_dataset, validate_plan


def create_demo(out, seconds=12.0):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 5.0625 <= seconds <= 300:
        raise ValueError("demo seconds must be finite in 5.0625..300")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rate, length = 16000, round(seconds * 16000)
    wav = out / "synthetic.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        for start in range(0, length, rate):
            values = []
            for index in range(start, min(start + rate, length)):
                t = index / rate
                pulse = 0.15 if (index % (rate // 2)) < rate // 25 else 0.025
                values.append(round(32767 * pulse * math.sin(2 * math.pi * 220 * t)))
            stream.writeframes(struct.pack("<%dh" % len(values), *values))
    clock = inspect_wav(wav)
    halfway = length // 2
    song = {"song_id": "demo-song", "split": "dev", "song_group_id": "demo-song-group",
            "artist_group_id": "synthetic-artist", "near_duplicate_group_id": "demo-near",
            "brief": "Synthetic plumbing fixture; do not interpret as a music or choreography quality result.",
            "audio": dict(clock, path="synthetic.wav"),
            "music": {"source": "synthetic-fixture-known-clock",
                      "beats": list(range(0, length, rate // 2)), "downbeats": list(range(0, length, rate * 2)),
                      "sections": [{"section_id": "s000", "start": 0, "end": halfway, "label": "intro", "repeat_group": None},
                                   {"section_id": "s001", "start": halfway, "end": length, "label": "chorus", "repeat_group": "chorus"}]}}
    dataset = {"schema": "choreobench.dataset.v1", "dataset_id": "synthetic-smoke-v1", "synthetic": True,
               "grouping": {"status": "unverified", "evidence": "Synthetic fixture, not curated songs."}, "songs": [song]}
    validate_dataset(dataset)
    write_json(out / "dataset.json", dataset)
    events = []
    for i, (start, end) in enumerate(((0, halfway), (halfway, length))):
        events.extend([
            {"event_id": "arm%d" % i, "start": start, "end": end, "track": "right_arm",
             "action": "Lift the right arm in a gentle arc.", "attributes": {"side": "right", "intensity": 0.4},
             "entry": {"hand_height": "waist"}, "exit": {"hand_height": "shoulder"}, "motif_id": "arm_theme", "evidence": ["fixture"]},
            {"event_id": "feet%d" % i, "start": start, "end": end, "track": "lower",
             "action": "Alternate small side steps.", "attributes": {"intensity": 0.4},
             "entry": {"support": "left"}, "exit": {"support": "left"}, "motif_id": None, "evidence": ["fixture"]}
        ])
    events.sort(key=lambda event: (event["start"], event["track"], event["event_id"]))
    plan = {"schema": "choreobench.plan.v1", "song_id": song["song_id"], "sample_rate": rate, "num_samples": length,
            "motifs": [{"motif_id": "arm_theme", "description": "Gentle arm arc"}],
            "events": events, "provenance": {"synthetic": True, "purpose": "editing contract fixture"}}
    validate_plan(plan, song)
    write_json(out / "source_plan.json", plan)
    protected = [{"event_id": "feet0", "field": "action"}, {"event_id": "*", "field": "start"},
                 {"event_id": "*", "field": "end"}]
    instruction = "In the first section lift the left arm instead of the right. Keep the steps, timing and second section unchanged."
    task = {"schema": "choreobench.edit_task.v1", "task_id": "demo-edit", "song_id": song["song_id"],
            "source_plan_sha256": object_sha(plan), "instruction": instruction,
            "allowed_event_ids": ["arm0"], "repair_event_ids": [], "protected": protected,
            "goals": [{"event_id": "arm0", "field": "track", "op": "eq", "value": "left_arm"},
                      {"event_id": "arm0", "field": "attributes.side", "op": "eq", "value": "left"},
                      {"event_id": "arm0", "field": "action", "op": "contains", "value": "left"}],
            "semantic_goals": ["The action describes the requested left arm change without altering the steps or timing."]}
    write_json(out / "edit_task.private.json", task)
    patch = {"schema": "choreobench.patch.v1", "base_plan_sha256": object_sha(plan),
             "instruction": instruction, "allowed_event_ids": ["arm0"], "repair_event_ids": [], "protected": protected,
             "operations": [{"event_id": "arm0", "field": "track", "value": "left_arm"},
                            {"event_id": "arm0", "field": "attributes.side", "value": "left"},
                            {"event_id": "arm0", "field": "action", "value": "Lift the left arm in a gentle arc."}]}
    write_json(out / "example_patch.json", patch)
    for mode in ("concat", "global", "hierarchical"):
        config = {"mode": mode, "local_provider": {"kind": "mock"}, "planner_provider": {"kind": "mock"},
                  "seed": 7, "max_calls": 1000, "max_output_tokens": 4096, "planner_max_output_tokens": 32768, "window_seconds": 5.0625,
                  "stride_seconds": 5.0625, "candidates_per_window": 1, "splits": ["dev"]}
        write_json(out / (mode + ".json"), config)
    write_json(out / "NOTICE.json", {"synthetic": True, "benchmark_results": False,
                                    "message": "No real model, human review or music quality evaluation is represented by these fixtures."})
    return {"dataset": str(out / "dataset.json"), "source_plan": str(out / "source_plan.json"),
            "task": str(out / "edit_task.private.json"), "out": str(out), "synthetic": True}
