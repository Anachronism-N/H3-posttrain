"""Module S (music structure) and Module R (whole-song revision) contracts."""
import copy
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from choreobench import composition_music as cm
from choreobench import music_structure_summary as mss
from choreobench import whole_song_revision as wsr
from choreobench.io import file_sha


RATE = 16000


def _wav(path, seconds=20.0, silent_tail=0.0):
    total = int(seconds * RATE)
    signal = (0.2 * np.sin(2 * np.pi * 220 * np.arange(total) / RATE)).astype(np.float32)
    tail = int(silent_tail * RATE)
    if tail:
        signal[-tail:] = 0.0
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(RATE)
        handle.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def test_summary_is_deterministic_and_declares_its_limits(tmp_path):
    wav = _wav(tmp_path / "a.wav", 20.0, silent_tail=5.0)
    first = mss.analyze(wav)
    second = mss.analyze(wav)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["schema"] == mss.SCHEMA
    assert len(first["blocks"]) == 2
    assert first["blocks"][1]["relative_energy_db"] < first["blocks"][0]["relative_energy_db"]
    assert first["blocks"][1]["energy_trend"] == "fall"
    assert first["functional_labels_verified"] is False
    assert first["beat_or_meter_assumed"] is False
    assert first["audio"]["sha256"] == file_sha(wav)
    assert all(r["interpretation"].startswith("acoustic_similarity_only") for r in first["relations"])


def _song():
    return dict(song_id="s1", brief="b", audio=dict(sample_rate=RATE, num_samples=320000),
                music=dict(source="public", beats=[], downbeats=[],
                           sections=[dict(section_id="a", start=0, end=160000, label="unknown", repeat_group=None),
                                     dict(section_id="b", start=160000, end=320000, label="unknown", repeat_group=None)]))


def _units(n=4):
    step = 320000 // n
    return dict(units=[dict(unit_id="u%02d" % i, section_id="a" if i < n // 2 else "b",
                            start=i * step, end=(i + 1) * step, kind="time_block") for i in range(n)])


def _summary():
    return dict(schema=mss.SCHEMA, blocks=[
        dict(block_id="b00", start=0, end=160000, start_seconds=0.0, end_seconds=10.0, relative_energy_db=1.2,
             onset_rate_per_second=4.0, spectral_centroid_hz=1000.0, energy_trend="unknown", energy_db=-20.0,
             onset_count=40),
        dict(block_id="b01", start=160000, end=320000, start_seconds=10.0, end_seconds=20.0, relative_energy_db=-1.4,
             onset_rate_per_second=1.5, spectral_centroid_hz=900.0, energy_trend="fall", energy_db=-22.6,
             onset_count=15)],
        relations=[], parameters=dict(block_seconds=10.0), functional_labels_verified=False, beat_or_meter_assumed=False)


def _outline_response(**changes):
    value = dict(
        motifs=[dict(motif_id="m1", description="Side step with arm swing")],
        sections=[dict(section_id="a", intent="Introduce m1 with two side steps", motif_ids=["m1"]),
                  dict(section_id="b", intent="Keep stepping but lower the arms", motif_ids=["m1"])],
        section_metadata=[
            dict(section_id="a", development="introduce", quantity_change="steady",
                 transition_from_previous=dict(kind="unknown", description="first section"),
                 music_blocks=["b00"], evidence=[dict(source="music_block", note="steady energy")],
                 uncertainty="no beat"),
            dict(section_id="b", development="vary", quantity_change="decrease",
                 transition_from_previous=dict(kind="continue", description="keep the same pulse"),
                 music_blocks=["b01"], evidence=[dict(source="music_block", note="energy falls")],
                 uncertainty="no beat")],
        block_guidance=[dict(unit_id=u["unit_id"], quantity_change="steady", motif_id="m1")
                        for u in _units()["units"]])
    value.update(changes)
    return json.dumps(value)


def test_music_outline_accepts_a_grounded_design():
    outline = cm.read_outline(_outline_response(), _song(), _summary(), _units())
    assert [s["section_id"] for s in outline["sections"]] == ["a", "b"]
    assert outline["sections"][1]["metadata"]["quantity_change"] == "decrease"


@pytest.mark.parametrize("mutate,fragment", [
    (lambda v: v["sections"][0].update(intent="Be more powerful and echo the music"), "body part"),
    (lambda v: v["section_metadata"][0].update(music_blocks=["b99"]), "existing summary block"),
    (lambda v: v["section_metadata"][0]["transition_from_previous"].update(kind="continue"), "first section"),
    (lambda v: v["section_metadata"][1].update(evidence=[]), "one or two evidence"),
])
def test_music_outline_refuses_unsupported_claims(mutate, fragment):
    value = json.loads(_outline_response())
    mutate(value)
    with pytest.raises(ValueError, match=fragment):
        cm.read_outline(json.dumps(value), _song(), _summary(), _units())


def _plan():
    return dict(schema="choreobench.plan.v1", song_id="s1", sample_rate=RATE, num_samples=320000,
                motifs=[dict(motif_id="m1", description="x")], provenance={},
                events=[dict(event_id="e01", start=0, end=80000, track="body", action="Two side steps left",
                             attributes={}, entry={}, exit={}, motif_id="m1", evidence=["c1"]),
                        dict(event_id="e02", start=80000, end=160000, track="upper", action="Raise both arms",
                             attributes={}, entry={}, exit={}, motif_id="m1", evidence=["c2"])])


def _outline():
    return cm.read_outline(_outline_response(), _song(), _summary(), _units())


def test_audit_requires_exact_quote_and_music_block():
    plan, outline, summary = _plan(), _outline(), _summary()
    good = json.dumps(dict(issues=[dict(event_id="e01", quote="Two side steps", problem_type="quantity_mismatch",
                                        target="shorten to one step", music_block="b00")]))
    issues = wsr.read_audit(good, plan, summary)
    assert issues[0]["event_id"] == "e01"
    with pytest.raises(ValueError, match="exact substring"):
        wsr.read_audit(json.dumps(dict(issues=[dict(event_id="e01", quote="not in the action",
                                                    problem_type="missing_motif", target="x", music_block=None)])),
                       plan, summary)
    with pytest.raises(ValueError, match="music block"):
        wsr.read_audit(json.dumps(dict(issues=[dict(event_id="e01", quote="Two side steps",
                                                    problem_type="quantity_mismatch", target="x", music_block=None)])),
                       plan, summary)
    with pytest.raises(ValueError, match="at most three"):
        wsr.read_audit(json.dumps(dict(issues=[dict(event_id="e01", quote="Two", problem_type="missing_motif",
                                                    target="x", music_block=None)] * 4)), plan, summary)
    assert outline and summary


def test_patch_is_action_only_and_cannot_touch_other_events():
    plan, issues = _plan(), [dict(event_id="e01")]
    patches = wsr.read_patches(json.dumps(dict(patches=[dict(event_id="e01", action="One side step left")])), issues)
    final, changed = wsr.apply_patches(plan, patches)
    assert changed == ["e01"] and final["events"][0]["action"] == "One side step left"
    assert final["events"][0]["start"] == plan["events"][0]["start"]
    assert final["events"][1] == plan["events"][1]
    with pytest.raises(ValueError, match="non-listed"):
        wsr.read_patches(json.dumps(dict(patches=[dict(event_id="e02", action="x")])), issues)


def test_apply_patches_rejects_a_time_change():
    plan = _plan()
    broken = copy.deepcopy(plan)
    broken["events"][0]["end"] = 90000
    with pytest.raises(ValueError):
        wsr.apply_patches(broken, [])


class _Provider:
    identity = dict(kind="replay", synthetic=True)

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def generate(self, request):
        self.calls.append(request)
        text = self.responses.pop(0)
        return dict(text=text, raw={}, identity=self.identity, synthetic=True,
                    stop_reason="stop", costs=dict(input_tokens=100, output_tokens=20))


def _draft(tmp_path, plan=None):
    run = tmp_path / "draft"
    run.mkdir()
    from choreobench.io import write_json
    write_json(run / "contract.json", dict(source_inputs_sha256={}))
    write_json(run / "plan.json", plan or _plan())
    write_json(run / "results.json", dict(complete=True, full_song_succeeded=True, calls_used=3,
                                          input_tokens=1000, output_tokens=100))
    write_json(run / "outline.json", _outline())
    return run


def test_revision_applies_one_patch_and_keeps_the_draft(tmp_path, monkeypatch):
    run = _draft(tmp_path)
    provider = _Provider([
        json.dumps(dict(issues=[dict(event_id="e01", quote="Two side steps", problem_type="quantity_mismatch",
                                     target="reduce", music_block="b00")])),
        json.dumps(dict(patches=[dict(event_id="e01", action="One slow side step")]))])
    monkeypatch.setattr(wsr, "create_provider", lambda *a, **k: provider)
    summary_path = tmp_path / "music.json"
    summary_path.write_text(json.dumps(_summary()))
    record = wsr.run_revision(run, dict(kind="replay"), summary_path, tmp_path / "rev",
                              song=_song(), outline=_outline(), max_calls=100)
    assert record["completed"] and record["outcome"] == "patched"
    assert record["changed_event_ids"] == ["e01"] and record["revision_calls"] == 2
    assert record["total_calls"] == 5  # draft calls are counted, not treated as free
    assert json.loads((run / "plan.json").read_text())["events"][0]["action"] == "Two side steps left"
    final = json.loads((tmp_path / "rev" / "final_plan.json").read_text())
    assert final["events"][0]["action"] == "One slow side step"


def test_revision_with_no_issue_changes_nothing(tmp_path, monkeypatch):
    run = _draft(tmp_path)
    provider = _Provider([json.dumps(dict(issues=[]))])
    monkeypatch.setattr(wsr, "create_provider", lambda *a, **k: provider)
    summary_path = tmp_path / "music.json"
    summary_path.write_text(json.dumps(_summary()))
    record = wsr.run_revision(run, dict(kind="replay"), summary_path, tmp_path / "rev",
                              song=_song(), outline=_outline(), max_calls=100)
    assert record["completed"] and record["outcome"] == "no_change" and record["revision_calls"] == 1


def test_revision_failure_keeps_the_original_plan(tmp_path, monkeypatch):
    run = _draft(tmp_path)
    provider = _Provider(["not json at all"])
    monkeypatch.setattr(wsr, "create_provider", lambda *a, **k: provider)
    summary_path = tmp_path / "music.json"
    summary_path.write_text(json.dumps(_summary()))
    record = wsr.run_revision(run, dict(kind="replay"), summary_path, tmp_path / "rev",
                              song=_song(), outline=_outline(), max_calls=100)
    assert record["completed"] is False and record["outcome"] == "revision_failed"
    assert json.loads((tmp_path / "rev" / "final_plan.json").read_text()) == _plan()


def test_revision_is_skipped_without_call_budget(tmp_path):
    run = _draft(tmp_path)
    record = wsr.run_revision(run, dict(kind="replay"), tmp_path / "missing.json", tmp_path / "rev",
                              song=_song(), outline=_outline(), max_calls=3)
    assert record["skipped"] == "no_call_budget"


def test_bar_blocks_cover_the_song_without_tiny_units(tmp_path):
    import wave
    rate, seconds = 16000, 24.0
    total = int(seconds * rate)
    t = np.arange(total) / rate
    # 120 BPM: a low thump every 0.5 s, accented every 4th beat.
    signal = 0.05 * np.sin(2 * np.pi * 220 * t)
    for beat in range(int(seconds / 0.5)):
        start = int(beat * 0.5 * rate)
        amp = 0.9 if beat % 4 == 0 else 0.4
        length = int(0.12 * rate)
        signal[start:start + length] += amp * np.sin(2 * np.pi * 60 * np.arange(length) / rate)
    path = tmp_path / "bars.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(rate)
        handle.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
    from choreobench import bar_structure
    structure = bar_structure.analyze(str(path))
    assert structure["meter_assumed"] == "4/4" and structure["meter_detected"] is False
    assert len(structure["downbeats"]) >= 3
    units = bar_structure.blocks(structure, "s000", total, rate)["units"]
    assert units[0]["start"] == 0 and units[-1]["end"] == total
    assert all(a["end"] == b["start"] for a, b in zip(units, units[1:]))
    assert min(u["end"] - u["start"] for u in units) >= int(0.5 * rate)
    assert any(u["kind"] == "bar_block" for u in units)


def test_block_motif_quota_cycles_and_returns_home():
    units = [dict(unit_id="u%02d" % i, section_id="a") for i in range(6)]
    design = dict(section_id="a", motif_ids=["m1", "m2"])
    picks = [cm.assigned_motifs(units, i, design) for i in range(6)]
    assert picks[:5] == [["m1"], ["m2"], ["m1"], ["m2"], ["m1"]]
    assert picks[5] == ["m1"]          # last block returns to the first motif
    assert cm.assigned_motifs(units, 0, dict(section_id="a", motif_ids=["m1"])) == ["m1"]


def test_null_motif_is_stripped_from_the_block_schema():
    schema = dict(type="object", properties=dict(events=dict(items=dict(
        properties=dict(motif_id=dict(enum=["m1", None]))))))
    cm._strip_null_motif(schema)
    assert schema["properties"]["events"]["items"]["properties"]["motif_id"]["enum"] == ["m1"]


def test_previous_motif_occurrence_returns_last_and_ignores_null():
    events = [dict(motif_id="m1", action="first", start=0, end=10),
              dict(motif_id="m2", action="other", start=10, end=20),
              dict(motif_id="m1", action="second", start=20, end=30)]
    assert cm.previous_motif_occurrence(events, "m1")["action"] == "second"
    assert cm.previous_motif_occurrence(events, "m2")["action"] == "other"
    assert cm.previous_motif_occurrence(events, "m9") is None
    assert cm.previous_motif_occurrence(events + [dict(motif_id=None, action="x")], None) is None


def test_motif_variation_adds_context_only_when_enabled(monkeypatch):
    import json as _json
    from choreobench import coordination

    def fake_unit_request(*args, **kwargs):
        return dict(messages=[dict(content="SYSTEM"),
                              dict(content='{"local_proposals": [], "output_contract": {}}')])

    monkeypatch.setattr(coordination, "_unit_request", fake_unit_request)
    unit = dict(unit_id="u0", section_id="a", start=0, end=16000)
    state = type("State", (), {})()
    state.source = dict(song=dict(song_id="s", sample_rate=16000, num_samples=16000), options={},
                        units=dict(units=[unit]), config={})
    state.outline = dict(motifs=[dict(motif_id="m1", description="travel")],
                         sections=[dict(section_id="a", intent="intro", motif_ids=["m1"],
                                        metadata={})],
                         block_guidance=None)
    state.plan = dict(sample_rate=16000,
                      events=[dict(event_id="e0", motif_id="m1", track="body", action="previous travel",
                                   start=0, end=8000)])
    summary = dict(blocks=[], relations=[], sample_rate=16000)
    cfg = dict(acoustic_guidance=False, motif_quota=True, motif_variation=True, max_events_per_unit=3,
               event_time_format="boundary_ids", event_grid_seconds=0.25, min_event_seconds=0.5,
               max_request_characters=200000)
    request = cm.unit_request(state, unit, summary, cfg)
    payload = _json.loads(request["messages"][-1]["content"])
    assert payload["previous_motif_occurrence"]["action"] == "previous travel"
    assert "return of that theme" in request["messages"][0]["content"]

    cfg_off = dict(cfg, motif_variation=False)
    request_off = cm.unit_request(state, unit, summary, cfg_off)
    assert _json.loads(request_off["messages"][-1]["content"])["previous_motif_occurrence"] is None
    assert "return of that theme" not in request_off["messages"][0]["content"]
