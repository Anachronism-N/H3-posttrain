"""Revision plumbing with synthetic PCM and observable fake providers only.

No weights, remote APIs, musical judgments or semantic success are represented.
"""
import copy
import json
from pathlib import Path

import pytest

from choreobench import revision
from choreobench.fixtures import create_demo
from choreobench.io import file_sha, object_sha, read_json, write_json
from choreobench.music_recheck import MUSIC_OBSERVATION_PROMPT
from choreobench.schema import validate_plan


class FakeProvider:
    def __init__(self, world, audio=False):
        self.world, self.audio = world, audio
        self.identity = dict(kind="replay", synthetic=True,
                             fixture="synthetic-audio-v1" if audio else "synthetic-text-v1")
        if audio:
            self.identity["task_prompt"] = MUSIC_OBSERVATION_PROMPT

    def generate(self, request):
        self.world.calls.append(("audio" if self.audio else "text", copy.deepcopy(request)))
        kind = request["response_kind"]
        payload = json.loads(request["messages"][-1]["content"]) if not self.audio else None
        event = payload.get("target_event_id", payload.get("event_id")) if payload else None
        if (kind, event) in self.world.interrupt:
            raise KeyboardInterrupt("fixture interruption after durable call start")
        if (kind, event) in self.world.raise_on:
            raise ValueError("fixture provider exception")
        if self.audio:
            assert kind == "local_caption"
            assert set(request) == {"call_id", "seed", "max_output_tokens", "response_kind", "audio"}
            assert request["audio"]["num_samples"] == 81000
            text = "A steady pulse and moderate energy; precise accents are uncertain."
        elif kind == "revision_diagnosis":
            text = json.dumps(dict(schema="choreobench.revision_diagnosis.v1",
                source_plan_sha256=payload["source_plan_sha256"], issues=copy.deepcopy(self.world.issues)))
        elif kind == "patch":
            patch = copy.deepcopy(payload["patch_template"])
            behavior = self.world.edit_behavior.get(event, "edit")
            if behavior == "malformed":
                text = "not a JSON patch"
            else:
                old = next(e["action"] for e in payload["source_plan"]["events"] if e["event_id"] == event)
                patch["operations"] = [] if behavior == "no_op" else [dict(
                    event_id=event, field="start" if behavior == "forbidden" else "action",
                    value=1 if behavior == "forbidden" else old + " Keep a controlled, clear motion.")]
                if behavior == "same_value":
                    patch["operations"][0]["value"] = old
                text = json.dumps(patch)
        elif kind == "revision_verdict":
            revised_label = next(label for label, plan in payload["pair"].items()
                if "Keep a controlled, clear motion." in next(e["action"] for e in plan["events"] if e["event_id"] == event))
            decision = self.world.verdicts.get(event, "prefer_revision")
            choice = ("uncertain" if decision == "uncertain" else revised_label if decision == "prefer_revision"
                      else "b" if revised_label == "a" else "a")
            assert "revision_label" not in payload and "original" not in payload and "revised" not in payload
            text = json.dumps(dict(schema="choreobench.revision_choice.v1",
                a_plan_sha256=payload["a_plan_sha256"], b_plan_sha256=payload["b_plan_sha256"],
                event_id=event, choice=choice,
                reason="Synthetic internal selection fixture, not a quality judgment."))
        else:
            raise AssertionError("Unexpected request kind: " + kind)
        if (kind, event) in self.world.malformed:
            text = "malformed fixture output"
        identity = copy.deepcopy(self.identity)
        if (kind, event) in self.world.identity_drift:
            identity["fixture"] = "different-model-identity"
        if (kind, event) in self.world.mutate_identity:
            self.identity["fixture"] = "in-place-provider-mutation"
        return dict(text=text, raw=dict(fixture=True), identity=identity,
                    synthetic=self.identity["synthetic"], stop_reason=self.world.stops.get((kind, event), "stop"),
                    costs=dict(input_tokens=2, output_tokens=1, latency_seconds=0.0))


class World:
    def __init__(self, root):
        made = create_demo(root / "demo", seconds=12)
        self.dataset, self.plan_path = Path(made["dataset"]), Path(made["source_plan"])
        self.plan = read_json(self.plan_path)
        self.events = [event["event_id"] for event in self.plan["events"]]
        self.issues = [dict(event_id=self.events[0], priority=3, reason="Synthetic check of this event."),
                       dict(event_id=self.events[1], priority=2, reason="Synthetic check of the next event.")]
        self.calls, self.factories, self.observer_factories = [], [], []
        self.edit_behavior, self.verdicts, self.stops = {}, {}, {}
        self.interrupt, self.raise_on, self.malformed, self.identity_drift = set(), set(), set(), set()
        self.mutate_identity = set()
        self.text, self.audio = FakeProvider(self), FakeProvider(self, audio=True)

    def config(self, **changes):
        value = dict(planner_provider={"kind": "replay", "fixture": "synthetic"},
                     seed=0, listen=False, max_repairs=2, max_calls=6, source_plan_synthetic=True)
        value.update(changes)
        if value["listen"]:
            value.setdefault("observer_provider", {"kind": "replay", "fixture": "synthetic-audio"})
        return value

    def diagnosis(self, out, config=None):
        result = revision.diagnose_plan(self.dataset, self.plan_path, out, config or self.config())
        return Path(out) / "diagnosis.json", result

    def payloads(self, kind):
        return [json.loads(req["messages"][-1]["content"]) for role, req in self.calls
                if role == "text" and req["response_kind"] == kind]


@pytest.fixture
def world(tmp_path, monkeypatch):
    value = World(tmp_path)
    def text_provider(spec, role, base_dir):
        assert role == "planner"
        value.factories.append(copy.deepcopy(spec))
        return value.text
    def audio_provider(spec, base_dir):
        value.observer_factories.append(copy.deepcopy(spec))
        return value.audio
    monkeypatch.setattr(revision, "create_provider", text_provider)
    monkeypatch.setattr(revision, "create_music_observer", audio_provider)
    return value


def test_diagnosis_is_shared_bytes_targeted_and_random_select_same_number_from_all_events(world, tmp_path):
    world.issues = [dict(event_id=world.events[0], priority=3, reason="Synthetic issue.")]
    diagnosis, collected = world.diagnosis(tmp_path / "diagnosis")
    original_sha = file_sha(diagnosis)
    targeted = revision.run_revision(world.dataset, world.plan_path, diagnosis, tmp_path / "targeted",
                                     world.config(max_repairs=1))
    random = revision.run_revision(world.dataset, world.plan_path, diagnosis, tmp_path / "random",
                                   world.config(selection_policy="random", max_repairs=1))
    assert len(world.payloads("revision_diagnosis")) == 1
    assert collected["calls_used"] == 1 and file_sha(diagnosis) == original_sha
    assert targeted["selected_count"] == random["selected_count"] == 1
    assert targeted["calls_used"] == random["calls_used"] == 2
    assert targeted["steps"][0]["event_id"] == world.events[0]
    assert random["steps"][0]["event_id"] not in {row["event_id"] for row in world.issues}
    patches = world.payloads("patch")
    assert patches[0]["initial_plan_diagnosis"] == patches[1]["initial_plan_diagnosis"]
    assert patches[0]["source_plan"] == patches[1]["source_plan"]
    assert targeted["diagnosis_new_calls"] == random["diagnosis_new_calls"] == 0
    assert targeted["diagnosis_collection_accounting"] == random["diagnosis_collection_accounting"]


@pytest.mark.parametrize("decision,accepted", [("prefer_revision", True), ("keep_original", False), ("uncertain", False)])
def test_action_only_candidate_and_internal_judge_do_not_assert_semantic_success(world, tmp_path, decision, accepted):
    world.issues = world.issues[:1]
    world.verdicts[world.events[0]] = decision
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    out = tmp_path / "revision"
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, out, world.config(max_repairs=1))
    after = read_json(out / "plan.json")
    validate_plan(after)
    expected = copy.deepcopy(world.plan)
    if accepted:
        expected["events"][0]["action"] += " Keep a controlled, clear motion."
    assert {k: v for k, v in after.items() if k != "provenance"} == {
        k: v for k, v in expected.items() if k != "provenance"}
    assert all(after["provenance"][k] == v for k, v in world.plan["provenance"].items())
    assert result["accepted_count"] == int(accepted)
    assert result["proposed_count"] == result["judged_count"] == 1
    assert result["semantic_success"] is None and result["quality_verified"] is False
    assert result["synthetic"] is True and result["model_weights_updated"] is False
    assert result["steps"][0]["status"] == decision


@pytest.mark.parametrize("behavior", ["no_op", "same_value"])
def test_no_op_does_not_call_judge_or_count_as_acceptance(world, tmp_path, behavior):
    world.issues = world.issues[:1]
    world.edit_behavior[world.events[0]] = behavior
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    out = tmp_path / "revision"
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, out, world.config(max_repairs=1))
    assert result["calls_used"] == 1
    assert result["accepted_count"] == result["judged_count"] == 0
    assert result["steps"][0]["status"] == "no_op"
    assert read_json(out / "plan.json") == world.plan
    assert not world.payloads("revision_verdict")


@pytest.mark.parametrize("later", ["uncertain", "bad_edit", "bad_judge"])
def test_later_uncertain_or_failed_attempt_does_not_erase_earlier_acceptance(world, tmp_path, later):
    second = world.events[1]
    if later == "uncertain":
        world.verdicts[second] = "uncertain"
    elif later == "bad_edit":
        world.edit_behavior[second] = "malformed"
    else:
        world.malformed.add(("revision_verdict", second))
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    out = tmp_path / "revision"
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, out, world.config())
    after = read_json(out / "plan.json")
    assert after["events"][0]["action"] != world.plan["events"][0]["action"]
    assert after["events"][1:] == world.plan["events"][1:]
    assert result["selected_count"] == 2 and result["accepted_count"] == 1
    assert result["failed_count"] == int(later != "uncertain")
    assert result["steps"][1]["before_plan_sha256"] == result["steps"][0]["after_plan_sha256"]
    assert result["steps"][1]["after_plan_sha256"] == result["steps"][0]["after_plan_sha256"]
    assert len(world.payloads("patch")) == 2  # No replacement event after failure.
    assert result["complete"] is True and result["semantic_success"] is None


def test_forbidden_change_is_retained_failed_without_sampling_a_replacement(world, tmp_path):
    world.edit_behavior[world.events[0]] = "forbidden"
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    out = tmp_path / "revision"
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, out, world.config(max_repairs=1))
    assert result["failed_count"] == result["selected_count"] == 1
    assert result["all_selected_attempts_usable"] is False
    assert result["calls_used"] == 1 and not world.payloads("revision_verdict")
    assert len(world.payloads("patch")) == 1 and read_json(out / "plan.json") == world.plan
    assert (out / "steps/00/patch.json").is_file()  # Invalid raw proposal retained.


def test_audio_observer_receives_audio_only_and_text_observation_is_unverified(world, tmp_path):
    world.issues = world.issues[:1]
    # Private provenance/evidence are allowed in the source file, not model views.
    world.plan["provenance"]["private_goals"] = "PRIVATE_SENTINEL"
    world.plan["events"][0]["evidence"] = ["PRIVATE_SENTINEL"]
    write_json(world.plan_path, world.plan, overwrite=True)
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    out = tmp_path / "revision"
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, out,
                                   world.config(listen=True, max_repairs=1, max_calls=3))
    observations = [request for role, request in world.calls if role == "audio"]
    assert len(observations) == 1 and result["calls_used"] == 3
    request = observations[0]
    assert "messages" not in request and "plan" not in json.dumps(request)
    assert request["audio"]["sha256"] == file_sha(request["audio"]["path"])
    assert world.audio.identity["task_prompt"] == MUSIC_OBSERVATION_PROMPT
    observation = read_json(out / "steps/00/observation.json")
    assert observation["musical_facts_verified"] is False
    assert observation["covers_entire_event"] is False  # 6s event, 5.0625s observation.
    assert observation["target_overlap_end"] - observation["target_overlap_start"] == 81000
    text_payloads = [request["messages"] for role, request in world.calls if role == "text"]
    assert "PRIVATE_SENTINEL" not in json.dumps(text_payloads)
    payload = world.payloads("patch")[0]
    assert payload["music_observation"]["musical_facts_verified"] is False
    assert "audio" not in payload["music_observation"]


def test_budget_preflight_blocks_all_revision_provider_calls(world, tmp_path):
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    count = len(world.calls)
    with pytest.raises(ValueError, match="cannot cover"):
        revision.run_revision(world.dataset, world.plan_path, diagnosis, tmp_path / "revision",
                              world.config(max_calls=3))
    assert len(world.calls) == count
    assert not (tmp_path / "revision").exists()


def test_no_issues_means_no_revision_or_observer_calls(world, tmp_path):
    world.issues = []
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    count = len(world.calls)
    out = tmp_path / "revision"
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, out, world.config(listen=True))
    assert result["selected_count"] == result["calls_used"] == result["accepted_count"] == 0
    assert len(world.calls) == count and not world.observer_factories
    assert read_json(out / "plan.json") == world.plan
    assert result["semantic_success"] is None


def test_diagnosis_and_revision_resume_make_zero_additional_calls(world, tmp_path):
    cfg = world.config(listen=True)
    diagnosis, first_diag = world.diagnosis(tmp_path / "diag", cfg)
    count = len(world.calls)
    assert revision.diagnose_plan(world.dataset, world.plan_path, tmp_path / "diag", cfg, resume=True) == first_diag
    assert len(world.calls) == count
    first = revision.run_revision(world.dataset, world.plan_path, diagnosis, tmp_path / "revision", cfg)
    count = len(world.calls)
    second = revision.run_revision(world.dataset, world.plan_path, diagnosis, tmp_path / "revision", cfg, resume=True)
    assert second == first and len(world.calls) == count
    assert first["calls_used"] == 6


@pytest.mark.parametrize("source", ["plan", "dataset", "audio"])
def test_changed_original_inputs_rejected_before_revision_dispatch(world, tmp_path, source):
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    if source == "plan":
        changed = copy.deepcopy(world.plan)
        changed["events"][0]["action"] += " Changed initial action."
        write_json(world.plan_path, changed, overwrite=True)
    elif source == "dataset":
        world.dataset.write_text(world.dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    else:
        path = world.dataset.parent / "synthetic.wav"
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
    count = len(world.calls)
    with pytest.raises(ValueError):
        revision.run_revision(world.dataset, world.plan_path, diagnosis, tmp_path / "revision", world.config())
    assert len(world.calls) == count


def test_diagnosis_record_tampering_is_not_accepted(world, tmp_path):
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    record = read_json(diagnosis)
    record["diagnosis"]["issues"][0]["reason"] = "Tampered after binding"
    write_json(diagnosis, record, overwrite=True)
    count = len(world.calls)
    with pytest.raises(ValueError, match="hash"):
        revision.run_revision(world.dataset, world.plan_path, diagnosis, tmp_path / "revision", world.config())
    assert len(world.calls) == count


@pytest.mark.parametrize("artifact", ["candidate", "raw_call", "observation"])
def test_changed_published_artifact_or_journal_blocks_resume(world, tmp_path, artifact):
    cfg = world.config(listen=True, max_repairs=1)
    diagnosis, _ = world.diagnosis(tmp_path / "diag", cfg)
    out = tmp_path / "revision"
    revision.run_revision(world.dataset, world.plan_path, diagnosis, out, cfg)
    if artifact == "candidate":
        path = out / "steps/00/candidate.json"
        changed = read_json(path); changed["events"][0]["action"] += " Changed."
        write_json(path, changed, overwrite=True)
    elif artifact == "raw_call":
        path = next((out / "calls").glob("*.raw.json"))
        changed = read_json(path); changed["result"]["text"] = "tampered"
        write_json(path, changed, overwrite=True)
    else:
        path = out / "steps/00/observation.wav"
        changed = bytearray(path.read_bytes()); changed[-1] ^= 1; path.write_bytes(changed)
    count = len(world.calls)
    with pytest.raises(ValueError):
        revision.run_revision(world.dataset, world.plan_path, diagnosis, out, cfg, resume=True)
    assert len(world.calls) == count


def test_changed_configuration_or_provider_identity_cannot_resume(world, tmp_path):
    cfg = world.config(max_repairs=1)
    diagnosis, _ = world.diagnosis(tmp_path / "diag", cfg)
    out = tmp_path / "revision"
    revision.run_revision(world.dataset, world.plan_path, diagnosis, out, cfg)
    count = len(world.calls)
    with pytest.raises(ValueError, match="identity changed"):
        revision.run_revision(world.dataset, world.plan_path, diagnosis, out, dict(cfg, seed=8), resume=True)
    world.text.identity["fixture"] = "changed-provider"
    with pytest.raises(ValueError, match="identity changed"):
        revision.run_revision(world.dataset, world.plan_path, diagnosis, out, cfg, resume=True)
    assert len(world.calls) == count


def test_pending_call_after_interruption_is_not_retried_or_replaced(world, tmp_path):
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    world.interrupt.add(("patch", world.events[0]))
    out = tmp_path / "revision"
    with pytest.raises(KeyboardInterrupt):
        revision.run_revision(world.dataset, world.plan_path, diagnosis, out, world.config())
    assert len(list((out / "calls").glob("*.started.json"))) == 1
    assert not list((out / "calls").glob("*.raw.json"))
    world.interrupt.clear()
    count = len(world.calls)
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, out, world.config(), resume=True)
    assert len(world.calls) == count
    assert result["complete"] is False and "Pending/unknown" in result["error"]
    assert result["unknown_outcomes_count"] == 1 and result["not_attempted_count"] == 1


@pytest.mark.parametrize("stage", ["diagnosis", "revision"])
def test_request_character_cap_rejects_before_dispatch_without_truncation(world, tmp_path, stage):
    if stage == "diagnosis":
        count = len(world.calls)
        with pytest.raises(ValueError, match="max_request_characters"):
            world.diagnosis(tmp_path / "diag", world.config(max_request_characters=1))
    else:
        diagnosis, _ = world.diagnosis(tmp_path / "diag")
        count = len(world.calls)
        result = revision.run_revision(world.dataset, world.plan_path, diagnosis, tmp_path / "revision",
                                       world.config(max_request_characters=1))
        assert result["complete"] is False and "max_request_characters" in result["error"]
        assert result["calls_used"] == 0
    assert len(world.calls) == count


@pytest.mark.parametrize("malformed", ["duplicate", "unknown", "priority"])
def test_bad_diagnosis_remains_a_failed_diagnosis_not_zero_issues(world, tmp_path, malformed):
    if malformed == "duplicate":
        world.issues = [world.issues[0], world.issues[0]]
    elif malformed == "unknown":
        world.issues[0]["event_id"] = "not-an-existing-event"
    else:
        world.issues[0]["priority"] = True
    out = tmp_path / "diag"
    with pytest.raises(ValueError):
        world.diagnosis(out)
    result = read_json(out / "results.json")
    assert result["complete"] is False and result["calls_used"] == 1
    assert not (out / "diagnosis.json").exists()
    count = len(world.calls)
    with pytest.raises(ValueError):
        revision.diagnose_plan(world.dataset, world.plan_path, out, world.config(), resume=True)
    assert len(world.calls) == count


def test_diagnosis_synthetic_taint_cannot_be_removed_by_revision_declarations(world, tmp_path):
    # False declarations here isolate the OR rule; these remain synthetic fixtures.
    dataset = read_json(world.dataset); dataset["synthetic"] = False
    write_json(world.dataset, dataset, overwrite=True)
    world.plan["provenance"] = {}
    write_json(world.plan_path, world.plan, overwrite=True)
    cfg = world.config(source_plan_synthetic=False, max_repairs=1)
    diagnosis, _ = world.diagnosis(tmp_path / "diag", cfg)
    assert read_json(diagnosis)["synthetic"] is True
    world.text.identity["synthetic"] = False
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, tmp_path / "revision", cfg)
    assert result["synthetic"] is True


def test_identity_drift_is_fatal_before_another_selected_event_is_dispatched(world, tmp_path):
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    world.identity_drift.add(("patch", world.events[0]))
    count = len(world.calls)
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, tmp_path / "revision", world.config())
    assert result["complete"] is False and "identity" in result["error"]
    assert result["failed_count"] == result["not_attempted_count"] == 1
    assert len(world.calls) == count + 1


@pytest.mark.parametrize("failure", ["exception", "http_error", "malformed_api_response", "infrastructure_error"])
def test_unknown_provider_outcome_stops_remaining_inventory_without_retry(world, tmp_path, failure):
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    if failure == "exception":
        world.raise_on.add(("patch", world.events[0]))
    else:
        world.stops[("patch", world.events[0])] = failure
    count = len(world.calls)
    out = tmp_path / "revision"
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, out, world.config())
    assert result["complete"] is False and result["not_attempted_count"] == 1
    assert result["failed_count"] == result["unknown_outcomes_count"] == 1
    assert len(world.calls) == count + 1 and read_json(out / "plan.json") == world.plan
    count = len(world.calls)
    again = revision.run_revision(world.dataset, world.plan_path, diagnosis, out, world.config(), resume=True)
    assert again == result and len(world.calls) == count


def test_completed_truncated_model_answer_can_fail_one_point_and_continue(world, tmp_path):
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    world.stops[("patch", world.events[0])] = "length"
    out = tmp_path / "revision"
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, out, world.config())
    assert result["complete"] is True and result["not_attempted_count"] == 0
    assert result["failed_count"] == result["accepted_count"] == 1
    assert result["calls_used"] == 3
    assert read_json(out / "plan.json")["events"][0] == world.plan["events"][0]
    assert read_json(out / "plan.json")["events"][1]["action"] != world.plan["events"][1]["action"]


def test_late_infrastructure_failure_keeps_prior_accepted_edit(world, tmp_path):
    world.issues.append(dict(event_id=world.events[2], priority=1, reason="Third synthetic selected point."))
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    world.raise_on.add(("patch", world.events[1]))
    out = tmp_path / "revision"
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, out, world.config(max_repairs=3))
    after = read_json(out / "plan.json")
    assert result["complete"] is False and result["accepted_count"] == 1
    assert result["failed_count"] == result["not_attempted_count"] == 1
    assert after["events"][0]["action"] != world.plan["events"][0]["action"]
    assert after["events"][1:] == world.plan["events"][1:]
    assert len(world.payloads("patch")) == 2


def test_anonymous_choice_mapping_is_saved_but_not_sent_to_judge(world, tmp_path):
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    out = tmp_path / "revision"
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, out, world.config())
    assert result["accepted_count"] == 2
    for index, payload in enumerate(world.payloads("revision_verdict")):
        comparison = read_json(out / ("steps/%02d/comparison.json" % index))
        assert set(payload["pair"]) == {"a", "b"}
        assert comparison["revision_label"] in ("a", "b")
        assert "revision_label" not in json.dumps(payload)
        verdict = read_json(out / ("steps/%02d/verdict.json" % index))
        assert verdict["decision"] == "prefer_revision"


def test_synthetic_observer_taints_otherwise_declared_real_revision(world, tmp_path):
    # Explicit declaration controls only, not evidence of a real model/data run.
    dataset = read_json(world.dataset); dataset["synthetic"] = False
    write_json(world.dataset, dataset, overwrite=True)
    world.plan["provenance"] = {}
    write_json(world.plan_path, world.plan, overwrite=True)
    world.text.identity["synthetic"] = False
    cfg = world.config(source_plan_synthetic=False, listen=True, max_repairs=1)
    diagnosis, _ = world.diagnosis(tmp_path / "diag", cfg)
    assert read_json(diagnosis)["synthetic"] is False
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, tmp_path / "revision", cfg)
    assert result["synthetic"] is True


def test_provider_in_place_identity_mutation_does_not_rewrite_contract_or_continue(world, tmp_path):
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    original_identity = copy.deepcopy(world.text.identity)
    world.mutate_identity.add(("patch", world.events[0]))
    count = len(world.calls)
    out = tmp_path / "revision"
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, out, world.config())
    assert result["complete"] is False and "identity changed during dispatch" in result["error"]
    assert result["failed_count"] == result["not_attempted_count"] == 1
    assert len(world.calls) == count + 1
    assert read_json(out / "contract.json")["provider_identity"] == original_identity
    assert read_json(out / "plan.json") == world.plan


def test_later_real_character_cap_preserves_prior_edit_and_full_selected_denominator(world, tmp_path):
    world.issues.append(dict(event_id=world.events[2], priority=1, reason="Third synthetic selected point."))
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    cfg = world.config(max_repairs=3)
    # A completed synthetic reference determines exact serialized sizes, without
    # hand-waving about message-only length. No real model is called here.
    revision.run_revision(world.dataset, world.plan_path, diagnosis, tmp_path / "size-reference", cfg)
    requests = [r for role, r in world.calls if role == "text" and r["response_kind"] != "revision_diagnosis"]
    size = lambda r: len(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
    limit = max(size(r) for r in requests[:2])
    assert max(size(r) for r in requests[2:4]) > limit
    count = len(world.calls)
    out = tmp_path / "bounded"
    result = revision.run_revision(world.dataset, world.plan_path, diagnosis, out,
                                   dict(cfg, max_request_characters=limit))
    assert result["complete"] is False and "max_request_characters" in result["error"]
    assert result["selected_count"] == 3 and result["accepted_count"] == 1
    assert result["failed_count"] == result["not_attempted_count"] == 1
    assert 2 <= len(world.calls) - count <= 3
    after = read_json(out / "plan.json")
    assert after["events"][0]["action"] != world.plan["events"][0]["action"]
    assert after["events"][1:] == world.plan["events"][1:]


def test_real_review_pack_cli_binds_original_and_revision_including_retained_failure(world, tmp_path, capsys):
    from choreobench.cli import main
    from choreobench.reviews import import_reviews, validate_review_pack

    world.issues = world.issues[:1]
    diagnosis, _ = world.diagnosis(tmp_path / "diag")
    cfg = world.config(max_repairs=1)
    for failed in (False, True):
        name = "failed" if failed else "accepted"
        if failed:
            world.raise_on.add(("patch", world.events[0]))
        out, pack_dir = tmp_path / (name + "-run"), tmp_path / (name + "-review")
        result = revision.run_revision(world.dataset, world.plan_path, diagnosis, out, cfg)
        assert result["complete"] is (not failed)
        if failed:
            assert read_json(out / "plan.json") == world.plan
        assert result["plans"] == [dict(song_id=world.plan["song_id"], path="plan.json",
                                        sha256=file_sha(out / "plan.json"))]
        count = len(world.calls)
        assert main(["review-pack", "--dataset", str(world.dataset), "--split", "dev",
                     "--song-id", world.plan["song_id"],
                     "--submission", str(out / "original.submission.json"),
                     "--run", "revision=" + str(out), "--copy-audio", "--out-dir", str(pack_dir)]) == 0
        assert len(world.calls) == count  # Packaging is read-only: no new provider calls.
        exported = json.loads(capsys.readouterr().out)
        assert exported["audio_copied"] is True
        public = read_json(pack_dir / "public/pack.json")
        bindings = read_json(pack_dir / "private/bindings.json")
        validate_review_pack(public)
        assert len(public["items"]) == len(bindings["items"]) == 2
        assert {b["method_id"] for b in bindings["items"]} == {"original", "revision"}
        assert bindings["public_pack_sha256"] == object_sha(public)
        original_audio_sha = read_json(world.dataset)["songs"][0]["audio"]["sha256"]
        for item in public["items"]:
            assert "method_id" not in item and "source_binding" not in item
            assert item["song"]["song_id"] != world.plan["song_id"]
            assert item["plan"]["provenance"] == {}
            assert file_sha(pack_dir / "public" / item["song"]["audio"]["path"]) == original_audio_sha
        by_method = {b["method_id"]: b for b in bindings["items"]}
        for method, source_path, plan_path, kind in (
            ("original", out / "original.submission.json", out / "before.json", "submission"),
            ("revision", out / "results.json", out / "plan.json", "run"),
        ):
            binding = by_method[method]
            source = binding["original"]["source_binding"]
            assert source == dict(kind=kind, sha256=file_sha(source_path),
                                  dataset_sha256=file_sha(world.dataset), synthetic=True)
            assert binding["plan_sha256"] == object_sha(read_json(plan_path))
        # Validate the complete private/public round trip without inventing scores.
        pending = import_reviews(public, bindings, dict(schema="choreobench.reviews.v1",
            pack_sha256=object_sha(public), reviews=[]))
        assert pending["pending_item_count"] == 2
