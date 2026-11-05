"""CPU request-size contracts; no model/API or musical quality claim."""
import copy
import json

import pytest

from choreobench.baseline import ModelOutputError
from choreobench.coordination import unit_response_schema
from choreobench.io import object_sha
from choreobench.request_budget import guard_request
from choreobench.timing import event_clock


def request_fixture():
    return {"call_id": "fixture/coordinate/u0", "seed": 7, "max_output_tokens": 512,
            "response_kind": "coordination_unit", "messages": [{"role": "user", "content": "Synthetic input."}]}


def encoded_characters(request):
    return len(json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))


def test_exact_complete_serialized_boundary_and_passthrough_identity():
    request = request_fixture()
    request["response_schema"] = {"type": "object", "additionalProperties": False,
                                  "properties": {"answer": {"type": "string"}}}
    before = object_sha(request)
    count = encoded_characters(request)
    assert guard_request(request, {"max_request_characters": count}) is request
    with pytest.raises(ModelOutputError, match="Complete serialized harness request exceeds"):
        guard_request(request, {"max_request_characters": count - 1})
    assert object_sha(request) == before


def test_schema_and_future_request_fields_are_charged_even_with_tiny_message():
    request = request_fixture()
    old_count = encoded_characters(request)
    request["response_schema"] = {"description": "x" * 1000}
    with pytest.raises(ModelOutputError, match="including response_schema"):
        guard_request(request, {"max_request_characters": old_count})
    request.pop("response_schema")
    request["future_request_metadata"] = {"description": "y" * 1000}
    with pytest.raises(ModelOutputError):
        guard_request(request, {"max_request_characters": old_count})


def test_json_escaping_and_unicode_characters_have_an_explicit_non_token_unit():
    request = request_fixture()
    request["messages"][0]["content"] = '音乐🎵\n"quoted"\\path'
    count = encoded_characters(request)
    encoded = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert len(encoded.encode("utf-8")) > count
    assert count > sum(len(m["content"]) for m in request["messages"])
    assert guard_request(request, {"max_request_characters": count}) is request
    with pytest.raises(ModelOutputError):
        guard_request(request, {"max_request_characters": count - 1})


@pytest.mark.parametrize("cfg", [None, {}, {"max_request_characters": True}, {"max_request_characters": False},
    {"max_request_characters": 0}, {"max_request_characters": -1}, {"max_request_characters": 100.0},
    {"max_request_characters": "100"}, {"max_request_characters": float("nan")}])
def test_invalid_budget_is_a_configuration_error(cfg):
    with pytest.raises(ValueError, match="explicit positive integer"):
        guard_request(request_fixture(), cfg)


@pytest.mark.parametrize("invalid", [None, [], "request", {1: "non-string JSON key"},
    {"value": float("nan")}, {"value": float("inf")}, {"value": (1, 2)}, {"value": b"bytes"}])
def test_non_json_request_is_rejected_without_dispatch(invalid):
    with pytest.raises(ModelOutputError, match="finite JSON"):
        guard_request(invalid, {"max_request_characters": 10000})


def test_cycle_is_a_controlled_request_error():
    request = request_fixture()
    request["cycle"] = request
    with pytest.raises(ModelOutputError, match="finite JSON"):
        guard_request(request, {"max_request_characters": 10000})


def structured_request(policy, step):
    unit = dict(unit_id="u0", start=0, end=20 * 16000)
    clock = event_clock(unit, 16000, dict(event_time_format="boundary_ids", event_grid_seconds=step,
        min_event_seconds=.5, event_schedule_policy=policy, max_events_per_unit=3))
    request = request_fixture()
    request["messages"][0]["content"] = json.dumps({"event_clock": clock, "text": "Synthetic proposals only."})
    request["response_schema"] = unit_response_schema(unit, [], ["c0", "c1"], max_events=3, clock=clock)
    return request


@pytest.mark.parametrize("policy,step", [("balanced_full_unit", .25), ("balanced_full_unit", .005), ("free", .25)])
def test_normal_schedules_fit_default_full_request_budget(policy, step):
    request = structured_request(policy, step)
    before = copy.deepcopy(request)
    assert guard_request(request, {"max_request_characters": 120000}) is request
    assert request == before


def test_dense_free_schema_regression_is_rejected_before_provider_dispatch():
    request = structured_request("free", .005)
    assert sum(len(m["content"]) for m in request["messages"]) < 120000
    assert encoded_characters(request) > 1_000_000
    called = []
    with pytest.raises(ModelOutputError, match="not a tokenizer/context limit"):
        checked = guard_request(request, {"max_request_characters": 120000})
        called.append(checked)  # A dispatch placed after the guard must be unreachable.
    assert called == []
