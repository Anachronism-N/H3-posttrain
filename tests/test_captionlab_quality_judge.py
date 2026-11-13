"""Synthetic CPU tests; these do not establish metric validity."""
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from captionlab import quality_judge as module
from captionlab.choreo_quality import make_review_item, validate_review
from choreobench.io import read_json, write_json


def item(reference=True, status="ok", case="c1"):
    return make_review_item(case, "source", "hidden-method", "The left wrist rises, then the elbow bends.",
        reference_text="The left wrist rises." if reference else None, audio_sha256="a" * 64,
        generation_status=status, duration_seconds=5.0625)


def answer(reference=True):
    rating = dict(status="rated", score=4, reason="Clear action progression.",
                  text_quotes=["The left wrist rises"], reference_quotes=[])
    value = {name: copy.deepcopy(rating) for name in module.TEXT_DIMENSIONS}
    value["reference_fidelity"] = dict(status="rated" if reference else "not_applicable",
        score=4 if reference else None, reason="Partial matching action." if reference else "No reference supplied.",
        text_quotes=["The left wrist rises"] if reference else [],
        reference_quotes=["The left wrist rises."] if reference else [])
    return value


def fake(mutate=None):
    calls = []
    identity = dict(kind="chat_completions", synthetic=True, model="synthetic-test", remote_weight_identity_verified=False)
    def generate(request):
        calls.append(copy.deepcopy(request))
        value = answer()
        if callable(mutate):
            mutate(value)
        if mutate == "exception":
            raise OSError("Synthetic transport failure")
        if mutate == "drift":
            identity["model"] = "changed"
        return dict(text=json.dumps(value) if mutate != "invalid_json" else "not JSON",
            identity=copy.deepcopy(identity) if mutate != "identity" else dict(kind="wrong", synthetic=True),
            synthetic=True, raw=dict(fixture=True), costs=dict(input_tokens=100, output_tokens=60),
            stop_reason={"truncated": "budget_truncation", "http": "http_error"}.get(mutate if isinstance(mutate, str) else "", "stop"))
    return calls, lambda *args, **kwargs: SimpleNamespace(identity=identity, generate=generate)


@pytest.fixture
def source(tmp_path):
    path, config, out = tmp_path / "items.json", tmp_path / "provider.json", tmp_path / "out"
    write_json(path, dict(items=[item(), item(status="failed", case="c2")]))
    write_json(config, dict(kind="chat_completions", model="synthetic-test", base_url="https://example.invalid/v1"))
    return path, config, out


def run(source, factory=None, **kw):
    return module.run_judge(source[0], source[1], "judge1", source[2], provider_factory=factory, **kw)


def forbidden(*args, **kw):
    pytest.fail("Provider unexpectedly constructed/called")


def change(path, fn):
    value = read_json(path); fn(value); write_json(path, value, overwrite=True)


def test_valid_review_has_no_audio_and_separate_reference():
    target = item()
    review = module.parse_review(target, json.dumps(answer()), "judge", "identity")
    assert review == validate_review(review, target)
    assert review["origin"] == "model_judge"
    assert review["ratings"]["music_fit"] == dict(status="not_applicable", score=None, reason="Text-only judge: no audio supplied or heard.")
    assert not review["evidence"]["audio_seen"] and review["evidence"]["audio_sha256"] is None
    assert review["reference_fidelity"]["score"] == 4


def test_no_reference():
    review = module.parse_review(item(False), json.dumps(answer(False)), "judge", "identity")
    assert review["reference_fidelity"]["status"] == "not_applicable"
    assert not review["evidence"]["reference_seen"]


@pytest.mark.parametrize("mutation", [
    lambda v: v.update(music_fit=dict(score=5)),
    lambda v: v["action_clarity"].update(score=True),
    lambda v: v["action_clarity"].update(score=6),
    lambda v: v["action_clarity"].update(status="unknown"),
    lambda v: v["action_clarity"].update(status="not_applicable", score=None),
    lambda v: v["action_clarity"].update(text_quotes=[]),
    lambda v: v["action_clarity"].update(text_quotes=["Invented quotation"]),
    lambda v: v["action_clarity"].update(text_quotes=[""]),
    lambda v: v["action_clarity"].update(reference_quotes=["The left wrist rises."]),
    lambda v: v["reference_fidelity"].update(reference_quotes=[]),
    lambda v: v["reference_fidelity"].update(reference_quotes=["No such reference phrase"]),
    lambda v: v["reference_fidelity"].update(status="not_applicable", score=None),
    lambda v: v["temporal_organization"].update(reason=""),
    lambda v: v["temporal_organization"].update(reason="x" * 1001),
    lambda v: v["temporal_organization"].update(extra=True),
])
def test_invalid_or_hallucinated_evidence_rejected(mutation):
    value = answer(); mutation(value)
    with pytest.raises(ValueError):
        module.parse_review(item(), json.dumps(value), "judge", "identity")


def test_unknown_not_filled_with_score():
    value = answer()
    value["action_clarity"].update(status="unknown", score=None, text_quotes=[])
    review = module.parse_review(item(), json.dumps(value), "judge", "identity")
    assert review["ratings"]["action_clarity"]["score"] is None


def test_duplicate_keys_rejected():
    raw = json.dumps(answer())
    with pytest.raises(ValueError, match="Duplicate"):
        module.parse_review(item(), raw[:-1] + ',"action_clarity":null}', "judge", "identity")


def test_absent_reference_cannot_be_scored():
    with pytest.raises(ValueError):
        module.parse_review(item(False), json.dumps(answer()), "judge", "identity")


@pytest.mark.parametrize("status", ["failed", "truncated", "unknown", "not_run"])
def test_generation_failures_cannot_be_scored(status):
    with pytest.raises(ValueError):
        module.parse_review(item(status=status), json.dumps(answer()), "judge", "identity")


def test_journal_keeps_denominator_and_resume_has_zero_calls(source):
    calls, factory = fake()
    result = run(source, factory)
    assert len(calls) == result["calls_used"] == result["valid_reviews"] == 1
    assert result["expected_items"] == 2 and result["generation_not_evaluable"] == 1
    assert result["human_reviews"] == 0 and not result["audio_supplied"] and not result["metric_validity_verified"]
    assert len(read_json(source[2] / "reviews.json")["reviews"]) == 1
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert set(payload) == {"text", "duration_seconds", "reference_text"}
    assert "hidden-method" not in json.dumps(calls[0])
    assert "untrusted DATA" in calls[0]["messages"][0]["content"]
    assert run(source, forbidden, resume=True) == result
    with pytest.raises(FileExistsError):
        run(source, forbidden)


@pytest.mark.parametrize("behavior,status", [("exception", "unknown"), ("http", "unknown"), ("truncated", "invalid"), ("invalid_json", "invalid"),
    (lambda v: v["action_clarity"].update(text_quotes=["Invented"]), "invalid")])
def test_failures_retained_and_never_retried(source, behavior, status):
    calls, factory = fake(behavior)
    result = run(source, factory)
    assert len(calls) == 1 and result["rows"][0]["status"] == status
    assert result["valid_reviews"] == 0
    assert read_json(source[2] / "reviews.json") == {"reviews": []}
    assert run(source, forbidden, resume=True) == result


@pytest.mark.parametrize("behavior", ["identity", "drift"])
def test_identity_drift_is_fatal(source, behavior):
    _, factory = fake(behavior)
    with pytest.raises(ValueError, match="identity"):
        run(source, factory)


def test_pending_call_is_not_dispatched(source):
    _, factory = fake()
    run(source, factory)
    (source[2] / "results.json").unlink()
    next((source[2] / "calls").glob("*.raw.json")).unlink()
    with pytest.raises(ValueError, match="Pending/unknown"):
        run(source, forbidden, resume=True)


@pytest.mark.parametrize("artifact", ["reviews.json", "results.json"])
def test_published_tamper_rejected_before_provider(source, artifact):
    _, factory = fake()
    run(source, factory)
    change(source[2] / artifact, lambda v: v.update(tampered=True))
    with pytest.raises(ValueError, match="evidence changed"):
        run(source, forbidden, resume=True)


def test_input_tamper_rejected_before_provider(source):
    _, factory = fake()
    run(source, factory)
    change(source[0], lambda v: v["items"].append(item(case="added")))
    with pytest.raises(ValueError, match="identity changed"):
        run(source, forbidden, resume=True)


def test_duplicate_item_rejected_before_provider(source):
    change(source[0], lambda v: v["items"].append(copy.deepcopy(v["items"][0])))
    with pytest.raises(ValueError, match="Duplicate"):
        run(source, forbidden)


def test_nonevaluable_inventory_does_not_call_provider_generate(source):
    write_json(source[0], [item(status="unknown")], overwrite=True)
    calls, factory = fake()
    result = run(source, factory)
    assert not calls and result["generation_not_evaluable"] == 1
    assert run(source, forbidden, resume=True) == result


def test_partial_resume_uses_sealed_raw_not_provider(source):
    _, factory = fake()
    result = run(source, factory)
    (source[2] / "results.json").unlink()
    (source[2] / "reviews.json").unlink()
    assert run(source, forbidden, resume=True) == result


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as error:
        module.main(["run", "--help"])
    assert error.value.code == 0 and "--provider-config" in capsys.readouterr().out


def test_real_chat_transport_is_text_only_and_closed_schema(source):
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(dict(choices=[dict(message=dict(content=json.dumps(answer())), finish_reason="stop")],
                                   usage=dict(prompt_tokens=100, completion_tokens=60))).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        write_json(source[1], dict(kind="chat_completions", model="cpu-http-fixture",
            base_url="http://127.0.0.1:%d/v1" % server.server_port, api_key_env="", timeout_seconds=3,
            temperature=0), overwrite=True)
        result = run(source)
        assert result["valid_reviews"] == 1 and len(received) == 1
        body = received[0]
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        assert all(isinstance(message["content"], str) for message in body["messages"])
        assert "audio_url" not in json.dumps(body)
        assert run(source, forbidden, resume=True) == result
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)


def test_cli_synthetic_replay_end_to_end(source, capsys):
    replay = source[0].parent / "replay.json"
    write_json(replay, dict(schema="choreobench.replay.v1", synthetic=True, source=dict(test=True),
        responses={"quality/" + item()["item_sha256"]: dict(text=json.dumps(answer()), stop_reason="stop")}))
    write_json(source[1], dict(kind="replay", path=str(replay)), overwrite=True)
    argv = ["run", "--items", str(source[0]), "--provider-config", str(source[1]),
            "--reviewer-id", "judge1", "--out-dir", str(source[2])]
    module.main(argv)
    assert json.loads(capsys.readouterr().out)["valid_reviews"] == 1
    assert read_json(source[2] / "results.json")["synthetic"] is True
    module.main(argv + ["--resume"])
    assert json.loads(capsys.readouterr().out)["calls_used"] == 1


def reference_item(reference=True):
    return make_review_item("c1", "source", "hidden-method", "The left wrist rises, then the elbow bends.",
        reference_text="The left wrist rises." if reference else None, audio_sha256="a" * 64,
        duration_seconds=5.0625, provenance={"review_phase": "reference"})


def reference_answer():
    value = answer()
    for name in module.TEXT_DIMENSIONS[:2]:
        value[name] = dict(status="not_applicable", score=None, reason="Reference-only phase.",
                           text_quotes=[], reference_quotes=[])
    return value


def test_reference_phase_scores_only_reference_and_has_correct_schema():
    target = reference_item()
    review = module.parse_review(target, json.dumps(reference_answer()), "judge", "identity")
    assert all(rating["status"] == "not_applicable" for rating in review["ratings"].values())
    assert review["reference_fidelity"]["score"] == 4
    assert review == validate_review(review, target)
    request = module._request(target, 42, 1800)
    assert "REFERENCE-ONLY" in request["messages"][0]["content"]
    assert request["response_schema"]["properties"]["action_clarity"]["properties"]["score"] == {"type": "null"}


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(action_clarity=answer()["action_clarity"]),
    lambda value: value["temporal_organization"].update(status="unknown"),
    lambda value: value["action_clarity"].update(text_quotes=["The left wrist rises"]),
])
def test_reference_phase_rejects_rescoring_quality(mutation):
    value = reference_answer(); mutation(value)
    with pytest.raises(ValueError, match="Reference-only"):
        module.parse_review(reference_item(), json.dumps(value), "judge", "identity")


def test_reference_phase_no_reference_rejected_before_dispatch(source):
    write_json(source[0], [reference_item(False)], overwrite=True)
    with pytest.raises(ValueError, match="nonempty bound reference"):
        run(source, forbidden)


def test_reference_phase_runner_retains_na_for_bundle_unblind(source):
    write_json(source[0], [reference_item()], overwrite=True)
    calls, factory = fake(lambda value: value.update(reference_answer()))
    result = run(source, factory)
    assert len(calls) == result["valid_reviews"] == 1
    row = read_json(source[2] / "reviews.json")["reviews"][0]
    assert all(rating["status"] == "not_applicable" for rating in row["ratings"].values())
    assert run(source, forbidden, resume=True) == result


def test_roster_no_generate_and_same_identity_as_actual_run(source, monkeypatch):
    calls, factory = fake()
    monkeypatch.setattr(module, "create_provider", factory)
    roster = module.make_roster(source[1], "judge1", ["h1", "h2"])
    assert not calls
    assert roster["h1"] == {"origin": "human", "judge_identity": None}
    result = run(source)
    assert roster["judge1"] == {"origin": "model_judge", "judge_identity": result["judge_identity"]}


@pytest.mark.parametrize("humans,judge", [(["h1", "h1"], "j"), (["h1"], "h1"), ([""], "j"), ([], "j"), (["h1"], "")])
def test_invalid_roster_identity_rejected_before_provider(source, monkeypatch, humans, judge):
    monkeypatch.setattr(module, "create_provider", forbidden)
    with pytest.raises(ValueError):
        module.make_roster(source[1], judge, humans)


def test_roster_cli_without_remote_calls(source, capsys):
    path = source[0].parent / "roster.json"
    module.main(["roster", "--provider-config", str(source[1]), "--reviewer-id", "judge1",
                 "--human", "h1", "--human", "h2", "--out", str(path)])
    assert read_json(path) == json.loads(capsys.readouterr().out)
    assert set(read_json(path)) == {"h1", "h2", "judge1"}


def test_request_quotation_inventory_is_exact_bounded_and_not_a_semantic_label():
    target = make_review_item("quote-control", "source", "hidden-method", "Arm moves. " + "x" * 805 + "!\nStep left?", duration_seconds=5.0625)
    request = module._request(target, 42, 1800)
    properties = request["response_schema"]["properties"]
    quotes = properties["action_clarity"]["properties"]["text_quotes"]["items"]["enum"]
    assert quotes and all(q in target["text"] and 0 < len(q) <= 400 for q in quotes)
    assert quotes == properties["temporal_organization"]["properties"]["text_quotes"]["items"]["enum"]
    assert "Arm moves." in quotes and "Step left?" in quotes
    assert properties["action_clarity"]["properties"]["status"]["enum"] == ["rated", "unknown"]
    assert properties["reference_fidelity"]["properties"]["status"]["enum"] == ["not_applicable"]
    assert all(p["properties"]["reference_quotes"]["maxItems"] == 0 for p in properties.values())
    assert json.loads(request["messages"][1]["content"]) == {"text": target["text"], "duration_seconds": 5.0625, "reference_text": None}
    assert "hidden-method" not in json.dumps(request)


def test_reference_only_quote_inventory_keeps_quality_dimensions_na():
    target = reference_item()
    properties = module._request(target, 42, 1800)["response_schema"]["properties"]
    for name in module.TEXT_DIMENSIONS[:2]:
        assert properties[name]["properties"]["text_quotes"]["maxItems"] == 0
        assert properties[name]["properties"]["reference_quotes"]["maxItems"] == 0
    for field, source in [("text_quotes", target["text"]), ("reference_quotes", target["reference_text"])]:
        assert all(q in source for q in properties["reference_fidelity"]["properties"][field]["items"]["enum"])
    assert module.parse_review(target, json.dumps(reference_answer()), "judge", "identity")["reference_fidelity"]["score"] == 4


def test_quotation_choices_strip_schedule_prefix_and_preserve_decimals():
    source = "[0.000-3.250 s] Hold for 1.5 seconds while arms circle.\n[3.250-5.062 s] Step left, then pause."
    choices = module.quotation_choices(source, 300)
    assert choices == ["Hold for 1.5 seconds while arms circle.", "Step left, then pause."]
    assert all(q in source for q in choices)
    assert "[0." not in choices and not any(q.startswith("[") for q in choices)
    assert module.quotation_choices("Use image-space motion (e.g., wrist leftward), not anatomical side.") == ["Use image-space motion (e.g., wrist leftward), not anatomical side."]
    assert module.quotation_choices("[0.000-5.062 s] ") == []
    item = make_review_item("timed-quotes", "source", "hidden", source, duration_seconds=5.0625)
    schema = module._request(item, 42, 1800)["response_schema"]
    for dimension in module.TEXT_DIMENSIONS[:2]:
        menu = schema["properties"][dimension]["properties"]["text_quotes"]["items"]["enum"]
        assert menu == choices


@pytest.mark.parametrize("limit", [0, -1, True])
def test_quotation_choices_reject_invalid_character_limit(limit):
    with pytest.raises(ValueError):
        module.quotation_choices("Move.", limit)


def paired_items():
    left = make_review_item("pair-source", "shared-source", "workflow-method", "[0.000-5.062 s] Step left while both arms circle.", duration_seconds=5.0625, audio_sha256="a" * 64)
    right = make_review_item("pair-source", "shared-source", "choreography-method", "[0.000-5.062 s] Hold both feet still while repeatedly jumping.", duration_seconds=5.0625, audio_sha256="a" * 64)
    return left, right


def test_pairwise_request_is_anonymous_order_invariant_and_excludes_clock_evidence():
    left, right = paired_items()
    request = module.pairwise_request(left, right)
    assert request == module.pairwise_request(right, left)
    payload = json.loads(request["messages"][-1]["content"])
    assert set(payload) == {"A", "B", "duration_seconds"}
    assert {payload["A"], payload["B"]} == {left["text"], right["text"]}
    serialized = json.dumps(request)
    assert "workflow-method" not in serialized and "choreography-method" not in serialized
    assert "shared-source" not in serialized and "a" * 64 not in serialized
    for label in ["A", "B"]:
        choices = request["response_schema"]["properties"]["quotes_" + label]["items"]["enum"]
        assert choices and all(q in payload[label] and not q.startswith("[") for q in choices)


@pytest.mark.parametrize("preference", ["A", "B", "tie", "unknown"])
def test_pairwise_parser_preserves_preference_side_tie_and_unknown(preference):
    left, right = paired_items();request = module.pairwise_request(left, right)
    payload = json.loads(request["messages"][-1]["content"])
    value = dict(preference=preference, reason="Synthetic comparison", quotes_A=[], quotes_B=[])
    if preference != "unknown":
        for label in ["A", "B"]:
            value["quotes_" + label] = request["response_schema"]["properties"]["quotes_" + label]["items"]["enum"][:1]
    response = dict(text=json.dumps(value), stop_reason="stop")
    review = module.parse_pairwise_review(left, right, response, "fixture-judge", "fixture-identity")
    assert review["human_reviews"] == 0 and not review["metric_validity_verified"]
    assert not review["audio_supplied"] and not review["reference_supplied"]
    if preference in ["tie", "unknown"]:
        assert review["preference"] == preference and review["preferred_item_sha256"] is None
    else:
        expected = "left" if payload[preference] == left["text"] else "right"
        assert review["preference"] == expected
        reversed_review = module.parse_pairwise_review(right, left, response, "fixture-judge", "fixture-identity")
        assert reversed_review["preference"] != review["preference"]
        assert reversed_review["preferred_item_sha256"] == review["preferred_item_sha256"]


@pytest.mark.parametrize("change", ["source", "audio", "duration", "same_method", "failed", "reference_phase"])
def test_pairwise_request_rejects_mismatched_or_unreviewable_sources(change):
    left, right = paired_items()
    kwargs = dict(audio_sha256=right["audio_sha256"], duration_seconds=right["duration_seconds"])
    case, group, method, status, provenance = right["case_id"], right["source_group_id"], right["method_id"], "ok", {}
    if change == "source": group = "other-source"
    elif change == "audio": kwargs["audio_sha256"] = "b" * 64
    elif change == "duration": kwargs["duration_seconds"] = 10.125
    elif change == "same_method": method = left["method_id"]
    elif change == "failed": status = "failed"
    else: provenance = {"review_phase": "reference"}
    changed = make_review_item(case, group, method, right["text"], generation_status=status, provenance=provenance, **kwargs)
    with pytest.raises(ValueError): module.pairwise_request(left, changed)


@pytest.mark.parametrize("change", ["clock", "wrong_side", "missing_side", "invalid_choice", "truncated", "extra"])
def test_pairwise_parser_rejects_fabricated_clock_wrong_side_and_incomplete_evidence(change):
    left, right = paired_items();request = module.pairwise_request(left, right)
    value = dict(preference="A", reason="Synthetic comparison", quotes_A=request["response_schema"]["properties"]["quotes_A"]["items"]["enum"][:1], quotes_B=request["response_schema"]["properties"]["quotes_B"]["items"]["enum"][:1])
    stop = "stop"
    if change == "clock": value["quotes_A"] = ["[0."]
    elif change == "wrong_side": value["quotes_A"] = value["quotes_B"]
    elif change == "missing_side": value["quotes_B"] = []
    elif change == "invalid_choice": value["preference"] = "best"
    elif change == "truncated": stop = "budget_truncation"
    else: value["music_fit"] = 5
    with pytest.raises(ValueError): module.parse_pairwise_review(left, right, dict(text=json.dumps(value), stop_reason=stop), "judge", "identity")


def pairwise_material(tmp_path, behavior=None):
    left, right = paired_items()
    missing_left = make_review_item("missing-source", "other-group", "workflow", "", generation_status="failed", duration_seconds=5.0625)
    missing_right = make_review_item("missing-source", "other-group", "harness", "Move left.", duration_seconds=5.0625)
    pairs = tmp_path / "pairs.json"; config = tmp_path / "provider.json"; out = tmp_path / "pairs-run"
    write_json(pairs, dict(pairs=[dict(left=left, right=right), dict(left=missing_left, right=missing_right)]))
    write_json(config, dict(kind="chat_completions", model="pairwise-fixture"))
    calls = []; identity = dict(kind="chat_completions", synthetic=True, model="pairwise-fixture")
    def generate(request):
        calls.append(copy.deepcopy(request))
        if behavior == "exception": raise OSError("Fixture transport failure")
        value = dict(preference="tie" if behavior != "unknown" else "unknown", reason="Fixture only",
            quotes_A=request["response_schema"]["properties"]["quotes_A"]["items"]["enum"][:1],
            quotes_B=request["response_schema"]["properties"]["quotes_B"]["items"]["enum"][:1])
        if behavior == "bad_quote": value["quotes_A"] = ["[0."]
        return dict(text=json.dumps(value), raw={}, identity=identity if behavior != "identity" else dict(identity, model="other"),
            synthetic=True, stop_reason={"truncated": "budget_truncation", "http": "http_error"}.get(behavior, "stop"),
            costs=dict(input_tokens=10, output_tokens=20))
    factory = lambda *args, **kwargs: SimpleNamespace(identity=identity, generate=generate)
    return pairs, config, out, calls, factory


@pytest.mark.parametrize("behavior,expected", [(None, "valid"), ("unknown", "valid"), ("bad_quote", "invalid"),
                                              ("truncated", "invalid"), ("http", "unknown_infrastructure"), ("exception", "unknown_infrastructure")])
def test_pairwise_runner_keeps_denominator_raw_failures_and_zero_call_resume(tmp_path, behavior, expected):
    pairs, config, out, calls, factory = pairwise_material(tmp_path, behavior)
    result = module.run_pairwise(pairs, config, "fixture-judge", out, provider_factory=factory)
    assert len(calls) == result["calls_used"] == 1 and result["expected_pairs"] == 2
    assert result["status_counts"] == {"generation_not_evaluable": 1, expected: 1}
    assert result["human_reviews"] == 0 and not result["metric_validity_verified"]
    assert module.run_pairwise(pairs, config, "fixture-judge", out, provider_factory=forbidden, resume=True) == result
    assert len(calls) == 1
    if behavior == "unknown": assert result["preference_counts"]["unknown"] == 1
    if behavior is None: assert result["preference_counts"]["tie"] == 1


@pytest.mark.parametrize("artifact", ["raw", "reviews", "source", "budget"])
def test_pairwise_completed_tamper_rejected_without_dispatch(tmp_path, artifact):
    pairs, config, out, calls, factory = pairwise_material(tmp_path)
    module.run_pairwise(pairs, config, "fixture-judge", out, provider_factory=factory)
    kw = {}
    if artifact == "raw": next((out / "calls").glob("*.raw.json")).unlink()
    elif artifact == "reviews": (out / "reviews.json").unlink()
    elif artifact == "source": change(pairs, lambda v: v["pairs"].reverse())
    else: kw["max_output_tokens"] = 1801
    with pytest.raises(ValueError):
        module.run_pairwise(pairs, config, "fixture-judge", out, resume=True, provider_factory=forbidden, **kw)
    assert len(calls) == 1


def test_pairwise_reversed_duplicate_slot_rejected_before_provider(tmp_path):
    pairs, config, out, calls, _ = pairwise_material(tmp_path)
    value = read_json(pairs); pair = value["pairs"][0]
    value["pairs"].append(dict(left=pair["right"], right=pair["left"]))
    write_json(pairs, value, overwrite=True)
    with pytest.raises(ValueError, match="Duplicate"):
        module.run_pairwise(pairs, config, "fixture-judge", out, provider_factory=forbidden)
    assert not calls


def test_pairwise_returned_identity_drift_cannot_become_quality_failure(tmp_path):
    pairs, config, out, calls, factory = pairwise_material(tmp_path, "identity")
    with pytest.raises(ValueError, match="identity"):
        module.run_pairwise(pairs, config, "fixture-judge", out, provider_factory=factory)
    assert len(calls) == 1 and not (out / "results.json").exists()


def test_pairwise_all_failed_generations_do_not_construct_provider(tmp_path):
    pairs, config, out, calls, _ = pairwise_material(tmp_path)
    value = read_json(pairs); value["pairs"] = value["pairs"][1:]; write_json(pairs, value, overwrite=True)
    result = module.run_pairwise(pairs, config, "fixture-judge", out, provider_factory=forbidden)
    assert result["calls_used"] == 0 and result["expected_pairs"] == 1
    assert result["status_counts"] == {"generation_not_evaluable": 1}
    assert module.run_pairwise(pairs, config, "fixture-judge", out, resume=True, provider_factory=forbidden) == result
    assert not calls


def test_pairwise_cli_dispatch_and_resume(tmp_path, monkeypatch, capsys):
    pairs, config, out, calls, factory = pairwise_material(tmp_path)
    monkeypatch.setattr(module, "create_provider", factory)
    argv = ["pairwise", "--pairs", str(pairs), "--provider-config", str(config), "--reviewer-id", "fixture", "--out-dir", str(out)]
    result = module.main(argv)
    assert json.loads(capsys.readouterr().out)["expected_pairs"] == 2
    assert result["calls_used"] == len(calls) == 1
    monkeypatch.setattr(module, "create_provider", forbidden)
    assert module.main(argv + ["--resume"]) == result
    assert json.loads(capsys.readouterr().out)["calls_used"] == 1


def test_pairwise_cli_help(capsys):
    with pytest.raises(SystemExit) as error:
        module.main(["pairwise", "--help"])
    assert error.value.code == 0 and "--pairs" in capsys.readouterr().out


def fullscore_inputs():
    def score(prefix):
        return dict(duration_seconds=9., events=[dict(index=i, start_seconds=i*3., end_seconds=(i+1)*3., track='body', action=prefix+f' movement {i}.') for i in range(3)])
    return score('Left'), score('Right')


def fullscore_answer(request, preference='A'):
    data=json.loads(request['messages'][-1]['content'])
    evidence={label:[dict(index=e['index'],quote=e['action']) for e in data[label]['events']] for label in ('A','B')}
    return {dim:dict(preference=preference,reason='Synthetic temporal coverage control',quotes_A=[] if preference=='unknown' else evidence['A'],quotes_B=[] if preference=='unknown' else evidence['B']) for dim in module.FULLSCORE_DIMENSIONS}


def test_fullscore_request_order_invariant_with_exact_three_third_evidence():
    left,right=fullscore_inputs();request=module.fullscore_request(left,right)
    assert request==module.fullscore_request(right,left)
    data=json.loads(request['messages'][-1]['content']);schema=request['response_schema']
    assert set(data)=={'A','B'} and set(schema['$defs'])=={'quotes_A','quotes_B'}
    for label in ('A','B'):
        array=schema['$defs']['quotes_'+label]['anyOf'][1]
        assert array['minItems']==array['maxItems']==3 and array['items'] is False
        for third,element in enumerate(array['prefixItems']):
            branch=element['anyOf'][0]
            assert branch['properties']['index']['enum']==[third]
            assert branch['properties']['quote']['enum']==[data[label]['events'][third]['action']]
    review=module.parse_fullscore_review(left,right,dict(text=json.dumps(fullscore_answer(request)),stop_reason='stop'),'fixture','identity')
    assert review['temporal_third_evidence_required'] and not review['metric_validity_verified']
    assert review['human_reviews']==0 and not review['audio_supplied']


@pytest.mark.parametrize('preference',['A','B','tie','unknown'])
def test_fullscore_preference_and_unknown_remain_separate(preference):
    left,right=fullscore_inputs();req=module.fullscore_request(left,right)
    response=dict(text=json.dumps(fullscore_answer(req,preference)),stop_reason='stop')
    review=module.parse_fullscore_review(left,right,response,'reviewer','identity')
    for v in review['preferences'].values():
        assert v['preference']==preference
        assert (v['preferred_score_sha256'] is None)==(preference in ('tie','unknown'))


@pytest.mark.parametrize('change',['first_only','duplicate_third','wrong_event_quote','clock_fragment','foreign_index','truncated'])
def test_fullscore_review_rejects_uncovered_or_unbound_evidence(change):
    left,right=fullscore_inputs();req=module.fullscore_request(left,right);value=fullscore_answer(req);refs=value['clarity']['quotes_A'];stop='stop'
    if change=='first_only':value['clarity']['quotes_A']=refs[:1]
    elif change=='duplicate_third':refs[1]=copy.deepcopy(refs[0])
    elif change=='wrong_event_quote':refs[1]['quote']=refs[0]['quote']
    elif change=='clock_fragment':refs[1]['quote']='[0.0-3.0 s]'
    elif change=='foreign_index':refs[1]['index']=10
    else:stop='budget_truncation'
    with pytest.raises(ValueError):module.parse_fullscore_review(left,right,dict(text=json.dumps(value),stop_reason=stop),'reviewer','identity')


@pytest.mark.parametrize('change',['missing_third','clock','index','duration','boolean_time'])
def test_fullscore_request_rejects_invalid_or_incomplete_clock(change):
    left,right=fullscore_inputs()
    if change=='missing_third':right['events']=right['events'][:2]
    elif change=='clock':right['events'][1]['end_seconds']=12.
    elif change=='index':right['events'][1]['index']=0
    elif change=='duration':right['duration_seconds']=10.
    else:right['events'][0]['start_seconds']=False
    with pytest.raises(ValueError):module.fullscore_request(left,right)
