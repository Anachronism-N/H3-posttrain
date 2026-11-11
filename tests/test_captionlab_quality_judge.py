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
