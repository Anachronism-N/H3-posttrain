"""Typed phase/concurrent/finish proposals versus same-budget free action text.

Fields encode proposed organization, not measured movement or physical state.
The compiler preserves text literally and does not create extra motion facts.
"""
import argparse
import copy
import json

from .duration_content import prepare_duration_requests
from .io import json_text, read_json
from .memory_replay import run_memory_replay
from .providers import parse_json_response
from .request_budget import guard_request
from .schema import keys, text


RULES = (
    " This is a MOVEMENT-PHASE experiment. Each event has a phase_budget from its fixed duration. "
    "A phase is one primary ongoing motion or one portion of a motion, not a full sequence of "
    "turning, landing, crouching and touching. Up to two concurrent modifiers may describe "
    "other body parts moving simultaneously, not successive new actions. A finish may be a "
    "brief hold or clear endpoint of that same movement, never a new jump/turn after stopping. "
    "Use at most phase_budget primary successive phases and communicate simultaneous relations "
    "explicitly. Do not compress several phases into one long phrase or opaque abbreviations. "
    "Use a maximum of eight whitespace-delimited words for each primary phase, four words for "
    "each concurrent modifier, and six words for a finish; total event action is still bounded "
    "by the common 32-word/240-character limit. Preserve meaningful motion, not only stillness. "
    "These phase counts are engineering hypotheses, not measured feasibility. Empty source "
    "entry/exit states remain unknown. Music and prior text remain unverified proposals. "
)
FORM = (
    " For this arm replace the output action field with phases. Each phase object has exactly "
    "main (string), concurrent (array of zero to two strings), finish (string or null). "
    "Return only those typed fields, not an action string or explanatory text. The controller "
    "joins them literally as main while concurrent; finish, using Then only between phases. "
)


def _phase_schema(budget):
    phase = dict(type="object", additionalProperties=False, required=["main", "concurrent", "finish"],
        properties=dict(main=dict(type="string", minLength=1, maxLength=64),
            concurrent=dict(type="array", maxItems=2, items=dict(type="string", minLength=1, maxLength=36)),
            finish=dict(anyOf=[dict(type="string", minLength=1, maxLength=48), dict(type="null")])))
    return dict(type="array", minItems=1, maxItems=budget, items=phase)


def prepare_phase_requests(inventory_path, sources, source_run):
    cases, original_requests = prepare_duration_requests(inventory_path, sources, source_run)
    requests = []
    for index, case in enumerate(cases):
        original = original_requests[2 * index][2]
        durations = [(i["end"] - i["start"]) / case["before"]["sample_rate"] for i in case["fixed_intervals"]]
        budgets = [1 if seconds <= 1.5 else 2 for seconds in durations]
        payload = json.loads(original["messages"][-1]["content"])
        payload["movement_phase_contract"] = dict(phase_budgets=budgets, max_main_words=8,
            max_concurrent_modifiers=2, max_modifier_words=4, max_finish_words=6,
            finish_cannot_introduce_new_movement=True, measured_feasibility=False)
        for mode in ("text_control", "phase_structured"):
            request = copy.deepcopy(original)
            request["call_id"] = case["song_id"] + "/motion-phase/" + case["unit_id"] + "/" + mode
            request["messages"][0]["content"] += RULES + (FORM if mode == "phase_structured" else "")
            view = copy.deepcopy(payload)
            probe = dict(copy.deepcopy(case), phase_budgets=budgets, phase_representation=mode == "phase_structured")
            if probe["phase_representation"]:
                fields = view["output_contract"]["fields"]
                fields[fields.index("action")] = "phases"
                alternatives = request["response_schema"]["properties"]["events"]["anyOf"]
                if len(alternatives) != 1 or len(alternatives[0]["prefixItems"]) != len(budgets):
                    raise ValueError("Phase probe needs the one frozen schedule")
                for slot, budget in zip(alternatives[0]["prefixItems"], budgets):
                    slot["properties"].pop("action")
                    slot["properties"]["phases"] = _phase_schema(budget)
                    slot["required"][slot["required"].index("action")] = "phases"
            request["messages"][-1]["content"] = json.dumps(view, ensure_ascii=False, sort_keys=True, allow_nan=False)
            guard_request(request, case["config"])
            requests.append((probe, mode, request))
    return cases, requests


def decode_phases(case, response):
    value = parse_json_response(response)
    keys(value, ("unit_id", "events"), where="phase unit")
    if not isinstance(value["events"], list) or len(value["events"]) != len(case["phase_budgets"]):
        raise ValueError("Typed phase response changed the frozen event count")
    for event, budget in zip(value["events"], case["phase_budgets"]):
        keys(event, ("start_boundary", "end_boundary", "track", "phases", "motif_id", "source_call_ids"), where="phase event")
        phases = event["phases"]
        if not isinstance(phases, list) or not 1 <= len(phases) <= budget:
            raise ValueError("Typed action exceeds the declared primary phase budget")
        phrases = []
        for phase in phases:
            keys(phase, ("main", "concurrent", "finish"), where="movement phase")
            text(phase["main"], "phase.main")
            if len(phase["main"]) > 64 or len(phase["main"].split()) > 8:
                raise ValueError("Primary phase exceeds the common component writing budget")
            concurrent = phase["concurrent"]
            if not isinstance(concurrent, list) or len(concurrent) > 2:
                raise ValueError("Too many concurrent modifiers")
            for modifier in concurrent:
                text(modifier, "phase.concurrent")
                if len(modifier) > 36 or len(modifier.split()) > 4:
                    raise ValueError("Concurrent modifier exceeds its writing budget")
            finish = phase["finish"]
            if finish is not None:
                text(finish, "phase.finish")
                if len(finish) > 48 or len(finish.split()) > 6:
                    raise ValueError("Finish exceeds its writing budget")
            phrase = phase["main"]
            if concurrent:
                phrase += " while " + " and ".join(concurrent)
            if finish is not None:
                phrase += "; " + finish
            phrases.append(phrase)
        event.pop("phases")
        event["action"] = ". Then ".join(phrases) + "."
    return json_text(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run_memory_replay(args.inventory, read_json(args.provider), args.out_dir, args.resume,
        duration_source_run=args.source_run, motion_phase_probe=True)
    print(json_text({k: result[k] for k in ("complete", "calls_used", "rows", "quality_verified")}))


if __name__ == "__main__":
    main()
