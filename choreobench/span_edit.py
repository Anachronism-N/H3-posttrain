"""Explicit source-text edit permissions, not automatically discovered facts.

A direct full-action arm and a replacement-only arm share the same span scope.
Immutable source wording is retained; contradictions introduced inside permitted
text remain a semantic evaluation problem rather than certified safe motion.
"""
import argparse
import copy
import json
from pathlib import Path
import re

from .baseline import _Journal, _sealed, _verify
from .editing import apply_task_patch, public_edit_task, validate_edit_task
from .io import atomic_write, file_lock, file_sha, object_sha, read_json, read_json_snapshot, write_json, json_text
from .memory_replay import _dispatch, _ReplayInfrastructureError, _accounting
from .providers import create_provider, parse_json_response
from .request_budget import guard_request
from .revision import _no_new_timing_literals
from .revision_protocol import public_plan
from .schema import keys, integer, text, validate_plan


MODES = ("whole_action", "source_spans")
COMMON = (
    "Edit only the supplied permitted SOURCE TEXT SPANS of one possible choreography event. "
    "The surrounding source action must remain byte-for-byte identical, in the same order. "
    "Both output formats have the SAME authorized scope: replacing only these spans. "
    "Do not answer with a promise to preserve facts; the final action must directly retain them. "
    "Use one concise replacement describing the requested upper-body movement, with no new "
    "leg/foot movements, repetition counts, timestamps, support-foot or verified-contact claims. "
    "Keep the specified concurrent/temporal relation to retained source movements. "
    "A replacement has at most 24 whitespace-delimited words and 180 Unicode characters. "
    "All other events/fields, including timing and stored states, remain unchanged. "
    "You have not heard audio or seen actual movement. Proposed source states are not ground truth. "
    "Return one complete closed JSON object according to output_contract, no Markdown or commentary."
)


def validate_span_task(document, plan):
    keys(document, ("schema", "task", "event_id", "source_action", "spans"), where="source span task")
    if document["schema"] != "choreobench.span_edit_task.v1":
        raise ValueError("Unknown span edit task")
    validate_plan(plan)
    validate_edit_task(document["task"], plan)
    event_id = document["event_id"]
    if document["task"]["allowed_event_ids"] != [event_id] or document["task"]["repair_event_ids"]:
        raise ValueError("Span editing requires exactly one explicit target without repair scope")
    event = next(e for e in plan["events"] if e["event_id"] == event_id)
    if event["action"] != document["source_action"]:
        raise ValueError("Span task source action differs from frozen score")
    if not isinstance(document["spans"], list) or not 1 <= len(document["spans"]) <= 3:
        raise ValueError("Span task requires one to three nonempty source spans")
    end, ids = 0, set()
    for span in document["spans"]:
        keys(span, ("span_id", "start", "end", "source_text"), where="authorized span")
        text(span["span_id"], "span_id")
        integer(span["start"], "span.start"); integer(span["end"], "span.end", 1)
        if span["span_id"] in ids or not end <= span["start"] < span["end"] <= len(event["action"]):
            raise ValueError("Duplicate, overlapping, unordered or out-of-range edit span")
        if event["action"][span["start"]:span["end"]] != span["source_text"]:
            raise ValueError("Span offsets do not bind actual source text")
        ids.add(span["span_id"]); end = span["end"]
    return document


def _replacement(value):
    text(value, "replacement")
    if len(value) > 180 or len(value.split()) > 24:
        raise ValueError("Replacement exceeds the common span writing ceiling")
    if any(ord(c) < 32 for c in value):
        raise ValueError("Replacement includes control/newline characters")
    return value


def compile_span_response(document, response, mode):
    value = parse_json_response(response)
    spans, source = document["spans"], document["source_action"]
    if mode == "whole_action":
        keys(value, ("event_id", "action"), where="whole action response")
        if value["event_id"] != document["event_id"]:
            raise ValueError("Whole-action response changed target identity")
        text(value["action"], "replacement action")
        cursor, pattern = 0, ""
        for span in spans:
            pattern += re.escape(source[cursor:span["start"]]) + "(.+?)"
            cursor = span["end"]
        pattern += re.escape(source[cursor:])
        match = re.fullmatch(pattern, value["action"])
        if match is None:
            raise ValueError("Whole action changes source text outside permitted spans")
        replacements = [_replacement(part) for part in match.groups()]
    elif mode == "source_spans":
        keys(value, ("event_id", "replacements"), where="span replacements")
        if value["event_id"] != document["event_id"] or not isinstance(value["replacements"], list):
            raise ValueError("Span response target or replacement array is invalid")
        if len(value["replacements"]) != len(spans):
            raise ValueError("Span response must replace each authorized span once")
        replacements = []
        for span, replacement in zip(spans, value["replacements"]):
            keys(replacement, ("span_id", "text"), where="span replacement")
            if replacement["span_id"] != span["span_id"]:
                raise ValueError("Span identity/order changed")
            replacements.append(_replacement(replacement["text"]))
    else:
        raise ValueError("Unknown span output mode")
    cursor, parts = 0, []
    for span, replacement in zip(spans, replacements):
        parts += [source[cursor:span["start"]], replacement]
        cursor = span["end"]
    parts.append(source[cursor:])
    action = "".join(parts)
    return action, [dict(span_id=s["span_id"], text=v) for s,v in zip(spans,replacements)]


def _request(plan, document, mode, seed, cap):
    task = document["task"]
    public = public_edit_task(task)
    if mode == "whole_action":
        schema = dict(type="object", additionalProperties=False, required=["event_id","action"],
            properties=dict(event_id=dict(type="string", enum=[document["event_id"]]),
                            action=dict(type="string", minLength=1, maxLength=len(document["source_action"])+540)))
        contract = dict(event_id=document["event_id"], fields=["event_id","action"],
                        instruction="Return full action; do not change any text outside the authorized spans.")
    else:
        slots = [dict(type="object", additionalProperties=False, required=["span_id","text"], properties=dict(
            span_id=dict(type="string", enum=[s["span_id"]]), text=dict(type="string", minLength=1,maxLength=180))) for s in document["spans"]]
        schema = dict(type="object", additionalProperties=False, required=["event_id","replacements"],properties=dict(
            event_id=dict(type="string",enum=[document["event_id"]]), replacements=dict(type="array",minItems=len(slots),
                maxItems=len(slots),prefixItems=slots,items=False)))
        contract = dict(event_id=document["event_id"], fields=["event_id","replacements"],
                        instruction="Return only ordered span_id/text replacements; controller preserves surrounding text.")
    payload = dict(source_plan=public_plan(plan), task=public, source_action=document["source_action"],
        authorized_spans=copy.deepcopy(document["spans"]), output_contract=contract,
        source_spans_are_author_supplied=True, numeric_offsets_are_character_positions_not_audio_times=True)
    return dict(call_id=task["task_id"]+"/"+mode,seed=seed,max_output_tokens=cap,response_kind="span_edit",
        messages=[dict(role="system",content=COMMON),dict(role="user",content=json_text(payload))],response_schema=schema)


def run_span_pair(inventory_path, provider_spec, output_dir, resume=False):
    output, path = Path(output_dir).resolve(), Path(inventory_path).resolve()
    inventory, digest = read_json_snapshot(path)
    keys(inventory, ("schema","cases","seed"), where="span experiment inventory")
    if inventory["schema"] != "choreobench.span_pair_inventory.v1" or not isinstance(inventory["cases"],list) or not 1 <= len(inventory["cases"]) <= 3:
        raise ValueError("Span pair permits one to three cases")
    integer(inventory["seed"],"seed")
    if inventory["seed"] >= 2**63:raise ValueError("Span seed outside range")
    sources, prepared, seen = {str(path):digest}, [], set()
    for case in inventory["cases"]:
        keys(case, ("plan_path","plan_sha256","task_path","task_sha256"), where="span input")
        plan, ps = read_json_snapshot(case["plan_path"]);document, ds = read_json_snapshot(case["task_path"])
        if (ps,ds) != (case["plan_sha256"],case["task_sha256"]):raise ValueError("Frozen span input bytes changed")
        validate_span_task(document,plan)
        if plan["song_id"] in seen:raise ValueError("Only one span case per song")
        seen.add(plan["song_id"])
        sources[str(Path(case["plan_path"]).resolve())]=ps;sources[str(Path(case["task_path"]).resolve())]=ds
        for mode in MODES:prepared.append((plan,document,mode,_request(plan,document,mode,inventory["seed"],2048)))
    def unchanged():
        for source,sha in sources.items():
            if file_sha(source)!=sha:raise ValueError("Span source changed: "+source)
    unchanged()
    provider=create_provider(provider_spec,role="planner",base_dir=path.parent)
    if provider.identity["kind"] not in ("chat_completions","replay"):raise ValueError("Span comparison requires text provider")
    contract=dict(schema="choreobench.span_pair_contract.v1",sources=sources,provider_identity=copy.deepcopy(provider.identity),
        code_sha256={p.name:file_sha(p) for p in Path(__file__).parent.glob("*.py")},maximum_calls=len(prepared),
        request_sha256={req["call_id"]:object_sha(req) for _,_,_,req in prepared},same_authorized_source_spans=True,
        modes=list(MODES),private_goals_visible=False,semantic_preservation_verified=False,quality_verified=False)
    if output.exists() and not resume:raise FileExistsError(str(output))
    if resume and not (output/"contract.json").is_file():raise ValueError("Resume requires bound span contract")
    output.mkdir(parents=True,exist_ok=True)
    with file_lock(output/".run.lock"):
        def save(relative,value):
            p=output/relative
            if p.exists():
                if read_json(p)!=value:raise ValueError("Span artifact changed")
            else:write_json(p,value)
        save("contract.json",contract)
        journal=_Journal(output,object_sha(contract),len(prepared))
        previous=None
        if (output/"results.json").exists():
            previous=_verify(read_json(output/"results.json"))
            if previous["complete"]:
                expected_ids={r["call_id"] for _,_,_,r in prepared}
                if set(journal.records)!=expected_ids or set(journal.started)!=expected_ids or previous["call_records"]!={k:v["record_sha256"] for k,v in journal.records.items()}:
                    raise ValueError("Completed span inventory changed; no replacement calls")
                for relative,sha in previous["artifacts"].items():
                    if file_sha(output/relative)!=sha:raise ValueError("Completed span artifact bytes changed")
        rows,artifacts, fatal=[],{},False
        for plan,document,mode,request in prepared:
            row=dict(song_id=plan["song_id"],mode=mode,event_id=document["event_id"],status="pending",new_plan_sha256=None)
            prefix=plan["song_id"]+"/"+mode+"/"
            try:
                guard_request(request,dict(max_request_characters=180000));unchanged()
                if previous and previous["complete"] and request["call_id"] not in journal.records:raise ValueError("No completed-run dispatch allowed")
                response=_dispatch(journal,provider,request,contract)
                action,replacements=compile_span_response(document,response["text"],mode)
                task=document["task"]
                patch=dict(schema="choreobench.patch.v1",base_plan_sha256=object_sha(plan),instruction=task["instruction"],
                    allowed_event_ids=task["allowed_event_ids"],repair_event_ids=task["repair_event_ids"],protected=task["protected"],
                    operations=[dict(event_id=document["event_id"],field="action",value=action)])
                candidate=apply_task_patch(plan,patch,task);_no_new_timing_literals(plan,candidate,document["event_id"])
                validate_plan(candidate)
                row.update(status="valid",new_plan_sha256=object_sha(candidate),changed=action!=document["source_action"],
                    replacements=replacements,final_action=action,immutable_source_text_preserved=True,semantic_quality_verified=False)
                for name,value in (("patch.json",patch),("plan.json",candidate),("replacements.json",replacements)):
                    relative=prefix+name;save(relative,value);artifacts[relative]=file_sha(output/relative)
            except (ValueError,TypeError,KeyError) as error:
                row.update(status="failed",error=str(error),retained_in_denominator=True)
                raw=journal.records.get(request["call_id"])
                info=raw.get("result") if raw else None
                if isinstance(error,_ReplayInfrastructureError) or not raw or not raw.get("success") or not isinstance(info,dict) or info.get("stop_reason") in (None,"http_error","infrastructure_error","malformed_api_response"):
                    fatal=True
            rows.append(row)
            if fatal:break
        unchanged()
        result=_sealed(dict(schema="choreobench.span_pair_result.v1",contract_sha256=object_sha(contract),rows=rows,artifacts=artifacts,
            complete=len(rows)==len(prepared) and not fatal,calls_used=len(journal.started),
            call_records={k:v["record_sha256"] for k,v in journal.records.items()},**_accounting(journal),
            same_source_text_permissions=True,source_spans_are_author_supplied=True,new_audio_calls=0,quality_verified=False,human_reviews=0))
        if previous and previous["complete"] and result!=previous:raise ValueError("Reconstructed span result differs")
        if previous is None:write_json(output/"results.json",result)
        elif not previous["complete"]:atomic_write(output/"results.json",result)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory",required=True);parser.add_argument("--provider",required=True)
    parser.add_argument("--out-dir",required=True);parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv)
    result=run_span_pair(args.inventory,read_json(args.provider),args.out_dir,args.resume)
    print(json_text({k:result[k] for k in ("complete","calls_used","rows","quality_verified")}))


if __name__=="__main__":main()
