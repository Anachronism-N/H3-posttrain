"""Same-tool edit policies with frozen scopes, budgets, and auditable stops.

The shared executor never judges musical fit or semantic protection. Generic
and choreography policies see the same score/task/context and differ in prompt
policy only. This is not a claim that a new model or whole-song agent exists.
"""
import argparse
import copy
from pathlib import Path

from .baseline import _Journal, _sealed, _verify
from .editing import apply_task_patch, public_edit_task, validate_edit_task
from .evaluation import evaluate_edit
from .io import file_lock, file_sha, json_text, object_sha, read_json_snapshot, read_json, write_json
from .providers import create_provider, parse_json_response
from .request_budget import guard_request
from .revision_protocol import public_plan
from .schema import keys, validate_plan, integer, text


POLICIES = ("generic", "choreography")
TOOLS = (
    "You edit an existing possible choreography score, not an observed performance. "
    "All times are canonical integer samples with half-open intervals. "
    "Only action text on the supplied allowed_event_ids may change. "
    "All other events, times, IDs, tracks, motifs, attributes, states and evidence stay unchanged. "
    "Unknown support, facing and contact stay unknown; musical metadata is predicted. "
    "You have the complete source plan, public instruction and target/neighbor context. "
    "Tools: inspect_context returns that target/neighbor view; check_patch validates proposed operations "
    "against the frozen initial plan without committing; submit_patch applies those operations and "
    "finishes only if mechanical checks pass; abstain finishes with the unchanged original. "
    "A check result is structural only, not a semantic or musical rating. "
    "Each operation has event_id,field=action,value. Empty operations are allowed. "
    "Return exactly one JSON object {tool,operations,reason}. For inspect_context/abstain operations must be []. "
    "Each check/submit is a cumulative proposal from the SAME INITIAL PLAN, not from a previous candidate. "
    "Tool results are recorded below. Use the remaining budget without inventing new observations. "
    "Do not add precise numeric timestamps in action prose or claim that a word limit proves feasibility. "
    "You may stop early; budget is a ceiling, not a target."
)
GENERIC = (
    " Use a general action-observation loop: understand the user's request and constraints, "
    "choose an available tool, interpret its result, and finish when the request is satisfied "
    "or no justified permitted edit exists. Prefer clear edits with low collateral change."
)
CHOREOGRAPHY = (
    " Use choreography-specific decision steps before tools: identify the requested motion facts "
    "and the facts the instruction requires preserving. Distinguish successive complete transitions "
    "from simultaneous body-part motion. Fit the primary movement to the target duration, preserving "
    "the requested footwork/directions rather than merely retaining a phrase saying 'unchanged'. "
    "Use the actual adjacent planned actions to avoid a finish-then-restart contradiction. "
    "A low-density edit may remove proposed unnecessary phases but must not remove a requested fact. "
    "Check the source-to-final cumulative difference against the original, then submit the smallest "
    "useful content change. If the source already meets the instruction or permissions conflict, abstain."
)


def _snapshot(path, sources):
    path = Path(path).resolve()
    value, sha = read_json_snapshot(path)
    sources[str(path)] = sha
    return value


def _unchanged(sources):
    for path, sha in sources.items():
        if file_sha(path) != sha:
            raise ValueError("Frozen edit-agent input changed: " + path)


def _context(plan, task):
    selected = set(task["allowed_event_ids"])
    indexes = {i for i,e in enumerate(plan["events"]) if e["event_id"] in selected}
    neighbors = indexes | {i-1 for i in indexes if i>0} | {i+1 for i in indexes if i+1<len(plan["events"])}
    return dict(events=[copy.deepcopy(plan["events"][i]) for i in sorted(neighbors)],
                duration_seconds={e["event_id"]:(e["end"]-e["start"])/plan["sample_rate"]
                                  for e in plan["events"] if e["event_id"] in selected},
                planned_not_observed=True)


def _decision(text_value, task):
    value = parse_json_response(text_value)
    keys(value, ("tool", "operations", "reason"), where="edit tool decision")
    if value["tool"] not in ("inspect_context", "check_patch", "submit_patch", "abstain"):
        raise ValueError("Unknown editing tool")
    text(value["reason"], "tool reason")
    if len(value["reason"])>1200 or not isinstance(value["operations"], list):
        raise ValueError("Decision exceeds the bounded protocol")
    if len(value["operations"])>len(task["allowed_event_ids"]):
        raise ValueError("Too many edit operations")
    if value["tool"] in ("inspect_context", "abstain") and value["operations"]:
        raise ValueError("Read/abstain tool cannot contain operations")
    for op in value["operations"]:
        keys(op, ("event_id", "field", "value"), where="edit operation")
        # Valid tool proposals can be rejected by the executor without widening
        # the schema or swallowing the failed attempt.
        text(op["event_id"], "operation event_id")
        if op["field"] != "action":
            raise ValueError("This shared editing track only replaces action")
        text(op["value"], "operation action")
        if len(op["value"])>1600:
            raise ValueError("Operation action exceeds the common safety ceiling")
    return value


def _execute(plan, task, context, decision):
    tool = decision["tool"]
    if tool == "inspect_context":
        return dict(tool=tool,status="observed",context=copy.deepcopy(context),semantic_verified=False), None
    if tool == "abstain":
        return dict(tool=tool,status="abstained",semantic_verified=False), copy.deepcopy(plan)
    template = dict(schema="choreobench.patch.v1",base_plan_sha256=object_sha(plan),instruction=task["instruction"],
        allowed_event_ids=task["allowed_event_ids"],repair_event_ids=task["repair_event_ids"],
        protected=task["protected"],operations=decision["operations"])
    try:
        candidate = apply_task_patch(plan, template, task)
        from .revision import _no_new_timing_literals
        for op in decision["operations"]:
            _no_new_timing_literals(plan,candidate,op["event_id"])
        report = evaluate_edit(plan, candidate, task)
        # The model-facing report never includes private goals/semantic goals.
        public_report = {k:copy.deepcopy(report[k]) for k in
            ("after_valid","errors","protections","changed_event_ids","out_of_scope_event_ids",
             "changed_immutable_fields","changed_evidence_event_ids","event_identity_preserved")}
        feedback = dict(tool=tool,status="structurally_valid",report=public_report,
                        candidate_plan_sha256=object_sha(candidate),semantic_verified=False)
        return feedback, candidate if tool == "submit_patch" else None
    except (ValueError,TypeError,KeyError) as error:
        return dict(tool=tool,status="rejected",error=str(error),semantic_verified=False), None


def _schema(task):
    operations = dict(type="array",maxItems=len(task["allowed_event_ids"]),items=dict(type="object",additionalProperties=False,
        required=["event_id","field","value"],properties=dict(event_id=dict(type="string",minLength=1,maxLength=128),
        field=dict(type="string",enum=["action"]),value=dict(type="string",minLength=1,maxLength=1600))))
    return dict(type="object",additionalProperties=False,required=["tool","operations","reason"],properties=dict(
        tool=dict(type="string",enum=["inspect_context","check_patch","submit_patch","abstain"]),
        operations=operations,reason=dict(type="string",minLength=1,maxLength=1200)))


def _save(path, value):
    if path.exists():
        if read_json(path)!=value:
            raise ValueError("Immutable edit-agent artifact changed: " + str(path))
    else:
        write_json(path,value)


def _accounting(journal):
    from .baseline import _journal_accounting
    value = _journal_accounting(journal)
    value["costs"].sort(key=lambda r:r["call_id"])
    for name in ("unknown_outcome_call_ids","unknown_usage_call_ids"):
        value[name].sort()
    return value


def run_edit_agent(plan_path, task_path, provider_spec, output_dir, *, policy="generic", seed=7,
                   max_calls=3, max_output_tokens=2048, max_request_characters=180000, resume=False):
    if policy not in POLICIES:raise ValueError("Unknown edit policy")
    for key,value,low,high in (("seed",seed,0,2**63-1),("max_calls",max_calls,1,8),
            ("max_output_tokens",max_output_tokens,1,8192),("max_request_characters",max_request_characters,1,2000000)):
        integer(value,key,low)
        if value>high:raise ValueError("Edit-agent budget outside bounds")
    sources={}
    plan=_snapshot(plan_path,sources);task=_snapshot(task_path,sources)
    validate_plan(plan);validate_edit_task(task,plan)
    if task["repair_event_ids"]:
        raise ValueError("Comparison track has no implicit neighboring repair scope")
    if not task["allowed_event_ids"]:
        raise ValueError("At least one publicly located target is required")
    provider=create_provider(provider_spec,role="planner",base_dir=Path(task_path).resolve().parent)
    if provider.identity["kind"] not in ("chat_completions","replay"):
        raise ValueError("Editing comparison requires real/replay text provider")
    context=_context(public_plan(plan),task)
    public=public_edit_task(task)
    contract=dict(schema="choreobench.edit_agent_contract.v1",sources=sources,policy=policy,
        source_plan_sha256=object_sha(plan),task_sha256=object_sha(task),provider_identity=copy.deepcopy(provider.identity),
        seed=seed,max_calls=max_calls,max_output_tokens=max_output_tokens,max_request_characters=max_request_characters,
        code_sha256={name:file_sha(Path(__file__).with_name(name)) for name in
            ("edit_agent.py","editing.py","evaluation.py","revision_protocol.py","schema.py","providers.py","baseline.py","io.py","request_budget.py")},
        tool_interface_shared=True,private_goals_visible=False,automatic_semantic_judge=False,
        scientific_quality_verified=False,synthetic=provider.identity["synthetic"] or plan.get("provenance",{}).get("synthetic",False))
    output=Path(output_dir).resolve()
    if output.exists() and not resume:raise FileExistsError(str(output))
    if resume and not (output/"contract.json").is_file():raise ValueError("Resume requires original edit-agent contract")
    _unchanged(sources)
    output.mkdir(parents=True,exist_ok=True)
    with file_lock(output/".run.lock"):
        _save(output/"contract.json",contract)
        journal=_Journal(output,object_sha(contract),max_calls)
        completed=None
        if (output/"results.json").is_file():
            previous=_verify(read_json(output/"results.json"))
            if previous.get("complete"):
                if previous["call_records"]!={k:v["record_sha256"] for k,v in journal.records.items()} or previous["calls_used"]!=len(journal.started):
                    raise ValueError("Completed edit-agent call inventory changed; no resampling")
                for relative,sha in previous["artifacts"].items():
                    if file_sha(output/relative)!=sha:raise ValueError("Completed edit-agent artifact changed")
                completed=previous
        _save(output/"before.json",plan)
        history=[];final=copy.deepcopy(plan);stop="budget_exhausted";failure=None;done=True
        for index in range(max_calls):
            payload=dict(source_plan=public_plan(plan),task=public,context=context,tool_history=history,
                remaining_model_calls=max_calls-index,tool_check_is_literal_only=True)
            request=dict(call_id=task["task_id"]+"/edit-agent/%02d"%index,seed=seed,
                max_output_tokens=max_output_tokens,response_kind="edit_agent_tool",messages=[
                    dict(role="system",content=TOOLS+(GENERIC if policy=="generic" else CHOREOGRAPHY)),
                    dict(role="user",content=json_text(payload))],response_schema=_schema(task))
            try:
                guard_request(request,contract);_unchanged(sources)
                if read_json(output/"contract.json")!=contract or provider.identity!=contract["provider_identity"]:
                    raise ValueError("Frozen edit-agent contract/provider changed")
                if completed is not None and request["call_id"] not in journal.records:
                    raise ValueError("Completed trajectory reconstruction would dispatch; refused")
                response=journal.call(provider,request)
                if provider.identity!=contract["provider_identity"]:raise ValueError("Provider identity drift during dispatch")
                decision=_decision(response["text"],task)
                observation,candidate=_execute(plan,task,context,decision)
                step=dict(index=index,decision=decision,observation=observation,source_plan_sha256=object_sha(plan))
                _save(output/("steps/%02d.json"%index),step);history.append(step)
                if candidate is not None:
                    final=candidate;stop="abstained" if decision["tool"]=="abstain" else "submitted";break
            except (ValueError,TypeError,KeyError) as error:
                failure=str(error);stop="execution_failed";done=False;break
        _unchanged(sources)
        _save(output/"plan.json",final)
        structural=evaluate_edit(plan,final,task)
        # Private evaluation is performed only after the final artifact is fixed.
        _save(output/"evaluation.private.json",structural)
        artifacts={str(p.relative_to(output)):file_sha(p) for p in
            [output/"before.json",output/"plan.json",output/"evaluation.private.json",*sorted((output/"steps").glob("*.json"))]}
        result=_sealed(dict(schema="choreobench.edit_agent_result.v1",complete=done,policy=policy,
            contract_sha256=object_sha(contract),stop_reason=stop,error=failure,steps=len(history),
            calls_used=len(journal.started),call_records={k:v["record_sha256"] for k,v in journal.records.items()},
            artifacts=artifacts,**_accounting(journal),source_plan_sha256=object_sha(plan),final_plan_sha256=object_sha(final),
            changed_event_ids=structural["changed_event_ids"],mechanical_scope_valid=not structural["out_of_scope_event_ids"] and structural["event_identity_preserved"],
            submitted=stop=="submitted",unchanged_original=plan["events"]==final["events"],
            budget_exhaustion_is_not_success=stop=="budget_exhausted",semantic_joint_success=None,
            quality_verified=False,human_reviews=0,synthetic=contract["synthetic"]))
        if completed is not None and result!=completed:raise ValueError("Completed edit-agent reconstructed result differs")
        if (output/"results.json").exists():
            if read_json(output/"results.json")!=result:raise ValueError("Edit-agent result immutable; use new directory")
        else:write_json(output/"results.json",result)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",required=True);parser.add_argument("--task",required=True)
    parser.add_argument("--provider",required=True);parser.add_argument("--out-dir",required=True)
    parser.add_argument("--policy",choices=POLICIES,required=True)
    parser.add_argument("--max-calls",type=int,default=3);parser.add_argument("--max-output-tokens",type=int,default=2048)
    parser.add_argument("--seed",type=int,default=7);parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv)
    result=run_edit_agent(args.plan,args.task,read_json(args.provider),args.out_dir,policy=args.policy,
        seed=args.seed,max_calls=args.max_calls,max_output_tokens=args.max_output_tokens,resume=args.resume)
    print(json_text({k:result[k] for k in ("complete","policy","stop_reason","calls_used","changed_event_ids","semantic_joint_success")}))


if __name__=="__main__":main()
