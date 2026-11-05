"""A bounded, journaled text-editor baseline with no access to private goals."""
import copy
from pathlib import Path

from .baseline import _Journal, _sealed, _verify
from .editing import apply_task_patch, public_edit_task, validate_edit_task
from .io import atomic_write, file_lock, file_sha, json_text, object_sha, read_json, read_json_snapshot, write_json
from .providers import create_provider, parse_json_response
from .schema import validate_plan


def propose_edit(plan_path, task_path, provider_spec, output_dir, *, seed=7, max_output_tokens=4096, resume=False):
    if type(seed) is not int or not 0 <= seed < 2 ** 63 or type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError("Invalid edit seed/output token budget")
    plan, plan_file_sha = read_json_snapshot(plan_path)
    task, task_file_sha = read_json_snapshot(task_path)
    validate_plan(plan)
    validate_edit_task(task, plan)
    provider = create_provider(provider_spec, role="planner", base_dir=Path(task_path).resolve().parent)
    if provider.identity["kind"] == "mock":
        raise ValueError("Use the manual example patch for smoke tests; natural-language editing requires chat_completions or replay")
    contract = {"schema": "choreobench.edit_run_contract.v1",
                "source_file_sha256": plan_file_sha, "task_file_sha256": task_file_sha,
                "source_plan_sha256": object_sha(plan), "task_sha256": object_sha(task),
                "provider_identity": provider.identity, "seed": seed, "max_output_tokens": max_output_tokens,
                "code_sha256": {name: file_sha(Path(__file__).with_name(name)) for name in
                               ("edit_runner.py", "editing.py", "schema.py", "io.py", "providers.py", "baseline.py")}}
    public = public_edit_task(task)
    model_plan = copy.deepcopy(plan)
    model_plan["provenance"] = {}
    for event in model_plan["events"]:
        event["evidence"] = []
    template = {"schema": "choreobench.patch.v1", "base_plan_sha256": object_sha(plan),
                "instruction": public["instruction"], "allowed_event_ids": public["allowed_event_ids"],
                "repair_event_ids": public["repair_event_ids"], "protected": public["protected"], "operations": []}
    request = {"call_id": task["task_id"] + "/edit", "seed": seed, "max_output_tokens": max_output_tokens,
               "response_kind": "patch", "messages": [
                   {"role": "system", "content": (
                       "Edit an existing choreography score according to the user instruction and immutable scope. "
                       "Return exactly one COMPLETE JSON patch with all seven keys from patch_template: "
                       "schema,base_plan_sha256,instruction,allowed_event_ids,repair_event_ids,protected,operations. "
                       "Copy the first six fields exactly from patch_template and replace only its operations array. "
                       "Do not return an operations-only object, the source plan, or a prose explanation. "
                       "Each operation is {event_id,field,value}. Existing scalar fields only: action,start,end,track,motif_id,"
                       "attributes.KEY,entry.KEY,exit.KEY. No event additions/deletions or attribute additions. "
                       "All times are integer samples; do not change protected fields or events outside allowed/repair IDs. "
                       "If the request is ambiguous or infeasible, return empty operations (abstention, never automatic success).")},
                   {"role": "user", "content": json_text({"source_plan": model_plan, "task": public, "patch_template": template})}]}
    output = Path(output_dir).resolve()
    if output.exists() and not resume:
        raise ValueError("Edit output exists; use unchanged --resume or a new directory")
    if resume and not (output / "contract.json").is_file():
        raise ValueError("Resume requires an edit contract")
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".run.lock"):
        if (output / "contract.json").exists():
            if read_json(output / "contract.json") != contract:
                raise ValueError("Edit resume model/input/code identity changed")
        else:
            write_json(output / "contract.json", contract)
        journal = _Journal(output, object_sha(contract), 1)
        try:
            response = journal.call(provider, request)
            patch = parse_json_response(response["text"])
            edited = apply_task_patch(plan, patch, task)
            for name, item in (("patch.json", patch), ("plan.json", edited)):
                if (output / name).exists():
                    if read_json(output / name) != item:
                        raise ValueError("Existing edit artifact changed")
                else:
                    write_json(output / name, item)
            result = _sealed({"schema": "choreobench.edit_run_result.v1", "complete": True,
                              "contract_sha256": object_sha(contract), "calls_used": len(journal.started),
                              "patch_sha256": object_sha(patch), "plan_sha256": object_sha(edited),
                              "abstained": not patch["operations"], "synthetic": response["synthetic"],
                              "costs": response["costs"], "semantic_success": None,
                              "semantic_status": "pending_independent_review"})
            if (output / "results.json").exists() and read_json(output / "results.json").get("complete"):
                if _verify(read_json(output / "results.json")) != result:
                    raise ValueError("Existing edit result changed")
            else:
                atomic_write(output / "results.json", result)
            return result
        except Exception as error:
            # Original call evidence stays immutable. No second model call is hidden.
            if not (output / "results.json").exists():
                atomic_write(output / "results.json", _sealed({
                    "schema": "choreobench.edit_run_result.v1", "complete": False,
                    "contract_sha256": object_sha(contract), "calls_used": len(journal.started),
                    "costs": [{"call_id": call_id,
                               "known_costs": record["result"].get("costs") if isinstance(record.get("result"), dict) else None,
                               "status": "reported" if isinstance(record.get("result"), dict) else "unknown"}
                              for call_id, record in journal.records.items()],
                    "unknown_outcome_call_ids": sorted(set(journal.started) - set(journal.records)),
                    "error": {"type": type(error).__name__, "message": str(error)}}))
            raise
