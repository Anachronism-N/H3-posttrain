"""Independent missing/durability closure after frozen v2 interruption."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
sys.dont_write_bytecode=True
import ms_vlm_archive_inputs_recovery_v2 as old
v=old.v
a=old.a
RT=Path(__file__).resolve().parent
ROOT=RT.parent
OUTPUT=ROOT/"ms_fullframe_vlm_archive_inputs_recovery_20261028_v3"
OLD_SHA="88af11490d88bedfda7ec87682cc23b671efc3217c4b17ea04f286c5b3be7946"


def sync_existing(path,raw):
    path=Path(path)
    v.need(path.is_absolute() and path.resolve(strict=True)==path,"Stable preserved file required")
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,"rb") as stream:
        before=os.fstat(stream.fileno())
        v.need(stat.S_ISREG(before.st_mode) and stream.read(len(raw)+1)==raw,"Preserved raw bytes differ")
        os.fsync(stream.fileno())
        v.need(a.signature(before)==a.signature(os.fstat(stream.fileno())),"Preserved file changed during durability closure")
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
    v.need(path.read_bytes()==raw,"Preserved file readback differs")
    return a.raw_ref(path,raw)


def strict_extra(row,draft,manifest,proof,commit):
    old.strict_archive(row,draft,manifest,proof,commit)
    members=commit["archive_index"][row["id"]]["members"]
    locator=manifest["locator"]
    v.need(draft.get("locator")==locator and draft.get("artifact_storage")=="archive_member",
           "Draft top-level original locator/storage required")
    v.need(manifest.get("artifact_members")==members,"Manifest original members required")


def check_row(row,commit_ref,commit):
    draft,manifest=v.input_pair(row)
    proof=v.checked(row["origin_proof_ref"])
    v.need(proof["commit_ref"]==commit_ref,"Original commit binding differs")
    strict_extra(row,draft,manifest,proof,commit)
    return row


def job_plan(job):
    cfg_path=Path(job["config_path"])
    if cfg_path.exists():
        cfg_ref=v.ref(cfg_path);cfg=v.checked(cfg_ref);plan_ref=cfg["plan"]
        if "plan_ref"in job:v.need(job["plan_ref"]==plan_ref,"Registry/config plan reference differs")
        selected_ids=cfg["selected_source_ids"]
    else:
        v.need(job.get("expected_running") is False and isinstance(job.get("plan_ref"),dict) and
               job.get("worker_status") in ("future_pending","pending_config"),
               "Only explicit nonrunning futurepending may lack config")
        cfg_ref=None;plan_ref=job["plan_ref"];selected_ids=job.get("selected_source_ids")
    p=Path(plan_ref["path"])
    v.need(p.is_absolute() and p.resolve(strict=True)==p,"Stable explicit frozen plan required")
    raw=p.read_bytes();v.need(hashlib.sha256(raw).hexdigest()==plan_ref["sha256"],"Frozen allocated plan changed")
    rows=[v.h.bench.strict(line)for line in raw.splitlines()if line.strip()]
    ids=[r["id"]for r in rows]
    v.need(type(job["sources"])is int and len(rows)==job["sources"] and len({i.casefold()for i in ids})==len(ids),
           "Exact finite unique allocated sources required")
    v.need(selected_ids is None or selected_ids==ids,"Allocated selected IDs differ")
    v.need(all(re.fullmatch(r"[A-Za-z0-9_-]+",i)for i in ids),"Unsafe allocated IDs")
    return rows,plan_ref,cfg_ref


def allocations():
    path=ROOT/"runtime/ms_vlm_postprocess_jobs_registry_v1.json"
    raw=path.read_bytes();registry=v.h.bench.strict(raw)
    keys,refs,configs,lanes=set(),[],[],[]
    for job in registry["jobs"]:
        rows,plan_ref,cfg_ref=job_plan(job)
        if cfg_ref:configs.append(cfg_ref)
        refs.append(plan_ref);keys.update(r["id"].casefold()for r in rows)
        if "_archive250_"in job["phase"]:lanes.append(dict(phase=job["phase"],plan_ref=plan_ref,source_keys=[[r["split"],r["id"]]for r in rows]))
    v.need(len(lanes)==4 and all(len(l["source_keys"])==250 for l in lanes) and
           len({i.casefold()for l in lanes for _,i in l["source_keys"]})==1000,"Four frozen allocated250 lanes required")
    return dict(registry_ref=a.raw_ref(path,raw),plan_refs=refs,config_refs=configs,archive1000_plans=lanes,
                allocated_ids=keys,jobs_count=len(registry["jobs"]))


def restore_row(id_,commit_ref,commit,batch,output,previous):
    item=commit["archive_index"][id_];source=item["source_row"];members=item["members"]
    receipt=next(r for r in commit["receipts"]if r["source_row"]["id"]==id_)
    a.eligible(source,receipt,set(),set())
    rawpair={n:batch["extracted"][members[n]]for n in ("draft.json","fullframe_manifest.json")}
    refs,adopted,zeros={},[],[]
    for n,raw in rawpair.items():
        path=previous/n
        if path.exists()and path.stat().st_size:
            refs[n]=sync_existing(path,raw);adopted.append(refs[n])
        else:
            if path.exists():zeros.append(dict(path=str(path),bytes=0))
            refs[n]=old.once(output/"sources"/source["split"]/id_/n,raw)
    originpath=previous/"origin_proof.json"
    if originpath.exists()and originpath.stat().st_size:
        proofraw=originpath.read_bytes();proof=v.h.bench.strict(proofraw)
        v.need(proof["physical_refs"]==refs,"Existing complete proof physical references differ")
        proof_ref=sync_existing(originpath,proofraw);adopted.append(proof_ref)
    else:
        if originpath.exists():zeros.append(dict(path=str(originpath),bytes=0))
        draft,manifest=(v.h.bench.strict(rawpair[n])for n in ("draft.json","fullframe_manifest.json"))
        origin=str(Path(commit_ref["path"]).parent/members["fullframe_manifest.json"])
        proof=dict(schema="ms_archive_vlm_input_origin_v1",commit_ref=commit_ref,source_row=source,
            members=members,source_case_prefix=item["source_case_prefix"],receipt=receipt,
            archive_path=commit["archive_path"],archive_sha256=commit["archive_sha256"],archive_bytes=commit["archive_bytes"],
            sha_verified_once=True,archive_verification_ref=batch["verification_ref"],
            member_metadata={n:dict(member=members[n],sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))for n,raw in rawpair.items()},
            physical_refs=refs,logical_fullframe_origin=origin,original_binding=draft["source"]["fullframe_binding"],
            original_npz_path=manifest["npz_path"],no_npz_decoded=True,no_media_input_read=True,no_gpu_calls=True,
            original_raw_bytes_preserved=True,hash_includes_npz_packaged_bytes=True)
        proof_ref=old.once(output/"sources"/source["split"]/id_/"origin_proof.json",a.encoded(proof))
    row=dict(id=id_,split=source["split"],video_path=source["video_path"],rule_draft_ref=refs["draft.json"],
             fullframe_manifest_ref=refs["fullframe_manifest.json"],origin_proof_ref=proof_ref,
             fullframe_origin_path=proof["logical_fullframe_origin"])
    return check_row(row,commit_ref,commit),adopted,zeros


def main():
    v.need(v.base.sha(old.__file__)==OLD_SHA,"Frozen v2 changed")
    selection_ref=dict(path=str(old.POOL/"source_selection_proof.json"),sha256=old.SELECTION_SHA)
    selection=v.checked(selection_ref);keys=selection["source_keys"]
    v.need(len(keys)==3000 and len(selection["selected_batches"])==150 and
           hashlib.sha256(a.encoded(keys)).hexdigest()==selection["source_keys_sha256"],"Frozen3000 selection differs")
    before=allocations()
    v.need(not OUTPUT.exists(),"New independent v3 output exists; no restart")
    OUTPUT.mkdir();code_ref=old.once(OUTPUT/"frozen_recovery_code.py",Path(__file__).read_bytes())
    cpu_ref=v.ref(RT/"ms_vlm_archive_inputs_recovery_v3_cpu_proof.json");cpu=v.checked(cpu_ref)
    v.need(cpu["code_sha256"]==code_ref["sha256"]and cpu["exitcode"]==0,"V3 CPU proof required before metadata work")
    old.once(OUTPUT/"cpu_proof.json",a.encoded(cpu))
    complete_names={p.name for p in (old.OUTPUT/"batch_receipts").iterdir()}
    proof_names={str(root):{p.name for p in (root/"batch_proofs").iterdir()}for root in (old.POOL,old.OUTPUT)}
    reused=[];missing=[]
    for entry in selection["selected_batches"]:
        name=hashlib.sha256(entry["commit_ref"]["path"].encode()).hexdigest()+".json"
        if name in complete_names:
            path=old.OUTPUT/"batch_receipts"/name;raw=path.read_bytes();r=v.h.bench.strict(raw)
            v.need(r["commit_ref"]==entry["commit_ref"]and r["selected_source_ids"]==entry["source_ids"]and
                   [row["id"]for row in r["rows"]]==entry["source_ids"]and
                   r["original_rows_reused"]+r["missing_rows_materialized"]==len(entry["source_ids"]),
                   "Previously successful receipt binding differs")
            commit=v.frozen_commit(entry["commit_ref"])
            old.verified_batch(entry["commit_ref"],commit,Path(r["archive_verification_ref"]["path"]))
            reused.append(dict(receipt_ref=a.raw_ref(path,raw),evidence=r))
        else:missing.append(entry)
    v.need(len(reused)==129 and len(missing)==21,"Authorized129 successes and21 unclosed batches required")
    print(json.dumps(dict(successful_v2_receipts_reused=129,previous_sources_reused=2580,remaining_sources=420)),flush=True)
    def process(entry):
        commit_ref=entry["commit_ref"];commit=v.frozen_commit(commit_ref);ids=entry["source_ids"]
        wanted={commit["archive_index"][i]["members"][n]for i in ids for n in ("draft.json","fullframe_manifest.json")}
        name=old.proof_path(old.POOL,commit["archive_path"]).name;verified=None
        for root in (old.POOL,old.OUTPUT):
            if name in proof_names[str(root)]:verified=root/"batch_proofs"/name;break
        if verified:
            batch=old.read_verified_members(commit_ref,commit,wanted,verified);newhash=0
        else:batch=a.read_archive(commit_ref,commit,wanted,OUTPUT,{});newhash=1
        rows,adopted,zeros=[],[],[]
        for id_ in ids:
            row,kept,zero=restore_row(id_,commit_ref,commit,batch,OUTPUT,old.OUTPUT/"sources/train"/id_)
            rows.append(row);adopted.extend(kept);zeros.extend(zero)
        proof=dict(schema="ms_archive_metadata_missing_durability_batch_v3",commit_ref=commit_ref,
                   selected_source_ids=ids,rows=rows,archive_verification_ref=batch["verification_ref"],
                   new_whole_archive_hash_reads=newhash,adopted_durability_refs=adopted,preserved_zero_files=zeros,
                   native_input_pair_pass_count=len(rows),strict_extra_pass_count=len(rows),models_called=0,
                   source_claims_modified=False,original_files_overwritten=0)
        ref=old.once(OUTPUT/"batch_receipts"/(hashlib.sha256(commit_ref["path"].encode()).hexdigest()+".json"),a.encoded(proof))
        print(json.dumps(dict(batch=commit["ticket_id"],sources=len(rows),first_hashes=newhash,receipt_ref=ref)),flush=True)
        return proof
    with ThreadPoolExecutor(max_workers=4)as executor:new=list(executor.map(process,missing))
    batches={x["evidence"]["commit_ref"]["path"]:x["evidence"]for x in reused}
    batches.update({x["commit_ref"]["path"]:x for x in new})
    ordered=[batches[e["commit_ref"]["path"]]for e in selection["selected_batches"]]
    rows=[r for b in ordered for r in b["rows"]]
    v.need([[r["split"],r["id"]]for r in rows]==keys,"Complete manifest differs from frozen selection")
    after=allocations();allocated=before["allocated_ids"]|after["allocated_ids"]
    available=[r for r in rows if r["id"].casefold()not in allocated]
    manifest=old.once(OUTPUT/"manifest.jsonl",b"".join(a.encoded(r)for r in rows))
    unallocated=old.once(OUTPUT/"unallocated_manifest.jsonl",b"".join(a.encoded(r)for r in available))
    proof=dict(schema="ms_archive_vlm_metadata_recovery_allocation_v3",status="complete_cpu_metadata_only_not_deployed",
        source_count=3000,source_keys=keys,source_keys_sha256=selection["source_keys_sha256"],source_selection_ref=selection_ref,
        manifest_ref=manifest,unallocated_manifest_ref=unallocated,unallocated_source_count=len(available),
        registry_before_ref=before["registry_ref"],registry_after_ref=after["registry_ref"],allocated_plan_refs=after["plan_refs"],
        archive1000_plans=after["archive1000_plans"],previous_successful_receipt_refs=[r["receipt_ref"]for r in reused],
        previous_native_strict_sources_reused=2580,new_native_and_extra_strict_sources=420,
        all3000_fresh_v4_input_pair_not_claimed=True,new_extra_strict_conditions_not_replayed_for2580=True,
        previous_frozen_native_input_pair_pass_total=3000,unallocated_plan_intersection=0,
        unallocated_observation_not_exclusive_reservation=True,archive_verification_refs=[b["archive_verification_ref"]for b in ordered],
        v3_new_once_only_archive_hash_reads=sum(b["new_whole_archive_hash_reads"]for b in new),existing_archive_rehashes=0,
        original_files_overwritten=0,source_claims_created=0,stage_budget_claims_created=0,run_configs_created=0,
        run_queues_created=0,models_called=0,no_gpu_calls=True,no_model_load=True,no_media_input_read=True,no_npz_decoded=True,
        code_ref=code_ref,cpu_ref=cpu_ref,**v.FLAGS)
    allocation=old.once(OUTPUT/"allocation.json",a.encoded(proof))
    completion=old.once(OUTPUT/"completion_receipt.json",a.encoded(dict(schema="ms_archive_metadata_recovery_complete_v3",
        source_count=3000,manifest_ref=manifest,allocation_ref=allocation,unallocated_manifest_ref=unallocated,
        unallocated_source_count=len(available),all_originals_preserved=True,existing_archive_rehashes=0,models_called=0)))
    print(json.dumps(dict(completion_ref=completion,manifest_ref=manifest,allocation_ref=allocation,
                         unallocated_manifest_ref=unallocated,unallocated_source_count=len(available)),sort_keys=True),flush=True)


if __name__=="__main__":main()
