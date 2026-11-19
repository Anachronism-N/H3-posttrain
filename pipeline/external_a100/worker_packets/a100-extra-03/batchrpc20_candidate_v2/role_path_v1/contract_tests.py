"""CPU-only original20-source positive and injected publication fault contracts."""
import copy
import io
import json
import os
from pathlib import Path
import tarfile
import time
import candidate as c

class PartialReader:
    def __init__(self, reader, limit=None, fragment=17):
        self.reader, self.limit, self.fragment, self.count = reader, limit, fragment, 0
    def read(self,n):
        if self.limit is not None and self.count>=self.limit:
            return b""
        n=min(n,self.fragment)
        if self.limit is not None:
            n=min(n,self.limit-self.count)
        raw=self.reader.read(n);self.count+=len(raw);return raw

def run_tests(run_id):
    start=time.monotonic();checks=[];failures=[];proofs={};caches={}
    def reject(label,action):
        try:
            action()
        except (ValueError,OSError,TypeError,KeyError,tarfile.TarError) as error:
            checks.append(label);failures.append(dict(test=label,status="sealed_reject",error=type(error).__name__+": "+str(error)))
        else:
            raise AssertionError("accepted forbidden "+label)
    def publish(proof,cache,arm,op="batch",names=None,fragment=None):
        q=c.request(proof,run_id,arm,op,names)
        reader=c.CacheReader(cache,q["files"])
        try:
            return c.receive(q,reader if fragment is None else PartialReader(reader,fragment=fragment),proof)
        finally:
            reader.close()
    for fixture,arm in (("mixed","cpu_mixed"),("all_success","cpu_all_success")):
        proof,cache=c.prepare_local(run_id,fixture);proofs[fixture]=proof;caches[fixture]=cache
        result=publish(proof,cache,arm,fragment=997)
        checked=c.receive(c.request(proof,run_id,arm,"readback"),io.BytesIO(),proof)
        assert result["commit"]==checked["commit"] and result["publications"]==checked["publications"]
        assert checked["all_file_tar_commit_readback"]
        root=c.target_root(run_id,arm,fixture)
        committed=c.strict(c.bounded_read(root/"commit.json",c.HEADER_CAP,result["commit"]["sha256"]))
        for key in proof["original_commit"]:
            if key!="publications":
                assert committed[key]==proof["original_commit"][key],key
        assert committed["source_count"]==20 and committed["test_only"] and committed["no_annotation"] and committed["newsource"]==0
        assert len(committed["original_to_test_paths"])==len(proof["boundaries"])
        checks.append(fixture+"_20source_complete_full_tar_commit_readback_partial_reads")
        c.local_artifact(run_id,fixture+"_contract.json",dict(fixture=fixture,result=result,original_commit_sha256=proof["original_commit_sha256"],
                        source_count=20,case_files=len(proof["boundaries"])-1,cache_bytes=sum(b["bytes"] for b in proof["boundaries"]),
                        receipts_payload_and_original_fields_unchanged=True))
    p,cache=proofs["mixed"],caches["mixed"]
    base=c.request(p,run_id,"batch","batch")
    for label,key,value in [("wrong_worker","worker","a100-extra-03"),("wrong_execution_worker","execution_node","a100-extra-03"),
                            ("wrong_GPU","gpu",0),("wrong_GPU_type","gpu",True),("production_mode","test_only",False),
                            ("annotation_mode","no_annotation",False),("new_source","newsource",1),("wrong_ticketSHA","ticket_sha256","0"*64),
                            ("wrong_assignmentSHA","assignment_sha256","0"*64),("wrong_queueSHA","queue_sha256","0"*64),
                            ("wrong_original_commitSHA","original_commit_sha256","0"*64)]:
        q=copy.deepcopy(base);q[key]=value;reject(label,lambda q=q:c.request_check(q,p))
    q=copy.deepcopy(base);q["assignment_indices"][0]+=1
    reject("wrong_index_mod4",lambda:c.request_check(q,p))
    q=copy.deepcopy(base);q["source_rows"][0]["id"]="altered"
    reject("wrong_source_row",lambda:c.request_check(q,p))
    q=copy.deepcopy(base);q["path"]="/tmp/production_target"
    reject("arbitrary_target",lambda:c.request_check(q,p))
    for label,mutate in [("duplicate_boundary",lambda q:q["files"].append(copy.deepcopy(q["files"][0]))),
                         ("over_file_budget",lambda q:q["files"][0].update(bytes=c.OUTPUT_CAP+1)),
                         ("too_many_files",lambda q:q.update(files=[copy.deepcopy(q["files"][0]) for _ in range(102)])),
                         ("bad_header_SHA",lambda q:q["files"][0].update(sha256="0"*64)),
                         ("short_header_set",lambda q:q.update(files=q["files"][:-1]))]:
        q=copy.deepcopy(base);mutate(q);reject(label,lambda q=q:c.request_check(q,p))
    huge=[dict(name="cases/train/id"+str(i)+"/pose.npz",bytes=c.OUTPUT_CAP,sha256="1"*64) for i in range(3)]
    reject("total_cap8GiB",lambda:c.boundaries_check(huge))
    for name in ["../cases/train/id/pose.npz","/cases/train/id/pose.npz","cases/train/../pose.npz",
                 "cases/train/id/../../outside","cases//id/pose.npz","cases/train/id/notallowed.json"]:
        reject("unsafe_path_"+str(len(checks)),lambda name=name:c.boundary_check(dict(name=name,bytes=1,sha256="1"*64)))
    reject("unsafe_run_path",lambda:c.run_root("../escape"))
    # Header budget independent of payload size.
    q=copy.deepcopy(base);q["source_rows"][0]["id"]="x"*(c.HEADER_CAP+1)
    reject("header_cap1MiB",lambda:c.request_check(q,p))
    # Mid-second-file disconnect retains first good path; same target and intent are mandatory for fallback.
    q=c.request(p,run_id,"fault_short","batch");r=c.CacheReader(cache,q["files"])
    cutoff=q["files"][0]["bytes"]+q["files"][1]["bytes"]//2
    try:
        reject("second_file_disconnect",lambda:c.receive(q,PartialReader(r,cutoff,fragment=997),p))
    finally:
        r.close()
    root=c.target_root(run_id,"fault_short","mixed")
    assert not (root/"commit.json").exists()
    first=q["files"][0];c.verify_file(root/first["name"],first)
    assert not (root/q["files"][1]["name"]).exists()
    fallback_calls=0
    for item in p["boundaries"]:
        ack=publish(p,cache,"fault_short","fallback_file",[item["name"]],fragment=997);fallback_calls+=1
        assert ack["publications"][0]["receipt"]==dict(path=str(root/item["name"]),bytes=item["bytes"],sha256=item["sha256"])
        assert not (root/"commit.json").exists()
    done=c.receive(c.request(p,run_id,"fault_short","finalize"),io.BytesIO(),p);fallback_calls+=1
    reread=c.receive(c.request(p,run_id,"fault_short","readback"),io.BytesIO(),p);fallback_calls+=1
    assert done["commit"]==reread["commit"] and done["publications"]==reread["publications"] and reread["all_file_tar_commit_readback"]
    checks.append("same_target_full_payload_sameSHA_file_fallback_tar_commit_last_complete")
    reject("fallback_cannot_change_target",lambda:publish(p,cache,"already_good","fallback_file",[first["name"]]))
    # Already-good same bytes is reusable, including immutable final commit.
    a=publish(p,cache,"already_good");b=publish(p,cache,"already_good")
    assert a==b;checks.append("already_good_same_bytes_reuse_full_commit")
    # Preexisting conflicting bytes remain byte-for-byte unchanged; no commit, no alternate output.
    root=c.target_root(run_id,"fault_bad","mixed");item=p["boundaries"][1];bad=b"Z"*item["bytes"]
    c.write_stream(root/item["name"],io.BytesIO(bad),len(bad),c.sha(bad),c.run_root(run_id))
    reject("already_bad_sealed_reject",lambda:publish(p,cache,"fault_bad"))
    assert c.bounded_read(root/item["name"],c.OUTPUT_CAP)==bad and not (root/"commit.json").exists()
    reject("fallback_bad_preserves_original",lambda:publish(p,cache,"fault_bad","fallback_file",[item["name"]]))
    assert c.bounded_read(root/item["name"],c.OUTPUT_CAP)==bad and not (root/"commit.json").exists()
    checks.append("already_bad_original_bytes_preserved_commit_absent")
    # Correct header but actual changed bytes must fail SHA, with no final commit.
    q=c.request(p,run_id,"fault_sha","batch")
    altered=c.bounded_read(cache/q["files"][0]["name"],c.OUTPUT_CAP)
    altered=bytes([altered[0]^1])+altered[1:]
    reject("actual_payload_bad_SHA",lambda:c.receive(q,io.BytesIO(altered),p))
    assert not (c.target_root(run_id,"fault_sha","mixed")/"commit.json").exists()
    # Extra post-boundary bytes fail before commit.
    class ExtraReader:
        def __init__(self,r):self.r=r;self.extra=False
        def read(self,n):
            raw=self.r.read(n)
            if raw:return raw
            if not self.extra:self.extra=True;return b"x"
            return b""
    q=c.request(p,run_id,"fault_extra","batch");r=c.CacheReader(cache,q["files"])
    try:
        reject("extra_final_stream_byte",lambda:c.receive(q,ExtraReader(r),p))
    finally:r.close()
    assert not (c.target_root(run_id,"fault_extra","mixed")/"commit.json").exists()
    # Nested symlink ancestor is rejected via O_NOFOLLOW directory FD; target never traversed.
    root=c.target_root(run_id,"fault_symlink","mixed");fd=c.directory_fd(root,create=True,scope=c.run_root(run_id))
    os.symlink(str(cache/"cases"),"cases",dir_fd=fd);os.close(fd)
    reject("nested_symlink_ancestor",lambda:publish(p,cache,"fault_symlink"))
    assert not (root/"commit.json").exists()
    reject("final_symlink_readback",lambda:c.open_regular(root/"cases"/Path(first["name"]).relative_to("cases")))
    # Tar safety checks read only mutated own cache; never extract unsafe members.
    scratch=c.OWN/"runs"/run_id/"negative_archives"
    for label,kind in [("tar_symlink",tarfile.SYMTYPE),("tar_hardlink",tarfile.LNKTYPE),("tar_directory",tarfile.DIRTYPE),
                       ("tar_duplicate",tarfile.REGTYPE),("tar_traversal",tarfile.REGTYPE)]:
        data=io.BytesIO()
        with tarfile.open(fileobj=data,mode="w:",format=tarfile.USTAR_FORMAT) as pack:
            name=first["name"] if label!="tar_traversal" else "../"+first["name"]
            member=tarfile.TarInfo(name);member.type=kind
            if kind==tarfile.REGTYPE:
                payload=c.bounded_read(cache/first["name"],c.OUTPUT_CAP);member.size=len(payload)
                pack.addfile(member,io.BytesIO(payload))
                if label=="tar_duplicate":pack.addfile(member,io.BytesIO(payload))
            else:
                member.linkname="outside" if kind!=tarfile.DIRTYPE else "";pack.addfile(member)
        raw=data.getvalue();fake=copy.deepcopy(p);fake["boundaries"][-1].update(bytes=len(raw),sha256=c.sha(raw))
        archive=scratch/(label+".tar");c.write_stream(archive,io.BytesIO(raw),len(raw),c.sha(raw),c.OWN)
        reject(label,lambda archive=archive,fake=fake:c.extract_cached_archive(archive,fake,scratch/"expanded"))
    # Tar wrong member length/SHA/short archive have distinct failures.
    fake=copy.deepcopy(p);fake["boundaries"][0]["sha256"]="0"*64
    reject("tar_member_expected_SHA",lambda:c.extract_cached_archive(cache/"batch.tar",fake,scratch/"bad_sha"))
    fake=copy.deepcopy(p);fake["boundaries"][0]["bytes"]+=1
    reject("tar_member_expected_size",lambda:c.extract_cached_archive(cache/"batch.tar",fake,scratch/"bad_size"))
    fake=copy.deepcopy(p);fake["boundaries"][-1]["bytes"]+=1
    reject("tar_short_stream",lambda:c.extract_cached_archive(cache/"batch.tar",fake,scratch/"short"))
    result=dict(test_only=True,no_annotation=True,newsource=0,model_calls=0,sourceclaims=0,media_reads=0,GPU_calls=0,
                actual_fullpipeline_speedup=False,run_id=run_id,passed=True,checks=checks,injected_failures_retained=failures,
                mixed_source_count=20,mixed_case_files=96,all_success_source_count=20,all_success_case_files=100,
                fallback_publication_file_calls=len(p["boundaries"]),fallback_calls_including_finalize_readback=fallback_calls,
                fallback_same_target=str(c.target_root(run_id,"fault_short","mixed")),
                bad_same_target=str(c.target_root(run_id,"fault_bad","mixed")),wall_seconds=time.monotonic()-start,
                output_root=str(c.run_root(run_id)),original_commit_sha256={k:v["original_commit_sha256"] for k,v in proofs.items()})
    c.local_artifact(run_id,"cpu_contracts_result.json",result)
    return result


def run_topology_client_contract(run_id, existing_cache_run_id):
    """Exercise the exact topology fault client using local receiver, zero actual RPC."""
    proof=c.fixture_proof("mixed");cache=c.local_cache(existing_cache_run_id,"mixed")
    original_rpc=c.rpc;operations=[]
    def local_rpc(q,cache=None,deadline=None,prepare=False,inject_second_file_short=False):
        assert not prepare and time.monotonic()<deadline
        operations.append(q["op"])
        reader=c.CacheReader(cache,q["files"]) if q["op"] in ("batch","file","fallback_file") else io.BytesIO()
        try:
            if inject_second_file_short:
                assert q["op"]=="batch" and q["arm"]=="fault_short"
                cutoff=q["files"][0]["bytes"]+q["files"][1]["bytes"]//2
                reader_for_receiver=PartialReader(reader,cutoff,fragment=997)
            else:reader_for_receiver=reader
            try:return c.receive(q,reader_for_receiver,proof)
            except ValueError as error:raise RuntimeError("sealed receiver reject; CPU synthetic handle: "+str(error)) from error
        finally:
            if hasattr(reader,"close"):reader.close()
    c.rpc=local_rpc
    try:
        result=c.topology_fault_fallback(proof,cache,run_id,time.monotonic()+600)
    finally:c.rpc=original_rpc
    assert result["complete"] and result["publication_only_same_payload_same_target"] and result["all_file_tar_commit_readback"]
    assert result["actual_rpc_calls"]==101 and len(operations)==101
    assert operations==["batch","fault_readback"]+["fallback_file"]*97+["finalize","readback"]
    # Returned topology actual_rpc_calls count refers to calls in the mocked client; actual SSH calls=0.
    contract=dict(passed=True,test_only=True,actual_SSH_RPC_calls=0,model_calls=0,sourceclaims=0,
                  client_contract_calls=len(operations),operations=operations,result=result,
                  unchanged_existing_cache=str(cache),original_commit_sha256=proof["original_commit_sha256"],
                  source_generation_repeated=False,actual_fullpipeline_speedup=False)
    c.local_artifact(run_id,"topology_client_cpu_contract.json",contract)
    return contract
