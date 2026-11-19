"""Batch-local inputs/intermediates/results with one reliable archive writeback."""
import copy
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time

import caption_two_stage_light_v1 as light


def stage_source(producer_root,meta,batch_root):
    sid,split=meta['id'],meta['split'];source=Path(producer_root)/'sources'/split/sid
    inputs=light.Inputs();original_plan=inputs.json(source/'bar_plan.json');actual=inputs.json(source/'actual_media.json')
    light.need(original_plan['id']==sid and original_plan['manifest_record']['split']==split,'Source ID/split mismatch')
    directory=Path(batch_root)/'inputs'/split/sid
    light.need(not directory.exists(),'Fresh local source bundle required')
    directory.mkdir(parents=True,exist_ok=False)
    row=copy.deepcopy(original_plan);copied={};size=0;begin=time.perf_counter()
    def asset(path,subdirectory):
        nonlocal size
        path=Path(path)
        if str(path) in copied:return copied[str(path)]
        target=directory/subdirectory/path.name
        light.need(not target.exists(),'Local source asset name collision')
        target.parent.mkdir(parents=True,exist_ok=True)
        with path.open('rb') as src,target.open('xb') as dst:
            shutil.copyfileobj(src,dst,8*1024*1024)
        size+=target.stat().st_size;copied[str(path)]=str(target)
        return str(target)
    for field in ('video_path','audio_path','nlf_path','rich_path'):
        path=row['manifest_record'].get(field)
        if path and Path(path).is_file():row['manifest_record'][field]=asset(path,'media/'+field)
    if row.get('raw2d_mask'):
        row['raw2d_mask']['path']=asset(row['raw2d_mask']['path'],'mask')
    roi_root=Path(producer_root)/'roi'/sid
    roi=inputs.json(roi_root/'roi_manifest.json') if (roi_root/'roi_manifest.json').exists() else None
    if roi:
        roi=copy.deepcopy(roi)
        light.need(roi['id']==sid,'ROI source ID mismatch')
        for crop in roi.get('crops',[]):crop['path']=asset(crop['path'],'roi/crops')
        if roi.get('raw'):roi['raw']['path']=asset(roi['raw']['path'],'roi/raw')
        for name in ('source_manifest.json','extract_receipt.json'):
            if (roi_root/name).exists():asset(roi_root/name,'roi')
    light.write_once(directory/'original_plan.json',original_plan)
    light.write_once(directory/'actual_media.json',actual)
    light.write_once(directory/'local_plan.json',row)
    light.write_once(directory/'path_map.json',copied)
    entry=dict(row=row,actual=actual,roi_manifest=roi,binding=dict(source_binding=row['binding'],split=split,
        original_source_record=original_plan['manifest_record'],local_execution=True),
        root=Path(batch_root)/'caption_two_stage_results/sources'/split/sid)
    return entry,dict(copy_seconds=time.perf_counter()-begin,files_copied=len(copied),bytes_copied=size,
        original_source=sid,split=split,media_SHA_reads=0)


def writeback(batch_root,shared_root,batch_id):
    batch_root=Path(batch_root);shared_root=Path(shared_root)
    light.need(light.bar_module().portable(batch_id),'Portable batch ID required')
    shared_root.mkdir(parents=True,exist_ok=True)
    final=shared_root/(batch_id+'.tar');receipt=shared_root/(batch_id+'.committed.json')
    light.need(not final.exists() and not receipt.exists(),'Batch already published; never overwrite')
    begin=time.perf_counter();local_archive=batch_root.with_suffix('.tar')
    light.need(not local_archive.exists(),'Archive path already exists')
    with tarfile.open(local_archive,'x') as archive:
        archive.add(batch_root,arcname=batch_id,recursive=True)
    size=local_archive.stat().st_size
    with local_archive.open('rb') as src,tempfile.NamedTemporaryFile(dir=shared_root,prefix='.'+batch_id,delete=False) as dst:
        temporary=Path(dst.name)
        try:
            shutil.copyfileobj(src,dst,8*1024*1024);dst.flush();os.fsync(dst.fileno())
            light.need(dst.tell()==size,'Incomplete archive transfer')
            os.link(temporary,final)
        finally:temporary.unlink(missing_ok=True)
    fd=os.open(shared_root,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    value=dict(batch_id=batch_id,archive_path=str(final),archive_bytes=size,
        source_inputs_intermediates_and_results_all_local=True,reliable_writeback=True,
        writeback_seconds=time.perf_counter()-begin,local_cache_reclaim_allowed=True,**light.FLAGS)
    light.write_once(receipt,value)
    return value


def run_local_batch(producer_root,metas,args,runtime,shared_root,batch_id,local_parent='/tmp',resource_admission=None):
    """Include copy, local processing and durable writeback in the measured wall."""
    light.need(callable(resource_admission) and resource_admission(args,runtime) is True,'Actual caller-owned resource admission required')
    shared_root=Path(shared_root);begin=time.perf_counter()
    with light.bar_module().batch.bundle_lock(shared_root/'budgets'/batch_id):
        committed=shared_root/(batch_id+'.committed.json')
        if committed.exists():return light.load(committed)
        claim=shared_root/'budgets'/batch_id/'claim.json'
        light.need(not claim.exists(),'Local batch already reserved; do not consume model budget again')
        light.write_once(claim,dict(batch_id=batch_id,metas=metas,recipe=light.RECIPE,**light.FLAGS))
        local=Path(tempfile.mkdtemp(prefix=batch_id+'-',dir=local_parent))
        entries=[];copies=[]
        try:
            for meta in metas:
                entry,timing=stage_source(producer_root,meta,local)
                entries.append(entry);copies.append(timing)
            out=local/'caption_two_stage_results'
            process_begin=time.perf_counter()
            result=light.generate_batch(entries,args,runtime,out,resource_admission)
            process_seconds=time.perf_counter()-process_begin
            light.write_once(local/'local_execution.json',dict(copy_timings=copies,
                processing_seconds=process_seconds,draft_counts={s:sum(d['status']==s for d in result['drafts'])
                    for s in {d['status'] for d in result['drafts']}},new_model_calls=result['new_model_calls'],**light.FLAGS))
            committed_value=writeback(local,shared_root,batch_id)
        except BaseException as error:
            light.write_once(shared_root/'budgets'/batch_id/'failure.json',dict(reason=type(error).__name__+': '+str(error),
                local_cache_path=str(local),budget_not_reset=True,**light.FLAGS))
            raise
        committed_value.update(copy_seconds=sum(t['copy_seconds'] for t in copies),processing_seconds=process_seconds,
            end_to_end_seconds=time.perf_counter()-begin,new_model_calls=result['new_model_calls'])
        light.write_once(shared_root/(batch_id+'.timing.json'),committed_value)
        # Exact owned temporary cache is reclaimed only after the durable commit.
        shutil.rmtree(local)
        local.with_suffix('.tar').unlink()
        return committed_value


def archive_drafts(archive_path):
    """Read revision targets without extracting or trusting local execution paths."""
    with tarfile.open(archive_path,'r') as archive:
        for member in archive.getmembers():
            if member.isfile() and '/caption_two_stage_results/drafts/' in member.name and member.name.endswith('/draft.json'):
                stream=archive.extractfile(member)
                yield member.name,light.schemas.strict(stream.read())
