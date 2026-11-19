"""Managed TP2 draft lane: one next-batch copy thread, all processing local."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types
sys.dont_write_bytecode=True
RT=Path(__file__).resolve().parent;BAR=RT.parent;SELF=Path(__file__).resolve()
sys.path.append('/opt/venv_eval/lib/python3.11/site-packages');sys.path.insert(0,str(RT))
import caption_local_source_bundle_v1 as local
s=importlib.util.spec_from_file_location('local_live_base',RT/'resource_manager.py');base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
s=importlib.util.spec_from_file_location('local_live_ownership',RT/'bench_manager_driver_v2.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
BASE_SHA='6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
HOSTS={'N3':'28.216.19.213','N4':'28.216.19.143','N6':'28.216.19.225'}
MODEL='/opt/models/Qwen3-VL-32B-Instruct'

def ref(path):return dict(path=str(Path(path).resolve()),sha256=base.sha(path))
def selected(cfg):
    rows=local.light.load(cfg['plan']['path']);assert len({(r['id'],r['split']) for r in rows})==len(rows)
    return [r['id'] for r in rows],len(rows)
def validate(cfg):
    local.light.need(base.sha(base.__file__)==BASE_SHA and cfg['producer']==cfg['resource_manager_ref']==ref(SELF),'Separate native worker identity required')
    local.light.need(HOSTS[cfg['node']] in h.subprocess.check_output(['hostname','-I'],text=True).split(),'Wrong execution node')
    local.light.need(cfg['release_authorized'] is True and cfg['gpus'] in ([0,1],[2,3],[4,5],[6,7]),'Explicit TP2 GPU release required')
    local.light.need(cfg['env']['CUDA_VISIBLE_DEVICES']==','.join(map(str,cfg['gpus'])) and cfg['targetcount']==selected(cfg)[1],'Source/GPU assignment differs')
    local.light.need(Path(cfg['draft_out']).parent==BAR and Path(cfg['draft_out']).name.startswith('caption_two_stage_local_'),'Independent local draft edition required')
    prior=Path(cfg['resource_previous_safe']['path']).parent;safe=local.light.load(prior/'resources_safe.json')
    local.light.need(base.sha(prior/'resources_safe.json')==cfg['resource_previous_safe']['sha256'] and
        all(safe[k] is True for k in ('all_holders_safe','locks_released','leases_restored')),'Native predecessor resources not safe')
    for gpu,owner in zip(cfg['gpus'],cfg['holders']):
        previous=local.light.load(prior/('holder_restored_gpu%d.json'%gpu))
        local.light.need(previous['uuid']==owner['uuid'] and all(previous['holder_identity'][k]==owner[k] for k in base.IDENTITY_KEYS),'Fresh restored holder differs')
        base.check_holder(owner,base.identity(owner['pid']),gpu,base.uuid_for(gpu),base.apps())
        local.light.need(not h.lock_snapshot(gpu)['held'],'Active GPU lock prevents release')
    source_safe=local.light.load(cfg['source_handoff']['path'])
    local.light.need(base.sha(cfg['source_handoff']['path'])==cfg['source_handoff']['sha256'] and source_safe['old_worker_tree_absent'] is True
        and source_safe['original_config_resume_forbidden'] is True,'Source handoff/exclusion proof required')
    allowed={(x['id'],x['split']) for x in source_safe['remaining_sources']}
    local.light.need(all((x['id'],x['split']) in allowed for x in local.light.load(cfg['plan']['path'])),'Source outside handed-off remaining assignment')

def gate(cfg,before=False):
    fn=types.FunctionType(h.parent_gate.__code__,dict(h.parent_gate.__globals__,SELF=SELF,base=base,NODE=cfg['node'],
        GPUS=cfg['gpus'],CVD=cfg['env']['CUDA_VISIBLE_DEVICES']),'parent_gate',h.parent_gate.__defaults__)
    return fn(cfg,before)

def worker(path):
    cfg=local.light.load(path);cfg.update(_config_path=str(Path(path).resolve()),_config_sha=base.sha(path));gate(cfg,True)
    from vllm import LLM,SamplingParams
    llm=LLM(model=MODEL,dtype='bfloat16',tensor_parallel_size=2,max_num_seqs=8,max_model_len=16384,gpu_memory_utilization=.88,
        limit_mm_per_prompt={'image':16,'video':0},mm_processor_kwargs={'max_pixels':200704},
        structured_outputs_config={'backend':'xgrammar','disable_any_whitespace':True},enable_prefix_caching=True,trust_remote_code=False,seed=0)
    llm._ms_bar_model_sha256=cfg['modelref']['model_identity_sha256']
    args=argparse.Namespace(stage='run',batch_size=8,max_num_seqs=8,retry_failed=False,model=MODEL,visual_max_tokens=1800,
        hand_roi_root=str(Path(cfg['source_root'])/'roi'),sample_fps=4.,ffmpeg='/usr/local/bin/ffmpeg')
    rows=local.light.load(cfg['plan']['path']);out=Path(cfg['draft_out']);out.mkdir(parents=True,exist_ok=True)
    local.light.write_once(out/'assignment.json',dict(rows=rows,source_handoff=cfg['source_handoff'],recipe=local.light.RECIPE,**local.light.FLAGS))
    complete=set();counts={};progress_path=out/'progress.json'
    if progress_path.exists():
        v=local.light.load(progress_path);complete={tuple(x) for x in v['completed_source_keys']};counts=v['counts']
    ready={};offsets=[0]*4;planned=set(complete);batch_number=0;start=time.perf_counter()
    def update_ready():
        source=Path(cfg['source_root']);paths=[source/'ready'/('producer-%04d.jsonl'%i) for i in range(4)] if (source/'ready').is_dir() else [source/'ready.jsonl']
        for qi,path in enumerate(paths):
            if not path.exists():continue
            with path.open('rb') as f:
                f.seek(offsets[qi])
                while True:
                    pos=f.tell();line=f.readline()
                    if not line or not line.endswith(bytes([10])):break
                    row=local.light.schemas.strict(line);ready[row['id']]=row;offsets[qi]=f.tell()
    def batch():
        nonlocal batch_number
        update_ready();chosen=[r for r in rows if (r['id'],r['split']) not in planned and r['id'] in ready][:3]
        if not chosen:return None
        batch_number+=1;name='batch-'+local.light.key(chosen)[:20]
        local.light.need(not (out/'budgets'/name/'claim.json').exists(),'Previously reserved local batch; no automatic retry')
        local.light.write_once(out/'budgets'/name/'claim.json',dict(metas=chosen,reserved_sources=len(chosen),**local.light.FLAGS))
        planned.update((r['id'],r['split']) for r in chosen)
        return name,chosen
    def copy_batch(job):
        name,metas=job;root=Path(tempfile.mkdtemp(prefix=name+'-',dir='/tmp'));entries=[];times=[];failures=[]
        for meta in metas:
            try:
                entry,t=local.stage_source(cfg['source_root'],meta,root);entries.append(entry);times.append(t)
            except (OSError,ValueError,KeyError,TypeError) as error:
                failure=dict(id=meta['id'],split=meta['split'],status='draft_needs_regeneration',reason=type(error).__name__+': '+str(error),**local.light.FLAGS)
                local.light.write_once(root/'caption_two_stage_results/drafts'/meta['split']/meta['id']/'draft.json',failure)
                failures.append(failure)
        return job,root,entries,times,failures
    def runtime():return llm,SamplingParams
    def admitted(a,r):return gate(cfg)['admitted'] is True
    with ThreadPoolExecutor(max_workers=1) as copier:
        pending=None
        while len(complete)<len(rows):
            if pending is None:
                job=batch()
                if not job:time.sleep(10);continue
                pending=copier.submit(copy_batch,job)
            job,root,entries,times,failures=pending.result();pending=None
            next_job=batch()
            if next_job:pending=copier.submit(copy_batch,next_job)
            begin=time.perf_counter()
            result=local.light.generate_batch(entries,args,runtime,root/'caption_two_stage_results',admitted) if entries else dict(drafts=[],new_model_calls=0)
            result['drafts'].extend(failures)
            local.light.write_once(root/'source_batch.json',dict(metas=job[1],copy_timings=times,**local.light.FLAGS))
            commit=local.writeback(root,out,job[0])
            for draft in result['drafts']:
                complete.add((draft['id'],draft['split']));status=draft['status'];counts[status]=counts.get(status,0)+1
            value=dict(recipe=local.light.RECIPE,node=cfg['node'],gpus=cfg['gpus'],processed=len(complete),selected=len(rows),counts=counts,
                completed_source_keys=sorted(complete),last_batch_processing_and_writeback_seconds=time.perf_counter()-begin,
                last_copy_seconds=sum(t['copy_seconds'] for t in times),new_window_calls=result['new_model_calls'],
                local_inputs_intermediates_results=True,next_batch_copy_overlaps_GPU=True,archive_path=commit['archive_path'],
                wall_seconds=time.perf_counter()-start,UTC=time.strftime('%FT%TZ',time.gmtime()),**local.light.FLAGS)
            local.light.bar_module().batch.atomic_write(progress_path,local.light.encoded(value));print(json.dumps(value),flush=True)
            shutil.rmtree(root);root.with_suffix('.tar').unlink()
    local.light.write_once(Path(cfg['out'])/'summary.json',value)

def summaries(cfg,out):
    r=local.light.load(out/'summary.json');local.light.need(r['processed']==r['selected']==cfg['targetcount'],'Local lane not fully terminal')
    return {str(out/'summary.json'):dict(summary=dict(stage='run',total_source_count=cfg['targetcount'],selected_source_count=cfg['targetcount'],
        processed_source_count=r['processed'],plan_sha256=cfg['plan']['sha256']),selection=dict(selected_ids=selected(cfg)[0]),draft_counts=r['counts'])}

def main():
    if len(sys.argv)>1 and sys.argv[1]=='worker':
        p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--config',required=True);worker(p.parse_args().config);return
    base.validate_pair=validate;base.selected_sources=selected;base.stage_summaries=summaries;base.main()
if __name__=='__main__':main()
