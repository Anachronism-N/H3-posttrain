"""Finite new-source metadata supply from completed native MS streams, no model calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
import ms_vlm_archive_inputs_v1 as a
import ms_fullframe_vlm_postprocess_batch_v8 as v

SELF=Path(__file__).resolve()
SLOTS=('n2g5','n3g0','n3g1','n6g2','n6g3')

def select(count):
    used=set();registry=v.load(v.RT/'ms_vlm_postprocess_jobs_registry_v1.json')
    for job in registry['jobs']:
        cp=Path(job['config_path'])
        if cp.exists():used.update(i.casefold() for i in v.load(cp)['selected_source_ids'])
        else:
            v.need(job.get('expected_running') is False and 'plan_ref' in job,'Missing unbound registered input config')
            ref=job['plan_ref'];v.need(v.base.sha(ref['path'])==ref['sha256'],'Pending plan changed')
            used.update(v.h.bench.strict(x)['id'].casefold() for x in Path(ref['path']).read_bytes().splitlines())
    used.update(i.casefold() for _,i in v.checked(dict(path=str(v.ROOT/'ms_fullframe_vlm_archive_inputs_20261027_v1/source_selection_proof.json'),sha256='d5edc46b0782a3c44837e72efcc555c3c46b05f882ecb05ba9011d904dca3d56'))['source_keys'])
    selected=[];keys=[];seen=set();refs=[]
    for slot in SLOTS:
        phase='ms_unified_native_v1_stream1000_'+slot+'_20261027';cp=v.RT/(phase+'_config.json');cfg=v.load(cp);summary_path=Path(cfg['out'])/'summary.json';summary=v.load(summary_path);ticket=v.checked(cfg['ticket'])
        v.need(cfg['phase']==summary['edition']==ticket['edition']==phase and summary['status']=='completed' and summary['processed']==1000 and
               summary['allprefetch_drained'] is True and summary['uncommitted_prefetch']==0 and summary['vlm_called'] is False and
               cfg['targetcount']==cfg['input_count']==1000 and summary['selected_source_ids']==cfg['selected_source_ids'] and
               len(summary['all_outcomes'])==1000 and len({r['id'].casefold() for r in summary['all_outcomes']})==1000, 'Exact complete native stream required')
        plan=[v.h.bench.strict(x) for x in Path(cfg['plan']['path']).read_bytes().splitlines() if x.strip()]
        v.need(v.base.sha(cfg['plan']['path'])==cfg['plan']['sha256'] and plan==ticket['sources'] and [r['id'] for r in plan]==cfg['selected_source_ids'],'Frozen native plan/ticket binding differs')
        planmap={r['id']:r for r in plan};outcomes={(r['batch_index'],r['id']):r for r in summary['all_outcomes']};refs.extend([v.ref(cp),v.ref(summary_path),cfg['ticket'],cfg['plan']])
        for batch in summary['completed_batches']:
            cr=dict(path=batch['commit_path'],sha256=batch['commit_sha256']);commit=v.frozen_commit(cr)
            v.need(commit['schema']=='ms_unified_novlm_batch_commit_v3' and commit['edition']==phase and commit['parent_ticket_id']==ticket['ticket_id'] and commit['batch_index']==batch['batch_index'],'Frozen native commit differs')
            ids=[]
            for receipt in commit['receipts']:
                row=receipt['source_row'];id_=row['id'];outcome=outcomes[(batch['batch_index'],id_)]
                v.need(row==planmap[id_] and row==commit['archive_index'][id_]['source_row'] and outcome['status']==receipt['status'] and outcome['split']==row['split'],'Source row/summary/receipt differs')
                if receipt['status']!=a.SUCCESS or row['split']!='train' or id_.casefold() in used:continue
                a.eligible(row,receipt,used,seen);seen.add(id_.casefold());ids.append(id_);keys.append([row['split'],id_])
                if len(keys)==count:break
            if ids:selected.append(dict(commit_ref=cr,source_ids=ids))
            if len(keys)==count:break
        if len(keys)==count:break
    v.need(len(keys)==count,'Completed explicit streams have insufficient eligible new sources')
    return selected,keys,refs


def run(count,output,shared):
    v.need(type(count) is int and 1<=count<=5000,'Finite new source metadata supply required')
    v.need(v.base.sha(a.__file__)=='21e908102b20ba91533953c1e6d46af36f32e1622e8b54cf808870bbff1dbf75','Frozen original bridge changed')
    output=Path(output);shared=Path(shared)
    v.need(str(output).startswith('/root/ms_vlm_fastcache/') and output.is_absolute() and str(output.resolve())==str(output) and not output.exists(),'Fresh owned local metadata directory required')
    v.need(shared.is_absolute() and shared.parent==v.ROOT and not shared.exists(),'Fresh shared proof directory required')
    selected,keys,refs=select(count);output.mkdir(parents=True);shared.mkdir()
    selection=v.write_once(shared/'source_selection.json',dict(schema='ms_vlm_completed_native_new_selection_v1',source_count=count,source_keys=keys,selected_batches=selected,source_refs=refs,source_code_ref=v.ref(SELF),no_source_claims_created=True,no_model_calls=True,observed_unix=time.time()))
    def materialize(entry):
        commit=v.frozen_commit(entry['commit_ref']);rows=a.materialize(entry['commit_ref'],commit,entry['source_ids'],output,{})
        for row in rows:v.input_pair(row)
        receipt=v.write_once(output/'batch_receipts'/(hashlib.sha256(entry['commit_ref']['path'].encode()).hexdigest()+'.json'),dict(commit_ref=entry['commit_ref'],source_ids=entry['source_ids'],rows=rows,v8_input_pair_verified_count=len(rows),once_only_archive_hash=True,source_claims_created=0,model_calls=0))
        print(json.dumps(dict(event='new_native_metadata_materialized',sources=len(rows),receipt=receipt)),flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=4) as executor:batches=list(executor.map(materialize,selected))
    rows=[r for b in batches for r in b];v.need([[r['split'],r['id']] for r in rows]==keys,'Source materialization selection/order differs')
    raw=b''.join(v.h.bench.encoded(r) for r in rows);plan=a.write_raw_once(shared/'manifest.jsonl',raw)
    proof=v.write_once(shared/'input_proof.json',dict(schema='ms_vlm_new_native_local_materialization_v1',source_count=count,plan_ref=plan,selection_ref=selection,selected_source_ids=[r['id'] for r in rows],local_input_root=str(output),v8_input_pair_verified_count=count,source_claims_created=0,model_calls=0,existing_archive_rehashes=0,new_archive_hash_reads=len(selected),no_npz_decoded=True,no_media_input_read=True,original_raw_bytes_preserved=True,quality_approved=False,training_eligible=False))
    print(json.dumps(dict(event='new_native_supply_complete',proof_ref=proof,plan_ref=plan,sources=count)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--shared',required=True);x=p.parse_args();run(x.count,x.output,x.shared)
