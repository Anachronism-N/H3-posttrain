"""Portable adapter with FP32 base storage on multiple GPUs for matched-dtype audits."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import torch
from captionlab.numeric_common import rows_from_manifest,fixed_positions,snapshot,compare,save_new,fixed_sequence_cache_logits


def main():
    p=argparse.ArgumentParser();p.add_argument('--original-code',type=Path,required=True);p.add_argument('--run-config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Preserve earlier portable storage evidence')
    args.out.mkdir(parents=True)
    sys.path.insert(0,str(args.original_code))
    tr=importlib.import_module('scripts_omnidance.exp.ms.music_planner_train');ad=importlib.import_module('scripts_omnidance.exp.ms.music_planner_adapters');schema=importlib.import_module('scripts_omnidance.exp.ms.music_planner_schema')
    cfg=json.loads(args.run_config.read_text());tr._force_fp32_moe_fallback()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # The official base is first loaded in BF16, exactly as in original training;
    # promoting that same rounded payload to FP32 adds no new base information.
    native=ad.load_planner_adapter('omni_captioner',cfg['model']['path'],checkpoint=args.checkpoint,device_map='balanced',max_length=8192)
    adapter=schema.ConfiguredPlannerAdapter(native,contract=cfg['data']['contract']);model=adapter.model
    model.float();model.eval()
    if not all(p.dtype==torch.float32 for p in model.parameters() if p.is_floating_point()):raise ValueError('Matched FP32 parameter storage failed')
    actual=model.hf_device_map
    if any(v in ('cpu','disk') for v in actual.values()):raise ValueError('Unexpected CPU/disk offload in parity model')
    results=[]
    for row in rows_from_manifest(args.manifest):
        batch=adapter.collate([row]);n,positions=fixed_positions(adapter,batch);inputs=adapter._move(batch)
        for mode in ['bf16_autocast','fp32_reference']:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=mode=='bf16_autocast'):
                output=model(**inputs,use_cache=False);first=snapshot(output.logits,batch,positions);del output
                output=model(**inputs,use_cache=False);repeat=snapshot(output.logits,batch,positions);del output
                short=[]
                for position in positions:
                    subset=dict(inputs)
                    for k in ['input_ids','attention_mask','token_type_ids','mm_token_type_ids']:
                        if k in subset:subset[k]=subset[k][:,:position+1]
                    output=model(**subset,use_cache=False);short.append(output.logits[0,-1].detach().float().cpu().clone());del output
                short=torch.stack(short)
                cached=fixed_sequence_cache_logits(adapter,batch,positions)
            targets=[int(batch['input_ids'][0,p+1]) for p in positions];control=compare(first['selected_logits'],repeat['selected_logits'],targets)
            torch.save(first,args.out/(row['id']+'-'+mode+'.pt'))
            r=dict(case_id=row['id'],mode=mode,repeat_control=control,full_shortened=compare(first['selected_logits'],short,targets),full_cached=compare(first['selected_logits'],cached,targets),shortened_cached=compare(short,cached,targets),input_feature_dtype=str(inputs['input_features'].dtype),target_logprob_mean=float(first['target_logprobs'].mean()),same_base_values_as_BF16_initialization=True)
            results.append(r);print(json.dumps(dict(case=row['id'],mode=mode,NLL=-r['target_logprob_mean'],repeat_max=max(c['max_logit_abs_difference'] for c in control))),flush=True)
    save_new(args.out/'summary.json',dict(schema='captionlab-portable-matched-FP32-storage-forward-v1',records=results,HF_device_map=actual,base_FP32_storage=True,learned_weights_restored_exact_FP32=True,new_training_updates=0,new_generation_calls=0,tf32=False,moe_fp32_guard='same_original_trainer_guard'))


if __name__=='__main__':main()
