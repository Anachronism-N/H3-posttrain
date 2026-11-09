"""Calibrate BF16 cached decoding against repeated and length-matched forwards."""
import argparse
import json
from pathlib import Path
import torch
from captionlab.numeric_common import rows_from_manifest,fixed_positions,snapshot,compare,save_new,digest_tensor
from captionlab.runner import audit_runtime,_load_runtime


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Preserve calibration output')
    args.out.mkdir(parents=True)
    config=json.loads(args.runtime.read_text());audit=audit_runtime(config,args.runtime.parent)
    loaded=_load_runtime(config,audit);a=loaded['adapter'];model=a.model;model.eval()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=rows_from_manifest(args.manifest);results=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for row in rows:
            batch=a.collate([row]);n,positions=fixed_positions(a,batch);inputs=a._move(batch)
            full=model(**inputs,use_cache=False);first=snapshot(full.logits,batch,positions);del full
            torch.save(first,args.out/(row['id']+'-portable_adapter_move.pt'))
            full=model(**inputs,use_cache=False);repeat=snapshot(full.logits,batch,positions);del full
            raw={k:v.to(next(model.parameters()).device) if isinstance(v,torch.Tensor) else v for k,v in batch.items() if k!='labels'}
            output=model(**raw,use_cache=False);raw_snapshot=snapshot(output.logits,batch,positions);del output
            torch.save(raw_snapshot,args.out/(row['id']+'-portable_fp32_features.pt'))
            shortened=[]
            for position in positions:
                subset=dict(inputs)
                for k in ['input_ids','attention_mask','token_type_ids','mm_token_type_ids']:
                    if k in subset:subset[k]=subset[k][:,:position+1]
                output=model(**subset,use_cache=False)
                shortened.append(output.logits[0,-1].float().cpu());del output
            shortened=torch.stack(shortened)
            cached_runs=[]
            for run in range(2):
                prefix=a._move(a.prefix(row));output=model(**prefix,use_cache=True);past=output.past_key_values
                selected={n-1:output.logits[0,-1].float().cpu()};del output
                native=model.get_base_model() if hasattr(model,'get_base_model') else model
                delta=native.rope_deltas.detach().clone()
                for position in range(n,batch['input_ids'].shape[1]-1):
                    token=batch['input_ids'][:,position:position+1].to(next(model.parameters()).device)
                    pos=torch.full((3,1,1),position,dtype=torch.long,device=token.device)+delta.to(token.device).unsqueeze(0)
                    output=model(input_ids=token,position_ids=pos,attention_mask=batch['attention_mask'][:,:position+1].to(token.device),past_key_values=past,use_cache=True)
                    past=output.past_key_values
                    if position in positions:selected[position]=output.logits[0,-1].float().cpu()
                    del output
                cached_runs.append(torch.stack([selected[p] for p in positions]));del past
            targets=[int(batch['input_ids'][0,pos+1]) for pos in positions]
            contrasts={
                'repeat_full':compare(first['selected_logits'],repeat['selected_logits'],targets),
                'full_vs_shortened':compare(first['selected_logits'],shortened,targets),
                'repeat_cache':compare(cached_runs[0],cached_runs[1],targets),
                'full_vs_cache':compare(first['selected_logits'],cached_runs[0],targets),
                'shortened_vs_cache':compare(shortened,cached_runs[0],targets),
                'fp32_features_vs_adapter_move':compare(raw_snapshot['selected_logits'],first['selected_logits'],targets)}
            record=dict(case_id=row['id'],positions=positions,target_ids=targets,contrasts=contrasts,
                full_target_logprob_mean=float(first['target_logprobs'].mean()),raw_feature_target_logprob_mean=float(raw_snapshot['target_logprobs'].mean()),
                adapter_moved_feature_dtype=str(inputs['input_features'].dtype),raw_feature_dtype=str(raw['input_features'].dtype),
                input_ids_sha256=first['input_ids_sha256'],labels_sha256=first['labels_sha256'],
                correct_fixed_sequence_and_native_rope_delta=True)
            results.append(record);save_new(args.out/(row['id']+'-calibration.json'),record)
            torch.save(dict(shortened=shortened,cached=cached_runs[0],cache_repeat=cached_runs[1]),args.out/(row['id']+'-controls.pt'))
            print(json.dumps(dict(case=row['id'],checked_positions=len(positions),full_repeat_max=max(c['max_logit_abs_difference'] for c in contrasts['repeat_full']),cache_repeat_max=max(c['max_logit_abs_difference'] for c in contrasts['repeat_cache']),cache_top1_changes=sum(not c['top1_equal'] for c in contrasts['full_vs_cache']))),flush=True)
    save_new(args.out/'summary.json',dict(schema='captionlab-cache-numeric-calibration-v1',cases=results,
        calibration_controls=['identical_full_repeat','same_target_prefix_full_no_cache','identical_cache_repeat','fp32_vs_bf16_input_features'],
        dtype='bf16-base-FP32-adapter-BF16-autocast',torch_tf32=False,forward_only=True,new_training_updates=0,new_generation_calls=0,
        model_receipt=loaded['receipt'],acceptance_not_assumed=True,scope='three_cases_selected_positions_not_all_models_or_all_tokens'))


if __name__=='__main__':main()
