"""Explain FP32 length sensitivity with observed router traces and a local intervention."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import torch
from captionlab.numeric_common import rows_from_manifest,fixed_positions,compare,save_new


def main():
    p=argparse.ArgumentParser();p.add_argument('--original-code',type=Path,required=True);p.add_argument('--run-config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--previous-calibration',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Preserve prior route trace')
    args.out.mkdir(parents=True)
    sys.path.insert(0,str(args.original_code));tr=importlib.import_module('scripts_omnidance.exp.ms.music_planner_train');ad=importlib.import_module('scripts_omnidance.exp.ms.music_planner_adapters');schema=importlib.import_module('scripts_omnidance.exp.ms.music_planner_schema');cfg=json.loads(args.run_config.read_text())
    tr._force_fp32_moe_fallback();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    native=ad.load_planner_adapter('omni_captioner',cfg['model']['path'],checkpoint=args.checkpoint,device_map='balanced',max_length=8192);a=schema.ConfiguredPlannerAdapter(native,contract=cfg['data']['contract']);model=a.model;model.float();model.eval()
    capture={};reference={};mode='full';handles=[]
    def hook(name):
        def run(module,inputs,output):
            logits,weights,indices=output
            capture[name]=dict(logits=logits.detach().float().cpu().clone(),indices=indices.detach().cpu().clone())
            if mode=='forced_reference_routes':
                ref=reference[name]['indices'][:len(logits)].to(logits.device)
                probs=logits.softmax(-1,dtype=torch.float32);scores=probs.gather(1,ref)
                if module.norm_topk_prob:scores=scores/scores.sum(-1,keepdim=True)
                return logits,scores.to(logits.dtype),ref
        return run
    for name,module in model.named_modules():
        if name.endswith('.mlp.gate') and type(module).__name__=='Qwen3OmniMoeThinkerTextTopKRouter':handles.append(module.register_forward_hook(hook(name)))
    if not handles:raise ValueError('No actual native MoE routers to trace')
    previous=json.loads(args.previous_calibration.read_text());records=[]
    with torch.no_grad():
        for row in rows_from_manifest(args.manifest):
            prior=next(r for r in previous['records'] if r['case_id']==row['id'] and r['mode']=='fp32_reference')
            batch=a.collate([row]);n,positions=fixed_positions(a,batch);worst=max(range(len(positions)),key=lambda i:abs(prior['full_shortened'][i]['target_logprob_difference']));position=positions[worst];target=int(batch['input_ids'][0,position+1]);inputs=a._move(batch)
            capture={};mode='full';out=model(**inputs,use_cache=False);full_logits=out.logits[0,position].float().cpu().unsqueeze(0);del out;reference=capture;capture={}
            short=dict(inputs)
            for k in ['input_ids','attention_mask','token_type_ids','mm_token_type_ids']:
                if k in short:short[k]=short[k][:,:position+1]
            mode='short';out=model(**short,use_cache=False);short_logits=out.logits[0,-1].float().cpu().unsqueeze(0);del out;actual=capture;capture={}
            route_changes=[]
            for name,ref in reference.items():
                cur=actual[name];limit=len(cur['indices']);rids=ref['indices'][:limit];cids=cur['indices'];different=(rids.sort(-1).values!=cids.sort(-1).values).any(-1)
                if bool(different.any()):
                    j=int(different.nonzero()[0]);refp=ref['logits'][j].softmax(-1).topk(9);curp=cur['logits'][j].softmax(-1).topk(9)
                    route_changes.append(dict(layer=name,changed_tokens=int(different.sum()),first_changed_token=j,reference_top8=rids[j].tolist(),candidate_top8=cids[j].tolist(),reference_8th_9th_gap=float(refp.values[7]-refp.values[8]),candidate_8th_9th_gap=float(curp.values[7]-curp.values[8]),max_gate_logit_difference_at_token=float((ref['logits'][j]-cur['logits'][j]).abs().max())))
            mode='forced_reference_routes';out=model(**short,use_cache=False);forced_logits=out.logits[0,-1].float().cpu().unsqueeze(0);del out
            record=dict(case_id=row['id'],tested_target_position=position,target_id=target,actual_native_router_count=len(handles),route_changes=route_changes,first_changed_layer=route_changes[0] if route_changes else None,
                full_vs_short=compare(full_logits,short_logits,[target])[0],full_vs_short_fixed_reference_routes=compare(full_logits,forced_logits,[target])[0],
                intervention='use_full_forward_top8_expert_IDs_on_same_prefix_only; recompute_current_probabilities; diagnostic_not_generation_policy')
            records.append(record);print(json.dumps(record),flush=True)
    for handle in handles:handle.remove()
    save_new(args.out/'summary.json',dict(schema='captionlab-FP32-router-boundary-calibration-v1',records=records,new_training_updates=0,new_generation_calls=0,model_weights_unchanged=True,intervention_not_deployed=True))


if __name__=='__main__':main()
