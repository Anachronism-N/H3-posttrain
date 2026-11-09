"""Replay exactly the original ZeRO feature tensors through ordinary portable inference."""
import argparse
import json
from pathlib import Path
import torch
from captionlab.numeric_common import rows_from_manifest,fixed_positions,snapshot,compare,save_new,digest_tensor
from captionlab.runner import audit_runtime,_load_runtime


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--frozen-batches',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--fp32-norms',action='store_true');p.add_argument('--fp32-embeddings',action='store_true')
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Do not overwrite old parity evidence')
    args.out.mkdir(parents=True)
    cfg=json.loads(args.runtime.read_text());loaded=_load_runtime(cfg,audit_runtime(cfg,args.runtime.parent));a=loaded['adapter'];a.model.eval()
    norm_modules=[];embedding_modules=[]
    if args.fp32_embeddings:
        for name,module in a.model.named_modules():
            if isinstance(module,torch.nn.Embedding):
                module.float();embedding_modules.append(name)
    if args.fp32_norms:
        for name,module in a.model.named_modules():
            if isinstance(module,torch.nn.LayerNorm) or type(module).__name__.endswith('RMSNorm'):
                module.float();norm_modules.append(name)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    records=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for row in rows_from_manifest(args.manifest):
            native=a.collate([row]);frozen=torch.load(args.frozen_batches/(row['id']+'.pt'),map_location='cpu',weights_only=True)
            for key in ['input_ids','labels','attention_mask','feature_attention_mask']:
                if not torch.equal(native[key],frozen[key]):raise ValueError('Frozen/native token/mask differs: '+key)
            diff=(native['input_features']-frozen['input_features']).abs()
            record=dict(case_id=row['id'],source_feature_max_difference=float(diff.max()),source_feature_mean_difference=float(diff.mean()),bf16_features_equal=torch.equal(native['input_features'].to(torch.bfloat16),frozen['input_features'].to(torch.bfloat16)),source_feature_hash=digest_tensor(native['input_features']),frozen_feature_hash=digest_tensor(frozen['input_features']),forwards=[])
            for name,batch in [('native_cpu_features',native),('frozen_zero_features',frozen)]:
                n,positions=fixed_positions(a,batch)
                for precision in ['adapter_move','preserve_fp32_features']:
                    inputs=a._move(batch)
                    if precision=='preserve_fp32_features':inputs['input_features']=batch['input_features'].to(next(a.model.parameters()).device)
                    output=a.model(**inputs,use_cache=False);s=snapshot(output.logits,batch,positions);del output
                    torch.save(s,args.out/(row['id']+'-'+name+'-'+precision+'.pt'))
                    record['forwards'].append(dict(name=name,precision=precision,target_NLL=-float(s['target_logprobs'].mean()),actual_input_features_hash=digest_tensor(inputs['input_features']),actual_input_dtype=str(inputs['input_features'].dtype)))
            records.append(record);print(json.dumps(record),flush=True)
    save_new(args.out/'summary.json',dict(schema='captionlab-portable-frozen-feature-replay-v1',cases=records,original_adapter_recipe_unchanged=not args.fp32_norms,FP32_norm_modules=norm_modules,FP32_embedding_modules=embedding_modules,norm_dtype_intervention_diagnostic_only=args.fp32_norms or args.fp32_embeddings,forward_only=True,model_receipt=loaded['receipt']))


if __name__=='__main__':main()
