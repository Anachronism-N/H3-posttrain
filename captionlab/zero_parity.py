"""Restore the original eight-rank ZeRO checkpoint; forward only with audits."""
import argparse
from contextlib import nullcontext
import importlib
import json
import os
from pathlib import Path
import sys
import torch
from captionlab.numeric_common import rows_from_manifest,fixed_positions,snapshot,compare,save_new


def main():
    p=argparse.ArgumentParser();p.add_argument('--original-code',type=Path,required=True);p.add_argument('--run-config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fp32-compute',action='store_true');args=p.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local_rank=int(os.environ['LOCAL_RANK'])
    if world!=8:raise ValueError('Exact original ZeRO world size8 required')
    torch.cuda.set_device(local_rank);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(args.original_code))
    tr=importlib.import_module('scripts_omnidance.exp.ms.music_planner_train')
    ad=importlib.import_module('scripts_omnidance.exp.ms.music_planner_adapters')
    schema=importlib.import_module('scripts_omnidance.exp.ms.music_planner_schema')
    cfg=json.loads(args.run_config.read_text())
    if tr.file_sha(Path(ad.__file__))!=cfg['adapter_source_sha256']:raise ValueError('Actual adapter code differs from trained snapshot')
    for name,expected in cfg['source_sha256'].items():
        module=tr if name=='music_planner_train' else tr.planner_module(name)
        if tr.file_sha(module.__file__)!=expected:raise ValueError('Actual original runtime differs: '+name)
    if cfg.get('jev') or cfg.get('count'):raise ValueError('This bounded audit uses the original standard no-head checkpoint')
    tr._allow_external_autocast();tr._force_fp32_moe_fallback()
    from accelerate import Accelerator,DeepSpeedPlugin
    from accelerate.utils import GradientAccumulationPlugin,set_seed
    ds=json.loads(json.dumps(cfg['resolved_deepspeed_config']))
    if args.fp32_compute:ds['torch_autocast']={'enabled':False}
    plugin=DeepSpeedPlugin(hf_ds_config=ds,zero3_init_flag=True)
    acc=Accelerator(mixed_precision='no',deepspeed_plugin=plugin,gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=cfg['accum'],sync_with_dataloader=False))
    set_seed(cfg['seed'])
    if rank==0:
        if args.out.exists():raise FileExistsError('Preserve previous ZeRO numeric evidence')
        args.out.mkdir(parents=True)
    acc.wait_for_everyone()
    with plugin.zero3_init_context_manager(enable=True):
        adapter=ad.load_planner_adapter('omni_captioner',cfg['model']['path'],lora_rank=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'],audio_last_n=cfg['audio_last_n'],max_length=cfg['max_length'])
        adapter=schema.ConfiguredPlannerAdapter(adapter,contract=cfg['data']['contract'])
    model=adapter.model;model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
    base_dtype=tr.prepare_training_storage(model,ad.load_planner_adapter,True)
    leaves=tr.configure_zero3_leaf_modules(model,'omni_captioner',True)
    groups,names=tr.optimizer_groups(model,cfg['lora_lr'],cfg['projector_lr'],base_dtype=base_dtype)
    optimizer=torch.optim.AdamW(groups,weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:tr.lr_multiplier(step,cfg['total_updates'],cfg['effective_warmup_updates'],cfg['lr_schedule']))
    model,optimizer=acc.prepare(model,optimizer)
    # Copy is in an independent audit directory. Original optimizer state and
    # checkpoints are never updated, overwritten or augmented by resume metadata.
    cursor=tr.load_checkpoint(args.checkpoint,model,optimizer,scheduler,cfg['fingerprint'],rank=rank,world=world,zero3=True,barrier=acc.wait_for_everyone)
    before=tr.optimizer_evidence(optimizer)
    if rank==0:save_new(args.out/'restore_receipt.json',dict(original_fingerprint=cfg['fingerprint'],cursor=cursor,actual_trainable_tensors_match_saved_FP32=True,optimizer_restored_exact=True,checkpoint_copy=str(args.checkpoint),ds_config=ds,moe_leaves=leaves,diagnostic_only=True,compute='float32' if args.fp32_compute else 'original_bf16_autocast'))
    model.eval();records=[]
    with torch.no_grad():
        for row in rows_from_manifest(args.manifest):
            batch=adapter.collate([row]);n,positions=fixed_positions(adapter,batch)
            for variant in (['original_fp32_features'] if args.fp32_compute else ['original_fp32_features','adapter_bf16_features']):
                inputs=tr.move_batch(batch,acc.device)
                if variant=='adapter_bf16_features':
                    for key,value in inputs.items():
                        if isinstance(value,torch.Tensor) and value.is_floating_point():inputs[key]=value.to(torch.bfloat16)
                output=model(**inputs,use_cache=False);raw_output_dtype=str(output.logits.dtype);first=snapshot(output.logits,batch,positions);del output
                tr.finish_zero3_forward(model,True)
                output=model(**inputs,use_cache=False);repeat=snapshot(output.logits,batch,positions);del output
                tr.finish_zero3_forward(model,True)
                targets=[int(batch['input_ids'][0,x+1]) for x in positions]
                controls=compare(first['selected_logits'],repeat['selected_logits'],targets)
                # All ranks consume the same actual waveform/sequence. Cross-rank
                # output differences therefore cannot be blamed on different cases.
                ref=first['selected_logits'].to(acc.device);torch.distributed.broadcast(ref,src=0)
                rank_max=(ref-first['selected_logits'].to(acc.device)).abs().max();torch.distributed.all_reduce(rank_max,op=torch.distributed.ReduceOp.MAX)
                record=dict(case_id=row['id'],variant=variant,positions=positions,input_feature_dtype=str(inputs['input_features'].dtype),
                    repeat_full_controls=controls,max_cross_rank_logit_difference=float(rank_max),input_ids_sha256=first['input_ids_sha256'],labels_sha256=first['labels_sha256'],target_logprob_mean=float(first['target_logprobs'].mean()),dtype='fp32_compute' if args.fp32_compute else 'bf16_autocast',raw_logit_dtype=raw_output_dtype)
                records.append(record)
                if rank==0:
                    torch.save(first,args.out/(row['id']+'-'+variant+'.pt'))
                    save_new(args.out/(row['id']+'-'+variant+'.json'),record)
                    print(json.dumps(dict(case=row['id'],variant=variant,cross_rank_difference=float(rank_max),repeat_difference=max(c['max_logit_abs_difference'] for c in controls),target_NLL=-float(first['target_logprobs'].mean()))),flush=True)
                del first,repeat,inputs,ref
    tr.finish_zero3_forward(model,True)
    after=tr.optimizer_evidence(optimizer)
    if before!=after:raise ValueError('Forward-only audit changed optimizer state')
    if rank==0:save_new(args.out/'summary.json',dict(schema='captionlab-original-zero3-forward-parity-v1',cases=records,actual_learned_FP32_tensor_restore_verified=True,optimizer_state_unchanged=True,original_cursor=cursor,new_training_updates=0,new_generation_calls=0,zero3_world_size=world,FP32_reference=args.fp32_compute,original_base_bf16_loaded_into_FP32_storage=True,raw_model_output_dtype_not_inferred=True))
    acc.wait_for_everyone()
    torch.distributed.destroy_process_group()


if __name__=='__main__':main()
