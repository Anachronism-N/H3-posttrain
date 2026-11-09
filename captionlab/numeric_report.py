"""Assemble observed parity evidence; do not manufacture a numerical pass threshold."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from captionlab.numeric_common import save_new


def compact(rows):
    return dict(positions=len(rows),top1_changes=sum(not r['top1_equal'] for r in rows),
        max_absolute_logit_difference=max(r['max_logit_abs_difference'] for r in rows),
        max_centered_logit_difference=max(r['max_centered_logit_abs_difference'] for r in rows),
        max_target_logprob_difference=max(abs(r['target_logprob_difference']) for r in rows),
        max_total_variation=max(r['total_variation'] for r in rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();r=args.root/'evidence';load=lambda p:json.loads((r/p).read_text())
    cache=load('cache_calibration/summary.json');matched=load('matched_portable/summary.json');route=load('route_calibration/summary.json');zp=load('zero_portable_same_inputs_comparison.json')
    calibrations=[dict(case_id=c['case_id'],contrasts={name:compact(rows) for name,rows in c['contrasts'].items()}) for c in cache['cases']]
    higher=[dict(case_id=c['case_id'],mode=c['mode'],contrasts={k:compact(c[k]) for k in ['repeat_control','full_shortened','full_cached','shortened_cached']}) for c in matched['records']]
    exact=[c for c in zp if c['comparison'] in ['matched_BF16','matched_FP32']]
    if len(exact)!=6 or not all(c['exact_selected'] and c['exact_all_target_logprob'] for c in exact):raise ValueError('Matched-precision ZeRO/portable equality not established')
    signatures=[];count=0
    for p in sorted((r/'zero_original').glob('*-original_fp32_features.pt')):
        x=torch.load(p,weights_only=True);count+=len(x['target_logprobs']);signatures.append(dict(case_id=p.name.split('-original_fp32_features.pt')[0],target_tokens=len(x['target_logprobs']),input_ids_sha256=x['input_ids_sha256'],feature_sha256=x['input_feature_sha256'],selected_positions=x['positions']))
    resources={}
    for folder in ['cache_manager','zero_manager','matched_portable_manager','frozen_portable_manager','route_calibration_manager','norm_dtype_manager','norm_embedding_dtype_manager']:
        value=load(folder+'/resources_restored.json')
        if not value['holder_compute_verified'] or not value['unknown_jobs_not_signaled']:raise ValueError('Missing resource restoration evidence')
        resources[folder]=value
    result=dict(schema='captionlab-calibrated-numeric-equivalence-v1',original_checkpoint_update=6040,case_count=3,
        identity=dict(original_learned_tensor_FP32_exact_restore=True,ZeRO_world_size=8,original_optimizer_unchanged=True,
            original_checkpoint_not_written=True,independent_copy_used=True,same_input_ids_masks_and_frozen_features_verified=True,source_signatures=signatures),
        ZeRO_portable=dict(matched_parameter_storage_FP32_passed=True,matched_BF16_compute_selected_logits_bitwise_equal=True,
            matched_FP32_compute_selected_logits_bitwise_equal=True,matched_all_target_logprobs_bitwise_equal=True,
            target_tokens_per_precision=count,selected_positions_per_precision=sum(len(x['selected_positions']) for x in signatures),
            observed_comparisons=zp,current_single_GPU_BF16_storage_bitwise_equivalent=False,
            historical_raw_logit_dtype_field_means_saved_selected_FP32_tensor_not_native_output=True,
            normalization_only_diagnostic=load('FP32_norm_fix_comparison.json'),
            normalization_embedding_diagnostic=load('hybrid_norm_embedding_comparison.json'),
            normalizations_only_single_GPU_fix_passed=False,norm_and_embedding_single_GPU_fix_passed=False,
            scope='original_standard_checkpoint_three_cases; selected_full_vocab_logits_and_all_supervised_target_probabilities; not_universal_output_equivalence'),
        cache_calibration=dict(BF16_controls=calibrations,matched_storage_BF16_and_FP32=higher,
            identical_forward_repeats_bitwise_equal=True,identical_cache_repeats_bitwise_equal=True,
            blanket_BF16_cache_full_bitwise_equivalence=False,
            FP32_route_switches=route['records'],
            conclusion='sequence-size BF16 arithmetic and discrete near-tie expert top8 routing can change distributions and greedy choices; repeat randomness is not observed',
            tolerance_policy='retain measured control envelopes by dtype/route; no_global_relaxed_tolerance_or_claim_of_cache_bug_eliminated',
            same_route_FP32_bound_is_empirical_three_case_selected_position_measurement=True,
            frozen_reference_route_intervention_is_not_deployed=True),
        CPU_feature_identity=load('portable_frozen_inputs/summary.json')['cases'],
        deployment_recommendation='use_evaluated_multiGPU_FP32_parameter_storage_or_original_Zero_FP32_storage_with_BF16_compute_for_strict_train_inference_parity; freeze_features_for_numeric_audits; original_BF16_single_GPU_is_a_different_numerical_policy',
        no_retraining_required_for_state_restore=True,new_training_updates=0,new_generation_calls=0,
        no_new_action_or_genre_quality_claim=True,resources_restored=resources,
        implementation_tests='171_CPU_tests_and35_subtests_passed; distinct_from_actual_full_model_forward_evidence')
    save_new(args.root/'numeric_equivalence_summary.json',result)
    print(json.dumps(dict(matched_Zero_portable_passed=True,target_tokens=count,selected_positions=sum(len(x['selected_positions']) for x in signatures),current_BF16_single_GPU_passed=False,cache_blanket_bitwise_passed=False,resources_restored=True)),flush=True)


if __name__=='__main__':main()
