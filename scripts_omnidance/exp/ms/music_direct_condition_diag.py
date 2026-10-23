"""Read-only fixed-three waveform condition diagnosis for complete format4 models."""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import time
import torch
from torch.nn import functional as F
from music_text_adapter import (DirectMusicToH3Text, load_direct_music_text_checkpoint,
                               resample_audio32k_to24k, save_direct_music_text_checkpoint)
from music_text_contrastive import _DirectCPUBackbone, score_samples
from music_clip_bridge import load_audio
from music_direct_compare_manager_v3 import identity, snapshot, sha, HOLDER

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
MS = ROOT / 'diff/logs_omnidance/exp/MS'
PROTOCOL = 'fixed3_complete_mean_token_condition_diag_v1'
IDS = ['wJPptuhclZmKAEWs','wJPrpFsThWAcBQIw','wJQTclrlOLEYYfTB']
SCOPE = [('mean', MS/'music_direct_mean100_n2_v3/last.pt',
          '417d64d2d7cbee3d44f36b82c4eaf192c3959caf9aface9e6411d4e9aef78a82'),
         ('token', MS/'music_direct_token100_n3_v3/last.pt',
          'f8350e14b44a882a02fb47c784c5875fd145e1dac94b5d35cfb44d1365f68473')]
MANIFEST = MS/'music_direct6k_v2_training_manifest_v2.json'
MANIFEST_SHA = '40df4ff51bb020a9296ad2c54e9c9c069ef9b1e9710ad2c234d3d7782ea2c6e6'
JOINT = MS/'music_direct_eval3_generation_joint_v1.json'
UUID = 'GPU-d89362d8-8b22-575c-2be3-86eba53ac29c'


def save(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def tensor_info(x):
    x = x.detach().cpu().contiguous()
    b = x.view(torch.uint8).numpy().tobytes()
    return dict(shape=list(x.shape), dtype=str(x.dtype), sha256=hashlib.sha256(b).hexdigest(),
                finite=bool(torch.isfinite(x).all()), rms=float(x.float().square().mean().sqrt()))


def inputs():
    joint = json.loads(JOINT.read_text())
    assert joint['condition']=='joint' and [r['sample_id'] for r in joint['cases']]==IDS
    split_source = joint['source_files']['split']
    assert sha(split_source['path'])==split_source['sha256']
    fixed_split = json.loads(Path(split_source['path']).read_text())
    assert fixed_split['val']==IDS and fixed_split['train']==[]
    assert sha(MANIFEST)==MANIFEST_SHA
    manifest = json.loads(MANIFEST.read_text())
    meta = manifest['metadata']
    original = meta['source_paths_hashes']['split']
    assert sha(original['path'])==original['sha256']
    original_split = json.loads(Path(original['path']).read_text())
    assert original_split['train']==meta['train_ids'] and original_split['val']==meta['val_ids']
    assert all(sid not in meta['train_ids'] and sid not in meta['val_ids'] for sid in IDS)
    stats = {k:torch.tensor(manifest['statistics'][k],dtype=torch.float32) for k in ('mean','std')}
    assert stats['mean'].shape==stats['std'].shape==(5120,) and (stats['std']>0).all()
    del manifest
    waves, provenance = {}, []
    for row in joint['cases']:
        sid, path = row['sample_id'], Path(row['audio_path'])
        assert sha(path)==row['audio_sha256']
        wave, effective = load_audio(path.stem, path.parent)
        assert wave.shape==(2,162000) and wave.dtype==torch.float32 and torch.isfinite(wave).all()
        resampled = resample_audio32k_to24k(wave)
        assert resampled.shape==(2,121500)
        waves[sid]=wave
        provenance.append(dict(sample_id=sid,source_audio=str(path),audio_sha256=sha(path),
                               input_wave=tensor_info(wave),resampled_24k_wave=tensor_info(resampled),
                               effective_frames=effective))
    assert len({v['input_wave']['sha256'] for v in provenance})==3
    isolation=dict(fixed_generation_split=split_source,original_train_val_split=original,
        fixed_split_partition='val',not_in_6000_train=True,not_in_500_validation=True,
        role='three extra original joint generation holdouts, not 500-validation retrieval evaluation',
        train_count=len(meta['train_ids']),validation_count=len(meta['val_ids']))
    return joint['cases'],waves,stats,provenance,isolation


def cpu_check(args):
    import tempfile
    rows,waves,stats,provenance,isolation=inputs()
    model=DirectMusicToH3Text(_DirectCPUBackbone(),freeze_backbone=False).eval()
    with tempfile.TemporaryDirectory(prefix='condition_loader_fixture_') as directory:
        checkpoint=Path(directory)/'model.pt'
        save_direct_music_text_checkpoint(checkpoint,model)
        fresh=load_direct_music_text_checkpoint(checkpoint,backbone=_DirectCPUBackbone()).eval()
        assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in model.state_dict().items())
        with torch.inference_mode():
            expected=model(waves[IDS[0]][None]);actual=fresh(waves[IDS[0]][None])
        assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    scopes=[]
    for arm,path,checksum in SCOPE:
        assert sha(path)==checksum
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        metadata,state=checkpoint['metadata'],checkpoint['model']
        assert metadata['format_version']==4 and len(state)==467
        assert metadata['context_policy']=='forbidden' and metadata['output_normalization']=='none_raw_hidden'
        assert set(k for k in state if k.startswith('adapter.'))=={'adapter.projection.weight','adapter.projection.bias'}
        assert all(torch.isfinite(v).all() for v in state.values())
        scopes.append(dict(arm=arm,path=str(path),sha256=checksum,state_tensors=467,
                           backbone_model_path=metadata['backbone_metadata']['model_path'],
                           all_linear_weight_and_bias_present=True))
        del checkpoint,state,metadata
        gc.collect()
    save(args.cpu_report,dict(protocol=PROTOCOL,complete=True,script_sha256=sha(__file__),
        actual_musicfm_forward=False,fixture_full_state_loader_exact=True,
        actual_checkpoint_source_and_full_state_structure_verified=scopes,
        actual_wave_loader=provenance,isolation=isolation))
    print(json.dumps(dict(event='cpu_contract_complete',report=str(args.cpu_report))),flush=True)


def full_set_score(x,y,mean,std):
    x=F.normalize((x.float()-mean)/std,dim=-1)
    y=F.normalize((y.float()-mean)/std,dim=-1)
    similarity=x@y.T
    return .5*(.1*(torch.logsumexp(similarity/.1,dim=1)-math.log(len(y))).mean()
               +.1*(torch.logsumexp(similarity/.1,dim=0)-math.log(len(x))).mean())


def sampled_score(x,y,mean,std):
    length=max(len(x),len(y))
    sx=F.pad(x,(0,0,0,length-len(x)))[None]
    ty=F.pad(y,(0,0,0,length-len(y)))[None]
    mask=torch.arange(length)<len(y)
    own=torch.arange(length)<len(x)
    return float(score_samples(sx,ty,mask[None],mean,std,limit=64,temperature=.1,student_mask=own[None])[0])


def token_summary(x,channel_major):
    x=x.float()
    center=x-x.mean(0)
    cosine=F.normalize(x,dim=-1)@F.normalize(x,dim=-1).T
    nonself=~torch.eye(len(x),dtype=torch.bool)
    if channel_major:
        assert len(x)==254
        seq=x.reshape(2,127,5120)
        delta=(seq[:,1:]-seq[:,:-1]).flatten(0,1)
        shuffled=seq[:,torch.randperm(127)]
        shuffled_delta=(shuffled[:,1:]-shuffled[:,:-1]).flatten(0,1)
        reverse_delta=(seq.flip(1)[:,1:]-seq.flip(1)[:,:-1]).flatten(0,1)
    else:
        delta=x[1:]-x[:-1]
        shuffled=x[torch.randperm(len(x))]
        shuffled_delta=shuffled[1:]-shuffled[:-1]
        reverse_delta=x.flip(0)[1:]-x.flip(0)[:-1]
    rms=float(x.square().mean().sqrt());centered=float(center.square().mean().sqrt())
    eig=torch.linalg.eigvalsh(center@center.T).clamp_min(0)
    probability=eig/eig.sum().clamp_min(1e-12)
    rank=float(torch.exp(-(probability*probability.clamp_min(1e-12).log()).sum())) if centered>0 else 0.
    return dict(tensor=tensor_info(x),token_centered_rms=centered,
        centered_over_total_rms=centered/max(rms,1e-12),effective_centered_rank=rank,
        exact_unique_tokens=int(torch.unique(x,dim=0).shape[0]),
        mean_offdiagonal_raw_token_cosine=float(cosine[nonself].mean()),
        near_duplicate_token_cosine_above_0999_fraction=float((cosine[nonself]>.999).float().mean()),
        adjacent_difference_rms=float(delta.square().mean().sqrt()),
        shuffled_adjacent_difference_rms=float(shuffled_delta.square().mean().sqrt()),
        reverse_vs_original_adjacent_difference_cosine=float(F.cosine_similarity(delta,reverse_delta,dim=-1).mean()),
        adjacency_boundary_policy='exclude left/right channel join' if channel_major else 'text index adjacency only; not music time')


def acquire_gpu(args,out):
    resource=snapshot()
    gpu=next(x for x in resource['gpus'] if x[0]=='0')
    assert gpu[1]==UUID and resource['memavailable_bytes']>=64*1024**3 and resource['cgroup_headroom_bytes']>=64*1024**3
    reserve=Path('/tmp/od_reserve_0')
    assert not reserve.exists() or (reserve.is_file() and not reserve.is_symlink() and reserve.stat().st_uid==os.getuid() and reserve.stat().st_size==0)
    reserve.touch()
    active=[x for x in resource['apps'] if x[1]==UUID]
    if active:
        assert len(active)==1
        pid=int(active[0][0]);holder=identity(pid)
        assert pid==args.holder_pid and holder['start_ticks']==args.holder_start_ticks
        assert holder['uids']==[0,0,0,0] and len(holder['argv'])==2
        assert holder['argv'][0].endswith('/envs/longlive/bin/python') and holder['argv'][1]==HOLDER
        assert holder['cuda']=='0' and holder['cgroup']==resource['cgroup'] and int(active[0][2])<=1200
        fd=os.pidfd_open(pid)
        try:
            assert identity(pid)==holder
            save(out/'resource_release.json',dict(snapshot=resource,holder=holder,signal='pidfd_SIGTERM_exact_project_holder_only'))
            signal.pidfd_send_signal(fd,signal.SIGTERM)
            assert select.select([fd],[],[],30)[0], 'Holder failed to exit; no escalation'
        finally:
            os.close(fd)
        time.sleep(3)
    fresh=snapshot()
    assert not any(x[1]==UUID for x in fresh['apps'])
    save(out/'fresh_resource.json',fresh)
    return reserve


def run(args):
    lock=Path('/tmp/music_direct_fixed3_condition_diag.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tested=json.loads(args.cpu_report.read_text())
    assert tested['complete'] and tested['protocol']==PROTOCOL and tested['script_sha256']==sha(__file__)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    args.out.mkdir(exist_ok=False)
    report=dict(protocol=PROTOCOL,complete=False,model_order=['mean','token'],actual_optimizer_updates=0,
        script_sha256=sha(__file__),cpu_report=str(args.cpu_report),cases=[],models=[],
        scientific_bounds=dict(mean_cosine=[-1,1],token_softchamfer_cosine=[-1,1],rms_minimum=0,
            duplicate_fraction=[0,1],softchamfer64='64 uniformly selected valid token indices; no temporal supervision',
            full_softchamfer='permutation invariant all actual tokens; does not establish choreography',
            adjacency='ordered hidden-index diagnostic only; text token index is NOT dance/musical time',
            video_quality=False,N4_native_AV_joint_fixed3_video_comparison_pending=True),
        teacher_only_for_post_forward_metrics=True,student_reads_cached_P_or_teacher_or_caption=False,
        student_length_from_actual_music_features_not_teacher=True,feature_cache=False,
        raw_deploy='uncropped unnormalized FP32 channel-major [254,5120]; no generator invoked')
    reserve=acquire_gpu(args,args.out)
    rows,waves,stats,provenance,isolation=inputs()
    report.update(input_provenance=provenance,isolation=isolation,manifest_sha256=MANIFEST_SHA)
    device=torch.device('cuda',0)
    torch.cuda.set_device(device)
    for arm,path,checksum in SCOPE:
        assert sha(path)==checksum
        print(json.dumps(dict(event='loading_complete_actual_model',arm=arm,path=str(path))),flush=True)
        model=load_direct_music_text_checkpoint(path).to(device).eval()
        metadata=model.checkpoint_metadata
        assert len(model.state_dict())==467 and metadata['format_version']==4
        outputs={};events=[]
        # Each unchanged/rotated assignment runs real loader + backbone forward; no exact-wave memoization.
        for i,sid in enumerate(IDS):
            for condition,source in [('original',sid),('cyclic_swap',IDS[(i+1)%3])]:
                row=rows[IDS.index(source)];audio=Path(row['audio_path'])
                wave,_=load_audio(audio.stem,audio.parent)
                assert torch.equal(wave,waves[source])
                observed={}
                def prehook(module,arguments):
                    observed['actual_backbone_input_24k']=tensor_info(arguments[0])
                def featurehook(module,arguments,output):
                    observed['actual_features']=tensor_info(output)
                h1=model.backbone.register_forward_pre_hook(prehook)
                h2=model.backbone.register_forward_hook(featurehook)
                try:
                    with torch.inference_mode():
                        raw,mask=model(wave[None].to(device))
                    torch.cuda.synchronize()
                finally:
                    h1.remove();h2.remove()
                assert raw.shape==(1,254,5120) and mask.all() and raw.dtype==torch.float32
                output=raw[0].cpu();outputs[(sid,condition)]=output
                event=dict(arm=arm,case_id=sid,condition=condition,source_audio_id=source,
                    audio_sha256=row['audio_sha256'],input_wave=tensor_info(wave),
                    raw_tokens=tensor_info(output),**observed)
                events.append(event)
                save(args.out/(arm+'_'+sid+'_'+condition+'_forward.json'),event)
                print(json.dumps(dict(event='actual_condition_forward',**event)),flush=True)
                del raw,mask,wave
                reserve.touch()
        # The teacher is first materialized AFTER this model's real waveform-only outputs.
        teachers=[];teacher_records=[]
        for row in rows:
            assert sha(row['target_path'])==row['target_sha256']
            record=torch.load(row['target_path'],map_location='cpu',weights_only=True)
            tail=record['tail']
            assert tail.dtype==torch.bfloat16 and tail.ndim==2 and tail.shape[1]==5120
            assert len(tail)==record['teacher_length'] and torch.isfinite(tail).all()
            teachers.append(tail.float())
            teacher_records.append(dict(sample_id=row['sample_id'],path=row['target_path'],
                sha256=row['target_sha256'],raw_teacher=tensor_info(tail)))
        mean,std=stats['mean'],stats['std']
        originals=[outputs[(sid,'original')] for sid in IDS]
        distinct_outputs=len({tensor_info(x)['sha256'] for x in originals})
        cross_mean=[];cross_full=[];cross64=[]
        for x in originals:
            cross_mean.append([float(F.cosine_similarity((x.mean(0)-mean)/std,(y.mean(0)-mean)/std,dim=0)) for y in teachers])
            cross_full.append([float(full_set_score(x,y,mean,std)) for y in teachers])
            cross64.append([sampled_score(x,y,mean,std) for y in teachers])
        case_results=[]
        for i,sid in enumerate(IDS):
            x=originals[i];other=outputs[(sid,'cyclic_swap')];teacher=teachers[i]
            assert torch.equal(other,originals[(i+1)%3]), 'Deterministic swap must match its actual source music'
            flat_delta=x-other
            xd=(x.reshape(2,127,5120)[:,1:]-x.reshape(2,127,5120)[:,:-1]).flatten(0,1)
            yd=(other.reshape(2,127,5120)[:,1:]-other.reshape(2,127,5120)[:,:-1]).flatten(0,1)
            shuffle=x[torch.randperm(len(x))]
            torch.testing.assert_close(full_set_score(shuffle,teacher,mean,std),full_set_score(x,teacher,mean,std),atol=2e-6,rtol=2e-6)
            collapsed=x.mean(0).expand_as(x)
            global_mean=mean.expand_as(x)
            result=dict(sample_id=sid,cyclic_swap_source=IDS[(i+1)%3],
                actual_original_vs_swap_raw_rms=float(flat_delta.square().mean().sqrt()),
                difference_over_original_rms=float(flat_delta.square().mean().sqrt()/x.square().mean().sqrt()),
                original_vs_swap_mean_raw_cosine=float(F.cosine_similarity(x.mean(0),other.mean(0),dim=0)),
                original_vs_swap_whitened_mean_cosine=float(F.cosine_similarity((x.mean(0)-mean)/std,(other.mean(0)-mean)/std,dim=0)),
                original_vs_swap_ordered_adjacent_difference_cosine=float(F.cosine_similarity(xd,yd,dim=-1).mean()),
                original_vs_swap_ordered_adjacent_difference_rms=float((xd-yd).square().mean().sqrt()),
                raw_token_summary=token_summary(x,True),teacher_token_summary=token_summary(teacher,False),
                original_matching_teacher=dict(mean_cosine=cross_mean[i][i],full_token_set=cross_full[i][i],sampled64=cross64[i][i]),
                swap_matching_same_teacher=dict(mean_cosine=cross_mean[(i+1)%3][i],full_token_set=cross_full[(i+1)%3][i],sampled64=cross64[(i+1)%3][i]),
                own_mean_repeated_baseline_full_token_set=float(full_set_score(collapsed,teacher,mean,std)),
                train_global_mean_repeated_baseline_full_token_set=float(full_set_score(global_mean,teacher,mean,std)),
                full_set_shuffle_invariant_verified=True,swap_equals_forward_of_actual_source=True)
            case_results.append(result)
        arm_report=dict(arm=arm,checkpoint=str(path),checkpoint_sha256=checksum,format_version=4,
            full_state_tensors=467,all_learned_backbone_and_linear_loaded_strict=True,
            dense_trainable_metadata_tensors=len(metadata['online_training']['trainable_backbone_parameters']),
            complete_model_parameters=sum(p.numel() for p in model.parameters()),
            distinct_original_music_output_hashes=distinct_outputs,
            identical_across_music=distinct_outputs==1,
            forward_events=events,teacher_records=teacher_records,cases=case_results,
            three_by_three_whitened_mean_teacher_cosine=cross_mean,
            three_by_three_full_token_set_teacher_match=cross_full,
            three_by_three_sampled64_teacher_match=cross64,
            mean_of_actual_original_swap_rms=sum(c['actual_original_vs_swap_raw_rms'] for c in case_results)/3,
            cuda_peak_allocated=torch.cuda.max_memory_allocated(),optimizer_updates=0)
        save(args.out/(arm+'_metrics.json'),arm_report)
        report['models'].append(arm_report)
        print(json.dumps(dict(event='arm_condition_diagnosis_complete',arm=arm,report=str(args.out/(arm+'_metrics.json')))),flush=True)
        del model,outputs,originals,teachers
        gc.collect();torch.cuda.empty_cache()
        reserve.touch()
    report['complete']=True
    report['cases']=IDS
    save(args.out/'report.json',report)
    print(json.dumps(dict(event='all_complete',report=str(args.out/'report.json'),video_quality_accepted=False)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu_check',action='store_true')
    parser.add_argument('--cpu_report',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--holder_pid',type=int)
    parser.add_argument('--holder_start_ticks')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if args.cpu_check:
        cpu_check(args)
    else:
        try:
            run(args)
        except BaseException as error:
            if args.out and args.out.is_dir():
                save(args.out/'failure.json',dict(protocol=PROTOCOL,complete=False,error=repr(error),optimizer_updates=0))
            raise


if __name__=='__main__':
    main()
