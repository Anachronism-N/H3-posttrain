"""One fixed native-music case: reuse P+y, compare multimodal y and pure-text y without P."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from music_generation_validate import atomic_write_new, sha256
from music_scale_generate import SSH, guard_code

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
REPO = ROOT / 'H3-posttrain'
MS = ROOT / 'diff/logs_omnidance/exp/MS'
EXP = MS.parent
UPSTREAM = ROOT / 'DiffSynth-Studio-upstream-20260930'
SID = 'wJPptuhclZmKAEWs'
PROTOCOL = 'h3_teacher_prefix_single_case_v1'


def prepare(out):
    import torch
    from music_text_oracle_generate import validate_direct_generation_manifest
    from music_text_condition import tensor_sha256
    manifest, digest = validate_direct_generation_manifest(MS / 'music_direct_eval3_generation_joint_v1.json')
    case = next(c for c in manifest['cases'] if c['sample_id'] == SID)
    multimodal = torch.load(case['target_path'], map_location='cpu', weights_only=True)
    pure_path = MS / 'music_text_pure_text_cases_v1' / (SID + '.pt')
    pure = torch.load(pure_path, map_location='cpu', weights_only=True)
    if (multimodal['metadata']['prompt'] != pure['metadata']['prompt']
            or not torch.equal(multimodal['source_token_ids'], pure['source_token_ids'])
            or multimodal['tail'].shape != pure['tail'].shape
            or torch.equal(multimodal['tail'], pure['tail'])
            or pure['metadata'].get('image_conditioned') is not False):
        raise ValueError('Require the same prompt/source IDs and separately encoded pure-text teacher')
    report_path = MS / 'music_direct_cached_teacher_identity_all3_audit_v1.json'
    baseline = json.loads(report_path.read_text())
    row = next(r for r in baseline['rows'] if r['sample_id'] == SID)
    if (baseline.get('complete') is not True or baseline.get('passed') != 3
            or baseline.get('complete_selected_audio_video_decode') is not True
            or row['integrity'].get('full_video_audio_decode') is not True
            or row['case'] != case or baseline['plan']['music_mode'] != 'ref'
            or baseline['plan']['output_audio'] != 'generated'):
        raise ValueError('Require the completed original native real-music P+y baseline')
    if sha256(Path(row['video_path'])) != row['video_sha256']:
        raise ValueError('Baseline video changed')
    for path, checksum in baseline['plan']['source_files'].items():
        if sha256(Path(path)) != checksum:
            raise ValueError('Baseline generation code changed: ' + path)
    specs = []
    for label, target_path, record, host in (
            ('B', case['target_path'], multimodal, '28.216.19.69'),
            ('C', str(pure_path), pure, '28.216.17.70')):
        directory = EXP / 'H3AR' / ('val_teacher_prefix_' + label.lower() + '_case1_v1')
        if directory.exists():
            raise FileExistsError(directory)
        specs.append(dict(label=label, host=host, gpu=6, target_path=str(target_path),
                          target_sha256=sha256(Path(target_path)), raw_sha256=tensor_sha256(record['tail']),
                          raw_shape=list(record['tail'].shape), out=str(directory),
                          uses_P=False, teacher_image_conditioned=label == 'B'))
    if out.exists() or not out.parent.is_dir():
        raise FileExistsError('Require a new queue and existing parent')
    out.mkdir()
    plan = dict(protocol=PROTOCOL, single_case=SID, case=case, manifest_sha256=digest,
                prompt=multimodal['metadata']['prompt'], prompt_sha256=multimodal['metadata']['caption_sha256'],
                baseline_report=str(report_path), baseline_report_sha256=sha256(report_path),
                baseline=row, specifications=specs,
                source_files={str(Path(__file__).resolve()): sha256(Path(__file__)),
                              str(UPSTREAM / 'diffsynth/pipelines/minimax_h3_audio_video.py'):
                                  sha256(UPSTREAM / 'diffsynth/pipelines/minimax_h3_audio_video.py')},
                comparison='A cached P+multimodal y reused; B same multimodal y without P; C same-caption pure y without P',
                no_music_student_or_new_training=True, no_new_teacher_encoding=True,
                P_removal_changes_packed_positions=True, human_quality_accepted=False)
    atomic_write_new(out / 'plan.json', plan)
    return plan


def worker(queue, label):
    import torch
    import librosa
    from PIL import Image
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig, MiniMaxH3Unit_PromptEmbedder
    from diffsynth.utils.data.audio_video import write_video_audio
    from music_text_condition import tensor_sha256
    plan = json.loads((queue / 'plan.json').read_text())
    spec = next(s for s in plan['specifications'] if s['label'] == label)
    for path, checksum in plan['source_files'].items():
        if sha256(Path(path)) != checksum:
            raise ValueError('Ablation code changed')
    if sha256(Path(spec['target_path'])) != spec['target_sha256']:
        raise ValueError('Teacher cache changed')
    record = torch.load(spec['target_path'], map_location='cpu', weights_only=True)
    raw = record['tail']
    if tensor_sha256(raw) != spec['raw_sha256'] or raw.dtype != torch.bfloat16:
        raise ValueError('Teacher tensor changed')
    case = plan['case']
    for kind in ('image', 'audio'):
        if sha256(Path(case[kind + '_path'])) != case[kind + '_sha256']:
            raise ValueError('Reference source changed')
    directory = Path(spec['out'])
    video_path = directory / 'joint' / (SID + '_' + Path(case['audio_path']).stem + '.mp4')
    if video_path.exists() or video_path.with_suffix('.audit.json').exists():
        raise FileExistsError('Never overwrite an ablation partial result')
    vram = dict(offload_dtype=torch.bfloat16, offload_device='cpu', onload_dtype=torch.bfloat16,
                onload_device='cpu', preparing_dtype=torch.bfloat16, preparing_device='cuda',
                computation_dtype=torch.bfloat16, computation_device='cuda')
    weights = Path('/dev/shm/h3/Ref2VA')
    configs = [ModelConfig(path=sorted(str(p) for p in (weights / 'transformer').glob('model*.safetensors')), **vram),
               ModelConfig(path=str(weights / 'video_vae/source/model.safetensors'), **vram),
               ModelConfig(path=str(weights / 'audio_vae/model.safetensors'), **vram)]
    pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda', model_configs=configs,
        processor_config=None, vram_limit=torch.cuda.mem_get_info('cuda')[1] / 1024 ** 3 - 5)
    if any(getattr(pipe, key) is not None for key in ('text_encoder', 'tokenizer', 'processor')):
        raise ValueError('Ablation must not load live text/vision language encoder')
    for unit in pipe.units:
        if isinstance(unit, MiniMaxH3Unit_PromptEmbedder):
            unit.onload_model_names = ()
    with Image.open(case['image_path']) as source:
        image = source.convert('RGB')
    values, rate = librosa.load(case['audio_path'], sr=pipe.audio_vae.sample_rate, mono=False, duration=124 / 24)
    waveform = torch.from_numpy(values).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).repeat(2, 1)
    baseline = plan['baseline']['audit']
    if tensor_sha256(waveform) != baseline['waveform_sha256']:
        raise ValueError('Native waveform differs from A')
    sampling = baseline['sampling']
    captured, updates = {}, dict(video=0, audio=0)
    original_runner, original_step = pipe.unit_runner, pipe.step

    def runner(unit, current_pipe, shared, positive, negative):
        shared, positive, negative = original_runner(unit, current_pipe, shared, positive, negative)
        if unit is pipe.units[-1]:
            actual, tags, packed = positive['prompt_embeds'], positive['text_token_tags'], positive['packed']
            if (not torch.equal(actual, raw.to(actual)) or not torch.all(tags == 1)
                    or len(packed['text_pos']) != len(raw)
                    or shared.get('input_latents_audio') is not None or shared.get('denoise_mask_audio') is not None):
                raise ValueError('No-P raw teacher route or reference-only audio changed')
            for key, base_key in (('ref_visual_anchor', 'visual_reference'), ('ref_audio_anchor', 'reference')):
                tensor = shared[key]
                expected = baseline[base_key]
                if list(tensor.shape) != expected['shape'] or tensor_sha256(tensor) != expected['sha256']:
                    raise ValueError('Independent VAE reference differs from A: ' + key)
            captured.update(raw_exact=True, raw_sha256=tensor_sha256(actual), raw_shape=list(actual.shape),
                            text_pos_count=len(packed['text_pos']), no_P=True, tags_all_text=True,
                            visual_reference=baseline['visual_reference'], audio_reference=baseline['reference'],
                            native_waveform_sha256=tensor_sha256(waveform), native_real_music=True,
                            text_encoder_loaded=False, text_encoder_calls=0, original_joint_pipeline=True,
                            sampling=sampling, position_layout_changed_from_A=True)
        return shared, positive, negative

    def step(scheduler, *args, **kwargs):
        key = 'video' if scheduler is pipe.scheduler else 'audio' if scheduler is pipe.scheduler_audio else None
        if key is None:
            raise ValueError('Unexpected scheduler')
        updates[key] += 1
        return original_step(scheduler, *args, **kwargs)

    pipe.unit_runner, pipe.step = runner, step
    try:
        video, audio = pipe(prompt=None, negative_prompt=None, text_embedding=raw.to(pipe.device), **sampling,
            references=[dict(type='image', image=image), dict(type='audio', audio=waveform, sample_rate=int(rate))])
    finally:
        pipe.unit_runner, pipe.step = original_runner, original_step
    if not captured or updates != dict(video=50, audio=50):
        raise ValueError('Incomplete original joint prediction/scheduler execution')
    write_video_audio(video, audio, str(video_path), fps=24, audio_sample_rate=pipe.audio_vae.sample_rate)
    result = dict(protocol=PROTOCOL, complete=True, label=label, sample_id=SID,
                  teacher_image_conditioned=spec['teacher_image_conditioned'], input_target_path=spec['target_path'],
                  input_target_sha256=spec['target_sha256'], audit=captured, scheduler_updates=updates,
                  video_path=str(video_path), video_sha256=sha256(video_path),
                  generated_audio_sha256=tensor_sha256(audio), generated_audio_preserved=True,
                  human_quality_accepted=False, plan_sha256=sha256(queue / 'plan.json'))
    atomic_write_new(video_path.with_suffix('.audit.json'), result)
    print('[teacher prefix saved] ' + json.dumps(result), flush=True)
    print('[h3ar] done', flush=True)


def run(queue):
    plan = prepare(queue)
    holder = ROOT / 'diff/scripts_omnidance/exp/hold_gpu.py'
    ssh = SSH + ['-p', '36000']

    def generate(spec):
        guard = guard_code(str(Path(__file__).resolve()), str(holder), 6).replace(
            'if holder in args:', 'if holder in args and p.stat().st_uid == os.getuid():').replace(
            'assert len(samplers)<=4', 'assert len(samplers)==0')
        admission = subprocess.run(ssh + [spec['host'], 'python3 -c ' + shlex.quote(guard)], capture_output=True, text=True)
        atomic_write_new(queue / ('admission_' + spec['label'] + '.json'),
                         dict(host=spec['host'], gpu=6, exit_code=admission.returncode,
                              stdout=admission.stdout, stderr=admission.stderr))
        if admission.returncode:
            return dict(label=spec['label'], exit_code=admission.returncode, complete=False)
        directory = Path(spec['out'])
        directory.mkdir()
        (directory / 'joint').mkdir()
        command = ['env', 'CUDA_VISIBLE_DEVICES=6', 'OMP_NUM_THREADS=2', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
                   'PYTHONPATH=' + str(UPSTREAM), '/opt/venv_h3/bin/python', str(Path(__file__).resolve()),
                   '--worker', spec['label'], '--queue', str(queue)]
        log = queue / ('generation_' + spec['label'] + '.log')
        with log.open('x') as stream:
            completed = subprocess.run(ssh + [spec['host'], shlex.join(command)], stdout=stream, stderr=subprocess.STDOUT)
        return dict(label=spec['label'], exit_code=completed.returncode, complete=completed.returncode == 0,
                    log=str(log), log_sha256=sha256(log))

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = list(pool.map(generate, plan['specifications']))
    atomic_write_new(queue / 'jobs.json', dict(jobs=jobs, complete=all(j['complete'] for j in jobs)))
    if not all(j['complete'] for j in jobs):
        raise RuntimeError('Ablation generation failed; preserve all partial outputs')
    sys.path.append('/opt/venv_eval/lib/python3.11/site-packages')
    from music_generation_validate import probe_video
    rows = [dict(label='A', reused=True, video_path=plan['baseline']['video_path'],
                 video_sha256=plan['baseline']['video_sha256'], integrity=plan['baseline']['integrity'])]
    for spec in plan['specifications']:
        video = next((Path(spec['out']) / 'joint').glob('*.mp4'))
        record = json.loads(video.with_suffix('.audit.json').read_text())
        if record['video_sha256'] != sha256(video) or not record['audit']['raw_exact']:
            raise ValueError('Ablation output changed')
        rows.append(dict(label=spec['label'], reused=False, video_path=str(video), video_sha256=sha256(video),
                         integrity=probe_video(video), audit=record['audit']))
    atomic_write_new(queue / 'finished.json', dict(protocol=PROTOCOL, complete=True, rows=rows,
                    generated_new_videos=2, reused_A_videos=1, fixed_people=1, human_review_required=True,
                    human_quality_accepted=False, comparison=plan['comparison']))
    print(json.dumps(dict(event='teacher_prefix_single_case_complete', complete=True, queue=str(queue))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--worker', choices=['B', 'C'])
    args = parser.parse_args()
    if args.worker:
        worker(args.queue, args.worker)
    else:
        run(args.queue)
