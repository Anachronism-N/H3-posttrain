"""Calibrate joint audio-video model opinions; never label them human reviews."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from music_generation_validate import atomic_write_new, sha256

SYSTEM = 'You are a careful audiovisual observer. Judge only evidence visible and audible in the supplied clip. Do not assume that dance must hit every musical beat.'
PROMPT = '''Observe this single dancer and listen to the supplied music together.
The input is a four-second crop; local time 0.0 corresponds to source-video time 1.0.
Describe visible dance events and audible accents, and whether their timing relates.
Ignore identity, clothing, background and production quality. Do not infer musical
synchrony from motion amount, tempo alone, or the fact that both modalities exist.
The video is sampled at 8 fps, so do not assert timing finer than 0.125 seconds.
No beat labels or information about original versus swapped music are provided.
Return only one JSON object with these fields:
{"music_description": "short audible description", "motion_description": "short visible description",
"correspondences": [{"local_time_s": 0.0, "motion_event": "observed event", "audible_event": "observed event or none", "confidence": "low|medium|high"}],
"synchrony_score": 1, "confidence": "low|medium|high", "limitations": "what is uncertain"}.
Score is an integer 1 to 5: 1 clear repeated mismatch, 3 unclear/mixed evidence,
5 repeated clearly observed alignment. Use low confidence when evidence is weak.
At most three correspondences. Do not invent exact beats or unseen movement.'''


def validate_opinion(raw):
    text = raw.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        end = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == '```'), None)
        if end is None:
            return None, 'unterminated JSON fence'
        text = '\n'.join(lines[1:end])
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        return None, str(error)
    required = {'music_description', 'motion_description', 'correspondences',
                'synchrony_score', 'confidence', 'limitations'}
    if not isinstance(value, dict) or set(value) != required:
        return None, 'unexpected JSON fields'
    if type(value['synchrony_score']) is not int or not 1 <= value['synchrony_score'] <= 5:
        return None, 'invalid ordinal score'
    if value['confidence'] not in ('low', 'medium', 'high'):
        return None, 'invalid confidence'
    if any(not isinstance(value[k], str) for k in ('music_description', 'motion_description', 'limitations')):
        return None, 'description fields must be strings'
    events = value['correspondences']
    if not isinstance(events, list) or len(events) > 3:
        return None, 'invalid event list'
    for event in events:
        if not isinstance(event, dict) or set(event) != {'local_time_s', 'motion_event', 'audible_event', 'confidence'}:
            return None, 'invalid event fields'
        stamp = event['local_time_s']
        if type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 <= stamp < 4:
            return None, 'event timestamp outside crop'
        if event['confidence'] not in ('low', 'medium', 'high'):
            return None, 'invalid event confidence'
        if not isinstance(event['motion_event'], str) or not isinstance(event['audible_event'], str):
            return None, 'event descriptions must be strings'
    return value, None


def prepare_input(case, audio_path, fps=8):
    import decord
    import soundfile as sf
    from scipy.signal import resample_poly

    reader = decord.VideoReader(case['video_path'], ctx=decord.cpu(0), num_threads=1, fault_tol=0)
    native_fps = float(reader.get_avg_fps())
    if not math.isfinite(native_fps) or native_fps <= 0 or len(reader) / native_fps < 5:
        raise ValueError('Video does not contain the full [1,5) crop')
    stamps = 1 + np.arange(4 * fps) / fps
    indices = np.rint(stamps * native_fps).astype(int)
    video = reader.get_batch(indices).asnumpy()
    actual_stamps = indices / native_fps
    if np.max(np.abs(actual_stamps - stamps)) > .5 / native_fps + 1e-6:
        raise ValueError('Unexpected sampled video timestamps')
    with sf.SoundFile(audio_path) as source:
        sample_rate = source.samplerate
        if source.frames < 5 * sample_rate:
            raise ValueError('Audio does not contain full [1,5) crop')
        source.seek(sample_rate)
        waveform = source.read(4 * sample_rate, dtype='float32', always_2d=True).mean(1)
    divisor = math.gcd(sample_rate, 16000)
    waveform = resample_poly(waveform, 16000 // divisor, sample_rate // divisor).astype('float32')
    if waveform.shape != (64000,) or not np.isfinite(waveform).all():
        raise ValueError('Expected finite four-second mono 16kHz audio')
    metadata = dict(video_sha256=sha256(Path(case['video_path'])),
                    audio_sha256=sha256(Path(audio_path)),
                    video_frame_indices=indices.tolist(), video_source_times=actual_stamps.tolist(),
                    native_fps=native_fps, source_window_seconds=[1., 5.],
                    sampled_video_fps=fps, frames=len(video), input_audio_sr=16000,
                    input_audio_samples=len(waveform),
                    waveform_crop_sha256=hashlib.sha256(waveform.tobytes()).hexdigest())
    return video, waveform, metadata


def main():
    import torch
    import transformers
    from music_counterfactual_eval import cases
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp_root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New directory, one atomic opinion file per item')
    parser.add_argument('--tags', nargs='+', default=[])
    parser.add_argument('--calibration', action='store_true', help='Only three original GT videos, correct and alternative music')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshard', type=int, default=1)
    args = parser.parse_args()
    if not args.out.parent.is_dir() or args.out.exists() or not args.model.is_dir():
        parser.error('Require existing input/output parents and a new opinion directory')
    if args.nshard < 1 or not 0 <= args.shard < args.nshard:
        parser.error('Invalid shard')
    selected = cases(args.exp_root, tags=args.tags)
    if args.calibration:
        selected = [c for c in selected if c['tag'] == 'GT']
    jobs = [(c, label, c[key]) for c in selected for label, key in
            (('correct', 'correct_audio'), ('alternative', 'alt_audio'))][args.shard::args.nshard]
    if not jobs:
        parser.error('No selected audiovisual pairs')
    args.out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map='cuda',
        attn_implementation='sdpa', local_files_only=True).eval().requires_grad_(False)
    model_config_hash = sha256(args.model / 'config.json')
    completed = []
    for number, (case, label, audio_path) in enumerate(jobs):
        start = time.perf_counter()
        video, waveform, metadata = prepare_input(case, audio_path)
        conversation = [dict(role='system', content=[dict(type='text', text=SYSTEM)]),
                        dict(role='user', content=[dict(type='video', video=case['video_path']),
                                                   dict(type='text', text=PROMPT)])]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, videos=[video], audio=[waveform],
                           return_tensors='pt', padding=True,
                           videos_kwargs=dict(fps=8., do_sample_frames=False,
                                              size=dict(shortest_edge=65536, longest_edge=100352),
                                              cap_pixels_per_frame=True,
                                              use_audio_in_video=True),
                           audio_kwargs=dict(sampling_rate=16000))
        required = ('pixel_values_videos', 'video_grid_thw', 'input_features',
                    'feature_attention_mask', 'video_second_per_grid')
        if any(key not in inputs for key in required):
            raise ValueError('Joint audio/video features missing from processor output')
        audio_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
        video_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
        token_counts = {name: int((inputs['input_ids'] == idx).sum()) for name, idx in
                        (('audio', audio_id), ('video', video_id))}
        if min(token_counts.values()) <= 0 or not torch.allclose(
                inputs['video_second_per_grid'], torch.tensor([.25])):
            raise ValueError('Missing audio/video tokens or incorrect temporal grid')
        metadata.update(joint_audio_video_tokens=token_counts,
                        video_grid_thw=inputs['video_grid_thw'].tolist(),
                        video_seconds_per_grid=inputs['video_second_per_grid'].tolist(),
                        context_tokens=int(inputs['input_ids'].shape[-1]))
        if metadata['context_tokens'] + 512 > 32768:
            raise ValueError('Joint input exceeds context limit')
        inputs = inputs.to(model.device)
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                inputs[key] = value.to(model.dtype)
        with torch.inference_mode():
            output = model.generate(**inputs, use_audio_in_video=True,
                                    max_new_tokens=512, do_sample=False)
        if not isinstance(output, torch.Tensor):
            raise ValueError('Expected text-only Thinker output')
        generated = output[:, inputs['input_ids'].shape[-1]:]
        raw = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        opinion, parse_error = validate_opinion(raw)
        item = dict(case=case, diagnostic_audio_label=label, audio_path=audio_path,
                    input_metadata=metadata, model_path=str(args.model), model_config_sha256=model_config_hash,
                    transformers_version=transformers.__version__, system=SYSTEM, prompt=PROMPT,
                    raw_output=raw, opinion=opinion, parse_error=parse_error,
                    wall_seconds=time.perf_counter() - start,
                    scope='Automatic joint audiovisual model opinion, not human review or calibrated beat ground truth.')
        filename = f'item_{number:03d}.json'
        atomic_write_new(args.out / filename, item)
        completed.append(dict(file=filename, case_key=case['key'], audio_label=label,
                              synchrony_score=opinion['synchrony_score'] if opinion else None,
                              confidence=opinion['confidence'] if opinion else None, parse_error=parse_error))
        print(json.dumps(completed[-1]), flush=True)
    summary = dict(complete=len(completed) == len(jobs), expected=len(jobs), completed=completed,
                   calibration_only=args.calibration, shard=args.shard, nshard=args.nshard,
                   human_review=False, protocol='Same [1,5) crop for video and audio; 8fps input, 16kHz mono; joint use_audio_in_video=True; no score/pair labels supplied to model.')
    atomic_write_new(args.out / 'summary.json', summary)
    print(json.dumps(dict(complete=summary['complete'], expected=summary['expected'])), flush=True)


if __name__ == '__main__':
    main()
