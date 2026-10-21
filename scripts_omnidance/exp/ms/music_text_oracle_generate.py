"""Managed three-person joint/aswap raw-text oracle, with fixed native silence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from music_scale_generate import SSH, guard_code


def teacher_tail_cache_source(cache_dir):
    from music_text_condition import PURE_TEXT_TEACHER_KIND, PURE_TEXT_BINDING_KIND, file_sha256
    cache = Path(cache_dir)
    candidates = [cache / 'finished.json', cache.parent / 'finished.json']
    for path in candidates:
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text())
        source = manifest.get('source', {})
        kind = manifest.get('kind') or source.get('kind')
        if kind not in (PURE_TEXT_TEACHER_KIND, PURE_TEXT_BINDING_KIND):
            continue
        if (manifest.get('complete') is not True or kind != PURE_TEXT_BINDING_KIND
                or source.get('kind') != PURE_TEXT_TEACHER_KIND
                or manifest.get('text_only') is not True or manifest.get('image_conditioned') is not False
                or type(manifest.get('prefix_length')) is not int or manifest['prefix_length'] != 0
                or manifest.get('expected') != 3 or len(manifest.get('rows', [])) != 3):
            raise ValueError('Pure-text teacher requires a complete explicit three-case binding manifest')
        rows = manifest['rows']
        stems = []
        for row in rows:
            case = row['case']
            stem = str(case['sample_id']) + '_' + str(case['audio_id'])
            record_path = cache / (stem + '.pt')
            if (Path(stem).name != stem or case.get('condition') != 'joint'
                    or Path(row['path']).resolve() != record_path.resolve()
                    or file_sha256(record_path) != row['file_sha256']):
                raise ValueError('Pure-text teacher binding manifest has changed case paths or file SHA')
            stems.append(stem)
        if len(set(stems)) != 3:
            raise ValueError('Pure-text teacher binding manifest must declare three distinct cases')
        return dict(kind=PURE_TEXT_TEACHER_KIND, source_scope='caption_only_no_reference_context',
                    manifest_path=str(path), manifest_sha256=file_sha256(path),
                    text_only=True, image_conditioned=False, prefix_length=0)
    return dict(kind='cached_image_conditioned_teacher',
                source_scope='cached_teacher_tail_original_reference_context',
                manifest_path=None, manifest_sha256=None)


DIRECT_PROTOCOL = 'musicfm_direct_tokens_cached_prefix_v1'


def direct_native_parser():
    parser = argparse.ArgumentParser(description='Direct MusicFM raw suffix plus offline native P/Ptags; original joint H3 pipeline, real reference audio, no TC')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cached_prefix_manifest', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--direct_music_ckpt', type=Path)
    source.add_argument('--direct_teacher_identity', action='store_true', help='Offline cached full teacher identity diagnostic; not music prediction')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshard', type=int, default=1)
    parser.add_argument('--rows', type=int, default=3, choices=[3])
    parser.add_argument('--conds', default='joint', choices=['joint', 'aswap'])
    parser.add_argument('--mode', default='bi', choices=['bi'])
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--shift', type=float, default=12.0)
    parser.add_argument('--audio_shift', type=float, default=3.0)
    parser.add_argument('--cfg', type=float, default=1.0, choices=[1.0])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--num_frames', type=int, default=124)
    parser.add_argument('--music_mode', default='ref', choices=['ref', 'retake'])
    parser.add_argument('--output_audio', default='generated', choices=['generated', 'original_mux'])
    parser.add_argument('--direct_zero_input', action='store_true')
    parser.add_argument('--no_text_encoder', action='store_true', help='Direct mode always omits TC/tokenizer/processor')
    parser.add_argument('--w', default='/dev/shm/h3/Ref2VA')
    return parser


def direct_sampling_settings(args, height, width):
    return dict(height=height, width=width, num_frames=args.num_frames,
                num_inference_steps=args.steps, seed=args.seed, rand_device='cpu',
                cfg_scale=args.cfg, flow_shift=args.shift, audio_flow_shift=args.audio_shift,
                tiled=True, tile_size=256, tile_overlap=64,
                use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
                ref_image_short_edge=2048, ref_video_short_edge=768,
                ref_video_max_pixels=768 * 1344)


def direct_native_call(pipe, embedder, image, native_waveform, sample_rate, sampling,
                       music_mode='ref', expected_prefix=None, identity_output=None):
    import torch
    from music_text_condition import tensor_sha256, difference
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PromptEmbedder, MiniMaxH3Unit_PackedSequenceBuilder
    if any(getattr(pipe, name) is not None for name in ('text_encoder', 'tokenizer', 'processor')):
        raise ValueError('Direct native generation must not load text encoder/tokenizer/processor')
    if embedder.onload_model_names:
        raise ValueError('Direct embedder must not request TC onload')
    matches = [index for index, unit in enumerate(pipe.units) if isinstance(unit, MiniMaxH3Unit_PromptEmbedder)]
    if len(matches) != 1:
        raise ValueError('Require exactly one original prompt embedder')
    original_unit = pipe.units[matches[0]]
    original_runner, original_step, original_model_fn = pipe.unit_runner, pipe.step, pipe.model_fn
    pipe.units[matches[0]] = embedder
    audit = {}
    updates = dict(video=0, audio=0)
    first_noise = {}

    def optional_tensor(value):
        return dict(exists=value is not None, shape=list(value.shape) if value is not None else None,
                    sha256=tensor_sha256(value) if value is not None else None)

    def runner(unit, current_pipe, shared, positive, negative):
        shared, positive, negative = original_runner(unit, current_pipe, shared, positive, negative)
        if unit is pipe.units[-1]:
            raw, tags, packed = positive['prompt_embeds'], positive['text_token_tags'], positive['packed']
            if raw.ndim != 2 or raw.shape[1] != 5120 or raw.shape[0] != len(packed['text_pos']):
                raise ValueError('Direct condition rows differ from native packed text layout')
            if expected_prefix is not None:
                prefix, prefix_tags = expected_prefix
                length = prefix.shape[0]
                if (not torch.equal(raw[:length], prefix.to(raw))
                        or not torch.equal(tags[:length], prefix_tags.to(tags))
                        or not torch.all(tags[length:] == 1)):
                    raise ValueError('Direct generation changed cached full P/Ptags or non-text suffix tags')
            else:
                raise ValueError('Direct native generation requires the explicitly cached full P/Ptags')
            if shared.get('ref_visual_anchor') is None or shared.get('ref_audio_anchor') is None:
                raise ValueError('Original independent visual/audio VAE reference must remain present')
            if music_mode == 'ref' and (shared.get('input_latents_audio') is not None or shared.get('denoise_mask_audio') is not None):
                raise ValueError('Reference-only E3 mode must not introduce audio retake')
            image_blocks = [block for block in shared['ref_blocks'] if block['kind'] == 'image']
            metadata = embedder.prefix_record['metadata']
            if len(image_blocks) != 1 or image_blocks[0].get('prepared_image') is None:
                raise ValueError('Require the actual original prepared reference image')
            prepared = image_blocks[0]['prepared_image']
            if (list(prepared.size) != metadata['prepared_image_size'] or
                    hashlib.sha256(prepared.tobytes()).hexdigest() != metadata['prepared_image_sha256']):
                raise ValueError('Native VAE prepared image differs from cached full P')
            if not torch.equal(packed['token_tags'][packed['text_pos']], tags.to(packed['token_tags'])):
                raise ValueError('Original packed text tags differ from cached P plus text suffix tags')
            audit.update(protocol=DIRECT_PROTOCOL, text_encoder_loaded=False, tokenizer_loaded=False,
                processor_loaded=False, text_encoder_calls=0, semantic_text=None,
                full_cached_prefix_preserved=True, original_cached_prefix_tags_preserved=True,
                prefix_tokens=length, prompt_embeds_shape=list(raw.shape), prompt_embeds_sha256=tensor_sha256(raw),
                text_token_tags_sha256=tensor_sha256(tags), packed_text_tags_exact=torch.equal(packed['token_tags'][packed['text_pos']], tags.to(packed['token_tags'])),
                waveform_sha256=tensor_sha256(native_waveform), waveform_samples=native_waveform.shape[-1],
                native_audio_condition='real_reference_audio', music_mode=music_mode,
                reference=optional_tensor(shared['ref_audio_anchor']),
                retake=optional_tensor(shared.get('input_latents_audio')),
                denoise_mask_audio=optional_tensor(shared.get('denoise_mask_audio')),
                visual_reference=optional_tensor(shared['ref_visual_anchor']),
                audio_cond_noise_aug=float(shared['audio_cond_noise_aug']),
                imgvid_cond_noise_aug=float(shared['imgvid_cond_noise_aug']),
                native_audio_vae_unchanged=True, original_joint_pipeline=True,
                audio_tail_policy='original_native_unmodified', sampling=dict(sampling))
            print('[direct cached prefix native] ' + json.dumps(audit), flush=True)
        return shared, positive, negative

    def step(scheduler, *args, **kwargs):
        key = 'video' if scheduler is pipe.scheduler else 'audio' if scheduler is pipe.scheduler_audio else None
        if key is None:
            raise ValueError('Unexpected scheduler in original joint pipeline')
        updates[key] += 1
        return original_step(scheduler, *args, **kwargs)

    def model_fn(*args, **kwargs):
        if not first_noise:
            first_noise.update(video=float(kwargs['timestep_video'].reshape(-1)[0]),
                               audio=float(kwargs['timestep_audio'].reshape(-1)[0]))
            mask = kwargs.get('denoise_mask_audio')
            first_noise['retake_effective_time_one_rows'] = int((mask == 0).sum()) if mask is not None else 0
            first_noise['retake_regenerated_rows'] = int((mask != 0).sum()) if mask is not None else None
            if identity_output is not None:
                teacher = dict(kwargs, **identity_output)
                teacher['packed'] = MiniMaxH3Unit_PackedSequenceBuilder().process(
                    pipe, prompt_embeds=teacher['prompt_embeds'], text_token_tags=teacher['text_token_tags'],
                    video_latents=kwargs['video_latents'], audio_latents=kwargs['audio_latents'],
                    keyframe_cond_anchor=kwargs.get('keyframe_cond_anchor'), keyframe_indices=kwargs.get('keyframe_indices'),
                    ref_blocks=kwargs['ref_blocks'])['packed']
                packed = kwargs['packed']
                packed_exact = teacher['packed'].keys() == packed.keys() and all(
                    torch.equal(value, packed[key]) if torch.is_tensor(value) else value == packed[key]
                    for key, value in teacher['packed'].items())
                raw_exact = torch.equal(teacher['prompt_embeds'], kwargs['prompt_embeds'])
                tags_exact = torch.equal(teacher['text_token_tags'], kwargs['text_token_tags'])
                teacher_video, teacher_audio = original_model_fn(*args, **teacher)
                student_video, student_audio = original_model_fn(*args, **kwargs)
                velocity = dict(video=difference(student_video, teacher_video), audio=difference(student_audio, teacher_audio))
                if not packed_exact or not raw_exact or not tags_exact or not torch.equal(student_video, teacher_video) or not torch.equal(student_audio, teacher_audio):
                    raise ValueError('Cached teacher P+y/tags/packed or original joint video/audio identity prediction mismatch')
                audit['identity'] = dict(raw_exact=raw_exact, tags_exact=tags_exact, packed_exact=packed_exact,
                                        velocity=velocity, comparison='paired_same_input_cached_full_teacher_vs_P_plus_y')
                print('[direct cached prefix identity] ' + json.dumps(audit['identity']), flush=True)
                return student_video, student_audio
        return original_model_fn(*args, **kwargs)

    pipe.unit_runner, pipe.step, pipe.model_fn = runner, step, model_fn
    try:
        reference_args = dict(references=[dict(type='image', image=image),
                                         dict(type='audio', audio=native_waveform, sample_rate=sample_rate)])
        if music_mode == 'retake':
            reference_args.update(retake_audio=native_waveform, retake_audio_sample_rate=sample_rate, seconds_regions_to_retake=[])
        video, audio = pipe(prompt=None, negative_prompt=None, **sampling, **reference_args)
        if not audit or updates != dict(video=sampling['num_inference_steps'], audio=sampling['num_inference_steps']):
            raise ValueError('Original joint video AND audio scheduler updates were not preserved')
        audit.update(scheduler_updates=updates, first_noise_timesteps=first_noise,
                     generated_audio_sha256=tensor_sha256(audio), generated_audio_shape=list(audio.shape))
        print('[direct cached prefix complete] ' + json.dumps(audit), flush=True)
        return video, audio, audit
    finally:
        pipe.unit_runner, pipe.step, pipe.model_fn = original_runner, original_step, original_model_fn
        pipe.units[matches[0]] = original_unit


def direct_generation_manifest(selection, cache_roots, condition='joint', csv_dir=None):
    """Join completed native teacher shards without reading a target prompt/tensor."""
    import torch
    from music_text_condition import validate_cached_reference_prefix, file_sha256
    selection = Path(selection).resolve(strict=True)
    paths = {name: selection / (name + '.json') for name in ('caps', 'split', 'references', 'source')}
    source_files = {name: dict(path=str(path), sha256=file_sha256(path)) for name, path in paths.items()}
    split = json.loads(paths['split'].read_text())
    refs = json.loads(paths['references'].read_text())
    ids = split.get('val', [])
    if (condition not in ('joint', 'aswap') or split.get('protocol') != DIRECT_PROTOCOL
            or refs.get('protocol') != DIRECT_PROTOCOL or split.get('train') != []
            or len(ids) != 3 or len(set(ids)) != 3 or set(refs.get('records', {})) != set(ids)
            or any(Path(sid).name != sid or sid in ('.', '..') for sid in ids)):
        raise ValueError('Generation requires exactly three explicit holdout cases, never a training split')
    native = refs['native_generation']
    if native != dict(height=1280, width=704, num_frames=124, fps=24):
        raise ValueError('This round explicitly selects 1280x704/124 frames/24fps')
    rows, caches = {}, []
    for root in cache_roots:
        root = Path(root).resolve(strict=True)
        plan_path, finished_path = root / 'plan.json', root / 'finished.json'
        plan, finished = json.loads(plan_path.read_text()), json.loads(finished_path.read_text())
        plan_sha = file_sha256(plan_path)
        if (plan.get('protocol') != DIRECT_PROTOCOL or plan.get('kind') != 'cached_reference_prompt_teacher'
                or finished.get('protocol') != DIRECT_PROTOCOL or finished.get('complete') is not True
                or finished.get('actual_lm_forward') is not True
                or finished.get('prefix_caption_independence_checked') is not True
                or finished.get('original_split_reconstruction_exact') is not True
                or finished.get('source_protocol_sha256') != plan_sha
                or finished.get('expected') != len(finished.get('rows', []))
                or plan.get('native_generation') != native):
            raise ValueError('Generation cache is not the complete original native P/teacher shard')
        for name in ('caps', 'split', 'references'):
            if plan.get(name) != source_files[name]:
                raise ValueError('Native shard selection source binding differs: ' + name)
        caches.append(dict(plan_path=str(plan_path), plan_sha256=plan_sha,
                           finished_path=str(finished_path), finished_sha256=file_sha256(finished_path),
                           model_weight_content_sha256=plan['model_weight_content_sha256']))
        for row in finished['rows']:
            sid = row['sample_id']
            if sid not in ids or sid in rows or row.get('partition') != 'val':
                raise ValueError('Extra, duplicate or training case in generation shard')
            binding = refs['records'][sid]
            for kind in ('image', 'audio'):
                if file_sha256(binding[kind + '_path']) != binding[kind + '_sha256']:
                    raise ValueError('Reference source content changed: ' + kind)
            for kind, folder in (('prefix', 'prefixes'), ('target', 'targets')):
                path = Path(row[kind + '_path']).resolve(strict=True)
                if path != root / folder / (sid + '.pt') or file_sha256(path) != row[kind + '_sha256']:
                    raise ValueError('Native cached P/teacher file binding changed')
            record = torch.load(row['prefix_path'], map_location='cpu', weights_only=True)
            metadata = validate_cached_reference_prefix(record, image_path=binding['image_path'],
                                                        reference_specs=binding['references'])
            if (metadata.get('sample_id') != sid or metadata.get('partition') != 'val'
                    or metadata.get('source_protocol_sha256') != plan_sha
                    or metadata.get('prefix_length') != row['prefix_length']
                    or metadata.get('model_weight_content_sha256') != plan['model_weight_content_sha256']):
                raise ValueError('Cached prefix metadata differs from completed shard')
            rows[sid] = dict(sample_id=sid, condition=condition, **{key: binding[key] for key in
                ('image_path', 'image_sha256', 'audio_path', 'audio_sha256')},
                prefix_path=row['prefix_path'], prefix_file_sha256=row['prefix_sha256'],
                target_path=row['target_path'], target_sha256=row['target_sha256'],
                prefix_length=row['prefix_length'], prefix_sha256=metadata['prefix_sha256'],
                prefix_tags_sha256=metadata['prefix_tags_sha256'],
                prepared_image_size=metadata['prepared_image_size'],
                prepared_image_sha256=metadata['prepared_image_sha256'],
                source_protocol_path=str(plan_path), source_protocol_sha256=plan_sha,
                finished_path=str(finished_path), finished_sha256=file_sha256(finished_path))
    if set(rows) != set(ids) or len({cache['model_weight_content_sha256'] for cache in caches}) != 1:
        raise ValueError('Missing holdout cases or inconsistent native teacher weights')
    if condition == 'aswap':
        import csv
        if csv_dir is None:
            raise ValueError('Aswap requires the existing E2 CSV directory, never an invented swap')
        csv_path = Path(csv_dir).resolve(strict=True) / 'full_aswap.csv'
        with csv_path.open(newline='', encoding='utf-8') as stream:
            swaps = [row for _, row in zip(range(3), csv.DictReader(stream))]
        if [row['sample_id'] for row in swaps] != ids:
            raise ValueError('E2 swapped cases differ from the original fixed three people/order')
        source_files['aswap_csv'] = dict(path=str(csv_path), sha256=file_sha256(csv_path))
        for swap in swaps:
            row = rows[swap['sample_id']]
            row['original_audio_path'], row['original_audio_sha256'] = row['audio_path'], row['audio_sha256']
            row['audio_path'] = str(Path(swap['audio_path']).resolve(strict=True))
            row['audio_sha256'] = file_sha256(row['audio_path'])
    return dict(protocol=DIRECT_PROTOCOL, kind='direct_cached_prefix_native_generation', complete=True,
                source_people=3, condition=condition, selection=str(selection),
                source_files=source_files, cache_roots=[str(Path(root).resolve()) for root in cache_roots],
                caches=caches, native_generation=native, cases=[rows[sid] for sid in ids],
                student_reads_target_or_caption=False, teacher_identity_scope='offline_cached_full_teacher_only')


def validate_direct_generation_manifest(path):
    from music_text_condition import file_sha256
    path = Path(path)
    manifest = json.loads(path.read_text())
    csv_source = manifest.get('source_files', {}).get('aswap_csv')
    actual = direct_generation_manifest(manifest['selection'], manifest['cache_roots'],
                                        manifest['condition'], Path(csv_source['path']).parent if csv_source else None)
    if actual != manifest:
        raise ValueError('Direct generation manifest differs from actual unchanged complete native sources')
    return manifest, file_sha256(path)


def prepare_direct_generation_manifest_main(argv=None):
    parser = argparse.ArgumentParser(description='CPU metadata-only native three-case binding; no target prompts or GPU')
    parser.add_argument('--prepare_direct_generation_manifest', action='store_true', required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--cache_roots', type=Path, nargs='+', required=True)
    parser.add_argument('--conds', choices=['joint', 'aswap'], default='joint')
    parser.add_argument('--csv_dir', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    from music_generation_validate import atomic_write_new
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require an unused output JSON and an existing parent; no overwrite')
    result = direct_generation_manifest(args.selection, args.cache_roots, args.conds, args.csv_dir)
    atomic_write_new(args.out, result)
    print(json.dumps(dict(complete=True, protocol=DIRECT_PROTOCOL, source_people=3,
                          condition=args.conds, out=str(args.out), no_gpu=True)), flush=True)


def direct_teacher_identity_embedder(prefix_record, target_path):
    import torch
    from music_text_condition import tensor_sha256, validate_cached_reference_prefix
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PromptEmbedder
    metadata = validate_cached_reference_prefix(prefix_record)
    target = torch.load(target_path, map_location='cpu', weights_only=True)
    tail, ids, meta = target['tail'], target['source_token_ids'], target['metadata']
    if (meta.get('protocol') != DIRECT_PROTOCOL or meta.get('kind') != 'cached_reference_prompt_tail'
            or meta.get('sample_id') != metadata['sample_id'] or tail.dtype != torch.bfloat16
            or tail.ndim != 2 or tail.shape[1] != 5120 or len(tail) < 1 or not torch.isfinite(tail).all()
            or ids.dtype != torch.long or ids.shape != (len(tail),)
            or target['teacher_length'] != len(tail) or meta.get('teacher_length') != len(tail)
            or meta.get('tail_sha256') != tensor_sha256(tail)
            or meta.get('source_token_ids_sha256') != tensor_sha256(ids)
            or any(meta.get(key) != metadata[key] for key in
                   ('source_protocol_sha256', 'prefix_sha256', 'image_sha256', 'model_weight_content_sha256'))):
        raise ValueError('Identity teacher raw tail/source IDs do not match the native cached prefix')
    full = torch.cat((prefix_record['prefix'], tail))
    tags = torch.cat((prefix_record['prefix_tags'], torch.ones(len(tail), dtype=torch.long)))

    class IdentityEmbedder(MiniMaxH3Unit_PromptEmbedder):
        def __init__(self):
            super().__init__()
            self.onload_model_names = ()
            self.prefix_record = prefix_record

        def process(self, pipe, prompt=None, keyframes=None, ref_blocks=None,
                    height=None, width=None, text_embedding=None):
            if prompt is not None or keyframes or text_embedding is not None:
                raise ValueError('Cached full teacher diagnostic must not read live semantic text')
            validate_cached_reference_prefix(prefix_record, reference_specs=[dict(kind=b['kind']) for b in ref_blocks])
            return dict(prompt_embeds=torch.cat((prefix_record['prefix'], tail)).to(pipe.device),
                        text_token_tags=tags.to(pipe.device))
    return IdentityEmbedder(), dict(prompt_embeds=full, text_token_tags=tags)


def direct_native_main(argv=None):
    parser = direct_native_parser()
    args = parser.parse_args(argv)
    manifest, manifest_sha = validate_direct_generation_manifest(args.cached_prefix_manifest)
    native = manifest['native_generation']
    height = args.height if args.height is not None else native['height']
    width = args.width if args.width is not None else native['width']
    if (manifest['condition'] != args.conds or not 1 <= args.nshard <= 3
            or not 0 <= args.shard < args.nshard
            or dict(height=height, width=width, num_frames=args.num_frames, fps=24) != native
            or (args.steps, args.shift, args.audio_shift, args.seed) != (50, 12.0, 3.0, 42)
            or (args.direct_teacher_identity and (args.conds != 'joint' or args.direct_zero_input))):
        parser.error('Require exact selected three-case native settings; identity is joint only, never a swapped/zero teacher')
    if not args.out.is_dir():
        parser.error('Require an explicit existing output tree; the managed driver creates only new trees')
    import torch
    import librosa
    from PIL import Image
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
    from diffsynth.utils.data.audio_video import write_video_audio
    from music_text_condition import file_sha256, tensor_sha256
    from music_text_infer import direct_student_text_embedder
    from music_clip_bridge import load_audio
    from music_generation_validate import atomic_write_new
    sampling = direct_sampling_settings(args, height, width)
    cases = manifest['cases'][args.shard::args.nshard]
    outputs = [args.out / args.conds / (case['sample_id'] + '_' + Path(case['audio_path']).stem + '.mp4') for case in cases]
    if any(path.exists() or path.with_suffix('.audit.json').exists() for path in outputs):
        raise FileExistsError('Refuse existing direct video/audit outputs; never silently skip partial runs')
    vram = dict(offload_dtype=torch.bfloat16, offload_device='cpu', onload_dtype=torch.bfloat16,
                onload_device='cpu', preparing_dtype=torch.bfloat16, preparing_device='cuda',
                computation_dtype=torch.bfloat16, computation_device='cuda')
    weights = Path(args.w)
    configs = [ModelConfig(path=sorted(str(path) for path in (weights / 'transformer').glob('model*.safetensors')), **vram),
               ModelConfig(path=str(weights / 'video_vae/source/model.safetensors'), **vram),
               ModelConfig(path=str(weights / 'audio_vae/model.safetensors'), **vram)]
    pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=configs, processor_config=None, vram_limit=torch.cuda.mem_get_info('cuda')[1] / 1024 ** 3 - 5)
    if any(getattr(pipe, name) is not None for name in ('text_encoder', 'tokenizer', 'processor')):
        raise ValueError('Direct native pipeline unexpectedly loaded text processing')
    import inspect
    code_files = [Path(__file__), Path(__file__).with_name('music_text_infer.py'),
                  Path(__file__).with_name('music_text_condition.py'),
                  Path(__file__).parent.parent / 'h3ar/sample_h3_ar.py',
                  Path(inspect.getfile(MiniMaxH3Pipeline)), Path(inspect.getfile(write_video_audio))]
    source_files = {path.name: dict(path=str(path), sha256=file_sha256(path)) for path in code_files}
    checkpoint_sha = file_sha256(args.direct_music_ckpt) if args.direct_music_ckpt else None
    print('[direct native loaded] ' + json.dumps(dict(protocol=DIRECT_PROTOCOL, text_encoder_loaded=False,
          tokenizer_loaded=False, processor_loaded=False, cached_prefix_manifest_sha256=manifest_sha,
          source_files=source_files, checkpoint_sha256=checkpoint_sha)), flush=True)
    for case, output in zip(cases, outputs):
        started = __import__('time').monotonic()
        prefix = torch.load(case['prefix_path'], map_location='cpu', weights_only=True)
        with Image.open(case['image_path']) as source_image:
            image = source_image.convert('RGB')
        native_values, rate = librosa.load(case['audio_path'], sr=pipe.audio_vae.sample_rate, mono=False, duration=args.num_frames / 24)
        native_waveform = torch.from_numpy(native_values).float()
        if native_waveform.ndim == 1:
            native_waveform = native_waveform.unsqueeze(0).repeat(2, 1)
        if native_waveform.ndim != 2 or native_waveform.shape[0] != 2 or not torch.isfinite(native_waveform).all():
            raise ValueError('Native audio requires finite original mono/stereo source waveform')
        identity = None
        student_sha = None
        if args.direct_teacher_identity:
            embedder, identity = direct_teacher_identity_embedder(prefix, case['target_path'])
            identity = {key: value.to(device=pipe.device) for key, value in identity.items()}
        else:
            embedder = direct_student_text_embedder(args.direct_music_ckpt, case['prefix_path'])
            audio_path = Path(case['audio_path'])
            student_waveform, _ = load_audio(audio_path.stem, audio_root=audio_path.parent)
            if args.direct_zero_input:
                student_waveform = torch.zeros_like(student_waveform)
            student_sha = tensor_sha256(student_waveform)
            embedder.set_waveform(student_waveform)
        input_record = dict(protocol=DIRECT_PROTOCOL, case=case, sampling=sampling,
            manifest_path=str(args.cached_prefix_manifest), manifest_sha256=manifest_sha,
            source_files=source_files, checkpoint=str(args.direct_music_ckpt) if args.direct_music_ckpt else None,
            checkpoint_sha256=checkpoint_sha, teacher_identity=args.direct_teacher_identity,
            student_zero_input=args.direct_zero_input, student_waveform_sha256=student_sha,
            student_waveform_loader='music_clip_bridge.load_audio_fixed_2x162000_32k',
            native_waveform_loader='librosa_native_32k_duration_num_frames_over_24',
            native_waveform_sha256=tensor_sha256(native_waveform), native_sample_rate=int(rate),
            output_audio=args.output_audio, music_mode=args.music_mode,
            student_reads_target_or_caption=False, semantic_text=None)
        print('[direct native input] ' + json.dumps(input_record), flush=True)
        video, audio, audit = direct_native_call(pipe, embedder, image, native_waveform, rate, sampling,
            args.music_mode, (prefix['prefix'], prefix['prefix_tags']), identity)
        output.parent.mkdir(exist_ok=True)
        mux_audio = audio if args.output_audio == 'generated' else native_waveform
        mux_rate = pipe.audio_vae.sample_rate if args.output_audio == 'generated' else rate
        write_video_audio(video, mux_audio, str(output), fps=24, audio_sample_rate=mux_rate)
        if file_sha256(args.cached_prefix_manifest) != manifest_sha:
            raise ValueError('Manifest changed during direct native inference')
        record = dict(complete=True, input=input_record, audit=audit, output_path=str(output),
            video_sha256=file_sha256(output), mux_audio_sha256=tensor_sha256(mux_audio),
            mux_audio_sample_rate=int(mux_rate), generated_audio_preserved=args.output_audio == 'generated',
            full_media_decode_checked=False, musical_quality_accepted=False)
        atomic_write_new(output.with_suffix('.audit.json'), record)
        print('[direct native saved] ' + json.dumps(record), flush=True)
        print('[h3ar] ' + args.conds + ' ' + case['sample_id'] + ' ' + str(round(__import__('time').monotonic() - started)) + 's', flush=True)
    print('[h3ar] done', flush=True)


def direct_managed_main(argv=None):
    parser = argparse.ArgumentParser(description='Managed direct native three-case generation; strict explicit admission')
    parser.add_argument('--direct_native', action='store_true', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cached_prefix_manifest', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--direct_teacher_identity', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hosts', nargs='+', required=True)
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--holder_path', type=Path, required=True)
    parser.add_argument('--music_mode', choices=['ref', 'retake'], default='ref')
    parser.add_argument('--output_audio', choices=['generated', 'original_mux'], default='generated')
    parser.add_argument('--zero_input', action='store_true')
    args = parser.parse_args(argv)
    allowed = {'28.216.19.69', '28.216.17.70', '28.216.19.213', '28.216.19.143', '28.216.19.137', '28.216.19.225'}
    if (args.out.exists() or not args.out.parent.is_dir() or not args.holder_path.is_file()
            or not 1 <= len(args.hosts) <= 3 or len(set(args.hosts)) != len(args.hosts)
            or not set(args.hosts) <= allowed or len(args.gpus) != len(args.hosts)
            or any(gpu not in range(8) for gpu in args.gpus)
            or (args.direct_teacher_identity and args.zero_input)):
        parser.error('Require a new output tree, exact own holder, 1-3 unique approved hosts and explicit GPU slots')
    manifest, manifest_sha = validate_direct_generation_manifest(args.cached_prefix_manifest)
    if args.direct_teacher_identity and manifest['condition'] != 'joint':
        parser.error('Cached teacher identity uses original joint only')
    from music_text_condition import file_sha256
    from music_generation_validate import atomic_write_new
    sampler = args.root / 'scripts_omnidance/exp/h3ar/sample_h3_ar.py'
    upstream = args.root.parent / 'DiffSynth-Studio-upstream-20260930'
    source_files = {str(path): file_sha256(path) for path in
                    (sampler, Path(__file__), Path(__file__).with_name('music_text_infer.py'),
                     Path(__file__).with_name('music_text_condition.py'),
                     upstream / 'diffsynth/pipelines/minimax_h3_audio_video.py',
                     upstream / 'diffsynth/utils/data/audio_video.py')}
    checkpoint_sha = file_sha256(args.checkpoint) if args.checkpoint else None
    args.out.mkdir()
    (args.out / 'logs').mkdir()
    (args.out / manifest['condition']).mkdir()
    plan = dict(protocol=DIRECT_PROTOCOL, kind='direct_cached_prefix_native_generation', source_people=3,
        expected=3, cached_prefix_manifest=str(args.cached_prefix_manifest), cached_prefix_manifest_sha256=manifest_sha,
        checkpoint=str(args.checkpoint) if args.checkpoint else None, checkpoint_sha256=checkpoint_sha,
        teacher_identity=args.direct_teacher_identity, student_zero_input=args.zero_input,
        native_generation=manifest['native_generation'], music_mode=args.music_mode, output_audio=args.output_audio,
        source_files=source_files, hosts=args.hosts, gpus=args.gpus, holder_path=str(args.holder_path),
        no_text_encoder=True, native_audio='real_reference_unmodified', student_reads_target_or_caption=False)
    atomic_write_new(args.out / 'plan.json', plan)
    queues = [list(range(3))[i::len(args.hosts)] for i in range(len(args.hosts))]
    ssh = SSH + ['-p', '36000']

    def run_queue(slot):
        host, gpu, shards = slot
        jobs = []
        for shard in shards:
            guard = guard_code(str(sampler), str(args.holder_path), gpu).replace(
                'if holder in args:', 'if holder in args and p.stat().st_uid == os.getuid():')
            check = subprocess.run(ssh + [host, 'python3 -c ' + shlex.quote(guard)], text=True, capture_output=True)
            atomic_write_new(args.out / 'logs' / ('admission_' + str(shard) + '.json'),
                             dict(host=host, gpu=gpu, exit_code=check.returncode, stdout=check.stdout, stderr=check.stderr))
            if check.returncode:
                jobs.append(dict(host=host, gpu=gpu, shard=shard, exit_code=check.returncode, admission_failed=True))
                break
            command = ['env', 'CUDA_VISIBLE_DEVICES=' + str(gpu), 'OMP_NUM_THREADS=2',
                'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
                'PYTHONPATH=' + str(args.root.parent / 'DiffSynth-Studio-upstream-20260930'),
                '/opt/venv_h3/bin/python', str(sampler), '--out', str(args.out),
                '--cached_prefix_manifest', str(args.cached_prefix_manifest), '--conds', manifest['condition'],
                '--shard', str(shard), '--nshard', '3', '--rows', '3', '--no_text_encoder',
                '--music_mode', args.music_mode, '--output_audio', args.output_audio]
            command += ['--direct_teacher_identity'] if args.direct_teacher_identity else ['--direct_music_ckpt', str(args.checkpoint)]
            if args.zero_input:
                command += ['--direct_zero_input']
            log = args.out / 'logs' / ('shard_' + str(shard) + '.log')
            with log.open('x') as stream:
                stream.write(json.dumps(dict(host=host, gpu=gpu, command=command,
                    manifest_sha256=manifest_sha, source_files=source_files, checkpoint_sha256=checkpoint_sha)) + '\n')
                stream.flush()
                result = subprocess.run(ssh + [host, shlex.join(command)], stdout=stream, stderr=subprocess.STDOUT)
            text = log.read_text()
            complete = result.returncode == 0 and '[h3ar] done' in text and text.count('[direct native saved] ') == 1
            jobs.append(dict(host=host, gpu=gpu, shard=shard, exit_code=result.returncode,
                             sampler_complete=complete, log=str(log), log_sha256=file_sha256(log)))
            if not complete:
                break
        return jobs
    with ThreadPoolExecutor(max_workers=len(args.hosts)) as pool:
        jobs = [job for queue in pool.map(run_queue, zip(args.hosts, args.gpus, queues)) for job in queue]
    complete = len(jobs) == 3 and all(job.get('sampler_complete') is True for job in jobs)
    complete = complete and file_sha256(args.cached_prefix_manifest) == manifest_sha
    complete = complete and all(file_sha256(path) == digest for path, digest in source_files.items())
    if args.checkpoint:
        complete = complete and file_sha256(args.checkpoint) == checkpoint_sha
    atomic_write_new(args.out / 'jobs.json', dict(protocol=DIRECT_PROTOCOL, complete=complete,
        all_sampler_jobs_exit_zero=complete, expected=3, jobs=jobs, plan_sha256=file_sha256(args.out / 'plan.json'),
        full_media_decode_checked=False, musical_quality_accepted=False))
    if not complete:
        raise SystemExit(1)


def main():
    import sys
    if '--prepare_direct_generation_manifest' in sys.argv:
        return prepare_direct_generation_manifest_main()
    if '--direct_native' in sys.argv:
        return direct_managed_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--caps', type=Path)
    source.add_argument('--checkpoint', type=Path, help='Deploy a trained complete raw-text model, never the oracle')
    source.add_argument('--projection', type=Path, help='Same-length projected teacher manifest; not trained music generation')
    source.add_argument('--teacher_tail_cache', type=Path, help='Existing offline teacher tails, without TC/prefix; teacher-only diagnostic')
    parser.add_argument('--no_text_encoder', action='store_true', help='Omit text encoder, tokenizer and processor entirely; no blank text input')
    parser.add_argument('--checkpoint_source', type=Path, help='Canonical completed training checkpoint when --checkpoint is a verified local byte-for-byte cache')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hosts', nargs='+', required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--gpus', nargs='+', type=int, help='Explicit GPU per host slot; permits one noTC case per unique host/GPU pair')
    parser.add_argument('--holder_path', type=Path, help='Explicit existing holder script when source tree and experiment tree differ')
    parser.add_argument('--joint_only', action='store_true', help='Three same-person teacher-label diagnostic clips only')
    parser.add_argument('--zero_input', action='store_true', help='Deploy model on zero music, original track only muxed')
    parser.add_argument('--legacy_clip', action='store_true', help='Reuse all old trained encoder/adapter/head weights as one global text token')
    args = parser.parse_args()
    expected = 3 if args.joint_only else 6
    conditions = ['joint'] if args.joint_only else ['joint', 'aswap']
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require a new output tree and existing parent')
    gpus = args.gpus if args.gpus is not None else [args.gpu] * len(args.hosts)
    if (len(gpus) != len(args.hosts) or any(not 0 <= gpu < 8 for gpu in gpus)
            or len(args.hosts) > expected or len(set(zip(args.hosts, gpus))) != len(args.hosts)
            or (args.gpus is None and len(set(args.hosts)) != len(args.hosts))
            or (args.gpus is not None and (not args.no_text_encoder or len(args.hosts) != expected))):
        parser.error('Require unique host/GPU slots; explicit GPU slots are noTC-only with exactly one case per slot')
    if (args.zero_input or args.legacy_clip) and not args.checkpoint:
        parser.error('Zero/legacy music input requires a deployed model checkpoint')
    if args.no_text_encoder and (args.caps or args.projection or args.legacy_clip):
        parser.error('No-TC cannot combine with caption oracle, projected teacher or legacy CLIP conditions')
    if args.teacher_tail_cache and (not args.no_text_encoder or not args.joint_only
            or not args.teacher_tail_cache.is_dir()
            or len(list(args.teacher_tail_cache.glob('*.pt'))) < 3):
        parser.error('Cached teacher tail requires no_text_encoder, exactly three joint cases and existing cache files')
    if args.caps:
        caps = json.loads(args.caps.read_text())
        if len(caps) != 6 or any(not value.get('source') for value in caps.values()):
            parser.error('Require six source-declared oracle supervision captions')
    elif args.projection:
        projected = json.loads(args.projection.read_text())
        if (not args.joint_only or projected.get('complete') is not True
                or projected.get('kind') != 'projected_teacher_not_music_prediction'
                or projected.get('diagnostic_only') is not True or len(projected.get('cases', {})) != 3):
            parser.error('Projection requires a complete three-joint diagnostic manifest')
    elif args.checkpoint and not args.checkpoint.is_file():
        parser.error('Require an existing complete music text checkpoint')
    source_summary_sha256 = None
    if args.checkpoint_source:
        if not args.checkpoint or not args.no_text_encoder or not args.checkpoint_source.is_file():
            parser.error('Checkpoint cache provenance requires a noTC model and an existing canonical source')
        summary_path = args.checkpoint_source.parent / 'finished.json'
        summary = json.loads(summary_path.read_text())
        if (summary.get('complete') is not True or summary['deployment'].get('independent_inference_tail_exact') is not True
                or summary['protocol'].get('context_policy') != 'music_only_global_bias'
                or hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != summary['checkpoint_sha256']):
            parser.error('Local inference checkpoint does not match its completed canonical source SHA')
        source_summary_sha256 = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    context_policy = 'tc_prefix_context'
    if args.no_text_encoder and args.checkpoint:
        from music_text_infer import _checkpoint_context_policy
        context_policy = _checkpoint_context_policy(args.checkpoint)
    teacher_source = teacher_tail_cache_source(args.teacher_tail_cache) if args.teacher_tail_cache else None
    pure_text_teacher = bool(teacher_source and teacher_source['kind'] == 'pure_text_h3_raw_tokens')
    old_no_tc_model = bool(args.no_text_encoder and args.checkpoint and context_policy == 'tc_prefix_context')
    no_tc_kind = ('pure_text_teacher_no_tc_diagnostic' if pure_text_teacher else
                  'cached_teacher_tail_only_no_tc_diagnostic' if args.teacher_tail_cache else
                  'legacy_checkpoint_no_tc_music_only_diagnostic' if old_no_tc_model else
                  'music_only_global_bias_no_tc_deployment')
    sampler = args.root / 'scripts_omnidance/exp/h3ar/sample_h3_ar.py'
    holder = args.holder_path or args.root / 'scripts_omnidance/exp/hold_gpu.py'
    if not holder.is_file():
        parser.error('Require an existing holder path for exact process admission')
    args.out.mkdir()
    (args.out / 'logs').mkdir()
    (args.out / 'cache').mkdir()
    from music_generation_validate import atomic_write_new as write_new
    code_sha = hashlib.sha256(sampler.read_bytes()).hexdigest()
    write_new(args.out / 'plan.json', dict(kind=no_tc_kind if args.no_text_encoder else 'projected_teacher_not_music_prediction' if args.projection else 'teacher_oracle_not_trained_music_model' if args.caps else 'trained_music_text_deployment',
              source_people=3, conditions=conditions, videos=expected, hosts=args.hosts, gpus=gpus,
              serialized_host_cold_loading=args.gpus is not None,
              sampler_sha256=code_sha, supervision_caps_sha256=hashlib.sha256(args.caps.read_bytes()).hexdigest() if args.caps else None,
              checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
              native_audio='same_fixed_silence_reference_and_retake', semantic_student_prompt=None if args.no_text_encoder else ' ',
              no_text_encoder=args.no_text_encoder, text_encoder_loaded=False if args.no_text_encoder else None,
              teacher_tail_cache=str(args.teacher_tail_cache) if args.teacher_tail_cache else None,
              teacher_source=teacher_source,
              source_scope=teacher_source['source_scope'] if teacher_source else None,
              old_model_not_trained_for_no_tc=old_no_tc_model, context_policy=context_policy,
              holder_path=str(holder), checkpoint_inference_path=str(args.checkpoint) if args.checkpoint else None,
              checkpoint_source=str(args.checkpoint_source or args.checkpoint) if args.checkpoint else None,
              checkpoint_source_summary_sha256=source_summary_sha256,
              prefix_length=0 if args.no_text_encoder else None,
              sampler_no_tc_arguments=(['--no_text_encoder'] + (['--no_tc_teacher_cache', str(args.teacher_tail_cache)] if args.teacher_tail_cache else [])) if args.no_text_encoder else [],
              raw_text_tail_replacement=True, no_training=True, student_uses_caption=bool(args.caps),
              zero_music_input=args.zero_input, legacy_clip_complete_chain=args.legacy_clip,
              projection_manifest=str(args.projection) if args.projection else None,
              projection_manifest_sha256=hashlib.sha256(args.projection.read_bytes()).hexdigest() if args.projection else None,
              projection_teacher_length_only=bool(args.projection)))
    queues = [list(range(expected))[i::len(args.hosts)] for i in range(len(args.hosts))]
    import threading
    host_locks = {host: threading.Lock() for host in args.hosts} if args.gpus is not None else {}

    def run_admitted_queue(item, release_cold_load):
        host, shards, gpu = item
        guard = guard_code(str(sampler), str(holder), gpu)
        admission = subprocess.run(SSH + [host, 'python3 -c ' + shlex.quote(guard)],
                                   text=True, capture_output=True)
        admission_name = 'admission_' + host + ('_gpu' + str(gpu) if args.gpus is not None else '') + '.json'
        write_new(args.out / 'logs' / admission_name,
                  dict(host=host, gpu=gpu, exit_code=admission.returncode,
                       stdout=admission.stdout, stderr=admission.stderr))
        if admission.returncode:
            return [dict(host=host, shards=shards, admission_failed=True, exit_code=admission.returncode)]
        result = []
        for shard in shards:
            command = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', 'OMP_NUM_THREADS=2',
                       'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
                       f'PYTHONPATH={args.root.parent / "DiffSynth-Studio-upstream-20260930"}',
                       '/opt/venv_h3/bin/python', str(sampler), '--out', str(args.out),
                       '--shard', str(shard), '--nshard', str(expected), '--rows', '3',
                       '--conds', ','.join(conditions), '--mode', 'bi', '--steps', '50',
                       '--height', '1280', '--width', '704', '--shift', '12',
                       '--audio_shift', '3', '--seed', '42', '--prompt_mode', 'empty',
                       '--audio_tail_mode', 'repeat_clean', '--fixed_native_silence']
            if args.no_text_encoder:
                command += ['--no_text_encoder']
            if args.teacher_tail_cache:
                command += ['--no_tc_teacher_cache', str(args.teacher_tail_cache)]
            elif args.caps:
                command += ['--text_hidden_oracle', str(args.caps), '--text_oracle_cache_dir', str(args.out / 'cache')]
            elif args.projection:
                command += ['--text_hidden_projection', str(args.projection)]
            else:
                command += ['--music_text_ckpt', str(args.checkpoint)]
                if args.zero_input:
                    command += ['--music_text_zero_input']
                if args.legacy_clip:
                    command += ['--music_text_legacy_clip']
            path = args.out / 'logs' / f'shard_{shard}.log'
            with path.open('x') as stream:
                stream.write(json.dumps(dict(host=host, command=command, sampler_sha256=code_sha)) + '\n')
                stream.flush()
                print(json.dumps(dict(event='launch', host=host, shard=shard)), flush=True)
                if args.gpus is None:
                    completed = subprocess.run(SSH + [host, shlex.join(command)], stdout=stream, stderr=subprocess.STDOUT)
                    returncode = completed.returncode
                else:
                    with subprocess.Popen(SSH + [host, shlex.join(command)], stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                        try:
                            for line in process.stdout:
                                stream.write(line)
                                stream.flush()
                                if line.startswith('[no tc native isolation] '):
                                    release_cold_load()
                            returncode = process.wait()
                        finally:
                            if process.poll() is None:
                                process.terminate()
                                process.wait()
            result.append(dict(host=host, gpu=gpu, shard=shard, exit_code=returncode, log=str(path)))
            print(json.dumps(dict(event='exit', **result[-1])), flush=True)
            if returncode:
                break
        return result

    def run_queue(item):
        lock = host_locks.get(item[0])
        released = False
        if lock is not None:
            lock.acquire()

        def release_cold_load():
            nonlocal released
            if lock is not None and not released:
                released = True
                lock.release()

        try:
            return run_admitted_queue(item, release_cold_load)
        finally:
            release_cold_load()

    with ThreadPoolExecutor(max_workers=len(args.hosts)) as executor:
        jobs = [job for queue in executor.map(run_queue, zip(args.hosts, queues, gpus)) for job in queue]
    reports = [json.loads(path.read_text()) for path in (args.out / 'cache').glob('*.json')]
    if args.projection:
        import re
        reports = []
        for job in jobs:
            text = Path(job['log']).read_text()
            records = re.findall(r'^\[music text projection\] (.+)$', text, flags=re.MULTILINE)
            if len(records) != 1 or '[h3ar] done' not in text:
                raise ValueError('Missing complete projected-teacher layout evidence')
            record = json.loads(records[0])
            if record.get('original_packed_exact') is not True or record.get('music_prediction') is not False:
                raise ValueError('Projected teacher layout changed or is mislabeled as music prediction')
            reports.append(dict(status='passed', native_audio=record['native_audio'], projection=record))
    if args.no_text_encoder:
        import re
        reports = []
        for job in jobs:
            if 'log' not in job:
                continue
            text = Path(job['log']).read_text()
            conditions_logged = re.findall(r'^\[no tc condition\] (.+)$', text, flags=re.MULTILINE)
            native = re.findall(r'^\[no tc native isolation\] (.+)$', text, flags=re.MULTILINE)
            if len(conditions_logged) != 1 or len(native) != 1 or '[h3ar] done' not in text:
                raise ValueError('Missing complete no-TC condition/native isolation evidence')
            condition = json.loads(conditions_logged[0])
            if (condition.get('loaded') is not False or type(condition.get('calls')) is not int
                    or condition['calls'] != 0 or condition.get('semantic_text', 'missing') is not None
                    or condition.get('processor_loaded') is not False
                    or condition.get('tokenizer_loaded') is not False
                    or type(condition.get('prefix_length')) is not int or condition['prefix_length'] != 0
                    or type(condition.get('predicted_length')) is not int or condition['predicted_length'] <= 0
                    or condition.get('tail_shape') != [condition['predicted_length'], 5120]
                    or not all(type(value) is int for value in condition['tail_shape'])
                    or re.fullmatch(r'[0-9a-f]{64}', str(condition.get('raw_sha256'))) is None):
                raise ValueError('No-TC loaded/called text processing or retained invalid text/image prefix/length/hash')
            native_record = json.loads(native[0])
            if (native_record.get('text_encoder_loaded') is not False
                    or native_record.get('tokenizer_loaded') is not False
                    or native_record.get('processor_loaded') is not False
                    or type(native_record.get('text_encoder_calls')) is not int
                    or native_record['text_encoder_calls'] != 0
                    or native_record.get('semantic_text', 'missing') is not None
                    or type(native_record.get('prefix_tokens')) is not int or native_record['prefix_tokens'] != 0
                    or native_record.get('prompt_embeds_shape') != condition['tail_shape']
                    or type(native_record.get('visual_reference_rows')) is not int
                    or native_record['visual_reference_rows'] <= 0
                    or native_record.get('native_music_bypass') is not False
                    or native_record.get('all_target_audio_clean') is not True
                    or any(re.fullmatch(r'[0-9a-f]{64}', str(native_record.get(key))) is None for key in
                           ('visual_reference_sha256', 'waveform_sha256', 'reference_sha256', 'retake_sha256', 'mask_sha256'))):
                raise ValueError('No-TC native log has text processing, mismatched tail, missing visual reference or native music bypass')
            report = dict(status='passed', native_audio=native_record,
                          no_tc_condition=condition, shard=job['shard'])
            if args.teacher_tail_cache:
                if (condition.get('teacher_only_diagnostic') is not True
                        or condition.get('music_prediction') is not False
                        or type(condition.get('original_teacher_tokens')) is not int
                        or condition['original_teacher_tokens'] != condition['predicted_length']
                        or re.fullmatch(r'[0-9a-f]{64}', str(condition.get('source_sha256'))) is None):
                    raise ValueError('Cached teacher tail lacks explicit teacher-only source/length evidence')
                if pure_text_teacher and (condition.get('kind') != no_tc_kind
                        or condition.get('source_scope') != 'caption_only_no_reference_context'
                        or condition.get('text_only') is not True
                        or condition.get('image_conditioned') is not False
                        or condition.get('teacher_length_for_diagnostic_only') is not True):
                    raise ValueError('Pure-text teacher source provenance or caption-only diagnostic scope differs')
                if not pure_text_teacher and condition.get('kind') == 'pure_text_teacher_no_tc_diagnostic':
                    raise ValueError('Pure-text teacher log requires explicit pure-text binding manifest provenance')
            else:
                deployed = re.findall(r'^\[music text deployment\] (.+)$', text, flags=re.MULTILINE)
                expected_context = 'zeros_legacy_checkpoint_compatibility' if old_no_tc_model else 'none_music_only_global_bias'
                if (len(deployed) != 1 or condition.get('old_model_not_trained_for_no_tc') is not old_no_tc_model
                        or condition.get('context_source') != expected_context
                        or condition.get('kind') != no_tc_kind):
                    raise ValueError('No-TC checkpoint context policy or deployment scope differs from its metadata')
                deployment = json.loads(deployed[0])
                if (re.fullmatch(r'[0-9a-f]{64}', str(condition.get('checkpoint_sha256'))) is None
                        or deployment.get('checkpoint_sha256') != condition['checkpoint_sha256']
                        or deployment.get('tail_shape') != condition['tail_shape']
                        or type(deployment.get('predicted_length')) is not int
                        or deployment['predicted_length'] != condition['predicted_length']
                        or deployment.get('uses_ground_truth_caption') is not False
                        or deployment.get('final_adapter_projection_executed') is not True):
                    raise ValueError('No-TC deployment checkpoint/shape/length differs or GT caption/full adapter contract failed')
                report['deployment'] = deployment
            reports.append(report)
    elif args.checkpoint:
        import re
        reports = []
        for job in jobs:
            text = Path(job['log']).read_text()
            native = re.findall(r'^\[music text native isolation\] (.+)$', text, flags=re.MULTILINE)
            deployed = re.findall(r'^\[music text deployment\] (.+)$', text, flags=re.MULTILINE)
            if len(native) != 1 or len(deployed) != 1 or '[h3ar] done' not in text:
                raise ValueError('Missing exact trained-model deployment/native isolation success evidence')
            reports.append(dict(status='passed', native_audio=json.loads(native[0]),
                                deployment=json.loads(deployed[0]), shard=job['shard']))
    native_keys = ('waveform_sha256', 'reference_sha256', 'retake_sha256', 'mask_sha256')
    native_same = len(reports) == expected and all(
        len({report.get('native_audio', {}).get(key) for report in reports}) == 1
        and all(report.get('native_audio', {}).get(key) for report in reports) for key in native_keys)
    passed = (len(jobs) == expected and all(job['exit_code'] == 0 for job in jobs)
              and len(reports) == expected and all(report['status'] == 'passed' for report in reports) and native_same)
    final = dict(all_sampler_jobs_exit_zero=passed, expected=expected, jobs=jobs, condition_reports=reports,
                 native_silence_identical_across_all_music_and_images=native_same,
                 complete_video_decode_checked=False,
                 no_text_encoder=args.no_text_encoder,
                 old_model_not_trained_for_no_tc=old_no_tc_model, context_policy=context_policy,
                 teacher_source=teacher_source,
                 scope='Pure caption-only offline H3 teacher tokens without any reference-context conditioning or TC/prefix; teacher diagnostic, not music prediction or quality acceptance' if pure_text_teacher else 'Cached offline teacher tail only without TC/prefix, not music prediction or a trained model result' if args.teacher_tail_cache else 'Old checkpoint music-only/no-TC zero-context diagnostic; old model not trained for no-TC, no new-interface success claim' if old_no_tc_model else 'Trained music-only global-bias model deployed without any TC/context/prefix; video integrity and quality acceptance are separate' if args.no_text_encoder else 'Same-length projected teacher, not music prediction or identity-velocity equivalence' if args.projection else 'Teacher hidden oracle, not music prediction or a trained model result' if args.caps else 'Trained shared raw-text adapter, original native music conditions disabled')
    write_new(args.out / 'jobs.json', final)
    print(json.dumps(dict(expected=expected, finished=len(jobs), oracle_conditions_passed=passed,
                          out=str(args.out))), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
