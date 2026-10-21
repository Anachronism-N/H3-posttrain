"""Event-driven cache audit -> real small-fit -> formal raw-token training, managed by caller."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import select
import shlex
import subprocess
import time

from music_scale_generate import SSH, guard_code


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare_contrastive_bundle(root, out):
    import torch
    from music_text_align import read_split, load_targets, load_features
    from music_text_adapter import load_music_text_checkpoint, PREPROCESSING
    from music_text_contrastive import native_statistics
    root, out = Path(root), Path(out)
    if out.exists() or not out.parent.is_dir():
        raise FileExistsError('Bundle requires a new path and existing parent')
    exp = root / 'logs_omnidance/exp/MS'
    split_path = exp / 'music_scale_v4_train1800_split.json'
    init = exp / 'music_text_train1800_raw_neutral_v2/candidate_step1875.pt'
    train, val = read_split(split_path)
    targets, protocols = load_targets([exp / f'music_text_targets_neutral_v1_s{i}' for i in range(4)],
                                      set(train + val), digest(split_path))
    model = load_music_text_checkpoint(init, expected_backend='musicfm')
    features = load_features([exp / f'music_text_features_v1_s{i}.pt' for i in range(2)],
                             set(train + val), digest(split_path), model.backbone)
    mean, std, position_mean, teacher_energy = native_statistics(targets, train, model.adapter.qmax)
    source_init_sha = digest(init)
    snapshot = out.with_name(out.stem + '_init.pt')
    if snapshot.exists():
        raise FileExistsError('Refuse existing bundle checkpoint snapshot')
    with init.open('rb') as source, snapshot.open('xb') as target:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            target.write(block)
    if digest(snapshot) != source_init_sha:
        raise ValueError('Local snapshot differs from audited initial checkpoint')
    state = dict(targets=targets, features=features, protocols=protocols,
                 mean=mean, std=std, position_mean=position_mean, teacher_energy=teacher_energy,
                 metadata=dict(format_version=1, split_sha256=digest(split_path), init_sha256=source_init_sha,
                               local_init=str(snapshot), count=len(targets), train_ids=train, val_ids=val,
                               query_max=model.adapter.qmax, preprocessing=PREPROCESSING,
                               targets_validated_by_original_file_sha=True, weights_frozen=True))
    with out.open('xb') as stream:
        torch.save(state, stream)
    reread = torch.load(out, map_location='cpu', weights_only=True)
    if reread['metadata'] != state['metadata'] or set(reread['targets']) != set(train + val):
        raise ValueError('Local bundle roundtrip failed')
    if not all(torch.equal(value, reread['features'][sid]) for sid, value in features.items()):
        raise ValueError('Feature bundle changed tensors')
    report = dict(complete=True, path=str(out), sha256=digest(out), local_init=str(snapshot),
                  init_sha256=source_init_sha, count=len(targets), train=len(train), val=len(val),
                  teacher_energy=teacher_energy, query_max=model.adapter.qmax,
                  scope='Validated numerical copies of existing inputs, no new training or data split')
    with out.with_suffix('.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(event='inputs_bundled', **report)), flush=True)


def wait_for_target_shards(directories, manager_pid):
    marker = 'finished.json'
    missing = [path for path in directories if not (path / marker).is_file()]
    if not missing:
        return
    pidfd = __import__('os').pidfd_open(manager_pid)
    cmdline = Path('/proc') / str(manager_pid) / 'cmdline'
    if b'music_text_targets_neutral_v1_s' not in cmdline.read_bytes():
        raise ValueError('Declared cache manager PID does not match target task')
    # File-close/move/create events wake the queue. No sleep/polling/repeated cache launches.
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(0x800 | 0x80000)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1')
    try:
        for directory in missing:
            if not directory.is_dir():
                raise FileNotFoundError('Target task has not created its declared directory: ' + str(directory))
            if libc.inotify_add_watch(fd, str(directory).encode(), 0x8 | 0x80 | 0x100) < 0:
                raise OSError(ctypes.get_errno(), 'inotify_add_watch')
        deadline = time.monotonic() + 6 * 3600
        while missing:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Verified cache wait expired; not training completion')
            events, _, _ = select.select([fd, pidfd], [], [], remaining)
            if fd in events:
                while True:
                    try:
                        __import__('os').read(fd, 65536)
                    except BlockingIOError:
                        break
            missing = [path for path in directories if not (path / marker).is_file()]
            if pidfd in events and missing:
                raise RuntimeError('Cache manager exited with missing finished shards')
    finally:
        __import__('os').close(fd)
        __import__('os').close(pidfd)


def prepare_online_bundle_main():
    import os
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from music_text_align import read_split, write_new
    from music_clip_bridge import load_audio
    from music_encoder_adapter import AUDIO_ROOT
    parser = argparse.ArgumentParser(description='Verified teacher labels and raw waveforms for online MusicFM')
    parser.add_argument('--prepare_online_bundle', action='store_true', required=True)
    parser.add_argument('--source_bundle', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix('.json').exists() or not args.out.parent.is_dir():
        parser.error('Require new bundle/report and an existing parent')
    torch.set_num_threads(2)
    report = json.loads(args.source_bundle.with_suffix('.json').read_text())
    source_sha = digest(args.source_bundle)
    if not report['complete'] or source_sha != report['sha256']:
        raise ValueError('Source teacher bundle fails SHA validation')
    source = torch.load(args.source_bundle, map_location='cpu', weights_only=True)
    train, val = read_split(args.split)
    initial_sha = digest(args.init)
    old = source['metadata']
    if (old['split_sha256'] != digest(args.split) or old['init_sha256'] != initial_sha
            or old['train_ids'] != train or old['val_ids'] != val
            or set(source['targets']) != set(train + val)):
        raise ValueError('Original audited teacher bundle does not match initialization/split')
    # The frozen music features must not survive into the online training bundle.
    source.pop('features')
    waveforms, rows = {}, []
    def read(sid):
        path = AUDIO_ROOT / (sid + '.wav')
        waveform, _ = load_audio(sid)
        return sid, waveform, dict(sample_id=sid, source_path=str(path), source_sha256=digest(path),
             waveform_sha256=__import__('hashlib').sha256(waveform.contiguous().numpy().tobytes()).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, (sid, waveform, row) in enumerate(executor.map(read, train + val), 1):
            waveforms[sid] = waveform
            rows.append(row)
            if index % 100 == 0:
                print(json.dumps(dict(event='online_waveform_cache', count=index, expected=len(train + val))), flush=True)
    metadata = dict(format_version=1, split_sha256=digest(args.split), init_sha256=initial_sha,
                    train_ids=train, val_ids=val, source_teacher_bundle_sha256=source_sha,
                    query_max=old['query_max'], student_feature_cache_used=False,
                    waveform_loader='music_clip_bridge.load_audio', waveform_shape=[2, 162000],
                    waveform_dtype='float32', waveform_rows=rows,
                    loader_source_sha256=digest(Path(__file__).with_name('music_clip_bridge.py')))
    bundle = dict(metadata=metadata, targets=source['targets'], waveforms=waveforms,
                  mean=source['mean'], std=source['std'], position_mean=source['position_mean'],
                  teacher_energy=source['teacher_energy'], protocols=source['protocols'])
    write_new(args.out, bundle, binary=True)
    result = dict(complete=True, path=str(args.out), sha256=digest(args.out), count=len(waveforms),
                  teacher_source_sha256=source_sha, init_sha256=initial_sha,
                  student_feature_cache_used=False, waveform_rows=rows)
    write_new(args.out.with_suffix('.json'), result)
    print(json.dumps(dict(event='online_bundle_ready', complete=True, path=str(args.out), sha256=result['sha256'])), flush=True)


def tensor_digest(value):
    import torch
    cpu = value.detach().cpu().contiguous()
    return hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def verified_bundle(path):
    import torch
    path = Path(path)
    report = json.loads(path.with_suffix('.json').read_text())
    actual = digest(path)
    if report.get('complete') is not True or report.get('sha256') != actual:
        raise ValueError('Bundle fails its completed report/content SHA gate')
    return torch.load(path, map_location='cpu', weights_only=True), actual, report


def pure_waveforms(source, report, train, val, split_sha):
    import torch
    metadata = source['metadata']
    expected = set(train + val)
    if (metadata.get('split_sha256') != split_sha or metadata.get('train_ids') != train
            or metadata.get('val_ids') != val or metadata.get('student_feature_cache_used') is not False
            or set(source.get('waveforms', {})) != expected
            or metadata.get('waveform_loader') != 'music_clip_bridge.load_audio'
            or metadata.get('waveform_shape') != [2, 162000]
            or metadata.get('waveform_dtype') != 'float32'
            or metadata.get('loader_source_sha256') != digest(Path(__file__).with_name('music_clip_bridge.py'))):
        raise ValueError('Source waveforms fail the exact full-split/32k-loader identity gate')
    rows = metadata.get('waveform_rows', [])
    if (rows != report.get('waveform_rows') or len(rows) != len(expected)
            or {row.get('sample_id') for row in rows} != expected):
        raise ValueError('Source waveform provenance must cover all IDs exactly once')
    for row in rows:
        sid = row['sample_id']
        value = source['waveforms'][sid]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.shape != (2, 162000) or not torch.isfinite(value).all()
                or tensor_digest(value) != row.get('waveform_sha256')
                or not isinstance(row.get('source_path'), str)
                or len(row.get('source_sha256', '')) != 64):
            raise ValueError('Source waveform tensor/provenance changed: ' + sid)
    return source['waveforms'], rows


def verified_pure_teacher_release(caps_path, teacher_release_report, split_sha):
    caps_path, teacher_release_report = Path(caps_path), Path(teacher_release_report)
    release_sha = digest(teacher_release_report)
    release = json.loads(teacher_release_report.read_text())
    caps_sha = digest(caps_path)
    if (release.get('complete') is not True or release.get('kind') != 'pure_text_h3_raw_tokens'
            or release.get('caps', {}).get('sha256') != caps_sha
            or Path(release.get('caps', {}).get('path', '')).resolve() != caps_path.resolve()
            or release.get('split', {}).get('sha256') != split_sha
            or digest(release['split']['path']) != split_sha):
        raise ValueError('Pure teacher release does not bind the specified actual captions/split')
    sources = release.get('source_files', {})
    for key, name in (('cache', 'music_text_cache.py'), ('text_embed', 'text_embed.py')):
        current_path = Path(__file__).with_name(name)
        identity = sources.get(key, {})
        if (Path(identity.get('path', '')).resolve() != current_path.resolve()
                or identity.get('sha256') != digest(current_path)):
            raise ValueError('Released pure teacher source code content changed: ' + key)
    caps = json.loads(caps_path.read_text())
    if digest(teacher_release_report) != release_sha or digest(caps_path) != caps_sha:
        raise ValueError('Pure teacher release/captions changed during identity validation')
    return dict(teacher_release=dict(path=str(teacher_release_report.resolve()), sha256=release_sha,
                                    content=release),
                caps=dict(path=str(caps_path.resolve()), sha256=caps_sha, content=caps))


def load_pure_targets(directories, train, val, split_sha, caps_path, teacher_release_report, read_root=None):
    import torch
    from text_embed import raw_tensor_sha256, load_pure_text_tokenizer
    from music_text_cache import pure_text_inputs, pure_token_length_report, target_prompt
    expected = set(train + val)
    artifacts = verified_pure_teacher_release(caps_path, teacher_release_report, split_sha)
    release = artifacts['teacher_release']['content']
    caps, caps_sha = artifacts['caps']['content'], artifacts['caps']['sha256']
    if (len(directories) != 6 or len(train) != 1800 or len(val) != 200 or len(expected) != 2000
            or release.get('teacher_expected_train') != len(train)
            or release.get('teacher_expected_val') != len(val) or set(caps) != expected):
        raise ValueError('Require six completed shards and the exact released train1800/val200 captions')
    targets, protocols, provenances = {}, [], []
    shared, shard_numbers, parsed_token_ids = None, set(), None
    if len(set(Path(path).resolve() for path in directories)) != len(directories):
        raise ValueError('Duplicate pure target-cache directory')
    for directory in map(Path, directories):
        plan_path, finished_path = directory / 'plan.json', directory / 'finished.json'
        plan = json.loads(plan_path.read_text())
        finished = json.loads(finished_path.read_text())
        plan_sha = digest(plan_path)
        finished_sha = digest(finished_path)
        read_directory = Path(read_root) / directory.name if read_root is not None else directory
        if read_root is not None:
            if (not read_directory.is_dir()
                    or digest(read_directory / 'plan.json') != plan_sha
                    or digest(read_directory / 'finished.json') != finished_sha):
                raise ValueError('Local pure teacher mirror plan/completion differs from canonical source')
        shard, nshard = plan.get('shard'), plan.get('nshard')
        if (type(shard) is not int or type(nshard) is not int or nshard != len(directories)
                or not 0 <= shard < nshard or shard in shard_numbers):
            raise ValueError('Require every unique complete declared pure target shard')
        shard_numbers.add(shard)
        ids = (train + val)[shard::nshard]
        if (finished.get('complete') is not True or finished.get('actual_lm_forward') is not True
                or finished.get('image_io') is not False or finished.get('exact_expected_full_shard_ids') is not True
                or finished.get('source') != plan or finished.get('source_protocol_sha256') != plan_sha
                or plan.get('split_sha256') != split_sha or plan.get('caps_sha256') != caps_sha
                or plan.get('train_ids') != train or plan.get('val_ids') != val
                or plan.get('original_full_train') != len(train) or plan.get('original_full_val') != len(val)
                or plan.get('token_check_train_count') != len(train) or plan.get('token_check_val_count') != len(val)
                or plan.get('caption_fields') != ['dims.genre', 'motion_original']
                or plan.get('ids') != ids or plan.get('expected_ids') != ids
                or finished.get('expected_ids') != ids or finished.get('expected') != len(ids)
                or plan.get('kind') != 'pure_text_h3_raw_tokens' or plan.get('text_only') is not True
                or plan.get('image_conditioned') is not False or plan.get('prefix_length') != 0
                or plan.get('interface') != 'raw_h3_last_hidden_5120_pure_text'
                or plan.get('final_norm') != 'Identity' or plan.get('special_template') is not False
                or plan.get('normalization') != 'none' or plan.get('pooling') != 'none'
                or plan.get('dtype') != 'torch.bfloat16' or plan.get('hidden_width') != 5120
                or plan.get('full_token_check') is not True or plan.get('validation_does_not_set_query_max') is not True
                or plan.get('tokenizer_padding_side') != 'right' or plan.get('add_special_tokens') is not False
                or plan.get('truncation') is not False):
            raise ValueError('Pure teacher shard fails complete/full-ID/no-image/no-prefix/native-raw protocol gate')
        ignored = {'shard', 'ids', 'expected_ids', 'tokenizer_root'}
        signature = {key: value for key, value in plan.items() if key not in ignored}
        if shared is not None and signature != shared:
            raise ValueError('Pure teacher shards disagree on full split/captions/loaded model/tokenizer/source hash protocol')
        shared = signature
        model_hashes = plan.get('model_weight_tensor_sha256')
        if (not isinstance(model_hashes, dict) or len(model_hashes) != 551
                or len(model_hashes) != plan.get('loaded_tensor_count')
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in '0123456789abcdef' for char in value)
                       for value in model_hashes.values())
                or hashlib.sha256(json.dumps(model_hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    != plan.get('model_weight_content_sha256')
                or plan.get('model_weight_hash_scope') != 'actual_loaded_H3_50_layer_language_model_state_BF16'
                or plan.get('retained_layers') != 50 or plan.get('strict_assign') is not True):
            raise ValueError('Teacher actual-loaded-model content/tensor hashes are absent or inconsistent')
        transform = plan.get('teacher_transform')
        if transform not in ('none', 'gender_pronouns_to_dancer_v1'):
            raise ValueError('Unsupported pure teacher caption transform')
        _, _, prompts, replacements = pure_text_inputs(caps, dict(train=train, val=val),
                                                       transform == 'gender_pronouns_to_dancer_v1')
        tokenizer_hashes = plan.get('tokenizer_metadata_file_sha256', {})
        if (not isinstance(tokenizer_hashes, dict) or 'tokenizer.json' not in tokenizer_hashes
                or 'tokenizer_config.json' not in tokenizer_hashes
                or any(Path(name).name != name or digest(Path(plan['tokenizer_root']) / name) != value
                       for name, value in tokenizer_hashes.items())):
            raise ValueError('Pure teacher tokenizer source content changed')
        if parsed_token_ids is None:
            tokenizer = load_pure_text_tokenizer(Path(plan['tokenizer_root']).parent)
            parsed_report, parsed_token_ids = pure_token_length_report(tokenizer, prompts, train, val)
        if any(plan.get(key) != value for key, value in parsed_report.items()):
            raise ValueError('Pure teacher full source-token audit differs from actual specified-caption tokenization')
        lengths, ids_hashes = plan.get('lengths', {}), plan.get('token_ids_sha256', {})
        if (set(lengths) != expected or set(ids_hashes) != expected
                or any(type(length) is not int or length < 1 for length in lengths.values())
                or plan.get('qmax') != max(lengths[sid] for sid in train)
                or plan.get('training_query_max') != plan['qmax']):
            raise ValueError('Pure teacher full token-length/source-ID audit is incomplete')
        rows = finished.get('rows', [])
        if ([row.get('sample_id') for row in rows] != ids
                or {path.stem for path in read_directory.glob('*.pt')} != set(ids)):
            raise ValueError('Pure teacher shard records/files do not match exact shard IDs')
        for row in rows:
            sid = row['sample_id']
            path = directory / (sid + '.pt')
            read_path = read_directory / path.name
            if sid in targets or Path(row['path']).resolve() != path.resolve() or digest(read_path) != row['file_sha256']:
                raise ValueError('Duplicate or altered pure teacher file: ' + sid)
            data = torch.load(read_path, map_location='cpu', weights_only=True)
            if not isinstance(data, dict) or not isinstance(data.get('metadata'), dict):
                raise ValueError('Pure teacher record/metadata must be objects: ' + sid)
            metadata = data.get('metadata', {})
            tail, token_ids = data.get('tail'), data.get('source_token_ids')
            length = lengths[sid]
            if (set(data) != {'metadata', 'tail', 'source_token_ids', 'teacher_length'}
                    or not isinstance(tail, torch.Tensor) or tail.dtype != torch.bfloat16
                    or tail.shape != (length, 5120) or not torch.isfinite(tail).all()
                    or not isinstance(token_ids, torch.Tensor) or token_ids.dtype != torch.int64
                    or token_ids.shape != (length,) or torch.any(token_ids < 0)
                    or type(data['teacher_length']) is not int or data['teacher_length'] != length
                    or not torch.equal(token_ids, parsed_token_ids[sid]) or row.get('tail_length') != length
                    or row.get('teacher_length') != length or row.get('prefix_length') != 0
                    or metadata.get('partition') != ('train' if sid in train else 'val')
                    or metadata.get('sample_id') != sid or metadata.get('kind') != plan['kind']
                    or metadata.get('text_only') is not True or metadata.get('image_conditioned') is not False
                    or metadata.get('prefix_length') != 0 or metadata.get('tail_length') != length
                    or metadata.get('teacher_length') != length
                    or metadata.get('source_protocol_sha256') != plan_sha
                    or Path(metadata.get('source_protocol_path', '')).resolve() != plan_path.resolve()
                    or metadata.get('caps_sha256') != plan['caps_sha256'] or metadata.get('split_sha256') != split_sha
                    or metadata.get('model_weight_content_sha256') != plan['model_weight_content_sha256']
                    or metadata.get('tail_sha256') != raw_tensor_sha256(tail)
                    or metadata.get('source_token_ids_sha256') != raw_tensor_sha256(token_ids)
                    or ids_hashes[sid] != raw_tensor_sha256(token_ids)
                    or metadata.get('prompt') != prompts[sid]
                    or metadata.get('original_prompt') != target_prompt(caps[sid])
                    or metadata.get('teacher_replacements') != replacements[sid]
                    or metadata.get('token_ids_sha256') != ids_hashes[sid]
                    or metadata.get('tokenizer_metadata_file_sha256') != tokenizer_hashes
                    or metadata.get('final_norm') != 'Identity' or metadata.get('normalization') != 'none'
                    or metadata.get('pooling') != 'none'
                    or hashlib.sha256(metadata.get('prompt', '').encode()).hexdigest() != metadata.get('caption_sha256')
                    or any('context' in key or 'image' in key and key != 'image_conditioned' for key in metadata)):
                raise ValueError('Pure raw target/native source-ID/length/caption/source/prefix gate failed: ' + sid)
            targets[sid] = data
        protocols.append(plan)
        provenances.append(dict(directory=str(directory), plan_sha256=plan_sha,
            finished_sha256=finished_sha, read_directory=str(read_directory),
            local_mirror_used=read_root is not None, mirror_file_sha256_verified=True,
            count=len(rows), shard=shard, nshard=nshard,
            model_weight_content_sha256=plan['model_weight_content_sha256'], caps_sha256=plan['caps_sha256'],
            split_sha256=split_sha, teacher_release_report=dict(path=artifacts['teacher_release']['path'],
                sha256=artifacts['teacher_release']['sha256']), source_files=release['source_files'],
            caps=dict(path=artifacts['caps']['path'], sha256=caps_sha),
            actual_specified_caption_token_ids_exact=True,
            rows=rows, exact_expected_full_shard_ids=True, actual_lm_forward=True,
            text_only=True, image_conditioned=False, prefix_length=0))
    if set(targets) != expected or shard_numbers != set(range(len(directories))):
        raise ValueError('Pure target cache must cover each full split ID once with every complete shard')
    if verified_pure_teacher_release(caps_path, teacher_release_report, split_sha) != artifacts:
        raise ValueError('Pure release/captions/source identity changed while validating shards')
    return targets, protocols, provenances


def prepare_pure_text_bundle_main():
    import torch
    from music_text_align import read_split, write_new
    from music_text_contrastive import native_statistics
    parser = argparse.ArgumentParser(description='Pure text-only H3 teacher labels plus audited raw 32k waveforms')
    parser.add_argument('--target_read_root', type=Path, help='Optional same-SHA local mirror; canonical target paths and metadata remain authoritative')
    parser.add_argument('--prepare_pure_text_bundle', action='store_true', required=True)
    parser.add_argument('--source_bundle', type=Path, required=True)
    parser.add_argument('--targets', type=Path, nargs='+', required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--caps', type=Path, required=True)
    parser.add_argument('--teacher_release_report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix('.json').exists() or not args.out.parent.is_dir():
        parser.error('Require new bundle/report paths and existing parent')
    torch.set_num_threads(2)
    train, val = read_split(args.split)
    if len(train) != 1800 or len(val) != 200:
        raise ValueError('Formal pure teacher bundle requires the complete train1800/val200 split')
    split_sha = digest(args.split)
    artifacts = verified_pure_teacher_release(args.caps, args.teacher_release_report, split_sha)
    source, source_sha, source_report = verified_bundle(args.source_bundle)
    waveforms, waveform_rows = pure_waveforms(source, source_report, train, val, split_sha)
    # Never inherit old image-conditioned labels, cached features, statistics or init bindings.
    del source
    targets, protocols, provenances = load_pure_targets(args.targets, train, val, split_sha,
                                                       args.caps, args.teacher_release_report,
                                                       read_root=args.target_read_root)
    qmax = max(len(targets[sid]['tail']) for sid in train)
    mean, std, position_mean, energy = native_statistics(targets, train, qmax)
    overflow = [dict(sample_id=sid, teacher_length=len(targets[sid]['tail']),
                     tokens_above_query_max=len(targets[sid]['tail']) - qmax)
                for sid in val if len(targets[sid]['tail']) > qmax]
    metadata = dict(format_version=2, protocol='musicfm_pure_text_h3_raw_tokens_v1',
        teacher_kind='pure_text_h3_raw_tokens',
        teacher_image_conditioned=False, teacher_prefix_length=0,
        student_feature_cache_used=False, train_ids=train, val_ids=val, query_max=qmax,
        split_sha256=split_sha, waveform_loader='music_clip_bridge.load_audio',
        waveform_shape=[2, 162000], waveform_dtype='float32', waveform_sample_rate=32000,
        waveform_rows=waveform_rows,
        loader_source_sha256=digest(Path(__file__).with_name('music_clip_bridge.py')),
        source_waveform_identity=dict(bundle=str(args.source_bundle), bundle_sha256=source_sha,
            report_sha256=digest(args.source_bundle.with_suffix('.json')),
            waveform_rows_sha256=hashlib.sha256(json.dumps(waveform_rows, sort_keys=True).encode()).hexdigest(),
            old_targets_reused=False, old_statistics_reused=False, old_init_binding_reused=False),
        pure_cache_provenances=provenances,
        teacher_release_report=dict(path=artifacts['teacher_release']['path'],
                                   sha256=artifacts['teacher_release']['sha256']),
        caps=dict(path=artifacts['caps']['path'], sha256=artifacts['caps']['sha256']),
        initialization_binding='external_pure_text_init_binding_manifest',
        validation_does_not_set_query_max=True, validation_overflow=overflow,
        statistics_source='all_valid_tokens_of_full_train_only_no_prefix_no_padding')
    bundle = dict(metadata=metadata, targets=targets, waveforms=waveforms, mean=mean, std=std,
                  position_mean=position_mean, teacher_energy=energy, protocols=protocols, artifacts=artifacts)
    write_new(args.out, bundle, binary=True)
    reread = torch.load(args.out, map_location='cpu', weights_only=True)
    if (reread['metadata'] != metadata or reread['artifacts'] != artifacts
            or set(reread['targets']) != set(train + val)
            or not all(torch.equal(value, reread['waveforms'][sid]) for sid, value in waveforms.items())
            or not all(torch.equal(value['tail'], reread['targets'][sid]['tail']) for sid, value in targets.items())):
        raise ValueError('Pure bundle serialization changed metadata or numerical inputs')
    result = dict(complete=True, path=str(args.out), sha256=digest(args.out), count=len(targets),
        train=len(train), val=len(val), query_max=qmax, teacher_energy=energy,
        teacher_kind=metadata['teacher_kind'], teacher_image_conditioned=False,
        student_feature_cache_used=False, source_waveform_bundle_sha256=source_sha,
        pure_cache_provenances=provenances, validation_overflow=overflow,
        teacher_release_report=metadata['teacher_release_report'], caps=metadata['caps'],
        source_files=artifacts['teacher_release']['content']['source_files'],
        waveform_rows=waveform_rows, optimizer_updates=0,
        scope='Verified pure teacher/raw-waveform inputs; no initialization or training acceptance')
    write_new(args.out.with_suffix('.json'), result)
    print(json.dumps(dict(event='pure_text_bundle_ready', path=str(args.out), sha256=result['sha256'],
                         query_max=qmax, validation_above_query_max=len(overflow))), flush=True)


def pure_init_probe(model, bundle, ids):
    import torch
    sums = torch.zeros(5120, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    n = 0
    rows = []
    with torch.no_grad():
        for sid in ids:
            device = model.adapter.raw_output.weight.device
            raw, logs = model(bundle['waveforms'][sid][None].to(device))
            indices = torch.linspace(0, model.adapter.qmax - 1, min(32, model.adapter.qmax), device=device).round().long()
            selected = raw[0].index_select(0, indices).double().cpu()
            if not torch.isfinite(selected).all() or not torch.isfinite(logs).all():
                raise ValueError('Nonfinite actual full-music initialization probe')
            sums += selected.sum(0)
            squares += selected.square().sum(0)
            n += len(selected)
            rows.append(dict(sample_id=sid, query_indices=indices.tolist(),
                             waveform_sha256=tensor_digest(bundle['waveforms'][sid])))
    mean = sums / n
    energy = float(squares.mean() / n)
    centered = float((squares / n - mean.square()).clamp_min(0).mean())
    return dict(mean=mean, raw_energy=energy, raw_rms=energy ** 0.5,
                centered_energy=centered, centered_rms=centered ** 0.5,
                sampled_tokens=n, samples=len(ids), rows=rows)


def calibrate_pure_head(model, bundle, probe_ids):
    import math
    import torch
    teacher_mean = bundle['mean'].double()
    teacher_energy = float(bundle['teacher_energy'])
    centered_energy = teacher_energy - float(teacher_mean.square().mean())
    if (teacher_mean.shape != (5120,) or not torch.isfinite(teacher_mean).all()
            or not math.isfinite(centered_energy) or centered_energy <= 1e-12):
        raise ValueError('Need positive finite full-train teacher centered energy')
    before = pure_init_probe(model, bundle, probe_ids)
    eps = 1e-12
    if before['centered_rms'] <= eps:
        raise ValueError('Degenerate random output probe; refuse nonfinite or huge head calibration')
    scale = math.sqrt(centered_energy) / before['centered_rms']
    if not math.isfinite(scale) or not 1e-6 <= scale <= 1e6:
        raise ValueError('Head calibration scale outside finite bounded range')
    head = model.adapter.raw_output
    with torch.no_grad():
        head.weight.mul_(scale)
        bias = teacher_mean + (head.bias.detach().double().cpu() - before['mean']) * scale
        head.bias.copy_(bias.to(device=head.bias.device, dtype=torch.float32))
    after = pure_init_probe(model, bundle, probe_ids)
    mean_error = float((after['mean'] - teacher_mean).square().mean().sqrt())
    rms_error = abs(after['centered_rms'] / math.sqrt(centered_energy) - 1)
    if mean_error > 1e-4 * max(1., math.sqrt(teacher_energy)) or rms_error > 1e-4:
        raise ValueError('Actual after-calibration probe does not match requested mean/global centered RMS')
    def report_probe(probe):
        return dict(probe, mean=probe['mean'].tolist(), mean_sha256=tensor_digest(probe['mean']))
    return dict(method='W*=scale;b=teacher_mean+(b-prediction_mean)*scale',
        scale=scale, epsilon=eps, scale_bounds=[1e-6, 1e6], teacher_raw_energy=teacher_energy,
        teacher_centered_energy=centered_energy, teacher_centered_rms=math.sqrt(centered_energy),
        teacher_mean=teacher_mean.tolist(), teacher_mean_sha256=tensor_digest(bundle['mean']),
        teacher_statistics_source='full_train_all_valid_raw_teacher_tokens',
        probe_source='first_32_train_samples_full_music_forward_uniform_up_to32_full_qmax_queries',
        before=report_probe(before), after=report_probe(after), after_mean_error_rms=mean_error,
        after_centered_rms_relative_error=rms_error, validation_used=False,
        covariance_matched=False, teacher_directions_matched=False, fixed_basis_installed=False,
        raw_output_weight_and_bias_trainable=True, optimizer_updates=0, backward_calls=0)


def resize_pure_queries(model, qmax):
    import torch
    from music_text_adapter import MusicTextAdapter
    old = model.adapter
    before = dict(old.state_dict())
    if old.qmax == qmax:
        return dict(old_query_max=old.qmax, new_query_max=qmax, changed_state=[], copied_rows=qmax,
                    new_rows=0, common_state_exact=True)
    torch.manual_seed(42)
    adapter = MusicTextAdapter(qmax, width=old.width, heads=old.heads,
                              context_policy=old.context_policy, head_normalization=old.head_normalization)
    adapter.to(device=old.raw_output.weight.device, dtype=old.raw_output.weight.dtype)
    if old.output_basis is not None:
        raise ValueError('Pure control must not install a fixed output basis')
    state = dict(before)
    query = adapter.query_positions.weight.detach().clone()
    copied = min(old.qmax, qmax)
    query[:copied].copy_(old.query_positions.weight[:copied])
    state['query_positions.weight'] = query
    adapter.load_state_dict(state, strict=True)
    common = [name for name in state if name != 'query_positions.weight']
    if not all(torch.equal(before[name], adapter.state_dict()[name]) for name in common):
        raise ValueError('Changing qmax changed other control state')
    model.adapter = adapter
    return dict(old_query_max=old.qmax, new_query_max=qmax, changed_state=['adapter.query_positions.weight'],
        copied_rows=copied, new_rows=max(0, qmax - old.qmax), new_row_seed=42,
        copied_rows_exact=torch.equal(query[:copied], before['query_positions.weight'][:copied]),
        common_state_exact=True, strict_state_loading=True,
        common_state_names=common, new_module='query_positions_only')


def prepare_pure_text_init_main():
    import torch
    from music_encoder_adapter import DenseMusicEncoder
    from music_text_adapter import (MusicTextAdapter, MusicToH3Text, save_music_text_checkpoint,
        load_music_text_checkpoint, convert_music_text_to_no_text_encoder)
    from music_text_contrastive import _state_hashes
    from music_text_align import write_new
    parser = argparse.ArgumentParser(description='Complete public-MusicFM/free raw-output initializer, without optimizer updates')
    parser.add_argument('--prepare_pure_text_init', action='store_true', required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--backbone_model_path', type=Path, default=Path('/opt/models/MusicFM-MSD'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source_checkpoint', type=Path, help='Optional completed legacy C300 control; preserve its weights and old head LayerNorm')
    parser.add_argument('--width', type=int, help='New model defaults to512; a C300 control must match its old width')
    parser.add_argument('--head_normalization', choices=['identity', 'layer_norm'], default='identity')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Full actual MusicFM probe device; caller schedules CUDA explicitly')
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix('.json').exists() or not args.out.parent.is_dir():
        parser.error('Require new complete checkpoint/report paths and existing parent')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    bundle, bundle_sha, _ = verified_bundle(args.bundle)
    metadata = bundle['metadata']
    if (metadata.get('format_version') != 2
            or metadata.get('protocol') != 'musicfm_pure_text_h3_raw_tokens_v1'
            or metadata.get('teacher_kind') != 'pure_text_h3_raw_tokens'
            or metadata.get('teacher_image_conditioned') is not False
            or metadata.get('student_feature_cache_used') is not False
            or 'init_sha256' in metadata or 'features' in bundle):
        raise ValueError('Initializer requires an unbound pure-text-only raw-waveform bundle')
    train, qmax = metadata['train_ids'], metadata['query_max']
    if qmax != max(len(bundle['targets'][sid]['tail']) for sid in train):
        raise ValueError('Initializer qmax must come from all train teacher labels only')
    backbone_path = str(args.backbone_model_path)
    weights_path = args.backbone_model_path / 'pretrained_msd.pt'
    stats_path = args.backbone_model_path / 'msd_stats.json'
    provenance = dict(backbone_model_path=backbone_path, public_weights_sha256=digest(weights_path),
                      public_stats_sha256=digest(stats_path), bundle_sha256=bundle_sha,
                      split_sha256=metadata['split_sha256'], teacher_kind=metadata['teacher_kind'])
    backbone = DenseMusicEncoder('musicfm', backbone_path)
    calibration, conversion, resize = None, None, None
    if args.source_checkpoint:
        source_sha = digest(args.source_checkpoint)
        finished_path = args.source_checkpoint.parent / 'finished.json'
        finished = json.loads(finished_path.read_text())
        if (args.source_checkpoint.name != 'last.pt' or finished.get('complete') is not True
                or finished.get('steps') != 300 or finished.get('protocol', {}).get('steps') != 300
                or finished.get('protocol', {}).get('online_bundle_sha256')
                   != metadata['source_waveform_identity']['bundle_sha256']
                or finished.get('checkpoint_sha256') != source_sha):
            raise ValueError('Control requires the exact completed C300 last.pt checkpoint')
        model = load_music_text_checkpoint(args.source_checkpoint, backbone=backbone, expected_backend='musicfm')
        if ((args.width is not None and args.width != model.adapter.width)
                or model.adapter.width != 256 or model.adapter.context_policy != 'tc_prefix_context'
                or model.adapter.head_normalization != 'layer_norm' or model.adapter.output_basis is not None):
            raise ValueError('C300 control must preserve its old width/LayerNorm/free raw-output weights')
        model.configure_online_training(True).to(args.device).eval()
        source_states = _state_hashes(model)
        with torch.no_grad():
            features, lengths = model.dense_features(bundle['waveforms'][train[0]][None].to(args.device))
        conversion = convert_music_text_to_no_text_encoder(model, features, lengths)
        conversion.update(probe_sample_id=train[0], source_checkpoint_sha256=source_sha)
        resize = resize_pure_queries(model, qmax)
        after = _state_hashes(model)
        changed = [name for name in source_states if name in after and source_states[name] != after[name]]
        if changed != (['adapter.query_positions.weight'] if resize['changed_state'] else []):
            raise ValueError('C300 common state changed beyond the declared query resizing')
        provenance.update(kind='c300_derived_no_tc_layer_norm_control', source_checkpoint=str(args.source_checkpoint),
            source_checkpoint_sha256=source_sha, source_finished_sha256=digest(finished_path),
            source_state_sha256=source_states, actual_shared_state_unchanged=True,
            init_output_scale_calibrated=False, head_normalization_preserved='layer_norm')
    else:
        if args.head_normalization != 'identity':
            raise ValueError('New pure public-MusicFM initializer requires an Identity output head')
        torch.manual_seed(42)
        adapter = MusicTextAdapter(qmax, width=args.width if args.width is not None else 512,
            context_policy='music_only_global_bias', head_normalization=args.head_normalization)
        model = MusicToH3Text(backbone, adapter, freeze_backbone=False).configure_online_training(True).to(args.device).eval()
        source_states = _state_hashes(model)
        calibration = calibrate_pure_head(model, bundle, train[:32])
        after = _state_hashes(model)
        changed = sorted(name for name in source_states if source_states[name] != after[name])
        if changed != ['adapter.raw_output.bias', 'adapter.raw_output.weight']:
            raise ValueError('Random-head calibration must change only the free raw head W/b')
        provenance.update(kind='public_musicfm_random_full_adapter_global_rms_calibrated',
            source_checkpoint=None, seed=42, source_state_sha256=source_states,
            init_output_scale_calibrated=True, changed_initialization_state=changed,
            old_checkpoint_weights_used=False)
    model.configure_online_training(True).eval()
    if (model.adapter.output_space_metadata() != {'mode': 'none'}
            or model.adapter.context_policy != 'music_only_global_bias'
            or not all(p.requires_grad for p in model.adapter.parameters())
            or any(p.grad is not None for p in model.parameters())):
        raise ValueError('Pure initialization must stay complete, music-only, free-head, without backward calls')
    state_hashes = _state_hashes(model)
    save_music_text_checkpoint(args.out, model)
    saved = torch.load(args.out, map_location='cpu', weights_only=True)
    if set(saved) != {'metadata', 'model'} or set(saved['model']) != set(state_hashes):
        raise ValueError('Initializer must contain complete wrapper state')
    if not all(tensor_digest(value) == state_hashes[name] for name, value in saved['model'].items()):
        raise ValueError('Complete initializer weight/buffer roundtrip differs')
    result = dict(complete=True, path=str(args.out), sha256=digest(args.out), init_sha256=digest(args.out),
        bundle_sha256=bundle_sha, query_max=qmax, width=model.adapter.width,
        head_normalization=model.adapter.head_normalization, output_space={'mode': 'none'},
        context_policy='music_only_global_bias', provenance=provenance, calibration=calibration,
        context_conversion=conversion, query_resize=resize, checkpoint_metadata=saved['metadata'],
        complete_weight_buffer_roundtrip_exact=True, state_sha256=state_hashes,
        source_sha256=digest(Path(__file__)), adapter_source_sha256=digest(Path(__file__).with_name('music_text_adapter.py')),
        optimizer_updates=0, backward_calls=0, teacher_statistics_train_only=True,
        no_fixed_projection_or_direction_claim=True, initialization_probe_device=args.device,
        scope='Complete trainable initializer, not fitted model or generation acceptance')
    write_new(args.out.with_suffix('.json'), result)
    print(json.dumps(dict(event='pure_text_init_ready', path=str(args.out), sha256=result['sha256'],
                         kind=provenance['kind'], optimizer_updates=0)), flush=True)


def bind_pure_init_manifest_main():
    from music_text_align import write_new
    parser = argparse.ArgumentParser(description='Bind one initializer to immutable pure teacher bundle without copying it')
    parser.add_argument('--bind_pure_init_manifest', action='store_true', required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--init', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require a new manifest path with existing parent')
    bundle, bundle_sha, _ = verified_bundle(args.bundle)
    init, init_sha, init_report = verified_bundle(args.init)
    metadata = bundle['metadata']
    if (metadata.get('teacher_kind') != 'pure_text_h3_raw_tokens'
            or metadata.get('teacher_image_conditioned') is not False
            or init_report.get('bundle_sha256') != bundle_sha
            or init_report.get('init_sha256') != init_sha
            or init['metadata'].get('query_max') != metadata['query_max']
            or init['metadata'].get('context_policy') != 'music_only_global_bias'
            or init['metadata'].get('output_space') != {'mode': 'none'}
            or init_report.get('optimizer_updates') != 0 or init_report.get('backward_calls') != 0):
        raise ValueError('Pure initializer report/checkpoint must bind this exact unmodified bundle')
    manifest = dict(format_version=1, kind='pure_text_init_binding', complete=True,
        bundle_path=str(args.bundle.resolve()), bundle_sha256=bundle_sha,
        init_path=str(args.init.resolve()), init_sha256=init_sha,
        init_report_path=str(args.init.with_suffix('.json').resolve()),
        init_report_sha256=digest(args.init.with_suffix('.json')),
        split_sha256=metadata['split_sha256'], query_max=metadata['query_max'],
        teacher_kind=metadata['teacher_kind'], teacher_image_conditioned=False,
        context_policy='music_only_global_bias', head_normalization=init['metadata']['head_normalization'],
        initialization_kind=init_report['provenance']['kind'], optimizer_updates=0,
        student_inputs='waveform_only', no_fixed_output_projection=True,
        bundle_copied=False, source_sha256=digest(Path(__file__)))
    write_new(args.out, manifest)
    print(json.dumps(dict(event='pure_text_init_bound', **manifest)), flush=True)


def prepare_direct_training_manifest(args, *, expected_train=6000, expected_val=500, audit_case_cache=False):
    """Stream complete cached teacher tails; prefix tensors are never deserialized."""
    import math
    import torch
    from transformers import AutoTokenizer
    from music_text_align import parallel_file_sha256, checked_file_sha256
    from music_scale_data import (DIRECT_PROTOCOL, read_json, source_record,
        assert_sources_unchanged, atomic_json, check_outputs, id_list)

    def cached_tensor_sha(value):
        value = value.detach().cpu().contiguous()
        h = hashlib.sha256()
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    out = check_outputs([args.out])[0]
    selection = Path(args.selection).resolve(strict=True)
    report_name = 'source' if audit_case_cache else 'selection_report'
    source_paths = {name: selection / (name + '.json')
                    for name in (report_name, 'split', 'caps', 'references')}
    sources = {name: source_record(path) for name, path in source_paths.items()}
    sources['pipeline_code'] = source_record(__file__)
    sources['file_hash_code'] = source_record(Path(__file__).with_name('music_text_align.py'))
    report, split, caps, references = [read_json(source_paths[name])
        for name in (report_name, 'split', 'caps', 'references')]
    train, val = split.get('train'), split.get('val')
    train_set = set() if audit_case_cache and train == [] else id_list(train, 'direct train')
    val_set = id_list(val, 'direct val')
    expected = train_set | val_set
    if audit_case_cache:
        if train or len(val) != 3 or split.get('purpose') != 'fixed_generation_holdout_only':
            raise ValueError('Case-cache audit is exclusively zero-train/three fixed generation holdouts')
        nshards = 3
    else:
        if (len(train) != expected_train or len(val) != expected_val
                or report.get('complete') is not True
                or report.get('train_count') != len(train) or report.get('val_count') != len(val)):
            raise ValueError('Require completed exact direct6k/val500 selection')
        for name in ('split', 'caps', 'references'):
            identity = report.get('outputs', {}).get(name, {})
            if (identity.get('sha256') != sources[name]['sha256']
                    or Path(identity.get('path', '')).resolve() != source_paths[name]):
                raise ValueError('Selection output SHA/path mismatch: ' + name)
        nshards = 6
    if (train_set & val_set or report.get('protocol') != DIRECT_PROTOCOL
            or split.get('protocol') != DIRECT_PROTOCOL or references.get('protocol') != DIRECT_PROTOCOL
            or set(caps) != expected or set(references.get('records', {})) != expected
            or references.get('native_generation', {}).get('num_frames') != 124):
        raise ValueError('Selection protocol/native-124/disjoint ID coverage mismatch')
    groups = ({sid: references['records'][sid]['source_group'] for sid in expected}
              if audit_case_cache else split.get('source_group_ids', {}))
    if (set(groups) != expected or len(set(groups.values())) != len(expected)
            or any(references['records'][sid].get('source_group') != groups[sid] for sid in expected)):
        raise ValueError('Selected teacher groups must be complete, unique and split-isolated')
    directories = [Path(path).resolve(strict=True) for path in args.cache_roots]
    if len(directories) != nshards or len(set(directories)) != nshards:
        raise ValueError('Require each distinct completed canonical cache shard')
    tokenizer_root = Path(args.tokenizer_root).resolve(strict=True)
    torch.set_num_threads(2)
    rows, cache_sources, shard_numbers = {}, [], set()
    sums = torch.zeros(5120, dtype=torch.float64)
    squares = torch.zeros_like(sums)
    token_count = 0
    shared_source, tokenizer = None, None
    teacher_sources = {}
    for directory in directories:
        plan_path, finished_path = directory / 'plan.json', directory / 'finished.json'
        identities = {'plan': source_record(plan_path), 'finished': source_record(finished_path)}
        plan, finished = read_json(plan_path), read_json(finished_path)
        shard, nshard = plan.get('shard'), plan.get('nshard')
        if (type(shard) is not int or nshard != nshards or not 0 <= shard < nshards or shard in shard_numbers):
            raise ValueError('Missing/duplicate/out-of-range direct cache shard')
        shard_numbers.add(shard)
        ids = (train + val)[shard::nshards]
        if (plan.get('protocol') != DIRECT_PROTOCOL or plan.get('kind') != 'cached_reference_prompt_teacher'
                or plan.get('train_ids') != train or plan.get('val_ids') != val or plan.get('ids') != ids
                or plan.get('max_items') != 0 or plan.get('raw_output_width') != 5120
                or plan.get('raw_output_dtype') != 'torch.bfloat16'
                or plan.get('teacher_fields') != ['dims.genre', 'motion_original']
                or plan.get('teacher_transform') != 'original_motion_already_neutralized_in_selected_caps'
                or plan.get('native_generation') != references['native_generation']
                or finished.get('protocol') != DIRECT_PROTOCOL or finished.get('complete') is not True
                or finished.get('actual_lm_forward') is not True
                or finished.get('prefix_caption_independence_checked') is not True
                or finished.get('original_split_reconstruction_exact') is not True
                or finished.get('expected') != len(ids)
                or finished.get('source_protocol_sha256') != identities['plan']['sha256']
                or finished.get('source_files') != plan.get('source_files')):
            raise ValueError('Direct cache plan/completion must bind actual full native teacher forwarding')
        for name in ('split', 'caps', 'references'):
            identity = plan.get(name, {})
            if (identity.get('sha256') != sources[name]['sha256']
                    or Path(identity.get('path', '')).resolve() != source_paths[name]):
                raise ValueError('Cache plan selection path/SHA mismatch: ' + name)
        source_meta = {key: value for key, value in plan.items() if key not in ('shard', 'ids', 'tokenizer_root')}
        if shared_source is not None and source_meta != shared_source:
            raise ValueError('Direct cache shards disagree on teacher/source preparation provenance')
        shared_source = source_meta
        model_hashes = plan.get('actual_loaded_tensor_sha256', {})
        if (not isinstance(model_hashes, dict) or not model_hashes
                or len(model_hashes) != plan.get('loaded_tensor_count')
                or any(not isinstance(value, str) or len(value) != 64 for value in model_hashes.values())
                or hashlib.sha256(json.dumps(model_hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                   != plan.get('model_weight_content_sha256')):
            raise ValueError('Teacher actual-loaded tensor/model content identity is missing or inconsistent')
        declared_sources = plan.get('source_files', {})
        if not isinstance(declared_sources, dict) or not {'music_text_cache.py', 'music_text_condition.py'} <= set(declared_sources):
            raise ValueError('Teacher cache/condition source content hashes are required')
        for name, identity in declared_sources.items():
            if (not isinstance(identity, dict) or not isinstance(identity.get('path'), str)
                    or not isinstance(identity.get('sha256'), str) or len(identity['sha256']) != 64):
                raise ValueError('Teacher source path/content hash binding missing: ' + name)
            if name not in teacher_sources:
                if not audit_case_cache and source_record(identity['path']) != identity:
                    raise ValueError('Frozen teacher source file content changed: ' + name)
                teacher_sources[name] = identity
            elif teacher_sources[name] != identity:
                raise ValueError('Cache shards use different teacher source files: ' + name)
        tokenizer_hashes = plan.get('tokenizer_files_sha256', {})
        if (not tokenizer_root.is_dir() or not isinstance(tokenizer_hashes, dict)
                or not {'tokenizer.json', 'tokenizer_config.json'} <= set(tokenizer_hashes)):
            raise ValueError('Require actual native tokenizer directory and audited JSON content hashes')
        for name, value in tokenizer_hashes.items():
            if Path(name).name != name:
                raise ValueError('Invalid tokenizer file name')
            identity = source_record(tokenizer_root / name)
            if identity['sha256'] != value:
                raise ValueError('Native tokenizer source content changed: ' + name)
            teacher_sources[str(tokenizer_root / name)] = identity
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_root), local_files_only=True)
        cache_rows = finished.get('rows')
        if not isinstance(cache_rows, list) or [row.get('sample_id') for row in cache_rows] != ids:
            raise ValueError('Finished rows must cover each exact shard ID in plan order')
        if ({path.stem for path in (directory / 'targets').glob('*.pt')} != set(ids)
                or {path.stem for path in (directory / 'prefixes').glob('*.pt')} != set(ids)):
            raise ValueError('Teacher/prefix files do not match exact completed shard IDs')
        file_hashes = parallel_file_sha256(
            [directory / folder / (sid + '.pt') for sid in ids for folder in ('targets', 'prefixes')]
            + [references['records'][sid]['audio_path'] for sid in ids])
        for record in cache_rows:
            sid = record['sample_id']
            partition = 'train' if sid in train_set else 'val'
            target_path, prefix_path = directory / 'targets' / (sid + '.pt'), directory / 'prefixes' / (sid + '.pt')
            if (sid in rows or Path(record.get('target_path', '')).resolve() != target_path
                    or Path(record.get('prefix_path', '')).resolve() != prefix_path
                    or record.get('partition') != partition):
                raise ValueError('Duplicate/direct target path/partition mismatch: ' + sid)
            target_stat, prefix_stat = target_path.stat(), prefix_path.stat()
            target_identity = dict(path=str(target_path), sha256=checked_file_sha256(target_path, file_hashes))
            prefix_identity = dict(path=str(prefix_path), sha256=checked_file_sha256(prefix_path, file_hashes))
            if (target_identity['sha256'] != record.get('target_sha256')
                    or prefix_identity['sha256'] != record.get('prefix_sha256')):
                raise ValueError('Actual full target/prefix file SHA mismatch: ' + sid)
            data = torch.load(target_path, map_location='cpu', weights_only=True)
            if not isinstance(data, dict) or set(data) != {'metadata', 'tail', 'source_token_ids', 'teacher_length'}:
                raise ValueError('Exact direct target record schema mismatch: ' + sid)
            metadata, tail, token_ids = data['metadata'], data['tail'], data['source_token_ids']
            length, prefix_length = record.get('teacher_length'), record.get('prefix_length')
            prompt = f"Dance style: {caps[sid]['dims']['genre']}. " + caps[sid]['motion_original']
            if (type(length) is not int or length < 1 or type(prefix_length) is not int or prefix_length < 1
                    or type(data['teacher_length']) is not int or data['teacher_length'] != length
                    or not isinstance(tail, torch.Tensor) or tail.dtype != torch.bfloat16
                    or tail.shape != (length, 5120) or not torch.isfinite(tail).all()
                    or not isinstance(token_ids, torch.Tensor) or token_ids.dtype != torch.int64
                    or token_ids.shape != (length,) or torch.any(token_ids < 0)
                    or not isinstance(metadata, dict) or metadata.get('kind') != 'cached_reference_prompt_tail'
                    or metadata.get('sample_id') != sid or metadata.get('partition') != partition
                    or metadata.get('teacher_length') != length or metadata.get('prefix_length') != prefix_length
                    or metadata.get('source_protocol_sha256') != identities['plan']['sha256']
                    or metadata.get('prompt') != prompt
                    or metadata.get('protocol') != DIRECT_PROTOCOL or metadata.get('image_conditioned') is not True
                    or metadata.get('model_weight_content_sha256') != plan['model_weight_content_sha256']
                    or metadata.get('source_token_ids_sha256') != cached_tensor_sha(token_ids)
                    or metadata.get('tail_sha256') != cached_tensor_sha(tail)
                    or metadata.get('caption_sha256') != hashlib.sha256(prompt.encode()).hexdigest()
                    or Path(metadata.get('source_protocol_path', '')).resolve() != plan_path):
                raise ValueError('Direct BF16/native token IDs/length/prompt/provenance gate failed: ' + sid)
            parsed_ids = torch.tensor(tokenizer(prompt, add_special_tokens=False)['input_ids'], dtype=torch.int64)
            if not torch.equal(token_ids, parsed_ids):
                raise ValueError('Actual native tokenizer original prompt source IDs mismatch: ' + sid)
            if (metadata.get('image_path') != references['records'][sid]['image_path']
                    or metadata.get('image_sha256') != references['records'][sid]['image_sha256']):
                raise ValueError('Teacher prefix/image selection binding mismatch: ' + sid)
            reference = references['records'][sid]
            audio_path = Path(reference['audio_path']).resolve(strict=True)
            audio_identity = dict(path=str(audio_path), sha256=checked_file_sha256(audio_path, file_hashes))
            if (audio_identity['sha256'] != reference.get('audio_sha256')
                    or (not audit_case_cache and audio_identity != report.get('artifacts', {}).get(sid, {}).get('audio'))):
                raise ValueError('Original selected audio path/content changed: ' + sid)
            value = tail.double()
            example_mean = value.mean(0)
            example_rms = float((value - example_mean).square().mean().sqrt())
            if not math.isfinite(example_rms):
                raise ValueError('Nonfinite raw teacher readout: ' + sid)
            if partition == 'train':
                sums += value.sum(0)
                squares += value.square().sum(0)
                token_count += length
            rows[sid] = dict(audio_path=str(audio_path), audio_sha256=audio_identity['sha256'],
                group_id=groups[sid], target_path=str(target_path), target_sha256=target_identity['sha256'],
                prefix_path=str(prefix_path), prefix_sha256=prefix_identity['sha256'], partition=partition,
                teacher_mean=example_mean.tolist(), teacher_centered_rms=example_rms, teacher_length=length)
            after_target, after_prefix = target_path.stat(), prefix_path.stat()
            if ((target_stat.st_size, target_stat.st_mtime_ns) != (after_target.st_size, after_target.st_mtime_ns)
                    or (prefix_stat.st_size, prefix_stat.st_mtime_ns) != (after_prefix.st_size, after_prefix.st_mtime_ns)):
                raise RuntimeError('Target/prefix changed during streaming validation: ' + sid)
            for path in (target_path, prefix_path, audio_path):
                checked_file_sha256(path, file_hashes)
            del data, metadata, tail, token_ids, value
            if len(rows) % 250 == 0:
                print(json.dumps(dict(event='direct_training_manifest_targets_verified', count=len(rows),
                                     expected=len(expected))), flush=True)
        cache_sources.append(dict(directory=str(directory), shard=shard, nshard=nshards, count=len(ids),
                                  **identities))
    if set(rows) != expected or shard_numbers != set(range(nshards)) or (not audit_case_cache and token_count < 2):
        raise ValueError('Completed direct cache must cover each train/validation ID exactly once')
    verified_sources = [identity for key, identity in teacher_sources.items()
                        if not audit_case_cache or key not in ('music_text_cache.py', 'music_text_condition.py')]
    assert_sources_unchanged(list(sources.values()) + verified_sources
                             + [item[key] for item in cache_sources for key in ('plan', 'finished')])
    if audit_case_cache:
        result = dict(metadata=dict(protocol=DIRECT_PROTOCOL, complete=True,
            purpose='actual_three_case_cache_compatibility_audit_only_not_training_manifest',
            train_ids=[], val_ids=val, trainable=False, prefix_tensors_loaded_by_builder=False,
            actual_native_tokenizer_source_ids_verified=True, tokenizer_root=str(tokenizer_root),
            source_paths_hashes=sources, cache_sources=cache_sources,
            historical_teacher_source_files_validated_against_current_code=False,
            teacher_source_files=list(teacher_sources.values()), native_generation=references['native_generation']),
            rows=rows, statistics=None)
        atomic_json(out, result)
        print(json.dumps(dict(event='actual_direct_case_cache_audit_ready', path=str(out),
                             sha256=digest(out), val=len(val), prefix_tensors_loaded=False,
                             training_manifest=False)), flush=True)
        return result
    mean = sums / token_count
    std = (squares / token_count - mean.square()).clamp_min(0).sqrt().clamp_min(.1)
    energy = float(squares.sum() / (token_count * 5120))
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not math.isfinite(energy):
        raise ValueError('Nonfinite train-only FP64 normalization statistics')
    result = dict(metadata=dict(format_version=1, protocol=DIRECT_PROTOCOL, complete=True,
        train_ids=train, val_ids=val, teacher_kind='cached_reference_prompt_tail',
        student_inputs='waveform_only', student_feature_cache_used=False, prefix_trainable=False,
        teacher_width=5120, teacher_dtype='torch.bfloat16', target_files_validated_by_actual_full_sha256=True,
        prefix_files_validated_by_actual_full_sha256=True, prefix_tensors_loaded_by_builder=False,
        statistics_source='all_valid_raw_teacher_tokens_of_train_only_no_prefix_no_padding_no_qmax',
        statistics_accumulation_dtype='torch.float64', fixed_std_floor=.1,
        per_example_readout='mean over all raw valid teacher tokens; RMS centered by each dimension mean',
        source_paths_hashes=sources, cache_sources=cache_sources, source_meta=shared_source,
        teacher_source_files=list(teacher_sources.values()), actual_native_tokenizer_source_ids_verified=True,
        tokenizer_root=str(tokenizer_root), native_generation=references['native_generation']), rows=rows,
        statistics=dict(mean=mean.tolist(), std=std.tolist(), teacher_energy=energy,
                        training_token_count=token_count))
    atomic_json(out, result)
    print(json.dumps(dict(event='direct_training_manifest_ready', path=str(out), sha256=digest(out),
                         train=len(train), val=len(val), training_token_count=token_count,
                         prefix_tensors_loaded=False)), flush=True)
    return result


def prepare_direct_training_manifest_main():
    parser = argparse.ArgumentParser(description='CPU streaming validated direct teacher cache manifest, no prefix tensor load')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare_direct_training_manifest', action='store_true')
    mode.add_argument('--audit_direct_case_cache', action='store_true')
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--cache_roots', type=Path, nargs='+', required=True)
    parser.add_argument('--tokenizer_root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    return prepare_direct_training_manifest(args, audit_case_cache=args.audit_direct_case_cache)


def main():
    if any(flag in __import__('sys').argv for flag in ('--prepare_direct_training_manifest', '--audit_direct_case_cache')):
        return prepare_direct_training_manifest_main()
    if '--prepare_direct_text_data' in __import__('sys').argv:
        from music_scale_data import direct_selection_main
        argv = __import__('sys').argv[1:]
        argv.remove('--prepare_direct_text_data')
        return direct_selection_main(argv)
    if '--prepare_pure_text_bundle' in __import__('sys').argv:
        return prepare_pure_text_bundle_main()
    if '--prepare_pure_text_init' in __import__('sys').argv:
        return prepare_pure_text_init_main()
    if '--bind_pure_init_manifest' in __import__('sys').argv:
        return bind_pure_init_manifest_main()
    if '--prepare_online_bundle' in __import__('sys').argv:
        return prepare_online_bundle_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache_manager_pid', type=int, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or not args.report.parent.is_dir() or not 0 <= args.gpu < 8:
        parser.error('Require a new report with existing parent and valid GPU')
    root = args.root
    exp = root / 'logs_omnidance/exp/MS'
    targets = [exp / f'music_text_targets_neutral_v1_s{i}' for i in range(4)]
    features = [exp / f'music_text_features_v1_s{i}.pt' for i in range(2)]
    wait_for_target_shards(targets, args.cache_manager_pid)
    reports = [json.loads((directory / 'finished.json').read_text()) for directory in targets]
    split = json.loads((exp / 'music_scale_v4_train1800_split.json').read_text())
    expected = set(split['train'] + split['val'])
    rows = [row for report in reports for row in report['rows']]
    if (any(not report['complete'] or report['source']['teacher_transform'] != 'gender_pronouns_to_dancer_v1' for report in reports)
            or len(rows) != len(expected) or {row['sample_id'] for row in rows} != expected):
        raise ValueError('Finished target shards fail complete/unique/split/teacher protocol gate')
    for row in rows:
        if digest(row['path']) != row['file_sha256']:
            raise ValueError('Teacher-cache file digest mismatch: ' + row['sample_id'])
    print(json.dumps(dict(event='teacher_cache_audited', count=len(rows),
                         training_query_max=max(row['tail_length'] for row in rows if row['sample_id'] in set(split['train'])))), flush=True)
    guard = guard_code(str(root / 'scripts_omnidance/exp/h3ar/sample_h3_ar.py'),
                       str(root / 'scripts_omnidance/exp/hold_gpu.py'), args.gpu)
    stages = [('smallfit', 300, 50, 8, exp / 'music_text_smallfit_neutral_v1'),
              ('formal1800', 1875, 250, 0, exp / 'music_text_train1800_neutral_v1')]
    completed = []
    for stage, steps, eval_every, small_fit, out in stages:
        if out.exists():
            raise FileExistsError('Refuse duplicate training output: ' + str(out))
        log = exp / (out.name + '.log')
        command = ['env', f'CUDA_VISIBLE_DEVICES={args.gpu}', 'OMP_NUM_THREADS=2',
                   'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', '/opt/venv_music/bin/python',
                   str(root / 'scripts_omnidance/exp/ms/music_text_align.py'), '--stage', 'train',
                   '--split', str(exp / 'music_scale_v4_train1800_split.json'), '--out', str(out),
                   '--steps', str(steps), '--eval_every', str(eval_every), '--batch_size', '8',
                   '--lr', '0.0001', '--small_fit', str(small_fit), '--targets']
        command += [str(directory) for directory in targets]
        command += ['--features'] + [str(path) for path in features]
        with log.open('x') as stream:
            stream.write(json.dumps(dict(stage=stage, host=args.host, command=command)) + '\n')
            stream.flush()
            print(json.dumps(dict(event='training_launch', stage=stage, host=args.host)), flush=True)
            process = subprocess.run(SSH + [args.host, 'python3 -c ' + shlex.quote(guard) + ' && ' + shlex.join(command)],
                                     stdout=stream, stderr=subprocess.STDOUT)
        entry = dict(stage=stage, exit_code=process.returncode, log=str(log), out=str(out))
        completed.append(entry)
        if process.returncode:
            with args.report.open('x') as stream:
                json.dump(dict(complete=False, stages=completed, failure=stage), stream, indent=2)
            raise SystemExit(process.returncode)
        summary = json.loads((out / 'finished.json').read_text())
        if not summary['complete'] or not summary['real_feature_checkpoint_roundtrip_exact']:
            raise ValueError('Training finished marker fails deployment-check gate')
        entry['summary_sha256'] = digest(out / 'finished.json')
        print(json.dumps(dict(event='training_finished', stage=stage, steps=summary['steps'])), flush=True)
    with args.report.open('x') as stream:
        json.dump(dict(complete=True, stages=completed, teacher_count=len(rows),
                       generation_not_yet_evaluated=True), stream, indent=2)
    print(json.dumps(dict(event='pipeline_finished', complete=True)), flush=True)


if __name__ == '__main__':
    main()
