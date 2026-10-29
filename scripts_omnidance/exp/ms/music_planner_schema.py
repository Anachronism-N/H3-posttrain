"""Versioned text-caption contracts; the historical fixed2s protocol stays opt-in compatible.

The contract validates existing natural-language targets. It never writes a genre,
rewrites choreography, derives labels from captions, crops audio, or truncates text.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import torch

from .music_planner_adapters import PlannerAdapter

SCHEMA = 'music-planner-caption-contract-v1'
CONDITIONED_SCHEMA = 'music-planner-caption-contract-v2'
TOKEN_PROTOCOL = 'header-then-actions-then-native-im-end-v1'
LENGTH_POLICY = 'one-complete-source-audio-one-verbatim-text-target-v1'


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    keys = {'schema', 'caption_version', 'task_instruction', 'audio_min_seconds',
            'audio_max_seconds', 'required_markers', 'require_target_windows'}
    if isinstance(value, dict) and value.get('schema') == CONDITIONED_SCHEMA:
        keys = keys | {'dance_caption_format'}
        if 'genre_supervision' in value:
            keys = keys | {'genre_supervision'}
    if (not isinstance(value, dict) or set(value) != keys or
            value.get('schema') not in (SCHEMA, CONDITIONED_SCHEMA)):
        raise ValueError('Caption contract needs exactly the documented v1 or v2 keys')
    if value['schema'] == CONDITIONED_SCHEMA:
        from .music_planner_caption_format import validate_format
        value['dance_caption_format'] = validate_format(value['dance_caption_format'])
        if 'genre_supervision' in value:
            policy = value['genre_supervision']
            if (not isinstance(policy, dict) or set(policy) != {'unknown_label', 'unknown_policy', 'mask_unknown_header'} or
                    policy['unknown_label'] != 'unknown' or policy['unknown_policy'] != 'mask' or
                    policy['mask_unknown_header'] is not True):
                raise ValueError('genre_supervision must explicitly mask visual unknown labels and their header loss')
    for key in ('caption_version', 'task_instruction'):
        if not isinstance(value[key], str) or not value[key].strip() or _reserved(value[key]):
            raise ValueError('Nonempty plain-text contract field required: ' + key)
    low, high = value['audio_min_seconds'], value['audio_max_seconds']
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (low, high)) or not 0 < low <= high:
        raise ValueError('Finite positive minimum/maximum audio duration required')
    markers = value['required_markers']
    if (not isinstance(markers, list) or any(not isinstance(m, str) or not m.strip() or _reserved(m)
                                             for m in markers) or len(set(markers)) != len(markers)):
        raise ValueError('required_markers must be unique literal text markers')
    if type(value['require_target_windows']) is not bool:
        raise ValueError('require_target_windows must be a boolean')
    return value


def _reserved(text):
    return '<|' in text or '<audio>' in text


def _windows(row, duration, required):
    windows = row.get('target_windows')
    if windows is None and not required:
        return
    if not isinstance(windows, list) or not windows:
        raise ValueError('Nonempty target_windows required by contract')
    previous_end = 0.
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError('Target window must be an object')
        start, end = window.get('start'), window.get('end')
        if (any(type(v) not in (float, int) or not math.isfinite(v) for v in (start, end)) or
                not 0 <= previous_end <= start < end <= duration):
            raise ValueError('Target windows must be ordered, nonoverlapping, within source audio')
        if not isinstance(window.get('caption'), str) or not window['caption'].strip() or _reserved(window['caption']):
            raise ValueError('Each target window needs a plain-text caption')
        previous_end = end


def read_contract_split(path, split, contract):
    if split not in ('train', 'val'):
        raise ValueError('Training accepts train/val only; test is final-only')
    path = Path(path).resolve()
    rows, seen = [], set()
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get('split') != split:
                raise ValueError(f'{path}:{number}: pure {split} split required')
            for key in ('id', 'source_group_id', 'audio_path', 'audio_sha256', 'target_text', 'caption_version'):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f'{path}:{number}: missing {key}')
            if row['id'] in seen:
                raise ValueError('Each complete source id must occur once per epoch')
            if not re.fullmatch('[0-9a-f]{64}', row['audio_sha256']):
                raise ValueError('Invalid declared audio SHA256')
            if row['caption_version'] != contract['caption_version']:
                raise ValueError('Caption version differs from the selected contract')
            if _reserved(row['target_text']) or any(m not in row['target_text'] for m in contract['required_markers']):
                raise ValueError('Reserved token or missing required caption marker')
            duration = row.get('audio_duration_seconds')
            if (type(duration) not in (int, float) or not math.isfinite(duration) or
                    not contract['audio_min_seconds'] <= duration <= contract['audio_max_seconds']):
                raise ValueError('Declared complete audio duration is outside caption contract')
            if row.get('audio_window_seconds') != [0., duration]:
                raise ValueError('Require complete source [0,duration], not physical chunk crops')
            for key in ('semantic_approval', 'human_review_completed'):
                if type(row.get(key)) is not bool:
                    raise ValueError('Explicit review boolean required: ' + key)
            if (row['semantic_approval'] or row['human_review_completed']) and (
                    not isinstance(row.get('review_provenance'), str) or not row['review_provenance'].strip()):
                raise ValueError('Review claims need independently supplied review_provenance')
            for key in ('song_id', 'near_duplicate_group_id'):
                if key in row and (not isinstance(row[key], str) or not row[key].strip()):
                    raise ValueError('Optional isolation identity must be nonempty: ' + key)
            if contract['schema'] == CONDITIONED_SCHEMA:
                # The v2 parser checks text/metadata together, with the declared
                # rounding tolerance. The historical strict checker would reject
                # a legitimate rounded end such as 5.063 for a 5.0625s source.
                if contract['require_target_windows'] and not row.get('target_windows'):
                    raise ValueError('Nonempty target_windows required by contract')
            else:
                _windows(row, duration, contract['require_target_windows'])
            audio = Path(row['audio_path']).expanduser()
            if not audio.is_absolute():
                audio = path.parent / audio
            row = dict(row, audio_path=os.path.abspath(audio))
            rows.append(row)
            seen.add(row['id'])
    if not rows:
        raise ValueError('Empty split: ' + split)
    return rows, dict(path=str(path), sha256=_file_hash(path), sources=len(rows),
        isolation_groups=len({r['source_group_id'] for r in rows}),
        source_groups_sha256=_hash(sorted({r['source_group_id'] for r in rows})),
        audio_identities_sha256=_hash(sorted({r['audio_sha256'] for r in rows})))


def read_training_data(args):
    path = getattr(args, 'caption_contract', None)
    if path is None:
        from .music_planner_train import check_data
        train, val, data = check_data(args.train, args.val)
        return _join_encoder(args, *_join_genre(args, train, val, data))
    contract = load_contract(path)
    train, train_id = read_contract_split(args.train, 'train', contract)
    val, val_id = read_contract_split(args.val, 'val', contract)
    isolation = {}
    for key in ('id', 'source_group_id', 'audio_sha256', 'audio_path', 'song_id', 'near_duplicate_group_id'):
        left, right = ({r[key] for r in values if key in r} for values in (train, val))
        if left & right:
            raise ValueError('Train/val leakage: ' + key)
        isolation[key] = 'complete-declared' if all(key in r for r in train + val) else 'partial-or-unknown'
    data = dict(train=train_id, val=val_id, length_policy=LENGTH_POLICY,
        contract=contract, contract_sha256=_hash(contract), contract_file_sha256=_file_hash(path),
        source_overlap=0, exact_audio_overlap=0, declared_isolation=isolation,
        song_near_duplicate_status='declared-groups-disjoint-content-similarity-not-audited',
        semantic_approval=False, human_review_completed=False,
        review_claims_verified_by_program=False, test_used=False)
    return _join_encoder(args, *_join_genre(args, train, val, data))


def jev_enabled(args):
    """Explicit opt-out separates the caption vocabulary from an auxiliary head.

    None keeps historical invocations unchanged. New encoder-only experiments
    use --no-enable-jev with the same formal caption vocabulary.
    """
    selected = getattr(args, 'enable_jev', None)
    return getattr(args, 'genre_classes', None) is not None if selected is None else selected


def _join_encoder(args, train, val, data):
    path = getattr(args, 'musicfm_cache', None)
    if path is None:
        return train, val, data
    if getattr(args, 'family', None) != 'moss':
        raise ValueError('MusicFM fusion is a MOSS-only experiment')
    from .music_planner_musicfm import FeatureStore
    store = FeatureStore(path)
    store.validate_rows(train + val, verify_bytes=False)
    data = dict(data, musicfm=store.identity)
    return train, val, data


def _join_genre(args, train, val, data):
    classes_path = getattr(args, 'genre_classes', None)
    label_paths = [getattr(args, key, None) for key in ('train_genre_labels', 'val_genre_labels')]
    if classes_path is None:
        if data.get('contract', {}).get('schema') == CONDITIONED_SCHEMA:
            raise ValueError('The v2 dance caption contract requires a formal --genre-classes vocabulary')
        if any(path is not None for path in label_paths):
            raise ValueError('Genre labels require an explicit --genre-classes vocabulary')
        return train, val, data
    from .music_planner_jev import read_classes, join_jev_labels, classes_identity
    classes = read_classes(classes_path)
    train, train_labels = join_jev_labels(train, label_paths[0], classes, 'train')
    val, val_labels = join_jev_labels(val, label_paths[1], classes, 'val')
    if jev_enabled(args) and train_labels['known_labels'] <= 0:
        raise ValueError('Enabled JEV branch needs at least one independently labelled training source')
    dance_format = data.get('contract', {}).get('dance_caption_format')
    policy = data.get('contract', {}).get('genre_supervision')
    if dance_format is not None:
        from .music_planner_caption_format import parse_caption
        for row in train + val:
            parsed = parse_caption(row['target_text'], classes, dance_format,
                row['audio_duration_seconds'], target_windows=row.get('target_windows'))
            target = row.get('jev_target')
            expected = [float(name == parsed['genre']) for name in classes]
            if target is not None and any(abs(a - b) > 1e-6 for a, b in zip(target, expected)):
                raise ValueError('Caption genre and independent discrete genre label disagree: ' + row['id'])
            row['caption_genre'] = parsed['genre']
    caption_classes = list(classes)
    if policy is not None:
        unknown = policy['unknown_label']
        if unknown not in caption_classes or len(caption_classes) < 3:
            raise ValueError('Abstention protocol needs unknown plus at least two real dance classes')
        classes = [name for name in caption_classes if name != unknown]
        indices = [caption_classes.index(name) for name in classes]
        for rows, metadata in ((train, train_labels), (val, val_labels)):
            missing = sum(row.get('jev_target') is None for row in rows)
            visual_unknown = 0
            for row in rows:
                if row['caption_genre'] == unknown:
                    visual_unknown += int(row.get('jev_target') is not None)
                    row['jev_target'] = None
                    row['genre_supervision_status'] = 'visual-label-abstention'
                elif row.get('jev_target') is not None:
                    row['jev_target'] = [row['jev_target'][index] for index in indices]
                    row['genre_supervision_status'] = 'labelled-real-dance-class'
                else:
                    row['genre_supervision_status'] = 'missing-independent-label'
            metadata.update(source_vocabulary_sha256=metadata['classes_sha256'], classes_sha256=classes_identity(classes),
                known_labels=sum(row.get('jev_target') is not None for row in rows),
                unknown_labels=sum(row.get('jev_target') is None for row in rows),
                missing_independent_labels=missing, masked_visual_unknown_labels=visual_unknown)
        if jev_enabled(args) and train_labels['known_labels'] <= 0:
            raise ValueError('JEV training requires at least one real dance label after masking visual unknown')
    counts = {split: {name: sum(row.get('caption_genre') == name for row in rows) for name in caption_classes}
              for split, rows in (('train', train), ('val', val))} if dance_format is not None else None
    data = dict(data, jev=dict(classes=classes, classes_sha256=classes_identity(classes),
        classes_file_sha256=_file_hash(classes_path), train=train_labels, val=val_labels,
        task='single-categorical-hard-v2' if dance_format is not None else 'single-categorical-hard-or-soft-v1',
        label_inference_from_caption=False))
    if dance_format is not None:
        data.update(caption_classes=caption_classes, caption_classes_sha256=classes_identity(caption_classes),
                    caption_style_counts=counts, style_count_scope='these-manifests-only-not-full-corpus')
    if policy is not None:
        data['genre_supervision'] = policy
    return train, val, data


class ConfiguredPlannerAdapter(PlannerAdapter):
    """Keep native audio prefixes and learn the target text exactly as supplied."""
    def __init__(self, native, contract=None, enable_jev=False, classes=None):
        super().__init__(native.family, native.model, native.processor, native.max_length)
        self.contract = contract
        self.enable_jev = enable_jev
        self.trainable_info = native.trainable_info
        self._audio_identities = native._audio_identities
        self.dance_format = contract.get('dance_caption_format') if contract is not None else None
        self.genre_classes = None
        self.header_token_ids = None
        self.token_protocol = None
        self.dance_caption_protocol = None
        self.genre_supervision = contract.get('genre_supervision') if contract is not None else None
        if self.dance_format is not None:
            from .music_planner_jev import validate_classes, classes_identity
            self.genre_classes = validate_classes(classes)
            self.header_token_ids = {}
            for name in self.genre_classes:
                header = self.dance_format['genre_header_template'].format(genre=name)
                ids = self.tokenizer.encode(header, add_special_tokens=False)
                if (not ids or any(type(token) is not int or token < 0 or token in self.eos_token_ids() for token in ids) or
                        self.tokenizer.decode(ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False) != header):
                    raise ValueError('Each genre header must round-trip exactly without native EOS')
                self.header_token_ids[name] = list(ids)
            self.token_protocol = TOKEN_PROTOCOL
            self.dance_caption_protocol = dict(schema='music-planner-conditioned-caption-v1',
                family=self.family, classes=self.genre_classes, classes_sha256=classes_identity(self.genre_classes),
                dance_format=self.dance_format, header_token_ids=self.header_token_ids, token_protocol=TOKEN_PROTOCOL)
            if self.genre_supervision is not None:
                self.dance_caption_protocol['genre_supervision'] = self.genre_supervision

    def _prompt(self):
        if self.contract is None or self.family == 'omni_captioner':
            return super()._prompt()
        if self.family != 'moss':
            raise ValueError('Configurable caption contracts support MOSS and Qwen Captioner only')
        return self.processor._build_default_prompt(self.contract['task_instruction'], has_audio=True)

    def prefix(self, row, silence=False):
        store = getattr(self, 'musicfm_store', None)
        if store is not None and silence:
            raise ValueError('Fusion ablations require matching re-extracted audio features; cached original audio cannot accompany silence')
        batch = super().prefix(row, silence=silence)
        if store is not None:
            batch['musicfm_features'] = store.load(row).unsqueeze(0)
            batch['musicfm_frame_times_seconds'] = store.frame_times(row).unsqueeze(0)
            batch['musicfm_duration_seconds'] = torch.tensor([row['audio_duration_seconds']], dtype=torch.float32)
        return batch

    def _audio(self, row, silence=False):
        if self.contract is None:
            return super()._audio(row, silence=silence)
        import soundfile as sf
        from scipy.signal import resample_poly
        path = Path(row['audio_path']).expanduser().resolve()
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        expected = row.get('audio_sha256')
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Declared audio_sha256 is required')
        if self._audio_identities.get(str(path)) != (identity, expected):
            if _file_hash(path) != expected:
                raise ValueError('Audio bytes differ from declared audio_sha256')
            self._audio_identities[str(path)] = (identity, expected)
        duration = row.get('audio_duration_seconds')
        if (type(duration) not in (int, float) or not math.isfinite(duration) or
                not self.contract['audio_min_seconds'] <= duration <= self.contract['audio_max_seconds'] or
                row.get('audio_window_seconds') != [0., duration]):
            raise ValueError('Complete source duration/window differs from caption contract')
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        if audio.shape[1] != 1 or not np.isfinite(audio).all():
            raise ValueError('Require a finite mono source waveform')
        if abs(len(audio) - duration * rate) > 1.000001:
            raise ValueError('Waveform length differs from complete declared duration; no crop/pad allowed')
        audio = audio[:, 0]
        if rate != 16000:
            divisor = math.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        return np.zeros_like(audio) if silence else audio

    def collate(self, rows):
        if self.dance_format is None:
            batch = super().collate(rows)
        else:
            from .music_planner_caption_format import parse_caption, encode_target
            if len(rows) != 1:
                raise ValueError('Native audio feature packing supports microbatch=1 only')
            row = rows[0]
            parsed = parse_caption(row['target_text'], self.genre_classes, self.dance_format,
                row['audio_duration_seconds'], target_windows=row.get('target_windows'))
            target = torch.tensor([encode_target(self.tokenizer, parsed, self.eos_token_ids())], dtype=torch.long)
            batch = self.prefix(row)
            prefix = batch['input_ids']
            if prefix.shape[1] + target.shape[1] > self.max_length:
                raise ValueError('Full prefix plus segmented caption exceeds max_length; truncation forbidden')
            batch['input_ids'] = torch.cat((prefix, target), dim=1)
            batch['attention_mask'] = torch.cat((batch['attention_mask'], torch.ones_like(target)), dim=1)
            for key in ('audio_input_mask', 'token_type_ids', 'mm_token_type_ids'):
                if key in batch:
                    batch[key] = torch.cat((batch[key], torch.zeros_like(target, dtype=batch[key].dtype)), dim=1)
            batch['labels'] = torch.cat((torch.full_like(prefix, -100), target.clone()), dim=1)
            if self.genre_supervision is not None and parsed['genre'] == self.genre_supervision['unknown_label']:
                # Unknown is a visual-annotation abstention, not an audio class.
                # Keep it as an action context, but do not teach music->unknown.
                header_end = prefix.shape[1] + len(self.header_token_ids[parsed['genre']])
                batch['labels'][:, prefix.shape[1]:header_end] = -100
        if self.enable_jev:
            # This is an exclusive native prefix length, never an answer token index.
            first = (torch.full((batch['input_ids'].shape[0],), prefix.shape[1], dtype=torch.long)
                     if self.dance_format is not None else batch['labels'].ne(-100).long().argmax(dim=1))
            if not bool(first.gt(0).all()):
                raise ValueError('A nonempty native input prefix is required for JEV')
            batch['jev_prefix_end'] = first
        return batch

    @torch.no_grad()
    def generate(self, row, max_new_tokens=2048, silence=False, seed=0, selected_genre=None, min_style_confidence=None):
        if self.dance_format is None:
            if selected_genre is not None or min_style_confidence is not None:
                raise ValueError('Selected-genre control requires a trained v2 caption protocol')
            return super().generate(row, max_new_tokens=max_new_tokens, silence=silence, seed=seed)
        if silence:
            raise ValueError('Conditioned v2 inference requires an explicit verified audio source')
        if not self.enable_jev and selected_genre is None and min_style_confidence is None:
            from .music_planner_encoder_generate import generate_free
            return dict(generate_free(self, row, max_new_tokens=max_new_tokens), seed=seed)
        from .music_planner_moss_generate import generate_conditioned
        options = dict(max_new_tokens=max_new_tokens, selected_genre=selected_genre)
        if min_style_confidence is not None:
            options['min_style_confidence'] = min_style_confidence
        result = generate_conditioned(self, row, **options)
        return dict(result, seed=seed)

    def save_pretrained(self, path, state_dict=None):
        super().save_pretrained(path, state_dict=state_dict)
        if self.contract is not None:
            (Path(path) / 'caption_contract.json').write_text(
                json.dumps(self.contract, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if self.dance_caption_protocol is not None:
            (Path(path) / 'dance_caption_protocol.json').write_text(
                json.dumps(self.dance_caption_protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def configure_adapter(native, args):
    contract_path = getattr(args, 'caption_contract', None)
    contract = load_contract(contract_path) if contract_path is not None else None
    enable_jev = jev_enabled(args)
    if contract is None and not enable_jev and getattr(args, 'musicfm_cache', None) is None:
        return native
    if native.family not in ('moss', 'omni_captioner'):
        raise ValueError('New planner training protocol supports MOSS and Qwen Captioner only')
    classes = None
    if contract is not None and contract.get('dance_caption_format') is not None:
        from .music_planner_jev import read_classes
        if getattr(args, 'genre_classes', None) is None:
            raise ValueError('The v2 caption adapter requires a formal --genre-classes vocabulary')
        classes = read_classes(args.genre_classes)
    adapter = ConfiguredPlannerAdapter(native, contract=contract, enable_jev=enable_jev, classes=classes)
    path = getattr(args, 'musicfm_cache', None)
    if path is not None:
        from .music_planner_musicfm import FeatureStore
        adapter.musicfm_store = FeatureStore(path)
    return adapter


def assert_adapter_data_identity(adapter, data):
    """Reject contract/class files changed while a large native model was loading."""
    expected = data.get('contract')
    if getattr(adapter, 'contract', None) != expected:
        raise ValueError('Caption contract changed between dataset validation and adapter initialization')
    if expected is not None and expected.get('schema') == CONDITIONED_SCHEMA:
        if getattr(adapter, 'genre_classes', None) != data.get('caption_classes', data.get('jev', {}).get('classes')):
            raise ValueError('Genre vocabulary/order changed between dataset validation and adapter initialization')
    store = getattr(adapter, 'musicfm_store', None)
    if (store.identity if store is not None else None) != data.get('musicfm'):
        raise ValueError('MusicFM cache changed between dataset validation and adapter initialization')


def restore_adapter_protocol(native, checkpoint, musicfm_cache=None):
    """Restore contract/head AFTER loading this checkpoint's native PEFT adapter.

    This verifies checkpoint/portable-artifact bindings, not the caller's base
    weight contents. Load the matching base and adapter before calling it. Native
    MOSS generation compatibility patches remain separate, as in the eval tools.
    """
    from .music_planner_train import object_sha
    checkpoint = Path(checkpoint).resolve()
    marker = json.loads((checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))
    config = json.loads((checkpoint.parent / 'run_config.json').read_text(encoding='utf-8'))
    canonical = {k: v for k, v in config.items() if k not in ('fingerprint', 'trainable_parameters')}
    if (marker.get('schema') != 'music-planner-train-v1' or marker.get('complete') is not True or
            marker.get('fingerprint') != object_sha(canonical) or
            config.get('fingerprint') != marker.get('fingerprint') or
            config.get('family') != native.family or config.get('data', {}).get('test_used') is not False or
            marker.get('trainable_sha256') != _file_hash(checkpoint / 'trainable.pt')):
        raise ValueError('Training checkpoint/config/family/portable identity differs')
    refs = {}
    members = marker.get('files', [])
    portable = [ref for ref in members if isinstance(ref, dict) and isinstance(ref.get('path'), str)
                and ref['path'].replace('\\', '/').endswith('/trainable.pt')]
    if len(portable) != 1 or portable[0].get('sha256') != marker['trainable_sha256']:
        raise ValueError('Checkpoint inventory requires one matching portable state entry')
    recorded_root = portable[0]['path'].replace('\\', '/')[:-len('/trainable.pt')]
    for ref in members:
        original = ref['path'].replace('\\', '/')
        if not original.startswith(recorded_root + '/'):
            raise ValueError('Checkpoint member is outside the recorded checkpoint directory')
        relative = original[len(recorded_root) + 1:]
        if any(part in ('', '.', '..') or ':' in part for part in relative.split('/')):
            raise ValueError('Checkpoint member escapes its directory')
        if relative.startswith(('adapter/', 'jev/', 'fusion/')):
            target = (checkpoint / relative).resolve()
            if checkpoint not in target.parents:
                raise ValueError('Checkpoint adapter/head symlink escapes its directory')
            if relative in refs or _file_hash(target) != ref['sha256']:
                raise ValueError('Checkpoint adapter/head member identity differs')
            refs[relative] = ref
    actual = {p.relative_to(checkpoint).as_posix() for folder in ('adapter', 'jev', 'fusion')
              for p in (checkpoint / folder).rglob('*') if p.is_file()}
    if actual != set(refs) or not any(k.startswith('adapter/') for k in refs):
        raise ValueError('Checkpoint adapter/head file membership differs')
    contract = config.get('data', {}).get('contract')
    exported = checkpoint / 'adapter' / 'caption_contract.json'
    if (contract is None) != (not exported.is_file()):
        raise ValueError('Checkpoint caption contract export is missing or unexpected')
    if contract is not None and load_contract(exported) != contract:
        raise ValueError('Exported caption contract differs from the training contract')
    enabled = config.get('jev') is not None
    if enabled != (checkpoint / 'jev').is_dir():
        raise ValueError('Checkpoint JEV head is missing or unexpected')
    classes = config.get('data', {}).get('caption_classes', config.get('data', {}).get('jev', {}).get('classes'))
    fusion = config.get('fusion')
    if (fusion is not None) != (checkpoint / 'fusion').is_dir():
        raise ValueError('Checkpoint encoder fusion is missing or unexpected')
    if musicfm_cache is not None and fusion is None:
        raise ValueError('MusicFM cache cannot be added to a checkpoint trained without fusion')
    adapter = (ConfiguredPlannerAdapter(native, contract=contract, enable_jev=enabled, classes=classes)
               if contract is not None or enabled or fusion is not None else native)
    protocol_path = checkpoint / 'adapter' / 'dance_caption_protocol.json'
    if getattr(adapter, 'dance_caption_protocol', None) is not None:
        if (not protocol_path.is_file() or json.loads(protocol_path.read_text(encoding='utf-8')) != adapter.dance_caption_protocol):
            raise ValueError('Exported dance caption token/class protocol differs from the restored model')
    elif protocol_path.exists():
        raise ValueError('Unexpected dance caption protocol for a historical checkpoint')
    adapter.max_length = config['max_length']
    if enabled:
        from .music_planner_jev import load_jev
        load_jev(adapter, checkpoint / 'jev', expected_identity=dict(
            run_fingerprint=marker['fingerprint'], trainable_sha256=marker['trainable_sha256'],
            installation=config['jev']))
    if fusion is not None:
        from .music_planner_musicfm import FeatureStore
        from .music_planner_encoder_fusion import load_fusion
        store = FeatureStore(musicfm_cache if musicfm_cache is not None else fusion['cache_path'])
        if store.identity['extractor_identity'] != fusion['cache_identity']['extractor_identity']:
            raise ValueError('Inference MusicFM extractor differs from the trained feature protocol')
        adapter.musicfm_store = store
        load_fusion(adapter, checkpoint / 'fusion', expected_identity=dict(
            run_fingerprint=marker['fingerprint'], trainable_sha256=marker['trainable_sha256'],
            installation=fusion['installation']))
        adapter.fusion_metadata = fusion['installation']
    adapter.training_protocol_identity = dict(fingerprint=marker['fingerprint'],
        checkpoint_path=str(checkpoint), marker_sha256=_file_hash(checkpoint / 'trainer_state.json'),
        cursor=dict(marker.get('cursor', {})), family=native.family, trainable_sha256=marker['trainable_sha256'],
        native_checkpoint_load_required=True, base_weight_content_verified_by_this_helper=False)
    return adapter
