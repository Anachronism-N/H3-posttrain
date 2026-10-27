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
    if not isinstance(value, dict) or set(value) != keys or value.get('schema') != SCHEMA:
        raise ValueError('Caption contract needs exactly the documented v1 keys')
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
        return _join_genre(args, train, val, data)
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
    return _join_genre(args, train, val, data)


def _join_genre(args, train, val, data):
    classes_path = getattr(args, 'genre_classes', None)
    label_paths = [getattr(args, key, None) for key in ('train_genre_labels', 'val_genre_labels')]
    if classes_path is None:
        if any(path is not None for path in label_paths):
            raise ValueError('Genre labels require an explicit --genre-classes vocabulary')
        return train, val, data
    from .music_planner_jev import read_classes, join_jev_labels, classes_identity
    classes = read_classes(classes_path)
    train, train_labels = join_jev_labels(train, label_paths[0], classes, 'train')
    val, val_labels = join_jev_labels(val, label_paths[1], classes, 'val')
    if train_labels['known_labels'] <= 0:
        raise ValueError('Enabled JEV branch needs at least one independently labelled training source')
    data = dict(data, jev=dict(classes=classes, classes_sha256=classes_identity(classes),
        classes_file_sha256=_file_hash(classes_path), train=train_labels, val=val_labels,
        task='single-categorical-hard-or-soft-v1', label_inference_from_caption=False))
    return train, val, data


class ConfiguredPlannerAdapter(PlannerAdapter):
    """Keep native audio prefixes and learn the target text exactly as supplied."""
    def __init__(self, native, contract=None, enable_jev=False):
        super().__init__(native.family, native.model, native.processor, native.max_length)
        self.contract = contract
        self.enable_jev = enable_jev
        self.trainable_info = native.trainable_info
        self._audio_identities = native._audio_identities

    def _prompt(self):
        if self.contract is None or self.family == 'omni_captioner':
            return super()._prompt()
        if self.family != 'moss':
            raise ValueError('Configurable caption contracts support MOSS and Qwen Captioner only')
        return self.processor._build_default_prompt(self.contract['task_instruction'], has_audio=True)

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
        batch = super().collate(rows)
        if self.enable_jev:
            # This is an exclusive native prefix length, never an answer token index.
            first = batch['labels'].ne(-100).long().argmax(dim=1)
            if not bool(first.gt(0).all()):
                raise ValueError('A nonempty native input prefix is required for JEV')
            batch['jev_prefix_end'] = first
        return batch

    def save_pretrained(self, path, state_dict=None):
        super().save_pretrained(path, state_dict=state_dict)
        if self.contract is not None:
            (Path(path) / 'caption_contract.json').write_text(
                json.dumps(self.contract, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def configure_adapter(native, args):
    contract_path = getattr(args, 'caption_contract', None)
    contract = load_contract(contract_path) if contract_path is not None else None
    enable_jev = getattr(args, 'genre_classes', None) is not None
    if contract is None and not enable_jev:
        return native
    if native.family not in ('moss', 'omni_captioner'):
        raise ValueError('New planner training protocol supports MOSS and Qwen Captioner only')
    return ConfiguredPlannerAdapter(native, contract=contract, enable_jev=enable_jev)


def restore_adapter_protocol(native, checkpoint):
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
    anchor = '/' + checkpoint.name + '/'
    for ref in marker.get('files', []):
        original = '/' + ref['path'].replace('\\', '/')
        if anchor not in original:
            raise ValueError('Checkpoint member is outside the recorded checkpoint directory')
        relative = original.rsplit(anchor, 1)[1]
        if '..' in relative.split('/'):
            raise ValueError('Checkpoint member escapes its directory')
        if relative.startswith(('adapter/', 'jev/')):
            if relative in refs or _file_hash(checkpoint / relative) != ref['sha256']:
                raise ValueError('Checkpoint adapter/head member identity differs')
            refs[relative] = ref
    actual = {p.relative_to(checkpoint).as_posix() for folder in ('adapter', 'jev')
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
    adapter = (ConfiguredPlannerAdapter(native, contract=contract, enable_jev=enabled)
               if contract is not None or enabled else native)
    adapter.max_length = config['max_length']
    if enabled:
        from .music_planner_jev import load_jev
        load_jev(adapter, checkpoint / 'jev', expected_identity=dict(
            run_fingerprint=marker['fingerprint'], trainable_sha256=marker['trainable_sha256'],
            installation=config['jev']))
    adapter.training_protocol_identity = dict(fingerprint=marker['fingerprint'],
        native_checkpoint_load_required=True, base_weight_content_verified_by_this_helper=False)
    return adapter
