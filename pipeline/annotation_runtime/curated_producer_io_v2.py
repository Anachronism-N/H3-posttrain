"""Frozen producer with per-source local native ROI and verified no-replace publication.

No source media/model staging, ROI thresholds, decoding or evidence rules change.
Only this new process wraps the frozen producer's two ROI entry points.
"""
from __future__ import annotations

import copy
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

sys.dont_write_bytecode = True
FROZEN = Path(__file__).resolve().parent / 'source_curated_producer_v1/curated_pipeline.py'


def load_pipeline(path=FROZEN):
    spec = importlib.util.spec_from_file_location('_curated_producer_io_frozen', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rename_noreplace(source, target):
    # Both paths are siblings on the shared filesystem, never a local->shared rename.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def remap_paths(value, old, new):
    if isinstance(value, dict):
        return {key: remap_paths(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [remap_paths(item, old, new) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        try:
            relative = Path(value).relative_to(old)
        except ValueError:
            return value
        return str(new / relative)
    return value


def publish_roi(roi, local, final, prepared, extraction_seconds):
    local, final = Path(local).resolve(), Path(final).resolve()
    manifest = roi.validate_completed(local, prepared)
    if json.loads((local / 'source_manifest.json').read_bytes()) != prepared:
        raise ValueError('Local source manifest differs from original source binding')
    assets = [manifest['raw']] + manifest['crops']
    allowed = {'source_manifest.json', 'prepare_receipt.json', 'raw_wholebody134.npz',
               'roi_manifest.json', 'extract_receipt.json'}
    for asset in assets:
        asset_path = Path(asset['path'])
        if asset_path.is_symlink() or asset_path.resolve() != asset_path:
            raise ValueError('ROI asset is not a real canonical file')
        relative = asset_path.relative_to(local)
        if asset is manifest['raw']:
            if str(relative) != 'raw_wholebody134.npz':
                raise ValueError('Unexpected raw ROI asset location')
        elif relative.parent != Path('crops') or relative.suffix != '.jpg':
            raise ValueError('Crop escaped the source-local crop directory')
        allowed.add(relative.as_posix())
    actual = set()
    for path in local.rglob('*'):
        if path.is_symlink() or (path.is_dir() and path != local / 'crops'):
            raise ValueError('Unexpected source-local directory/symlink')
        if path.is_file():
            actual.add(path.relative_to(local).as_posix())
    if actual != allowed:
        raise ValueError('Unexpected/missing local ROI files')
    if final.exists():
        raise FileExistsError('Refusing to overwrite any existing ROI directory: ' + str(final))
    mapped = remap_paths(manifest, local, final)
    # Output references change, but original video identity, source manifest and raw NPZ do not.
    for key in ('source', 'video_sha256', 'binding_sha256', 'model_ref', 'probe', 'selected_frames'):
        if mapped[key] != manifest[key]:
            raise ValueError('Publication changed source evidence: ' + key)
    manifest_bytes = roi.json_bytes(mapped)
    receipt = json.loads((local / 'extract_receipt.json').read_bytes())
    import hashlib
    receipt['roi_manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
    replacements = {'roi_manifest.json': manifest_bytes,
                    'extract_receipt.json': roi.json_bytes(receipt)}
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + final.name + '.io-v2-', dir=final.parent))
    started = time.monotonic()
    owned_files, owned_dirs = [], []
    try:
        crop_dir = stage / 'crops'
        crop_dir.mkdir()
        owned_dirs.append(crop_dir)
        verified = {}
        for relative in sorted(allowed):
            target = stage / relative
            with target.open('xb') as output:
                owned_files.append(target)
                if relative in replacements:
                    output.write(replacements[relative])
                else:
                    with (local / relative).open('rb') as source:
                        shutil.copyfileobj(source, output, 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            expected = (hashlib.sha256(replacements[relative]).hexdigest() if relative in replacements
                        else roi.sha256(local / relative))
            if roi.sha256(target) != expected:
                raise ValueError('Shared stage content SHA mismatch: ' + relative)
            verified[relative] = expected
        roi.verify_identity(prepared['source'])
        roi.verify_model(prepared['model_ref'])
        evidence = {'schema': 'curated_producer_io_v2', 'source_id': prepared['id'],
                    'source_ordinal': prepared['input_row']['manifest_record']['source_ordinal'],
                    'source_binding_sha256': prepared['binding_sha256'],
                    'source_manifest_sha256_unchanged': verified['source_manifest.json'],
                    'raw_sha256_unchanged': verified['raw_wholebody134.npz'],
                    'roi_model_sha256': prepared['model_ref']['sha256'],
                    'output_path_mapping': {'local_roi_directory': str(local), 'shared_roi_directory': str(final)},
                    'source_media_staged': False, 'source_manifest_rewritten': False,
                    'all_shared_assets_sha256': verified,
                    'local_extract_seconds': extraction_seconds,
                    'shared_copy_verify_seconds': time.monotonic() - started,
                    'per_file_shared_fsync': True, 'atomic_publish': 'same_filesystem_renameat2_noreplace',
                    'semantic_approval': False, 'human_review_completed': False, 'formal_training': False}
        path = stage / 'io_publish_receipt.json'
        with path.open('xb') as stream:
            owned_files.append(path)
            stream.write(roi.json_bytes(evidence))
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(crop_dir)
        fsync_directory(stage)
        rename_noreplace(stage, final)
        fsync_directory(final.parent)
        return roi.validate_completed(final, prepared)
    finally:
        # Only this invocation's explicitly created paths are eligible for cleanup.
        if stage.exists():
            for path in reversed(owned_files):
                path.unlink(missing_ok=True)
            for path in reversed(owned_dirs):
                path.rmdir()
            stage.rmdir()


def cleanup_local_generated(root, source):
    directory = root / source['id']
    allowed = {'source_manifest.json', 'prepare_receipt.json', 'raw_wholebody134.npz',
               'roi_manifest.json', 'extract_receipt.json'}
    allowed.update('crops/f%06d_%s.jpg' % (frame['frame_index'], part)
                   for frame in source['selected_frames'] for part in ('left_hand', 'right_hand'))
    generated = []
    for path in root.rglob('*'):
        if path.is_symlink():
            return False
        if path.is_dir():
            if path not in (directory, directory / 'crops'):
                return False
        elif path.is_file() and path.is_relative_to(directory) and path.relative_to(directory).as_posix() in allowed:
            generated.append(path)
        else:
            return False
    for path in generated:
        path.unlink()
    if (directory / 'crops').exists():
        (directory / 'crops').rmdir()
    if directory.exists():
        directory.rmdir()
    root.rmdir()
    return True


class LocalROI:
    def __init__(self, roi, local_root):
        self.roi = roi
        self.local_root = Path(local_root).resolve(strict=True)

    def __getattr__(self, name):
        return getattr(self.roi, name)

    def ensure_prepared(self, out, source):
        final = Path(out) / source['id']
        if final.exists():
            old = json.loads((final / 'source_manifest.json').read_bytes())
            if old != source:
                raise ValueError('Existing ROI source/SHA/PTS/model/params differ')
            self.roi.validate_completed(final, source)
        return final  # No incomplete source directory is published during preparation.

    def extract_source(self, directory, source, model):
        if directory.exists():
            return self.roi.validate_completed(directory, source)
        temporary = Path(tempfile.mkdtemp(prefix='curated-roi-io-v2-', dir=self.local_root))
        try:
            local = self.roi.ensure_prepared(temporary, source)
            started = time.monotonic()
            self.roi.extract_source(local, source, model)
            seconds = time.monotonic() - started
            return publish_roi(self.roi, local, directory, source, seconds)
        finally:
            # Do not recursively remove an unexpected file/directory, even under our own temp root.
            cleanup_local_generated(temporary, source)


def select_ids(pipeline, args):
    ids = list(args.ids or [])
    if args.source_ids_file:
        ids.extend(i.strip() for i in Path(args.source_ids_file).read_text().splitlines() if i.strip())
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', i) for i in ids):
        raise ValueError('Source IDs must be safe and unique')
    ranged = args.ordinal_start is not None or args.ordinal_end is not None
    if ranged and (args.ordinal_start is None or args.ordinal_end is None or
                   not 0 <= args.ordinal_start < args.ordinal_end):
        raise ValueError('Both half-open ordinal bounds are required')
    if not ids and not ranged:
        raise ValueError('Explicit IDs/file or ordinal bounds required; no implicit full-source launch')
    summary = pipeline.load(args.manifest)
    wanted = set(ids)
    selected, found = [], set()
    for row, batch in pipeline.iter_rows(summary):
        if wanted and found == wanted:
            break
        if ranged and not wanted and row['source_ordinal'] >= args.ordinal_end:
            break
        if wanted and row['id'] not in wanted:
            continue
        found.add(row['id'])
        if ranged and not args.ordinal_start <= row['source_ordinal'] < args.ordinal_end:
            continue
        if row['source_ordinal'] % args.nshard == args.shard and (not args.limit or len(selected) < args.limit):
            selected.append(row['id'])
    if wanted - found:
        raise ValueError('Requested IDs absent from verified manifest: ' + str(sorted(set(ids) - found)))
    if not selected:
        raise ValueError('Explicit selection is empty for this shard')
    return selected


def main(argv=None):
    pipeline = load_pipeline()
    parser = pipeline.parser()
    producer = parser._subparsers._group_actions[0].choices['producer']
    producer.add_argument('--local-work-root', required=True, help='Existing node-local directory, not CephFS')
    producer.add_argument('--source-ids-file')
    producer.add_argument('--ordinal-start', type=int)
    producer.add_argument('--ordinal-end', type=int)
    args = parser.parse_args(argv)
    if args.mode != 'producer' or args.stage != 'run':
        raise ValueError('IO v2 only accepts producer --stage run')
    if not 0 <= args.shard < args.nshard or args.limit < 0:
        raise ValueError('Invalid shard/limit')
    local_root = Path(args.local_work_root).resolve(strict=True)
    if not local_root.is_dir() or os.stat(local_root).st_dev == os.stat(Path(args.out).resolve().parent).st_dev:
        raise ValueError('local-work-root must be a different node-local filesystem from out')
    args.ids = select_ids(pipeline, args)
    args.limit = len(args.ids)  # Stop the frozen iterator after exactly this verified selection.
    bar, roi, visual = pipeline.modules(args.repo)
    wrapped = LocalROI(roi, local_root)
    pipeline.modules = lambda repo: (bar, wrapped, visual)
    original_save, original_append, original_rows = pipeline.save, pipeline.append, pipeline.iter_rows
    current_row = [None]

    def iter_rows(*positional, **keywords):
        for row, batch in original_rows(*positional, **keywords):
            current_row[0] = row
            yield row, batch

    pipeline.iter_rows = iter_rows

    def save(path, value):
        if Path(path).name == 'producer.terminal.json' and value.get('status') == 'ready':
            value['roi_model_sha256'] = pipeline.load(value['roi_ref']['path'])['model_ref']['sha256']
        if Path(path).name == 'protocol.json':
            value = {**value, 'producer_io_code': pipeline.ref(__file__),
                     'source_selection': {'ids': args.ids, 'ordinal_start': args.ordinal_start,
                                          'ordinal_end': args.ordinal_end},
                     'io_policy': 'source_identity_unchanged_local_roi_verified_atomic_publish_v2'}
        return original_save(path, value)

    def append(path, value):
        queue_hint = Path(path) == Path(args.out).resolve() / 'ready.jsonl'
        if queue_hint:
            row = current_row[0]
            if row is None or any(value.get(key) != row[key] for key in ('id', 'split', 'batch_id', 'source_ordinal')):
                raise ValueError('Ready/resume hint differs from selected original manifest row')
            if value['status'] == 'failed' and value.get('manifest_record') != row:
                raise ValueError('Failed terminal original manifest differs')
        if queue_hint and value['status'] == 'ready':
            expected = {'plan_ref': pipeline.source_dir(args.out, row) / 'bar_plan.json',
                        'roi_ref': Path(args.out) / 'roi' / row['id'] / 'roi_manifest.json'}
            if any(Path(value[key]['path']) != target.resolve() for key, target in expected.items()):
                raise ValueError('Ready/resume references escaped selected source output')
            for key in ('plan_ref', 'roi_ref'):
                artifact = value[key]
                if pipeline.sha(artifact['path']) != artifact['sha256']:
                    raise ValueError('Ready/resume artifact SHA mismatch: ' + key)
            plan = pipeline.load(value['plan_ref']['path'])
            if plan['binding'] != value['source_binding'] or plan['manifest_record'] != row:
                raise ValueError('Ready/resume source binding mismatch')
            directory = Path(value['roi_ref']['path']).parent
            prepared = pipeline.load(directory / 'source_manifest.json')
            if prepared['input_row']['manifest_record'] != plan['manifest_record']:
                raise ValueError('Ready/resume original manifest binding mismatch')
            roi.validate_completed(directory, prepared)
            roi.verify_identity(prepared['source'])
            roi.verify_model(prepared['model_ref'])
            if roi.sha256(prepared['source']['path']) != prepared['video_sha256']:
                raise ValueError('Ready/resume original video SHA mismatch')
        return original_append(path, value)

    pipeline.save, pipeline.append = save, append
    return pipeline.run_producer(args)


if __name__ == '__main__':
    try:
        print(json.dumps(main(), ensure_ascii=False), flush=True)
    except Exception as error:
        print(json.dumps({'status': 'worker_stopped', 'error': type(error).__name__ + ': ' + str(error),
                          'semantic_approval': False, 'formal_training': False}), file=sys.stderr, flush=True)
        sys.exit(2)
