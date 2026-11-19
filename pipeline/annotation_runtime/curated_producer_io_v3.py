"""Ceph-compatible exclusive directory plus atomic last-extraction-receipt commit.

Reuses immutable IO v2 local extraction/selection and the original frozen producer.
Directory visibility is NOT atomic. Consumers remain gated by producer.terminal.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import uuid

sys.dont_write_bytecode = True
import curated_producer_io_v2 as base

BASE_SHA256 = '5fd8f933116a1e78d0bcfd391fa37df48730a6ade8704b0dbab575b62600c5dd'


def publish_roi(roi, local, final, prepared, extraction_seconds):
    local, final = Path(local).resolve(), Path(final).resolve()
    started = time.monotonic()
    manifest = roi.validate_completed(local, prepared)
    if json.loads((local / 'source_manifest.json').read_bytes()) != prepared:
        raise ValueError('Local source manifest differs from original source binding')
    assets = [manifest['raw']] + manifest['crops']
    allowed = {'source_manifest.json', 'prepare_receipt.json', 'raw_wholebody134.npz',
               'roi_manifest.json', 'extract_receipt.json'}
    for asset in assets:
        path = Path(asset['path'])
        if path.is_symlink() or path.resolve() != path:
            raise ValueError('ROI asset is not canonical')
        relative = path.relative_to(local)
        if asset is manifest['raw']:
            if relative != Path('raw_wholebody134.npz'):
                raise ValueError('Unexpected raw ROI path')
        elif relative.parent != Path('crops') or relative.suffix != '.jpg':
            raise ValueError('Crop escaped source directory')
        allowed.add(relative.as_posix())
    actual = set()
    for path in local.rglob('*'):
        if path.is_symlink() or (path.is_dir() and path != local / 'crops'):
            raise ValueError('Unexpected local directory/symlink')
        if path.is_file():
            actual.add(path.relative_to(local).as_posix())
    if actual != allowed:
        raise ValueError('Unexpected/missing local ROI files')
    if final.exists():
        raise FileExistsError('Refusing any existing ROI directory: ' + str(final))
    mapped = base.remap_paths(manifest, local, final)
    for key in ('source', 'video_sha256', 'binding_sha256', 'model_ref', 'probe', 'selected_frames'):
        if mapped[key] != manifest[key]:
            raise ValueError('Publication changed source evidence: ' + key)
    manifest_bytes = roi.json_bytes(mapped)
    receipt = json.loads((local / 'extract_receipt.json').read_bytes())
    receipt['roi_manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
    replacements = {'roi_manifest.json': manifest_bytes, 'extract_receipt.json': roi.json_bytes(receipt)}
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + final.name + '.io-v3-', dir=final.parent))
    owned_files, owned_dirs = [], []
    owner = uuid.uuid4().hex
    try:
        (stage / 'crops').mkdir()
        owned_dirs.append(stage / 'crops')
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
        copied_seconds = time.monotonic() - started
        roi.verify_identity(prepared['source'])
        roi.verify_model(prepared['model_ref'])
        ownership = {'schema': 'curated_producer_io_v3_ownership', 'owner_token': owner,
                     'owner_pid': os.getpid(), 'source_id': prepared['id'],
                     'source_ordinal': prepared['input_row']['manifest_record']['source_ordinal'],
                     'source_binding_sha256': prepared['binding_sha256'],
                     'shared_roi_directory': str(final), 'generated_assets_sha256': verified,
                     'whole_directory_atomic': False, 'exclusive_directory': True,
                     'commit_marker': 'extract_receipt.json',
                     'consumer_gate': 'producer.terminal.json status=ready after full validation',
                     'partial_failure_policy': 'retain_final_directory_refuse_incomplete_resume',
                     'semantic_approval': False, 'formal_training': False}

        def stage_json(name, value):
            path = stage / name
            with path.open('xb') as stream:
                owned_files.append(path)
                stream.write(roi.json_bytes(value))
                stream.flush()
                os.fsync(stream.fileno())
            return path

        stage_json('io_publish_ownership.json', ownership)
        base.fsync_directory(stage / 'crops')
        base.fsync_directory(stage)
        # This is the only claim on final. No rename, replacement, or reuse of an empty directory.
        final.mkdir(exist_ok=False)
        os.link(stage / 'io_publish_ownership.json', final / 'io_publish_ownership.json')
        base.fsync_directory(final)
        base.fsync_directory(final.parent)
        (final / 'crops').mkdir(exist_ok=False)
        before_commit = sorted(allowed - {'roi_manifest.json', 'extract_receipt.json'})
        for relative in before_commit:
            os.link(stage / relative, final / relative)
            if roi.sha256(final / relative) != verified[relative]:
                raise ValueError('Final linked asset SHA mismatch: ' + relative)
        base.fsync_directory(final / 'crops')
        base.fsync_directory(final)
        base.fsync_directory(final.parent)
        roi.verify_identity(prepared['source'])
        roi.verify_model(prepared['model_ref'])
        os.link(stage / 'roi_manifest.json', final / 'roi_manifest.json')
        base.fsync_directory(final)
        # Atomic no-replace file link is the last ROI schema artifact to be made visible.
        os.link(stage / 'extract_receipt.json', final / 'extract_receipt.json')
        base.fsync_directory(final)
        base.fsync_directory(final.parent)
        result = roi.validate_completed(final, prepared)
        committed_seconds = time.monotonic() - started
        evidence = {'schema': 'curated_producer_io_v3', 'owner_token': owner,
                    'source_id': prepared['id'], 'source_ordinal': ownership['source_ordinal'],
                    'source_binding_sha256': prepared['binding_sha256'],
                    'roi_model_sha256': prepared['model_ref']['sha256'],
                    'source_manifest_sha256_unchanged': verified['source_manifest.json'],
                    'raw_sha256_unchanged': verified['raw_wholebody134.npz'],
                    'output_path_mapping': {'local_roi_directory': str(local), 'shared_roi_directory': str(final)},
                    'source_media_staged': False, 'source_manifest_rewritten': False,
                    'all_shared_assets_sha256': verified,
                    'local_extract_seconds': extraction_seconds,
                    'shared_copy_verify_seconds': copied_seconds,
                    'shared_publish_seconds': committed_seconds,
                    'shared_publish_timer_end': 'after last ROI receipt link, directory fsync, full validate_completed',
                    'whole_directory_atomic': False, 'per_file_shared_fsync': True,
                    'publication_protocol': 'exclusive_mkdir_samefs_hardlink_noreplace_last_extract_receipt_v3',
                    'complete_roi_validated': True,
                    'semantic_approval': False, 'human_review_completed': False, 'formal_training': False}
        stage_json('io_publish_receipt.json', evidence)
        os.link(stage / 'io_publish_receipt.json', final / 'io_publish_receipt.json')
        base.fsync_directory(final)
        base.fsync_directory(final.parent)
        return result
    finally:
        # Failed final directories, including any partial hardlinks, are intentionally retained.
        # Cleanup is limited to paths explicitly generated in this unique shared stage.
        for path in reversed(owned_files):
            path.unlink(missing_ok=True)
        for path in reversed(owned_dirs):
            path.rmdir()
        stage.rmdir()


def main(argv=None):
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError('Immutable IO v2 base SHA changed')
    original_loader = base.load_pipeline

    def load_pipeline(path=base.FROZEN):
        pipeline = original_loader(path)
        original_save = pipeline.save

        def save(path, value):
            if Path(path).name == 'protocol.json':
                value = {**value, 'producer_io_publication_code': pipeline.ref(__file__),
                         'producer_io_base_sha256': BASE_SHA256,
                         'io_policy': 'native_roi_exclusive_directory_last_extract_receipt_commit_v3',
                         'whole_directory_atomic': False}
            return original_save(path, value)

        pipeline.save = save
        return pipeline

    base.load_pipeline = load_pipeline
    base.publish_roi = publish_roi
    return base.main(argv)


if __name__ == '__main__':
    try:
        print(json.dumps(main(), ensure_ascii=False), flush=True)
    except Exception as error:
        print(json.dumps({'status': 'worker_stopped', 'error': type(error).__name__ + ': ' + str(error),
                          'semantic_approval': False, 'formal_training': False}), file=sys.stderr, flush=True)
        sys.exit(2)
