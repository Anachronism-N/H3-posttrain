"""CPU preflight and original-domain claim primitive, not a production launcher.
Central retirement/release trust and fresh local resource checks remain external.
"""
import hashlib
import json
import os
import re
from pathlib import Path


class ConsumptionUncertain(RuntimeError):
    pass


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def verify_boundary(manifest_bytes, release_bytes, trusted, original_rows):
    if not trusted or set(trusted) != {'manifest_sha256', 'release_sha256', 'predecessor_epoch', 'next_epoch', 'original_claim_root', 'donor_assignment_sha256'}:
        raise PermissionError('Independent central expected boundary required')
    for raw, expected in [(manifest_bytes, trusted['manifest_sha256']), (release_bytes, trusted['release_sha256'])]:
        if not isinstance(raw, bytes) or len(raw) > 32 * 1024**2 or hashlib.sha256(raw).hexdigest() != expected:
            raise PermissionError('Independently anchored evidence SHA mismatch')
    manifest = json.loads(manifest_bytes, object_pairs_hook=_unique_pairs)
    release = json.loads(release_bytes, object_pairs_hook=_unique_pairs)
    if trusted['next_epoch'] == trusted['predecessor_epoch'] or manifest['next_epoch'] != trusted['next_epoch']:
        raise PermissionError('Independent successor epoch required')
    if release['predecessor_epoch'] != trusted['predecessor_epoch'] or manifest['predecessor_epoch'] != trusted['predecessor_epoch']:
        raise PermissionError('Predecessor binding mismatch')
    if release['original_claim_root'] != trusted['original_claim_root'] or manifest['original_claim_root'] != trusted['original_claim_root']:
        raise PermissionError('Original claim domain cannot be replaced')
    if release['donor_assignment_sha256'] != trusted['donor_assignment_sha256']:
        raise PermissionError('Original donor assignment SHA mismatch')
    if release.get('durable_enforcement_receipt_verified') is not True or not release.get('enforcement_evidence_ref'):
        raise PermissionError('Central durable retirement enforcement evidence required')
    if manifest.get('automatic_retry') is not False or manifest.get('old_queue_append') is not False:
        raise PermissionError('Old queue append or automatic retry forbidden')
    rows = manifest['rows']
    seen = set()
    for index, item in enumerate(rows):
        worker, original_index = item['original_worker'], item['original_assignment_index']
        if worker not in trusted['donor_assignment_sha256'] or type(original_index) is not int or original_index < 0:
            raise PermissionError('Original source owner/index missing')
        row = item['source_row']
        key = row['split'] + '/' + row['id']
        if key in seen or row != original_rows[worker][original_index]:
            raise PermissionError('Duplicate or nonexact original source row')
        if item['receiver'] not in ['a100-extra-01', 'a100-extra-02']:
            raise PermissionError('Receiver outside transferred scope')
        if item.get('claim_absence_verified_after_retirement') is not True:
            raise PermissionError('Fresh absence after retirement required')
        seen.add(key)
    if len(rows) != manifest['source_count']:
        raise PermissionError('Frozen source count mismatch')
    return {'structural_boundary_verified': True, 'source_count': len(rows), 'production_permission': False,
            'fresh_actual_durable_enforcement_and_local_resource_checks_required': True}


def consume_original_claim(claim_root, source_row, record, *, write=os.write, fsync=os.fsync):
    split, identity = source_row['split'], source_row['id']
    if not all(isinstance(v, str) and re.fullmatch('[A-Za-z0-9_-]+', v) for v in [split, identity]):
        raise ValueError('Unsafe source key')
    if record.get('source_row') != source_row or record.get('automatic_retry') is not False:
        raise PermissionError('Claim source binding/no-retry required')
    raw = (json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    rootfd = os.open(claim_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    directory = None
    descriptor = None
    created = False
    try:
        directory = os.open(split, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=rootfd)
        descriptor = os.open(identity + '.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
        created = True
        offset = 0
        while offset < len(raw):
            count = write(descriptor, raw[offset:])
            if count <= 0:
                raise OSError('Claim write made no progress')
            offset += count
        fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        fsync(directory)
        fd = os.open(identity + '.json', os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        try:
            with os.fdopen(fd, 'rb') as stream:
                verified = stream.read()
        except BaseException:
            raise
        if verified != raw:
            raise OSError('Claim readback mismatch')
        return {'source_key': split + '/' + identity, 'claim_sha256': hashlib.sha256(raw).hexdigest(),
                'claim_bytes': len(raw), 'created_exactly_once': True, 'inference_started': False}
    except BaseException as error:
        if created:
            raise ConsumptionUncertain('Original claim retained; no replay, reset, or inference') from error
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory is not None:
            os.close(directory)
        os.close(rootfd)
