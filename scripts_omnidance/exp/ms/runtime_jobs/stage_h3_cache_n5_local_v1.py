import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

SOURCE = Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/h3_ms_style_pilot_cache512_v2')
PARENT = Path('/opt/h3cache')
STAGE = PARENT / 'ms_pilot512_stage_20261023_v1'
DESTINATION = PARENT / 'ms_pilot512_local_20261023_v1'
RECEIPT = PARENT / 'ms_pilot512_local_20261023_v1_verified.json'
MARKER_SHA = '6eba363545c7717155a141755d1c47ed03f1659d028d12305f27d8db6f0e30dc'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    marker_bytes = (SOURCE / 'complete.json').read_bytes()
    assert hashlib.sha256(marker_bytes).hexdigest() == MARKER_SHA
    marker = json.loads(marker_bytes)
    assert marker['scope'] == 'source_label_train' and marker['signature']['dataset_size'] == 512
    files = marker['files']
    assert len([name for name in files if name.endswith('.pth')]) == 512
    for name in files:
        p = Path(name)
        assert not p.is_absolute() and '..' not in p.parts
    required = sum(info['bytes'] for info in files.values())
    assert shutil.disk_usage('/opt').free > required + 10 * 2**30
    assert not STAGE.exists() and not DESTINATION.exists() and not RECEIPT.exists()
    PARENT.mkdir(exist_ok=True)
    STAGE.mkdir()
    print(json.dumps({'event': 'copy_started', 'source': str(SOURCE), 'destination': str(DESTINATION), 'bytes': required, 'GPU_count': 0}), flush=True)
    subprocess.run(['scp', '-O', '-rp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                    '28.216.19.213:' + str(SOURCE), str(STAGE)], check=True)
    local = STAGE / SOURCE.name
    assert (local / 'complete.json').read_bytes() == marker_bytes
    assert not any(p.is_symlink() for p in local.rglob('*'))
    actual_files = {str(p.relative_to(local)) for p in local.rglob('*') if p.is_file()}
    assert actual_files == set(files) | {'complete.json'}
    for index, (name, expected) in enumerate(sorted(files.items()), 1):
        p = local / name
        before = p.stat()
        assert before.st_size == expected['bytes'] and digest(p) == expected['sha256']
        after = p.stat()
        assert (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino)
        if index % 32 == 0 or index == len(files):
            print(json.dumps({'event': 'local_SHA_verified', 'files': index, 'total': len(files), 'GPU_count': 0}), flush=True)
    assert (SOURCE / 'complete.json').read_bytes() == marker_bytes
    assert not DESTINATION.exists()
    local.rename(DESTINATION)
    for p in DESTINATION.rglob('*'):
        if p.is_file():
            p.chmod(0o444)
    for p in sorted([p for p in DESTINATION.rglob('*') if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
        p.chmod(0o555)
    DESTINATION.chmod(0o555)
    report = {'complete': True, 'source': str(SOURCE), 'destination': str(DESTINATION), 'marker_sha256': MARKER_SHA,
              'verified_file_count': len(files), 'samples': 512, 'verified_bytes': required, 'copied_exact_bytes': True,
              'source_code_sha256': digest(Path(__file__)), 'GPU_count': 0, 'immutable_local_copy': True}
    with RECEIPT.open('x') as f:
        json.dump(report, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({'event': 'local_cache_ready', **report}), flush=True)

if __name__ == '__main__':
    main()
