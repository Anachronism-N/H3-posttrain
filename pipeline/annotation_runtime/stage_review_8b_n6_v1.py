"""N6-only, no GPU allocation: copy 8B with source/destination SHA verification."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

RT = Path(__file__).resolve().parent
SOURCE = Path('/apdcephfs_gy4/share_302533218/cedricnie/models/Qwen3-VL-8B-Instruct')
DEST = Path('/opt/models/Qwen3-VL-8B-Instruct')
RECEIPT = RT / 'reviewer8b_stage_N6_extension_v1.json'


def raw(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def snapshot(p):
    s = p.stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                device=s.st_dev, inode=s.st_ino)


def digest(p):
    before = snapshot(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    if snapshot(p) != before:
        raise RuntimeError('model file changed while hashing: ' + str(p))
    return h.hexdigest()


def main():
    if '28.216.19.225' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('8B extension staging is N6-only')
    files = sorted(p for p in SOURCE.iterdir() if p.is_file() and not p.name.startswith('.'))
    if len([p for p in files if p.suffix == '.safetensors']) != 4 or DEST.is_symlink():
        raise RuntimeError('unexpected source inventory or symlink destination')
    DEST.mkdir(parents=True, exist_ok=True)
    if {p.name for p in DEST.iterdir() if not p.name.startswith('.')} - {p.name for p in files}:
        raise RuntimeError('unknown files in local model; never delete/overwrite')
    inventory = {}
    for source in files:
        target = DEST / source.name
        expected = digest(source)
        if target.exists():
            if target.is_symlink() or target.stat().st_size != source.stat().st_size or digest(target) != expected:
                raise RuntimeError('existing target differs; never overwrite: ' + str(target))
        else:
            # Exclusive temporary files retain interrupted copies for diagnosis.
            temporary = DEST / ('.' + source.name + '.extension-copy')
            with source.open('rb') as src, temporary.open('xb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
                if hasattr(os, 'posix_fadvise'):
                    os.posix_fadvise(src.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    os.posix_fadvise(dst.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            if digest(temporary) != expected:
                raise RuntimeError('new target SHA mismatch')
            os.link(temporary, target)
            temporary.unlink()
        inventory[source.name] = dict(path=str(target), sha256=expected, stat=snapshot(target))
        print(json.dumps(dict(verified_file=source.name, bytes=target.stat().st_size)), flush=True)
    content = {name: dict(path=v['path'], sha256=v['sha256']) for name, v in inventory.items()}
    identity = dict(root=str(DEST), files=inventory,
                    sha256=hashlib.sha256(raw(dict(root=str(DEST), files=content))).hexdigest(),
                    full_content_hash_computed_once_per_process=True)
    value = dict(schema='reviewer8b_stage_N6_extension_v1', node='N6', complete=True,
                 model_identity=identity, source_path=str(SOURCE), full_copy_SHA_verified=True,
                 source_and_destination_SHA_verified=True, GPU_model_loaded=False)
    if RECEIPT.exists():
        if json.loads(RECEIPT.read_text()) != value:
            raise RuntimeError('existing N6 receipt differs; never overwrite')
    else:
        with RECEIPT.open('xb') as f:
            f.write(raw(value))
    print(json.dumps(dict(complete=True, receipt=str(RECEIPT), sha256=digest(RECEIPT))), flush=True)


if __name__ == '__main__':
    main()
