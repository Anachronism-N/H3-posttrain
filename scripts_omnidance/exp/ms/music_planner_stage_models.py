import argparse
import fcntl
import hashlib
import json
import os
import shutil
import socket
import stat
import time
from pathlib import Path

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
EXPERIMENT = ROOT / 'diff/logs_omnidance/exp/MS/music_planner_fixed2s_training_20261024_v1'
MODEL_BASE = Path('/opt/music_planner_models')
AUTHORIZED = {
    'N2': ('28.216.17.70', 'music_flamingo'),
    'N4': ('28.216.19.143', 'omni_captioner'),
    'N7': ('28.216.18.144', 'moss'),
}
MODELS = {
    'moss': ('OpenMOSS-Team/MOSS-Music-8B-Instruct', 'fce7f8304e96cc2d3398b8106456cbb2ecec3139', 'MOSS-Music-8B-Instruct'),
    'music_flamingo': ('nvidia/music-flamingo-2601-hf', '6b5be086d52f65a1e204cb0faf70bf54e2741ecd', 'music-flamingo-2601-hf'),
    'omni_captioner': ('Qwen/Qwen3-Omni-30B-A3B-Captioner', 'a2bd106cbf527db5676e79662674da22b0545ec0', 'Qwen3-Omni-30B-A3B-Captioner'),
}
SAFETY = 80 * 2**30
ADMISSION = 160 * 2**30
BLOCK = 8 * 2**20


class WaitingWeights(RuntimeError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            h.update(block)
        drop_cache(stream)
    return h.hexdigest()


def drop_cache(stream):
    if hasattr(os, 'posix_fadvise'):
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def headroom():
    p = Path('/sys/fs/cgroup/memory')
    current = int((p / 'memory.current').read_text())
    maximum = (p / 'memory.max').read_text().strip()
    available = int(next(x for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:')).split()[1]) * 1024
    cap = int(maximum) if maximum != 'max' else current + available
    return dict(current=current, maximum=maximum, headroom=min(cap-current, available))


def validate_scope(node, family, host=None):
    if AUTHORIZED.get(node) != (host or AUTHORIZED.get(node, ('', ''))[0], family):
        raise RuntimeError('outside authorized node/family/host scope')
    expected = AUTHORIZED[node][0]
    hostname = socket.gethostname()
    local_addresses = {row[4][0] for row in socket.getaddrinfo(hostname, None, socket.AF_INET)}
    if expected not in local_addresses:
        raise RuntimeError('manager must execute on authorized host, not coordinator')


def inventory(receipt, family):
    repo, revision, dirname = MODELS[family]
    source = ROOT / 'models/music_planner_official' / dirname
    if (receipt.get('family') != family or receipt.get('repo_id') != repo
            or receipt.get('revision') != revision or receipt.get('destination') != str(source)):
        raise RuntimeError('download receipt not the pinned official model')
    if receipt.get('status') not in ('complete', 'completed') or receipt.get('complete') is False:
        raise WaitingWeights('download is not complete')
    files = {}
    for row in receipt['files_expected']:
        name = row['rfilename']
        rel = Path(name)
        if (not isinstance(name, str) or not name or rel.is_absolute()
                or any(part in ('..', '.') or part.startswith('.') for part in rel.parts)):
            raise RuntimeError('unsafe model manifest path')
        if name in files:
            raise RuntimeError('duplicate model manifest path')
        size = row['size']
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise RuntimeError('invalid manifest file size')
        lfs = row.get('lfs')
        digest = lfs['sha256'] if lfs else row['blobId']
        length = 64 if lfs else 40
        if len(digest) != length or any(c not in '0123456789abcdef' for c in digest):
            raise RuntimeError('invalid pinned model digest')
        if lfs and lfs['size'] != size:
            raise RuntimeError('LFS length mismatch')
        files[name] = dict(size=size, sha256=digest if lfs else None, git_blob=digest if not lfs else None)
    if 'config.json' not in files or not any(name.endswith('.safetensors') for name in files):
        raise RuntimeError('incomplete model inventory')
    for name, expected in files.items():
        path = source / name
        if not path.is_file() or path.stat().st_size != expected['size']:
            raise WaitingWeights('model file missing or partial: ' + name)
    index = source / 'model.safetensors.index.json'
    if 'model.safetensors.index.json' in files:
        shards = set(json.loads(index.read_text())['weight_map'].values())
        if not shards or not shards.issubset(files):
            raise RuntimeError('weight index names missing shards')
    return source, files


def read_download(modelref, family):
    path = Path(modelref['path'])
    if path.resolve() != (EXPERIMENT / (family + '_download.json')).resolve():
        raise RuntimeError('download receipt outside model experiment')
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise WaitingWeights('download receipt absent') from error
    digest = hashlib.sha256(raw).hexdigest()
    if modelref.get('sha256') and digest != modelref['sha256']:
        raise RuntimeError('download receipt changed from final pinned reference')
    receipt = json.loads(raw)
    source, files = inventory(receipt, family)
    return path, digest, receipt, source, files


def file_stat(path):
    st = path.stat()
    return dict(size=st.st_size, device=st.st_dev, inode=st.st_ino, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


def copy_verified(source, dest, expected):
    before = file_stat(source)
    if before['size'] != expected['size']:
        raise RuntimeError('source length changed')
    h, git = hashlib.sha256(), hashlib.sha1()
    git.update(('blob %d\0' % before['size']).encode())
    with source.open('rb') as reader, dest.open('xb') as writer:
        for block in iter(lambda: reader.read(BLOCK), b''):
            h.update(block)
            git.update(block)
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
        drop_cache(reader)
        drop_cache(writer)
    digest = h.hexdigest()
    if (expected['sha256'] and digest != expected['sha256']) or (expected['git_blob'] and git.hexdigest() != expected['git_blob']):
        raise RuntimeError('official source content digest mismatch: ' + str(source))
    if file_stat(source) != before:
        raise RuntimeError('source changed during copy')
    if sha(dest) != digest:
        raise RuntimeError('local copied file full SHA256 mismatch')
    return dict(sha256=digest, size=before['size'], source_stat=before, local_stat=file_stat(dest))


def check_staged(marker, download_sha, family, files, destination):
    if (marker.get('complete') is not True or marker.get('family') != family
            or marker.get('revision') != MODELS[family][1]
            or marker.get('download_receipt_sha256') != download_sha
            or marker.get('destination') != str(destination) or set(marker['files']) != set(files)):
        raise RuntimeError('staged model receipt binding mismatch')
    actual = {str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()}
    if actual != set(files):
        raise RuntimeError('staged whole-model inventory differs')
    for name, row in marker['files'].items():
        path = destination / name
        if path.is_symlink() or file_stat(path) != row['local_stat'] or row['size'] != files[name]['size']:
            raise RuntimeError('staged file changed after full verification: ' + name)
        if files[name]['sha256'] and row['sha256'] != files[name]['sha256']:
            raise RuntimeError('staged LFS digest not official')
    # Only small metadata is rehashed at launch; weights were fully verified at copy.
    for name in files:
        if not name.endswith('.safetensors'):
            if sha(destination / name) != marker['files'][name]['sha256']:
                raise RuntimeError('staged model code/config changed: ' + name)
    return marker


def stage_model(node, family, modelref, progress=None):
    validate_scope(node, family)
    receipt_path, digest, receipt, source, files = read_download(modelref, family)
    destination = MODEL_BASE / family
    marker_path = Path(str(destination) + '.stage_receipt.json')
    MODEL_BASE.mkdir(parents=True, exist_ok=True)
    if MODEL_BASE.is_symlink() or destination.is_symlink() or marker_path.is_symlink():
        raise RuntimeError('unsafe local staging path')
    lock_path = MODEL_BASE / (family + '.stage.lock')
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_size:
            raise RuntimeError('unsafe local model stage lock')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if marker_path.exists():
            return check_staged(json.loads(marker_path.read_text()), digest, family, files, destination)
        if headroom()['headroom'] < ADMISSION:
            raise RuntimeError('local stage headroom below 160 GiB; no GPU holder released')
        required = sum(v['size'] for v in files.values()) + 8 * 2**30
        if shutil.disk_usage(MODEL_BASE).free < required:
            raise RuntimeError('insufficient local disk; no cache/model deletion permitted')
        destination.mkdir(exist_ok=False)
        copied = {}
        for name, expected in files.items():
            if headroom()['headroom'] < SAFETY:
                raise RuntimeError('stage memory safety floor crossed; partial files preserved')
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            copied[name] = copy_verified(source / name, path, expected)
            if progress:
                progress('staging_weights', dict(file=name, copied_files=len(copied), total_files=len(files)))
        if sha(receipt_path) != digest:
            raise RuntimeError('download receipt changed during stage')
        marker = dict(complete=True, family=family, repo_id=receipt['repo_id'], revision=receipt['revision'],
                      node=node, source=str(source), destination=str(destination),
                      download_receipt=str(receipt_path), download_receipt_sha256=digest,
                      files=copied, single_file_concurrency=1, full_source_and_local_sha_verified=True,
                      completed_unix=time.time())
        with marker_path.open('x') as stream:
            json.dump(marker, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        return check_staged(marker, digest, family, files, destination)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--node', required=True, choices=AUTHORIZED)
    ap.add_argument('--family', required=True, choices=MODELS)
    ap.add_argument('--download-receipt', required=True)
    ap.add_argument('--download-receipt-sha')
    ap.add_argument('--wait-seconds', type=float, default=3600)
    ap.add_argument('--poll-seconds', type=float, default=10)
    args = ap.parse_args()
    validate_scope(args.node, args.family)
    def progress(status, value):
        try:
            print(json.dumps(dict(status=status, **value)), flush=True)
        except OSError:
            pass
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            marker = stage_model(args.node, args.family,
                                 dict(path=args.download_receipt, sha256=args.download_receipt_sha), progress)
            progress('staged', dict(receipt=str(MODEL_BASE / (args.family + '.stage_receipt.json')),
                                    files=len(marker['files']), training_started=False))
            return
        except WaitingWeights as error:
            progress('waiting_weights', dict(reason=str(error), no_holder_released=True, training_started=False))
            if time.monotonic() >= deadline:
                raise RuntimeError('bounded weights wait expired; no GPU holder released') from error
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
