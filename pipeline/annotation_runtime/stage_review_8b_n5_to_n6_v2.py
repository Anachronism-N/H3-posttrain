"""N5 verified LOCAL 8B -> exclusive N6 new directory; no shared weight reads."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

RT = Path(__file__).resolve().parent
ORIGIN = RT / 'reviewer8b_stage_TENCENT64.site.json'
RECEIPT = RT / 'reviewer8b_stage_N6_extension_v2.json'
SOURCE = Path('/opt/models/Qwen3-VL-8B-Instruct')
DEST = '/opt/models/Qwen3-VL-8B-Instruct-review-extension-v2'
N6 = '28.216.19.225'


def raw(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def digest(path):
    h = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino, before.st_dev) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino, after.st_dev):
        raise RuntimeError('local source changed during SHA')
    return h.hexdigest()


REMOTE = r'''
import hashlib,json,os,pathlib,sys,subprocess
cfg=json.loads(sys.argv[1])
if '28.216.19.225' not in subprocess.check_output(['hostname','-I'],text=True).split():
 raise RuntimeError('wrong target node')
root=pathlib.Path(cfg['root'])
if root.is_symlink():raise RuntimeError('symlink target rejected')
root.mkdir(parents=True,exist_ok=True)
if {p.name for p in root.iterdir() if not p.name.startswith('.')} - set(cfg['inventory']):
 raise RuntimeError('unknown target files; never remove/overwrite')
p=root/cfg['name']
if p.exists():raise RuntimeError('target already exists; no-replace transfer')
tmp=root/('.'+cfg['name']+'.transfer-v2')
h=hashlib.sha256();total=0
with tmp.open('xb') as f:
 while True:
  block=sys.stdin.buffer.read(8*1024*1024)
  if not block:break
  f.write(block);h.update(block);total+=len(block)
 f.flush();os.fsync(f.fileno())
 if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
if total!=cfg['bytes'] or h.hexdigest()!=cfg['sha256']:raise RuntimeError('stream bytes/SHA mismatch')
h=hashlib.sha256()
with tmp.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
if h.hexdigest()!=cfg['sha256']:raise RuntimeError('target disk SHA mismatch')
os.link(tmp,p);tmp.unlink();s=p.stat()
print(json.dumps(dict(path=str(p),sha256=h.hexdigest(),stat=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,device=s.st_dev,inode=s.st_ino))))
'''


def main():
    if '28.216.19.137' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('fast transfer must run on N5 local verified model node')
    origin_raw = ORIGIN.read_bytes()
    origin_sha = hashlib.sha256(origin_raw).hexdigest()
    origin = json.loads(origin_raw)
    if not origin.get('complete') or not origin.get('full_copy_SHA_verified'):
        raise RuntimeError('verified original N5 receipt required')
    inventory = origin['files']
    if len([n for n in inventory if n.endswith('.safetensors')]) != 4:
        raise RuntimeError('unexpected N5 8B inventory')
    if {p.name for p in SOURCE.iterdir() if p.is_file() and not p.name.startswith('.')} != set(inventory):
        raise RuntimeError('N5 local model inventory changed')
    if RECEIPT.exists():
        raise RuntimeError('new stage receipt already exists; do not transfer twice')
    verified = {}
    for name, expected in inventory.items():
        source = SOURCE / name
        st = source.stat()
        if source.is_symlink() or expected['path'] != str(source) or st.st_size != expected['bytes']:
            raise RuntimeError('N5 source identity differs from prior receipt')
        if (st.st_mtime_ns, st.st_ino, st.st_dev) != (expected['mtime_ns'], expected['inode'], expected['device']):
            raise RuntimeError('N5 source stat differs from prior receipt')
        local_sha = digest(source)
        if local_sha != expected['sha256']:
            raise RuntimeError('fresh N5 LOCAL SHA differs from verified origin: ' + name)
        cfg = dict(root=DEST, name=name, inventory=list(inventory), bytes=st.st_size, sha256=local_sha)
        command = '/opt/venv_eval/bin/python -B -c ' + shlex.quote(REMOTE) + ' ' + shlex.quote(json.dumps(cfg))
        with source.open('rb') as f:
            result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', N6, command],
                                    stdin=f, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        entry = json.loads(result.stdout)
        if entry['sha256'] != local_sha or entry['path'] != DEST + '/' + name or entry['stat']['size'] != st.st_size:
            raise RuntimeError('target verifier returned different file identity')
        after = source.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino, after.st_dev) != (
                st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev):
            raise RuntimeError('N5 source changed during transfer')
        verified[name] = entry
        print(json.dumps(dict(verified_file=name, bytes=st.st_size, source='N5_local', target='N6_new_directory')), flush=True)
    if hashlib.sha256(ORIGIN.read_bytes()).hexdigest() != origin_sha:
        raise RuntimeError('origin receipt changed during transfer')
    content = {name: dict(path=v['path'], sha256=v['sha256']) for name, v in verified.items()}
    identity = dict(root=DEST, files=verified, sha256=hashlib.sha256(raw(dict(root=DEST, files=content))).hexdigest(),
                    full_content_hash_computed_once_per_process=True)
    value = dict(schema='reviewer8b_stage_N6_extension_v2', node='N6', complete=True, model_identity=identity,
        origin_ref=dict(path=str(ORIGIN), sha256=origin_sha), origin_verified_node='N5',
        origin_local_model_root=str(SOURCE), fresh_N5_local_file_SHA_verified=True,
        source_shared_weights_rehashed=False, source_shared_SHA_attestation_carried_from_origin=True,
        source_and_destination_SHA_verified=True, destination_disk_SHA_rehashed=True,
        full_copy_SHA_verified=True, GPU_model_loaded=False)
    with RECEIPT.open('xb') as f:
        f.write(raw(value));f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(complete=True, receipt=str(RECEIPT), sha256=digest(RECEIPT), model_identity_sha256=identity['sha256'])), flush=True)


if __name__ == '__main__':
    main()
