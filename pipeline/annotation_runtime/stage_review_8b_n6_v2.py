"""Foreground main-node SSH transfer of exact N5 local files to exclusive N6 storage.

Never overwrites or resumes an existing target/receipt; partials remain for diagnosis.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import uuid

RT = Path(__file__).resolve().parent
ORIGIN = RT / 'reviewer8b_stage_TENCENT64.site.json'
ORIGIN_SHA = '76ed7d54737f787434ad8153a0910dfc8d024f146533e9b01443d0b3cfe2dd15'
RECEIPT = RT / 'reviewer8b_stage_N6_extension_v2.json'
SOURCE = '/opt/models/Qwen3-VL-8B-Instruct'
DEST = '/opt/models/Qwen3-VL-8B-Instruct-review-extension-v2'
N5 = '28.216.19.137'
N6 = '28.216.19.225'
PYTHON = '/opt/venv_eval/bin/python'


def raw(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


COMMON = r'''
import hashlib,json,os,pathlib,stat,subprocess,sys
cfg=json.loads(sys.argv[1])
def need(ok, message):
 if not ok: raise RuntimeError(message)
def snapshot(p):
 s=p.lstat()
 need(stat.S_ISREG(s.st_mode), 'not a regular file: '+str(p))
 return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,device=s.st_dev,inode=s.st_ino)
def digest(p):
 before=snapshot(p); h=hashlib.sha256()
 with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
  need(os.fstat(f.fileno()).st_ino==before['inode'], 'file replaced before open')
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
 need(snapshot(p)==before,'file changed during hashing: '+str(p))
 return h.hexdigest(),before
def directory(p):
 s=p.lstat();need(stat.S_ISDIR(s.st_mode),'symlink/non-directory root rejected')
 return dict(device=s.st_dev,inode=s.st_ino)
need(cfg['node'] in subprocess.check_output(['hostname','-I'],text=True).split(),'wrong physical node')
root=pathlib.Path(cfg['root'])
'''

SOURCE_VERIFY = COMMON + r'''
root_stat=directory(root)
need({p.name for p in root.iterdir() if not p.name.startswith('.')}==set(cfg['files']),'N5 visible inventory changed')
verified={}
for name,expected in cfg['files'].items():
 p=root/name
 h,s=digest(p)
 need(h==expected['sha256'] and s['size']==expected['bytes'],'N5 local SHA/bytes mismatch: '+name)
 verified[name]=s
need(directory(root)==root_stat,'N5 source root changed')
for name,s in verified.items():need(snapshot(root/name)==s,'N5 source changed after preverification: '+name)
print(json.dumps(dict(files=verified,root_stat=root_stat)))
'''

SOURCE_SEND = COMMON + r'''
need(directory(root)==cfg['root_stat'],'N5 source root replaced')
p=root/cfg['name'];before=snapshot(p)
need(before==cfg['stat'],'N5 source changed before streaming')
h=hashlib.sha256();total=0
with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
 need(os.fstat(f.fileno()).st_ino==before['inode'],'N5 file replaced before streaming')
 for b in iter(lambda:f.read(8*1024*1024),b''):
  sys.stdout.buffer.write(b);h.update(b);total+=len(b)
 sys.stdout.buffer.flush()
 if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
need(snapshot(p)==before and directory(root)==cfg['root_stat'],'N5 source changed during streaming')
need(total==cfg['bytes'] and h.hexdigest()==cfg['sha256'],'N5 streaming SHA/bytes mismatch')
'''

SOURCE_FINAL = COMMON + r'''
need(directory(root)==cfg['root_stat'],'N5 source root replaced')
need({p.name for p in root.iterdir() if not p.name.startswith('.')}==set(cfg['files']),'N5 final inventory changed')
for name,s in cfg['files'].items():need(snapshot(root/name)==s,'N5 source changed after transfer: '+name)
print(json.dumps(dict(stable=True)))
'''

DEST_CREATE = COMMON + r'''
directory(root.parent)
os.mkdir(root,0o700)
owner=root/'.review-extension-v2-owner.json'
with owner.open('xb') as f:
 f.write(json.dumps(cfg['owner'],sort_keys=True).encode());f.flush();os.fsync(f.fileno())
print(json.dumps(dict(root_stat=directory(root))))
'''

DEST_OWNED = COMMON + r'''
need(directory(root)==cfg['root_stat'],'N6 target root changed')
owner=root/'.review-extension-v2-owner.json'
snapshot(owner)
with os.fdopen(os.open(owner,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
 need(json.load(f)==cfg['owner'],'N6 target owner changed')
allowed=set(cfg['files'])|{owner.name}
need({p.name for p in root.iterdir()}<=allowed,'unknown N6 destination entries; never remove/overwrite')
'''

DEST_RECEIVE = DEST_OWNED + r'''
p=root/cfg['name'];tmp=root/('.'+cfg['name']+'.partial-'+cfg['owner']['token'])
need(not os.path.lexists(p),'existing N6 destination file refused')
h=hashlib.sha256();total=0
with tmp.open('xb') as f:
 while total<cfg['bytes']:
  b=sys.stdin.buffer.read(min(8*1024*1024,cfg['bytes']-total))
  need(bool(b),'interrupted stream; partial retained')
  f.write(b);h.update(b);total+=len(b)
 need(not sys.stdin.buffer.read(1),'stream exceeds expected bytes; partial retained')
 f.flush();os.fsync(f.fileno())
 if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
need(h.hexdigest()==cfg['sha256'],'stream SHA mismatch; partial retained')
os.link(tmp,p)
tmp.unlink()
print(json.dumps(dict(copied=cfg['name'],bytes=total)))
'''

DEST_FINAL = DEST_OWNED + r'''
need({p.name for p in root.iterdir()}==set(cfg['files'])|{owner.name},'N6 final inventory mismatch')
verified={}
for name,expected in cfg['files'].items():
 p=root/name;h,s=digest(p)
 need(h==expected['sha256'] and s['size']==expected['bytes'],'N6 final disk SHA/bytes mismatch: '+name)
 verified[name]=dict(path=str(p),sha256=h,stat=s)
for name,entry in verified.items():need(snapshot(root/name)==entry['stat'],'N6 changed during final full verification: '+name)
need(directory(root)==cfg['root_stat'],'N6 root changed during final full verification')
print(json.dumps(verified))
'''


def ssh(node, code, cfg):
    command = shlex.join([PYTHON, '-B', '-u', '-c', code, json.dumps(cfg)])
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=3', node, command]


def run(node, code, cfg):
    result = subprocess.run(ssh(node, code, cfg), check=True, stdout=subprocess.PIPE)
    return json.loads(result.stdout)


def main():
    if os.path.lexists(RECEIPT):
        raise RuntimeError('stage receipt already exists; never overwrite or retransfer')
    origin_bytes = ORIGIN.read_bytes()
    if hashlib.sha256(origin_bytes).hexdigest() != ORIGIN_SHA:
        raise RuntimeError('origin receipt SHA differs from exact approved manifest')
    origin = json.loads(origin_bytes)
    if origin.get('complete') is not True or origin.get('full_copy_SHA_verified') is not True or origin['destination'] != SOURCE:
        raise RuntimeError('verified local origin receipt required')
    files = origin['files']
    if not files or len([n for n in files if n.endswith('.safetensors')]) != 4:
        raise RuntimeError('unexpected exact origin inventory')
    for name, entry in files.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or entry['path'] != SOURCE + '/' + name:
            raise RuntimeError('unsafe or mismatched inventory path')
        if not re.fullmatch(r'[0-9a-f]{64}', entry['sha256']) or type(entry['bytes']) is not int or entry['bytes'] < 0:
            raise RuntimeError('invalid inventory SHA/bytes')
    expected = {n: dict(sha256=e['sha256'], bytes=e['bytes']) for n, e in files.items()}
    source_cfg = dict(node=N5, root=SOURCE, files=expected)
    fresh = run(N5, SOURCE_VERIFY, source_cfg)
    print(json.dumps(dict(N5_fresh_SHA_verified=True, file_count=len(expected))), flush=True)
    owner = dict(token=uuid.uuid4().hex, origin_sha256=ORIGIN_SHA, source=N5, destination=N6)
    dest_cfg = dict(node=N6, root=DEST, files=expected, owner=owner)
    dest_cfg.update(run(N6, DEST_CREATE, dest_cfg))
    for name, entry in expected.items():
        send_cfg = dict(source_cfg, name=name, stat=fresh['files'][name], root_stat=fresh['root_stat'], **entry)
        recv_cfg = dict(dest_cfg, name=name, **entry)
        # Both SSH children stay in the foreground task's process tree; no detach/signals.
        with subprocess.Popen(ssh(N5, SOURCE_SEND, send_cfg), stdout=subprocess.PIPE) as sender:
            try:
                received = subprocess.run(ssh(N6, DEST_RECEIVE, recv_cfg), stdin=sender.stdout,
                                          stdout=subprocess.PIPE)
            finally:
                sender.stdout.close()
            source_rc = sender.wait()
        if received.returncode or source_rc:
            raise RuntimeError('stream failed; exclusive destination/partials retained: ' + name)
        print(received.stdout.decode().strip(), flush=True)
    stable_cfg = dict(source_cfg, root_stat=fresh['root_stat'], files=fresh['files'])
    run(N5, SOURCE_FINAL, stable_cfg)
    verified = run(N6, DEST_FINAL, dest_cfg)
    run(N5, SOURCE_FINAL, stable_cfg)
    if ORIGIN.read_bytes() != origin_bytes:
        raise RuntimeError('origin receipt changed during transfer')
    content = {n: dict(path=e['path'], sha256=e['sha256']) for n, e in verified.items()}
    identity = dict(root=DEST, files=verified,
                    sha256=hashlib.sha256(raw(dict(root=DEST, files=content))).hexdigest(),
                    full_content_hash_computed_once_per_process=True)
    value = dict(schema='reviewer8b_stage_N6_extension_v2', node='N6', complete=True,
                 model_identity=identity, origin_ref=dict(path=str(ORIGIN), sha256=ORIGIN_SHA),
                 origin_verified_node='N5', origin_local_model_root=SOURCE,
                 fresh_N5_local_file_SHA_verified=True, source_fresh_stat=fresh,
                 destination_exclusive_owner=owner, exact_file_count=len(files),
                 source_shared_weights_rehashed=False, source_shared_SHA_attestation_carried_from_origin=True,
                 source_and_destination_SHA_verified=True, destination_disk_SHA_rehashed=True,
                 full_copy_SHA_verified=True, GPU_model_loaded=False)
    with RECEIPT.open('xb') as f:
        f.write(raw(value));f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(complete=True, receipt=str(RECEIPT),
                         sha256=hashlib.sha256(raw(value)).hexdigest(),
                         model_identity_sha256=identity['sha256'])), flush=True)


if __name__ == '__main__':
    main()
