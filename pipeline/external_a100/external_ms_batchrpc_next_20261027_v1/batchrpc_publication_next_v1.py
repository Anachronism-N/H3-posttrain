"""Publication-only plans: immutable bytes, fixed target, no model/claim callbacks."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile

CHUNK = 4 * 2**20

def encoded(value):
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(CHUNK),b''): h.update(chunk)
    return h.hexdigest()

def save_once(path,raw):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.plan-',delete=False) as f:
            temporary=Path(f.name); f.write(raw); f.flush(); os.fsync(f.fileno())
        try: os.link(temporary,path,follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes()!=raw: raise ValueError('publication plan write-once conflict')
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)

def prepare(local_batch,target,commit):
    local_batch,target=Path(local_batch),Path(target)
    paths=sorted([p for group in ('cases','diagnostics') for p in (local_batch/group).glob('*/*/*') if p.is_file()]+[local_batch/'batch.tar'],key=lambda p:str(target/p.relative_to(local_batch)))
    manifest=[]; local_files={}
    for p in paths:
        if p.is_symlink() or p.resolve()!=p: raise ValueError('redirected local publication member')
        m=dict(path=str(target/p.relative_to(local_batch)),bytes=p.stat().st_size,sha256=sha(p))
        manifest.append(m); local_files[m['path']]=str(p)
    c=dict(commit); c['publications']=[dict(path=m['path'],sha256=m['sha256'],receipt=m) for m in manifest]
    c['batch_sha256']=next(m['sha256'] for m in manifest if m['path']==str(target/'batch.tar'))
    for r in c['receipts']:
        if r.get('diagnostic_publications'):
            by_path={m['path']:m for m in manifest}
            r['diagnostic_publications']=[by_path[ref['path']] for ref in r['diagnostic_publications']]
    plan=dict(schema='ms_batchrpc_publication_plan_v1',target=str(target),manifest=manifest,commit=c,commit_sha256=hashlib.sha256(encoded(c)).hexdigest())
    saved=dict(schema='ms_batchrpc_local_prepared_v1',plan=plan,local_files=local_files)
    save_once(local_batch/'publication_plan.json',encoded(saved))
    return saved

class FileSequence:
    def __init__(self,manifest,local_files):
        self.manifest,self.files=manifest,local_files; self.index=0; self.stream=None; self.before=None; self.count=0; self.h=None
    def read(self,size):
        if type(size) is not int or not 0<size<=CHUNK: raise ValueError('unbounded publication read forbidden')
        while self.index<len(self.manifest):
            m=self.manifest[self.index]; p=Path(self.files[m['path']])
            if self.stream is None:
                if p.resolve()!=p or not p.is_file(): raise ValueError('local source redirected/missing')
                self.stream=p.open('rb'); self.before=os.fstat(self.stream.fileno()); self.count=0; self.h=hashlib.sha256()
                if self.before.st_size!=m['bytes']: raise ValueError('local member size changed')
            chunk=self.stream.read(min(size,m['bytes']-self.count))
            if chunk: self.count+=len(chunk); self.h.update(chunk); return chunk
            after=os.fstat(self.stream.fileno()); fields=('st_size','st_mtime_ns','st_ctime_ns','st_dev','st_ino')
            if self.count!=m['bytes'] or self.stream.read(1) or self.h.hexdigest()!=m['sha256'] or any(getattr(self.before,k)!=getattr(after,k) or getattr(after,k)!=getattr(p.stat(),k) for k in fields): raise ValueError('local bytes/source changed')
            self.stream.close(); self.stream=None; self.index+=1
        return b''
    def close(self):
        if self.stream is not None: self.stream.close(); self.stream=None

def recover(access,saved):
    plan=saved['plan']; state=access.inspect_plan(plan)
    if state['state']=='committed_verified': return dict(state='already_committed_verified',verification=state)
    # All source bytes must still exist and match the frozen prepared plan.
    for m in plan['manifest']:
        p=Path(saved['local_files'][m['path']])
        if p.resolve()!=p or p.stat().st_size!=m['bytes'] or sha(p)!=m['sha256']: raise ValueError('prepared source missing/changed; no generation or claim reset')
    for m in plan['manifest']:
        if m['path'] not in state['existing']: access.publish_file(saved['local_files'][m['path']],m['path'],m['sha256'])
    # Verify members/archive through the batch contract BEFORE commit-last.
    access.inspect_members(plan)
    access.publish_json(plan['commit'],str(Path(plan['target'])/'commit.json'))
    return dict(state='publication_only_recovered_verified',verification=access.verify_plan(plan))

def publish(access,saved):
    state=access.inspect_plan(saved['plan'])
    if state['state']=='committed_verified': return dict(state='already_committed_verified',verification=state)
    stream=FileSequence(saved['plan']['manifest'],saved['local_files'])
    try:
        access.batch_exchange(saved['plan'],stream)
    except (OSError,RuntimeError,ValueError):
        # A lost acknowledgement is not permission to regenerate. Existing commit first.
        return recover(access,saved)
    finally: stream.close()
    return dict(state='batch_committed_verified',verification=access.verify_plan(saved['plan']))
