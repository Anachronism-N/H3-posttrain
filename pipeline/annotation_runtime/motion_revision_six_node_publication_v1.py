"""Local durable results and one crash-resumable metadata uploader per node."""
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import time


def require(ok,message):
    if not ok: raise ValueError(message)


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':'))+chr(10)).encode()


def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    require(path.is_absolute() and path.parent.resolve()==path.parent,'Stable absolute publication parent required')
    raw=encoded(value);fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w+b') as f:
        f.write(raw);f.flush();os.fsync(f.fileno());df=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.fsync(df)
        finally:os.close(df)
        f.seek(0);require(f.read()==raw,'Durable local readback differs')
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def checked(ref):
    p=Path(ref['path']);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        s=os.fstat(f.fileno());require(stat.S_ISREG(s.st_mode),'Regular frozen file required');raw=f.read()
    require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Frozen file changed')
    if 'bytes' in ref: require(len(raw)==ref['bytes'],'Frozen size changed')
    return json.loads(raw)


def rename_noreplace(source,target):
    library=ctypes.CDLL(None,use_errno=True)
    fn=getattr(library,'renameat2',None)
    if fn is None:raise RuntimeError('Kernel no-clobber rename is unavailable; publication remains local')
    fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];fn.restype=ctypes.c_int
    if fn(-100,os.fsencode(source),-100,os.fsencode(target),1)!=0:
        e=ctypes.get_errno()
        if e==errno.EEXIST:raise FileExistsError(e,os.strerror(e),str(target))
        if e in (errno.EINVAL,errno.ENOSYS,errno.EOPNOTSUPP):
            source=Path(source);target=Path(target)
            require(source.parent==target.parent and source.parent.resolve()==source.parent and source.is_file() and not source.is_symlink(),'Stable same-directory regular no-clobber publication required')
            os.link(source,target,follow_symlinks=False)
            df=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:os.fsync(df)
            finally:os.close(df)
            source.unlink()
            return
        raise OSError(e,os.strerror(e),str(target))


class LocalOutbox:
    def __init__(self,local_root,shared_root,node):
        self.local=Path(local_root);self.shared=Path(shared_root);self.node=node
        require(self.local.is_absolute() and self.shared.is_absolute() and node in tuple('a100-extra-0'+str(i) for i in range(1,7)),'Explicit local/shared roots and node required')
        require(str(self.local).startswith(('/root/','/tmp/')),'GPU work must use node-local root or tmp storage')
        require(not self.local.is_relative_to(self.shared) and not self.shared.is_relative_to(self.local),'Local work must be disjoint from shared output')
        self.local.mkdir(parents=True,exist_ok=True)
        require(self.local.resolve()==self.local and stat.S_ISDIR(self.local.lstat().st_mode) and self.local.stat().st_uid==os.getuid(),'Owned regular local work root required')
        (self.local/'outbox').mkdir(exist_ok=True)

    def seal(self,source_keys):
        require(1<=len(source_keys)<=50 and len({tuple(k) for k in source_keys})==len(source_keys),'One-to-fifty unique source results required')
        files=[];statuses={}
        for split,id_ in source_keys:
            require(all(isinstance(x,str) and re.fullmatch('[A-Za-z0-9_-]+',x) for x in (split,id_)),'Safe source identity required')
            directory=self.local/'sources'/split/id_;commit=directory/'commit.json';raw=commit.read_bytes();c=json.loads(raw)
            require(c['id']==id_ and c['split']==split and c.get('quality_approved') is False and c.get('training_eligible') is False,'Reliable unapproved terminal source commit required')
            statuses[c['status']]=statuses.get(c['status'],0)+1
            require(Path(c['result_ref']['path'])==directory/'postprocessed.json','Owned source result reference required')
            result=checked(c['result_ref']);require(result['id']==id_ and result['split']==split and result.get('quality_approved') is False and result.get('training_eligible') is False,'Result/commit identity differs')
            for folder in (directory,self.local/'budgets'/split/id_):
                require(folder.resolve()==folder,'Local source directory changed')
                for p in sorted(folder.iterdir()):
                    require(p.suffix=='.json' and not p.is_symlink() and p.is_file(),'Only regular local JSON evidence may enter metadata archive')
                    require(p.stat().st_size<=16*2**20,'Source metadata exceeds bound');b=p.read_bytes();files.append(dict(path=str(p.relative_to(self.local)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),raw=b))
            packet_ref=result.get('original_pool_claim_ref')
            if packet_ref:
                p=Path(packet_ref['path']);require(p.is_relative_to(self.local/'packets') and p.resolve()==p,'Owned original claim packet reference required')
                b=p.read_bytes();require(hashlib.sha256(b).hexdigest()==packet_ref['sha256'],'Original claim packet changed')
                relative=str(p.relative_to(self.local))
                if not any(item['path']==relative for item in files):files.append(dict(path=relative,sha256=packet_ref['sha256'],bytes=len(b),raw=b))
            for stage in result.get('stages',[]):
                for name in ('batch_request_ref','batch_response_ref'):
                    if name not in stage:continue
                    ref=stage[name];p=Path(ref['path']);require(p.is_relative_to(self.local/'batch_calls') and p.resolve()==p,'Owned batch-call evidence required')
                    b=p.read_bytes();require(hashlib.sha256(b).hexdigest()==ref['sha256'],'Batch-call evidence differs')
                    relative=str(p.relative_to(self.local))
                    if not any(item['path']==relative for item in files):files.append(dict(path=relative,sha256=ref['sha256'],bytes=len(b),raw=b))
        identity=dict(node=self.node,source_keys=source_keys,files=[{k:v for k,v in f.items() if k!='raw'} for f in files])
        batch_id=self.node+'-'+hashlib.sha256(encoded(identity)).hexdigest()[:32];directory=self.local/'outbox'/batch_id;directory.mkdir(exist_ok=False)
        archive=directory/'batch.tar';fd=os.open(archive,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:
            with tarfile.open(fileobj=f,mode='w',format=tarfile.PAX_FORMAT) as tar:
                for item in files:
                    m=tarfile.TarInfo(item['path']);m.size=item['bytes'];tar.addfile(m,io.BytesIO(item['raw']))
                manifest=encoded(identity);m=tarfile.TarInfo('batch_manifest.json');m.size=len(manifest);tar.addfile(m,io.BytesIO(manifest))
            f.flush();os.fsync(f.fileno())
        raw=archive.read_bytes();require(len(raw)<=1024*2**20,'Metadata batch archive exceeds bound')
        ref=write_once(directory/'sealed.json',dict(schema='stage2_local_sealed_batch_v1',batch_id=batch_id,node=self.node,source_count=len(source_keys),source_keys=source_keys,status_counts=statuses,archive_sha256=hashlib.sha256(raw).hexdigest(),archive_bytes=len(raw),manifest=identity,sealed_unix=time.time(),quality_approved=False,training_eligible=False))
        return ref

    def upload(self,sealed_ref):
        value=checked(sealed_ref);batch=value['batch_id'];require(re.fullmatch(self.node+'-[a-f0-9]{32}',batch),'Owned batch identity required')
        localdir=self.local/'outbox'/batch;require(Path(sealed_ref['path'])==localdir/'sealed.json','Owned sealed receipt path required')
        target=self.shared/self.node/batch;target.mkdir(parents=True,exist_ok=True)
        require(target.resolve()==target,'Stable shared batch destination required')
        commit=target/'commit.json';ack=localdir/'upload_receipt.json'
        expected=dict(schema='stage2_shared_metadata_commit_v1',batch_id=batch,node=self.node,source_count=value['source_count'],source_keys=value['source_keys'],status_counts=value['status_counts'],archive=dict(path=str(target/'batch.tar'),sha256=value['archive_sha256'],bytes=value['archive_bytes']),local_seal_ref=sealed_ref,quality_approved=False,training_eligible=False)
        if commit.exists():
            require(json.loads(commit.read_bytes())==expected and (target/'batch.tar').stat().st_size==value['archive_bytes'],'Existing publication differs; never overwrite or call models')
        else:
            archive=target/'batch.tar'
            if not archive.exists():
                temporary=target/(batch+'.partial-'+str(os.getpid()));fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);digest=hashlib.sha256();size=0
                with (localdir/'batch.tar').open('rb') as source,os.fdopen(fd,'wb') as output:
                    for block in iter(lambda:source.read(8*2**20),b''):digest.update(block);size+=len(block);output.write(block)
                    output.flush();os.fsync(output.fileno())
                require(size==value['archive_bytes'] and digest.hexdigest()==value['archive_sha256'],'Local sealed archive changed; preserve both copies')
                rename_noreplace(temporary,archive);df=os.open(target,os.O_RDONLY|os.O_DIRECTORY);os.fsync(df);os.close(df)
            else:
                b=archive.read_bytes();require(len(b)==value['archive_bytes'] and hashlib.sha256(b).hexdigest()==value['archive_sha256'],'Partial previously published archive changed')
            write_once(commit,expected)
        require(json.loads(commit.read_bytes())==expected,'Published commit readback differs')
        if ack.exists():
            result=json.loads(ack.read_bytes());require(result['batch_id']==batch and result['archive_sha256']==value['archive_sha256'],'Existing local ACK differs');return result
        receipt=dict(schema='stage2_local_upload_receipt_v1',batch_id=batch,source_count=value['source_count'],archive_sha256=value['archive_sha256'],shared_commit=dict(path=str(commit),sha256=hashlib.sha256(commit.read_bytes()).hexdigest()),committed_unix=time.time(),may_cleanup_local=True,models_called=0)
        write_once(ack,receipt);return receipt

    def uploader_lock(self):
        fd=os.open(self.local/'uploader.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);s=os.fstat(fd)
        require(stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_size==0,'Owned single-uploader lock required')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:os.close(fd);raise
        return fd
