"""Only shared-I/O reconciliation; never re-executes model work or resets claims."""
import hashlib
import json
import os
from pathlib import Path
import time

B=Path('/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp/MS/bar_annotation_optimized_20261024_v1')
TRANSIENT={5,103,104,107,110,116}


def io_wait(operation,label):
    warning=False
    while True:
        try:
            result=operation()
            if warning:print(json.dumps({'event':'shared_io_recovered','operation':label,'no_model_retry':True}),flush=True)
            return result
        except OSError as e:
            if e.errno not in TRANSIENT:raise
            if not warning:print(json.dumps({'event':'shared_io_wait','operation':label,'errno':e.errno,'error':str(e),'no_model_retry':True,'no_claim_reset':True}),flush=True);warning=True
            time.sleep(5)


def read_json_shared(path):
    path=Path(path);deadline=time.monotonic()+120
    while True:
        try:return json.loads(io_wait(lambda:path.read_bytes(),'read '+str(path)))
        except FileNotFoundError:
            if path.name!='commit.json' or time.monotonic()>=deadline:raise
            print(json.dumps({'event':'commit_readback_visibility_wait','path':str(path),'no_model_retry':True}),flush=True)
            time.sleep(5)


def fsync_dir(path):
    def action():
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    return io_wait(action,'fsync '+str(path))


def mkdir_sync(path):
    path=Path(path)
    def action():
        if path.is_dir():return
        mkdir_sync(path.parent)
        try:path.mkdir()
        except FileExistsError:
            if not path.is_dir():raise
        fsync_dir(path.parent)
    return io_wait(action,'mkdir '+str(path))


def digest_file(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def publish_file(src,dst):
    src,dst=Path(src),Path(dst);sha=digest_file(src)
    def attempt():
        mkdir_sync(dst.parent)
        if dst.exists():
            if dst.stat().st_size!=src.stat().st_size or digest_file(dst)!=sha:raise ValueError('Existing publication differs; never overwritten: '+str(dst))
            fsync_dir(dst.parent)
            return dict(path=str(dst),bytes=src.stat().st_size,sha256=sha,existing_identical_preserved=True)
        tmp=dst.parent/('.publish-'+str(os.getpid())+'-'+str(time.time_ns())+'-'+dst.name)
        with src.open('rb') as a,tmp.open('xb') as b:
            while True:
                raw=a.read(4*1024**2)
                if not raw:break
                b.write(raw)
            b.flush();os.fsync(b.fileno())
        if digest_file(tmp)!=sha:raise ValueError('Shared publication readback mismatch')
        try:os.link(tmp,dst)
        except FileExistsError:
            if digest_file(dst)!=sha:raise ValueError('Concurrent publication differs; preserved')
        tmp.unlink();fsync_dir(dst.parent)
        return dict(path=str(dst),bytes=src.stat().st_size,sha256=sha)
    return io_wait(attempt,'publish '+str(dst))


def install_shared_io_adapters(core):
    original_write,original_claim=core.write_once,core.claim_once
    def write(path,raw):
        path=Path(path)
        if not path.is_relative_to(B):return original_write(path,raw)
        def attempt():
            mkdir_sync(path.parent)
            if path.exists():
                if path.read_bytes()!=raw:raise ValueError('Existing shared result differs; no overwrite '+str(path))
                fsync_dir(path.parent);return
            original_write(path,raw)
        return io_wait(attempt,'write_once '+str(path))
    def claim(path,value):
        path=Path(path);mkdir_sync(path.parent);uncertain=False;warned=False
        while True:
            try:
                if path.exists():
                    if not uncertain:raise FileExistsError(str(path))
                    if path.read_bytes()!=core.json_bytes(value):raise PermissionError('Uncertain claim differs/partial; retained without replay '+str(path))
                    fsync_dir(path.parent);return
                original_claim(path,value);return
            except FileExistsError:raise
            except OSError as e:
                if e.errno not in TRANSIENT:raise
                uncertain=True
                if not warned:print(json.dumps({'event':'claim_io_wait','path':str(path),'errno':e.errno,'consumption_uncertain':True,'no_delete_or_reset':True}),flush=True);warned=True
                time.sleep(5)
    core.write_once=write;core.claim_once=claim
