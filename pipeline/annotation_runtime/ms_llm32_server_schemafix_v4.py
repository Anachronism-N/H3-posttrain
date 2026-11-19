"""03-only shared-input relay, one durable source pool, and no-clobber batch publication."""
import argparse
from collections import OrderedDict
from collections import Counter
import fcntl
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import re
import struct
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit,parse_qs
from ms_llm32_pool_v4 import DynamicPool
from stage2_local_publication_v1 import write_once,encoded,require,rename_noreplace

EXTERNAL=tuple('a100-extra-0'+str(i) for i in range(1,7))
MAIN=('N1','N2','N3','N6','N8')
NODES=MAIN+EXTERNAL
STATUSES={'postprocessed_unapproved':'published_unapproved','postprocessed_needs_visual_revision':'needs_visual_revision','insufficient_visible_evidence':'insufficient_visible_evidence','failed_sealed_no_retry':'failed_sealed_no_retry'}


def source_row(cfg,owner,id_):
    p=DynamicPool(cfg['database'])
    try:r=p.db.execute('SELECT payload,state,owner FROM sources WHERE source_key=?',(id_.casefold(),)).fetchone()
    finally:p.close()
    require(r is not None and r[2]==owner,'Only permanently owned original inputs may be fetched');return json.loads(r[0])


def acknowledge(cfg,value,archive):
    # Validate the uploaded evidence once; retries only reconcile the existing commit.
    keys=set(map(tuple,value['source_keys']));require(1<=len(keys)==value['source_count']<=50 and len(value['source_keys'])==len(keys),'Unique source-bound upload required')
    require(value['batch_id']==value['node']+'-'+hashlib.sha256(encoded(value['manifest'])).hexdigest()[:32],'Sealed batch identity differs')
    manifest=None;commits={};names=set()
    with tarfile.open(archive,mode='r:') as tar:
        for m in tar:
            parts=m.name.split('/');require(m.isfile() and not m.issparse() and m.sparse is None and not any(k.startswith(('GNU.sparse','SCHILY.realsize')) for k in m.pax_headers),'Only regular nonsparse evidence members allowed')
            require(not m.name.startswith('/') and all(x not in ('','.','..') for x in parts) and '\\' not in m.name and m.name not in names and 0<=m.size<=16*2**20,'Safe bounded unique evidence member required');names.add(m.name)
            if m.name=='batch_manifest.json':manifest=json.loads(tar.extractfile(m).read(16*2**20+1))
            if len(parts)==4 and parts[0]=='sources' and parts[3]=='commit.json':commits[(parts[1],parts[2])]=json.loads(tar.extractfile(m).read(16*2**20+1))
    require(manifest==value['manifest'] and set(commits)==keys,'Sealed archive/source manifest differs')
    require(dict(Counter(c['status'] for c in commits.values()))==value['status_counts'],'Uploaded per-source status count differs')
    outcomes=[];pool=DynamicPool(cfg['database'])
    try:
        for split,id_ in keys:
            c=commits[(split,id_)];require(c['id']==id_ and c['split']==split and c['owner'] in cfg['allowed_owners'] and c['owner'].startswith(value['node']+'-g') and c.get('revision')=='english_merged_edges_MS_LLM32B_v4' and c['quality_approved'] is False and c['training_eligible'] is False and c['new_model_calls']<=1,'Source result identity or budget differs')
            r=pool.db.execute('SELECT state,owner,batch_id,split,id FROM sources WHERE source_key=?',(id_.casefold(),)).fetchone();require(r is not None and r[1:]==(c['owner'],c['pool_claim_batch'],split,id_),'Uploaded source is outside original claim')
            outcomes.append((c,r[0],STATUSES[c['status']]))
    finally:pool.close()
    return outcomes


def finish_ledger(cfg,outcomes):
    pool=DynamicPool(cfg['database'])
    try:
        for c,old,status in outcomes:
            r=pool.db.execute('SELECT state FROM sources WHERE source_key=?',(c['id'].casefold(),)).fetchone()[0]
            if r=='assigned':pool.terminal(c['owner'],c['pool_claim_batch'],c['split'],c['id'],status)
            else:require(r==status,'Existing terminal outcome cannot change')
    finally:pool.close()


def serve(path):
    cfg=json.loads(Path(path).read_bytes());token=Path(cfg['token_path']).read_text().strip();require(len(token)>=32 and set(cfg['allowed_owners'])==({n+'-g'+pair+'-msllm4' for n in EXTERNAL for pair in ('0123','4567')} | {n+'-g'+pair+'-msllm4' for n in MAIN for pair in ('01','23','45','67')}),'Exact external source-group owners required')
    root=Path(cfg['local_root']);root.mkdir(parents=True,exist_ok=True);lf=os.open(root/'relay.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)
    shared=Path(cfg['shared_root']);require(shared.is_absolute() and shared.resolve()==shared,'Stable authorized shared publication root required');shared.mkdir(parents=True,exist_ok=True)
    input_limit=threading.BoundedSemaphore(16);upload_limit=threading.BoundedSemaphore(4)
    metadata_cache=OrderedDict();metadata_lock=threading.Lock()
    def metadata(row):
        ref=row['revision_input_archive'];key=ref['path']
        with metadata_lock:
            values=metadata_cache.get(key)
            if values is None:
                raw=Path(key).read_bytes();require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'Frozen original revision pack differs');values={}
                with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tar:
                    for m in tar:
                        require(m.isfile() and not m.issparse() and m.name not in values and 0<m.size<=2*2**20,'Bounded original input member required');values[m.name]=tar.extractfile(m).read(2*2**20+1)
                metadata_cache[key]=values
                if len(metadata_cache)>32:metadata_cache.popitem(last=False)
            else:metadata_cache.move_to_end(key)
            raw=values[row['revision_input_member']];require(hashlib.sha256(raw).hexdigest()==row['revision_input_sha256'],'Original metadata bytes differ');v=json.loads(raw);v['input_metadata_revision']=v['revision'];v['revision']='english_merged_edges_MS_LLM32B_v4';return v
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.0'
        def setup(self):
            super().setup();self.connection.settimeout(180)
        def log_message(self,*args):pass
        def auth(self):require(hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token),'Unauthorized relay')
        def reply(self,value):
            raw=encoded(value);self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            try:
                self.auth();u=urlsplit(self.path);x=parse_qs(u.query,strict_parsing=True);require(u.path=='/input' and set(x)=={'owner','id','kind'} and all(len(v)==1 for v in x.values()),'Exact relay input query required')
                owner,id_,kind=x['owner'][0],x['id'][0],x['kind'][0];require(owner in cfg['allowed_owners'] and re.fullmatch('[A-Za-z0-9_-]+',id_) and kind=='video','Authorized input lookup required');row=source_row(cfg,owner,id_)
                with input_limit:
                    if kind=='video':p=Path(row['video_path']);maximum=2**30
                    else:p=Path(row['rule_draft_ref' if kind=='draft' else 'fullframe_manifest_ref']['path']);maximum=2**20
                    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
                    with os.fdopen(fd,'rb') as f:
                        before=os.fstat(f.fileno());require(0<before.st_size<=maximum,'Bounded original relay input required')
                        if kind!='video':
                            raw=f.read(maximum+1);ref=row['rule_draft_ref' if kind=='draft' else 'fullframe_manifest_ref'];require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Frozen original metadata changed')
                        self.send_response(200);self.send_header('Content-Length',str(before.st_size));self.end_headers()
                        if kind!='video':self.wfile.write(raw)
                        else:
                            for block in iter(lambda:f.read(8*2**20),b''):self.wfile.write(block)
                        after=os.fstat(f.fileno());require((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'Source changed during relay')
            except Exception as e:
                self.close_connection=True
                try:self.send_error(409,'Original input unavailable; keep claim sealed, no retry')
                except OSError:pass
        def do_POST(self):
            try:
                self.auth()
                if self.path in ('/claim','/inspect','/progress'):
                    n=int(self.headers.get('Content-Length','0'));require(0<n<=8192,'Small claim metadata required');x=json.loads(self.rfile.read(n));owner=x['owner'];require(owner in cfg['allowed_owners'],'Unknown external owner');p=DynamicPool(cfg['database'])
                    try:
                        if self.path=='/claim':
                            v=p.claim(owner,x['request_id'],x.get('size',8))
                            if v is None and p.db.execute('SELECT 1 FROM sources WHERE state="paused_unused_schema_preflight" LIMIT 1').fetchone():v=dict(waiting_for_original_inputs=True,new_claim_created=False,models_called=0)
                        elif self.path=='/inspect':v=p.inspect_claim(owner,x['request_id'])
                        else:v=dict(states=dict(p.db.execute('SELECT s.state,count(*) FROM sources s JOIN external_allocations a ON a.source_key=s.source_key WHERE a.node=? GROUP BY s.state',('-'.join(owner.split('-')[:3]),)).fetchall()) if owner.startswith('a100-extra-') else p.progress(),source_count=12000 if owner.startswith('a100-extra-') else cfg['source_count'],revision='english_merged_edges_MS_LLM32B_v4',observed_unix=time.time())
                    finally:p.close()
                    if self.path=='/claim' and v and 'rows' in v:v['prepared_metadata']=[metadata(row) for row in v['rows']]
                    self.reply(v);return
                require(self.path=='/upload','Unknown relay operation');length=int(self.headers.get('Content-Length','0'));require(8<length<=1025*2**20,'Bounded sealed upload required')
                with upload_limit:
                    prefix=self.rfile.read(8);require(len(prefix)==8,'Missing upload seal framing');n=struct.unpack('!Q',prefix)[0];require(0<n<=2**20 and n+8<length,'Bounded seal metadata required');raw=self.rfile.read(n);require(len(raw)==n,'Torn upload seal');value=json.loads(raw);node=value['node'];require(node in NODES and re.fullmatch(re.escape(node)+'-[a-f0-9]{32}',value['batch_id']),'Exact external publication identity required')
                    require(value['quality_approved'] is False and value['training_eligible'] is False and value['archive_bytes']==length-8-n and 0<value['archive_bytes']<=1024*2**20,'Unapproved sealed upload required')
                    batch=value['batch_id'];target=shared/node/batch;target.mkdir(parents=True,exist_ok=True);require(target.resolve()==target,'Stable no-clobber upload destination required');lockfd=os.open(target/'upload.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
                    try:
                        fcntl.flock(lockfd,fcntl.LOCK_EX);archive=target/'batch.tar';expected=dict(schema='stage2_shared_metadata_commit_v1',batch_id=batch,node=node,source_count=value['source_count'],source_keys=value['source_keys'],status_counts=value['status_counts'],archive=dict(path=str(archive),sha256=value['archive_sha256'],bytes=value['archive_bytes']),local_seal_ref=value['_upload_seal_ref'],quality_approved=False,training_eligible=False);commit=target/'commit.json'
                        # Receive to a unique owned temporary, even when retrying a prior uncertain upload.
                        temporary=target/('incoming-'+str(os.getpid())+'-'+__import__('secrets').token_hex(8));fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);remaining=value['archive_bytes'];digest=hashlib.sha256()
                        with os.fdopen(fd,'wb') as f:
                            while remaining:
                                b=self.rfile.read(min(8*2**20,remaining));require(b,'Truncated upload; source remains locally sealed');remaining-=len(b);digest.update(b);f.write(b)
                            f.flush();os.fsync(f.fileno())
                        require(digest.hexdigest()==value['archive_sha256'],'Sealed archive SHA differs')
                        if not archive.exists():rename_noreplace(temporary,archive)
                        else:require(archive.stat().st_size==value['archive_bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==value['archive_sha256'],'Existing upload differs; never overwrite');temporary.unlink()
                        outcomes=acknowledge(cfg,value,archive)
                        if commit.exists():require(json.loads(commit.read_bytes())==expected,'Existing shared commit differs')
                        else:write_once(commit,expected)
                        finish_ledger(cfg,outcomes);ref=dict(path=str(commit),sha256=hashlib.sha256(commit.read_bytes()).hexdigest());self.reply(dict(schema='stage2_local_upload_receipt_v1',batch_id=batch,source_count=value['source_count'],archive_sha256=value['archive_sha256'],shared_commit=ref,committed_unix=time.time(),may_cleanup_local=True,models_called=0));print(json.dumps(dict(event='external_shared_batch_committed',node=node,batch=batch,sources=value['source_count'],committed_unix=time.time())),flush=True)
                    finally:os.close(lockfd)
            except Exception as e:
                self.close_connection=True
                try:self.send_error(409,'Relay operation rejected; do not repeat model or claim')
                except OSError:pass
                print(json.dumps(dict(event='relay_rejected',path=urlsplit(self.path).path,error=type(e).__name__+': '+str(e)[:500])),flush=True)
    ThreadingHTTPServer((cfg['bind'],cfg['port']),Handler).serve_forever()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();serve(a.config)
