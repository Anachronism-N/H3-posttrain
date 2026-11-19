"""One new-revision ledger; claims include locally decoded original metadata, no shared GPU I/O."""
import argparse
import os
from urllib.parse import urlsplit,parse_qs
from collections import OrderedDict
import hashlib
import hmac
import io
import json
import sqlite3
from pathlib import Path
import threading
import tarfile
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from motion_revision_external_pool_v1 import DynamicPool
from stage2_local_publication_v1 import require


def serve(path):
    cfg=json.loads(Path(path).read_bytes());token=Path(cfg['token_path']).read_text().strip();cache=OrderedDict();lock=threading.Lock();sync_lock=threading.Lock()
    def sync_inputs():
        with sync_lock:
            source=sqlite3.connect('file:'+cfg['input_catalog']+'?mode=ro',uri=True,timeout=30);pool=DynamicPool(cfg['database'])
            try:
                count=pool.db.execute('SELECT count(*) FROM sources').fetchone()[0];rows=source.execute('SELECT payload FROM sources ORDER BY ordinal LIMIT 4000 OFFSET ?',(count,)).fetchall()
                if rows:
                    pool.db.execute('BEGIN IMMEDIATE')
                    try:
                        for i,(raw,) in enumerate(rows):
                            row=json.loads(raw);pool.db.execute('INSERT INTO sources(ordinal,source_key,split,id,payload) VALUES(?,?,?,?,?)',(count+i,row['id'].casefold(),row['split'],row['id'],raw))
                        pool.db.execute('COMMIT')
                    except BaseException:pool.db.execute('ROLLBACK');raise
            finally:source.close();pool.close()
    def metadata(row):
        ref=row['revision_input_archive'];key=ref['path']
        with lock:
            values=cache.get(key)
            if values is None:
                raw=Path(key).read_bytes();require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'Frozen original revision metadata pack changed');values={}
                with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as tar:
                    for m in tar:
                        require(m.isfile() and not m.issparse() and m.name not in values and 0<m.size<=2*2**20,'Safe bounded input metadata member required');values[m.name]=tar.extractfile(m).read(2*2**20+1)
                cache[key]=values
                if len(cache)>16:cache.popitem(last=False)
            else:cache.move_to_end(key)
            raw=values[row['revision_input_member']];require(hashlib.sha256(raw).hexdigest()==row['revision_input_sha256'],'Original revision input differs');value=json.loads(raw);value['input_metadata_revision']=value['revision'];value['revision']=cfg['revision'];return value
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            try:
                require(hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token),'Unauthorized media request');u=urlsplit(self.path);args=parse_qs(u.query,strict_parsing=True);require(u.path=='/input' and set(args)=={'owner','id'} and all(len(v)==1 for v in args.values()),'Exact owner/source video query required');owner=args['owner'][0];id_=args['id'][0];require(owner in cfg['allowed_owners'],'Unknown source owner');pool=DynamicPool(cfg['database'])
                try:row=pool.db.execute('SELECT payload,owner FROM sources WHERE source_key=?',(id_.casefold(),)).fetchone()
                finally:pool.close()
                require(row is not None and row[1]==owner,'Only originally assigned source media may be fetched');data=json.loads(row[0]);fd=os.open(data['video_path'],os.O_RDONLY|os.O_NOFOLLOW)
                with os.fdopen(fd,'rb') as f:
                    before=os.fstat(f.fileno());require(0<before.st_size<=2**30,'Bounded original video required');self.send_response(200);self.send_header('Content-Length',str(before.st_size));self.end_headers()
                    for b in iter(lambda:f.read(8*2**20),b''):self.wfile.write(b)
                    after=os.fstat(f.fileno());require((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'Source changed while relaying')
            except Exception as e:
                self.close_connection=True
                try:self.send_error(409,'Owned original source input rejected')
                except OSError:pass

        def do_POST(self):
            try:
                require(self.path in ('/claim','/progress','/inspect'),'Unknown metadata operation');require(hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token),'Unauthorized revision request');n=int(self.headers.get('Content-Length','0'));require(0<n<=8192,'Bounded request required');x=json.loads(self.rfile.read(n));sync_inputs();require(x['owner'] in cfg['allowed_owners'],'Explicit new-revision owner required');p=DynamicPool(cfg['database'])
                try:
                    if self.path=='/claim':v=p.claim(x['owner'],x['request_id'],x.get('size',8))
                    elif self.path=='/inspect':v=p.inspect_claim(x['owner'],x['request_id'])
                    else:v=p.progress()
                finally:p.close()
                if self.path=='/claim' and v:v['prepared_metadata']=[metadata(r) for r in v['rows']]
                elif self.path=='/claim' and not Path(cfg['input_preparation_complete']).exists():v=dict(waiting_for_original_inputs=True,models_called=0,new_claim_created=False)
                raw=json.dumps(v,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode();require(len(raw)<=16*2**20,'Bounded claim+metadata response required');self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(raw)
            except Exception as e:
                self.send_error(409,'Claim/metadata boundary failed; do not automatically reclaim');print(json.dumps(dict(event='new_revision_claim_rejected',error=repr(e))),flush=True)
    ThreadingHTTPServer((cfg['bind'],cfg['port']),Handler).serve_forever()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();serve(a.config)
