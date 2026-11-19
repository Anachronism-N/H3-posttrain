"""Node-local durable dynamic allocator: batches of 3-8, assigned sources never reissued."""
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time


def need(ok,message):
    if not ok:raise ValueError(message)


def text(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':'))


class DynamicPool:
    def __init__(self,path):
        self.path=Path(path)
        need(self.path.is_absolute() and str(self.path).startswith(('/root/','/tmp/')) and self.path.resolve()==self.path and self.path.is_file() and not self.path.is_symlink(),'Pool ledger must be an existing stable local-disk file')
        self.db=sqlite3.connect(str(self.path),timeout=30,isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA foreign_keys=ON')

    @classmethod
    def create(cls,path,rows,excluded_keys):
        path=Path(path);need(1<=len(rows)<=500000,'Finite existing input pool required')
        excluded={id_.casefold() for split,id_ in excluded_keys};ids=set();selected=[]
        for row in rows:
            id_=row['id'];split=row['split'];need(isinstance(id_,str) and re.fullmatch('[A-Za-z0-9_-]+',id_) and isinstance(split,str) and re.fullmatch('[A-Za-z0-9_-]+',split),'Safe exact source identity required')
            need(id_.casefold() not in ids,'Casefold duplicate source forbidden');ids.add(id_.casefold())
            if id_.casefold() not in excluded:selected.append(row)
        need(selected,'No unused qualified source inputs available')
        path.parent.mkdir(parents=True,exist_ok=True);need(path.parent.resolve()==path.parent,'Stable ledger parent required');fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.close(fd)
        pool=cls(path)
        pool.db.executescript('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE sources(ordinal INTEGER PRIMARY KEY,source_key TEXT UNIQUE NOT NULL,split TEXT NOT NULL,id TEXT NOT NULL,payload TEXT NOT NULL,state TEXT NOT NULL DEFAULT "unused",owner TEXT,batch_id TEXT); CREATE TABLE batches(batch_id TEXT PRIMARY KEY,owner TEXT NOT NULL,requested_size INTEGER NOT NULL,source_keys TEXT NOT NULL,assigned_unix REAL NOT NULL);')
        pool.db.execute('BEGIN IMMEDIATE')
        try:
            pool.db.executemany('INSERT INTO sources(ordinal,source_key,split,id,payload) VALUES(?,?,?,?,?)',[(i,r['id'].casefold(),r['split'],r['id'],text(r)) for i,r in enumerate(selected)])
            pool.db.execute('INSERT INTO meta VALUES(?,?)',('source_manifest_sha256',hashlib.sha256(text(selected).encode()).hexdigest()))
            pool.db.execute('INSERT INTO meta VALUES(?,?)',('original_source_count',str(len(rows))))
            pool.db.execute('INSERT INTO meta VALUES(?,?)',('excluded_count',str(len(rows)-len(selected))))
            pool.db.execute('COMMIT')
        except BaseException:pool.db.execute('ROLLBACK');pool.close();raise
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
        return pool

    def claim(self,owner,batch_id,size=8):
        need(isinstance(owner,str) and re.fullmatch('a100-extra-0[3-6]-g(?:0123|4567)-external50000',owner),'Explicit external03-06 TP4 owner required')
        need(isinstance(batch_id,str) and re.fullmatch('[A-Za-z0-9_-]{8,100}',batch_id),'Stable new claim request identity required')
        need(type(size) is int and 3<=size<=8,'Atomic dynamic source batches must be 3-to-8')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            existing=self.db.execute('SELECT owner FROM batches WHERE batch_id=?',(batch_id,)).fetchone()
            need(existing is None,'Existing claim receipt must be inspected, never automatically re-claimed or assigned')
            rows=self.db.execute('SELECT ordinal,split,id,payload FROM sources WHERE state="unused" ORDER BY ordinal LIMIT ?',(size,)).fetchall()
            if not rows:self.db.execute('COMMIT');return None
            keys=[[r[1],r[2]] for r in rows];now=time.time()
            self.db.execute('INSERT INTO batches VALUES(?,?,?,?,?)',(batch_id,owner,size,text(keys),now))
            for ordinal,split,id_,payload in rows:
                changed=self.db.execute('UPDATE sources SET state="assigned",owner=?,batch_id=? WHERE ordinal=? AND state="unused"',(owner,batch_id,ordinal))
                need(changed.rowcount==1,'Source ownership changed during transaction')
            self.db.execute('COMMIT')
            return dict(schema='stage2_dynamic_claim_receipt_v1',batch_id=batch_id,owner=owner,source_keys=keys,rows=[json.loads(r[3]) for r in rows],source_count=len(rows),assigned_unix=now,automatic_retry=False,maximum_model_calls_per_source=2)
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise

    def inspect_claim(self,owner,batch_id):
        row=self.db.execute('SELECT owner,source_keys,assigned_unix FROM batches WHERE batch_id=?',(batch_id,)).fetchone();need(row is not None and row[0]==owner,'Claim is absent or belongs to another worker')
        return dict(batch_id=batch_id,owner=row[0],source_keys=json.loads(row[1]),assigned_unix=row[2],may_run_models=False,metadata_only_recovery=True)

    def terminal(self,owner,batch_id,split,id_,status):
        need(status in ('published_unapproved','needs_visual_revision','insufficient_visible_evidence','failed_sealed_no_retry'),'Explicit unapproved or sealed outcome required')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row=self.db.execute('SELECT state,owner,batch_id,split,id FROM sources WHERE source_key=?',(id_.casefold(),)).fetchone();need(row is not None and row[1:]==(owner,batch_id,split,id_),'Source outcome does not belong to this original claim')
            need(row[0]=='assigned','Terminal outcome is permanently sealed, never changed or regenerated')
            self.db.execute('UPDATE sources SET state=? WHERE source_key=?',(status,id_.casefold()));self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise

    def progress(self):
        return dict(self.db.execute('SELECT state,count(*) FROM sources GROUP BY state').fetchall())

    def close(self):self.db.close()
