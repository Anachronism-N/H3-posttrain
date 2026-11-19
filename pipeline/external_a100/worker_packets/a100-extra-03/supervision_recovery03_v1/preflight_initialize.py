"""Initialize independent state from explicitly SHA-pinned original checkpoints."""
from pathlib import Path
import argparse, datetime, json, os
from supervision_common import OLD_ROOT,REPORT_ROOT,P,WORKERS,strict_json,sha,assignment_scope,validate_cursor,bind_ledger

def initialize(old_root,root,cursor_sha,ledger_sha,check_only=False,p=P):
    if root.resolve()==old_root.resolve() or not root.resolve().is_relative_to(REPORT_ROOT.resolve()): raise ValueError('report root must remain in independent candidate')
    names=['rollover_verified_commit_cursor.json','rollover_payload_audit_ledger.json']
    raw=[(old_root/'logs'/n).read_bytes() for n in names]
    if [sha(b) for b in raw]!=[cursor_sha,ledger_sha]: raise ValueError('original cursor/ledger SHA mismatch; fail closed')
    state,ledger=map(strict_json,raw); assignments,tickets=assignment_scope(p)
    commits=validate_cursor(state,assignments,tickets,p); copied_ledger=bind_ledger(ledger,state,commits)
    if any((old_root/'logs'/n).read_bytes()!=b for n,b in zip(names,raw)): raise ValueError('original checkpoints changed during preflight')
    marker={'schema':'independent_supervision_initialization_v1','created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_cursor_sha256':cursor_sha,'original_deep_ledger_sha256':ledger_sha,'assignment_sha256':{w:tickets[w]['assignment_sha256'] for w in WORKERS},'validated_commit_count':len(commits),'copied_deep_verified_commit_count':len(copied_ledger['commits']),'committed_unique_sources':sum(len(s) for s in state['sources'].values()),'deep_verified_unique_sources':sum(len(s) for s in ledger['sources'].values()),'input_media_read':False,'old_output_payloads_reread':False,'GPU_models_loaded':False,'worker_lifecycle_changed':False,'check_only':check_only}
    if not check_only:
        logs=root/'logs'; logs.mkdir(parents=True,exist_ok=True)
        paths=[logs/n for n in names]+[logs/'initialization.json']
        if any(path.exists() for path in paths): raise FileExistsError('candidate state exists; no overwrite')
        payloads=[raw[0],(json.dumps(copied_ledger,allow_nan=False)+'\n').encode(),(json.dumps(marker,allow_nan=False)+'\n').encode()]
        # Marker is last: any interrupted initialization remains fail-closed.
        for path,content in zip(paths,payloads):
            with path.open('xb') as stream: stream.write(content);stream.flush();os.fsync(stream.fileno())
    return marker

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cursor-sha256',required=True);parser.add_argument('--ledger-sha256',required=True);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();print(json.dumps(initialize(OLD_ROOT,REPORT_ROOT,args.cursor_sha256,args.ledger_sha256,args.check_only),allow_nan=False))
