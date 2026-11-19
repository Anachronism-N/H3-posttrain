"""A short authenticated-SSH metadata CLI; GPU workers never touch the shared filesystem."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from stage2_dynamic_pool_v1 import DynamicPool,need


def load_manifest(path,digest):
    p=Path(path);need(p.is_absolute() and p.resolve()==p and p.is_file(),'Frozen source manifest required');raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==digest,'Source manifest SHA differs')
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()];need(rows and len(rows)<=500000,'Finite source pool required');return rows


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='operation',required=True)
    p=sub.add_parser('init');p.add_argument('--database',required=True);p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--exclusions',required=True);p.add_argument('--exclusions-sha',required=True)
    p=sub.add_parser('claim');p.add_argument('--database',required=True);p.add_argument('--owner',required=True);p.add_argument('--request-id',required=True);p.add_argument('--size',type=int,default=8)
    p=sub.add_parser('inspect');p.add_argument('--database',required=True);p.add_argument('--owner',required=True);p.add_argument('--request-id',required=True)
    p=sub.add_parser('progress');p.add_argument('--database',required=True)
    args=parser.parse_args()
    if args.operation=='init':
        rows=load_manifest(args.manifest,args.manifest_sha);p=Path(args.exclusions);raw=p.read_bytes();need(hashlib.sha256(raw).hexdigest()==args.exclusions_sha,'Global exclusion evidence differs');e=json.loads(raw);need(e.get('schema')=='stage2_exact_exclusion_keys_v1' and e.get('full_allocated_and_consumed_stage_keys_verified') is True,'Exact old allocation and consumed-source exclusion required')
        pool=DynamicPool.create(args.database,rows,e['source_keys'])
        try:result=dict(schema='stage2_pool_initialized_v1',source_states=pool.progress(),database=str(pool.path),manifest_sha256=args.manifest_sha,exclusion_sha256=args.exclusions_sha,source_claims_created=0,models_called=0)
        finally:pool.close()
    else:
        pool=DynamicPool(args.database)
        try:
            if args.operation=='claim':result=pool.claim(args.owner,args.request_id,args.size)
            elif args.operation=='inspect':result=pool.inspect_claim(args.owner,args.request_id)
            else:result=dict(schema='stage2_pool_progress_v1',source_states=pool.progress(),models_called=0)
        finally:pool.close()
    sys.stdout.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+chr(10));sys.stdout.flush()

if __name__=='__main__':main()
