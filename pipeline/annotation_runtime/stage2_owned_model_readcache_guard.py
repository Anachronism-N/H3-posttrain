"""Avoid retaining loaded model disk-page cache in N1 cgroup; never lower safety limits."""
import fcntl
import json
import os
from pathlib import Path
import time

LOCAL=Path('/root/stage2_owned_readcache_service_N1_20261028_v1')


def memory():
    p=Path('/sys/fs/cgroup/memory');return int((p/'memory.limit_in_bytes').read_text())-int((p/'memory.usage_in_bytes').read_text())


def run():
    LOCAL.mkdir(exist_ok=True);fd=os.open(LOCAL/'service.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (LOCAL/'STOP').exists():
        before=memory();roots=[p for p in Path('/root').glob('stage2_dynamic_*_N1_20261028_v1') if (p/'deployment.json').exists()];active=[];loading=False
        for root in roots:
            d=json.loads((root/'deployment.json').read_bytes())
            for e in d['configs']:
                c=json.loads(Path(e['config_ref']['path']).read_bytes());b=root/'runtime_bindings'/(c['owner']+'.json')
                if b.exists():c=json.loads(Path(json.loads(b.read_bytes())['config_ref']['path']).read_bytes())
                q=Path(c['queue'])
                if not (q/'live.json').exists():continue
                identity=json.loads((q/'live.json').read_bytes())['child_identity'];proc=Path('/proc',str(identity['pid']))
                try:s=(proc/'stat').read_text().rsplit(') ',1)[1].split()
                except FileNotFoundError:continue
                if int(s[19])!=identity['start_ticks'] or s[0] in ('Z','X','x'):continue
                state=root/'worker_state'/c['owner'];state=state/c['phase'] if c.get('worker_state_layout')=='by_phase' else state
                if not (state/'load.receipt.json').exists():loading=True
                else:active.append(c['owner'])
        if active and not loading and before<200*2**30:
            count=0;bytes_=0
            for p in Path('/opt/models/Qwen3-VL-32B-Instruct').glob('*.safetensors'):
                f=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
                try:os.posix_fadvise(f,0,0,os.POSIX_FADV_DONTNEED);bytes_+=os.fstat(f).st_size;count+=1
                finally:os.close(f)
            print(json.dumps(dict(event='owned_loaded_model_readcache_advised',before_headroom_gib=before/2**30,after_headroom_gib=memory()/2**30,model_readcache_bytes=bytes_,shards=count,active_owners=active,loading_skipped=True,safety_thresholds_changed=False,global_drop_caches_used=False,files_deleted=0,GPU_signals=0)),flush=True)
        time.sleep(120)
    os.close(fd)

if __name__=='__main__':run()
