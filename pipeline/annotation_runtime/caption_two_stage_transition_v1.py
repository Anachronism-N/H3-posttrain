"""Exact old-lane TERM, native safe, one-index source exclusion, new managed lane."""
import argparse
import os
from pathlib import Path
import select
import signal
import time
import caption_two_stage_live_manager_v5 as m


def main():
    p=argparse.ArgumentParser();p.add_argument('--old-config',required=True);p.add_argument('--name',required=True);p.add_argument('--source-root',required=True)
    args=p.parse_args();old_path=Path(args.old_config);old=m.light.load(old_path);q=Path(old['queue']);oldout=Path(old['out'])
    admission=m.light.load(q/'admission.json');live=m.light.load(q/'live.json')
    identities=[]
    if not (q/'resources_safe.json').exists():
        manager=m.base.identity(admission['manager_identity']['pid'])
        assert all(manager[k]==admission['manager_identity'][k] for k in m.base.IDENTITY_KEYS)
        worker=m.base.identity(live['child_identity']['pid'])
        assert all(worker[k]==live['child_identity'][k] for k in m.base.IDENTITY_KEYS) and worker['ppid']==manager['pid']
        uuids={m.base.uuid_for(g) for g in old['gpus']}
        for app in m.base.apps():
            if app['uuid'] in uuids:
                i=m.base.identity(app['pid']);assert i['uid']==manager['uid'] and i['cgroup']==manager['cgroup']
                ancestor=i
                for _ in range(20):
                    if ancestor['pid']==manager['pid']:break
                    ancestor=m.base.identity(ancestor['ppid'])
                assert ancestor['pid']==manager['pid'],'Unknown GPU tree; no signal'
                identities.append(i)
        fd=os.pidfd_open(manager['pid'])
        try:signal.pidfd_send_signal(fd,signal.SIGTERM)
        finally:os.close(fd)
        print(m.json.dumps({'event':'authorized_old_lane_TERM','node':old['node'],'gpus':old['gpus']}),flush=True)
    deadline=time.monotonic()+240
    while not (q/'resources_safe.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Native resources not safe; no new model started')
        try:child=m.base.identity(live['child_identity']['pid'])
        except FileNotFoundError:child=None
        if child is None:
            for expected in identities:
                try:now=m.base.identity(expected['pid'])
                except FileNotFoundError:continue
                assert all(now[k]==expected[k] for k in m.base.IDENTITY_KEYS)
                assert not (now['sig_ign']|now['sig_blk'])&(1<<(15-1))
                fd=os.pidfd_open(now['pid'])
                try:signal.pidfd_send_signal(fd,signal.SIGTERM)
                finally:os.close(fd)
        time.sleep(5)
    assert m.light.load(q/'resources_safe.json')['all_holders_safe'] is True
    rows=m.light.load(old['selected_sources']['path']);shard=old['source_selection']['shard']
    started=set(os.listdir(oldout/'visual'/('shard-%04d'%shard)/'sources'))
    completed={m.json.loads(x)['id'] for x in (oldout/'completed.jsonl').read_bytes().splitlines()}
    remaining=[r for r in rows if r['id'] not in started|completed];assert remaining
    name=args.name;plan=m.RT/(name+'_selected.json');prefix=m.RT/(name+'_old_completed_prefix.jsonl');cfgpath=m.RT/(name+'_launch.json')
    m.light.write_once(plan,remaining)
    with prefix.open('xb') as f:f.write((oldout/'completed.jsonl').read_bytes());f.flush();os.fsync(f.fileno())
    owners=[]
    for g in old['gpus']:
        v=m.light.load(q/('holder_restored_gpu%d.json'%g));i=m.base.identity(v['holder_identity']['pid']);i.update(gpu=g,uuid=m.base.uuid_for(g));owners.append(i)
    paths=[Path(m.__file__),Path(m.light.__file__),m.RT/'resource_manager.py',m.RT/'bench_manager_driver_v2.py',old_path,q/'resources_safe.json',plan,prefix]
    cfg=dict(node=old['node'],gpus=old['gpus'],mode='consumer',release_authorized=True,phase=name,targetcount=len(remaining),input_count=len(remaining),holders=owners,
        out=str(m.BAR/(name+'_execution')),draft_out=str(m.BAR/name),queue=str(m.BAR/'queues'/name),cwd=str(m.RT),config_path=str(cfgpath),reuse_existing_results=False,
        previous_safe=m.ref(q/'resources_safe.json'),previous_config=m.ref(old_path),old_out=str(oldout),old_completed_prefix=str(prefix),source_root=args.source_root,
        producer=m.ref(m.__file__),resource_manager_ref=m.ref(m.__file__),plan=m.ref(plan),modelref=old['modelref'],
        env=dict(CUDA_VISIBLE_DEVICES=','.join(map(str,old['gpus'])),MS_VISUAL_CPU_SITEPATH=m.CPU_SITE,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',VLLM_NO_USAGE_STATS='1'),
        source_sha256={str(p.resolve()):m.base.sha(p) for p in paths})
    cfg['argv']=['/opt/venv_vllm/bin/python','-B',str(Path(m.__file__).resolve()),'worker','--config',str(cfgpath)]
    m.validate(cfg)
    m.light.write_once(m.RT/(name+'_exclusion.json'),dict(original_config_resume_forbidden=True,old_config=m.ref(old_path),new_assignment=m.ref(plan)))
    m.light.write_once(cfgpath,cfg)
    print(m.json.dumps({'event':'start_light_draft_lane','node':cfg['node'],'gpus':cfg['gpus'],'selected':len(remaining)}),flush=True)
    a=['/opt/venv_vllm/bin/python','-B',str(Path(m.__file__).resolve()),'--config',str(cfgpath),'--config-sha',m.base.sha(cfgpath),'--manager-sha',m.BASE_SHA]
    os.execv(a[0],a)
if __name__=='__main__':main()
