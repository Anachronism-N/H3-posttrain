"""Start each frozen staged native configuration once; never restart a used queue."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def start(root, pair):
    root=Path(root);sys.path.insert(0,str(root/'code'))
    from stage2_local_publication_v1 import checked,write_once
    fd=os.open(root/'deployment_start.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    d=json.loads((root/'deployment.json').read_bytes());d['configs']+= [json.loads(p.read_bytes()) for p in sorted((root/'extra_deployments').glob('*.json'))];e=next(x for x in d['configs'] if x['gpus']==pair);c=checked(e['config_ref']);spec=importlib.util.spec_from_file_location('staged_native_manager',c['producer']['path']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);phase=e['phase'];logs=root/'logs';logs.mkdir(exist_ok=True)
    # Finish the previous node-local load before charging another loading budget.
    deadline=time.monotonic()+360
    for other in d['configs']:
        if other==e:continue
        p=logs/(other['phase']+'_launch.json')
        if not p.exists():continue
        oc=checked(other['config_ref']);q=Path(oc['queue']);state=root/'worker_state'/oc['owner'];state=state/oc['phase'] if oc.get('worker_state_layout')=='by_phase' else state
        while not (state/'load.receipt.json').exists():
            if (q/'failure.json').exists():raise RuntimeError('Previous admitted native job failed; do not expand')
            if time.monotonic()>deadline:raise RuntimeError('Previous model load still pending; no extra admission')
            time.sleep(1)
    m.check_sources(c,e['config_ref']['path'],e['config_ref']['sha256'],m.ORIGINAL_BASE_SHA);m.validate(c)
    for other in d['configs']:(root/'out'/'ready'/other['owner']).mkdir(parents=True,exist_ok=True)
    if not (logs/'uploader_launch.json').exists():raise RuntimeError('Start the node single uploader first')
    write_once(logs/(phase+'_launch.json'),dict(node=c['node'],owner=c['owner'],gpus=pair,config_ref=e['config_ref'],start_unix=time.time(),fresh_headroom_gib=m.headroom()['headroom']/2**30,launch_once=True,global_source_count=d['global_pool_count'],training_eligible=False))
    sock=str(root/'tmux.sock');session='main3936'
    def window(name,argv,log):
        cmd=shlex.join(argv)+' > '+shlex.quote(str(log))+' 2>&1'
        subprocess.run(['tmux','-S',sock,'new-window','-d','-t',session,'-n',name,'-c','/tmp',cmd],check=True)
    window(phase,['/opt/venv_vllm/bin/python','-B',c['producer']['path'],'--config',c['config_path'],'--config-sha',e['config_ref']['sha256'],'--manager-sha',m.ORIGINAL_BASE_SHA],logs/(phase+'_manager.log'))
    q=Path(c['queue']);deadline=time.monotonic()+60
    while not (q/'live.json').exists():
        if (q/'failure.json').exists():raise RuntimeError((q/'failure.json').read_text())
        if time.monotonic()>deadline:raise RuntimeError('Manager started; live receipt pending; feeder not launched')
        time.sleep(.25)
    write_once(logs/(phase+'_feeder_launch.json'),dict(owner=c['owner'],live_ref=str(q/'live.json'),start_unix=time.time()))
    window(phase+'-feeder',['/opt/venv_eval/bin/python','-B',str(root/'code'/('stage2_cpu_feeder_v3.py' if (root/'code/stage2_cpu_feeder_v3.py').exists() else 'stage2_cpu_feeder_v2.py' if (root/'code/stage2_cpu_feeder_v2.py').exists() else 'stage2_cpu_feeder_v1.py')),'--config',e['feeder_ref']['path']],logs/(phase+'_feeder.log'))
    print(json.dumps(dict(node=c['node'],owner=c['owner'],gpus=pair,manager_and_feeder_started=True,worker_pid=json.loads((q/'live.json').read_bytes())['child_identity']['pid'])),flush=True)
    os.close(fd)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--pair',required=True,choices=('01','23','45','67'));a=p.parse_args();start(a.root,list(map(int,a.pair)))
