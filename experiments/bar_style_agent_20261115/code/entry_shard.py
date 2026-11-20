"""Generic cells entry: read per-song catalogs from an OSS root, run variants in parallel."""
import json, os, shutil, subprocess, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
JOB=Path(__file__).resolve().parents[1]; CFG=json.loads((JOB/'config.json').read_text())
SONGS=CFG['songs']; VARIANTS=CFG['variants']; CATROOT=Path(CFG['catalog_root']); ROOTS=CFG.get('catalog_root_by_song',{})
LOCAL=Path('/tmp')/(CFG['run_id']); REMOTE=Path(CFG['remote_root'])
REPLICAS=[('0,1',8101),('2,3',8102),('4,5',8103),('6,7',8104)]
def wait(port,seconds=900):
    for _ in range(seconds//5):
        try:
            with urllib.request.urlopen('http://127.0.0.1:%d/v1/models'%port,timeout=3) as r: json.loads(r.read()); return
        except Exception: time.sleep(5)
    raise RuntimeError('unhealthy %d'%port)
def copytree(src,dst):
    dst.mkdir(parents=True,exist_ok=True)
    for item in sorted(Path(src).rglob('*')):
        if item.is_file() and '.cache' not in item.parts:
            t=dst/item.relative_to(src); t.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(item,t)
def main():
    LOCAL.mkdir(parents=True,exist_ok=False); REMOTE.mkdir(parents=True,exist_ok=True)
    venv=LOCAL/'venv'; venv.mkdir(); subprocess.run(['tar','-xf',CFG['venv_archive'],'-C',str(venv)],check=True)
    python=next(venv.glob('*/bin/python'))
    vlm=LOCAL/'vl30b'; copytree(Path(CFG['models_dir'])/'Qwen3-VL-30B-A3B-Instruct',vlm)
    for sid in SONGS: copytree(JOB/'assets/panel43_songs'/sid, LOCAL/'songs'/sid)
    env=dict(os.environ,PYTHONPATH=str(JOB),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
             OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    servers=[]; providers=[]
    for devices,port in REPLICAS:
        log=(LOCAL/('vl_%d.log'%port)).open('w')
        p=subprocess.Popen([str(python),str(JOB/'cloud/text_server.py'),'--model',str(vlm),'--device','auto','--port',str(port)],
            stdout=log,stderr=subprocess.STDOUT,env=dict(env,CUDA_VISIBLE_DEVICES=devices,
            PYTORCH_ALLOC_CONF='expandable_segments:True',TEXT_SERVER_MAX_MEMORY='0:36GiB,1:36GiB'))
        servers.append((p,log)); wait(port)
        path=LOCAL/('provider_%d.json'%port)
        path.write_text(json.dumps(dict(kind='chat_completions',base_url='http://127.0.0.1:%d/v1'%port,
            model='model',api_key_env='',timeout_seconds=2400,temperature=0,json_mode=True,token_parameter='max_tokens')))
        providers.append(path)
    print('CELLS_SERVICES_READY',flush=True)
    def cell(args):
        sid,arm,flags,idx=args; d=LOCAL/'songs'/sid; out=LOCAL/'runs'/sid/arm
        log=(LOCAL/('cell_%s_%s.log'%(sid,arm))).open('w')
        run_arm=arm if arm in ('W','SF','RA','O') else 'O'
        cmd=[str(python),str(JOB/'cloud/run_cell.py'),str(Path(ROOTS.get(sid,str(CATROOT)))/sid/'results.json'),sid,run_arm,
             str(providers[idx%len(providers)]),str(out),'--music',str(d/'music.json'),'--style',str(d/'style_prior.json')]+flags
        r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=log,env=env,text=True); log.close()
        line=next((x for x in r.stdout.splitlines() if x.startswith('CELL ')),None)
        print('CELL_DONE %s %s %s'%(sid,arm,'ok' if line else 'fail'),flush=True)
        return json.loads(line[5:]) if line else dict(song_id=sid,arm=arm,status='dispatch_error',error='no CELL',result=None,logical_results=[])
    tasks=[(sid,arm,flags) for sid in SONGS for arm,flags in VARIANTS]
    rows=[]
    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        for wi in range(0,len(tasks),len(providers)):
            batch=tasks[wi:wi+len(providers)]
            rows.extend(pool.map(cell, [(t[0],t[1],t[2],i) for i,t in enumerate(batch)]))
    registry=dict(schema='choreobench.cells_registry.v1',songs=SONGS,variants=[v[0] for v in VARIANTS],runs=rows)
    (LOCAL/'runs').mkdir(parents=True,exist_ok=True)
    (LOCAL/'runs/registry.json').write_text(json.dumps(registry,ensure_ascii=False)+'\n')
    print('FINAL_REGISTRY '+json.dumps([dict(song=r['song_id'],arm=r['arm'],
        complete=(r.get('result') or {}).get('full_song_succeeded'),
        calls=(r.get('result') or {}).get('calls_used'),
        stop=(r.get('result') or {}).get('stop_reason')) for r in rows],ensure_ascii=False),flush=True)
    shutil.copytree(LOCAL/'runs',REMOTE/'runs',dirs_exist_ok=True)
    for f in LOCAL.glob('*.log'): shutil.copyfile(f,REMOTE/f.name)
    for p,log in servers: log.close(); p.terminate()
if __name__=='__main__': main()
