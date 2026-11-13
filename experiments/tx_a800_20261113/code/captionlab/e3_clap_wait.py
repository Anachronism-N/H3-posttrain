"""Await original registered outputs before independent non-generative calculation."""
from pathlib import Path
import subprocess,json,time,shlex,shutil
r=Path('/opt/music_action_runs/20261113_e3_clap_metrics_v1');formal=Path('/apdcephfs_gy4/share_302533218/cedricnie/choreobench_runs/20261113_sol61_e3_formal_v1');remote='/opt/music_action_runs/20261113_e3_observe_catalog_v1';start=time.monotonic()
while True:
 code="from pathlib import Path; import json; r=Path("+repr(remote)+"); print(json.dumps({'ready':(r/'bindings.json').exists(),'failed':(r/'manager/observe_exit.json').exists() and json.loads((r/'manager/observe_exit.json').read_text())['exit_code']!=0}))"
 q=subprocess.run(['ssh','-o','ConnectTimeout=6','28.216.19.213','python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=15)
 if q.returncode:raise RuntimeError('Original observation node inaccessible')
 ready=json.loads(q.stdout)
 if (formal/'execution_results.json').exists() and ready['ready']:break
 if time.monotonic()-start>86400:raise TimeoutError('Registered final plans not available; no substitute outputs')
 time.sleep(45)
obs=r/'observations';obs.mkdir(exist_ok=False);src=subprocess.Popen(['ssh','-o','ConnectTimeout=6','28.216.19.213','tar -C '+remote+' -cf - bindings.json audio_assets.json'],stdout=subprocess.PIPE);tar=subprocess.run(['tar','-C',str(obs),'-xf','-'],stdin=src.stdout);src.stdout.close()
if src.wait()!=0 or tar.returncode!=0:raise RuntimeError('Observation metadata transfer failed')
catalog='/opt/music_action_runs/20261113_e3_moss_catalog_v1'
src=subprocess.Popen(['ssh','-o','ConnectTimeout=6','28.216.19.213','tar -C '+catalog+' -cf - '+' '.join('shard'+str(i)+'/catalog/audio' for i in range(4))],stdout=subprocess.PIPE)
tar=subprocess.run(['tar','-C',str(obs),'-xf','-'],stdin=src.stdout);src.stdout.close()
if src.wait()!=0 or tar.returncode!=0:raise RuntimeError('Original PCM batch transfer failed')
assets=json.loads((obs/'audio_assets.json').read_text())
for asset in assets['assets']:
 path=obs/Path(asset['path']).relative_to(catalog);import hashlib
 if hashlib.sha256(path.read_bytes()).hexdigest()!=asset['sha256']:raise ValueError('Copied original PCM differs')
 asset['path']=str(path)
(obs/'audio_assets.json').write_text(json.dumps(assets))
subprocess.run(['/opt/venv_music_action_20261106/bin/python','-m','captionlab.e3_clap_metrics','--formal',str(formal),'--observations',str(obs),'--model','/opt/sol61_metric_models/clap-htsat-unfused','--device','cpu','--out',str(r/'metrics')],cwd=r/'code',check=True)
shutil.copytree(r/'metrics',formal/'clap_metrics')
print('Full registered non-generative audio-text diagnostics completed',flush=True)
