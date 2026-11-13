"""Finalize only after every registered graph suffix is terminal, no redispatch."""
import time,json,subprocess,shlex
from pathlib import Path

root=Path('/opt/music_action_runs/20261113_graph_final_metrics_v5')
shared=Path('/apdcephfs_gy4/share_302533218/cedricnie/music_action_runs/20261113_e2_metrics_v1/graph_parallel_v5')
remote='/opt/music_action_runs/20261113_graph_parallel_v5'
start=time.monotonic()
while True:
 code="from pathlib import Path; import json; r=Path("+repr(remote)+"); print(json.dumps([json.loads((r/('recovery'+str(i))/'manager/graph_suffix_exit.json').read_text())['exit_code'] if (r/('recovery'+str(i))/'manager/graph_suffix_exit.json').exists() else None for i in range(4)]))"
 result=subprocess.run(['ssh','-o','ConnectTimeout=6','28.216.19.143','python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=15)
 if result.returncode:raise RuntimeError('Graph node status inaccessible; no inferred completion')
 exits=json.loads(result.stdout)
 if any(x is not None and x!=0 for x in exits):raise RuntimeError('Registered graph suffix failed; preserve raw and diagnose')
 if all(x==0 for x in exits):break
 if time.monotonic()-start>86400:raise TimeoutError('Graph suffix not completed')
 time.sleep(45)
collected=root/'collected';collected.mkdir(exist_ok=False)
command=['ssh','-o','ConnectTimeout=6','28.216.19.143','tar -C '+shlex.quote(remote)+' -cf - '+' '.join('recovery'+str(i)+'/extractions/graphs.jsonl' for i in range(4))]
source=subprocess.Popen(command,stdout=subprocess.PIPE);target=subprocess.run(['tar','-C',str(collected),'-xf','-'],stdin=source.stdout);source.stdout.close()
if source.wait()!=0 or target.returncode!=0:raise RuntimeError('Graph result transfer failed')
paths=[str(collected/('recovery'+str(i))/'extractions/graphs.jsonl') for i in range(4)]
result=subprocess.run(['/opt/venv_music_action_20261106/bin/python','-m','captionlab.graph_finalize','--run',str(shared),'--graphs',*paths,'--nli-model','/opt/sol61_metric_models/nli-deberta-v3-base','--device','cpu','--out',str(root/'metrics')],cwd=root/'code',check=True)
import shutil
shutil.copytree(root/'metrics',shared/'final_metrics_v5')
print('Registered graph comparison completed, explicit unknowns retained',flush=True)
