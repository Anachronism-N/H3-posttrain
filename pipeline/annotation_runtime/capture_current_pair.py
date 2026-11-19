import argparse,importlib.util,json
from pathlib import Path
RT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('capture_build_current',RT/'capture_build.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ap=argparse.ArgumentParser();ap.add_argument('--node',required=True);ap.add_argument('--gpus',nargs=2,type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
pids=[]
for gpu in a.gpus:
 uuid=m.manager.uuid_for(gpu);apps=[p for p in m.manager.apps() if p['uuid']==uuid]
 if len(apps)!=1:raise RuntimeError('Expected one known project holder, not unknown compute')
 identity=m.manager.identity(apps[0]['pid'])
 if identity['argv'][-1:]!=[str(m.manager.HOLDER)]:raise RuntimeError('GPU is occupied by a non-holder; refusing capture')
 pids.append(apps[0]['pid'])
result=m.capture(a.node,a.gpus,pids);m.write_new(a.out,result)
print(json.dumps({'path':a.out,'sha256':m.manager.sha(a.out),'holders':[{'gpu':h['gpu'],'pid':h['pid'],'start':h['start_ticks']} for h in result['holders']],'signals_sent':False}))
