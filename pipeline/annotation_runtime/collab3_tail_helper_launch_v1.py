"""Launch one collab3 tail helper on a node (same GPU as that shard's producer). Usage: launch <member> <helper-id>"""
import json, os, shlex, subprocess, sys
from pathlib import Path
RT = Path(__file__).resolve().parent
sys.path.insert(0, str(RT))
import curated_collaborative3_plan_v1 as plan

member, helper = sys.argv[1], sys.argv[2]
node, mode, gpus, shard, nshard = plan.MEMBERS[member]
base = json.loads((RT / 'curated_collaborative3_n2p1_v1_launch.json').read_text())
argv = list(base['argv'])
argv[3] = str(RT / 'collab3_tail_helper_v1.py')
argv[argv.index('--member') + 1] = member
argv[argv.index('--shard') + 1] = str(shard)
argv[argv.index('--roi-model-reference') + 1] = str(RT / ('curated_collaborative3_roi_%s_v1.json' % node))
argv += ['--helper-id', helper]
env = dict(base['env'], CUDA_VISIBLE_DEVICES=str(gpus[0]))
logs = plan.OUT / 'tail_helper_v1' / 'logs'
logs.mkdir(parents=True, exist_ok=True)
log = logs / (helper + '.log')
if log.exists():
    raise SystemExit('helper id already used: ' + helper)
envs = ' '.join('%s=%s' % (k, shlex.quote(v)) for k, v in env.items())
remote = 'cd %s && env %s setsid nohup %s > %s 2>&1 < /dev/null & echo $!' % (
    shlex.quote(base['cwd']), envs, ' '.join(shlex.quote(a) for a in argv), shlex.quote(str(log)))
out = subprocess.run(['ssh', '-f', '-n', '-o', 'BatchMode=yes', plan.HOSTS[node], remote], capture_output=True, text=True, timeout=60)
print(json.dumps(dict(member=member, helper=helper, node=node, gpu=gpus[0], shard=shard, pid=out.stdout.strip(), err=out.stderr.strip(), log=str(log))))
