"""Submit the recovery job with OSS mount credentials from the local secret file."""
import configparser, subprocess
from pathlib import Path

cfg = configparser.ConfigParser()
cfg.read('/ssd/2/niezichao.nzc/.secrets/ossutilconfig_beijing')
c = cfg['Credentials']
pkg = Path(__file__).resolve().parents[1]
job = __import__('json').loads((pkg / 'config.json').read_text())['run_id']
cmd = ['nebula-cli', 'run', 'mdl', '--user_params=', '--queue=future_hz_qwenvideo',
       '--worker_count=1', '--worker_gpu=8', '--worker_cpu=1600', '--worker_memory=262144',
       '--algo_name=pytorch280', '--entry=cloud/entry_shard.py',
       '--job_name=' + job, '--max_failover_times=0',
       '--oss_bucket=futurelab-dpro-hz', '--oss_endpoint=https://' + c['endpoint'],
       '--oss_access_id=' + c['accessKeyID'], '--oss_access_key=' + c['accessKeySecret'],
       '--ignore=*.pyc,__pycache__/*']
raise SystemExit(subprocess.run(cmd, cwd=pkg).returncode)
