"""Generate six fixed joint/aswap clips per audited configuration; one cold loader per host."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess

SSH = ['ssh', '-o', 'StrictHostKeyChecking=yes', '-o',
       'UserKnownHostsFile=/tmp/od_known_hosts_20261012', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=10']


def guard_code(sampler, holder, gpu):
    return f'''import pathlib,subprocess,json,os,signal,select
sampler={sampler!r};holder={holder!r};gpu={gpu}
mem={{r.split()[0][:-1]:int(r.split()[1]) for r in pathlib.Path('/proc/meminfo').read_text().splitlines()}}
c=pathlib.Path('/sys/fs/cgroup/memory')
limit=int((c/'memory.limit_in_bytes').read_text());used=int((c/'memory.usage_in_bytes').read_text())
assert mem['MemAvailable']>=180*2**20 and limit-used>=180*2**30, 'Host/cgroup memory admission failed'
samplers=[];holders={{}}
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
  if sampler in args:samplers.append(int(p.name))
  if holder in args:
   env={{s.split('=',1)[0]:s.split('=',1)[1] for s in (p/'environ').read_bytes().decode().split(chr(0)) if '=' in s}}
   if env.get('CUDA_VISIBLE_DEVICES',env.get('OCCUPY_GPU'))==str(gpu):holders[int(p.name)]=args
 except (OSError,UnicodeError):pass
assert len(samplers)<=4, 'Too many existing H3 samplers'
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','-i',str(gpu)],text=True).strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
active=[int(r.split(',')[1].strip()) for r in apps if r.split(',')[0].strip()==uuid]
assert all(pid in holders for pid in active), ('Nonholder GPU compute process',active)
pathlib.Path('/tmp/od_reserve_'+str(gpu)).touch()
for pid in holders:
 try:
  assert holder.encode() in pathlib.Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(bytes([0]))
  fd=os.pidfd_open(pid);os.kill(pid,signal.SIGTERM)
  assert select.select([fd],[],[],20)[0], ('Holder did not exit',pid)
  os.close(fd)
 except ProcessLookupError:pass
print(json.dumps(dict(admission='passed',memavailable_gib=mem['MemAvailable']/2**20,cgroup_free_gib=(limit-used)/2**30,existing_samplers=samplers,released_holder_pids=list(holders))),flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Experiment code tree')
    parser.add_argument('--audit_report', type=Path, required=True)
    parser.add_argument('--configs', nargs='+', required=True, help='Audited training directory names')
    parser.add_argument('--hosts', nargs='+', required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--integrity', type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.configs)) != len(args.configs) or len(set(args.hosts)) != len(args.hosts):
        parser.error('Require unique configuration names and hosts')
    if not 0 <= args.gpu < 8:
        parser.error('GPU must be in [0,7]')
    if args.report.exists() or args.integrity.exists() or not args.report.parent.is_dir() or not args.integrity.parent.is_dir():
        parser.error('Require existing parents and new report paths')
    audit = json.loads(args.audit_report.read_text())
    assert audit['status'] == 'verified'
    verified = {r['name']: r for r in audit['configurations']}
    exp = args.root / 'logs_omnidance/exp'
    sampler = args.root / 'scripts_omnidance/exp/h3ar/sample_h3_ar.py'
    holder = args.root / 'scripts_omnidance/exp/hold_gpu.py'
    jobs = []
    for name in args.configs:
        if name not in verified or verified[name]['stage'] != 'bridge_clip' or not name.startswith('musicfm_scale_'):
            parser.error('Require audited CLIP scale configurations')
        assert Path(name).name == name
        checkpoint = exp / 'MS/music_encoders' / name / 'best.pt'
        summary = json.loads((checkpoint.parent / 'summary.json').read_text())
        claimed = next(c['sha256'] for c in verified[name]['checkpoints'] if c['name'] == 'best')
        assert summary['checkpoint_sha256']['best'] == claimed
        out = exp / 'H3AR' / ('val_musicreplace_' + name)
        if out.exists():
            parser.error(f'Refuse existing output tree: {out}')
        (out / 'logs').mkdir(parents=True, exist_ok=False)
        for shard in range(6):
            jobs.append(dict(name=name, checkpoint=str(checkpoint), checkpoint_sha256=claimed,
                             out=str(out), shard=shard, log=str(out / 'logs' / f'shard_{shard}.log')))
    queues = [jobs[i::len(args.hosts)] for i in range(len(args.hosts))]
    guard = guard_code(str(sampler), str(holder), args.gpu)

    def run_queue(item):
        host, queue = item
        result = []
        for job in queue:
            command = ['env', f'CUDA_VISIBLE_DEVICES={args.gpu}', 'OMP_NUM_THREADS=2',
                       'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
                       f'PYTHONPATH={args.root.parent / "DiffSynth-Studio-upstream-20260930"}',
                       '/opt/venv_h3/bin/python', str(sampler), '--out', job['out'],
                       '--music_encoder_ckpt', job['checkpoint'], '--shard', str(job['shard']),
                       '--nshard', '6', '--rows', '3', '--conds', 'joint,aswap',
                       '--mode', 'bi', '--steps', '50', '--height', '1280', '--width', '704',
                       '--shift', '12', '--audio_shift', '3', '--seed', '42',
                       '--prompt_mode', 'generic', '--audio_tail_mode', 'repeat_clean']
            remote = 'python3 -c ' + shlex.quote(guard) + ' && ' + shlex.join(command)
            with Path(job['log']).open('x') as log:
                log.write(json.dumps(dict(host=host, command=command,
                                          checkpoint_sha256=job['checkpoint_sha256'])) + '\n')
                log.flush()
                print(json.dumps(dict(event='launch', host=host, name=job['name'], shard=job['shard'])), flush=True)
                process = subprocess.run(SSH + [host, remote], stdout=log, stderr=subprocess.STDOUT)
            result.append(dict(**job, host=host, exit_code=process.returncode))
            print(json.dumps(dict(event='exit', host=host, name=job['name'], shard=job['shard'],
                                  exit_code=process.returncode)), flush=True)
            if process.returncode:
                break
        return result

    with ThreadPoolExecutor(max_workers=len(args.hosts)) as executor:
        results = [r for group in executor.map(run_queue, zip(args.hosts, queues)) for r in group]
    from music_generation_validate import atomic_write_new
    complete = len(results) == len(jobs) and all(r['exit_code'] == 0 for r in results)
    report = dict(expected=len(jobs), finished=len(results), all_jobs_exit_zero=complete,
                  configurations=args.configs, jobs=results, fixed_people=3,
                  conditions=['joint', 'aswap'], integrity_report=str(args.integrity),
                  scope='Launch completion only until independent complete video/audio validation succeeds.')
    atomic_write_new(args.report, report)
    assert json.loads(args.report.read_text()) == report
    if not complete:
        raise SystemExit(1)
    validate = ['/opt/venv_eval/bin/python', str(args.root / 'scripts_omnidance/exp/ms/music_generation_validate.py'),
                '--exp_root', str(exp), '--out', str(args.integrity), '--tags', *args.configs]
    process = subprocess.run(SSH + [args.hosts[0], shlex.join(validate)])
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
