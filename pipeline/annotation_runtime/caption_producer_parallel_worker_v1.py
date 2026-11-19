"""K-way parallel native IOv4 producer on one admitted GPU (N2G1, collab3 shard 1).
Parent = sole native-admitted child: gates, spawns K native producer children on disjoint id files,
forwards TERM, and verifies every GPU app on the GPU is its own descendant.  Children are unmodified
native IOv4 producer runs; ready lines use the native flock append to the original shard ready file.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import caption_producer_parallel_handoff_v1 as hand

SELF = Path(__file__).resolve()
NATIVE_IO_SHA = '2dbf2b3da4bf838132b54ba8bcfbed90f5a3e9f8775c7a4ada56aaf0add6e735'


def contract(cfg):
    old = hand.checked(cfg['old_config'])
    if cfg['old_config'] != {'path': str(hand.OLD_CONFIG), 'sha256': hand.OLD_SHA}:
        raise RuntimeError('Only frozen old n2p1 config accepted')
    if (cfg['node'] != hand.NODE or cfg['gpus'] != [hand.GPU] or cfg['mode'] != 'producer'
            or cfg['source_root'] != str(hand.SOURCE_ROOT) or cfg['input_count'] != 222939
            or cfg['env']['CUDA_VISIBLE_DEVICES'] != str(hand.GPU) or cfg['phase'] != hand.PHASE
            or cfg['out'] != str(hand.B / hand.PHASE) or cfg['queue'] != str(hand.B / 'queues' / hand.PHASE)
            or cfg['source_selection'] != {'shard': hand.SHARD, 'nshard': hand.NSHARD, 'limit': 0}
            or cfg['old_config_resume_forbidden'] is not True or cfg['parallel_children'] != hand.K):
        raise RuntimeError('Fixed parallel producer node/GPU/source scope changed')
    proof = hand.checked(cfg['source_handoff'])
    if (proof['schema'] != 'caption_producer_parallel_handoff_v1' or proof['old_config'] != cfg['old_config']
            or proof['remaining_meta'] != cfg['remaining_meta'] or proof['child_ids_files'] != cfg['child_ids_files']
            or proof['source_owner_absence'] is not True or proof['old_gpu_tree_absent'] is not True
            or proof['old_config_resume_forbidden'] is not True or proof['K'] != hand.K):
        raise RuntimeError('Exact source handoff proof changed')
    original = hand.checked(old['selected_sources'])
    rows = hand.validate_meta(hand.checked(cfg['remaining_meta']), original)
    excluded = {r['id'] for r in hand.validate_meta(proof['excluded_started_meta'], original)}
    if rows != [r for r in original if r['id'] not in excluded] or len(rows) != cfg['targetcount'] or cfg['plan'] != cfg['remaining_meta']:
        raise RuntimeError('Remaining must be exactly original 7500 minus all started sources')
    parts = hand.partition(rows)
    for i, (part, binding) in enumerate(zip(parts, cfg['child_ids_files'])):
        if Path(binding['path']).read_text().splitlines() != [r['id'] for r in part] or hand.sha(binding['path']) != binding['sha256']:
            raise RuntimeError('Child id partition differs from round-robin remaining order')
        if cfg['child_native_argv'][i] != hand.native_args(old, binding['path']):
            raise RuntimeError('Native pipeline/ROI/BeatThis args changed')
    if (cfg['producer'] != hand.ref(SELF) or cfg['resource_manager'] != hand.ref(hand.RT / 'caption_producer_parallel_manager_v1.py')
            or cfg['handoff_code'] != hand.ref(hand.RT / 'caption_producer_parallel_handoff_v1.py')):
        raise RuntimeError('Actual parallel worker/manager/handoff code changed')
    hand.verify_prefix(proof['old_ready_prefix'])
    return rows, parts, proof


def parent_gate(base, cfg, config_path, proof):
    hand.fixed_node(cfg)
    hand.old_owner_absence(base, proof)
    queue = Path(cfg['queue'])
    admission = hand.load(queue / 'admission.json')
    parent = base.identity(os.getppid())
    hand.exact_identity(base, admission['manager_identity'], parent)
    expected_argv = cfg['argv'][:3] + [cfg['resource_manager']['path'], '--config', str(config_path),
                                      '--config-sha', hand.sha(config_path), '--manager-sha', hand.BASE_SHA]
    if (parent['argv'] != expected_argv or admission['node'] != hand.NODE or admission['gpus'] != [hand.GPU]
            or admission['config_sha256'] != hand.sha(config_path) or admission['manager_sha256'] != hand.BASE_SHA
            or admission['producer'] != cfg['producer'] or admission['plan'] != cfg['plan']
            or admission['out'] != cfg['out'] or admission['targetcount'] != cfg['targetcount']):
        raise RuntimeError('Exact native admission parent/argv/source binding mismatch')
    current = base.identity(os.getpid())
    if (current['ppid'] != parent['pid'] or current['argv'] != cfg['argv'] or current['uid'] != parent['uid']
            or current['cgroup'] != parent['cgroup'] or current['CVD'] != ['CUDA_VISIBLE_DEVICES=%d' % hand.GPU]
            or os.environ.get('CUDA_VISIBLE_DEVICES') != str(hand.GPU)):
        raise RuntimeError('Exact parallel parent identity/CVD/parent chain changed')
    hand.lock_state(parent['pid'])
    hand.fresh_lease()
    if base.uuid_for(hand.GPU) != proof['gpu_uuid']:
        raise RuntimeError('Actual GPU UUID changed')
    if [a for a in base.apps() if a['uuid'] == proof['gpu_uuid']]:
        raise RuntimeError('GPU must be empty before parallel children load models')
    deadline = time.monotonic() + 5
    while not (queue / 'live.json').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Native parent live receipt missing')
        time.sleep(.05)
    live = hand.load(queue / 'live.json')
    hand.exact_identity(base, live['child_identity'], current)
    hand.exact_identity(base, live['manager_identity'], parent)
    if live['argv'] != cfg['argv']:
        raise RuntimeError('Native child live argv mismatch')
    return {'parent': parent, 'child': current, 'gpu_uuid': proof['gpu_uuid']}


def own_gpu_apps(base, uuid):
    """Every app on the admitted GPU must be this parent or a descendant (no foreign jobs)."""
    values = hand.proc_inventory(base)
    mine = hand.descendants(values, os.getpid())
    running = [a for a in base.apps() if a['uuid'] == uuid]
    foreign = [a['pid'] for a in running if a['pid'] not in mine]
    if foreign:
        raise RuntimeError('Foreign GPU app on admitted GPU: ' + str(foreign))
    return len(running)


def child_preexec():
    ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def run_parent(config_path):
    cfg = hand.load(config_path)
    rows, parts, proof = contract(cfg)
    if str(config_path) != str(hand.RT / (hand.PHASE + '_launch.json')):
        raise RuntimeError('Only immutable fixed launch config is accepted')
    base = hand.native_base()
    owner = parent_gate(base, cfg, config_path, proof)
    available, blocked = hand.split_remaining(rows)
    if blocked or len(available) != len(rows):
        raise RuntimeError('Remaining source has started since handoff; no replay permitted')
    execution = Path(cfg['out'])
    if not execution.is_dir() or execution.is_symlink():
        raise RuntimeError('Native parent-owned execution directory missing or unsafe')
    hand.write_once(execution / 'producer_owner_claim.json', {'config': hand.ref(config_path), **owner,
                    'ready_writer': str(hand.READY), 'K': hand.K, 'old_config_resume_forbidden': True,
                    'no_retry_after_failure': True})
    children, stopping = [], []

    def forward(signum, frame):
        stopping.append(signum)
        for child in children:
            if child.poll() is None:
                child.terminate()  # Only our own direct Popen children.

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    for i in range(hand.K):
        log = (execution / ('child%d.log' % i)).open('xb')
        argv = cfg['argv'][:3] + [str(SELF), '--config', str(config_path), '--child', str(i)]
        children.append(subprocess.Popen(argv, cwd=cfg['cwd'], stdout=log, stderr=subprocess.STDOUT,
                                         preexec_fn=child_preexec))
        log.close()
    hand.write_once(execution / 'children.json', [dict(index=i, pid=c.pid, identity=base.identity(c.pid))
                                                  for i, c in enumerate(children)])
    try:
        while any(c.poll() is None for c in children):
            if not stopping:
                own_gpu_apps(base, owner['gpu_uuid'])
            time.sleep(30 if not stopping else 1)
    except BaseException:
        forward(signal.SIGTERM, None)
        for child in children:
            child.wait()
        raise
    codes = [c.wait() for c in children]
    results = []
    for i in range(hand.K):
        text = (execution / ('child%d.log' % i)).read_text(errors='replace').strip().splitlines()
        try:
            results.append(json.loads(text[-1]))
        except (IndexError, json.JSONDecodeError):
            results.append(None)
    if stopping or any(codes):
        raise SystemExit('Parallel producer children exit codes %s stopping=%s' % (codes, bool(stopping)))
    counts = {}
    for result in results:
        for key, value in result['counts'].items():
            counts[key] = counts.get(key, 0) + value
    summary = {'summary': {'stage': 'run', 'plan_sha256': cfg['plan']['sha256'], 'total_source_count': 222939,
                           'selected_source_count': len(rows), 'processed_source_count': sum(counts.values())},
               'selection': {'selected_ids': [r['id'] for r in rows]}, 'child_results': results, 'counts': counts,
               'source_handoff': cfg['source_handoff'], 'semantic_approval': False, 'source_failures_preserved': True}
    hand.write_once(execution / 'run.summary.json', summary)
    return summary


def run_child(config_path, index):
    cfg = hand.load(config_path)
    base = hand.native_base()
    parent = base.identity(os.getppid())
    if parent['argv'] != cfg['argv'] or parent['pid'] != os.getppid() or not 0 <= index < hand.K:
        raise RuntimeError('Child must be a direct child of the admitted parallel parent')
    binding = cfg['child_ids_files'][index]
    if hand.sha(binding['path']) != binding['sha256']:
        raise RuntimeError('Child id file changed')
    expected = {r['id']: r for r in hand.load(cfg['remaining_meta']['path'])}
    mine = set(Path(binding['path']).read_text().splitlines())
    if not mine <= set(expected):
        raise RuntimeError('Child ids outside remaining')
    if hand.sha(hand.RT / 'curated_producer_io_v4.py') != NATIVE_IO_SHA:
        raise RuntimeError('Native IOv4 SHA changed')
    import curated_producer_io_v4 as io
    native = io.base.load_pipeline
    root = hand.SOURCE_ROOT

    def load_pipeline(path=io.base.FROZEN):
        pipeline = native(path)
        append, save = pipeline.append, pipeline.save

        def routed_append(path, record):
            if Path(path) == root / 'ready.jsonl':
                meta = expected.get(record.get('id'))
                if record.get('id') not in mine or meta is None or any(record.get(k) != meta[k] for k in hand.KEYS):
                    raise RuntimeError('Ready publication outside this child partition')
                path = hand.READY
            return append(path, record)

        def bound_save(path, record):
            if Path(path).name == 'protocol.json':
                record = {**record, 'collab3_member': hand.MEMBER, 'caption_producer_parallel_worker': cfg['producer'],
                          'caption_producer_parallel_child': index, 'caption_producer_parallel_K': hand.K,
                          'caption_producer_source_handoff': cfg['source_handoff'], 'caption_producer_child_ids': binding}
            return save(path, record)

        pipeline.append, pipeline.save = routed_append, bound_save
        return pipeline

    io.base.load_pipeline = load_pipeline
    return io.main(cfg['child_native_argv'][index])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--child', type=int)
    args = parser.parse_args(argv)
    if args.child is None:
        return run_parent(Path(args.config))
    return run_child(Path(args.config), args.child)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False), flush=True)
