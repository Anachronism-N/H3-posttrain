"""One-shot native light-to-local handoff; no unknown signals or budget replay."""
import argparse
import ast
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import signal
import stat
import sys
import time

sys.dont_write_bytecode = True
import caption_local_live_manager_v1 as m

RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
LIGHT_MANAGER = RT / 'caption_two_stage_live_manager_v5.py'
PYTHON = '/opt/venv_vllm/bin/python'
SAFE_KEYS = ('all_holders_safe', 'locks_released', 'leases_restored')
SPLITS = ('train', 'val', 'test')
DEADLINE_SECONDS = 240
MIN_DISK = 8 * 2**30
MIN_HEADROOM = 160 * 2**30
need = m.local.light.need
load = m.local.light.load
write_once = m.local.light.write_once


def file_identity(path):
    path = Path(path).resolve()
    st = path.stat()
    need(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid(), 'Unsafe proof file owner/type')
    digest = m.base.sha(path)
    after = path.stat()
    need((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_uid) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_uid), 'Proof changed during capture')
    return dict(path=str(path), sha256=digest, device=st.st_dev,
                inode=st.st_ino, size=st.st_size, mtime_ns=st.st_mtime_ns, uid=st.st_uid)


def check_files(proofs):
    for proof in proofs:
        need(file_identity(proof['path']) == proof, 'Old proof/config/progress changed after capture')


def same_process(expected, current):
    return current is not None and current['pid'] == expected['pid'] and current['start_ticks'] == expected['start_ticks']


def current_process(expected):
    try:
        return m.base.identity(expected['pid'])
    except FileNotFoundError:
        return None


def exact_process(expected):
    current = current_process(expected)
    need(current is not None and all(current[k] == expected[k] for k in m.base.IDENTITY_KEYS)
         and current['ppid'] == expected['ppid'], 'Recorded process identity/PPID changed')
    need(current['uid'] == os.getuid(), 'Foreign process UID')
    return current


def old_tree_absent(expected):
    need(all(not same_process(item, current_process(item)) for item in expected),
         'Old manager/child/GPU worker still alive; no source handoff')
    return True


def manager_argv(path, producer, digest):
    return [PYTHON, '-B', str(producer), '--config', str(path), '--config-sha', digest,
            '--manager-sha', m.BASE_SHA]


def check_old_receipts(old, old_path, admission, live):
    digest = m.base.sha(old_path)
    producer = m.ref(LIGHT_MANAGER)
    need(old['producer'] == old['resource_manager_ref'] == producer and
         old['config_path'] == str(old_path) and old['cwd'] == str(RT), 'Old native producer/config path differs')
    need(old['release_authorized'] is True and old['gpus'] in ([0, 1], [2, 3], [4, 5], [6, 7]) and
         old['env']['CUDA_VISIBLE_DEVICES'] == ','.join(map(str, old['gpus'])), 'Old authorized GPU assignment differs')
    need(old['argv'] == [PYTHON, '-B', str(LIGHT_MANAGER), 'worker', '--config', str(old_path)],
         'Old native worker argv differs')
    need(admission['config_sha256'] == digest and admission['manager_sha256'] == m.BASE_SHA and
         admission['producer'] == producer and admission['plan'] == old['plan'] and
         admission['source_sha256'] == old['source_sha256'] and
         admission['source_sha256'].get(str(LIGHT_MANAGER)) == producer['sha256'] and
         admission['out'] == old['out'] and admission['node'] == old['node'] and
         admission['gpus'] == old['gpus'] and admission['phase'] == old['phase'] and
         admission['targetcount'] == old['targetcount'] and
         admission['model_identity_sha256'] == old['modelref']['model_identity_sha256'], 'Old admission binding differs')
    parent = admission['manager_identity']
    child = live['child_identity']
    need(child is not None and all(live['manager_identity'][k] == parent[k] for k in m.base.IDENTITY_KEYS) and
         live['manager_identity']['ppid'] == parent['ppid'] and live['argv'] == old['argv'] and
         live['phase'] == old['phase'] and live['targetcount'] == old['targetcount'], 'Old live binding differs')
    need(parent['argv'] == manager_argv(old_path, LIGHT_MANAGER, digest) and child['argv'] == old['argv'] and
         child['ppid'] == parent['pid'] and parent['uid'] == child['uid'] == os.getuid() and
         parent['cgroup'] == child['cgroup'] and child['CVD'] == ['CUDA_VISIBLE_DEVICES=' + old['env']['CUDA_VISIBLE_DEVICES']],
         'Old parent/worker native argv/UID/cgroup/PPID differs')
    need(m.base.sha(old['plan']['path']) == old['plan']['sha256'], 'Old plan SHA changed')
    return parent, child


def check_node(old):
    need(old['node'] in m.HOSTS and m.HOSTS[old['node']] in
         m.h.subprocess.check_output(['hostname', '-I'], text=True).split(), 'Wrong execution node; no signal')


def capture_gpu_tree(old, parent, child):
    exact_process(parent)
    exact_process(child)
    running = m.base.apps()
    captured = {}
    for gpu in old['gpus']:
        uuid = m.base.uuid_for(gpu)
        need(uuid == old['holders'][old['gpus'].index(gpu)]['uuid'], 'Old GPU UUID differs')
        apps = [app for app in running if app['uuid'] == uuid]
        need(len(apps) == 1, 'Old GPU must have exactly one compute app')
        item = m.base.identity(apps[0]['pid'])
        cursor = item
        seen = set()
        for _ in range(64):
            need(cursor['uid'] == parent['uid'] and cursor['cgroup'] == parent['cgroup'], 'Unknown GPU UID/cgroup; no signal')
            if cursor['pid'] == child['pid']:
                need(all(cursor[k] == child[k] for k in m.base.IDENTITY_KEYS), 'GPU worker ancestor identity changed')
                break
            need(cursor['pid'] not in seen and cursor['ppid'] > 1, 'Unknown GPU tree; no signal')
            seen.add(cursor['pid'])
            cursor = m.base.identity(cursor['ppid'])
        else:
            raise ValueError('GPU ancestor bound exceeded; no signal')
        captured[item['pid']] = item
    exact_process(parent)
    exact_process(child)
    return list(captured.values())


def term_exact(expected, deadline):
    need(time.monotonic() < deadline, 'Transition deadline expired; no signal')
    fd = os.pidfd_open(expected['pid'])
    try:
        current = exact_process(expected)
        need(not (current['sig_ign'] | current['sig_blk']) & (1 << (signal.SIGTERM - 1)),
             'Authorized process ignores/blocks TERM; no escalation')
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)


def safe_proof(path):
    safe = load(path)
    need(all(safe.get(key) is True for key in SAFE_KEYS), 'Incomplete native safe proof; no new model')
    return m.ref(path)


def safe_parent_wait(old, parent):
    safe_proof(Path(old['queue']) / 'resources_safe.json')
    restored_owners(old)
    current = current_process(parent)
    if current is not None:
        exact_process(parent)
        need(current['state'] == 'S' and (Path('/proc') / str(parent['pid']) / 'wchan').read_text().strip() == 'do_wait',
             'Safe old manager not in quiescent native holder wait')


def exclusion_details(index):
    def keys(category):
        return {(sid, split) for split, ids in index[category].items() for sid in ids}
    started = keys('sources')
    drafted = keys('drafts')
    return dict(excluded_started_keys=sorted(started), excluded_draft_keys=sorted(drafted),
                excluded_started_without_draft=sorted(started - drafted), budget_not_reset=True)


def park_old(old, parent, child, proofs, record_path):
    deadline = time.monotonic() + DEADLINE_SECONDS
    safe_path = Path(old['queue']) / 'resources_safe.json'
    gpu_tree = []
    if not safe_path.exists():
        check_files(proofs)
        gpu_tree = capture_gpu_tree(old, parent, child)
        write_once(record_path, dict(manager=parent, child=child, confirmed_gpu_tree=gpu_tree,
                                    signals_allowed='TERM only; manager first, confirmed GPU remnants after child exit'))
        check_files(proofs)
        term_exact(parent, deadline)
    else:
        safe_proof(safe_path)
        old_tree_absent([child])
        safe_parent_wait(old, parent)
        write_once(record_path, dict(manager=parent, child=child, confirmed_gpu_tree=[], already_native_safe=True))
    signaled = set()
    expected = [child] + gpu_tree
    while True:
        check_files(proofs)
        if safe_path.exists():
            safe_proof(safe_path)
            if all(not same_process(item, current_process(item)) for item in expected):
                old_tree_absent(expected)
                safe_parent_wait(old, parent)
                return expected, safe_proof(safe_path)
        need(time.monotonic() < deadline, 'Native restoration/old worker exit deadline exceeded; no new model')
        if not same_process(child, current_process(child)):
            for item in gpu_tree:
                if item['pid'] in signaled or not same_process(item, current_process(item)):
                    continue
                # Reparenting after the original direct child exit is expected; all other identity fields remain exact.
                now = current_process(item)
                need(all(now[k] == item[k] for k in m.base.IDENTITY_KEYS), 'GPU remnant identity changed; no signal')
                expected_remnant = dict(item, ppid=now['ppid'])
                term_exact(expected_remnant, deadline)
                signaled.add(item['pid'])
        time.sleep(1)


def row_key(row):
    need(isinstance(row, dict) and row.get('split') in SPLITS and isinstance(row.get('id'), str) and
         row['id'] not in ('', '.', '..') and all(c.isascii() and (c.isalnum() or c in '_-') for c in row['id']),
         'Invalid source key')
    return row['id'], row['split']


def remaining_sources(old):
    rows = load(old['plan']['path'])
    need(isinstance(rows, list) and rows, 'Nonempty old selected plan required')
    keys = [row_key(row) for row in rows]
    need(len(set(keys)) == len(keys) and len(keys) == old['targetcount'], 'Duplicate/count-mismatched old source plan')
    allowed = set(keys)
    excluded = set()
    index = {}
    root = Path(old['draft_out'])
    for category in ('sources', 'drafts'):
        directory = root / category
        category_index = {}
        if directory.exists():
            need(directory.is_dir() and not directory.is_symlink(), 'Unsafe source/draft index directory')
            with os.scandir(directory) as entries:
                splits = {entry.name: entry for entry in entries}
            need(set(splits) <= set(SPLITS), 'Unknown source split in old output')
            for split, entry in splits.items():
                need(entry.is_dir(follow_symlinks=False), 'Unsafe split index')
                with os.scandir(directory / split) as entries:
                    names = []
                    for entry in entries:
                        need(entry.is_dir(follow_symlinks=False), 'Unsafe source entry')
                        key = row_key(dict(id=entry.name, split=split))
                        need(key in allowed, 'Unknown source outside old selected plan')
                        names.append(entry.name)
                        excluded.add(key)
                category_index[split] = sorted(names)
        index[category] = category_index
    # Any directory means started, even only prepared/raw/claim/failed or an interrupted empty directory.
    remaining = [row for row in rows if row_key(row) not in excluded]
    need(remaining, 'No remaining unstarted sources; no new model')
    return remaining, sorted(excluded), index


def code_bindings(old):
    bindings = dict(old['source_sha256'])
    seeds = [SELF, Path(m.__file__), Path(m.local.__file__), Path(m.local.light.__file__),
             Path(m.base.__file__), Path(m.h.__file__), LIGHT_MANAGER,
             Path(m.local.light.native.__file__), Path(m.local.light.schemas.__file__), RT / 'curated_resource_manager.py']
    native = m.local.light.native
    for name, digest in native.FROZEN.items():
        path = native.SOURCE / name
        need(m.base.sha(path) == digest, 'Original frozen native helper changed')
        seeds.append(path)
    seen = set()
    while seeds:
        path = seeds.pop().resolve()
        if path in seen:
            continue
        seen.add(path)
        digest = m.base.sha(path)
        need(str(path) not in bindings or bindings[str(path)] == digest, 'Existing source code binding changed')
        bindings[str(path)] = digest
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for name in names:
                for directory in (path.parent, RT, native.SOURCE):
                    dependency = directory / (name.replace('.', '/') + '.py')
                    if dependency.is_file():
                        seeds.append(dependency)
                        break
    return bindings


def capacity():
    disk = shutil.disk_usage('/tmp')
    memory = m.base.headroom()
    need(disk.free >= MIN_DISK, 'Local /tmp free space below 8 GiB')
    need(memory['headroom'] >= MIN_HEADROOM, 'Headroom below native 160 GiB budget')
    return dict(local_disk_free_bytes=disk.free, memory=memory,
                existing_copy_threads=3, added_copy_threads=1, total_copy_threads=4,
                source_batch_size=3, max_inflight_batches=2)


def build_config(old, old_path, name, owners, plan_path, handoff_path, safe, bindings):
    path = RT / (name + '_launch.json')
    rows = load(plan_path)
    producer = m.ref(m.SELF)
    cfg = dict(node=old['node'], gpus=old['gpus'], mode='consumer', release_authorized=True,
               phase=name, targetcount=len(rows), input_count=len(rows), holders=owners,
               out=str(m.BAR / (name + '_execution')), draft_out=str(m.BAR / name),
               queue=str(m.BAR / 'queues' / name), cwd=str(RT), config_path=str(path),
               reuse_existing_results=False, resource_previous_safe=safe,
               source_handoff=m.ref(handoff_path), previous_config=m.ref(old_path), old_out=old['draft_out'],
               source_root=old['source_root'], modelref=old['modelref'], producer=producer,
               resource_manager_ref=producer, plan=m.ref(plan_path), env=dict(old['env']))
    cfg['argv'] = [PYTHON, '-B', str(m.SELF), 'worker', '--config', str(path)]
    cfg['source_sha256'] = dict(bindings)
    for reference in (cfg['producer'], cfg['plan'], cfg['source_handoff'], safe, cfg['previous_config']):
        cfg['source_sha256'][reference['path']] = reference['sha256']
    return cfg


def restored_owners(old):
    owners = []
    running = m.base.apps()
    for gpu in old['gpus']:
        proof = load(Path(old['queue']) / ('holder_restored_gpu%d.json' % gpu))
        need(proof['gpu'] == gpu, 'Restored GPU index differs')
        owner = dict(proof['holder_identity'], gpu=gpu, uuid=proof['uuid'])
        m.base.check_holder(owner, m.base.identity(owner['pid']), gpu, m.base.uuid_for(gpu), running)
        need(not m.h.lock_snapshot(gpu)['held'], 'Old GPU lock still held')
        m.h.lease_snapshot(gpu)
        owners.append(owner)
    need(len({owner['pid'] for owner in owners}) == len(owners) and
         len({owner['uuid'] for owner in owners}) == len(owners), 'Duplicate restored holder identity')
    return owners


@contextmanager
def handoff_lock(old_phase):
    need(old_phase and all(c.isascii() and (c.isalnum() or c in '_-') for c in old_phase), 'Invalid old phase')
    path = RT / (old_phase + '_light_to_local_handoff.lock')
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        st = os.fstat(fd)
        need(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid() and st.st_size == 0, 'Unsafe transition lock')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)  # Persistent transition lock inode is never deleted.


def final_check(cfg, config_path, expected, proofs, budgets):
    check_files(proofs)
    old_tree_absent(expected)
    need(capacity()['memory']['headroom'] >= MIN_HEADROOM, 'Fresh capacity required')
    old = load(cfg['previous_config']['path'])
    parent = load(Path(old['queue']) / 'admission.json')['manager_identity']
    safe_parent_wait(old, parent)
    # Use the actual local source selector with the original, unmodified base check_sources interface.
    original = m.base.selected_sources
    try:
        m.base.selected_sources = m.selected
        m.base.check_sources(cfg, str(config_path), m.base.sha(config_path), m.BASE_SHA)
    finally:
        m.base.selected_sources = original
    m.validate(cfg)
    need(budgets['total_copy_threads'] == 4, 'Copy concurrency budget changed')


def run(old_path, name, prepare_only=False):
    old_path = Path(old_path).resolve()
    old_proof = file_identity(old_path)
    old = load(old_path)
    need(name.startswith('caption_two_stage_local_') and
         all(c.isascii() and (c.isalnum() or c in '_-') for c in name), 'Independent local lane name required')
    check_node(old)
    with handoff_lock(old['phase']):
        claim_path = RT / (old['phase'] + '_light_to_local_handoff_claim.json')
        need(not claim_path.exists(), 'Old lane handoff already claimed; never execute twice')
        for path in (RT / (name + suffix) for suffix in ('_launch.json', '_remaining.json', '_handoff.json')):
            need(not path.exists(), 'New immutable artifact already exists')
        for path in (m.BAR / name, m.BAR / (name + '_execution'), m.BAR / 'queues' / name):
            need(not path.exists(), 'New output/queue already exists')
        write_once(claim_path, dict(old_config=old_proof, name=name, transition=m.ref(SELF), pid=os.getpid()))
        try:
            queue = Path(old['queue'])
            proofs = [old_proof, file_identity(queue / 'admission.json'), file_identity(queue / 'live.json'),
                      file_identity(old['plan']['path'])]
            admission = load(queue / 'admission.json')
            live = load(queue / 'live.json')
            parent, child = check_old_receipts(old, old_path, admission, live)
            bindings = code_bindings(old)
            expected, safe = park_old(old, parent, child, proofs, RT / (name + '_old_tree.json'))
            owners = restored_owners(old)
            budgets = capacity()
            check_files(proofs)
            old_tree_absent(expected)
            remaining, excluded, index = remaining_sources(old)
            progress = Path(old['draft_out']) / 'progress.json'
            progress_identity = file_identity(progress) if progress.exists() else dict(path=str(progress), absent=True)
            if progress.exists():
                snapshot = RT / (name + '_old_progress_snapshot.json')
                with snapshot.open('xb') as stream:
                    stream.write(progress.read_bytes()); stream.flush(); os.fsync(stream.fileno())
                proofs.append(dict(progress_identity))
                bindings[str(snapshot)] = m.base.sha(snapshot)
                progress_identity['snapshot'] = m.ref(snapshot)
            plan_path = RT / (name + '_remaining.json')
            write_once(plan_path, remaining)
            handoff_path = RT / (name + '_handoff.json')
            write_once(handoff_path, dict(remaining_sources=remaining, excluded_source_keys=excluded,
                directory_index=index, old_worker_tree_absent=True, original_config_resume_forbidden=True,
                old_config=old_proof, old_plan=old['plan'], old_draft_out=old['draft_out'],
                old_progress_identity=progress_identity, old_native_admission=proofs[1], old_native_live=proofs[2],
                source_safety_proof_path=str(handoff_path), resource_safety_proof=safe,
                old_tree=expected, old_manager_identity=parent,
                old_worker_tree_absence_scope='recorded direct child and preconfirmed GPU model apps only; not all CPU descendants',
                safe_old_manager_quiescent_wait_allowed=True,
                remaining_plan=m.ref(plan_path), resource_budget=budgets, **exclusion_details(index)))
            for proof in proofs:
                bindings[proof['path']] = proof['sha256']
            for path in [RT / (name + '_old_tree.json')] + [queue / ('holder_restored_gpu%d.json' % gpu) for gpu in old['gpus']]:
                bindings[str(path.resolve())] = m.base.sha(path)
            cfg = build_config(old, old_path, name, owners, plan_path, handoff_path, safe, bindings)
            config_path = Path(cfg['config_path'])
            write_once(config_path, cfg)
            final_check(cfg, config_path, expected, proofs, budgets)
            argv = manager_argv(config_path, m.SELF, m.base.sha(config_path))
            print(m.json.dumps(dict(event='prepared_local_handoff', selected=len(remaining), excluded=len(excluded),
                config=str(config_path), prepare_only=prepare_only, native_manager_argv=argv)), flush=True)
            if not prepare_only:
                final_check(cfg, config_path, expected, proofs, budgets)
                os.execv(argv[0], argv)
            return cfg
        except BaseException as error:
            write_once(RT / (name + '_handoff_failed.json'), dict(reason=type(error).__name__ + ': ' + str(error),
                original_config_resume_forbidden=True, locks_leases_budgets_not_deleted=True, new_model_not_started=True))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-config', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--prepare-only', action='store_true', help='Still parks the old lane; does not exec the new manager')
    args = parser.parse_args()
    run(args.old_config, args.name, args.prepare_only)


if __name__ == '__main__':
    main()
