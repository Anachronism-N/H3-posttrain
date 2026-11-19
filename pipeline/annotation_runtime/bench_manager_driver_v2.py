"""Native manager lifecycle, fixed visual benchmark driver, and CPU-only fresh config builder."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
BAR = RT.parent
SELF = Path(__file__).resolve()
SOURCE_CFG = RT / 'curated_consumer_prepare_ahead_n8s0_v1_launch_main_v1.json'
SOURCE_CFG_SHA = '5042c5049a865dbacda30dfd314fec058038786ba338461ed3ae7dd8c67c185e'
BASE_SHA = '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
NODE, GPUS, CVD = 'N6', [2, 3], '2,3'
HOST = '28.216.19.225'
SAFE_PHASES = {
    'visual_batch_fixed3_n6g23_main_v1': 'visual_batch_fixed3_n6g23_main_v1',
}
CAPTURE_TTL = 1200


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


bench = module('fixed_visual_bench', RT / 'visual_batch_fixed3_benchmark_v1.py')
c = module('bench_native_curated', RT / 'curated_resource_manager.py')
base = c.base
need = bench.need
need(Path(base.__file__).resolve() == RT / 'resource_manager.py' and base.sha(base.__file__) == BASE_SHA, 'original native lifecycle file identity differs')
c.HOSTS[NODE] = HOST
c.ALLOWED['consumer'][NODE] = [tuple(GPUS)]
base.AUTHORIZED_PAIRS[NODE] = (tuple(GPUS),)


def checked(ref):
    need(base.sha(ref['path']) == ref['sha256'], 'frozen input changed: ' + ref['path'])
    return bench.strict(Path(ref['path']).read_bytes())


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=base.sha(path))


def fixed_selected(cfg):
    manifest = checked(cfg['benchmark_manifest'])
    summary = checked(cfg['manifest_summary'])
    need(summary['total'] == cfg['input_count'] == 222939 and
         summary['manifest'] == cfg['original_full_manifest'], 'full original denominator/provenance differs')
    need(tuple(x['source_id'] for x in manifest['cases']) == bench.SOURCES and
         [x['window_count'] for x in manifest['cases']] == [2, 3, 3] and
         manifest['request_count'] == 8 and manifest['planned_model_request_count'] == 16,
         'fixed three-source/eight-window benchmark scope differs')
    return list(bench.SOURCES), 222939


def summaries(cfg, out):
    fixed_selected(cfg)
    value = bench.strict((out / 'result.json').read_bytes())
    need(value['status'] == 'completed' and value['actual_model_request_count'] == 16 and
         value['original_boundary_unchanged'] and value['production_counts'] == value['production_claims'] == 0,
         'benchmark completion/count/original boundary not verified')
    planned = checked(cfg['benchmark_manifest'])
    bar = bench.pure_bar()
    for name, sizes in [('A', [2, 2, 2, 2]), ('B', [8])]:
        variant = value['variants'][name]
        need([x['size'] for x in variant['batches']] == sizes and len(variant['records']) == 8 and
             [x['key'] for x in variant['records']] == [x['key'] for x in planned['requests']], 'actual batch/request mapping differs')
        need(all(x['finish_reason'] == 'stop' and 0 < x['completion_tokens'] < 1800 for x in variant['records']),
             'actual generation truncated or missing')
        for item, record in zip(planned['requests'], variant['records']):
            generation = checked(record['generation_ref'])
            raw_ref = record['raw_response_ref']
            need(Path(raw_ref['path']).resolve().is_relative_to(out.resolve() / 'generations') and
                 base.sha(raw_ref['path']) == raw_ref['sha256'] == generation['raw_response_sha256'], 'native benchmark raw response binding differs')
            raw = Path(raw_ref['path']).read_text()
            need(raw == record['raw_response'] and generation['model_call_performed'] is True and
                 generation['model_identity_sha256'] == bench.MODEL_SHA and generation['key'] == item['key'] and
                 generation['request_sha256'] == item['refs']['request.json']['sha256'] and
                 generation['binding_sha256'] == item['refs']['binding.json']['sha256'] and
                 generation['completion_tokens'] == record['completion_tokens'] and generation['finish_reason'] == 'stop',
                 'actual native generation receipt differs')
            request = checked(item['refs']['request.json'])
            response_value = bench.strict(raw)
            bench.schema_check(response_value, request['json_schema'])
            filtered = bar.filter_supported_observation(response_value, item['manifest'], {p: set(v) for p, v in item['forbidden'].items()})
            need(filtered == checked(record['filtered_ref']) == record['filtered_observation'], 'original support filter replay differs')
    need(len(list((out / 'generations').glob('*/*/*/generation.json'))) == 16, 'actual generation file count differs')
    guard = bench.strict((Path(cfg['execution_root']) / 'original_budget_guard.exit.json').read_bytes())
    need(guard['unchanged'] is True, 'original absent/present budget guard changed')
    return {str(out / 'result.json'): dict(summary=dict(stage='run', total_source_count=222939,
        selected_source_count=3, processed_source_count=3, plan_sha256=cfg['plan']['sha256']),
        selection=dict(selected_ids=list(bench.SOURCES)), model_request_count=16,
        semantic_approval=False, production_counts=0)}


def safe_proofs(refs):
    expected = {BAR / 'queues' / name / 'resources_safe.json': phase for name, phase in SAFE_PHASES.items()}
    need(len(refs) == len(expected) and {Path(r['path']).resolve() for r in refs} == set(expected),
         'exact N6 GPU2/3 restored resource proofs required')
    for reference in refs:
        path = Path(reference['path']).resolve()
        value = checked(reference)
        need(value.get('phase') == expected[path] and value.get('all_holders_safe') is True and
             value.get('locks_released') is True and value.get('leases_restored') is True,
             'N6 prior resources not safely restored')


def kernel_lock_matches(line, st):
    fields = line.split()
    if len(fields) < 6 or fields[1] != 'FLOCK':
        return False
    major, minor, inode = fields[5].split(':')
    return int(major, 16) == os.major(st.st_dev) and int(minor, 16) == os.minor(st.st_dev) and int(inode) == st.st_ino


def lock_snapshot(gpu):
    path = Path('/tmp/ms12k_gpu_%d.lock' % gpu)
    if not path.exists():
        return dict(exists=False, held=[])
    st = path.lstat()
    need(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid() and st.st_size == 0, 'unsafe capture FLOCK type/owner')
    held = [line for line in Path('/proc/locks').read_text().splitlines() if kernel_lock_matches(line, st)]
    return dict(exists=True, device=st.st_dev, inode=st.st_ino, mode=st.st_mode, uid=st.st_uid, size=st.st_size, held=held)


def lease_snapshot(gpu):
    st = base.lease_snapshot(Path('/tmp/od_reserve_%d' % gpu))
    if st is None:
        return dict(exists=False)
    return dict(exists=True, device=st.st_dev, inode=st.st_ino, uid=st.st_uid, mode=st.st_mode,
                size=st.st_size, mtime_ns=st.st_mtime_ns)


def validate_capture(capture):
    need(capture['schema'] == 'visual_bench_native_capture_v1' and capture['node'] == NODE and
         capture['gpus'] == GPUS and capture['mode'] == 'consumer' and capture.get('signals_sent') is False,
         'fresh explicit N6 native capture required')
    need(0 <= time.time() - capture['captured_unix'] <= CAPTURE_TTL and HOST in capture['physical_addresses'],
         'capture TTL or physical N6 host differs')
    holders = capture['holders']
    need(len(holders) == 2 and [h['gpu'] for h in holders] == GPUS and
         len({h['pid'] for h in holders}) == len({h['uuid'] for h in holders}) == 2,
         'exact unique N6 pair holder inventory required')
    need(capture['headroom']['headroom'] >= base.SAFETY + base.LOAD, 'capture native admission headroom insufficient')
    for gpu in GPUS:
        need(capture['lock_inventory'][str(gpu)]['held'] == [] and str(gpu) in capture['lease_inventory'],
             'capture FLOCK occupied or lease snapshot missing')


def verify_capture_current(capture):
    validate_capture(capture)
    need(HOST in subprocess.check_output(['hostname', '-I'], text=True).split(), 'current physical N6 host differs')
    running = base.apps()
    for gpu, holder in zip(GPUS, capture['holders']):
        base.check_holder(holder, base.identity(holder['pid']), gpu, base.uuid_for(gpu), running)
        need(lock_snapshot(gpu) == capture['lock_inventory'][str(gpu)] and
             lease_snapshot(gpu) == capture['lease_inventory'][str(gpu)], 'current exact FLOCK/lease differs from fresh capture')
    need(base.headroom()['headroom'] >= base.SAFETY + base.LOAD, 'current native admission headroom insufficient')


def capture_current(out, proof_refs):
    safe_proofs(proof_refs)
    addresses = subprocess.check_output(['hostname', '-I'], text=True).split()
    need(HOST in addresses, 'capture executing outside actual N6')
    running, holders = base.apps(), []
    for gpu in GPUS:
        uuid = base.uuid_for(gpu)
        apps = [a for a in running if a['uuid'] == uuid]
        need(len(apps) == 1, 'capture requires one sole authorized holder per GPU')
        holder = base.identity(apps[0]['pid'])
        holder.update(gpu=gpu, uuid=uuid)
        base.check_holder(holder, base.identity(holder['pid']), gpu, uuid, running)
        holders.append(holder)
    capture = dict(schema='visual_bench_native_capture_v1', node=NODE, gpus=list(GPUS), mode='consumer',
        captured_unix=time.time(), physical_addresses=addresses, holders=holders, headroom=base.headroom(),
        lock_inventory={str(g): lock_snapshot(g) for g in GPUS}, lease_inventory={str(g): lease_snapshot(g) for g in GPUS},
        resource_safe_refs=proof_refs, signals_sent=False)
    verify_capture_current(capture)
    bench.exclusive(out, bench.encoded(capture))
    return ref(out)


def validate(cfg):
    c.validate(cfg)
    need(cfg['node'] == NODE and cfg['gpus'] == GPUS and cfg['targetcount'] == 3,
         'benchmark permits only explicit N6 GPU2/3 three-source assignment')
    need(cfg['resource_manager_ref'] == ref(SELF) and cfg['source_sha256'].get(str(RT / 'resource_manager.py')) == BASE_SHA,
         'wrapper or original native lifecycle SHA binding differs')
    validate_capture(checked(cfg['holder_capture']))
    safe_proofs(cfg['resource_safe_refs'])
    out = bench.isolated_out(cfg['out'])
    need(Path(cfg['benchmark_manifest']['path']).resolve() == out / 'manifest.json' and
         not (out / 'started.json').exists() and not (Path(cfg['execution_root']) / 'driver.started.json').exists(),
         'benchmark already started or output differs')
    need(cfg['env']['CUDA_VISIBLE_DEVICES'] == CVD and '--bench-driver' in cfg['argv'], 'native benchmark driver required')
    fixed_selected(cfg)
    verify_zero_dispatch(cfg)


def verify_zero_dispatch(cfg):
    prior = checked(cfg['prior_zero_dispatch_ref'])
    need(cfg['prior_zero_dispatch_ref']['sha256'] == '633b1ee462a9263acf0b763fe0a7b8610621af58b596cf779b1c38dcd89af670'
         and prior['node'] == NODE and prior['gpus'] == GPUS and prior['out'] == cfg['out']
         and prior['benchmark_manifest'] == cfg['benchmark_manifest'], 'exact zero-dispatch predecessor differs')
    queue, out = Path(prior['queue']), Path(prior['out'])
    need(cfg['resource_safe_refs'] == [ref(queue / 'resources_safe.json')], 'exact predecessor safe reference required')
    safe_proofs(cfg['resource_safe_refs'])
    admission = bench.strict((queue / 'admission.json').read_bytes())
    need(admission['config_sha256'] == cfg['prior_zero_dispatch_ref']['sha256']
         and admission['node'] == NODE and admission['gpus'] == GPUS
         and admission['manager_sha256'] == BASE_SHA, 'predecessor native admission differs')
    need((queue / 'failure.json').exists() and "KeyError: 'max_pixels'" in (queue / 'worker.runtime.log').read_text(),
         'only diagnosed configuration-read failure can recover')
    need(all(not (out / name).exists() for name in ('started.json', 'calls.jsonl', 'result.json', 'generations')),
         'predecessor dispatched or benchmark already started; never reset/resubmit')
    load = bench.strict((out / 'native_model_load.receipt.json').read_bytes())
    need(load['model_load_completed'] is True and load['model_identity_sha256'] == bench.MODEL_SHA,
         'predecessor actual loaded model identity differs')
    need(bench.strict((out / 'original_budget_guard.exit.json').read_bytes())['unchanged'] is True,
         'predecessor original budget guard missing')
    need(Path(cfg['execution_root']) == out / 'zero_dispatch_recovery_v2', 'fresh execution directory differs')


def build_config(source_cfg, source_sha, capture_ref, manifest_path, config_out, queue, phase):
    source = checked(dict(path=str(Path(source_cfg).resolve()), sha256=source_sha))
    need(source_sha == SOURCE_CFG_SHA and Path(source_cfg).resolve() == SOURCE_CFG, 'exact original N8 consumer config required')
    capture = checked(capture_ref)
    manifest = bench.strict(Path(manifest_path).read_bytes())
    validate_capture(capture)
    safe_proofs(capture['resource_safe_refs'])
    cfg = copy.deepcopy(source)
    for key in ('handoff', 'preparation_ref', 'original_consumer_config_ref', 'fresh_capture'):
        cfg.pop(key, None)
    cfg['env']['CUDA_VISIBLE_DEVICES'] = CVD
    cfg.update(node=NODE, gpus=list(GPUS), phase=phase, targetcount=3, input_count=222939, holders=capture['holders'],
        resource_safe_refs=capture['resource_safe_refs'],
        queue=str(Path(queue).resolve()), out=manifest['output'], cwd=str(RT), reuse_existing_results=True,
        producer=ref(SELF), benchmark_manifest=ref(manifest_path),
        original_consumer_config=dict(path=str(Path(source_cfg).resolve()), sha256=source_sha),
        holder_capture=capture_ref, resource_manager_ref=ref(SELF), fresh_capture=capture_ref,
        execution_root=str(Path(manifest['output']) / 'zero_dispatch_recovery_v2'),
        prior_zero_dispatch_ref=ref(RT / 'visual_batch_fixed3_n6g23_main_v1_launch.json'),
        original_consumer_argv=source['argv'], original_consumer_env=source['env'],
        original_consumer_out=source['out'],
        comparison_caveats=dict(order=['A', 'B'], prefix_caching_enabled=True,
            image_processor_and_grammar_cache_may_favor_B=True, pure_batch_effect_not_isolated=True))
    cfg['argv'] = ['/opt/venv_vllm/bin/python', '-B', str(SELF), 'consumer', '--bench-driver',
        '--config-file', str(Path(config_out).resolve()), '--bench-manifest', str(Path(manifest_path).resolve()),
        '--manifest', cfg['manifest_summary']['path'], '--stage', 'run']
    paths = [SELF, RT / 'visual_batch_fixed3_benchmark_v1.py', RT / 'resource_manager.py',
        RT / 'curated_resource_manager.py', Path(source_cfg), Path(manifest_path), Path(capture_ref['path']),
        Path(cfg['plan']['path']), Path(cfg['manifest_summary']['path']), Path(cfg['modelref']['path']),
        RT / 'source_curated_consumer_v1/ms_bar_annotation.py', RT / 'source_curated_consumer_v1/ms_caption_rewrite.py']
    paths.extend(Path(r['path']) for r in capture['resource_safe_refs'])
    prior = checked(cfg['prior_zero_dispatch_ref'])
    paths.extend([Path(cfg['prior_zero_dispatch_ref']['path']), Path(prior['queue']) / 'admission.json', Path(prior['queue']) / 'failure.json',
        Path(prior['queue']) / 'worker.runtime.log', Path(prior['out']) / 'driver.started.json',
        Path(prior['out']) / 'native_model_load.receipt.json', Path(prior['out']) / 'original_budget_guard.exit.json'])
    cfg['source_sha256'] = {str(p.resolve()): base.sha(p) for p in paths}
    cfg['original_retry_budget_paths'] = [str(BAR / 'curated222939_repair_v1/attempt-budget/train' / (sid + '.json'))
                                           for sid in bench.SOURCES]
    validate(cfg)
    need(not Path(queue).exists(), 'fresh independent benchmark queue required')
    bench.exclusive(config_out, bench.encoded(cfg))
    return cfg



def parent_gate(cfg, before_load=False):
    queue = Path(cfg['queue'])
    admission = bench.strict((queue / 'admission.json').read_bytes())
    parent_pid = os.getppid()
    parent = base.identity(parent_pid)
    current = base.identity(os.getpid())
    expected = admission['manager_identity']
    need(all(parent[k] == expected[k] for k in base.IDENTITY_KEYS) and parent_pid == expected['pid'] and
         parent['uid'] == current['uid'] == os.getuid() and parent['cgroup'] == current['cgroup'], 'direct native parent identity/UID/cgroup differs')
    need(parent['argv'] == ['/opt/venv_vllm/bin/python', '-B', str(SELF), '--config', cfg['_config_path'],
                           '--config-sha', cfg['_config_sha'], '--manager-sha', BASE_SHA], 'direct parent native manager argv differs')
    need(admission['manager_sha256'] == BASE_SHA and admission['source_sha256'].get(str(SELF)) == cfg['resource_manager_ref']['sha256'] == base.sha(SELF) and
         admission['config_sha256'] == cfg['_config_sha'] and admission['out'] == cfg['out'] and
         admission['gpus'] == GPUS and admission['node'] == NODE and
         current['argv'] == cfg['argv'] and current['CVD'] == ['CUDA_VISIBLE_DEVICES=' + CVD], 'native admitted driver argv/CVD/config differs')
    deadline = time.monotonic() + 15
    while not (queue / 'live.json').exists() and time.monotonic() < deadline:
        time.sleep(.05)
    live = bench.strict((queue / 'live.json').read_bytes())
    need(live['child_identity']['pid'] == current['pid'] and
         all(live['child_identity'][k] == current[k] for k in base.IDENTITY_KEYS), 'native direct child live receipt differs')
    parent_fds = []
    for fd in (Path('/proc') / str(parent_pid) / 'fd').iterdir():
        try:
            parent_fds.append((os.readlink(fd), fd.stat()))
        except FileNotFoundError:
            pass
    locks = Path('/proc/locks').read_text().splitlines()
    uuids = []
    for gpu, holder in zip(cfg['gpus'], admission['holders']):
        uuid = base.uuid_for(gpu)
        need(uuid == holder['uuid'], 'native GPU UUID differs')
        uuids.append(uuid)
        lock_path = '/tmp/ms12k_gpu_%d.lock' % gpu
        lock = Path(lock_path).lstat()
        need(stat.S_ISREG(lock.st_mode) and lock.st_uid == os.getuid() and lock.st_size == 0, 'unsafe native FLOCK inode/type/owner')
        need(any(name == lock_path and st.st_ino == lock.st_ino and st.st_dev == lock.st_dev for name, st in parent_fds) and
             any(kernel_lock_matches(line, lock) and line.split()[4] == str(parent_pid) for line in locks), 'native parent FLOCK fd/owner missing')
        lease_path = '/tmp/od_reserve_%d' % gpu
        lease = Path(lease_path).lstat()
        need(stat.S_ISREG(lease.st_mode) and lease.st_uid == os.getuid() and lease.st_size == 0 and
             0 <= time.time() - lease.st_mtime < 30 and
             any(name == lease_path and st.st_ino == lease.st_ino and st.st_dev == lease.st_dev for name, st in parent_fds),
             'native parent lease inode/owner/live heartbeat missing')
    for app in base.apps():
        if app['uuid'] not in uuids:
            continue
        need(not before_load, 'GPU occupied before admitted model load')
        pid = app['pid']
        for _ in range(20):
            if pid == current['pid']:
                break
            ancestor = base.identity(pid)
            need(ancestor['uid'] == os.getuid() and ancestor['cgroup'] == current['cgroup'], 'unknown GPU process identity')
            pid = ancestor['ppid']
        need(pid == current['pid'], 'GPU app outside sole admitted driver tree; never signal unknown job')
    return dict(admitted=True, caller_owned=True, parent_identity=parent, child_identity=current,
                gpu_uuids=uuids, admission_ref=ref(queue / 'admission.json'))


def actual_state(llm, cfg, load_ref):
    load = checked(load_ref)
    vc = llm.llm_engine.vllm_config
    model, scheduler = vc.model_config, vc.scheduler_config
    mm = model.multimodal_config
    kwargs = mm.mm_processor_kwargs or {}
    need('max_pixels' in kwargs, 'actual multimodal processor config is missing max_pixels')
    structured = vc.structured_outputs_config
    need(str(Path(model.model).resolve()) == '/opt/models/Qwen3-VL-32B-Instruct' and
         vc.cache_config.gpu_memory_utilization == 0.88 and vc.cache_config.enable_prefix_caching is True and
         load['model_identity_sha256'] == cfg['modelref']['model_identity_sha256'],
         'actual model root/cache policy/load receipt differs')
    dtype = str(model.dtype).removeprefix('torch.')
    return dict(model_identity_sha256=llm._ms_bar_model_sha256,
        source_reference_sha256=load['source_reference_sha256'], dtype=dtype, quantization=model.quantization,
        tensor_parallel_size=vc.parallel_config.tensor_parallel_size, max_num_seqs=scheduler.max_num_seqs,
        max_model_len=model.max_model_len, max_pixels=kwargs['max_pixels'],
        max_images_per_prompt=mm.get_limit_per_prompt('image'), structured_outputs_backend=structured.backend,
        disable_any_whitespace=structured.disable_any_whitespace, seed=model.seed,
        model_load_completed=load['model_load_completed'])


def memory_snapshot(llm):
    text = subprocess.check_output(['nvidia-smi', '-i', CVD, '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
                                  '--format=csv,noheader,nounits'], text=True)
    return dict(utc=time.strftime('%FT%TZ', time.gmtime()), gpus=[dict(gpu=int(v[0]), uuid=v[1], used_MiB=int(v[2]),
        total_MiB=int(v[3]), utilization_pct=int(v[4])) for line in text.splitlines()
        for v in [[x.strip() for x in line.split(',')]]], peak_source='native manager resource_samples/worker_exit.gpu_peak_MiB')


def driver(config_path, manifest_path):
    cfg = bench.strict(Path(config_path).read_bytes())
    cfg.update(_config_path=str(Path(config_path).resolve()), _config_sha=base.sha(config_path))
    need(Path(manifest_path).resolve() == Path(cfg['benchmark_manifest']['path']).resolve(), 'driver manifest/config differs')
    checked(cfg['benchmark_manifest'])
    validate(cfg)
    out = Path(cfg['out'])
    execution = Path(cfg['execution_root'])
    owned = parent_gate(cfg, before_load=True)
    execution.mkdir(exist_ok=False)
    bench.exclusive(execution / 'driver.started.json', bench.encoded(dict(parent=owned, **bench.FLAGS)))
    guards = bench.Boundary()
    for path in cfg['original_retry_budget_paths']:
        guards.optional(path)
    bench.exclusive(execution / 'original_budget_guard.entry.json', bench.encoded(dict(refs=list(guards.refs.values()), **bench.FLAGS)))
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model='/opt/models/Qwen3-VL-32B-Instruct', tensor_parallel_size=2, max_model_len=16384,
            max_num_seqs=8, gpu_memory_utilization=0.88, dtype='bfloat16', quantization=None,
            limit_mm_per_prompt={'image': 16, 'video': 0}, mm_processor_kwargs={'max_pixels': 200704},
            structured_outputs_config={'backend': 'xgrammar', 'disable_any_whitespace': True},
            enable_prefix_caching=True, trust_remote_code=False, seed=0)
        llm._ms_bar_model_sha256 = cfg['modelref']['model_identity_sha256']
        load_path = execution / 'native_model_load.receipt.json'
        bench.exclusive(load_path, bench.encoded(dict(model_load_completed=True, model_identity_sha256=llm._ms_bar_model_sha256,
            source_reference_sha256=cfg['modelref']['sha256'], driver_identity=base.identity(os.getpid()), parent=parent_gate(cfg), **bench.FLAGS)))
        load_ref = ref(load_path)
        runtime = bench.CallerOwnedRuntime(llm, SamplingParams, lambda model: actual_state(model, cfg, load_ref), memory_snapshot)
        return bench.run_benchmark(manifest_path, runtime, admission=lambda manifest, runtime: parent_gate(cfg))
    finally:
        bench.verify_boundary(list(guards.refs.values()))
        bench.exclusive(execution / 'original_budget_guard.exit.json', bench.encoded(dict(refs=list(guards.refs.values()), unchanged=True, **bench.FLAGS)))


def manager():
    need(Path(base.__file__).resolve() == RT / 'resource_manager.py' and base.sha(base.__file__) == BASE_SHA,
         'original native base identity must never be rebound')
    def manager_validate(cfg):
        validate(cfg)
        verify_capture_current(checked(cfg['holder_capture']))
    base.validate_pair = manager_validate
    base.selected_sources = fixed_selected
    base.stage_summaries = summaries
    original_check = base.check_sources
    boundary_stats = {}
    def cached_check_sources(config, config_path, config_sha, manager_sha):
        need(manager_sha == BASE_SHA, 'native --manager-sha must bind original resource_manager.py')
        paths = set(config['source_sha256']) | {config_path, str(SELF), str(RT / 'resource_manager.py'), config['producer']['path'], config['plan']['path']}
        if not boundary_stats or (Path(config['execution_root']) / 'driver.started.json').exists():
            original_check(config, config_path, config_sha, manager_sha)
        for name in paths:
            st = Path(name).stat()
            snapshot = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            if name in boundary_stats:
                need(snapshot == boundary_stats[name], 'cached native stage source changed: ' + name)
            else:
                boundary_stats[name] = snapshot
    base.check_sources = cached_check_sources
    base.main()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'capture':
        parser = argparse.ArgumentParser()
        parser.add_argument('stage', choices=['capture'])
        parser.add_argument('--out', type=Path, required=True)
        parser.add_argument('--resource-safe', type=Path, action='append', required=True)
        args = parser.parse_args()
        print(json.dumps(capture_current(args.out, [ref(path) for path in args.resource_safe])))
    elif '--bench-driver' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('mode', choices=['consumer'])
        parser.add_argument('--bench-driver', action='store_true')
        parser.add_argument('--config-file', required=True)
        parser.add_argument('--bench-manifest', required=True)
        parser.add_argument('--manifest', required=True)
        parser.add_argument('--stage', choices=['run'], required=True)
        args = parser.parse_args()
        driver(args.config_file, args.bench_manifest)
    elif len(sys.argv) > 1 and sys.argv[1] == 'build':
        parser = argparse.ArgumentParser()
        parser.add_argument('stage', choices=['build'])
        parser.add_argument('--source-config', type=Path, default=SOURCE_CFG)
        parser.add_argument('--source-sha', default=SOURCE_CFG_SHA)
        parser.add_argument('--capture', type=Path, required=True)
        parser.add_argument('--capture-sha', required=True)
        parser.add_argument('--bench-manifest', type=Path, default=bench.DEFAULT_OUT / 'manifest.json')
        parser.add_argument('--config-out', type=Path, required=True)
        parser.add_argument('--queue', type=Path, required=True)
        parser.add_argument('--phase', required=True)
        args = parser.parse_args()
        capture_ref = dict(path=str(args.capture.resolve()), sha256=args.capture_sha)
        build_config(args.source_config, args.source_sha, capture_ref, args.bench_manifest, args.config_out, args.queue, args.phase)
        print(json.dumps(dict(config=str(args.config_out), config_sha=base.sha(args.config_out), manager_sha=BASE_SHA,
                              resource_wrapper_sha=base.sha(SELF))))
    else:
        manager()


if __name__ == '__main__':
    main()
