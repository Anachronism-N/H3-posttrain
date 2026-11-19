"""Strict N6/GPU3 primary annotation driver using the unchanged native lifecycle."""
import argparse
from contextlib import contextmanager
import copy
import importlib.util
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
BASE_SHA = '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
MANIFEST_SHA = '4c9fff89871ad9e2783ba679fbf7b4581aa8f28695cf6725809915a9ec2786f1'
PREPARED_MANIFEST = BAR / 'caption_8b_primary_pilot_cpu_v2/manifest.json'
STAGE_SHA = '5607668eb5c476308f20ca20b2853c97bc846f5d43182df2a66729b3c9c4385f'
NODE, GPUS, CVD, HOST = 'N6', [3], '3', '28.216.19.225'
CAPTURE_TTL = 1200
PYTHON = '/opt/venv_vllm/bin/python'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


p = module('primary_pilot_native', RT / 'caption_8b_primary_pilot_v1.py')
base = module('primary_original_resource', RT / 'resource_manager.py')
need = p.need


def original_base():
    need(Path(base.__file__).resolve() == RT / 'resource_manager.py' and
         base.sha(base.__file__) == BASE_SHA, 'original native base identity must never be rebound')


def ref(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=base.sha(path))


def checked(reference):
    return p.Reader().json(reference['path'], reference['sha256'])[0]


def predecessor(admission_ref, safe_ref):
    queue = Path(safe_ref['path']).resolve().parent
    need(queue.is_relative_to(BAR / 'queues') and Path(safe_ref['path']).name == 'resources_safe.json' and
         Path(admission_ref['path']).resolve() == queue / 'admission.json', 'same native prior queue admission/safe required')
    admission, safe = checked(admission_ref), checked(safe_ref)
    need(admission['node'] == NODE and admission['gpus'] == [2, 3] and
         admission['model_identity_sha256'] == p.SOURCE_MODEL_SHA and admission['manager_sha256'] == BASE_SHA,
         'prior admission must be actual N6 GPU2/3 original 32B native stage')
    need(all(safe.get(k) is True for k in ('all_holders_safe', 'locks_released', 'leases_restored',
         'managed_parent_wait', 'unknown_jobs_not_signaled')) and safe['phase'] == admission['phase'],
         'prior native stage is not safely restored')
    need(not (queue / 'failure.json').exists(), 'prior native stage failure requires coordinator review')
    exited = p.strict((queue / 'worker_exit.json').read_bytes())
    need(exited['exit_code'] == 0 and exited['phase'] == admission['phase'] and
         exited['source_unchanged'] is True and bool(exited['summaries']) and
         exited['producer'] == admission['producer'] and exited['plan'] == admission['plan'],
         'prior 32B worker not completed with native summary')
    restored = p.strict((queue / 'holder_restored_gpu3.json').read_bytes())
    need([h['gpu'] for h in admission['holders']] == [2, 3] and
         len({h['uuid'] for h in admission['holders']}) == 2 and
         len({h['identity']['pid'] for h in admission['holders']}) == 2, 'prior complete native pair inventory differs')
    original = next(h for h in admission['holders'] if h['gpu'] == 3)
    need(restored['gpu'] == 3 and restored['uuid'] == original['uuid'] and
         all(restored['original_identity'][k] == original['identity'][k] for k in base.IDENTITY_KEYS) and
         restored['managed_parent_wait'] is True and restored['restore_argv'] == base.RESTORE_ARGV,
         'prior actual native G3 restoration identity differs')
    need(Path(safe_ref['path']).stat().st_mtime_ns >= (queue / 'worker_exit.json').stat().st_mtime_ns >=
         Path(admission_ref['path']).stat().st_mtime_ns, 'native stage receipt order differs')
    return admission, safe


def validate_capture(capture, admission_ref, safe_ref):
    predecessor(admission_ref, safe_ref)
    need(capture.get('schema') in ('caption_8b_primary_native_capture_v1', 'visual_bench_native_capture_v1') and
         capture['node'] == NODE and capture['gpus'] in ([3], [2, 3]) and
         capture.get('signals_sent') is False and capture['mode'] == 'consumer', 'actual N6 G3 holder capture required')
    need(HOST in capture['physical_addresses'] and 0 <= time.time() - capture['captured_unix'] <= CAPTURE_TTL,
         'capture physical node/TTL differs')
    need(capture['captured_unix'] * 1e9 >= Path(safe_ref['path']).stat().st_mtime_ns,
         'capture must follow prior 32B resources_safe, never reuse its prelaunch capture')
    holders = capture['holders']
    need([h['gpu'] for h in holders] == capture['gpus'] and
         len({h['pid'] for h in holders}) == len({h['uuid'] for h in holders}) == len(holders), 'capture holder inventory differs')
    holder = next(h for h in holders if h['gpu'] == 3)
    restored = p.strict((Path(safe_ref['path']).parent / 'holder_restored_gpu3.json').read_bytes())
    need(all(holder[k] == restored['holder_identity'][k] for k in base.IDENTITY_KEYS) and
         holder['uuid'] == restored['uuid'], 'capture is not the prior actual native restored G3 holder')
    need(holder['uid'] == os.getuid() and holder['CVD'] == ['CUDA_VISIBLE_DEVICES=3'] and
         holder['argv'] == base.RESTORE_ARGV and
         not (holder['sig_ign'] | holder['sig_blk']) & (1 << (15 - 1)), 'capture G3 is not exact authorized native holder')
    need(capture['lock_inventory']['3']['held'] == [] and '3' in capture['lease_inventory'] and
         capture['headroom']['headroom'] >= base.SAFETY + base.LOAD, 'capture lock/lease/native headroom differs')
    return holder


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
    need(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid() and st.st_size == 0, 'unsafe capture FLOCK')
    return dict(exists=True, device=st.st_dev, inode=st.st_ino, mode=st.st_mode, uid=st.st_uid, size=st.st_size,
                held=[line for line in Path('/proc/locks').read_text().splitlines() if kernel_lock_matches(line, st)])


def lease_snapshot(gpu):
    st = base.lease_snapshot(Path('/tmp/od_reserve_%d' % gpu))
    return dict(exists=False) if st is None else dict(exists=True, device=st.st_dev, inode=st.st_ino,
        uid=st.st_uid, mode=st.st_mode, size=st.st_size, mtime_ns=st.st_mtime_ns)


def verify_capture_current(cfg):
    holder = validate_capture(checked(cfg['holder_capture']), cfg['prior_admission'], cfg['prior_resources_safe'])
    need(HOST in subprocess.check_output(['hostname', '-I'], text=True).split(), 'current physical node is not N6')
    base.check_holder(holder, base.identity(holder['pid']), 3, base.uuid_for(3), base.apps())
    capture = checked(cfg['holder_capture'])
    need(lock_snapshot(3) == capture['lock_inventory']['3'] and lease_snapshot(3) == capture['lease_inventory']['3'],
         'current exact FLOCK/lease differs from capture')
    need(base.headroom()['headroom'] >= base.SAFETY + base.LOAD, 'current native admission headroom insufficient')


def capture_current(output, admission_ref, safe_ref):
    predecessor(admission_ref, safe_ref)
    addresses = subprocess.check_output(['hostname', '-I'], text=True).split()
    need(HOST in addresses, 'capture executing outside N6')
    uuid, running = base.uuid_for(3), base.apps()
    local = [a for a in running if a['uuid'] == uuid]
    need(len(local) == 1, 'one sole native G3 holder required, never capture occupied worker')
    holder = dict(base.identity(local[0]['pid']), gpu=3, uuid=uuid)
    base.check_holder(holder, base.identity(holder['pid']), 3, uuid, running)
    value = dict(schema='caption_8b_primary_native_capture_v1', node=NODE, gpus=GPUS, mode='consumer',
        captured_unix=time.time(), physical_addresses=addresses, holders=[holder], headroom=base.headroom(),
        lock_inventory={'3': lock_snapshot(3)}, lease_inventory={'3': lease_snapshot(3)}, signals_sent=False,
        prior_admission=admission_ref, prior_resources_safe=safe_ref)
    validate_capture(value, admission_ref, safe_ref)
    return p.write_once(output, value)


CPU_FAKE_RECEIPTS = {
    'cpu_fake_batch8': '752e641c58c84815f56387f6f4742df6c263d1b4988a959dc5ab9cca1cd69e98',
    'cpu_fake_failed_structure': 'b77b701e11c63c812410b4cbd59a547029cc28dd7fef64ec1b1ddded65969fcb',
    'cpu_fake_conflicting_count': 'bae0013d0461337f45c02a737ef37e11982e57de9631ead978e09ce36d1d4264',
}
CPU_FAKE_CLAIM_SHA = 'bf12b66d03b9bdb4cb895c098e28509042ed9449646b707d81cef39425c6433f'


def no_claims(manifest_path):
    need(Path(manifest_path).resolve() == PREPARED_MANIFEST, 'only unique original prepared manifest/root permitted')
    root = PREPARED_MANIFEST.parent
    need(not (root / 'native_primary.claim.json').exists(),
         'prepared root already reserved/claimed; failure is terminal, no redispatch')
    for path in root.rglob('run.claim.json'):
        name = path.parent.name
        need(name in CPU_FAKE_RECEIPTS and path == root / name / 'run.claim.json' and
             base.sha(path) == CPU_FAKE_CLAIM_SHA,
             'prepared root already reserved/claimed; no redispatch of real or unknown claims')
        receipt_path = path.parent / 'benchmark.receipt.json'
        need(base.sha(receipt_path) == CPU_FAKE_RECEIPTS[name], 'pinned CPU-only receipt changed')
        claim, receipt = p.strict(path.read_bytes()), p.strict(receipt_path.read_bytes())
        need(claim['admission'].get('test_only') is True and receipt['measured_real_model'] is False and
             claim['manifest_ref'] == dict(path=str(PREPARED_MANIFEST), sha256=MANIFEST_SHA),
             'only exact completed CPU Fake contracts are separate from native GPU budget')


def prepared_scope(reference):
    need(Path(reference['path']).resolve() == PREPARED_MANIFEST, 'only unique original prepared manifest/root permitted')
    need(reference['sha256'] == MANIFEST_SHA, 'exact original manifestCPU_v2 SHA required')
    prepared = p.load_prepared(reference['path'], reference['sha256'])
    m = prepared.manifest
    need(m['case_ids'] == list(p.CASES) and len(m['requests']) == 8 and m['settings'] == p.SETTINGS and
         m['model_identity']['sha256'] == p.MODEL_SHA and m['request_delta'] == ['model'] and
         m['budgets'] == dict(new_8B_window_calls=8, independent_32B_reviews=0, source_style_reruns=0,
                             production_attempt_calls=0, production_repair_budget=0), 'fixed original pilot settings/scope changed')
    for record in m['requests']:
        request = record['request']
        need(request['model'] == p.MODEL and request['sampling'] == dict(temperature=0.0, seed=0, max_tokens=1800) and
             p.digest({k: v for k, v in request.items() if k != 'model'}) == record['source_request_without_model_sha256'],
             'original request schema/token budget changed')
    return prepared


def check_model(cfg):
    stage = checked(cfg['modelref'])
    model = stage['model_identity']
    need(cfg['modelref']['sha256'] == STAGE_SHA and
         Path(cfg['modelref']['path']).resolve() == RT / 'reviewer8b_stage_N6_extension_v2.json' and
         stage['node'] == NODE and stage.get('complete') is True and stage.get('full_copy_SHA_verified') is True and
         stage.get('destination_disk_SHA_rehashed') is True and stage.get('source_and_destination_SHA_verified') is True,
         'complete real N6 extension-v2 staging required')
    need(model['root'] == p.MODEL and model['sha256'] == cfg['modelref']['model_identity_sha256'] == p.MODEL_SHA and
         p.identity_digest(model) == p.MODEL_SHA, 'raw 8B model identity differs')
    origin = checked(stage['origin_ref'])
    need(origin.get('complete') is True and origin.get('full_copy_SHA_verified') is True and
         origin['model'] == 'Qwen3-VL-8B-Instruct' and len(origin['files']) == 15, 'bound origin staging incomplete')
    weights = {name: row['sha256'] for name, row in model['files'].items() if name.endswith('.safetensors')}
    origin_weights = {name: row['sha256'] for name, row in origin['files'].items() if name.endswith('.safetensors')}
    need(len(model['files']) == 15 and len(weights) == 4 and weights == origin_weights and
         p.checkpoint(model) == prepared_scope(cfg['pilot_manifest']).manifest['declared_checkpoint'],
         'full 15-file/four-weight origin/checkpoint identity differs')
    root = Path(p.MODEL)
    inventory = {str(f.relative_to(root)) for f in root.rglob('*') if f.is_file() and
                 not any(part.startswith('.') for part in f.relative_to(root).parts)}
    need(inventory == set(model['files']), 'local full model inventory differs')
    for name, row in model['files'].items():
        path, expected = root / name, row['stat']
        st = path.stat()
        need(str(path.resolve(strict=True)) == row['path'] and
             dict(size=st.st_size, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns,
                  device=st.st_dev, inode=st.st_ino) == expected, 'local staged model stat changed: ' + name)
        if not name.endswith('.safetensors'):
            need(base.sha(path) == row['sha256'], 'local model config/token schema changed: ' + name)
    return stage


def selected(cfg):
    prepared_scope(cfg['pilot_manifest'])
    return list(p.CASES), 3


def validate(cfg, unclaimed=True):
    original_base()
    need(cfg['node'] == NODE and cfg['gpus'] == GPUS and cfg['mode'] == 'consumer' and
         cfg['release_authorized'] is True and cfg['targetcount'] == cfg['input_count'] == 3 and
         cfg['env']['CUDA_VISIBLE_DEVICES'] == CVD, 'explicit N6 G3 primary driver assignment required')
    need(cfg['resource_manager_ref'] == ref(SELF) and cfg['source_sha256'].get(str(RT / 'resource_manager.py')) == BASE_SHA,
         'wrapper/base separate SHA binding differs')
    holder = validate_capture(checked(cfg['holder_capture']), cfg['prior_admission'], cfg['prior_resources_safe'])
    need(cfg['holders'] == [holder], 'config holder differs from actual capture')
    expected_argv = [PYTHON, '-B', str(SELF), 'worker', '--config-file', cfg['config_path']]
    need(cfg['argv'] == expected_argv and cfg['producer'] == cfg['resource_manager_ref'] and
         cfg['plan'] == cfg['pilot_manifest'] and cfg['cwd'] == str(RT), 'exact frozen direct worker argv required')
    need(isinstance(cfg['phase'], str) and cfg['phase'] and all(c.isalnum() or c in '_-' for c in cfg['phase']), 'invalid native phase')
    root, out, queue = Path(cfg['pilot_manifest']['path']).resolve().parent, Path(cfg['out']).resolve(), Path(cfg['queue']).resolve()
    need(root.is_relative_to(BAR) and not root.is_relative_to(BAR / 'curated222939_collaborative_v3') and
         out.parent == root and out != root and queue.is_relative_to(BAR / 'queues') and
         not out.is_relative_to(queue) and not queue.is_relative_to(out), 'new independent root/queue required')
    if unclaimed:
        no_claims(cfg['pilot_manifest']['path'])
    selected(cfg)


def build_config(args):
    manifest = dict(path=str(args.pilot_manifest.resolve(strict=True)), sha256=args.manifest_sha)
    capture = dict(path=str(args.capture.resolve(strict=True)), sha256=args.capture_sha)
    previous_admission = dict(path=str(args.prior_admission.resolve(strict=True)), sha256=args.prior_admission_sha)
    previous_safe = dict(path=str(args.prior_resources_safe.resolve(strict=True)), sha256=args.prior_resources_safe_sha)
    holder = validate_capture(checked(capture), previous_admission, previous_safe)
    prepared = prepared_scope(manifest)
    modelref = dict(path=str(args.modelref.resolve(strict=True)), sha256=args.modelref_sha, model_identity_sha256=p.MODEL_SHA)
    paths = [SELF, RT / 'resource_manager.py', RT / 'caption_8b_primary_pilot_v1.py', Path(modelref['path']),
             Path(manifest['path']), Path(capture['path']), Path(previous_admission['path']), Path(previous_safe['path'])]
    paths.extend(Path(r['path']) for r in prepared.manifest['original_filter_code_refs'])
    for name in CPU_FAKE_RECEIPTS:
        directory = Path(manifest['path']).parent / name
        if directory.exists():
            paths.extend([directory / 'run.claim.json', directory / 'benchmark.receipt.json'])
    stage = checked(modelref)
    paths.extend([Path(stage['origin_ref']['path']), Path(previous_safe['path']).parent / 'worker_exit.json',
                  Path(previous_safe['path']).parent / 'holder_restored_gpu3.json'])
    cfg = dict(node=NODE, gpus=GPUS, mode='consumer', release_authorized=True, phase=args.phase,
        targetcount=3, input_count=3, env=dict(CUDA_VISIBLE_DEVICES=CVD, TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1',
        VLLM_NO_USAGE_STATS='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', OMP_NUM_THREADS='2'),
        config_path=str(args.config_out.resolve()), holders=[holder], queue=str(args.queue.resolve()),
        out=str(args.out.resolve()), cwd=str(RT), reuse_existing_results=False,
        resource_manager_ref=ref(SELF), producer=ref(SELF), plan=manifest, pilot_manifest=manifest,
        modelref=modelref, holder_capture=capture, prior_admission=previous_admission, prior_resources_safe=previous_safe,
        source_sha256={str(path.resolve()): base.sha(path) for path in paths})
    cfg['argv'] = [PYTHON, '-B', str(SELF), 'worker', '--config-file', cfg['config_path']]
    need(not args.queue.exists() and not args.out.exists(), 'new native queue/output only; no resume')
    validate(cfg)
    return p.write_once(args.config_out, cfg)


def parent_gate(cfg, before_load=False):
    admission = p.strict((Path(cfg['queue']) / 'admission.json').read_bytes())
    parent_pid, current = os.getppid(), base.identity(os.getpid())
    parent, expected = base.identity(parent_pid), admission['manager_identity']
    need(all(parent[k] == expected[k] for k in base.IDENTITY_KEYS) and parent_pid == expected['pid'] and
         parent['uid'] == current['uid'] == os.getuid() and parent['cgroup'] == current['cgroup'], 'native direct parent identity differs')
    need(parent['argv'] == [PYTHON, '-B', str(SELF), '--config', cfg['config_path'], '--config-sha', cfg['_config_sha'],
         '--manager-sha', BASE_SHA] and current['argv'] == cfg['argv'] and current['CVD'] == ['CUDA_VISIBLE_DEVICES=3'],
         'native parent/worker argv or CVD differs')
    need(admission['node'] == NODE and admission['gpus'] == GPUS and admission['out'] == cfg['out'] and
         admission['manager_sha256'] == BASE_SHA and admission['config_sha256'] == cfg['_config_sha'] and
         admission['model_identity_sha256'] == p.MODEL_SHA and
         admission['source_sha256'] == cfg['source_sha256'] and admission['producer'] == cfg['producer'],
         'native admission model/config/wrapper binding differs')
    deadline = time.monotonic() + 15
    while not (Path(cfg['queue']) / 'live.json').exists() and time.monotonic() < deadline:
        time.sleep(.05)
    live = p.strict((Path(cfg['queue']) / 'live.json').read_bytes())
    need(all(live['child_identity'][k] == current[k] for k in base.IDENTITY_KEYS), 'actual direct child live identity differs')
    fds = []
    for fd in (Path('/proc') / str(parent_pid) / 'fd').iterdir():
        try:
            fds.append((os.readlink(fd), fd.stat()))
        except FileNotFoundError:
            pass
    locks = Path('/proc/locks').read_text().splitlines()
    for filename in ('/tmp/ms12k_gpu_3.lock', '/tmp/od_reserve_3'):
        st = Path(filename).lstat()
        need(stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid() and st.st_size == 0 and
             any(name == filename and f.st_dev == st.st_dev and f.st_ino == st.st_ino for name, f in fds),
             'native parent FLOCK/lease FD inode missing')
        if filename.endswith('.lock'):
            need(any(kernel_lock_matches(line, st) and line.split()[4] == str(parent_pid) for line in locks), 'parent kernel FLOCK ownership missing')
        else:
            need(0 <= time.time() - st.st_mtime < 30, 'native lease heartbeat stale')
    uuid = base.uuid_for(3)
    need(len(admission['holders']) == 1 and admission['holders'][0]['uuid'] == uuid and
         admission['holders'][0]['gpu'] == 3, 'admitted physical GPU UUID differs')
    for app in base.apps():
        if app['uuid'] != uuid:
            continue
        need(not before_load, 'GPU app exists before native admitted load')
        pid = app['pid']
        for _ in range(20):
            if pid == current['pid']:
                break
            ancestor = base.identity(pid)
            need(ancestor['uid'] == current['uid'] and ancestor['cgroup'] == current['cgroup'], 'unknown GPU app identity')
            pid = ancestor['ppid']
        need(pid == current['pid'], 'GPU app outside admitted worker tree; never signal unknown job')
    return dict(admitted=True, node=NODE, gpus=GPUS, tp=1, test_only=False,
                parent_identity=parent, child_identity=current, admission_ref=ref(Path(cfg['queue']) / 'admission.json'))


def actual_state(llm):
    vc = getattr(llm, 'vllm_config', None)
    if vc is None:
        vc = llm.llm_engine.vllm_config
    model, scheduler, parallel = vc.model_config, vc.scheduler_config, vc.parallel_config
    structured, cache = vc.structured_outputs_config, vc.cache_config
    kwargs = model.multimodal_config.mm_processor_kwargs or {}
    need(str(Path(model.model).resolve()) == p.MODEL and str(Path(model.tokenizer).resolve()) == p.MODEL and
         str(model.dtype).removeprefix('torch.') == 'bfloat16' and model.quantization is None and
         parallel.tensor_parallel_size == 1 and scheduler.max_num_seqs == 8 and model.max_model_len == 16384 and
         model.seed == 0 and model.trust_remote_code is False and kwargs == {'max_pixels': 200704} and
         model.multimodal_config.get_limit_per_prompt('image') == 16 and
         model.multimodal_config.get_limit_per_prompt('video') == 0 and structured.backend == 'xgrammar' and
         structured.disable_any_whitespace is True and cache.enable_prefix_caching is True and
         cache.gpu_memory_utilization == 0.88, 'actual vllm_config model/dtype/TP/capacity/token-schema settings differ')
    return dict(model=p.MODEL, tokenizer=model.tokenizer, model_identity_sha256=p.MODEL_SHA, dtype='bfloat16',
        tensor_parallel_size=1, max_num_seqs=8, max_model_len=16384, seed=0, max_pixels=200704,
        structured_outputs_backend='xgrammar', disable_any_whitespace=True, verified_actual_vllm_config=True)


def sampling(request, SamplingParams, StructuredOutputsParams):
    need(request['sampling'] == dict(temperature=0.0, seed=0, max_tokens=1800), 'original sampling budget differs')
    result = SamplingParams(temperature=0.0, seed=0, max_tokens=1800, n=1,
        structured_outputs=StructuredOutputsParams(json=copy.deepcopy(request['json_schema']), disable_any_whitespace=True))
    need(result.temperature == 0.0 and result.seed == 0 and result.max_tokens == 1800 and result.n == 1 and
         result.structured_outputs.json == request['json_schema'] and result.structured_outputs.disable_any_whitespace is True,
         'actual sampling token/schema settings differ')
    return result


class Runtime:
    def __init__(self, llm, gate):
        self.llm, self.gate, self.attempted_window_calls = llm, gate, 0
        actual_state(llm)
        self._ms_bar_model_sha256 = p.MODEL_SHA
        self._ms_8b_pilot_settings = copy.deepcopy(p.SETTINGS)

    def chat(self, messages, samples, **kwargs):
        self.gate()
        actual_state(self.llm)
        need(self.attempted_window_calls + len(messages) <= 8, 'eight new-call budget exhausted')
        self.attempted_window_calls += len(messages)
        return self.llm.chat(messages, samples, **kwargs)


def worker(config_path, factory=None):
    cfg = p.strict(Path(config_path).read_bytes())
    cfg['_config_sha'] = base.sha(config_path)
    need(str(Path(config_path).resolve()) == cfg['config_path'], 'worker config path differs')
    validate(cfg)
    prepared = prepared_scope(cfg['pilot_manifest'])
    owned = parent_gate(cfg, before_load=True)
    out = Path(cfg['out'])
    p.write_once(out / 'native_driver.started.json', dict(parent=owned, **p.FLAGS))
    p.write_once(Path(cfg['pilot_manifest']['path']).parent / 'native_primary.claim.json',
        dict(manifest_ref=cfg['pilot_manifest'], out=str(out), reserved_unique_window_calls=8, no_redispatch=True, **p.FLAGS))
    guard = []
    for sid in p.CASES:
        path = BAR / 'curated222939_repair_v1/attempt-budget/train' / (sid + '.json')
        guard.append(dict(path=str(path), reference=ref(path) if path.exists() else None))
    p.write_once(out / 'original_budget_guard.entry.json', guard)
    holder = {'runtime': None}
    try:
        @contextmanager
        def admitted(manifest):
            yield parent_gate(cfg, before_load=True)

        def load():
            check_model(cfg)
            if factory is None:
                from vllm import LLM, SamplingParams
                from vllm.sampling_params import StructuredOutputsParams
            else:
                LLM, SamplingParams, StructuredOutputsParams = factory
            llm = LLM(model=p.MODEL, tokenizer=p.MODEL, dtype='bfloat16', quantization=None,
                tensor_parallel_size=1, max_model_len=16384, max_num_seqs=8, gpu_memory_utilization=0.88,
                limit_mm_per_prompt={'image': 16, 'video': 0}, mm_processor_kwargs={'max_pixels': 200704},
                structured_outputs_config={'backend': 'xgrammar', 'disable_any_whitespace': True},
                enable_prefix_caching=True, trust_remote_code=False, seed=0)
            state = actual_state(llm)
            holder.update(runtime=Runtime(llm, lambda: parent_gate(cfg)), SamplingParams=SamplingParams,
                          StructuredOutputsParams=StructuredOutputsParams)
            p.write_once(out / 'native_model_load.receipt.json', dict(model_load_completed=True,
                model_identity_sha256=p.MODEL_SHA, source_reference_sha256=cfg['modelref']['sha256'], actual_runtime=state,
                parent=parent_gate(cfg), **p.FLAGS))
            return holder['runtime']

        receipt = p.benchmark(prepared, out / 'annotation', admission=admitted, runtime=load,
            sampling=lambda request: sampling(request, holder['SamplingParams'], holder['StructuredOutputsParams']), batch_size=8)
        p.write_once(out / 'native_driver.summary.json', dict(status='completed', manifest_ref=cfg['pilot_manifest'],
            benchmark_receipt_ref=receipt['receipt_ref'], attempted_window_calls=holder['runtime'].attempted_window_calls,
            reserved_unique_window_calls=8, independent_review_calls=0, **p.FLAGS))
        return receipt
    except BaseException as error:
        runtime = holder['runtime']
        p.write_once(out / 'native_driver.failure.json', dict(error=repr(error), manifest_ref=cfg['pilot_manifest'],
            attempted_window_calls=runtime.attempted_window_calls if runtime else 0,
            reserved_unique_window_calls=8, failure_is_terminal_no_redispatch=True, native_parent_restores_resources=True, **p.FLAGS))
        raise
    finally:
        for row in guard:
            path = Path(row['path'])
            need((ref(path) if path.exists() else None) == row['reference'], 'original source attempt-budget changed')
        p.write_once(out / 'original_budget_guard.exit.json', dict(unchanged=True, refs=guard, **p.FLAGS))


def summaries(cfg, out):
    summary = p.strict((out / 'native_driver.summary.json').read_bytes())
    receipt = checked(summary['benchmark_receipt_ref'])
    need(summary['status'] == 'completed' and summary['attempted_window_calls'] == 8 and
         receipt['new_window_calls'] == receipt['requests_count'] == 8 and receipt['independent_review_calls'] == 0 and
         receipt['measured_real_model'] is True and summary['manifest_ref'] == cfg['pilot_manifest'] and
         receipt['manifest_ref'] == cfg['pilot_manifest'] and all(receipt[k] is False for k in p.FLAGS),
         'native eight-call summary/binding differs')
    need(p.strict((out / 'original_budget_guard.exit.json').read_bytes())['unchanged'] is True,
         'original source attempt-budget guard missing')
    return {str(out / 'native_driver.summary.json'): dict(summary=dict(stage='run', total_source_count=3,
        selected_source_count=3, processed_source_count=3, plan_sha256=cfg['plan']['sha256']),
        selection=dict(selected_ids=list(p.CASES)), primary_8b_window_calls=8, semantic_approval=False)}


def manager():
    original_base()
    def admit(cfg):
        validate(cfg)
        verify_capture_current(cfg)
    original_check = base.check_sources
    cache = {}
    def sources(cfg, config_path, config_sha, manager_sha):
        need(manager_sha == BASE_SHA, 'native --manager-sha must bind original base, not wrapper')
        if not cache or (Path(cfg['out']) / 'native_driver.started.json').exists():
            original_check(cfg, config_path, config_sha, manager_sha)
        for path in set(cfg['source_sha256']) | {config_path}:
            st = Path(path).stat()
            value = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            need(path not in cache or cache[path] == value, 'cached frozen input changed: ' + path)
            cache[path] = value
    base.validate_pair = admit
    base.selected_sources = selected
    base.check_model = check_model
    base.stage_summaries = summaries
    base.check_sources = sources
    base.main()


def proof_args(parser):
    parser.add_argument('--prior-admission', type=Path, required=True)
    parser.add_argument('--prior-admission-sha', required=True)
    parser.add_argument('--prior-resources-safe', type=Path, required=True)
    parser.add_argument('--prior-resources-safe-sha', required=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('build', 'capture', 'worker'):
        parser = argparse.ArgumentParser()
        parser.add_argument('command', choices=['build', 'capture', 'worker'])
        if sys.argv[1] == 'worker':
            parser.add_argument('--config-file', type=Path, required=True)
            args = parser.parse_args()
            worker(args.config_file)
            return
        proof_args(parser)
        if sys.argv[1] == 'capture':
            parser.add_argument('--out', type=Path, required=True)
            args = parser.parse_args()
            result = capture_current(args.out, dict(path=str(args.prior_admission.resolve()), sha256=args.prior_admission_sha),
                dict(path=str(args.prior_resources_safe.resolve()), sha256=args.prior_resources_safe_sha))
        else:
            for name in ('pilot-manifest', 'capture', 'modelref', 'config-out', 'queue', 'out'):
                parser.add_argument('--' + name, type=Path, required=True)
            for name in ('manifest-sha', 'capture-sha', 'modelref-sha', 'phase'):
                parser.add_argument('--' + name, required=True)
            args = parser.parse_args()
            result = build_config(args)
        print(p.encoded(dict(reference=result, manager_sha=BASE_SHA, wrapper_sha=base.sha(SELF))).decode(), end='')
    else:
        manager()


if __name__ == '__main__':
    main()
