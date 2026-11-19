"""Caller-owned N6 G2/3 native lifecycle for the fixed full direct pilot."""
import argparse
import copy
import importlib.util
import os
from pathlib import Path
import sys
import time
import types

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
sys.path.insert(0, str(RT))
import caption_batch8_direct_full_pilot_v1 as pilot

spec = importlib.util.spec_from_file_location('full_pilot_native_resource_helpers', RT / 'bench_manager_driver_v2.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
base, BAR = helpers.base, RT.parent
BASE_SHA = '6bd8a6c51d2b8423f481daae13222afa0724248c9bac5d1b9661ef8aba7727c6'
PILOT_SHA = 'ae0077a5a11332c15e4cc56d74c35bd919b6bb705b513890cd804b872f7fc462'
MODEL = '/opt/models/Qwen3-VL-32B-Instruct'
MODEL_SHA = '26722d150f791da5ff1befaf753efd651a105b17849adb74ea07cfe0ba01779c'
PRIOR_8B_SHA = '220d9152229696caec4decebb89a160afc8ac9b6ba91ab1b9d790343a82d733a'
PYTHON = '/opt/venv_vllm/bin/python'
NODE, GPUS, CVD = 'N6', [2, 3], '2,3'
PRIOR = {g: ('caption_batch8_direct_full_n6_main_v2', 'caption_batch8_direct_full_n6_main_v2') for g in GPUS}
FAILED_CONFIG_SHA = '9fbdc42452c23cee8409395d4552232285230e3ebc1d2ff3af547eafcd537164'
need, ref, checked = pilot.need, helpers.ref, helpers.checked
# Clone only the existing native parent/FD/FLOCK/lease/CVD/UUID ownership helper.
parent_gate = types.FunctionType(helpers.parent_gate.__code__,
    dict(helpers.parent_gate.__globals__, SELF=SELF, base=base), 'parent_gate', helpers.parent_gate.__defaults__)


def frozen():
    need(Path(base.__file__).resolve() == RT / 'resource_manager.py' and base.sha(base.__file__) == BASE_SHA,
         'Original native base file identity must never be rebound')
    need(base.sha(pilot.__file__) == PILOT_SHA, 'Frozen full pilot changed')


def verify_preload_failure(queue, admission):
    previous = dict(path=str(RT / 'caption_batch8_direct_full_n6_launch_main_v2.json'),
        sha256='9fbdc42452c23cee8409395d4552232285230e3ebc1d2ff3af547eafcd537164')
    prior = checked(previous)
    need(admission['config_sha256'] == previous['sha256'] and prior['node'] == NODE
         and prior['gpus'] == GPUS and Path(prior['queue']) == queue, 'Exact zero-call predecessor differs')
    failure = checked(ref(queue / 'failure.json'))
    need(failure['worker_started'] is True and failure['phase'] == prior['phase'] and
         'ValueError: Actual vLLM0.21 runtime required' in (queue / 'worker.runtime.log').read_text(),
         'Only diagnosed pre-load version rejection may recover')
    execution = Path(prior['out'])
    need((execution / 'driver.started.json').exists() and not (execution / 'model_load.receipt.json').exists()
         and not (execution / 'model.identity.json').exists() and not pilot.DEFAULT_OUT.exists(),
         'Predecessor already loaded/dispatched or pilot output exists; never resubmit')


def restored(gpu, safe_ref):
    name, phase = PRIOR[gpu]
    queue = BAR / 'queues' / name
    need(Path(safe_ref['path']).resolve() == queue / 'resources_safe.json', 'Wrong GPU predecessor safe queue')
    safe, admission, value = checked(safe_ref), checked(ref(queue / 'admission.json')), checked(ref(queue / ('holder_restored_gpu%d.json' % gpu)))
    verify_preload_failure(queue, admission)
    need(safe['phase'] == admission['phase'] == phase and admission['node'] == NODE and admission['gpus'] == GPUS
         and admission['manager_sha256'] == BASE_SHA and admission['model_identity_sha256'] == MODEL_SHA
         and all(safe.get(k) is True for k in ('all_holders_safe', 'locks_released', 'leases_restored', 'managed_parent_wait', 'unknown_jobs_not_signaled')),
         'Actual predecessor not safely restored')
    old = next(h for h in admission['holders'] if h['gpu'] == gpu)
    need(value['gpu'] == gpu and value['uuid'] == old['uuid'] and value['managed_parent_wait'] is True
         and value['restore_argv'] == base.RESTORE_ARGV
         and all(value['original_identity'][k] == old['identity'][k] for k in base.IDENTITY_KEYS), 'Actual restored holder binding differs')
    need(Path(safe_ref['path']).stat().st_mtime_ns >= (queue / 'failure.json').stat().st_mtime_ns >=
         (queue / 'admission.json').stat().st_mtime_ns, 'Predecessor receipt order differs')
    return value

def validate_capture(capture):
    helpers.validate_capture(capture)
    need(set(capture['predecessor_safe_refs']) == {'2', '3'}, 'Two distinct actual predecessor proofs required')
    for gpu, holder in zip(GPUS, capture['holders']):
        safe_ref = capture['predecessor_safe_refs'][str(gpu)]
        previous = restored(gpu, safe_ref)
        need(capture['captured_unix'] * 1e9 >= Path(safe_ref['path']).stat().st_mtime_ns and
             holder['uuid'] == previous['uuid'] and
             all(holder[k] == previous['holder_identity'][k] for k in base.IDENTITY_KEYS), 'Capture must name actual restored GPU holder')
    return capture


def capture_current(out):
    frozen()
    proofs = {str(g): ref(BAR / 'queues' / PRIOR[g][0] / 'resources_safe.json') for g in GPUS}
    for gpu in GPUS:
        restored(gpu, proofs[str(gpu)])
    addresses = helpers.subprocess.check_output(['hostname', '-I'], text=True).split()
    need(helpers.HOST in addresses, 'Capture must execute on actual N6')
    running, holders = base.apps(), []
    for gpu in GPUS:
        uuid = base.uuid_for(gpu)
        apps = [a for a in running if a['uuid'] == uuid]
        need(len(apps) == 1, 'Only sole restored native holder can be captured')
        holder = dict(base.identity(apps[0]['pid']), gpu=gpu, uuid=uuid)
        base.check_holder(holder, base.identity(holder['pid']), gpu, uuid, running)
        holders.append(holder)
    capture = dict(schema='visual_bench_native_capture_v1', node=NODE, gpus=GPUS, mode='consumer',
        captured_unix=time.time(), physical_addresses=addresses, holders=holders, headroom=base.headroom(),
        lock_inventory={str(g): helpers.lock_snapshot(g) for g in GPUS},
        lease_inventory={str(g): helpers.lease_snapshot(g) for g in GPUS},
        predecessor_safe_refs=proofs, signals_sent=False)
    validate_capture(capture)
    helpers.verify_capture_current(capture)
    return pilot.direct.native()[1].ref(pilot.direct.native()[1].save(out, capture))


def selected(cfg):
    return list(pilot.SOURCES), 3


def validate(cfg):
    frozen()
    need(not (Path(cfg['out']) / 'driver.started.json').exists(), 'Execution already started; no resume')
    need(cfg['node'] == NODE and cfg['gpus'] == GPUS and cfg['mode'] == 'consumer' and
         cfg['release_authorized'] is True and cfg['env']['CUDA_VISIBLE_DEVICES'] == CVD and
         cfg['targetcount'] == cfg['input_count'] == 3, 'Fixed actual N6 G2/3 three-source scope required')
    need(cfg['resource_manager_ref'] == ref(SELF) and cfg['producer'] == cfg['resource_manager_ref'] and
         cfg['plan'] == ref(pilot.__file__) and cfg['source_sha256'][str(RT / 'resource_manager.py')] == BASE_SHA and
         cfg['modelref']['model_identity_sha256'] == MODEL_SHA, 'Separate wrapper/base/recipe/model binding differs')
    capture = validate_capture(checked(cfg['holder_capture']))
    need(cfg['holders'] == capture['holders'], 'Config capture holders differ')
    need(cfg['argv'] == [PYTHON, '-B', str(SELF), 'worker', '--config-file', cfg['config_path']] and
         cfg['cwd'] == str(RT) and cfg['pilot_out'] == str(pilot.DEFAULT_OUT), 'Exact full pilot worker argv/output required')
    queue, execution = Path(cfg['queue']).resolve(), Path(cfg['out']).resolve()
    need(queue.is_relative_to(BAR / 'queues') and execution.parent == BAR and
         execution.name.startswith('caption_batch8_direct_full_execution_') and
         queue != execution and not queue.is_relative_to(execution) and not execution.is_relative_to(queue) and
         execution != pilot.DEFAULT_OUT, 'Fresh independent execution root/queue required')
    need(cfg['phase'] and all(c.isalnum() or c in '_-' for c in cfg['phase']), 'Invalid native phase')


def build_config(args):
    capture_ref = dict(path=str(args.capture.resolve()), sha256=args.capture_sha)
    capture = validate_capture(checked(capture_ref))
    modelref = dict(path=str(args.modelref.resolve()), sha256=args.modelref_sha, model_identity_sha256=MODEL_SHA)
    paths = [SELF, Path(pilot.__file__), RT / 'resource_manager.py', RT / 'bench_manager_driver_v2.py',
             RT / 'curated_resource_manager.py', RT / 'visual_batch_fixed3_benchmark_v1.py',
             RT / 'caption_cross_source_stage_v1.py', RT / 'caption_direct_facts_v1.py',
             Path(capture_ref['path']), Path(modelref['path'])]
    paths.extend(pilot.direct.SOURCE / name for name in pilot.direct.FROZEN)
    paths.append(RT / 'caption_batch8_direct_full_n6_launch_main_v2.json')
    for gpu in GPUS:
        queue = Path(capture['predecessor_safe_refs'][str(gpu)]['path']).parent
        paths.extend(queue / name for name in ('admission.json', 'resources_safe.json', 'failure.json', 'worker.runtime.log',
                                               'holder_restored_gpu%d.json' % gpu))
    cfg = dict(node=NODE, gpus=GPUS, mode='consumer', release_authorized=True, phase=args.phase,
        targetcount=3, input_count=3, holders=capture['holders'], holder_capture=capture_ref, modelref=modelref,
        out=str(args.execution_root.resolve()), pilot_out=str(pilot.DEFAULT_OUT), queue=str(args.queue.resolve()),
        config_path=str(args.config_out.resolve()), cwd=str(RT), reuse_existing_results=False,
        resource_manager_ref=ref(SELF), producer=ref(SELF), plan=ref(pilot.__file__),
        env=dict(CUDA_VISIBLE_DEVICES=CVD, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                 VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1', OMP_NUM_THREADS='2'),
        source_sha256={str(p.resolve()): base.sha(p) for p in paths})
    cfg['argv'] = [PYTHON, '-B', str(SELF), 'worker', '--config-file', cfg['config_path']]
    need(not args.queue.exists() and not args.execution_root.exists() and not pilot.DEFAULT_OUT.exists(), 'New queue/execution/pilot output only')
    validate(cfg)
    return ref(pilot.direct.native()[1].save(args.config_out, cfg))


def actual_state(llm):
    vc = getattr(llm, 'vllm_config', None)
    if vc is None:
        vc = llm.llm_engine.vllm_config
    m, s, mm = vc.model_config, vc.scheduler_config, vc.model_config.multimodal_config
    need(str(Path(m.model).resolve()) == MODEL and str(m.dtype).removeprefix('torch.') == 'bfloat16' and
         m.quantization is None and m.trust_remote_code is False and m.seed == 0 and m.max_model_len == 16384 and
         vc.parallel_config.tensor_parallel_size == 2 and s.max_num_seqs == 8 and
         mm.mm_processor_kwargs == {'max_pixels': 200704} and mm.get_limit_per_prompt('image') == 16 and
         mm.get_limit_per_prompt('video') == 0 and vc.structured_outputs_config.backend == 'xgrammar' and
         vc.structured_outputs_config.disable_any_whitespace is True and vc.cache_config.enable_prefix_caching is True and
         vc.cache_config.gpu_memory_utilization == .88, 'Actual loaded vLLM0.21 config differs')
    return dict(verified_actual_vllm_config=True, model=MODEL, tp=2, max_num_seqs=8, max_pixels=200704)


class Runtime:
    def __init__(self, llm, gate):
        self.llm, self.gate, self.calls = llm, gate, 0
        self._ms_bar_model_sha256 = MODEL_SHA

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def chat(self, messages, *a, **kw):
        self.gate()
        actual_state(self.llm)
        bar, _ = pilot.direct.native()
        need(self.calls + len(messages) <= 8 and all(m[0]['content'] == bar.SYSTEM for m in messages),
             'Only eight new native window calls; no boundary/rewrite/style')
        self.calls += len(messages)
        return self.llm.chat(messages, *a, **kw)


def worker(config_path, factory=None):
    cfg = pilot.direct.native()[0].core.strict_json(Path(config_path).read_bytes())
    cfg.update(_config_path=str(Path(config_path).resolve()), _config_sha=base.sha(config_path))
    need(cfg['_config_path'] == cfg['config_path'], 'Worker config path differs')
    validate(cfg)
    need(not pilot.DEFAULT_OUT.exists(), 'Full pilot already started; no redispatch')
    parent_gate(cfg, before_load=True)
    bar, pipeline = pilot.direct.native()
    execution = Path(cfg['out'])
    pipeline.save(execution / 'driver.started.json', dict(parent=parent_gate(cfg, before_load=True), **pilot.FLAGS))
    args = pilot.pilot_args(MODEL, cfg['modelref']['path'])
    model = bar.model_reference(args)
    need(model['sha256'] == MODEL_SHA, 'Actual staged native model identity differs')
    if factory is None:
        from vllm import LLM, SamplingParams
    else:
        LLM, SamplingParams = factory
    llm = LLM(model=MODEL, dtype='bfloat16', quantization=None, tensor_parallel_size=2, max_model_len=16384,
        max_num_seqs=8, gpu_memory_utilization=.88, limit_mm_per_prompt={'image': 16, 'video': 0},
        mm_processor_kwargs={'max_pixels': 200704}, structured_outputs_config={'backend': 'xgrammar', 'disable_any_whitespace': True},
        enable_prefix_caching=True, trust_remote_code=False, seed=0)
    runtime = Runtime(llm, lambda: parent_gate(cfg))
    pipeline.save(execution / 'model.identity.json', model)
    pipeline.save(execution / 'model_load.receipt.json', dict(actual_runtime=actual_state(llm),
        parent=parent_gate(cfg), model_identity_sha256=MODEL_SHA, model_load_completed=True, **pilot.FLAGS))
    def loaded():
        # First call is inside direct_recipe; never populate pilot out in admission.
        bar.core.save_json(Path(args.out) / 'model.identity.json', model)
        return runtime, SamplingParams
    def admit(current_args, current_runtime):
        need(current_args is args and current_runtime is loaded, 'Caller runtime/config changed')
        parent_gate(cfg)
        actual_state(llm)
        return True
    report = pilot.run_pilot(args, loaded, resource_admission=admit)
    need(runtime.calls == 8, 'Actual eight-call native budget not completed')
    pipeline.save(execution / 'driver.summary.json', dict(status='completed', attempted_window_calls=runtime.calls,
        report_ref=ref(pilot.DEFAULT_OUT / 'report.json'), timing_ref=ref(pilot.DEFAULT_OUT / 'timing.json'), **pilot.FLAGS))
    return report


def summaries(cfg, out):
    value = checked(ref(out / 'driver.summary.json'))
    report = checked(value['report_ref'])
    need(value['status'] == 'completed' and value['attempted_window_calls'] == 8 and
         report['source_ids'] == list(pilot.SOURCES) and report['source_count'] == 3 and report['window_count'] == 8 and
         report['visual_chat_batch_sizes'] == [8] and report['boundary_model_calls'] == report['rewrite_model_calls'] ==
         report['new_style_model_calls'] == 0 and all(report[k] is False or report[k] == 0 for k in pilot.FLAGS),
         'Full native actual-run summary/scope differs')
    for result in report['results']:
        for observation in result['visual']['window_observations']:
            generation = checked(ref(Path(observation['attempt_path']) / 'generation.json'))
            need(generation['model_call_performed'] is True and generation['model_identity_sha256'] == MODEL_SHA,
                 'Missing actual fresh native window generation')
    return {str(out / 'driver.summary.json'): dict(summary=dict(stage='run', total_source_count=3,
        selected_source_count=3, processed_source_count=3, plan_sha256=cfg['plan']['sha256']),
        selection=dict(selected_ids=list(pilot.SOURCES)), actual_window_calls=8, semantic_approval=False, production_counts=0)}


def manager():
    frozen()
    def admit(cfg):
        validate(cfg)
        helpers.verify_capture_current(checked(cfg['holder_capture']))
    base.validate_pair, base.selected_sources, base.stage_summaries = admit, selected, summaries
    base.main()  # Preserves original __file__, --manager-sha and cleanup/resources_safe.


def main():
    if len(sys.argv) <= 1 or sys.argv[1].startswith('--'):
        manager()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('capture', 'build', 'worker'))
    command = sys.argv[1]
    if command == 'capture':
        parser.add_argument('--out', type=Path, required=True)
    elif command == 'worker':
        parser.add_argument('--config-file', type=Path, required=True)
    else:
        for name in ('capture', 'modelref', 'config-out', 'execution-root', 'queue'):
            parser.add_argument('--' + name, type=Path, required=True)
        for name in ('capture-sha', 'modelref-sha', 'phase'):
            parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = capture_current(args.out) if command == 'capture' else build_config(args) if command == 'build' else worker(args.config_file)
    if command != 'worker':
        import json
        print(json.dumps(result))


if __name__ == '__main__':
    main()
