"""Finite N6 production caller; native resources remain in the original manager."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
sys.path.insert(0, str(RT))
import caption_batch8_production_consumer_v1 as consumer
spec = importlib.util.spec_from_file_location('production_actual_full_helpers', RT / 'caption_batch8_direct_full_manager_v3.py')
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)
base, helpers, BAR = full.base, full.helpers, RT.parent
BASE_SHA, MODEL, MODEL_SHA, PYTHON = full.BASE_SHA, full.MODEL, full.MODEL_SHA, full.PYTHON
NODE, GPUS, CVD = 'N6', [2, 3], '2,3'
PREVIOUS = 'caption_batch8_direct_full_n6_main_v3'
FULL_SHA = 'f00dae339c16323ecbfef51cd12f9c2a7ac2fa443c95124822c818aaefaec4e6'
need, ref, checked = consumer.need, full.ref, full.checked
FLAGS = dict(consumer.FLAGS, production_counts=0, production_claims=0)
_HANDOFFS = {}


def clone(fn, **replace):
    return types.FunctionType(fn.__code__, dict(fn.__globals__, **replace), fn.__name__, fn.__defaults__)


parent_gate = clone(full.parent_gate, SELF=SELF)
actual_state = full.actual_state


def frozen():
    need(Path(base.__file__).resolve() == RT / 'resource_manager.py' and base.sha(base.__file__) == BASE_SHA,
         'Original base identity must never be rebound')
    need(base.sha(full.__file__) == FULL_SHA, 'Actual native helper changed')


def restored(gpu, safe_ref):
    queue = BAR / 'queues' / PREVIOUS
    need(Path(safe_ref['path']).resolve() == queue / 'resources_safe.json', 'Only actual full main_v3 predecessor safe required')
    safe, admission, exited, value = checked(safe_ref), checked(ref(queue / 'admission.json')), checked(ref(queue / 'worker_exit.json')), checked(ref(queue / ('holder_restored_gpu%d.json' % gpu)))
    need(safe['phase'] == admission['phase'] == exited['phase'] == PREVIOUS and admission['node'] == NODE and
         admission['gpus'] == GPUS and admission['manager_sha256'] == BASE_SHA and admission['model_identity_sha256'] == MODEL_SHA and
         all(safe.get(k) is True for k in ('all_holders_safe', 'locks_released', 'leases_restored', 'managed_parent_wait', 'unknown_jobs_not_signaled')),
         'Actual full predecessor not safely restored')
    need(exited['exit_code'] == 0 and exited['source_unchanged'] is True and bool(exited['summaries']) and
         exited['producer'] == admission['producer'] and exited['plan'] == admission['plan'] and
         exited['producer']['sha256'] == FULL_SHA, 'Actual full worker did not complete')
    old = next(h for h in admission['holders'] if h['gpu'] == gpu)
    need(value['gpu'] == gpu and value['uuid'] == old['uuid'] and value['managed_parent_wait'] is True and
         value['restore_argv'] == base.RESTORE_ARGV and
         all(value['original_identity'][k] == old['identity'][k] for k in base.IDENTITY_KEYS), 'Actual full restored holder differs')
    need(Path(safe_ref['path']).stat().st_mtime_ns >= (queue / 'worker_exit.json').stat().st_mtime_ns >=
         (queue / 'admission.json').stat().st_mtime_ns, 'Actual full receipt order differs')
    return value


validate_capture = clone(full.validate_capture, restored=restored)
_capture = clone(full.capture_current, frozen=frozen, restored=restored, validate_capture=validate_capture,
                 PRIOR={g: (PREVIOUS, PREVIOUS) for g in GPUS}, BAR=BAR)


def capture_current(out):
    need(not Path(out).exists(), 'Fresh capture output required')
    return _capture(out)


def consumer_args(cfg):
    args = consumer.parser().parse_args(cfg['consumer_argv'])
    args.handoff_ref = dict(path=str(Path(args.handoff).resolve()), sha256=args.handoff_sha)
    return args


def handoff(cfg):
    """Mandatory exact handoff is verified by the finite consumer, never a boolean."""
    need(isinstance(cfg.get('handoff_ref'), dict), 'Exact stopped old-lane handoff artifact required')
    args = consumer_args(cfg)
    need(args.handoff_ref == cfg['handoff_ref'], 'Config/consumer handoff reference differs')
    key = json.dumps(vars(args), sort_keys=True)
    if key not in _HANDOFFS:
        inputs = consumer.Inputs()
        _, _, selected_meta, _ = consumer.assignment(args, inputs)
        _HANDOFFS[key] = inputs, selected_meta
    inputs, selected_meta = _HANDOFFS[key]
    inputs.verify()
    return dict(selected_meta=copy.deepcopy(selected_meta))


def selected(cfg):
    value = handoff(cfg)
    ids = [m['id'] for m in value['selected_meta']]
    need(0 < len(ids) <= 500 and len(ids) == len(set(ids)), 'Finite exact handoff selected count required')
    return ids, len(ids)


def validate(cfg):
    frozen()
    need(cfg['phase'] and all(c.isalnum() or c in '_-' for c in cfg['phase']), 'Invalid native phase')
    need(cfg['node'] == NODE and cfg['gpus'] == GPUS and cfg['mode'] == 'consumer' and cfg['release_authorized'] is True and
         cfg['env']['CUDA_VISIBLE_DEVICES'] == CVD, 'Only exact native N6 G2/3 assignment required')
    need(cfg['resource_manager_ref'] == ref(SELF) and cfg['producer'] == cfg['resource_manager_ref'] and cfg['plan'] == cfg['handoff_ref'] and
         cfg['source_sha256'][str(RT / 'resource_manager.py')] == BASE_SHA and cfg['consumer_ref'] == ref(consumer.__file__) and
         cfg['modelref']['model_identity_sha256'] == MODEL_SHA, 'Separate wrapper/base/consumer/model binding differs')
    ids, count = selected(cfg)
    need(cfg['targetcount'] == cfg['input_count'] == count, 'Native selected count is handoff count, not full denominator')
    capture = validate_capture(checked(cfg['holder_capture']))
    need(cfg['holders'] == capture['holders'], 'Config/restored actual holders differ')
    need(cfg['argv'] == [PYTHON, '-B', str(SELF), 'worker', '--config-file', cfg['config_path']] and cfg['cwd'] == str(RT), 'Exact finite worker argv required')
    args = consumer_args(cfg)
    queue, execution, out = Path(cfg['queue']).resolve(), Path(cfg['out']).resolve(), Path(args.out).resolve()
    need(queue.is_relative_to(BAR / 'queues') and execution.parent == BAR and execution.name.startswith('caption_batch8_production_execution_') and
         out.parent == BAR and out.name.startswith('caption_batch8_production_') and execution != out and
         not out.is_relative_to(execution) and not execution.is_relative_to(out) and
         not queue.is_relative_to(out) and not out.is_relative_to(queue) and not queue.is_relative_to(execution) and not execution.is_relative_to(queue), 'Separate native execution/caption/queue roots required')
    need(args.model == MODEL and args.model_reference == cfg['modelref']['path'] and args.tp == 2 and args.batch_size == 8 and
         args.max_num_seqs == 8 and args.stage == 'run' and args.boundary_review == 'off' and not args.retry_failed,
         'Original native model/TP2/batch8/direct configuration required')
    need(not (execution / 'driver.started.json').exists(), 'New execution root only; source resume uses separate owned consumer output')


def build_config(args):
    need(not args.config_out.exists() and not args.execution_root.exists() and not args.queue.exists(), 'Fresh launch/execution/queue required')
    capture_ref = dict(path=str(args.capture.resolve()), sha256=args.capture_sha)
    capture = validate_capture(checked(capture_ref))
    handoff_ref = dict(path=str(args.handoff_ref.resolve()), sha256=args.handoff_sha)
    modelref = dict(path=str(args.modelref.resolve()), sha256=args.modelref_sha, model_identity_sha256=MODEL_SHA)
    argv = list(args.consumer_args)
    need(argv and argv[0] == '--', 'Consumer native options must follow --')
    argv = argv[1:]
    cfg = dict(node=NODE, gpus=GPUS, mode='consumer', release_authorized=True, phase=args.phase, holders=capture['holders'],
        out=str(args.execution_root.resolve()), queue=str(args.queue.resolve()), config_path=str(args.config_out.resolve()), cwd=str(RT),
        resource_manager_ref=ref(SELF), producer=ref(SELF), plan=handoff_ref, handoff_ref=handoff_ref, modelref=modelref,
        consumer_ref=ref(consumer.__file__), consumer_argv=argv, holder_capture=capture_ref, reuse_existing_results=False,
        env=dict(CUDA_VISIBLE_DEVICES=CVD, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1', OMP_NUM_THREADS='2'))
    cfg['argv'] = [PYTHON, '-B', str(SELF), 'worker', '--config-file', cfg['config_path']]
    cfg['targetcount'] = cfg['input_count'] = selected(cfg)[1]
    paths = [SELF, Path(consumer.__file__), RT / 'resource_manager.py', Path(full.__file__), RT / 'bench_manager_driver_v2.py',
             RT / 'curated_resource_manager.py', RT / 'caption_cross_source_stage_v1.py', RT / 'caption_direct_facts_v1.py',
             RT / 'curated_transaction_cache_v1.py', RT / 'curated_fast_source_v1.py', RT / 'curated_request_image_cache_v1.py',
             Path(capture_ref['path']), Path(handoff_ref['path']), Path(modelref['path'])]
    native_args = consumer_args(cfg)
    paths.extend([Path(native_args.source_plan), Path(native_args.manifest)])
    paths.extend(consumer.direct.SOURCE / name for name in consumer.direct.FROZEN)
    queue = BAR / 'queues' / PREVIOUS
    paths.extend(queue / name for name in ('admission.json', 'worker_exit.json', 'resources_safe.json', 'holder_restored_gpu2.json', 'holder_restored_gpu3.json'))
    cfg['source_sha256'] = {str(p.resolve()): base.sha(p) for p in paths}
    validate(cfg)
    return ref(consumer.direct.native()[1].save(args.config_out, cfg))


class Runtime:
    def __init__(self, llm, gate):
        self.llm, self.gate, self.calls = llm, gate, {'visual': [], 'style': []}
        self._ms_bar_model_sha256 = MODEL_SHA
    def __getattr__(self, name):
        return getattr(self.llm, name)
    def chat(self, messages, *args, **kwargs):
        self.gate()
        actual_state(self.llm)
        bar, _ = consumer.direct.native()
        import ms_dance_style as style
        system = messages[0][0]['content']
        need(system in (bar.SYSTEM, style.SYSTEM) and all(m[0]['content'] == system for m in messages), 'No boundary/rewrite or mixed protocol dispatch allowed')
        self.calls['visual' if system == bar.SYSTEM else 'style'].append(len(messages))
        return self.llm.chat(messages, *args, **kwargs)


def worker(config_path, factory=None):
    bar, pipeline = consumer.direct.native()
    cfg = bar.core.strict_json(Path(config_path).read_bytes())
    cfg.update(_config_path=str(Path(config_path).resolve()), _config_sha=base.sha(config_path))
    need(cfg['_config_path'] == cfg['config_path'], 'Worker config path differs')
    validate(cfg)
    parent_gate(cfg, before_load=True)
    args = consumer_args(cfg)
    execution = Path(cfg['out'])
    pipeline.save(execution / 'driver.started.json', dict(parent=parent_gate(cfg, before_load=True), **FLAGS))
    model = bar.model_reference(args)
    need(model['sha256'] == MODEL_SHA, 'Actual staged native model identity differs')
    if factory is None:
        from vllm import LLM, SamplingParams
    else:
        LLM, SamplingParams = factory
    llm = LLM(model=MODEL, dtype='bfloat16', quantization=None, tensor_parallel_size=2, max_model_len=16384, max_num_seqs=8,
        gpu_memory_utilization=.88, limit_mm_per_prompt={'image': 16, 'video': 0}, mm_processor_kwargs={'max_pixels': 200704},
        structured_outputs_config={'backend': 'xgrammar', 'disable_any_whitespace': True}, enable_prefix_caching=True, trust_remote_code=False, seed=0)
    runtime = Runtime(llm, lambda: parent_gate(cfg))
    pipeline.save(execution / 'model.identity.json', model)
    pipeline.save(execution / 'model_load.receipt.json', dict(actual_runtime=actual_state(llm), parent=parent_gate(cfg), model_identity_sha256=MODEL_SHA, **FLAGS))
    identity_saved = False
    def loaded():
        nonlocal identity_saved
        if not identity_saved:
            bar.core.save_json(Path(args.out) / 'visual/model.identity.json', model)
            identity_saved = True
        return runtime, SamplingParams
    def admit(current_args, current_runtime):
        need(current_args is args and current_runtime is loaded, 'Caller runtime/config changed')
        parent_gate(cfg)
        actual_state(llm)
        return True
    report = consumer.run_consumer(args, loaded, resource_admission=admit)
    ids, count = selected(cfg)
    need(report['selected_source_count'] == report['terminal_count'] == count, 'Actual finite terminal/selected count differs')
    report_path = pipeline.save(execution / 'consumer.report.json', report)
    pipeline.save(execution / 'transaction.timings.json', report['transaction_cache_events'])
    pipeline.save(execution / 'driver.summary.json', dict(status='completed', report_ref=ref(report_path), selected_ids=ids,
        selected_source_count=count, calls=runtime.calls, consumer_out=str(Path(args.out).resolve()), **FLAGS))
    return report


def summaries(cfg, out):
    value = checked(ref(out / 'driver.summary.json'))
    report = checked(value['report_ref'])
    ids, count = selected(cfg)
    need(value['status'] == 'completed' and value['selected_ids'] == ids and report['selected_source_count'] == report['terminal_count'] == count and
         report['boundary_model_calls'] == report['rewrite_model_calls'] == 0 and all(report[k] is False for k in consumer.FLAGS), 'Actual finite summary/terminal protocol differs')
    bar, pipeline = consumer.direct.native()
    consumer_out = Path(value['consumer_out'])
    admission = checked(report['admission_ref'])
    meta = {r['meta']['id']: r['meta'] for r in admission['selected']}
    notices = list(consumer.ready_lines(consumer_out / 'completed.jsonl', bar))
    need(len(notices) == count and {r['id'] for r in notices} == set(ids), 'Actual finite completed membership/count differs')
    for notice in notices:
        need(pipeline.load(pipeline.source_dir(consumer_out, meta[notice['id']]) / 'consumer.terminal.json') == notice,
             'Actual finite terminal differs from completed notice')
    need(report['counts'] == {s: sum(r['status'] == s for r in notices) for s in sorted({r['status'] for r in notices})}, 'Actual finite partial/terminal status counts differ')
    return {str(out / 'driver.summary.json'): dict(summary=dict(stage='run', total_source_count=count,
        selected_source_count=count, processed_source_count=report['terminal_count'], plan_sha256=cfg['plan']['sha256']),
        selection=dict(selected_ids=ids), finite_source_counts=report['counts'], semantic_approval=False, production_counts=0)}


def manager():
    frozen()
    def admit(cfg):
        validate(cfg)
        helpers.verify_capture_current(checked(cfg['holder_capture']))
    base.validate_pair, base.selected_sources, base.stage_summaries = admit, selected, summaries
    base.main()


def main():
    if len(sys.argv) <= 1 or sys.argv[1].startswith('--'):
        manager()
        return
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('capture', 'build', 'worker'))
    if sys.argv[1] == 'capture':
        p.add_argument('--out', type=Path, required=True)
    elif sys.argv[1] == 'worker':
        p.add_argument('--config-file', type=Path, required=True)
    else:
        for name in ('capture', 'modelref', 'handoff-ref', 'config-out', 'execution-root', 'queue'):
            p.add_argument('--' + name, type=Path, required=True)
        for name in ('capture-sha', 'modelref-sha', 'handoff-sha', 'phase'):
            p.add_argument('--' + name, required=True)
    argv = sys.argv[1:]
    remainder = []
    if sys.argv[1] == 'build' and '--' in argv:
        split = argv.index('--')
        argv, remainder = argv[:split], argv[split:]
    args = p.parse_args(argv)
    if args.command == 'build':
        args.consumer_args = remainder
    result = capture_current(args.out) if args.command == 'capture' else build_config(args) if args.command == 'build' else worker(args.config_file)
    if args.command != 'worker':
        print(json.dumps(result))


if __name__ == '__main__':
    main()
