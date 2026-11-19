"""Fresh fixed3 grounded/native trial; caller owns native N6 G2/3 lifecycle."""
import copy
import os
from pathlib import Path
import sys
import time
import types
from types import SimpleNamespace as NS

sys.dont_write_bytecode = True
RT, SELF = Path(__file__).resolve().parent, Path(__file__).resolve()
sys.path.insert(0, str(RT))
CPU_SITEPATH = os.environ.get('MS_VISUAL_CPU_SITEPATH', '/opt/venv_eval/lib/python3.11/site-packages')
if CPU_SITEPATH not in sys.path:
    sys.path.append(CPU_SITEPATH)  # Existing vLLM/torch sites retain precedence.
import caption_batch8_direct_full_manager_v1 as old
import caption_motion_grounded_request_v1 as grounded

base, helpers, pilot = old.base, old.helpers, old.pilot
BAR, MODEL, MODEL_SHA, BASE_SHA = RT.parent, old.MODEL, old.MODEL_SHA, old.BASE_SHA
NODE, GPUS, CVD, PYTHON = old.NODE, old.GPUS, old.CVD, old.PYTHON
need, ref, checked = old.need, old.ref, old.checked
RECIPE = grounded.RECIPE
OUT = BAR / 'caption_motion_grounded_fixed3_v1'
EXECUTION = BAR / 'caption_motion_grounded_execution_v1'
MANIFEST = BAR / 'visual_batch_fixed3_benchmark_v1/manifest.json'
MANIFEST_SHA = '399786d3f4102cb086042d3a5b8a7c67763579c3485f1917d839e4aecd7be7e0'
STYLE_REPORT = pilot.DEFAULT_OUT / 'report.json'
PRIOR = {g: ('caption_motion_grounded_full_n6_v1', 'caption_motion_grounded_full_n6_v1') for g in GPUS}
PRIOR_CONFIG = RT / 'caption_motion_grounded_full_launch_main_v1.json'
PRIOR_CONFIG_SHA = '71af58f45a6fa0f38716534ffba3b45af38a4d7bc0791b9ceb381a3225ca1b24'
FLAGS = dict(pilot.FLAGS, quality_approved=False)


def clone(fn, **changes):
    return types.FunctionType(fn.__code__, dict(fn.__globals__, **changes), fn.__name__, fn.__defaults__)


parent_gate = clone(helpers.parent_gate, SELF=SELF, base=base)
def actual_state(llm):
    value = old.actual_state(llm)  # Real config checks; no runtime-version prefix gate.
    vc = getattr(llm, 'vllm_config', None)
    vc = vc if vc is not None else llm.llm_engine.vllm_config
    m, mm = vc.model_config, vc.model_config.multimodal_config
    return dict(value, dtype=str(m.dtype), quantization=m.quantization, max_model_len=m.max_model_len,
        seed=m.seed, trust_remote_code=m.trust_remote_code, image_limit=mm.get_limit_per_prompt('image'),
        video_limit=mm.get_limit_per_prompt('video'), structured_backend=vc.structured_outputs_config.backend,
        disable_any_whitespace=vc.structured_outputs_config.disable_any_whitespace,
        prefix_caching=vc.cache_config.enable_prefix_caching, gpu_memory_utilization=vc.cache_config.gpu_memory_utilization)


def frozen():
    old.frozen()
    need(base.sha(old.__file__) == 'dc626f989f681143e9017aca9ce707dcd6001ff43c8d28561c0acf4a3f102b2f' and
         base.sha(grounded.__file__) == '4e646d9b564e4a070afc010f2a12fbad58a26f1c0bbc64f96c8096d97c5aa07e',
         'Frozen lifecycle/request adapter changed')


def restored(gpu, safe_ref):
    queue = BAR / 'queues' / PRIOR[gpu][0]
    need(Path(safe_ref['path']).resolve() == queue / 'resources_safe.json', 'Exact zero-preload G2/3 predecessor required')
    safe, admission, failure, value = (checked(r) for r in (safe_ref, ref(queue / 'admission.json'),
        ref(queue / 'failure.json'), ref(queue / ('holder_restored_gpu%d.json' % gpu))))
    prior = checked(dict(path=str(PRIOR_CONFIG), sha256=PRIOR_CONFIG_SHA))
    need(safe['phase'] == admission['phase'] == failure['phase'] == PRIOR[gpu][1] and
         admission['config_sha256'] == PRIOR_CONFIG_SHA and admission['node'] == NODE and admission['gpus'] == GPUS and
         admission['manager_sha256'] == BASE_SHA and admission['model_identity_sha256'] == MODEL_SHA and
         prior['node'] == NODE and prior['gpus'] == GPUS and prior['out'] == admission['out'] and
         Path(prior['queue']).resolve() == queue and failure['worker_started'] is True and
         all(safe.get(k) is True for k in ('all_holders_safe', 'locks_released', 'leases_restored',
                                         'managed_parent_wait', 'unknown_jobs_not_signaled')), 'Zero-preload failure/safe binding differs')
    log = (queue / 'worker.runtime.log').read_text()
    need("ModuleNotFoundError: No module named 'scipy'" in log and 'install_requests' in log and
         'caption_direct_facts_v1.py' in log and not any(Path(prior['out']).iterdir()) and
         not Path(prior['pilot_out']).exists() and not OUT.exists(),
         'Only zero-preload scipy failure may recover; old/new output must remain unclaimed')
    need(admission['producer'] == ref(RT / 'caption_motion_grounded_full_manager_v1.py') and
         admission['plan'] == ref(pilot.__file__), 'Zero-preload predecessor source/recipe differs')
    original = next(h for h in admission['holders'] if h['gpu'] == gpu)
    need(value['gpu'] == gpu and value['uuid'] == original['uuid'] and value['managed_parent_wait'] is True and
         value['restore_argv'] == base.RESTORE_ARGV and
         all(value['original_identity'][k] == original['identity'][k] for k in base.IDENTITY_KEYS), 'Restored identity binding differs')
    need(Path(safe_ref['path']).stat().st_mtime_ns >= (queue / 'failure.json').stat().st_mtime_ns >=
         (queue / 'admission.json').stat().st_mtime_ns, 'Native predecessor receipt order differs')
    return value


validate_capture = clone(old.validate_capture, BAR=BAR, restored=restored)
capture_current = clone(old.capture_current, SELF=SELF, BAR=BAR, PRIOR=PRIOR,
                        frozen=frozen, restored=restored, validate_capture=validate_capture)


def inputs(manifest_ref, style_ref):
    need(Path(manifest_ref['path']).resolve() == MANIFEST.resolve() and manifest_ref['sha256'] == MANIFEST_SHA,
         'Unique fixed benchmark manifest required')
    manifest, report = checked(manifest_ref), checked(style_ref)
    need([c['source_id'] for c in manifest['cases']] == list(pilot.SOURCES) and
         [c['window_count'] for c in manifest['cases']] == [2, 3, 3] and manifest['request_count'] == 8 and
         report['source_ids'] == list(pilot.SOURCES) and report['window_count'] == 8 and
         manifest['model_ref']['model_identity_sha256'] == MODEL_SHA, 'Fixed three-source mapping differs')
    styles = {r['id']: r['joint_candidate'] for r in report['results']}
    keys = [sid + ':w%05d' % w for sid, count in zip(pilot.SOURCES, (2, 3, 3)) for w in range(count)]
    need([r['key'] for r in manifest['requests']] == keys, 'Wrong source/window mapping')
    entries, refs = [], [manifest_ref, style_ref]
    bar, _ = pilot.direct.native()
    for record in manifest['requests']:
        sid, wid, m = record['source_id'], record['window_id'], record['manifest']
        need(record['key'] == sid + ':' + wid and m['window']['window_id'] == wid, 'Wrong native source/window mapping')
        request_ref, binding_ref = (record['refs'][n] for n in ('request.json', 'binding.json'))
        original, binding = checked(request_ref), checked(binding_ref)
        refs.extend((request_ref, binding_ref))
        style = styles[sid]
        need(binding['source_binding']['id'] == sid and binding['window'] == m['window'] and
             binding['frames_sha256'] == bar.batch.digest([{k: f[k] for k in
                 ('index', 'time', 'source_timestamp', 'file', 'sha256')} for f in m['frames']]) and original['model'] == MODEL and
             style['annotator_model_sha256'] == MODEL_SHA and
             all(binding[k] == v for k, v in style['source_binding'].items() if k != 'params_sha256') and
             style['style_result']['id'] == sid and style['style_result']['status'] in ('resolved', 'candidate') and
             style['style_result']['dance_style'] not in ('', 'unknown'), 'Original source/model/style binding differs')
        indices = [f['index'] for f in m['frames']]
        need(len(indices) == 8 and len(set(indices)) == 8, 'Original eight distinct global frames required')
        header = grounded.json.loads(original['messages'][1]['content'][0]['text'])
        denied = {p: sorted(set(indices) & set(v)) for p, v in record['forbidden'].items() if set(indices) & set(v)}
        need(header['allowed_source_frame_indices'] == indices and header['explicit_raw2d_forbidden_part_frames'] == denied,
             'Original global frames/hard mask mapping differs')
        request = grounded.prepare_request(original, m)
        content = original['messages'][1]['content']
        urls = [x['image_url']['url'] for x in content if x['type'] == 'image_url']
        body_count = sum(x['type'] == 'text' and x['text'].startswith('Source frame ') for x in content)
        need(len(urls) == record['image_count'] <= 16 and body_count == 8 and
             all(u.startswith('data:image/jpeg;base64,') for u in urls), 'Original body/crop image mapping differs')
        forbidden = {p: set(v) for p, v in record['forbidden'].items()}
        def validator(value, m=m, f=forbidden, schema=request['json_schema']):
            helpers.bench.schema_check(value, schema)
            return bar.filter_supported_observation(value, m, f)
        new_binding = dict(binding, caption_recipe=RECIPE, original_request_ref=request_ref,
            original_binding_ref=binding_ref, recipe_identity=request['recipe_identity'],
            recipe_identity_sha256=request['recipe_identity_sha256'])
        entries.append(dict(root=OUT / 'sources' / sid / 'windows' / wid / 'attempt-0', request=request,
            binding=new_binding, validator=validator, record=record,
            image_count=len(urls), body_image_count=body_count, crop_image_count=len(urls) - body_count))
    return entries, styles, refs


class Runtime:
    def __init__(self, llm, gate, entries):
        self.llm, self.gate, self.expected = llm, gate, [e['request']['messages'] for e in entries]
        self.calls, self.chatwall = 0, 0.0
        self._ms_bar_model_sha256 = MODEL_SHA
    def __getattr__(self, name):
        return getattr(self.llm, name)
    def chat(self, messages, *a, **kw):
        need(self.calls == 0 and messages == self.expected and len(messages) == 8,
             'Exactly eight reserved grounded windows; no boundary/rewrite/style/retry')
        need(self.gate() is True, 'Caller ownership denied before dispatch')
        actual_state(self.llm)
        self.calls = 8  # Consumed before dispatch, including exceptions/bad outputs.
        begin = time.perf_counter()
        try:
            return self.llm.chat(messages, *a, **kw)
        finally:
            self.chatwall = time.perf_counter() - begin


def run_fixed(manifest_ref, style_ref, model, loader=None, admission=None):
    need(callable(loader) and callable(admission) and admission() is True, 'Caller default denied')
    need(not OUT.exists(), 'Old output/claim/attempt declined; no retry or resume')
    entries, styles, refs = inputs(manifest_ref, style_ref)
    bar, pipeline = pilot.direct.native()
    need(model['root'] == MODEL and model['sha256'] == MODEL_SHA, 'Actual model identity differs')
    OUT.mkdir(parents=True, exist_ok=False)
    helpers.bench.exclusive(OUT / 'claim.json', helpers.bench.encoded(dict(recipe=RECIPE, reserved_window_calls=8,
        max_completion_tokens=14400, original_manifest_ref=manifest_ref, no_retry=True, **FLAGS)))
    bar.core.save_json(OUT / 'model.identity.json', model)
    begin = time.perf_counter()
    llm, sampling = loader()
    runtime = Runtime(llm, admission, entries)
    def persist(root, request, output, completion, llm):
        def save(path, value):
            if Path(path).name == 'generation.json':
                value = dict(value, generation_recipe=RECIPE, fact_source_recipe=RECIPE,
                    recipe_identity_sha256=request['recipe_identity_sha256'])
            bar.core.save_json(path, value)
        original = clone(bar.persist_completion, core=NS(**dict(bar.core.__dict__, save_json=save)))
        original(root, request, output, completion, llm)
    transactions = clone(bar.transactions, persist_completion=persist)
    observations = transactions(entries, lambda: (runtime, sampling), NS(stage='run', batch_size=8))
    need(runtime.calls == 8 and all(observations[str(e['root'])]['status'] in ('candidate', 'abstain') for e in entries),
         'Native output failed; eight-call budget consumed, never retry')
    results, generations = [], []
    for e in entries:
        o = observations[str(e['root'])]
        o.update(window=e['record']['manifest']['window'], attempt_path=str(e['root']), caption_recipe=RECIPE,
                 native_visual_model_protocol=False, **FLAGS)
        if o['json']['abstain']:
            o['status'] = 'abstain'
        generation = bar.core.validate_generation(e['root'], e['request'])
        need(generation['model_identity_sha256'] == MODEL_SHA, 'Fresh native generation model identity differs')
        bar.core.save_json(e['root'] / 'recipe.receipt.json', dict(recipe=RECIPE,
            generation_ref=ref(e['root'] / 'generation.json'), recipe_identity=e['request']['recipe_identity'], **FLAGS))
        generations.append(dict(key=e['record']['key'], generation_ref=ref(e['root'] / 'generation.json'),
            prompt_tokens=generation['prompt_tokens'], completion_tokens=generation['completion_tokens'],
            image_count=e['image_count'], body_image_count=e['body_image_count'], crop_image_count=e['crop_image_count']))
    for sid in pilot.SOURCES:
        local = [observations[str(e['root'])] for e in entries if e['record']['source_id'] == sid]
        windows = [o['window'] for o in local]
        facts = bar.facts_from_observations(windows, local)
        final = pilot.direct.direct_final(windows, facts, [])
        for fact in facts:
            fact.update(source=RECIPE, caption_recipe=RECIPE, native_visual_model_protocol=False, **FLAGS)
        final.update(derivation=RECIPE, caption_recipe=RECIPE, **FLAGS)
        captions = final.get('json', {}).get('windows', [])
        body = bar.core.timed_text({'windows': windows}, captions) if captions else ''
        style = styles[sid]['style_result']
        results.append(dict(id=sid, windows=local, facts=facts, final=final, final_timed_caption=body,
            uncertainties=[dict(window_id=o['window']['window_id'], text=o['json']['uncertainties']) for o in local],
            style_result=style, original_style_report_ref=style_ref, original_style_reused_without_model_call=True,
            target_text='Dance style: ' + style['dance_style'] + '\n' + body if body else '',
            all_windows_nonempty=len(captions) == len(windows) and all(w['caption'].strip() for w in captions), **FLAGS))
    need(admission() is True, 'Caller ownership denied at commit')
    actual = actual_state(llm)
    for r in refs:
        need(base.sha(r['path']) == r['sha256'], 'Original request/metadata changed at commit')
    report = dict(schema=RECIPE, status='completed', source_ids=list(pilot.SOURCES), source_count=3, window_count=8,
        attempted_window_calls=8, visual_chat_batch_sizes=[8], boundary_model_calls=0, rewrite_model_calls=0,
        new_style_model_calls=0, model_identity=model, actual_runtime=actual,
        runtime_version=getattr(llm, '_grounded_runtime_version', 'CPU_fake_not_actual_GPU'),
        chatwall_seconds=runtime.chatwall, transaction_wall_seconds=time.perf_counter() - begin,
        generation_records=generations, results=results, independent_pixel_review_required=True, **FLAGS)
    pipeline.save(OUT / 'report.json', report)
    return report


def selected(cfg):
    return list(pilot.SOURCES), 3


def validate(cfg):
    frozen()
    need(not (Path(cfg['out']) / 'driver.started.json').exists() and not OUT.exists(), 'Execution/output already claimed')
    need(cfg['node'] == NODE and cfg['gpus'] == GPUS and cfg['mode'] == 'consumer' and cfg['release_authorized'] is True and
         cfg['env']['CUDA_VISIBLE_DEVICES'] == CVD and cfg['env']['MS_VISUAL_CPU_SITEPATH'] == CPU_SITEPATH and cfg['input_count'] == cfg['targetcount'] == 3,
         'Exact N6 pair/source scope required')
    need(cfg['resource_manager_ref'] == cfg['producer'] == ref(SELF) and cfg['plan'] == ref(grounded.__file__) and
         cfg['source_sha256'][str(RT / 'resource_manager.py')] == BASE_SHA and cfg['modelref']['model_identity_sha256'] == MODEL_SHA,
         'Wrapper/base/recipe/model identity differs')
    capture = validate_capture(checked(cfg['holder_capture']))
    need(cfg['holders'] == capture['holders'] and cfg['manifest_ref'] == ref(MANIFEST) and
         cfg['modelref'] == checked(cfg['manifest_ref'])['model_ref'] and cfg['style_ref'] == ref(STYLE_REPORT),
         'Capture/model/source/style references differ')
    need(cfg['out'] == str(EXECUTION) and cfg['trial_out'] == str(OUT) and Path(cfg['queue']).resolve().parent == BAR / 'queues' and
         cfg['argv'] == [PYTHON, '-B', str(SELF), 'worker', '--config-file', cfg['config_path']] and cfg['cwd'] == str(RT) and
         cfg['phase'] and all(c.isalnum() or c in '_-' for c in cfg['phase']), 'Independent worker/root/phase differs')


def build_config(args):
    frozen()
    capture_ref = dict(path=str(args.capture.resolve()), sha256=args.capture_sha)
    capture = validate_capture(checked(capture_ref))
    need(args.execution_root.resolve() == EXECUTION and not args.execution_root.exists() and not args.queue.exists() and
         not OUT.exists() and not args.config_out.exists(), 'Fresh fixed execution/queue/trial/config required')
    manifest_ref, style_ref = ref(MANIFEST), ref(STYLE_REPORT)
    _, _, inputs_refs = inputs(manifest_ref, style_ref)
    modelref = dict(path=str(args.modelref.resolve()), sha256=args.modelref_sha, model_identity_sha256=MODEL_SHA)
    paths = [SELF, Path(grounded.__file__), Path(old.__file__), Path(pilot.__file__), Path(pilot.direct.__file__),
        Path(grounded.compact.__file__), RT / 'resource_manager.py', RT / 'bench_manager_driver_v2.py', RT / 'curated_resource_manager.py']
    paths.extend(pilot.direct.SOURCE / name for name in pilot.direct.FROZEN)
    paths.extend(RT / n for n in ('caption_motion_grounded_full_manager_v1.py', 'visual_batch_fixed3_benchmark_v1.py'))
    paths.append(PRIOR_CONFIG)
    paths.extend(pilot.direct.SOURCE / n for n in ('ms_visual_batch.py', 'ms_visual_observe.py'))
    paths.extend(Path(r['path']) for r in inputs_refs + [capture_ref, modelref])
    paths.extend(BAR / 'queues' / PRIOR[2][0] / n for n in ('admission.json', 'failure.json', 'worker.runtime.log', 'resources_safe.json',
        'holder_restored_gpu2.json', 'holder_restored_gpu3.json'))
    cfg = dict(node=NODE, gpus=GPUS, mode='consumer', release_authorized=True, phase=args.phase, targetcount=3, input_count=3,
        holders=capture['holders'], holder_capture=capture_ref, modelref=modelref, manifest_ref=manifest_ref, style_ref=style_ref,
        out=str(EXECUTION), trial_out=str(OUT), queue=str(args.queue.resolve()), config_path=str(args.config_out.resolve()), cwd=str(RT),
        reuse_existing_results=False, resource_manager_ref=ref(SELF), producer=ref(SELF), plan=ref(grounded.__file__),
        env=dict(CUDA_VISIBLE_DEVICES=CVD, MS_VISUAL_CPU_SITEPATH=CPU_SITEPATH, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1', OMP_NUM_THREADS='2'),
        source_sha256={str(p.resolve()): base.sha(p) for p in paths})
    cfg['argv'] = [PYTHON, '-B', str(SELF), 'worker', '--config-file', cfg['config_path']]
    validate(cfg)
    _, pipeline = pilot.direct.native()
    return ref(pipeline.save(args.config_out, cfg))


def worker(config_path, factory=None):
    cfg = helpers.bench.strict(Path(config_path).read_bytes())
    cfg.update(_config_path=str(Path(config_path).resolve()), _config_sha=base.sha(config_path))
    need(cfg['_config_path'] == cfg['config_path'], 'Exact worker config path required')
    validate(cfg)
    parent_gate(cfg, before_load=True)
    bar, pipeline = pilot.direct.native()
    pipeline.save(EXECUTION / 'driver.started.json', dict(parent=parent_gate(cfg, before_load=True), **FLAGS))
    model = bar.model_reference(NS(model=MODEL, model_reference=cfg['modelref']['path']))
    def load():
        if factory is None:
            import vllm
            LLM, SamplingParams, version = vllm.LLM, vllm.SamplingParams, vllm.__version__
        else:
            LLM, SamplingParams, version = factory[0], factory[1], 'CPU_fake_not_actual_GPU'
        llm = LLM(model=MODEL, dtype='bfloat16', quantization=None, tensor_parallel_size=2, max_model_len=16384,
            max_num_seqs=8, gpu_memory_utilization=.88, limit_mm_per_prompt={'image': 16, 'video': 0},
            mm_processor_kwargs={'max_pixels': 200704}, structured_outputs_config={'backend': 'xgrammar', 'disable_any_whitespace': True},
            enable_prefix_caching=True, trust_remote_code=False, seed=0)
        llm._grounded_runtime_version = version
        pipeline.save(EXECUTION / 'model_load.receipt.json', dict(actual_runtime=actual_state(llm), runtime_version=version,
            model_identity_sha256=model['sha256'], parent=parent_gate(cfg), **FLAGS))
        return llm, SamplingParams
    report = run_fixed(cfg['manifest_ref'], cfg['style_ref'], model, load, lambda: bool(parent_gate(cfg)))
    pipeline.save(EXECUTION / 'driver.summary.json', dict(status='completed', attempted_window_calls=8,
        report_ref=ref(OUT / 'report.json'), **FLAGS))
    return report


def summaries(cfg, out):
    value = checked(ref(out / 'driver.summary.json'))
    report = checked(value['report_ref'])
    need(value['status'] == report['status'] == 'completed' and report['source_ids'] == list(pilot.SOURCES) and
         report['source_count'] == 3 and report['window_count'] == report['attempted_window_calls'] == 8 and
         len(report['generation_records']) == 8 and report['visual_chat_batch_sizes'] == [8] and
         report['boundary_model_calls'] == report['rewrite_model_calls'] == report['new_style_model_calls'] == 0 and
         all(report[k] is False or report[k] == 0 for k in FLAGS), 'Fresh actual summary differs')
    for record in report['generation_records']:
        g = checked(record['generation_ref'])
        need(g['model_call_performed'] is True and g['model_identity_sha256'] == MODEL_SHA and
             g['generation_recipe'] == g['fact_source_recipe'] == RECIPE, 'Fresh native generation missing')
    return {str(out / 'driver.summary.json'): dict(summary=dict(stage='run', total_source_count=3, selected_source_count=3,
        processed_source_count=3, plan_sha256=cfg['plan']['sha256']), selection=dict(selected_ids=list(pilot.SOURCES)),
        actual_window_calls=8, semantic_approval=False, production_counts=0)}


def manager():
    frozen()
    def admit(cfg):
        validate(cfg)
        helpers.verify_capture_current(checked(cfg['holder_capture']))
    base.validate_pair, base.selected_sources, base.stage_summaries = admit, selected, summaries
    base.main()  # Original file/SHA, owned pair recovery, safe receipts and normal holder wait.


main = clone(old.main, manager=manager, capture_current=capture_current, build_config=build_config, worker=worker)
if __name__ == '__main__':
    main()
