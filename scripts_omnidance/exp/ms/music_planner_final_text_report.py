"""Read-only first final-test500 weak-reference agreement; never a quality ranking."""
import argparse
from collections import Counter
import hashlib
import importlib.util
from itertools import combinations
import json
import os
from pathlib import Path
import random
import stat
import statistics
import sys

sys.dont_write_bytecode = True
FAMILIES = ('music_flamingo', 'moss', 'omni_captioner')
COUNT = 500
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 42
PINS = {
    'music_planner_eval.py': '55ca82f4ec117ffa05087e7dba54c1787de15ce6522ab0cd69e95f8f1270bea6',
    'planner_eval_manager_v3.py': 'c387edfd246fd25ab3ac73d00bffe2f41d5f1450cba15df9509b53192ac63c9d',
    'music_planner_parallel_test.py': '21f3815c248bd20ba5893aedbc5d53e8a3be2904e338f693e8a53cc68988fc92',
    'music_planner_parallel_test_manager.py': '91f0e90017238a2dd4b774694947a0d4670ddcc2f4df307d499feaea9e8b81c7',
    'music_planner_reference_f1.py': '3cdbba545b861839978d553b00935966163fa20e294b7b388550f806dd546ced',
}
CHANNELS = ('rouge1', 'rouge2', 'rougeL', 'window_token',
            'window_action_categories', 'window_body_categories')
PRF = ('precision', 'recall', 'f1')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    path = Path(path)
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)),
            'absolute nonsymlink input required: ' + str(path))
    require(stat.S_ISREG(path.lstat().st_mode), 'regular input required: ' + str(path))
    return path


class Inputs:
    """Snapshot every consumed input, then recheck before the exclusive report write."""
    def __init__(self):
        self.refs = {}

    def read(self, path, expected=None):
        path = regular(path)
        payload = path.read_bytes()
        ref = dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))
        if expected is not None:
            require(all(ref.get(k) == v for k, v in expected.items()), 'input SHA/ref mismatch: ' + str(path))
        if str(path) in self.refs:
            require(self.refs[str(path)] == ref, 'input changed during report: ' + str(path))
        self.refs[str(path)] = ref
        return payload

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))

    def ref(self, ref):
        self.read(ref['path'], ref)
        return self.refs[ref['path']]

    def recheck(self):
        for ref in list(self.refs.values()):
            self.read(ref['path'], ref)


def load_pinned(inputs, name, path):
    inputs.read(path, dict(path=str(path), sha256=PINS[Path(path).name]))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_completion(complete, exited, safe, parallel):
    require(complete.get('technical_collection_complete') is True
            and complete.get('complete_native_GPU_outputs_verified') is True,
            'missing technical/native completion')
    require(type(exited.get('exit_code')) is int and exited['exit_code'] == 0, 'worker did not exit0')
    require(all(safe.get(k) is True for k in ('all_holders_safe', 'locks_released', 'leases_restored',
            'managed_parent_wait', 'unknown_jobs_not_signaled'))
            and type(safe.get('admitted_gpu_count')) is int
            and safe['admitted_gpu_count'] == (8 if parallel else 1), 'resources not safe')
    if parallel:
        require(complete.get('worker_exit_code') == 0
                and complete.get('all_eight_real_GPU_manifests_verified') is True
                and complete.get('world_size') == 8 and complete.get('source_count') == COUNT
                and complete.get('each_source_once') is True, 'eight-rank complete receipt missing')


def validate_selection(family, cfg, claim, manifest, identity, selected, protocol_ref, selection_ref):
    require(cfg.get('job') == 'test' and cfg.get('family') == family, 'not same-family final test')
    for key in ('training_launch', 'full_complete', 'resources_safe', 'checkpoint_selection',
                'checkpoint_marker', 'fingerprint'):
        require(cfg.get(key) == selected[key], 'test differs from pre-test final selection: ' + key)
    for key, expected in (('protocol', protocol_ref), ('final_selection', selection_ref)):
        require(cfg.get(key) == expected, 'config uses different ' + key)
    require(claim.get('family') == family and claim.get('out') == cfg['out']
            and claim.get('automatic_retry') is False and claim.get('test_used_for_selection') is False,
            'not original once-only test claim')
    for key in ('checkpoint_selection', 'checkpoint_marker', 'final_selection'):
        require(claim.get(key) == cfg[key], 'test claim uses different selection')
    require(manifest.get('family') == family and manifest.get('checkpoint_identity') == identity
            and identity.get('fingerprint') == selected['fingerprint']
            and identity.get('trainer_state', {}).get('path') == selected['checkpoint_marker']['path']
            and identity.get('trainer_state', {}).get('sha256') == selected['checkpoint_marker']['sha256']
            and manifest.get('checkpoint') == str(Path(selected['checkpoint_marker']['path']).parent)
            and identity.get('training_config', {}).get('family') == family,
            'manifest/checkpoint differs from frozen family selection')
    require(manifest.get('complete') is True and manifest.get('attempted') == COUNT
            and manifest.get('test_used_for_checkpoint_selection') is False
            and manifest.get('human_quality_accepted') is not True, 'incomplete/final-test selection used')


def validate_rows(records, sources, evaluator):
    ids = [source['id'] for source in sources]
    require(len(ids) == COUNT and len(set(ids)) == COUNT, 'frozen test must contain 500 unique IDs')
    require(len({source['source_group_id'] for source in sources}) == COUNT,
            'test source groups not unique; source-cluster bootstrap contract differs')
    require(len(records) == COUNT and len({r['id'] for r in records}) == COUNT,
            'test output must contain 500 unique IDs')
    require([r['id'] for r in records] == ids, 'test source order differs from frozen select_rows')
    for record, source in zip(records, sources):
        require(record.get('failure') is None, 'interface failure in final test')
        require(record.get('reference_target') == source['target_text']
                and record.get('reference_target_windows') == source['target_windows'], 'reference swapped')
        require(record.get('condition') == 'original' and record.get('audio_transform') == 'none'
                and record.get('audio_source_id') == source['id']
                and record.get('source_group_id') == source['source_group_id']
                and record.get('audio_source_group_id') == source['source_group_id']
                and record.get('audio_sha256') == source['audio_sha256']
                and record.get('audio_window_seconds') == source['audio_window_seconds']
                and record.get('audio_ref', {}).get('path') == source['audio_path']
                and record.get('audio_ref', {}).get('sha256') == source['audio_sha256']
                and 'teacher_forced' not in record and 'teacher_forced_failure' not in record,
                'source/audio-only binding differs from frozen test')
        require(record.get('metrics') == evaluator.text_metrics(record['raw_text']),
                'stored diagnostics differ from frozen text_metrics')


def channel_values(case, channel):
    if not channel.startswith('window_'):
        return [case['agreement'][channel]]
    key = channel[len('window_'):]
    return [window[key] for window in case['agreement']['windows']]


def summarize_cases(cases, metric):
    result = {channel: metric.aggregate([v for c in cases for v in channel_values(c, channel)])
              for channel in CHANNELS}
    stops = Counter(c['stop_reason'] for c in cases)
    full_predictions = Counter(c['prediction_text_sha256'] for c in cases)
    result['diagnostics'] = dict(source_count=len(cases), failure_count=0,
        unique_full_prediction_count=len(full_predictions),
        exact_full_prediction_duplicate_case_fraction=sum(n for n in full_predictions.values() if n > 1) / len(cases),
        full_prediction_exact_match_definition='SHA256 of raw UTF8 text, no normalization; all cases in count>1 groups / source_count',
        stop_reason_counts={s: stops[s] for s in ('eos', 'native_stop', 'budget_truncation', 'unknown_stop')},
        empty_raw_count=sum(c['empty_raw'] for c in cases),
        valid_two_window_fraction=statistics.mean(c['diagnostics']['temporal']['matches_fixed_windows'] for c in cases),
        nonempty_two_window_fraction=statistics.mean(
            c['diagnostics']['temporal']['matches_fixed_windows'] and
            c['diagnostics']['temporal']['nonempty_window_text'] for c in cases),
        **{'mean_' + key: statistics.mean(c['diagnostics'][key] for c in cases) for key in
           ('word_count', 'exact_sentence_max_count', 'repeated_sentence_fraction',
            'repeated_4gram_fraction', 'distinct_1', 'action_category_count', 'body_category_count')})
    return result


def bootstrap_stat(cases, channel, indices, metric):
    values = [v for i in indices for v in channel_values(cases[i], channel)]
    return metric.aggregate(values)


def percentile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def paired_bootstrap(reports, metric, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    families = list(reports)
    ids = [c['id'] for c in reports[families[0]]]
    require(ids and len(set(ids)) == len(ids), 'bootstrap needs unique source IDs')
    require(all([c['id'] for c in reports[f]] == ids for f in families), 'bootstrap source pairing differs')
    # Per-source sufficient statistics keep both windows together. Equal fixed window
    # counts make these macro means and micro sums identical to aggregate(flattened).
    packed = {}
    draws = {f: {channel: {level: {k: [] for k in PRF} for level in ('macro', 'micro')}
                  for channel in CHANNELS} for f in families}
    for family in families:
        packed[family] = {}
        for channel in CHANNELS:
            packed[family][channel] = []
            for case in reports[family]:
                values = channel_values(case, channel)
                require(len(values) == (2 if channel.startswith('window_') else 1), 'expected fixed two windows')
                summary = metric.aggregate(values)
                packed[family][channel].append(tuple(summary['macro'][k] for k in PRF) +
                    tuple(summary['micro'][k] for k in ('overlap', 'predicted', 'reference')))
    rng = random.Random(seed)
    for _ in range(replicates):
        weights = list(Counter(rng.randrange(len(ids)) for _ in ids).items())
        for family in families:
            for channel in CHANNELS:
                values = packed[family][channel]
                sums = [sum(values[i][k] * weight for i, weight in weights) for k in range(6)]
                micro = metric.precision_recall_f1(*sums[3:])
                for k, key in enumerate(PRF):
                    draws[family][channel]['macro'][key].append(sums[k] / len(ids))
                    draws[family][channel]['micro'][key].append(micro[key])
    def interval(values):
        return dict(low=percentile(values, 0.025), high=percentile(values, 0.975))
    intervals = {f: {ch: {level: {k: interval(draws[f][ch][level][k]) for k in PRF}
                  for level in ('macro', 'micro')} for ch in CHANNELS} for f in families}
    differences = {}
    for a, b in combinations(families, 2):
        differences[a + '_minus_' + b] = {ch: {level: {k: interval([x-y for x, y in zip(
            draws[a][ch][level][k], draws[b][ch][level][k])]) for k in PRF}
            for level in ('macro', 'micro')} for ch in CHANNELS}
    return dict(replicates=replicates, seed=seed, unit='one source ID with both windows intact',
                paired_same_draws_across_families=True, interval='two-sided percentile 95%, linear interpolation',
                family_intervals=intervals, paired_differences=differences,
                not_a_quality_ranking_or_multiple_comparison_adjusted_significance_test=True)


def validate_family(inputs, handoff, family, sources, protocol, protocol_ref, selection, selection_ref, ev, serial):
    parallel = family != 'music_flamingo'
    sub = handoff / ('parallel_final_test_v2' if parallel else 'manual_recovery_v1')
    queue, out = sub / (family + '_test_queue'), sub / (family + '_test_out')
    require(not (queue / 'failure.json').exists() and not (queue / 'failure.json').is_symlink(),
            'manager recorded technical failure')
    complete = inputs.json(queue / 'eval.complete.json')
    exited = inputs.json(queue / 'test.exit.json')
    safe = inputs.json(queue / 'resources_safe.json')
    validate_completion(complete, exited, safe, parallel)
    manifest = inputs.json(out / 'manifest.json', complete['manifest'])
    raw = inputs.read(out / 'raw.jsonl', complete['raw'])
    require(raw.endswith(b'\n') and all(line.strip() for line in raw.splitlines()), 'partial/empty raw line')
    records = [json.loads(line) for line in raw.splitlines()]
    integrity = inputs.json(queue / 'integrity_receipt.json')
    cfg = inputs.json(sub / (family + '_test.run_config.json'),
                      dict(sha256=integrity['config_sha256']))
    require(cfg['out'] == str(out) and cfg['queue'] == str(queue), 'wrong first-test queue/output')
    require(cfg['gpus'] == (list(range(8)) if parallel else [0]), 'wrong GPU scope')
    admission = inputs.json(queue / 'admission.json')
    require(admission.get('config_sha256') == integrity['config_sha256'], 'admission config SHA differs')
    for path, digest in cfg['source_sha256'].items():
        inputs.ref(dict(path=path, sha256=digest))
    for key in ('training_launch', 'full_complete', 'resources_safe', 'checkpoint_selection', 'checkpoint_marker'):
        inputs.ref(cfg[key])
    local_claim = inputs.json(queue / 'test_claim.json')
    claim_path = handoff / ('test_once_' + family + '.json')
    require(local_claim.get('path') == str(claim_path), 'not shared family once-only ledger')
    claim = inputs.json(claim_path)
    identity = integrity['checkpoint_identity']
    validate_selection(family, cfg, claim, manifest, identity, selection['families'][family], protocol_ref, selection_ref)
    run_config = inputs.json(identity['run_config']['path'], identity['run_config'])
    require(run_config == identity['training_config'], 'recorded checkpoint training config differs')
    best = inputs.json(cfg['checkpoint_selection']['path'], cfg['checkpoint_selection'])
    require(best.get('checkpoint') == manifest['checkpoint'], 'actual frozen best checkpoint differs')
    validate_rows(records, sources, ev)
    require(manifest.get('source_ids') == [s['id'] for s in sources], 'manifest source order differs')
    if parallel:
        dispatch = inputs.json(queue / 'dispatch.json', admission['dispatch'])
        require(dispatch.get('source_ids') == [s['id'] for s in sources]
                and dispatch.get('checkpoint_identity') == identity
                and dispatch.get('final_selection') == selection_ref
                and dispatch.get('protocol') == protocol_ref
                and dispatch.get('config', {}).get('sha256') == integrity['config_sha256']
                and dispatch.get('out') == str(out)
                and dispatch.get('source_refs') == integrity.get('source_refs'), 'dispatch differs from first frozen test')
        require(claim.get('config_sha256') == integrity['config_sha256'], 'claim config SHA differs')
        for ref in dispatch['source_refs'].values():
            inputs.ref(ref)
        manager = load_pinned(inputs, '_final_parallel_manager', dispatch['source_refs']['manager']['path'])
        producer = load_pinned(inputs, '_final_parallel_producer', dispatch['source_refs']['producer']['path'])
        require(integrity['manager_sha256'] == PINS['music_planner_parallel_test_manager.py'], 'wrong actual manager SHA')
        evidence = manager.check_worker_outputs(cfg, dispatch, producer, ev)
        for receipt in manifest['rank_receipts']:
            for key in ('manifest', 'raw', 'native_source_loaded'):
                inputs.ref(receipt[key])
        for rank in range(8):
            for phase in ('start', 'loaded', 'exit'):
                inputs.json(out / ('%s-rank%d.json' % (phase, rank)))
        inputs.ref(manifest['summary'])
        evidence_scope = dict(world_size=8, rank_count=8, actual_forward_count=manifest['actual_forward_count'],
            finite_native_logits_forward_count=manifest['finite_native_logits_forward_count'],
            native_logits_all_finite=manifest['native_logits_all_finite'],
            revalidated_using='actual frozen v2 manager check_worker_outputs + producer collect_shards')
    else:
        require(integrity['manager_sha256'] == PINS['planner_eval_manager_v3.py'], 'wrong actual serial manager SHA')
        require(integrity.get('protocol') == protocol_ref, 'serial integrity protocol differs')
        evidence = serial.check_worker_outputs(cfg, Path(manifest['checkpoint']), identity, protocol, ev)
        evidence_scope = dict(world_size=1, physical_gpu=0, native_token_and_stop_evidence_revalidated=True,
            actual_forward_count=None, finite_native_logits_forward_count=None,
            hook_evidence_not_recorded_by_this_frozen_serial_producer=True,
            revalidated_using='actual frozen v3 manager check_worker_outputs; no invented hooks')
    require(all(complete.get(k) == value for k, value in evidence.items()), 'completion evidence differs from actual outputs')
    require(manifest.get('actual_generated_tokens') == sum(r['generated_tokens'] for r in records),
            'aggregate generated token evidence differs')
    return records, evidence_scope


def build_report(handoff, protocol_sha, selection_sha):
    handoff = Path(handoff)
    inputs = Inputs()
    protocol_ref = dict(path=str(handoff / 'protocol.json'), sha256=protocol_sha)
    selection_ref = dict(path=str(handoff / 'final_selection.json'), sha256=selection_sha)
    protocol = inputs.json(protocol_ref['path'], protocol_ref)
    selection = inputs.json(selection_ref['path'], selection_ref)
    require(protocol.get('schema') == 'music_planner_fixed3_protocol_v1'
            and protocol['test']['limit'] == COUNT and protocol['test']['seed'] == 42
            and protocol['test']['final_evaluations_per_family'] == 1
            and protocol['test']['consumed_for_model_selection'] is False, 'wrong final test500 protocol')
    require(selection.get('schema') == 'music-planner-final-selection-v1'
            and selection.get('selection_frozen_before_test') is True and selection.get('test_used') is False
            and selection.get('protocol') == protocol_ref and selection.get('data') == protocol['data']
            and selection.get('decoding') == protocol['decoding']
            and set(selection.get('families', {})) == set(FAMILIES), 'not identical pre-test frozen three-family selection')
    runtime = handoff.parent / 'runtime'
    evaluator_path = runtime / 'source_v11/music_planner_eval.py'
    require(any(ref['path'] == str(evaluator_path) and ref['sha256'] == PINS['music_planner_eval.py']
                for ref in protocol['code']), 'protocol does not pin actual frozen evaluator')
    ev = load_pinned(inputs, '_final_frozen_eval', evaluator_path)
    serial = load_pinned(inputs, '_final_serial_manager', runtime / 'planner_eval_manager_v3.py')
    metric = load_pinned(inputs, '_final_reference_f1', Path(__file__).with_name('music_planner_reference_f1.py'))
    data_summary = inputs.json(protocol['data']['summary']['path'], protocol['data']['summary'])
    exclusion_ref = data_summary['training_exclusion_receipt_ref']
    exclusions = inputs.json(exclusion_ref['path'], exclusion_ref)
    parent_ref = exclusions['parent_summary_ref']
    parent = inputs.json(parent_ref['path'], parent_ref)
    inputs.ref(parent['split_artifacts']['train'])
    for ref in protocol['data']['splits'].values():
        inputs.ref(ref)
    ev.validate_data(protocol['data']['path'])
    test_ref = protocol['data']['splits']['test']
    source_rows = [json.loads(line) for line in inputs.read(test_ref['path'], test_ref).splitlines() if line.strip()]
    require(len(source_rows) == COUNT and len({r['id'] for r in source_rows}) == COUNT, 'not unique frozen test500')
    sources = ev.select_rows(source_rows, 'test', limit=COUNT, seed=42)
    cases, summaries, technical = {}, {}, {}
    for family in FAMILIES:
        records, technical[family] = validate_family(inputs, handoff, family, sources, protocol,
            protocol_ref, selection, selection_ref, ev, serial)
        cases[family] = [dict(id=r['id'], source_group_id=s['source_group_id'],
            source_test_file=test_ref, source_row_sha256=hashlib.sha256(json.dumps(s, sort_keys=True,
                ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
            reference_text_sha256=hashlib.sha256(s['target_text'].encode()).hexdigest(),
            prediction_text_sha256=hashlib.sha256(r['raw_text'].encode()).hexdigest(),
            agreement=metric.compare_text(r['raw_text'], s['target_text'], ev),
            diagnostics=ev.text_metrics(r['raw_text']), stop_reason=r['stop_reason'],
            generated_tokens=r['generated_tokens'], empty_raw=not r['raw_text'].strip())
            for r, s in zip(records, sources)]
        summaries[family] = summarize_cases(cases[family], metric)
        summaries[family]['frozen_text_summary'] = ev.summarize(records)
        for key in ('corpus_distinct_1', 'corpus_distinct_2'):
            summaries[family]['diagnostics'][key] = summaries[family]['frozen_text_summary'][key]
    bootstrap = paired_bootstrap(cases, metric)
    inputs.read(Path(__file__).resolve())
    inputs.recheck()
    report = dict(schema='music-planner-first-final-test500-text-report-v1',
        scope='first once-only test500; original audio only; same sources and frozen order for all three families',
        source_ids=[r['id'] for r in sources], reports=summaries, cases=cases,
        technical_evidence=technical, bootstrap=bootstrap,
        formulas=dict(reused_compare_text_and_aggregate=True, token_f1='multiset English word/CJK character overlap; timestamps removed',
            action_body_f1='frozen per-window regex category sets; invalid windows get zero overlap',
            rougeL='sequence LCS P/R harmonic F1, not ROUGE-Lsum', zero_division=0),
        single_weak_reference_not_unique_valid_choreography=True,
        metrics_are_weak_reference_agreement_not_creative_choreography_accuracy=True,
        dance_style_f1=None, independent_dance_style_ground_truth_available=False,
        music_or_H3_human_review_performed=False, human_quality_accepted=False,
        test_used_only_for_final_evaluation=True, test_used_for_metric_or_threshold_selection=False,
        checkpoint_reselection_performed=False, training_or_generation_performed=False,
        automatic_quality_ranking=False, winner=None,
        input_sources_sha256=list(inputs.refs.values()))
    return report, inputs


def save_report(path, report, inputs):
    path = Path(path)
    require(path.is_absolute() and not any(p.is_symlink() for p in path.parents), 'absolute nonsymlink output required')
    require(str(path) not in inputs.refs, 'report cannot replace an input')
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    inputs.recheck()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', required=True, type=Path)
    parser.add_argument('--protocol-sha', required=True)
    parser.add_argument('--selection-sha', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    report, inputs = build_report(args.handoff, args.protocol_sha, args.selection_sha)
    save_report(args.out, report, inputs)
    print(json.dumps(dict(out=str(args.out), weak_reference_agreement_only=True, winner=None)))


if __name__ == '__main__':
    main()
