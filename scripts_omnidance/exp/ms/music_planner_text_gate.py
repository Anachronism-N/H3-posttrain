"""Frozen val32 text gates; never use final test outputs to set these thresholds."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

THRESHOLDS = dict(min_normal_stop_fraction=0.95, min_valid_two_window_fraction=0.95,
                  max_severe_sentence_repeat_fraction=0.05)
CONDITIONS = ('original', 'mismatched', 'silence')


def ref(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def analyze_records(records, ids, evaluator, family):
    expected = [(sid, condition) for sid in ids for condition in CONDITIONS]
    actual = [(row['id'], row['condition']) for row in records]
    membership = actual == expected
    errors = sum(bool(row.get('failure')) for row in records)
    empty = sum(not isinstance(row.get('raw_text'), str) or not row['raw_text'].strip() for row in records)
    invalid_stops, nonfinite, finite_observed = 0, 0, 0
    subsets, paired = defaultdict(list), defaultdict(dict)
    for row in records:
        text = row.get('raw_text', '')
        native = row.get('native_output', {})
        tokens, eos = row.get('new_token_ids', []), row.get('eos_token_ids', [])
        interface_ok = (isinstance(tokens, list) and isinstance(eos, list) and bool(tokens) and
                        all(type(t) is int and t >= 0 for t in tokens + eos) and
                        len(tokens) <= 2048 and native.get('new_token_ids') == tokens and
                        native.get('eos_token_ids') == eos and native.get('raw_text') == text)
        stop = 'interface_failure'
        if interface_ok:
            stop = evaluator.classify_stop(tokens, eos, 2048, native.get('native_stop', False))
        proper_boundary = interface_ok and not any(token in eos for token in tokens[:-1])
        if stop != row.get('stop_reason') or not proper_boundary:
            invalid_stops += 1
        observed = native.get('native_logits_all_finite')
        finite_observed += int(observed is True)
        if observed is False or (family == 'moss' and (observed is not True or
                type(native.get('actual_forward_count')) is not int or native['actual_forward_count'] <= 0 or
                native.get('finite_native_logits_forward_count') != native['actual_forward_count'])):
            nonfinite += 1
        metrics = evaluator.text_metrics(text) if isinstance(text, str) else evaluator.text_metrics('')
        item = dict(id=row['id'], stop=stop, boundary=proper_boundary, metrics=metrics, raw_text=text)
        subsets[row['condition']].append(item)
        paired[row['id']][row['condition']] = text
    diagnostics, checks = {}, {}
    for condition in CONDITIONS:
        items = subsets[condition]
        count = len(items)
        fraction = lambda flag: sum(bool(flag(item)) for item in items) / count if count else None
        normal = fraction(lambda item: item['boundary'] and item['stop'] in ('eos', 'native_stop'))
        windows = fraction(lambda item: item['metrics']['temporal']['matches_fixed_windows'] and
                           item['metrics']['temporal']['nonempty_window_text'])
        severe = fraction(lambda item: item['metrics']['exact_sentence_max_count'] >= 3)
        diagnostics[condition] = dict(sources=count, normal_stop_fraction=normal,
            valid_two_window_fraction=windows, severe_exact_sentence_repeat_fraction=severe,
            stop_counts=dict(Counter(item['stop'] for item in items)),
            mean_words=statistics.mean(item['metrics']['word_count'] for item in items) if items else None,
            mean_repeated_4gram_fraction=statistics.mean(item['metrics']['repeated_4gram_fraction'] for item in items) if items else None,
            mean_repeated_sentence_fraction=statistics.mean(item['metrics']['repeated_sentence_fraction'] for item in items) if items else None,
            mean_action_categories=statistics.mean(item['metrics']['action_category_count'] for item in items) if items else None,
            mean_body_categories=statistics.mean(item['metrics']['body_category_count'] for item in items) if items else None)
        checks[condition] = dict(normal_stop=normal is not None and normal >= THRESHOLDS['min_normal_stop_fraction'],
            two_windows=windows is not None and windows >= THRESHOLDS['min_valid_two_window_fraction'],
            severe_repeat=severe is not None and severe <= THRESHOLDS['max_severe_sentence_repeat_fraction'])
    originals = [item['raw_text'] for item in subsets['original']]
    dependence = {}
    for condition in ('mismatched', 'silence'):
        valid = [group for group in paired.values() if 'original' in group and condition in group]
        dependence[condition] = dict(pairs=len(valid), raw_identical_fraction=
            sum(group['original'] == group[condition] for group in valid) / len(valid) if valid else None)
    passed = membership and not (errors or empty or invalid_stops or nonfinite) and all(
        all(check.values()) for check in checks.values())
    return dict(expected_records=len(expected), actual_records=len(records), exact_membership=membership,
        interface_failures=errors, empty_outputs=empty, invalid_stop_or_earlier_EOS=invalid_stops,
        recorded_nonfinite_or_missing_required_MOSS_proof=nonfinite,
        native_finite_proof_records=finite_observed, native_finite_proof_coverage=finite_observed / len(records) if records else None,
        by_condition=diagnostics, gate_checks=checks, text_gate_passed=passed,
        original_exact_duplicate_fraction=1 - len(set(originals)) / len(originals) if originals else None,
        music_dependency=dependence, music_dependency_change_is_not_synchronization_score=True,
        dance_style_accuracy=None, dance_style_ground_truth_available=False,
        semantic_choreography_quality_accepted=False, automatic_model_ranking=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['thresholds'] != THRESHOLDS or config.get('test_used_for_threshold_selection') is not False:
        raise ValueError('Only the preregistered val text gates are accepted')
    source = config['frozen_eval_source']
    if ref(source['path'])['sha256'] != source['sha256']:
        raise ValueError('Frozen text metric code changed')
    spec = importlib.util.spec_from_file_location('frozen_val_text_metrics', source['path'])
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    protocol = json.loads(Path(config['protocol']['path']).read_text())
    if ref(config['protocol']['path'])['sha256'] != config['protocol']['sha256']:
        raise ValueError('Counterfactual panel protocol changed')
    reports = {}
    for family, out in config['panels'].items():
        out = Path(out)
        if not (out / 'manifest.json').exists() or not (out / 'raw.jsonl').exists():
            reports[family] = dict(status='pending', text_gate_passed=None)
            continue
        manifest = json.loads((out / 'manifest.json').read_text())
        if manifest.get('mode') != 'val' or manifest['family'] != family or manifest['split']['sha256'] != protocol['data']['splits']['val']['sha256']:
            raise ValueError('Text gates accept the declared val panel only, never test')
        raw = (out / 'raw.jsonl').read_bytes()
        try:
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        except ValueError:
            reports[family] = dict(status='pending_partial_write', text_gate_passed=None)
            continue
        analysis = analyze_records(records, protocol['val32']['ids'], evaluator, family)
        donors = {(item['id'], item['condition']): item for item in protocol['val32']['donor_mapping']}
        donor_matches = all((row['id'], row['condition']) in donors and
            all(row.get(key) == donors[(row['id'], row['condition'])][key]
                for key in ('source_group_id', 'audio_source_group_id', 'audio_source_id', 'audio_sha256')) and
            row.get('audio_ref', {}).get('sha256') == donors[(row['id'], row['condition'])]['audio_sha256']
            for row in records)
        analysis['exact_frozen_audio_donor_mapping'] = donor_matches
        analysis['text_gate_passed'] = analysis['text_gate_passed'] and donor_matches
        queue = Path(config['panel_queues'][family])
        complete_path, safe_path = queue / 'eval.complete.json', queue / 'resources_safe.json'
        exit_path = queue / 'val_panel.exit.json'
        collector_verified = False
        if all(path.is_file() for path in (complete_path, safe_path, exit_path)):
            complete = json.loads(complete_path.read_text())
            safe = json.loads(safe_path.read_text())
            exit_ = json.loads(exit_path.read_text())
            collector_verified = (complete.get('technical_collection_complete') is True and
                complete.get('manifest') == ref(out / 'manifest.json') and complete.get('raw') == ref(out / 'raw.jsonl') and
                all(safe.get(key) is True for key in ('all_holders_safe', 'locks_released', 'leases_restored')) and
                type(exit_.get('exit_code')) is int and exit_['exit_code'] == 0)
        analysis['collector_completion_and_resource_safe_verified'] = collector_verified
        analysis['status'] = 'complete' if manifest.get('complete') is True and len(records) == 96 and collector_verified else 'pending'
        if analysis['status'] != 'complete':
            analysis['text_gate_passed'] = None
        analysis['source_refs'] = dict(manifest=ref(out / 'manifest.json'), raw=ref(out / 'raw.jsonl'))
        reports[family] = analysis
    report = dict(schema='music-planner-val-text-gates-v1', thresholds=THRESHOLDS,
        gate_scope='complete native outputs, assistant EOS, two windows and severe exact sentence repetition only',
        severe_repeat_definition='same normalized sentence occurs at least three times within one caption',
        music_semantic_synchronization_accuracy_evaluated=False,
        silence_corpus_duplicate_not_used_as_diversity_failure=True,
        gate_config=ref(args.config), code=ref(__file__), reports=reports,
        test_used_for_threshold_selection=False, independent_dance_style_ground_truth=False,
        H3_dispatch_deferred_until_text_review=True, winner=None, human_quality_accepted=False)
    with args.out.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write(chr(10))
    print(json.dumps(dict(out=str(args.out), statuses={family: row['status'] for family, row in reports.items()},
                         text_gate={family: row['text_gate_passed'] for family, row in reports.items()}, winner=None)))


if __name__ == '__main__':
    main()
