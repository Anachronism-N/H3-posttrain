"""Weak-reference agreement metrics, not correctness of creative choreography."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


def precision_recall_f1(overlap, predicted, reference):
    precision = overlap / predicted if predicted else 0.0
    recall = overlap / reference if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=precision, recall=recall, f1=f1,
                overlap=overlap, predicted=predicted, reference=reference)


def multiset_f1(prediction, reference):
    predicted, gold = Counter(prediction), Counter(reference)
    return precision_recall_f1(sum((predicted & gold).values()), sum(predicted.values()), sum(gold.values()))


def set_f1(prediction, reference):
    predicted, gold = set(prediction), set(reference)
    return precision_recall_f1(len(predicted & gold), len(predicted), len(gold))


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]


def lcs_length(prediction, reference):
    previous = [0] * (len(reference) + 1)
    for token in prediction:
        current = [0]
        for i, gold in enumerate(reference):
            current.append(previous[i] + 1 if token == gold else max(previous[i + 1], current[-1]))
        previous = current
    return previous[-1]


def compare_text(prediction, reference, evaluator):
    # Formatting/time labels cannot inflate the content overlap metrics.
    predicted = evaluator.word_units(evaluator.MARKER.sub('', prediction))
    gold = evaluator.word_units(evaluator.MARKER.sub('', reference))
    result = dict(rouge1=multiset_f1(predicted, gold),
                  rouge2=multiset_f1(ngrams(predicted, 2), ngrams(gold, 2)),
                  rougeL=precision_recall_f1(lcs_length(predicted, gold), len(predicted), len(gold)))
    p = evaluator.parse_windows(prediction)
    r = evaluator.parse_windows(reference)
    if not r['matches_fixed_windows']:
        raise ValueError('Weak reference windows differ from the frozen benchmark')
    windows = []
    for index, source in enumerate(r['windows']):
        text = p['windows'][index]['text'] if p['matches_fixed_windows'] else ''
        pm, rm = evaluator.text_metrics(text), evaluator.text_metrics(source['text'])
        words, targets = evaluator.word_units(text), evaluator.word_units(source['text'])
        windows.append(dict(index=index, predicted_window_valid=p['matches_fixed_windows'],
            token=multiset_f1(words, targets),
            action_categories=set_f1(pm['action_categories'], rm['action_categories']),
            body_categories=set_f1(pm['body_categories'], rm['body_categories'])))
    result['windows'] = windows
    return result


def aggregate(values):
    if not values:
        return dict(count=0, macro=None, micro=None)
    return dict(count=len(values),
        macro={key: statistics.mean(value[key] for value in values) for key in ('precision', 'recall', 'f1')},
        micro=precision_recall_f1(sum(v['overlap'] for v in values), sum(v['predicted'] for v in values),
                                  sum(v['reference'] for v in values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source = cfg['frozen_eval_source']
    if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:
        raise ValueError('Metric code differs from the frozen protocol')
    spec = importlib.util.spec_from_file_location('weak_reference_frozen_eval', source['path'])
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    protocol_path = Path(cfg['protocol']['path'])
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != cfg['protocol']['sha256']:
        raise ValueError('Panel protocol changed')
    protocol = json.loads(protocol_path.read_text())
    val = Path(protocol['data']['splits']['val']['path'])
    if hashlib.sha256(val.read_bytes()).hexdigest() != protocol['data']['splits']['val']['sha256']:
        raise ValueError('Weak val reference bytes changed')
    references = {row['id']: row['target_text'] for row in map(json.loads, val.read_text().splitlines())}
    reports = {}
    for family, path in cfg['panels'].items():
        path = Path(path)
        manifest = json.loads((path / 'manifest.json').read_text())
        if (manifest['family'] != family or manifest['mode'] != 'val' or manifest['complete'] is not True):
            raise ValueError('Only the complete same val panel is accepted, never final-test tuning')
        rows = [json.loads(line) for line in (path / 'raw.jsonl').read_text().splitlines() if line.strip()]
        expected = [(sid, condition) for sid in protocol['val32']['ids'] for condition in ('original', 'mismatched', 'silence')]
        if [(row['id'], row['condition']) for row in rows] != expected:
            raise ValueError('Panel source/condition membership differs')
        buckets = defaultdict(list)
        for row in rows:
            if row.get('failure') or row['reference_target'] != references[row['id']]:
                raise ValueError('Interface failure or reference swapped')
            buckets[row['condition']].append(compare_text(row['raw_text'], references[row['id']], evaluator))
        reports[family] = {}
        for condition, values in buckets.items():
            reports[family][condition] = dict(
                rouge1=aggregate([v['rouge1'] for v in values]),
                rouge2=aggregate([v['rouge2'] for v in values]),
                rougeL=aggregate([v['rougeL'] for v in values]),
                window_token=aggregate([window['token'] for v in values for window in v['windows']]),
                window_action_categories=aggregate([window['action_categories'] for v in values for window in v['windows']]),
                window_body_categories=aggregate([window['body_categories'] for v in values for window in v['windows']]))
    report = dict(schema='music-planner-weak-reference-F1-v1', scope='complete val32 by condition, each matched to original weak source reference',
        reports=reports, formulas=dict(token_f1='multiset English word/CJK-character overlap; formatting timestamps removed',
            action_body_f1='frozen regex category sets per expected window; invalid predicted windows get zero overlap',
            rougeL='sequence LCS P/R harmonic F1, not ROUGE-Lsum', zero_division=0),
        reference_is_one_observed_choreography_not_only_valid_solution=True,
        metrics_are_weak_reference_agreement_not_choreography_accuracy=True,
        mismatch_silence_reference_remains_original_source=True,
        dance_style_f1=None, independent_dance_style_ground_truth_available=False,
        test_used_for_metric_or_threshold_selection=False, winner=None, human_quality_accepted=False,
        code=dict(path=str(Path(__file__)), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    with args.out.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write(chr(10))
    print(json.dumps({'out': str(args.out), 'models': list(reports), 'weak_reference_metrics_only': True}))


if __name__ == '__main__':
    main()
