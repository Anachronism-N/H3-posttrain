"""Frozen E2 reference-text metrics; agreement is not choreography or musical quality."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean


def inputs(root):
    manifest = json.loads((root / 'test384/manifest.public.json').read_text())
    references = {r['case_id']: r for r in json.loads((root / 'test384/references.private.json').read_text())['records']}
    entries = json.loads((root / 'test384-review-6methods/private/mapping.json').read_text())['entries']
    case_ids = {r['case_id'] for r in manifest['cases']}
    if len(case_ids) != 384 or set(references) != case_ids or len(entries) != 2304:
        raise ValueError('Frozen384 reference inventory differs')
    rows, seen = [], set()
    for entry in entries:
        item = entry['true_item']; key = (item['case_id'], item['method_id'])
        if key in seen or key[0] not in references:
            raise ValueError('Duplicated or unbound metric row')
        seen.add(key); ref = references[key[0]]
        if item['source_group_id'] != ref['source_group_id']:
            raise ValueError('Cross-source reference binding')
        text = item['text']
        if hashlib.sha256(text.encode()).hexdigest() != item['text_sha256']:
            raise ValueError('Candidate text changed')
        rows.append(dict(case_id=key[0], method_id=key[1], source_group_id=item['source_group_id'],
            item_sha256=entry['blind_item_sha256'], text=text, reference=ref['action_reference'],
            generation_status=item['generation_status']))
    if len({r['method_id'] for r in rows}) != 6:
        raise ValueError('Require all six frozen methods')
    return rows


def compute(rows, *, meteor=False):
    from sacrebleu.metrics import BLEU
    from rouge_score.rouge_scorer import RougeScorer
    sentence_bleu = BLEU(lowercase=False, tokenize='13a', smooth_method='exp', effective_order=True)
    rouge = RougeScorer(['rougeL'], use_stemmer=True)
    if meteor:
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import TreebankWordTokenizer
        tokenize = TreebankWordTokenizer().tokenize
        from nltk.corpus import wordnet
        wordnet.ensure_loaded()
    for row in rows:
        row['metrics'] = {}
        if row['generation_status'] != 'ok':
            row['metric_status'] = 'generation_not_evaluable'; continue
        candidate, reference = row['text'], row['reference']
        if not reference.strip():
            raise ValueError('Empty frozen reference')
        row['metric_status'] = 'computed'
        value = rouge.score(reference, candidate)['rougeL']
        row['metrics'].update(sentence_bleu4=sentence_bleu.sentence_score(candidate, [reference]).score,
            rougeL_precision=value.precision, rougeL_recall=value.recall, rougeL_f1=value.fmeasure)
        if meteor:
            row['metrics']['meteor'] = meteor_score([tokenize(reference.lower())], tokenize(candidate.lower()))
    return rows


def embedding(rows, model, device):
    import torch
    from transformers import AutoTokenizer
    from bert_score import score
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=False)
    eligible = []
    for row in rows:
        if row['generation_status'] != 'ok': continue
        lengths = [len(tokenizer.encode(row[k], add_special_tokens=True, truncation=False)) for k in ('text', 'reference')]
        row['bert_input_token_lengths'] = lengths
        if max(lengths) > 512:
            row['bert_status'] = 'context_overflow_not_truncated'; continue
        row['bert_status'] = 'computed'; eligible.append(row)
    if eligible:
        p, r, f = score([x['text'] for x in eligible], [x['reference'] for x in eligible],
            model_type=str(model), num_layers=17, lang='en', batch_size=16, nthreads=2,
            idf=False, rescale_with_baseline=False, device=device, use_fast_tokenizer=False)
        for row, precision, recall, f1 in zip(eligible, p.tolist(), r.tolist(), f.tolist()):
            row['metrics'].update(bertscore_precision=precision, bertscore_recall=recall, bertscore_f1=f1)
    return rows


def summarize(rows):
    from sacrebleu.metrics import BLEU
    out = {}
    for method in sorted({r['method_id'] for r in rows}):
        group = [r for r in rows if r['method_id'] == method]
        valid = [r for r in group if r['metric_status'] == 'computed']
        corpus = BLEU(lowercase=False, tokenize='13a', smooth_method='exp', effective_order=False)
        result = dict(registered=len(group), evaluable=len(valid), failed=len(group)-len(valid),
            corpus_bleu4=corpus.corpus_score([r['text'] for r in valid], [[r['reference'] for r in valid]]).score,
            bleu_signature=str(corpus.get_signature()), metrics={})
        for name in sorted({k for r in valid for k in r['metrics']}):
            values = [r['metrics'][name] for r in valid if name in r['metrics']]
            result['metrics'][name] = dict(n=len(values), mean=mean(values),
                missing_registered=len(group)-len(values))
        out[method] = result
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--meteor', action='store_true')
    p.add_argument('--embedding-model', type=Path); p.add_argument('--device', default='cpu')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    rows = compute(inputs(a.root), meteor=a.meteor)
    if a.embedding_model: rows = embedding(rows, a.embedding_model, a.device)
    versions = {n: importlib.metadata.version(n) for n in ('sacrebleu','rouge-score','nltk')}
    receipt = dict(scope='single sol61 action_reference agreement only; not musical fit or video truth',
        metric_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={str(rel):hashlib.sha256((a.root/rel).read_bytes()).hexdigest() for rel in (
            'test384/references.private.json','test384-review-6methods/private/mapping.json')},
        versions=versions, meteor_wordnet=a.meteor, bert_model=str(a.embedding_model),
        bert_num_layers=17, bert_idf=False, bert_rescale=False, silent_truncation=False,
        failed_generations_not_imputed=True, human_validation=False)
    with (a.out/'records.jsonl').open('x') as f:
        for row in rows:
            output = {k:v for k,v in row.items() if k not in ('text','reference')}
            f.write(json.dumps(output,allow_nan=False)+'\n')
    (a.out/'summary.json').write_text(json.dumps(dict(receipt=receipt,methods=summarize(rows)),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(completed=True,registered=len(rows),output=str(a.out))),flush=True)


if __name__ == '__main__': main()
