"""Teacher-free validation generation audit for separately trained planner protocols."""
import argparse
import json
from pathlib import Path
import time

import torch

from scripts_omnidance.exp.ms import music_planner_cot as cot

COMPARISON_KEYS = ('seed', 'rank', 'alpha', 'lora_lr', 'projector_lr', 'accum', 'epochs',
                   'dtype', 'grad_clip', 'warmup_updates', 'max_length', 'gradient_checkpointing')


def training_contract(config):
    return dict(settings={key: config[key] for key in COMPARISON_KEYS},
                initialization='fresh-local-base-and-LoRA-no-warm-start', base_identity=config['model'])


def audio_rows(path):
    path = Path(path).resolve()
    rows, seen = [], set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            source = json.loads(line)
            if source.get('split') != 'val' or source.get('id') in seen:
                raise ValueError('Audit requires unique validation sources; train/test cannot tune this audit')
            row = {key: source[key] for key in ('id', 'split', 'audio_path', 'audio_sha256', 'audio_window_seconds')}
            audio = Path(row['audio_path']).expanduser()
            row['audio_path'] = str((path.parent / audio).resolve()) if not audio.is_absolute() else str(audio.resolve())
            if not isinstance(row['id'], str) or not row['id'].strip():
                raise ValueError('Audit source ID must be nonempty')
            seen.add(row['id'])
            rows.append(row)
    if not rows:
        raise ValueError('Empty validation audit')
    return rows


def _source_panel(rows):
    return sorted([dict(id=row['id'], audio_sha256=row['audio_sha256'],
                        audio_window_seconds=row['audio_window_seconds']) for row in rows], key=lambda row: row['id'])


def summarize(records):
    valid = [row for row in records if row['generation']['caption_valid']]
    return dict(sources=len(records), valid_final_captions=len(valid),
        invalid_final_captions=len(records) - len(valid),
        input_tokens=sum(row['generation']['input_tokens'] for row in records),
        raw_generated_tokens=sum(len(row['generation']['new_token_ids']) for row in records),
        raw_characters=sum(len(row['generation']['raw_text']) for row in records),
        final_caption_characters=sum(len(row['generation']['caption']) for row in valid),
        wall_seconds=sum(row['wall_seconds'] for row in records),
        stop_reasons={reason: sum(row['generation']['stop_reason'] == reason for row in records)
                      for reason in sorted({row['generation']['stop_reason'] for row in records})},
        interpretation='Format, length and repetition diagnostics only; no choreography quality score')


def generate_audit(adapter, rows, *, max_new_tokens, checkpoint_identity, training_data, input_ref,
                   base_identity, comparison_contract=None, source_exposures=None):
    records = []
    for row in rows:
        audio = {key: row[key] for key in ('audio_path', 'audio_sha256', 'audio_window_seconds')}
        started = time.perf_counter()
        generated = adapter.generate_caption(audio, max_new_tokens=max_new_tokens)
        elapsed = time.perf_counter() - started
        records.append(dict(source=dict(id=row['id'], split='val', **audio), generation=generated,
            wall_seconds=elapsed, raw_text_metrics=cot.evaluation.text_metrics(generated['raw_text']),
            final_caption_metrics=(cot.evaluation.text_metrics(generated['caption'])
                                   if generated['caption_valid'] else None)))
    report = dict(schema='music-planner-teacher-free-audit-v1', mode=adapter.mode,
        family=adapter.family, base_identity=base_identity,
        require_shared_annotations=adapter.require_shared_annotations,
        checkpoint_identity=checkpoint_identity, training_data=training_data, input_ref=input_ref,
        training_comparison_contract=comparison_contract, source_exposures=source_exposures,
        panel=_source_panel(rows), max_new_tokens=max_new_tokens, max_length=adapter.max_length,
        generation_policy=('native-audio-only-prefix' if adapter.family == 'omni_captioner'
                           else 'native-audio-plus-task-text-prefix') + '-greedy-first-EOS-no-gold-intermediate-input',
        audit_source_sha256=cot.base.file_sha(__file__), records=records, summary=summarize(records),
        teacher_forced_nll_used=False, test_used=False,
        comparison_scope='Separate learned protocols require separate checkpoints; no system-prompt mode switch')
    report['audit_sha256'] = cot.base.object_sha(report)
    return report


def validate_audit(report):
    if (report.get('schema') != 'music-planner-teacher-free-audit-v1' or
            report.get('audit_sha256') != cot.base.object_sha({k: v for k, v in report.items() if k != 'audit_sha256'}) or
            report.get('teacher_forced_nll_used') is not False or report.get('test_used') is not False):
        raise ValueError('Invalid or changed teacher-free audit')
    if report.get('mode') not in cot.MODES or not report.get('records'):
        raise ValueError('Audit mode/records invalid')
    sources = []
    for record in report['records']:
        generated = record['generation']
        if record['source'].get('split') != 'val' or generated.get('mode') != report['mode']:
            raise ValueError('Audit source split or output protocol differs')
        if generated['caption_valid']:
            cot.validate_completion_tokens(generated, report['max_new_tokens'])
            parsed = (cot.extract_caption(generated['raw_text']) if report['mode'] == 'short-plan'
                      else dict(caption=cot._text(generated['raw_text'], 'caption')))
            timing = cot.evaluation.parse_windows(parsed['caption'])
            if (generated['caption'] != parsed['caption'] or not timing['matches_fixed_windows'] or
                    not timing['nonempty_window_text']):
                raise ValueError('Audit final caption differs from raw generation')
        sources.append(dict(id=record['source']['id'], audio_sha256=record['source']['audio_sha256'],
                            audio_window_seconds=record['source']['audio_window_seconds']))
    if (len({row['id'] for row in sources}) != len(sources) or _source_panel(sources) != report['panel'] or
            report['summary'] != summarize(report['records'])):
        raise ValueError('Audit panel or summary changed')
    return report


def compare_audits(left, right, require_shared_annotations=False):
    left, right = validate_audit(left), validate_audit(right)
    if {left['mode'], right['mode']} != set(cot.MODES):
        raise ValueError('Compare separately trained direct-caption and short-plan checkpoints')
    if (left['panel'] != right['panel'] or left['max_new_tokens'] != right['max_new_tokens'] or
            left['max_length'] != right['max_length'] or left['family'] != right['family'] or
            left['base_identity'] != right['base_identity']):
        raise ValueError('Comparison requires identical validation audio panel and generation budgets')
    shared = (left['require_shared_annotations'] and right['require_shared_annotations'] and
              all(left['training_data'][split]['sha256'] == right['training_data'][split]['sha256']
                  for split in ('train', 'val')))
    matched_contract = (left.get('training_comparison_contract') is not None and
                        left['training_comparison_contract'] == right.get('training_comparison_contract'))
    matched_exposures = (type(left.get('source_exposures')) is int and left['source_exposures'] > 0 and
                         left['source_exposures'] == right.get('source_exposures'))
    controlled = bool(shared and matched_contract and matched_exposures)
    if require_shared_annotations and not controlled:
        raise ValueError('Controlled ablation requires the same reviewed manifests, training settings and actual source exposures')
    return dict(schema='music-planner-teacher-free-comparison-v1',
        audit_refs=[dict(mode=item['mode'], sha256=item['audit_sha256'],
                         checkpoint_identity=item['checkpoint_identity']) for item in (left, right)],
        shared_reviewed_training_manifests=bool(shared), matched_validation_panel_and_budgets=True,
        matched_training_contract=bool(matched_contract), matched_source_exposures=bool(matched_exposures),
        controlled_ablation=controlled,
        training_comparison={item['mode']: dict(contract=item.get('training_comparison_contract'),
            source_exposures=item.get('source_exposures')) for item in (left, right)},
        modes={item['mode']: item['summary'] for item in (left, right)}, test_used=False,
        teacher_forced_nll_used=False, quality_winner=None,
        interpretation='Use raw/final captions and independent audio review; these diagnostics cannot decide CoT quality')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('audit')
    run.add_argument('--checkpoint', required=True, type=Path)
    run.add_argument('--model', required=True, type=Path)
    run.add_argument('--family', choices=cot.FAMILIES, default='omni_captioner')
    run.add_argument('--mode', choices=cot.MODES, required=True)
    run.add_argument('--input', required=True, type=Path)
    run.add_argument('--out', required=True, type=Path)
    run.add_argument('--max-new-tokens', type=int, default=2048)
    run.add_argument('--max-length', type=int, default=8192)
    run.add_argument('--device', default='cuda:0')
    run.add_argument('--dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    run.add_argument('--require-shared-annotations', action='store_true')
    compare = sub.add_parser('compare')
    compare.add_argument('--left', required=True, type=Path)
    compare.add_argument('--right', required=True, type=Path)
    compare.add_argument('--out', required=True, type=Path)
    compare.add_argument('--require-shared-annotations', action='store_true')
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError('Audit output must be fresh')
    if args.command == 'compare':
        report = compare_audits(json.loads(args.left.read_text(encoding='utf-8')),
            json.loads(args.right.read_text(encoding='utf-8')), args.require_shared_annotations)
    else:
        if min(args.max_new_tokens, args.max_length) <= 0:
            raise ValueError('Positive audit budgets required')
        rows = audio_rows(args.input)
        config = json.loads((args.checkpoint.parent / 'run_config.json').read_text(encoding='utf-8'))
        trained_val = Path(config['data']['val']['path'])
        if cot.base.file_sha(trained_val) != config['data']['val']['sha256']:
            raise ValueError('Training validation manifest bytes changed')
        allowed = {row['id']: row for row in audio_rows(trained_val)}
        if any(row['id'] not in allowed or _source_panel([row]) != _source_panel([allowed[row['id']]]) for row in rows):
            raise ValueError('Audit input is not a bound subset of the checkpoint validation audio')
        adapter = cot.load_short_plan_adapter(args.family, args.model, checkpoint=args.checkpoint,
            mode=args.mode, require_shared_annotations=args.require_shared_annotations,
            device=args.device, dtype=getattr(torch, args.dtype), max_length=args.max_length)
        report = generate_audit(adapter, rows, max_new_tokens=args.max_new_tokens,
            checkpoint_identity=cot.evaluation.checkpoint_identity(args.checkpoint),
            training_data=config['data'], input_ref=cot.base.file_ref(args.input), base_identity=config['model'],
            comparison_contract=training_contract(config),
            source_exposures=cot.evaluation.checkpoint_identity(args.checkpoint)['cursor']['source_exposures'])
        validate_audit(report)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report


if __name__ == '__main__':
    main()
