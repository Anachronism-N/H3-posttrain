"""Snapshot existing fusion outputs; source actions are assumed correct by user policy."""
import argparse
import hashlib
import json
from pathlib import Path
import re

POLICY = 'user_authorized_source_actions_assumed_correct_20261023'
TIMING = re.compile(r'\b(?:a few seconds (?:earlier|later)|(?:one|a) second (?:earlier|later)|in the second (?:right )?(?:before|after)|(?:one|a) second (?:before|after))\b\s*[,;:]?\s*', re.I)

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def clean(text):
    text = TIMING.sub('', text)
    text = re.sub(r'\b(?:her|his)\b', "the dancer's", text, flags=re.I)
    text = re.sub(r'\b(?:she|he)\b', 'the dancer', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    return re.sub(r'(^|[.!?]\s+)the dancer\b', lambda m: m.group(1) + 'The dancer', text)

def self_test():
    source = 'A few seconds earlier, her left hand moves away from the right shoulder. In the second right before, she turns 180 degrees clockwise.'
    assert clean(source) == "The dancer's left hand moves away from the right shoulder. The dancer turns 180 degrees clockwise."
    assert clean('Both elbows extend completely, then return.') == 'Both elbows extend completely, then return.'
    assert clean('The dancer repeats two steps twice.') == 'The dancer repeats two steps twice.'
    assert clean('The second turn is slower.') == 'The second turn is slower.'
    assert clean('The dancer bends the left knee slightly.') == 'The dancer bends the left knee slightly.'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    self_test()
    assert args.source.is_dir() and not args.out.exists()
    args.out.mkdir()
    seen = set()
    included = excluded = changed = 0
    with (args.out / 'captions.jsonl').open('x', encoding='utf-8') as captions, (args.out / 'lineage.jsonl').open('x', encoding='utf-8') as lineage, (args.out / 'unavailable.jsonl').open('x', encoding='utf-8') as unavailable:
        for path in sorted(args.source.glob('*.json')):
            try:
                data = path.read_bytes()
                record = json.loads(data)
            except (OSError, ValueError):
                continue
            if not isinstance(record, dict) or not record.get('id'):
                continue
            sid = record['id']
            assert sid not in seen
            seen.add(sid)
            target = record.get('target')
            body = target.get('motion_caption') if isinstance(target, dict) else None
            source_record = record.get('original_source_record') or {}
            sources = (record.get('actual_text_input') or {}).get('sources', [])
            have_rich = any(str(row.get('ref', '')).startswith('rich:') for row in sources)
            have_ms = any(row.get('ref') == 'MS:original' for row in sources)
            if not isinstance(body, str) or not body.strip() or source_record.get('id') != sid or not have_rich or not have_ms:
                excluded += 1
                unavailable.write(json.dumps({'id': sid, 'source_path': str(path), 'source_sha256': sha_bytes(data), 'reason': 'missing_parsed_caption_or_two_source_binding'}, ensure_ascii=False) + '\n')
                continue
            normalized = clean(body)
            changed += normalized != body
            included += 1
            common = {'id': sid, 'split': record.get('split'), 'source_group': record.get('source_group'), 'caption_version': 'rich_ms_fusion_source_assumed_text_cleanup_v1', 'source_record_path': str(path), 'source_record_sha256': sha_bytes(data), 'source_policy': POLICY, 'source_action_correctness_assumed': True, 'semantic_review_performed': False, 'human_review_completed': False, 'original_source_window_mapping': 'unknown', 'faithfulness_status': 'not_individually_verified'}
            captions.write(json.dumps({**common, 'motion_caption': normalized, 'genre': record.get('genre')}, ensure_ascii=False) + '\n')
            lineage.write(json.dumps({**common, 'raw_fused_caption': body, 'exported_caption': normalized, 'text_changed': normalized != body, 'original_completion_status': record.get('completion_status'), 'original_rule_flags': record.get('validation_errors', []), 'original_flags_do_not_block_export': True, 'rich_ms_source_provenance': source_record.get('source_provenance'), 'source_refs': target.get('source_refs'), 'conflicts': target.get('conflicts'), 'uncertainty': target.get('uncertainty')}, ensure_ascii=False) + '\n')
    report = {'complete': True, 'source_policy': POLICY, 'source_action_correctness_assumed': True, 'semantic_review_performed': False, 'human_review_completed': False, 'snapshot_records': len(seen), 'included': included, 'unavailable': excluded, 'deterministically_cleaned': changed, 'raw_outputs_unchanged': True, 'batch6000_complete': False, 'cleanup': 'performer pronouns and listed synthetic timing phrases only; directions, degrees, amplitude and repeated steps retained', 'script_sha256': sha_bytes(Path(__file__).read_bytes()), 'files': {name: sha_bytes((args.out / name).read_bytes()) for name in ['captions.jsonl', 'lineage.jsonl', 'unavailable.jsonl']}}
    (args.out / 'complete.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
