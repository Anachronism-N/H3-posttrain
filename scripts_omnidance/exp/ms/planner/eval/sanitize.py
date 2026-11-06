#!/usr/bin/env python3
"""Truncate planner output at the first post-caption control-token leak."""
import json
import re
import sys
from pathlib import Path

STOPS = ('\nassistant', '\nAssistant', 'Assistant:', 'End of interaction', '###', '<|im_end|>')
TRAIL = re.compile(r'[\s.;,:\-]*$')


JUNK_LINE = re.compile(r'^[-|=#.:!?@%$*+\[\]{}<>~^&(),;\s]+$')


def clean(text):
    cut = len(text)
    for token in STOPS:
        index = text.find(token)
        if index != -1:
            cut = min(cut, index)
    kept = []
    for line in text[:cut].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r'[\u4e00-\u9fff]', stripped):
            break
        if JUNK_LINE.fullmatch(stripped):
            break
        if len(stripped) < 12 and not stripped.startswith('['):
            break
        if re.search(r'(\.\.\.|>>|<<|\|\||##)', stripped):
            break
        kept.append(stripped)
    result = '\n'.join(kept)
    result = re.split(r'[\u4e00-\u9fff]', result)[0]
    # Drop a trailing incomplete timestamp header like "[2.520-3.440s]"
    result = re.sub(r'\[\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?\s*s?\]\s*$', '', result)
    return TRAIL.sub('', result).strip()


def main():
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    with source.open(encoding='utf-8') as stream, target.open('w', encoding='utf-8') as out:
        for line in stream:
            record = json.loads(line)
            raw = record.get('raw_text', '')
            record['raw_text_untruncated'] = raw
            record['raw_text'] = clean(raw)
            record['truncated_chars'] = len(raw) - len(record['raw_text'])
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            rows.append(record)
    print(json.dumps(dict(rows=len(rows),
                          mean_truncated=sum(r['truncated_chars'] for r in rows) / max(len(rows), 1),
                          mean_clean_chars=sum(len(r['raw_text']) for r in rows) / max(len(rows), 1))))


if __name__ == '__main__':
    main()
