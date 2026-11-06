#!/usr/bin/env python3
"""Build a 10-case holdout benchmark strictly from the full-v3 test split."""
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import av

ROOT = Path('/ssd/2/niezichao.nzc/artifacts/captioner-jev-data-v1')
LABELS = Path('/ssd/2/niezichao.nzc/artifacts/planner_eval_local/vlm_v4_peek')
CLIPS = Path('/ssd/2/niezichao.nzc/artifacts/h3-nebula-prep-20261030/full_data/clips')
OUT = Path('/ssd/2/niezichao.nzc/artifacts/benchmark10_holdout')
CLASSES = ['pop / casual', 'hip-hop', 'jazz funk / heels', 'chinese classical', 'chinese folk',
           'contemporary', 'k-pop', 'belly dance', 'ballet']
EXTRA = ['jazz funk / heels', 'contemporary', 'ballet', 'chinese folk']
PROMPT = {
    'ballet': '一个人正在跳舞，舞蹈种类是古典舞。',
    'chinese classical': '一个人正在跳舞，舞蹈种类是古典舞。',
    'chinese folk': '一个人正在跳舞，舞蹈种类是古典舞。',
    'k-pop': '一个人正在跳舞，舞蹈种类是韩舞。',
    'contemporary': '一个人正在跳舞，舞蹈种类是现代舞。',
}


def first_frame(video, image):
    container = av.open(str(video))
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        frame.to_image().convert('RGB').save(image)
        break
    container.close()


def main():
    labels = {}
    for path in LABELS.glob('*.jsonl'):
        for line in path.read_text(errors='replace').splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get('parse_ok') and row.get('genre') in CLASSES:
                labels[row['id']] = row['genre']
    rows = {}
    for line in (ROOT / 'full-v3' / 'test.jsonl').read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row['id']] = row
    by_genre = defaultdict(list)
    for sid, genre in labels.items():
        if sid not in rows:
            continue
        if not (CLIPS / (sid + '.mp4')).is_file():
            continue
        audio = rows[sid]['audio_path']
        audio = (ROOT / 'full-v3' / audio).resolve() if not Path(audio).is_absolute() else Path(audio)
        if not audio.is_file():
            continue
        by_genre[genre].append(sid)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'images').mkdir(parents=True)
    (OUT / 'audio').mkdir(parents=True)
    manifest = []
    seen = set()
    plan = [(genre, 1) for genre in CLASSES if by_genre[genre]]
    extra_genres = [genre for genre in EXTRA if len(by_genre[genre]) >= 2]
    while sum(count for _, count in plan) < 10 and extra_genres:
        genre = extra_genres.pop(0)
        plan.append((genre, 1))
    for genre, _ in plan:
        candidates = sorted(by_genre[genre],
                            key=lambda sid: hashlib.sha256(('holdout10:' + sid).encode()).hexdigest())
        sid = next(candidate for candidate in candidates if candidate not in seen)
        seen.add(sid)
        row = rows[sid]
        audio_src = Path(row['audio_path'])
        if not audio_src.is_absolute():
            audio_src = (ROOT / 'full-v3' / audio_src).resolve()
        shutil.copyfile(audio_src, OUT / 'audio' / (sid + '.wav'))
        first_frame(CLIPS / (sid + '.mp4'), OUT / 'images' / (sid + '.jpg'))
        manifest.append(dict(id=sid, genre=genre, motion_score=None, prompt=PROMPT.get(genre, '一个人正在跳舞。'),
                             image='images/' + sid + '.jpg', audio='audio/' + sid + '.wav',
                             difficulty='holdout', source_split='test',
                             reference_caption=row['target_text']))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(cases=len(manifest), ids=[m['id'] for m in manifest]), ensure_ascii=False))


if __name__ == '__main__':
    main()
