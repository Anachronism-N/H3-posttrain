import json, av, numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor  # noqa: F401  (import check only)

root = Path('/ssd/2/niezichao.nzc/artifacts/captioner-jev-data-v1')
labels = {}
for p in (root / 'vlm_labels_v2').glob('*.jsonl'):
    for line in p.read_text(errors='replace').splitlines():
        if line.strip():
            try: r = json.loads(line)
            except: continue
            if r.get('genre') and r.get('parse_ok'):
                labels[r['id']] = r
rows = {}
for split in ('train', 'val', 'test'):
    for line in (root / 'full-v2' / (split + '.jsonl')).read_text().splitlines():
        if line.strip():
            r = json.loads(line); rows[r['id']] = r
classes = ['pop / casual', 'hip-hop', 'jazz funk / heels', 'chinese classical', 'chinese folk',
           'contemporary', 'k-pop', 'belly dance', 'other ethnic / folk', 'ballet']
prompt_map = {
 'ballet':'一个人正在跳舞，舞蹈种类是古典舞。', 'chinese classical':'一个人正在跳舞，舞蹈种类是古典舞。',
 'chinese folk':'一个人正在跳舞，舞蹈种类是古典舞。', 'k-pop':'一个人正在跳舞，舞蹈种类是韩舞。',
 'contemporary':'一个人正在跳舞，舞蹈种类是现代舞。', 'lyrical':'一个人正在跳舞，舞蹈种类是现代舞。',
}
out = Path('/ssd/2/niezichao.nzc/artifacts/benchmark10'); (out/'images').mkdir(parents=True, exist_ok=True); (out/'audio').mkdir(parents=True, exist_ok=True)
by = defaultdict(list)
for sid, lab in labels.items():
    if sid in rows and lab['genre'] in classes:
        by[lab['genre']].append(sid)
chosen = []
for genre in classes:
    ids = sorted(by[genre], key=lambda x: __import__('hashlib').sha256(('bench10:'+x).encode()).hexdigest())
    for sid in ids:
        if all(c['id'] != sid for c in chosen):
            chosen.append({'id': sid, 'genre': genre}); break
    if len(chosen) >= 10: break
if len(chosen) < 10:
    raise SystemExit('not enough classes: '+str([(c, len(by[c])) for c in classes]))
clip_root = Path('/ssd/2/niezichao.nzc/artifacts/h3-nebula-prep-20261030/full_data/clips')
audio_root = Path('/ssd/2/niezichao.nzc/artifacts/h3-nebula-prep-20261030/full_data/audio')
manifest = []
for case in chosen:
    sid = case['id']; video = clip_root / (sid + '.mp4'); audio = audio_root / (sid + '.wav')
    if not video.is_file() or not audio.is_file(): raise SystemExit('missing media '+sid)
    c = av.open(str(video)); st = c.streams.video[0]; frames = []
    for i, f in enumerate(c.decode(st)):
        if i % 5 == 0: frames.append(np.asarray(f.to_image().convert('L').resize((64, 128)), dtype=np.float32))
        if i >= 80: break
    c.close()
    diffs = [float(np.abs(frames[i+1]-frames[i]).mean()) for i in range(min(len(frames)-1, 15))]
    score = float(np.mean(diffs)) if diffs else 0.0
    image = out/'images'/(sid+'.jpg'); Image.open(video).save(image) if False else None
    # extract first frame via av
    c = av.open(str(video)); st = c.streams.video[0]
    for f in c.decode(st):
        f.to_image().convert('RGB').save(image); break
    c.close()
    import shutil; shutil.copyfile(audio, out/'audio'/(sid+'.wav'))
    prompt = prompt_map.get(case['genre'], '一个人正在跳舞，舞蹈种类是街舞。')
    manifest.append(dict(id=sid, genre=case['genre'], motion_score=score,
                         prompt=prompt, image='images/'+sid+'.jpg', audio='audio/'+sid+'.wav'))
manifest.sort(key=lambda x: x['motion_score'])
for i, c in enumerate(manifest):
    c['difficulty'] = 'easy' if i < 3 else 'medium' if i < 7 else 'hard'
(out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
