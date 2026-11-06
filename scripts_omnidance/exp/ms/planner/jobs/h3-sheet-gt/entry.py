#!/usr/bin/env python3
"""Build the reference | Wan-Dancer(GT) | H3+LoRA(GT) | H3(planner) comparison sheet."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

OSS = Path(os.environ.get('WD_OSS_ROOT', '/data/oss_bucket_0/niezichao.nzc'))
VENV_TAR = Path('/data/oss_bucket_0/niezichao.nzc/h3/env/cache/venv_h3_3a418017cc795bd1.tar')
RUN_ID = os.environ.get('SHEET_RUN_ID') or time.strftime('h3sheet_%Y%m%d_%H%M%S')
LOCAL = Path('/checkpoint/binary') / RUN_ID
OUT = OSS / 'h3' / 'review' / 'sheets'
PLANNER = OSS / 'h3/review/videos_planner'
GT_H3 = OSS / 'h3/runs/h3jobs_20261103_220042/outputs'
WAN = OSS / 'baselines/Wan-Dancer-runs-GT/wandancer_gt_bench10_v1'
BENCH = OSS / 'baselines/benchmark10'


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


SCRIPT = r'''
import json, sys
from pathlib import Path
import av
from PIL import Image, ImageDraw
bench, pl, gt, wan, out = map(Path, sys.argv[1:6])
TH = (200, 356)
ids = [r['id'] for r in json.loads((bench / 'manifest.json').read_text())]
def mid(p):
    c = av.open(str(p)); s = c.streams.video[0]
    fs = [f.to_image().convert('RGB') for f in c.decode(s)]; c.close()
    return fs[len(fs) // 2].resize(TH)
cols = ('reference', 'Wan-Dancer (GT)', 'H3+LoRA1000 (GT)', 'H3 (planner)')
rows = []
for sid in ids:
    ref = bench / 'images' / (sid + '.jpg')
    w = wan / (sid + '.mp4')
    g = gt / (sid + '__gt_caption.mp4')
    p = next(pl.glob(sid + '*.mp4'), None)
    if not (ref.is_file() and w.is_file() and g.is_file() and p):
        print('skip', sid, ref.is_file(), w.is_file(), g.is_file(), bool(p)); continue
    rows.append((sid, [Image.open(ref).convert('RGB').resize(TH), mid(w), mid(g), mid(p)]))
L = 24
sheet = Image.new('RGB', (len(cols) * TH[0], len(rows) * (TH[1] + L)), 'white')
d = ImageDraw.Draw(sheet)
for i, (sid, cells) in enumerate(rows):
    y = i * (TH[1] + L)
    for c, im in enumerate(cells): sheet.paste(im, (c * TH[0], y))
    d.text((6, y + TH[1] + 4), sid, fill='black')
d.text((6, 4), ' | '.join(cols), fill='red')
out.mkdir(parents=True, exist_ok=True)
path = out / 'compare_gt_4col.png'
sheet.save(path)
print(path, len(rows))
'''


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    venv = LOCAL / 'venv'
    python = venv / 'venv_h3/bin/python'
    if not python.is_file():
        venv.mkdir(exist_ok=True)
        log('extract venv')
        subprocess.run(['tar', '-xf', str(VENV_TAR), '-C', str(venv)], check=True)
    script = LOCAL / 'sheet.py'
    script.write_text(SCRIPT)
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(python), str(script), str(BENCH), str(PLANNER), str(GT_H3), str(WAN), str(OUT)], check=True)
    log('sheet done')


if __name__ == '__main__':
    main()
