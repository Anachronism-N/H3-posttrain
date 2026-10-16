"""把 VLM 标签 + 3D 运动学(MS-lite 事件/kin) + 音频特征 + 已有 rich caption 关键词 合成为结构化维度与英文 caption。
维度来源：
  舞种/子风格/情绪/质感  ← Qwen3-VL-235B（低置信时用 rich caption 关键词兜底）
  能量/位移/转身/跳跃/高低/身体主导 ← 3D 关节实测（分位数分档），VLM 作对照
  BPM/速度感 ← librosa
  动作描述 ← MS-lite 三段事件 + VLM key_moves
输出: captions.json {sid: {dims:{...}, caption: str}}
"""
import argparse, glob, json, os, re, collections
import numpy as np

GENRE_PHRASE = {
    "chinese classical": "Chinese classical dance", "chinese folk": "Chinese folk dance", "ballet": "ballet",
    "contemporary": "contemporary dance", "lyrical": "lyrical dance", "modern jazz": "modern jazz dance",
    "jazz funk / heels": "jazz-funk heels dance", "k-pop": "a K-pop dance", "hip-hop": "a hip-hop dance",
    "popping": "a popping dance", "locking": "a locking dance", "breaking": "breakdance", "house": "a house dance",
    "krump": "a krump dance", "waacking": "a waacking dance", "street jazz": "a street jazz dance", "latin": "a Latin dance",
    "belly dance": "belly dance", "indian / bollywood": "an Indian Bollywood dance", "other ethnic / folk": "an ethnic folk dance",
    "pop / casual": "a casual pop dance", "freestyle / other": "a freestyle dance"}
# H3 可区分的粗类（人评：中国古典 / 现代 / 街舞 三类可区分）
COARSE = {"chinese classical": "chinese", "chinese folk": "chinese", "ballet": "classical_western", "contemporary": "contemporary",
          "lyrical": "contemporary", "modern jazz": "contemporary", "jazz funk / heels": "street", "k-pop": "street",
          "hip-hop": "street", "popping": "street", "locking": "street", "breaking": "street", "house": "street",
          "krump": "street", "waacking": "street", "street jazz": "street", "latin": "latin", "belly dance": "ethnic",
          "indian / bollywood": "ethnic", "other ethnic / folk": "ethnic", "pop / casual": "pop", "freestyle / other": "pop"}
RICH_KW = [("classical chinese", "chinese classical"), ("chinese classical", "chinese classical"), ("chinese", "chinese folk"),
           ("hip-hop", "hip-hop"), ("k-pop", "k-pop"), ("popping", "popping"), ("locking", "locking"), ("ballet", "ballet"),
           ("lyrical", "lyrical"), ("latin", "latin"), ("belly", "belly dance"), ("folk", "other ethnic / folk"),
           ("street", "hip-hop"), ("contemporary", "contemporary")]


def rich_genre(t):
    t = t.lower(); m = re.search(r'choreographic content[^\]]*\]\s*(.*?)\[3\.', t, re.S); s = m.group(1) if m else ''
    for k, g in RICH_KW:
        if k in s: return g
    return None


def tert(v, q):  # 0/1/2 by global terciles
    return int(v > q[0]) + int(v > q[1])


def join_events(evs, adv):
    t = [e['text'] for e in evs[:3]]
    if not t: return None
    s = t[0] if len(t) == 1 else (', '.join(t[:-1]) + ' and ' + t[-1])
    return s + (' ' + adv if adv else '')


def main():
    ap = argparse.ArgumentParser()
    E = '/apdcephfs_gy4/share_302533218/cedricnie/diff/logs_omnidance/exp'
    ap.add_argument('--vlm', default=f'{E}/MS/vlm/labels_s*.jsonl'); ap.add_argument('--ms', default=f'{E}/MS/ms_orig')
    ap.add_argument('--motion_source', choices=['original', 'lite'], default='original')
    ap.add_argument('--audio', default=f'{E}/MS/audio_feats.jsonl'); ap.add_argument('--rich', default=f'{E}/H3AR/full12k_rich_prompts.json')
    ap.add_argument('--out', default=f'{E}/MS/captions_v1.json')
    ap.add_argument('--format_probe', action='store_true', help='Prepare same-content original/structured bodies from the supplied original MS directory')
    a = ap.parse_args()
    if a.format_probe:
        original, structured = {}, {}
        for path in sorted(glob.glob(os.path.join(a.ms, '*.json'))):
            item = json.load(open(path))
            sentences = [sentence.strip() for sentence in item.get('ms', []) if sentence.strip()]
            if not sentences:
                raise ValueError(f'Cannot prepare a format probe from an empty caption: {path}')
            prefix = '<Subject 1> performs a dance.'
            original[item['sid']] = prefix + ' ' + ' '.join(sentences)
            structured[item['sid']] = 'Dance style:\n' + prefix + '\nMotion sequence:\n' + '\n'.join(sentences)
        if not original:
            raise ValueError('No original MotionScript captions available for the format probe')
        with open(a.out, 'w') as stream:
            json.dump(dict(original=original, structured=structured), stream, indent=2)
        print('Same-content format probe:', sorted(original))
        return
    V = {}
    for f in glob.glob(a.vlm):
        for l in open(f):
            d = json.loads(l)
            if d.get('label'): V[d['sid']] = d['label']
    A = {}
    if os.path.exists(a.audio):
        for l in open(a.audio):
            d = json.loads(l)
            if 'bpm' in d: A[d['sid']] = d
    R = json.load(open(a.rich))
    M = {}
    for f in glob.glob(os.path.join(a.ms, '*.json')):
        d = json.load(open(f))
        if a.motion_source == 'original':
            if d.get('ms') and any(sentence.strip() for sentence in d['ms']):
                M[d['sid']] = d
        elif d.get('segs'):
            M[d['sid']] = d
    sids = sorted(set(V) & set(M))
    print('vlm', len(V), 'ms', len(M), 'audio', len(A), 'both', len(sids))
    K = {k: np.array([M[s]['kin'][k] for s in sids]) for k in ['speed', 'travel', 'disp', 'pelvis_h_med', 'arm_speed', 'leg_speed', 'torso_speed']}
    q = {k: np.percentile(v, [33.3, 66.7]) for k, v in K.items()}
    bpms = np.array([A[s]['bpm'] for s in sids if s in A]) if A else np.array([100.])
    out, stat = {}, collections.Counter()
    for s in sids:
        v, m, kin = V[s], M[s], M[s]['kin']
        g = v.get('genre') if v.get('genre') in GENRE_PHRASE else None
        rg = rich_genre(R.get(s, ''))
        if (g is None or v.get('genre_conf') == 'low') and rg: g = rg; stat['genre_from_rich'] += 1
        g = g or 'freestyle / other'
        stat['agree_rich' if rg and COARSE.get(rg) == COARSE[g] else ('disagree_rich' if rg else 'no_rich')] += 1
        energy = ['low', 'medium', 'high'][tert(kin['speed'], q['speed'])]
        travel = ('travels across the floor' if kin['disp'] > 0.35 or kin['travel'] > 1.6 else
                  ('moves around a little' if kin['travel'] > q['travel'][1] else 'stays mostly in place'))
        level = ('includes low floor work' if kin['pelvis_h_min'] < 0.45 else
                 ('goes down into low crouches' if kin['pelvis_h_min'] < 0.7 else 'stays upright'))
        jumps = 'jumps several times' if kin['jumps'] >= 2 else ('jumps once' if kin['jumps'] == 1 else None)
        if v.get('jumps') == 'none' and jumps: jumps = None; stat['jump_vetoed_by_vlm'] += 1
        turns = 'spins around' if abs(kin['net_turn']) > 270 else ('turns around' if abs(kin['net_turn']) > 120 else None)
        if turns is None and v.get('turns') == 'spins': turns = 'spins around'
        ra, rl, rt = kin['arm_speed'] / (q['arm_speed'][1] + 1e-6), kin['leg_speed'] / (q['leg_speed'][1] + 1e-6), kin['torso_speed'] / (q['torso_speed'][1] + 1e-6)
        focus = max([('the arms', ra), ('the legs and footwork', rl), ('the torso and hips', rt)], key=lambda x: x[1])[0]
        if min(ra, rl, rt) > 0.95: focus = 'the whole body'
        bpm = A[s]['bpm'] if s in A else None
        tempo = v.get('tempo_feel', 'moderate')
        if bpm: tempo = 'slow' if bpm < 85 else ('fast' if bpm > 125 else 'moderate')
        quality = v.get('quality', 'mixed'); mood = v.get('mood', '')
        dims = dict(genre=g, coarse=COARSE[g], sub_style=v.get('sub_style', ''), genre_conf=v.get('genre_conf'), rich_genre=rg,
                    energy=energy, energy_vlm=v.get('energy'), tempo=tempo, bpm=bpm, quality=quality, mood=mood,
                    focus=focus, focus_vlm=v.get('body_focus'), level=level, level_vlm=v.get('level'), travel=travel,
                    travel_vlm=v.get('travel'), jumps=jumps, jumps_vlm=v.get('jumps'), turns=turns, turns_vlm=v.get('turns'),
                    props=v.get('props', 'none'), key_moves=v.get('key_moves', []),
                    motion_source=a.motion_source,
                    seg_events=[[e['k'] for e in sg['events']] for sg in m.get('segs', [])])
        # ---- caption ----
        S = '<Subject 1>'
        c = f"{S} performs {GENRE_PHRASE[g]}"
        if dims['sub_style'] and dims['sub_style'].lower() not in GENRE_PHRASE[g].lower(): c += f" ({dims['sub_style']})"
        c += '.'
        c += f" The music tempo is {tempo}" + (f" (about {bpm:.0f} BPM)" if bpm else '') + '.'
        c += f" The dancer's movement energy is {energy}" + (f"; the expression is {mood}." if mood else '.')
        body = [travel, level] + [x for x in (jumps, turns) if x]
        c += f" {S} " + ', '.join(body[:-1]) + ' and ' + body[-1] + '.'
        c += f" The movement is {quality}, led by {focus}."
        if dims['props'] and dims['props'] != 'none': c += f" {S} dances with {dims['props']}."
        context = c
        if a.motion_source == 'original':
            motion = ' '.join(sentence.strip() for sentence in m['ms'] if sentence.strip())
        else:
            segs = [join_events(sg['events'], sg['adv']) for sg in m['segs']]
            conn = ['First', 'Then', 'Finally']
            motion = ' '.join(f"{conn[i]}, {S} {t}." for i, t in enumerate(segs) if t)
        c += ' ' + motion
        # Identical context and movement text; only headings/line breaks differ.
        structured = 'Dance context:\n' + context + '\nMovement description:\n' + motion
        out[s] = dict(dims=dims, caption=c, caption_structured=structured, motion_original=motion)
    json.dump(out, open(a.out, 'w'), ensure_ascii=False)
    print(len(out), dict(stat))
    print('genre', collections.Counter(o['dims']['genre'] for o in out.values()).most_common())
    print('coarse', collections.Counter(o['dims']['coarse'] for o in out.values()).most_common())
    for s in list(out)[:3]: print(out[s]['caption'], '\n')


if __name__ == '__main__':
    main()
