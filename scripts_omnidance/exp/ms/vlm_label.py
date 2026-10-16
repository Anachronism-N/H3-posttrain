"""Qwen3-VL-235B-A22B (vLLM, TP=8) 舞蹈视频结构化标注：舞种/风格/情绪/质感/动作要点。
只看视频（不喂已有 caption，避免偏置；已有 caption 之后做交叉核对）。
用法: python vlm_label.py --ids ids.txt --out labels.jsonl [--shard i --nshard n] [--limit N]
"""
import argparse, json, os, re, time
VID = '/apdcephfs_gy4/share_302533218/cedricnie/OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/videos/{}.mp4'
GENRES = ["chinese classical", "chinese folk", "ballet", "contemporary", "lyrical", "modern jazz", "jazz funk / heels",
          "k-pop", "hip-hop", "popping", "locking", "breaking", "house", "krump", "waacking", "street jazz",
          "latin", "belly dance", "indian / bollywood", "other ethnic / folk", "pop / casual", "freestyle / other"]
PROMPT = f"""You are an expert dance analyst. Watch this short solo dance clip carefully (about 5 seconds) and describe the dance.
Answer with ONE JSON object only, no extra text, using exactly these keys:
{{
 "genre": one of {json.dumps(GENRES)},
 "genre_2nd": second most likely genre from the same list, or "none",
 "genre_conf": "high" | "medium" | "low",
 "sub_style": short free text, e.g. "Chinese classical sleeve dance", "K-pop girl-group choreography",
 "energy": "low" | "medium" | "high",
 "tempo_feel": "slow" | "moderate" | "fast",
 "quality": "sharp and percussive" | "smooth and flowing" | "bouncy" | "mixed",
 "mood": 1-3 words, e.g. "graceful, elegant",
 "body_focus": "arms" | "legs and footwork" | "torso and hips" | "whole body",
 "level": "upright" | "crouched or low" | "includes floor work",
 "travel": "stays in place" | "moves around a little" | "travels across the floor",
 "turns": "none" | "some turns" | "spins",
 "jumps": "none" | "some jumps" | "many jumps",
 "key_moves": list of 3 to 5 short concrete movement phrases in time order, e.g. "raises both arms overhead", "body roll", "kicks the right leg",
 "props": "none" or the prop used (fan, sleeves, chair, ...)
}}"""


def parse(t):
    m = re.search(r'\{.*\}', t, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--model', default='/apdcephfs_gy4/share_302533218/cedricnie/models/Qwen3-VL-235B-A22B-Instruct')
    ap.add_argument('--tp', type=int, default=8); ap.add_argument('--frames', type=int, default=16)
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.ids) if l.strip()][a.shard::a.nshard]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try: done.add(json.loads(l)['sid'])
            except Exception: pass
    ids = [i for i in ids if i not in done]
    if a.limit: ids = ids[:a.limit]
    print('todo', len(ids), flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=16384, gpu_memory_utilization=0.88,
              limit_mm_per_prompt={"video": 1}, allowed_local_media_path='/apdcephfs_gy4',
              media_io_kwargs={"video": {"num_frames": a.frames}},
              mm_processor_kwargs={"max_pixels": 448 * 800}, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=400)
    fo = open(a.out, 'a'); t0 = time.time(); n = 0
    for s in range(0, len(ids), a.bs):
        chunk = ids[s:s + a.bs]
        msgs = [[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "file://" + VID.format(i)}},
                                               {"type": "text", "text": PROMPT}]}] for i in chunk]
        outs = llm.chat(msgs, sp, use_tqdm=False)
        for i, o in zip(chunk, outs):
            t = o.outputs[0].text
            fo.write(json.dumps(dict(sid=i, label=parse(t), raw=t), ensure_ascii=False) + '\n')
        fo.flush(); n += len(chunk)
        print(f'{n}/{len(ids)} {(time.time() - t0) / n:.2f}s/clip', flush=True)
    print('DONE', flush=True)
