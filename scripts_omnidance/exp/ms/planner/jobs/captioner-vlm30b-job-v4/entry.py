#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

JOB = Path(__file__).resolve().parent
OSS = Path(os.environ.get('VLM_OSS_ROOT', '/data/oss_bucket_0/niezichao.nzc'))
MODE = (sys.argv[1] if len(sys.argv) > 1 else ((JOB / 'mode.txt').read_text().strip() if (JOB / 'mode.txt').is_file() else 'probe')).strip()
VENV_TAR = OSS / 'm2d_data/env/venv_h3_3a418017cc795bd1.tar'
if os.environ.get('VLM_REEXEC') != '1':
    local = Path(os.environ.get('VLM_VENV', '/checkpoint/binary/vlm_venv'))
    python = local / 'venv_h3' / 'bin' / 'python'
    if not python.is_file():
        local.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tar', '-xf', str(VENV_TAR), '-C', str(local)], check=True)
    env = dict(os.environ, VLM_REEXEC='1')
    os.execve(str(python), [str(python), __file__, MODE], env)

import decord
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

MODEL_OSS = OSS / 'm2d_data/models/Qwen3-VL-30B-A3B-Instruct'
TAR_DIR = OSS / 'm2d_data/videos_tars'
OUT_DIR = OSS / 'm2d_output/vlm_genre_v4_qwen3vl30b'
IDS_FILE = OSS / 'm2d_output/_completed_ids_198k.txt'
RUN_ID = os.environ.get('VLM_RUN_ID') or time.strftime('vlm_%Y%m%d_%H%M%S')
LOCAL = Path(os.environ.get('VLM_LOCAL', '/checkpoint/binary/vlm30bv4_' + RUN_ID))
GENRES = ["chinese classical", "chinese folk", "ballet", "contemporary", "lyrical", "modern jazz",
          "jazz funk / heels", "k-pop", "hip-hop", "popping", "locking", "breaking", "house",
          "krump", "waacking", "street jazz", "latin", "belly dance", "indian / bollywood",
          "other ethnic / folk", "pop / casual", "freestyle / other"]
PROMPT = """You are a professional dance analyst. Watch these 16 frames sampled from one short solo dance clip and classify its dance style.

Choose exactly one category from this fixed list:
""" + json.dumps(GENRES) + """

Definitions and priority:
- contemporary: release/contraction, floor work, weight shifts, expressive or lyrical modern technique.
- k-pop: idol/group choreography, point dances, sharp formations, polished performance.
- hip-hop: groove, bounce, isolations, urban choreography, breaking/popping/locking/waacking/house/krump when clearly present.
- jazz funk / heels: commercial jazz, sharp isolations, heel work.
- chinese classical / chinese folk: Chinese classical or ethnic folk vocabulary, sleeves, regional folk steps.
- ballet / latin / belly dance / indian / other ethnic: use only with clear technical evidence.
- pop / casual: simple social-media, party, or casual dancing WITHOUT a specific technical vocabulary. Do NOT use this as a default.
- unknown: not enough dance or too ambiguous.

Rules:
- Choose the MOST SPECIFIC style supported by visible movement. Only choose pop / casual if no specific style fits.
- Do not infer style from music, clothing, or setting.
- Return one JSON object only, with keys: genre, confidence (high/medium/low), secondary (a genre or "none"), evidence (one sentence citing observed movement).
Do not add markdown."""


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def load_ids():
    return {line.strip() for line in IDS_FILE.read_text().splitlines() if line.strip()}


def build_local():
    LOCAL.mkdir(parents=True, exist_ok=True)
    model = LOCAL / 'model'
    if not (model / 'model.safetensors.index.json').is_file():
        log('copying model')
        subprocess.run(['cp', '-a', str(MODEL_OSS) + '/.', str(model)], check=True)
    return model


def load_model(model_path, device):
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa', device_map='auto')
    model.eval()
    return processor, model


def frames_of(path, count=16):
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=2)
    total = len(reader)
    if total < 1:
        raise ValueError('empty video')
    indices = [round(i * (total - 1) / (count - 1)) for i in range(count)]
    return [Image.fromarray(reader[i].asnumpy()).convert('RGB') for i in indices]


def parse_json(text):
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except Exception:
        return None
    if not isinstance(value, dict) or value.get('genre') not in GENRES + ['unknown']:
        return None
    return value


@torch.inference_mode()
def generate(model, processor, frames, device):
    conversation = [{'role': 'user', 'content': [{'type': 'video', 'video': frames},
                                                {'type': 'text', 'text': PROMPT}]}]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], videos=[frames], return_tensors='pt', padding=True)
    inputs = {key: value.to(model.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    text = processor.tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    parsed = parse_json(text)
    return parsed, text


def label_one(processor, model, path, sid, device):
    frames = frames_of(path)
    parsed, raw = generate(model, processor, frames, device)
    return dict(id=sid, genre=(parsed or {}).get('genre', 'unknown'),
                confidence=(parsed or {}).get('confidence', 'unknown'),
                secondary=(parsed or {}).get('secondary', 'none'),
                evidence=(parsed or {}).get('evidence', ''),
                parse_ok=parsed is not None, raw=raw, model='Qwen3-VL-30B-A3B-Instruct',
                prompt_version='vlm-genre-22-v1', video_bytes=path.stat().st_size)


def process_rank(rank, world, device, ids, model_path):
    processor, model = load_model(model_path, device)
    tars = sorted(TAR_DIR.glob('part_*.tar'))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = 0
    for index in range(rank, len(tars), world):
        tar_path = tars[index]
        out_path = OUT_DIR / (tar_path.stem + '.jsonl')
        if out_path.is_file() and out_path.with_suffix('.jsonl.done').is_file():
            log('skip complete ' + tar_path.name)
            continue
        local_tar = LOCAL / tar_path.name
        if not local_tar.is_file():
            log('download ' + tar_path.name)
            shutil.copyfile(tar_path, local_tar)
        video_dir = LOCAL / ('videos_' + tar_path.stem)
        video_dir.mkdir(exist_ok=True)
        found = []
        with tarfile.open(local_tar, 'r:') as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith('.mp4'):
                    continue
                sid = Path(member.name).stem
                if sid not in ids:
                    continue
                target = video_dir / (sid + '.mp4')
                if not target.is_file():
                    source = tar.extractfile(member)
                    with target.open('wb') as stream:
                        shutil.copyfileobj(source, stream)
                found.append(sid)
        existing = set()
        if out_path.is_file():
            for line in out_path.read_text(encoding='utf-8', errors='replace').splitlines():
                if line.strip():
                    try:
                        existing.add(json.loads(line).get('id'))
                    except Exception:
                        pass
        found = [sid for sid in found if sid not in existing]
        log('%s target videos %d (already %d)' % (tar_path.name, len(found), len(existing)))
        with out_path.open('a', encoding='utf-8') as stream:
            for sid in found:
                try:
                    result = label_one(processor, model, video_dir / (sid + '.mp4'), sid, device)
                except Exception as error:
                    result = dict(id=sid, genre='unknown', confidence='unknown', secondary='none',
                                  evidence='', parse_ok=False, raw='', error=repr(error),
                                  model='Qwen3-VL-30B-A3B-Instruct', prompt_version='vlm-genre-22-v1')
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
                done += 1
                if done % 50 == 0:
                    log('labeled %d' % done)
        out_path.with_suffix('.jsonl.done').write_text(json.dumps({'complete': True, 'videos': len(found)}) + '\n')
        shutil.rmtree(video_dir, ignore_errors=True)
        local_tar.unlink(missing_ok=True)
        log('complete ' + tar_path.name)


def main():
    if MODE == 'probe':
        report = dict(host=os.uname().nodename, gpu=torch.cuda.is_available(), gpus=torch.cuda.device_count(),
                      oss=True, ids_file=IDS_FILE.is_file(), tars=len(list(TAR_DIR.glob('part_*.tar'))),
                      model=MODEL_OSS.is_dir())
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report['gpu'] or not report['oss'] or not report['ids_file'] or not report['model']:
            return 1
        device = torch.device('cuda:0')
        model_path = build_local()
        processor, model = load_model(model_path, device)
        ids = load_ids()
        tars = sorted(TAR_DIR.glob('part_*.tar'))
        with tarfile.open(tars[0], 'r:') as tar:
            for member in tar:
                if member.isfile() and member.name.endswith('.mp4') and Path(member.name).stem in ids:
                    sid = Path(member.name).stem
                    path = LOCAL / (sid + '.mp4')
                    with path.open('wb') as stream:
                        shutil.copyfileobj(tar.extractfile(member), stream)
                    result = label_one(processor, model, path, sid, device)
                    print(json.dumps(result, ensure_ascii=False)[:2000])
                    return 0
        return 2
    if MODE == 'worker':
        local_rank = int(os.environ['VLM_LOCAL_RANK'])
        global_rank = int(os.environ['VLM_GLOBAL_RANK'])
        world = int(os.environ['VLM_WORLD'])
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        model_path = build_local()
        ids = load_ids()
        process_rank(global_rank, world, device, ids, model_path)
        return 0
    if MODE != 'label':
        raise ValueError('mode must be probe, label or worker')
    model_path = build_local()
    os.environ['VLM_MODEL_READY'] = '1'
    os.environ['VLM_LOCAL'] = str(LOCAL)
    pods = int(os.environ.get('WORLD_SIZE', '1'))
    pod_rank = int(os.environ.get('RANK', '0'))
    children = []
    for local_rank in range(4):
        env = dict(os.environ, VLM_REEXEC='1', VLM_LOCAL_RANK=str(local_rank),
                   VLM_GLOBAL_RANK=str(pod_rank * 4 + local_rank), VLM_WORLD=str(pods * 4),
                   CUDA_VISIBLE_DEVICES='%d,%d' % (2 * local_rank, 2 * local_rank + 1))
        children.append(subprocess.Popen([sys.executable, __file__, 'worker'], env=env))
    codes = [child.wait() for child in children]
    return 0 if all(code == 0 for code in codes) else 1


if __name__ == '__main__':
    sys.exit(main())
