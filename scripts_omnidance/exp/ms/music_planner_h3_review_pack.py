"""Build a new, offline, anonymous fixed-three H3 review pack after complete/safe.

Only the new --out directory is public. Keep the exclusive 0600 --key-out file
outside it. No GPU calls, services, source edits, weight rehash, or human verdicts.
"""
import argparse
import hashlib
import importlib.util
import json
from fractions import Fraction
import os
from pathlib import Path
import secrets
import shutil
import re
import tempfile
import subprocess
import sys
import wave

sys.dont_write_bytecode = True
ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
EXP = ROOT / 'diff/logs_omnidance/exp/MS/music_planner_fixed2s_training_20261024_v1'
HANDOFF = EXP / 'eval_handoff_clean_v11'
CONFIG = HANDOFF / 'h3_compare_v1/run_config_v3.json'
MANAGER = EXP / 'runtime/planner_h3_render_manager_v3.py'
CONFIG_SHA = '55f7fd1c75b996a43aacfdbbb275a7ffd0cdf813991680f66c9311b4ceb5a332'
MANAGER_SHA = 'bba5d6f6cc42e1a7ac2818d41ed738ce0e53c074d3044d7a73558417042320bf'
FAMILIES = {'music_flamingo', 'moss', 'omni_captioner'}
IDS = ['wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB']
GEOMETRY = dict(height=1280, width=704, num_frames=124, fps=24)
AXES = ['clarity', 'continuity', 'choreography', 'amplitude', 'music_response']


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def source(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise RuntimeError('regular absolute nonsymlink source required')
    before, digest = path.stat(), hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError('source changed during hash')
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=after.st_size)


def frozen_manager():
    if source(MANAGER)['sha256'] != MANAGER_SHA:
        raise RuntimeError('frozen manager SHA changed')
    spec = importlib.util.spec_from_file_location('offline_review_frozen_manager', MANAGER)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def require(value, expected, message):
    if any(value.get(key) != item or type(value.get(key)) is not type(item) for key, item in expected.items()):
        raise RuntimeError(message)


def verify_complete():
    """Reuse frozen read-only validators; never call manage/capture/weight_sources."""
    config_ref = source(CONFIG)
    if config_ref['sha256'] != CONFIG_SHA:
        raise RuntimeError('frozen config SHA changed')
    cfg = read_json(CONFIG)
    queue, out = Path(cfg['queue']), Path(cfg['out'])
    expected = HANDOFF / 'h3_compare_v1'
    if queue != expected / 'queue' or out != expected / 'out':
        raise RuntimeError('actual frozen render paths required')
    complete_ref = source(queue / 'h3.compare.complete.json')
    safe_ref = source(queue / 'resources_safe.json')
    complete = read_json(complete_ref['path'])
    require(complete, dict(technical_video_complete=True, completed_ids=IDS,
            human_quality_accepted=False, human_quality=False, winner=None,
            config_sha256=CONFIG_SHA, manager_sha256=MANAGER_SHA,
            other_seven_GPUs_untouched=True), 'technical comparison is incomplete')
    if (len(complete.get('families', [])) != 3 or set(complete['families']) != FAMILIES
            or set(cfg['families']) != FAMILIES or (queue / 'failure.json').exists()):
        raise RuntimeError('exact three completed families without failure required')
    manager = frozen_manager()
    audit, resource_manager, fixed, ev, h3 = manager.bootstrap()
    manager.safe_resources(safe_ref, 1, audit)
    manager.check_sources(cfg, CONFIG, CONFIG_SHA, MANAGER_SHA, audit)
    records = manager.prerequisites(cfg, audit, resource_manager, fixed, ev, h3)
    models = {mode: audit.ref_json(cfg['weight_pre_audit'][mode]) for mode in ('cache', 'video')}
    if complete['model_sources'] != models:
        raise RuntimeError('complete models differ from existing full-SHA preaudit')
    vae = lambda model: [r for r in model['files'] if Path(r['path']).is_relative_to(manager.WEIGHTS / 'video_vae')]
    if vae(models['cache']) != vae(models['video']):
        raise RuntimeError('cache/video VAE sources differ')
    receipts = [config_ref, source(MANAGER), complete_ref, safe_ref,
                *[source(cfg['weight_pre_audit'][mode]['path']) for mode in ('cache', 'video')],
                source(cfg['protocol']['path'])]
    videos = {}
    for family in complete['families']:
        if [r['id'] for r in records[family]] != IDS or len(records[family]) != 3:
            raise RuntimeError('exact fixed-three records required')
        cache = out / family / 'cache'
        video = out / family / 'video'
        caption_ref = h3.source(cfg['families'][family]['h3_records']['path'])
        for stage, directory, mode in (('cache', cache, 'condition_cache_only'), ('video', video, 'video_generation')):
            exit_ref = source(queue / (family + '.' + stage + '.exit.json'))
            require(read_json(exit_ref['path']), dict(exit_code=0), 'renderer exit is not zero')
            receipt_ref = source(queue / (family + '.' + stage + '.complete.json'))
            receipt = read_json(receipt_ref['path'])
            require(receipt, dict(completed_ids=IDS), 'renderer stage not complete')
            run_ref = source(directory / 'run_complete.json')
            if receipt.get('report') != run_ref:
                raise RuntimeError('stage receipt does not bind actual run_complete SHA')
            audit.ref_json(receipt['report'])
            manager.check_stage(directory, mode, records[family], caption_ref, models[stage], audit, h3,
                                cache if stage == 'video' else None)
            receipts.extend([exit_ref, receipt_ref, run_ref])
        for sid in IDS:
            videos[family, sid] = dict(video=source(video / (sid + '.mp4')),
                                      case_audit=source(video / (sid + '.audit.json')))
    protocol = audit.ref_json(cfg['protocol'])
    cases = protocol['fixed3']
    if [c['id'] for c in cases] != IDS:
        raise RuntimeError('fixed cases differ')
    for case in cases:
        if source(case['audio_ref']['path']) != case['audio_ref']:
            raise RuntimeError('frozen full review audio SHA differs')
        with wave.open(case['audio_ref']['path'], 'rb') as stream:
            header = dict(channels=stream.getnchannels(), sample_rate=stream.getframerate(),
                          frames=stream.getnframes(), duration_seconds=stream.getnframes() / stream.getframerate())
        if header != case['audio_header']:
            raise RuntimeError('frozen complete audio header differs')
    return dict(cases=cases, videos=videos, receipts=receipts), h3


def mux_command(ffmpeg, video, audio, destination):
    return [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
            '-i', str(video), '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'copy', '-c:a', 'aac', '-map_metadata', '-1', '-map_chapters', '-1',
            '-metadata', 'encoder=', '-metadata:s:v:0', 'encoder=', '-metadata:s:a:0', 'encoder=',
            '-metadata:s:v:0', 'handler_name=', '-metadata:s:a:0', 'handler_name=', str(destination)]


def packet_frames(path, ffmpeg):
    result = subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
                             '-i', str(path), '-map', '0:v:0', '-c:v', 'copy',
                             '-f', 'framehash', '-hash', 'sha256', 'pipe:1'],
                            capture_output=True, text=True, check=True)
    if result.stderr.strip():
        raise RuntimeError('video packet inspection failed')
    base, frames = None, []
    for line in result.stdout.splitlines():
        if line.startswith('#tb 0:'):
            base = Fraction(line.split(':', 1)[1].strip())
        elif line and not line.startswith('#'):
            fields = [s.strip() for s in line.split(',')]
            if base is None or len(fields) != 6 or fields[0] != '0':
                raise RuntimeError('unexpected video packet audit format')
            frames.append((int(fields[1]) * base, int(fields[2]) * base,
                           int(fields[3]) * base, int(fields[4]), fields[5]))
    if len(frames) != 124:
        raise RuntimeError('video packet frame count changed')
    return frames


def verify_copy(original, copy, audio_header, ffmpeg, h3, forbidden):
    inspection = subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-i', str(copy)],
                                capture_output=True, text=True)
    log = inspection.stderr
    streams = [line for line in log.splitlines() if re.search(r'^\s*Stream #\d+:\d+', line)]
    audios = [line for line in streams if 'Audio:' in line]
    if len(streams) != 2 or len(audios) != 1 or sum('Video:' in s for s in streams) != 1:
        raise RuntimeError('copy needs exactly one video and one audio stream')
    layout = {1: 'mono', 2: 'stereo'}.get(audio_header['channels'])
    if (layout is None or not re.search(r'Audio: aac\b', audios[0])
            or str(audio_header['sample_rate']) + ' Hz, ' + layout not in audios[0]):
        raise RuntimeError('full audio codec/sample rate/channels differ')
    if any(token in log for token in forbidden):
        raise RuntimeError('identity leaked in copied media metadata')
    before, after = packet_frames(original, ffmpeg), packet_frames(copy, ffmpeg)
    if before != after:
        raise RuntimeError('copied video packets/PTS/duration changed')
    duration = max(row[1] + row[2] for row in after) - min(row[1] for row in after)
    if duration != Fraction(124, 24):
        raise RuntimeError('full video duration changed')
    with tempfile.TemporaryDirectory(prefix='anonymous-video-check-') as temporary:
        track = Path(temporary) / 'video.mp4'
        subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
                        '-i', str(copy), '-map', '0:v:0', '-c:v', 'copy', '-an',
                        '-map_metadata', '-1', str(track)], capture_output=True, text=True, check=True)
        h3.verify_media(track, GEOMETRY)
    decoded = subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror',
                              '-i', str(copy), '-map', '0:a:0', '-c:a', 'pcm_s16le',
                              '-f', 's16le', 'pipe:1'], capture_output=True, check=True)
    if decoded.stderr.strip():
        raise RuntimeError('complete AAC decode emitted errors')
    frame_bytes = 2 * audio_header['channels']
    count, remainder = divmod(len(decoded.stdout), frame_bytes)
    if remainder or not audio_header['frames'] <= count <= audio_header['frames'] + 2048:
        raise RuntimeError('decoded AAC sample count outside expected duration/padding range')
    return dict(width=704, height=1280, frames=124, fps=24,
                video_duration_seconds=float(duration),
                audio_decoded_duration_seconds=count / audio_header['sample_rate'],
                aac_decoded_samples=count, original_audio_samples=audio_header['frames'],
                aac_padding_samples=count - audio_header['frames'], audio_lossless=False)


def new_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def output_paths(out, key_out):
    out, key_out = Path(out).absolute(), Path(key_out).absolute()
    for path in (out, key_out):
        if path != path.resolve() or not path.parent.is_dir() or os.path.lexists(path):
            raise RuntimeError('fresh canonical output with existing parent required; no overwrite')
        if path.is_relative_to(EXP) or path.is_relative_to(ROOT / 'H3-posttrain'):
            raise RuntimeError('keep public/key outputs outside experiment and repository')
    if key_out.is_relative_to(out) or out.is_relative_to(key_out):
        raise RuntimeError('private key must be outside public pack')
    return out, key_out


def build_pack(out, key_out):
    out, key_out = output_paths(out, key_out)
    verified, h3 = verify_complete()  # No destination writes before all nine are verified safe.
    ffmpeg = h3.ffmpeg_binary()
    families, cases = sorted(FAMILIES), list(verified['cases'])
    secrets.SystemRandom().shuffle(families)
    secrets.SystemRandom().shuffle(cases)
    bundle_id = secrets.token_hex(16)
    page = html_page(bundle_id)
    forbidden = [*FAMILIES, *IDS, str(ROOT), str(EXP)]
    for (family, sid), row in verified['videos'].items():
        forbidden.extend([row['video']['path'], row['case_audit']['path']])
    # Reserve the private key exclusively before the new public directory. Any failure
    # leaves partial new outputs unaccepted; no cleanup/overwrite of existing artifacts.
    key_fd = os.open(key_out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(key_fd, 'w', encoding='utf-8') as key_stream:
        out.mkdir(mode=0o700, exist_ok=False)
        public = dict(schema='anonymous-offline-video-review-v1', bundle_id=bundle_id, cases=[], axes=AXES)
        private = dict(schema='private-offline-video-review-key-v1', bundle_id=bundle_id,
                       audio_policy='AAC review track is lossy; complete original WAV copy is byte-identical',
                       receipts=verified['receipts'], cases=[])
        for index, case in enumerate(cases, 1):
            alias = 'case' + str(index)
            case_out = out / alias
            case_out.mkdir(mode=0o700)
            public_case = dict(case=alias, videos=[])
            private_case = dict(case=alias, id=case['id'], audio_ref=case['audio_ref'],
                                audio_header=case['audio_header'], videos=[])
            audio = Path(case['audio_ref']['path'])
            if source(audio) != case['audio_ref']:
                raise RuntimeError('verified original audio changed before copy')
            copied_audio = case_out / 'music.wav'
            audio_fd = os.open(copied_audio, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(audio_fd, 'wb') as destination_audio, audio.open('rb') as original_audio:
                shutil.copyfileobj(original_audio, destination_audio)
                destination_audio.flush(); os.fsync(destination_audio.fileno())
            copied_audio_ref = source(copied_audio)
            if (source(audio) != case['audio_ref']
                    or copied_audio_ref['sha256'] != case['audio_ref']['sha256']
                    or copied_audio_ref['size_bytes'] != case['audio_ref']['size_bytes']):
                raise RuntimeError('complete original WAV copy SHA differs')
            public_case['original_audio'] = dict(file=alias + '/music.wav', sha256=copied_audio_ref['sha256'])
            private_case['original_audio_copy'] = copied_audio_ref
            for label, family in zip('ABC', families):
                row = verified['videos'][family, case['id']]
                original = Path(row['video']['path'])
                if (source(original) != row['video'] or source(audio) != case['audio_ref']
                        or source(row['case_audit']['path']) != row['case_audit']):
                    raise RuntimeError('verified original video/audio changed before mux')
                destination = case_out / (label + '.mp4')
                command = mux_command(ffmpeg, original, audio, destination)
                subprocess.run(command, capture_output=True, text=True, check=True)
                media = verify_copy(original, destination, case['audio_header'], ffmpeg, h3, forbidden)
                if (source(original) != row['video'] or source(audio) != case['audio_ref']
                        or source(row['case_audit']['path']) != row['case_audit']):
                    raise RuntimeError('original video/audio changed during mux')
                copied_ref = source(destination)
                public_case['videos'].append(dict(label=label, file=alias + '/' + label + '.mp4',
                                                  sha256=copied_ref['sha256'], **media))
                private_case['videos'].append(dict(label=label, family=family, **row, copy=copied_ref,
                                                   mux_command=command, copy_verification=media))
            public['cases'].append(public_case)
            private['cases'].append(private_case)
        for receipt in verified['receipts']:
            if source(receipt['path']) != receipt:
                raise RuntimeError('verified receipt changed during copy')
        text = json.dumps(public, ensure_ascii=False, allow_nan=False) + page
        if any(token in text for token in forbidden):
            raise RuntimeError('identity in public text')
        json.dump(private, key_stream, ensure_ascii=False, allow_nan=False, indent=2)
        key_stream.write('\n'); key_stream.flush(); os.fsync(key_stream.fileno())
        new_json(out / 'manifest.json', public)
        fd = os.open(out / 'index.html', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(page)
            stream.flush(); os.fsync(stream.fileno())
    return dict(public_pack=str(out), private_key_written=True, videos=9)


def html_page(bundle_id):
    if not re.fullmatch(r'[a-f0-9]{32}', bundle_id):
        raise ValueError('neutral bundle ID required')
    return r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src 'self' file:; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>匿名视频人工评审</title>
<style>body{font:16px sans-serif;margin:20px;background:#171717;color:#eee}section{border-top:1px solid #777;margin:24px 0;padding:15px 0}.videos{display:flex;gap:12px}.videos article{flex:1;min-width:0}video{width:100%;aspect-ratio:704/1280;background:#000;object-fit:contain}button,select,input,textarea{font:inherit;margin:6px;padding:6px}textarea{width:90%;height:65px}.axis{display:inline-block;margin:8px}input[type=range]{width:55%}:fullscreen{width:100%;height:100%;object-fit:contain}</style>
<h1>匿名视频人工评审</h1>
<p>三个 case，每组 A/B/C 使用同一音乐。请共同播放、暂停和拖动进度；仅所选一路有声。评清晰度请分别全屏、按原分辨率观察。全部 124 帧，704×1280，24fps。视频内 AAC 音轨供评审播放，是有损编码；每组另存完整原 WAV（字节一致），未截取、未拉伸。音乐比视频略短属于原协议，不裁尾帧。先看视频再评分；可选相近、都不可用、无法判断，不强制选胜者。</p>
<p>此包只隐藏展示标签，不声称抵抗主动追踪源码或原目录。包标识用于将本地评分绑定对应解盲文件。</p>
<p>五轴独立比较：清晰（细节/手脸）、连续（时序稳定）、编排（动作结构）、幅度（运动强度）、音乐响应（节拍/变化对应）。此页完全离线，不上传、不自动保存；刷新会丢失未下载评分。下载 JSON 保存人工选择及理由，空白保留为未知。</p>
<label>评审标记（可留空）<input id="reviewer" autocomplete="off"></label><button id="download">下载本地评分 JSON</button><p id="status" role="status"></p><main id="cases"></main>
<script>
'use strict';
const bundleId='__BUNDLE_ID__';
const axes=[['clarity','清晰'],['continuity','连续'],['choreography','编排'],['amplitude','幅度'],['music_response','音乐响应']];
const choices=[['','未填写'],['A','A'],['B','B'],['C','C'],['similar','相近'],['all_unusable','都不可用'],['unknown','无法判断']];
const groups=[];
function option(value,text){const o=document.createElement('option');o.value=value;o.textContent=text;return o;}
for(let n=1;n<=3;n++){
 const name='case'+n,section=document.createElement('section');section.dataset.case=name;
 const title=document.createElement('h2');title.textContent=name;section.append(title);const music=document.createElement('a');music.href=name+'/music.wav';music.download=name+'-original.wav';music.textContent='下载完整原 WAV';section.append(music);
 const grid=document.createElement('div');grid.className='videos';section.append(grid);
 const videos=[];
 for(const label of ['A','B','C']){const card=document.createElement('article'),head=document.createElement('h3'),v=document.createElement('video'),full=document.createElement('button');head.textContent=label;v.src=name+'/'+label+'.mp4';v.preload='metadata';v.playsInline=true;v.muted=true;full.textContent='全屏 '+label;full.onclick=()=>v.requestFullscreen().catch(()=>{document.getElementById('status').textContent='浏览器不支持全屏，请放大窗口观察。';});card.append(head,v,full);grid.append(card);videos.push(v);}
 const play=document.createElement('button'),pause=document.createElement('button'),seek=document.createElement('input'),sound=document.createElement('select');play.textContent='共同播放';pause.textContent='共同暂停';seek.type='range';seek.min='0';seek.max=String(124/24);seek.step='0.01';seek.value='0';seek.setAttribute('aria-label',name+' 共同进度');for(const label of ['A','B','C'])sound.append(option(label,'声音 '+label));section.append(play,pause,seek,sound);
 const group={name,section,videos,seek,sound};groups.push(group);
 function stop(){videos.forEach(v=>v.pause());}
 play.onclick=async()=>{groups.forEach(g=>{if(g!==group)g.videos.forEach(v=>v.pause());});const t=Number(seek.value);videos.forEach((v,i)=>{v.currentTime=t;v.muted='ABC'[i]!==sound.value;});try{await Promise.all(videos.map(v=>v.play()));}catch(e){stop();document.getElementById('status').textContent='播放失败，请检查本地媒体文件和浏览器支持。';}};
 pause.onclick=stop;seek.oninput=()=>videos.forEach(v=>{v.currentTime=Number(seek.value);});sound.onchange=()=>videos.forEach((v,i)=>{v.muted='ABC'[i]!==sound.value;});
 videos[0].ontimeupdate=()=>{seek.value=String(videos[0].currentTime);videos.slice(1).forEach(v=>{if(Math.abs(v.currentTime-videos[0].currentTime)>0.08)v.currentTime=videos[0].currentTime;});};videos[0].onended=stop;
 for(const [key,label] of axes){const l=document.createElement('label'),s=document.createElement('select');l.className='axis';l.append(document.createTextNode(label+' '));s.dataset.axis=key;choices.forEach(([value,text])=>s.append(option(value,text)));l.append(s);section.append(l);}
 const reason=document.createElement('textarea');reason.dataset.reason='true';reason.placeholder='自由理由：可描述不可用、未知或不同维度的取舍';reason.setAttribute('aria-label',name+' 理由');section.append(document.createElement('br'),reason);document.getElementById('cases').append(section);
}
document.getElementById('download').onclick=()=>{
 const result={schema:'anonymous-offline-human-scores-v1',bundle_id:bundleId,reviewer:document.getElementById('reviewer').value||null,downloaded_at:new Date().toISOString(),cases:groups.map(g=>({case:g.name,axes:Object.fromEntries(axes.map(([key])=>[key,g.section.querySelector('[data-axis="'+key+'"]').value||null])),reason:g.section.querySelector('textarea').value||null}))};
 const blob=new Blob([JSON.stringify(result,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='anonymous-human-scores-'+bundleId+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
</script></html>'''.replace('__BUNDLE_ID__', bundle_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--key-out', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(build_pack(args.out, args.key_out)))
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError):
        # Do not print ffmpeg arguments or private source identities on failure.
        print('Review pack not completed; gate/copy failed. Existing originals untouched; do not use partial new outputs.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
