"""Add audited text-condition experiments to a focused review manifest without rescanning."""
import argparse
import copy
import json
from pathlib import Path
import time

from music_generation_validate import fixed_cases, atomic_write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.parent.is_dir():
        parser.error('Require an existing parent and new focused manifest')
    base = json.loads(args.base_manifest.read_text())
    manifest = copy.deepcopy(base)
    exp = Path(manifest['exp'])
    configurations = [
        ('text_legacy_fullclip_v1', ['joint', 'aswap'],
         '旧训练MusicFM+32Dadapter+池化+CLIPhead全部执行，单token文本注入；没有外部语义文本，原生两路静音；尺度来自train统计，非原生逐token对齐'),
        ('text_oracle_v1', ['joint', 'aswap'],
         '教师oracle：真实配对caption生成raw hidden回填，并非音乐encoder预测；joint原MS，aswap明确VLM fallback；只用于接口/可达效果诊断'),
        ('text_oracle_neutral_v1', ['joint'],
         '中性教师oracle：原MS仅去性别代词，未加学生ID/动作文字；原生静音；不是音乐预测，仅检验教师措辞/人物保真'),
    ]
    new_models = []
    for tag, conditions, description in configurations:
        group = 'H3AR/val_musicreplace_' + tag
        directory = exp / group
        marker = directory / ('all6_integrity.json' if len(conditions) == 2 else 'joint_integrity.json')
        integrity = json.loads(marker.read_text())
        if not integrity['complete']:
            raise ValueError('Incomplete video integrity report: ' + str(marker))
        entries = {}
        for condition in conditions:
            entries[condition] = {}
            for case in fixed_cases(exp):
                if case['condition'] != condition:
                    continue
                video = directory / condition / case['filename']
                if not video.is_file():
                    raise FileNotFoundError(video)
                audio_id = Path(case['audio_path']).stem
                item = [case['filename']]
                if audio_id != case['sample_id']:
                    item.append(audio_id)
                entries[condition][case['sample_id']] = item
        new_models.append(dict(id=group, label=group, group='H3AR', desc=description,
                               conds=entries, metrics={}))
    current = {model['id'] for model in manifest['models']}
    if current & {model['id'] for model in new_models}:
        raise ValueError('Focused manifest already contains new configurations')
    manifest['models'] += new_models
    new_ids = [model['id'] for model in new_models]
    manifest['presets'] = [
        dict(name='【10-18 文本侧·旧完整CLIP链】原生空文本 / 原MusicFM音频替换 / 旧encoder+adapter+head单token；joint和aswap，未新增训练',
             cond='joint', view='models', models=[
                 'H3AR/val_musicreplace_empty_native_h3_v1',
                 'H3AR/val_musicreplace_empty_musicfm_scale_d6000_l0_s1875_v4', new_ids[0]]),
        dict(name='【10-18 接口教师oracle·不是音乐预测】原版MS raw hidden / 中性代词raw hidden / 旧完整CLIP链；看人物和动作，音乐控制不能拿teacher证明',
             cond='joint', view='models', models=[new_ids[1], new_ids[2], new_ids[0]]),
    ] + manifest['presets']
    manifest['built'] = time.strftime('%Y-%m-%d %H:%M:%S')
    manifest['scope'] += ' New legacy-global-token text injection and explicitly labeled teacher oracles appended. Teacher captions are not music predictions; trained raw-token model is not yet present.'
    if manifest['samples'] != base['samples'] or len(manifest['samples']) != 3:
        raise ValueError('Original three source people or audio mapping changed')
    atomic_write_new(args.out, manifest)
    print(json.dumps(dict(complete=True, models=len(manifest['models']), source_people=3,
                          presets=len(manifest['presets']), original_models_preserved=True,
                          manifest=str(args.out))), flush=True)


if __name__ == '__main__':
    main()
