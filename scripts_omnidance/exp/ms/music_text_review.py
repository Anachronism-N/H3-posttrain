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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--trained_only', action='store_true', help='Append completed raw-token and zero-music model to the existing text review')
    mode.add_argument('--diagnostic_only', action='store_true', help='Append unchanged native-H CLIP candidate and zero-input diagnostic; keep quality gate false')
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
    if args.trained_only:
        configurations = [
            ('text_trained1800_v2', ['joint', 'aswap'],
             '正式MusicFM冻结骨干+raw文本token adapter，1800/1875更新；仅音乐+参考图，两路原生静音；非teacher，文本均值/错配审计尚未建立音乐收益'),
            ('text_trained1800_zero_v2', ['joint'],
             '同一正式adapter只把音乐输入置零；参考图/seed/native静音不变，输出视频仍配原音轨；仅joint三条，非动作条件文本'),
        ]
    if args.diagnostic_only:
        configurations = [
            ('text_Hclip_realneg_diag_v1', ['joint', 'aswap'],
             '上一轮双真实音乐负例H序列CLIP原权重直接推理；未校准、不重训；原质量门槛false，仅诊断；完整rawH进入DiT文本入口，原生两路静音'),
            ('text_Hclip_realneg_diag_zero_v1', ['joint'],
             '同CLIP候选仅将音乐输入置零；三参考图/seed/固定native静音不变，按生成路径配原音轨（未独立验证mux逐值匹配）；未校准模型诊断，不代表验收成功，仅joint三条'),
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
        record = dict(id=group, label=group, group='H3AR', desc=description, conds=entries, metrics={})
        if args.diagnostic_only:
            final_report = Path(integrity.get('final_diagnostic_report', ''))
            if not final_report.is_file():
                raise ValueError('Diagnostic group cannot be reviewed before the final nine-video report exists')
            final = json.loads(final_report.read_text())
            if (integrity.get('diagnostic_only') is not True or integrity.get('quality_gate_passed') is not False
                    or not final.get('complete') or final.get('passed') != 9
                    or final.get('quality_gate_passed') is not False):
                raise ValueError('Candidate diagnostics must preserve the failed quality gate and complete nine-video audit')
            record.update(diagnostic_only=True, quality_gate_passed=False)
        new_models.append(record)
    current = {model['id'] for model in manifest['models']}
    if current & {model['id'] for model in new_models}:
        raise ValueError('Focused manifest already contains new configurations')
    manifest['models'] += new_models
    new_ids = [model['id'] for model in new_models]
    if args.diagnostic_only:
        presets = [dict(name='【H序列CLIP·训练后直接推理】MSE基线 / 原权重CLIP候选 / 候选零音乐；未校准、原生音频均静音；joint后切aswap，零音乐仅joint',
                        cond='joint', view='models', models=[
                            'H3AR/val_musicreplace_text_trained1800_v2', new_ids[0], new_ids[1]])]
    elif args.trained_only:
        presets = [dict(name='【10-18 正式文本token·仅音乐+参考图】原生空文本 / 旧完整CLIP链 / 新rawtoken / 新rawtoken零音乐；先joint，再aswap，零音乐仅joint',
                        cond='joint', view='models', models=[
                            'H3AR/val_musicreplace_empty_native_h3_v1',
                            'H3AR/val_musicreplace_text_legacy_fullclip_v1', new_ids[0], new_ids[1]])]
    else:
        presets = [
        dict(name='【10-18 文本侧·旧完整CLIP链】原生空文本 / 原MusicFM音频替换 / 旧encoder+adapter+head单token；joint和aswap，未新增训练',
             cond='joint', view='models', models=[
                 'H3AR/val_musicreplace_empty_native_h3_v1',
                 'H3AR/val_musicreplace_empty_musicfm_scale_d6000_l0_s1875_v4', new_ids[0]]),
        dict(name='【10-18 接口教师oracle·不是音乐预测】原版MS raw hidden / 中性代词raw hidden / 旧完整CLIP链；看人物和动作，音乐控制不能拿teacher证明',
             cond='joint', view='models', models=[new_ids[1], new_ids[2], new_ids[0]]),
        ]
    manifest['presets'] = presets + manifest['presets']
    manifest['built'] = time.strftime('%Y-%m-%d %H:%M:%S')
    if args.diagnostic_only:
        manifest['scope'] += ' Unchanged previous native-H CLIP candidate and zero-music diagnostic appended; original failed quality gate is unchanged, not a model acceptance claim.'
    else:
        manifest['scope'] += (' Completed trained raw-token and zero-music controls appended; no claim of established musical gain.'
                              if args.trained_only else ' New legacy-global-token text injection and explicitly labeled teacher oracles appended. Teacher captions are not music predictions.')
    if manifest['samples'] != base['samples'] or len(manifest['samples']) != 3:
        raise ValueError('Original three source people or audio mapping changed')
    atomic_write_new(args.out, manifest)
    print(json.dumps(dict(complete=True, models=len(manifest['models']), source_people=3,
                          presets=len(manifest['presets']), original_models_preserved=True,
                          manifest=str(args.out))), flush=True)


if __name__ == '__main__':
    main()
