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
    mode.add_argument('--online_only', action='store_true', help='Append completed online A/B and their zero-input controls')
    mode.add_argument('--online_abc', action='store_true', help='Append completed A/B/C budget comparison and all zero-input controls')
    mode.add_argument('--length_ablation', action='store_true', help='Append C length-weight0.01 and its zero-input control')
    mode.add_argument('--behavior_only', action='store_true', help='Append audited six-update control/prediction-supervised music models')
    mode.add_argument('--projection_only', action='store_true', help='Append audited same-length projected teacher diagnostics, not music models')
    parser.add_argument('--online_steps', type=int, default=100)
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
    if args.online_only:
        if args.online_steps < 1:
            parser.error('Online update count must be positive')
        configurations = []
        for name, zero in (('A', False), ('B', False), ('A', True), ('B', True)):
            tag = f'text_online_{name}_s{args.online_steps}' + ('_zero' if zero else '') + '_v1'
            description = (f'在线{name} {args.online_steps}更新：' + ('实际MusicFM dense骨干+完整adapter共同训练' if name == 'B' else '在线冻结骨干，仅训练完整adapter的同协议对照')
                + '；学生全量raw H/自己预测有效长度→DiT文本入口，无语义文字/native真实音乐旁路；'
                + ('仅将学生音乐置零，仍配原曲，仅joint三条' if zero else '固定三人物joint/aswap，技术验收不代表舞蹈质量通过'))
            configurations.append((tag, ['joint'] if zero else ['joint', 'aswap'], description))
    if args.length_ablation:
        if args.online_steps < 1:
            parser.error('Online update count must be positive')
        configurations = [(f'text_online_C_len001_s{args.online_steps}' + ('_zero' if zero else '') + '_v1',
            ['joint'] if zero else ['joint', 'aswap'],
            f'在线C {args.online_steps}步：MusicFM dense骨干+完整adapter共同训练；仅把length loss权重1改为0.01，其余同B；'
            + ('学生音乐置零、仍配原曲，仅joint三条' if zero else '仅音乐+参考图，完整raw H文本侧部署，技术验收不代表音乐控制通过'))
            for zero in (False, True)]
    if args.online_abc:
        if args.online_steps < 1:
            parser.error('Online update count must be positive')
        configurations = []
        for name, zero in (('A', False), ('B', False), ('C_len001', False), ('A', True), ('B', True), ('C_len001', True)):
            tag = f'text_online_{name}_s{args.online_steps}' + ('_zero' if zero else '') + '_v1'
            description = f'在线{name} {args.online_steps}步：' + ('骨干冻结同协议对照' if name == 'A' else '骨干和完整adapter联训，长度权重0.01' if name == 'C_len001' else '骨干和完整adapter联训，长度权重1')
            description += '；仅音乐+参考图，学生自己预测raw H有效序列，原生音频固定静音；' + ('学生音乐置零，仍配原曲，仅joint三条' if zero else 'joint/aswap固定三人物，技术验收不代表音乐质量通过')
            configurations.append((tag, ['joint'] if zero else ['joint', 'aswap'], description))
    if args.behavior_only:
        configurations = []
        for name, zero in (('control', False), ('prediction', False), ('control', True), ('prediction', True)):
            tag = f'text_behavior_{name}_s6' + ('_zero' if zero else '') + '_v1'
            description = ('同C300起点，3train/6更新，骨干+完整adapter共同训练；'
                + ('新增冻结H3原124帧预测一致性loss1，显式条件VJP' if name == 'prediction' else '原表示loss对照，无H3预测loss')
                + '；学生仅音乐+参考图，自身长度和完整raw H，native固定静音；'
                + ('学生音乐置零、仍配原曲，仅joint三条' if zero else '原/换音乐各三条，技术验收不是舞蹈收益声明'))
            configurations.append((tag, ['joint'] if zero else ['joint', 'aswap'], description))
    if args.projection_only:
        configurations = [('text_projection_C300_' + name + '_v2', ['joint'],
            '教师输出空间投影诊断，不是音乐预测：' + label
            + '；同neutral教师token长度/图prefix/tags/原packed/native静音，零训练，不声明舞蹈质量通过')
            for name, label in [('w256', '当前W256宽松仿射包络'), ('ln255', '包含LayerNorm零均值约束的255维仿射包络'),
                                ('pca256', '仅1800train有效tail拟合的近似PCA256，全train中心')]]
    new_models = []
    for tag, conditions, description in configurations:
        group = 'H3AR/val_musicreplace_' + tag
        directory = exp / group
        marker = directory / ('projection_integrity.json' if args.projection_only else 'all6_integrity.json' if len(conditions) == 2 else 'joint_integrity.json')
        integrity = json.loads(marker.read_text())
        if integrity.get('complete') is not True:
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
        if args.online_only or args.length_ablation or args.online_abc or args.behavior_only:
            final_report = Path(integrity.get('final_online_report', ''))
            if not final_report.is_file():
                raise ValueError('Online review requires completed all18 input/audio/video audit')
            final = json.loads(final_report.read_text())
            expected_count = 27 if args.online_abc else 9 if args.length_ablation else 18
            if (integrity.get('complete') is not True or final.get('complete') is not True
                    or final.get('all_native_audio_hashes_identical') is not True
                    or type(final.get('passed')) is not int or final['passed'] != expected_count
                    or type(final.get('expected')) is not int or final['expected'] != expected_count
                    or not isinstance(final.get('rows'), list) or len(final['rows']) != expected_count
                    or any(not isinstance(row, dict) for row in final['rows'])
                    or {row.get('tag') for row in final['rows']} != {entry[0] for entry in configurations}):
                raise ValueError('Online technical audit is incomplete')
            group_rows = [row for row in final.get('rows', []) if row.get('tag') == tag]
            expected_rows = {(case['condition'], case['sample_id'], str(directory / case['condition'] / case['filename']), case['audio_path'])
                             for case in fixed_cases(exp) if case['condition'] in conditions}
            actual_rows = {(row.get('condition'), row.get('sample_id'), row.get('video_path'), row.get('audio_path'))
                           for row in group_rows}
            count = 3 * len(conditions)
            if (len(group_rows) != count or integrity.get('rows') != group_rows or actual_rows != expected_rows
                    or integrity.get('expected') != count or integrity.get('passed') != count):
                raise ValueError('Online group marker does not match the displayed fixed cases and completed counts')
            name = 'C' if args.length_ablation or '_C_len001_' in tag else 'A' if '_A_' in tag else 'B'
            if any(row.get('backbone_train') is not (name != 'A')
                   or row.get('zero_input') is not ('_zero_' in tag) for row in group_rows):
                raise ValueError('Online displayed training scope or zero-input variant differs from the audited videos')
            record['label'] = f'在线{name}·' + ('长度权重0.01' if name == 'C' else '骨干联合训练' if name == 'B' else '骨干冻结对照') + f'·{args.online_steps}步' + ('·零音乐' if '_zero_' in tag else '')
            record.update(musical_quality_accepted=False, online_backbone_train=name != 'A')
            if args.behavior_only:
                variant = 'prediction' if '_prediction_' in tag else 'control'
                checkpoint_dir = exp / 'MS' / f'music_text_behavior_{variant}_s6_v1_step6'
                summary = json.loads((checkpoint_dir / 'finished.json').read_text())
                if (summary.get('complete') is not True or summary.get('steps') != 6
                        or summary['protocol'].get('backbone_train') is not True
                        or summary['protocol'].get('behavior_weight') != float(variant == 'prediction')
                        or any(row['inputs'].get('music_text_ckpt') != str(checkpoint_dir / 'last.pt')
                               or row['deployment'].get('checkpoint_sha256') != summary['checkpoint_sha256'] for row in group_rows)):
                    raise ValueError('Behavior supervision variant is not bound to the audited full model')
                record['label'] = ('预测监督loss1' if variant == 'prediction' else '原表示loss对照') + '·6更新' + ('·零音乐' if '_zero_' in tag else '')
                record['frozen_h3_prediction_loss_weight'] = float(variant == 'prediction')
        if args.projection_only:
            final_path = Path(integrity['final_projection_report'])
            final = json.loads(final_path.read_text())
            variant = next(name for name in ('w256', 'ln255', 'pca256') if '_' + name + '_' in tag)
            rows = [row for row in final['rows'] if row['variant'] == variant]
            expected_rows = {(case['sample_id'], str(directory / 'joint' / case['filename']), case['audio_path'])
                             for case in fixed_cases(exp) if case['condition'] == 'joint'}
            if (final.get('complete') is not True or final.get('expected') != 12 or final.get('passed') != 12
                    or final.get('passed_new') != 9 or final.get('baseline_reused') != 3
                    or final.get('same_teacher_lengths_and_original_packed_verified') is not True
                    or final.get('all_native_silence_hashes_identical') is not True
                    or len(final['rows']) != 12 or len(rows) != 3 or integrity.get('expected') != 3
                    or integrity.get('passed') != 3 or integrity.get('rows') != rows
                    or {(row['sample_id'], row['video_path'], row['audio_path']) for row in rows} != expected_rows
                    or any(row['projection'].get('original_packed_exact') is not True
                           or row['projection'].get('variant') != variant or row['projection'].get('music_prediction') is not False
                           for row in rows)):
                raise ValueError('Projection review is not bound to completed same-layout teacher cases')
            record.update(label='教师空间·' + variant, diagnostic_only=True, music_prediction=False,
                          teacher_length_for_diagnostic_only=True, musical_quality_accepted=False)
        new_models.append(record)
    current = {model['id'] for model in manifest['models']}
    if current & {model['id'] for model in new_models}:
        raise ValueError('Focused manifest already contains new configurations')
    manifest['models'] += new_models
    new_ids = [model['id'] for model in new_models]
    if args.projection_only:
        teacher = 'H3AR/val_musicreplace_text_oracle_neutral_v1'
        if teacher not in current:
            raise ValueError('Projection review must retain the existing neutral teacher baseline')
        presets = [dict(name='【教师输出空间·零训练同长度】原neutral教师 / 当前W256 / LN255 / train-PCA256；仅joint，不是音乐预测',
                        cond='joint', view='models', models=[teacher, *new_ids]),
                   dict(name='【先复审教师舞蹈上界】原neutral教师 / C300 / 六步control / 六步prediction；仅joint',
                        cond='joint', view='models', models=[teacher, 'H3AR/val_musicreplace_text_online_C_len001_s300_v1',
                            'H3AR/val_musicreplace_text_behavior_control_s6_v1', 'H3AR/val_musicreplace_text_behavior_prediction_s6_v1'])]
    elif args.behavior_only:
        presets = [dict(name='【冻结H3预测监督·6更新】原C300 / 原表示loss对照 / 预测loss1；先joint，再aswap',
                       cond='joint', view='models', models=['H3AR/val_musicreplace_text_online_C_len001_s300_v1', new_ids[0], new_ids[1]]),
                   dict(name='【冻结H3预测监督·零音乐】对照 / 对照零音乐 / 预测loss1 / 预测loss1零音乐；仅joint',
                       cond='joint', view='models', models=[new_ids[0], new_ids[2], new_ids[1], new_ids[3]])]
    elif args.online_abc:
        presets = [dict(name=f'【在线骨干预算·{args.online_steps}步】A骨干冻结 / B骨干联训length1 / C骨干联训length0.01；先joint后aswap',
                       cond='joint', view='models', models=new_ids[:3]),
                   dict(name=f'【在线骨干预算·{args.online_steps}步·零音乐】B / B零音乐 / C / C零音乐；仅joint',
                       cond='joint', view='models', models=[new_ids[1], new_ids[4], new_ids[2], new_ids[5]])]
    elif args.length_ablation:
        baseline = f'H3AR/val_musicreplace_text_online_B_s{args.online_steps}_v1'
        if baseline not in current:
            baseline = 'H3AR/val_musicreplace_text_trained1800_v2'
        presets = [dict(name=f'【在线骨干·长度监督单变量·{args.online_steps}步】已有基线 / C长度权重0.01 / C零音乐；先joint再aswap，零音乐仅joint',
                    cond='joint', view='models', models=[baseline, new_ids[0], new_ids[1]])]
    elif args.online_only:
        presets = [dict(name=f'【在线骨干联训·{args.online_steps}步】旧MSE / A骨干冻结 / B骨干联训 / B零音乐；先joint，再aswap，零音乐仅joint',
                    cond='joint', view='models', models=['H3AR/val_musicreplace_text_trained1800_v2', new_ids[0], new_ids[1], new_ids[3]]),
                   dict(name=f'【在线骨干联训·{args.online_steps}步·零音乐对照】A / A零音乐 / B / B零音乐；仅joint，比较音乐作用和人物保真',
                    cond='joint', view='models', models=[new_ids[0], new_ids[2], new_ids[1], new_ids[3]])]
    elif args.diagnostic_only:
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
    if args.projection_only:
        manifest['scope'] += ' Same-length teacher projection diagnostics and reused neutral teacher added after full12 audit; no music-model update or continuous human acceptance claim.'
    elif args.behavior_only:
        manifest['scope'] += ' Six-update representation-only control and frozen-H3 prediction-supervised full music models appended after all18 video audit; small fitting or condition gradients do not establish choreography gain.'
    elif args.online_abc:
        manifest['scope'] += ' A/B/C budget comparison and actual zero-input controls appended after all27 technical audit; musical gain is not established by representation scores.'
    elif args.length_ablation:
        manifest['scope'] += ' C length-weight0.01 single-variable backbone/adapter run and actual zero-input control appended after all9 technical audit; no established musical-quality claim.'
    elif args.online_only:
        manifest['scope'] += ' Online A-frozen/B-dense-backbone full-adapter models and actual zero-input controls appended after all18 technical audit; no established musical-quality claim.'
    elif args.diagnostic_only:
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
