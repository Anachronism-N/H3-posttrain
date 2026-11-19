"""Package one explicitly authorized RAQ offline visual revision with original pilot style.

No new VLM generation, no original file changes, no human approval fabrication.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys

RT = Path(__file__).resolve().parent
ROOT = RT.parent
SID = 'RAQoxqRxCfMvTHSx'
EDITION = 'raq-scope-offline-20261025-v3'
VISUAL_SHA = '55efb59a85811d1033b286184941a03e8f6fe489384c51da991ea608f376ab01'
VISUAL_PATH = ROOT / 'legacy_raq_repair_v1/revisions' / EDITION / 'train' / SID / 'result.json'
OUT = ROOT / 'curated_review_extra_visual_v1'
spec = importlib.util.spec_from_file_location('extra_watch_protocol', RT / 'curated_review_extension_watch_v1.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
sys.path.insert(0, str(RT / 'source_curated_review_v2'))
import ms_batch_review as review


def identity(path):
    return dict(path=str(path), sha256=w.sha(path))


def save_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    if path.exists():
        w.need(path.read_bytes() == data, 'existing extra joint record changed')
    else:
        with path.open('xb') as f:
            f.write(data)


def main():
    pilot = RT / 'independent_review_pilot25_rows_v1.jsonl'
    pilot_ref = identity(pilot)
    originals = [json.loads(l) for l in pilot.read_text().splitlines() if l]
    originals = [r for r in originals if r['id'] == SID]
    w.need(len(originals) == 1, 'one original pilot RAQ source/style required')
    original = originals[0]
    visual_ref = dict(path=str(VISUAL_PATH), sha256=VISUAL_SHA)
    visual = w.checked(visual_ref)
    w.need(visual['id'] == SID and visual['global_revision'] == EDITION
           and visual['final']['status'] == 'deterministic_derived_candidate'
           and visual['all_windows_nonempty'] is True and visual['final']['model_call_performed'] is False
           and visual['final']['model_rewrite_performed'] is False, 'explicit zero-new-VLM RAQ visual candidate required')
    old = w.checked(visual['original_result_ref'])
    for k in ('id', 'split', 'video_path', 'audio_path'):
        w.need(old['manifest_record'][k] == original[k], 'RAQ original video/source mismatch')
    windows = [o['window'] for o in visual['window_observations']]
    captions = visual['final']['json']['windows']
    w.need(len(windows) == len(captions) == 3 and all(wi['window_id'] == ci['window_id'] for wi, ci in zip(windows, captions)),
           'all three fixed RAQ windows required')
    timed = review.core.timed_text(dict(windows=windows), captions)
    payload = review.label_payload(original)
    w.need(payload['dance_style'] and payload['dance_style'] != 'unknown', 'original pilot style required')
    lines = payload['timed_motion_script'].splitlines()
    newlines = timed.splitlines()
    w.need(len(lines) == len(newlines) == 3 and lines[0] == newlines[0] and lines[2] == newlines[2],
           'RAQ unchanged w0/w2 labels must remain verbatim')
    root = OUT / 'joint-revisions' / EDITION / 'train' / SID
    source = root / 'source_record.json'
    save_once(source, original)
    row = copy.deepcopy(original)
    row.update(schema='independent_extra_visual_joint_v1', label=dict(dance_style=payload['dance_style'], timed_motion_script=timed),
        target_text='Dance style: ' + payload['dance_style'] + '\n' + timed,
        label_status='completed', joint_complete=True, all_windows_nonempty=True,
        visual_status='deterministic_derived_candidate', global_revision=EDITION,
        extra_visual_revision_ref=visual_ref, original_visual_result_ref=visual['original_result_ref'],
        original_source_binding=visual['original_source_binding'],
        raw_rgb_manifest=old.get('raw_rgb_manifest'), hand_roi=old.get('hand_roi'),
        original_pilot_rows_ref=pilot_ref, original_source_record_ref=identity(source),
        source_row_sha256=review.batch.digest(original), semantic_label_sha256=review.batch.digest(dict(dance_style=payload['dance_style'], timed_motion_script=timed)),
        risk_review_case=True, style_label_changed=False, model_rewrite_performed=False,
        new_model_calls=0, unchanged_window_ids=['w00000', 'w00002'],
        unchanged_windows_old_review_scores_reused=False, **w.FLAGS)
    row['source_refs']['independent_extra_visual_lineage'] = dict(original_pilot=pilot_ref,
        original_source_record=identity(source), offline_visual=visual_ref, original_visual=visual['original_result_ref'])
    save_once(root / 'result.json', row)
    notice = dict(schema='independent_extra_visual_joint_v1', id=SID, split='train', source_key='train/' + SID,
        batch_id=original['batch_id'], global_revision=EDITION, result_ref=identity(root / 'result.json'),
        source_record_ref=identity(source), visual_revision_ref=visual_ref, risk_review_case=True, **w.FLAGS)
    stream = OUT / 'review-required.jsonl'
    data = (json.dumps(notice, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if stream.exists():
        w.need(stream.read_bytes() == data, 'extra review stream differs; do not overwrite')
    else:
        with stream.open('xb') as f:
            f.write(data)
    w.need(w.sha(pilot) == pilot_ref['sha256'], 'pilot source changed during packaging')
    print(json.dumps(dict(extra_review_stream=str(stream), result_ref=notice['result_ref'],
        semantic_label_sha256=row['semantic_label_sha256'], original_style_preserved=True,
        original_w0_w2_preserved=True, new_model_calls=0, human_approval_automatically_granted=False)), flush=True)


if __name__ == '__main__':
    main()
