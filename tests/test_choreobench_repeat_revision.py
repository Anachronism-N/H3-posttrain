import json
from choreobench import repeat_revision as rr
from choreobench.baseline import _empty_plan

def song():
    return dict(schema='choreobench.plan.v1', song_id='s1', sample_rate=16000, num_samples=320000,
                provenance=dict(method='test'), motifs=[dict(motif_id='m0', description='travel')], events=[
        dict(event_id='e0', start=0, end=80000, track='body', action='Arm sweeps left and right.', motif_id='m0', evidence=[], attributes={}, entry={}, exit={}),
        dict(event_id='e1', start=80000, end=160000, track='body', action='arm sweeps LEFT and right', motif_id='m0', evidence=[], attributes={}, entry={}, exit={}),
        dict(event_id='e2', start=160000, end=240000, track='body', action='Knee bends slowly.', motif_id='m0', evidence=[], attributes={}, entry={}, exit={}),
    ])

def test_groups_detect_normalised_and_near_duplicates():
    groups = rr.duplicate_groups(song())
    assert len(groups) == 1 and groups[0]['event_ids'] == ['e0', 'e1']

def test_read_rewrites_rejects_unchanged_and_missing():
    group = rr.duplicate_groups(song())[0]
    accepted, skipped = rr.read_rewrites(
        {'rewrites': [dict(event_id='e1', action='Arm sweeps overhead then down.')]}, group, min_overlap=0.2)
    assert accepted[0]['event_id'] == 'e1' and not skipped
    accepted, skipped = rr.read_rewrites(
        {'rewrites': [dict(event_id='e1', action='Knee bends and the torso rotates.')]}, group)
    assert not accepted and skipped[0]['reason'] == 'vocabulary_drift'
    empty, skipped = rr.read_rewrites({'rewrites': []}, group)
    assert not empty and not skipped
    for bad in ({'rewrites': [dict(event_id='e0', action='x')]},):
        try:
            rr.read_rewrites(bad, group)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted bad rewrite: %r' % bad)

def test_apply_rewrites_changes_only_actions():
    plan = song(); group = rr.duplicate_groups(plan)[0]
    rewrites = [dict(event_id='e1', action='Arm sweeps overhead then folds down.')]
    assert rr.overlap('Arm sweeps overhead then folds down.', 'Arm sweeps left and right.') > 0.2
    final, changed = rr.apply_rewrites(plan, rewrites)
    assert changed == ['e1'] and final['events'][1]['action'].startswith('Arm sweeps overhead')
    assert [(e['start'], e['end']) for e in final['events']] == [(e['start'], e['end']) for e in plan['events']]
    assert rr.summary(plan, final)['duplicate_rate_after'] < rr.summary(plan, final)['duplicate_rate_before']


def test_near_duplicate_rewrite_is_rejected():
    plan = song(); group = rr.duplicate_groups(plan)[0]
    accepted, skipped = rr.read_rewrites(
        {'rewrites': [dict(event_id='e1', action='Arm sweeps left and right again.')]}, group, min_overlap=0.2)
    assert not accepted and skipped[0]['reason'] == 'near_duplicate'
