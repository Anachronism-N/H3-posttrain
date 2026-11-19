import ast
import hashlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
from rejected_beats import ARRAY_CAP, FLAGS, rejection_predicates, validate_and_record

CORE = Path('/tmp/ms_pose2d_draft_20261026T1427Z/launcher/unified_code_snapshot_v1/ms_unified_novlm_v1.py')
raw = CORE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '85ea6781a433a7f3cfe50fa074f56e7d7c291a815dfbf7b0220d4eb9caf8dda9'
tree = ast.parse(raw)
nodes = [n for n in ast.walk(tree) if isinstance(n, ast.If) and any(
    isinstance(v, ast.Constant) and v.value == 'Invalid actual BeatThis beat/downbeat output'
    for body in n.body for v in ast.walk(body))]
assert len(nodes) == 1
condition = compile(ast.Expression(nodes[0].test), str(CORE), 'eval')
context = dict(source_row=dict(id='synthetic_contract', split='train', audio_path='/synthetic/audio.wav'),
               worker='a100-extra-03', gpu=2, assignment_index=10,
               assignment_sha256='1'*64, algorithm_sha256=hashlib.sha256(raw).hexdigest(),
               model_sha256='2'*64, device='cuda:0', float16=False, dbn=False)
cases = [
    ('empty', [], [], False),
    ('valid', [.5, 1., 1.5], [.5, 1.5], False),
    ('audio_tolerance', [5.02], [5.02], False),
    ('membership_tolerance', [.5], [.5000005], False),
    ('scalar', np.array(.5, dtype=np.float32), [], True),
    ('2d', np.array([[.5, 1.]], dtype=np.float32), [], True),
    ('nan', np.array([np.nan], dtype=np.float32), [], True),
    ('inf', [.5], np.array([np.inf], dtype=np.float64), True),
    ('negative', [-.1], [], True),
    ('end_range', [5.021], [], True),
    ('duplicate', [.5, .5], [], True),
    ('reverse', [1., .5], [], True),
    ('duplicate_downbeats', [.5, 1.], [.5, .5], True),
    ('missing_member', [.5, 1.], [.7], True),
    ('outside_membership_tolerance', [.5], [.500002], True),
    ('downbeat_without_beats', [], [.5], True),
    ('multifailure', [-1., -1., 6., np.nan], [-2., 6.], True),
]
reports = []
with tempfile.TemporaryDirectory(prefix='ms-rejected-beats-tests-') as temporary:
    root = Path(temporary)
    for name, beats, downbeats, rejected in cases:
        rb, rd = np.asarray(beats), np.asarray(downbeats)
        cb, cd = (np.asarray(v, dtype=float) for v in (rb, rd))
        original_rejected = bool(eval(condition, dict(np=np, beats=cb, downbeats=cd, audio_duration=5.)))
        assert original_rejected == rejected == bool(rejection_predicates(cb, cd, 5.))
        p = root/name
        p.mkdir()
        if not rejected:
            b, d = validate_and_record(beats, downbeats, 5., p, context)
            np.testing.assert_array_equal(b, cb)
            np.testing.assert_array_equal(d, cd)
            assert not list(p.iterdir())
        else:
            try:
                validate_and_record(beats, downbeats, 5., p, context)
            except ValueError as e:
                assert str(e) == 'Invalid actual BeatThis beat/downbeat output'
            else:
                raise AssertionError('Refusal disappeared: ' + name)
            archive = (p/'rejected_beats.npz').read_bytes()
            raw_metadata = (p/'rejection.json').read_bytes()
            assert raw_metadata.endswith(bytes([10])) and bytes([0]) not in raw_metadata
            record = json.loads(raw_metadata)
            assert record['predicates'] == rejection_predicates(cb, cd, 5.)
            assert record['archive']['sha256'] == hashlib.sha256(archive).hexdigest()
            assert record['archive']['bytes'] == len(archive)
            assert record['audio_duration_seconds'] == 5.
            assert all(record[k] == v for k, v in context.items())
            assert all(record[k] == v for k, v in FLAGS.items())
            with np.load(io.BytesIO(archive), allow_pickle=False) as z:
                for key, expected in [('beats', rb), ('downbeats', rd)]:
                    assert z[key].shape == expected.shape and z[key].dtype == expected.dtype
                    np.testing.assert_array_equal(z[key], expected)
            before = {f.name:f.read_bytes() for f in p.iterdir()}
            try:
                validate_and_record(beats, downbeats, 5., p, context)
            except FileExistsError:
                pass
            else:
                raise AssertionError('Existing refusal evidence overwritten')
            assert before == {f.name:f.read_bytes() for f in p.iterdir()}
        reports.append(name)
    rng = np.random.default_rng(20261027)
    for i in range(300):
        beats = rng.uniform(-1., 6., size=int(rng.integers(0, 20))).astype(np.float32)
        if i%3 == 0:
            beats.sort()
        downbeats = beats[::4].copy() if i%2 else rng.uniform(-1., 6., size=3)
        if i%11 == 0 and beats.size:
            beats[0] = np.nan
        cb, cd = (np.asarray(v, dtype=float) for v in (beats, downbeats))
        expected = bool(eval(condition, dict(np=np, beats=cb, downbeats=cd, audio_duration=5.)))
        assert bool(rejection_predicates(cb, cd, 5.)) == expected
    reports.append('300_randomized_frozen_predicate_equivalence')
    # Existing conflicting evidence is never overwritten, regardless of incoming contents.
    p = root/'preexisting';p.mkdir();(p/'rejected_beats.npz').write_bytes(b'existing evidence')
    try:validate_and_record([.5, .5], [], 5., p, context)
    except FileExistsError:pass
    else:raise AssertionError('Conflicting existing evidence accepted')
    assert (p/'rejected_beats.npz').read_bytes() == b'existing evidence'
    assert not (p/'rejection.json').exists()
    reports.append('conflicting_evidence_preserved')
    p = root/'budget';p.mkdir()
    try:validate_and_record(np.zeros(ARRAY_CAP//8+1), [], 5., p, context)
    except ValueError as e:assert 'budget' in str(e)
    else:raise AssertionError('Unbounded refusal evidence accepted')
    assert not list(p.iterdir());reports.append('evidence_budget_refused_no_output')
    p = root/'wrong_binding';p.mkdir();bad = dict(context, assignment_index=11)
    try:validate_and_record([.5, .5], [], 5., p, bad)
    except ValueError as e:assert 'binding' in str(e)
    else:raise AssertionError('Wrong GPU row binding accepted')
    assert not list(p.iterdir());reports.append('wrong_GPU_binding_refused_no_output')
    p = root/'redirect';p.symlink_to(root/'budget', target_is_directory=True)
    try:validate_and_record([.5, .5], [], 5., p, context)
    except ValueError as e:assert 'canonical' in str(e)
    else:raise AssertionError('Redirected evidence directory accepted')
    reports.append('symlink_directory_refused')
    p = root/'duration';p.mkdir()
    for duration in [0., -1., np.nan, np.inf]:
        try:validate_and_record([.5], [.5], duration, p, context)
        except ValueError as e:assert str(e) == 'Actual audio duration must be finite and positive'
        else:raise AssertionError('Invalid duration accepted')
    assert not list(p.iterdir());reports.append('original_invalid_audio_duration_refusal')
print(json.dumps(dict(passed=True,checks=reports,checks_count=len(reports),randomized_predicate_cases=300,
                     original_algorithm_SHA256=hashlib.sha256(raw).hexdigest(),
                     no_models_GPU_media_or_claim_calls=True,production_deployed=False,
                     old_failures_reexecuted=False,old_failure_root_cause_fixed=False)), flush=True)
