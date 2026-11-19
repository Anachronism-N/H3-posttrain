"""Targeted CPU-only real bad-line, terminal, half-line and cached-revision checks."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

RT = Path(__file__).resolve().parent
ROOT = RT.parent
spec = importlib.util.spec_from_file_location('verified_review_stream_smoke', RT/'curated_review_watch_v4.py')
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)
sys.path.insert(0, str(RT/'source_curated_review_v2'))
import ms_batch_review as review


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--receipt', required=True)
    args = p.parse_args()
    production = ROOT/'curated222939_production_v1'
    membership = watch.Membership(ROOT/'curated222939_inputs_v1/manifest.summary.json')
    originals = membership.batch('train-b000001')
    assert len(originals) == 500
    with (production/'completed.jsonl').open('rb') as source:
        source.seek(532233)
        raw = source.readline()
    try:
        json.loads(raw)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError('Real source bad line must remain malformed')
    terminals = [watch.evidence.load(production/'sources/train'/sid/'consumer.terminal.json') for sid in ('MVuipekTKQDQprcC', 'ytykxzHnSDAecSEP')]
    rows = [watch.terminal_record(production, row, membership) for row in terminals]
    assert all(row['joint_complete'] is True for row in rows)
    chunks = [watch.evidence.enc(row) for row in terminals]
    plan_ref = {'path': 'fixture-only', 'sha256': 'fixture-only'}
    results = {}
    with tempfile.TemporaryDirectory(prefix='main-review-v4-stream-smoke-') as tmp:
        stream = Path(tmp)/'completed.shadow.jsonl'
        stream.write_bytes(raw+b''.join(chunks))
        state = {'shadow_offset': stream.stat().st_size, 'shadow_sha256': watch.sha(stream)}
        with patch.object(watch, 'prepared_stream', side_effect=lambda *unused: (plan_ref, dict(state))):
            reader = watch.VerifiedReader(stream, production, membership, plan_ref)
            assert reader.poll() == set() and reader.offset == 0 and reader.pending['skipped'] is False
            assert sum(len(x) for x in reader.completed.values()) == 0
            results['real_bad_line_pending_without_skip'] = True
        stream.write_bytes(b''.join(chunks)+b'{"unfinished":')
        state.update(shadow_offset=stream.stat().st_size, shadow_sha256=watch.sha(stream))
        with patch.object(watch, 'prepared_stream', side_effect=lambda *unused: (plan_ref, dict(state))):
            reader = watch.VerifiedReader(stream, production, membership, plan_ref)
            assert reader.poll() == {'train-b000001'}
            assert len(reader.completed['train-b000001']) == 2 and reader.offset == sum(map(len, chunks))
            assert reader.pending['reason'] == 'incomplete committed line'
            assert reader.poll() == set() and reader.offset == sum(map(len, chunks))
            results['two_real_terminal_refs_reader_verified'] = True
            results['half_line_remains_pending'] = True
            value = watch.progress(membership, reader, [])
            assert value['all_source_labels_completed'] is False and value['all_batches_reviewed'] is False
            # Same-inode rewriting with a different state SHA must not hide a prior-prefix mutation.
            altered = stream.read_bytes().replace(b'train', b'traix', 1)
            stream.write_bytes(altered)
            state.update(shadow_sha256=watch.sha(stream))
            assert reader.poll() == set() and 'Previously verified shadow bytes changed' in reader.pending['reason']
            results['previous_committed_prefix_mutation_rejected'] = True
        stream.write_bytes(b''.join(chunks))
        state.update(shadow_offset=stream.stat().st_size, shadow_sha256=watch.sha(stream))
        with patch.object(watch, 'prepared_stream', side_effect=lambda *unused: (plan_ref, dict(state))):
            reader = watch.VerifiedReader(stream, production, membership, plan_ref)
            assert reader.poll() == {'train-b000001'} and reader.pending is None
            results['both_recovered_sources_not_lost'] = len(reader.completed['train-b000001']) == 2
    with tempfile.TemporaryDirectory(prefix='additional-edition-review-smoke-') as tmp:
        other = Path(tmp)/'production'
        other.mkdir()
        stream = other/'completed.jsonl'
        derived = []
        for terminal in terminals:
            joint = watch.evidence.checked(terminal['joint_candidate_ref'])
            joint_path = other/'joint'/terminal['id']/'candidate.json'
            joint_path.parent.mkdir(parents=True)
            joint_path.write_bytes(joint)
            value = copy.deepcopy(terminal)
            value['joint_candidate_ref'] = watch.evidence.ref(joint_path)
            terminal_path = other/'sources'/value['split']/value['id']/'consumer.terminal.json'
            terminal_path.parent.mkdir(parents=True)
            terminal_path.write_bytes(watch.evidence.enc(value))
            derived.append(watch.evidence.enc(value))
        stream.write_bytes(b''.join(derived)+b'partial')
        extra_reader = watch.AdditionalReader(stream, other, membership)
        assert extra_reader.poll() == {'train-b000001'} and len(extra_reader.completed['train-b000001']) == 2
        assert extra_reader.pending['reason'] == 'incomplete committed line'
        assert all(Path(row['label_path']).is_relative_to(other) for row in extra_reader.completed['train-b000001'].values())
        assert watch.progress(membership, extra_reader, [])['all_source_labels_completed'] is False
        results['additional_edition_canonical_refs_separate_and_halfline_pending'] = True
    for row in rows:
        unknown, partial = copy.deepcopy(row), copy.deepcopy(row)
        unknown['label']['dance_style'] = 'unknown'
        partial['joint_complete'] = False
        partial['label_status'] = 'partial'
        assert review.label_complete(unknown) is False and review.label_complete(partial) is False
        plan = {'records': [row, unknown], 'sampling': {'selected_source_ids': []}}
        assert review.batch_acceptance(plan, {}, expansion_complete=True)['eligible_for_training'] is False
    results['unknown_partial_training_gate_closed'] = True
    out = ROOT/'curated_independent_reviews_v1'
    cached = {}
    full = []
    original_batch = membership.batch('train-b000000')
    for sid, original in original_batch.items():
        terminal_path = production/'sources'/original['split']/sid/'consumer.terminal.json'
        row = watch.terminal_record(production, watch.evidence.load(terminal_path), membership) if terminal_path.exists() else {**original, 'label_status': 'pending'}
        full.append(review.normalized_record(row))
    retained, errors = watch.retained_selection(review, out, 'train-b000000', full)
    assert retained and errors and len(full) == 500
    for receipt_path in sorted((out/'train-b000000/sources').glob('*/*/review/receipt.json')):
        receipt = watch.evidence.load(receipt_path)
        if receipt['status'] != 'candidate':
            continue
        final = watch.evidence.load(receipt_path.parent/'final.json')
        status = final['json']['status']
        if status not in ('pass', 'confirmed_error') or status in cached:
            continue
        binding = receipt['binding']
        terminal = watch.evidence.load(production/'sources'/binding['split']/binding['source_id']/'consumer.terminal.json')
        row = review.normalized_record(watch.terminal_record(production, terminal, membership))
        assert row['source_id'] in retained
        if status == 'confirmed_error':
            assert row['source_id'] in errors
        changed = copy.deepcopy(full)
        changed_row = next(r for r in changed if r['source_id'] == row['source_id'])
        changed_row['label']['timed_motion_script'] += ' changed current source label'
        try:
            watch.retained_selection(review, out, 'train-b000000', changed)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed historical source must fail closed without reusing or deleting history')
        if status == 'confirmed_error':
            missing = [r if r['source_id'] != row['source_id'] else review.normalized_record({**original_batch[row['source_id']], 'label_status': 'pending'}) for r in full]
            try:
                watch.retained_selection(review, out, 'train-b000000', missing)
            except ValueError:
                results['old_error_current_pending_fails_closed'] = True
            else:
                raise AssertionError('Pending historical error must fail closed; preserve summaries')
        cached[status] = {'receipt_sha256': watch.sha(receipt_path), 'label_sha256': binding['label_sha256']}
        if len(cached) == 2:
            break
    assert len(cached) == 2
    results['old_pass_and_AI_error_current_SHA_retained'] = True
    results['changed_current_label_SHA_not_reused'] = True
    old_sampling = review.sampling_plan
    watch.install_history_selection(review, out)
    selection = review.sampling_plan(full, 'train-b000000', sample_count=25, risk_count=5)
    original_membership = watch.evidence.load(out/'train-b000000/batch.membership.json')
    assert selection['policy'] == original_membership['policy'], 'Frozen membership policy must not change on resume'
    assert set(retained) <= set(selection['selected_source_ids']) and set(errors) <= set(selection['risk_source_ids'])
    review.sampling_plan = old_sampling
    results['frozen_membership_policy_unchanged_history_union'] = True
    value = {'schema': 'curated_review_watch_v4_stream_smoke_v1', 'worker_sha256': watch.sha(RT/'curated_review_watch_v4.py'),
        'history_helper_sha256': watch.sha(RT/'main_review_history_v1.py'),
        'real_bad_offset': 532233, 'real_bad_line_length': len(raw), 'fixed_batch_source_count': len(originals),
        'total_source_count': 222939, 'expected_batch_count': 448, 'source_count_by_split': dict(membership.by_split),
        'results': results, 'verified_cached_receipts': cached, 'GPU_model_load_performed': False,
        'new_VLM_calls': 0, 'production_and_frozen_reviews_mutated': False,
        'stream_fixture_only_isolated_tempfiles': True}
    watch.evidence.once(Path(args.receipt), value)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
