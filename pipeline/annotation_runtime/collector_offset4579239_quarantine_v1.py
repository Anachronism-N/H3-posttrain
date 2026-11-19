#!/usr/bin/env python3
"""Quarantine one reviewed source-unknown fragment; inherit the original CPU collector."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

RT = Path(__file__).resolve().parent
OLD_WRAPPER = RT / 'collector_offset2881725_reconciled_v1.py'
OLD_WRAPPER_SHA = '0cb3499a90da535b97d4e934a9543c20f73eba58d6434d6f01aec4d513fd271a'
if hashlib.sha256(OLD_WRAPPER.read_bytes()).hexdigest() != OLD_WRAPPER_SHA:
    raise RuntimeError('Frozen inherited collector changed')
spec = importlib.util.spec_from_file_location('private_inherited_offset4579239', OLD_WRAPPER)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
r, q = old.r, old.q
old_recover_line = r.recover_line
SELF_REF = q.identity(__file__)
SCHEMA = 'exact_source_unknown_fragment_quarantine_offset4579239_v1'
OFFSET = 4579239
RAW = bytes.fromhex('7465227d0a')
RAW_SHA = '7561b973a7f36b54382420878e894bbb5cd9e7bcd7e5a8db6ec27f883315593c'
INPUT_IDENTITY = [1048620, 9895974566444]
PREFIX_SHA = '061fc80752ba7afe0b7e66399908c5e32494a48ab10c1014c226629c2d233dec'
PRIOR_OFFSETS = (r.BAD_OFFSET, old.v2.EMPTY_OFFSET, old.old.OFFSET, old.OFFSET)


def namespace(production, out):
    production, out = Path(production).resolve(), Path(out).resolve()
    r.same(production, RT.parent / 'curated222939_production_v1', 'Incident production namespace changed')
    r.same(out, RT.parent / 'curated222939_repair_v1', 'Incident repair namespace changed')
    return production, out, out / 'collector-offset4579239-quarantine-v1'


def verify_incident(production, begin, line, snapshots):
    if begin != OFFSET or line != RAW or hashlib.sha256(line).hexdigest() != RAW_SHA:
        raise r.UnrecoverableLine('Unreviewed incident bytes/offset; retain original cursor')
    if (len(snapshots) != 3 or any(s['line_sha256'] != RAW_SHA for s in snapshots)
            or any(len(s['identity']) != 5 or s['identity'] != snapshots[0]['identity'] for s in snapshots)):
        raise r.StreamBusy('Incident lacks three equal original stable byte snapshots')
    for sample in snapshots:
        if sample['identity'][:2] != INPUT_IDENTITY or sample['identity'][2] < OFFSET + len(RAW):
            raise q.SourceChanged('Incident snapshot stream replaced/truncated')
    queue = production / 'completed.jsonl'
    r.original_stream(queue, INPUT_IDENTITY, OFFSET + len(RAW))
    with queue.open('rb') as stream:
        stat = os.fstat(stream.fileno())
        r.same([stat.st_dev, stat.st_ino], INPUT_IDENTITY, 'Opened incident stream identity changed')
        stream.seek(OFFSET)
        r.same(stream.readline(), RAW, 'Original fragment changed after stable snapshots')
    r.original_stream(queue, INPUT_IDENTITY, OFFSET + len(RAW))


def verified_state(production, out, revision, evidence):
    state_path = out / 'collector.recovery.state.json'
    state, state_ref = r.document(state_path)
    origin, origin_ref = r.document(out / 'collector.state.json')
    r.same(state['origin_state_ref'], origin_ref, 'Original collector state drifted')
    r.same((state['input_identity'], origin['input_identity']), (INPUT_IDENTITY, INPUT_IDENTITY), 'Original stream identity changed')
    r.same((state['input_path'], origin['input_path']), (str(production / 'completed.jsonl'),) * 2, 'Original stream path changed')
    if revision is not None:  # The handler inherits r.collect's already-checked CLI revision.
        r.same(state['global_revision'], revision, 'Original recovery revision changed')
    r.same(evidence.production, production, 'Original evidence namespace changed')
    r.same(evidence.summary_ref['path'], str(RT.parent / 'curated222939_inputs_v1/manifest.summary.json'), 'Original manifest namespace changed')
    r.same(state['manifest_summary_ref'], evidence.summary_ref, 'Original manifest summary changed')
    if state['offset'] < OFFSET:
        raise r.UnrecoverableLine('Current recovery checkpoint precedes reviewed quarantine')
    r.same(r.prefix_sha(production / 'completed.jsonl', state['offset'], INPUT_IDENTITY), state['input_prefix_sha256'], 'Current original committed prefix changed')
    if state['offset'] == OFFSET:
        r.same(state['input_prefix_sha256'], PREFIX_SHA, 'Exact reviewed incident prefix changed')
    else:
        r.same(r.prefix_sha(production / 'completed.jsonl', OFFSET, INPUT_IDENTITY), PREFIX_SHA, 'Historical reviewed incident prefix changed')
        if OFFSET not in state['recovered_badline_offsets']:
            raise q.SourceChanged('Advanced cursor lacks original quarantine recovery marker')
    return state, state_ref, origin_ref


def receipt(production, evidence, origin_ref, snapshots, raw_path):
    return {'schema': SCHEMA, 'source_input_path': str(production / 'completed.jsonl'),
        'input_identity': INPUT_IDENTITY, 'physical_byte_begin': OFFSET, 'physical_byte_end': OFFSET + len(RAW),
        'committed_prefix_sha256': PREFIX_SHA, 'raw_ref': {'path': str(raw_path), 'sha256': RAW_SHA},
        'raw_size': len(RAW), 'raw_sha256': RAW_SHA, 'source_association_unknown': True,
        'fragment_source_association': 'unknown', 'contained_complete_sources': 0,
        'reconstructed_source_records': 0, 'source_population_completeness_not_proven': True,
        'source_population_completeness_inferred': False, 'lost_sources_need_separate_reconciliation': True,
        'source_terminal_or_label_generated': False, 'original_decode_error_preserved': True,
        'original_log_modified': False, 'no_model_calls': True, 'model_calls': 0,
        'no_budget_reset': True, 'retry_budget_reset': False, 'GPU_dispatch_performed': False,
        'original_collector_origin_state_ref': origin_ref, 'manifest_summary_ref': evidence.summary_ref,
        'inherited_collector_ref': {'path': str(OLD_WRAPPER), 'sha256': OLD_WRAPPER_SHA},
        'wrapper_ref': SELF_REF, 'stable_badline_snapshots': snapshots, **q.FLAGS}


def verify_receipt(saved, expected, production, raw_path):
    r.same({k: v for k, v in saved.items() if k != 'stable_badline_snapshots'},
           {k: v for k, v in expected.items() if k != 'stable_badline_snapshots'}, 'Immutable quarantine metadata changed')
    verify_incident(production, OFFSET, RAW, saved['stable_badline_snapshots'])
    r.same(raw_path.read_bytes(), RAW, 'Immutable quarantine raw evidence changed')


def recover_line(production, out, begin, line, evidence, snapshots, rules=r.KNOWN):
    production, out, root = namespace(production, out)
    if begin != OFFSET:
        if begin not in PRIOR_OFFSETS:
            raise r.UnrecoverableLine('Unreviewed malformed original line at ' + str(begin))
        return old_recover_line(production, out, begin, line, evidence, snapshots, rules)
    verify_incident(production, begin, line, snapshots)
    state, state_ref, origin_ref = verified_state(production, out, None, evidence)
    r.same(state['offset'], OFFSET, 'Quarantine must inherit the exact original committed checkpoint')
    raw_path, receipt_path = root / 'badline-4579239.raw', root / 'snapshot.json'
    value = receipt(production, evidence, origin_ref, snapshots, raw_path)
    if receipt_path.exists():
        saved = q.load(receipt_path)
        verify_receipt(saved, value, production, raw_path)
        value = saved
    r.same(q.identity(out / 'collector.recovery.state.json'), state_ref, 'Original collector state changed during quarantine')
    r.save_raw_once(raw_path, RAW)
    q.save_once(receipt_path, value)
    return []  # Original r.collect alone owns prefix/cursor, seen, counts, jobs, and recovery markers.


def check_only(production, out, revision, manifest, stable_seconds=0.25):
    production, out, root = namespace(production, out)
    evidence = r.Evidence(production, manifest)
    state, state_ref, origin_ref = verified_state(production, out, revision, evidence)
    snapshots = r.stable_badline(production / 'completed.jsonl', OFFSET, RAW, INPUT_IDENTITY, stable_seconds)
    verify_incident(production, OFFSET, RAW, snapshots)
    raw_path, receipt_path = root / 'badline-4579239.raw', root / 'snapshot.json'
    value = receipt(production, evidence, origin_ref, snapshots, raw_path)
    if receipt_path.exists():
        verify_receipt(q.load(receipt_path), value, production, raw_path)
    elif state['offset'] != OFFSET:
        raise q.SourceChanged('Advanced cursor lacks immutable quarantine receipt')
    r.same(q.identity(out / 'collector.recovery.state.json'), state_ref, 'Recovery checkpoint changed during read-only check')
    return {'schema': SCHEMA, 'check_only': True, 'state_modified': False, 'metadata_written': False,
            'input_byte_offset': state['offset'], 'reviewed_fragment_offset': OFFSET, 'raw_size': len(RAW),
            'raw_sha256': RAW_SHA, 'source_association_unknown': True, 'contained_complete_sources': 0,
            'source_population_completeness_not_proven': True, 'lost_sources_need_separate_reconciliation': True,
            'no_model_calls': True, 'no_budget_reset': True, 'GPU_dispatch_performed': False, **q.FLAGS}


r.recover_line = recover_line


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('production', 'out', 'global-revision', 'manifest'):
        parser.add_argument('--' + name, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('once', 'watch', 'check-only'):
        mode.add_argument('--' + name, action='store_true')
    parser.add_argument('--stable-seconds', type=float, default=0.25)
    parser.add_argument('--interval', '--poll-seconds', dest='interval', type=float, default=10)
    args = parser.parse_args()
    if any(not math.isfinite(x) or x < 0 for x in (args.stable_seconds, args.interval)) or (args.watch and args.interval == 0):
        parser.error('Stability delay must be finite/nonnegative; watch interval must be finite/positive')
    production, out, _ = namespace(args.production, args.out)
    if args.check_only:
        result = check_only(production, out, args.global_revision, args.manifest, args.stable_seconds)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return
    while True:
        result = r.collect(production, out, args.global_revision, args.manifest, args.stable_seconds)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
