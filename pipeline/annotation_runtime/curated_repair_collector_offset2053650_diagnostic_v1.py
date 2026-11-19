#!/usr/bin/env python3
"""CPU-only, read-only exact incident diagnosis; never advance a collector cursor."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

RT = Path(__file__).resolve().parent
FROZEN = {
    'curated_repair_collector_recovery_v1.py': 'f85efb1441930cfc31bfe4cafacf25e0fdb18aeb9da6ad80fb540b58519237cd',
    'curated_repair_collector_emptyline_v2.py': '5c6c01e2f9ad70fc6515ef43384e81af2cface3b52f716c67496a24e1565752a',
    'source_repair_gpu_v2/curated_repair_queue_v1.py': '94b00ad0b503053f4b9bd7f7c08097bf9b6480f8a9aa248e971126a149c78b4c',
}
for name, sha in FROZEN.items():
    if hashlib.sha256((RT / name).read_bytes()).hexdigest() != sha:
        raise RuntimeError('Frozen collector source changed: ' + name)
spec = importlib.util.spec_from_file_location('frozen_offset2053650_diagnostic', RT / 'curated_repair_collector_emptyline_v2.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
r = v2.recovery

OFFSET = 2053650
RAW = bytes.fromhex('7465227d0a')
RAW_SHA = '7561b973a7f36b54382420878e894bbb5cd9e7bcd7e5a8db6ec27f883315593c'
INPUT_IDENTITY = [1048620, 9895974566444]
PREFIX_SHA = '7fb515967552cd1916581019b1a13f1f83f07e2da263424a4ed6af93ec80864c'
OLD_OFFSETS = [532233, 1192186]
INDEX_PREFIXES = {
    'train-b000005': (17010, '711c461338909a9ab2bde993867dfdadc267dd8cd61670f180267ee21bc3a900'),
    'train-b000006': (2730, 'c15a862202c0516aa846511a97c327052fc55e33be162f56e8420d606514354f'),
}
CANDIDATES = [
    ('afgToKIsEUTkAyDP', 'train-b000005', 286, 2723, '0f5b43a56bbc660d43f11d2cbf0fa6f3954ca516d433400da23ca56c362a6079'),
    ('SQQbWIbvUGqzZMGP', 'train-b000005', 344, 2775, '709dc468c11b2d5bf154039e4a391ebc3feca1cabd6165d2ee6bfa1c2a81e7bd'),
    ('QIWvouIspDKiagWW', 'train-b000005', 403, 2883, '166d4a8fcd806bb906f3762609ca53680c656a53cf51bb9fab424d113256dd75'),
    ('VHFSAcOmzvvCYiGJ', 'train-b000006', 64, 3140, '244156218a17f42782b25347685156b2dc67951d4211040e9f4277a507f9974c'),
]


def verify_incident(path, begin, delay=0.25):
    if begin != OFFSET:
        raise r.UnrecoverableLine('Incident offset is not the frozen original cursor')
    r.same(r.prefix_sha(path, OFFSET, INPUT_IDENTITY), PREFIX_SHA, 'Original incident prefix changed')
    with path.open('rb') as stream:
        r.same([os.fstat(stream.fileno()).st_dev, os.fstat(stream.fileno()).st_ino],
               INPUT_IDENTITY, 'Opened incident stream identity differs')
        stream.seek(begin)
        actual = stream.readline()
    if actual != RAW or hashlib.sha256(actual).hexdigest() != RAW_SHA:
        raise r.UnrecoverableLine('Frozen incident bytes changed; do not advance cursor')
    samples = r.stable_badline(path, begin, actual, INPUT_IDENTITY, delay)
    if len(samples) != 3 or any(s['line_sha256'] != RAW_SHA for s in samples):
        raise r.StreamBusy('Exact incident lacks three stable byte snapshots')
    if any(s['identity'] != samples[0]['identity'] for s in samples):
        raise r.StreamBusy('Exact incident snapshots are unstable')
    for sample in samples:
        if sample['identity'][:2] != INPUT_IDENTITY or sample['identity'][2] < begin + len(RAW):
            raise r.q.SourceChanged('Snapshot identity changed or incident truncated')
    r.same(r.prefix_sha(path, OFFSET, INPUT_IDENTITY), PREFIX_SHA, 'Original prefix changed after snapshots')
    return {'byte_begin': begin, 'byte_end': begin + len(actual), 'raw_sha256': RAW_SHA,
            'raw_hex': actual.hex(), 'input_identity': INPUT_IDENTITY,
            'input_prefix_sha256': PREFIX_SHA, 'stable_badline_snapshots': samples}


def diagnose(delay=0.25):
    production = RT.parent / 'curated222939_production_v1'
    out = RT.parent / 'curated222939_repair_v1'
    queue = production / 'completed.jsonl'
    state, state_ref = r.document(out / 'collector.recovery.state.json')
    r.same(state['offset'], OFFSET, 'Collector no longer has the incident cursor')
    r.same(state['input_identity'], INPUT_IDENTITY, 'Original state inode changed')
    r.same(state['input_path'], str(queue), 'Original state stream changed')
    r.same(state['input_prefix_sha256'], PREFIX_SHA, 'Original state prefix changed')
    r.same(state['recovered_badline_offsets'], OLD_OFFSETS, 'Prior reviewed incidents changed')
    r.document(state['origin_state_ref']['path'], state['origin_state_ref']['sha256'])
    incident = verify_incident(queue, OFFSET, delay)
    evidence = r.Evidence(production, state['manifest_summary_ref']['path'])
    r.same(evidence.summary_ref, state['manifest_summary_ref'], 'Original manifest summary changed')
    indices = {}
    for batch, (size, expected_sha) in INDEX_PREFIXES.items():
        path = production / 'batches' / batch / 'terminal_index.jsonl'
        identity = [path.stat().st_dev, path.stat().st_ino]
        r.same(r.prefix_sha(path, size, identity), expected_sha, 'Finite batch index prefix changed')
        with path.open('rb') as stream:
            prefix = stream.read(size)
        r.same(hashlib.sha256(prefix).hexdigest(), expected_sha, 'Read finite index prefix changed')
        indices[batch] = [json.loads(line) for line in prefix.splitlines()]
    candidates = []
    for sid, batch, index, ordinal, terminal_sha in CANDIDATES:
        r.same(indices[batch][index], {'id': sid, 'split': 'train'}, 'Original terminal index member changed')
        directory = production / 'sources/train' / sid
        terminal, terminal_ref = r.document(directory / 'consumer.terminal.json', terminal_sha)
        r.source_identity(terminal, sid, 'train', batch)
        member, manifest_ref = evidence.member(sid, 'train', batch)
        producer, producer_ref = r.document(directory / 'producer.terminal.json')
        r.source_identity(producer, sid, 'train', batch)
        r.same((member['source_ordinal'], producer['source_ordinal']), (ordinal, ordinal), 'Original ordinal changed')
        plan, plan_ref = evidence.artifact(producer['plan_ref'], directory)
        r.same(plan['manifest_record'], member, 'Original manifest member differs from plan')
        r.same(plan['source_refs']['manifest'], manifest_ref, 'Plan manifest reference changed')
        r.same(plan['source_refs']['original_csv'], member['source_refs']['csv'], 'Plan original CSV reference changed')
        joint, joint_ref = evidence.artifact(terminal['joint_candidate_ref'], directory / 'joint-results')
        r.source_identity(joint, sid, 'train', batch)
        r.same(joint['source_refs'], member['source_refs'], 'JOINT original CSV references changed')
        candidates.append({'source_key': 'train/' + sid, 'batch_id': batch,
            'terminal_index_position': index, 'original_source_ordinal': ordinal,
            'consumer_terminal_ref': terminal_ref, 'producer_terminal_ref': producer_ref,
            'plan_ref': plan_ref, 'joint_ref': joint_ref, 'manifest_ref': manifest_ref,
            'original_csv_ref': member['source_refs']['csv'],
            'serialized_terminal_suffix_matches': r.q.encoded(terminal).endswith(RAW),
            'present_in_inherited_seen': 'train/' + sid in state['seen'],
            'bound_to_incident_offset': False})
    return {'schema': 'exact_offset2053650_diagnostic_v1', **incident,
        'collector_state_ref': state_ref, 'prior_reviewed_incident_offsets': OLD_OFFSETS,
        'collector_code_refs': [{'path': str(RT / name), 'sha256': sha} for name, sha in FROZEN.items()],
        'limited_terminal_candidates': candidates,
        'decision': 'STOP_UNREVIEWED_NONUNIQUE_TERMINAL_SUFFIX',
        'cursor_advanced': False, 'source_rows_reconstructed': 0,
        'source_terminal_or_label_generated': False, 'source_population_completeness_inferred': False,
        'unseen_or_lost_terminals_require_separate_reconciliation': True,
        'production_and_collector_state_modified': False,
        'whole_media_content_hashed': False, 'GPU_dispatch_performed': False, **r.q.FLAGS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stable-seconds', type=float, default=0.25)
    args = parser.parse_args()
    if args.stable_seconds < 0:
        parser.error('Negative stability delay')
    print(json.dumps(diagnose(args.stable_seconds), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
