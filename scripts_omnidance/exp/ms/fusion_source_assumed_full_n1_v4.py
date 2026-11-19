"""Full-6000 source-assumed export; legacy reference/keyword flags are sidecars."""
import argparse
import concurrent.futures
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).absolute().parent
V3_SHA = '8917ef1d90094d1c392f0dd3097d546371ce62d81ec5fca2b5cfbcb23ed1e682'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(name, path, expected=None):
    if expected:
        assert sha(path.read_bytes()) == expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V3 = load('frozen_fusion_parallel', HERE / 'fusion_complete_parallel_n1_v3.py', V3_SHA)
CLEAN = load('source_assumed_clean', HERE / 'export_fusion_source_assumed_v1.py',
    '742dcfacf9d5dd2a3d9f6ef64ae80246f5a5f4b77ae4de03e624a1448d670cc0')
EXPORT = V3.MS / 'fusion_source_assumed_full6000_v4'


def recover_target(record, unique_object):
    target = record.get('target')
    if isinstance(target, dict) and isinstance(target.get('motion_caption'), str) and target['motion_caption'].strip():
        return target, None
    raw = record.get('raw_generation')
    if not isinstance(raw, str):
        raise ValueError('missing_raw_generation')
    # Recover a complete emitted JSON object with trailing text, never synthesize a caption.
    target, end = json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(raw.lstrip())
    if not isinstance(target, dict) or not isinstance(target.get('motion_caption'), str) or not target['motion_caption'].strip():
        raise ValueError('no_complete_nonempty_emitted_caption')
    return target, dict(method='first_complete_emitted_JSON_object', trailing_text=raw.lstrip()[end:],
        raw_generation_sha256=sha(raw.encode()), raw_unchanged=True)


def inspect(item, legacy, helper, plan):
    index, row = item
    sid = row['id']
    paths = [path for path in [V3.OUT / 'aggregate' / (sid + '.json')] +
        [V3.OUT / ('shard' + str(gpu)) / (sid + '.json') for gpu in V3.UUIDS] if path.is_file()]
    if len(paths) != 1:
        return None, dict(id=sid, reason='missing_or_multiple_result', paths=[str(p) for p in paths])
    path = paths[0]
    try:
        data = path.read_bytes()
        record = json.loads(data, object_pairs_hook=helper.unique_object)
        assert record['id'] == sid and record['split'] == row['split'] and record['source_group'] == row['source_group']
        assert record['original_source_record'] == row and record['source_records_sha256'] == V3.SOURCE_SHA
        assert record['source_sha256'] in (V3.LEGACY_SHA, plan['new_source_sha256'])
        material = legacy.actual_material(row)
        assert record['actual_text_input'] == material and record['actual_text_input_sha256'] == legacy.object_sha(material)
        assert record['raw_source_text_input_sha256'] == row['actual_text_input_sha256']
        assert record['semantic_approval'] is False and record['human_review_completed'] is False
        assert record['window_mapping'] == 'unknown'
        refs = {source['ref'] for source in material['sources']}
        assert 'MS:original' in refs and any(ref.startswith('rich:') for ref in refs)
        target, recovery = recover_target(record, helper.unique_object)
        assert record['generated_tokens'] < legacy.MAX_TOKENS, 'generation_limit_requires_individual_retry'
        assert record['generated_tokens'] == len(record['raw_generated_token_ids'])
        return dict(index=index, row=row, path=path, data=data, record=record, target=target, recovery=recovery), None
    except (AssertionError, ValueError, TypeError, KeyError, OSError) as exc:
        return None, dict(id=sid, path=str(path), reason=str(exc), raw_preserved=True)


def export():
    CLEAN.self_test()
    legacy = V3.load_module('legacy_policy_export', V3.LEGACY, V3.LEGACY_SHA)
    helper = legacy.helper()
    assert V3.sha(V3.SOURCE) == V3.SOURCE_SHA
    rows = [json.loads(line, object_pairs_hook=helper.unique_object) for line in V3.SOURCE.read_text().splitlines()]
    assert len(rows) == len({row['id'] for row in rows}) == 6000
    plan = json.loads((V3.OUT / 'plan.json').read_text())
    assert plan['new_source_sha256'] == V3_SHA
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda item: inspect(item, legacy, helper, plan), enumerate(rows)))
    unavailable = [error for _, error in results if error]
    if unavailable:
        path = V3.OUT / ('source_assumed_unavailable_' + str(time.time_ns()) + '.json')
        V3.save(path, unavailable)
        print(json.dumps(dict(event='source_assumed_export_blocked', included=6000-len(unavailable),
            unavailable=len(unavailable), unavailable_path=str(path), ids=[r['id'] for r in unavailable])), flush=True)
        return False
    records = [record for record, _ in results]
    assert len(records) == len({rec['row']['id'] for rec in records}) == 6000
    EXPORT.mkdir(exist_ok=False)
    flags = Counter()
    changed = recovered = 0
    with (EXPORT / 'captions.jsonl').open('x', encoding='utf-8') as caps, \
         (EXPORT / 'lineage.jsonl').open('x', encoding='utf-8') as lineage:
        for value in records:
            row, record, target = value['row'], value['record'], value['target']
            body = target['motion_caption']
            cleaned = CLEAN.clean(body)
            changed += cleaned != body
            recovered += value['recovery'] is not None
            flags.update(record.get('validation_errors', []))
            common = dict(id=row['id'], split=row['split'], source_group=row['source_group'], genre=row['genre'],
                caption_version='rich_ms_fusion_source_assumed_full6000_v4', source_policy=CLEAN.POLICY,
                source_action_correctness_assumed=True, semantic_review_performed=False, human_review_completed=False,
                original_source_window_mapping='unknown', faithfulness_status='not_individually_verified',
                source_record_path=str(value['path']), source_record_sha256=sha(value['data']))
            caps.write(json.dumps(dict(common, motion_caption=cleaned), ensure_ascii=False) + '\n')
            lineage.write(json.dumps(dict(common, raw_fused_caption=body, exported_caption=cleaned, text_changed=cleaned != body,
                original_source_record=row, actual_text_input=record['actual_text_input'],
                actual_text_input_sha256=record['actual_text_input_sha256'],
                raw_source_text_input_sha256=record['raw_source_text_input_sha256'],
                raw_rich_text_sha256=row['raw_rich_text_sha256'], raw_MS_text_sha256=row['raw_MS_text_sha256'],
                source_records_sha256=V3.SOURCE_SHA, generation_source_code_sha256=record['source_sha256'],
                legacy_source_code_sha256=V3.LEGACY_SHA, raw_generation_sha256=sha(record['raw_generation'].encode()),
                raw_generated_token_ids_sha256=legacy.object_sha(record['raw_generated_token_ids']),
                original_completion_status=record.get('completion_status'), original_rule_flags=record.get('validation_errors', []),
                original_flags_do_not_block_export=True, original_generated_target=record.get('target'),
                source_refs=target.get('source_refs'), source_refs_status='generated_unverified_sidecar_only',
                actual_bound_source_refs=[s['ref'] for s in record['actual_text_input']['sources']],
                conflicts=target.get('conflicts'), uncertainty=target.get('uncertainty'),
                raw_JSON_recovery=value['recovery']), ensure_ascii=False) + '\n')
        for stream in (caps, lineage):
            stream.flush()
            os.fsync(stream.fileno())
    report = dict(complete=True, batch6000_complete=True, included=6000, unique_ids=6000, unavailable=0, split='train',
        source_policy=CLEAN.POLICY, source_action_correctness_assumed=True, semantic_review_performed=False,
        human_review_completed=False, source_records_sha256=V3.SOURCE_SHA, new_generation_source_sha256=V3_SHA,
        export_source_code_sha256=sha(Path(__file__).read_bytes()), legacy_source_code_sha256=V3.LEGACY_SHA,
        reused_old=plan['reused'], new_sharded=plan['missing'], shard_counts=plan['shard_counts'],
        legacy_reference_flag_only_regenerated_count=sum(t['reason']['reason'] != 'missing' for t in plan['tasks']),
        original_rule_flag_counts=dict(flags), deterministic_cleaned=changed, raw_JSON_recovered=recovered,
        raw_outputs_unchanged=True, val500_deferred=True,
        eligibility='complete emitted nonempty motion body plus frozen exact SID/source/input-hash and real rich+MS binding; generated reference and keyword flags are nonblocking sidecars',
        cleanup='original clean(): pronouns/listed synthetic timing only; directions/degrees/amplitude/real repetition retained',
        exporter_cleanup_sha256=sha((HERE / 'export_fusion_source_assumed_v1.py').read_bytes()),
        files={name: V3.sha(EXPORT / name) for name in ('captions.jsonl', 'lineage.jsonl')})
    V3.save(EXPORT / 'complete.json', report)
    print(json.dumps(dict(event='all6000_source_assumed_complete', output=str(EXPORT), report=report)), flush=True)
    return True


def wait():
    last = None
    while True:
        if (V3.OUT / 'plan.json').is_file():
            plan = json.loads((V3.OUT / 'plan.json').read_text())
            counts = {}
            done = True
            for gpu in V3.UUIDS:
                folder = V3.OUT / ('shard' + str(gpu))
                count = sum(p.suffix == '.json' and len(p.stem) == 16 for p in folder.iterdir())
                counts[str(gpu)] = count
                done &= count == plan['shard_counts'][str(gpu)]
            if counts != last:
                last = counts
                print(json.dumps(dict(event='actual_full6000_progress', old_reused=plan['reused'],
                    new_counts=counts, available=plan['reused']+sum(counts.values()), target=6000)), flush=True)
            if done:
                # All JSON file bodies must pass before any complete marker is written.
                time.sleep(10)
                if export():
                    return
                raise SystemExit('Only listed unavailable IDs require recovery/retry; full batch retained')
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if args.wait:
        wait()
    elif not export():
        raise SystemExit(1)
