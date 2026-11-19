"""Tail helper for one collab3 producer shard (native IOv4 pipeline, reverse order, no model changes).

The original producer keeps running forward through its shard. Helpers process the same shard from the
END backwards with per-source O_EXCL claims, and stop before they get within GAP sources of the original
producer's frontier (observed from the shard ready journal), so the two never touch the same source.
Ready records go to the same shard ready journal via the native flock append.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

RT = Path(__file__).resolve().parent
sys.path.insert(0, str(RT))
import curated_collaborative3_plan_v1 as plan

HELPER_ROOT = plan.OUT / 'tail_helper_v1'
CLAIMS = HELPER_ROOT / 'claims'


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--member', required=True)
    parser.add_argument('--helper-id', required=True)
    parser.add_argument('--gap', type=int, default=200)
    extra, remaining = parser.parse_known_args(argv)
    node, mode, gpus, shard, nshard = plan.MEMBERS[extra.member]
    if mode != 'producer' or extra.gap < 20:
        raise RuntimeError('Producer member and gap>=20 required')
    plan.verify_plan(plan.load(plan.PLAN))
    import curated_producer_io_v4 as io
    native = io.base.load_pipeline
    ready_path = plan.OUT / 'ready' / ('producer-%04d.jsonl' % shard)
    stop_file = HELPER_ROOT / 'STOP'
    CLAIMS.mkdir(parents=True, exist_ok=True)
    stats = dict(helper=extra.helper_id, member=extra.member, claimed=0, skipped_terminal=0, stop_reason=None)

    def load_pipeline(path=io.base.FROZEN):
        pipeline = native(path)
        append, save, original_rows = pipeline.append, pipeline.save, pipeline.iter_rows

        def routed_append(path, record):
            if Path(path) == plan.OUT / 'ready.jsonl':
                if record['source_ordinal'] % nshard != shard or not plan.START <= record['source_ordinal'] < plan.END:
                    raise RuntimeError('Ready publication outside fixed producer member')
                path = ready_path
            return append(path, record)

        def bound_save(path, record):
            if Path(path).name == 'protocol.json':
                record = {**record, 'collab3_member': extra.member, 'collab3_tail_helper': extra.helper_id,
                          'collab3_tail_helper_code': plan.ref(__file__), 'collab3_tail_helper_gap': extra.gap}
            return save(path, record)

        def tail_rows(summary, shard_arg=0, nshard_arg=1, limit=0, ids=None):
            if ids is None:  # Native manifest scan (IO v2 select_ids), not the producer loop.
                yield from original_rows(summary, shard_arg, nshard_arg, limit, ids)
                return
            order = [(row, batch) for row, batch in original_rows(summary, shard_arg, nshard_arg, 0, ids)
                     if plan.START <= row['source_ordinal'] < plan.END]
            index = {row['id']: i for i, (row, _) in enumerate(order)}
            cursor = [0, -1]  # ready offset, original-producer frontier

            def frontier():
                claimed = {p.stem for p in CLAIMS.iterdir()}
                with ready_path.open('rb') as stream:
                    stream.seek(cursor[0])
                    while True:
                        line = stream.readline()
                        if not line.endswith(b'\n'):
                            break
                        cursor[0] += len(line)
                        try:
                            sid = json.loads(line).get('id')
                        except ValueError:
                            continue
                        if sid in index and sid not in claimed:
                            cursor[1] = max(cursor[1], index[sid])
                return cursor[1]

            for j in range(len(order) - 1, -1, -1):
                row, batch = order[j]
                if stop_file.exists():
                    stats['stop_reason'] = 'stop_file'
                    return
                if j - frontier() <= extra.gap:
                    stats['stop_reason'] = 'reached_gap_to_original_producer'
                    return
                if (pipeline.source_dir(plan.OUT, row) / 'producer.terminal.json').exists():
                    stats['skipped_terminal'] += 1
                    continue
                try:
                    fd = os.open(CLAIMS / (row['id'] + '.json'), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                except FileExistsError:
                    continue
                with os.fdopen(fd, 'w') as stream:
                    json.dump(dict(id=row['id'], index=j, member=extra.member, helper=extra.helper_id,
                                   pid=os.getpid(), utc=time.strftime('%FT%TZ', time.gmtime())), stream)
                if j - frontier() <= extra.gap or (pipeline.source_dir(plan.OUT, row) / 'producer.terminal.json').exists():
                    stats['stop_reason'] = 'gap_or_terminal_after_claim'
                    return  # Claim stays; original producer will still process this source normally.
                stats['claimed'] += 1
                yield row, batch
            stats['stop_reason'] = 'shard_start_reached'

        pipeline.append, pipeline.save, pipeline.iter_rows = routed_append, bound_save, tail_rows
        return pipeline

    io.base.load_pipeline = load_pipeline

    def option(name):
        if remaining.count(name) != 1:
            raise RuntimeError('Unique explicit producer option required: ' + name)
        return remaining[remaining.index(name) + 1]
    if (option('--out') != str(plan.OUT) or option('--manifest') != str(plan.MANIFEST)
            or int(option('--shard')) != shard or int(option('--nshard')) != nshard
            or int(option('--ordinal-start')) != plan.START or int(option('--ordinal-end')) != plan.END
            or '--source-ids-file' in remaining or '--ids' in remaining):
        raise RuntimeError('Fixed producer source selection changed')
    result = io.main(remaining)
    return dict(result=result, **stats)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False), flush=True)
