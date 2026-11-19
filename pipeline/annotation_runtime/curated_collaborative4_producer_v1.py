"""Native IOv4 producer with one ready writer for the fixed 20k range."""
import argparse
import json
import sys
from pathlib import Path
import curated_collaborative4_plan_v1 as plan

NATIVE_SHA = '2dbf2b3da4bf838132b54ba8bcfbed90f5a3e9f8775c7a4ada56aaf0add6e735'


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--member', required=True)
    extra, remaining = parser.parse_known_args(argv)
    node, mode, gpus, shard, nshard = plan.MEMBERS[extra.member]
    if mode != 'producer':
        raise RuntimeError('Producer member required')
    value = plan.verify_plan(plan.load(plan.PLAN))
    if plan.sha(plan.RT / 'curated_producer_io_v4.py') != NATIVE_SHA:
        raise RuntimeError('Native IOv4 SHA changed')
    import curated_producer_io_v4 as io
    native = io.base.load_pipeline

    def load_pipeline(path=io.base.FROZEN):
        pipeline = native(path)
        append, save = pipeline.append, pipeline.save

        def routed_append(path, record):
            if Path(path) == plan.OUT / 'ready.jsonl':
                if (record['source_ordinal'] % nshard != shard
                        or not plan.START <= record['source_ordinal'] < plan.END):
                    raise RuntimeError('Ready publication outside fixed producer member')
                path = plan.OUT / 'ready' / ('producer-%04d.jsonl' % shard)
            return append(path, record)

        def bound_save(path, record):
            if Path(path).name == 'protocol.json':
                record = {**record, 'collab4_producer_wrapper': plan.ref(__file__),
                          'collab4_source_plan': plan.ref(plan.PLAN), 'collab4_member': extra.member}
            return save(path, record)

        pipeline.append, pipeline.save = routed_append, bound_save
        return pipeline

    io.base.load_pipeline = load_pipeline
    # The native parser validates all arguments; these bindings must precede any model load.
    def option(name):
        if remaining.count(name) != 1:
            raise RuntimeError('Unique explicit producer option required: ' + name)
        return remaining[remaining.index(name) + 1]
    if (option('--out') != str(plan.OUT) or option('--manifest') != str(plan.MANIFEST)
            or int(option('--shard')) != shard or int(option('--nshard')) != nshard
            or int(option('--ordinal-start')) != plan.START or int(option('--ordinal-end')) != plan.END
            or int(option('--limit')) != 0 or '--source-ids-file' in remaining or '--ids' in remaining):
        raise RuntimeError('Fixed producer source selection changed')
    return io.main(remaining)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False), flush=True)
