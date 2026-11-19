"""Native TP2 consumer, four single-writer inputs and isolated shard output."""
import argparse
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import curated_collaborative3_plan_v1 as plan


def configuration(pipeline, args):
    bar, _, _ = pipeline.modules(args.repo, args.ms_source)
    args.hand_roi_root = str(plan.OUT / 'roi')
    value = bar.params(args, bar.code_identity())
    style = Path(bar.__file__).with_name('ms_dance_style.py')
    value['style_module_sha256'] = pipeline.sha(style) if style.exists() else None
    return value


def install(pipeline):
    source = inspect.getsource(pipeline.run_consumer)
    changes = [
        ("args.hand_roi_root = str(out / 'roi')", "args.hand_roi_root = str(SOURCE_OUT / 'roi')"),
        ("'configuration': configuration, 'pipeline_code': ref(__file__), **FLAGS",
         "'configuration': configuration, 'pipeline_code': ref(__file__), 'consumer_wrapper': ref(CONSUMER_WRAPPER), 'collab3_source_plan': ref(SOURCE_PLAN), **FLAGS"),
        ("    position = 0", "    position = 0\n    positions = [0] * 4\n    queue_index = -1"),
        ("assigned_total = sum(1 for i in range(summary['total']) if i % args.nshard == args.shard)", "assigned_total = len(args.ids)"),
        ("assigned_total = sum(1 for _ in iter_rows(summary, args.shard, args.nshard, ids=wanted))", "assigned_total = len(wanted)"),
        ("        queue = out / 'ready.jsonl'", "        queue_index = (queue_index + 1) % 4\n        position = positions[queue_index]\n        queue = SOURCE_OUT / 'ready' / ('producer-%04d.jsonl' % queue_index)"),
        ("                    position = begin\n                    break", "                    position = begin\n                    positions[queue_index] = position\n                    break"),
        ("                position = stream.tell()", "                position = stream.tell()\n                positions[queue_index] = position"),
        ("                directory = out / 'sources' / hint['split'] / hint['id']\n                terminal = load(directory / 'producer.terminal.json')",
         "                producer_directory = SOURCE_OUT / 'sources' / hint['split'] / hint['id']\n                terminal = load(producer_directory / 'producer.terminal.json')\n                directory = out / 'sources' / hint['split'] / hint['id']"),
        ("ref(directory / 'producer.terminal.json')", "ref(producer_directory / 'producer.terminal.json')"),
        ("actual_media = load(directory / 'actual_media.json')", "actual_media = load(producer_directory / 'actual_media.json')"),
        ("        if not args.watch:\n            break", "        if not args.watch and queue_index == 3:\n            break"),
    ]
    for old, new in changes:
        if source.count(old) != 1:
            raise RuntimeError('Native consumer patch anchor changed: ' + old)
        source = source.replace(old, new)
    pipeline.SOURCE_OUT = plan.OUT
    pipeline.SOURCE_PLAN = plan.PLAN
    pipeline.CONSUMER_WRAPPER = str(Path(__file__).resolve())
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), pipeline.__dict__)
    return source


@contextmanager
def optimization(pipeline, args):
    import curated_fast_source_v1 as fast
    import curated_transaction_cache_v1 as transactions
    bar, _, _ = pipeline.modules(args.repo, args.ms_source)
    original_process, original_save = bar.process_source, pipeline.save
    metadata = {"fast_source_execution_ref": plan.ref(plan.RT / "curated_fast_source_v1.py"),
                "transaction_cache_execution_ref": plan.ref(plan.RT / "curated_transaction_cache_v1.py"),
                "caption_optimization": "source_input_reuse_and_lock_scoped_transactions",
                "native_request_sampling_and_labels_unchanged": True}

    def process(*positional, **keywords):
        with fast.fast_source(bar, positional[2]) as cache:
            try:
                return original_process(*positional, **keywords)
            finally:
                print(json.dumps({"event": "source_input_cache", **cache.stats}, sort_keys=True), flush=True)

    def save(path, value):
        if Path(path).name == "protocol.json" and value.get("mode") == "consumer":
            value = {**value, **metadata}
        return original_save(path, value)

    def emit(value):
        print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)

    bar.process_source, pipeline.save = process, save
    try:
        with transactions.transaction_cache(bar, emit):
            yield
    finally:
        bar.process_source, pipeline.save = original_process, original_save


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--member', required=True)
    parser.add_argument('--selected-sources', required=True)
    extra, remaining = parser.parse_known_args(argv)
    value = plan.verify_plan(plan.load(plan.PLAN))
    node, mode, gpus, shard, nshard = plan.MEMBERS[extra.member]
    pipeline = plan.pipeline()
    args = pipeline.parser().parse_args(remaining)
    if (mode != 'consumer' or args.mode != mode or args.shard != shard or args.nshard != nshard
            or args.limit != 0 or args.ids or Path(args.out).resolve() != plan.paths(extra.member)['out']
            or Path(args.manifest) != plan.MANIFEST or extra.selected_sources != value['selected_sources'][extra.member]['path']):
        raise RuntimeError('Exact native isolated consumer assignment changed')
    args.ids = [r['id'] for r in plan.checked(value['selected_sources'][extra.member])]
    install(pipeline)
    with optimization(pipeline, args):
        return pipeline.run_consumer(args)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False), flush=True)
