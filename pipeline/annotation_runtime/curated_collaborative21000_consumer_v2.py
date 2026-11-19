"""Finite native TP2 consumer; only the independent collaborative output is used."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

RT = Path(__file__).resolve().parent
BASE = RT / 'source_curated_consumer_v1/curated_pipeline.py'
BASE_SHA = 'd5cd607d4789e3b3cb361b2f32c35c4adc438cda8b9d2f5262dc143e61e952fc'
OUT = RT.parent / 'curated222939_collaborative_v2'
START, END = 21000, 31000


def load_pipeline():
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
        raise RuntimeError('Original curated consumer snapshot changed')
    spec = importlib.util.spec_from_file_location('curated_consumer_original_v1', BASE)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    return pipeline


def selected_rows(pipeline, manifest):
    summary = pipeline.load(manifest)
    if summary['total'] != 222939:
        raise ValueError('Full curated provenance count changed')
    rows = []
    for batch in summary['batches']:
        start = batch['first_source_ordinal']
        if start >= END or start + batch['count'] <= START:
            continue
        if pipeline.sha(batch['path']) != batch['sha256']:
            raise ValueError('Selected original batch SHA changed')
        values = [json.loads(line) for line in Path(batch['path']).read_text().splitlines() if line.strip()]
        if len(values) != batch['count']:
            raise ValueError('Selected original batch count changed')
        for offset, row in enumerate(values):
            if (row['split'] != batch['split'] or row['batch_id'] != batch['batch_id']
                    or row['source_ordinal'] != start + offset):
                raise ValueError('Selected batch binding/ordinal changed')
            if START <= row['source_ordinal'] < END:
                rows.append({key: row[key] for key in ('id', 'split', 'batch_id', 'source_ordinal')})
    if ([row['source_ordinal'] for row in rows] != list(range(START, END))
            or len({row['id'].casefold() for row in rows}) != END - START):
        raise ValueError('Expansion requires exactly 10000 unique ordinals 21000:31000')
    return rows


def configuration(pipeline, args):
    bar, _, _ = pipeline.modules(args.repo, args.ms_source)
    args.hand_roi_root = str(Path(args.out).resolve() / 'roi')
    value = bar.params(args, bar.code_identity())
    style = Path(bar.__file__).with_name('ms_dance_style.py')
    value['style_module_sha256'] = pipeline.sha(style) if style.exists() else None
    return value


def install(pipeline):
    source = inspect.getsource(pipeline.run_consumer)
    changes = [
        ("visual_out = out / 'visual' / f'shard-{args.shard:04d}'",
         "visual_out = out / 'visual' / 'collaborative-n3g67-v2'"),
        ("'configuration': configuration, 'pipeline_code': ref(__file__), **FLAGS",
         "'configuration': configuration, 'pipeline_code': ref(__file__), 'consumer_wrapper': ref(CONSUMER_WRAPPER), **FLAGS"),
        ("assigned_total = sum(1 for i in range(summary['total']) if i % args.nshard == args.shard)",
         "assigned_total = len(args.ids)"),
        ("assigned_total = sum(1 for _ in iter_rows(summary, args.shard, args.nshard, ids=wanted))",
         "assigned_total = len(wanted)"),
    ]
    old = "                with visual.bundle_lock(directory):\n                    with visual.bundle_lock(root):"
    new = """                with visual.bundle_lock(directory):
                    if consumer_terminal.exists():
                        previous = load(consumer_terminal)
                        if previous.get('params_sha256') != visual.digest(configuration):
                            raise ValueError('Locked consumer parameter fingerprint differs')
                        artifact = previous.get('joint_candidate_ref')
                        if artifact and sha(artifact['path']) != artifact['sha256']:
                            raise ValueError('Locked resumed joint candidate changed')
                        seen.add(hint['id'])
                        counts['resumed_terminal'] += 1
                        continue
                    with visual.bundle_lock(root):"""
    changes.append((old, new))
    for old, new in changes:
        if source.count(old) != 1:
            raise RuntimeError('Consumer wrapper patch anchor changed: ' + old)
        source = source.replace(old, new)
    if source.count("queue = out / 'ready.jsonl'") != 1 or "PRIVATE_READY" in source:
        raise RuntimeError('Native public-ready input changed')
    pipeline.CONSUMER_WRAPPER = str(Path(__file__).resolve())
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), pipeline.__dict__)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--selected-sources', required=True)
    extra, remaining = parser.parse_known_args(argv)
    pipeline = load_pipeline()
    args = pipeline.parser().parse_args(remaining)
    if (args.mode != 'consumer' or args.shard != 0 or args.nshard != 1 or args.limit != 0
            or args.ids or Path(args.out).resolve() != OUT):
        raise ValueError('Only isolated collaborative consumer shard0/nshard1/limit0 admitted')
    rows = selected_rows(pipeline, args.manifest)
    if pipeline.load(extra.selected_sources) != rows:
        raise ValueError('Explicit selected sources differ from original curated batches')
    args.ids = [row['id'] for row in rows]
    install(pipeline)
    return pipeline.run_consumer(args)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False), flush=True)
