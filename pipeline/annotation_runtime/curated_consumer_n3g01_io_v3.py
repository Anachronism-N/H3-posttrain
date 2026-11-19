"""Finite TP2 consumer for the isolated 1000:2000 curated I/O expansion."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

RT = Path(__file__).resolve().parent
BASE = RT / 'source_curated_consumer_v1/curated_pipeline.py'
BASE_SHA = 'd5cd607d4789e3b3cb361b2f32c35c4adc438cda8b9d2f5262dc143e61e952fc'
PRIVATE_READY = 'ready.curated_io_v3_n3g01.jsonl'


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
        if start >= 2000 or start + batch['count'] <= 1000:
            continue
        if pipeline.sha(batch['path']) != batch['sha256']:
            raise ValueError('Selected original batch SHA changed')
        values = [json.loads(line) for line in Path(batch['path']).read_text().splitlines() if line.strip()]
        if len(values) != batch['count']:
            raise ValueError('Selected original batch count changed')
        for row in values:
            if row['split'] != batch['split'] or row['batch_id'] != batch['batch_id']:
                raise ValueError('Selected batch binding changed')
            if 1000 <= row['source_ordinal'] < 2000:
                rows.append({key: row[key] for key in ('id', 'split', 'batch_id', 'source_ordinal')})
    if ([row['source_ordinal'] for row in rows] != list(range(1000, 2000))
            or len({row['id'].casefold() for row in rows}) != 1000):
        raise ValueError('Expansion must be exactly 1000 unique ordinals 1000:2000')
    return rows


def install(pipeline):
    source = inspect.getsource(pipeline.run_consumer)
    changes = [
        ("visual_out = out / 'visual' / f'shard-{args.shard:04d}'",
         "visual_out = out / 'visual' / 'collaborative-n3g01-v1'"),
        ("'configuration': configuration, 'pipeline_code': ref(__file__), **FLAGS",
         "'configuration': configuration, 'pipeline_code': ref(__file__), 'consumer_wrapper': ref(CONSUMER_WRAPPER), **FLAGS"),
        ("assigned_total = sum(1 for i in range(summary['total']) if i % args.nshard == args.shard)",
         "assigned_total = len(args.ids)"),
        ("assigned_total = sum(1 for _ in iter_rows(summary, args.shard, args.nshard, ids=wanted))",
         "assigned_total = len(wanted)"),
        ("queue = out / 'ready.jsonl'", "queue = out / PRIVATE_READY"),
        ("counts['resumed_terminal'] += 1\n                    continue",
         "counts['resumed_terminal'] += 1\n                    append(out / 'ready.jsonl', terminal)\n                    continue"),
    ]
    # Terminal recheck is inside the source lock, before any visual work.
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
                        append(out / 'ready.jsonl', terminal)
                        continue
                    with visual.bundle_lock(root):"""
    changes.append((old, new))
    # Publishing occurs only after the immutable consumer terminal exists.
    changes.append(("notify_batch_terminal(args, summary, value)",
                    "append(out / 'ready.jsonl', terminal)\n"))
    for old, new in changes:
        expected = 2 if old in ("append(out / 'completed.jsonl', value)", "notify_batch_terminal(args, summary, value)") else 1
        if source.count(old) != expected:
            raise RuntimeError('Consumer wrapper patch anchor changed: ' + old)
        if old == "notify_batch_terminal(args, summary, value)":
            # Preserve batch publication and its two different indentation levels.
            source = source.replace("                    " + old,
                                    "                    append(out / 'ready.jsonl', terminal)\n                    " + old, 1)
            source = source.replace("                        " + old,
                                    "                        append(out / 'ready.jsonl', terminal)\n                        " + old, 1)
        else:
            source = source.replace(old, new)
    pipeline.CONSUMER_WRAPPER = str(Path(__file__).resolve())
    pipeline.PRIVATE_READY = PRIVATE_READY
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), pipeline.__dict__)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--selected-sources', required=True)
    extra, remaining = parser.parse_known_args(argv)
    pipeline = load_pipeline()
    args = pipeline.parser().parse_args(remaining)
    if args.mode != 'consumer' or args.shard != 0 or args.nshard != 1 or args.limit != 0 or args.ids:
        raise ValueError('Collaborative wrapper requires consumer shard0/nshard1/limit0 without explicit IDs')
    rows = selected_rows(pipeline, args.manifest)
    if pipeline.load(extra.selected_sources) != rows:
        raise ValueError('Explicit selected sources differ from original curated batches')
    args.ids = [row['id'] for row in rows]
    install(pipeline)
    return pipeline.run_consumer(args)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False), flush=True)
