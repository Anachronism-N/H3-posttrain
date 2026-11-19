"""Immutable 30k source reservation and explicit three-node writer assignments."""
import hashlib
import importlib.util
import json
from pathlib import Path

RT = Path(__file__).resolve().parent
OUT = RT.parent / 'curated222939_collaborative_v3'
START, END = 32000, 62000
PLAN = RT / 'curated_collaborative3_source_plan_v1.json'
MANIFEST = RT.parent / 'curated222939_inputs_v1/manifest.summary.json'
BASE = RT / 'source_curated_consumer_v1/curated_pipeline.py'
BASE_SHA = 'd5cd607d4789e3b3cb361b2f32c35c4adc438cda8b9d2f5262dc143e61e952fc'
HOSTS = {'N2': '28.216.17.70', 'N4': '28.216.19.143', 'N7': '28.216.18.144'}
MEMBERS = {
    'n2p0': ('N2', 'producer', [0], 0, 4),
    'n2p1': ('N2', 'producer', [1], 1, 4),
    'n7p2': ('N7', 'producer', [0], 2, 4),
    'n7p3': ('N7', 'producer', [1], 3, 4),
    'n2c0': ('N2', 'consumer', [2, 3], 0, 9),
    'n2c1': ('N2', 'consumer', [4, 5], 1, 9),
    'n2c2': ('N2', 'consumer', [6, 7], 2, 9),
    'n4c3': ('N4', 'consumer', [0, 1], 3, 9),
    'n4c4': ('N4', 'consumer', [2, 3], 4, 9),
    'n4c5': ('N4', 'consumer', [4, 5], 5, 9),
    'n4c6': ('N4', 'consumer', [6, 7], 6, 9),
    'n7c7': ('N7', 'consumer', [2, 3], 7, 9),
    'n7c8': ('N7', 'consumer', [4, 5], 8, 9),
}
RESERVATIONS = [RT / 'curated_collaborative21000_n5g0_launch_v2.json',
                RT / 'curated_collaborative30000_n5g0_launch_v1.json']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def ref(path):
    return {'path': str(Path(path)), 'sha256': sha(path)}


def checked(reference):
    if sha(reference['path']) != reference['sha256']:
        raise RuntimeError('Frozen reference SHA changed: ' + reference['path'])
    return load(reference['path'])


def write(path, value, identical=False):
    path = Path(path)
    if identical and path.exists():
        if load(path) != value:
            raise RuntimeError('Immutable artifact changed: ' + str(path))
        return ref(path)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    return ref(path)


def pipeline():
    if sha(BASE) != BASE_SHA:
        raise RuntimeError('Native consumer snapshot SHA changed')
    spec = importlib.util.spec_from_file_location('collab3_native', BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows_from_manifest(summary):
    if (summary['total'] != 222939 or summary['counts'] != {'train': 200568, 'val': 11110, 'test': 11261}
            or len(summary['batches']) != 448 or summary['batch_size'] != 500):
        raise RuntimeError('Entire 222939/448-batch descriptor changed')
    batches = [b for b in summary['batches'] if START <= b['first_source_ordinal'] < END]
    if ([b['first_source_ordinal'] for b in batches] != list(range(START, END, 500))
            or any(b['count'] != 500 or b['split'] != 'train' for b in batches)):
        raise RuntimeError('Original 60 x 500 train source batches changed')
    rows = []
    for batch in batches:
        if sha(batch['path']) != batch['sha256']:
            raise RuntimeError('Selected original batch SHA changed')
        values = [json.loads(line) for line in Path(batch['path']).read_text().splitlines() if line.strip()]
        if len(values) != batch['count']:
            raise RuntimeError('Selected batch count changed')
        for offset, row in enumerate(values):
            if (row['source_ordinal'] != batch['first_source_ordinal'] + offset
                    or row['split'] != 'train' or row['batch_id'] != batch['batch_id']):
                raise RuntimeError('Original row ordinal/split/batch binding changed')
            rows.append({k: row[k] for k in ('id', 'split', 'batch_id', 'source_ordinal')})
    if ([r['source_ordinal'] for r in rows] != list(range(START, END))
            or len({r['id'].casefold() for r in rows}) != END - START):
        raise RuntimeError('Exactly 30k unique original source ordinals required')
    return rows, batches


def paths(label):
    node, mode, gpus, shard, nshard = MEMBERS[label]
    stem = 'curated_collaborative3_' + label + '_v1'
    return {'selected': RT / (stem + '_selected.json'), 'launch': RT / (stem + '_launch.json'),
            'queue': RT.parent / 'queues' / stem,
            'out': OUT if mode == 'producer' else OUT / 'consumers' / ('shard-%04d' % shard)}


def verify_plan(value, verify_sources=True):
    expected = {label: {'node': n, 'mode': m, 'gpus': g, 'shard': s, 'nshard': ns}
                for label, (n, m, g, s, ns) in MEMBERS.items()}
    if (value['source_range'] != {'start': START, 'end': END} or value['members'] != expected
            or value['selected_count'] != 30000 or value['input_count'] != 222939
            or value['manifest_summary'] != ref(MANIFEST) or value['out'] != str(OUT)
            or value['reserved_review'] != {'N7': [6, 7]}):
        raise RuntimeError('Fixed range/member/output assignment changed')
    previous = value['previous_reservations']
    if {r['path'] for r in previous} != {str(p) for p in RESERVATIONS}:
        raise RuntimeError('Prior reservation inventory changed')
    for reference in previous:
        old = checked(reference)
        if old['source_range']['end'] > START:
            raise RuntimeError('Expansion overlaps previously reserved ordinals')
    if verify_sources:
        rows, batches = rows_from_manifest(checked(value['manifest_summary']))
        if batches != value['selected_original_batches']:
            raise RuntimeError('Original batch descriptor membership changed')
        for label, (_, _, _, shard, nshard) in MEMBERS.items():
            if checked(value['selected_sources'][label]) != [r for r in rows if r['source_ordinal'] % nshard == shard]:
                raise RuntimeError('Shard selected-source membership changed')
    return value


def prepare_plan():
    summary = load(MANIFEST)
    rows, batches = rows_from_manifest(summary)
    refs = {}
    for label, (_, _, _, shard, nshard) in MEMBERS.items():
        refs[label] = write(paths(label)['selected'], [r for r in rows if r['source_ordinal'] % nshard == shard], identical=True)
    value = {'schema': 'curated_collaborative3_fixed_source_plan_v1',
             'source_range': {'start': START, 'end': END}, 'selected_count': len(rows), 'input_count': 222939,
             'full_counts': summary['counts'], 'full_batch_count': len(summary['batches']),
             'manifest_summary': ref(MANIFEST), 'original_full_manifest': summary['manifest'],
             'original_csv_inputs': summary['inputs'], 'selected_original_batches': batches,
             'previous_reservations': [ref(p) for p in RESERVATIONS],
             'members': {label: {'node': n, 'mode': m, 'gpus': g, 'shard': s, 'nshard': ns}
                         for label, (n, m, g, s, ns) in MEMBERS.items()},
             'selected_sources': refs, 'out': str(OUT), 'reserved_review': {'N7': [6, 7]},
             'single_ready_writer_per_file': True, 'single_completed_writer_per_shard': True,
             'semantic_approval': False, 'formal_training': False}
    verify_plan(value)
    return write(PLAN, value, identical=True)
