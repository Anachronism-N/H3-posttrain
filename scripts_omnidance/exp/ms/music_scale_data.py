"""Curated genre + original MotionScript pool and conservative nested scale splits.

Candidate acoustic components are not complete song IDs or strict song isolation.
Only this tool's explicit output files are created; existing files are never replaced.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import glob
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import tempfile

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
EXP = ROOT / 'diff/logs_omnidance/exp'
AUDIO_ROOT = ROOT / 'OmniDance_processed/pose_preprocess_outputs/mainline/normalized_704x1280/main/audio'
SECONDS = 5.0625


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=unique_object)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def valid_id(sid):
    if (not isinstance(sid, str) or not sid or '/' in sid or '\\' in sid or
            '\0' in sid or sid in ('.', '..')):
        raise ValueError(f'Invalid original clip ID: {sid!r}')
    return sid


def check_outputs(paths):
    paths = [Path(path).absolute() for path in paths]
    if len(set(paths)) != len(paths):
        raise ValueError('Output paths must be distinct')
    for path in paths:
        if not path.parent.is_dir():
            raise FileNotFoundError(path.parent)
        if os.path.lexists(path):
            raise FileExistsError(path)
    return paths


def atomic_json(path, value):
    path = check_outputs([path])[0]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic no-clobber, including a racing writer.
    finally:
        if temporary is not None:
            os.unlink(temporary)


def source_record(path):
    return dict(path=str(Path(path).absolute()), sha256=sha256(path))


def assert_sources_unchanged(records):
    for record in records:
        if sha256(record['path']) != record['sha256']:
            raise RuntimeError(f"Input changed during preparation: {record['path']}")


def audio_metadata(sid, audio_root):
    import soundfile as sf
    path = Path(audio_root) / (valid_id(sid) + '.wav')
    if path.resolve().parent != Path(audio_root).resolve():
        raise ValueError(f'Audio path escapes root: {sid}')
    stat = path.stat()
    with sf.SoundFile(str(path)) as stream:
        frames, rate, channels = len(stream), stream.samplerate, stream.channels
    if rate <= 0 or frames < math.ceil(SECONDS * rate) - 1:
        raise ValueError(f'Audio shorter than {SECONDS}s beyond one-sample rounding tolerance: {sid}')
    if channels not in (1, 2):
        raise ValueError(f'Expected mono/stereo original audio: {sid}, channels={channels}')
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Audio changed during metadata read: {path}')
    return dict(id=sid, audio_path=str(path.absolute()), source_frames=frames,
                source_sr=rate, source_channels=channels, duration_seconds=frames / rate,
                size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                target_shortfall_samples=max(0, math.ceil(SECONDS * rate) - frames),
                allowed_rounding_shortfall_samples=1,
                validation='SoundFile metadata only; one-sample rounding tolerance; actual duration retained for downstream masks, no full waveform decode or content hash')


def curated_train(path):
    all_rows, passed = {}, set()
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'video_stem', 'overall_pass'} <= set(reader.fieldnames or []):
            raise ValueError('Curated train.csv requires video_stem and overall_pass')
        for row in reader:
            sid = valid_id(row['video_stem'])
            if sid in all_rows:
                raise ValueError(f'Duplicate curated train ID: {sid}')
            all_rows[sid] = row
            if row['overall_pass'].strip().lower() in ('true', '1', '1.0', 'yes'):
                passed.add(sid)
    return all_rows, passed


def fixed_holdouts(csv_paths, audio_root):
    ids, cases = set(), []
    for path in csv_paths:
        with open(path, newline='', encoding='utf-8') as stream:
            rows = list(itertools.islice(csv.DictReader(stream), 3))
        if len(rows) != 3:
            raise ValueError(f'Fixed-case CSV needs its first three cases: {path}')
        for index, row in enumerate(rows):
            sid = valid_id(Path(row['audio_path']).stem)
            if Path(row['audio_path']).resolve() != (Path(audio_root) / (sid + '.wav')).resolve():
                raise ValueError(f'Fixed case audio is not under original root: {row["audio_path"]}')
            ids.add(sid)
            cases.append(dict(csv=str(Path(path).absolute()), row=index + 2,
                              sample_id=valid_id(row['sample_id']), audio_id=sid))
    if len(ids) != 6:
        raise ValueError(f'Exactly six distinct fixed audio tracks required, found {len(ids)}')
    return ids, cases


def make_pool(args):
    out, report_path, audit_split_path = check_outputs([args.out, args.report, args.audit_split_file])
    from compose_caption import GENRE_PHRASE, COARSE

    paths = sorted(Path(args.ms).glob('*.json'))
    vlm_paths = sorted(glob.glob(args.vlm))
    if not paths or not vlm_paths:
        raise ValueError('No MotionScript JSONs or VLM JSONL sources found')
    holdout_csvs = args.holdout_csv or [str(EXP / 'E2/full_joint.csv'), str(EXP / 'E2/full_aswap.csv')]
    named = {'train_csv': args.train_csv, 'legacy_caps': args.legacy_caps, 'legacy_split': args.legacy_split}
    sources = {name: source_record(path) for name, path in named.items()}
    vlm_sources = [source_record(path) for path in vlm_paths]
    holdout_sources = [source_record(path) for path in holdout_csvs]
    train_rows, train_passed = curated_train(args.train_csv)
    legacy, legacy_split = read_json(args.legacy_caps), read_json(args.legacy_split)
    if not isinstance(legacy, dict) or not isinstance(legacy_split, dict):
        raise ValueError('Legacy captions and split must be dictionaries')
    legacy_train = id_list(legacy_split.get('train'), 'legacy train')
    legacy_val = id_list(legacy_split.get('val'), 'legacy val')
    if (legacy_split.get('seed') != 42 or legacy_train & legacy_val or
            legacy_train | legacy_val != set(legacy) or len(legacy_val) != args.val_size):
        raise ValueError('Require complete seed42 legacy split and unchanged validation IDs')
    if len(legacy) != args.legacy_size:
        raise ValueError(f'Expected {args.legacy_size} legacy captions, got {len(legacy)}')
    for sid, item in legacy.items():
        valid_id(sid)
        if (not isinstance(item, dict) or not isinstance(item.get('dims'), dict) or
                item['dims'].get('motion_source') != 'original' or
                not isinstance(item.get('caption'), str) or not item['caption'].strip() or
                not isinstance(item.get('motion_original'), str) or not item['motion_original'].strip()):
            raise ValueError(f'Invalid original legacy caption: {sid}')
    holdout_ids, holdout_cases = fixed_holdouts(holdout_csvs, args.audio_root)
    if holdout_ids & set(legacy):
        raise ValueError('Fixed generation audio unexpectedly overlaps legacy captions')

    labels, label_sources, label_ids = {}, defaultdict(list), set()
    for path in vlm_paths:
        with open(path, encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line, object_pairs_hook=unique_object)
                sid = valid_id(row['sid'])
                label_ids.add(sid)
                label_sources[sid].append(dict(path=str(Path(path).absolute()), line=line_number))
                label = row.get('label')
                if isinstance(label, dict) and label:
                    if sid in labels and labels[sid] != label:
                        raise ValueError(f'Conflicting VLM labels for {sid}')
                    labels[sid] = label

    def load_ms(path):
        data = path.read_bytes()
        item = json.loads(data, object_pairs_hook=unique_object)
        sid = valid_id(item.get('sid'))
        if sid != path.stem:
            raise ValueError(f'MotionScript ID/filename mismatch: {path}')
        return sid, item, dict(path=str(path.absolute()), sha256=hashlib.sha256(data).hexdigest())

    motion, ms_sources = {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        for sid, item, source in workers.map(load_ms, paths):
            if sid in motion:
                raise ValueError(f'Duplicate MotionScript ID: {sid}')
            motion[sid], ms_sources[sid] = item, source
    caps, rejected, provenance, legacy_exemptions = {}, {}, {}, []
    universe = sorted(set(motion) | label_ids | set(legacy))
    for sid in universe:
        reasons = []
        if sid not in train_rows:
            reasons.append('not_in_curated_train')
        elif sid not in train_passed:
            reasons.append('curated_overall_pass_false')
        item = motion.get(sid)
        if item is None:
            reasons.append('missing_ms')
            sentences = []
        else:
            raw = item.get('ms')
            if not isinstance(raw, list) or any(not isinstance(text, str) for text in raw):
                reasons.append('empty_or_invalid_ms')
                sentences = []
            else:
                sentences = [text.strip() for text in raw if text.strip()]
                if not sentences:
                    reasons.append('empty_ms')
        label = labels.get(sid)
        genre = label.get('genre') if label else None
        confidence = label.get('genre_conf') if label else None
        if sid not in legacy:
            if label is None:
                reasons.append('unlabeled')
            elif confidence not in ('high', 'medium'):
                reasons.append('genre_conf_not_high_or_medium')
            if genre not in GENRE_PHRASE:
                reasons.append('unrecognized_vlm_genre')
        elif label is None or confidence not in ('high', 'medium'):
            legacy_exemptions.append(sid)
        if sid in holdout_ids:
            reasons.append('fixed_generation_holdout')
        if reasons:
            rejected[sid] = reasons
            continue
        if sid in legacy:
            caps[sid] = legacy[sid]  # All original dimensions/text are preserved verbatim.
            genre_source = 'legacy_unchanged'
        else:
            body = ' '.join(sentences)
            caps[sid] = dict(dims=dict(genre=genre, coarse=COARSE[genre], genre_conf=confidence,
                                       motion_source='original'),
                             caption=f'<Subject 1> performs {GENRE_PHRASE[genre]}. ' + body,
                             motion_original=body)
            genre_source = 'vlm'
        provenance[sid] = dict(ms=ms_sources[sid], vlm=label_sources.get(sid, []),
                               genre_source=genre_source, legacy_caption=sid in legacy)

    def inspect(sid):
        try:
            return sid, audio_metadata(sid, args.audio_root), None
        except Exception as error:
            return sid, None, f'{type(error).__name__}: {error}'

    audio = {}
    audio_errors = {}
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        for sid, metadata, error in workers.map(inspect, sorted(set(caps) | holdout_ids)):
            if error is not None:
                audio_errors[sid] = error
                rejected.setdefault(sid, []).append('audio_metadata_error')
                caps.pop(sid, None)
                provenance.pop(sid, None)
            else:
                audio[sid] = metadata
    for sid in sorted(holdout_ids):
        if sid in audio:
            caps[sid] = dict(dims=dict(genre='generation_holdout'), motion_original='', audit_only=True)
            provenance[sid] = dict(audit_only=True, fixed_cases=[c for c in holdout_cases if c['audio_id'] == sid])
    missing_legacy = sorted(set(legacy) - set(caps))
    counts = Counter(reason for reasons in rejected.values() for reason in reasons)
    report = dict(stage='music_scale_pool', args=vars(args), source_paths_hashes=sources,
                  vlm_sources=vlm_sources, fixed_case_sources=holdout_sources,
                  ms_source_manifest_sha256=object_sha256(ms_sources), ms_sources=ms_sources,
                  n_ms=len(motion), n_vlm_ids=len(label_ids), n_vlm_labeled=len(labels),
                  n_curated_train=len(train_rows), n_curated_train_pass=len(train_passed),
                  n_pool=len(caps), n_train_eligible=len(caps) - len(holdout_ids & set(caps)),
                  n_audit_only=len(holdout_ids & set(caps)), reject_reason_counts=dict(sorted(counts.items())),
                  rejected_ids=rejected, audio_metadata_errors=audio_errors,
                  missing_legacy_ids=missing_legacy, legacy_confidence_exemptions=legacy_exemptions,
                  legacy_policy='Preserve original caption values/dimensions exactly; genre confidence gate applies only to new IDs.',
                  generation_holdout_ids=sorted(holdout_ids), fixed_cases=holdout_cases,
                  id_sha256=object_sha256(sorted(caps)), audio_metadata=[audio[sid] for sid in sorted(audio)],
                  provenance=provenance,
                  genre_distribution=dict(sorted(Counter(c['dims']['genre'] for c in caps.values()
                                                          if not c.get('audit_only')).items())),
                  scope='Curated train overall_pass only; new IDs require known VLM genre with high/medium confidence, no fallback. '
                        'New supervision is genre + original MS, no audio tempo or new dimensions. '
                        'Original 2k captions intentionally retain existing dimensions and text; use dance_only downstream. '
                        'Audio checks read metadata only, not complete decode. Fixed six tracks have no artificial captions.')
    records = list(sources.values()) + vlm_sources + holdout_sources + list(ms_sources.values())
    assert_sources_unchanged(records)
    if missing_legacy or holdout_ids - set(caps):
        report['status'] = 'invalid_missing_legacy_or_holdout'
        atomic_json(report_path, report)
        raise ValueError('Pool incomplete: required legacy captions/fixed tracks rejected; see report')
    caps = {sid: caps[sid] for sid in sorted(caps)}
    assert legacy_val <= set(caps)
    audit_val = legacy_split['val'] + sorted(holdout_ids)
    audit_train = sorted(set(caps) - set(audit_val))
    assert set(audit_train).isdisjoint(audit_val) and set(audit_train) | set(audit_val) == set(caps)
    audit_split = dict(train=audit_train, val=audit_val, seed=42,
                       scope='Preliminary candidate-audit partition only; not clean scale-training split. '
                             'Legacy val IDs followed by fixed six generation audio IDs are holdouts.')
    atomic_json(out, caps)
    atomic_json(audit_split_path, audit_split)
    report.update(status='complete', pool_path=str(out), pool_sha256=sha256(out),
                  audit_split=audit_split, audit_split_path=str(audit_split_path),
                  audit_split_sha256=sha256(audit_split_path))
    atomic_json(report_path, report)
    print(f'pool={len(caps)} train_eligible={report["n_train_eligible"]} audit_only={report["n_audit_only"]}')


def id_list(values, name):
    if not isinstance(values, list) or not values:
        raise ValueError(f'{name} must be a nonempty ID list')
    for sid in values:
        valid_id(sid)
    if len(set(values)) != len(values):
        raise ValueError(f'Duplicate IDs in {name}')
    return set(values)


def candidate_groups(ids, audit):
    parent = {sid: sid for sid in ids}

    def root(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    edges = []
    for field in ('exact_groups', 'suspected_pairs'):
        rows = audit.get(field)
        if not isinstance(rows, list):
            raise ValueError(f'Audit missing complete {field}')
        for row in rows:
            members = row.get('ids')
            group = id_list(members, field)
            if len(group) < 2 or not group <= ids:
                raise ValueError(f'Invalid/unknown IDs in audit {field}: {members}')
            if field == 'suspected_pairs' and row.get('suspected') is not True:
                raise ValueError('suspected_pairs contains a row not explicitly suspected')
            edges.extend((members[0], sid) for sid in members[1:])
    for a, b in edges:
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for sid in sorted(ids):
        groups[root(sid)].append(sid)
    return sorted(groups.values(), key=lambda members: members[0]), len(edges)


def genre_round_robin(ids, caps, seed):
    buckets = defaultdict(list)
    for sid in sorted(ids):
        buckets[caps[sid]['dims']['genre']].append(sid)
    rng = random.Random(seed)
    for key in sorted(buckets):
        rng.shuffle(buckets[key])
    order = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                order.append(buckets[key].pop())
    return order


def make_split(args):
    prefix = Path(args.prefix).absolute()
    paths = {name: Path(str(prefix) + suffix) for name, suffix in (
        ('subset_split', f'_train{args.subset_size}_split.json'),
        ('train_split', f'_train{args.train_size}_split.json'),
        ('caps', f'_caps{args.train_size + args.val_size}.json'),
        ('norm_ids', f'_norm_ids{args.subset_size}.json'), ('report', '_report.json'))}
    check_outputs(paths.values())
    sources = {name: source_record(path) for name, path in
               (('pool', args.pool), ('audit', args.audit), ('legacy_split', args.legacy_split))}
    caps, audit, legacy = read_json(args.pool), read_json(args.audit), read_json(args.legacy_split)
    if not isinstance(caps, dict) or not caps or not isinstance(audit, dict) or not isinstance(legacy, dict):
        raise ValueError('Pool, audit and legacy split must be dictionaries')
    ids = set(caps)
    for sid, item in caps.items():
        valid_id(sid)
        if not isinstance(item, dict) or not isinstance(item.get('dims'), dict):
            raise ValueError(f'Missing caption dims: {sid}')
        if not isinstance(item['dims'].get('genre'), str) or not item['dims']['genre']:
            raise ValueError(f'Missing genre: {sid}')
        if not item.get('audit_only') and (item['dims'].get('motion_source') != 'original' or
                                           not isinstance(item.get('motion_original'), str) or
                                           not item['motion_original'].strip()):
            raise ValueError(f'Missing original MotionScript body: {sid}')
    if audit.get('input_sha256', {}).get('caps') != sources['pool']['sha256']:
        raise ValueError('Audit caps SHA256 does not match this exact pool file')
    metadata = audit.get('audio_metadata')
    if not isinstance(metadata, list):
        raise ValueError('Audit must include full audio_metadata ID coverage')
    audited_ids = id_list([row['id'] for row in metadata], 'audit audio_metadata')
    if audited_ids != ids or audit.get('n_ids') != len(ids):
        raise ValueError('Audit must cover exactly the complete pool ID union, including generation holdouts')
    if 'ids' in audit and id_list(audit['ids'], 'audit ids') != ids:
        raise ValueError('Explicit audit IDs differ from complete pool')
    legacy_train, val = id_list(legacy.get('train'), 'legacy train'), id_list(legacy.get('val'), 'legacy val')
    if len(val) != args.val_size or len(legacy_train) != args.subset_size:
        raise ValueError('Legacy split counts must match subset_size and val_size')
    if legacy.get('seed') != 42 or legacy_train & val or not (legacy_train | val) <= ids:
        raise ValueError('Require intact, disjoint seed42 legacy split IDs present in pool')
    audit_only = {sid for sid in ids if caps[sid].get('audit_only') or
                  caps[sid]['dims']['genre'] == 'generation_holdout'}
    if (legacy_train | val) & audit_only:
        raise ValueError('Legacy split contains generation holdout/audit-only IDs')
    groups, n_edges = candidate_groups(ids, audit)
    excluded_groups, within_val, representatives, deduplicated = [], [], [], []
    for members in groups:
        group = set(members)
        val_members, generation_members = sorted(group & val), sorted(group & audit_only)
        if len(val_members) > 1:
            within_val.append(dict(ids=members, val_ids=val_members,
                                   reason='Legacy validation kept unchanged despite known candidate cluster'))
        if val_members or generation_members:
            excluded_groups.append(dict(ids=members, val_ids=val_members,
                                        generation_holdout_ids=generation_members,
                                        excluded_train_ids=sorted(group - val - audit_only),
                                        reasons=(['touches_fixed_validation'] if val_members else []) +
                                                (['touches_generation_holdout'] if generation_members else [])))
            continue
        original = sorted(group & legacy_train)
        representative = original[0] if original else members[0]
        representatives.append(representative)
        if len(members) > 1:
            deduplicated.append(dict(ids=members, representative=representative,
                                     excluded_train_ids=sorted(group - {representative})))
    clean_original = sorted(set(representatives) & legacy_train)
    additions = genre_round_robin(set(representatives) - set(clean_original), caps, args.seed)
    subset = clean_original + additions[:max(0, args.subset_size - len(clean_original))]
    train = subset + genre_round_robin(set(representatives) - set(subset), caps, args.seed)[:args.train_size - len(subset)]
    report = dict(stage='music_scale_nested_split', args=vars(args), source_paths_hashes=sources,
                  output_paths={name: str(path) for name, path in paths.items()},
                  n_pool=len(ids), n_audit_only=len(audit_only), n_candidate_edges=n_edges,
                  n_candidate_groups=sum(len(group) > 1 for group in groups),
                  n_train_representatives=len(representatives), legacy_train_n=len(legacy_train),
                  retained_original_clean_train_ids=clean_original,
                  removed_original_train_ids=sorted(legacy_train - set(clean_original)),
                  n_original_clean_retained=len(clean_original),
                  excluded_groups=excluded_groups, deduplicated_train_groups=deduplicated,
                  known_withinval_clusters=within_val, generation_holdout_ids=sorted(audit_only),
                  fixed_val_ids=legacy['val'], fixed_val_order_sha256=object_sha256(legacy['val']),
                  fixed_val_set_sha256=object_sha256(sorted(val)),
                  pool_id_sha256=object_sha256(sorted(ids)), source_audit_sha256=sources['audit']['sha256'],
                  requested_train_size=args.train_size, requested_subset_size=args.subset_size,
                  scope='Conservative exact+suspected candidate graph, not full songID or strict song isolation. '
                        'Entire groups touching fixed val or generation audio are excluded from training. '
                        'Only one representative per remaining group; prioritize original train IDs. '
                        'Fixed legacy val (including known within-val clusters) is never changed; no legacy result is rewritten. '
                        'Norm IDs are the clean nested training subset only.')
    if len(subset) != args.subset_size or len(train) != args.train_size:
        report.update(status='insufficient_train_representatives', available=len(representatives))
        assert_sources_unchanged(sources.values())
        atomic_json(paths['report'], report)
        raise ValueError(f'Only {len(representatives)} candidate-group representatives, need {args.train_size}; see report')
    assert set(subset) <= set(train) and not set(train) & (val | audit_only)
    group_map = {sid: index for index, members in enumerate(groups) for sid in members}
    assert len({group_map[sid] for sid in train}) == len(train)
    assert not {group_map[sid] for sid in train} & {group_map[sid] for sid in val | audit_only}
    selected_caps = {sid: caps[sid] for sid in sorted(set(train) | val)}
    train_split = dict(train=train, val=legacy['val'], seed=args.seed)
    subset_split = dict(train=subset, val=legacy['val'], seed=args.seed)
    report.update(status='complete', n_train=len(train), n_subset=len(subset), n_val=len(val),
                  same_validation_assertion=True, nested_train_assertion=True,
                  one_train_representative_per_candidate_group=True,
                  nested_split_sha256={'subset': object_sha256(subset_split), 'train': object_sha256(train_split)},
                  norm_ids_sha256=object_sha256(subset),
                  genre_distributions={name: dict(sorted(Counter(caps[sid]['dims']['genre'] for sid in selected).items()))
                                       for name, selected in (('train', train), ('subset', subset), ('val', legacy['val']))})
    assert_sources_unchanged(sources.values())
    for name, value in (('subset_split', subset_split), ('train_split', train_split),
                        ('caps', selected_caps), ('norm_ids', subset)):
        atomic_json(paths[name], value)
    report['output_sha256'] = {name: sha256(path) for name, path in paths.items() if name != 'report'}
    atomic_json(paths['report'], report)
    print(f'train={len(train)} nested_subset={len(subset)} fixed_val={len(val)} known_withinval_clusters={len(within_val)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pool = commands.add_parser('pool', help='Build flat caption pool and separate provenance report')
    pool.add_argument('--out', required=True)
    pool.add_argument('--report', required=True)
    pool.add_argument('--audit_split_file', required=True, help='Fresh JSON for preliminary full-pool audit train/val partition')
    pool.add_argument('--ms', default=str(EXP / 'MS/ms_orig'))
    pool.add_argument('--vlm', default=str(EXP / 'MS/vlm/labels_s[0-3].jsonl'))
    pool.add_argument('--train_csv', default=str(ROOT / 'OmniDance_processed/pose_preprocess_outputs/mainline/vlm_filter_235b/splits/train.csv'))
    pool.add_argument('--legacy_caps', default=str(EXP / 'MS/captions_orig_2k.json'))
    pool.add_argument('--legacy_split', default=str(EXP / 'MS/music_encoders/muq_linear_probe/split.json'))
    pool.add_argument('--val_size', type=int, default=200, help='Override only for small synthetic fixtures')
    pool.add_argument('--audio_root', default=str(AUDIO_ROOT))
    pool.add_argument('--holdout_csv', action='append', help='Two fixed-case CSVs; use first 3 tracks from each')
    pool.add_argument('--workers', type=int, default=8)
    pool.add_argument('--legacy_size', type=int, default=2000, help='Override only for small synthetic fixtures')
    split = commands.add_parser('split', help='Consume full-pool audit; never run an audit or train')
    split.add_argument('--pool', required=True)
    split.add_argument('--audit', required=True)
    split.add_argument('--legacy_split', default=str(EXP / 'MS/music_encoders/muq_linear_probe/split.json'))
    split.add_argument('--prefix', required=True, help='Existing parent plus output basename, not a directory')
    split.add_argument('--train_size', type=int, default=6000)
    split.add_argument('--subset_size', type=int, default=1800)
    split.add_argument('--val_size', type=int, default=200, help='Override only for small synthetic fixtures')
    split.add_argument('--seed', type=int, choices=[42], default=42)
    args = parser.parse_args()
    if args.command == 'pool':
        if not 1 <= args.workers <= 64 or args.legacy_size < 1 or args.val_size < 1:
            parser.error('workers must be in [1,64], legacy_size and val_size positive')
        make_pool(args)
    else:
        if not 0 < args.subset_size < args.train_size or args.val_size < 1:
            parser.error('Require 0 < subset_size < train_size and positive val_size')
        make_split(args)


if __name__ == '__main__':
    main()
