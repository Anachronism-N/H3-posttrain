"""Finite, resumable candidate collection with immutable raw/complete journals."""
import json
import os
from pathlib import Path
import tempfile
import platform
from importlib import metadata
from contextlib import contextmanager

from scripts_omnidance.exp.ms.music_planner_search import (
    _keys, candidate_seed, file_sha, make_candidate, make_pool, object_sha, read_json,
    validate_generator, validate_sampling, validate_selected_rows,
)


def durable_create(path, value):
    """Publish a fully flushed immutable file without replacing an existing raw."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary), str(path))  # Atomic, exclusive, same-filesystem publication.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _status(root, value):
    # This progress snapshot is mutable; raw/attempt/complete evidence never is.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=root, suffix='.tmp', delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, root / 'status.json')


def runtime_sources():
    root = Path(__file__).resolve().parent
    names = ('music_planner_search.py', 'music_planner_search_collection.py', 'music_planner_search_identity.py',
             'music_planner_adapters.py', 'music_planner_moss_generation_patch.py',
             'music_planner_moss_position_buffer_patch.py', 'music_planner_generation_eos_patch.py')
    return {name: dict(path=str(root / name), sha256=file_sha(root / name), size_bytes=(root / name).stat().st_size)
            for name in names}


def runtime_environment():
    versions = {}
    for package in ('torch', 'transformers', 'peft', 'numpy', 'scipy', 'soundfile', 'safetensors'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return dict(python=platform.python_version(), packages=versions)


def _runtime_matches(generation, sources, generator):
    runtime = generation['runtime_provenance']
    for actual, name in ((runtime['adapter_source_ref'], 'music_planner_adapters.py'),
                         (runtime['moss_cache']['requested_ref'], 'music_planner_moss_generation_patch.py'),
                         (runtime['moss_position_buffer']['requested_ref'], 'music_planner_moss_position_buffer_patch.py'),
                         (runtime['eos_bridge']['requested_ref'], 'music_planner_generation_eos_patch.py')):
        if actual != sources[name]:
            raise ValueError('Candidate runtime source differs from collection contract: ' + name)
    if 'artifact_identity' in generator:
        receipt = generation.get('loaded_artifact_verification')
        expected = dict(schema='music_planner_loaded_artifact_verification_v1',
                        mode=generator['artifact_identity']['mode'], artifact_identity_sha256=generator['model_sha256'],
                        loaded_portable_state_verified=generator['artifact_identity']['mode'] == 'fair_portable_checkpoint')
        if receipt != expected:
            raise ValueError('Candidate lacks actual loaded artifact/portable verification')


@contextmanager
def _collector_lock(root):
    with (root / 'collector.lock').open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if stream.tell() == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError('Another collector currently owns this journal') from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ValueError('Another collector currently owns this journal') from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def collect_candidates(rows, isolation, generator, **options):
    validate_selected_rows(rows, isolation)
    root = Path(options['journal_dir']).resolve()
    new = not root.exists()
    if new:
        if options.get('resume'):
            raise ValueError('--resume requires an existing candidate journal')
        root.mkdir(parents=True)
    with _collector_lock(root):
        return _collect_locked(rows, isolation, generator, _new_journal=new, **options)


def _collect_locked(rows, isolation, generator, *, journal_dir, out, sampler,
                    count=4, seed=42, sampling=None, resume=False, context=None, sources=None,
                    _new_journal=False):
    selected = validate_selected_rows(rows, isolation)
    validate_generator(generator)
    sampling = sampling or dict(temperature=.9, top_p=.95, top_k=50, max_new_tokens=2048)
    validate_sampling(sampling)
    if type(count) is not int or not 2 <= count <= 32 or type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError('Require 2..32 candidates and a valid integer seed')
    root, out = Path(journal_dir).resolve(), Path(out).resolve()
    sources = runtime_sources() if sources is None else sources
    contract = dict(schema='music_planner_collection_contract_v1', generator=generator,
                    isolation=isolation, selected_sources=[dict(member, **audio) for member, audio in selected],
                    sampling=sampling, candidates_per_source=count, seed=seed,
                    context=context or {}, runtime_sources=sources,
                    runtime_environment=runtime_environment(),
                    protocol='native-direct-action-fixed-two-windows-v1')
    job = object_sha(contract)
    if not _new_journal:
        if not resume or read_json(root / 'contract.json') != contract:
            raise ValueError('Existing journal requires --resume and the identical model/source/params/runtime contract')
    else:
        if resume:
            raise ValueError('--resume requires an existing candidate journal')
        if out.exists():
            raise ValueError('Final pool already exists; choose a new output/journal')
        durable_create(root / 'contract.json', contract)
    tasks, completed = [], {}
    for member, audio in selected:
        for index in range(count):
            derived = candidate_seed(seed, member['id'], index)
            slot = object_sha(dict(source=member, index=index, seed=derived))
            tasks.append((slot, member, audio, index, derived))
    if len({task[0] for task in tasks}) != len(tasks):
        raise ValueError('Collection slot collision')
    # Validate ALL existing completed evidence before scheduling any new model call.
    for slot, member, audio, index, derived in tasks:
        path = root / 'candidates' / (slot + '.json')
        if not path.exists():
            continue
        stored = read_json(path)
        _keys(stored, ('schema', 'job_sha256', 'slot', 'candidate'), 'completed journal candidate')
        candidate = make_candidate(member, audio, index, generator, stored['candidate']['generation'], sampling, derived)
        if (stored['schema'] != 'music_planner_completed_candidate_v1' or stored['job_sha256'] != job
                or stored['slot'] != slot or stored['candidate'] != candidate):
            raise ValueError('Completed candidate journal identity/raw differs')
        _runtime_matches(candidate['generation'], sources, generator)
        completed[slot] = candidate
    # A process can stop after durable raw publication but before the complete
    # marker. Validate and promote that exact raw; never sample a nicer replacement.
    for slot, member, audio, index, derived in tasks:
        if slot in completed:
            continue
        raw_paths = sorted((root / 'attempts').glob(f'{slot}-*.raw.json'))
        if not raw_paths:
            continue
        recovered = None
        try:
            for path in raw_paths:
                raw = read_json(path)
                expected = dict(job_sha256=job, slot=slot, source_id=member['id'], index=index, seed=derived)
                _keys(raw, tuple(expected) + ('generation',), 'durable raw attempt')
                if any(raw[key] != value for key, value in expected.items()):
                    raise ValueError('Durable raw attempt belongs to another job/slot/seed')
                candidate = make_candidate(member, audio, index, generator, raw['generation'], sampling, derived)
                _runtime_matches(candidate['generation'], sources, generator)
                if recovered is not None and recovered != candidate:
                    raise ValueError('Multiple different durable raws exist for one slot; refusing outcome selection')
                recovered = candidate
            durable_create(root / 'candidates' / (slot + '.json'),
                           dict(schema='music_planner_completed_candidate_v1', job_sha256=job,
                                slot=slot, candidate=recovered))
            completed[slot] = recovered
        except BaseException as error:
            _status(root, dict(schema='music_planner_collection_status_v1', job_sha256=job,
                               state='incomplete', complete=len(completed), requested=len(tasks),
                               invalid_raw_slot=slot, error_type=type(error).__name__, error=str(error),
                               final_pool_published=False))
            raise
    _status(root, dict(schema='music_planner_collection_status_v1', job_sha256=job, state='collecting',
                       complete=len(completed), requested=len(tasks), final_pool_published=False))
    for slot, member, audio, index, derived in tasks:
        if slot in completed:
            continue
        attempts = root / 'attempts'
        attempts.mkdir(exist_ok=True)
        attempt = 1
        while (attempts / f'{slot}-{attempt:04d}.started.json').exists():
            attempt += 1
        stem = attempts / f'{slot}-{attempt:04d}'
        binding = dict(job_sha256=job, slot=slot, source_id=member['id'], index=index, seed=derived)
        durable_create(str(stem) + '.started.json', binding)
        try:
            generation = sampler(audio, seed=derived, **sampling)
            durable_create(str(stem) + '.raw.json', dict(binding, generation=generation))
            candidate = make_candidate(member, audio, index, generator, generation, sampling, derived)
            _runtime_matches(generation, sources, generator)
            durable_create(root / 'candidates' / (slot + '.json'),
                           dict(schema='music_planner_completed_candidate_v1', job_sha256=job,
                                slot=slot, candidate=candidate))
            completed[slot] = candidate
        except BaseException as error:
            durable_create(str(stem) + '.failed.json', dict(binding, error_type=type(error).__name__, error=str(error)))
            _status(root, dict(schema='music_planner_collection_status_v1', job_sha256=job, state='incomplete',
                               complete=len(completed), requested=len(tasks), failed_slot=slot,
                               final_pool_published=False))
            raise
    try:
        pool = make_pool(isolation, generator, sampling, count, [completed[task[0]] for task in tasks])
        if out.exists():
            if not resume or read_json(out) != pool:
                raise ValueError('Existing final pool differs; raw evidence was not overwritten')
        else:
            durable_create(out, pool)
    except BaseException as error:
        _status(root, dict(schema='music_planner_collection_status_v1', job_sha256=job, state='incomplete',
                           complete=len(completed), requested=len(tasks), final_pool_published=False,
                           publication_error=type(error).__name__ + ': ' + str(error)))
        raise
    _status(root, dict(schema='music_planner_collection_status_v1', job_sha256=job, state='complete',
                       complete=len(completed), requested=len(tasks), final_pool_published=True,
                       final_pool_sha256=file_sha(out), run_sha256=pool['run_sha256']))
    return pool
