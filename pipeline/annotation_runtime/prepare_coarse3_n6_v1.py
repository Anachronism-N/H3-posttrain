"""CPU plan and one-shot N6 physical GPU3 independent-three admission builder."""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import stat
import subprocess
import time
import uuid
import prepare_curated_coarse_empty_n5_v1 as h

RT = Path(__file__).resolve().parent
BAR = RT.parent
OUT = BAR / 'coarse3_filter_recovery_independent_reviews_v1'
MANAGER = RT / 'coarse3_n6_resource_manager_v1.py'
DRIVER = RT / 'coarse3_n6_driver_v1.py'
WORKER = RT / 'coarse3_independent_review_v1.py'
WORKER_SHA = 'd1a924d5fbac924f37fcb1a149e5e1929c37419009a4b7e73d691a49e5009f1d'
BASE_SHA = h.BASE_SHA
MODEL_REF = RT / 'reviewer8b_stage_N6_extension_v2.json'
MODEL_REF_SHA = '5607668eb5c476308f20ca20b2853c97bc846f5d43182df2a66729b3c9c4385f'
MODEL_SHA = '220d9152229696caec4decebb89a160afc8ac9b6ba91ab1b9d790343a82d733a'
MODEL = '/opt/models/Qwen3-VL-8B-Instruct-review-extension-v2'
OLD_OUT = BAR / 'curated_clause_salvage_reviews_v1'
STREAM = BAR / 'curated222939_clause_salvage_v1/review_required.jsonl'
_SALVAGE = None
PHASE = 'coarse3_n6_independent_review_v1'
IDS = ('AuuitdcBJyGOnWaH', 'AzPomlvsANsevIFb', 'DxNCjtNsxcazNpGb')
sha, ref, load, checked, save_once = h.sha, h.ref, h.load, h.checked, h.save_once
lease_value, verify_lease_value = h.lease_value, h.verify_lease_value
fd_inode_owned, proc_lock_owned = h.fd_inode_owned, h.proc_lock_owned


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def base_module():
    need(sha(RT / 'resource_manager.py') == BASE_SHA, 'original resource manager changed')
    base = module('coarse3_original_resource', RT / 'resource_manager.py')
    base.sha = sha
    return base


def worker_module():
    need(sha(WORKER) == WORKER_SHA, 'frozen independent-three worker changed')
    return module('coarse3_finite_independent', WORKER)


def modelref():
    return dict(path=str(MODEL_REF), sha256=MODEL_REF_SHA, model_identity_sha256=MODEL_SHA)


def checkpoint(identity):
    weights = sorted(v['sha256'] for n, v in identity['files'].items()
                     if Path(n).suffix in ('.safetensors', '.bin', '.pt', '.gguf'))
    need(bool(weights), 'actual checkpoint weight content inventory required')
    raw = (json.dumps({'weight_file_content_sha256': weights}, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    return hashlib.sha256(raw).hexdigest()


def verify_model_reference():
    value = checked(modelref())
    need(value.get('node') == 'N6' and value.get('complete') is True and value.get('full_copy_SHA_verified') is True,
         'complete verified N6 staging required')
    identity = value['model_identity']
    need(identity['root'] == MODEL and identity['sha256'] == MODEL_SHA, 'scheduled N6 8B model changed')
    teacher_ref = ref(BAR.parent / '32b_replica_stage_receipts/source_reference_v1.json')
    teacher = checked(teacher_ref)['model_identity']
    need(checkpoint(identity) != checkpoint(teacher), 'reviewer checkpoint content equals original annotator')
    return identity, teacher_ref


def verify_local_model(cfg=None):
    cfg = cfg or {'modelref': modelref()}
    need(cfg['modelref'] == modelref(), 'only staged N6 reviewer allowed')
    extension = module('coarse3_extension_storage', RT / 'curated_review_extension_resource_manager_v1.py')
    extension.base.sha = sha
    extension.check_model(cfg)  # Does not import annotation/review ms namespaces.
    identity, _ = verify_model_reference()
    for entry in identity['files'].values():
        need(sha(entry['path']) == entry['sha256'], 'full local checkpoint content changed: ' + entry['path'])
    if cfg.get('local_model_identity') is not None:
        need(identity == cfg['local_model_identity'], 'local checkpoint identity changed')
    return identity


def source_plan(m=None, *, probe=True):
    m = m or worker_module()
    lib = m.libraries()
    cfg, contract, launch_ref = m.source_contract(lib)
    population_ref = ref(OUT / 'frozen.population.json')
    population = checked(population_ref)
    need(population.get('source_ids') == list(IDS) and population.get('population_source_count') == 3
         and population.get('maximum_review_calls') == 3
         and population.get('jobs_manifest_ref') == contract['manifest_ref'], 'complete frozen independent population changed')
    sources, bindings = [], dict(contract['bindings'])
    bindings.update({str(WORKER): WORKER_SHA, str(RT / 'test_coarse3_independent_review_v1.py'): sha(RT / 'test_coarse3_independent_review_v1.py')})
    for hint, sid in zip(contract['manifest']['jobs'], IDS):
        candidate, verified = m.validate_candidate(lib, cfg, hint, ffprobe='/opt/ms_tools/ffprobe' if probe else None)
        need(candidate is not None and verified['status'] == 'verified', 'source recovery not complete: ' + sid)
        row = m.review_row(lib, candidate, verified, population_ref)
        digest = lib.batch.digest(row)
        root = OUT / 'sources' / ('train-' + sid) / digest[:24]
        evidence = load(root / 'evidence.json')
        m.verify_evidence(lib, evidence, row, root, verified['probe'])
        need(load(root / 'input.row.json') == row, 'CPU prepared current input row changed')
        need(load(root / 'independent.request.preview.json') == lib.audit.review_request(row, evidence, root, '/CPU-preview-only/no-checkpoint-loaded'),
             'CPU native preview differs from reconstructed complete label/evidence')
        prepared = load(root / 'independent.prepare.receipt.json')
        need(prepared.get('status') == 'prepared' and prepared.get('input_row_sha256') == digest
             and prepared.get('candidate_ref') == verified['candidate_ref'] and prepared.get('evidence_ref') == ref(root / 'evidence.json')
             and prepared.get('actual_media_SHA_PTS_RGB_ROI_checked') is True
             and all(prepared.get(k) is False for k in m.NO_APPROVAL), 'CPU prepared receipt changed')
        names = ('input.row.json', 'evidence.json', 'evidence.receipt.json', 'independent.request.preview.json', 'independent.prepare.receipt.json')
        refs = [ref(root / n) for n in names]
        for name, digest_value in h.references(verified).items():
            need(sha(name) == digest_value, 'original six-file/recovery source reference changed')
            bindings[name] = digest_value
        for f in evidence['frames']:
            need(sha(root / f['file']) == f['sha256'], 'prepared RGB changed')
            bindings[str(root / f['file'])] = f['sha256']
        for roi in evidence['hand_rois']:
            need(sha(roi['path']) == roi['sha256'], 'prepared ROI changed')
            bindings[roi['path']] = roi['sha256']
        bindings.update(h.references(refs))
        sources.append({'source_id': sid, 'root': str(root), 'row_sha256': digest,
                        'source_job_ref': hint['job_ref'], 'candidate_ref': verified['candidate_ref'],
                        'evidence_ref': ref(root / 'evidence.json'), 'prepared_refs': refs})
    identity, teacher_ref = verify_model_reference()
    bindings.update(h.references([population_ref, modelref(), teacher_ref, m.RECOVERY_REFS]))
    for path in (Path(__file__).resolve(), MANAGER, DRIVER, RT / 'test_coarse3_n6_v1.py', RT / 'curated_review_extension_resource_manager_v1.py'):
        bindings[str(path)] = sha(path)
    freeze = load(RT / 'source_curated_review_v2/freeze.receipt.json')
    bindings.update(freeze['files'])
    bindings[str(RT / 'source_curated_review_v2/freeze.receipt.json')] = m.LIB_SHA
    for name, digest in bindings.items():
        need(sha(name) == digest, 'source plan full SHA changed: ' + name)
    return {'schema': PHASE, 'population_ref': population_ref, 'jobs_manifest_ref': contract['manifest_ref'],
            'source_job_refs': [h['job_ref'] for h in contract['manifest']['jobs']], 'sources': sources,
            'source_sha256': bindings, 'modelref': modelref(), 'annotator_model_reference': teacher_ref,
            'reviewer_checkpoint_sha256': checkpoint(identity), 'maximum_review_calls': 3,
            'semantic_approval': False, 'training_eligible': False}


def require_unreviewed(plan):
    for source in plan['sources']:
        root = Path(source['root'])
        claim = OUT / 'review-attempts' / ('train-' + source['source_id'] + '.json')
        need(not os.path.lexists(claim) and (not (root / 'review').exists() or not any((root / 'review').iterdir())),
             'prior/partial independent attempt preserved; no redispatch')
    need(not (OUT / 'finite.stage.summary.json').exists(), 'finite stage already attempted')


def driver_env(launch):
    return dict(CUDA_VISIBLE_DEVICES='3', COARSE3_N6_CONFIG=str(launch),
                OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                HF_HUB_DISABLE_TELEMETRY='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1',
                PYTHONPATH='', PYTHONHOME='', CONDA_PREFIX='',
                MS_VISUAL_CPU_SITEPATH='/opt/venv_eval/lib/python3.11/site-packages')


def verify_config(cfg, plan, *, verify_history=True):
    launch = Path(cfg['launch_path'])
    need(launch.parent == RT and launch.name.startswith('coarse3_n6_launch_'), 'new exact finite launch path required')
    exact = dict(node='N6', gpus=[3], targetcount=3, input_count=222939, phase=PHASE,
                 out=str(OUT), cwd=str(RT), release_authorized=True, reuse_existing_results=True,
                 maximum_review_calls=3, tensor_parallel_size=1, semantic_approval=False, training_eligible=False,
                 env=driver_env(launch), argv=['/opt/venv_vllm/bin/python', '-B', '-u', str(DRIVER), '--launch', str(launch)],
                 producer=ref(DRIVER), modelref=modelref(), jobs_manifest_ref=plan['jobs_manifest_ref'],
                 population_ref=plan['population_ref'])
    need(all(cfg.get(k) == v for k, v in exact.items()), 'exact N6 GPU3 finite config/argv/environment changed')
    need(Path(cfg['queue']).parent == BAR / 'queues' and Path(cfg['queue']).name.startswith('coarse3_n6_independent_'), 'isolated queue required')
    need(checked(cfg['plan']) == plan, 'current full source plan differs from frozen launch plan')
    need(cfg.get('resource_manager') == ref(MANAGER), 'exact wrapper SHA required')
    for name, digest in plan['source_sha256'].items():
        need(cfg['source_sha256'].get(name) == digest, 'complete source SHA binding missing')
    need(str(STREAM) not in cfg['source_sha256'], 'append-only stream must not enter whole-file source freeze')
    if verify_history:
        verify_stream_prefix(cfg['stream_prefix_proof'])
        for name, digest in cfg['source_sha256'].items():
            need(sha(name) == digest, 'launch source changed: ' + name)


def salvage_modules():
    global _SALVAGE
    if _SALVAGE is None:
        path = RT / 'curated_clause_salvage_review_watch_v5.py'
        need(sha(path) == 'c79d1428c0650bce5616d28c96c34abc62677d1798703037d9f92f993cf9b6d2', 'original v5 watcher changed')
        watcher = module('coarse3_salvage_prefix_watch', path)
        resume_path = RT / 'curated_clause_salvage_review_resume_validate_v5.py'
        need(sha(resume_path) == '54cefaa1ff132118235ef443249dd6127c26b40b1bb58832ea7b4ee8408a6635', 'original v5 resume validator changed')
        resume = module('coarse3_salvage_prefix_resume', resume_path)
        _SALVAGE = watcher, resume
    return _SALVAGE


def read_stream_prefix(state):
    watcher, _ = salvage_modules()
    cursor = copy.deepcopy(state['cursor'])
    need(state['input_stream'] == str(STREAM) and type(cursor.get('offset')) is int and cursor['offset'] >= 0
         and isinstance(cursor.get('identity'), list) and len(cursor['identity']) == 2
         and isinstance(cursor.get('committed_prefix_sha256'), str), 'original committed prefix identity/SHA required')
    watcher.AppendCursor(STREAM, cursor).next()  # Real v5 inode/size/anchor/partial/full-prefix checks; no commit.
    with STREAM.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        prefix = stream.read(cursor['offset'])
        finished = os.fstat(stream.fileno())
    current = STREAM.lstat()
    need(stat.S_ISREG(current.st_mode) and current.st_uid == os.getuid()
         and all([s.st_dev, s.st_ino] == state['cursor']['identity'] for s in (opened, finished, current))
         and all(s.st_size >= state['cursor'].get('observed_size', cursor['offset']) for s in (opened, finished, current))
         and len(prefix) == cursor['offset'] and (not prefix or prefix.endswith(b'\n'))
         and hashlib.sha256(prefix).hexdigest() == state['cursor']['committed_prefix_sha256'],
         'original stream physical identity/committed prefix changed')
    return prefix


def verify_notice_binding(key, entry, transaction, row, notice, line_ref):
    watcher, _ = salvage_modules()
    identity = transaction['identity']
    need(identity['source_key'] == key == row['split'] + '/' + row['id'] == notice['source_key']
         and identity['notice_sha256'] == watcher.digest(notice)
         and identity['input_row_sha256'] == hashlib.sha256((json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()).hexdigest()
         and identity['semantic_label_sha256'] == hashlib.sha256((json.dumps(row['label'], ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()).hexdigest()
         and notice['joint_candidate_ref'] == identity['label_result_ref'] == ref(row['label_path'])
         and entry['status'] == transaction['status'], 'original prefix notice/source/input/label transaction mismatch')
    provenance = row['review_provenance']
    historical = provenance.get('original_notice_stream_ref')
    if historical is not None:
        need(historical == line_ref, 'original notice byte offset/length/line SHA changed')
    if provenance.get('original_notice_ref') is not None:
        need(checked(provenance['original_notice_ref']) == notice, 'original persisted notice differs from committed stream line')


def committed_stream_proof(state_ref):
    state = checked(state_ref)
    prefix = read_stream_prefix(state)
    seen, lines, ranges = {}, {}, []
    offset = 0
    for raw in prefix.splitlines(keepends=True):
        need(raw.endswith(b'\n'), 'incomplete committed source line')
        notice = json.loads(raw)
        key = notice['source_key']
        line_ref = {'path': str(STREAM), 'byte_offset': offset, 'byte_length': len(raw),
                    'sha256': hashlib.sha256(raw).hexdigest()}
        need(key in state['sources'], 'committed prefix source lacks original transaction')
        entry = state['sources'][key]
        transaction = checked(entry['transaction_ref'])
        row = checked(transaction['input_ref'])
        watcher, _ = salvage_modules()
        if key in seen:
            need(seen[key] == watcher.digest(notice), 'original duplicate source notice changed')
        else:
            verify_notice_binding(key, entry, transaction, row, notice, line_ref)
            seen[key] = watcher.digest(notice)
        lines[(offset, len(raw), line_ref['sha256'])] = key
        ranges.append({**line_ref, 'source_key': key, 'transaction_ref': entry['transaction_ref'],
                       'input_ref': transaction['input_ref']})
        offset += len(raw)
    need(set(seen) == set(state['sources']), 'original state/prefix source sets differ')
    need(sha(state_ref['path']) == state_ref['sha256'], 'original stable state changed during prefix proof')
    return {'schema': 'coarse3_owned_salvage_prefix_v1', 'stream_path': str(STREAM),
            'physical_identity': state['cursor']['identity'], 'original_cursor': state['cursor'],
            'state_ref': state_ref, 'consumed_prefix_bytes': len(prefix),
            'consumed_prefix_sha256': hashlib.sha256(prefix).hexdigest(), 'source_line_refs': ranges,
            'original_source_count': len(seen), 'subsequent_append_allowed': True}, prefix, lines


def preserved_references(value, line_sources, source_key=None):
    watcher, resume = salvage_modules()
    whole = {}
    for reference in resume.references(value):  # Original v5 excludes byte-range identities from whole-file SHA.
        need(reference['path'] != str(STREAM), 'historical whole-stream ref lacks bounded-prefix evidence')
        name, digest = reference['path'], reference['sha256']
        need(name not in whole or whole[name] == digest, 'conflicting original whole-file reference')
        whole[name] = digest
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if isinstance(item.get('path'), str) and isinstance(item.get('sha256'), str) and 'byte_offset' in item:
                need(item['path'] == str(STREAM) and type(item['byte_offset']) is int
                     and type(item.get('byte_length')) is int
                     and (item['byte_offset'], item['byte_length'], item['sha256']) in line_sources,
                     'unknown or changed historical source byte-range ref')
                if source_key is not None:
                    need(line_sources[(item['byte_offset'], item['byte_length'], item['sha256'])] == source_key,
                         'historical source byte-range belongs to another transaction')
                if item.get('source_key') is not None:
                    need(item['source_key'] == line_sources[(item['byte_offset'], item['byte_length'], item['sha256'])],
                         'historical byte-range source mismatch')
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return whole


def verify_stream_prefix(proof):
    need(proof['schema'] == 'coarse3_owned_salvage_prefix_v1' and proof['stream_path'] == str(STREAM)
         and proof['physical_identity'] == proof['original_cursor']['identity']
         and proof['consumed_prefix_bytes'] == proof['original_cursor']['offset']
         and proof['consumed_prefix_sha256'] == proof['original_cursor']['committed_prefix_sha256'],
         'exact original prefix proof identity changed')
    snapshot = proof['prefix_snapshot_ref']
    path = Path(snapshot['path'])
    need(path.parent == RT and path.name == 'coarse3_n6_salvage_prefix_' + proof['consumed_prefix_sha256'] + '.jsonl'
         and snapshot['sha256'] == proof['consumed_prefix_sha256'] and sha(path) == snapshot['sha256'],
         'immutable content-addressed prefix snapshot changed')
    state = checked(proof['state_ref'])
    need(state['cursor'] == proof['original_cursor'], 'original preserved state cursor changed')
    prefix = read_stream_prefix(state)
    need(path.read_bytes() == prefix, 'current physical stream differs from original prefix snapshot')


def save_prefix_snapshot(proof, prefix):
    path = RT / ('coarse3_n6_salvage_prefix_' + proof['consumed_prefix_sha256'] + '.jsonl')
    if path.exists():
        need(sha(path) == proof['consumed_prefix_sha256'], 'existing prefix snapshot differs; never overwrite')
    else:
        with path.open('xb') as stream:
            stream.write(prefix)
            stream.flush()
            os.fsync(stream.fileno())
    value = {**proof, 'prefix_snapshot_ref': ref(path)}
    verify_stream_prefix(value)
    return value


def prior_safe(transition_ref, base=None):
    base = base or base_module()
    transition = checked(transition_ref)
    need(transition.get('schema') == 'owned_salvage_finite_review_transition_v1', 'actual salvage transition required')
    admission_ref = transition['admission_ref']
    old_queue = Path(admission_ref['path']).parent
    need(old_queue.parent == BAR / 'queues' and old_queue.name.startswith('curated_clause_salvage_review_watch_n6g3_v5-')
         and Path(admission_ref['path']).name == 'admission.json', 'exact original salvage queue required')
    admission = checked(admission_ref)
    argv = admission['manager_identity']['argv']
    need(argv.count('--config') == 1, 'actual original config argv required')
    old_ref = {'path': argv[argv.index('--config') + 1], 'sha256': admission['config_sha256']}
    need(Path(old_ref['path']).parent == RT and len(old_ref['sha256']) == 64, 'actual original runtime config ref required')
    old = checked(old_ref)
    original_manager = RT / 'curated_clause_salvage_review_watch_resource_manager_v5.py'
    need(sha(original_manager) == '76d48559e7dc89f13f9835d2484757acad9ab89fc3ebf7108f4134ed83bbf983',
         'original v5 resource wrapper changed')
    need(transition['old_config_ref'] == old_ref and transition['old_manager_identity'] == admission['manager_identity'], 'old manager/config binding changed')
    need(admission['manager_sha256'] == BASE_SHA
         and old['node'] == admission['node'] == 'N6' and old['gpus'] == admission['gpus'] == [3]
         and old['queue'] == str(old_queue) and old['out'] == admission['out'] == str(OLD_OUT), 'exact old admission scope changed')
    expected_argv = ['/opt/venv_eval/bin/python', '-B', '-u', str(RT / 'curated_clause_salvage_review_watch_resource_manager_v5.py'),
                     '--config', old_ref['path'], '--config-sha', old_ref['sha256'], '--manager-sha', BASE_SHA]
    need(argv == expected_argv and admission['manager_identity']['uid'] == os.getuid(), 'old owned manager identity argv mismatch')
    need(all(admission[k] == old[k] for k in ('producer', 'plan', 'source_sha256'))
         and old['producer'] == {'path': str(RT / 'curated_clause_salvage_review_watch_v5.py'),
                                 'sha256': 'c79d1428c0650bce5616d28c96c34abc62677d1798703037d9f92f993cf9b6d2'},
         'old actual v5 producer/source/plan changed')
    need(old['modelref'] == modelref() and admission['model_identity_sha256'] == MODEL_SHA, 'old staged reviewer changed')
    for key, name in (('live_ref', 'live.json'), ('safe_ref', 'resources_safe.json'), ('restored_ref', 'holder_restored_gpu3.json')):
        need(transition[key]['path'] == str(old_queue / name), 'transition actual queue ref changed')
    live, safe, restored = (checked(transition[k]) for k in ('live_ref', 'safe_ref', 'restored_ref'))
    need(all(live['manager_identity'][k] == admission['manager_identity'][k] for k in base.IDENTITY_KEYS)
         and live['argv'] == old['argv'], 'old actual live provenance changed')
    need(safe.get('phase') == old['phase'] and all(safe.get(k) is True for k in
         ('all_holders_safe', 'locks_released', 'leases_restored', 'managed_parent_wait', 'unknown_jobs_not_signaled')), 'original resources not actually safely restored')
    need(restored['gpu'] == 3 and restored['uuid'] == admission['holders'][0]['uuid']
         and restored['original_identity'] == admission['holders'][0]['identity']
         and restored['restore_argv'] == base.RESTORE_ARGV, 'actual restored original project holder required')
    need((old_queue / 'admission.json').stat().st_mtime_ns <= (old_queue / 'holder_restored_gpu3.json').stat().st_mtime_ns
         <= (old_queue / 'resources_safe.json').stat().st_mtime_ns, 'restoration evidence ordering changed')
    signal_capture = checked(transition['signal_capture_ref'])
    need(signal_capture.get('signal') == 'SIGTERM' and signal_capture.get('ownership_verified') is True
         and isinstance(signal_capture.get('target_identity'), dict)
         and all(signal_capture['target_identity'].get(k) == admission['manager_identity'][k] for k in base.IDENTITY_KEYS)
         and signal_capture.get('unknown_jobs_not_signaled') is True, 'exact owned manager TERM capture required')
    need(transition['state_ref']['path'] == str(OLD_OUT / 'state.json'), 'exact preserved old state required')
    state = checked(transition['state_ref'])
    need(state['total_source_count'] == 222939 and state['expected_batch_count'] == 448
         and len(state['sources']) >= 107 and state.get('semantic_approval') is False, 'original 107+ source state not preserved')
    prefix_proof, prefix, line_sources = committed_stream_proof(transition['state_ref'])
    ancestor_ref = {'path': str(RT / 'curated_clause_salvage_review_watch_main_fresh_v5_1792907677082418170.json'),
                    'sha256': '49eb26d2a647ca85b481297722519491a2db303a5ba34a7ae85ebc2cd3802d44'}
    checked(ancestor_ref)
    preserved = {transition['state_ref']['path']: transition['state_ref']['sha256'],
                 ancestor_ref['path']: ancestor_ref['sha256'], str(original_manager): sha(original_manager)}
    old_review_count = 0
    for source_key, entry in state['sources'].items():
        need(Path(entry['transaction_ref']['path']).resolve().is_relative_to(OLD_OUT), 'old source transaction outside original output')
        transaction = checked(entry['transaction_ref'])
        input_row = checked(transaction['input_ref'])
        summary = checked(transaction['summary_ref'])
        preserved.update(preserved_references([entry, transaction, input_row, summary], line_sources, source_key))
        if transaction.get('ledger_ref'):
            ledger_ref = transaction['ledger_ref']
            need(sha(ledger_ref['path']) == ledger_ref['sha256'], 'old committed source ledger changed')
            for raw in Path(ledger_ref['path']).read_bytes().splitlines():
                if not raw.strip():
                    continue
                score = json.loads(raw)
                need(score['source_id'] == input_row['id'] and score['split'] == input_row['split'],
                     'original score source differs from committed input')
                preserved.update(preserved_references(score, line_sources, source_key))
                if score.get('review_receipt'):
                    receipt = checked(score['review_receipt'])
                    attempt = Path(score['review_receipt']['path']).parent
                    need(attempt.resolve().is_relative_to(BAR), 'old score receipt outside original experiment')
                    names = set(receipt.get('files_sha256', {}))
                    need(names <= {'request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'response.json', 'final.json'},
                         'unsafe old transaction file inventory')
                    if receipt.get('status') == 'candidate':
                        need(len(names) == 6, 'old completed review requires original six artifacts')
                        old_review_count += 1
                    for name, digest in receipt.get('files_sha256', {}).items():
                        preserved[str(attempt / name)] = digest
    need(old_review_count >= 107, 'original 107+ existing review scores not preserved')
    for name, digest in preserved.items():
        need(Path(name).is_absolute(), 'absolute original state reference required')
        need(sha(name) == digest, 'old existing state/score/ledger changed: ' + name)
    for name, digest in old['source_sha256'].items():
        need(name != str(STREAM), 'dynamic stream cannot enter whole-file frozen sources')
        need(Path(name).is_absolute() and sha(name) == digest, 'original frozen salvage source changed: ' + name)
        preserved[name] = digest
    for reference in transition.get('old_score_refs', []) + transition.get('ledger_refs', []):
        need(sha(reference['path']) == reference['sha256'], 'explicit old score/ledger changed')
        preserved.update(preserved_references(reference, line_sources))
    if transition.get('ledger_ref'):
        need(sha(transition['ledger_ref']['path']) == transition['ledger_ref']['sha256'], 'old ledger changed')
        preserved.update(h.references(transition['ledger_ref']))
    for name in ('failure.json', 'worker_exit.json'):
        if (old_queue / name).exists():
            preserved.update(h.references(ref(old_queue / name)))
    return {'transition_ref': transition_ref, 'restored': restored, 'old_state_source_count': len(state['sources']),
            'preserved_refs': preserved, 'ancestor_config_ref': ancestor_ref, 'stream_prefix_proof': prefix_proof, 'prefix_bytes': prefix,
            'old_review_receipt_count': old_review_count, 'old_partial_failure_preserved': True, 'old_worker_success_not_required': True}


def check_node():
    need('28.216.19.225' in subprocess.check_output(['hostname', '-I'], text=True).split(), 'only local physical N6 permitted')


def fresh_holder(base, prior):
    check_node()
    fd = os.open('/tmp/ms12k_gpu_3.lock', os.O_RDWR | os.O_NOFOLLOW)
    try:
        s = os.fstat(fd)
        need(stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid() and s.st_size == 0, 'unsafe GPU3 lock')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reserve = Path('/tmp/od_reserve_3')
        lease = lease_value(base.lease_snapshot(reserve))
        expected = dict(prior['restored']['holder_identity'], gpu=3, uuid=prior['restored']['uuid'])
        holder = dict(base.identity(expected['pid']), gpu=3, uuid=base.uuid_for(3))
        base.check_holder(expected, holder, 3, holder['uuid'], base.apps())
        need(holder['argv'] == base.RESTORE_ARGV, 'only exact restored local project holder admitted')
        pidfd = os.pidfd_open(holder['pid'])
        try:
            need(not select.select([pidfd], [], [], 0)[0], 'restored holder already exited')
        finally:
            os.close(pidfd)
        budget = base.headroom()
        need(budget['headroom'] >= 160 * 2**30, 'under 160GiB; no holder released')
        base.check_holder(holder, base.identity(holder['pid']), 3, base.uuid_for(3), base.apps())
        need(lease_value(base.lease_snapshot(reserve)) == lease, 'lease changed during CPU capture')
        return {'holders': [holder], 'lease_snapshot_ns': {str(reserve): lease}, 'budget': budget,
                'captured_unix_ns': time.time_ns(), 'signals_sent': False}
    finally:
        os.close(fd)


def build(launch, transition):
    launch = Path(launch).resolve()
    need(launch.parent == RT and launch.name.startswith('coarse3_n6_launch_') and not launch.exists(), 'new RT launch required')
    plan = source_plan()
    require_unreviewed(plan)
    identity = verify_local_model()
    prior = prior_safe(ref(Path(transition).resolve()))  # Refuses until actual safe/restored proof exists.
    fresh = fresh_holder(base_module(), prior)
    stream_prefix_proof = save_prefix_snapshot(prior['stream_prefix_proof'], prior['prefix_bytes'])
    suffix = str(time.time_ns()) + '-' + uuid.uuid4().hex[:12]
    plan_path, capture_path = RT / ('coarse3_n6_plan_' + suffix + '.json'), RT / ('coarse3_n6_capture_' + suffix + '.json')
    save_once(plan_path, plan)
    save_once(capture_path, fresh)
    cfg = dict(node='N6', gpus=[3], phase=PHASE, targetcount=3, input_count=222939,
               maximum_review_calls=3, tensor_parallel_size=1, release_authorized=True,
               queue=str(BAR / 'queues' / ('coarse3_n6_independent_' + suffix)), out=str(OUT), cwd=str(RT),
               launch_path=str(launch), env=driver_env(launch), reuse_existing_results=True,
               argv=['/opt/venv_vllm/bin/python', '-B', '-u', str(DRIVER), '--launch', str(launch)],
               producer=ref(DRIVER), resource_manager=ref(MANAGER), plan=ref(plan_path), modelref=modelref(),
               local_model_identity=identity, population_ref=plan['population_ref'], jobs_manifest_ref=plan['jobs_manifest_ref'],
               transition_ref=prior['transition_ref'], stream_prefix_proof=stream_prefix_proof, holder_capture=ref(capture_path),
               holders=fresh['holders'], lease_snapshot_ns=fresh['lease_snapshot_ns'],
               source_sha256=dict(plan['source_sha256']), semantic_approval=False, training_eligible=False)
    cfg['source_sha256'].update(prior['preserved_refs'])
    cfg['source_sha256'].update(preserved_references(
        [cfg['plan'], cfg['holder_capture'], checked(prior['transition_ref']), prior['transition_ref'], stream_prefix_proof],
        {(r['byte_offset'], r['byte_length'], r['sha256']): r['source_key'] for r in stream_prefix_proof['source_line_refs']}))
    verify_config(cfg, plan)
    save_once(launch, cfg)
    return {'launch_ref': ref(launch), 'wrapper_ref': ref(MANAGER), 'base_manager_sha256': BASE_SHA,
            'signals_sent': False, 'GPU_model_calls': 0, 'command': ['/opt/venv_eval/bin/python', '-B', '-u', str(MANAGER),
            '--config', str(launch), '--config-sha', sha(launch), '--manager-sha', BASE_SHA]}


def main():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-plan', action='store_true')
    mode.add_argument('--build', action='store_true')
    p.add_argument('--check-local-model', action='store_true')
    p.add_argument('--launch')
    p.add_argument('--transition', default=str(RT / 'coarse3_n6_salvage_transition_main_v1.json'))
    a = p.parse_args()
    if a.check_plan:
        plan = source_plan()
        require_unreviewed(plan)
        if a.check_local_model:
            verify_local_model()
        report = {'source_ids': list(IDS), 'prepared_count': 3, 'bound_file_count': len(plan['source_sha256']),
                  'population_ref': plan['population_ref'], 'modelref': plan['modelref'],
                  'checkpoint_content_independent': True, 'local_model_content_SHA_verified': a.check_local_model,
                  'GPU_model_calls': 0, 'signals_sent': False, 'lease_checks_performed': False, 'launch_written': False}
    else:
        need(a.launch is not None, '--build requires --launch')
        report = build(a.launch, a.transition)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
