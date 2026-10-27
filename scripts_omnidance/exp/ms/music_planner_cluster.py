"""Produce and audit fixed-size multi-node planner jobs; never dispatch training.

JSON spec: {"layout":"split32"|"single64", "jobs":[{"family":"moss"|
"omni_captioner", "hosts":[canonical host names], "model":"...", "train":"...",
"val":"...", "out":"...", "master_port":29551, "job_id":"unique-id", ...}]}.
split32 requires two disjoint four-host jobs, one per family; single64 one eight-host
job. Each host has eight A800 80GB GPUs, verified only by a real preflight. The
default micro=accum=1 preserves the old global batch 32 on 32 ranks; 64 ranks use
global batch 64 and consequently a different update budget. No elastic resizing.

Usage: plan --spec spec.json --output cluster.json; preflight --plan cluster.json
--receipts receipts [--dry-run]; verify --plan cluster.json --receipts receipts.
commands --plan cluster.json [--receipts receipts] prints quoted shell commands.
Run preflight once on every expected host, then verify before manually dispatching.
Smoke: phase=smoke,max_updates=2. Fresh-process resume smoke: phase=smoke,
max_updates=3,resume=<checkpoint-0000002>. Full continuation: phase=full,resume=...
Every phase gets its own plan/receipts; --resume retains the original training out.

torchrun static rendezvous fixes node_rank/global rank ranges. max_restarts=0 is
deliberate: the trainer resumes explicit complete ZeRO checkpoints, not elastic
world-size changes (https://docs.pytorch.org/docs/stable/elastic/run.html). ZeRO-3
shards parameters as well as optimizer state; all ranks must save/load together
(https://deepspeed.readthedocs.io/en/stable/zero3.html). This tool does not prove
model training, GPU memory fit, network bandwidth, or model quality.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid


SCHEMA = 'music-planner-cluster-plan-v1'
RECEIPT = 'music-planner-cluster-preflight-v1'
FAMILIES = ('moss', 'omni_captioner')
HOST = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$')
DEPENDENCIES = ('torch', 'transformers', 'accelerate', 'deepspeed', 'peft',
                'numpy', 'scipy', 'soundfile', 'librosa')
APIS = ('native_captioner', 'processor', 'accelerator', 'zero3_plugin', 'zero_init',
        'gathered_parameters', 'peft_fused_expert_targets', 'nccl_backend')
CORE = ('music_planner_train.py', 'music_planner_adapters.py',
        'music_planner_cluster.py', 'music_planner_jev.py',
        'music_planner_segment_loss.py', 'music_planner_schema.py',
        'music_planner_token_loss.py')
REQUIRED_JOB = {'family', 'hosts', 'model', 'train', 'val', 'out', 'master_port', 'job_id'}
OPTIONAL_JOB = {'master_addr', 'python', 'repo_root', 'phase', 'max_updates', 'resume',
                'epochs', 'seed', 'rank', 'alpha', 'lora_lr', 'projector_lr',
                'grad_clip', 'warmup_updates', 'max_length', 'save_every', 'eval_every',
                'audio_last_n', 'loss_chunk_size', 'caption_contract', 'genre_classes',
                'train_genre_labels', 'val_genre_labels', 'genre_weight', 'genre_lr',
                'genre_head_dim', 'lr_schedule', 'warmup_ratio',
                'weights_manifest', 'deepspeed', 'expected_versions'}
FILE_FLAGS = {'caption_contract': '--caption-contract', 'genre_classes': '--genre-classes',
              'train_genre_labels': '--train-genre-labels', 'val_genre_labels': '--val-genre-labels',
              'weights_manifest': '--weights-manifest', 'deepspeed': '--deepspeed'}
VALUE_FLAGS = {key: '--' + key.replace('_', '-') for key in
               ('epochs', 'seed', 'rank', 'alpha', 'lora_lr', 'projector_lr', 'grad_clip',
                'warmup_updates', 'max_length', 'save_every', 'eval_every', 'audio_last_n',
                'loss_chunk_size', 'genre_weight', 'genre_lr', 'genre_head_dim', 'lr_schedule', 'warmup_ratio')}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only replace this explicitly named generated artifact, never arbitrary trees.
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def absolute(value, directory=False):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('A nonempty local path is required')
    path = Path(value).expanduser().resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError('Missing local ' + ('directory: ' if directory else 'file: ') + str(path))
    return path


def file_ref(path):
    path = absolute(str(path))
    return dict(path=str(path), size=path.stat().st_size, sha256=file_sha(path))


def model_inventory(path):
    """Hash actual local payload bytes, without importing or instantiating the model."""
    path = absolute(str(path), directory=True)
    refs = []
    for item in sorted(path.rglob('*')):
        relative = item.relative_to(path)
        if any(part in ('.git', '.cache', '__pycache__') for part in relative.parts):
            continue
        if item.is_file():
            if path not in item.resolve().parents:
                raise ValueError('Model symlink escapes declared model directory: ' + str(item))
            refs.append(dict(relative=relative.as_posix(), size=item.stat().st_size, sha256=file_sha(item)))
    refs.sort(key=lambda ref: ref['relative'])
    if not refs or not any(ref['relative'] == 'config.json' for ref in refs):
        raise ValueError('Local model must include config.json and payload files')
    if not any(ref['relative'].endswith(('.safetensors', '.bin')) for ref in refs):
        raise ValueError('Local model has no actual checkpoint weight files')
    return dict(path=str(path), files=refs, inventory_sha256=digest(refs), bytes_verified=True)


def manifest_ref(path):
    ref = file_ref(path)
    ids, seen = [], set()
    with Path(ref['path']).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('Every manifest row must be a JSON object')
            if 'id' in row and 'source_id' in row and row['id'] != row['source_id']:
                raise ValueError('Manifest id and source_id disagree')
            source = row.get('id', row.get('source_id'))
            if not isinstance(source, str) or not source.strip() or source in seen:
                raise ValueError('Manifests require unique nonempty id values')
            ids.append(source)
            seen.add(source)
    if not ids:
        raise ValueError('Empty training/validation manifest')
    return dict(ref, source_count=len(ids), source_ids_sha256=digest(ids)), set(ids)


def classes_identity(path):
    classes = read_json(path)
    if (not isinstance(classes, list) or len(classes) < 2 or
            any(not isinstance(item, str) or not item.strip() or item != item.strip() or
                len(item) > 80 or any(ord(char) < 32 for char in item) for item in classes) or
            len(set(classes)) != len(classes)):
        raise ValueError('genre_classes must contain at least two unique stripped text names')
    return dict(classes=classes, classes_sha256=digest(classes))


def _positive_int(value, field, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(field + ' must be a ' + ('nonnegative' if allow_zero else 'positive') + ' integer')
    return value


def _job(spec, nodes, default_repo):
    if not isinstance(spec, dict) or not REQUIRED_JOB.issubset(spec) or set(spec) - REQUIRED_JOB - OPTIONAL_JOB:
        raise ValueError('Job has missing or unknown fields')
    family = spec['family']
    hosts = spec['hosts']
    if family not in FAMILIES:
        raise ValueError('Only MOSS and Qwen Captioner planner jobs are supported')
    if (not isinstance(hosts, list) or len(hosts) != nodes or
            any(not isinstance(host, str) or not HOST.fullmatch(host) for host in hosts) or
            len({host.lower() for host in hosts}) != nodes):
        raise ValueError('Expected canonical unique hosts for every node')
    if not isinstance(spec['job_id'], str) or not NAME.fullmatch(spec['job_id']):
        raise ValueError('job_id must be a nonempty safe rendezvous identifier')
    port = _positive_int(spec['master_port'], 'master_port')
    if port < 1024 or port > 65535:
        raise ValueError('master_port must be 1024..65535')
    master = spec.get('master_addr', hosts[0])
    if master not in hosts:
        raise ValueError('master_addr must name a host in this job')
    repo = absolute(str(spec.get('repo_root', default_repo)), directory=True)
    script_dir = repo / 'scripts_omnidance' / 'exp' / 'ms'
    for name in CORE[:3]:
        absolute(str(script_dir / name))
    code = [file_ref(script_dir / name) for name in CORE if (script_dir / name).is_file()]
    python = absolute(str(spec.get('python', sys.executable)))
    train, train_ids = manifest_ref(spec['train'])
    val, val_ids = manifest_ref(spec['val'])
    if train_ids & val_ids:
        raise ValueError('Training/validation source_id overlap')
    out = Path(spec['out']).expanduser().resolve()
    if out in (repo, repo.parent) or out == out.parent:
        raise ValueError('Training out must be a dedicated run directory')
    model_root = absolute(spec['model'], directory=True)
    if out == model_root or model_root in out.parents:
        raise ValueError('Training output must not be inside the immutable model directory')
    phase = spec.get('phase', 'smoke')
    if phase not in ('smoke', 'full'):
        raise ValueError('phase must be smoke or full')
    maximum = spec.get('max_updates', 2 if phase == 'smoke' else 0)
    _positive_int(maximum, 'max_updates', allow_zero=True)
    if (phase == 'smoke') != (maximum > 0):
        raise ValueError('Smoke requires positive max_updates; full requires zero')
    resume = absolute(spec['resume'], directory=True) if spec.get('resume') else None
    if resume and (not out.is_dir() or resume.parent != out):
        raise ValueError('Resume checkpoint must be inside the original out directory')
    if not resume and out.exists():
        raise ValueError('New training out already exists; use an explicit completed resume checkpoint')
    if resume:
        state = read_json(resume / 'trainer_state.json')
        if state.get('complete') is not True or state.get('world_size') != nodes * 8:
            raise ValueError('Resume requires a complete checkpoint with the same world_size')
        update = state.get('cursor', {}).get('update')
        if type(update) is not int or update < 0 or (maximum and maximum <= update):
            raise ValueError('Resume smoke max_updates must exceed the actual completed checkpoint update')
    files = {key: file_ref(value) for key, value in spec.items() if key in FILE_FLAGS and value is not None}
    values = {key: value for key, value in spec.items() if key in VALUE_FLAGS}
    for key in ('epochs', 'rank', 'alpha', 'max_length', 'audio_last_n', 'genre_head_dim'):
        if key in values:
            _positive_int(values[key], key)
    for key in ('seed', 'warmup_updates', 'save_every', 'eval_every', 'loss_chunk_size'):
        if key in values:
            _positive_int(values[key], key, allow_zero=True)
    for key in ('lora_lr', 'projector_lr', 'grad_clip', 'genre_lr', 'genre_weight'):
        if key in values and (type(values[key]) not in (int, float) or
                              not math.isfinite(values[key]) or values[key] < 0 or
                              (key != 'genre_weight' and values[key] == 0)):
            raise ValueError('Invalid finite positive training value: ' + key)
    if values.get('lr_schedule', 'constant') not in ('constant', 'cosine'):
        raise ValueError('lr_schedule must be constant or cosine')
    if 'warmup_ratio' in values and (type(values['warmup_ratio']) not in (int, float) or
                                    not math.isfinite(values['warmup_ratio']) or
                                    not 0 <= values['warmup_ratio'] < 1):
        raise ValueError('warmup_ratio must be finite and in [0, 1)')
    genre = values.get('genre_weight', 0) > 0
    if genre and not {'genre_classes', 'train_genre_labels', 'val_genre_labels'}.issubset(files):
        raise ValueError('JEV requires actual classes plus train and val genre label files')
    if not genre and any(key in files for key in ('genre_classes', 'train_genre_labels', 'val_genre_labels')):
        raise ValueError('Genre files supplied but genre_weight is zero; choose the protocol explicitly')
    versions = spec.get('expected_versions', {})
    if not isinstance(versions, dict) or any(key not in DEPENDENCIES or not isinstance(value, str) or
                                           not value for key, value in versions.items()):
        raise ValueError('expected_versions must map known package names to exact version strings')
    job = dict(family=family, job_id=spec['job_id'], hosts=hosts, master_addr=master,
               master_port=port, nodes=nodes, gpus_per_node=8, world_size=nodes * 8,
               micro=1, accum=1, global_batch=nodes * 8, phase=phase, max_updates=maximum,
               resume=str(resume) if resume else None, python=str(python), repo_root=str(repo),
               trainer=str(script_dir / 'music_planner_train.py'), code=code,
               resume_marker=file_ref(resume / 'trainer_state.json') if resume else None,
               resume_run_config=file_ref(out / 'run_config.json') if resume else None,
               model=model_inventory(spec['model']), train=train, val=val, out=str(out),
               files=files, values=values, expected_versions=versions, genre_enabled=genre,
               genre_vocabulary=classes_identity(files['genre_classes']['path']) if genre else None,
               checkpoint_world_size_changes_supported=False)
    job['train_argv'] = train_argv(job)
    return job


def train_argv(job):
    result = [job['trainer'], '--family', job['family'], '--model', job['model']['path'],
              '--train', job['train']['path'], '--val', job['val']['path'], '--out', job['out'],
              '--backend', 'deepspeed', '--micro', '1', '--accum', '1', '--batch', str(job['global_batch'])]
    for key in sorted(job['values']):
        result.extend((VALUE_FLAGS[key], str(job['values'][key])))
    for key in sorted(job['files']):
        result.extend((FILE_FLAGS[key], job['files'][key]['path']))
    if job['max_updates']:
        result.extend(('--max-updates', str(job['max_updates'])))
    if job['resume']:
        result.extend(('--resume', job['resume']))
    return result


def build_plan(spec, repo_root=None):
    if not isinstance(spec, dict) or set(spec) != {'layout', 'jobs'}:
        raise ValueError('Spec requires exactly layout and jobs')
    layout = spec['layout']
    if layout not in ('split32', 'single64') or not isinstance(spec['jobs'], list):
        raise ValueError('Unknown cluster layout')
    if len(spec['jobs']) != (2 if layout == 'split32' else 1):
        raise ValueError('split32 requires two jobs; single64 requires one')
    repo_root = repo_root or Path(__file__).resolve().parents[3]
    jobs = [_job(item, 4 if layout == 'split32' else 8, repo_root) for item in spec['jobs']]
    if layout == 'split32' and {job['family'] for job in jobs} != set(FAMILIES):
        raise ValueError('split32 must contain exactly MOSS and Captioner')
    hosts = [host.lower() for job in jobs for host in job['hosts']]
    if len(set(hosts)) != len(hosts):
        raise ValueError('Concurrent jobs cannot share any node')
    if len({job['master_port'] for job in jobs}) != len(jobs):
        raise ValueError('Concurrent jobs must use distinct rendezvous ports')
    for key in ('job_id', 'out'):
        if len({job[key] for job in jobs}) != len(jobs):
            raise ValueError('Concurrent jobs must have distinct ' + key)
    for left in jobs:
        for right in jobs:
            if left is not right and Path(left['out']) in Path(right['out']).parents:
                raise ValueError('Concurrent output directories cannot contain each other')
    if len({(job['train']['sha256'], job['val']['sha256']) for job in jobs}) > 1:
        raise ValueError('Parallel model comparison requires identical train and val manifests')
    plan = dict(schema=SCHEMA, layout=layout, jobs=jobs, dispatch_performed=False,
                actual_training_started=False, gpu_runtime_verified=False,
                minimum_gpu_memory_bytes=80_000_000_000, gpu_name_contains='A800',
                qualification='Plan only; every host needs real preflight and multi-node smoke/resume acceptance')
    # Training resume must survive a NEW launch phase/checkpoint/rdzv plan. Keep
    # training configuration/source/data/topology identity separate from dispatch.
    excluded = {'job_id', 'master_addr', 'master_port', 'phase', 'max_updates', 'resume', 'train_argv',
                'resume_marker', 'resume_run_config'}
    plan['training_contract_sha256'] = digest(dict(layout=layout, jobs=[
        {key: value for key, value in job.items() if key not in excluded} for job in jobs]))
    plan['contract_sha256'] = digest(plan)
    return plan


def load_plan(path):
    plan = read_json(path)
    core = {key: value for key, value in plan.items() if key != 'contract_sha256'}
    if plan.get('schema') != SCHEMA or plan.get('contract_sha256') != digest(core):
        raise ValueError('Cluster plan schema/hash mismatch')
    return plan


def command_plan(plan):
    commands = []
    for job in plan['jobs']:
        for node_rank, host in enumerate(job['hosts']):
            argv = [job['python'], '-m', 'torch.distributed.run', '--nnodes', str(job['nodes']),
                    '--nproc-per-node', '8', '--node-rank', str(node_rank), '--rdzv-backend', 'static',
                    '--rdzv-id', job['job_id'], '--master-addr', job['master_addr'], '--master-port',
                    str(job['master_port']), '--max-restarts', '0'] + job['train_argv']
            commands.append(dict(host=host, family=job['family'], job_id=job['job_id'], node_rank=node_rank,
                                 expected_ranks=list(range(node_rank * 8, (node_rank + 1) * 8)),
                                 cwd=job['repo_root'], argv=argv,
                                 env=dict(PLANNER_CLUSTER_PLAN_SHA256=plan['training_contract_sha256'],
                                          PLANNER_BASE_MODEL_INVENTORY_SHA256=job['model']['inventory_sha256'],
                                          PLANNER_EXPECTED_WORLD_SIZE=str(job['world_size']),
                                          PLANNER_EXPECTED_HOSTNAME=host)))
            environment = ['env'] + [key + '=' + value for key, value in commands[-1]['env'].items()]
            commands[-1]['shell'] = 'cd ' + shlex.quote(job['repo_root']) + ' && ' + shlex.join(environment + argv)
    return commands


def verify_files(job):
    if not job['resume'] and Path(job['out']).exists():
        raise ValueError('New training out now exists; create an explicit resume phase plan')
    refs = job['code'] + [job['train'], job['val']] + list(job['files'].values())
    refs += [ref for ref in (job['resume_marker'], job['resume_run_config']) if ref]
    for ref in refs:
        if file_ref(ref['path']) != {key: ref[key] for key in ('path', 'size', 'sha256')}:
            raise ValueError('Declared local bytes changed: ' + ref['path'])
    actual = model_inventory(job['model']['path'])
    if actual != job['model']:
        raise ValueError('Local model bytes/inventory differ from the shared plan')
    if not Path(job['python']).is_file():
        raise ValueError('Declared Python executable missing')


def runtime_snapshot():
    versions = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    import torch
    from transformers import AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration
    from accelerate import Accelerator, DeepSpeedPlugin
    from peft import LoraConfig
    import deepspeed
    apis = dict(native_captioner=callable(Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained),
                processor=callable(AutoProcessor.from_pretrained), accelerator=callable(Accelerator),
                zero3_plugin='zero3_init_flag' in inspect.signature(DeepSpeedPlugin).parameters,
                zero_init=callable(deepspeed.zero.Init), gathered_parameters=callable(deepspeed.zero.GatheredParameters),
                peft_fused_expert_targets='target_parameters' in inspect.signature(LoraConfig).parameters,
                nccl_backend=torch.distributed.is_nccl_available())
    if not all(apis.values()):
        raise ValueError('Runtime is missing a required native training API')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise ValueError('Real preflight requires exactly eight visible CUDA GPUs')
    gpus = []
    for index in range(8):
        props = torch.cuda.get_device_properties(index)
        if 'A800' not in props.name.upper() or props.total_memory < 80_000_000_000:
            raise ValueError('GPU does not meet the declared A800 80GB hardware target')
        gpus.append(dict(local_rank=index, name=props.name, total_memory=props.total_memory,
                         uuid=str(getattr(props, 'uuid', 'unavailable'))))
    return dict(python_version=sys.version.split()[0], versions=versions,
                torch_cuda=torch.version.cuda, nccl_version=list(torch.cuda.nccl.version())
                if isinstance(torch.cuda.nccl.version(), tuple) else torch.cuda.nccl.version(),
                apis=apis, gpus=gpus, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def native_data_check(job):
    result = subprocess.run([job['python']] + job['train_argv'] + ['--check-data-only'],
                            cwd=job['repo_root'], capture_output=True, text=True, timeout=1800,
                            check=False)
    if result.returncode:
        raise ValueError('Trainer data-only check failed: ' + result.stderr[-4000:])
    try:
        result = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise ValueError('Trainer data-only check returned no JSON receipt') from error
    if result.get('actual_training_started') is not False:
        raise ValueError('Trainer data-only check did not prove no training started')
    return result


def validate_runtime(snapshot, plan):
    if (not isinstance(snapshot, dict) or
            any(snapshot.get('apis', {}).get(name) is not True for name in APIS) or
            any(not isinstance(snapshot.get('versions', {}).get(name), str) or
                not snapshot['versions'][name] for name in DEPENDENCIES) or
            len(snapshot.get('gpus', [])) != 8 or any(
                'A800' not in gpu['name'].upper() or gpu['total_memory'] < plan['minimum_gpu_memory_bytes']
                for gpu in snapshot.get('gpus', []))):
        raise ValueError('Runtime snapshot does not prove required APIs and eight A800 80GB GPUs')


def preflight(plan, receipts, dry_run=False, hostname=None, runtime=None, data_checker=None):
    """hostname/runtime injections are for CPU tests; the CLI offers no host spoofing."""
    hostname = hostname or socket.gethostname()
    identities = {hostname}
    matches = [(job, index, host) for job in plan['jobs'] for index, host in enumerate(job['hosts'])
               if host in identities]
    if len(matches) != 1:
        raise ValueError('Actual hostname must match exactly one expected canonical host')
    job, node_rank, expected_host = matches[0]
    receipt_path = Path(receipts) / (expected_host + '.json')
    if receipt_path.exists():
        raise ValueError('Preflight receipt exists; choose a fresh receipts directory and preserve prior evidence')
    verify_files(job)
    data = (data_checker or native_data_check)(job)
    snapshot = None if dry_run else (runtime or runtime_snapshot)()
    if not dry_run:
        validate_runtime(snapshot, plan)
        for package, version in job['expected_versions'].items():
            if snapshot['versions'].get(package) != version:
                raise ValueError('Runtime package version differs: ' + package)
    probe_dir = Path(job['out']).parent / '.planner-cluster-preflight' / plan['contract_sha256']
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe = probe_dir / (job['job_id'] + '-' + expected_host + '-' + uuid.uuid4().hex + '.json')
    probe_value = dict(contract_sha256=plan['contract_sha256'], host=expected_host,
                       out_parent=str(Path(job['out']).parent), nonce=uuid.uuid4().hex)
    write_json(probe, probe_value)
    if read_json(probe) != probe_value:
        raise ValueError('Output parent read/write verification failed')
    receipt = dict(schema=RECEIPT, contract_sha256=plan['contract_sha256'], job_id=job['job_id'],
                   host=expected_host, actual_hostname=hostname, node_rank=node_rank,
                   expected_ranks=list(range(node_rank * 8, (node_rank + 1) * 8)),
                   world_size=job['world_size'], local_bytes_verified=True, data_check=data,
                   dry_run=bool(dry_run), gpu_runtime_verified=not dry_run, runtime=snapshot,
                   output_parent_rw_verified=True, output_probe=file_ref(probe),
                   output_parent_free_bytes=shutil.disk_usage(probe_dir).free,
                   network_bandwidth_tested=False, training_smoke_passed=False,
                   actual_training_started=False, dispatch_performed=False, passed=not dry_run)
    receipt['receipt_sha256'] = digest(receipt)
    write_json(receipt_path, receipt)
    return receipt


def verify_receipts(plan, receipts):
    software = {}
    checked = []
    # Check the coordinator's current shared view too: a previously valid receipt
    # must not admit source/data/model changes made after the host preflight.
    for job in plan['jobs']:
        verify_files(job)
    for command in command_plan(plan):
        path = Path(receipts) / (command['host'] + '.json')
        receipt = read_json(path)
        core = {key: value for key, value in receipt.items() if key != 'receipt_sha256'}
        if receipt.get('schema') != RECEIPT or receipt.get('receipt_sha256') != digest(core):
            raise ValueError('Preflight receipt schema/hash mismatch: ' + str(path))
        expected = dict(contract_sha256=plan['contract_sha256'], job_id=command['job_id'],
                        host=command['host'], node_rank=command['node_rank'], expected_ranks=command['expected_ranks'],
                        passed=True, dry_run=False, gpu_runtime_verified=True, local_bytes_verified=True,
                        output_parent_rw_verified=True, actual_training_started=False, dispatch_performed=False)
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError('Missing, dry-run, wrong-host, or stale preflight evidence: ' + str(path))
        job = next(item for item in plan['jobs'] if item['job_id'] == command['job_id'])
        if (receipt.get('world_size') != job['world_size'] or receipt.get('actual_hostname') != command['host'] or
                receipt.get('data_check', {}).get('actual_training_started') is not False):
            raise ValueError('Preflight topology/data-check evidence differs: ' + str(path))
        probe = receipt['output_probe']
        if file_ref(probe['path']) != probe:
            raise ValueError('Output parent probe is not shared/unchanged: ' + str(path))
        runtime = receipt['runtime']
        validate_runtime(runtime, plan)
        identity = {key: runtime[key] for key in ('python_version', 'versions', 'torch_cuda', 'nccl_version')}
        previous = software.setdefault(command['job_id'], identity)
        if previous != identity:
            raise ValueError('Runtime package/CUDA/NCCL identity differs between nodes in one job')
        if any(runtime['versions'].get(package) != version for package, version in job['expected_versions'].items()):
            raise ValueError('Runtime exact expected package version mismatch: ' + str(path))
        checked.append(file_ref(path))
    return dict(schema='music-planner-cluster-admission-v1', contract_sha256=plan['contract_sha256'],
                hosts_checked=len(checked), receipt_refs=checked, local_preflight_passed=True,
                network_bandwidth_tested=False, training_smoke_passed=False,
                actual_training_started=False, dispatch_performed=False)


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = result.add_subparsers(dest='action', required=True)
    build = actions.add_parser('plan')
    build.add_argument('--spec', required=True, type=Path)
    build.add_argument('--output', required=True, type=Path)
    check = actions.add_parser('preflight')
    check.add_argument('--plan', required=True, type=Path)
    check.add_argument('--receipts', required=True, type=Path)
    check.add_argument('--dry-run', action='store_true', help='No CUDA/runtime check; cannot produce admission')
    verify = actions.add_parser('verify')
    verify.add_argument('--plan', required=True, type=Path)
    verify.add_argument('--receipts', required=True, type=Path)
    commands = actions.add_parser('commands')
    commands.add_argument('--plan', required=True, type=Path)
    commands.add_argument('--receipts', type=Path, help='If supplied, require all real host preflights')
    commands.add_argument('--format', choices=('json', 'shell'), default='shell')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.action == 'plan':
        if args.output.exists():
            raise ValueError('Plan output already exists; use a new immutable phase plan')
        plan = build_plan(read_json(args.spec))
        write_json(args.output, plan)
        print(json.dumps(dict(plan=str(args.output.resolve()), contract_sha256=plan['contract_sha256'],
                              gpu_runtime_verified=False, dispatch_performed=False), ensure_ascii=False))
        return
    plan = load_plan(args.plan)
    if args.action == 'preflight':
        value = preflight(plan, args.receipts, dry_run=args.dry_run)
    elif args.action == 'verify':
        value = verify_receipts(plan, args.receipts)
    else:
        if args.receipts:
            verify_receipts(plan, args.receipts)
        commands = command_plan(plan)
        if args.format == 'shell':
            print('# Commands only; run the matching line on each named host after preflight. No dispatch performed.')
            for command in commands:
                print('# host=' + command['host'] + ' ranks=' + str(command['expected_ranks']))
                print(command['shell'])
            return
        value = dict(contract_sha256=plan['contract_sha256'], commands=commands, dispatch_performed=False)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
