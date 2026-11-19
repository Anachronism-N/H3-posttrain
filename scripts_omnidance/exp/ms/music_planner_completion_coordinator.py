"""Independent clean-v10 text-eval handoff; CPU checks are not GPU success.

Run only as a managed background parent. Never kill SSH/holders, clear claims,
retry a stage, render H3, rank tokenizers, or claim human acceptance.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import stat
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
EXP = ROOT / 'diff/logs_omnidance/exp/MS/music_planner_fixed2s_training_20261024_v1'
MANAGER = EXP / 'runtime/planner_eval_manager_v3.py'
MANAGER_SHA = 'c387edfd246fd25ab3ac73d00bffe2f41d5f1450cba15df9509b53192ac63c9d'
WORK = EXP / 'eval_handoff_clean_v11/coordinator_v1'
LOCK = Path('/tmp/music_planner_completion_coordinator_v1_%d.lock' % os.getuid())
SELECTION_KEYS = ('training_launch', 'full_complete', 'resources_safe',
                  'checkpoint_selection', 'checkpoint_marker', 'fingerprint')
STAGES = ('val_panel', 'fixed3', 'test')
SAFE_KEYS = ('all_holders_safe', 'locks_released', 'leases_restored',
             'managed_parent_wait', 'unknown_jobs_not_signaled')


def regular(path):
    path = Path(path)
    if not path.is_absolute() or path != path.resolve() or any(p.is_symlink() for p in (path, *path.parents)):
        raise RuntimeError('absolute canonical nonsymlink path required: ' + str(path))
    if not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError('regular file required: ' + str(path))
    return path


def file_ref(path):
    path = regular(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def verify_file(ref):
    if file_ref(ref['path']) != {k: ref[k] for k in ('path', 'sha256')}:
        raise RuntimeError('frozen SHA changed: ' + ref['path'])
    if 'size_bytes' in ref and regular(ref['path']).stat().st_size != ref['size_bytes']:
        raise RuntimeError('frozen size changed: ' + ref['path'])


def load_manager(path=MANAGER, expected=MANAGER_SHA):
    if file_ref(path)['sha256'] != expected:
        raise RuntimeError('frozen manager SHA changed before import')
    spec = importlib.util.spec_from_file_location('completion_eval_manager_v2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != Path(path):
        raise RuntimeError('foreign manager import')
    return module


def coordinator_lock(path=LOCK):
    if path != path.resolve() or any(p.is_symlink() for p in path.parents):
        raise RuntimeError('canonical local coordinator lock required')
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        def check():
            opened, current = os.fstat(fd), regular(path).lstat()
            if (not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.getuid() or opened.st_size != 0
                    or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)):
                raise RuntimeError('coordinator lock type/owner/content/inode changed')
        check()
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check()
        return fd
    except BaseException:
        os.close(fd)
        raise


def static_inputs(m, fixed, ev):
    protocol_ref = file_ref(m.HANDOFF / 'protocol.json')
    protocol = m.ref_json(protocol_ref)
    if (protocol.get('schema') != 'music_planner_fixed3_protocol_v1'
            or protocol.get('decoding') != m.DECODE or protocol['data']['path'] != str(m.DATA)
            or protocol.get('fixed3_source_id_and_exact_audio_exclusion_verified') is not True
            or protocol.get('human_quality_accepted') is True):
        raise RuntimeError('wrong sameDATA/decoding/holdout protocol')
    refs = protocol['code'] + protocol['historical_refs'] + list(protocol['data']['splits'].values())
    refs += [protocol['data']['summary'], protocol['training_templates'], protocol['val32']['selection_ref']]
    for ref in refs:
        verify_file(ref)
    code = {r['path']: r['sha256'] for r in protocol['code']}
    for path, sha in m.PINNED.items():
        verify_file(dict(path=path, sha256=sha))
        if path not in (str(m.MANAGER), str(m.HOLDER_SOURCE)) and code.get(path) != sha:
            raise RuntimeError('protocol source identity mismatch')
    splits, summary = ev.validate_data(m.DATA)
    if {k: len(v) for k, v in splits.items()} != dict(train=10713, val=490, test=500):
        raise RuntimeError('wrong clean split counts')
    metadata = [json.loads(line) for line in Path(protocol['historical_refs'][0]['path']).read_text().splitlines() if line.strip()]
    images = json.loads(Path(protocol['historical_refs'][1]['path']).read_text())['records']
    if fixed.holdout_rows(metadata, images, splits, ev) != protocol['fixed3']:
        raise RuntimeError('incomplete/changed fixed3 holdout evidence')
    val = ev.select_rows(splits['val'], 'val', limit=32, seed=42)
    ids = [r['id'] for r in val]
    mapping = [dict(id=source['id'], condition=condition, audio_source_id=donor['id'],
                    source_group_id=source['source_group_id'], audio_source_group_id=donor['source_group_id'],
                    audio_sha256=donor['audio_sha256'])
               for source, condition, donor in ev.make_conditions(val, counterfactual=True)]
    if (protocol['val32']['ids'] != ids or m.ref_json(file_ref(protocol['val32']['selection_ref']['path'])) != ids
            or protocol['val32']['donor_mapping'] != mapping or protocol['val32']['seed'] != 42):
        raise RuntimeError('val32/donor protocol changed')
    if (protocol['test']['limit'] != 500 or protocol['test']['seed'] != 42
            or protocol['test']['final_evaluations_per_family'] != 1
            or protocol['test']['requires_all_three_val_best_checkpoint_identities_frozen_before_first_test'] is not True):
        raise RuntimeError('wrong final once-only test protocol')
    verify_file(summary['training_exclusion_receipt_ref'])
    verify_file(summary['parent_baseline_ref'])
    return protocol_ref, protocol, splits


def no_failure(queue):
    if queue.exists() and (any(queue.glob('*failure*.json')) or (queue / 'child_cleanup_blocked.json').exists()):
        raise RuntimeError('failure retained; no automatic dispatch: ' + str(queue))


def validate_paths(cfg, handoff):
    queue, out = Path(cfg['queue']), Path(cfg['out'])
    for path in (queue, out):
        if (not path.is_absolute() or path != path.resolve() or path == handoff
                or not path.is_relative_to(handoff) or path.exists() or path.is_symlink()):
            raise RuntimeError('fresh canonical handoff child required')
    if queue == out or queue.is_relative_to(out) or out.is_relative_to(queue):
        raise RuntimeError('queue/output overlap')


class Coordinator:
    def __init__(self, m, helpers, protocol_ref, protocol, splits, work=WORK):
        self.m = m
        self.manager, self.fixed, self.ev = helpers
        self.protocol_ref, self.protocol, self.splits, self.work = protocol_ref, protocol, splits, work
        self.frozen, self.freeze_refs, self.active, self.done, self.handles, self.states = {}, {}, {}, {}, [], {}
        self.final_ref = None
        self.blocked = False

    def status(self, key, value):
        if self.states.get(key) != value:
            self.states[key] = value
            print(json.dumps(dict(scope=key, status=value), ensure_ascii=False), flush=True)

    def sources(self):
        return {**self.m.PINNED, str(Path(self.m.__file__).resolve()): MANAGER_SHA,
                self.protocol_ref['path']: self.protocol_ref['sha256']}

    def paths(self, family, stage):
        prefix = family + '_' + stage
        return dict(queue=str(self.work / (prefix + '_queue')), out=str(self.work / (prefix + '_out')))

    def freeze_family(self, node):
        m = self.m
        host, family = m.SCOPE[node]
        queue = m.EXP / 'queues' / (family + '_fixed2s_clean_v11')
        no_failure(queue)
        if not all((queue / name).is_file() for name in ('full.complete.json', 'resources_safe.json')):
            self.status(family, 'waiting_actual_training_complete_and_resources_safe')
            return None
        row = dict(training_launch=file_ref(m.RT / (family + '_launch_clean_v11.json')),
                   full_complete=file_ref(queue / 'full.complete.json'), resources_safe=file_ref(queue / 'resources_safe.json'),
                   checkpoint_selection=file_ref(m.EXP / 'runs' / (family + '_fixed2s_clean_v11') / 'train/best.json'))
        best = m.ref_json(row['checkpoint_selection'])
        row['checkpoint_marker'] = file_ref(Path(best['checkpoint']) / 'trainer_state.json')
        row['fingerprint'] = m.ref_json(row['checkpoint_marker'])['fingerprint']
        cfg = dict(schema='music-planner-eval-manager-v1', node=node, host=host, family=family, gpus=[0],
                   release_authorized=True, job='val_panel', protocol=self.protocol_ref, source_sha256=self.sources(),
                   **row, **self.paths(family, 'val_panel'))
        launch, result = m.check_completed(cfg, self.manager, local=False)
        for split in ('train', 'val'):
            if launch['dataset'][split]['sha256'] != self.protocol['data']['splits'][split]['sha256']:
                raise RuntimeError('actual training differs from sameDATA')
        path, identity, audit = m.selected_checkpoint(cfg, result, self.protocol, self.fixed, self.ev, full_hash=False)
        no_failure(queue)
        for key in SELECTION_KEYS[:-1]:
            m.ref_json(row[key])
        frozen = dict(selection={k: row[k] for k in SELECTION_KEYS}, checkpoint=str(path), identity=identity,
                      audit=audit, protocol=self.protocol_ref, frozen_before_any_test=True)
        freeze_path = self.work / (family + '.selection.json')
        m.new_json(freeze_path, frozen)
        self.freeze_refs[family] = file_ref(freeze_path)
        self.frozen[family] = frozen
        self.status(family, 'actual_training_and_selection_frozen')
        return frozen

    def build_final_selection(self):
        m = self.m
        families = {v[1] for v in m.SCOPE.values()}
        if set(self.frozen) != families:
            raise RuntimeError('all three actual selections must be frozen before final selection')
        for family, frozen in self.frozen.items():
            verify_file(self.freeze_refs[family])
            if frozen['protocol'] != self.protocol_ref:
                raise RuntimeError('frozen protocol differs')
            for key in SELECTION_KEYS[:-1]:
                m.ref_json(frozen['selection'][key])
        final = dict(schema='music-planner-final-selection-v1', test_used=False,
                     selection_frozen_before_test=True, decoding=m.DECODE, data=self.protocol['data'],
                     protocol=self.protocol_ref, families={f: self.frozen[f]['selection'] for f in sorted(families)})
        # Build from prior independent freezes, not a newly calculated checkpoint identity.
        path = m.HANDOFF / 'final_selection.json'
        m.new_json(path, final)
        self.final_ref = file_ref(path)
        for node in m.SCOPE:
            m.check_final_selection(self.config(node, 'test'), self.manager, self.fixed, self.ev, self.protocol)
        self.status('shared', 'all_three_final_selection_frozen_and_rechecked')
        return final

    def config(self, node, stage):
        host, family = self.m.SCOPE[node]
        cfg = dict(schema='music-planner-eval-manager-v1', node=node, host=host, family=family,
                   gpus=[0], release_authorized=True, job=stage, protocol=self.protocol_ref,
                   source_sha256=self.sources(), **self.frozen[family]['selection'], **self.paths(family, stage))
        if stage == 'test':
            if self.final_ref is None or len(self.frozen) != 3:
                raise RuntimeError('no three-family final selection')
            cfg['final_selection'] = self.final_ref
        return cfg

    def check_config(self, cfg, ref):
        m = self.m
        validate_paths(cfg, m.HANDOFF)
        m.check_sources(cfg, ref['path'], ref['sha256'], MANAGER_SHA)
        no_failure(Path(cfg['full_complete']['path']).parent)
        _, result = m.check_completed(cfg, self.manager, local=False)
        path, identity, audit = m.selected_checkpoint(cfg, result, self.protocol, self.fixed, self.ev, full_hash=False)
        frozen = self.frozen[cfg['family']]
        verify_file(self.freeze_refs[cfg['family']])
        if str(path) != frozen['checkpoint'] or identity != frozen['identity'] or audit != frozen['audit']:
            raise RuntimeError('selection identity changed after independent freeze')
        if cfg['job'] == 'test':
            m.check_final_selection(cfg, self.manager, self.fixed, self.ev, self.protocol)
            if (m.HANDOFF / ('test_once_' + cfg['family'] + '.json')).exists():
                raise RuntimeError('family test already claimed; retained without retry')

    def ssh(self, action, cfg, ref):
        # A remote CPU guard checks bytes BEFORE invoking the frozen manager file.
        guard = ("import hashlib,pathlib,runpy,sys; p=pathlib.Path(sys.argv[1]); "
                 "hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2] or sys.exit('manager SHA changed'); "
                 "sys.argv=[str(p)]+sys.argv[3:]; runpy.run_path(str(p),run_name='__main__')")
        remote = ['/opt/venv_music_planner_v1/bin/python', '-u', '-c', guard,
                  str(Path(self.m.__file__).resolve()), MANAGER_SHA, action,
                  '--config', ref['path'], '--config-sha', ref['sha256'], '--manager-sha', MANAGER_SHA]
        return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', cfg['host'], shlex.join(remote)]

    def dispatch(self, node, stage):
        host, family = self.m.SCOPE[node]
        if self.blocked or family in self.active:
            raise RuntimeError('blocked or own-family stage already active')
        if stage != 'val_panel' and (family, STAGES[STAGES.index(stage) - 1]) not in self.done:
            raise RuntimeError('own-family stages must be serial and safely complete')
        claim = self.work / (family + '_' + stage + '.dispatch.json')
        if claim.exists() or claim.is_symlink():
            raise RuntimeError('existing dispatch claim needs diagnosis; never duplicate')
        cfg = self.config(node, stage)
        capture_path = self.work / (family + '_' + stage + '.capture_config.json')
        self.m.new_json(capture_path, cfg)
        capture_ref = file_ref(capture_path)
        self.check_config(cfg, capture_ref)
        captured = subprocess.run(self.ssh('capture', cfg, capture_ref), capture_output=True, text=True, check=True)
        evidence = json.loads(captured.stdout)
        if (evidence.get('signals_sent') is not False or evidence.get('other_seven_GPUs_untouched') is not True
                or evidence.get('holder', {}).get('gpu') != 0):
            raise RuntimeError('capture did not prove exact GPU0/no signals/other seven untouched')
        capture_receipt = self.work / (family + '_' + stage + '.capture.json')
        self.m.new_json(capture_receipt, evidence)
        evidence_ref = file_ref(capture_receipt)
        cfg['holder'] = evidence['holder']
        run_path = self.work / (family + '_' + stage + '.run_config.json')
        self.m.new_json(run_path, cfg)
        run_ref = file_ref(run_path)
        self.check_config(cfg, run_ref)
        stdout = (self.work / (family + '_' + stage + '.ssh.stdout.log')).open('xb')
        try:
            stderr = (self.work / (family + '_' + stage + '.ssh.stderr.log')).open('xb')
            try:
                self.m.new_json(claim, dict(config=run_ref, capture=evidence_ref, capture_config=capture_ref,
                    automatic_retry=False, existing_claim_blocks_dispatch=True, claimed_unix=time.time()))
                child = subprocess.Popen(self.ssh('run', cfg, run_ref), stdout=stdout, stderr=stderr,
                                         stdin=subprocess.DEVNULL, close_fds=True)
            finally:
                stderr.close()
        finally:
            stdout.close()
        self.handles.append(child)
        self.active[family] = dict(cfg=cfg, config=run_ref, capture=evidence_ref, claim=file_ref(claim), child=child)
        self.status(family, stage + '_dispatched_owned_SSH_parent')

    def stage_receipt(self, item):
        cfg, m = item['cfg'], self.m
        queue, out = Path(cfg['queue']), Path(cfg['out'])
        no_failure(queue)
        required = [queue / name for name in ('eval.complete.json', 'resources_safe.json',
                    cfg['job'] + '.exit.json', 'integrity_receipt.json', 'admission.json')]
        if not all(p.is_file() for p in required):
            if item['child'].poll() is not None:
                raise RuntimeError('owned SSH exited without complete receipts; dispatch stopped')
            return None
        refs = [file_ref(p) for p in required]
        complete, safe, exit_, integrity, admission = [m.ref_json(r) for r in refs]
        if (any(safe.get(k) is not True for k in SAFE_KEYS) or safe.get('admitted_gpu_count') != 1
                or type(exit_.get('exit_code')) is not int or exit_['exit_code'] != 0
                or complete.get('technical_collection_complete') is not True
                or complete.get('complete_native_GPU_outputs_verified') is not True
                or complete.get('quality_accepted') is not False or complete.get('human_quality_accepted') is not False):
            raise RuntimeError('worker exit/technical completion/resource-safe gate failed')
        if item['child'].poll() not in (None, 0):
            raise RuntimeError('owned SSH failed; no further dispatch')
        for ref in (item['config'], item['capture'], item['claim']):
            verify_file(ref)
        m.check_sources(cfg, item['config']['path'], item['config']['sha256'], MANAGER_SHA)
        frozen = self.frozen[cfg['family']]
        if (integrity.get('config_sha256') != item['config']['sha256'] or integrity.get('manager_sha256') != MANAGER_SHA
                or integrity.get('checkpoint_identity') != frozen['identity'] or integrity.get('protocol') != self.protocol_ref
                or any(integrity.get(k) != cfg[k] for k in ('training_launch', 'full_complete', 'resources_safe'))
                or admission.get('gpus') != [0] or admission.get('other_seven_GPUs_untouched') is not True
                or admission.get('config_sha256') != item['config']['sha256']
                or admission.get('source_sha256') != cfg['source_sha256']
                or admission.get('holder') != {k: v for k, v in cfg['holder'].items() if k not in ('gpu', 'uuid')}):
            raise RuntimeError('actual manager integrity/admission identity differs')
        verify_file(complete['manifest'])
        verify_file(complete['raw'])
        if complete['manifest']['path'] != str(out / 'manifest.json') or complete['raw']['path'] != str(out / 'raw.jsonl'):
            raise RuntimeError('technical receipts point to another output')
        manifest = m.ref_json(complete['manifest'])
        raw = [json.loads(line) for line in (out / 'raw.jsonl').read_text().splitlines() if line.strip()]
        self.check_outputs(cfg, manifest, raw)
        if cfg['job'] == 'test':
            m.check_final_selection(cfg, self.manager, self.fixed, self.ev, self.protocol)
            claim_ref = file_ref(queue / 'test_claim.json')
            actual_claim = m.ref_json(claim_ref)
            expected_path = m.HANDOFF / ('test_once_' + cfg['family'] + '.json')
            if actual_claim.get('path') != str(expected_path) or actual_claim.get('final_selection') != self.final_ref:
                raise RuntimeError('test did not bind the real once-only family claim')
            once_ref = file_ref(expected_path)
            once = m.ref_json(once_ref)
            if (once.get('family') != cfg['family'] or once.get('final_selection') != self.final_ref
                    or once.get('checkpoint_selection') != cfg['checkpoint_selection']
                    or once.get('checkpoint_marker') != cfg['checkpoint_marker'] or once.get('out') != cfg['out']
                    or once.get('automatic_retry') is not False or once.get('test_used_for_selection') is not False):
                raise RuntimeError('actual once-only test selection differs')
            refs.extend([claim_ref, once_ref])
        no_failure(queue)
        return dict(config=item['config'], capture=item['capture'], dispatch=item['claim'],
                    actual_receipts=refs, manifest=complete['manifest'], raw=complete['raw'],
                    technical_collection_complete=True, resources_safe=True, h3_rendered=False,
                    human_quality_accepted=False, automatic_retry=False)

    def check_outputs(self, cfg, manifest, raw):
        family, stage, m = cfg['family'], cfg['job'], self.m
        frozen = self.frozen[family]
        fixed3 = stage == 'fixed3'
        producer = m.FROZEN / ('music_planner_fixed3_eval.py' if fixed3 else 'music_planner_eval.py')
        if (manifest.get('complete') is not True or manifest.get('family') != family
                or manifest.get('checkpoint_identity') != frozen['identity']
                or manifest.get('human_quality_accepted') is True or manifest.get('failure') or manifest.get('load_failure')):
            raise RuntimeError('manifest not actual complete matching checkpoint')
        verify_file(manifest['code'])
        if manifest['code']['path'] != str(producer):
            raise RuntimeError('wrong worker code')
        if fixed3:
            rows = self.protocol['fixed3']
            expected = [(row, 'original', row) for row in rows]
            if (manifest.get('schema') != 'music_planner_fixed3_native_eval_v1'
                    or manifest.get('h3_handoff_approved') is not True
                    or manifest.get('pretrained_GPU_generation_executed') is not True
                    or manifest.get('protocol') != self.ev.file_ref(self.protocol_ref['path'])
                    or manifest.get('checkpoint_selection', {}).get('receipt') != self.ev.file_ref(cfg['checkpoint_selection']['path'])
                    or manifest.get('device') != 'cuda:0' or manifest.get('decoding') != m.DECODE):
                raise RuntimeError('fixed3 not strict effective H3 caption handoff')
            verify_file(manifest['h3_records'])
            h3rows = m.ref_json(manifest['h3_records'])['records']
            version = 'music-planner-fixed3-native-v1:' + frozen['identity']['fingerprint'] + ':' + frozen['identity']['trainer_state']['sha256']
            if any(r.get('caption_version') != version or not isinstance(r.get('motion_caption'), str) for r in h3rows):
                raise RuntimeError('fixed3 strict caption schema/version mismatch')
            if [(r['id'], r['image_path'], r['motion_caption']) for r in h3rows] != [
                    (row['id'], row['image_path'], record['raw_text']) for row, record in zip(rows, raw)] or len(h3rows) != 3:
                raise RuntimeError('fixed3 caption schema/raw passthrough mismatch')
        else:
            rows = self.ev.select_rows(self.splits['test' if stage == 'test' else 'val'],
                    'test' if stage == 'test' else 'val', limit=500 if stage == 'test' else 32, seed=42)
            expected = self.ev.make_conditions(rows, counterfactual=stage == 'val_panel')
            if (manifest.get('schema') != 'music_planner_ar_eval_v1' or manifest.get('device') != 'cuda'
                    or manifest.get('mode') != ('test' if stage == 'test' else 'val')
                    or manifest.get('checkpoint') != frozen['checkpoint']
                    or manifest.get('test_used_for_checkpoint_selection') is not False
                    or manifest.get('data_summary') != self.protocol['data']['summary']
                    or manifest.get('split') != self.protocol['data']['splits']['test' if stage == 'test' else 'val']
                    or manifest.get('adapter_code') != self.ev.file_ref(m.FROZEN / 'music_planner_adapters.py')
                    or manifest.get('selection_ref') != (self.protocol['val32']['selection_ref'] if stage == 'val_panel' else None)
                    or any(manifest.get('decoding', {}).get(k) != m.DECODE[k] for k in
                           ('seed', 'do_sample', 'max_new_tokens', 'added_penalties', 'GT_length_truncation', 'raw_output_cleaning'))):
                raise RuntimeError('val/test budget/source/decoding mismatch')
        if len(raw) != (3 if fixed3 else 500 if stage == 'test' else 96):
            raise RuntimeError('incomplete actual raw collection')
        if manifest.get('source_ids') != [r['id'] for r in rows]:
            raise RuntimeError('manifest source membership mismatch')
        for record, (source, condition, donor) in zip(raw, expected):
            if (record.get('id') != source['id'] or record.get('condition', 'original') != condition
                    or record.get('failure') is not None):
                raise RuntimeError('raw source/failure identity mismatch')
            if fixed3:
                if record.get('audio_ref') != source['audio_ref'] or record.get('image_ref') != source['image_ref']:
                    raise RuntimeError('fixed3 audio/image binding mismatch')
                verify_file(record['image_ref'])
            elif (record.get('audio_sha256') != donor['audio_sha256']
                    or record.get('audio_window_seconds') != donor['audio_window_seconds']):
                raise RuntimeError('raw audio/window identity mismatch')
            verify_file(record['audio_ref'])
            if record['audio_ref']['path'] != donor['audio_path'] or record['audio_ref']['sha256'] != donor['audio_sha256']:
                raise RuntimeError('raw actual audio differs')
            if not fixed3 and (record.get('audio_source_id') != donor['id']
                    or record.get('source_group_id') != source['source_group_id']
                    or record.get('audio_source_group_id') != donor['source_group_id']
                    or record.get('audio_transform') != ('zero_waveform_preserve_length' if condition == 'silence' else 'none')):
                raise RuntimeError('raw counterfactual identity mismatch')
            native, tokens, eos, text = (record.get(k) for k in ('native_output', 'new_token_ids', 'eos_token_ids', 'raw_text'))
            if (not isinstance(native, dict) or not isinstance(text, str) or native.get('raw_text') != text
                    or not isinstance(tokens, list) or not isinstance(eos, list) or not 0 < len(tokens) <= 2048
                    or any(type(t) is not int or t < 0 for t in tokens + eos)
                    or native.get('new_token_ids') != tokens or native.get('eos_token_ids') != eos
                    or record.get('generation_budget') != 2048 or record.get('generated_tokens') != len(tokens)):
                raise RuntimeError('raw native interface/budget mismatch')
            stop = self.ev.classify_stop(tokens, eos, 2048, native.get('native_stop', False))
            if record.get('stop_reason') != stop:
                raise RuntimeError('raw stop evidence mismatch')
            if fixed3 and (stop not in ('eos', 'native_stop') or not text.strip()):
                raise RuntimeError('fixed3 strict stop/empty schema failure; retained, no retry')
            # val/test cap, unknown stop and empty text are quality diagnostics, not retries.

    def tick(self):
        if self.blocked:
            return
        for child in self.handles:
            if child.poll() not in (None, 0):
                raise RuntimeError('owned SSH failed; dispatch stopped without retry')
        for family, item in list(self.active.items()):
            receipt = self.stage_receipt(item)
            if receipt is not None:
                stage = item['cfg']['job']
                path = self.work / (family + '_' + stage + '.done.json')
                self.m.new_json(path, receipt)
                self.done[family, stage] = file_ref(path)
                del self.active[family]
                self.status(family, stage + '_technical_complete_resources_safe')
        for node, (_, family) in self.m.SCOPE.items():
            if family not in self.frozen:
                self.freeze_family(node)
        if len(self.frozen) == 3 and self.final_ref is None:
            self.build_final_selection()
        for ref in self.done.values():
            receipt = self.m.ref_json(ref)
            for binding in receipt['actual_receipts'] + [receipt[k] for k in ('config', 'capture', 'dispatch', 'manifest', 'raw')]:
                verify_file(binding)
        if self.final_ref is not None:
            verify_file(self.final_ref)
        for node, (_, family) in self.m.SCOPE.items():
            if family not in self.frozen or family in self.active:
                continue
            stage = next((s for s in STAGES if (family, s) not in self.done), None)
            if stage is not None and (stage != 'test' or self.final_ref is not None):
                self.dispatch(node, stage)
        if len(self.done) == 9 and not (self.work / 'text_evals_complete.json').exists():
            self.m.new_json(self.work / 'text_evals_complete.json', dict(text_evals_complete=True,
                stages={f + '/' + s: ref for (f, s), ref in sorted(self.done.items())}, final_selection=self.final_ref,
                h3_rendered=False, human_quality_accepted=False, experiment_goal_complete=False))
            self.status('shared', 'text_evals_complete_only_H3_and_human_review_pending')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--check-config', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args(argv)
    if not 1 <= args.poll_seconds <= 3600:
        parser.error('poll seconds must be between 1 and 3600')
    m = load_manager()
    helpers = m.load_helpers()
    protocol_ref, protocol, splits = static_inputs(m, helpers[1], helpers[2])
    c = Coordinator(m, helpers, protocol_ref, protocol, splits)
    if not args.run:
        for node, (_, family) in m.SCOPE.items():
            no_failure(m.EXP / 'queues' / (family + '_fixed2s_clean_v11'))
            cfg = dict(c.paths(family, 'val_panel'))
            validate_paths(cfg, m.HANDOFF)
        print(json.dumps(dict(static_config_valid=True, mode='CPU_read_only', SSH_executed=False,
            dispatch_written=False, real_GPU_success=False, h3_rendered=False, human_quality_accepted=False)), flush=True)
        return 0
    lockfd = coordinator_lock()
    def stop_dispatch(signum, frame):
        c.blocked = True
        c.status('shared', 'signal_dispatch_stopped_owned_parents_retained')
    signal.signal(signal.SIGTERM, stop_dispatch)
    signal.signal(signal.SIGINT, stop_dispatch)
    try:
        if WORK != WORK.resolve() or not WORK.is_relative_to(m.HANDOFF) or WORK.is_symlink():
            raise RuntimeError('wrong coordinator work path')
        WORK.mkdir(exist_ok=True)
        if any(WORK.iterdir()) or (m.HANDOFF / 'final_selection.json').exists():
            raise RuntimeError('existing coordinator artifacts/claims block restart; diagnose, never duplicate')
        for _, family in m.SCOPE.values():
            if (m.HANDOFF / ('test_once_' + family + '.json')).exists():
                raise RuntimeError('existing once-only test claim; no retry')
        while True:
            if not c.blocked:
                try:
                    c.tick()
                except BaseException as error:
                    c.blocked = True
                    c.status('shared', 'dispatch_stopped_failure_retained_no_retry')
                    m.new_json(WORK / 'failure.json', dict(error=repr(error), automatic_retry=False,
                        owned_SSH_parents_retained=True, unknown_jobs_not_signaled=True))
            # Resource managers wait on restored holders. Keep this managed parent and
            # every Popen handle alive; never wait synchronously for a run to finish.
            if (c.blocked or len(c.done) == 9) and all(p.poll() is not None for p in c.handles):
                return 1 if c.blocked else 0
            try:
                time.sleep(args.poll_seconds)
            except KeyboardInterrupt:
                c.blocked = True
                c.status('shared', 'interrupted_dispatch_stopped_owned_parents_retained')
    finally:
        os.close(lockfd)


if __name__ == '__main__':
    raise SystemExit(main())
