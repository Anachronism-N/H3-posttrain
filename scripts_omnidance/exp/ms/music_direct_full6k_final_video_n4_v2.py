"""Resume final2250 after proven CPU state audit; frozen v1 source and outputs."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
UPSTREAM = ROOT / 'DiffSynth-Studio-upstream-20260930'
sys.path.insert(0, str(UPSTREAM))
os.environ['PYTHONPATH'] = str(UPSTREAM)
import music_direct_full6k_final_video_n4_v1 as original
from music_direct_compare_manager_v3 import identity, snapshot, sha

ORIGINAL_SOURCE = Path(original.__file__)
ORIGINAL_SOURCE_SHA = 'b1069ea048b6f694713b7871851760eabcff06b1e0a92323180cb12a77910458'
ORIGINAL_QUEUE = original.QUEUE
RECEIPT = ORIGINAL_QUEUE / 'final_cpu_checkpoint.json'
EXPECTED_SHA = '127a570f1422ee2f28f78588ace1c886da7ebc68e76ae421672148c4ab0f6cf6'
original.OUT = original.MS / 'music_direct_full6k_final2250_video_n4_v2'
original.QUEUE = original.MS / 'music_direct_full6k_final2250_video_n4_queue_v2'


def receipt_gate(receipt, finished, pointer):
    if not isinstance(receipt, dict):
        raise ValueError('Actual fullstate receipt missing; no automatic Adam coldread')
    for key in ['complete', 'actual_three_unique_epochs', 'full_saved_Adam_RNG_sampler_pointer',
                'strict_loaded_complete_MusicFM']:
        if receipt.get(key) is not True:
            raise ValueError('Incomplete actual receipt: ' + key)
    for key, value in dict(optimizer_states=402, steps=2250, exposures=18000,
                           state_tensors=467, trainable_tensors=402).items():
        if type(receipt.get(key)) is not int or receipt[key] != value:
            raise ValueError('Wrong actual receipt scope: ' + key)
    if receipt.get('training_finished') != finished or receipt.get('train_state') != str(original.STATE):
        raise ValueError('Actual receipt does not match immutable full6k finished/state')
    original.metadata_gate(receipt.get('metadata'), receipt['state_tensors'], receipt['trainable_tensors'])
    if (receipt.get('checkpoint_sha256') != EXPECTED_SHA or pointer.get('checkpoint_sha256') != EXPECTED_SHA
            or receipt.get('local_stage') != str(original.STAGED)
            or receipt.get('local_stage_actual_sha256') != EXPECTED_SHA):
        raise ValueError('Receipt/checkpoint/stage identity mismatch')
    original.terminal_gate(finished)


def reuse_full_cpu_checkpoint(finished, pointer):
    if sha(ORIGINAL_SOURCE) != ORIGINAL_SOURCE_SHA:
        raise ValueError('Frozen original manager source differs')
    receipt = original.small_json(RECEIPT)
    receipt_gate(receipt, finished, pointer)
    final_sha, stage_sha = sha(original.FINAL), sha(original.STAGED)
    if final_sha != EXPECTED_SHA or stage_sha != EXPECTED_SHA:
        raise ValueError('Fresh actual final/local stage SHA differs from strict-load receipt')
    actual = dict(receipt, reused_actual_complete_receipt=True, original_receipt=str(RECEIPT),
                  original_receipt_sha256=sha(RECEIPT), frozen_original_source=str(ORIGINAL_SOURCE),
                  frozen_original_source_sha256=ORIGINAL_SOURCE_SHA,
                  fresh_final_sha256=final_sha, fresh_local_stage_sha256=stage_sha,
                  Adam_coldread_repeated=False)
    original.atomic(original.QUEUE / 'final_cpu_checkpoint.json', actual)
    return EXPECTED_SHA


def acquire_coordinated_gpu3():
    lease = Path('/tmp/music_direct_full6k_final2250_n4_gpu3.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reserve = Path('/tmp/od_reserve_3')
    if reserve.exists() and (reserve.is_symlink() or not reserve.is_file()
                            or reserve.stat().st_uid != 0 or reserve.stat().st_size != 0):
        raise RuntimeError('Unfamiliar GPU3 reserve; do not modify')
    while True:
        resource, models = snapshot(), original.model_processes()
        selected = next(row for row in resource['gpus'] if row[0] == '3')
        if selected[1] != original.UUID:
            raise RuntimeError('Fresh N4 GPU3 UUID mismatch')
        active = [row for row in resource['apps'] if row[1] == original.UUID]
        sufficient = (len(models) < 2 and min(resource['memavailable_bytes'],
                      resource['cgroup_headroom_bytes']) >= 200 * 1024**3)
        if not sufficient:
            original.live('waiting_resource_no_signal', resource=resource, model_processes=models)
            time.sleep(30)
            continue
        if active:
            if len(active) != 1:
                original.live('waiting_unknown_GPU3_no_signal', active=active)
                time.sleep(30)
                continue
            holder = identity(int(active[0][0]))
            known = (all(holder[key] == value for key, value in original.EXPECTED_HOLDER.items())
                     and holder['cgroup'] == resource['cgroup'] and int(active[0][2]) < 1200)
            if known:
                reserve.touch()
                original.atomic(original.QUEUE / 'ready_for_resource.json',
                    dict(complete=True, cpu_preflight=str(original.OUT / 'cpu_preflight.json'),
                         manager_identity=identity(os.getpid()), exact_holder=holder, gpu=3,
                         gpu_uuid=original.UUID, coordinator='goal-live-pipelines-2',
                         manager_sends_no_signal=True, resource=resource))
                original.live('waiting_coordinator_exact_holder_release_no_signal', exact_holder=holder)
            else:
                original.live('waiting_GPU3_identity_changed_no_signal', process=holder)
            time.sleep(10)
            continue
        reserve.touch()
        fresh = snapshot()
        if any(row[1] == original.UUID for row in fresh['apps']):
            original.live('waiting_GPU3_process_appeared_no_signal', resource=fresh)
            time.sleep(10)
            continue
        if len(original.model_processes()) >= 2:
            raise RuntimeError('Another pipeline appeared; do not start a third')
        original.atomic(original.QUEUE / 'fresh_resource.json', fresh)
        return lease


def main():
    if sha(ORIGINAL_SOURCE) != ORIGINAL_SOURCE_SHA:
        raise RuntimeError('Frozen v1 manager unexpectedly changed')
    original.full_cpu_checkpoint = reuse_full_cpu_checkpoint
    original.acquire = acquire_coordinated_gpu3
    original.__file__ = __file__
    original.main()


if __name__ == '__main__':
    main()
