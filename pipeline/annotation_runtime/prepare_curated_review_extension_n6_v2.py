"""Read-only N6 holder/lease capture for the v2 staged model; no GPU signals."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time

RT = Path(__file__).resolve().parent
ROOT = RT.parent
spec = importlib.util.spec_from_file_location('prepare_extension_resource', RT / 'curated_review_extension_resource_manager_v2.py')
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)
base = manager.base


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--authorize-release', action='store_true')
    a = p.parse_args()
    if '28.216.19.225' not in subprocess.check_output(['hostname', '-I'], text=True).split():
        raise RuntimeError('N6-only fresh read-only config capture')
    uuid = base.uuid_for(2)
    apps = base.apps()
    selected = [v for v in apps if v['uuid'] == uuid]
    if len(selected) != 1:
        raise RuntimeError('GPU2 must contain exactly the expected project holder')
    holder = dict(base.identity(selected[0]['pid']), gpu=2, uuid=uuid)
    base.check_holder(holder, base.identity(holder['pid']), 2, uuid, apps)
    lease = base.lease_snapshot(Path('/tmp/od_reserve_2'))
    stage = RT / 'reviewer8b_stage_N6_extension_v2.json'
    blocked = []
    modelref = dict(path=str(stage), sha256=None, model_identity_sha256=None)
    if stage.exists():
        staged = json.loads(stage.read_text())
        modelref.update(sha256=base.sha(stage), model_identity_sha256=staged['model_identity']['sha256'])
    else:
        blocked.append('8B_N6_stage_not_complete')
    if a.authorize_release and blocked:
        raise RuntimeError('cannot authorize launch before complete N6 8B stage')
    watcher = RT / 'curated_review_extension_watch_v1.py'
    wrapper = RT / 'curated_review_extension_resource_manager_v2.py'
    out = ROOT / 'curated_independent_reviews_extension_v1'
    annotator = ROOT.parent / '32b_replica_stage_receipts/source_reference_v1.json'
    manifest = ROOT / 'curated222939_inputs_v1/manifest.summary.json'
    plan = RT / 'curated222939_admission_ids_v1.jsonl'
    old = json.loads((RT / 'curated_review_watch_launch_v3.json').read_text())
    sources = {k: v for k, v in old['source_sha256'].items()
               if str(RT / 'source_curated_review_v2') + '/' in k}
    for path in (watcher, wrapper, RT / 'resource_manager.py', Path(__file__), manifest, plan, annotator):
        sources[str(path)] = base.sha(path)
    if stage.exists():
        sources[str(stage)] = modelref['sha256']
    cfg = dict(node='N6', gpus=[2], holders=[holder], release_authorized=a.authorize_release,
        launch_ready=not blocked and a.authorize_release, launch_blocked_reasons=blocked,
        phase='curated_review_extension_v1', queue=str(ROOT / 'queues/curated_review_extension_n6g2_v1'),
        out=str(out), cwd=str(RT / 'source_curated_review_v2'),
        producer=dict(path=str(watcher), sha256=base.sha(watcher)), plan=dict(path=str(plan), sha256=base.sha(plan)),
        input_count=222939, targetcount=222939, source_selection=dict(shard=0, nshard=1, limit=0),
        modelref=modelref, source_sha256=sources, reuse_existing_results=True, semantic_approval=False,
        fresh_capture=dict(unix_time=time.time(), GPU_signals_sent=False, lease_modified=False,
                           lease_mtime_ns=lease.st_mtime_ns if lease else None, budget=base.headroom()),
        env=dict(CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
                 HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MS_VISUAL_CPU_SITEPATH='/opt/venv_eval/lib/python3.11/site-packages'),
        argv=['/opt/venv_vllm/bin/python', '-B', '-u', str(watcher),
              '--manifest', str(manifest), '--collaborative-root', str(ROOT / 'curated222939_collaborative_v1'),
              '--repair-root', str(ROOT / 'curated222939_repair_v1'),
              '--extra-visual-root', str(ROOT / 'curated_review_extra_visual_v1'), '--out', str(out),
              '--model', '/opt/models/Qwen3-VL-8B-Instruct-review-extension-v2', '--model-reference', str(stage),
              '--ms-source', str(RT / 'source_curated_review_v2'), '--annotator-model-reference', str(annotator),
              '--global-revision', 'repair-20261025-v1', '--poll-seconds', '60', '--watch'])
    if a.authorize_release:
        manager.validate(cfg)
        manager.check_model(cfg)
    with Path(a.out).open('x') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(config=a.out, sha256=base.sha(a.out), launch_ready=cfg['launch_ready'],
                          blocked_reasons=blocked, GPU_signals_sent=False, lease_modified=False)), flush=True)


if __name__ == '__main__':
    main()
