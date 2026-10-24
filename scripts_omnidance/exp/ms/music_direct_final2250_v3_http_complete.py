"""Independent actual-v3 completion: immutable native AV receipts and real HTTP."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from music_direct_compare_manager_v3 import identity, sha
from music_direct_full6k_final_video_n4_v1 import atomic, small_json

ROOT = Path('/apdcephfs_gy4/share_302533218/cedricnie')
EXP = ROOT / 'diff/logs_omnidance/exp'
MODEL_ID = 'MS/music_direct_full6k_final2250_video_n4_v3'
OUT = EXP / MODEL_ID
QUEUE = EXP / 'MS/music_direct_full6k_final2250_video_n4_queue_v3'
BASE = 'http://28.216.19.69:8775'
IDS = ['wJPptuhclZmKAEWs', 'wJPrpFsThWAcBQIw', 'wJQTclrlOLEYYfTB']
CHECKPOINT_SHA = '127a570f1422ee2f28f78588ace1c886da7ebc68e76ae421672148c4ab0f6cf6'
PUBLISHER = Path(__file__).parent / 'runtime_jobs/publish_ready_eval_review_v6.py'
sys.path.insert(0, str(PUBLISHER.parent))
import publish_ready_eval_review_v6 as publisher


def report_gate():
    failure = small_json(QUEUE / 'partial_failure.json')
    report = small_json(OUT / 'report.json')
    if failure is not None:
        live = small_json(QUEUE / 'live.json')
        publication_only = (failure.get('error') == "BrokenPipeError(32, 'Broken pipe')"
                            and failure.get('upstream_partial_failure') is None
                            and isinstance(live, dict) and live.get('phase') == 'waiting_review_HTTP_retry'
                            and live.get('error') == failure['error']
                            and report is not None
                            and (OUT / 'report.json').stat().st_mtime < (QUEUE / 'partial_failure.json').stat().st_mtime)
        if not publication_only:
            raise ValueError('Actual generation failure is terminal: ' + repr(failure))
    if report is None:
        return None
    if (report.get('complete') is not True or report.get('expected') != 3
            or report.get('actual_final_updates') != 2250
            or report.get('checkpoint_sha256') != CHECKPOINT_SHA
            or report.get('full_media_decode_checked') is not True
            or [case.get('sample_id') for case in report.get('cases', [])] != IDS):
        raise ValueError('Actual report final checkpoint/budget/case binding differs')
    generation_exit = small_json(QUEUE / 'generation_exit.json')
    if generation_exit != dict(exit_code=0):
        raise ValueError('Actual native sampler exit is not exactly zero')
    text = (OUT / 'logs/generation.log').read_text()
    if ('[h3ar] done' not in text or text.count('[direct native saved] ') != 3
            or 'Traceback (most recent call last)' in text):
        raise ValueError('Actual three native-save/done evidence missing')
    preflight = small_json(OUT / 'cpu_preflight.json')
    if sha(OUT / 'cpu_preflight.json') != report.get('preflight_sha256'):
        raise ValueError('Actual generation preflight changed')
    if not preflight or preflight.get('complete') is not True or preflight.get('expected_videos') != 3:
        raise ValueError('Actual native preflight incomplete')
    if sha(publisher.EXP / 'MS/music_direct_eval3_generation_joint_v1.json') != preflight['manifest_sha256']:
        raise ValueError('Actual fixed3 source manifest changed')
    for path, digest in preflight['source_files'].items():
        if sha(path) != digest:
            raise ValueError('Frozen runtime source changed: ' + path)
    job = next(job for job in publisher.JOBS if job['id'] == MODEL_ID)
    sources = []
    for case in report['cases']:
        sid = case['sample_id']
        checked = small_json(OUT / (sid + '.validated.json'))
        if checked != case:
            raise ValueError('Full decoded case differs from complete report: ' + sid)
        names, provenance = publisher.native_available(job, sid, EXP)
        if names != [Path(case['path']).name]:
            raise ValueError('Publisher filename differs from actual decoded file')
        audit = small_json(Path(case['audit_path']))['audit']
        if audit.get('native_audio_vae_unchanged') is not True:
            raise ValueError('Native audio VAE scope changed')
        sources.append(dict(sample_id=sid, **provenance))
    return report, sources


def http_gate(report, sources):
    with urllib.request.urlopen(BASE + '/manifest', timeout=30) as response:
        status, data = response.status, response.read()
        if response.headers.get('Content-Encoding') == 'gzip':
            data = gzip.decompress(data)
    manifest = json.loads(data)
    models = {item['id']: item for item in manifest['models']}
    model = models.get(MODEL_ID)
    if status != 200 or model is None or set(model.get('conds', {}).get('joint', {})) != set(IDS):
        return None
    if not all(name in models for name in ['MS/music_direct_video_compare_n4_v1/mean',
                                           'MS/music_direct_video_compare_n4_v1/token']):
        raise ValueError('Historical mean/token review was not preserved')
    if not any(MODEL_ID in preset.get('models', []) and '2250' in preset['name']
               for preset in manifest['presets']):
        return None
    ranges = []
    for case in report['cases']:
        path = Path(case['path'])
        if model['conds']['joint'][case['sample_id']] != [path.name]:
            raise ValueError('HTTP manifest filename differs from actual final media')
        if path != OUT / 'joint' / path.name:
            raise ValueError('Actual output path differs from new-v3 UI path')
        request = urllib.request.Request(BASE + '/media?' + urllib.parse.urlencode(dict(p=str(path))),
                                         headers={'Range': 'bytes=0-1023'})
        with urllib.request.urlopen(request, timeout=30) as response:
            chunk, code = response.read(), response.status
            content_range = response.headers.get('Content-Range')
        with path.open('rb') as stream:
            expected = stream.read(1024)
        if code != 206 or chunk != expected or content_range != 'bytes 0-1023/' + str(path.stat().st_size):
            raise ValueError('Actual HTTP Range/media byte identity failed')
        ranges.append(dict(sample_id=case['sample_id'], status=code, bytes_verified=1024,
                           content_range=content_range, video_sha256=case['sha256']))
    return dict(complete=True, model_id=MODEL_ID, manifest_status=200, full3_registered=True,
                video_range_checks=ranges, historical_mean_token_preserved=True,
                actual_final_checkpoint_sha256=CHECKPOINT_SHA,
                actual_complete_native_source_provenance=sources,
                no_v1_alias=True, no_review_score_write=True, full_media_decode_checked=True,
                human_quality_pending=True, musical_quality_accepted=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checker_sha', required=True)
    args = parser.parse_args()
    if os.getuid() != 0 or sha(__file__) != args.checker_sha:
        raise RuntimeError('Require exact independent checker source and UID0')
    if (OUT / 'review_http.json').exists() or (QUEUE / 'finished.json').exists():
        raise FileExistsError('Inspect existing completion; never overwrite or repeat')
    plan = dict(identity=identity(os.getpid()), checker_source=str(Path(__file__).resolve()),
                checker_source_sha256=args.checker_sha, publisher_source_sha256=sha(PUBLISHER),
                new_model_id=MODEL_ID, no_GPU=True, no_training=True, no_Adam_coldread=True,
                no_runtime_source_mutation=True, no_review_score_write=True)
    atomic(QUEUE / 'http_checker_plan.json', plan)
    while True:
        try:
            actual = report_gate()
            checked = http_gate(*actual) if actual is not None else None
            if checked is not None:
                checked['checker'] = plan
                checked['actual_report_sha256'] = sha(OUT / 'report.json')
                failure = small_json(QUEUE / 'partial_failure.json')
                if failure is not None:
                    checked['preserved_original_manager_publication_failure'] = dict(
                        path=str(QUEUE / 'partial_failure.json'), sha256=sha(QUEUE / 'partial_failure.json'),
                        record=failure, classified_as='stdout_BrokenPipe_after_complete_native_sampler_and_report',
                        actual_sampler_exit_code=0, actual_new_v3_HTTP_independently_verified=True)
                atomic(OUT / 'review_http.json', checked)
                finished = dict(complete=True, actual_final_video_count=3, review_http=checked,
                                report=str(OUT / 'report.json'), actual_training_updates=2250,
                                actual_model_id=MODEL_ID, independent_HTTP_checker=True,
                                original_manager_HTTP_id_is_obsolete=True, human_quality_pending=True)
                atomic(QUEUE / 'finished.json', finished)
                atomic(QUEUE / 'http_checker_finished.json', finished)
                print(json.dumps(dict(complete=True, phase='technical_complete_actual_v3_final3_HTTP',
                                      model_id=MODEL_ID, actual_videos=3, human_quality_pending=True)), flush=True)
                return
            phase = 'waiting_actual_complete_report3' if actual is None else 'waiting_publisher_actual_v3_full3'
            atomic(QUEUE / 'http_checker_live.json', dict(complete=False, phase=phase, time=time.time(), pid=os.getpid()))
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            atomic(QUEUE / 'http_checker_live.json', dict(complete=False, phase='HTTP_transport_retry',
                                                        error=repr(error), time=time.time(), pid=os.getpid()))
        time.sleep(20)


if __name__ == '__main__':
    main()
