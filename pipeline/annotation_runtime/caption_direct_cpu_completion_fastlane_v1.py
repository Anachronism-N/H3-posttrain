"""CPU derivation of immutable admitted annotation facts, NOT a new pixel review."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
BAR = RT.parent
sys.path.insert(0, str(RT))
import caption_direct_facts_v1 as direct

RECIPE = 'caption_direct_cpu_completion_fastlane_v1'
FLAGS = dict(direct.FLAGS, semantic_validation_completed=False, pixel_quality_review_completed=False)
REQUIRED = {'request.json', 'binding.json', 'response.raw.txt', 'generation.json', 'response.json', 'final.json'}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def stat(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, device=s.st_dev, inode=s.st_ino)


class Evidence:
    def __init__(self):
        self.cache = {}
        self.read_count = 0

    def raw(self, path, expected=None):
        path = Path(path).resolve()
        if path not in self.cache:
            need(path.suffix in ('.json', '.jsonl', '.txt', '.py'), 'Only annotation/code artifacts may be read')
            before = stat(path)
            data = path.read_bytes()
            after = stat(path)
            need(before == after, 'Annotation changed during first read: ' + str(path))
            self.cache[path] = (data, hashlib.sha256(data).hexdigest(), after)
            self.read_count += 1
        data, actual, _ = self.cache[path]
        need(expected is None or expected == actual, 'Annotation SHA mismatch: ' + str(path))
        return data

    def load(self, path, expected=None):
        bar, _ = direct.native()
        return bar.core.strict_json(self.raw(path, expected))

    def checked(self, ref):
        return self.load(ref['path'], ref['sha256'])

    def ref(self, path):
        path = Path(path).resolve()
        self.raw(path)
        return dict(path=str(path), sha256=self.cache[path][1])

    def boundary(self):
        rows = []
        for path, (_, sha, old) in self.cache.items():
            need(stat(path) == old, 'Annotation changed after admission: ' + str(path))
            rows.append(dict(path=str(path), sha256=sha, stat=old))
        return rows


def source_boundary(identity, record, binding, crossmount):
    bar, _ = direct.native()
    content = {k: dict(path=v['path'], sha256=v['sha256']) if v is not None else None
               for k, v in identity['files'].items()}
    need(bar.batch.digest(dict(row=record, files=content)) == identity['sha256'], 'Source identity digest differs')
    rows = []
    for field, old in identity['files'].items():
        if old is None:
            need(record[field] is None, 'Null source field differs')
            continue
        need(old['path'] == record[field], 'Media/source path identity differs')
        if field in ('video_path', 'audio_path'):
            need(old['sha256'] == binding['source_binding'][field.replace('_path', '_sha256')], 'Media binding SHA differs')
        current = stat(old['path'])
        same = all(current[k] == old['stat'][k] for k in ('size', 'mtime_ns', 'ctime_ns', 'inode'))
        device_changed = current['device'] != old['stat']['device']
        need(same and (not device_changed or crossmount), 'Source stat changed beyond explicit crossmount device: ' + old['path'])
        rows.append(dict(path=old['path'], original_stat=copy.deepcopy(old['stat']), current_stat=current,
                         crossmount_device_only=device_changed, content_read=False))
    return rows


def generation_check(evidence, attempt, request, model):
    bar, _ = direct.native()
    g = evidence.load(attempt / 'generation.json')
    need(g.get('request_sha256') == evidence.ref(attempt / 'request.json')['sha256']
         and g.get('raw_response_sha256') == evidence.ref(attempt / 'response.raw.txt')['sha256']
         and g.get('max_tokens') == request['sampling']['max_tokens']
         and g.get('model_call_performed') is True
         and ('json_schema' not in request or g.get('structured_output_schema_sha256') == bar.core.digest(request['json_schema'])),
         'Generation does not bind request/raw/schema/token budget')
    need(g.get('finish_reason') == 'stop' and type(g.get('completion_tokens')) is int
         and 0 < g['completion_tokens'] < g['max_tokens'], 'Generation not normally stopped')
    need(g.get('model_identity_sha256') == model['sha256']
         and request['model'] == model['root'], 'Original generation/model identity differs')
    return g


def replay(evidence, original, prepared, root):
    bar, _ = direct.native()
    windows = [m['window'] for m in prepared['windows']]
    observations = original['window_observations']
    need(prepared['original_windows'] == original['original_windows']
         and len(windows) == len(observations) == len(original['original_windows']) and bool(windows),
         'Window population or original identity differs')
    union = evidence.checked(original['raw_rgb_manifest'])
    need(union['binding'] == original['binding'] == prepared['binding']
         and prepared['union_manifest_sha256'] == original['raw_rgb_manifest']['sha256'], 'Union/prepared/source binding differs')
    model = evidence.checked(original['model_identity_ref'])
    model_content = {name: dict(path=row['path'], sha256=row['sha256']) for name, row in model['files'].items()}
    need(bar.batch.digest(dict(root=model['root'], files=model_content)) == model['sha256'], 'Model metadata identity digest differs')
    forbidden = {p: set(v) for p, v in prepared['forbidden'].items()}
    replayed, provenance = [], []
    for manifest, obs in zip(prepared['windows'], observations):
        wid = manifest['window']['window_id']
        need(re.fullmatch(r'[A-Za-z0-9_-]+', wid) is not None, 'Unsafe window ID')
        need(obs['window'] == manifest['window'] and obs['status'] == 'candidate', 'Window is not a complete admitted candidate')
        local = root / 'windows' / wid / 'manifest.json'
        need(evidence.load(local) == dict(manifest, binding=original['binding'],
             union_manifest_sha256=prepared['union_manifest_sha256'], raw2d_mask=prepared['raw2d_mask'], **bar.FLAGS),
             'Window manifest differs from immutable prepared annotation')
        need(all(f in union['frames'] for f in manifest['frames']), 'Frame refs differ from original union')
        attempt = Path(obs['attempt_path']).resolve()
        need(attempt.parent == root / 'windows' / wid and re.fullmatch(r'attempt-[0-9]+', attempt.name), 'Wrong source/window attempt')
        receipt = evidence.load(attempt / 'receipt.json')
        need(receipt['status'] == 'candidate' and REQUIRED <= set(receipt['files_sha256']), 'Incomplete window receipt')
        for name, sha in receipt['files_sha256'].items():
            need(Path(name).name == name and name in REQUIRED, 'Unexpected receipt artifact')
            evidence.raw(attempt / name, sha)
        expected = dict(original['binding'], window=manifest['window'], frames_sha256=bar.batch.digest(manifest['frames']))
        need(evidence.load(attempt / 'binding.json') == receipt['binding'] == expected, 'Window/frame/source generation binding differs')
        request = evidence.load(attempt / 'request.json')
        generation_check(evidence, attempt, request, model)
        filtered = bar.filter_supported_observation(evidence.load(attempt / 'response.raw.txt'), manifest, forbidden)
        response = evidence.load(attempt / 'response.json')
        final = evidence.load(attempt / 'final.json')
        need(filtered == response == obs['json'] == final['json']
             and final['status'] == 'candidate' and filtered['abstain'] is False, 'Native support-filter admission replay differs')
        replayed.append(dict(window=manifest['window'], status='candidate', json=filtered))
        provenance.append(dict(window=copy.deepcopy(manifest['window']), manifest_ref=evidence.ref(local),
            receipt_ref=evidence.ref(attempt / 'receipt.json'),
            artifact_refs={name: evidence.ref(attempt / name) for name in sorted(REQUIRED)},
            original_frame_refs=copy.deepcopy(manifest['frames']),
            original_hand_roi_evidence=copy.deepcopy(manifest.get('hand_roi_evidence', {}))))
    facts = bar.facts_from_observations(windows, replayed)
    need(facts == original['facts'] and all(any(f['window_id'] == w['window_id'] for f in facts) for w in windows),
         'Admitted facts text/IDs/window/parts were changed or empty')
    return windows, facts, provenance


def derive(evidence, notice, consumer, version, crossmount=False):
    bar, pipeline = direct.native()
    sid, split = notice['id'], notice['split']
    need(consumer.parent.parent == BAR / ('curated222939_collaborative_v%d' % version), 'Consumer/source version differs')
    need(re.fullmatch(r'[A-Za-z0-9_-]+', sid) and split in ('train', 'val', 'test'), 'Unsafe source ID/split')
    joint_path = Path(notice['joint_candidate_ref']['path']).resolve()
    need(joint_path.parent == consumer / 'sources' / split / sid / 'joint-results'
         and joint_path.stem == notice['joint_candidate_ref']['sha256'], 'Wrong source/version joint reference')
    joint = evidence.checked(notice['joint_candidate_ref'])
    need(joint['id'] == sid and joint['split'] == split and joint['batch_id'] == notice['batch_id']
         and joint['joint_complete'] is False and notice['joint_complete'] is False, 'Original joint/index identity or completeness differs')
    style = joint['style_result']
    need(style.get('id') == sid and style.get('dance_style') == joint['dance_style_label'], 'Known-style source/label identity differs')
    known = style.get('status') in ('candidate', 'resolved') and isinstance(style.get('dance_style'), str)
    known = known and style['dance_style'] not in ('', 'unknown')
    if not known:
        return dict(status='partial_unchanged', reason='unknown_style', id=sid, joint_complete=False, **FLAGS)
    visual_path = Path(joint['visual_result_ref']['path']).resolve()
    need(visual_path.is_relative_to(consumer / 'visual'), 'Wrong version/source visual reference')
    original = evidence.checked(joint['visual_result_ref'])
    need(original['id'] == sid and original['binding'] == joint['source_binding']
         and original['manifest_record']['id'] == sid and original['binding']['source_binding']['id'] == sid
         and original['binding']['params_sha256'] == notice['params_sha256'], 'Original visual/index/source binding differs')
    if any(o.get('status') != 'candidate' for o in original['window_observations']):
        return dict(status='partial_unchanged', reason='abstain_or_incomplete_window', id=sid, joint_complete=False, **FLAGS)
    if original['final'].get('status') != 'failed':
        return dict(status='partial_unchanged', reason='not_original_rewrite_failure', id=sid, joint_complete=False, **FLAGS)
    source_plan = evidence.load(RT / ('curated_collaborative%d_source_plan_v1.json' % version))
    member = next(name for name, row in source_plan['members'].items()
                  if row['mode'] == 'consumer' and row['shard'] == int(consumer.name.split('-')[-1]))
    selected = evidence.checked(source_plan['selected_sources'][member])
    need(dict(id=sid, split=split, batch_id=notice['batch_id'], source_ordinal=original['manifest_record']['source_ordinal']) in selected,
         'Source not selected in original version/shard')
    root = Path(original['raw_rgb_manifest']['path']).resolve().parent.parent
    need(root.is_relative_to(consumer / 'visual') and root.name == sid and visual_path.is_relative_to(root), 'Visual/prepared source root differs')
    prepared_receipt = evidence.load(root / 'prepared.receipt.json')
    prepared = evidence.load(root / 'prepared.json', prepared_receipt['prepared_sha256'])
    plan = evidence.load(root / 'plan.json')
    need(prepared_receipt['binding'] == original['binding'] == prepared['binding']
         and bar.batch.digest(plan) == original['binding']['full_source_row_sha256']
         and bar.batch.digest(original['manifest_record']) == original['binding']['manifest_record_sha256']
         and plan['binding'] == original['binding']['source_binding']
         and plan['manifest_record'] == original['manifest_record']
         and plan['windows'] == prepared['original_windows'] == original['original_windows']
         and bar.normalized_windows(plan) == [m['window'] for m in prepared['windows']]
         and plan['window_manifest'] == prepared['window_manifest'] == original['original_window_manifest'], 'Plan/prepared/manifest SHA binding differs')
    need(evidence.checked(original['model_identity_ref'])['sha256'] == joint['annotator_model_sha256'],
         'Original joint/model identity differs')
    media = source_boundary(prepared['source_identity'], original['manifest_record'], original['binding'], crossmount)
    windows, facts, observation_refs = replay(evidence, original, prepared, root)
    final = direct.direct_final(windows, facts, [])
    final['derivation'] = RECIPE
    need(final['status'] == 'candidate', 'Direct final native supported rewrite validation failed')
    need(all(w['caption'].strip() for w in final['json']['windows']), 'Direct caption has an empty window')
    visual = dict(id=sid, status='candidate', binding=copy.deepcopy(original['binding']),
        original_windows=copy.deepcopy(windows), final=final, facts=copy.deepcopy(facts),
        final_timed_caption=bar.core.timed_text(dict(windows=windows), final['json']['windows']),
        coverage=dict(full_window_caption=True), raw_rgb_manifest=copy.deepcopy(original['raw_rgb_manifest']),
        hand_roi=copy.deepcopy(original.get('hand_roi')), **FLAGS)
    result = pipeline.joint_candidate(joint, visual, copy.deepcopy(style), joint['annotator_model_sha256'])
    need(result['joint_complete'] is True and result['style_result'] == style, 'Known-style joint derivation not complete')
    result.update(schema=RECIPE, status='direct_cpu_joint_candidate', caption_recipe=RECIPE,
        direct_final=final, admitted_facts=copy.deepcopy(facts), original_joint_ref=copy.deepcopy(notice['joint_candidate_ref']),
        original_visual_result_ref=copy.deepcopy(joint['visual_result_ref']), original_final=copy.deepcopy(original['final']),
        original_model_identity_ref=copy.deepcopy(original['model_identity_ref']),
        original_window_manifest=copy.deepcopy(original['original_window_manifest']), observation_provenance=observation_refs,
        original_source_identity=copy.deepcopy(prepared['source_identity']), source_stat_boundary=media,
        original_source_refs=copy.deepcopy(original['source_refs']),
        original_boundary_observations=copy.deepcopy(original['boundary_observations']),
        derivation_of_immutable_admitted_annotation_facts=True, new_visual_review_performed=False,
        new_model_calls=0, new_gpu_use=False, new_budget_reservations=0, new_source_attempts=0,
        original_generation_not_copied_to_new_identity=True, original_production_counts_changed=False,
        quality_review_status='qualitylater', **FLAGS)
    for row in media:
        need(stat(row['path']) == row['current_stat'], 'Media changed across derivation boundary')
    return result


def write_once(path, value):
    bar, pipeline = direct.native()
    need(not Path(path).exists(), 'Refusing overwrite: ' + str(path))
    return pipeline.save(path, value)


class Fastlane:
    def __init__(self, output, crossmount=False):
        self.output = Path(output).resolve()
        need(self.output.is_relative_to(BAR) and self.output.name.startswith('direct_cpu_completion_fastlane_v1'),
             'Output must be an independent new fastlane directory under BAR')
        self.crossmount = crossmount
        self.writes = 0
        self.execution = Evidence()
        self.identity = dict(schema=RECIPE, execution_ref=self.execution.ref(__file__),
            direct_helper_ref=self.execution.ref(direct.__file__),
            frozen_native_refs=[self.execution.ref(direct.SOURCE / name) for name in direct.FROZEN],
            allow_crossmount_device_only=crossmount,
            immutable_annotation_derivation_not_pixel_review=True, **FLAGS)
        config = self.output / 'recipe.json'
        if config.exists():
            need(self.execution.load(config) == self.identity, 'Existing output recipe/execution version differs')
        else:
            write_once(config, self.identity)
            self.writes += 1

    def process(self, notice, consumer, version, input_ref):
        bar, pipeline = direct.native()
        key = bar.batch.digest(dict(id=notice['id'], split=notice['split'],
            original_joint_ref=notice['joint_candidate_ref'], recipe=self.identity))
        receipt_path = self.output / 'sources' / ('v%d' % version) / consumer.name / key / 'receipt.json'
        evidence = Evidence()
        if receipt_path.exists():
            receipt = evidence.load(receipt_path)
            need(receipt['recipe_identity'] == self.identity and receipt['source_key'] == key, 'Existing source recipe identity differs')
            for row in receipt['annotation_stat_boundary']:
                need(stat(row['path']) == row['stat'], 'Previously admitted annotation changed')
            for row in receipt.get('source_stat_boundary', []):
                need(stat(row['path']) == row['current_stat'], 'Previously admitted source changed')
            if receipt.get('candidate_ref'):
                evidence.checked(receipt['candidate_ref'])
            return dict(id=notice['id'], status='existing_exact', joint_complete=receipt['joint_complete'], writes=0)
        started = time.perf_counter()
        try:
            result = derive(evidence, notice, consumer, version, self.crossmount)
            boundary = evidence.boundary()
            self.execution.boundary()
            candidate_ref = None
            if result['joint_complete']:
                candidate_path = receipt_path.parent / 'candidate.json'
                # The recipe identity belongs to the CPU derivation, never old generation.
                result['recipe_identity'] = copy.deepcopy(self.identity)
                result['annotation_stat_boundary'] = boundary
                write_once(candidate_path, result)
                self.writes += 1
                candidate_ref = dict(path=str(candidate_path), sha256=pipeline.digest(result))
            receipt = dict(schema=RECIPE, source_key=key, recipe_identity=self.identity,
                status=result['status'], id=notice['id'], joint_complete=result['joint_complete'],
                reason=result.get('reason'), candidate_ref=candidate_ref, original_joint_ref=notice['joint_candidate_ref'],
                completed_input_prefix_ref=input_ref, annotation_stat_boundary=boundary,
                source_stat_boundary=result.get('source_stat_boundary', []),
                annotation_artifacts_read_once=evidence.read_count, rgb_jpeg_media_weight_content_reads=0,
                cpu_seconds=time.perf_counter()-started, new_model_calls=0, new_budget_reservations=0, **FLAGS)
            write_once(receipt_path, receipt)
            self.writes += 1
            return dict(id=notice['id'], status=result['status'], joint_complete=result['joint_complete'],
                        candidate_ref=candidate_ref, receipt_path=str(receipt_path), cpu_seconds=receipt['cpu_seconds'])
        except (ValueError, OSError, KeyError, TypeError, StopIteration) as error:
            failure = dict(schema=RECIPE, source_key=key, recipe_identity=self.identity, id=notice['id'],
                status='source_rejected', joint_complete=False, error=type(error).__name__ + ': ' + str(error),
                original_joint_ref=notice['joint_candidate_ref'], completed_input_prefix_ref=input_ref,
                annotation_stat_boundary=[], new_model_calls=0, new_budget_reservations=0, **FLAGS)
            write_once(receipt_path, failure)
            self.writes += 1
            return dict(id=notice['id'], status='source_rejected', joint_complete=False, error=failure['error'])


def completed_snapshot(path):
    path = Path(path).resolve()
    rel = path.relative_to(BAR)
    need(len(rel.parts) == 4 and rel.parts[1] == 'consumers' and rel.parts[-1] == 'completed.jsonl'
         and re.fullmatch(r'curated222939_collaborative_v[34]', rel.parts[0])
         and re.fullmatch(r'shard-[0-9]{4}', rel.parts[2]), 'Only explicit collab3/4 single-writer completed indexes admitted')
    size = path.stat().st_size
    with path.open('rb') as stream:
        raw = stream.read(size)
    raw = raw[:raw.rfind(b'\n')+1]
    bar, _ = direct.native()
    rows = [bar.core.strict_json(line) for line in raw.splitlines()]
    ref = dict(path=str(path), prefix_bytes=len(raw), prefix_sha256=hashlib.sha256(raw).hexdigest(),
               append_only_hint_not_full_population_manifest=True)
    return rows, ref, path.parent, int(rel.parts[0][-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed', action='append', default=[])
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--out', default=str(BAR / 'direct_cpu_completion_fastlane_v1'))
    parser.add_argument('--pilot3', action='store_true')
    parser.add_argument('--allow-crossmount-device-only', action='store_true')
    args = parser.parse_args()
    need(args.limit > 0, 'Positive incremental source limit required')
    if args.pilot3:
        need(not args.completed and args.limit == 3, 'Pilot requires fixed three-source inputs only')
        args.completed = [str(BAR / ('curated222939_collaborative_v%d' % v) / 'consumers' / ('shard-%04d' % s) / 'completed.jsonl')
                          for v, s, _, _ in direct.CASES]
    need(bool(args.completed), 'Explicit completed input paths required')
    snapshots = [completed_snapshot(path) for path in dict.fromkeys(args.completed)]
    items = []
    if args.pilot3:
        for v, s, sid, _ in direct.CASES:
            rows, ref, consumer, version = next(t for t in snapshots if t[3] == v and t[2].name == 'shard-%04d' % s)
            notice = next(row for row in rows if row['id'] == sid)
            items.append((notice, consumer, version, ref))
    else:
        for rows, ref, consumer, version in snapshots:
            for row in rows:
                if (row.get('joint_complete') is False and row.get('visual_status') == 'failed'
                    and row.get('style_status') in ('candidate', 'resolved')):
                    items.append((row, consumer, version, ref))
    lane = Fastlane(args.out, args.allow_crossmount_device_only)
    results, new = [], 0
    for item in items:
        if new >= args.limit:
            break
        result = lane.process(*item)
        results.append(result)
        if result['status'] != 'existing_exact':
            new += 1
    print(json.dumps(dict(schema=RECIPE, results=results, writes=lane.writes, new_sources=new,
        new_model_calls=0, new_budget_reservations=0, original_production_counts_changed=False, **FLAGS), ensure_ascii=False))


if __name__ == '__main__':
    main()
