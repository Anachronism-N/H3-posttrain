"""Draft generation first, pixel-grounded revision later; never a training export."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
RT = Path(__file__).resolve().parent
BAR = RT.parent
sys.path.insert(0, str(RT))
import caption_direct_facts_v1 as native
import visual_batch_fixed3_benchmark_v1 as schema

RECIPE = 'caption_two_stage_draft_v1'
FLAGS = dict(training_eligible=False, semantic_approval=False, quality_approved=False,
             pixel_quality_review_completed=False, needs_visual_revision=True,
             formal_training=False, human_review_completed=False)
SOURCES = ('VGfurWgIdrVQNyhI','HVsstNGHcxHNsWkM','VrydLuJOvAnzhiQt')


def need(value, message):
    if not value:
        raise ValueError(message)


class Reader:
    def __init__(self):
        self.cache = {}
        self.reads = 0
    def raw(self,path,expected=None):
        path = Path(path).resolve(strict=True)
        if path not in self.cache:
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
                 (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'Input changed during read')
            self.cache[path] = raw, hashlib.sha256(raw).hexdigest(), after
            self.reads += 1
        raw,sha,_ = self.cache[path]
        need(expected is None or expected == sha,'Original annotation SHA differs')
        return raw
    def load(self,path,expected=None):
        bar,_ = native.native()
        return bar.core.strict_json(self.raw(path,expected))
    def checked(self,ref):
        return self.load(ref['path'],ref['sha256'])
    def ref(self,path):
        path = Path(path).resolve(strict=True)
        self.raw(path)
        return dict(path=str(path),sha256=self.cache[path][1])
    def verify(self):
        for path,(_,_,before) in self.cache.items():
            after = path.stat()
            need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
                 (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'Input changed before draft publication')


def schema_only(value,manifest):
    bar,_ = native.native()
    schema.schema_check(value,bar.observation_schema([f['index'] for f in manifest['frames']]))
    return value


def window_draft(reader,observation,binding,model):
    bar,_ = native.native()
    window = observation['window']
    attempt = Path(observation.get('attempt_path',''))
    result = dict(window=copy.deepcopy(window), original_status=observation['status'],
                  original_filtered_observation=copy.deepcopy(observation.get('json')), **FLAGS)
    if not observation.get('attempt_path'):
        result.update(status='draft_needs_regeneration',draft_text='',reason='no_model_observation',raw_text=None)
        return result
    need(attempt.parent.name == window['window_id'], 'Wrong original window attempt')
    request = reader.load(attempt/'request.json')
    original_binding = reader.load(attempt/'binding.json')
    receipt = reader.load(attempt/'receipt.json')
    need(receipt['binding'] == original_binding and original_binding['source_binding'] == binding['source_binding']
         and original_binding['window'] == window,'Original source/window binding differs')
    for name in ('request.json','binding.json','response.raw.txt','generation.json'):
        if name in receipt['files_sha256']:
            reader.raw(attempt/name,receipt['files_sha256'][name])
    raw = reader.raw(attempt/'response.raw.txt')
    result.update(raw_text=raw.decode('utf-8',errors='replace'),
        original_request_ref=reader.ref(attempt/'request.json'), original_receipt_ref=reader.ref(attempt/'receipt.json'),
        original_binding_ref=reader.ref(attempt/'binding.json'),original_raw_ref=reader.ref(attempt/'response.raw.txt'))
    try:
        g = reader.load(attempt/'generation.json')
        need(g['request_sha256'] == reader.ref(attempt/'request.json')['sha256'] and
             g['raw_response_sha256'] == hashlib.sha256(raw).hexdigest() and g['model_call_performed'] is True and
             g['model_identity_sha256'] == model['sha256'] and request['model'] == model['root'] and
             g['max_tokens'] == request['sampling']['max_tokens'] and
             g['structured_output_schema_sha256'] == bar.core.digest(request['json_schema']), 'Generation identity/budget differs')
        need(g['finish_reason'] == 'stop' and type(g['completion_tokens']) is int and
             0 < g['completion_tokens'] < g['max_tokens'],'Truncated or abnormal generation')
        value = bar.core.strict_json(raw)
        schema.schema_check(value,request['json_schema'])
    except (ValueError,KeyError,OSError) as error:
        result.update(status='draft_needs_regeneration',draft_text='',reason=str(error))
    else:
        result.update(status='draft_generated' if value['events'] else 'draft_partial',
            raw_observation=value, draft_text=' '.join(e['description'] for e in value['events']),
            uncertainties=copy.deepcopy(value['uncertainties']),
            original_generation_ref=reader.ref(attempt/'generation.json'),
            support_and_semantic_filter_deferred=True)
    return result


def reuse_source(visual_ref,style,reader):
    bar,_ = native.native()
    visual = reader.checked(visual_ref)
    binding = visual['binding']
    need(visual['id'] == binding['source_binding']['id'],'Original visual source differs')
    model = reader.checked(visual['model_identity_ref'])
    windows = [window_draft(reader,o,binding,model) for o in visual['window_observations']]
    return assemble(visual['id'],visual['manifest_record'],binding,windows,style,
        original_visual_ref=visual_ref, original_model_ref=visual['model_identity_ref'],
        raw_rgb_manifest=visual['raw_rgb_manifest'], hand_roi=visual.get('hand_roi'))


def assemble(sid,record,binding,windows,style,**provenance):
    bar,_ = native.native()
    known = (style and style.get('id') == sid and style.get('status') in ('candidate','resolved')
        and style.get('dance_style') not in (None,'','unknown')
        and style.get('source') in ('historical_RGB_label','rich_chapter2','independent_RGB')
        and style.get('metadata',{}).get('video_sha256')==binding['source_binding'].get('video_sha256'))
    label = style['dance_style'] if known else 'unknown'
    texts = [dict(window_id=w['window']['window_id'],caption=w['draft_text']) for w in windows]
    body = bar.core.timed_text(dict(windows=[w['window'] for w in windows]),texts)
    all_generated = bool(windows) and all(w['status']=='draft_generated' for w in windows)
    return dict(schema=RECIPE,id=sid,split=record['split'],batch_id=record['batch_id'],
        source_binding=copy.deepcopy(binding), manifest_record=copy.deepcopy(record), source_refs=record.get('source_refs'),
        status='draft_generated' if all_generated else 'draft_needs_regeneration' if any(w['status']=='draft_needs_regeneration' for w in windows) else 'draft_partial',
        windows=windows, style_result=copy.deepcopy(style), dance_style=label,
        style_revision_pending=not bool(known), unverified_motion_script=body,
        unverified_draft_text=('Dance style: '+label+'\n'+body) if known else body,
        next_stage='pixel_grounded_vlm_revision', no_empty_window_filled_with_stillness=True,
        support_and_semantic_filter_deferred=True, new_model_calls=0, **provenance, **FLAGS)


def publish(draft,out,reader):
    bar,pipeline = native.native()
    key = bar.batch.digest(dict(recipe=RECIPE,id=draft['id'],split=draft['split'],
        source_binding=draft['source_binding'],origin=draft.get('original_visual_ref')))
    directory = out/'drafts'/draft['split']/draft['id']/key
    receipt_path = directory/'receipt.json'
    if receipt_path.exists():
        receipt = reader.load(receipt_path)
        reader.checked(receipt['draft_ref'])
        need(receipt['source_key'] == key,'Existing draft version differs')
        return receipt
    reader.verify()
    path = pipeline.save(directory/'draft.json',draft)
    receipt = dict(schema=RECIPE,source_key=key,id=draft['id'],split=draft['split'],
        draft_ref=pipeline.ref(path),status=draft['status'],next_stage='pixel_grounded_vlm_revision',**FLAGS)
    pipeline.save(receipt_path,receipt)
    return receipt


def export_existing(index_paths,out,limit=3,ids=None):
    bar,pipeline = native.native()
    out = Path(out).resolve()
    need(out.parent == BAR and out.name.startswith('caption_two_stage_'),'Independent two-stage output required')
    reader=Reader()
    started=time.perf_counter()
    with bar.batch.bundle_lock(out):
        queued=set()
        queue=out/'revision_queue.jsonl'
        if queue.exists():
            queued={json.loads(line)['source_key'] for line in queue.read_text().splitlines() if line.strip()}
        seen=set();results=[];new=0;read_count=0
        for path in index_paths:
            path=Path(path).resolve()
            need(path.name=='completed.jsonl' and path.parent.name.startswith('shard-') and path.is_relative_to(BAR), 'Explicit original shard completed index required')
            raw=path.read_bytes();raw=raw[:raw.rfind(b'\n')+1]
            for line in raw.splitlines():
                row=json.loads(line);key=(row.get('split'),row.get('id'))
                if key in seen or (ids is not None and row.get('id') not in ids) or not row.get('joint_candidate_ref'):continue
                seen.add(key)
                reader=Reader()
                joint=reader.checked(row['joint_candidate_ref'])
                need(joint['id']==row['id'] and joint['split']==row['split'],'Joint source/index mismatch')
                need(Path(joint['visual_result_ref']['path']).resolve().is_relative_to(path.parent/'visual'), 'Cross-shard visual reference')
                draft=reuse_source(joint['visual_result_ref'],joint.get('style_result'),reader)
                receipt=publish(draft,out,reader)
                read_count+=reader.reads
                if receipt['source_key'] not in queued:
                    pipeline.append(queue,receipt);queued.add(receipt['source_key']);new+=1
                results.append(dict(id=draft['id'],status=draft['status'],draft_ref=receipt['draft_ref']))
                if new>=limit:break
            if new>=limit:break
        return dict(recipe=RECIPE,new_drafts=new,visited_sources=len(results),results=results,
            draft_complete_count=sum(x['status']=='draft_generated' for x in results),
            metadata_reads=read_count,media_or_weight_content_reads=0,new_model_calls=0,
            wall_seconds=time.perf_counter()-started,revision_queue=str(queue),**FLAGS)


def generate_batch(entries,args,runtime,out,resource_admission=None,styles=None,ref_cache=None):
    """Caller retains actual resource ownership and an exclusive source assignment.

    Only the original output schema is checked here. Semantic support, motion,
    style completion and fluent rewriting belong to the revision stage.
    """
    from contextlib import ExitStack
    bar,pipeline=native.native()
    out=Path(out).resolve()
    need(callable(resource_admission) and resource_admission(args,runtime) is True,'Caller resource admission required')
    need(out.parent==BAR and out.name.startswith('caption_two_stage_'),'Independent draft output required')
    need(args.stage=='run' and args.batch_size==8 and args.max_num_seqs>=8 and not args.retry_failed,
         'Explicit native batch8 run without retries required')
    need(callable(runtime) and 0<len(entries)<=3,'Caller-owned runtime and at most three whole sources required')
    need(len({e['row']['id'] for e in entries})==len(entries),'Duplicate draft source')
    for entry in entries:
        need(Path(entry['root']).resolve()==out/'sources'/entry['row']['id'],'Draft source output differs')
    started=time.perf_counter();ref_cache={} if ref_cache is None else ref_cache
    prepared_rows=[];items=[];positions=[];loaded=[]
    with bar.batch.bundle_lock(out), ExitStack() as locks:
        need(not (out/'batch.claim.json').exists(),'Draft batch already claimed; do not redispatch')
        for entry in entries:
            locks.enter_context(bar.batch.bundle_lock(entry['root']))
            prepared=bar.prepare_source(entry['row'],entry['raw_line'],Path(entry['root']),args,entry['binding'],ref_cache)
            prepared_rows.append(prepared)
            for manifest in prepared['windows']:
                root=Path(entry['root'])/'windows'/manifest['window']['window_id']/'attempt-0'
                if len(manifest['frames'])<2:
                    positions.append((entry['row']['id'],manifest,None))
                    continue
                request=bar.visual_request(args,Path(entry['root'])/'rgb',manifest,{k:set(v) for k,v in prepared['forbidden'].items()})
                request.update(caption_recipe=RECIPE,generation_recipe=RECIPE,semantic_filter_deferred=True)
                items.append(dict(root=root,request=request,validator=lambda value,m=manifest:schema_only(value,m),
                    binding=dict(entry['binding'],window=manifest['window'],frames_sha256=bar.batch.digest(manifest['frames']),draft_recipe=RECIPE)))
                positions.append((entry['row']['id'],manifest,root))
        pipeline.save(out/'batch.claim.json',dict(recipe=RECIPE,reserved_window_calls=len(items),
            source_ids=[e['row']['id'] for e in entries],**FLAGS))
        def once():
            if not loaded:
                loaded.append(runtime())
            return loaded[0]
        results=bar.transactions(items,once,args)
        published=[]
        for entry,prepared in zip(entries,prepared_rows):
            windows=[]
            for sid,manifest,root in positions:
                if sid!=entry['row']['id']:continue
                w=dict(window=manifest['window'],original_status='new_draft',**FLAGS)
                if root is None:
                    w.update(status='draft_partial',draft_text='',raw_text=None,reason='fewer_than_two_actual_frames')
                else:
                    result=results[str(root)]
                    w.update(original_raw_ref=pipeline.ref(root/'response.raw.txt'),
                        original_request_ref=pipeline.ref(root/'request.json'),original_receipt_ref=pipeline.ref(root/'receipt.json'),
                        raw_text=(root/'response.raw.txt').read_text(errors='replace'))
                    if result['status']=='candidate':
                        value=result['json']
                        w.update(raw_observation=value,uncertainties=value['uncertainties'],
                            status='draft_generated' if value['events'] else 'draft_partial',
                            draft_text=' '.join(e['description'] for e in value['events']))
                    else:
                        w.update(status='draft_needs_regeneration',draft_text='',reason=result.get('error','generation_failed'))
                windows.append(w)
            bar.batch.verify_source(prepared['source_identity'])
            bar.core.verify_references(prepared['source_refs'])
            draft=assemble(entry['row']['id'],entry['row']['manifest_record'],entry['binding'],windows,
                (styles or {}).get(entry['row']['id']),source_identity=prepared['source_identity'],
                raw_rgb_manifest=pipeline.ref(Path(entry['root'])/'rgb/manifest.json'),hand_roi=prepared['hand_roi'])
            draft['new_model_calls']=sum(root is not None for sid,_,root in positions if sid==entry['row']['id'])
            receipt=publish(draft,out,Reader());pipeline.append(out/'revision_queue.jsonl',receipt);published.append(receipt)
        return dict(recipe=RECIPE,source_count=len(entries),window_call_count=len(items),published=published,
            synchronous_rewrite_calls=0,synchronous_review_calls=0,new_style_model_calls=0,
            wall_seconds=time.perf_counter()-started,**FLAGS)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--completed',action='append',default=[])
    parser.add_argument('--out',default=str(BAR/'caption_two_stage_drafts_reuse_v1'))
    parser.add_argument('--limit',type=int,default=3)
    parser.add_argument('--pilot3',action='store_true')
    args=parser.parse_args()
    need(args.limit>0,'Positive finite batch limit required')
    if args.pilot3:
        need(not args.completed and args.limit==3,'Fixed three-source pilot only')
        args.completed=[str(BAR/'curated222939_collaborative_v3/consumers'/('shard-%04d'%i)/'completed.jsonl') for i in range(3)]
    need(args.completed,'Explicit completed input required')
    print(json.dumps(export_existing(args.completed,args.out,args.limit,set(SOURCES) if args.pilot3 else None),ensure_ascii=False))

if __name__=='__main__':main()
