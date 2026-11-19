"""Stage-one drafts: basic inputs, source isolation, and write-once budgets only."""
import base64
import copy
from functools import lru_cache
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from contextlib import ExitStack

import numpy as np
from PIL import Image
import caption_direct_facts_v1 as native
import visual_batch_fixed3_benchmark_v1 as schemas

RECIPE='caption_two_stage_light_draft_v1'
FLAGS=dict(training_eligible=False,semantic_approval=False,quality_approved=False,
    needs_visual_revision=True,pixel_quality_review_completed=False)


def need(ok,message):
    if not ok:raise ValueError(message)


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()


def key(value):
    # Small request/assignment identity, never a media or JPEG checksum.
    return hashlib.sha256(encoded(value)).hexdigest()


def write_once(path,value):
    path=Path(path);data=encoded(value);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        need(path.read_bytes()==data,'Refusing changed existing output: '+str(path))
        return
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name,delete=False) as stream:
        temporary=Path(stream.name)
        try:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
            try:os.link(temporary,path)
            except FileExistsError:
                need(path.read_bytes()==data,'Refusing concurrent changed output: '+str(path))
        finally:temporary.unlink(missing_ok=True)


def load(path):
    return schemas.strict(Path(path).read_bytes())


@lru_cache(maxsize=1)
def bar_module():
    return native.native()[0]  # One startup code binding, not once per window.


def validate_source(row,binding):
    sid=row['id'];record=row['manifest_record']
    need(bar_module().portable(sid) and sid==record['id']==row['binding']['id']==binding['source_binding']['id'],
        'Source ID mismatch')
    need(record['split'] in ('train','val','test'),'Invalid source split')
    if 'split' in binding:need(binding['split']==record['split'],'Source split mismatch')
    return sid,record['split']


def valid_points(points,scores):
    points=np.asarray(points);scores=np.asarray(scores)
    need(points.ndim>=2 and points.shape[-1]==2 and points.shape[:-1]==scores.shape,
        'Invalid keypoint/score array shapes')
    valid=np.isfinite(scores)&(scores>0)
    need(np.all(np.isfinite(points[valid])) and np.all(np.isfinite(scores)),
        'Invalid active points or scores')
    return valid


class Inputs:
    def __init__(self):self.values={};self.shapes={};self.jpeg_reads=0
    def json(self,path):
        path=Path(path)
        if path not in self.values:self.values[path]=load(path)
        return self.values[path]
    def jpeg(self,path,expected_size=None):
        path=Path(path)
        if path not in self.values:
            raw=path.read_bytes()
            with Image.open(BytesIO(raw)) as image:
                need(image.format=='JPEG' and image.width>0 and image.height>0,'Invalid readable JPEG')
                self.shapes[path]=(image.width,image.height)
            self.values[path]='data:image/jpeg;base64,'+base64.b64encode(raw).decode()
            self.jpeg_reads+=1
        need(expected_size is None or self.shapes[path]==tuple(expected_size),'JPEG array shape differs from input')
        return self.values[path]


def validate_prepared(prepared,row,binding):
    validate_source(row,binding)
    need(prepared['binding']==binding,'Prepared source binding mismatch')
    times=prepared['probe']['times'];pts=prepared['probe']['source_timestamps']
    need(len(times)==len(pts)>0 and all(type(t) in (float,int) and math.isfinite(t) for t in times+pts)
        and all(a<b for a,b in zip(times,times[1:])) and all(a<b for a,b in zip(pts,pts[1:])),
        'Invalid or unordered source times')
    ids=[];previous_end=-math.inf
    for item in prepared['windows']:
        w=item['window'];start,end=w['start'],w['end'];ids.append(w['window_id'])
        need(bar_module().portable(w['window_id']) and math.isfinite(start) and math.isfinite(end)
            and start<end and start>=previous_end,'Invalid or unordered windows')
        previous_end=end
        indices=[f['index'] for f in item['frames']]
        need(all(type(i) is int and 0<=i<len(times) for i in indices) and indices==sorted(set(indices)),
            'Invalid or unordered frame indices')
        for f in item['frames']:
            need(start<=f['time']<end and f['time']==times[f['index']]
                and f['source_timestamp']==pts[f['index']],'Frame time/source mismatch')
    need(len(ids)==len(set(ids)),'Duplicate window IDs')
    return prepared


def prepare_entry(entry,args,inputs):
    row,binding,root=entry['row'],entry['binding'],Path(entry['root'])
    sid,split=validate_source(row,binding)
    if 'prepared' in entry:
        return validate_prepared(entry['prepared'],row,binding)
    if (root/'prepared.draft.json').exists():
        return validate_prepared(inputs.json(root/'prepared.draft.json'),row,binding)
    if 'keypoints' in entry or 'scores' in entry:
        valid_points(entry['keypoints'],entry['scores'])
    actual=entry.get('actual')
    need(actual is not None,'Existing producer actual-media timeline required; no repeat probe')
    probe=actual['video_probe']
    if 'time' in probe:
        probe=dict(probe,times=probe['time'],source_timestamps=probe.get('source_timestamps',
            [t+probe.get('pts_offset',0.) for t in probe['time']]))
    times=probe['times'];pts=probe['source_timestamps']
    need(len(times)==len(pts)>0 and all(math.isfinite(t) for t in times+pts)
        and all(a<b for a,b in zip(times,times[1:])) and all(a<b for a,b in zip(pts,pts[1:])),
        'Invalid producer timeline')
    roi=entry.get('roi_manifest')
    if roi is None and args.hand_roi_root:
        path=Path(args.hand_roi_root)/sid/'roi_manifest.json'
        if path.exists():roi=inputs.json(path)
    if roi:
        need(roi['id']==sid and roi.get('video_sha256')==row['binding']['video_sha256'],'ROI source mismatch')
        if roi.get('raw') and entry.get('validate_pose_arrays',False):
            with np.load(roi['raw']['path'],allow_pickle=False) as raw:
                valid_points(raw['keypoints'],raw['scores'])
    forbidden={}
    if row.get('raw2d_mask'):
        mask=inputs.json(row['raw2d_mask']['path'])
        need(mask['id']==sid,'Mask source mismatch')
        for part,values in mask.get('parts',{}).items():
            need(part in bar_module().PARTS and isinstance(values,dict),'Invalid part mask shape')
            need(all(str(i).isdigit() and 0<=int(i)<len(times) and type(v) is bool for i,v in values.items()),'Invalid part mask points')
            forbidden[part]=[int(i) for i,v in values.items() if v is False]
    windows=bar_module().normalized_windows(row);selections=[];roi_groups=[]
    for w in windows:
        selected,_=bar_module().select_window_frames(times,w['start'],w['end'],args.sample_fps)
        groups={}
        for part in ('left_hand','right_hand'):
            crops=[c for c in (roi or {}).get('crops',[]) if c['part']==part
                and w['start']<=c['time']<w['end']]
            need(len({c['frame_index'] for c in crops})==len(crops),'Duplicate ROI frame')
            crops.sort(key=lambda c:c['frame_index'])
            if len(crops)<2:crops=[]
            elif len(crops)>3:crops=[crops[int(i*(len(crops)-1)/2)] for i in range(3)]
            for c in crops:
                i=c['frame_index']
                need(type(i) is int and 0<=i<len(times) and c['time']==times[i] and c['source_timestamp']==pts[i]
                    and len(c['crop_pixel_size'])==2 and all(type(n) is int and n>0 for n in c['crop_pixel_size']),
                    'Invalid ROI index/time/shape')
            groups[part]=dict(frame_indices=[c['frame_index'] for c in crops],crops=crops)
        required={i for g in groups.values() for i in g['frame_indices']}
        if required:
            available=[i for i,t in enumerate(times) if w['start']<=t<w['end']]
            candidates=sorted(set(selected)|{available[0],available[-1]})
            body=set(required)
            for k in range(len(candidates)):
                if len(body)>=8:break
                pos=0 if k==0 else len(candidates)-1 if k==1 else round((k-1)*(len(candidates)-1)/max(1,len(candidates)-2))
                body.add(candidates[pos])
            for i in candidates:
                if len(body)>=min(8,len(set(candidates)|required)):break
                body.add(i)
            need(2<=len(body)<=8,'Invalid ROI/body input count')
            selected=sorted(body)
        selections.append(selected);roi_groups.append(groups)
    union=sorted({i for indices in selections for i in indices})
    rgb=root/'rgb';rgb.mkdir(parents=True,exist_ok=True)
    if union:
        expression='+'.join('eq(n\\,%d)'%i for i in union)
        script=rgb/'select.ffilter'
        write_once(root/'decode.assignment.json',dict(id=sid,split=split,indices=union,binding=binding))
        if not script.exists():script.write_text('select='+expression+',scale=640:640:force_original_aspect_ratio=decrease')
        # Decode once on fresh assignment. Interrupted outputs are never silently overwritten.
        need(not list(rgb.glob('*.jpg')),'Existing decoded frames require explicit prepared-input resume')
        subprocess.run([args.ffmpeg,'-nostdin','-v','error','-i',row['manifest_record']['video_path'],
            '-map','0:v:0','-filter_script:v',str(script),'-fps_mode','vfr','-q:v','3',str(rgb/'%06d.jpg')],
            check=True,capture_output=True)
    images=sorted(rgb.glob('*.jpg'))
    need(len(images)==len(union),'Decoded frame array count differs')
    frames={i:dict(index=i,time=times[i],source_timestamp=pts[i],file=p.name) for i,p in zip(union,images)}
    prepared=dict(schema=RECIPE,binding=binding,probe=probe,
        windows=[dict(window=w,frames=[frames[i] for i in indices],hand_roi_evidence=groups)
            for w,indices,groups in zip(windows,selections,roi_groups)],
        hand_roi=dict(available=bool(roi)),forbidden=forbidden,**FLAGS)
    validate_prepared(prepared,row,binding)
    write_once(root/'prepared.draft.json',prepared)
    return prepared


def request_for(root,manifest,args,inputs,forbidden=None):
    bar=bar_module();frames=manifest['frames'];indices=[f['index'] for f in frames]
    header=dict(allowed_source_frame_indices=indices,parts=list(bar.PARTS),
        explicit_raw2d_forbidden_part_frames={p:sorted(set(indices)&set(v)) for p,v in (forbidden or {}).items() if set(indices)&set(v)},
        window_is_only_container_not_new_action=True,ROI_same_frame_is_not_extra_temporal_support=True)
    content=[dict(type='text',text=json.dumps(header))]
    for f in frames:
        filename=Path(f['file'])
        need(not filename.is_absolute() and '..' not in filename.parts,'Body image escapes source')
        content.extend([dict(type='text',text='Source frame %d, time %.9fs, original PTS %.9fs.'%
            (f['index'],f['time'],f['source_timestamp'])),dict(type='image_url',image_url={'url':inputs.jpeg(Path(root)/'rgb'/f['file'])})])
        for part in ('left_hand','right_hand'):
            for crop in (manifest.get('hand_roi_evidence') or {}).get(part,{}).get('crops',[]):
                if crop['frame_index']==f['index']:
                    content.extend([dict(type='text',text=json.dumps(dict(part=part,frame_index=f['index'],
                        time=f['time'],source_timestamp=f['source_timestamp'],same_source_frame_as_body_context=True,
                        anatomical_side_is_unreviewed=True,not_another_time_observation=True))),
                        dict(type='image_url',image_url={'url':inputs.jpeg(crop['path'],crop['crop_pixel_size'])})])
    need(sum(c['type']=='image_url' for c in content)<=16,'Input image count exceeds runtime limit')
    return dict(model=str(args.model),messages=[dict(role='system',content=bar.SYSTEM),dict(role='user',content=content)],
        sampling=dict(temperature=0.,seed=0,max_tokens=args.visual_max_tokens),json_schema=bar.observation_schema(indices),
        caption_recipe=RECIPE,draft_only=True)


def transactions(items,runtime,args):
    bar=bar_module();results={};pending=[]
    for item in items:
        root=Path(item['root'])
        with bar.batch.bundle_lock(root):
            write_once(root/'request.json',dict(request=item['request'],binding=item['binding']))
            if (root/'result.json').exists():
                value=load(root/'result.json');need(value['binding']==item['binding'],'Result source binding differs')
                need(value['request_key']==key(item['request']),'Result request identity differs')
                results[str(root)]=value
            elif (root/'claim.json').exists():
                value=dict(status='draft_needs_regeneration',binding=item['binding'],request_key=key(item['request']),
                    reason='interrupted_reserved_call_not_resubmitted',raw_text=None,**FLAGS)
                write_once(root/'result.json',value);results[str(root)]=value
            else:pending.append(item)
    loaded=[]
    for offset in range(0,len(pending),args.batch_size):
        group=pending[offset:offset+args.batch_size]
        with ExitStack() as locks:
            for item in group:locks.enter_context(bar.batch.bundle_lock(item['root']))
            for item in group:
                root=Path(item['root'])
                need(not (root/'claim.json').exists() and not (root/'result.json').exists(),'Concurrent budget consumption')
                write_once(root/'claim.json',dict(binding=item['binding'],request_key=key(item['request']),reserved_calls=1,**FLAGS))
            try:
                if not loaded:loaded.append(runtime())
                llm,sampling_cls=loaded[0]
                sampling=[bar.core.structured_sampling(sampling_cls,i['request']['json_schema'],i['request']['sampling']['max_tokens']) for i in group]
                outputs=llm.chat([i['request']['messages'] for i in group],sampling,use_tqdm=False)
                need(len(outputs)==len(group) and all(len(o.outputs)==1 for o in outputs),'Unexpected model output cardinality')
            except Exception as error:
                for item in group:
                    value=dict(status='draft_needs_regeneration',binding=item['binding'],request_key=key(item['request']),
                        reason=type(error).__name__+': '+str(error),raw_text=None,**FLAGS)
                    write_once(Path(item['root'])/'result.json',value);results[str(item['root'])]=value
                continue
            for item,output in zip(group,outputs):
                c=output.outputs[0];raw=c.text
                value=dict(binding=item['binding'],request_key=key(item['request']),raw_text=raw,
                    generation=dict(finish_reason=c.finish_reason,prompt_tokens=len(output.prompt_token_ids or []),
                        completion_tokens=len(c.token_ids),max_tokens=item['request']['sampling']['max_tokens'],
                        model_identity=getattr(llm,'_ms_bar_model_sha256',None)),**FLAGS)
                try:
                    need(c.finish_reason=='stop' and 0<len(c.token_ids)<value['generation']['max_tokens'],'Truncated/abnormal generation')
                    observation=schemas.strict(raw);schemas.schema_check(observation,item['request']['json_schema'])
                except (ValueError,KeyError,TypeError) as error:value.update(status='draft_needs_regeneration',reason=str(error))
                else:value.update(status='draft_generated' if observation['events'] else 'draft_partial',raw_observation=observation,
                    draft_text=' '.join(e['description'] for e in observation['events']))
                write_once(Path(item['root'])/'result.json',value);results[str(item['root'])]=value
    return results


def load_entry(producer_root,meta,out,inputs):
    sid,split=meta['id'],meta['split']
    need(bar_module().portable(sid) and split in ('train','val','test'),'Invalid source ID/split')
    root=Path(producer_root)/'sources'/split/sid
    producer=inputs.json(root/'producer.terminal.json')
    need(producer['id']==sid and producer['split']==split and producer['status']=='ready','Producer source/status mismatch')
    row=inputs.json(producer['plan_ref']['path'])
    need(row['id']==sid and row['manifest_record']['id']==sid and row['manifest_record']['split']==split
        and row['binding']==producer['source_binding'],'Producer plan source/split mismatch')
    actual=inputs.json(root/'actual_media.json')
    identity=actual['source_identity']['files']['video_path']
    need(identity['path']==row['manifest_record']['video_path'],'Actual video source mismatch')
    return dict(row=row,actual=actual,root=Path(out)/'sources'/split/sid,
        binding=dict(source_binding=row['binding'],split=split,draft_recipe=RECIPE),
        original_producer_path=str(root/'producer.terminal.json'))


def export_existing(index_paths,out,limit=500):
    """Reuse raw drafts with identity/basic checks, never traverse historical reviews."""
    out=Path(out).resolve()
    need(out.name.startswith('caption_two_stage_'),'Independent draft output required')
    inputs=Inputs();new=0;visited=set();states={};index_failures=0
    with bar_module().batch.bundle_lock(out):
        queue=out/'revision_queue.jsonl';queued=set()
        if queue.exists():
            for line in queue.read_bytes().splitlines():
                item=schemas.strict(line)
                need(item['recipe']==RECIPE and not item['training_eligible'],'Unexpected draft queue recipe')
                queued.add(item['source_key'])
        for index_path in index_paths:
            index_path=Path(index_path).resolve()
            need(index_path.name=='completed.jsonl' and index_path.parent.name.startswith('shard-'),'Explicit source shard index required')
            raw=index_path.read_bytes();raw=raw[:raw.rfind(bytes([10]))+1]
            for line_number,line in enumerate(raw.splitlines(),1):
                try:
                    notice=schemas.strict(line);sid=notice['id'];split=notice['split']
                    need(bar_module().portable(sid) and split in ('train','val','test'),'Invalid source ID/split')
                except (ValueError,KeyError,TypeError) as error:
                    write_once(out/'failures'/(key(dict(index=str(index_path),line=line_number))+'.json'),
                        dict(index=str(index_path),line_number=line_number,status='invalid_source_index',
                            reason=str(error),source_id_unknown=True,**FLAGS))
                    index_failures+=1
                    continue
                if (split,sid) in visited or not notice.get('joint_candidate_ref'):continue
                visited.add((split,sid))
                source_key=key(dict(id=sid,split=split,original_joint_ref=notice['joint_candidate_ref'],recipe=RECIPE))
                if source_key in queued:continue
                try:
                    joint_path=Path(notice['joint_candidate_ref']['path']).resolve()
                    need(joint_path.is_relative_to(index_path.parent/'sources'/split/sid),'Joint result crosses source')
                    joint=inputs.json(joint_path)
                    need(joint['id']==sid and joint['split']==split,'Joint source ID/split mismatch')
                    visual_path=Path(joint['visual_result_ref']['path']).resolve()
                    need(visual_path.is_relative_to(index_path.parent/'visual') and visual_path.parent.parent.name==sid,
                        'Visual result crosses source')
                    visual=inputs.json(visual_path)
                    need(visual['id']==sid and visual['manifest_record']['id']==sid and visual['manifest_record']['split']==split
                        and visual['binding']['source_binding']['id']==sid,'Visual source ID/split mismatch')
                    windows=[];last=-math.inf
                    for observation in visual['window_observations']:
                        window=observation['window']
                        need(bar_module().portable(window['window_id']) and math.isfinite(window['start']) and math.isfinite(window['end'])
                            and last<=window['start']<window['end'],'Invalid window times')
                        last=window['end'];w=dict(window=window,**FLAGS)
                        attempt=observation.get('attempt_path')
                        if attempt:
                            attempt=Path(attempt).resolve()
                            need(attempt.is_relative_to(visual_path.parent.parent/'windows'/window['window_id']),
                                'Window attempt crosses source')
                            binding=inputs.json(attempt/'binding.json')
                            need(binding['source_binding']==visual['binding']['source_binding'] and binding['window']==window,
                                'Window source/time binding mismatch')
                            response=(attempt/'response.raw.txt').read_bytes().decode('utf-8',errors='replace')
                            w.update(raw_text=response,revision_request_path=str(attempt/'request.json'),
                                original_generation_path=str(attempt/'generation.json'))
                            try:
                                generation=inputs.json(attempt/'generation.json')
                                need(generation['finish_reason']=='stop' and 0<generation['completion_tokens']<generation['max_tokens'],
                                    'Truncated/abnormal observation')
                                value=schemas.strict(response)
                                need(isinstance(value,dict) and isinstance(value.get('events'),list),'Invalid observation shape')
                                descriptions=[e['description'] for e in value['events']]
                                need(all(isinstance(s,str) for s in descriptions),'Invalid event text shape')
                            except (ValueError,KeyError,TypeError,OSError) as error:
                                w.update(status='draft_needs_regeneration',draft_text='',reason=str(error))
                            else:
                                w.update(status='draft_generated' if descriptions else 'draft_partial',raw_observation=value,
                                    draft_text=' '.join(descriptions),uncertainties=value.get('uncertainties',[]))
                        else:w.update(status='draft_partial',draft_text='',reason='no_original_model_observation')
                        windows.append(w)
                    source=dict(id=sid,split=split,binding=visual['binding'],original_joint_ref=notice['joint_candidate_ref'],
                        original_visual_ref=joint['visual_result_ref'],windows=windows,style_result=joint.get('style_result'),
                        original_model_ref=visual.get('model_identity_ref'),**FLAGS)
                    source['status']='draft_generated' if windows and all(w['status']=='draft_generated' for w in windows) else 'draft_partial'
                except (OSError,ValueError,KeyError,TypeError) as error:
                    source=dict(id=sid,split=split,status='draft_needs_regeneration',reason=type(error).__name__+': '+str(error),
                        original_joint_ref=notice.get('joint_candidate_ref'),windows=[],**FLAGS)
                draft_path=out/'drafts'/split/sid/source_key/'draft.json';write_once(draft_path,source)
                item=dict(recipe=RECIPE,source_key=source_key,id=sid,split=split,draft_path=str(draft_path),
                    status=source['status'],next_stage='pixel_grounded_vlm_revision',**FLAGS)
                with queue.open('ab') as stream:
                    stream.write(encoded(item));stream.flush();os.fsync(stream.fileno())
                queued.add(source_key);new+=1;states[source['status']]=states.get(source['status'],0)+1
                # Keep only one source's metadata in memory; no all-history revalidation.
                inputs=Inputs()
                if new>=limit:break
            if new>=limit:break
    return dict(recipe=RECIPE,new_drafts=new,states=states,index_failures=index_failures,media_sha_reads=0,media_probes=0,
        jpeg_reads=0,history_review_reads=0,new_model_calls=0,**FLAGS)


def generate_batch(entries,args,runtime,out,resource_admission=None):
    need(callable(resource_admission) and resource_admission(args,runtime) is True,'Caller resource safety admission required')
    need(args.stage=='run' and args.batch_size==8 and args.max_num_seqs>=8 and not args.retry_failed,'Native batch8 without retries required')
    out=Path(out).resolve()
    need(out.name.startswith('caption_two_stage_'),'Independent draft output required')
    need(len(entries)>0 and len({validate_source(e['row'],e['binding']) for e in entries})==len(entries),'Duplicate source/split assignment')
    inputs=Inputs();items=[];sources=[]
    with bar_module().batch.bundle_lock(out):
        for entry in entries:
            sid,split=validate_source(entry['row'],entry['binding']);root=Path(entry['root']).resolve()
            need(root==out/'sources'/split/sid,'Source output isolation differs')
            source=dict(id=sid,split=split,binding=copy.deepcopy(entry['binding']),windows=[],**FLAGS)
            try:
                prepared=prepare_entry(entry,args,inputs)
                for manifest in prepared['windows']:
                    w=dict(window=copy.deepcopy(manifest['window']),**FLAGS)
                    if len(manifest['frames'])<2:w.update(status='draft_partial',draft_text='',reason='fewer_than_two_actual_frames')
                    else:
                        request=request_for(root,manifest,args,inputs,prepared.get('forbidden',{}));attempt=root/'windows'/manifest['window']['window_id']
                        items.append(dict(root=attempt,request=request,binding=dict(entry['binding'],window=manifest['window'])))
                        w['result_path']=str(attempt/'result.json')
                        w['revision_request_path']=str(attempt/'request.json')
                    source['windows'].append(w)
                source['prepared_path']=str(root/'prepared.draft.json') if 'prepared' not in entry else None
            except (OSError,ValueError,KeyError,TypeError,subprocess.CalledProcessError) as error:
                source.update(status='draft_needs_regeneration',reason=type(error).__name__+': '+str(error))
            sources.append(source)
        calls=[]
        class ModelCalls:
            def __init__(self,llm):self.llm=llm
            def __getattr__(self,name):return getattr(self.llm,name)
            def chat(self,messages,*a,**kw):
                calls.append(len(messages))
                return self.llm.chat(messages,*a,**kw)
        def counted_runtime():
            llm,sampling=runtime()
            return ModelCalls(llm),sampling
        results=transactions(items,counted_runtime,args)
        queue=out/'revision_queue.jsonl'
        queued=set()
        if queue.exists():
            for line in queue.read_bytes().splitlines():
                row=schemas.strict(line)
                need(row['recipe']==RECIPE and row['training_eligible'] is False,'Unexpected revision queue recipe')
                queued.add(row['source_key'])
        for source in sources:
            for w in source['windows']:
                if 'result_path' in w:w.update(results[str(Path(w['result_path']).parent)])
            source.setdefault('status','draft_needs_regeneration' if any(w['status']=='draft_needs_regeneration' for w in source['windows']) else
                'draft_generated' if source['windows'] and all(w['status']=='draft_generated' for w in source['windows']) else 'draft_partial')
            path=out/'drafts'/source['split']/source['id']/'draft.json'
            write_once(path,source)
            source_key=key(dict(id=source['id'],split=source['split'],binding=source['binding'],recipe=RECIPE))
            if source_key not in queued:
                with queue.open('ab') as stream:
                    stream.write(encoded(dict(recipe=RECIPE,source_key=source_key,id=source['id'],split=source['split'],
                        draft_path=str(path),status=source['status'],next_stage='pixel_grounded_vlm_revision',**FLAGS)))
                    stream.flush();os.fsync(stream.fileno())
                queued.add(source_key)
        return dict(recipe=RECIPE,drafts=sources,source_count=len(sources),window_calls=len(items),
            new_model_calls=sum(calls),chat_batch_sizes=calls,
            jpeg_reads=inputs.jpeg_reads,media_sha_reads=0,media_probes=0,
            synchronous_semantic_checks=0,synchronous_rewrite_calls=0,**FLAGS)
