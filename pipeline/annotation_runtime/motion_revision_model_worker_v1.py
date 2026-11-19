"""Resident corrected motion revision: audit all original windows then improve continuity."""
import hashlib
import json
from pathlib import Path
import time
import motion_revision_codec_v1 as codec
from stage2_local_publication_v1 import write_once,checked,require

FLAGS=dict(quality_approved=False,semantic_approval=False,training_eligible=False,human_review_completed=False,needs_visual_revision=True)


def strict(raw):
    def pairs(xs):
        out={}
        for k,v in xs:require(k not in out,'Duplicate model JSON key');out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite model output')))


def run_packet(ref,out,owner,llm,sampler,gate):
    root=Path(out);packet=checked(ref);require(packet['schema']=='motion_revision_prepared_packet_v1' and packet['revision']==codec.REVISION and packet['owner']==owner and packet['claim']['owner']==owner,'Correct revision original owner required');require(1<=len(packet['context_refs'])==packet['source_count']<=8,'Bounded eight-source model batch required');contexts=[]
    for r,row in zip(packet['context_refs'],packet['claim']['rows']):
        context=checked(r);require(context['revision']==codec.REVISION and context['row']['id']==row['id'] and context['row']['split']==row['split'] and context['rich_body_forwarded_to_VLM'] is False,'Source window context differs')
        require(context['window_plan_sha256']==hashlib.sha256(codec.encoded(context['plan'])).hexdigest() and context['plan']['policy']=='merge_short_edges','Original merged-edge plan changed');require(codec.select_window_frames(context['manifest'],context['plan'])==context['window_frames'],'Native per-window frames changed')
        images={}
        for i,x in context['image_refs'].items():p=Path(x['path']);require(p.is_relative_to(root/'prepared') and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],'Owned original local image changed');images[int(i)]=str(p)
        context.update(images=images,error=None,stages=[],directory=root/'sources'/row['split']/row['id']);contexts.append(context)
    gate()
    for c in contexts:write_once(root/'budgets'/c['row']['split']/c['row']['id']/'source.claim.json',dict(revision=codec.REVISION,owner=owner,source=c['row'],original_pool_claim=packet['claim'],maximum_model_calls=2,automatic_retry=False))
    batch='rv-'+hashlib.sha256(json.dumps(ref,sort_keys=True).encode()).hexdigest()[:32]
    for stage in ('visual_audit','caption_fusion'):
        active=[c for c in contexts if c['error'] is None]
        if not active:continue
        requests=[];params=[]
        for c in active:
            write_once(root/'budgets'/c['row']['split']/c['row']['id']/(stage+'.claim.json'),dict(revision=codec.REVISION,stage=stage,maximum_calls=1,automatic_retry=False));request=codec.request_for(c,stage,c.get('audit'));request_ref=write_once(c['directory']/(stage+'.request.json'),request);requests.append(request);params.append(sampler(codec.response_schema(c,stage)));c['stages'].append(dict(stage=stage,request_sha256=request_ref['sha256'],model_call_performed=True))
        directory=root/'batch_calls'/(batch+'-'+stage);br=write_once(directory/'request.json',dict(revision=codec.REVISION,stage=stage,entries=[dict(id=c['row']['id'],split=c['row']['split'],position=i) for i,c in enumerate(active)],requests=requests));gate();start=time.monotonic();responses=[];error=None
        try:
            values=llm.chat(requests,sampling_params=params,use_tqdm=False);require(len(values)==len(active),'Response count/source mapping differs')
            for v in values:require(len(v.outputs)==1,'Single source candidate required');x=v.outputs[0];responses.append(dict(raw_text=x.text,completion_tokens=len(x.token_ids),finish_reason=x.finish_reason))
        except Exception as e:error=repr(e)
        resp=write_once(directory/'response.json',dict(revision=codec.REVISION,stage=stage,request_ref=br,responses=responses,error=error,wall_seconds=time.monotonic()-start))
        for i,c in enumerate(active):
            record=c['stages'][-1];record.update(batch_request_ref=br,batch_response_ref=resp,batch_position=i)
            try:
                require(error is None,'Model batch error '+str(error));x=responses[i];record.update(x);require(x['finish_reason']=='stop' and 0<x['completion_tokens']<6144,'Truncated/incomplete revision must be sealed, never success');value=codec.validate(strict(x['raw_text']),c,stage);record['parsed']=value
                if stage=='visual_audit':c['audit']=value;write_once(c['directory']/'visual_audit.normalized.json',value)
                else:c['caption']=value
            except Exception as e:c['error']=repr(e);record['error']=c['error']
            write_once(c['directory']/(stage+'.raw.json'),{k:v for k,v in record.items() if k!='parsed'})
    outcomes=[]
    for c in contexts:
        row=c['row'];status='failed_sealed_no_retry' if c['error'] else 'postprocessed_unapproved';value=dict(id=row['id'],split=row['split'],revision=codec.REVISION,status=status,owner=owner,input_refs=row,original_pool_claim_ref=ref,pool_claim_batch=packet['claim']['batch_id'],window_plan=c['plan'],window_plan_sha256=c['window_plan_sha256'],selected_window_frames=c['window_frames'],single_source_video_copy=True,stages=c['stages'],error=c['error'],**FLAGS)
        if not c['error']:value.update(dance_style=c['caption']['dance_style'],windows=c['caption']['windows'],fused_caption=codec.render(c['caption']),visual_audit=c['audit'],continuity_notes=c['caption']['continuity_notes'],rich_role='dance_style_only',language='English',format_approved_by_protocol=True)
        rr=write_once(c['directory']/'postprocessed.json',value);cr=dict(id=row['id'],split=row['split'],revision=codec.REVISION,status=status,result_ref=rr,new_model_calls=len(c['stages']),owner=owner,pool_claim_batch=packet['claim']['batch_id'],**FLAGS);refc=write_once(c['directory']/'commit.json',cr);outcomes.append(dict(commit_ref=refc,**cr))
    return outcomes
