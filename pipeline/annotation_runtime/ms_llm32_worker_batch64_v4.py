"""One text-only32B wording call per original source; immutable local budgets and outbox."""
import hashlib,json,time
from pathlib import Path
import ms_llm32_codec_v4 as codec
from stage2_local_publication_v1 import write_once,checked,require
FLAGS=dict(quality_approved=False,semantic_approval=False,training_eligible=False,human_review_completed=False)


def strict(raw):
    def pairs(xs):
        out={}
        for k,v in xs:require(k not in out,'Duplicate JSON key');out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite output')))


def run_packets(refs,out,owner,llm,sampler,gate):
    root=Path(out);contexts=[];keys=set()
    for ref in refs:
        packet=checked(ref);require(packet['schema']=='MS_LLM32_text_packet_v4' and packet['revision']==codec.REVISION and packet['owner']==owner and packet['claim']['owner']==owner,'Exact text-stage source ownership')
        claim=packet['claim'];require(1<=len(packet['context_refs'])==len(claim['rows'])==packet['source_count']<=8 and claim['source_keys']==[[r['split'],r['id']] for r in claim['rows']],'Original microclaim source mapping')
        for ctxref,row in zip(packet['context_refs'],claim['rows']):
            c=checked(ctxref);require(c['revision']==codec.REVISION and c['row']['id']==row['id'] and c['row']['split']==row['split'] and c['rich_body_forwarded_to_VLM'] is False,'Original text input identity')
            require(row['id'].casefold() not in keys,'Duplicate source in model batch');keys.add(row['id'].casefold());require(c['plan']['policy']=='merge_short_edges' and c['window_plan_sha256']==hashlib.sha256(codec.encoded(c['plan'])).hexdigest(),'Original window plan unchanged')
            c.update(error=None,stage_records=[],directory=root/'sources'/row['split']/row['id'],original_claim=claim,original_packet_ref=ref);contexts.append(c)
    require(1<=len(contexts)<=64,'At most64 independent text requests');gate();requests=[];params=[]
    for c in contexts:
        d=root/'budgets'/c['row']['split']/c['row']['id'];write_once(d/'source.claim.json',dict(revision=codec.REVISION,owner=owner,source=c['row'],original_pool_claim=c['original_claim'],maximum_model_calls=1,automatic_retry=False));write_once(d/'caption_polish.claim.json',dict(revision=codec.REVISION,stage='caption_polish',maximum_calls=1,automatic_retry=False))
        request=codec.request_for(c);r=write_once(c['directory']/'caption_polish.request.json',request);requests.append(request);params.append(sampler(codec.response_schema(c)));c['stage_records'].append(dict(stage='caption_polish',request_sha256=r['sha256'],model_call_performed=True))
    batch='text-'+hashlib.sha256(codec.encoded(refs)).hexdigest()[:32];d=root/'batch_calls'/batch;br=write_once(d/'request.json',dict(revision=codec.REVISION,stage='caption_polish',source_ids=[c['row']['id'] for c in contexts],requests=requests));gate();start=time.monotonic();responses=[];error=None
    try:
        values=llm.chat(requests,sampling_params=params,use_tqdm=False);require(len(values)==len(contexts),'Model/source response count differs')
        for v in values:
            require(len(v.outputs)==1,'One source candidate');x=v.outputs[0];responses.append(dict(raw_text=x.text,completion_tokens=len(x.token_ids),finish_reason=x.finish_reason))
    except Exception as e:error=repr(e)
    resp=write_once(d/'response.json',dict(revision=codec.REVISION,stage='caption_polish',request_ref=br,responses=responses,error=error,wall_seconds=time.monotonic()-start));outcomes=[]
    for i,c in enumerate(contexts):
        record=c['stage_records'][0];record.update(batch_request_ref=br,batch_response_ref=resp,batch_position=i)
        try:
            require(error is None,'Model batch error '+str(error));x=responses[i];record.update(x);require(x['finish_reason']=='stop' and 0<x['completion_tokens']<4096,'Incomplete output sealed, no retry');value=codec.validate(strict(x['raw_text']),c);c['caption']=codec.validate(codec.normalize(value,c),c);write_once(c['directory']/'caption_polish.normalized.json',c['caption'])
        except Exception as e:c['error']=repr(e);record['error']=c['error']
        write_once(c['directory']/'caption_polish.raw.json',record);row=c['row'];status='failed_sealed_no_retry' if c['error'] else 'postprocessed_unapproved';v=dict(id=row['id'],split=row['split'],revision=codec.REVISION,status=status,owner=owner,input_refs=row,original_pool_claim_ref=c['original_packet_ref'],pool_claim_batch=c['original_claim']['batch_id'],window_plan=c['plan'],window_plan_sha256=c['window_plan_sha256'],stages=c['stage_records'],error=c['error'],pipeline='original_MS_LLM32B_text_only',image_inputs=0,video_inputs=0,VLM_patches_applied=0,rich_role='dance_style_only',language='English',maximum_model_calls_per_source=1,**FLAGS)
        if not c['error']:v.update(dance_style=c['caption']['dance_style'],windows=c['caption']['windows'],fused_caption=codec.render(c['caption']),continuity_notes=[])
        rr=write_once(c['directory']/'postprocessed.json',v);cr=dict(id=row['id'],split=row['split'],revision=codec.REVISION,status=status,result_ref=rr,new_model_calls=1,owner=owner,pool_claim_batch=c['original_claim']['batch_id'],**FLAGS);rc=write_once(c['directory']/'commit.json',cr);outcomes.append(dict(commit_ref=rc,**cr))
    return outcomes
