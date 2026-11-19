"""One resident original VL model consumes prepared local packets and writes only local results."""
import hashlib
import json
from pathlib import Path
import time
from stage2_local_publication_v1 import write_once,checked,require
from stage2_local_packet_v1 import validate_packet
import stage2_model_codec_v1 as model

FLAGS=dict(quality_approved=False,semantic_approval=False,training_eligible=False,needs_visual_revision=True,human_review_completed=False)


def parse_refs(refs):
    return json.dumps(refs,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':'))


def run_packet(packet_ref,local_root,owner,llm,sampler,admission_gate):
    return run_packets([packet_ref],local_root,owner,llm,sampler,admission_gate)


def parse_strict(raw):
    def pairs(values):
        result={}
        for key,value in values:
            require(key not in result,'Duplicate model response key');result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite model JSON')))


def run_packets(packet_refs,local_root,owner,llm,sampler,admission_gate):
    root=Path(local_root);require(root.is_absolute() and str(root).startswith(('/root/','/tmp/')) and root.resolve()==root,'Owned local-disk pipeline root required')
    require(1<=len(packet_refs)<=4,'GPU group may contain up to four atomic source claims')
    contexts=[];source_keys=set()
    for packet_ref in packet_refs:
        packet=checked(packet_ref);current=validate_packet(packet,root/'prepared',owner);require(1<=len(current)<=8,'Dynamic prepared packet must contain at most eight sources')
        for ctx in current:
            key=(ctx['original_row']['split'],ctx['original_row']['id']);require(key not in source_keys,'Duplicate source across GPU microclaims')
            source_keys.add(key);ctx['original_claim']=packet['claim'];ctx['original_packet_ref']=packet_ref;contexts.append(ctx)
    require(len(contexts)<=32,'At most thirty-two active conversations per resident model')
    batch_id='gpu-'+hashlib.sha256(parse_refs(packet_refs).encode()).hexdigest()[:32];out=root
    admission_gate()
    own=[]
    for ctx in contexts:
        row=ctx['original_row'];id_=row['id'];split=row['split'];directory=out/'sources'/split/id_
        try:
            write_once(out/'budgets'/split/id_/'source.claim.json',dict(source=row,original_pool_claim=ctx['original_claim'],maximum_model_calls=2,automatic_retry=False))
        except FileExistsError:
            raise RuntimeError('Permanent existing local source budget; never retry model generation')
        ctx.update(directory=directory,records=[],error=None)
        own.append(ctx)
    for stage in ('visual_audit','caption_fusion'):
        active=[ctx for ctx in own if ctx['error'] is None]
        if not active:continue
        conversations=[];params=[]
        for ctx in active:
            row=ctx['original_row'];frames=ctx['selected_frames'];write_once(out/'budgets'/row['split']/row['id']/(stage+'.claim.json'),dict(stage=stage,id=row['id'],split=row['split'],maximum_calls=1,automatic_retry=False,original_pool_batch=batch_id))
            request=model.request_for(row,ctx['draft'],frames,ctx['images'] if stage=='visual_audit' else [],stage,ctx.get('audit'))
            ref=write_once(ctx['directory']/(stage+'.request.json'),request);schema=model.response_schema(row,stage,frames if stage=='visual_audit' else None)
            ctx['current_schema']=schema;conversations.append(request);params.append(sampler(schema));ctx['records'].append(dict(stage=stage,request=request,request_sha256=ref['sha256'],model_call_performed=True))
        directory=out/'batch_calls'/(batch_id+'-'+stage);entries=[dict(id=c['original_row']['id'],split=c['original_row']['split'],position=i,request_ref=dict(path=str(c['directory']/(stage+'.request.json')),sha256=c['records'][-1]['request_sha256'])) for i,c in enumerate(active)]
        request_ref=write_once(directory/'request.json',dict(schema='stage2_local_batch_request_v1',stage=stage,batch_size=len(active),entries=entries,original_pool_batch=batch_id))
        admission_gate();begin=time.monotonic();error=None;responses=[]
        try:
            outputs=llm.chat(conversations,sampling_params=params,use_tqdm=False)
            for output in outputs:
                responses.append(dict(candidate_count=len(output.outputs),candidates=[dict(raw_text=x.text,completion_tokens=len(x.token_ids),finish_reason=x.finish_reason) for x in output.outputs]))
            require(len(responses)==len(active),'Batch output mapping count differs')
        except Exception as e:error=type(e).__name__+': '+str(e)
        seconds=time.monotonic()-begin
        response_ref=write_once(directory/'response.json',dict(schema='stage2_local_batch_response_v1',request_ref=request_ref,stage=stage,input_count=len(active),output_count=len(responses),responses=responses,error=error,wall_seconds=seconds))
        for position,ctx in enumerate(active):
            record=ctx['records'][-1];record.update(batch_position=position,batch_request_ref=request_ref,batch_response_ref=response_ref)
            if error:record['error']=error;ctx['error']=error
            elif responses[position]['candidate_count']!=1:record['error']='Expected one source-bound candidate';ctx['error']=record['error']
            else:
                candidate=responses[position]['candidates'][0];record.update(candidate,raw_sha256=hashlib.sha256(candidate['raw_text'].encode()).hexdigest())
                try:
                    require(candidate['finish_reason']=='stop' and 0<candidate['completion_tokens']<1800,'Incomplete model generation must be sealed')
                    parsed=parse_strict(candidate['raw_text']);model.schema_check(parsed,ctx['current_schema']);record['parsed']=parsed
                    if stage=='visual_audit':ctx['raw_audit']=parsed;ctx['audit']=model.audit_check(parsed,ctx['original_row'],ctx['selected_frames']);write_once(ctx['directory']/'visual_audit.normalized.json',ctx['audit'])
                    else:ctx['caption']=parsed
                except Exception as e:ctx['error']=type(e).__name__+': '+str(e)
            write_once(ctx['directory']/(stage+'.raw.json'),{k:v for k,v in record.items() if k!='parsed'})
    commits=[]
    for ctx in own:
        row=ctx['original_row'];directory=ctx['directory'];status='failed_sealed_no_retry';extra={}
        if ctx['error'] is None:
            audit=ctx['audit'];caption=ctx['caption']
            try:
                require(bool(audit['supported_actions'])==bool(caption['fused_caption'].strip()),'Caption must be empty exactly when visual evidence is empty')
                violations=model.evidence_policy_violations(audit,caption);insufficient=not bool(audit['supported_actions']);status='postprocessed_needs_visual_revision' if violations else 'insufficient_visible_evidence' if insufficient else 'postprocessed_unapproved'
                extra=dict(visual_audit_raw=ctx['raw_audit'],visual_audit=audit,fused_caption=caption['fused_caption'],corrections=caption['corrections'],limitations=caption['limitations'],evidence_policy_violations=violations)
            except Exception as e:ctx['error']=type(e).__name__+': '+str(e)
        value=dict(id=row['id'],split=row['split'],status=status,input_refs=row,original_pool_claim_ref=ctx['original_packet_ref'],pool_claim_batch=ctx['original_claim']['batch_id'],owner=owner,selected_frames=ctx['selected_frames'],single_source_video_copy=True,stages=ctx['records'],model_revision_performed=ctx['error'] is None,error=ctx['error'],**extra,**FLAGS)
        result_ref=write_once(directory/'postprocessed.json',value);commit=dict(id=row['id'],split=row['split'],status=status,result_ref=result_ref,new_model_calls=len(ctx['records']),pool_claim_batch=ctx['original_claim']['batch_id'],owner=owner,**FLAGS);commit_ref=write_once(directory/'commit.json',commit);commits.append(dict(commit_ref=commit_ref,**commit))
    write_once(out/'completed_packets'/(batch_id+'.json'),dict(schema='stage2_local_completed_packet_v1',batch_id=batch_id,owner=owner,source_count=len(commits),outcomes=commits,model_requests=sum(c['new_model_calls'] for c in commits),completed_unix=time.time(),**FLAGS))
    return commits
