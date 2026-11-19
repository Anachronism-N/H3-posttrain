"""CPU integration for original microclaims merged16, same prompts/windows/one-budget."""
import ast,hashlib,json,sys,tempfile,types
from pathlib import Path
import motion_revision_codec_evidence_v1 as codec
sys.modules['motion_revision_codec_v1']=codec
from stage2_local_publication_v1 import write_once,checked,LocalOutbox
import motion_revision_model_worker_evidence_v1 as w
RT=Path(__file__).parent
for n in ('motion_revision_codec_batch32_v1.py','motion_revision_model_worker_batch32_v1.py','motion_revision_native_manager_batch32_v1.py','motion_revision_feeder_batch16_v1.py'):ast.parse((RT/n).read_text())
assert codec.EVIDENCE_POLICY=='window_visibility_boundary_evidence_v1'
class Fake:
    def __init__(self):self.calls=[]
    def chat(self,requests,sampling_params,use_tqdm):
        self.calls.append(len(requests));out=[]
        for req in requests:
            raw=req[1]['content'];p=json.loads(raw[0]['text']) if isinstance(raw,list) else json.loads(raw);visual=isinstance(raw,list);windows=[]
            for plan in p['immutable_window_plan']:
                x=dict(window_id=plan['window_id'],start=plan['start'],end=plan['end']);x.update(dict(motion_description='The arms sweep outward, then lower.',corrections=[],evidence_limits=[],visibility={part:'visible' for part in codec.BODY_PARTS},supported_details=[],boundary_evidence=dict(entry_motion='The arms extend outward.',exit_motion='The arms lower.',relation_to_next='no_next_window',support_frame_indices=[])) if visual else dict(caption='The arms sweep outward, then lower.'));windows.append(x)
            v=dict(id=p['id'],split=p['split'],dance_style='urban',windows=windows)
            if not visual:v['continuity_notes']=[]
            out.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(text=json.dumps(v),token_ids=[1]*40,finish_reason='stop')]))
        return out
with tempfile.TemporaryDirectory(prefix='motion16CPU-',dir='/tmp') as d:
    root=Path(d)/'local';root.mkdir();prepared=root/'prepared';prepared.mkdir();owner='N2-g45-rv3main';refs=[]
    for b in range(4):
        rows=[];contexts=[]
        for i in range(8):
            id_='CPU'+str(b)+'Source'+str(i);p=prepared/id_;p.mkdir();img=p/'frame.jpg';img.write_bytes(bytes([255,216])+b'fixture'+bytes([255,217]));row=dict(id=id_,split='train',video_path='/synthetic/'+id_+'.mp4');plan=dict(policy='merge_short_edges',windows=[dict(window_id=0,start=0.,end=1.)]);m=dict(times=[0.,.04,.08],source_timestamps=[0.,.04,.08]);c=dict(row=row,revision=codec.REVISION,rich_body_forwarded_to_VLM=False,manifest=m,plan=plan,window_plan_sha256=hashlib.sha256(codec.encoded(plan)).hexdigest(),draft=dict(events=[],observation={}),window_frames=codec.select_window_frames(m,plan),style_hint=dict(candidates=['urban']),image_refs={str(j):dict(path=str(img),sha256=hashlib.sha256(img.read_bytes()).hexdigest()) for j in range(3)});rows.append(row);contexts.append(write_once(p/'context.json',c))
        claim=dict(owner=owner,batch_id='originalclaim'+str(b),source_count=8,rows=rows,source_keys=[[r['split'],r['id']] for r in rows],automatic_retry=False);refs.append(write_once(root/'packets'/owner/('claim'+str(b)+'.json'),dict(schema='motion_revision_prepared_packet_v1',revision=codec.REVISION,owner=owner,claim=claim,context_refs=contexts,source_count=8)))
    f=Fake();result=w.run_packets(refs,root,owner,f,lambda s:s,lambda:None);assert len(result)==32 and f.calls==[32,32];assert [r['pool_claim_batch'] for r in result]==sum(([('originalclaim'+str(i))]*8 for i in range(4)),[]);assert all(checked(r['result_ref'])['original_pool_claim_ref']==refs[i//8] for i,r in enumerate(result));assert all(checked(r['result_ref'])['fused_caption'].startswith('Dance style: urban') for r in result)
    fake=Fake()
    try:w.run_packets(refs,root,owner,fake,lambda s:s,lambda:None);raise AssertionError('Models replayed')
    except FileExistsError:pass
    assert fake.calls==[]
    try:w.run_packets([refs[0],refs[0]],root,owner,fake,lambda s:s,lambda:None);raise AssertionError('Duplicate microclaim accepted')
    except ValueError:pass
    assert fake.calls==[];box=LocalOutbox(root,Path(d)/'shared','N2');ack=box.upload(box.seal([[r['split'],r['id']] for r in result]));assert ack['source_count']==32 and ack['models_called']==0
print(json.dumps(dict(original_claim_mapping32=True,two_model_calls=[32,32],immutable_windows_preserved=True,updated_evidence_prompts=True,source_budget_retries_blocked=True,duplicate_microclaims_rejected=True,shared_commit_ACK=True,production_GPU_calls=0)))

import copy
vis={part:'visible' for part in codec.BODY_PARTS}
window=dict(window_id=0,start=0.,end=1.,motion_description='The arms sweep outward.',corrections=[],evidence_limits=[],visibility=vis,supported_details=[],boundary_evidence=dict(entry_motion='The arms extend.',exit_motion='The arms lower.',relation_to_next='no_next_window',support_frame_indices=[]))
context=dict(draft=dict(events=[],observation={}),style_hint=dict(candidates=[]),row=dict(id='evidenceFixture',split='train'),plan=dict(windows=[dict(window_id=0,start=0.,end=1.)]),window_frames=[dict(frame_indices=[0,1,2])])
audit=dict(id='evidenceFixture',split='train',dance_style='Unknown',windows=[window])
codec.validate(audit,context,'visual_audit');context['audit']=audit
rejected=0
for part,text in [('feet','There is no footwork.'),('legs','The legs remain stationary.'),('feet','The feet step outward.'),('hands','The fingers curl inward.'),('arms','The left arm extends.'),('torso','The weight shifts onto the supporting leg.')]:
    w=copy.deepcopy(window);w['visibility'][part]='not_visible' if part in ('feet','legs') else 'visible'
    try:codec.check_claims(text,w);raise AssertionError('Unsupported claim accepted: '+text)
    except ValueError:rejected+=1
w=copy.deepcopy(window);w['visibility']['feet']='partial'
try:codec.check_claims('The feet remain still.',w);raise AssertionError('Partial view used as stillness evidence')
except ValueError:rejected+=1
codec.check_claims('The image-left arm extends and the arms move in a coordinated sweep.',window)
codec.check_claims('The head turns while the chest rotates.',window)
final=dict(id='evidenceFixture',split='train',dance_style='Unknown',windows=[dict(window_id=0,start=0.,end=1.,caption='The arms sweep outward.')],continuity_notes=['The arms continue toward'])
try:codec.validate(final,context,'caption_fusion');raise AssertionError('Truncated continuity note accepted')
except ValueError:rejected+=1
p=codec.request_for(context,'caption_fusion',audit);payload=json.loads(p[1]['content']);assert 'original_timed_events' not in payload and payload['visual_audit']==audit
schema=codec.response_schema(context,'visual_audit');assert set(schema['properties']['windows']['prefixItems'][0]['properties']['visibility']['required'])==set(codec.BODY_PARTS)
print(json.dumps(dict(evidence_negative_cases_rejected=rejected,fusion_original_pose_text_withheld=True,legitimate_motion_landmarks_accepted=True)))
