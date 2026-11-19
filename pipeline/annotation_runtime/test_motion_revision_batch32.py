"""CPU integration for original microclaims merged16, same prompts/windows/one-budget."""
import ast,hashlib,json,sys,tempfile,types
from pathlib import Path
import motion_revision_codec_v3 as codec
sys.modules['motion_revision_codec_v1']=codec
from stage2_local_publication_v1 import write_once,checked,LocalOutbox
import motion_revision_model_worker_batch32_v1 as w
RT=Path(__file__).parent
for n in ('motion_revision_codec_batch32_v1.py','motion_revision_model_worker_batch32_v1.py','motion_revision_native_manager_batch32_v1.py','motion_revision_feeder_batch16_v1.py'):ast.parse((RT/n).read_text())
assert (RT/'motion_revision_codec_batch32_v1.py').read_text()==(RT/'motion_revision_codec_v3.py').read_text().replace('max_num_seqs=8,max_model_len','max_num_seqs=32,max_model_len')
class Fake:
    def __init__(self):self.calls=[]
    def chat(self,requests,sampling_params,use_tqdm):
        self.calls.append(len(requests));out=[]
        for req in requests:
            raw=req[1]['content'];p=json.loads(raw[0]['text']) if isinstance(raw,list) else json.loads(raw);visual=isinstance(raw,list);windows=[]
            for plan in p['immutable_window_plan']:
                x=dict(window_id=plan['window_id'],start=plan['start'],end=plan['end']);x.update(dict(motion_description='The arms sweep outward, then lower.',corrections=[],evidence_limits=[]) if visual else dict(caption='The arms sweep outward, then lower.'));windows.append(x)
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
print(json.dumps(dict(original_claim_mapping32=True,two_model_calls=[32,32],unchanged_window_and_prompt_bytes=True,source_budget_retries_blocked=True,duplicate_microclaims_rejected=True,shared_commit_ACK=True,production_GPU_calls=0)))
