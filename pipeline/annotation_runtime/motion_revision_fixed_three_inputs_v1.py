"""Transfer three existing diagnostic cases, preserving original contexts and image evidence."""
import argparse,hashlib,json,os,shutil,sys,time
from pathlib import Path
RT=Path(__file__).parent
IDS=('wToyUHvrVoLgBvUd','XiCMwgEceOlirSYX','pImhVAYNzZmZiVZX')
SHARED=RT.parent/'motion_revision_english_merged_edges_final_20261028_v1/fixed_three_evidence_diagnostics_v1'

def prepare():
    root=Path('/root/stage2_motion_revision_final_main_N3_20261028_v1/out');sys.path.insert(0,str(root.parent/'batch32code'))
    from stage2_local_publication_v1 import checked,write_once
    SHARED.mkdir(exist_ok=False);entries=[]
    for sid in IDS:
        ps=list((root/'sources').glob('*/'+sid+'/postprocessed.json'));assert len(ps)==1;p=ps[0];result=json.loads(p.read_bytes());packet=checked(result['original_pool_claim_ref']);refs=[]
        for ref in packet['context_refs']:
            c=checked(ref)
            if c['row']['id']==sid:refs.append((ref,c))
        assert len(refs)==1;original,c=refs[0];d=SHARED/sid;d.mkdir();images={}
        for index,ref in c['image_refs'].items():
            raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];p=d/('frame-'+index+'.jpg');fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            images[index]=dict(path=str(p),sha256=ref['sha256'])
        cr=write_once(d/'context.json',dict(c,image_refs=images,diagnostic_provenance=dict(original_context_ref=original,original_result_path=str(ps[0]),original_result_sha256=hashlib.sha256(ps[0].read_bytes()).hexdigest(),original_packet_ref=result['original_pool_claim_ref'])))
        rr=write_once(d/'original_result.json',result);entries.append(dict(id=sid,context_ref=cr,original_result_ref=rr))
        from PIL import Image,ImageDraw
        groups=c['window_frames'];canvas=Image.new('RGB',(8*180,len(groups)*350),'white');draw=ImageDraw.Draw(canvas)
        for y,w in enumerate(groups):
            for x,frame in enumerate(w['frames']):
                im=Image.open(images[str(frame['frame_index'])]['path']);im.thumbnail((176,320));canvas.paste(im,(x*180,y*350+24));draw.text((x*180,y*350+4),'W%d / %.4fs'%(w['window_id'],frame['time']),fill='black')
        canvas.save(d/'all_native_samples.jpg',quality=90)
    write_once(SHARED/'manifest.json',dict(schema='existing_fixed_three_diagnostic_inputs_v1',source_count=3,entries=entries,model_pool_claims=0,new_model_calls=0,created_unix=time.time()));print(json.dumps(dict(prepared=3,shared=str(SHARED),new_model_calls=0)))

def receive():
    root=Path('/root/stage2_motion_revision_final_main_N2_20261028_v1');sys.path.insert(0,str(root/'code'))
    from stage2_local_publication_v1 import checked,write_once
    manifest=json.loads((SHARED/'manifest.json').read_bytes());assert manifest['source_count']==3 and tuple(e['id'] for e in manifest['entries'])==IDS
    local=root/'evidence_fixed_three_inputs_v1';local.mkdir(exist_ok=False);refs=[]
    for e in manifest['entries']:
        c=checked(e['context_ref']);d=local/e['id'];d.mkdir();images={}
        for index,ref in c['image_refs'].items():
            raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];p=d/('frame-'+index+'.jpg');fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            images[index]=dict(path=str(p),sha256=ref['sha256'])
        refs.append(write_once(d/'context.json',dict(c,image_refs=images,diagnostic_input_reference=e)))
    write_once(root/'evidence_fixed_three_manifest_v1.json',dict(schema='existing_fixed_three_diagnostic_inputs_v1',source_count=3,context_refs=refs,source_ids=list(IDS),diagnostic_only=True,model_pool_claims=0,original_model_budgets_untouched=True,stage_budget_each=1,automatic_retry=False,created_unix=time.time()));print(json.dumps(dict(received=3,no_model_pool_claims=True)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','receive'));a=p.parse_args();prepare() if a.mode=='prepare' else receive()
