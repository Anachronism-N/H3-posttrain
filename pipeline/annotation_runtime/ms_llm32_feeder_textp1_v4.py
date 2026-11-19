"""Local JSON-only input feeder: no video copying, decode, frames, image arrays or VLM summaries."""
import argparse,json,secrets,time,urllib.request
from pathlib import Path
import ms_llm32_codec_v4 as codec
from stage2_local_publication_v1 import write_once,require


def run(path):
    cfg=json.loads(Path(path).read_bytes());out=Path(cfg['out']);owner=cfg['owner'];q=out/'ready'/owner;q.mkdir(parents=True,exist_ok=True);claims=out/'feeder_claims'/owner;claims.mkdir(parents=True,exist_ok=True);token=Path(cfg['token_path']).read_text().strip()
    while not (out/'FEEDER_STOP_TEXTP1').exists():
        ready=[p for p in q.glob('*.ready.json') if not (out/'consumed_ready'/owner/p.name).exists()]
        if len(ready)>=8:time.sleep(.5);continue
        request_id=owner+'-'+secrets.token_hex(16);request=urllib.request.Request(cfg['claim_endpoint'],data=json.dumps(dict(owner=owner,request_id=request_id,size=8)).encode(),headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request,timeout=60) as f:raw=f.read(16*2**20+1)
            require(len(raw)<=16*2**20,'Bounded metadata response');claim=json.loads(raw)
        except Exception as e:
            write_once(claims/(request_id+'_uncertain.json'),dict(original_request_id=request_id,error=repr(e),automatic_retry=False,models_called=0));write_once(q/'FEEDER_FAILED_TEXTP1',dict(error=repr(e),claims_preserved=True));raise
        if claim is None:break
        if claim.get('waiting_for_original_inputs'):time.sleep(2);continue
        require(claim['owner']==owner and claim['automatic_retry'] is False and len(claim['prepared_metadata'])==claim['source_count'],'New text-stage owner and metadata')
        metadata=claim.pop('prepared_metadata');write_once(claims/(claim['batch_id']+'.json'),claim);refs=[];begin=time.monotonic()
        try:
            for c in metadata:
                require(c['revision']==codec.REVISION and c['rich_body_forwarded_to_VLM'] is False,'Exact new text-stage input')
                value={k:c[k] for k in ('revision','row','plan','window_plan_sha256','draft','style_hint','rich_body_forwarded_to_VLM')};value['input_metadata_revision']=c.get('input_metadata_revision');value['no_video_or_images_read']=True
                refs.append(write_once(out/'prepared'/c['row']['id']/'text_context.json',value))
            p=out/'packets'/owner/(claim['batch_id']+'.json');ref=write_once(p,dict(schema='MS_LLM32_text_packet_v4',revision=codec.REVISION,owner=owner,claim=claim,context_refs=refs,source_count=len(refs)));write_once(q/(p.name+'.ready.json'),dict(packet_ref=ref,source_count=len(refs),owner=owner,seconds=time.monotonic()-begin));print(json.dumps(dict(event='MS_LLM_text_packet_ready',owner=owner,sources=len(refs),seconds=time.monotonic()-begin)),flush=True)
        except Exception as e:
            write_once(claims/(claim['batch_id']+'_prepare_failure.json'),dict(original_claim=claim,error=repr(e),automatic_retry=False,models_called=0));write_once(q/'FEEDER_FAILED_TEXTP1',dict(error=repr(e),claims_preserved=True));raise
    write_once(q/'FEEDER_DRAINED_TEXTP1',dict(owner=owner,models_called=0,ended_unix=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
