"""Prepare exact historical requests/schemas for a paired pureLM trial, no new source claims."""
import pathlib,json,hashlib,sys,os
P=pathlib.Path;RT=P(__file__).parent
REVIEW=RT.parent/'MS_LLM32_english_merged_edges_v4/current_three_content_review_v1'
TARGET=RT.parent/'MS_LLM32_english_merged_edges_v4/Qwen3_sameprompt_three_v1'

def run():
 TARGET.mkdir(exist_ok=False);sys.path.insert(0,'/root/stage2_MS_LLM32_main_N6_v4/textp2code');import ms_llm32_codec_v4 as codec
 entries=[]
 for row in json.loads((REVIEW/'selected.json').read_bytes()):
  v=json.loads((REVIEW/(row['id']+'.json')).read_bytes());result=P(v['result_path']);old=json.loads(result.read_bytes());request_path=result.parent/'caption_polish.request.json';request_raw=request_path.read_bytes();context_raw=P(v['context_path']).read_bytes();context=json.loads(context_raw);schema=codec.response_schema(context);sampling=dict(temperature=0,max_tokens=4096,seed=0)
  payload=dict(id=v['id'],context=context,request=json.loads(request_raw),schema=schema,sampling=sampling,original_request_ref=dict(path=str(request_path),sha256=hashlib.sha256(request_raw).hexdigest()),original_result_ref=dict(path=str(result),sha256=hashlib.sha256(result.read_bytes()).hexdigest()),original_caption=old['windows'],prompt_messages_sha256=hashlib.sha256(json.dumps(json.loads(request_raw),ensure_ascii=False,sort_keys=True).encode()).hexdigest(),reference_model='Qwen3-VL-32B-Instruct',source_case_category=row['category'])
  p=TARGET/(v['id']+'.input.json');fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(fd,'w') as f:json.dump(payload,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
  entries.append(dict(id=v['id'],path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 (TARGET/'inputs_ready.json').write_text(json.dumps(dict(source_count=3,entries=entries,same_messages=True,same_schema=True,same_temperature=True,max_tokens=4096,new_source_claims=0,new_model_calls=0)));print(json.dumps(dict(prepared=3,target=str(TARGET),new_source_claims=0)))
if __name__=='__main__':run()
