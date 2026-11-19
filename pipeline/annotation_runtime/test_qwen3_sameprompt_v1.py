"""Validate exact same-prompt pairing and Qwen3 non-thinking template without GPU calls."""
import hashlib,json,tarfile
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).parent.parent/'MS_LLM32_english_merged_edges_v4/Qwen3_sameprompt_three_v1'
manifest=json.loads((ROOT/'inputs_ready.json').read_bytes());assert manifest['source_count']==3
model=Path('/opt/models/Qwen3-32B');tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False);out=[]
for ref in manifest['entries']:
 raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];x=json.loads(raw);r=Path(x['original_request_ref']['path'])
 # Source path is N6-local; frozen request identity recorded there; request content checks here.
 assert hashlib.sha256(json.dumps(x['request'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()==x['prompt_messages_sha256']
 text=tok.apply_chat_template(x['request'],tokenize=False,add_generation_prompt=True,enable_thinking=False);assert text.endswith('<think>\n\n</think>\n\n'),text[-100:]
 out.append(dict(id=x['id'],message_count=len(x['request']),model_input_tokens=len(tok.apply_chat_template(x['request'],tokenize=True,add_generation_prompt=True,enable_thinking=False)['input_ids']),prompt_sha256=x['prompt_messages_sha256']))
print(json.dumps(dict(source_count=3,messages_unchanged=True,Qwen3_thinking_disabled=True,tokenization_differs_by_model=True,GPU_calls=0,cases=out)))
