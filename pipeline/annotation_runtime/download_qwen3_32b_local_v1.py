"""Download official pure-language Qwen3-32B once on local disk; no GPU or production change."""
import json,os,time
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
ROOT=Path('/opt/models/Qwen3-32B')
STATE=Path('/root/stage2_Qwen3_32B_download_v1')

def put(name,value):
 p=STATE/name;raw=json.dumps(value,ensure_ascii=False).encode();fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def run():
 STATE.mkdir(exist_ok=False);start=time.time();api=HfApi();info=api.model_info('Qwen/Qwen3-32B',files_metadata=True);revision=info.sha;files=[f for f in info.siblings if f.rfilename.endswith(('.json','.safetensors','.txt','.model','.jinja','.md')) and not f.rfilename.startswith('.')];expected={f.rfilename:f.size for f in files};put('download_started.json',dict(repo_id='Qwen/Qwen3-32B',resolved_revision=revision,target=str(ROOT),expected_files=expected,expected_bytes=sum(n or 0 for n in expected.values()),started_unix=start,GPU_used=0,current_models_untouched=True));print(json.dumps(dict(event='Qwen3_32B_download_started',revision=revision,files=len(files),bytes=sum(n or 0 for n in expected.values()),target=str(ROOT))),flush=True)
 snapshot_download(repo_id='Qwen/Qwen3-32B',revision=revision,local_dir=str(ROOT),allow_patterns=list(expected),max_workers=4)
 missing=[name for name,size in expected.items() if not (ROOT/name).is_file() or size is not None and (ROOT/name).stat().st_size!=size];assert not missing,missing
 config=json.loads((ROOT/'config.json').read_bytes());index=json.loads((ROOT/'model.safetensors.index.json').read_bytes());shards=sorted(set(index['weight_map'].values()));assert config['model_type']=='qwen3' and all((ROOT/f).is_file() for f in shards);put('DOWNLOAD_COMPLETE.json',dict(repo_id='Qwen/Qwen3-32B',resolved_revision=revision,target=str(ROOT),model_type=config['model_type'],architectures=config.get('architectures'),weight_shards=len(shards),files_verified_by_size=len(expected),total_bytes=sum((ROOT/name).stat().st_size for name in expected),download_wall_seconds=time.time()-start,completed_unix=time.time(),GPU_used=0,not_deployed=True));print(json.dumps(dict(event='Qwen3_32B_download_complete',target=str(ROOT),shards=len(shards),wall_seconds=time.time()-start,not_deployed=True)),flush=True)
if __name__=='__main__':run()
