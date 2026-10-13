#!/usr/bin/env python3
"""List all public repos of NZC415 with sizes."""
from huggingface_hub import HfApi

api = HfApi(token='<HF_TOKEN_REDACTED>')
try:
    models = api.list_models(author='NZC415')
    for m in models:
        print(f'MODEL {m.id}', flush=True)
        try:
            info = api.repo_info(m.id, repo_type='model', files_metadata=True)
            total = sum((f.size or 0) for f in info.siblings)
            print(f'  files={len(info.siblings)} total={total/1e9:.1f}GB', flush=True)
        except Exception as e:
            print(f'  detail fail: {str(e)[:100]}', flush=True)
except Exception as e:
    print('LIST_FAIL:', str(e)[:200], flush=True)
