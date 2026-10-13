#!/usr/bin/env python3
"""List both HF repos' contents to decide where stage4 can go."""
from huggingface_hub import HfApi

api_m2d = HfApi(token='<HF_TOKEN_REDACTED>')
api_abl = HfApi(token='<HF_TOKEN_REDACTED>')

for name, api, repo in [('NZC415/m2d', api_m2d, 'NZC415/m2d'),
                        ('Minervaaa/AblationDance', api_abl, 'Minervaaa/AblationDance')]:
    try:
        info = api.repo_info(repo, repo_type='model', files_metadata=True)
        files = info.siblings
        total = sum((f.size or 0) for f in files)
        print(f'=== {name}: {len(files)} files, total {total/1e9:.1f}GB ===', flush=True)
        for f in sorted(files, key=lambda x: x.rfilename):
            print(f'  {f.rfilename}  {(f.size or 0)/1e9:.2f}GB', flush=True)
    except Exception as e:
        print(f'=== {name} FAIL: {str(e)[:200]} ===', flush=True)
