#!/usr/bin/env python3
"""Check NZC415/m2d repo state: does stage4 args.json exist remotely?"""
import sys
from huggingface_hub import HfApi

api = HfApi(token='<HF_TOKEN_REDACTED>')
try:
    files = api.list_repo_files('NZC415/m2d', repo_type='model')
    stage4 = [f for f in files if 'stage4' in f]
    total = len(files)
    print(f'REPO_OK total_files={total} stage4_files={stage4}', flush=True)
except Exception as e:
    print('REPO_FAIL:', str(e)[:300], flush=True)
